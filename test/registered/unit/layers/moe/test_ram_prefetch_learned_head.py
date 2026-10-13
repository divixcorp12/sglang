"""The learned L+2 head file format and its serving loader (CPU)."""

from contextlib import ExitStack

import pytest
import torch

from sglang.srt.arg_groups.expert_stream_requirements_exl3 import _check_ram_prefetch
from sglang.srt.environ import envs
from sglang.srt.layers.moe import ram_prefetch_learned_head as head_file
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SOURCE = {"path": "f109/result.json", "split": "calib", "scorer": "L2_T3", "budget": 0.75}
ROWS = (head_file.HeadRow(0, 2, 0), head_file.HeadRow(1, 3, 1))


def _shapes(rows: int) -> dict[str, tuple[int, ...]]:
    return {
        "w_pca": (4, 6), "w_in_pca": (5, 4), "b_in": (5,), "w_in_sparse": (5, 3), "row_bias": (rows, 5),
        "pos_bias": (0,), "w_r1": (5, 5), "b_r1": (5,), "w_r2": (5, 5), "b_r2": (5,), "w_gate": (5, 5),
        "b_gate": (5,), "w_out": (7, 5), "b_out": (7,), "agg_w1": (3, 2), "agg_row_bias": (rows, 3),
        "agg_w2": (1, 3), "agg_b2": (1,),
    }  # fmt: skip


def _tensors(rows: int = len(ROWS)) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(7)
    out = {}
    for name, shape in _shapes(rows).items():
        dtype = torch.bfloat16 if name in head_file.MATRIX_FIELDS else torch.float32
        out[name] = torch.randn(shape, generator=gen).to(dtype)
    return out


def _meta(**over) -> head_file.HeadMeta:
    fields = dict(
        format_version=head_file.HEAD_FORMAT_VERSION, checkpoint_sha256="ab" * 32, threshold=0.1301608383655548,
        threshold_source=SOURCE, horizon=2, rows=ROWS,
    )  # fmt: skip
    return head_file.HeadMeta(**{**fields, **over})


def _write(tmp_path, *, tensors=None, meta=None) -> str:
    path = str(tmp_path / "head.safetensors")
    head_file.write_head_file(path, tensors=tensors or _tensors(), meta=meta or _meta())
    return path


def _targets(pairs: dict[int, int], rows: int = 4) -> torch.Tensor:
    t = torch.full((rows, 2), -1, dtype=torch.int64)
    for src, dst in pairs.items():
        t[src, 0], t[src, 1] = dst, 0
    return t


def test_round_trip_is_bit_exact(tmp_path):
    """Mutant: the writer casts a tensor or rounds the threshold through float32; the read-back differs."""
    tensors = _tensors()
    meta = _meta()
    got, got_meta = head_file.read_head_file(_write(tmp_path, tensors=tensors, meta=meta))
    assert got_meta == meta
    assert set(got) == set(tensors)
    for name, want in tensors.items():
        assert got[name].dtype == want.dtype and torch.equal(got[name], want), name


def test_version_mismatch_refused(tmp_path):
    """Mutant: read_head_file stops comparing format_version; a future file would load as v1."""
    path = _write(tmp_path, meta=_meta(format_version=head_file.HEAD_FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="format_version"):
        head_file.read_head_file(path)


def test_missing_tensor_refused(tmp_path):
    """Mutant: the tensor check iterates the file's names instead of SpecHeadWeights' fields."""
    tensors = _tensors()
    del tensors["w_gate"]
    with pytest.raises(ValueError, match="w_gate"):
        head_file.read_head_file(_write(tmp_path, tensors=tensors))


def test_wrong_dtype_refused(tmp_path):
    """Mutant: the dtype check is dropped; an fp32 matrix would reach a kernel that reads bf16."""
    tensors = _tensors()
    tensors["w_out"] = tensors["w_out"].float()
    with pytest.raises(ValueError, match="w_out"):
        head_file.read_head_file(_write(tmp_path, tensors=tensors))


def test_absent_aggregator_refused(tmp_path):
    """Mutant: an empty agg_* (a T0 head) is accepted by the T3 loader."""
    tensors = _tensors()
    tensors["agg_w1"] = torch.empty(0, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="aggregator"):
        head_file.read_head_file(_write(tmp_path, tensors=tensors))


def test_wrong_horizon_refused(tmp_path):
    """Mutant: the horizon check is dropped; an L+1 head would serve as L+2."""
    with pytest.raises(ValueError, match="horizon"):
        head_file.read_head_file(_write(tmp_path, meta=_meta(horizon=1)))


def test_missing_file_refused(tmp_path):
    """Mutant: an OSError from the safetensors open escapes instead of a ValueError naming the file."""
    with pytest.raises(ValueError, match="nope.safetensors"):
        head_file.read_head_file(str(tmp_path / "nope.safetensors"))


def test_uncovered_layer_pair_refused(tmp_path):
    """Mutant: coverage matches the target layer only; (1, 3) then hits row (7, 3) of the file."""
    meta = _meta(rows=(head_file.HeadRow(0, 2, 0), head_file.HeadRow(7, 3, 1)))
    path = _write(tmp_path, meta=meta)
    targets = _targets({0: 2, 1: 3})
    with pytest.raises(ValueError, match=r"\(1, 3\)"):
        head_file.load_learned_head(path, layer_ids=[0, 1, 2, 3], targets=targets)


def test_rows_map_by_layer_pair_not_map_row(tmp_path):
    """Mutant: head rows are indexed by map row; layer ids offset from the map rows would pick the wrong bias."""
    path = _write(tmp_path)
    targets = _targets({1: 3, 2: 4}, rows=5)
    loaded = head_file.load_learned_head(path, layer_ids=[9, 0, 1, 2, 3], targets=targets)
    assert loaded.row_of_target == {1: 0, 2: 1}
    assert loaded.head_row_by_target == {3: 0, 4: 1}


def test_two_sources_sharing_a_target_row_refused(tmp_path):
    """Mutant: the per-target index silently keeps the last source; the device would score one with the other's row."""
    meta = _meta(rows=(head_file.HeadRow(0, 2, 0), head_file.HeadRow(1, 2, 1)))
    with pytest.raises(ValueError, match="target map row 2"):
        head_file.load_learned_head(
            _write(tmp_path, meta=meta), layer_ids=[0, 1, 2, 3], targets=_targets({0: 2, 1: 2})
        )


def test_serving_head_holds_static_buffers(tmp_path):
    """Mutant: positions or threshold built with the wrong dtype or length; the kernels read them raw."""
    loaded = head_file.load_learned_head(
        _write(tmp_path), layer_ids=[0, 1, 2, 3], targets=_targets({0: 2, 1: 3})
    )
    serving = head_file.serving_head(loaded, device="cpu", top_k=3)
    assert serving.positions.tolist() == list(range(8)) and serving.positions.dtype == torch.int64
    assert serving.threshold.dtype == torch.float32 and serving.threshold.shape == (1,)
    assert serving.src_w.shape == (8, 3) and serving.weights.rows == 2
    text = head_file.describe(loaded)
    assert "learned L+2 scorer active" in text and "ab" * 32 in text and "0.1301608383655548" in text


LAUNCH_ENV = {
    "SGLANG_DSV41_RAM_PREFETCH": True,
    "SGLANG_DSV41_CPU_EXPERTS": True,
    "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": True,
    "SGLANG_DSV41_RAM_PREFETCH_SCORER": "gpu",
    "SGLANG_DSV41_RAM_PREFETCH_HORIZON": 2,
    "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": "",
    "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": 1,
}  # fmt: skip


def _gate(path: str, **changes) -> None:
    with ExitStack() as stack:
        stack.enter_context(envs.SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD.override(path))
        for name, value in {**LAUNCH_ENV, **changes}.items():
            stack.enter_context(getattr(envs, name).override(value))
        _check_ram_prefetch()


def test_gate_accepts_a_complete_learned_head_launch(tmp_path):
    """Mutant: a new condition refuses the reference launch (T3 arm: gpu scorer, horizon 2, no floors, per-layer 1)."""
    _gate(_write(tmp_path))


def test_gate_refuses_the_head_without_the_prefetch(tmp_path):
    """Mutant: the prefetch-off early return skips the head check; the variable would pass unused."""
    with pytest.raises(ValueError, match="LEARNED_HEAD needs SGLANG_DSV41_RAM_PREFETCH=1"):
        _gate(
            _write(tmp_path),
            SGLANG_DSV41_RAM_PREFETCH=False,
            SGLANG_DSV41_RAM_PREFETCH_HORIZON=1,
            SGLANG_DSV41_RAM_PREFETCH_SCORER="cpu",
        )


def test_gate_refuses_the_head_with_the_cpu_scorer(tmp_path):
    """Mutant: the scorer condition is dropped; the head would be set but the CPU scorer would run."""
    with pytest.raises(ValueError, match="LEARNED_HEAD needs SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu"):
        _gate(_write(tmp_path), SGLANG_DSV41_RAM_PREFETCH_SCORER="cpu")


def test_gate_refuses_the_head_at_horizon_one(tmp_path):
    """Mutant: the horizon condition is dropped; an L+2 head would score L+1 targets."""
    with pytest.raises(ValueError, match="LEARNED_HEAD needs SGLANG_DSV41_RAM_PREFETCH_HORIZON=2"):
        _gate(_write(tmp_path), SGLANG_DSV41_RAM_PREFETCH_HORIZON=1)


def test_gate_refuses_the_head_with_a_floors_table(tmp_path):
    """Mutant: the floors condition is dropped; a floors file set alongside the head would pass."""
    floors = tmp_path / "floors.json"
    floors.write_text('{"min_margin": {"1": 0.25}}')
    with pytest.raises(ValueError, match="LEARNED_HEAD needs SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"):
        _gate(_write(tmp_path), SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS=str(floors))


def test_gate_refuses_the_head_with_per_layer_above_one(tmp_path):
    """Mutant: the per-layer condition is dropped; the head publishes one candidate per layer only."""
    with pytest.raises(ValueError, match="LEARNED_HEAD needs SGLANG_DSV41_RAM_PREFETCH_PER_LAYER=1"):
        _gate(_write(tmp_path), SGLANG_DSV41_RAM_PREFETCH_PER_LAYER=2)


def test_gate_refuses_an_unreadable_head_file(tmp_path):
    """Mutant: the gate stops reading the file; a bad path would only fail at service start."""
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD: .*nope.safetensors"):
        _gate(str(tmp_path / "nope.safetensors"))


def test_gate_refuses_a_head_file_of_another_format_version(tmp_path):
    """Mutant: the version check is dropped from the gate's read; a future file would be accepted."""
    path = _write(tmp_path, meta=_meta(format_version=head_file.HEAD_FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD: .*format_version"):
        _gate(path)


def test_gate_ignores_an_empty_head_variable():
    """Mutant: '' is treated as a path; the off state (unset or empty) would refuse launches."""
    _gate("", SGLANG_DSV41_RAM_PREFETCH_SCORER="cpu", SGLANG_DSV41_RAM_PREFETCH_HORIZON=1)
