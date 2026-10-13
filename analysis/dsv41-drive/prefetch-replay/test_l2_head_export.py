"""CPU tests for l2_head_export: the head file holds the kernel-precision fold of the checkpoint and its provenance."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l2_head_export  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import b1_head  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402

from sglang.srt.layers.moe import ram_prefetch_learned_head as head_file  # noqa: E402
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

E = 384
RANK = 128
LAYERS = (19, 20, 21, 22)
STRATA = ("a", "b")
ROWS = (l2_fast_torch.FoldRow(source_layer=19, target_layer=21, stratum="b"),
        l2_fast_torch.FoldRow(source_layer=20, target_layer=22, stratum="a"))
SHA = "ab" * 32
THRESHOLD = 0.1301608383655548
SOURCE = {"path": "f109/result.json", "split": "calib", "scorer": "L2_T3", "budget": "0.75"}


def _model(*, t3: bool) -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=64, emb_dim=4, num_experts=E)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(mean=torch.randn(r1_cost.X_DIM, generator=g) * 0.5,
                   components=torch.randn((RANK, r1_cost.X_DIM), generator=g) / r1_cost.X_DIM ** 0.5,
                   explained_variance=torch.ones(RANK))
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=LAYERS, strata=STRATA))
    model = r1_models.R1Model(r1_models.R1Config(head=cfg, position=t3, aggregator=t3), head=head)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model.eval()


def _export(model, path) -> head_file.HeadMeta:
    return l2_head_export.export_head(model, rows=ROWS, checkpoint_sha256=SHA, threshold=THRESHOLD,
                                      threshold_source=SOURCE, out=str(path))


def test_file_holds_kernel_precision_fold_and_provenance(tmp_path):
    """A mutant that stores fp32 matrices, drops a tensor or loses the sha/threshold/source/rows turns this red."""
    model = _model(t3=True)
    _export(model, tmp_path / "head.safetensors")
    tensors, meta = head_file.read_head_file(str(tmp_path / "head.safetensors"))
    want = l2_fast_kernels.to_kernel_weights(l2_fast_torch.fold(model, rows=ROWS), device="cpu")
    for name in head_file.TENSOR_NAMES:
        expected = getattr(want, name)
        assert tensors[name].dtype == expected.dtype
        assert torch.equal(tensors[name], expected), name
    assert meta.checkpoint_sha256 == SHA
    assert meta.threshold == THRESHOLD
    assert meta.threshold_source == SOURCE
    assert meta.horizon == 2
    assert [(r.source_layer, r.target_layer, r.stratum) for r in meta.rows] == [(19, 21, 1), (20, 22, 0)]


def test_t0_model_is_refused_by_the_reader(tmp_path):
    """A mutant that exports a model without an aggregator as servable turns this red."""
    _export(_model(t3=False), tmp_path / "t0.safetensors")
    with pytest.raises(ValueError, match="aggregator"):
        head_file.read_head_file(str(tmp_path / "t0.safetensors"))


def test_infeasible_threshold_is_refused(tmp_path):
    """A mutant that writes an inf/None threshold into the file turns this red."""
    with pytest.raises(ValueError, match="threshold"):
        l2_head_export.export_head(_model(t3=True), rows=ROWS, checkpoint_sha256=SHA, threshold=float("inf"),
                                   threshold_source=SOURCE, out=str(tmp_path / "x.safetensors"))
    assert not (tmp_path / "x.safetensors").exists()


def test_existing_out_is_refused_and_left_untouched(tmp_path, capsys):
    """A mutant that drops the existence check and rewrites the file turns this red."""
    out = tmp_path / "head.safetensors"
    out.write_bytes(b"existing head")
    rc = l2_head_export.main(["/nonexistent/run", "--dataset", "d", "--f109", "/nonexistent/f109", "--out", str(out)])
    assert rc == 2
    assert str(out) in capsys.readouterr().err
    with pytest.raises(FileExistsError):
        _export(_model(t3=True), out)
    assert out.read_bytes() == b"existing head"


def test_guard_accepts_the_fixed_serving_path():
    """A mutant that guards --out only against l2_fast_common.OUT_ROOT refuses both_cpu_ab.T3_HEAD and turns this red."""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("both_cpu_ab", HERE.parent / "dspark" / "both_cpu_ab.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    assert os.path.dirname(ab.T3_HEAD) == l2_head_export.HEAD_ROOT
    assert l2_head_export.guard_out(ab.T3_HEAD, run_dir="/nonexistent/run") == os.path.realpath(ab.T3_HEAD)


def test_out_inside_protected_run_dir_exits_2(capsys):
    """A mutant that skips guard_out would write into the F109 directory instead of exiting 2."""
    out = "/mnt/nvme1/nvfp4-work/f109-l2-cap1-20261013/head.safetensors"
    rc = l2_head_export.main(["/nonexistent/run", "--dataset", "d", "--f109", "/nonexistent/f109", "--out", out])
    assert rc == 2
    assert "refused" in capsys.readouterr().err
