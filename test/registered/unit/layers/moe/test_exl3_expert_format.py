"""The EXL3 expert format's row schema: six streamed names and the segment map."""

import contextlib
import dataclasses
import os
import shutil

import pytest
import torch

from sglang.srt.environ import envs

from sglang.srt.layers.moe.exl3_expert_format import (
    EXL3_MAX_GATHER_ROWS,
    EXL3_STREAMED_NAMES,
    Exl3ExpertFormat,
)
from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertLayout,
    Exl3ExpertRecord,
    Exl3TensorSpan,
    build_exl3_expert_layout,
)
from sglang.srt.layers.moe.exl3_mirror_row_source import Exl3MirrorRowSource
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import expert_spec, write_fake_exl3

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layout(tmp_path, num_layers=2, num_experts=4):
    write_fake_exl3(str(tmp_path), num_layers=num_layers, num_experts=num_experts)
    return build_exl3_expert_layout(str(tmp_path))


def _synthetic_layout(hidden, inter):
    """A layout with the real export's tensor order and sizes, and no files."""
    spans, cursor = [], 0
    for suffix, dtype, shape, nbytes in expert_spec(hidden, inter):
        spans.append(Exl3TensorSpan(suffix, cursor, nbytes, dtype, tuple(shape)))
        cursor += nbytes
    record = Exl3ExpertRecord(0, 0, "unused.safetensors", 0, cursor)
    return Exl3ExpertLayout(tuple(spans), cursor, {(0, 0): record}, 1, 1)


def test_specs_are_the_six_streamed_names(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 1)
    specs = fmt.tensor_specs(None)
    assert tuple(spec.name for spec in specs) == EXL3_STREAMED_NAMES == fmt.names
    shapes = {spec.name: (spec.row_shape, spec.dtype) for spec in specs}
    assert shapes == {
        "w13_trellis": ((2, 8, 8, 48), torch.int16),
        "w13_suh": ((2, 128), torch.float16),
        "w13_svh": ((2, 128), torch.float16),
        "w2_trellis": ((1, 8, 8, 48), torch.int16),
        "w2_suh": ((1, 128), torch.float16),
        "w2_svh": ((1, 128), torch.float16),
    }
    assert all(spec.residence == "host" for spec in specs)
    # Every byte of the row except the three 4-byte mul1 scalars is streamed.
    assert sum(spec.row_bytes for spec in specs) == layout.row_bytes - 12


def test_segment_map_covers_every_streamed_byte_once(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 0)
    row_bytes = {spec.name: spec.row_bytes for spec in fmt.tensor_specs(None)}
    source = [0] * layout.row_bytes
    target = {name: [0] * nbytes for name, nbytes in row_bytes.items()}
    for segment in fmt.segment_map():
        for i in range(segment.nbytes):
            source[segment.src_offset + i] += 1
            target[segment.name][segment.dst_offset + i] += 1
    mul1 = {span.rel_offset + i for span in layout.tensors if span.name.endswith("mul1") for i in range(4)}
    assert [i for i, n in enumerate(source) if n != 1] == sorted(mul1)
    assert all(n == 1 for counts in target.values() for n in counts)


def test_w1_and_w3_are_parts_zero_and_one(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 0)
    by_offset = {span.rel_offset: span.name for span in layout.tensors}
    parts = {by_offset[s.src_offset]: (s.name, s.part) for s in fmt.segment_map()}
    for kind in ("trellis", "suh", "svh"):
        assert parts[f"w1.{kind}"] == (f"w13_{kind}", 0)
        assert parts[f"w3.{kind}"] == (f"w13_{kind}", 1)
        assert parts[f"w2.{kind}"] == (f"w2_{kind}", 0)
    assert len(fmt.segment_map()) == 9


def test_real_export_row_sizes():
    fmt = Exl3ExpertFormat(_synthetic_layout(5120, 2304), 0)
    sizes = {spec.name: spec.row_bytes for spec in fmt.tensor_specs(None)}
    assert sizes == {
        "w13_trellis": 8_847_360,
        "w13_suh": 20_480,
        "w13_svh": 9_216,
        "w2_trellis": 4_423_680,
        "w2_suh": 4_608,
        "w2_svh": 10_240,
    }
    assert sum(sizes.values()) == 13_315_584
    assert all(size % 16 == 0 for size in sizes.values())


def test_format_flags_and_sources(tmp_path):
    layout = _layout(tmp_path)
    fmt = Exl3ExpertFormat(layout, 1)
    assert fmt.key == "exl3"
    assert fmt.supports_graph_gather is False
    assert fmt.supports_host_arena is False
    assert fmt.max_gather_rows == EXL3_MAX_GATHER_ROWS == 64
    assert fmt.num_experts(None) == 4
    assert fmt.source(None, "w13_trellis") is None


def test_rejects_a_layer_outside_the_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="outside the checkpoint"):
        Exl3ExpertFormat(_layout(tmp_path), 2)


def test_rejects_an_unexpected_row(tmp_path):
    layout = _layout(tmp_path)
    broken = dataclasses.replace(layout, tensors=layout.tensors[:-1])
    with pytest.raises(ValueError, match="expected"):
        Exl3ExpertFormat(broken, 0)


def test_the_exl3_format_serves_graph_gathers_from_its_pinned_tier():
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.expert_format import graph_source_kind_of

    assert graph_source_kind_of(Exl3ExpertFormat) == "pinned_tier"


# --- Mirror row source selection (SGLANG_MOE_EXPERT_MIRROR_DIRS / _WEIGHTS) ----


def _mirrored(tmp_path, num_roots=2):
    """A fake checkpoint and ``num_roots`` byte-identical copies of it."""
    source = tmp_path / "ckpt"
    source.mkdir()
    write_fake_exl3(str(source), num_layers=2, num_experts=4)
    layout = build_exl3_expert_layout(str(source))
    roots = []
    for i in range(num_roots):
        root = tmp_path / f"drive_{i}" / "copy"
        shutil.copytree(source, root)
        roots.append(str(root))
    fmt = Exl3ExpertFormat(layout, 1, direct=False, source_root=str(source))
    return fmt, roots


def _row_source(fmt):
    return fmt.default_row_source(None, fmt.tensor_specs(None), "auto")


def _mirror_env(dirs, weights=""):
    stack = contextlib.ExitStack()
    stack.enter_context(envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.override(dirs))
    stack.enter_context(envs.SGLANG_MOE_EXPERT_MIRROR_WEIGHTS.override(weights))
    return stack


def test_mirror_dirs_select_the_mirror_source_with_equal_weights(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    with _mirror_env(os.pathsep.join(roots)):
        source = _row_source(fmt)
    assert isinstance(source, Exl3MirrorRowSource)
    assert source.roots == tuple(roots) and source.layer_id == 1
    assert isinstance(source.policy, StaticSplitPolicy)
    # Empty weights: every root gets an equal share of a row's pages.
    assert source.policy.plan(8 * PAGE_BYTES).part_bytes == (4 * PAGE_BYTES,) * 2
    assert source.reader.source_root == fmt.source_root


def test_mirror_weights_reach_the_split_policy(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    with _mirror_env(os.pathsep.join(roots), "3:1"):
        source = _row_source(fmt)
    assert source.policy.plan(8 * PAGE_BYTES).part_bytes == (
        6 * PAGE_BYTES,
        2 * PAGE_BYTES,
    )
    with _mirror_env(os.pathsep.join(roots), " 0.5 : 1.5 "):
        source = _row_source(fmt)
    assert source.policy.plan(8 * PAGE_BYTES).part_bytes == (
        2 * PAGE_BYTES,
        6 * PAGE_BYTES,
    )


def test_one_mirror_root_reads_everything_from_it(tmp_path):
    fmt, roots = _mirrored(tmp_path, num_roots=1)
    with _mirror_env(roots[0]):
        source = _row_source(fmt)
    assert isinstance(source, Exl3MirrorRowSource) and source.roots == (roots[0],)
    assert source.policy.plan(8 * PAGE_BYTES).part_bytes == (8 * PAGE_BYTES,)
    with _mirror_env(roots[0], "5"):
        assert _row_source(fmt).policy.plan(2 * PAGE_BYTES).part_bytes == (
            2 * PAGE_BYTES,
        )


def test_mirror_dirs_unset_keeps_the_shard_source(tmp_path):
    fmt, _ = _mirrored(tmp_path)
    for kind in ("auto", "shards"):
        source = fmt.default_row_source(None, fmt.tensor_specs(None), kind)
        assert type(source) is Exl3ShardRowSource
    # Unset is the default, and a source root is not needed to read one root.
    assert envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.get() == ""
    assert envs.SGLANG_MOE_EXPERT_MIRROR_WEIGHTS.get() == ""
    plain = Exl3ExpertFormat(fmt.layout, 1, direct=False)
    assert type(_row_source(plain)) is Exl3ShardRowSource


def test_mirror_dirs_with_a_tensor_source_kind_are_still_refused(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    with _mirror_env(os.pathsep.join(roots)):
        with pytest.raises(ValueError, match="no row source kind 'tensor'"):
            fmt.default_row_source(None, fmt.tensor_specs(None), "tensor")


def test_weights_and_roots_of_different_length_are_refused_naming_both(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    with _mirror_env(os.pathsep.join(roots), "1:1:1"):
        with pytest.raises(ValueError) as caught:
            _row_source(fmt)
    message = str(caught.value)
    assert "SGLANG_MOE_EXPERT_MIRROR_WEIGHTS lists 3 weights" in message
    assert "SGLANG_MOE_EXPERT_MIRROR_DIRS lists 2 roots" in message
    with _mirror_env(os.pathsep.join(roots), "1"):
        with pytest.raises(ValueError, match="lists 1 weights.*lists 2 roots"):
            _row_source(fmt)


@pytest.mark.parametrize(
    "weights", ["1:x", "1:", ":1", "-1:2", "nan:1", "inf:1", "0:0"]
)
def test_unusable_weights_are_refused_naming_the_knob(tmp_path, weights):
    fmt, roots = _mirrored(tmp_path)
    with _mirror_env(os.pathsep.join(roots), weights):
        with pytest.raises(ValueError, match="SGLANG_MOE_EXPERT_MIRROR_WEIGHTS"):
            _row_source(fmt)


def test_weights_without_roots_are_refused_rather_than_ignored(tmp_path):
    fmt, _ = _mirrored(tmp_path)
    with _mirror_env("", "1:1"):
        with pytest.raises(ValueError, match="MIRROR_WEIGHTS is set.*MIRROR_DIRS"):
            _row_source(fmt)


def test_a_root_that_is_not_a_directory_is_refused_naming_it(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    missing = str(tmp_path / "no_such_drive" / "copy")
    with _mirror_env(os.pathsep.join([roots[0], missing])):
        with pytest.raises(ValueError, match="not a readable directory") as caught:
            _row_source(fmt)
    assert missing in str(caught.value) and roots[0] not in str(caught.value)
    a_file = tmp_path / "a_file"
    a_file.write_text("x")
    with _mirror_env(os.pathsep.join([str(a_file), roots[1]])):
        with pytest.raises(ValueError, match="not a readable directory") as caught:
            _row_source(fmt)
    assert str(a_file) in str(caught.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_an_unreadable_root_directory_is_refused_naming_it(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    os.chmod(roots[1], 0)
    try:
        with _mirror_env(os.pathsep.join(roots)):
            with pytest.raises(ValueError, match="not a readable directory") as caught:
                _row_source(fmt)
    finally:
        os.chmod(roots[1], 0o755)
    assert roots[1] in str(caught.value)


@pytest.mark.parametrize("shape", ["{a}::{b}", "{a}:", ":{a}", ":"])
def test_an_empty_root_entry_is_refused(tmp_path, shape):
    fmt, roots = _mirrored(tmp_path)
    with _mirror_env(shape.format(a=roots[0], b=roots[1])):
        with pytest.raises(ValueError, match="empty entry"):
            _row_source(fmt)


def test_a_mirror_source_needs_the_source_root(tmp_path):
    fmt, roots = _mirrored(tmp_path)
    bare = Exl3ExpertFormat(fmt.layout, 1, direct=False)
    with _mirror_env(os.pathsep.join(roots)):
        with pytest.raises(ValueError, match="source_root"):
            _row_source(bare)


def test_mirror_roots_are_checked_against_the_source_at_selection(tmp_path):
    """The source's own size checks are not bypassed: a truncated copy is refused."""
    fmt, roots = _mirrored(tmp_path)
    victim = os.path.join(roots[1], "model-00002.safetensors")
    with open(victim, "r+b") as f:
        f.truncate(os.path.getsize(victim) - 1)
    with _mirror_env(os.pathsep.join(roots)):
        with pytest.raises(RuntimeError, match="incomplete or stale") as caught:
            _row_source(fmt)
    assert roots[1] in str(caught.value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
