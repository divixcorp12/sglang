"""EXL3 expert rows read from the shards and split into the six streamed names."""

import os
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.exl3_shard_row_source import (
    Exl3ShardRowSource,
    shared_row_reader,
)
from sglang.srt.layers.moe.expert_row_source import ExpertRowSource, HostSlotLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PARTS = {"w13": ("w1", "w3"), "w2": ("w2",)}


def _checkpoint(tmp_path, num_layers=2, num_experts=5):
    # 3 experts per shard: a layer's rows span shards, and some rows end a shard.
    write_fake_exl3(str(tmp_path), num_layers=num_layers, num_experts=num_experts)
    return build_exl3_expert_layout(str(tmp_path))


def _reference(tmp_path, layout, layer):
    """{name: [experts, parts, ...]} read with safetensors, independent of the source."""
    import json

    with open(tmp_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    out = {}
    for prefix, linears in PARTS.items():
        for kind in ("trellis", "suh", "svh"):
            rows = []
            for expert in range(layout.num_experts):
                parts = []
                for w in linears:
                    name = f"layers.{layer}.ffn.experts.{expert}.{w}.{kind}"
                    with safe_open(str(tmp_path / weight_map[name]), "pt") as f:
                        parts.append(f.get_tensor(name))
                rows.append(torch.stack(parts))
            out[f"{prefix}_{kind}"] = torch.stack(rows)
    return out


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _source(layout, layer=1, bounce_rows=2):
    fmt = Exl3ExpertFormat(layout, layer, direct=False)
    reader = Exl3RowReader(layout, direct=False)
    return fmt, Exl3ShardRowSource(reader, layer, fmt.segment_map(), bounce_rows=bounce_rows)


def _zeros(fmt, rows, names=None):
    return {
        spec.name: torch.zeros((rows,) + spec.row_shape, dtype=spec.dtype)
        for spec in fmt.tensor_specs(None)
        if names is None or spec.name in names
    }


def test_reads_every_name_into_destination_rows(tmp_path):
    layout = _checkpoint(tmp_path)
    reference = _reference(tmp_path, layout, 1)
    fmt, source = _source(layout)
    destinations = _zeros(fmt, 3)
    experts, slots = [4, 0, 3], [2, 0, 1]
    stats = source.read(torch.tensor(experts), destinations, torch.tensor(slots))
    for name, destination in destinations.items():
        for expert, slot in zip(experts, slots):
            assert _same(destination[slot], reference[name][expert]), (name, expert)
    assert stats.rows == 3
    assert stats.split_bytes == 3 * (layout.row_bytes - 12)
    def read_bytes(e):
        record = layout.records[(1, e)]
        offset, length, _ = record.aligned_read(4096)
        return min(length, os.path.getsize(record.path) - offset)

    assert stats.file_bytes == sum(read_bytes(e) for e in experts)


def test_a_read_naming_two_names_fills_only_those(tmp_path):
    layout = _checkpoint(tmp_path)
    reference = _reference(tmp_path, layout, 1)
    fmt, source = _source(layout)
    destinations = _zeros(fmt, 2, names=("w13_suh", "w2_trellis"))
    stats = source.read(torch.tensor([1, 2]), destinations)
    for name in ("w13_suh", "w2_trellis"):
        assert _same(destinations[name], reference[name][[1, 2]]), name
    per_row = sum(destinations[n][0].numel() * destinations[n].element_size() for n in destinations)
    assert stats.split_bytes == 2 * per_row


def test_is_an_expert_row_source(tmp_path):
    layout = _checkpoint(tmp_path)
    _fmt, source = _source(layout)
    assert isinstance(source, ExpertRowSource)
    assert source.covers("w13_trellis") and not source.covers("w13_mul1")
    assert source.host_layouts == frozenset({HostSlotLayout.PER_NAME})
    assert source.preferred_batch_rows == 2
    assert source.num_experts == 5
    assert source.requires_page_aligned_destinations is False
    assert source.register_destinations([torch.zeros(4)]) == 0
    assert layout.row_bytes <= source.file_bytes_per_expert <= layout.row_bytes + 2 * 4096
    assert source.bounce.data_ptr() % 4096 == 0


def test_rejects_bad_requests(tmp_path):
    layout = _checkpoint(tmp_path)
    fmt, source = _source(layout)
    with pytest.raises(ValueError, match="does not cover"):
        source.read(torch.tensor([0]), {"w13_mul1": torch.zeros(1, 2, dtype=torch.int32)})
    with pytest.raises(ValueError, match="outside"):
        source.read(torch.tensor([5]), _zeros(fmt, 1))
    with pytest.raises(ValueError, match="expected"):
        source.read(torch.tensor([0]), {"w2_suh": torch.zeros(1, 3, dtype=torch.float16)})
    with pytest.raises(ValueError, match="match in length"):
        source.read(torch.tensor([0, 1]), _zeros(fmt, 2), torch.tensor([0]))
    assert source.read(torch.tensor([], dtype=torch.long), _zeros(fmt, 1)).rows == 0


def test_submit_returns_a_completed_ticket(tmp_path):
    layout = _checkpoint(tmp_path)
    reference = _reference(tmp_path, layout, 1)
    fmt, source = _source(layout)
    destinations = _zeros(fmt, 1)
    ticket = source.submit(torch.tensor([3]), destinations)
    assert ticket.done()
    assert ticket.wait().rows == 1
    assert _same(destinations["w2_svh"][0], reference["w2_svh"][3])


def test_every_layer_shares_the_bounce_ring(tmp_path):
    layout = _checkpoint(tmp_path)
    fmt0 = Exl3ExpertFormat(layout, 0, direct=False)
    fmt1 = Exl3ExpertFormat(layout, 1, direct=False)
    a = fmt0.default_row_source(None, fmt0.tensor_specs(None), "shards")
    b = fmt1.default_row_source(None, fmt1.tensor_specs(None), "shards")
    assert a.bounce.data_ptr() == b.bounce.data_ptr()
    assert a.reader is b.reader


def test_format_selects_the_shard_source(tmp_path):
    layout = _checkpoint(tmp_path)
    fmt = Exl3ExpertFormat(layout, 1, direct=False)
    for kind in ("auto", "shards"):
        source = fmt.default_row_source(None, fmt.tensor_specs(None), kind)
        assert isinstance(source, Exl3ShardRowSource) and source.layer_id == 1
    assert fmt.file_source_bytes_per_expert(None, source) == source.file_bytes_per_expert
    assert fmt.file_source_bytes_per_expert(None, None) is None
    for kind in ("tensor", "bogus"):
        with pytest.raises(ValueError, match=f"no row source kind '{kind}'"):
            fmt.default_row_source(None, fmt.tensor_specs(None), kind)


def test_reader_mode_follows_the_file_reader_knob(tmp_path):
    layout = _checkpoint(tmp_path)
    fmt = Exl3ExpertFormat(layout, 0)
    specs = fmt.tensor_specs(None)
    with envs.SGLANG_MOE_EXPERT_FILE_READER.override("mmap"):
        with pytest.raises(ValueError, match="SGLANG_MOE_EXPERT_FILE_READER=uring_direct"):
            fmt.default_row_source(None, specs, "shards")
    with envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring_direct"):
        assert fmt.default_row_source(None, specs, "shards").reader is shared_row_reader(layout, True)
    with envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"):
        assert fmt.default_row_source(None, specs, "shards").reader is shared_row_reader(layout, False)


def test_shared_row_reader_is_keyed_by_source_root(tmp_path):
    layout = _checkpoint(tmp_path)
    plain = shared_row_reader(layout, True)
    rooted = shared_row_reader(layout, True, source_root=str(tmp_path))
    assert plain.source_root is None
    assert rooted.source_root == str(tmp_path)
    assert rooted is not plain
    assert shared_row_reader(layout, True, source_root=str(tmp_path)) is rooted
    assert shared_row_reader(layout, True) is plain


def test_pinned_tier_options_protect_hot_cache_experts(tmp_path):
    layout = _checkpoint(tmp_path)
    fmt = Exl3ExpertFormat(layout, 1, direct=False)
    assert fmt.inclusive_pinned_tier is True
    layer = torch.nn.Module()
    is_pinned = fmt.pinned_tier_options(layer)["is_pinned"]
    assert not is_pinned(3)  # no streamer attached yet
    layer._nvfp4_expert_streamer = SimpleNamespace(hot_cache=None)
    assert not is_pinned(3)  # the pinned tier is built before the hot cache
    layer._nvfp4_expert_streamer.hot_cache = SimpleNamespace(slot_to_expert=[3, -1])
    assert is_pinned(3) and not is_pinned(4)
    layer._nvfp4_expert_streamer.hot_cache.slot_to_expert[0] = 4  # residency moved
    assert is_pinned(4) and not is_pinned(3)


def test_streamer_reads_shards_through_a_cpu_pinned_tier(tmp_path):
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer

    layout = _checkpoint(tmp_path)
    reference = _reference(tmp_path, layout, 1)
    layer = torch.nn.Module()
    layer.layer_id = 1
    fmt = Exl3ExpertFormat(layout, 1)
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"):
        streamer = ExpertStreamer(layer, fmt.names, layer_id=1, format=fmt)
    assert isinstance(streamer.row_source, Exl3ShardRowSource)
    assert streamer.has_spec_only_tensors
    assert streamer.num_experts == 5
    assert streamer.bytes_per_expert == layout.row_bytes - 12
    assert streamer.file_source_bytes_per_expert == streamer.row_source.file_bytes_per_expert
    cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
    outputs = _zeros(fmt, 3)
    ids = torch.tensor([3, 0, 4])
    result = cache.gather_rows(ids, outputs)
    assert (result.hit_rows, result.miss_rows) == (0, 3)
    for name, output in outputs.items():
        assert _same(output, reference[name][ids]), name


def test_an_inclusive_cpu_pinned_tier_keeps_hot_experts(tmp_path):
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer

    layout = _checkpoint(tmp_path)
    reference = _reference(tmp_path, layout, 1)
    layer = torch.nn.Module()
    layer.layer_id = 1
    fmt = Exl3ExpertFormat(layout, 1, direct=False)
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
        streamer = ExpertStreamer(layer, fmt.names, layer_id=1, format=fmt)
    layer._nvfp4_expert_streamer = streamer
    cache = ExpertPinnedHostCache(streamer, 3, device="cpu", **fmt.pinned_tier_options(layer))
    cache.gather_rows(torch.tensor([0, 1, 2]), _zeros(fmt, 3))
    # Expert 0 is now "in VRAM": the tier must keep it while 3 and 4 come in.
    streamer.hot_cache = SimpleNamespace(slot_to_expert=[0])
    outputs = _zeros(fmt, 2)
    cache.gather_rows(torch.tensor([3, 4]), outputs)
    assert 0 in cache.slot_to_expert
    assert {3, 4} <= set(cache.slot_to_expert)
    for name, output in outputs.items():
        assert _same(output, reference[name][[3, 4]]), name


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
