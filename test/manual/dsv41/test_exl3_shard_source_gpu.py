"""The EXL3 shard row source under the framework's real pinned and hot tiers (GPU, Window C)."""

import json

import pytest
import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_EXPERTS = 6
LAYER = 1


def _reference(ckpt):
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    def get(expert, w, kind):
        name = f"layers.{LAYER}.ffn.experts.{expert}.{w}.{kind}"
        with safe_open(str(ckpt / weight_map[name]), "pt") as f:
            return f.get_tensor(name)

    rows = {}
    for prefix, linears in (("w13", ("w1", "w3")), ("w2", ("w2",))):
        for kind in ("trellis", "suh", "svh"):
            rows[f"{prefix}_{kind}"] = torch.stack(
                [torch.stack([get(e, w, kind) for w in linears]) for e in range(NUM_EXPERTS)]
            )
    return rows


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _streamer(ckpt):
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

    write_fake_exl3(str(ckpt), num_layers=2, num_experts=NUM_EXPERTS)
    layer = torch.nn.Module()
    layer.layer_id = LAYER
    fmt = Exl3ExpertFormat(build_exl3_expert_layout(str(ckpt)), LAYER, direct=False)
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
        return ExpertStreamer(layer, fmt.names, layer_id=LAYER, format=fmt)


def test_gather_through_hot_pinned_and_shards(tmp_path):
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

    streamer = _streamer(tmp_path)
    reference = _reference(tmp_path)
    hot = ExpertHotCache(streamer, 2)
    hot.reassign([0, 1])
    pinned = ExpertPinnedHostCache(streamer, 2)
    pinned.ensure_rows(torch.tensor([2], device="cuda"))
    ids = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    chunks = list(streamer.iter_gather_experts(ids))
    assert len(chunks) == 1
    chunk, row_of_source, rows = chunks[0]
    for i, expert in enumerate(chunk.tolist()):
        for name, tensor in reference.items():
            assert _same(rows[name][row_of_source[i]].cpu(), tensor[expert]), (name, expert)
    stats = streamer.last_gather_stats
    assert stats.hot_hit_rows == 1
    assert (stats.pinned_host_hit_rows, stats.pinned_host_miss_rows) == (1, 1)
    assert stats.host_read_rows == 1


def test_chunked_gather_of_more_rows_than_the_pinned_tier(tmp_path):
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

    streamer = _streamer(tmp_path)
    reference = _reference(tmp_path)
    ExpertPinnedHostCache(streamer, 2)
    ids = torch.arange(NUM_EXPERTS, device="cuda", dtype=torch.int32)
    seen = []
    for chunk, row_of_source, rows in streamer.iter_gather_experts(ids, chunk_rows=4):
        for i, expert in enumerate(chunk.tolist()):
            seen.append(expert)
            for name, tensor in reference.items():
                assert _same(rows[name][row_of_source[i]].cpu(), tensor[expert]), (name, expert)
    assert seen == list(range(NUM_EXPERTS))
    assert streamer.last_gather_stats.miss_rows == NUM_EXPERTS


def test_the_managers_build_inclusive_tiers(tmp_path):
    """The production startup path: both tier managers over two EXL3 layers."""
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import (
        ExpertPinnedHostCacheManager,
        ExpertStreamer,
    )

    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=NUM_EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    layers = []
    for layer_id in (0, 1):
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        fmt = Exl3ExpertFormat(layout, layer_id, direct=False)
        # A 6-expert fake: shrink the gather reserve so the clamp leaves hot slots.
        fmt.max_gather_rows = 2
        with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
            layer._nvfp4_expert_streamer = ExpertStreamer(
                layer, fmt.names, layer_id=layer_id, format=fmt
            )
        layers.append(layer)
    model = torch.nn.Sequential(*layers)
    row_bytes = layout.row_bytes - 12
    pinned = ExpertPinnedHostCacheManager.from_model(model, budget_bytes=2 * 5 * row_bytes)
    assert sorted(cache.capacity for cache in pinned.caches.values()) == [5, 5]
    assert all(cache.is_pinned is not None for cache in pinned.caches.values())
    ExpertHotCacheManager.from_model(
        model,
        budget_bytes=2 * 4 * row_bytes,
        seed_path=None,
        dynamic=False,
        update_prefill_tokens=16,
        min_residence_forwards=0,
        benefit_ratio=1.0,
    )
    for layer in layers:
        streamer = layer._nvfp4_expert_streamer
        resident = streamer.hot_cache.resident_experts()
        # Clamped to 5 pinned rows - max_gather_rows 2; the unused budget goes nowhere.
        assert len(resident) == 3
        assert resident <= set(streamer.pinned_host_cache.slot_to_expert)
        ids = torch.arange(NUM_EXPERTS, device="cuda", dtype=torch.int32)
        for _chunk, _row_of_source, _rows in streamer.iter_gather_experts(ids):
            pass
        assert resident <= set(streamer.pinned_host_cache.slot_to_expert)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
