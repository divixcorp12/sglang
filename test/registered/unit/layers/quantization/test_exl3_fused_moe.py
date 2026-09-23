"""Pointer tables and route tables of the fused EXL3 MoE over hot-cache slots (CPU)."""

import pytest
import torch

from sglang.srt.layers.quantization.exl3_fused_moe import route_tables, slot_pointer_tables
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


def test_pointer_tables_address_each_slots_part():
    tensors = {name: torch.zeros((5, 2 if name.startswith("w13") else 1, 16), dtype=torch.int16) for name in NAMES}
    tables = slot_pointer_tables(tensors, 5)
    assert sorted(tables) == sorted(f"{p}_{k}" for p in ("gate", "up", "down") for k in ("trellis", "suh", "svh"))
    for slot in range(5):
        assert tables["gate_trellis"][slot].item() == tensors["w13_trellis"][slot, 0].data_ptr()
        assert tables["up_svh"][slot].item() == tensors["w13_svh"][slot, 1].data_ptr()
        assert tables["down_suh"][slot].item() == tensors["w2_suh"][slot, 0].data_ptr()
    assert all(t.dtype == torch.int64 and t.shape == (5,) for t in tables.values())


def test_route_tables_sort_routes_by_slot_and_scale_by_keep():
    remap = torch.tensor([4, 1, 3])
    count = torch.zeros(6, dtype=torch.long)
    weights = torch.tensor([0.5, 0.25, 0.125])
    inv_order, weight_sorted, det = route_tables(remap, count, torch.ones(3, dtype=torch.long), weights, torch.tensor([1.0]))
    assert count.tolist() == [0, 1, 0, 1, 1, 0]
    order = torch.argsort(remap)
    assert torch.equal(inv_order[order], torch.arange(3))
    assert weight_sorted.dtype == torch.float16 and weight_sorted.tolist() == [0.25, 0.125, 0.5]
    assert det[0].tolist() == [0, 0, 1, 1, 2, 3] and det[2].tolist() == [0, 1, 0, 1, 1, 0]
    _, dropped, _ = route_tables(remap, count, torch.ones(3, dtype=torch.long), weights, torch.tensor([0.0]))
    assert dropped.tolist() == [0.0, 0.0, 0.0]
    assert count.tolist() == [0] * 6  # a dropped layer runs no expert at all


class _GatherReached(Exception):
    """The stub streamer's gather ran: _apply_graph got past its setup checks."""


def _stub_streamer(backend, graph_gather_rows=6, scratch_rows=6, pull_row=False):
    from types import SimpleNamespace

    def gather(topk_ids):
        raise _GatherReached

    cache = SimpleNamespace(capacity=3, scratch_rows=scratch_rows, reserves_prefetch_pull_row=pull_row)
    return SimpleNamespace(
        row_backend=backend, gather=gather, hot_cache=cache, graph_gather_rows=graph_gather_rows
    )


def _apply_graph(layer, streamer):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    return Exl3MoEMethod._apply_graph(
        layer, streamer, torch.zeros((1, 8)), torch.ones((1, 6)), torch.zeros((1, 6), dtype=torch.long), 10.0
    )


def _plain_pinned_tier_backend():
    from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend

    return PinnedTierRowBackend({0: None}, torch.full((6,), -1, dtype=torch.int64), 6)


def test_apply_graph_refuses_a_plain_pinned_tier_backend():
    # Without option C a RAM miss silently drops the layer and nothing fail-stops.
    layer = torch.nn.Module()
    with pytest.raises(RuntimeError, match="option C"):
        _apply_graph(layer, _stub_streamer(_plain_pinned_tier_backend()))
    layer._exl3_allow_p3_only = True  # the explicit test-only configuration
    with pytest.raises(_GatherReached):
        _apply_graph(layer, _stub_streamer(_plain_pinned_tier_backend()))


def test_apply_graph_accepts_an_option_c_backend():
    from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend

    class OptionCBackend(PinnedTierRowBackend):  # stands in for Task 14's Exl3RamMissRowBackend
        pass

    backend = OptionCBackend({0: None}, torch.full((6,), -1, dtype=torch.int64), 6)
    with pytest.raises(_GatherReached):
        _apply_graph(torch.nn.Module(), _stub_streamer(backend))


@pytest.mark.parametrize(
    "streamer_changes, match",
    [
        ({"graph_gather_rows": 4}, "top_k"),
        ({"scratch_rows": 5}, "scratch"),
        ({"pull_row": True}, "prefetch"),
    ],
)
def test_the_fused_moe_refuses_shapes_its_tables_do_not_cover(streamer_changes, match):
    from sglang.srt.layers.quantization.exl3_fused_moe import exl3_fused_moe_for

    layer = torch.nn.Module()
    layer.top_k = 6
    with pytest.raises(ValueError, match=match):
        exl3_fused_moe_for(layer, _stub_streamer(None, **streamer_changes))


def test_direct_fused_moe_covers_resident_slots_with_zero_scratch(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.quantization import exl3_fused_moe as module

    calls = []
    monkeypatch.setattr(module, "Exl3FusedMoE", lambda tensors, slots, **kw: calls.append(slots) or object())
    layer = torch.nn.Module()
    layer.top_k = 6
    backend = SimpleNamespace(name="exl3_ram_miss")
    streamer = _stub_streamer(backend, scratch_rows=0)
    streamer.hot_cache.capacity = 6
    streamer.hot_cache.device_residency = SimpleNamespace(insert_on_miss=2)
    streamer.hot_cache.reserves_prefetch_pull_row = False
    streamer.hot_cache.device = torch.device("cpu")
    streamer.hot_cache.tensors = {
        "w13_suh": torch.zeros((6, 2, 8)),
        "w2_suh": torch.zeros((6, 1, 8)),
    }
    assert module.exl3_fused_moe_for(layer, streamer) is layer._exl3_fused_moe
    assert calls == [6]
    streamer.hot_cache.capacity = 5
    del layer._exl3_fused_moe
    with pytest.raises(ValueError, match="DIRECT needs at least top_k"):
        module.exl3_fused_moe_for(layer, streamer)
    streamer.hot_cache.capacity = 6
    streamer.row_backend = SimpleNamespace(name="other")
    with pytest.raises(ValueError, match="scratch"):
        module.exl3_fused_moe_for(layer, streamer)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
