"""Pointer tables and route tables of the fused EXL3 MoE over hot-cache slots (CPU)."""

import pytest
import torch

from sglang.srt.layers.quantization.exl3.fused_moe import route_tables, slot_pointer_tables
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


def test_route_tables_rank_cross_token_duplicates_by_route_and_name_their_tokens():
    # Two tokens of top-3; slot 4 is routed by both.
    remap = torch.tensor([4, 1, 3, 2, 4, 0])
    count = torch.zeros(6, dtype=torch.long)
    token_sorted = torch.full((6,), -1, dtype=torch.long)
    weights = torch.tensor([0.5, 0.25, 0.125, 1.0, 2.0, 4.0])
    inv, ws, det = route_tables(
        remap, count, torch.ones(6, dtype=torch.long), weights, torch.tensor([1.0]), token_sorted=token_sorted, top_k=3
    )
    assert count.tolist() == [1, 1, 1, 1, 2, 0]
    assert inv.tolist() == [4, 1, 3, 2, 5, 0]  # slot order, route order within slot 4
    assert token_sorted.tolist() == [1, 0, 1, 0, 0, 1]
    assert ws.tolist() == [4.0, 0.25, 1.0, 0.125, 0.5, 2.0]
    assert det[0].tolist() == [0, 1, 2, 3, 4, 6] and det[2].tolist() == [1, 1, 1, 1, 1, 0]


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
    from sglang.srt.layers.quantization.exl3.fused_moe import exl3_fused_moe_for

    layer = torch.nn.Module()
    layer.top_k = 6
    with pytest.raises(ValueError, match=match):
        exl3_fused_moe_for(layer, _stub_streamer(None, **streamer_changes))


def test_direct_fused_moe_covers_resident_slots_with_zero_scratch(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.exl3 import fused_moe as module

    calls = []
    monkeypatch.setattr(module, "Exl3FusedMoE", lambda tensors, slots, **kw: calls.append(slots) or object())
    layer = torch.nn.Module()
    layer.top_k = 6
    backend = SimpleNamespace(name="exl3_ram_miss")
    streamer = _stub_streamer(backend, scratch_rows=0)
    streamer.hot_cache.capacity = 6
    streamer.hot_cache.device_residency = SimpleNamespace(insert_on_miss=True, insert_direct=True)
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


class _FakeExt:
    """exl3_ext() stand-in: records what the fused MoE hands exllamav3."""

    def __init__(self):
        self.moe, self.gather = [], []

    def exl3_moe_max_concurrency(self, device):
        return 2

    def exl3_moe(self, *args):
        self.moe.append(args)

    def exl3_moe_gather(self, *args):
        self.gather.append(args)


def _fused(monkeypatch, tokens, hidden=8, slots=10):
    from sglang.srt.layers.quantization.exl3 import fused_moe as module

    ext = _FakeExt()
    monkeypatch.setattr(module, "exl3_ext", lambda: ext)
    module._SHARED_TEMPS.clear()
    tensors = {n: torch.zeros((slots, 2 if n.startswith("w13") else 1, hidden), dtype=torch.int16) for n in NAMES}
    tensors["w13_trellis"] = torch.zeros((slots, 2, 16 * 3), dtype=torch.int16)
    tensors["w2_trellis"] = torch.zeros((slots, 1, 16 * 3), dtype=torch.int16)
    fused = module.Exl3FusedMoE(tensors, slots, hidden=hidden, inter=hidden, top_k=6, device="cpu", tokens=tokens)
    return fused, ext


def test_fused_moe_hands_exllamav3_m_tokens_with_their_token_index(monkeypatch):
    fused, ext = _fused(monkeypatch, tokens=6)
    remap = torch.tensor([0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8])  # token 1 shares slots 3-5 with token 0
    out = fused.run(torch.ones((2, 8)), torch.ones(12), remap, torch.ones(1), 10.0)
    args = ext.moe[-1]
    assert args[0].shape == (2, 8) and args[1].shape == (2, 8) and out.shape == (2, 8)
    assert args[2][:9].tolist() == [1, 1, 1, 2, 2, 2, 1, 1, 1]
    assert args[3].tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]  # token of each rank
    assert args[4].shape == (12,) and args[30].shape == (12, 8)
    assert args[29] == -1  # num_active: launch-sized for any count of active slots
    assert ext.gather[-1][2].shape == (12,)


def test_fused_moe_one_token_is_todays_launch(monkeypatch):
    fused, ext = _fused(monkeypatch, tokens=6)
    fused.run(torch.ones((1, 8)), torch.ones(6), torch.tensor([5, 1, 3, 0, 2, 4]), torch.ones(1), 10.0)
    args = ext.moe[-1]
    assert args[0].shape == (1, 8) and args[3].tolist() == [0] * 6 and args[29] == 6 and args[30].shape == (6, 8)


def test_fused_moe_drops_every_token_when_keep_is_zero(monkeypatch):
    fused, ext = _fused(monkeypatch, tokens=2)
    out = fused.run(torch.ones((2, 8)), torch.ones(12), torch.arange(12) % 9, torch.zeros(1), 10.0)
    assert ext.moe[-1][2].sum().item() == 0 and torch.equal(out, torch.zeros((2, 8)))


@pytest.mark.parametrize("m", [0, 3])
def test_fused_moe_refuses_more_tokens_than_its_buffers(monkeypatch, m):
    fused, _ = _fused(monkeypatch, tokens=2)
    with pytest.raises(ValueError, match="1-2 tokens"):
        fused.run(torch.ones((m, 8)), torch.ones(6 * m), torch.zeros(6 * m, dtype=torch.long), torch.ones(1), 10.0)


@pytest.mark.parametrize("tokens", [0, 17])
def test_fused_moe_refuses_more_tokens_than_its_row_tile(monkeypatch, tokens):
    # Past 16, a slot shared by every token is skipped by exllamav3 and its stale scratch is summed: no error.
    with pytest.raises(ValueError, match="row tile"):
        _fused(monkeypatch, tokens=tokens)


def test_fused_moe_refuses_cpu_experts_for_several_tokens(monkeypatch):
    fused, _ = _fused(monkeypatch, tokens=2)
    with pytest.raises(RuntimeError, match="one token"):
        fused.run(
            torch.ones((2, 8)), torch.ones(12), torch.zeros(12, dtype=torch.long), torch.ones(1), 10.0,
            cpu=(None, None, 16, 0),
        )


def test_fused_moe_for_sizes_tokens_from_the_gather_rows(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.exl3 import fused_moe as module

    calls = []
    monkeypatch.setattr(module, "Exl3FusedMoE", lambda tensors, slots, **kw: calls.append(kw) or object())
    layer = torch.nn.Module()
    layer.top_k = 6
    streamer = _stub_streamer(SimpleNamespace(name="other"), graph_gather_rows=36, scratch_rows=36)
    streamer.hot_cache.device = torch.device("cpu")
    streamer.hot_cache.tensors = {"w13_suh": torch.zeros((39, 2, 8)), "w2_suh": torch.zeros((39, 1, 8))}
    module.exl3_fused_moe_for(layer, streamer)
    assert calls[-1]["tokens"] == 6 and calls[-1]["top_k"] == 6
    del layer._exl3_fused_moe
    streamer.graph_gather_rows = 6 * 17
    streamer.hot_cache.scratch_rows = 6 * 17
    with pytest.raises(ValueError, match="row tile"):
        module.exl3_fused_moe_for(layer, streamer)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
