"""Trace replay through the expert framework's hot-cache and pinned-tier policies (CPU)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41"))

from tier_sim import (  # noqa: E402
    live_summary,
    ms_per_token,
    ram_rows_per_layer,
    seed_from_trace,
    select_hot,
    simulate,
)


def _call(forward, layer, experts, tokens=1, counts=None, vram_miss=0, ram_miss=0):
    return {
        "forward": forward,
        "layer": layer,
        "tokens": tokens,
        "experts": experts,
        "counts": counts or [1] * len(experts),
        "vram_miss": vram_miss,
        "ram_miss": ram_miss,
    }


def test_cold_decode_misses_vram_and_then_hits_ram():
    calls = [_call(1, 0, [1, 2]), _call(2, 0, [1, 2])]
    out = simulate(calls, num_layers=1, num_experts=4, vram_slots=0, ram_slots=4)
    assert out["decode_tokens"] == 2
    assert (out["vram_misses"], out["ram_misses"]) == (4, 2)
    assert out["G"] == pytest.approx(2.0) and out["f"] == pytest.approx(0.5)


def test_ram_rows_are_split_per_layer():
    # Two layers share 2 RAM rows as one each: layer 0 cannot keep expert 1 past expert 2.
    # A single global LRU of 2 rows would have kept it (2 RAM misses, not 3).
    calls = [_call(1, 0, [1]), _call(2, 0, [2]), _call(3, 0, [1])]
    out = simulate(calls, num_layers=2, num_experts=4, vram_slots=0, ram_slots=2)
    assert (out["vram_misses"], out["ram_misses"]) == (3, 3)
    assert ram_rows_per_layer(5, 2, 4) == [3, 2]
    assert ram_rows_per_layer(100, 2, 4) == [4, 4]


def test_ram_keeps_the_rows_vram_holds():
    # Expert 0 is VRAM-resident from startup, so it is RAM's oldest row, never
    # touched. When expert 2 needs a row, an inclusive tier evicts 1 instead of 0,
    # and 1 then misses again: 3 RAM misses. A tier without is_pinned would evict 0
    # and hit 1 on the third call (2 misses).
    calls = [_call(1, 0, [1]), _call(2, 0, [2]), _call(3, 0, [1])]
    out = simulate(
        calls, num_layers=1, num_experts=4, vram_slots=1, ram_slots=2,
        max_gather_rows=1, dynamic=False,
    )
    assert out["hot_slots"] == 1
    assert (out["vram_misses"], out["ram_misses"]) == (3, 3)


def test_hot_allocation_follows_the_manager_order_and_clamp():
    assert select_hot(None, 3, 2, 4) == {0: [0, 1], 1: [0]}
    seed = np.array([[0, 5, 0, 0], [7, 0, 0, 0]], dtype=float)
    assert select_hot(seed, 2, 2, 4) == {0: [1], 1: [0]}
    # A clamped layer's budget flows to the next candidates.
    assert select_hot(None, 4, 2, 4, limits=[1, 4]) == {0: [0], 1: [0, 1, 2]}


def test_the_inclusive_clamp_moves_slots_to_other_layers():
    seed = np.zeros((2, 8))
    seed[0, :3] = [9, 8, 7]
    seed[1, 0] = 1
    out = simulate(
        [_call(1, 0, [0])], num_layers=2, num_experts=8, vram_slots=3, ram_slots=8,
        seed=seed, max_gather_rows=2, dynamic=False,
    )
    # 4 RAM rows per layer - 2 gather rows = 2 hot slots at most: layer 0 gets
    # experts 0 and 1, and the third slot goes to layer 1's expert 0.
    assert (out["hot_slots"], out["clamped_layers"]) == (3, 1)
    assert out["vram_misses"] == 0


def test_seeded_static_residency_serves_hits():
    seed = np.array([[0, 9, 1, 5]], dtype=float)
    calls = [_call(1, 0, [1, 2]), _call(2, 0, [3])]
    out = simulate(
        calls, num_layers=1, num_experts=4, vram_slots=2, ram_slots=4, seed=seed,
        max_gather_rows=2, dynamic=False,
    )
    # VRAM holds 1 and 3; RAM starts with them too (startup promotions go through it).
    assert (out["vram_misses"], out["ram_misses"]) == (1, 1)
    assert out["G"] == pytest.approx(0.5) and out["f"] == pytest.approx(1.0)
    assert out["boundaries"] == 0


def test_decode_boundaries_promote_a_frequent_expert():
    # One VRAM slot starts on expert 0. Expert 3 scores 1.0 after forward 1, not
    # above 0 + benefit_ratio 1.0; after forward 2 it scores 0.95 + 1 and replaces 0.
    calls = [_call(1, 0, [3]), _call(2, 0, [3]), _call(3, 0, [3])]
    out = simulate(
        calls, num_layers=1, num_experts=4, vram_slots=1, ram_slots=4, max_gather_rows=3,
        update_decode_forwards=1, min_residence_forwards=0, benefit_ratio=1.0,
    )
    assert out["boundaries"] == 3
    assert (out["vram_misses"], out["ram_misses"]) == (2, 1)
    assert (out["decode_promotion_rows"], out["decode_promotion_ram_misses"]) == (1, 0)
    assert out["G"] == pytest.approx(2 / 3) and out["f"] == pytest.approx(0.5)
    static = simulate(
        calls, num_layers=1, num_experts=4, vram_slots=1, ram_slots=4, max_gather_rows=3,
        dynamic=False,
    )
    assert static["vram_misses"] == 3 and static["decode_promotion_rows"] == 0


def test_a_call_touches_its_ram_hits_before_admitting_its_misses():
    # ExpertPinnedHostCache.gather_rows looks up the whole chunk first, so call
    # [0, 1] touches expert 1 and then evicts 2 for expert 0: 3 RAM misses. Walking
    # the experts one by one would evict 1 for 0 and miss on 1 again (4).
    calls = [_call(1, 0, [1]), _call(2, 0, [2]), _call(3, 0, [0, 1])]
    out = simulate(calls, num_layers=1, num_experts=4, vram_slots=0, ram_slots=2)
    assert (out["vram_misses"], out["ram_misses"]) == (4, 3)


def test_a_promotion_admits_a_missing_row_but_never_touches_a_held_one():
    # One VRAM slot. Expert 1 is promoted at forward 2 while RAM already holds it
    # (order 0, 1, 2), and 3 replaces it at forward 5. ensure_rows leaves 1's place
    # in the LRU order alone, so once it is unpinned it is the oldest evictable row:
    # forward 7 (expert 5) evicts 1 (after 0 went at forward 6), and forward 8 misses
    # on 1. A promotion that touched it would have evicted 2 instead.
    calls = [
        _call(1, 0, [1, 2]),
        _call(2, 0, [1, 2]),
        _call(3, 0, [3]),
        _call(4, 0, [3]),
        _call(5, 0, [3]),
        _call(6, 0, [4]),
        _call(7, 0, [5]),
        _call(8, 0, [1]),
    ]
    out = simulate(
        calls, num_layers=1, num_experts=6, vram_slots=1, ram_slots=4, max_gather_rows=2,
        update_decode_forwards=1, min_residence_forwards=0, benefit_ratio=1.0,
    )
    assert (out["decode_promotion_rows"], out["decode_promotion_ram_misses"]) == (2, 0)
    assert (out["vram_misses"], out["ram_misses"]) == (10, 6)


def test_a_ram_hit_refreshes_its_lru_position():
    # RAM holds 3 rows. Call [1, 4] hits 1 (touch: order 2, 3, 1 -> then 4 evicts 2),
    # so expert 5 evicts 3 and the last call hits 1: 5 RAM misses. Without the hit's
    # touch, 5 would evict 1 and the last call would miss again (6).
    calls = [_call(1, 0, [1]), _call(2, 0, [2]), _call(3, 0, [3]), _call(4, 0, [1, 4]), _call(5, 0, [5]), _call(6, 0, [1])]
    out = simulate(calls, num_layers=1, num_experts=8, vram_slots=0, ram_slots=3)
    assert (out["vram_misses"], out["ram_misses"]) == (7, 5)


def test_prefill_counts_apart_from_decode():
    calls = [_call(1, 0, [5, 6], tokens=4), _call(2, 0, [5])]
    out = simulate(calls, num_layers=1, num_experts=8, vram_slots=0, ram_slots=4)
    assert (out["prefill_vram_misses"], out["prefill_ram_misses"]) == (2, 2)
    assert (out["decode_tokens"], out["vram_misses"], out["ram_misses"]) == (1, 1, 0)


def test_the_no_prefill_admission_arm_keeps_decode_rows():
    # Decode brings expert 1 into a 2-row tier; a prefill of 2 and 3 flushes it
    # (framework), unless prefill misses do not enter RAM (simulation-only arm).
    calls = [_call(1, 0, [1]), _call(2, 0, [2, 3], tokens=4), _call(3, 0, [1])]
    kwargs = dict(num_layers=1, num_experts=8, vram_slots=0, ram_slots=2)
    framework = simulate(calls, **kwargs)
    arm = simulate(calls, prefill_admits=False, **kwargs)
    assert (framework["ram_misses"], arm["ram_misses"]) == (2, 1)
    assert framework["prefill_ram_misses"] == arm["prefill_ram_misses"] == 2


def test_live_summary_uses_the_traced_misses():
    calls = [
        _call(1, 0, [1, 2], tokens=4, vram_miss=2, ram_miss=2),
        _call(2, 0, [1], vram_miss=2, ram_miss=2),
        _call(3, 0, [1], vram_miss=2, ram_miss=1),
    ]
    live = live_summary(calls, warmup=1)
    assert live["decode_tokens"] == 2
    assert (live["G"], live["f"], live["f_after_warmup"]) == (2.0, 0.75, 0.5)


def test_ms_per_token_adds_decode_promotions():
    row = {"G": 2.0, "f": 0.5, "decode_tokens": 4, "decode_promotion_rows": 2, "decode_promotion_ram_misses": 1}
    assert ms_per_token(row, 7.0) == pytest.approx(2 * 1.11 + 0.5 * 2 * 7.0 + (2 * 1.11 + 7.0) / 4)


def test_seed_counts_decode_routes_with_multiplicity():
    calls = [_call(1, 0, [1, 2], counts=[3, 1]), _call(2, 1, [2]), _call(3, 0, [0], tokens=9)]
    assert np.array_equal(seed_from_trace(calls, 2, 3), np.array([[0, 3, 1], [0, 0, 1]]))


def test_cli_writes_rows_live_numbers_and_a_hot_seed(tmp_path):
    trace = tmp_path / "trace.jsonl"
    calls = [_call(1, 0, [1, 2], vram_miss=2, ram_miss=2), _call(2, 0, [1], vram_miss=1)]
    trace.write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    seed = tmp_path / "seed.json"
    script = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41", "tier_sim.py")
    result = subprocess.run(
        [sys.executable, script, str(trace), "--vram-slots", "0", "1", "--ram-slots", "4",
         "--update-decode-forwards", "0", "--num-layers", "1", "--num-experts", "4",
         "--max-gather-rows", "2", "--emit-seed", str(seed)],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert [(r["vram_slots"], r["prefill_admits"]) for r in report["rows"]] == [
        (0, True), (0, False), (1, True), (1, False)
    ]
    assert {"ms_per_token_nvme2", "ms_per_token_x4", "G", "f", "policy"} <= set(report["rows"][0])
    assert report["live"]["G"] == pytest.approx(1.5) and report["live"]["f"] == pytest.approx(2 / 3)
    assert json.loads(seed.read_text()) == {"count": [[0, 2, 1, 0]]}


def test_a_trace_missing_miss_counts_on_a_later_line_reports_no_live_numbers(tmp_path):
    trace = tmp_path / "trace.jsonl"
    late = _call(2, 0, [1])
    del late["vram_miss"], late["ram_miss"]
    calls = [_call(1, 0, [1], vram_miss=1, ram_miss=1), late]
    trace.write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    script = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41", "tier_sim.py")
    result = subprocess.run(
        [sys.executable, script, str(trace), "--vram-slots", "0", "--ram-slots", "4",
         "--update-decode-forwards", "0", "--num-layers", "1", "--num-experts", "4",
         "--max-gather-rows", "2"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["live"] is None


def test_graph_steps_mix_with_eager_prefill_lines():
    calls = [
        {"forward": 1, "layer": 0, "tokens": 256, "vram_miss": 9, "ram_miss": 4, "experts": [1], "counts": [9]},
        {"forward": 1, "layer": 1, "tokens": 256, "vram_miss": 9, "ram_miss": 4, "experts": [2], "counts": [9]},
        {"forward": 2, "layer": -1, "tokens": 1, "kind": "graph_step", "vram_miss": 6, "ram_miss": 3, "experts": [], "counts": []},
        {"forward": 3, "layer": -1, "tokens": 1, "kind": "graph_step", "vram_miss": 4, "ram_miss": 1, "experts": [], "counts": []},
    ]
    live = live_summary(calls, warmup=0)
    assert live["decode_tokens"] == 2 and live["G"] == 5.0 and live["f"] == 0.4


def test_direct_replay_evicts_a_resident_the_closing_window_did_not_route_before_a_lower_score():
    """GpuResidencyUpdater._rank_victims ranks residents routed in the closing window last, whatever
    their score: expert 0 (decayed score ~2.9) goes before expert 1 (score 1, routed one forward ago).
    A plain decayed-LFU would evict 1."""
    from tier_sim import DirectInsertReplay

    sim = DirectInsertReplay({0: [0, 1]}, {0: 2}, 8, miss_rows=1)
    for routes in ([0], [0], [0], [1]):
        assert sim.graph_forward({0: routes}) == {0: 0}
    assert sim.graph_forward({0: [5]}) == {0: 1}
    assert sim.resident(0) == {1, 5}
    assert sim.scores[0, 0] > sim.scores[0, 1] > 0


def test_direct_replay_victims_skip_the_slots_the_forward_reads():
    """The shortlist is ranked before the forward's routes are known; a listed slot the forward hits is
    dropped (gather_destinations), so the miss takes the next one."""
    from tier_sim import DirectInsertReplay

    sim = DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=2)
    sim.graph_forward({0: [0]})  # expert 0 scores; 2 then 1 rank lowest (ties evict the higher id)
    assert sim.graph_forward({0: [2, 6]}) == {0: 1}
    assert sim.resident(0) == {0, 2, 6}


def test_direct_replay_ranks_a_short_prefills_routes_before_it_scores_them():
    """GpuResidencyUpdater._apply re-ranks the shortlist on every graph forward; only the score update is
    gated. After a prefill below the 256-token boundary no boundary is due, yet the first decode's victims
    already rank the prefill's experts as routed: expert 2 survives and 1 goes. Found replaying smoke6,
    where the old replay kept the startup shortlist there and drifted from the logged hot sets."""
    from tier_sim import DirectInsertReplay

    sim = DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=1)
    sim.eager_forward(20, {0: ([2], [1])}, "extend")
    assert sim.graph_forward({0: [5]}) == {0: 1}
    assert sim.resident(0) == {0, 2, 5}
    assert sim.scores.sum() == 0  # the prefill's count is scored at the next forward's boundary


def test_direct_replay_eager_prefill_inserts_nothing():
    from tier_sim import DirectInsertReplay

    sim = DirectInsertReplay({0: [0, 1], 1: [0, 1]}, {0: 2, 1: 2}, 8, miss_rows=1)
    assert sim.eager_forward(40, {0: ([1, 7], [30, 10]), 1: ([3], [40])}) == {0: 1, 1: 1}
    assert sim.resident(0) == {0, 1} and sim.resident(1) == {0, 1}
    assert sim.scores.sum() == 0  # a 40-token prefill is below the 256-token boundary: counts wait
    sim.eager_forward(300, {0: ([7], [300])})
    assert sim.scores[0, 7] == 310 and sim.resident(0) == {0, 1}


def test_direct_allocation_gives_every_layer_its_floor_then_lowest_layers_the_rest():
    from tier_sim import direct_hot_allocation

    hot = direct_hot_allocation(1128, list(range(40)), 384, floor=12)
    assert [len(hot[layer]) for layer in range(40)] == [29] * 8 + [28] * 32
    assert all(hot[layer] == list(range(len(hot[layer]))) for layer in range(40))
    with pytest.raises(ValueError, match="floor"):
        direct_hot_allocation(100, list(range(40)), 384, floor=12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
