"""hot_cache_calibration: the measured sizing, the retire plan, and the calibration forward's no-state guards.

* Calibration's derived values from given measurements: non-torch growth, lowest free, the margin (allocator gap or
  repeat growth), and the slots to retire (trim-only: none when the lowest free covers the margin).
* plan_keep: retires one slot at a time from the fullest layer, keeps the floor, refuses an impossible retire.
* The guards: inside calibration_context, ExpertStreamer.record_routes does not reach the residency policy and
  EngramFileTable.lookup returns zero rows without touching its store; outside it both behave as before.
* The off path: maybe_calibrate_hot_cache returns None without reading the runner; require_hot_cache_headroom does
  nothing without a calibration and raises when free memory is below the margin.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import hot_cache_calibration as hc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

M = 1 << 20


def _cal(free=4000, reserved=1000, p1=2500, a1=2100, r1=2500, f1=None, p2=2500, slot=13):
    f1 = 4000 - 1500 - 20 if f1 is None else f1  # 1500 reserve grown + 20 outside torch
    runs = (hc.RunPeak(p1 * M, a1 * M, r1 * M, f1 * M), hc.RunPeak(p2 * M, a1 * M, r1 * M, f1 * M))
    return hc.Calibration(4096, free * M, reserved * M, runs, slot * M)


def test_lowest_free_counts_the_reserve_peak_and_non_torch_growth():
    c = _cal()
    assert c.nontorch_growth == 20 * M
    assert c.lowest_free == (4000 - 1500 - 20) * M


def test_margin_is_the_allocator_gap_or_the_repeat_growth():
    assert _cal().margin == 400 * M  # P1 - A1
    assert _cal(p2=3100).margin == 600 * M  # run 2 grew 600 past run 1's reserve


def test_trim_only_retires_just_enough_slots():
    assert _cal().slots_to_retire == 0  # 2480 MiB free over a 400 MiB margin
    tight = _cal(free=1900, f1=1900 - 1500 - 20)  # lowest free 380 < margin 400
    assert tight.lowest_free == 380 * M and tight.slots_to_retire == 2  # ceil(20 / 13)


def test_the_capture_estimate_is_part_of_what_the_prefill_must_leave():
    c = _cal()
    with_capture = hc.Calibration(c.tokens, c.free_before, c.reserved_before, c.runs, c.slot_bytes, 2200 * M)
    assert with_capture.required == (400 + 2200) * M  # margin + capture
    assert with_capture.slots_to_retire == 10  # ceil((2600 - 2480) / 13)
    assert with_capture.summary()["capture_estimate_mib"] == 2200


def test_a_verify_sized_route_set_goes_to_vram_residents_inside_calibration():
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])  # 6 routes <= 36 graph rows
    streamer = SimpleNamespace(
        hot_cache=SimpleNamespace(slot_to_expert=[20, 21, 22, 23, 24, 25, 26, -1]),
        graph_gather_rows=36,
        pinned_host_cache=SimpleNamespace(slot_to_expert=list(range(100, 140))),
    )
    with hc.calibration_context():
        verify = hc.calibration_routes(ids, streamer)
        prefill = hc.calibration_routes(torch.arange(4096 * 6).reshape(4096, 6) % 200, streamer)
    assert set(verify.reshape(-1).tolist()) <= set(range(20, 27))  # hot residents: every row a hit
    assert set(prefill.reshape(-1).tolist()) <= set(range(100, 140))  # RAM residents: no admissions


def test_forget_graph_path_state_drops_the_cached_fused_moe():
    from sglang.srt.layers.moe.hot_cache_dummy_extend import forget_graph_path_state

    layers = [torch.nn.Module() for _ in range(3)]
    layers[0]._exl3_fused_moe, layers[2]._exl3_fused_moe = object(), object()
    model = torch.nn.ModuleList(layers)
    assert forget_graph_path_state(model) == 2
    assert not any(hasattr(m, "_exl3_fused_moe") for m in layers)


def test_slot_delta_grows_by_the_room_above_required_and_trims_below_it():
    assert _cal().slot_delta == (2480 - 400) // 13  # 160 slots of room
    tight = _cal(free=1900, f1=1900 - 1500 - 20)
    assert tight.slot_delta == -2


def test_calibration_shapes_follow_the_warmup_largest_then_repeat_the_chunk():
    assert hc.calibration_shapes(4096) == (2048, 2049, 4096, 4096)
    assert hc.calibration_shapes(2) == (2, 2)


def test_the_highest_run_sets_lowest_free_and_the_last_repeat_the_margin():
    runs = (
        hc.RunPeak(2000 * M, 1900 * M, 2000 * M, 3000 * M),  # 2048
        hc.RunPeak(2100 * M, 1900 * M, 2100 * M, 2900 * M),  # 2049: fragmentation
        hc.RunPeak(2500 * M, 2300 * M, 2600 * M, 2400 * M),  # 4096
        hc.RunPeak(2650 * M, 2300 * M, 2650 * M, 2350 * M),  # 4096 again: grew 50 past 2600
    )
    c = hc.Calibration(4096, 4000 * M, 1000 * M, runs, 13 * M, 0, 900 * M)
    assert c.peak_run is runs[3]
    assert c.nontorch_growth == (1650 - 1650) * M  # (4000 - 2350) - (2650 - 1000)
    assert c.lowest_free == (4000 - 1650) * M
    assert c.margin == 350 * M  # the highest run's gap; the repeat grew 50


def test_compare_uses_transients_when_both_entries_are_known():
    runs = (hc.RunPeak(2500 * M, 2100 * M, 2500 * M, 0), hc.RunPeak(2500 * M, 2100 * M, 2500 * M, 0))
    c = hc.Calibration(4096, 4000 * M, 1000 * M, runs, 13 * M, 0, 900 * M)
    r = hc.compare_with_real(c, 3000 * M, 2600 * M, real_allocated_in=1700 * M)
    assert r["calibration_transient_mib"] == 1200 and r["real_transient_mib"] == 1300
    assert r["delta_transient_mib"] == 100 and r["delta_reserved_mib"] == 100


def test_plan_capacity_grows_the_emptiest_layers_up_to_their_bound():
    # 20 -> 21 -> 22, then the 22s tie and the lower layer id takes the third.
    assert hc.plan_capacity({0: 20, 1: 22}, 3, 16, {0: 30, 1: 30}) == {0: 23, 1: 22}
    assert hc.plan_capacity({0: 20, 1: 22}, 10, 16, {0: 21, 1: 23}) == {0: 21, 1: 23}  # bounds cap it
    assert hc.plan_capacity({0: 20, 1: 22}, -2, 16, {0: 30, 1: 30}) == {0: 20, 1: 20}


def test_plan_keep_takes_from_the_fullest_layer_and_keeps_the_floor():
    # 27 -> 26 first, then the 26s from the lowest layer id.
    assert hc.plan_keep({0: 26, 1: 27, 2: 26}, 3, 16) == {0: 25, 1: 25, 2: 26}
    assert hc.plan_keep({0: 20, 1: 20}, 0, 16) == {0: 20, 1: 20}
    with pytest.raises(ValueError, match="only 8 sit above"):
        hc.plan_keep({0: 20, 1: 20}, 9, 16)


def test_record_routes_skips_the_policy_only_inside_calibration():
    from sglang.srt.layers.moe.expert_stream import ExpertStreamer

    seen = []
    streamer = SimpleNamespace(residency_policy=SimpleNamespace(record_routes=lambda ids: seen.append(ids.numel())))
    ids = torch.tensor([[1, 2], [3, 4]])
    with hc.calibration_context():
        assert hc.calibrating()
        ExpertStreamer.record_routes(streamer, ids)
    assert not hc.calibrating() and seen == []
    ExpertStreamer.record_routes(streamer, ids)
    assert seen == [4]


def test_engram_lookup_returns_zero_rows_without_touching_the_store_inside_calibration():
    from sglang.srt.layers.engram_file_table import EngramFileTable

    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"store touched: {name}")

    table = SimpleNamespace(dim=8, _native_store=Untouchable(), cache=Untouchable())
    ids = torch.zeros((3, 2), dtype=torch.long)
    with hc.calibration_context():
        out = EngramFileTable.lookup(table, ids)
    assert out.shape == (3, 2, 8) and out.dtype == torch.bfloat16 and not out.any()
    with pytest.raises(AssertionError, match="store touched"):
        EngramFileTable.lookup(table, ids)


def test_off_returns_none_without_reading_the_runner():
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"runner read: {name}")

    with envs.SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP.override(False):
        assert hc.maybe_calibrate_hot_cache(Untouchable()) is None
    hc.require_hot_cache_headroom("after capture", None)  # no calibration: no check


def test_resident_routes_keep_the_shape_and_distinct_count_and_use_only_residents():
    real = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [0, 1, 2, 3, 4, 5]])  # 12 distinct
    residents = [-1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, -1]
    out = hc.resident_routes(real, residents)
    assert out.shape == real.shape and out.dtype == real.dtype
    assert set(out.reshape(-1).tolist()) <= set(range(40, 54)) and torch.unique(out).numel() == 12
    assert all(len(set(row)) == 6 for row in out.tolist())  # each token's top-k distinct
    with pytest.raises(RuntimeError, match="cannot fill a top-6"):
        hc.resident_routes(real, [1, 2, 3, -1])


def test_calibration_routes_change_nothing_outside_the_context():
    ids = torch.tensor([[3, 4]])
    tier = SimpleNamespace(slot_to_expert=[7, 8, 9])
    assert hc.calibration_routes(ids, SimpleNamespace(pinned_host_cache=tier)) is ids
    with hc.calibration_context():
        assert set(hc.calibration_routes(ids, SimpleNamespace(pinned_host_cache=tier)).tolist()[0]) <= {7, 8, 9}


def test_compare_flags_a_real_peak_above_the_calibration_by_more_than_the_margin():
    c = _cal()  # run 1 reserved peak 2500, allocated 2100, margin 400
    ok = hc.compare_with_real(c, 2200 * M, 2800 * M)
    assert ok["delta_reserved_mib"] == 300 and not ok["exceeds"]
    bad = hc.compare_with_real(c, 2200 * M, 3000 * M)
    assert bad["exceeds"] and bad["delta_allocated_mib"] == 100


def test_the_watch_measures_only_the_first_full_size_extend_then_removes_itself(monkeypatch):
    calls = []

    class Runner:
        def forward(self, fb):
            calls.append(fb.extend_num_tokens)
            return "out"

    runner = Runner()
    result = hc.CalibrationResult(_cal(), {0: 20})
    for name, value in (("synchronize", None), ("reset_peak_memory_stats", None)):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: value)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 2200 * M)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 1000 * M)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 2600 * M)
    hc.watch_first_full_prefill(runner, result)
    extend = lambda n: SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True), extend_num_tokens=n)
    assert runner.forward(extend(16)) == "out" and "forward" in vars(runner)  # not the calibrated size
    assert runner.forward(extend(4096)) == "out" and "forward" not in vars(runner)  # measured, then removed
    assert result.extra["first_real_prefill"]["delta_reserved_mib"] == 100
    assert calls == [16, 4096]


def test_headroom_check_raises_below_the_margin(monkeypatch):
    result = hc.CalibrationResult(_cal(), {0: 20})  # run-1 reserve peak 2500, margin 400
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 2500 * M)  # reserve still held
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (500 * M, 32000 * M))
    hc.require_hot_cache_headroom("after capture", result)  # 500 >= 400
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (300 * M, 32000 * M))
    with pytest.raises(hc.HotCacheHeadroomShort):
        hc.require_hot_cache_headroom("after capture", result)


def test_kv_pin_is_refused_when_the_pool_came_out_smaller():
    hc.check_kv_pin(pinned=381184, sized=381184)
    hc.check_kv_pin(pinned=None, sized=10)  # no pin, nothing to check
    with pytest.raises(hc.KvPinMissed, match="372480"):
        hc.check_kv_pin(pinned=381184, sized=372480)


def test_headroom_projection_charges_back_a_released_reserve(monkeypatch):
    # The smoke's case: capture released the calibration's cached reserve, so free memory alone looked roomy.
    result = hc.CalibrationResult(_cal(), {0: 20})
    assert hc.projected_free(result.calibration, free=2000 * M, reserved=1300 * M) == 800 * M  # 1200 to re-grow
    assert hc.projected_free(result.calibration, free=2000 * M, reserved=2600 * M) == 2000 * M
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 1000 * M)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1800 * M, 32000 * M))
    with pytest.raises(hc.HotCacheHeadroomShort):  # 1800 - 1500 = 300 < 400
        hc.require_hot_cache_headroom("after capture", result)
