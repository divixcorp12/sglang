"""DSpark verify windows over a one-token decode trace, and the union curve (CPU only)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

from verify_union import (  # noqa: E402
    baseline,
    gate,
    overflow_rate,
    project,
    shrink_hot,
    union_stats,
    verify_ms,
    window_forwards,
)

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _decode(seq, routes, rid="a"):
    return {
        "kind": "graph", "seq": seq, "phase": "decode", "tokens": 1, "rids": [rid], "forward_pass_id": seq,
        "routes": {0: routes}, "misses": {0: len(routes)}, "hot": {0: [0]},
    }


def _eager(forward):
    return {"kind": "eager", "forward": forward, "phase": "extend", "tokens": 256, "rids": ["a"],
            "forward_pass_id": None, "counts": {0: ([1], [1])}, "misses": {0: 1}}


def _loaded(forwards):
    return {"layer_ids": [0], "hot_capacity": {0: 8}, "hot_layer_ids": [0], "forwards": forwards}


def test_windows_union_routes_in_first_appearance_order_and_advance_by_stride():
    loaded = _loaded([_decode(i, r) for i, r in enumerate([[1, 2], [2, 3], [4], [5, 1], [6]])])
    out = window_forwards(loaded, width=3, stride=2)["forwards"]
    assert [f["routes"][0] for f in out] == [[1, 2, 3, 4], [4, 5, 1, 6], [6]]
    assert [f["tokens"] for f in out] == [3, 3, 1]
    assert [f["seq"] for f in out] == [0, 2, 4]
    assert [f["misses"][0] for f in out] == [5, 4, 1]


def test_a_window_never_spans_a_request_or_an_eager_forward():
    forwards = [_decode(0, [1]), _decode(1, [2]), _eager(7), _decode(2, [3]), _decode(3, [4], rid="b")]
    out = window_forwards(_loaded(forwards), width=4, stride=4)["forwards"]
    assert [(f["kind"], f.get("routes", {}).get(0)) for f in out] == [
        ("graph", [1, 2]), ("eager", None), ("graph", [3]), ("graph", [4]),
    ]


def test_width_one_is_the_trace_itself():
    loaded = _loaded([_decode(i, [i, i + 1]) for i in range(4)])
    assert window_forwards(loaded, width=1, stride=1)["forwards"] == loaded["forwards"]


@pytest.mark.parametrize("width,stride", [(0, 1), (3, 0), (3, 4)])
def test_a_stride_outside_one_to_width_is_refused(width, stride):
    with pytest.raises(ValueError, match="stride"):
        window_forwards(_loaded([_decode(0, [1])]), width=width, stride=stride)


def test_union_stats_count_distinct_experts_per_verify_and_layer():
    loaded = _loaded([_decode(i, r) for i, r in enumerate([[1, 2, 2], [3], [4, 5, 6, 7]])])
    stats = union_stats(loaded)
    assert stats["verifies"] == 3 and stats["max"] == 4
    assert stats["mean"] == pytest.approx(7 / 3) and stats["per_layer_mean"] == [pytest.approx(7 / 3)]


def test_shrink_hot_takes_slots_off_every_layer_and_refuses_an_empty_layer():
    loaded = _loaded([_decode(0, [1])])
    assert shrink_hot(loaded, 3)["hot_capacity"] == {0: 5}
    assert shrink_hot(loaded, 0) is loaded
    with pytest.raises(ValueError, match="slots"):
        shrink_hot(loaded, 8)


def test_verify_ms_adds_link_rows_nvme_waits_and_gpu_once_per_verify():
    n, m = np.array([[1, 2], [0, 0]]), np.array([[1, 0], [0, 1]])
    out = verify_ms(n, m, c_link=1.0, nvme_ms=1.5, gpu_ms=14.0)
    assert out.tolist() == [14.0 + 4 * 1.0 + 1 * 1.5, 14.0 + 1 * 1.0 + 1 * 1.5]


def test_overflow_rate_is_the_share_of_verify_layers_needing_more_lanes_than_the_record_has():
    n, m = np.array([[8, 3], [2, 2]]), np.array([[1, 0], [0, 0]])
    assert overflow_rate(n, m, lanes=8) == pytest.approx(1 / 4)


def test_project_costs_windows_and_flags_a_width_the_vram_cannot_hold():
    loaded = _loaded([_decode(i, [i % 5, 10 + i % 7]) for i in range(12)])
    row = project(loaded, width=4, stride=2, lanes=4, draft_slots=1, ram_rows=64, num_experts=64,
                  c_link=1.0, nvme_ms=1.5, gpu_ms=14.0)
    assert row["verifies"] == 6 and row["capacity_ok"] is False  # 8 - 1 = 7 slots < 2 * 4
    assert row["tok_s_no_draft"] == pytest.approx(2 * 1000.0 / row["verify_ms"])
    assert row["union"]["max"] <= 8


def test_baseline_is_the_slot_map_hits_arm():
    loaded = _loaded([_decode(i, [i % 5, 10 + i % 7]) for i in range(12)])
    base = baseline(loaded, ram_rows=64, num_experts=64, c_link=1.0, nvme_ms=1.5, gpu_ms=14.0, handoff=0.02,
                    c_cpu=0.63, split_c_cpu=0.52)
    assert base["tok_s"] == pytest.approx(1000.0 / base["ms_per_token"]) and base["ms_per_token"] >= 14.0


def _row(lanes, verify, overflow, ok=True, width=6, stride=3, draft_slots=4):
    return {"width": width, "stride": stride, "draft_slots": draft_slots, "lanes": lanes, "verify_ms": verify,
            "overflow": overflow, "capacity_ok": ok}


def test_gate_takes_the_cheapest_admissible_lane_count_and_needs_draft_room():
    base = {"ms_per_token": 76.0}
    rows = [_row(8, 150.0, 0.30), _row(16, 160.0, 0.01), _row(24, 158.0, 0.0, ok=False), _row(32, 170.0, 0.0)]
    out = gate(rows, base, width=6, stride=3, draft_slots=4, gain=1.10, draft_ms_floor=5.0, max_overflow=0.02)
    assert out["lanes"] == 16
    assert out["draft_budget_ms"] == pytest.approx(3 * 76.0 / 1.10 - 160.0)
    assert out["go"] is (out["draft_budget_ms"] >= 5.0)


def test_gate_says_no_when_no_lane_count_is_admissible():
    out = gate([_row(8, 100.0, 0.5)], {"ms_per_token": 76.0}, width=6, stride=3, draft_slots=4, gain=1.10,
               draft_ms_floor=5.0, max_overflow=0.02)
    assert out["go"] is False and "overflow" in out["why"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
