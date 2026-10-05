"""DSpark verify windows over a one-token decode trace, and the union curve (CPU only)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

from verify_union import shrink_hot, union_stats, window_forwards  # noqa: E402

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
