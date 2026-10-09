"""Per layer record of a served run's job trace: forced misses, prefetch cover, the layer's wait (CPU)."""

import importlib.util
import json
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _module():
    spec = importlib.util.spec_from_file_location(
        "layer_misses", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "layer_misses.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ev(kind, ns, row, gen, group=0, b=0):
    return {"event": kind, "ns": ns, "row": row, "gen": gen, "seq": 0, "group": group, "a": 0, "b": b, "c": 0}


def _record(row, gen, misses, used=0, start=0, ms=1.0):
    """copy_submit per group (misses split over both), spec_use per prefetched miss, then gate_open."""
    out = [_ev("copy_submit", start + 5, row, gen, 0, misses - misses // 2),
           _ev("copy_submit", start, row, gen, 1, misses // 2)]
    out += [_ev("spec_use", start + 1, row, gen) for _ in range(used)]
    return out + [_ev("gate_open", start + int(ms * 1e6), row, gen)]


def _write(path, events, dropped=0):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write(json.dumps({"dropped": dropped}) + "\n")


def test_a_record_sums_both_groups_misses_and_counts_its_swap_ins(tmp_path):
    m = _module()
    _write(tmp_path / "e.0.jsonl", _record(3, 7, misses=5, used=2, start=100, ms=4.0))
    (rec,) = m.load([str(tmp_path / "e.0.jsonl")])
    assert (rec["row"], rec["miss"], rec["used"], rec["left"]) == (3, 5, 2, 3)
    assert rec["ms"] == pytest.approx(4.0)  # from the earlier group's submit


def test_a_trace_that_dropped_events_is_refused(tmp_path):
    m = _module()
    _write(tmp_path / "e.0.jsonl", _record(0, 1, 1), dropped=2)
    with pytest.raises(ValueError, match="dropped"):
        m.load([str(tmp_path / "e.0.jsonl")])


def test_a_forward_is_every_row_once_at_consecutive_gens():
    m = _module()
    recs = [{"row": r, "gen": 10 + r} for r in range(3)] + [{"row": 0, "gen": 20}, {"row": 2, "gen": 22}]
    assert [[x["gen"] for x in f] for f in m.forwards(recs, rows=3)] == [[10, 11, 12]]


def test_the_within_stratum_slope_ignores_differences_between_strata():
    m = _module()
    # Row 1 costs 10 ms more than row 2 at any count; within each row a remaining read adds 0.5 ms.
    recs = [{"row": r, "miss": 2, "left": k, "ms": (10.0 if r == 1 else 0.0) + 0.5 * k} for r in (1, 2) for k in
            (0, 1, 2) for _ in range(3)]
    assert m.within_slope(recs, key=lambda x: (x["row"], x["miss"]), x="left") == pytest.approx(0.5)


def test_compare_matches_records_by_row_and_misses_without_prefetch_use():
    m = _module()
    base = [{"row": 1, "miss": k, "used": 0, "left": k, "ms": 2.0 + k} for k in (0, 1, 2) for _ in range(4)]
    other = [{"row": 1, "miss": k, "used": 0, "left": k, "ms": 3.0 + k} for k in (0, 1, 2) for _ in range(4)]
    other += [{"row": 1, "miss": 2, "used": 1, "left": 1, "ms": 0.0}] * 4  # swapped-in records are not matched
    c = m.compare(base, other)
    assert c["matched_records"] == 12
    assert c["ms_per_record_slower"] == pytest.approx(1.0)
    assert c["by_misses"]["0"]["ms_slower"] == pytest.approx(1.0)
