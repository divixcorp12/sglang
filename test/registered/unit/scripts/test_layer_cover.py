"""Per layer prefetch cover and read value (layer_cover.py) and the layer-completing oracle's rules (CPU)."""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DSPARK = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark")


def _module():
    sys.path.insert(0, DSPARK)  # layer_cover imports layer_misses
    spec = importlib.util.spec_from_file_location("layer_cover", os.path.join(DSPARK, "layer_cover.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ev(kind, ns, row, gen=0, seq=0, group=0, a=0, b=0, c=0):
    return {"event": kind, "ns": ns, "row": row, "gen": gen, "seq": seq, "group": group, "a": a, "b": b, "c": c}


def _layer(row, gen, misses, start, ms, reads=()):
    """A record with ``misses`` forced misses; ``reads`` = (expert, land_before_use_us) covered by the pool."""
    out = [_ev("copy_submit", start, row, gen, b=misses)]
    for expert, slack_us in reads:
        use = start + 500_000
        out += [_ev("spec_submit", use - 3_000_000, row, seq=gen - 1, a=expert),
                _ev("spec_land", use - int(slack_us * 1e3), row, seq=gen - 1, a=expert),
                _ev("spec_use", use, row, gen, a=expert, c=gen - 1)]
    return out + [_ev("gate_open", start + int(ms * 1e6), row, gen)]


def _write(path, events):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_cover_splits_all_some_none_and_flags_a_read_still_in_flight(tmp_path):
    m = _module()
    events = _layer(0, 10, 0, 0, 1.0)
    events += _layer(1, 11, 1, 10**7, 2.0, reads=[(5, 2000)])  # all, in time
    events += _layer(2, 12, 1, 2 * 10**7, 2.0, reads=[(6, 20)])  # all, the demand waited (late)
    events += _layer(3, 13, 2, 3 * 10**7, 3.0, reads=[(7, 2000)])  # some
    events += _layer(4, 14, 3, 4 * 10**7, 4.0)  # none
    _write(tmp_path / "e.0.jsonl", events)
    out = m.summarize(m.load([str(tmp_path / "e.0.jsonl")]), late_us=100)
    cov = out["coverage"]
    assert (cov["layers"], cov["all"], cov["all_in_time"], cov["all_some_late"]) == (4, 0.5, 0.25, 0.25)
    assert (cov["some"], cov["none"]) == (0.25, 0.25)
    assert out["miss_hist"] == {"1": 2, "2": 1, "3": 1}  # rows 1 and up only
    assert out["timing"]["late_share"] == pytest.approx(1 / 3, abs=1e-3)


def test_read_steps_compare_layers_of_one_row_and_miss_count_by_reads_left():
    m = _module()
    rows = [{"row": 2, "miss": 2, "left": left, "ms": ms} for left, ms in ((0, 3.0), (1, 3.5), (1, 3.7), (2, 4.6))]
    rows += [{"row": 3, "miss": 2, "left": 2, "ms": 9.0}]  # no partner cell in row 3: ignored
    steps = m.read_steps(rows, max_misses=2)
    assert steps["2 misses, left 0->1"]["ms"] == pytest.approx(0.6)
    assert steps["2 misses, left 1->2"] == {"ms": pytest.approx(1.0), "weight": 1}


def test_a_trace_that_dropped_events_is_refused(tmp_path):
    m = _module()
    _write(tmp_path / "e.0.jsonl", _layer(0, 1, 1, 0, 1.0) + [{"dropped": 3}])
    with pytest.raises(ValueError, match="dropped"):
        m.load([str(tmp_path / "e.0.jsonl")])


def _oracle():
    sys.path.insert(0, os.path.join(ROOT, "analysis", "dsv41-drive", "prefetch-replay"))
    sys.path.insert(0, os.path.join(ROOT, "scripts", "dsv41"))
    import layer_cover_oracle

    return layer_cover_oracle


def test_candidates_merge_tokens_by_best_margin_past_skipped_rows():
    o = _oracle()
    order = np.array([[1, 2, 3, 4, 5, 6, 7], [7, 1, 8, 9, 10, 11, 12]])
    score = np.array([[9.0, 8, 7, 6, 5, 4, 3.5], [9.5, 9.0, 8, 7, 6, 5, 1]])
    cand = o.candidate_list(order, score, tokens=2, skip=lambda e: e == 2)
    # expert 1: margins 5.0 (token 0) and 4.0 (token 1); expert 7: -0.5 and 4.5. Each keeps its best.
    assert cand[:2] == [(1, pytest.approx(5.0)), (7, pytest.approx(4.5))]
    assert 2 not in dict(cand)


def test_set_rule_reads_all_or_nothing_and_clears_only_complete_layers():
    o = _oracle()
    rows = [{"s": 0, "li": 0, "misses": [9], "cand": []},
            {"s": 0, "li": 1, "misses": [1, 2], "cand": [(1, 0.5), (2, 0.2), (3, -0.4)]},
            {"s": 0, "li": 2, "misses": [4], "cand": [(5, 0.3), (6, 0.2), (4, 0.1)]}]
    top2 = o.score_rule(rows, lambda c: [e for e, _ in c[:2]])
    assert (top2["reads"], top2["useful"], top2["cleared"]) == (4, 2, 1)
    set2 = o.score_rule(rows, lambda c: (lambda s: s if len(s) <= 2 else [])([e for e, mg in c if mg >= 0.0]))
    assert (set2["reads"], set2["useful"], set2["cleared"]) == (2, 2, 1)  # layer 2's set of 3 is refused
    assert set2["saved_ms"] == pytest.approx(2 * 0.67)
