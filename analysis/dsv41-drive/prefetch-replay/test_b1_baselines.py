"""b1_baselines.py on synthetic groups: gate margin ordering and floors, the packaged floors table, prev's
no-lookahead, and pop fit on train rows only. CPU only."""

import sys
from pathlib import Path

import msgspec
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_baselines  # noqa: E402
import b1_rows  # noqa: E402
from test_b1_rows import _group  # noqa: E402


def _with(g, **changes):
    return msgspec.structs.replace(g, **changes)


def test_gate_margin_is_score_minus_the_sixth_and_keeps_order():
    g = _group(n=2)
    scores = np.array([[5, 4, 3.5, 3, 2.5, 2, 1.5, 1, .5, 0, -.5, -1]] * 2, dtype=np.float32)
    ids = np.array([[9, 4, 0, 11, 2, 7, 1, 3, 5, 6, 8, 10]] * 2, dtype=np.int16)
    out = b1_baselines.GateScorer()(_with(g, gate_scores=scores, gate_ids=ids))
    assert out.rule == "margin" and out.floor is None
    assert out.scores.shape == (2, b1_rows.NUM_EXPERTS)
    assert out.scores[0, 7] == 0.0  # the 6th
    assert out.scores[0, 9] == 3.0 and out.scores[0, 10] == -3.0
    ranked = np.argsort(-out.scores[0])[:12]
    assert ranked.tolist() == ids[0].tolist()
    assert np.isneginf(out.scores[0, 100])


def test_gate_without_a_bias_has_no_candidates():
    out = b1_baselines.GateScorer()(_group(has_bias=False))
    assert np.isneginf(out.scores).all()


def test_floors_apply_at_l_plus_1_only_and_name_follows_floors():
    plain = b1_baselines.GateScorer()
    floored = b1_baselines.GateScorer(floors={1: 0.25})
    assert (plain.name, floored.name) == ("gate", "gate_floors")
    assert floored(_group(horizon=1)).floor == 0.25  # target_row = source_row + 1
    assert floored(_group(horizon=2)).floor is None
    assert floored(_group(horizon=1, source_row=3)).floor is None  # row 4 not in the table
    assert plain(_group(horizon=1)).floor is None


def test_packaged_floors_table_loads_by_target_row():
    floors = b1_baselines.load_floors()
    assert floors[5] == 0.0 and floors[14] == 0.25 and floors[1] == 0.1


def test_prev_uses_only_the_earlier_forward_and_resets_per_session():
    base = _group(n=1)
    ids = lambda *e: np.array([list(e)], dtype=np.int16)  # noqa: E731
    g0 = _with(base, seq=0, target_ids=ids(1, 2, 3, 4, 5, 6))
    g1 = _with(base, seq=1, target_ids=ids(10, 11, 12, 13, 14, 15))
    g2 = _with(base, seq=2, target_ids=ids(20, 21, 22, 23, 24, 25))
    prev = b1_baselines.PrevScorer()
    s0, s1, s2 = prev(g0), prev(g1), prev(g2)
    assert s0.rule == "count" and np.isneginf(s0.scores).all()
    assert np.flatnonzero(np.isfinite(s1.scores[0])).tolist() == [1, 2, 3, 4, 5, 6]
    assert np.flatnonzero(np.isfinite(s2.scores[0])).tolist() == [10, 11, 12, 13, 14, 15]
    assert np.isneginf(s2.scores[0, 20])  # its own forward is never visible
    other = prev(_with(g1, session="s2"))
    assert np.isneginf(other.scores).all()
    with pytest.raises(ValueError):
        prev(_with(g1, session="s2", seq=0))


def test_prev_counts_tokens_per_expert():
    base = _group(n=2)
    ids = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]], dtype=np.int16)
    prev = b1_baselines.PrevScorer()
    prev(_with(base, seq=0, target_ids=ids))
    out = prev(_with(base, seq=1))
    assert out.scores[0, 1] == 2.0 and out.scores[0, 3] == 1.0


def test_pop_counts_eligible_rows_and_never_ranks_an_unseen_expert():
    train = _group(split="train")
    pop = b1_baselines.PopScorer.fit([train, _with(train, seq=1)])
    out = pop(train)
    assert out.rule == "count"
    assert np.flatnonzero(np.isfinite(out.scores[0])).tolist() == [3, 7, 300]
    assert out.scores[0, 7] == 2.0
    dev_only = _with(train, split="dev", target_layer=train.target_layer)
    assert np.flatnonzero(np.isfinite(pop(dev_only).scores[0])).tolist() == [3, 7, 300]
    other_layer = _with(train, target_layer=99)
    assert np.isneginf(pop(other_layer).scores).all()


def test_pop_refuses_a_non_train_group_and_round_trips_json():
    with pytest.raises(ValueError):
        b1_baselines.PopScorer.fit([_group(split="train"), _group(split="dev")])
    pop = b1_baselines.PopScorer.fit([_group(split="train")])
    again = b1_baselines.PopScorer.from_json(pop.to_json())
    assert again.counts.keys() == pop.counts.keys()
    assert all(np.array_equal(again.counts[k], pop.counts[k]) for k in pop.counts)


def test_baseline_scorers_by_horizon():
    pop = b1_baselines.PopScorer.fit([])
    floors = {1: 0.1}
    assert [s.name for s in b1_baselines.baseline_scorers(horizon=1, floors=floors, pop=pop)] == [
        "gate", "gate_floors", "prev", "pop"]
    assert [s.name for s in b1_baselines.baseline_scorers(horizon=2, floors=floors, pop=pop)] == [
        "gate", "prev", "pop"]
