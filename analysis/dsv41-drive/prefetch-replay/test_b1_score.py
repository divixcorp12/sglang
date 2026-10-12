"""b1_score.py on hand-built 16-expert Groups: the margin, noisy-or and count aggregations, hot and RAM rows skipped,
the read budgets and their metrics, shared denominators across scorers, the cut sweep and calibration. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import b1_score  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from test_b0_integrity import EXPERTS, LAYERS  # noqa: E402

MISS_GPU, MISS_CPU, HIT, HOT = int(LaneKind.MISS_GPU), int(LaneKind.MISS_CPU), int(LaneKind.HIT_COPY), 0
NEG = -np.inf


def _group(*, session="s1", seq=0, flagged=True, target_layer=LAYERS[1]):
    # Eligible {1, 2, 5, 7}; 3 is routed VRAM-hot; 7 is outside every token's gate top-12 (the never-candidate).
    target_ids = np.array([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]], np.int16)
    kinds = np.array([[MISS_GPU, MISS_CPU, HOT, HIT, MISS_GPU, HIT], [MISS_GPU, MISS_GPU, HIT, HIT, HIT, HIT]],
                     np.uint8)
    gate_ids = np.tile(np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12], np.int16), (2, 1))
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 4)  # expert 4 is in RAM
    return b1_rows.Group(
        session=session, split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=flagged,
        horizon=1, source_row=0, source_layer=LAYERS[0], source_stratum="a", gate_has_bias=True, target_row=1,
        target_layer=target_layer, target_stratum="b", tokens=np.arange(2, dtype=np.int16), x_offset=0,
        source_ids=np.zeros((2, 6), np.int16), source_weights=np.zeros((2, 6), np.float32),
        target_ids=target_ids, target_kinds=kinds, ram_mask=mask, hot=np.array([0], np.int16),
        gate_ids=gate_ids, gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(2, 0))


def _scores(rows: list[dict[int, float]]) -> np.ndarray:
    out = np.full((len(rows), EXPERTS), NEG, np.float32)
    for m, row in enumerate(rows):
        for e, v in row.items():
            out[m, e] = v
    return out


# Token 0 scores 16 - e for every e (6th = 11); token 1 has 7 scorers (6th = 0.5).
MARGIN_SCORES = _scores([{e: 16.0 - e for e in range(EXPERTS)},
                         {7: 5.0, 9: 4.0, 1: 3.0, 2: 2.0, 10: 1.0, 11: 0.5, 12: 0.0}])


class _Fixed:
    def __init__(self, name, scores, rule, floor=None):
        self.name, self.ts = name, b1_rows.TokenScores(scores=scores, rule=rule, floor=floor)

    def __call__(self, g):
        return self.ts


def test_margin_keeps_k_per_token_by_margin_to_the_sixth_and_skips_hot_and_ram_rows():
    g = _group()
    ts = b1_rows.TokenScores(scores=MARGIN_SCORES, rule="margin")
    # Token 0 keeps 1, 2, 3 (0 is hot, 4 in RAM); token 1 keeps 7, 9, 1. Expert 1 takes its best margin, 4.
    assert b1_score.candidates(g, ts, k=3) == [(7, 4.5), (1, 4.0), (9, 3.5), (2, 3.0), (3, 2.0)]
    floored = b1_rows.TokenScores(scores=MARGIN_SCORES, rule="margin", floor=3.0)
    assert [e for e, _ in b1_score.candidates(g, floored, k=3)] == [7, 1, 9, 2]


def test_noisy_or_and_count_aggregate_over_tokens_with_ties_by_id():
    g = _group()
    p = _scores([{7: 0.5, 9: 0.2, 0: 0.9}, {7: 0.5, 2: 0.2}])
    got = dict(b1_score.candidates(g, b1_rows.TokenScores(scores=p, rule="noisy_or")))
    assert got.keys() == {7, 9, 2}
    assert got[7] == pytest.approx(0.75) and got[9] == pytest.approx(0.2)
    counts = _scores([{5: 1.0, 2: 1.0, 4: 3.0}, {5: 1.0, 2: 1.0}])
    assert b1_score.candidates(g, b1_rows.TokenScores(scores=counts, rule="count")) == [(2, 2.0), (5, 2.0)]


def test_read_budgets_count_recall_precision_waste_and_never_recovery():
    t = b1_score.evaluate([_group()], _Fixed("m", MARGIN_SCORES, "margin"), cuts=(2.0,), k=3)
    want = {"reads=1": (1, 1, 0, 1), "reads=2": (2, 2, 0, 1), "reads=3": (3, 2, 0, 1), "cut=2.0": (5, 3, 1, 1)}
    for budget, (read, useful, wasted, recovered) in want.items():
        c = t.cells[budget]["pooled"]
        assert (c.forwards, c.eligible, c.never_candidates) == (1, 4, 1)
        assert (c.read, c.useful, c.wasted_hot, c.never_recovered) == (read, useful, wasted, recovered), budget
        assert c.saved_ms_lo == pytest.approx(0.33 * useful) and c.saved_ms_hi == pytest.approx(0.52 * useful)
    c = t.cells["cut=2.0"]["pooled"]
    assert (c.recall, c.precision, c.never_recovery) == (0.75, 0.6, 1.0)


def test_cells_are_keyed_pooled_noflag_layer_stratum_and_session():
    groups = [_group(), _group(session="s2", flagged=False, target_layer=LAYERS[0])]
    cells = b1_score.evaluate(groups, _Fixed("m", MARGIN_SCORES, "margin"), k=3).cells["reads=1"]
    assert set(cells) == {"pooled", "pooled_noflag", f"layer:{LAYERS[0]}", f"layer:{LAYERS[1]}", "stratum:b",
                          "session:s1", "session:s2"}
    assert cells["pooled"].forwards == 2 and cells["pooled_noflag"].forwards == 1
    assert cells["session:s2"].eligible == 4 and cells["stratum:b"].eligible == 8


def test_eligible_denominators_match_across_scorers():
    groups = [_group(), _group(session="s2", seq=1, flagged=False, target_layer=LAYERS[0])]
    scorers = [_Fixed("margin", MARGIN_SCORES, "margin"),
               _Fixed("floored", MARGIN_SCORES, "margin", floor=10.0),
               _Fixed("head", _scores([{7: 0.9, 3: 0.4}, {1: 0.6}]), "noisy_or"),
               _Fixed("prev", _scores([{2: 1.0}, {2: 1.0, 5: 1.0}]), "count"),
               _Fixed("no_gate", np.full((2, EXPERTS), NEG, np.float32), "margin")]
    tables = b1_score.evaluate_many(iter(groups), scorers, cuts=(0.5, 2.0))
    ref = tables["margin"]
    for name, t in tables.items():
        assert t.cells.keys() == ref.cells.keys(), name
        for budget, keyed in t.cells.items():
            assert keyed.keys() == ref.cells[budget].keys(), name
            for key, c in keyed.items():
                r = ref.cells[budget][key]
                assert (c.forwards, c.eligible, c.never_candidates) == (r.forwards, r.eligible, r.never_candidates)
    assert tables["no_gate"].cells["reads=3"]["pooled"].read == 0


class _Batched(_Fixed):
    """Per-group scores keyed by seq, offered both one at a time and through score_many."""

    def __init__(self, name, by_seq, rule):
        self.name, self.by_seq, self.rule, self.calls = name, by_seq, rule, []

    def __call__(self, g):
        return b1_rows.TokenScores(scores=self.by_seq[g.seq], rule=self.rule)

    def score_many(self, groups):
        self.calls.append(len(groups))
        return [self(g) for g in groups]


@pytest.mark.parametrize("chunk", [1, 2, 64])
def test_score_many_gives_the_same_tables_as_one_group_at_a_time(chunk):
    groups = [_group(seq=i, session=f"s{i % 2}", flagged=bool(i % 3), target_layer=LAYERS[i % 2]) for i in range(5)]
    by_seq = {i: _scores([{7: 0.9 - 0.1 * i, 1: 0.5, 2: 0.05 * i}, {5: 0.3, 2: 0.6}]) for i in range(5)}
    batched = _Batched("b", by_seq, "noisy_or")
    single = type("Single", (), {"name": "b", "__call__": lambda self, g: batched(g)})()
    want = b1_score.evaluate_many(iter(groups), [single], cuts=(0.5, 0.9))["b"]
    got = b1_score.evaluate_many(iter(groups), [batched], cuts=(0.5, 0.9), chunk=chunk)["b"]
    assert b1_score.table_to_json(got) == b1_score.table_to_json(want)
    assert sum(batched.calls) == len(groups) and max(batched.calls) <= chunk


def test_evaluate_matches_evaluate_many_and_reports_split_and_horizon():
    s = _Fixed("m", MARGIN_SCORES, "margin")
    one = b1_score.evaluate([_group()], s, cuts=(2.0,))
    many = b1_score.evaluate_many([_group()], [s], cuts=(2.0,))["m"]
    assert b1_score.table_to_json(one) == b1_score.table_to_json(many)
    assert (one.split, one.horizon) == ("dev", 1)


def test_cut_sweep_finds_the_lowest_cut_reaching_a_precision():
    groups = [_group(), _group(seq=1)]
    t = b1_score.evaluate(groups, _Fixed("m", MARGIN_SCORES, "margin"), cuts=(2.0, 3.0, 4.5), k=3)
    assert b1_score.cut_for_precision(t, 0.7) == 3.0
    assert b1_score.recall_at_precision(t, 0.7) == 0.75
    assert b1_score.cut_for_precision(t, 0.95) == 4.5
    assert b1_score.recall_at_precision(t, 0.95) == 0.25
    assert b1_score.cut_for_precision(t, 1.01) is None and b1_score.recall_at_precision(t, 1.01) is None


def test_calibration_bins_probability_scorers_only():
    p = _scores([{7: 0.5, 9: 0.15}, {7: 0.5}])
    tables = b1_score.evaluate_many([_group()], [_Fixed("head", p, "noisy_or"),
                                                 _Fixed("m", MARGIN_SCORES, "margin")])
    bins = tables["head"].calibration
    assert len(bins) == b1_score.CALIBRATION_BINS
    assert bins[7][2:] == (1, 1) and bins[1][2:] == (1, 0)
    assert sum(b[2] for b in bins) == 2
    assert tables["m"].calibration == []


def test_table_json_is_plain_json():
    t = b1_score.evaluate([_group()], _Fixed("m", MARGIN_SCORES, "margin"), cuts=(2.0,), k=3)
    doc = json.loads(json.dumps(b1_score.table_to_json(t), indent=1))
    assert doc["cells"]["cut=2.0"]["pooled"]["precision"] == 0.6
    assert doc["cells"]["reads=1"]["session:s1"]["read"] == 1
