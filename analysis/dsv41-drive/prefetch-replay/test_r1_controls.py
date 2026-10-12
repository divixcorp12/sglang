"""r1_controls.py on synthetic groups: gate margins and floors, single pooling, the hybrid's one-row admission, the
prev/pop sanity rows and the control list. CPU only."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_baselines  # noqa: E402
import b1_rows  # noqa: E402
import r1_controls  # noqa: E402
import r1_groups  # noqa: E402
import r1_policy  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

E = r1_groups.NUM_EXPERTS
MISS_GPU, HIT = int(LaneKind.MISS_GPU), int(LaneKind.HIT_COPY)


def _group(*, seq):
    n = 2
    target_ids = np.array([[7, 3, 9, 1, 2, 0], [10, 11, 12, 13, 14, -1]], dtype=np.int16)
    kinds = np.array([[MISS_GPU, MISS_GPU, 0, 0, 0, 0], [MISS_GPU, HIT, HIT, HIT, HIT, 0]], dtype=np.uint8)
    mask = np.zeros(6, np.uint64)
    mask[0] = np.uint64(1 << 3)
    return b1_rows.Group(
        session="s1", split="dev", block="b000", seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=1, source_row=0, source_layer=3, source_stratum="a", gate_has_bias=True, target_row=1,
        target_layer=4, target_stratum="b", tokens=np.array([0, 1], np.int16), x_offset=n * seq,
        source_ids=np.arange(6 * n, dtype=np.int16).reshape(n, 6), source_weights=np.full((n, 6), 0.25, np.float32),
        target_ids=target_ids, target_kinds=kinds, ram_mask=mask, hot=np.array([0, 1], np.int16),
        gate_ids=np.tile(np.arange(12, dtype=np.int16), (n, 1)),
        gate_scores=np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0))


@pytest.fixture
def groups(tmp_path):
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=[_group(seq=0), _group(seq=1)])
    return list(r1_groups.read_r1_groups(tmp_path, split="dev", horizon=1, exclusions=r1_groups.Exclusions.empty()))


class _Probs:
    """Token scorer: probability 0.5 for expert 9 on token 0, 0.5 for experts 9 and 20 on token 1."""

    name = "tok"

    def __call__(self, g):
        s = np.full((len(g.tokens), E), -np.inf, np.float32)
        s[0, 9] = 0.5
        s[1, [9, 20]] = 0.5
        return b1_rows.TokenScores(scores=s, rule="noisy_or")


class _Learned:
    name = "learned"

    def score_groups(self, groups):
        out = []
        for _ in groups:
            s = np.full(E, -np.inf, np.float32)
            s[[2, 5, 9]] = [0.3, 0.9, 0.8]
            out.append(s)
        return out


def test_gate_block_is_best_margin_to_the_sixth_score_with_skip_rows_removed(groups):
    block = r1_controls.GateGroupScorer().score_groups(groups)[0]
    sixth = np.linspace(2.0, 1.0, 12, dtype=np.float32)[5]
    assert block.dtype == np.float32 and block.shape == (E,)
    # 0 and 1 are VRAM-hot and 3 is in the RAM snapshot.
    assert np.isneginf(block[[0, 1, 3]]).all()
    assert block[2] == pytest.approx(np.linspace(2.0, 1.0, 12, dtype=np.float32)[2] - sixth)
    assert np.isneginf(block[12:]).all()


def test_gate_floors_blank_rows_below_the_packaged_floor(groups):
    scorer = r1_controls.GateGroupScorer(floors={1: 0.0})
    assert scorer.name == "gate_floors"
    block = scorer.score_groups(groups)[0]
    assert np.isfinite(block).sum() == 3  # ids 2, 4, 5 have margin >= 0 among the unskipped
    assert np.isneginf(r1_controls.GateGroupScorer(floors={1: 10.0}).score_groups(groups)[0]).all()
    assert r1_controls.GateGroupScorer().name == "gate"


def test_token_pool_scorer_pools_once_by_max_or_noisy_or(groups):
    mx = r1_controls.TokenPoolScorer(_Probs(), rule="max", name="m").score_groups(groups)[0]
    no = r1_controls.TokenPoolScorer(_Probs(), rule="noisy_or", name="n").score_groups(groups)[0]
    assert mx[9] == pytest.approx(0.5) and no[9] == pytest.approx(0.75)
    assert mx[20] == no[20] == pytest.approx(0.5)
    assert np.isneginf(mx[0]) and np.isneginf(no[0])


def test_hybrid_scores_only_the_gates_top_available_row_with_the_learned_score(groups):
    hybrid = r1_controls.HybridScorer(gate=r1_controls.GateGroupScorer(), learned=_Learned())
    block = hybrid.score_groups(groups)[0]
    assert np.isfinite(block).sum() == 1
    assert block[2] == pytest.approx(0.3)  # gate's top available row is 2, not the learned scorer's best (5)
    assert hybrid.name == "hybrid"


def test_prev_and_pop_pool_counts_over_tokens(groups):
    pop = b1_baselines.PopScorer(counts={4: np.bincount([5, 5, 5, 0], minlength=E)})
    block = r1_controls.SanityScorer(pop).score_groups(groups)[0]
    assert block[5] == 6.0 and np.isneginf(block[0]) and np.isneginf(block[4])
    prev = r1_controls.SanityScorer(b1_baselines.PrevScorer()).score_groups(groups)
    assert np.isneginf(prev[0]).all()
    assert prev[1][7] == 2.0 and np.isneginf(prev[1][3])  # seq 0 routed 7 once; 3 is in the RAM snapshot


def test_control_scorers_list_and_shared_scoring_path(groups):
    pop = b1_baselines.PopScorer(counts={4: np.ones(E, np.int64)})
    names = [s.name for s in r1_controls.control_scorers(floors={1: 0.0}, pop=pop, f0_token_scorer=_Probs(),
                                                         learned=None)]
    assert names == ["gate", "gate_floors", "prev", "pop", "tok_max", "tok_noisy_or", "hybrid"]
    with_learned = r1_controls.control_scorers(floors=None, pop=pop, f0_token_scorer=_Probs(), learned=_Learned())
    assert [s.name for s in with_learned][-3:] == ["hybrid", "hybrid_f0", "learned"]
    out = r1_policy.score_split(groups, with_learned)
    assert set(out) == {s.name for s in with_learned}
    assert all(len(c) == len(groups) for c in out.values())
