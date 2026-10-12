"""R1 control GroupScorers (D6): native gate, gate + floors, pooled token scorers, the hybrid and the prev/pop rows.

Each scorer returns one f32 [384] block score per group (-inf = not a candidate) for r1_policy.score_split, which
consumes it as is. Token-score controls are pooled exactly once, inside the scorer that owns them: margins by the
b1_score margin rule (best margin over tokens), counts by sum, token probabilities by r1_policy.pool_tokens. The hybrid
ranks by the native gate and takes the learned block score of the gate's top available row only.

Inputs: R1Groups, b1_baselines scorers (GateScorer, PrevScorer, PopScorer) and any token-probability scorer such as
b1_head.HeadScorer built with an x_of that reads through r1_groups.x_rows (dropped tokens stay aligned).
Output: GroupScorers for r1_policy.score_split.

Usage: scorers = r1_controls.control_scorers(floors=load_floors(), pop=pop, f0_token_scorer=f0, learned=None)
"""

from __future__ import annotations

import os
import sys
from typing import Literal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_baselines  # noqa: E402
import b1_rows  # noqa: E402
import b1_score  # noqa: E402
import r1_groups  # noqa: E402
import r1_policy  # noqa: E402
from r1_groups import GroupScorer, R1Group  # noqa: E402

NUM_EXPERTS = r1_groups.NUM_EXPERTS


def _block_from_token_scores(ts: b1_rows.TokenScores, *, r: R1Group, rule: Literal["margin", "count"]) -> np.ndarray:
    skip = r1_groups.skip_mask(r.g)
    return b1_score._aggregate(b1_rows.TokenScores(scores=ts.scores, rule=rule), skip=skip,
                               k=b1_rows.GATE_DEPTH).astype(np.float32)


class GateGroupScorer:
    """Native gate on x[T]: best margin over live tokens, with the packaged L+1 floors when floors are given."""

    def __init__(self, *, floors: dict[int, float] | None = None):
        self._scorer = b1_baselines.GateScorer(floors=floors)
        self.name = self._scorer.name

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        out = []
        for r in groups:
            ts = self._scorer(r.g)
            block = _block_from_token_scores(ts, r=r, rule="margin")
            if ts.floor is not None:
                block = np.where(block >= ts.floor, block, -np.inf).astype(np.float32)
            out.append(block)
        return out


class TokenPoolScorer:
    """Pools a token-probability scorer's per-token scores once, by max or noisy-or."""

    def __init__(self, token_scorer, *, rule: Literal["max", "noisy_or"], name: str):
        self._token_scorer = token_scorer
        self._rule = rule
        self.name = name

    def _token_scores(self, groups: list[R1Group]) -> list[b1_rows.TokenScores]:
        raw = [r.g for r in groups]
        if isinstance(self._token_scorer, b1_score.BatchScorer):
            return self._token_scorer.score_many(raw)
        return [self._token_scorer(g) for g in raw]

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        out = []
        for r, ts in zip(groups, self._token_scores(groups)):
            scores = np.asarray(ts.scores, dtype=np.float32)
            if scores.shape != (len(r.g.tokens), NUM_EXPERTS):
                raise AssertionError(f"{self.name}: token scores {scores.shape} for {len(r.g.tokens)} kept tokens")
            out.append(r1_policy.pool_tokens(scores, rule=self._rule))
        return out


class HybridScorer:
    """The gate's top available row, scored by the learned block score; every other row is not a candidate."""

    def __init__(self, *, gate: GroupScorer, learned: GroupScorer, name: str = "hybrid"):
        self._gate = gate
        self._learned = learned
        self.name = name

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        gate = self._gate.score_groups(groups)
        learned = self._learned.score_groups(groups)
        out = []
        for r, g, v in zip(groups, gate, learned):
            block = np.full(NUM_EXPERTS, -np.inf, dtype=np.float32)
            top = r1_policy.candidates(g, skip=r1_groups.skip_mask(r.g)).ids[:1]
            block[top] = np.asarray(v, dtype=np.float32)[top]
            out.append(block)
        return out


class SanityScorer:
    """Wraps a b1_baselines count scorer (prev, pop): block score = summed per-token count, -inf where zero."""

    def __init__(self, scorer):
        self._scorer = scorer
        self.name = scorer.name

    def score_groups(self, groups: list[R1Group]) -> list[np.ndarray]:
        # prev is stateful over seq-ordered groups, so call it in the given order.
        return [_block_from_token_scores(self._scorer(r.g), r=r, rule="count") for r in groups]


def control_scorers(*, floors: dict[int, float] | None, pop: b1_baselines.PopScorer, f0_token_scorer,
                    learned: GroupScorer | None) -> list[GroupScorer]:
    """gate, gate_floors (when floors), prev, pop, <f0>_max, <f0>_noisy_or, hybrid and the learned scorer if given.

    The hybrid takes its admission score from `learned`, or from F0's noisy-or pooling when there is none; with a
    learned scorer, hybrid_f0 keeps the F0 hybrid beside it. prev is a fresh stateful instance.
    """
    gate = GateGroupScorer()
    scorers: list[GroupScorer] = [gate]
    if floors is not None:
        scorers.append(GateGroupScorer(floors=floors))
    scorers.append(SanityScorer(b1_baselines.PrevScorer()))
    scorers.append(SanityScorer(pop))
    name = f0_token_scorer.name
    f0_noisy = TokenPoolScorer(f0_token_scorer, rule="noisy_or", name=f"{name}_noisy_or")
    scorers.append(TokenPoolScorer(f0_token_scorer, rule="max", name=f"{name}_max"))
    scorers.append(f0_noisy)
    if learned is None:
        scorers.append(HybridScorer(gate=gate, learned=f0_noisy))
    else:
        scorers.append(HybridScorer(gate=gate, learned=learned))
        scorers.append(HybridScorer(gate=gate, learned=f0_noisy, name="hybrid_f0"))
        scorers.append(learned)
    return scorers
