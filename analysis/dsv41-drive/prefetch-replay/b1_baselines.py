"""B1 baseline scorers: the target gate on x[T] (with and without the packaged margin floors), prev and pop.

Each scorer maps one Group to TokenScores [n, num_experts] (-inf = not a candidate); the shared scoring function in
b1_score.py applies hot/RAM skipping, the read budget and the floor. Inputs are Groups from b1_rows.py. pop is fit on
train-split groups only (R2, C4). prev is stateful over one session's seq-ordered stream.

Usage: scorers = b1_baselines.baseline_scorers(horizon=1, floors=b1_baselines.load_floors(), pop=pop)
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_rows  # noqa: E402

from sglang.srt.layers.moe.ram_prefetch import MARGIN_FLOORS, read_margin_floors  # noqa: E402

_NO_CANDIDATE = np.float32(-np.inf)
_SIXTH = 5  # the token's 6th best gate score is its top-k boundary


def load_floors(path: str = MARGIN_FLOORS) -> dict[int, float]:
    """Packaged per-target-row margin floors (row -> least margin a candidate needs to be read)."""
    return read_margin_floors(path)


def _empty(*, n: int, num_experts: int) -> np.ndarray:
    return np.full((n, num_experts), _NO_CANDIDATE, dtype=np.float32)


class GateScorer:
    """Target gate on x[T]: per token, margin = biased score - the token's 6th score, over its top-12."""

    def __init__(self, *, floors: dict[int, float] | None = None, num_experts: int = b1_rows.NUM_EXPERTS):
        self.floors = floors
        self.num_experts = num_experts
        self.name = "gate" if floors is None else "gate_floors"

    def __call__(self, g: b1_rows.Group) -> b1_rows.TokenScores:
        n = len(g.tokens)
        scores = _empty(n=n, num_experts=self.num_experts)
        if g.gate_has_bias:
            margin = g.gate_scores - g.gate_scores[:, _SIXTH:_SIXTH + 1]
            rows, cols = np.nonzero(g.gate_ids >= 0)
            scores[rows, g.gate_ids[rows, cols]] = margin[rows, cols]
        return b1_rows.TokenScores(scores=scores, rule="margin", floor=self._floor(g))

    def _floor(self, g: b1_rows.Group) -> float | None:
        # The packaged floors were fit for the L+1 target row only.
        if self.floors is None or g.horizon != 1:
            return None
        return self.floors.get(g.target_row)


class PrevScorer:
    """The previous verify forward's per-expert token counts at the target row; only earlier forwards are used."""

    name = "prev"

    def __init__(self, *, num_experts: int = b1_rows.NUM_EXPERTS):
        self.num_experts = num_experts
        self._session: str | None = None
        # target row -> (seq of the forward being accumulated, its counts, counts of the forward before it)
        self._rows: dict[int, tuple[int, np.ndarray, np.ndarray | None]] = {}

    def __call__(self, g: b1_rows.Group) -> b1_rows.TokenScores:
        if g.session != self._session:
            self._session, self._rows = g.session, {}
        prior = self._prior(g)
        self._record(g)
        n = len(g.tokens)
        scores = _empty(n=n, num_experts=self.num_experts)
        if prior is not None:
            seen = prior > 0
            scores[:, seen] = prior[seen].astype(np.float32)
        return b1_rows.TokenScores(scores=scores, rule="count")

    def _prior(self, g: b1_rows.Group) -> np.ndarray | None:
        state = self._rows.get(g.target_row)
        if state is None:
            return None
        seq, counts, before = state
        if g.seq < seq:
            raise ValueError(f"prev: seq {g.seq} after {seq} in session {g.session}; groups must be seq-ordered")
        return before if seq == g.seq else counts

    def _record(self, g: b1_rows.Group) -> None:
        routed = g.target_ids[g.target_ids >= 0].astype(np.int64)
        counts = np.bincount(routed, minlength=self.num_experts).astype(np.int64)
        state = self._rows.get(g.target_row)
        if state is not None and state[0] == g.seq:
            self._rows[g.target_row] = (g.seq, state[1] + counts, state[2])
        else:
            self._rows[g.target_row] = (g.seq, counts, None if state is None else state[1])


class PopScorer:
    """Per target layer, how many train groups had each expert as an eligible miss (zero count = no candidate)."""

    name = "pop"

    def __init__(self, *, counts: dict[int, np.ndarray], num_experts: int = b1_rows.NUM_EXPERTS):
        self.counts = counts
        self.num_experts = num_experts

    @classmethod
    def fit(cls, groups: Iterable[b1_rows.Group], *, num_experts: int = b1_rows.NUM_EXPERTS) -> PopScorer:
        counts: dict[int, np.ndarray] = {}
        for g in groups:
            if g.split != "train":
                raise ValueError(f"pop is fit on train groups only, got split {g.split!r} (session {g.session})")
            layer = counts.setdefault(g.target_layer, np.zeros(num_experts, dtype=np.int64))
            layer[b1_rows.eligible(g).astype(np.int64)] += 1
        return cls(counts=counts, num_experts=num_experts)

    def __call__(self, g: b1_rows.Group) -> b1_rows.TokenScores:
        n = len(g.tokens)
        scores = _empty(n=n, num_experts=self.num_experts)
        counts = self.counts.get(g.target_layer)
        if counts is not None:
            seen = counts > 0
            scores[:, seen] = counts[seen].astype(np.float32)
        return b1_rows.TokenScores(scores=scores, rule="count")

    def to_json(self) -> str:
        table = {str(layer): c.tolist() for layer, c in sorted(self.counts.items())}
        return json.dumps({"num_experts": self.num_experts, "counts": table}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> PopScorer:
        doc = json.loads(text)
        counts = {int(layer): np.asarray(c, dtype=np.int64) for layer, c in doc["counts"].items()}
        return cls(counts=counts, num_experts=int(doc["num_experts"]))


def baseline_scorers(*, horizon: int, floors: dict[int, float] | None, pop: PopScorer,
                     num_experts: int = b1_rows.NUM_EXPERTS) -> list[b1_rows.Scorer]:
    """gate, gate_floors (L+1 only, when floors are given), prev, pop; prev is a fresh instance."""
    scorers: list[b1_rows.Scorer] = [GateScorer(num_experts=num_experts)]
    if horizon == 1 and floors is not None:
        scorers.append(GateScorer(floors=floors, num_experts=num_experts))
    scorers.append(PrevScorer(num_experts=num_experts))
    scorers.append(pop)
    return scorers
