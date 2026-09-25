"""Offline VRAM-prefetch replay over the graph route log: DIRECT insert-on-miss plus predicted rows.

The question: if, while layer ``T - h`` runs, the ``K`` best predicted rows of layer ``T`` that are not
in VRAM are copied into its hot cache, how many decode misses go away, and what does the link pay?

- ``PrefetchReplay`` is ``tier_sim.DirectInsertReplay`` (which reproduces the measured misses) with one
  addition: just before a layer's gather, up to ``k`` predicted non-resident rows are inserted. Each
  evicts a resident row: the lowest-ranked slot by the forward's own victim order (the one the demand
  shortlist comes from) that is *not* in the demand shortlist, so the demand path inserts exactly as it
  would have, and does not hold a row the predictor ranked at or above its last pick. A wrongly predicted row gets no routes, scores low, and is the next victim. Demand misses
  and prefetched rows are both link rows (one ~13.3 MB row each).
- ``link_exposed`` prices a run on one PCIe link. Per layer: the demand phase (the layer's misses, plus
  whatever is left of prefetches aimed at this layer: the gather waits for them on the serial link) is
  exposed; then the compute window has ``budget`` rows of idle link, which drains the prefetch queue
  first in, first out. A prefetch for layer ``T`` is issued at the start of layer ``T - h``'s window (its
  routes are then known), so it can use the ``h`` windows before its target. A prefetch that does not
  fit is not free: its remainder delays the target layer's demand exactly like a miss.
- Predictors rank candidate experts for (decode step, target layer) using only what exists at issue
  time: routes of earlier tokens, and of this token's layers up to ``T - h``. ``h`` may reach past layer
  0 into the previous token's tail, but only when that previous decode step belongs to the same request.

Everything here is CPU-only replay; no runtime code is involved.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Sequence

import msgspec
import numpy as np

import tier_sim

NUM_EXPERTS = 384


# ------------------------------------------------------------------ replay


class PrefetchReplay(tier_sim.DirectInsertReplay):
    """DirectInsertReplay plus per-layer prefetch before each graph gather.

    ``graph_forward(routes, phase, candidates, k)``: ``candidates(layer)`` returns that layer's ranked
    expert predictions (evaluated when the layer is reached; the replay itself filters out resident rows
    and takes the first ``k``). ``last_prefetched`` holds, per layer, the rows inserted for the latest
    forward.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_prefetched: dict[int, list[int]] = {}
        self.prefetched = 0

    def _ranked_slots(self, row: int) -> list[int]:
        """Every slot in the victim order of ``_rank`` (not cut to ``miss_rows``)."""
        slots, scores, routed = self.slots[row], self.scores[row], self.routed[row]
        free = [slot for slot, expert in enumerate(slots) if expert < 0]
        held = sorted(
            (slot for slot, expert in enumerate(slots) if expert >= 0),
            key=lambda slot: (routed[slots[slot]], scores[slots[slot]], -slots[slot]),
        )
        return free + held

    def prefetch(self, layer: int, candidates: Sequence[int], k: int) -> list[int]:
        """Insert the first ``k`` non-resident ``candidates`` into ``layer``; returns the rows inserted."""
        if k <= 0:
            return []
        row = self.row[layer]
        where, slots = self.where[row], self.slots[row]
        rows, predicted = [], set()
        for expert in candidates:
            expert = int(expert)
            predicted.add(expert)
            if where[expert] < 0 and expert not in rows:
                rows.append(expert)
                if len(rows) == k:
                    break
        if not rows:
            return []
        # Never evict the demand shortlist, nor a resident row the predictor ranked above its last pick.
        shortlist = set(self.shortlist[row])
        victims = [
            slot for slot in self._ranked_slots(row) if slot not in shortlist and slots[slot] not in predicted
        ][: len(rows)]
        rows = rows[: len(victims)]
        for expert, slot in zip(rows, victims):
            old = slots[slot]
            if old >= 0:
                where[old] = -1
            slots[slot] = expert
            where[expert] = slot
        self.prefetched += len(rows)
        return rows

    def graph_forward(
        self,
        routes: dict[int, list[int]],
        phase: str = "decode",
        candidates: Optional[Callable[[int], Sequence[int]]] = None,
        k: int = 0,
    ) -> dict[int, int]:
        self.last_prefetched = {}
        if candidates is None or k <= 0:
            return super().graph_forward(routes, phase)
        # DirectInsertReplay.graph_forward, with the prefetch inserted before each layer's gather.
        self._apply(self.boundary_pending)
        self._count(1, decode=True)
        misses = {}
        for layer, experts in routes.items():
            self.last_prefetched[layer] = self.prefetch(layer, candidates(layer), k)
            row = self.row[layer]
            where, slots = self.where[row], self.slots[row]
            hits = {int(where[e]) for e in experts if where[e] >= 0}
            missing = list(dict.fromkeys(e for e in experts if where[e] < 0))
            np.add.at(self.route_counts[row], experts, np.float32(1.0))
            usable = [slot for slot in self.shortlist[row] if slot not in hits]
            for expert, slot in zip(missing, usable):
                old = slots[slot]
                if old >= 0:
                    where[old] = -1
                slots[slot] = expert
                where[expert] = slot
            self.truncated += max(len(missing) - len(usable), 0)
            misses[layer] = len(missing)
        self.host_pending = phase == "decode"
        if phase != "decode":
            self.decode_forwards -= 1
            self.boundary_pending = self.decode_forwards >= 1
        return misses


# ------------------------------------------------------------------ link model


class LinkModel(msgspec.Struct, frozen=True):
    """One serial RAM->VRAM link. ``budget`` is idle link per layer compute window, in rows."""

    budget: float = 0.85
    ms_per_row: float = 1.0


def link_exposed(
    demand: np.ndarray, prefetch: np.ndarray, continuous: np.ndarray, horizon: int, link: LinkModel
) -> np.ndarray:
    """Exposed link rows per decode step (demand plus the prefetch the idle windows could not absorb).

    ``demand`` and ``prefetch`` are ``[steps, layers]`` row counts; ``prefetch[s, t]`` rows target layer
    ``t`` of step ``s`` and were issued in the window of layer ``t - horizon`` (of step ``s - 1`` when that
    is negative, which requires ``continuous[s]``: step ``s - 1`` is the same request's previous token).
    """
    steps, layers = demand.shape
    exposed = np.zeros(steps, dtype=np.float64)
    queue: deque = deque()  # [target step, target layer, rows left], in issue (= target) order
    for s in range(steps):
        if not continuous[s]:
            queue.clear()  # nothing crosses into a new request (and nothing was issued across it)
        for t in range(layers):
            late = 0.0
            while queue and (queue[0][0], queue[0][1]) <= (s, t):
                late += queue.popleft()[2]
            exposed[s] += demand[s, t] + late
            # Window t: issue the prefetch aimed at t + horizon, then drain the idle link.
            target = t + horizon
            if target < layers:
                rows = prefetch[s, target]
                if rows:
                    queue.append([s, target, float(rows)])
            elif s + 1 < steps and continuous[s + 1]:
                rows = prefetch[s + 1, target - layers]
                if rows:
                    queue.append([s + 1, target - layers, float(rows)])
            idle = link.budget
            while queue and idle > 0:
                take = min(idle, queue[0][2])
                queue[0][2] -= take
                idle -= take
                if queue[0][2] <= 1e-12:
                    queue.popleft()
    return exposed


# ------------------------------------------------------------------ stream and predictors


class DecodeStream(msgspec.Struct):
    """Decode graph forwards in execution order: routes ``[steps, layers, 6]``, request per step, and
    ``continuous[s]`` (step ``s - 1`` is the same request's previous token)."""

    layers: list[int]
    routes: np.ndarray
    rids: list
    continuous: np.ndarray


def decode_stream(loaded: dict) -> DecodeStream:
    layers = loaded["layer_ids"]
    routes, rids = [], []
    for forward in loaded["forwards"]:
        if forward["phase"] != "decode":
            continue
        if forward["kind"] != "graph":
            raise ValueError("an eager decode forward; this study replays graph decode only")
        routes.append([forward["routes"][layer] for layer in layers])
        rids.append(forward["rids"][0] if forward["rids"] else None)
    continuous = np.array([s > 0 and rids[s] is not None and rids[s] == rids[s - 1] for s in range(len(rids))])
    return DecodeStream(layers, np.asarray(routes, dtype=np.int64), rids, continuous)


def source_of(stream: DecodeStream, step: int, target: int, horizon: int) -> Optional[tuple[int, int]]:
    """The (step, layer) whose routes are the newest known when layer ``target``'s prefetch is issued."""
    layer = target - horizon
    if layer >= 0:
        return step, layer
    if not stream.continuous[step]:
        return None
    return step - 1, layer + len(stream.layers)


class Predictor:
    """Ranks candidate experts for (decode step, target layer index, horizon). ``observe(step)`` runs
    after each decode step, so state never holds the step being predicted."""

    name = "none"

    def rank(self, step: int, target: int, horizon: int) -> Sequence[int]:
        return ()

    def observe(self, step: int) -> None:
        pass

    def bind(self, sim: "PrefetchReplay", index: dict[int, int]) -> None:
        """Called once by run_arm; only predictors that model a quality level read residency."""


class Oracle(Predictor):
    """The target's true routes: a non-deployable upper bound on what prefetch can do."""

    name = "oracle"

    def __init__(self, stream: DecodeStream) -> None:
        self.stream = stream

    def rank(self, step, target, horizon):
        return self.stream.routes[step, target]


class NoisyOracle(Predictor):
    """A stand-in for a predictor of known quality, to price one before building it (not deployable).

    Per target layer it names ``k`` rows; each is, with probability ``precision``, one of the layer's
    true routes that is not resident (while any remain), and otherwise a random expert that is neither
    routed nor resident: a wrong prefetch. It reads residency, so ``precision`` is the precision of the
    rows actually prefetched (short only where a layer has fewer true misses than hits drawn).
    """

    def __init__(self, stream: DecodeStream, precision: float, k: int, seed: int = 0) -> None:
        self.stream, self.precision, self.k = stream, precision, k
        self.rng = np.random.default_rng(seed)
        self.name = f"noisy_oracle_p{precision:g}"
        self.sim: Optional["PrefetchReplay"] = None

    def bind(self, sim, index):
        self.sim, self.layer_of = sim, {li: layer for layer, li in index.items()}

    def rank(self, step, target, horizon):
        layer = self.layer_of[target]
        resident = self.sim.resident(layer)
        routed = [int(e) for e in self.stream.routes[step, target]]
        missing = [e for e in routed if e not in resident]
        out = []
        for _ in range(self.k):
            if missing and self.rng.random() < self.precision:
                out.append(missing.pop(0))
                continue
            while True:
                wrong = int(self.rng.integers(NUM_EXPERTS))
                if wrong not in resident and wrong not in routed and wrong not in out:
                    out.append(wrong)
                    break
        return out


class PreviousToken(Predictor):
    """The same layer's routes of the same request's previous token."""

    name = "prev_token"

    def __init__(self, stream: DecodeStream) -> None:
        self.stream = stream

    def rank(self, step, target, horizon):
        if not self.stream.continuous[step]:
            return ()
        return self.stream.routes[step - 1, target]


class Popularity(Predictor):
    """Per-layer route counts over training steps, frozen."""

    name = "popularity"

    def __init__(self, stream: DecodeStream, train: np.ndarray, depth: int = 48) -> None:
        counts = np.zeros((len(stream.layers), NUM_EXPERTS), dtype=np.int64)
        for step in np.flatnonzero(train):
            for li in range(len(stream.layers)):
                counts[li, stream.routes[step, li]] += 1
        self.order = np.argsort(-counts, axis=1, kind="stable")[:, :depth]

    def rank(self, step, target, horizon):
        return self.order[target]


class Recency(Predictor):
    """Per-layer route scores decayed by ``decay`` per decode token (all requests, as the cache sees them)."""

    def __init__(self, stream: DecodeStream, decay: float, depth: int = 48) -> None:
        self.stream, self.decay, self.depth = stream, decay, depth
        self.scores = np.zeros((len(stream.layers), NUM_EXPERTS), dtype=np.float64)
        self.name = f"recency{decay:g}"
        self._order: Optional[np.ndarray] = None

    def observe(self, step):
        self.scores *= self.decay
        for li in range(len(self.stream.layers)):
            self.scores[li, self.stream.routes[step, li]] += 1.0
        self._order = None

    def rank(self, step, target, horizon):
        if self._order is None:
            self._order = np.argsort(-self.scores, axis=1, kind="stable")[:, : self.depth]
        return self._order[target]


class CrossLayer(Predictor):
    """Co-occurrence from the issue point's routes to the target's, fitted on training steps.

    Score of target expert ``j`` = sum over the source's six experts ``i`` of count(i at source, j at
    target), ties to training popularity. The source is ``source_of``: the same token ``h`` layers back,
    or, past layer 0, the previous token's tail.

    With ``min_prob`` the score is instead the best single-source estimate max_i P(j at target | i at
    source) (sources seen at least ``min_support`` times in training), and only experts scoring at least
    ``min_prob`` are candidates: a confidence gate, so a layer may get no prefetch at all.
    """

    def __init__(
        self,
        stream: DecodeStream,
        train: np.ndarray,
        horizon: int,
        depth: int = 48,
        min_prob: float = 0.0,
        min_support: int = 5,
    ) -> None:
        self.stream, self.horizon, self.depth, self.min_prob = stream, horizon, depth, min_prob
        self.name = "cross_layer" if not min_prob else f"cross_layer_p{min_prob:g}"
        layers = len(stream.layers)
        self.counts = np.zeros((layers, NUM_EXPERTS, NUM_EXPERTS), dtype=np.float32)
        self.support = np.zeros((layers, NUM_EXPERTS), dtype=np.float32)
        popularity = np.zeros((layers, NUM_EXPERTS), dtype=np.float64)
        for step in np.flatnonzero(train):
            for target in range(layers):
                popularity[target, stream.routes[step, target]] += 1
                source = source_of(stream, step, target, horizon)
                if source is None:
                    continue
                src = stream.routes[source[0], source[1]]
                dst = stream.routes[step, target]
                self.counts[target][np.ix_(src, dst)] += 1.0
                self.support[target, src] += 1.0
        self.tiebreak = (popularity / (popularity.max() + 1.0) * 1e-3).astype(np.float32)
        if min_prob:
            enough = self.support >= min_support
            self.prob = np.where(enough[:, :, None], self.counts / np.maximum(self.support, 1.0)[:, :, None], 0.0)
            self.prob = self.prob.astype(np.float32)
            del self.counts

    def rank(self, step, target, horizon):
        source = source_of(self.stream, step, target, self.horizon)
        if source is None:
            if self.min_prob:
                return ()
            return np.argsort(-self.tiebreak[target], kind="stable")[: self.depth]
        src = self.stream.routes[source[0], source[1]]
        if self.min_prob:
            score = self.prob[target][src].max(axis=0)
            chosen = np.flatnonzero(score >= self.min_prob)
            return chosen[np.argsort(-score[chosen], kind="stable")]
        score = self.counts[target][src].sum(axis=0) + self.tiebreak[target]
        top = np.argpartition(-score, self.depth)[: self.depth]
        return top[np.argsort(-score[top], kind="stable")]


# ------------------------------------------------------------------ one arm


class ArmResult(msgspec.Struct):
    """Per decode step and layer: demand misses, prefetched rows, and prefetched rows the same forward
    routed (immediately useful). ``[steps, layers]`` each."""

    predictor: str
    k: int
    horizon: int
    demand: np.ndarray
    prefetch: np.ndarray
    useful: np.ndarray


def run_arm(
    loaded: dict,
    stream: DecodeStream,
    predictor: Predictor,
    k: int,
    horizon: int,
    capacity: Optional[dict[int, int]] = None,
) -> ArmResult:
    """Replay every forward (prefill included, as replay_direct does) with prefetch on decode steps."""
    capacity = capacity or loaded["hot_capacity"]
    initial = {layer: list(range(slots)) for layer, slots in capacity.items()}
    sim = PrefetchReplay(initial, capacity, NUM_EXPERTS)
    layers = stream.layers
    index = {layer: li for li, layer in enumerate(layers)}
    predictor.bind(sim, index)
    steps = len(stream.rids)
    demand = np.zeros((steps, len(layers)), dtype=np.int64)
    prefetch = np.zeros_like(demand)
    useful = np.zeros_like(demand)
    step = 0
    for forward in loaded["forwards"]:
        if forward["phase"] == "capture":
            continue
        if forward["kind"] != "graph":
            sim.eager_forward(forward["tokens"], forward["counts"], forward["phase"])
            continue
        if forward["phase"] != "decode":
            sim.graph_forward(forward["routes"], forward["phase"])
            continue
        current = step

        def candidates(layer: int, current=current) -> Sequence[int]:
            target = index[layer]
            if source_of(stream, current, target, horizon) is None:
                return ()
            return predictor.rank(current, target, horizon)

        misses = sim.graph_forward(forward["routes"], "decode", candidates, k)
        for layer, count in misses.items():
            li = index[layer]
            demand[step, li] = count
            rows = sim.last_prefetched.get(layer, [])
            prefetch[step, li] = len(rows)
            routed = set(forward["routes"][layer])
            useful[step, li] = sum(1 for row in rows if row in routed)
        predictor.observe(step)
        step += 1
    if step != steps:
        raise ValueError(f"replayed {step} decode steps, the stream has {steps}")
    return ArmResult(predictor.name, k, horizon, demand, prefetch, useful)
