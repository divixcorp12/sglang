"""Replay an EXL3 expert-stream trace through the expert framework's own tier policies.

G = VRAM misses per decode token, f = the share of those that also miss RAM
(DSV41_REFERENCE §9.4). The simulated tiers are the framework's, per layer:

- RAM is the pinned host tier. ExpertPinnedHostCacheManager splits the rows
  evenly over layers, and each layer is a PinnedSlotLRU. It is inclusive, as
  the EXL3 format asks (spec §9.1): its ``is_pinned`` protects every expert the
  layer's hot cache holds. A VRAM miss looks RAM up (hit: touch; miss: shard
  read and admit). Per call, the experts go through it in chunks of
  ``max_gather_rows`` as ExpertPinnedHostCache.gather_rows does: a chunk's hits
  are touched first, then its misses are admitted with the chunk protected.
  Promotions and the startup residency only admit the rows RAM lacks
  (ensure_rows) and never touch a row it holds.
- VRAM is the hot cache. Its startup allocation copies
  ExpertHotCacheManager.from_model: global slots go to the (layer, expert)
  pairs with the highest seed count (ties: lower expert, then lower layer),
  and an inclusive layer holds at most ``ram rows - max_gather_rows`` of them
  (the framework's clamp; the rest of the budget goes to other layers).
  With dynamic residency, every call records its routes into the layer's
  ExpertResidencyPolicy, a ResidencyBoundaryClock decides boundaries, and at a
  boundary every policy advances and the layers past min_residence_forwards
  take their decided set. A miss does not insert: eager gathers stage it.

``prefill_admits=False`` is a simulation-only arm, not a framework policy:
prefill misses read the shards without entering (or reordering) RAM. It shows
how much a 512-token prefill, which touches most of a layer's experts in
ascending id order, flushes the per-layer RAM tier before decode.

Only decode calls (one token row) count toward G and f; every call updates
the tiers. Promotions at decode boundaries are reported per decode token.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np
import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_MAX_GATHER_ROWS
from sglang.srt.layers.moe.expert_host_tier import PinnedSlotLRU
from sglang.srt.layers.moe.expert_residency import (
    ExpertResidencyPolicy,
    advance_residency_policies,
    decide_residency_policies,
)
from sglang.srt.layers.moe.expert_residency_clock import (
    ForwardKind,
    ResidencyBoundaryClock,
)

LINK_MS = 1.11
NVME_MS = {"nvme2": 7.0, "x4": 3.4}
# Exl3ExpertFormat.max_gather_rows: the rows an eager gather needs beside the hot rows.
MAX_GATHER_ROWS = EXL3_MAX_GATHER_ROWS


# Lines that are not forward calls: stage timing, and the graph route log that load_forwards reads.
_NOT_CALLS = ("ram_miss_request", "graph_routes", "graph_routes_header")


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    # Only forward calls and graph steps.
    return [line for line in lines if line.get("kind") not in _NOT_CALLS]


def load_forwards(path: str, *, allow_dropped: bool = False) -> dict:
    """Every serving forward of a trace with graph routes, in execution order.

    A replayed decode graph writes one ``graph_routes`` line per forward (GraphRouteLog): its routed
    experts per streamed layer, the misses its gathers found and, when logged, the hot set each layer
    held as it started. Eager forwards (prefill) write their per-layer calls, each stamped with
    ``graph_seq``, the graph forwards that ran before it. ``phase`` is the scheduler's forward mode
    where the trace has it (``decode``, ``extend``...); only a trace without it falls back to one token
    meaning decode. A run whose reader lost entries (``dropped_before``) is refused unless
    ``allow_dropped``: its replay would silently miss accesses.

    Returns ``run``, ``layer_ids``, ``hot_capacity`` per layer, ``hot_layer_ids``, ``dropped`` and
    ``forwards``, each ``{"kind": "graph"|"eager", "phase", "tokens", "rids", "forward_pass_id",
    "misses": {layer: n}}`` plus, for a graph forward, ``seq``, ``routes: {layer: [ids]}`` and ``hot:
    {layer: [ids]}`` (or None), and for an eager one ``forward`` and ``counts: {layer: (ids, counts)}``.
    """
    headers = []
    graph, eager, dropped = [], {}, 0
    with open(path) as f:
        for text in f:
            if not text.strip():
                continue
            line = json.loads(text)
            kind = line.get("kind")
            if kind == "graph_routes_header":
                headers.append(line)
            elif kind == "graph_routes":
                graph.append(line)
                dropped += line.get("dropped_before", 0)
            elif kind is None and "tokens" in line:
                eager.setdefault(line["forward"], []).append(line)
    if not headers:
        raise ValueError(f"{path} has no graph route log (a trace from before GraphRouteLog, or trace off)")
    if len({header.get("run") for header in headers}) != 1:
        raise ValueError(f"{path} holds more than one run's route log; split it by run first")
    header = headers[0]
    if dropped and not allow_dropped:
        raise ValueError(f"{path}: the route reader lost {dropped} graph forwards; the run cannot be replayed")
    layer_ids = header["layer_ids"]
    hot_layer_ids = header.get("hot_layer_ids") or []
    events = []
    for line in graph:
        hot = line.get("hot")
        events.append(((line["seq"], 1), {
            "kind": "graph",
            "seq": line["seq"],
            "phase": line.get("phase", "decode"),
            "tokens": line.get("forward_tokens", 1),
            "rids": line.get("rids", []),
            "forward_pass_id": line.get("forward_pass_id"),
            "routes": dict(zip(layer_ids, line["routes"])),
            "misses": dict(zip(layer_ids, line["misses"])),
            "hot": dict(zip(hot_layer_ids, hot)) if hot is not None else None,
        }))
    for forward, calls in eager.items():
        seqs = {call.get("graph_seq") for call in calls}
        if None in seqs or len(seqs) != 1:
            raise ValueError(f"eager forward {forward} lacks a single graph_seq stamp")
        first = calls[0]
        tokens = first["tokens"]
        events.append(((seqs.pop(), 0, forward), {
            "kind": "eager",
            "forward": forward,
            "phase": first.get("phase", "decode" if tokens == 1 else "extend"),
            "tokens": tokens,
            "rids": first.get("rids", []),
            "forward_pass_id": first.get("forward_pass_id"),
            "counts": {call["layer"]: (call["experts"], call["counts"]) for call in calls},
            "misses": {call["layer"]: call["vram_miss"] for call in calls},
        }))
    events.sort(key=lambda event: event[0])
    seqs = [line["seq"] for line in graph]
    if len(set(seqs)) != len(seqs):
        raise ValueError("a graph forward was logged twice")
    passes = [event["forward_pass_id"] for _, event in events if event["forward_pass_id"] not in (None, -1)]
    if passes != sorted(passes):
        raise ValueError("forward pass ids are out of execution order: the pre-forward stamp and the ring disagree")
    return {
        "run": header.get("run"),
        "schema": header.get("schema", 0),
        "layer_ids": layer_ids,
        "hot_capacity": dict(zip(layer_ids, header["hot_capacity"])),
        "hot_layer_ids": hot_layer_ids,
        "dropped": dropped,
        "forwards": [event for _, event in events],
    }


def direct_hot_allocation(
    slots: int, layer_ids: list[int], num_experts: int, floor: int, seed: Optional[np.ndarray] = None
) -> dict[int, list[int]]:
    """ExpertHotCacheManager.from_model's startup residency under stage DIRECT, equal-size rows, no clamp.

    Every layer first takes ``floor`` (twice its graph-gather rows) of its own best experts; the rest
    of the budget goes by seed count, ties to the lower expert, then the lower layer. Without a seed
    that is experts 0, 1, ... in every layer, and the remainder lands on the lowest layer ids.
    """
    candidates = sorted(
        (-(float(seed[layer, expert]) if seed is not None else 1.0), expert, layer)
        for layer in layer_ids
        for expert in range(num_experts)
    )
    chosen: dict[int, list[int]] = {layer: [] for layer in layer_ids}
    used = 0
    for floor_pass in (True, False):
        for _, expert, layer in candidates:
            if used >= slots:
                break
            if expert in chosen[layer] or (floor_pass and len(chosen[layer]) >= floor):
                continue
            chosen[layer].append(expert)
            used += 1
    if any(len(experts) < floor for experts in chosen.values()):
        raise ValueError(f"{slots} slots cannot give every layer its floor of {floor}")
    return chosen


class DirectInsertReplay:
    """The running recipe's residency, stage DIRECT insert-on-miss (GpuResidencyUpdater), per layer.

    Each graph decode forward applies the boundary its predecessor left pending, then every layer's
    gather sends its misses into the layer's victim shortlist: the ``miss_rows`` resident slots ranked
    lowest at that boundary by (routed in the closing window, insert score, -expert), less any slot this
    forward hits. Insert scores decay by ``decay`` per token and add each window's route counts. An eager
    forward inserts nothing; it adds its route counts, and its first layer's first gather applies a
    decode boundary still pending (after that layer recorded its counts); a prefill of at least
    ``update_prefill_tokens`` tokens is a boundary of its own. The clocks are the device mirror of
    ResidencyBoundaryClock at SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1. Float32 throughout, as on the
    device, so ties break the same way.
    """

    def __init__(
        self,
        initial: dict[int, list[int]],
        capacity: dict[int, int],
        num_experts: int,
        *,
        decay: float = 0.98,
        update_prefill_tokens: int = 256,
        miss_rows: int = 6,
    ) -> None:
        self.layer_ids = sorted(capacity)
        self.row = {layer: row for row, layer in enumerate(self.layer_ids)}
        self.num_experts, self.miss_rows = num_experts, miss_rows
        self.decay, self.update_prefill_tokens = decay, update_prefill_tokens
        layers = len(self.layer_ids)
        self.slots = [[-1] * capacity[layer] for layer in self.layer_ids]
        self.where = np.full((layers, num_experts), -1, dtype=np.int64)
        for layer in self.layer_ids:
            row = self.row[layer]
            for slot, expert in enumerate(initial[layer][: capacity[layer]]):
                self.slots[row][slot] = expert
                self.where[row, expert] = slot
        self.scores = np.zeros((layers, num_experts), dtype=np.float32)
        self.route_counts = np.zeros((layers, num_experts), dtype=np.float32)
        self.routed = np.zeros((layers, num_experts), dtype=bool)
        self.tokens = 0
        self.decode_forwards = 0
        self.boundary_pending = False
        self.host_pending = False
        self.truncated = 0
        self.shortlist = [self._rank(row) for row in range(layers)]  # reset_after_capture

    def _decay(self, tokens: int) -> np.float32:
        return np.float32(self.decay**tokens) if self.decay < 1.0 else np.float32(1.0)

    def _rank(self, row: int) -> list[int]:
        slots, scores, routed = self.slots[row], self.scores[row], self.routed[row]
        free = [slot for slot, expert in enumerate(slots) if expert < 0]
        held = sorted(
            (slot for slot, expert in enumerate(slots) if expert >= 0),
            key=lambda slot: (routed[slots[slot]], scores[slots[slot]], -slots[slot]),
        )
        return (free + held)[: self.miss_rows]

    def _apply(self) -> None:
        self.scores = self.scores * self._decay(self.tokens) + self.route_counts
        self.routed = self.route_counts > 0
        self.route_counts = np.zeros_like(self.route_counts)
        self.shortlist = [self._rank(row) for row in range(len(self.layer_ids))]
        self.tokens = 0
        self.decode_forwards = 0
        self.boundary_pending = False

    def _count(self, tokens: int, decode: bool) -> None:
        self.tokens += tokens
        if decode:
            self.decode_forwards += 1
            self.boundary_pending = self.boundary_pending or self.decode_forwards >= 1

    def _flush(self) -> None:
        if self.host_pending:
            self.host_pending = False
            if self.boundary_pending:
                self._apply()

    def graph_forward(self, routes: dict[int, list[int]], phase: str = "decode") -> dict[int, int]:
        """One forward served by the graph gather (a replay, or a one-token extend run eagerly through
        it); returns each layer's misses (its plan count)."""
        if self.boundary_pending:
            self._apply()
        self._count(1, decode=True)  # on_graph_forward counts every graph-served forward as decode
        misses = {}
        for layer, experts in routes.items():
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
        # observe_forward(graph_served=True): a decode is a boundary at interval 1; a graph-served
        # prefill is taken back out of the decode count and is a boundary only past the prefill threshold.
        self.host_pending = phase == "decode"
        if phase != "decode":
            self.decode_forwards -= 1
            self.boundary_pending = self.decode_forwards >= 1
        return misses

    def eager_forward(
        self, tokens: int, counts: dict[int, tuple[list[int], list[int]]], phase: Optional[str] = None
    ) -> dict[int, int]:
        """One eager forward (prefill, or a decode the graph did not serve); returns its misses per layer."""
        misses = {}
        for index, layer in enumerate(sorted(counts)):
            experts, weights = counts[layer]
            row = self.row[layer]
            np.add.at(self.route_counts[row], experts, np.asarray(weights, dtype=np.float32))
            if index == 0:  # streamers[0].before_eager_gather runs after that layer's record_routes
                self._flush()
            misses[layer] = sum(1 for e in experts if self.where[row, e] < 0)
        decode = phase == "decode" if phase is not None else tokens == 1
        self._flush()
        self._count(tokens, decode=decode)
        if not decode and tokens >= self.update_prefill_tokens:
            self._apply()
        elif decode:
            self.host_pending = True
        return misses

    def resident(self, layer: int) -> set[int]:
        return {e for e in self.slots[self.row[layer]] if e >= 0}


def replay_direct(
    loaded: dict,
    *,
    capacity: Optional[dict[int, int]] = None,
    num_experts: int = 384,
    initial: Optional[dict[int, list[int]]] = None,
    **options,
) -> dict:
    """Replay ``load_forwards`` output through DirectInsertReplay, next to what the run measured.

    ``capacity`` defaults to the run's own hot slots per layer and ``initial`` to the framework's
    unseeded startup residency (experts 0..capacity-1). Returns decode tokens, simulated and measured
    decode misses in total and per decode forward, and, where the trace logged hot sets, how many
    (graph forward, layer) pairs started with a different hot set than the replay's.
    """
    capacity = capacity or loaded["hot_capacity"]
    if initial is None:
        initial = {layer: list(range(slots)) for layer, slots in capacity.items()}
    sim = DirectInsertReplay(initial, capacity, num_experts, **options)
    out = {"decode_tokens": 0, "vram_misses": 0, "measured_vram_misses": 0, "per_forward": [],
           "hot_checked": 0, "hot_mismatched": 0, "first_hot_mismatch": None}
    for forward in loaded["forwards"]:
        if forward["kind"] == "graph":
            if forward.get("hot") is not None:
                for layer, experts in forward["hot"].items():
                    out["hot_checked"] += 1
                    if set(experts) != sim.resident(layer):
                        out["hot_mismatched"] += 1
                        if out["first_hot_mismatch"] is None:
                            out["first_hot_mismatch"] = (forward["seq"], layer)
            simulated = sim.graph_forward(forward["routes"], forward["phase"])
        else:
            simulated = sim.eager_forward(forward["tokens"], forward["counts"], forward["phase"])
        if forward["phase"] != "decode":
            continue
        out["decode_tokens"] += 1
        out["vram_misses"] += sum(simulated.values())
        out["measured_vram_misses"] += sum(forward["misses"].values())
        out["per_forward"].append((forward.get("seq"), simulated, forward["misses"]))
    tokens = out["decode_tokens"] or 1
    out["G"] = out["vram_misses"] / tokens
    out["measured_G"] = out["measured_vram_misses"] / tokens
    out["truncated"] = sim.truncated
    return out


def ram_rows_per_layer(rows: int, num_layers: int, num_experts: int) -> list[int]:
    """The pinned tier's even split, as ExpertPinnedHostCacheManager.from_model makes it."""
    return [
        min(num_experts, rows // num_layers + (1 if layer < rows % num_layers else 0))
        for layer in range(num_layers)
    ]


def select_hot(
    seed: Optional[np.ndarray],
    slots: int,
    num_layers: int,
    num_experts: int,
    limits: Optional[list[int]] = None,
) -> dict[int, list[int]]:
    """Startup hot residency, in the order ExpertHotCacheManager.from_model picks it.

    ``limits[layer]`` caps a layer's slots (the inclusive clamp); the budget a
    capped layer cannot use goes to the next candidates.
    """
    candidates = sorted(
        (-(float(seed[layer, expert]) if seed is not None else 1.0), expert, layer)
        for layer in range(num_layers)
        for expert in range(num_experts)
    )
    chosen: dict[int, list[int]] = {layer: [] for layer in range(num_layers)}
    used = 0
    for _, expert, layer in candidates:
        if used >= slots:
            break
        if limits is not None and len(chosen[layer]) >= limits[layer]:
            continue
        chosen[layer].append(expert)
        used += 1
    return chosen


def _forwards(calls):
    group: list[dict] = []
    for call in calls:
        if group and call["forward"] != group[0]["forward"]:
            yield group
            group = []
        group.append(call)
    if group:
        yield group


def simulate(
    calls,
    *,
    num_layers: int,
    num_experts: int,
    vram_slots: int,
    ram_slots: int,
    seed: Optional[np.ndarray] = None,
    dynamic: bool = True,
    update_prefill_tokens: int = 1024,
    update_decode_forwards: int = 0,
    min_residence_forwards: int = 8,
    benefit_ratio: float = 1.0,
    decay_tokens: int = 0,
    promotion_sigmas: float = 0.0,
    max_gather_rows: int = MAX_GATHER_ROWS,
    prefill_admits: bool = True,
) -> dict:
    # Graph decode steps carry no per-layer routes; only live_summary reads them.
    calls = [call for call in calls if call.get("kind") != "graph_step"]
    ram_rows = ram_rows_per_layer(ram_slots, num_layers, num_experts)
    limits = [max(rows - max_gather_rows, 0) if rows else num_experts for rows in ram_rows]
    hot = select_hot(seed, vram_slots, num_layers, num_experts, limits)
    resident = {layer: set(experts) for layer, experts in hot.items()}
    policies = {}
    if dynamic:
        for layer, experts in hot.items():
            if experts:
                policies[layer] = ExpertResidencyPolicy(
                    num_experts,
                    len(experts),
                    promotion_margin=benefit_ratio,
                    decay_tokens=decay_tokens or None,
                    promotion_sigmas=promotion_sigmas,
                    initial_scores=None if seed is None else seed[layer],
                )
    ram = {
        layer: PinnedSlotLRU(rows, is_pinned=resident[layer].__contains__) if rows else None
        for layer, rows in enumerate(ram_rows)
    }

    def evictable_rows(layer: int) -> int:
        """ExpertPinnedHostCache.evictable_rows: capacity minus the rows is_pinned protects."""
        tier = ram[layer]
        return tier.capacity - sum(1 for expert in tier.expert_to_slot if expert in resident[layer])

    def ram_gather(layer: int, experts: list[int], admit: bool) -> int:
        """ExpertPinnedHostCache.gather_rows: RAM misses of one hot-miss set.

        Each chunk is sized from the evictable rows when it starts; its hits are
        touched first, then its misses are admitted with the whole chunk protected.
        """
        tier = ram[layer]
        if tier is None:
            return len(experts)
        if not admit:
            return sum(1 for expert in experts if expert not in tier)
        misses = 0
        start = 0
        while start < len(experts):
            room = evictable_rows(layer)
            rows = len(experts) - start
            if room >= 1 and rows > room and len(set(experts[start:])) > room:
                rows = room
            chunk = experts[start : start + rows]
            hits = [expert for expert in chunk if expert in tier]
            for expert in hits:
                tier.touch(expert)
            missing = [expert for expert in chunk if expert not in tier]
            if missing:
                if room < 1:
                    raise RuntimeError("every pinned host slot holds a protected expert")
                misses += len(missing)
                protected = frozenset(chunk)
                for expert in missing:
                    tier.assign(expert, protected)
            start += rows
        return misses

    def ram_ensure(layer: int, experts) -> int:
        """ExpertHotCacheManager._load_reserved_in_chunks: admit rows, touching none.

        Chunks hold at most the evictable rows read when they start; each goes
        through ``ensure_rows``, which assigns only the experts the tier lacks.
        Returns how many it had to read.
        """
        experts = list(experts)
        tier = ram[layer]
        if tier is None:
            return len(experts)
        misses = 0
        start = 0
        while start < len(experts):
            room = evictable_rows(layer)
            if room < 1:
                raise RuntimeError("the pinned host tier has no evictable slots for hot cache promotions")
            chunk = experts[start : start + room]
            protected = frozenset(chunk)
            for expert in chunk:
                if expert not in tier:
                    misses += 1
                    tier.assign(expert, protected)
            start += len(chunk)
        return misses

    for layer, experts in hot.items():
        ram_ensure(layer, experts)

    clock = ResidencyBoundaryClock(update_prefill_tokens, update_decode_forwards, enabled=dynamic)
    last_update = {layer: 0 for layer in policies}
    out = {
        "decode_tokens": 0,
        "vram_misses": 0,
        "ram_misses": 0,
        "prefill_vram_misses": 0,
        "prefill_ram_misses": 0,
        "boundaries": 0,
        "decode_promotion_rows": 0,
        "decode_promotion_ram_misses": 0,
        "prefill_promotion_rows": 0,
        "prefill_promotion_ram_misses": 0,
        "hot_slots": sum(len(experts) for experts in hot.values()),
        "clamped_layers": sum(
            1 for layer in range(num_layers) if len(hot[layer]) >= limits[layer] and ram_rows[layer]
        ),
    }
    for forward in _forwards(calls):
        tokens = forward[0]["tokens"]
        decode = tokens == 1
        out["decode_tokens"] += int(decode)
        phase = "" if decode else "prefill_"
        for call in forward:
            layer = call["layer"]
            policy = policies.get(layer)
            if policy is not None:
                counts = torch.zeros(num_experts)
                counts[torch.tensor(call["experts"], dtype=torch.long)] = torch.tensor(
                    call["counts"], dtype=torch.float32
                )
                policy.record_counts(counts)
            # iter_gather_experts: chunks of max_gather_rows over the call's distinct
            # experts; each chunk's hot misses go to the pinned tier together.
            experts = call["experts"]
            for start in range(0, len(experts), max_gather_rows):
                missing = [
                    expert
                    for expert in experts[start : start + max_gather_rows]
                    if expert not in resident[layer]
                ]
                out[f"{phase}vram_misses"] += len(missing)
                out[f"{phase}ram_misses"] += ram_gather(
                    layer, missing, admit=decode or prefill_admits
                )
        kind = ForwardKind.DECODE if decode else ForwardKind.PREFILL
        boundary = clock.observe(kind, tokens)
        if boundary is None or not policies:
            continue
        out["boundaries"] += 1
        advance_residency_policies(list(policies.values()), boundary)
        deciding = [
            layer
            for layer in policies
            if clock.forwards - last_update[layer] >= min_residence_forwards
        ]
        decisions = decide_residency_policies(
            [policies[layer] for layer in deciding],
            [tuple(sorted(resident[layer])) for layer in deciding],
        )
        for layer, decision in zip(deciding, decisions):
            if not decision.promotions and not decision.evictions:
                continue
            # An evicted expert keeps its RAM row, now unprotected; the new set is
            # protected while its promotions are admitted through RAM.
            resident[layer].clear()
            resident[layer].update(decision.desired_experts)
            boundary_phase = "decode" if decode else "prefill"
            out[f"{boundary_phase}_promotion_rows"] += len(decision.promotions)
            out[f"{boundary_phase}_promotion_ram_misses"] += ram_ensure(
                layer, decision.promotions
            )
            last_update[layer] = clock.forwards
    tokens = out["decode_tokens"]
    out["G"] = out["vram_misses"] / tokens if tokens else 0.0
    out["f"] = out["ram_misses"] / out["vram_misses"] if out["vram_misses"] else 0.0
    return out


def ms_per_token(row: dict, nvme_ms: float) -> float:
    """§9.4's model plus decode-boundary promotions, amortized per decode token."""
    tokens = row["decode_tokens"] or 1
    promotions = (
        row["decode_promotion_rows"] * LINK_MS
        + row["decode_promotion_ram_misses"] * nvme_ms
    ) / tokens
    return row["G"] * LINK_MS + row["f"] * row["G"] * nvme_ms + promotions


def live_summary(calls, warmup: int = 16) -> dict:
    """G and f as the run measured them (the trace's own miss counts).

    ``f_after_warmup`` leaves out each session's first ``warmup`` decode tokens:
    a session starts at a prefill forward, and right after one the RAM tier
    holds what the prefill touched last.
    """
    totals = {"decode_tokens": 0, "vram": 0, "ram": 0, "late_vram": 0, "late_ram": 0}
    since_prefill = None
    for forward in _forwards(calls):
        if forward[0]["tokens"] != 1:
            since_prefill = 0
            continue
        vram = sum(call["vram_miss"] for call in forward)
        ram = sum(call["ram_miss"] for call in forward)
        # A graph_step line may cover more than one decode step (its "steps").
        steps = int(forward[0].get("steps", 1))
        totals["decode_tokens"] += steps
        totals["vram"] += vram
        totals["ram"] += ram
        if since_prefill is None or since_prefill >= warmup:
            totals["late_vram"] += vram
            totals["late_ram"] += ram
        if since_prefill is not None:
            since_prefill += steps
    tokens = totals["decode_tokens"]
    return {
        "decode_tokens": tokens,
        "G": totals["vram"] / tokens if tokens else 0.0,
        "f": totals["ram"] / totals["vram"] if totals["vram"] else 0.0,
        "f_after_warmup": totals["late_ram"] / totals["late_vram"] if totals["late_vram"] else 0.0,
        "warmup_tokens": warmup,
    }


def seed_from_trace(calls, num_layers: int, num_experts: int) -> np.ndarray:
    """Decode route counts per (layer, expert), with multiplicity."""
    counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    for call in calls:
        if call["tokens"] == 1:
            for expert, count in zip(call["experts"], call["counts"]):
                counts[call["layer"], expert] += count
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("--vram-slots", type=int, nargs="+", default=[1220, 1290, 1540, 1770])
    p.add_argument("--ram-slots", type=int, nargs="+", default=[3000, 5630])
    p.add_argument("--update-decode-forwards", type=int, nargs="+", default=[0, 32])
    p.add_argument("--update-prefill-tokens", type=int, default=256)
    p.add_argument("--min-residence-forwards", type=int, default=8)
    p.add_argument("--benefit-ratio", type=float, default=1.0)
    p.add_argument("--max-gather-rows", type=int, default=MAX_GATHER_ROWS)
    p.add_argument("--static", action="store_true", help="no dynamic residency")
    p.add_argument("--seed", help="a {\"count\": [[...]]} JSON to seed the hot cache with")
    p.add_argument("--emit-seed", help="write the trace's decode route counts here as a hot seed")
    p.add_argument("--num-layers", type=int, default=40)
    p.add_argument("--num-experts", type=int, default=384)
    args = p.parse_args()
    calls = load_trace(args.trace)
    seed = None
    if args.seed:
        with open(args.seed) as f:
            seed = np.asarray(json.load(f)["count"], dtype=np.float64)
    rows = []
    for vram in args.vram_slots:
        for ram in args.ram_slots:
            for decode_forwards in args.update_decode_forwards:
                for prefill_admits in (True, False):
                    r = simulate(
                        calls,
                        num_layers=args.num_layers,
                        num_experts=args.num_experts,
                        vram_slots=vram,
                        ram_slots=ram,
                        seed=seed,
                        dynamic=not args.static,
                        update_prefill_tokens=args.update_prefill_tokens,
                        update_decode_forwards=decode_forwards,
                        min_residence_forwards=args.min_residence_forwards,
                        benefit_ratio=args.benefit_ratio,
                        max_gather_rows=args.max_gather_rows,
                        prefill_admits=prefill_admits,
                    )
                    r.update(
                        {
                            "vram_slots": vram,
                            "ram_slots": ram,
                            "update_decode_forwards": decode_forwards,
                            "prefill_admits": prefill_admits,
                            "policy": "framework" if prefill_admits else "simulation only",
                        }
                    )
                    for drive, t in NVME_MS.items():
                        r[f"ms_per_token_{drive}"] = ms_per_token(r, t)
                    rows.append(r)
    report = {
        "trace": args.trace,
        "calls": len(calls),
        "seeded": bool(args.seed),
        "live": (
            live_summary(calls)
            if calls and all("vram_miss" in c and "ram_miss" in c for c in calls)
            else None
        ),
        "rows": rows,
    }
    print(json.dumps(report, indent=2))
    if args.emit_seed:
        counts = seed_from_trace(calls, args.num_layers, args.num_experts)
        with open(args.emit_seed, "w") as f:
            json.dump({"count": counts.tolist()}, f)


if __name__ == "__main__":
    main()
