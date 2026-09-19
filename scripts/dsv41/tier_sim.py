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


def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


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
        totals["decode_tokens"] += 1
        totals["vram"] += vram
        totals["ram"] += ram
        if since_prefill is None or since_prefill >= warmup:
            totals["late_vram"] += vram
            totals["late_ram"] += ram
        if since_prefill is not None:
            since_prefill += 1
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
