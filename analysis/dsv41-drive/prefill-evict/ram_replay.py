"""Replay a graph route capture through the C++ RAM tier's victim rule, with three prefill admission policies.

Per streamed layer the pinned tier is RamTier (exl3_ram_miss_host.cpp): a free slot first, else the unleased READY
slot with the lowest stamp that is neither VRAM-hot nor wanted by the request. A decode forward stamps every routed
expert the tier holds and admits the rest (the RAM misses). An eager prefill forward goes through
``iter_gather_experts`` chunks of 64 distinct experts; each chunk's VRAM misses reach ``gather_rows``, which touches
the tier's hits and admits the misses with the chunk protected. The prefill policies:

- ``base``: today's rule, prefill admits at the MRU end.
- ``cold``: prefill admits with stamp 0, below every decode stamp (SGLANG_DSV41_PREFILL_COLD_ADMIT).
- ``noadmit``: prefill misses are staged outside the tier and never enter it (option (a)).

``--no-touch`` also stops prefill hits from stamping. VRAM hot sets are the logged ones (graph forwards log the set
each layer held as the forward started; an eager forward uses the next graph forward's). Leases are not modelled:
between forwards every lease has retired.

Reports decode RAM misses per token by steps since the session's prefill, next to what the run measured
(graph_step ``layer_ram_rows``, aligned to the decode forwards by their VRAM miss counts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
from tier_sim import load_forwards, ram_rows_per_layer  # noqa: E402

CHUNK = 64
BUCKETS = ((0, 5), (5, 15), (15, 70), (70, 10**9))


class Tier:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.slot_expert = [-1] * capacity
        self.stamp = [0] * capacity
        self.where: dict[int, int] = {}

    def take(self, hot: set, protect: set) -> int:
        best = -1
        for slot, expert in enumerate(self.slot_expert):
            if expert < 0:
                return slot
            if expert in hot or expert in protect:
                continue
            if best < 0 or self.stamp[slot] < self.stamp[best]:
                best = slot
        if best < 0:
            raise RuntimeError("no victim")
        del self.where[self.slot_expert[best]]
        self.slot_expert[best] = -1
        return best

    def admit(self, expert: int, stamp: int, hot: set, protect: set) -> None:
        slot = self.take(hot, protect)
        self.slot_expert[slot] = expert
        self.stamp[slot] = stamp
        self.where[expert] = slot


class Replay:
    def __init__(self, capacity: list[int], policy: str, touch: bool):
        self.tiers = [Tier(c) for c in capacity]
        self.policy, self.touch, self.tick = policy, touch, 0

    def _next(self) -> int:
        self.tick += 1
        return self.tick

    def decode(self, layer: int, routes: list[int], hot: set) -> int:
        tier, wanted = self.tiers[layer], set(routes)
        missing = []
        for expert in dict.fromkeys(routes):
            slot = tier.where.get(expert)
            if slot is None:
                missing.append(expert)
            else:
                tier.stamp[slot] = self._next()
        for expert in missing:
            tier.admit(expert, self._next(), hot, wanted)
        return len(missing)

    def prefill(self, layer: int, experts: list[int], hot: set) -> int:
        tier, misses = self.tiers[layer], 0
        for start in range(0, len(experts), CHUNK):
            chunk = [e for e in experts[start : start + CHUNK] if e not in hot]
            missing = []
            for expert in chunk:
                slot = tier.where.get(expert)
                if slot is None:
                    missing.append(expert)
                elif self.touch:
                    tier.stamp[slot] = self._next()
            misses += len(missing)
            if self.policy == "noadmit":
                continue
            protect = set(chunk)
            for expert in missing:
                tier.admit(expert, 0 if self.policy == "cold" else self._next(), hot, protect)
        return misses


def measured_ram_rows(path: str) -> list[tuple[int, int]]:
    rows = []
    with open(path) as f:
        for text in f:
            if '"graph_step"' in text:
                line = json.loads(text)
                rows.append((int(line["vram_miss"]), int(line["ram_miss"])))
    return rows


def run(loaded: dict, capacity: list[int], policy: str, touch: bool) -> dict:
    replay = Replay(capacity, policy, touch)
    forwards = [f for f in loaded["forwards"] if f["phase"] != "capture"]
    next_hot: list = [None] * len(forwards)
    upcoming = None
    for i in range(len(forwards) - 1, -1, -1):
        next_hot[i] = upcoming
        if forwards[i]["kind"] == "graph" and forwards[i].get("hot") is not None:
            upcoming = forwards[i]["hot"]
    steps, since, prefill_misses = [], None, 0
    for i, forward in enumerate(forwards):
        if forward["kind"] == "graph":
            hot = forward["hot"]
            misses = sum(replay.decode(layer, forward["routes"][layer], set(hot[layer])) for layer in forward["routes"])
            if forward["phase"] == "decode":
                steps.append((since, misses, sum(forward["misses"].values())))
                since = None if since is None else since + 1
        else:
            hot = next_hot[i] or {}
            prefill_misses += sum(
                replay.prefill(layer, list(experts), set(hot.get(layer, ()))) for layer, (experts, _) in forward["counts"].items()
            )
            if forward["phase"] != "decode":
                since = 0
    return {"steps": steps, "prefill_ram_misses": prefill_misses}


def bucketize(values: list[tuple]) -> dict:
    out = {}
    for lo, hi in BUCKETS:
        part = [v for s, v in values if s is not None and lo <= s < hi]
        out[f"{lo}-{min(hi, 999) - 1}"] = (sum(part) / len(part) if part else None, len(part))
    everything = [v for _, v in values]
    out["all"] = (sum(everything) / len(everything), len(everything))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("--ram-rows", type=int, default=8063)
    p.add_argument("--num-experts", type=int, default=384)
    args = p.parse_args()
    loaded = load_forwards(args.trace)
    layers = len(loaded["layer_ids"])
    capacity = ram_rows_per_layer(args.ram_rows, layers, args.num_experts)
    report = {"trace": args.trace, "ram_rows": args.ram_rows, "arms": {}}
    measured = measured_ram_rows(args.trace)
    for policy in ("base", "cold", "noadmit"):
        for touch in (True, False):
            result = run(loaded, capacity, policy, touch)
            name = policy + ("" if touch else "-notouch")
            report["arms"][name] = {
                "decode_ram_misses_per_token": bucketize([(s, m) for s, m, _ in result["steps"]]),
                "prefill_ram_misses": result["prefill_ram_misses"],
            }
            if name == "base":
                # Align graph_step registers to decode forwards by VRAM miss count (the registers lag the route log).
                steps, aligned, j = result["steps"], [], 0
                for vram, ram in measured:
                    k = j
                    while k < len(steps) and steps[k][2] != vram:
                        k += 1
                    if k == len(steps):
                        continue
                    aligned.append((steps[k][0], ram))
                    j = k + 1
                report["measured"] = {"aligned_steps": len(aligned), "decode_ram_misses_per_token": bucketize(aligned)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
