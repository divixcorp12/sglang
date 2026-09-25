"""VRAM hot-cache policies replayed over logged decode routes: misses per decode token against capacity.

Input: stage traces with the graph route log (tier_sim.load_forwards). Every policy sees the same decode
steps in execution order; the cache persists across requests, as in serving. A miss is a routed (layer,
expert) row not in VRAM when its layer runs: about 13.3 MB over PCIe, ~1 ms of decode time each.

Policies (``capacity`` is total slots; ~80 slots per GiB):

- ``current``: tier_sim.DirectInsertReplay, the running recipe (stage DIRECT insert-on-miss, per-token
  decayed insert scores, 0.98), with ExpertHotCacheManager's unseeded allocation (floor 12 per layer,
  the rest round-robin). It sees prefill routes as the framework does; every other policy sees decode only.
- ``current_alloc``: the same policy, with per-layer capacities chosen greedily from each layer's
  train-prompt miss curve under that policy (a startup allocation, the only change).
- ``lru`` / ``lru_global``: insert every miss, evict the least recently routed row, per layer (the
  framework's allocation) or from one pool over all layers (the split floats).
- ``lfu<d>`` / ``lfu<d>_global``: insert every miss, evict the lowest score, score decayed by ``d`` per
  decode step plus one per route.
- ``static`` / ``static_layer``: the train prompts' most routed (layer, expert) rows, fixed; one pool or
  the framework's per-layer split. No transfers after startup.
- ``tinylfu<E>`` (plan P2): per layer a 6-row demand window (a miss must land somewhere) plus an LRU
  main; a window row leaving competes with main's LRU row on exact per-layer route counts halved every
  ``E`` decode steps; the candidate must beat the victim, a tie keeps the incumbent.
- ``wtinylfu<E>`` (plan P3): the same window and admission, main a segmented LRU (probation, and a
  protected segment of 0.8 of main that a probation hit promotes into).
- ``opt_insert`` / ``opt_bypass`` (+ ``_global``): Belady. ``insert`` must place every miss in VRAM (as
  DIRECT does) and evicts the row routed furthest in the future; ``bypass`` may leave a miss out (it would
  need a scratch row, ~6 per the whole model, since layers run in turn). ``opt_bypass_global`` is the
  lower bound on misses for any policy that does not prefetch.

Held-out scoring: requests are split into train and test (even and odd prompts of the run); every policy
replays the whole stream, but only test steps are scored, and anything tuned or fitted (decay, epoch,
static set, allocation) is chosen on train steps.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "dsv41"))
import tier_sim  # noqa: E402

NUM_EXPERTS = 384
FLOOR = 12  # DIRECT: every layer holds twice its 6 graph-gather rows
WINDOW = 6  # a DIRECT miss needs a slot; 6 routes per layer per step
INF = 1 << 60


# ------------------------------------------------------------------ stream


def decode_stream(loaded: dict) -> dict:
    """The decode steps in execution order: per step, its request and each layer's routes."""
    layers = loaded["layer_ids"]
    steps, rids = [], []
    for forward in loaded["forwards"]:
        if forward["phase"] != "decode":
            continue
        if forward["kind"] == "graph":
            routes = [forward["routes"][layer] for layer in layers]
        else:
            routes = [list(forward["counts"][layer][0]) for layer in layers]
        steps.append(routes)
        rids.append(forward["rids"][0] if forward["rids"] else None)
    return {"layers": layers, "steps": steps, "rids": rids}


def split_requests(rids: list, prompts: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Train/test masks over steps: the run's last ``prompts`` requests are the driver's prompts in order
    (earlier ones are health checks); even prompts train, odd prompts test. Returns (train, test, rids)."""
    order = list(dict.fromkeys(rid for rid in rids))
    driver = order[-prompts:]
    prompt_of = {rid: index for index, rid in enumerate(driver)}
    test = np.array([rid in prompt_of and prompt_of[rid] % 2 == 1 for rid in rids])
    train = np.array([rid in prompt_of and prompt_of[rid] % 2 == 0 for rid in rids])
    return train, test, driver


def framework_capacity(total: int, layers: list) -> dict:
    alloc = tier_sim.direct_hot_allocation(total, layers, NUM_EXPERTS, FLOOR)
    return {layer: len(experts) for layer, experts in alloc.items()}


# ------------------------------------------------------------------ policies


def run_current(loaded: dict, capacity: dict) -> np.ndarray:
    out = tier_sim.replay_direct(loaded, capacity=capacity, num_experts=NUM_EXPERTS)
    return np.array([sum(sim.values()) for _, sim, _ in out["per_forward"]], dtype=np.int64)


def run_current_per_layer(loaded: dict, capacity: dict) -> np.ndarray:
    """[steps, layers] misses of the current policy (layers are independent under DIRECT)."""
    out = tier_sim.replay_direct(loaded, capacity=capacity, num_experts=NUM_EXPERTS)
    layers = loaded["layer_ids"]
    return np.array([[sim[layer] for layer in layers] for _, sim, _ in out["per_forward"]], dtype=np.int64)


class _Pools:
    """Residency pools: one per layer (``capacity`` a dict) or one for all (an int)."""

    def __init__(self, layers: list, capacity):
        self.global_pool = not isinstance(capacity, dict)
        self.capacity = capacity if self.global_pool else {layer: capacity[layer] for layer in layers}

    def pool(self, layer):
        return None if self.global_pool else layer

    def cap(self, layer):
        return self.capacity if self.global_pool else self.capacity[layer]


def run_lru(stream: dict, capacity) -> np.ndarray:
    pools = _Pools(stream["layers"], capacity)
    caches: dict = {}
    misses = np.zeros(len(stream["steps"]), dtype=np.int64)
    for step, routes in enumerate(stream["steps"]):
        for layer, experts in zip(stream["layers"], routes):
            cache = caches.setdefault(pools.pool(layer), OrderedDict())
            current = {(layer, e) for e in experts}
            for e in experts:
                key = (layer, e)
                if key in cache:
                    cache.move_to_end(key)
                    continue
                misses[step] += 1
                while len(cache) >= pools.cap(layer):
                    victim = next(k for k in cache if k not in current)
                    del cache[victim]
                cache[key] = True
    return misses


def run_lfu(stream: dict, capacity, decay: float) -> np.ndarray:
    """Insert every miss; evict the lowest decayed route score. Scores are kept inflated (each route adds
    decay**-step) so no decay pass runs; a min-heap with lazy entries finds the victim."""
    pools = _Pools(stream["layers"], capacity)
    score: dict = {}
    heaps: dict = {}
    resident: dict = {}
    misses = np.zeros(len(stream["steps"]), dtype=np.int64)
    log_inc = -math.log(decay)
    base = 0.0  # scores are exp(log_inc * step - base); renormalised by shifting base
    for step, routes in enumerate(stream["steps"]):
        inc = math.exp(log_inc * step - base)
        if inc > 1e200:
            factor = inc
            base += math.log(factor)
            for key in score:
                score[key] /= factor
            for pool, heap in heaps.items():
                heaps[pool] = [(s / factor, k) for s, k in heap]
                heapq.heapify(heaps[pool])
            inc = 1.0
        for layer, experts in zip(stream["layers"], routes):
            pool = pools.pool(layer)
            heap = heaps.setdefault(pool, [])
            members = resident.setdefault(pool, set())
            current = {(layer, e) for e in experts}
            for key in current:
                score[key] = score.get(key, 0.0) + inc
            for e in experts:
                key = (layer, e)
                if key in members:
                    heapq.heappush(heap, (score[key], key))
                    continue
                misses[step] += 1
                if len(members) >= pools.cap(layer):
                    held = []
                    while True:
                        s, victim = heapq.heappop(heap)
                        if victim not in members or s != score[victim]:
                            continue  # stale
                        if victim in current:
                            held.append((s, victim))
                            continue
                        break
                    for entry in held:
                        heapq.heappush(heap, entry)
                    members.discard(victim)
                members.add(key)
                heapq.heappush(heap, (score[key], key))
    return misses


def run_static(stream: dict, keep: set) -> np.ndarray:
    return np.array(
        [sum(1 for layer, experts in zip(stream["layers"], routes) for e in experts if (layer, e) not in keep)
         for routes in stream["steps"]],
        dtype=np.int64,
    )


def static_set(stream: dict, train: np.ndarray, capacity) -> set:
    counts: dict = {}
    for step in np.flatnonzero(train):
        for layer, experts in zip(stream["layers"], stream["steps"][step]):
            for e in experts:
                counts[(layer, e)] = counts.get((layer, e), 0) + 1
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    if not isinstance(capacity, dict):
        return set(ranked[:capacity])
    keep, used = set(), {layer: 0 for layer in stream["layers"]}
    for key in ranked:
        if used[key[0]] < capacity[key[0]]:
            keep.add(key)
            used[key[0]] += 1
    return keep


def run_tinylfu(stream: dict, capacity: dict, epoch: int, segmented: bool, protected: float = 0.8) -> np.ndarray:
    """Plan P2 (LRU main) or P3 (segmented main) per layer, 6-row window, exact aged counts."""
    misses = np.zeros(len(stream["steps"]), dtype=np.int64)
    state = {}
    for layer in stream["layers"]:
        main_cap = capacity[layer] - WINDOW
        state[layer] = {
            "counts": np.zeros(NUM_EXPERTS, dtype=np.int64),
            "window": OrderedDict(),
            "probation": OrderedDict(),
            "protected": OrderedDict(),
            "main_cap": main_cap,
            "protected_cap": int(protected * main_cap) if segmented else 0,
        }
    for step, routes in enumerate(stream["steps"]):
        for layer, experts in zip(stream["layers"], routes):
            st = state[layer]
            counts, window, probation, prot = st["counts"], st["window"], st["probation"], st["protected"]
            if step and step % epoch == 0:
                counts >>= 1
            for e in set(experts):
                counts[e] = min(counts[e] + 1, 65535)
            current = set(experts)
            for e in experts:
                if e in window:
                    window.move_to_end(e)
                    continue
                if e in prot:
                    prot.move_to_end(e)
                    continue
                if e in probation:
                    del probation[e]
                    if segmented:
                        prot[e] = True
                        while len(prot) > st["protected_cap"]:
                            demoted = next(iter(prot))
                            del prot[demoted]
                            probation[demoted] = True
                    else:
                        probation[e] = True
                    continue
                misses[step] += 1
                window[e] = True
                if len(window) <= WINDOW:
                    continue
                candidate = next(k for k in window if k not in current)
                del window[candidate]
                if len(probation) + len(prot) < st["main_cap"]:
                    probation[candidate] = True
                    continue
                victims = [k for k in probation if k not in current] or [k for k in prot if k not in current]
                if not victims:
                    continue  # nothing evictable in main: the candidate is dropped
                victim = victims[0]
                if counts[candidate] > counts[victim]:
                    (probation if victim in probation else prot).pop(victim)
                    probation[candidate] = True
    return misses


def next_uses(stream: dict, positions_global: bool) -> list:
    """For each step and layer, each route's next use (a step, or a global position step*L+layer)."""
    layers = stream["layers"]
    width = len(layers)
    nxt = [[None] * width for _ in stream["steps"]]
    seen: dict = {}
    for step in range(len(stream["steps"]) - 1, -1, -1):
        for li, (layer, experts) in enumerate(zip(layers, stream["steps"][step])):
            row = []
            for e in experts:
                row.append(seen.get((layer, e), INF))
            nxt[step][li] = row
            here = step * width + li if positions_global else step
            for e in experts:
                seen[(layer, e)] = here
    return nxt


def run_opt(stream: dict, capacity, bypass: bool) -> np.ndarray:
    """Belady: evict (or, with ``bypass``, decline to insert) the row whose next route is furthest."""
    pools = _Pools(stream["layers"], capacity)
    nxt = next_uses(stream, positions_global=pools.global_pool)
    resident: dict = {}
    heaps: dict = {}
    misses = np.zeros(len(stream["steps"]), dtype=np.int64)
    for step, routes in enumerate(stream["steps"]):
        for li, (layer, experts) in enumerate(zip(stream["layers"], routes)):
            pool = pools.pool(layer)
            members = resident.setdefault(pool, {})
            heap = heaps.setdefault(pool, [])
            missing = [e for e in experts if (layer, e) not in members]
            misses[step] += len(missing)
            for e, use in zip(experts, nxt[step][li]):  # hits: refresh next use first
                if (layer, e) in members:
                    members[(layer, e)] = use
                    heapq.heappush(heap, (-use, (layer, e)))
            current = {(layer, e) for e in experts}
            for e, use in zip(experts, nxt[step][li]):
                key = (layer, e)
                if key in members:
                    continue
                if len(members) >= pools.cap(layer):
                    held = []
                    while True:
                        negative, victim = heapq.heappop(heap)
                        if victim not in members or members[victim] != -negative:
                            continue
                        if victim in current:
                            held.append((negative, victim))
                            continue
                        break
                    if bypass and -negative <= use:
                        heapq.heappush(heap, (negative, victim))  # keep it: the new row returns later
                        for entry in held:
                            heapq.heappush(heap, entry)
                        continue
                    del members[victim]
                    for entry in held:
                        heapq.heappush(heap, entry)
                members[key] = use
                heapq.heappush(heap, (-use, key))
    return misses


def greedy_allocation(curves: dict, total: int, floor: int) -> dict:
    """Per-layer capacities maximising train misses saved, from each layer's miss curve (capacity -> misses)."""
    alloc = {layer: floor for layer in curves}
    budget = total - floor * len(curves)
    heap = []
    for layer, curve in curves.items():
        heapq.heappush(heap, (-(curve[floor] - curve.get(floor + 1, curve[floor])), layer))
    while budget > 0 and heap:
        _, layer = heapq.heappop(heap)
        c = alloc[layer] + 1
        if c not in curves[layer]:
            continue
        alloc[layer] = c
        budget -= 1
        nxt = curves[layer].get(c + 1)
        if nxt is not None:
            heapq.heappush(heap, (-(curves[layer][c] - nxt), layer))
    return alloc


# ------------------------------------------------------------------ driver

_LOADED: dict = {}


def _curve_point(c: int) -> np.ndarray:
    return run_current_per_layer(_LOADED, {layer: c for layer in _LOADED["layer_ids"]})



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--prompts", type=int, required=True, help="driver requests at the end of the run")
    p.add_argument("--capacities", type=int, nargs="+", default=[1128, 1208, 1288, 1368, 1448, 1528, 1608, 1688])
    p.add_argument("--decays", type=float, nargs="+", default=[0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    p.add_argument("--epochs", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--curve-span", type=int, default=60, help="per-layer capacities 12..12+span for current_alloc")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    loaded = tier_sim.load_forwards(args.trace)
    stream = decode_stream(loaded)
    train, test, driver = split_requests(stream["rids"], args.prompts)
    layers = stream["layers"]
    print(f"{len(stream['steps'])} decode steps, {len(driver)} driver requests; "
          f"{int(train.sum())} train / {int(test.sum())} test steps", flush=True)

    def score(misses: np.ndarray) -> dict:
        return {
            "all": float(misses.mean()),
            "train": float(misses[train].mean()) if train.any() else None,
            "test": float(misses[test].mean()) if test.any() else None,
        }

    results: dict = {}

    def put(name: str, capacity: int, misses: np.ndarray, **extra) -> None:
        results.setdefault(name, {})[capacity] = {**score(misses), **extra}
        print(f"  {name:>22} {capacity:5d}  all {misses.mean():6.2f}  test "
              f"{misses[test].mean() if test.any() else float('nan'):6.2f}", flush=True)

    # Current policy's per-layer miss curves on train steps, for current_alloc (layers are independent).
    global _LOADED
    _LOADED = loaded
    curve_caps = list(range(FLOOR, FLOOR + args.curve_span))
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        per_cap = pool.map(_curve_point, curve_caps)
    curves = {layer: {} for layer in layers}
    for c, per_layer in zip(curve_caps, per_cap):
        for li, layer in enumerate(layers):
            curves[layer][c] = int(per_layer[train, li].sum())

    for total in args.capacities:
        cap = framework_capacity(total, layers)
        put("current", total, run_current(loaded, cap))
        alloc = greedy_allocation(curves, total, FLOOR)
        put("current_alloc", total, run_current(loaded, alloc), allocation=alloc)
        put("lru", total, run_lru(stream, cap))
        put("lru_global", total, run_lru(stream, total))
        for decay in args.decays:
            put(f"lfu{decay}", total, run_lfu(stream, cap, decay))
            put(f"lfu{decay}_global", total, run_lfu(stream, total, decay))
        put("static", total, run_static(stream, static_set(stream, train, total)))
        put("static_layer", total, run_static(stream, static_set(stream, train, cap)))
        for epoch in args.epochs:
            put(f"tinylfu{epoch}", total, run_tinylfu(stream, cap, epoch, segmented=False))
            put(f"wtinylfu{epoch}", total, run_tinylfu(stream, cap, epoch, segmented=True))
        put("opt_insert", total, run_opt(stream, cap, bypass=False))
        put("opt_bypass", total, run_opt(stream, cap, bypass=True))
        put("opt_insert_global", total, run_opt(stream, total, bypass=False))
        put("opt_bypass_global", total, run_opt(stream, total, bypass=True))

    # Tuned families: the decay / epoch with the lowest train misses at each capacity, scored on test.
    tuned = {}
    for family, members in (
        ("lfu_best", [f"lfu{d}" for d in args.decays]),
        ("lfu_best_global", [f"lfu{d}_global" for d in args.decays]),
        ("tinylfu_best", [f"tinylfu{e}" for e in args.epochs]),
        ("wtinylfu_best", [f"wtinylfu{e}" for e in args.epochs]),
    ):
        tuned[family] = {}
        for total in args.capacities:
            pick = min(members, key=lambda name: results[name][total]["train"])
            tuned[family][total] = {**results[pick][total], "picked": pick}
    report = {
        "trace": args.trace,
        "decode_steps": len(stream["steps"]),
        "train_steps": int(train.sum()),
        "test_steps": int(test.sum()),
        "driver_requests": len(driver),
        "results": results,
        "tuned": tuned,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
