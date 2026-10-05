"""Predict decode ms/token with CPU-computed RAM-tier experts, per policy (plan 2026-09-29-dsv41-cpu-experts, P1).

Routing comes from a trace with the graph route log (tier_sim.load_forwards). Each decode forward, each layer:

- n = RAM-tier hits: routed experts that miss VRAM and hit the pinned tier;
- m = NVMe misses: routed experts that miss both.

VRAM is tier_sim.DirectInsertReplay (the running recipe, SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2). The pinned tier is
the RamTier replay of analysis/dsv41-drive/prefill-evict/ram_replay.py (base policy: validated 11.33 against 11.26
measured RAM misses per token), whose victims spare the VRAM-hot set. A decode forward classifies against the tier
as it stood before the forward touched it.

Per-layer cost model (ms), for k of the n RAM hits computed on the CPU:

    T_cpu  = h + rows_cpu * c_cpu(rows_cpu)        (0 when rows_cpu == 0)
    T_link = rows_link * c_link
    layer  = max(T_cpu, T_link) + m * nvme_ms
    token  = sum over layers + gpu_ms

c_cpu(k) is the per-expert cost at k experts in one call, interpolated linearly through the calibrated points at
1, 2 and 6 experts per call (held flat outside). Without CPU experts rows_link = n + m: a miss lands in the tier and
then still crosses the link (tier_sim.ms_per_token does the same). Assumptions:

- NVMe exposure is additive and per miss (``--nvme-ms``, default 1.5: 11.3 misses/token * 1.5 = ~17 ms, inside
  DSV41_REFERENCE 27.3's 13-20 ms). tier_sim's whole-row serial figures (3.4, 7.0) are reported as upper bounds.
- ``k*+nvme`` computes the k*(n) RAM hits AND all m landed rows on the CPU (rows_cpu = k + m). The landed rows'
  CPU time starts after the NVMe read, whose latency stays fully exposed, so the two add.
- Residency (``insert_policies``): each policy replays its own residency, since a CPU lane is never inserted into
  VRAM (direct_commit_gather_kernel). ``insert_all`` is the P1 assumption (every miss inserted, the headline table
  above). ``cpu_lane_order`` puts split[n] of a layer's n RAM hits on the CPU in plan-lane order (the misses in
  route order); ``cpu_by_score_asc_head`` / ``_desc_tail`` sort every miss lane by the victim key on the device
  and take the first / last split[n] job lanes, so the pairing with the victims follows the sorted order;
  ``cpu_numa_local`` takes the hits whose replayed pinned slot is on the CPU's node first, as
  choose_cpu_lanes_locked does; ``cpu_by_score`` takes the lowest by the victim ranking's key. The
  ``cpu_deferred*`` policies insert ``cpu_lane_order``'s lanes one graph forward later, into shortlist entries that
  forward left unused, if the row is still in the pinned tier after that forward's admissions; the variants
  queue by score or spare only unrouted entries. ``<base>_promote_N<n>_P<p>`` adds DirectInsertReplay.promote
  (GpuResidencyUpdater._promote on DIRECT decode boundaries, off today) every n decode forwards. Background and
  promoted rows are costed three ways (``split_costs``; promotions amortise over their n forwards).
  Scores count every route.
- Not modeled: the real pinned slot of each row (``cpu_numa_local`` uses the replay's, which starts empty), c_cpu
  inflation under NVMe DMA or copy-engine load (P0 open item), link idle time outside the per-layer CPU slack
  (GPU compute, NVMe waits), which the amortised background cost leaves unused.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "analysis", "dsv41-drive", "prefill-evict"))

import tier_sim  # noqa: E402
from ram_replay import Replay  # noqa: E402

from sglang.srt.layers.moe.cpu_experts.policy import split_table  # noqa: E402

MAX_ROUTES = 6
CALIBRATED_KS = (1, 2, 6)
# P0 (2026-09-29): ms per expert at 1 / 2 / 6 experts per call. 12 is the resid+b128 flavor, 8 and 10 upstream native.
P0_C_CPU_MS = {
    "8": [0.717, 0.658, 0.644],
    "10": [0.622, 0.594, 0.543],
    "12": [0.575, 0.518, 0.475],
}


def c_cpu_at(k: float, points: list[float]) -> float:
    """Per-expert CPU cost with k experts in one call, interpolated through the calibrated points."""
    return float(np.interp(k, CALIBRATED_KS, points))


class CostModel:
    def __init__(self, points: list[float], *, c_link: float, handoff: float, nvme_ms: float, gpu_ms: float):
        self.points, self.c_link, self.h = points, c_link, handoff
        self.nvme_ms, self.gpu_ms = nvme_ms, gpu_ms

    def layer_ms(self, n: int, m: int, k: int, cpu_nvme: bool = False) -> float:
        rows_cpu = k + (m if cpu_nvme else 0)
        rows_link = (n - k) + (0 if cpu_nvme else m)
        t_cpu = self.h + rows_cpu * c_cpu_at(rows_cpu, self.points) if rows_cpu else 0.0
        return max(t_cpu, rows_link * self.c_link) + m * self.nvme_ms

    def best_k(self, n: int, m: int) -> int:
        """argmin_k of the exact layer time under the k-dependent c_cpu; a tie goes to the larger k."""
        best_k, best = 0, self.layer_ms(n, m, 0)
        for k in range(1, n + 1):
            cost = self.layer_ms(n, m, k)
            if cost <= best:
                best_k, best = k, cost
        return best_k

    def policy_tables(self) -> dict[str, np.ndarray]:
        """Per policy, layer ms for every (n, m), shape [MAX_ROUTES + 1, MAX_ROUTES + 1] (n + m <= 6 is used)."""
        # The runtime table (P3) is built from a scalar c_cpu: take the per-expert cost at decode's 1-2 per call.
        kstar = split_table(MAX_ROUTES, c_cpu_at(2, self.points), self.c_link, self.h).tolist()
        pick: dict[str, Callable[[int, int], tuple[int, bool]]] = {
            "off": lambda n, m: (0, False),
            "all-cpu": lambda n, m: (n, False),
            "k*": lambda n, m: (kstar[n], False),
            "k*-cap1": lambda n, m: (min(kstar[n], 1), False),
            "k*-cap2": lambda n, m: (min(kstar[n], 2), False),
            "k*-exact": lambda n, m: (self.best_k(n, m), False),
            "k*+nvme": lambda n, m: (kstar[n], True),
        }
        size = MAX_ROUTES + 1
        tables = {}
        for name, choose in pick.items():
            table = np.zeros((size, size))
            for n in range(size):
                for m in range(size):
                    k, cpu_nvme = choose(n, m)
                    table[n, m] = self.layer_ms(n, m, k, cpu_nvme)
            tables[name] = table
        return tables


# choice: which RAM hits the CPU takes (lane order, lowest victim rank, or NUMA-local first then lane order).
# queue: cpu_deferred's insertion order when spare entries run short. unrouted: spare entries exclude the open
# window's routed experts.
INSERT_POLICIES = {
    "insert_all": {"choice": "lane", "insert": True},
    "cpu_lane_order": {"choice": "lane"},
    "cpu_by_score": {"choice": "score"},
    # The device sorts the plan's miss lanes by the victim key (NVMe misses included) and the host takes the first
    # (asc) or last (desc) split[n] job lanes; lane j pairs with usable[j] in the sorted order.
    "cpu_by_score_asc_head": {"choice": "head", "sort": "asc"},
    "cpu_by_score_desc_tail": {"choice": "tail", "sort": "desc"},
    "cpu_numa_local": {"choice": "numa"},
    "cpu_deferred": {"choice": "lane", "queue": "lane"},
    "cpu_deferred_scoreq": {"choice": "lane", "queue": "score"},
    "cpu_deferred_unrouted": {"choice": "lane", "queue": "lane", "unrouted": True},
}
# Where a decode NVMe miss lands. miss: the victim slot of the RAM tier, as today. none: VRAM only (a staging
# buffer feeds it); prefill still admits. deferred: staged, then copied into the tier after the forward. never: as
# none, and prefill admits nothing either (a tier frozen at its startup contents).
RAM_INSERTS = ("miss", "none", "deferred", "never")
PROMOTE_NAME = re.compile(r"^(?P<base>\w+?)_promote_N(?P<n>\d+)_P(?P<p>\d+)$")


def policy_config(policy: str) -> dict:
    """INSERT_POLICIES[policy], or ``<base>_promote_N<n>_P<p>``: base, plus up to p promotions per layer every n
    decode forwards (DirectInsertReplay.promote)."""
    if policy in INSERT_POLICIES:
        return INSERT_POLICIES[policy]
    match = PROMOTE_NAME.match(policy)
    if match is None or match["base"] not in INSERT_POLICIES or int(match["n"]) < 1 or int(match["p"]) < 1:
        raise ValueError(
            f"unknown insertion policy {policy!r}, want one of {sorted(INSERT_POLICIES)} or <base>_promote_N<n>_P<p>"
        )
    return {**INSERT_POLICIES[match["base"]], "promote_n": int(match["n"]), "promote_p": int(match["p"])}


# arm_env.PINNED_HOST_NUMA_MB and the CPU cores 18-29 (node 1) of the P2 arms.
DEFAULT_NUMA_MB = "0:61440,1:40960"
DEFAULT_CPU_NODE = 1


def _victim_key(sim, row: int):
    # DirectInsertReplay._rank's key: the lowest would be evicted first.
    return lambda e: (sim.routed[row, e], sim.scores[row, e], -e)


def _cpu_chooser(
    policy: str,
    sim,
    split: list[int],
    ram_hits: dict[int, list[int]],
    chosen: dict[int, list[int]],
    local: Optional[dict[int, set[int]]] = None,
    cpu_misses: bool = False,
):
    """The graph_forward cpu_lanes hook: split[n] of a layer's n RAM hits, recorded into ``chosen`` in lane order.

    ``cpu_misses`` (head/tail policies): n counts every residual lane, NVMe misses too, as the slot-map post does."""
    config = policy_config(policy)
    choice = config["choice"]

    def choose(layer: int, missing: list[int]) -> set[int]:
        hits = ram_hits[layer]
        k = split[min(len(hits), len(split) - 1)]
        if choice in ("head", "tail"):
            # Job lanes in the device's lane order: pinned-tier hits, or every lane with cpu_misses.
            held = set(hits)
            jobs = list(missing) if cpu_misses else [e for e in missing if e in held]
            k = split[min(len(jobs), len(split) - 1)]
            picked = set(jobs[:k] if choice == "head" else jobs[len(jobs) - k :])
            chosen[layer] = [e for e in jobs if e in picked]
            return picked
        if choice == "score":
            ranked = sorted(hits, key=_victim_key(sim, sim.row[layer]))
        elif choice == "numa":
            near = local[layer] if local is not None else set(hits)
            ranked = [e for e in hits if e in near] + [e for e in hits if e not in near]
        else:
            ranked = hits
        picked = set(ranked[:k])
        chosen[layer] = [e for e in hits if e in picked]
        return set() if config.get("insert") else picked

    return choose


def _lane_sorter(policy: str, sim) -> Optional[Callable[[int, list[int]], list[int]]]:
    """The graph_forward lane_order hook of a device-sorted policy: miss lanes by the victim key."""
    order = policy_config(policy).get("sort")
    if order is None:
        return None
    return lambda layer, missing: sorted(missing, key=_victim_key(sim, sim.row[layer]), reverse=order == "desc")


def _queue(policy: str, sim, chosen: dict[int, list[int]]) -> dict[int, list[int]]:
    """cpu_deferred's pending insertions: lane order, or the highest victim rank first."""
    if policy_config(policy).get("queue") == "score":
        return {layer: sorted(e, key=_victim_key(sim, sim.row[layer]), reverse=True) for layer, e in chosen.items() if e}
    return {layer: list(e) for layer, e in chosen.items() if e}


def slot_nodes_per_row(capacity: list[int], numa_mb: str) -> list[list[int]]:
    """Each replayed host slot's node, as the pinned tier's host_numa.split_rows binds a row's slots."""
    from sglang.srt.layers.moe.host_numa import split_rows

    placement = [(int(node), int(mb)) for node, mb in (part.split(":") for part in numa_mb.split(","))]
    out = []
    for rows in capacity:
        nodes = [-1] * rows
        for node, first, count in split_rows(rows, placement):
            nodes[first : first + count] = [node] * count
        out.append(nodes)
    return out


def replay_nm(
    loaded: dict,
    ram_rows: int,
    num_experts: int = 384,
    initial_from_log: bool = True,
    policy: str = "insert_all",
    split: Optional[list[int]] = None,
    numa_mb: str = DEFAULT_NUMA_MB,
    cpu_node: int = DEFAULT_CPU_NODE,
    promotion_margin: float = 1.0,
    promotion_sigmas: float = 0.0,
    ram_insert: str = "miss",
    staging_reserve: int = 0,
    cpu_misses: bool = False,
    protect_reads: bool = True,
    miss_rows: int = 6,
) -> dict:
    """Per decode token and layer, n (VRAM miss, RAM hit) and m (miss both), from the DIRECT + RamTier replays.

    ``policy`` feeds the CPU lane choice (``split[n]`` of a layer's n RAM hits) back into VRAM residency;
    ``insert_all`` inserts every miss, as the P1 replay did. ``b`` is the deferred rows each decode forward
    inserts or promotes, per layer: rows copied off the critical path. Promotions (``_promote_N<n>_P<p>``) run
    before every n-th decode forward's gathers, from the pinned tier as that forward found it.

    ``ram_insert`` is where a decode NVMe miss goes (see RAM_INSERTS); prefill admits as today except under ``never``.
    ``staging_reserve`` takes K rows of every layer's tier for staging slots, never mapped. ``cpu_misses`` lets a
    head/tail policy put NVMe misses on the CPU (``kh``/``km``: CPU hits and CPU misses per token and layer).
    ``protect_reads=False`` reads only VRAM-missing experts into the tier on decode, not VRAM-hot routed ones.
    ``miss_rows`` is the DIRECT victim shortlist, the most misses a layer inserts per forward (6 at batch size 1; a
    verify's gather width W)."""
    if ram_insert not in RAM_INSERTS:
        raise ValueError(f"unknown ram_insert {ram_insert!r}, want one of {RAM_INSERTS}")
    config = policy_config(policy)
    defer = "queue" in config
    split = split if split is not None else [0] * (MAX_ROUTES + 1)
    layer_ids = loaded["layer_ids"]
    capacity = loaded["hot_capacity"]
    initial = None
    if initial_from_log:
        first = next((f for f in loaded["forwards"] if f["kind"] == "graph" and f.get("hot") is not None), None)
        if first is not None:
            initial = {layer: list(experts) for layer, experts in first["hot"].items()}
    if initial is None:
        initial = {layer: list(range(slots)) for layer, slots in capacity.items()}
    sim = tier_sim.DirectInsertReplay(initial, capacity, num_experts, miss_rows=miss_rows)
    ram_capacity = [c - staging_reserve for c in tier_sim.ram_rows_per_layer(ram_rows, len(layer_ids), num_experts)]
    if min(ram_capacity) < 1:
        raise ValueError(f"staging_reserve {staging_reserve} leaves a layer with {min(ram_capacity)} mappable rows")
    ram = Replay(ram_capacity, "noadmit" if ram_insert == "never" else "base", True)
    nodes = slot_nodes_per_row(ram_capacity, numa_mb) if config["choice"] == "numa" else None
    n_tok, m_tok, b_tok, cpu_tok, kh_tok, km_tok, sim_vram, measured_vram = [], [], [], [], [], [], 0, 0
    hits_total, unique_total, evicted, decodes, admitted = 0, 0, 0, 0, 0
    pending: dict[int, list[int]] = {}
    for forward in loaded["forwards"]:
        if forward["phase"] == "capture":
            continue
        if forward["kind"] == "graph":
            decode = forward["phase"] == "decode"
            promoted: dict[int, int] = {}
            if decode and "promote_n" in config and decodes and decodes % config["promote_n"] == 0:
                pinned = {layer: ram.tiers[sim.row[layer]].where for layer in forward["routes"]}
                promoted = sim.promote(
                    pinned, config["promote_p"], margin=promotion_margin, sigmas=promotion_sigmas
                )
            decodes += decode
            pre = {layer: sim.resident(layer) for layer in forward["routes"]}
            n_row, m_row, ram_hits, local = [], [], {}, {}
            staged: dict[int, tuple[list[int], set[int]]] = {}
            for layer, experts in forward["routes"].items():
                row = sim.row[layer]
                if decode:
                    vram_miss = [e for e in dict.fromkeys(experts) if e not in pre[layer]]
                    # Plan-lane order: the misses in first-appearance route order (plan_unique_routes_kernel).
                    ram_hits[layer] = [e for e in vram_miss if e in ram.tiers[row].where]
                    if nodes is not None:
                        where = ram.tiers[row].where
                        local[layer] = {e for e in ram_hits[layer] if nodes[row][where[e]] == cpu_node}
                    n_row.append(len(ram_hits[layer]))
                    m_row.append(len(vram_miss) - len(ram_hits[layer]))
                    unique_total += len(set(experts))
                    hits_total += len(set(experts)) - len(vram_miss)
                missing = ram.lookup(row, experts)
                if decode and not protect_reads:
                    missing = [e for e in missing if e not in pre[layer]]
                if ram_insert == "miss" or (ram_insert in ("none", "deferred") and not decode):
                    ram.insert(row, missing, pre[layer], set(experts))
                    admitted += len(missing) * decode
                elif ram_insert == "deferred":
                    staged[layer] = (missing, set(experts))
                if pending.get(layer):
                    # The background copy reads the pinned row during this forward, after its admissions.
                    kept = [e for e in pending[layer] if e in ram.tiers[row].where]
                    evicted += len(pending[layer]) - len(kept)
                    pending[layer] = kept
            chosen: dict[int, list[int]] = {}
            # CPU experts run only in a captured decode graph.
            hook = (
                _cpu_chooser(policy, sim, split, ram_hits, chosen, local, cpu_misses=cpu_misses)
                if decode and not config.get("insert") else None
            )
            simulated = sim.graph_forward(
                forward["routes"],
                forward["phase"],
                cpu_lanes=hook,
                deferred=pending if defer else None,
                deferred_unrouted=bool(config.get("unrouted")),
                lane_order=_lane_sorter(policy, sim),
            )
            for layer, (missing, wanted) in staged.items():
                # The copy runs after the forward: its victim is not VRAM-hot as the forward left it, nor routed by it.
                ram.insert(sim.row[layer], missing, sim.resident(layer), wanted)
                admitted += len(missing)
            if defer:
                # Every graph forward's commit lands the queue (a graph-served prefill too), so it restarts here.
                pending = _queue(policy, sim, chosen)
            if decode:
                n_tok.append(n_row)
                m_tok.append(m_row)
                b_tok.append([sim.deferred_last.get(layer, 0) + promoted.get(layer, 0) for layer in forward["routes"]])
                cpu_tok.append(sum(split[min(n, len(split) - 1)] for n in n_row))
                kh_row = [sum(e in set(ram_hits[layer]) for e in chosen.get(layer, [])) for layer in forward["routes"]]
                kh_tok.append(kh_row)
                km_tok.append([len(chosen.get(layer, [])) - h for layer, h in zip(forward["routes"], kh_row)])
                sim_vram += sum(simulated.values())
                measured_vram += sum(forward["misses"].values())
        else:
            pre = {layer: sim.resident(layer) for layer in forward["counts"]}
            for layer, (experts, _) in forward["counts"].items():
                ram.prefill(sim.row[layer], list(experts), pre[layer])
            sim.eager_forward(forward["tokens"], forward["counts"], forward["phase"])
    tokens = len(n_tok)
    n_arr = np.asarray(n_tok, dtype=np.int64).reshape(tokens, -1)
    m_arr = np.asarray(m_tok, dtype=np.int64).reshape(tokens, -1)
    b_arr = np.asarray(b_tok, dtype=np.int64).reshape(tokens, -1)
    kh_arr = np.asarray(kh_tok, dtype=np.int64).reshape(tokens, -1)
    km_arr = np.asarray(km_tok, dtype=np.int64).reshape(tokens, -1)
    return {
        "layer_ids": layer_ids,
        "n": n_arr,
        "m": m_arr,
        "b": b_arr,
        "kh": kh_arr,
        "km": km_arr,
        "validation": {
            "decode_tokens": tokens,
            "sim_vram_misses_per_token": sim_vram / max(tokens, 1),
            "measured_vram_misses_per_token": measured_vram / max(tokens, 1),
            "n_plus_m_per_token": float((n_arr + m_arr).sum() / max(tokens, 1)),
            "ram_misses_per_token_m": float(m_arr.sum() / max(tokens, 1)),
        },
        "residency": {
            "policy": policy,
            # The served metric: unique hot hits over unique requested rows (expert_hot_cache.py, graph rows).
            "hot_hit_rate": hits_total / max(unique_total, 1),
            "vram_misses_per_token": sim_vram / max(tokens, 1),
            "cpu_lanes_per_token": float(np.mean(cpu_tok)) if cpu_tok else 0.0,
            "deferred_link_rows_per_token": sim.deferred_inserted / max(tokens, 1),
            "deferred_dropped_per_token": sim.deferred_dropped / max(tokens, 1),
            "deferred_evicted_per_token": evicted / max(tokens, 1),
            "promoted_rows_per_token": sim.promoted / max(tokens, 1),
            "unique_routes_per_token": unique_total / max(tokens, 1),
            "ram_inserted_rows_per_token": admitted / max(tokens, 1),
            "ram_capacity_per_row": list(ram_capacity),
        },
    }


def histogram(values: np.ndarray, size: int = MAX_ROUTES + 1) -> list[int]:
    return np.bincount(values.reshape(-1), minlength=size)[:size].tolist()


def predict(n_arr: np.ndarray, m_arr: np.ndarray, model: CostModel) -> dict[str, dict]:
    """Mean and spread of ms/token per policy, and the gain over ``off``."""
    out = {}
    for name, table in model.policy_tables().items():
        per_token = table[n_arr, m_arr].sum(axis=1) + model.gpu_ms
        out[name] = {
            "ms_per_token": float(per_token.mean()),
            "p50": float(np.percentile(per_token, 50)),
            "p90": float(np.percentile(per_token, 90)),
        }
    off = out["off"]["ms_per_token"]
    for row in out.values():
        row["gain_vs_off_pct"] = 100.0 * (off - row["ms_per_token"]) / off
    return out


def split_costs(
    n_arr: np.ndarray,
    m_arr: np.ndarray,
    model: CostModel,
    split: list[int],
    b_arr: Optional[np.ndarray] = None,
    window: int = 1,
) -> dict[str, float]:
    """Mean ms/token with split[n] of each layer's n RAM hits on the CPU, and b background rows per layer.

    ``uncosted`` ignores b; ``worst`` adds b to the same layer's link rows; ``amortised`` spreads the b of each
    block of ``window`` tokens over the link slack its layers leave under the CPU term, and adds the rest;
    ``optimistic`` also credits the block's GPU compute and NVMe waits (NVMe DMA lands in host RAM) as idle link."""
    k = np.asarray(split)[n_arr]
    c_cpu = np.interp(k, CALIBRATED_KS, model.points)
    t_cpu = np.where(k > 0, model.h + k * c_cpu, 0.0)
    t_link = (n_arr - k + m_arr) * model.c_link
    core = np.maximum(t_cpu, t_link)
    nvme = m_arr * model.nvme_ms
    uncosted = (core + nvme).sum(axis=1) + model.gpu_ms
    if b_arr is None:
        b_arr = np.zeros_like(n_arr)
    t_bg = b_arr * model.c_link
    worst = (np.maximum(t_cpu, t_link + t_bg) + nvme).sum(axis=1) + model.gpu_ms
    block = np.arange(len(uncosted)) // window
    bg, slack = np.bincount(block, t_bg.sum(axis=1)), np.bincount(block, (core - t_link).sum(axis=1))
    idle = slack + np.bincount(block, nvme.sum(axis=1) + model.gpu_ms)
    amortised = uncosted.mean() + np.maximum(bg - slack, 0.0).sum() / len(uncosted)
    optimistic = uncosted.mean() + np.maximum(bg - idle, 0.0).sum() / len(uncosted)
    return {
        "uncosted": float(uncosted.mean()),
        "worst": float(worst.mean()),
        "amortised": float(amortised),
        "optimistic": float(optimistic),
    }


def slot_map_costs(
    n_arr: np.ndarray, m_arr: np.ndarray, kh_arr: np.ndarray, km_arr: np.ndarray, model: CostModel
) -> dict[str, float]:
    """Mean ms/token with kh CPU hits and km CPU-computed NVMe misses per layer (slot-map plan, Task 0).

    A CPU miss starts after its read. ``pessimistic`` runs it after the layer's link and hit work, ``optimistic``
    overlaps it with them; with km = 0 both equal split_costs' ``uncosted``."""
    c_h = np.interp(kh_arr, CALIBRATED_KS, model.points)
    c_m = np.interp(km_arr, CALIBRATED_KS, model.points)
    t_hits = np.where(kh_arr > 0, model.h + kh_arr * c_h, 0.0)
    t_link = (n_arr - kh_arr + m_arr - km_arr) * model.c_link
    core = np.maximum(t_hits, t_link)
    nvme = m_arr * model.nvme_ms
    t_miss = np.where(km_arr > 0, model.h + km_arr * c_m, 0.0)
    return {
        "pessimistic": float(((core + nvme + t_miss).sum(axis=1) + model.gpu_ms).mean()),
        "optimistic": float(((nvme + np.maximum(core, t_miss)).sum(axis=1) + model.gpu_ms).mean()),
    }


SLOT_MAP_CPU = {"off": ("insert_all", False), "hits": (None, False), "hits+misses": (None, True)}


def slot_map_results(loaded: dict, args, split: list[int]) -> list[dict]:
    """Deferred RAM inserts per staging reserve K, protect reads on/off, CPU experts off / hits / hits and misses."""
    model = CostModel(
        [args.measured_c_cpu] * len(CALIBRATED_KS), c_link=args.c_link, handoff=args.handoff,
        nvme_ms=args.nvme_ms[0], gpu_ms=args.gpu_ms,
    )
    rows = []
    for k in args.staging_reserve:
        for protect in (True, False):
            for cpu, (policy, misses) in SLOT_MAP_CPU.items():
                nm = replay_nm(
                    loaded, args.ram_rows, args.num_experts, not args.no_initial_from_log, policy or STAGING_MERGED,
                    None if cpu == "off" else split, numa_mb=args.numa_mb, cpu_node=args.cpu_node,
                    ram_insert="deferred", staging_reserve=k, cpu_misses=misses, protect_reads=protect,
                )
                n_sum, m_sum = int(nm["n"].sum()), int(nm["m"].sum())
                res, tokens = nm["residency"], nm["validation"]["decode_tokens"]
                cost = slot_map_costs(nm["n"], nm["m"], nm["kh"], nm["km"], model)
                rows.append({
                    "staging_reserve": k,
                    "protect_reads": protect,
                    "cpu": cpu,
                    "hot_hit_rate": res["hot_hit_rate"],
                    "ram_hit_of_lanes": n_sum / max(n_sum + m_sum, 1),
                    "nvme_reads_per_token": m_sum / max(tokens, 1),
                    "cpu_hits_per_token": float(nm["kh"].sum()) / max(tokens, 1),
                    "cpu_misses_per_token": float(nm["km"].sum()) / max(tokens, 1),
                    "ram_inserted_rows_per_token": res["ram_inserted_rows_per_token"],
                    "ms_per_token_pessimistic": cost["pessimistic"],
                    "ms_per_token_optimistic": cost["optimistic"],
                })
    return rows


def insert_policy_results(
    loaded: dict, args, split: list[int], tables: dict[str, list[float]], policies: list[str]
) -> dict[str, dict]:
    """Per insertion policy: its own replay's residency, n/m, and ms/token against ``off`` (insert_all's n/m)."""
    out, off_nm = {}, None
    for policy in policies:
        nm = replay_nm(
            loaded, args.ram_rows, args.num_experts, not args.no_initial_from_log, policy, split,
            numa_mb=args.numa_mb, cpu_node=args.cpu_node,
            promotion_margin=args.promotion_margin, promotion_sigmas=args.promotion_sigmas,
        )
        window = policy_config(policy).get("promote_n", 1)
        if off_nm is None:
            off_nm = (nm["n"], nm["m"])
        costs = {}
        for name, points in tables.items():
            for nvme in args.nvme_ms:
                model = CostModel(points, c_link=args.c_link, handoff=args.handoff, nvme_ms=nvme, gpu_ms=args.gpu_ms)
                off = split_costs(*off_nm, model, [0] * len(split))["uncosted"]
                on = split_costs(nm["n"], nm["m"], model, split, nm["b"], window)
                costs.setdefault(name, {})[str(nvme)] = {
                    "off_ms_per_token": off,
                    "ms_per_token": on["uncosted"],
                    "ms_per_token_bg_worst": on["worst"],
                    "ms_per_token_bg_amortised": on["amortised"],
                    "ms_per_token_bg_optimistic": on["optimistic"],
                    "gain_vs_off_pct": 100.0 * (off - on["uncosted"]) / off,
                    "gain_bg_worst_pct": 100.0 * (off - on["worst"]) / off,
                    "gain_bg_amortised_pct": 100.0 * (off - on["amortised"]) / off,
                    "gain_bg_optimistic_pct": 100.0 * (off - on["optimistic"]) / off,
                }
        out[policy] = {
            **nm["residency"],
            "n_per_token": float(nm["n"].sum(axis=1).mean()),
            "m_per_token": float(nm["m"].sum(axis=1).mean()),
            "costs": costs,
        }
    if "cpu_by_score" in out:
        for row in out.values():
            for table, by_nvme in row["costs"].items():
                for nvme, c in by_nvme.items():
                    ref = out["cpu_by_score"]["costs"][table][nvme]["ms_per_token"]
                    for key in (
                        "ms_per_token", "ms_per_token_bg_worst", "ms_per_token_bg_amortised", "ms_per_token_bg_optimistic"
                    ):
                        c[key.replace("ms_per_token", "gain_vs_by_score") + "_pct"] = 100.0 * (ref - c[key]) / ref
    return out


STAGING_ARMS = {
    "A_baseline": "miss",
    "B_staging_no_ram_insert": "none",
    "C_staging_deferred_insert": "deferred",
    "B0_tier_frozen_at_startup": "never",
}
STAGING_MERGED = "cpu_by_score_desc_tail"


def staging_results(loaded: dict, args, split: list[int]) -> dict[str, dict]:
    """Each RAM-insert arm with CPU experts off (insert_all, no split) and on (the merged policy).

    Lanes are the routed experts that miss VRAM (n + m); routes are the unique routed experts. ms/token uses the flat
    ``--measured-c-cpu`` and the first ``--nvme-ms``; the C host copy is reported in bytes, not folded in."""
    model = CostModel(
        [args.measured_c_cpu] * len(CALIBRATED_KS), c_link=args.c_link, handoff=args.handoff,
        nvme_ms=args.nvme_ms[0], gpu_ms=args.gpu_ms,
    )
    out: dict[str, dict] = {}
    for arm, ram_insert in STAGING_ARMS.items():
        for cpu, policy in (("off", "insert_all"), ("on", STAGING_MERGED)):
            nm = replay_nm(
                loaded, args.ram_rows, args.num_experts, not args.no_initial_from_log, policy,
                split if cpu == "on" else None, numa_mb=args.numa_mb, cpu_node=args.cpu_node, ram_insert=ram_insert,
            )
            n_sum, m_sum = int(nm["n"].sum()), int(nm["m"].sum())
            res, tokens = nm["residency"], nm["validation"]["decode_tokens"]
            cost = split_costs(nm["n"], nm["m"], model, split if cpu == "on" else [0] * len(split))["uncosted"]
            out.setdefault(arm, {})[cpu] = {
                "ram_insert": ram_insert,
                "policy": policy,
                "hot_hit_rate": res["hot_hit_rate"],
                "ram_hit_of_lanes": n_sum / max(n_sum + m_sum, 1),
                "ram_hit_of_routes": n_sum / max(res["unique_routes_per_token"] * tokens, 1),
                "n_per_token": n_sum / max(tokens, 1),
                "nvme_reads_per_token": m_sum / max(tokens, 1),
                "cpu_lanes_per_token": res["cpu_lanes_per_token"],
                "ms_per_token": cost,
                "ram_inserted_rows_per_token": res["ram_inserted_rows_per_token"],
                "host_memcpy_bytes_per_token": res["ram_inserted_rows_per_token"] * args.row_bytes
                if ram_insert == "deferred" else 0.0,
            }
    return out


def load_c_cpu_table(spec: Optional[str]) -> dict[str, list[float]]:
    if not spec:
        return P0_C_CPU_MS
    text = open(spec).read() if os.path.exists(spec) else spec
    table = {str(threads): [float(x) for x in points] for threads, points in json.loads(text).items()}
    for threads, points in table.items():
        if len(points) != len(CALIBRATED_KS):
            raise ValueError(f"threads {threads}: want ms at {CALIBRATED_KS} experts per call, got {points}")
    return table


def summary(report: dict) -> str:
    v, g = report["validation"], report["gate"]
    lines = [
        f"trace {report['trace']}",
        f"decode tokens {v['decode_tokens']}; VRAM misses/token sim {v['sim_vram_misses_per_token']:.2f} "
        f"vs route log {v['measured_vram_misses_per_token']:.2f}; NVMe misses/token (m) {v['ram_misses_per_token_m']:.2f}",
        f"mean n/token {report['mean_n_per_token']:.2f}, mean m/token {report['mean_m_per_token']:.2f}; NUMA preference: not modeled",
        "n histogram over (token, layer), n=0..6: " + " ".join(str(x) for x in report["n_hist_all"]),
        "m histogram over (token, layer), m=0..6: " + " ".join(str(x) for x in report["m_hist_all"]),
        "mean n per layer: " + " ".join(f"{x:.2f}" for x in report["mean_n_per_layer"]),
    ]
    for threads, by_nvme in report["results"].items():
        for nvme, rows in by_nvme.items():
            lines.append(f"\nthreads {threads}, nvme {nvme} ms/miss (c_cpu at 1/2/6 = {report['c_cpu_table'][threads]})")
            lines.append(f"{'policy':10s} {'ms/token':>9s} {'p50':>7s} {'p90':>7s} {'gain':>7s}")
            for name, row in rows.items():
                lines.append(
                    f"{name:10s} {row['ms_per_token']:9.2f} {row['p50']:7.2f} {row['p90']:7.2f} {row['gain_vs_off_pct']:6.1f}%"
                )
    if report.get("insert_policies"):
        lines.append(
            f"\ninsertion policies (split {report['params']['split']}; cpu_numa_local approximates the NUMA "
            f"preference from the replayed slots, {report['params']['numa_mb']} preferring node {report['params']['cpu_node']})"
        )
        lines.append(
            f"{'policy':30s} {'hot hit':>7s} {'VRAM miss':>9s} {'n':>6s} {'m':>6s} {'cpu':>6s} "
            f"{'bg rows':>7s} {'bg drop':>7s} {'bg evict':>8s} {'promoted':>8s}"
        )
        for name, row in report["insert_policies"].items():
            lines.append(
                f"{name:30s} {row['hot_hit_rate']:7.3f} {row['vram_misses_per_token']:9.2f} {row['n_per_token']:6.2f} "
                f"{row['m_per_token']:6.2f} {row['cpu_lanes_per_token']:6.2f} {row['deferred_link_rows_per_token']:7.2f} "
                f"{row['deferred_dropped_per_token']:7.2f} {row['deferred_evicted_per_token']:8.3f} "
                f"{row['promoted_rows_per_token']:8.2f}"
            )
        first = next(iter(report["insert_policies"].values()))
        for table in first["costs"]:
            for nvme in report["params"]["nvme_ms"]:
                off = first["costs"][table][str(nvme)]["off_ms_per_token"]
                lines.append(f"c_cpu {table}, nvme {nvme}: off {off:.2f} ms/token; ms/token (gain) uncosted | bg worst | bg amortised | bg optimistic")
                for name, row in report["insert_policies"].items():
                    c = row["costs"][table][str(nvme)]
                    lines.append(
                        f"  {name:30s} {c['ms_per_token']:7.2f} ({c['gain_vs_off_pct']:4.1f}%) | "
                        f"{c['ms_per_token_bg_worst']:7.2f} ({c['gain_bg_worst_pct']:4.1f}%) | "
                        f"{c['ms_per_token_bg_amortised']:7.2f} ({c['gain_bg_amortised_pct']:4.1f}%) | "
                        f"{c['ms_per_token_bg_optimistic']:7.2f} ({c['gain_bg_optimistic_pct']:4.1f}%)"
                    )
    if report.get("staging"):
        params = report["params"]
        lines.append(
            f"\nRAM-insert arms (flat c_cpu {params['measured_c_cpu']}, nvme {params['nvme_ms'][0]} ms/miss, merged policy "
            f"{STAGING_MERGED}, split {params['split']}); lanes = VRAM-missing experts, routes = unique routed experts"
        )
        lines.append(
            f"{'arm':28s} {'cpu':>3s} {'hot hit':>7s} {'RAM/lane':>8s} {'RAM/route':>9s} {'n/tok':>6s} "
            f"{'NVMe/tok':>8s} {'cpu/tok':>7s} {'ms/token':>8s} {'memcpy MB/tok':>13s}"
        )
        for arm, by_cpu in report["staging"].items():
            for cpu, r in by_cpu.items():
                lines.append(
                    f"{arm:28s} {cpu:>3s} {r['hot_hit_rate']:7.3f} {r['ram_hit_of_lanes']:8.3f} {r['ram_hit_of_routes']:9.3f} "
                    f"{r['n_per_token']:6.2f} {r['nvme_reads_per_token']:8.2f} {r['cpu_lanes_per_token']:7.2f} "
                    f"{r['ms_per_token']:8.2f} {r['host_memcpy_bytes_per_token'] / 1e6:13.1f}"
                )
    if report.get("slot_map"):
        params = report["params"]
        lines.append(
            f"\nslot-map arms (ram_insert=deferred, flat c_cpu {params['measured_c_cpu']}, nvme {params['nvme_ms'][0]} "
            f"ms/miss, split {params['split']}; ms/token pessimistic | optimistic)"
        )
        lines.append(
            f"{'K':>2s} {'protect':>7s} {'cpu':>11s} {'hot hit':>7s} {'RAM/lane':>8s} {'NVMe/tok':>8s} "
            f"{'cpuH/tok':>8s} {'cpuM/tok':>8s} {'pess':>7s} {'opt':>7s}"
        )
        for r in report["slot_map"]:
            lines.append(
                f"{r['staging_reserve']:2d} {('on' if r['protect_reads'] else 'off'):>7s} {r['cpu']:>11s} "
                f"{r['hot_hit_rate']:7.3f} {r['ram_hit_of_lanes']:8.3f} {r['nvme_reads_per_token']:8.2f} "
                f"{r['cpu_hits_per_token']:8.2f} {r['cpu_misses_per_token']:8.2f} "
                f"{r['ms_per_token_pessimistic']:7.2f} {r['ms_per_token_optimistic']:7.2f}"
            )
    lines.append(
        f"\ngate (>= {g['threshold_pct']}% for k*, threads {g['headline']['threads']}, nvme {g['headline']['nvme_ms']}): "
        f"k* {g['k_star_gain_pct']:.1f}% -> {'PASS' if g['passes'] else 'FAIL'} (best {g['best_policy']} {g['best_gain_pct']:.1f}%)"
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--ram-rows", type=int, default=8063, help="pinned tier rows (100 GiB / 13.3 MB)")
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--c-link", type=float, default=1.0, help="ms per row over the link")
    p.add_argument("--handoff", type=float, default=0.02, help="h, ms per layer")
    p.add_argument("--gpu-ms", type=float, default=14.0, help="fixed GPU compute per token")
    p.add_argument("--c-cpu-table", help="JSON (file or inline) threads -> [ms/expert at 1, 2, 6 per call]")
    p.add_argument("--threads", default="12", help="thread count the headline table and gate use")
    p.add_argument(
        "--nvme-ms", type=float, nargs="+", default=[1.5, 3.4, 7.0], help="exposed ms per NVMe miss; first is headline"
    )
    p.add_argument("--gate", type=float, default=15.0, help="pre-registered gain, percent")
    p.add_argument("--no-initial-from-log", action="store_true")
    p.add_argument(
        "--split-c-cpu", type=float, default=0.52, help="c_cpu the insertion replays' split is built from (served default)"
    )
    p.add_argument(
        "--measured-c-cpu", type=float, default=0.63, help="flat per-lane cost the insertion policies are also costed at"
    )
    p.add_argument("--no-insert-policies", action="store_true")
    p.add_argument("--staging", action="store_true", help="also replay the NVMe-miss staging arms (STAGING_ARMS)")
    p.add_argument(
        "--slot-map", action="store_true", help="also replay the slot-map arms (slot_map_results)"
    )
    p.add_argument(
        "--staging-reserve", type=lambda v: [int(x) for x in v.split(",")], default=[0, 6, 8],
        help="comma-separated K staging rows per layer for --slot-map",
    )
    p.add_argument("--row-bytes", type=float, default=13.3e6, help="bytes per expert row (100 GiB / 8063 rows)")
    p.add_argument("--numa-mb", default=DEFAULT_NUMA_MB, help="pinned tier placement node:MiB,... (cpu_numa_local)")
    p.add_argument("--cpu-node", type=int, default=DEFAULT_CPU_NODE, help="the CPU expert cores' node (cpu_numa_local)")
    p.add_argument("--promote-n", type=int, nargs="*", default=[], help="promotion intervals (decode forwards) to sweep")
    p.add_argument("--promote-p", type=int, nargs="*", default=[], help="promotions per layer and boundary to sweep")
    p.add_argument(
        "--promote-bases", nargs="*", default=["cpu_by_score", "insert_all"], help="policies the sweep adds promotion to"
    )
    p.add_argument("--promotion-margin", type=float, default=1.0, help="SGLANG_MOE_HOT_BENEFIT_RATIO (default 1.0)")
    p.add_argument("--promotion-sigmas", type=float, default=0.0, help="SGLANG_MOE_HOT_PROMOTION_SIGMAS (default 0)")
    p.add_argument("--out")
    args = p.parse_args()

    table = load_c_cpu_table(args.c_cpu_table)
    if args.threads not in table:
        raise SystemExit(f"--threads {args.threads} not in the c_cpu table {sorted(table)}")
    loaded = tier_sim.load_forwards(args.trace)
    nm = replay_nm(loaded, args.ram_rows, args.num_experts, not args.no_initial_from_log)
    n_arr, m_arr = nm["n"], nm["m"]

    results = {}
    for threads, points in table.items():
        for nvme in args.nvme_ms:
            model = CostModel(points, c_link=args.c_link, handoff=args.handoff, nvme_ms=nvme, gpu_ms=args.gpu_ms)
            results.setdefault(threads, {})[str(nvme)] = predict(n_arr, m_arr, model)
    head = results[args.threads][str(args.nvme_ms[0])]
    gains = {name: row["gain_vs_off_pct"] for name, row in head.items() if name != "off"}
    split = split_table(MAX_ROUTES, args.split_c_cpu, args.c_link, args.handoff).tolist()
    insert_tables = {f"flat{args.measured_c_cpu}": [args.measured_c_cpu] * len(CALIBRATED_KS), **table}
    report = {
        "trace": args.trace,
        "params": {
            **{k: getattr(args, k) for k in ("ram_rows", "c_link", "handoff", "gpu_ms", "threads", "nvme_ms", "gate")},
            "split": split,
            "measured_c_cpu": args.measured_c_cpu,
            "row_bytes": args.row_bytes,
            "numa_mb": args.numa_mb,
            "cpu_node": args.cpu_node,
            "promotion_margin": args.promotion_margin,
            "promotion_sigmas": args.promotion_sigmas,
        },
        "c_cpu_table": table,
        "numa": "cpu_numa_local approximates it from the replayed pinned slots; the other policies ignore it",
        "validation": nm["validation"],
        "n_hist_all": histogram(n_arr),
        "m_hist_all": histogram(m_arr),
        "n_hist_per_layer": [histogram(n_arr[:, i]) for i in range(n_arr.shape[1])],
        "mean_n_per_layer": n_arr.mean(axis=0).tolist(),
        "mean_m_per_layer": m_arr.mean(axis=0).tolist(),
        "mean_n_per_token": float(n_arr.sum(axis=1).mean()),
        "mean_m_per_token": float(m_arr.sum(axis=1).mean()),
        "results": results,
        "gate": {
            "threshold_pct": args.gate,
            "headline": {"threads": args.threads, "nvme_ms": args.nvme_ms[0]},
            "k_star_gain_pct": gains["k*"],
            "best_policy": max(gains, key=gains.get),
            "best_gain_pct": max(gains.values()),
            "passes": gains["k*"] >= args.gate,
        },
    }
    if not args.no_insert_policies:
        policies = list(INSERT_POLICIES) + [
            f"{base}_promote_N{n}_P{p}" for base in args.promote_bases for n in args.promote_n for p in args.promote_p
        ]
        report["insert_policies"] = insert_policy_results(loaded, args, split, insert_tables, policies)
    if args.staging:
        report["staging"] = staging_results(loaded, args, split)
    if args.slot_map:
        report["slot_map"] = slot_map_results(loaded, args, split)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(summary(report))


if __name__ == "__main__":
    main()
