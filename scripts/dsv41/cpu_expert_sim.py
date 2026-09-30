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
  route order), as choose_cpu_lanes_locked does without NUMA; ``cpu_by_score`` takes the lowest by the victim
  ranking's key; ``cpu_deferred`` inserts ``cpu_lane_order``'s lanes one graph forward later, into shortlist
  entries that forward left unused, and counts those rows as background link rows. Scores count every route.
- Not modeled: NUMA preference (needs each host slot's node, which the trace lacks), c_cpu inflation under NVMe DMA
  or copy-engine load (P0 open item), the link time of ``cpu_deferred``'s background rows (reported, not costed).
"""

from __future__ import annotations

import argparse
import json
import os
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


INSERT_POLICIES = ("insert_all", "cpu_lane_order", "cpu_by_score", "cpu_deferred")


def _cpu_chooser(policy: str, sim, split: list[int], ram_hits: dict[int, list[int]], chosen: dict[int, set[int]]):
    """The graph_forward cpu_lanes hook: split[n] of a layer's n RAM hits, recorded into ``chosen``."""

    def choose(layer: int, missing: list[int]) -> set[int]:
        hits = ram_hits[layer]
        k = split[min(len(hits), len(split) - 1)]
        if policy == "cpu_by_score":
            row = sim.row[layer]
            # The victim ranking's key (DirectInsertReplay._rank): the lowest-ranked would be evicted first.
            hits = sorted(hits, key=lambda e: (sim.routed[row, e], sim.scores[row, e], -e))
        chosen[layer] = set(hits[:k])
        return chosen[layer] if policy != "insert_all" else set()

    return choose


def replay_nm(
    loaded: dict,
    ram_rows: int,
    num_experts: int = 384,
    initial_from_log: bool = True,
    policy: str = "insert_all",
    split: Optional[list[int]] = None,
) -> dict:
    """Per decode token and layer, n (VRAM miss, RAM hit) and m (miss both), from the DIRECT + RamTier replays.

    ``policy`` feeds the CPU lane choice (``split[n]`` of a layer's n RAM hits) back into VRAM residency;
    ``insert_all`` inserts every miss, as the P1 replay did."""
    if policy not in INSERT_POLICIES:
        raise ValueError(f"unknown insertion policy {policy!r}, want one of {INSERT_POLICIES}")
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
    sim = tier_sim.DirectInsertReplay(initial, capacity, num_experts)
    ram = Replay(tier_sim.ram_rows_per_layer(ram_rows, len(layer_ids), num_experts), "base", True)
    n_tok, m_tok, cpu_tok, sim_vram, measured_vram = [], [], [], 0, 0
    hits_total, unique_total = 0, 0
    deferred: dict[int, list[int]] = {}
    for forward in loaded["forwards"]:
        if forward["phase"] == "capture":
            continue
        if forward["kind"] == "graph":
            pre = {layer: sim.resident(layer) for layer in forward["routes"]}
            decode = forward["phase"] == "decode"
            n_row, m_row, ram_hits = [], [], {}
            for layer, experts in forward["routes"].items():
                row = sim.row[layer]
                if decode:
                    vram_miss = [e for e in dict.fromkeys(experts) if e not in pre[layer]]
                    # Plan-lane order: the misses in first-appearance route order (plan_unique_routes_kernel).
                    ram_hits[layer] = [e for e in vram_miss if e in ram.tiers[row].where]
                    n_row.append(len(ram_hits[layer]))
                    m_row.append(len(vram_miss) - len(ram_hits[layer]))
                    unique_total += len(set(experts))
                    hits_total += len(set(experts)) - len(vram_miss)
                ram.decode(row, experts, pre[layer])
            chosen: dict[int, set[int]] = {}
            # CPU experts run only in a captured decode graph.
            hook = _cpu_chooser(policy, sim, split, ram_hits, chosen) if decode and policy != "insert_all" else None
            late = deferred if policy == "cpu_deferred" else None
            simulated = sim.graph_forward(forward["routes"], forward["phase"], cpu_lanes=hook, deferred=late)
            if late is not None:
                deferred = {layer: sorted(experts) for layer, experts in chosen.items() if experts}
            if decode:
                n_tok.append(n_row)
                m_tok.append(m_row)
                cpu_tok.append(sum(split[min(n, len(split) - 1)] for n in n_row))
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
    return {
        "layer_ids": layer_ids,
        "n": n_arr,
        "m": m_arr,
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


def split_cost(n_arr: np.ndarray, m_arr: np.ndarray, model: CostModel, split: list[int]) -> float:
    """Mean ms/token with split[n] of each layer's n RAM hits on the CPU (``split`` is the residency pass's)."""
    table = np.array([[model.layer_ms(n, m, split[n]) for m in range(MAX_ROUTES + 1)] for n in range(MAX_ROUTES + 1)])
    return float((table[n_arr, m_arr].sum(axis=1) + model.gpu_ms).mean())


def insert_policy_results(
    loaded: dict, args, split: list[int], tables: dict[str, list[float]]
) -> dict[str, dict]:
    """Per insertion policy: its own replay's residency, n/m, and ms/token against ``off`` (insert_all's n/m)."""
    out, off_nm = {}, None
    for policy in INSERT_POLICIES:
        nm = replay_nm(loaded, args.ram_rows, args.num_experts, not args.no_initial_from_log, policy, split)
        if off_nm is None:
            off_nm = (nm["n"], nm["m"])
        costs = {}
        for name, points in tables.items():
            for nvme in args.nvme_ms:
                model = CostModel(points, c_link=args.c_link, handoff=args.handoff, nvme_ms=nvme, gpu_ms=args.gpu_ms)
                off = split_cost(*off_nm, model, [0] * len(split))
                on = split_cost(nm["n"], nm["m"], model, split)
                costs.setdefault(name, {})[str(nvme)] = {
                    "off_ms_per_token": off,
                    "ms_per_token": on,
                    "gain_vs_off_pct": 100.0 * (off - on) / off,
                }
        out[policy] = {
            **nm["residency"],
            "n_per_token": float(nm["n"].sum(axis=1).mean()),
            "m_per_token": float(nm["m"].sum(axis=1).mean()),
            "costs": costs,
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
        lines.append(f"\ninsertion policies (split {report['params']['split']}; NUMA preference: not modeled)")
        lines.append(
            f"{'policy':15s} {'hot hit':>7s} {'VRAM miss':>9s} {'n':>6s} {'m':>6s} {'cpu':>6s} {'bg rows':>7s}"
        )
        for name, row in report["insert_policies"].items():
            lines.append(
                f"{name:15s} {row['hot_hit_rate']:7.3f} {row['vram_misses_per_token']:9.2f} {row['n_per_token']:6.2f} "
                f"{row['m_per_token']:6.2f} {row['cpu_lanes_per_token']:6.2f} {row['deferred_link_rows_per_token']:7.2f}"
            )
        for table in next(iter(report["insert_policies"].values()))["costs"]:
            for nvme in report["params"]["nvme_ms"]:
                cells = " ".join(
                    f"{name} {row['costs'][table][str(nvme)]['ms_per_token']:.2f} "
                    f"({row['costs'][table][str(nvme)]['gain_vs_off_pct']:.1f}%)"
                    for name, row in report["insert_policies"].items()
                )
                off = next(iter(report["insert_policies"].values()))["costs"][table][str(nvme)]["off_ms_per_token"]
                lines.append(f"c_cpu {table}, nvme {nvme}: off {off:.2f}; {cells}")
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
        },
        "c_cpu_table": table,
        "numa": "not modeled (host-slot node is not in the trace)",
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
        report["insert_policies"] = insert_policy_results(loaded, args, split, insert_tables)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(summary(report))


if __name__ == "__main__":
    main()
