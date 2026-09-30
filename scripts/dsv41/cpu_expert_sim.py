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
- Not modeled: NUMA preference (needs each host slot's node, which the trace lacks), c_cpu inflation under NVMe DMA
  or copy-engine load (P0 open item), the residency effect of CPU rows (a CPU-computed expert would still count as
  a hit for scoring; the replay keeps today's insert-on-miss).
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


def replay_nm(loaded: dict, ram_rows: int, num_experts: int = 384, initial_from_log: bool = True) -> dict:
    """Per decode token and layer, n (VRAM miss, RAM hit) and m (miss both), from the DIRECT + RamTier replays."""
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
    n_tok, m_tok, sim_vram, measured_vram = [], [], 0, 0
    for forward in loaded["forwards"]:
        if forward["phase"] == "capture":
            continue
        if forward["kind"] == "graph":
            pre = {layer: sim.resident(layer) for layer in forward["routes"]}
            decode = forward["phase"] == "decode"
            n_row, m_row = [], []
            for layer, experts in forward["routes"].items():
                row = sim.row[layer]
                if decode:
                    vram_miss = [e for e in dict.fromkeys(experts) if e not in pre[layer]]
                    hits = sum(1 for e in vram_miss if e in ram.tiers[row].where)
                    n_row.append(hits)
                    m_row.append(len(vram_miss) - hits)
                ram.decode(row, experts, pre[layer])
            simulated = sim.graph_forward(forward["routes"], forward["phase"])
            if decode:
                n_tok.append(n_row)
                m_tok.append(m_row)
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
    report = {
        "trace": args.trace,
        "params": {
            k: getattr(args, k) for k in ("ram_rows", "c_link", "handoff", "gpu_ms", "threads", "nvme_ms", "gate")
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
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(summary(report))


if __name__ == "__main__":
    main()
