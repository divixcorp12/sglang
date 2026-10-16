"""The lane head's served A/B, the mechanism beside serve_compare.py's ms/token verdict: per run, over its serving
prompts (serve_compare's sessions), VRAM hits, misses and unique routed experts per verify from the route log
(``stages.jsonl``, spec_margin_capture.py --route-log), the CPU-lane proxies from the job trace (lanes, CPU ms and ms per
lane per verify; copy-engine submits per verify), the hot-cache slots, the lane head's gather count, and the nvme3n1
read await in each prefill window (the gap before each serving prompt's first verify; serve_compare's STALL flag covers
decode only).

The route log's ``hot`` is each layer's residents when the forward was planned, so a VRAM hit is a routed expert in it
and a miss is a routed expert outside it; the gather's own ``misses`` count is reported as a cross-check.

Usage: lane_head_ab_report.py RUN_DIR RUN_DIR [...] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_compare  # noqa: E402

SESSION_GAP_S = 1.0


def route_log(d: str) -> tuple[dict, list[list[dict]]]:
    """The header and the target-verify records split into sessions (gaps over SESSION_GAP_S)."""
    header, recs = None, []
    with open(os.path.join(d, "stages.jsonl")) as f:
        for line in f:
            if '"graph_routes' not in line:
                continue
            r = json.loads(line)
            if r.get("kind") == "graph_routes_header":
                header = r
            elif r.get("kind") == "graph_routes" and r.get("phase") == "target_verify" and "hot" in r:
                recs.append(r)
    if header is None or not recs:
        raise SystemExit(f"{d}: no route log")
    gaps = [b["t"] - a["t"] for a, b in zip(recs, recs[1:])]
    # ``t`` is in ms when the median verify-to-verify step is over 10 (DSpark verifies are 50-300 ms apart).
    scale = 1e-3 if statistics.median(gaps) > 10 else 1.0
    sessions = [[recs[0]]]
    for a, b in zip(recs, recs[1:]):
        if (b["t"] - a["t"]) * scale > SESSION_GAP_S:
            sessions.append([])
        sessions[-1].append(b)
    return header, sessions


def verify_hits(r: dict) -> tuple[int, int, int]:
    """(hits, misses, unique routed) over the layers of one verify record."""
    hits = misses = unique = 0
    for routes, hot in zip(r["routes"], r["hot"]):
        u = {e for e in routes if e >= 0}
        h = len(u & set(hot))
        hits += h
        misses += len(u) - h
        unique += len(u)
    return hits, misses, unique


def cpu_jobs(d: str) -> list[tuple[int, float, int]]:
    """(start ns, ms, lanes) of each completed CPU job, both teams."""
    out = []
    for path in sorted(glob.glob(os.path.join(d, "events.*.exl3-cpu-exp*.jsonl"))):
        jobs: dict = {}
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                k = e.get("event")
                if k in ("cpu_start", "cpu_end", "cpu_shape"):
                    jobs.setdefault((e["row"], e["seq"]), {})[k] = e
        out += [(j["cpu_start"]["ns"], (j["cpu_end"]["ns"] - j["cpu_start"]["ns"]) / 1e6, j["cpu_shape"]["b"])
                for j in jobs.values() if all(k in j for k in ("cpu_start", "cpu_end", "cpu_shape"))]
    return sorted(out)


def copy_submits(d: str) -> list[int]:
    out = []
    for path in glob.glob(os.path.join(d, "events.*.exl3-copy-eng*.jsonl")):
        with open(path) as f:
            for line in f:
                if '"copy_submit"' in line:
                    out.append(json.loads(line)["ns"])
    return sorted(out)


def nvme3n1_samples(d: str) -> list[tuple[float, int, int]]:
    """(epoch s, reads completed, ms reading) from <run>.nvme3n1.samples (a /proc/diskstats line every 2 s)."""
    path = os.path.realpath(d) + ".nvme3n1.samples"
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) > 8:
                out.append((float(p[0]), int(p[4]), int(p[7])))
    return out


def window_await(samples: list[tuple[float, int, int]], t0: float, t1: float) -> tuple[float | None, int]:
    """Read await ms and reads between the samples bracketing [t0, t1] (epoch s)."""
    inside = [s for s in samples if t0 - 2.0 <= s[0] <= t1 + 2.0]
    if len(inside) < 2:
        return None, 0
    reads = inside[-1][1] - inside[0][1]
    return ((inside[-1][2] - inside[0][2]) / reads if reads else None), reads


def counters_from_log(d: str) -> dict:
    out = {"lane_head_active": False, "slots": None, "lane_head_gathers": None}
    with open(os.path.join(d, "server.log"), errors="replace") as f:
        for line in f:
            if "lane head active" in line:
                out["lane_head_active"] = True
            if "Expert hot cache startup" in line:
                try:
                    out["startup"] = json.loads(line[line.index("{"):])
                except ValueError:
                    pass
    for path in (os.path.join(d, "metrics.jsonl"),):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    i = line.find('"lane_head_gathers"')
                    if i >= 0:
                        gathers = json.loads(line)
                        out["lane_head_gathers"] = _find(gathers, "lane_head_gathers")
    return out


def _find(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def run(d: str) -> dict:
    per = serve_compare.prompts(d, 1.3, 1.2)
    _, to_epoch = serve_compare.load_verifies(d)
    header, sessions = route_log(d)
    serving = sessions[-len(per):]
    jobs = cpu_jobs(d)
    copies = copy_submits(d)
    disk = nvme3n1_samples(d)
    out = {"slots": sum(header["hot_capacity"]), "prompts": []}
    prev_t1 = None
    for p, recs in zip(per, serving):
        hm = [verify_hits(r) for r in recs]
        v = p["verifies"]
        t0, t1 = p["t0"], p["t1"]
        cj = [j for j in jobs if t0 <= j[0] <= t1]
        cs = sum(1 for c in copies if t0 <= c <= t1)
        w0 = (prev_t1 if prev_t1 is not None else t0 - 30e9) / 1e9 + to_epoch
        aw, reads = window_await(disk, w0, t0 / 1e9 + to_epoch)
        prev_t1 = t1
        out["prompts"].append({
            "verifies": v, "route_log_verifies": len(recs),
            "hits_per_verify": sum(h for h, _, _ in hm) / len(hm),
            "misses_per_verify": sum(m for _, m, _ in hm) / len(hm),
            "unique_per_verify": sum(u for _, _, u in hm) / len(hm),
            "gather_misses_per_verify": sum(sum(r["misses"]) for r in recs) / len(recs),
            "cpu_lanes_per_verify": sum(j[2] for j in cj) / v,
            "cpu_ms_per_verify": sum(j[1] for j in cj) / v,
            "cpu_ms_per_lane": sum(j[1] for j in cj) / max(1, sum(j[2] for j in cj)),
            "copy_submits_per_verify": cs / v,
            "prefill_nvme3n1_await_ms": aw, "prefill_nvme3n1_reads": reads,
        })
    keys = [k for k in out["prompts"][0] if k.endswith("_verify") or k == "cpu_ms_per_lane"]
    w = [p["verifies"] for p in out["prompts"]]
    out["all"] = {k: sum(p[k] * n for p, n in zip(out["prompts"], w)) / sum(w) for k in keys}
    out["log"] = counters_from_log(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {serve_compare._name(d): run(d) for d in a.runs}
    for name, r in res.items():
        print(f"== {name}: slots {r['slots']}, lane head active {r['log']['lane_head_active']}, "
              f"lane_head_gathers {r['log']['lane_head_gathers']}")
        print("   prompt verifies  hits/v  miss/v  uniq/v  gmiss/v  cpu lanes/v  cpu ms/v  ms/lane  copies/v  prefill await ms (reads)")
        for i, p in enumerate(r["prompts"]):
            aw = "-" if p["prefill_nvme3n1_await_ms"] is None else f"{p['prefill_nvme3n1_await_ms']:.1f}"
            print(f"   {i:6d} {p['verifies']:8d} {p['hits_per_verify']:7.1f} {p['misses_per_verify']:7.1f} "
                  f"{p['unique_per_verify']:7.1f} {p['gather_misses_per_verify']:8.1f} {p['cpu_lanes_per_verify']:12.1f} "
                  f"{p['cpu_ms_per_verify']:9.1f} {p['cpu_ms_per_lane']:8.3f} {p['copy_submits_per_verify']:9.1f}  "
                  f"{aw} ({p['prefill_nvme3n1_reads']})")
        print("   all: " + "  ".join(f"{k} {v:.2f}" for k, v in r["all"].items()))
    names = list(res)
    for other in names[1:]:
        a0, b0 = res[names[0]]["all"], res[other]["all"]
        print(f"== {other} - {names[0]}: " + "  ".join(f"{k} {b0[k] - a0[k]:+.2f}" for k in a0))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
