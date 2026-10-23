"""Marginal cost of each expert source in a DSpark verify layer, fitted on an InstrBuild capture's job trace.

Per record (row, gen) the trace's ``lane_kind`` events give every non-VRAM lane's kind (b & 255: 1 DMA, 2 SM copy,
3 CPU RAM hit, 4 NVMe miss streamed to the GPU, 5 NVMe miss to the CPU, 6 drip scratch hit), ``pool_point`` /
``spec_use`` reclassify an NVMe miss served from the prefetch pool, ``svc_routed`` gives the routed count (so
``vram = routed - lanes``) and the request seq that the device stamps use. The device stamps give the layer's expert
span: ``post_start`` to ``wait_end`` (the gate wait's end) or, without a gate wait, to ``stream_end``.

The fit is ordinary least squares of that span (ms) on the lane counts by kind, per record, after the warm-up
verifies. The coefficients are the traced marginal cost of one more lane of each kind, holding the others fixed;
the intercept is the layer's expert cost with every route a VRAM hit. Traced times carry the trace's own overhead.
Also reported: the span by total non-VRAM lanes (median), to show the fit is not driven by a few outliers.

Usage: lane_cost_fit.py CAPTURE [--skip 8]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

KIND = {1: "dma", 2: "sm", 3: "cpu_ram", 4: "nvme_gpu", 5: "cpu_nvme", 6: "drip"}
COLS = ("cpu_ram", "cpu_nvme", "cpu_pool", "dma", "nvme_gpu", "pool", "sm", "drip")


def load_events(d: str) -> dict:
    by = collections.defaultdict(list)
    for path in glob.glob(os.path.join(d, "events.*.jsonl")):
        if path.endswith((".device-stamps.jsonl", ".verify.jsonl")):
            continue
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if "event" in e:
                    by[e["event"]].append(e)
    if not by["lane_kind"]:
        raise ValueError("no lane_kind events: needs an InstrBuild capture (branch expert-breakdown)")
    return by


def load_stamps(d: str) -> dict:
    (path,) = glob.glob(os.path.join(d, "events.*.device-stamps.jsonl"))
    out = {}
    with open(path) as f:
        points = json.loads(f.readline())["points"]
        for line in f:
            r = json.loads(line)
            out[r["seq"]] = {p: r[p] for p in points}
    return out


def verify_starts(d: str) -> list:
    (path,) = glob.glob(os.path.join(d, "events.*.verify.jsonl"))
    with open(path) as f:
        f.readline()
        return [e["ns"] for e in map(json.loads, f) if e["event"] == "verify_begin"]


def records(ev: dict) -> dict:
    rec = collections.defaultdict(lambda: {"lanes": {}, "pool": set(), "spec": set(), "routed": None, "seq": None,
                                           "ns": None})
    for e in ev["svc_routed"]:
        r = rec[(e["row"], e["gen"])]
        r["routed"], r["seq"], r["ns"] = e["b"], e["seq"], e["ns"]
    for e in ev["lane_kind"]:
        rec[(e["row"], e["gen"])]["lanes"][e["a"]] = e["b"] & 0xFF
    for e in ev.get("pool_point", []):
        rec[(e["row"], e["gen"])]["pool"].add(e["a"])
    for e in ev.get("spec_use", []):
        rec[(e["row"], e["gen"])]["spec"].add(e["a"])
    return {k: r for k, r in rec.items() if r["routed"] is not None}


def lstsq(X, y):
    """Plain normal-equation least squares (small, well-conditioned design)."""
    import numpy as np

    A = np.asarray(X, dtype=float)
    b = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    ss_res = float(((b - pred) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    return coef.tolist(), 1 - ss_res / ss_tot if ss_tot else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ev = load_events(args.capture)
    stamps = load_stamps(args.capture)
    starts = verify_starts(args.capture)
    cut = starts[args.skip] if len(starts) > args.skip else 0
    rows = []
    for (row, gen), r in records(ev).items():
        if r["ns"] < cut:
            continue
        s = stamps.get(r["seq"])
        if not s or not s["post_start"]:
            continue
        end = s["wait_end"] if s["wait_end"] else s["stream_end"]
        if not end:
            continue
        n = collections.Counter()
        for e, k in r["lanes"].items():
            cat = KIND[k]
            if cat == "nvme_gpu" and e in r["pool"]:
                cat = "pool"
            elif cat == "cpu_nvme" and e in r["spec"]:
                cat = "cpu_pool"
            n[cat] += 1
        n["vram"] = r["routed"] - len(r["lanes"])
        span_ms = (end - s["post_start"]) / 1e6
        wait_ms = (s["wait_end"] - s["wait_start"]) / 1e6 if s["wait_end"] else 0.0
        stream_ms = (s["stream_end"] - s["stream_start"]) / 1e6 if s["stream_end"] else 0.0
        rows.append((row, n, span_ms, wait_ms, stream_ms))
    if not rows:
        raise ValueError("no stamped records after the skip")
    X = [[1.0] + [n[c] for c in COLS] for _, n, *_ in rows]
    y = [span for _, _, span, _, _ in rows]
    coef, r2 = lstsq(X, y)
    fit = dict(zip(("intercept",) + COLS, coef))
    # a coarser fit: CPU lanes (any), GPU-copied lanes (dma+sm+pool+nvme_gpu), NVMe-read lanes (nvme_gpu+cpu_nvme)
    X2 = [[1.0, n["cpu_ram"] + n["cpu_nvme"] + n["cpu_pool"], n["dma"] + n["sm"] + n["pool"] + n["nvme_gpu"],
           n["nvme_gpu"] + n["cpu_nvme"]] for _, n, *_ in rows]
    coef2, r2b = lstsq(X2, y)
    by_lanes = collections.defaultdict(list)
    for _, n, span, _, _ in rows:
        by_lanes[sum(n[c] for c in COLS)].append(span)
    res = {
        "capture": os.path.abspath(args.capture), "records": len(rows), "skip": args.skip,
        "fit_ms_per_lane": fit, "r2": r2,
        "coarse_fit": dict(zip(("intercept", "cpu_any", "gpu_copied", "nvme_read"), coef2)), "coarse_r2": r2b,
        "span_ms_median": statistics.median(y),
        "wait_ms_median": statistics.median(w for *_, w, _ in rows),
        "stream_ms_median": statistics.median(s for *_, s in rows),
        "span_by_lanes": {str(k): {"n": len(v), "median_ms": round(statistics.median(v), 2)}
                          for k, v in sorted(by_lanes.items())},
        "mean_lanes": {c: statistics.mean(n[c] for _, n, *_ in rows) for c in COLS + ("vram",)},
    }
    print(f"{res['capture']}: {len(rows)} records, span median {res['span_ms_median']:.2f} ms "
          f"(gate wait {res['wait_ms_median']:.2f}, S {res['stream_ms_median']:.2f})")
    print("mean lanes per record: " + ", ".join(f"{c}={v:.2f}" for c, v in res["mean_lanes"].items()))
    print(f"fit (ms per lane, r2={r2:.3f}): " + ", ".join(f"{c}={v:.2f}" for c, v in fit.items()))
    print(f"coarse (r2={r2b:.3f}): " + ", ".join(f"{c}={v:.2f}" for c, v in res["coarse_fit"].items()))
    print("span by non-VRAM lanes: " + " ".join(f"{k}:{v['median_ms']}({v['n']})" for k, v in res["span_by_lanes"].items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
