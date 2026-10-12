"""Summarize mirror_bench JSONL (run_matrix.sh) into the tables of results.md.

Per (root set, QD): the median over reps of row p50/p90/p99/max and GB/s, and per drive the median SQE/part
latencies, utilization, GB/s and straggler share. Then the per-drive-type fit part_time = a + b * part_bytes at QD1
(least squares over every rep and root set), and the prediction it gives for each root set: max over its drives.
Usage: python3 analyze.py results/matrix.jsonl.gz [results/matrix-raw.csv.gz]
"""

import csv
import gzip
import json
import statistics as st
import sys
from collections import defaultdict

SAMSUNG = {"nvme0n1", "nvme3n1"}
NAME = {"nvme0n1": "nvme0", "nvme2n1": "SPCC", "nvme3n1": "nvme2"}
ORDER = ["s-nvme0", "s-spcc", "s-nvme2", "p-nvme0+spcc", "p-nvme0+nvme2", "p-spcc+nvme2", "t-equal", "t-spcc0.5"]


def load(path):
    cells = defaultdict(list)
    for line in (gzip.open(path, "rt") if path.endswith(".gz") else open(path)):
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        if j["errors"] or j["short"]:
            raise SystemExit(f"{j['label']}: errors {j['errors']} short {j['short']}")
        name, qd, rep = j["label"].split("/")
        cells[(name, j["qd"])].append(j)
    return cells


def med(xs):
    return st.median(xs) if xs else float("nan")


def fit(points):
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in points)
    ss_tot = sum((y - my) ** 2 for _, y in points)
    return a, b, 1 - ss_res / ss_tot


def main():
    cells = load(sys.argv[1])
    print("## Scaling (median over reps)\n")
    print("| root set | QD | reps | row p50 ms | p90 | p99 | max | GB/s | per drive: GB/s, util, part p50 ms, last share |")
    print("|---|---|---|---|---|---|---|---|---|")
    for qd in (1, 2, 4):
        for name in ORDER:
            runs = cells.get((name, qd))
            if not runs:
                continue
            drives = []
            for k, d in enumerate(runs[0]["drives"]):
                col = [r["drives"][k] for r in runs]
                drives.append(
                    f"{NAME[d['disk']]} {med([c['GBps'] for c in col]):.2f} GB/s u{med([c['util'] for c in col]):.2f}"
                    f" {med([c['part_p50_us'] for c in col]) / 1e3:.2f} ms"
                    + (f" last {med([c['last_share'] for c in col]):.2f}" if len(runs[0]["drives"]) > 1 else "")
                )
            print(
                f"| {name} | {qd} | {len(runs)} | {med([r['p50_us'] for r in runs]) / 1e3:.2f} |"
                f" {med([r['p90_us'] for r in runs]) / 1e3:.2f} | {med([r['p99_us'] for r in runs]) / 1e3:.2f} |"
                f" {med([r['max_us'] for r in runs]) / 1e3:.1f} | {med([r['GBps'] for r in runs]):.2f} | {'; '.join(drives)} |"
            )
    print("\n## Per-rep row p50 ms at QD1 (spread)\n")
    for name in ORDER:
        runs = cells.get((name, 1), [])
        print(f"- {name}: " + ", ".join(f"{r['p50_us'] / 1e3:.2f}" for r in runs)
              + " | p99: " + ", ".join(f"{r['p99_us'] / 1e3:.2f}" for r in runs))

    # Fit per drive type at QD1: part completion p50 vs the part's bytes.
    print("\n## Fit at QD1: part p50 = a + b * part_bytes\n")
    pts = {"samsung": [], "spcc": []}
    for (name, qd), runs in cells.items():
        if qd != 1:
            continue
        for r in runs:
            for d in r["drives"]:
                if d["part_bytes"]:
                    pts["samsung" if d["disk"] in SAMSUNG else "spcc"].append((d["part_bytes"], d["part_p50_us"]))
    coef = {}
    for kind, p in pts.items():
        a, b, r2 = fit(p)
        coef[kind] = (a, b)
        print(f"- {kind}: a = {a:.0f} us, b = {b * 1e3:.3f} us/KB -> {1e-3 / b:.2f} GB/s marginal, R^2 = {r2:.3f}, n = {len(p)}")
    print("\n| root set | predicted row (max over drives of a + b*bytes) ms | measured p50 ms | measured/predicted |")
    print("|---|---|---|---|")
    for name in ORDER:
        runs = cells.get((name, 1))
        if not runs:
            continue
        pred = max(
            coef["samsung" if d["disk"] in SAMSUNG else "spcc"][0]
            + coef["samsung" if d["disk"] in SAMSUNG else "spcc"][1] * d["part_bytes"]
            for d in runs[0]["drives"]
            if d["part_bytes"]
        )
        m = med([r["p50_us"] for r in runs])
        print(f"| {name} | {pred / 1e3:.2f} | {m / 1e3:.2f} | {m / pred:.2f} |")

    if len(sys.argv) > 2:
        straggler(sys.argv[2])


def straggler(path):
    """From the raw rows: at QD1, the row time vs the max of the non-SPCC parts (what the row would take if the SPCC
    part had been free), and per-set p50/p99 of each part."""
    rows = defaultdict(list)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for rec in csv.reader(f):
            label, _, layer, expert, lat, p0, p1, p2 = rec
            name, qd, rep = label.split("/")
            rows[(name, qd)].append((float(lat), float(p0), float(p1), float(p2)))
    print("\n## Straggler detail at QD1 (raw rows, all reps pooled)\n")
    print("| root set | rows | row p50 | row p99 | part p50/p99 by root (ms) | row p50/p99 if the slowest part were free |")
    print("|---|---|---|---|---|---|")
    for name in ORDER:
        rs = rows.get((name, "qd1"))
        if not rs or name.startswith("s-"):
            continue
        lat = sorted(r[0] for r in rs)
        n = len(lat)
        parts = []
        k = 3 if name.startswith("t-") else 2
        for i in range(k):
            v = sorted(r[1 + i] for r in rs)
            parts.append(f"{v[n // 2] / 1e3:.2f}/{v[int(.99 * (n - 1))] / 1e3:.2f}")
        second = sorted(sorted(r[1:1 + k])[-2] for r in rs)
        print(f"| {name} | {n} | {lat[n // 2] / 1e3:.2f} | {lat[int(.99 * (n - 1))] / 1e3:.2f} | {' ; '.join(parts)} |"
              f" {second[n // 2] / 1e3:.2f}/{second[int(.99 * (n - 1))] / 1e3:.2f} |")


if __name__ == "__main__":
    main()
