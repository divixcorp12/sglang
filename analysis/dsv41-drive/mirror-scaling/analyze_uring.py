"""Rank the io_uring sweep (run_uring_sweep.py JSONL): combos by QD1 row p50, with QD2 and CPU cost alongside.

Per (combo, weight, QD, ring) the reps are averaged. The rank key is the mean of the two weights' QD1 p50 (each
weight reads its own rows, the same rows for every combo). The noise line is the median |rep1 - rep2| per cell.
A cell caught in an SPCC slow episode (see episode()) is excluded; a later file's re-run of it replaces it.
Usage: python3 analyze_uring.py uring-main.jsonl [uring-repair*.jsonl ...] [uring-top*.jsonl ...] [--list-episodes]
"""

import json
import statistics as st
import sys
from collections import defaultdict

BASE = "default.block.cuts0.normal"
SPCC = "nvme2n1"


def episode(j, med):
    """An SPCC slow episode: the row far above its class median, or the Samsungs starved (util < 0.65) while the
    SPCC is saturated. Clean cells sit within 1.12x of the median with Samsung util >= 0.71 (a clear gap)."""
    sams = min(d["util"] for d in j["drives"] if d["disk"] != SPCC)
    return j["p50_us"] > 1.5 * med or (j["qd"] <= 2 and sams < 0.65)


def load(*paths, flagged_out=None):
    """Every cell's latest clean record (later files re-run earlier files' episode cells)."""
    recs = []
    for path in paths:
        for line in open(path):
            if line.startswith("{"):
                j = json.loads(line)
                if j["errors"] or j["short"]:
                    raise SystemExit(f"{j['label']}: errors {j['errors']} short {j['short']}")
                recs.append(j)
    cls = defaultdict(list)
    for j in recs:
        _, w, qd, _, ring = j["label"].split("/")
        cls[(w, qd, ring)].append(j["p50_us"])
    med = {k: st.median(v) for k, v in cls.items()}
    latest, bad = {}, {}
    for j in recs:
        _, w, qd, _, ring = j["label"].split("/")
        if episode(j, med[(w, qd, ring)]):
            bad[j["label"]] = j
            latest.pop(j["label"], None)
        else:
            latest[j["label"]] = j
            bad.pop(j["label"], None)
    if flagged_out is not None:
        flagged_out.extend(sorted(bad))
    g = defaultdict(list)
    for j in latest.values():
        combo, w, qd, rep, ring = j["label"].split("/")
        g[(combo, w, int(qd[2:]), int(ring[4:]))].append(j)
    return g, len(recs), len(bad)


def m(runs, f):
    return st.mean(f(r) for r in runs)


def drive(r, disk):
    return next(d for d in r["drives"] if d["disk"] == disk)


def main():
    files = [f for f in sys.argv[1:] if not f.startswith("--") and "top" not in f]
    tops = [f for f in sys.argv[1:] if not f.startswith("--") and "top" in f]
    flagged = []
    g, nrec, nbad = load(*files, flagged_out=flagged)
    if "--list-episodes" in sys.argv:
        print("\n".join(flagged))
        return
    print(f"records {nrec}; cells still in an SPCC episode (excluded): {nbad}")
    diffs = [abs(v[0]["p50_us"] - v[1]["p50_us"]) for v in g.values() if len(v) == 2]
    diffs = diffs or [0.0]
    print(f"cells {sum(len(v) for v in g.values())}; rep-to-rep |dp50| median {st.median(diffs):.0f} us, "
          f"p90 {sorted(diffs)[int(.9 * (len(diffs) - 1))]:.0f} us\n")
    combos = sorted({k[0] for k in g})
    rows = []
    for c in combos:
        q1 = {w: g.get((c, w, 1, 0)) for w in ("w1", "w0.9")}
        q2 = {w: g.get((c, w, 2, 0)) for w in ("w1", "w0.9")}
        if not all(q1.values()) or not all(q2.values()):
            continue
        allq1 = q1["w1"] + q1["w0.9"]
        allq2 = q2["w1"] + q2["w0.9"]
        rows.append(dict(
            combo=c,
            key=(m(q1["w1"], lambda r: r["p50_us"]) + m(q1["w0.9"], lambda r: r["p50_us"])) / 2,
            q1w1=m(q1["w1"], lambda r: r["p50_us"]), q1w09=m(q1["w0.9"], lambda r: r["p50_us"]),
            q1p99=m(allq1, lambda r: r["p99_us"]), q1p90=m(allq1, lambda r: r["p90_us"]),
            q2w1=m(q2["w1"], lambda r: r["p50_us"]), q2w09=m(q2["w0.9"], lambda r: r["p50_us"]),
            q2p99=m(allq2, lambda r: r["p99_us"]), q2gb=m(allq2, lambda r: r["GBps"]),
            sub=m(allq1, lambda r: r["submitter_cpu_per_GB"]), sqp=m(allq1, lambda r: r["sqthread_cpu_per_GB"]),
            wq=m(allq1, lambda r: r["iowq_cpu_per_GB"]), tot=m(allq1, lambda r: r["total_cpu_per_GB"]),
            tot2=m(allq2, lambda r: r["total_cpu_per_GB"]),
            nwrk=max(r["iowq_workers"] for r in allq1 + allq2),
            spcc_last=m(q1["w1"], lambda r: drive(r, SPCC)["last_share"]),
            spcc_last09=m(q1["w0.9"], lambda r: drive(r, SPCC)["last_share"]),
            sqe=" / ".join(f"{m(q1['w1'], lambda r, d=d: drive(r, d)['sqe_p50_us']):.0f}"
                           for d in ("nvme0n1", SPCC, "nvme3n1")),
            spr=m(allq1, lambda r: r["sqes_per_row"]),
        ))
    rows.sort(key=lambda x: x["key"])
    base = next((r for r in rows if r["combo"] == BASE), rows[0])
    print("| # | combo | QD1 p50 1:1:1 / 1:0.9:1 (us) | vs base | QD1 p90 / p99 | QD2 p50 1:1:1 / 1:0.9:1 | QD2 p99 | QD2 GB/s |"
          " CPU-s/GB QD1: submit + SQ + io-wq = total | QD2 total | io-wq workers | SPCC last (1:1:1 / 1:0.9:1) |"
          " SQE p50 nvme0/SPCC/nvme2 (1:1:1) | SQEs/row |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | `{r['combo']}` | {r['q1w1']:.0f} / {r['q1w09']:.0f} | {r['key'] - base['key']:+.0f} |"
              f" {r['q1p90']:.0f} / {r['q1p99']:.0f} | {r['q2w1']:.0f} / {r['q2w09']:.0f} | {r['q2p99']:.0f} | {r['q2gb']:.2f} |"
              f" {r['sub']:.3f} + {r['sqp']:.3f} + {r['wq']:.3f} = {r['tot']:.3f} | {r['tot2']:.3f} | {r['nwrk']} |"
              f" {r['spcc_last']:.2f} / {r['spcc_last09']:.2f} | {r['sqe']} | {r['spr']:.1f} |")
    # Main effects: mean key over combos sharing one factor value.
    print("\n## Main effects (mean of QD1 p50 over the two weights, us; over the combos that have the value)\n")
    fac = {
        "mode": lambda c: c.split(".")[0].split("@")[0],
        "sq cpu": lambda c: c.split(".")[0].split("@")[1] if "@" in c else "-",
        "wait": lambda c: c.split(".")[1],
        "cuts": lambda c: c.split(".")[2],
        "buffers": lambda c: ".".join(c.split(".")[3:]),
    }
    for name, f in fac.items():
        vals = defaultdict(list)
        for r in rows:
            vals[f(r["combo"])].append(r["key"])
        print(f"- {name}: " + ", ".join(f"{k} {st.mean(v):.0f} (n={len(v)})" for k, v in sorted(vals.items())))
    if tops:
        top(tops, base)


def top(paths, base):
    g, nrec, nbad = load(*paths)
    print(f"\ntop records {nrec}; excluded as SPCC episodes: {nbad}")
    print("\n## Top combos: QD4, and a 4096-entry ring\n")
    print("| combo | weights | QD4 p50 / p99 / GB/s (default ring) | QD1 p50 (ring 4096) | QD4 p50 / p99 / GB/s (ring 4096) |")
    print("|---|---|---|---|---|")
    for c in sorted({k[0] for k in g}):
        for w in ("w1", "w0.9"):
            a, b, d = g.get((c, w, 4, 0)), g.get((c, w, 1, 4096)), g.get((c, w, 4, 4096))
            if not (a and b and d):
                continue
            print(f"| `{c}` | {w} | {m(a, lambda r: r['p50_us']):.0f} / {m(a, lambda r: r['p99_us']):.0f} / {m(a, lambda r: r['GBps']):.2f} |"
                  f" {m(b, lambda r: r['p50_us']):.0f} | {m(d, lambda r: r['p50_us']):.0f} / {m(d, lambda r: r['p99_us']):.0f} / {m(d, lambda r: r['GBps']):.2f} |")


if __name__ == "__main__":
    main()
