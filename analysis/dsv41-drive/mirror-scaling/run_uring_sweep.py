"""io_uring settings sweep for mirror_bench on the production 3-root set (nvme0, SPCC, nvme2), weights 1:1:1 and 1:0.9:1.

Every cell takes rowimg-disk.lock (blocking, per cell, so decode arms can interleave between cells but never overlap
one) and runs on cores 19-35 (NUMA node 1, inside 0-63; core 18 is kept for the node-1 SQ thread, core 2 is the node-0
one). Cells are shuffled with a fixed seed to spread drift; every combo reads the same rows for a (weight, QD, rep).

  phase main: 48 combos x 2 weights x QD {1, 2} x 2 reps.
  phase top:  the combos named by --top (their labels), QD 4 at the default ring, and QD 1 and 4 at a 4096-entry ring.

Usage: python3 run_uring_sweep.py <bench> <out.jsonl> main|top [--top LABEL ...]
"""

import argparse
import itertools
import random
import subprocess
import sys

ROOTS = ["--root", "/mnt/nvme0/dsv41_flash", "--root", "/mnt/nvme4/dsv41_flash", "--root", "/mnt/nvme2/dsv41_flash"]
LOCK = "/data/models/slang/nvfp4-work/rowimg-disk.lock"
CORES = "19-35"
SQ_CPUS = {"n1": 18, "n0": 2}
BUFS = {
    "normal": [],
    "normal+ff": ["--fixed-files"],
    "rvf-slabs+ff": ["--read-mode", "readv_fixed", "--arena", "0", "--fixed-files"],
    "rvf-arena+ff": ["--read-mode", "readv_fixed", "--arena", "1", "--fixed-files"],
}


def combos():
    out = []
    for wait, cuts, buf in itertools.product(["block", "spin"], [0, 1], BUFS):
        out.append(("default", None, wait, cuts, buf))
    for cuts, buf in itertools.product([0, 1], BUFS):
        out.append(("iopoll", None, "reap", cuts, buf))
    for cpu, wait, cuts, buf in itertools.product(SQ_CPUS, ["block", "spin"], [0, 1], ["normal", "rvf-arena+ff"]):
        out.append(("sqpoll", cpu, wait, cuts, buf))
    for cpu, cuts, buf in itertools.product(SQ_CPUS, [0, 1], ["normal", "rvf-arena+ff"]):
        out.append(("sqpoll_iopoll", cpu, "block", cuts, buf))
    return out


def label(c):
    mode, cpu, wait, cuts, buf = c
    return f"{mode}{'@' + cpu if cpu else ''}.{wait}.cuts{cuts}.{buf}"


def args_for(c):
    mode, cpu, wait, cuts, buf = c
    a = ["--mode", mode, "--wait", "spin" if wait == "spin" else "block"]
    if cpu:
        a += ["--sq-cpu", str(SQ_CPUS[cpu])]
    if cuts:
        a += ["--cuts"]
    return a + BUFS[buf]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bench")
    p.add_argument("out")
    p.add_argument("phase", choices=["main", "top"])
    p.add_argument("--top", nargs="*", default=[])
    p.add_argument("--reps", type=int, default=2)
    a = p.parse_args()
    by_label = {label(c): c for c in combos()}
    cells = []
    weights = {"w1": "1:1:1", "w0.9": "1:0.9:1"}
    if a.phase == "main":
        for c, (wk, w), qd, rep in itertools.product(combos(), weights.items(), [1, 2], range(1, a.reps + 1)):
            cells.append((c, wk, w, qd, rep, 0))
    else:
        for name in a.top:
            c = by_label[name]
            for (wk, w), rep in itertools.product(weights.items(), range(1, a.reps + 1)):
                cells += [(c, wk, w, 4, rep, 0), (c, wk, w, 1, rep, 4096), (c, wk, w, 4, rep, 4096)]
    random.Random(20260928).shuffle(cells)
    print(f"{len(cells)} cells", file=sys.stderr, flush=True)
    with open(a.out, "a") as out:
        for i, (c, wk, w, qd, rep, ring) in enumerate(cells):
            seed = 7000 + (100 if wk == "w0.9" else 0) + 10 * qd + rep
            rows = 2000 if qd < 4 else 3000
            cmd = ["flock", LOCK, "taskset", "-c", CORES, a.bench, "--label", f"{label(c)}/{wk}/qd{qd}/r{rep}/ring{ring}",
                   *ROOTS, "--weights", w, "--qd", str(qd), "--rows", str(rows), "--seed", str(seed), *args_for(c)]
            if ring:
                cmd += ["--ring", str(ring)]
            r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
            out.flush()
            print(f"[{i + 1}/{len(cells)}] rc={r.returncode} {label(c)} {wk} qd{qd} r{rep} ring{ring} {r.stderr.strip()}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
