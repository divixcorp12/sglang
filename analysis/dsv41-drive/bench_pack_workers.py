"""Packing throughput and exposed tail of the RAM-miss reader: inline on the owner vs packing workers.

CPU only, no GPU. Reads production-sized EXL3 rows (a row is ~12.7 MiB) from page-cached files under
/dev/shm into pinned-style slabs through the C++ reader's traced test entry point, and reads the
reader's own stage stamps:

  tail      pack_end - last_cqe: packing left after the final completion. Without a hold this is the
            natural overlap regime (cached reads finish a few ms apart, so early rows pack while later
            ones read); with the LAST row's completions withheld until every other row has packed
            (scenario "last": a simulated slow drive) it is exactly that one row's copy; with every row
            withheld and released together (scenario "burst") it is the time to pack n rows that are
            ready at once, the spread Task 6 wants to exploit.
  row_pack  pack_ns / rows: the mean time one row's copy takes, first byte to last.
  read      last_cqe - submit: the read window the packing overlaps.

What this cannot say: real NVMe completion timing (cached reads complete a few ms apart, set by the page
cache copy and the kernel's io workers, not by a drive), and O_DIRECT (buffered reads leave each bounce slot warm in the cache of the
core that read it; rows are larger than L3 so most of it is evicted before the copy, but the effect is
not removed). Every number is written with its conditions.

Run under `taskset -c 0-63` with OMP/MKL threads capped; the script then confines itself to the
least-busy cores of that set and refuses cores 64-71.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch

RESERVED = set(range(64, 72))
ROW_HIDDEN, ROW_INTER = 4096, 2880  # trellis 3 * 4096 * 2880 * 3 / 8 = 13.27 MB: a production row is 13.3 MB padded


def cpu_busy(seconds: float = 1.0) -> dict[int, float]:
    def snap():
        out = {}
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                vals = list(map(int, f[1:9]))
                out[int(f[0][3:])] = (sum(vals) - vals[3] - vals[4], sum(vals))
        return out

    a = snap()
    time.sleep(seconds)
    b = snap()
    return {c: (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a}


def foreign_processes(top: int = 8) -> list[dict]:
    out = subprocess.run(["ps", "-eo", "pid,pcpu,psr,comm", "--sort=-pcpu"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1 : top + 4]:
        pid, pcpu, psr, comm = line.split(None, 3)
        if comm in ("ps", "pgrep") or int(pid) == os.getpid():
            continue  # the measurement's own tools are not foreign load
        rows.append({"pid": int(pid), "pcpu": float(pcpu), "core": int(psr), "comm": comm})
    return rows


def conditions(cores: list[int], busy: dict[int, float]) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "load1_start": os.getloadavg()[0],
        "foreign_top_processes_start": foreign_processes(),
        "affinity_at_start": sorted(os.sched_getaffinity(0)),
        "cores_used": cores,
        "busy_of_cores_used": {c: round(busy[c], 3) for c in cores},
        "kernel": platform.release(),
        "git_head": sha,
        "build": "c++20 -O3 (sglang JIT), liburing",
        "row_hidden_inter": [ROW_HIDDEN, ROW_INTER],
        "torch_threads": torch.get_num_threads(),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "mkl": os.environ.get("MKL_NUM_THREADS"),
    }


def build_setup(root: Path, experts: int, mirror_roots: tuple[Path, Path] | None):
    """``mirror_roots``, when given, is where mirror 0 and mirror 1 are copied -- pass roots on two
    different drives to measure with the mirrors split the way production splits them (the registered
    O_DIRECT run put both under the same ``--work-dir`` instead; see PACK_WORKERS.md). ``None`` disables
    mirroring."""
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    root.mkdir(parents=True, exist_ok=True)
    write_fake_exl3(str(root), num_layers=2, num_experts=experts, hidden=ROW_HIDDEN, inter=ROW_INTER)
    layout = build_exl3_expert_layout(str(root))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    capacity = 8
    slabs = {
        layer: {
            name: allocate_host_slab(capacity, specs[name].row_shape, specs[name].dtype, register=False)
            for name in EXL3_STREAMED_NAMES
        }
        for layer in range(2)
    }
    mirror_args = {}
    if mirror_roots is not None:
        roots = tuple(str(r) for r in mirror_roots)
        for r in roots:
            shutil.copytree(root, r)
        mirror_args = dict(roots=roots, policy=StaticSplitPolicy((1.0, 1.0)), source_root=str(root))
    tables = exl3_ram_miss_tables(layout, fmt.segment_map(), slabs, **mirror_args)
    return tables, slabs


def diskstats_of(path: str) -> str:
    """The /proc/diskstats line of the device holding `path` (empty when it has none, e.g. tmpfs)."""
    st = os.stat(path)
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if int(f[0]) == os.major(st.st_dev) and int(f[1]) == os.minor(st.st_dev):
            return line
    return ""


def one_read(tables, experts, mode, *, scenario: str, direct: bool = False, owner_core: int = -1):
    from sglang.kernels.ops.moe.exl3_ram_miss import read_rows_traced

    workers, split = mode
    n = len(experts)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    result, rec = read_rows_traced(
        tables, 1, experts, list(range(n)), direct=direct, step=8,
        hold_ordinal={"natural": -1, "last": n - 1, "burst": 0}[scenario], hold_rest=scenario == "burst",
        pack_workers=workers, pack_split=split, owner_core=owner_core,
    )
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    assert result == 1, rec
    rows = rec["row_pack"]
    last = rows[n - 1]
    return {
        "tail_ns": rec["pack_end"] - rec["last_cqe"],
        "row_pack_ns": rec["pack_ns"] / n,
        "last_ns": last["end"] - last["start"],
        "read_ns": rec["last_cqe"] - rec["submit"],
        # Whole process, so it counts the owner's polling and every worker; the reader's setup (open, bounce
        # allocation and its page faults) is in it too, identically for every mode.
        "cpu_ns": cpu * 1e9,
        "wall_ns": wall * 1e9,
    }


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--modes", default="0:0,1:1,2:1,4:1,2:2,4:4,8:8", help="workers:split, 0:0 is inline")
    ap.add_argument("--cores", type=int, default=12, help="least-busy cores of the allowed set to use")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--work-dir", default="/dev/shm")
    ap.add_argument(
        "--work-dir2", default=None,
        help="drive for mirror 1 (mirror 0 stays under --work-dir); default puts both mirrors under "
        "--work-dir, as the registered run did",
    )
    ap.add_argument("--direct", action="store_true", help="O_DIRECT reads: needs --work-dir on a real drive")
    ap.add_argument("--scenarios", default="natural,last,burst", help="natural is the only one a real drive needs")
    ap.add_argument(
        "--owner-core", type=int, default=-1,
        help="test-only scaffold (PACK_WORKERS.md): pin the owner thread to this core, excluded from the "
        "packing pool's mask; -1 (default) leaves the reader unpinned, unchanged from before this flag existed",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    allowed = sorted(os.sched_getaffinity(0) - RESERVED)
    assert allowed and not (os.sched_getaffinity(0) & RESERVED), "run under taskset -c 0-63"
    busy = cpu_busy()
    if args.owner_core >= 0:
        # The least-busy set is picked fresh each run, so a fixed --owner-core would fail the old
        # membership check most of the time by luck alone. Force it in instead, keeping the same total
        # core count (so the pinned and unpinned runs use equally many cores): the owner core plus the
        # `--cores - 1` least-busy of the rest.
        assert args.owner_core in allowed, f"--owner-core {args.owner_core} is not in the allowed set {allowed}"
        rest = [c for c in sorted(allowed, key=lambda c: busy[c]) if c != args.owner_core]
        cores = sorted([args.owner_core] + rest[: args.cores - 1])
    else:
        cores = sorted(sorted(allowed, key=lambda c: busy[c])[: args.cores])
    cond = conditions(cores, busy)
    cond["owner_core"] = args.owner_core
    os.sched_setaffinity(0, cores)  # workers inherit this mask minus 64-71 (and minus owner_core, if pinned)
    modes = [tuple(map(int, m.split(":"))) for m in args.modes.split(",")]

    root = Path(tempfile.mkdtemp(dir=args.work_dir, prefix="packbench_"))
    root2 = Path(tempfile.mkdtemp(dir=args.work_dir2, prefix="packbench2_")) if args.work_dir2 else None
    try:
        mirror_roots = (root / "ckpt_mirror0", (root2 or root) / "ckpt_mirror1")
        tables, slabs = build_setup(root / "ckpt", args.experts, mirror_roots)
        cond["diskstats_start"] = diskstats_of(str(root))
        if root2 is not None:
            cond["diskstats2_start"] = diskstats_of(str(root2))
        cond["direct"] = args.direct
        results = {}
        for _ in range(2):  # warm the page cache and the JIT
            one_read(tables, list(range(2)), modes[0], scenario="natural", direct=args.direct, owner_core=args.owner_core)
        for rep in range(args.reps):
            for n in args.rows:
                experts = [(rep * n + i) % args.experts for i in range(n)]
                for mode in modes:  # every mode in every round: drift hits them all alike
                    for scenario in args.scenarios.split(","):
                        if scenario != "natural" and n == 1:
                            continue
                        got = one_read(tables, experts, mode, scenario=scenario, direct=args.direct, owner_core=args.owner_core)
                        results.setdefault((mode, n, scenario), []).append(got)
        cond["diskstats_end"] = diskstats_of(str(root))
        if root2 is not None:
            cond["diskstats2_end"] = diskstats_of(str(root2))
        cond["load1_end"] = os.getloadavg()[0]
        cond["foreign_top_processes_end"] = foreign_processes()
        summary = []
        for (mode, n, scenario), got in sorted(results.items()):
            row = {"workers": mode[0], "split": mode[1], "rows": n, "scenario": scenario, "n": len(got)}
            for key in ("tail_ns", "row_pack_ns", "last_ns", "read_ns", "cpu_ns", "wall_ns"):
                values = [g[key] / 1e6 for g in got]
                row[key[:-3]] = {"p50_ms": round(pct(values, 0.5), 3), "p90_ms": round(pct(values, 0.9), 3), "min_ms": round(min(values), 3)}
            summary.append(row)
        Path(args.out).write_text(json.dumps({"conditions": cond, "results": summary}, indent=1))
        print(json.dumps(cond, indent=1))
        for row in summary:
            print(row)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if root2 is not None:
            shutil.rmtree(root2, ignore_errors=True)


if __name__ == "__main__":
    main()
