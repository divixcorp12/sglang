#!/usr/bin/env python3
"""Background NVMe reader for the `nvme` load arm of C_MEASUREMENT_PREREG.md. CPU only, no CUDA.

It reads EXL3 rows from the two mirror roots with the production reader (`UringFileReader` through
`bench_row_scheduling.Harness`, the within-row split, the production scatter into the bounce), in back-to-back
batches of 6 rows, so the drives DMA into host memory as hard as one submitter thread can make them. **This is an
upper-bound load, not a decode duty cycle**, and it is not the service: it bounds the effect of concurrent
drive-to-host DMA on the GPU's host reads; it does not reproduce a decode step.

It reads only while GO exists, exits when STOP exists, and writes a log of (CLOCK_MONOTONIC ns, bytes requested)
per batch so the harness can attribute reader bytes to each cell window.

    numactl --membind=1 taskset -c <cores> python3 nvme_load_reader.py --go G --stop S --ready R --out LOG.json
"""
import argparse, json, os, sys, time, itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # bench_row_scheduling.py, bench_mirror_rows.py, drive_conditions.py

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--go", required=True); p.add_argument("--stop", required=True); p.add_argument("--ready", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", nargs="+", type=int, default=[3, 7, 11, 15, 19, 23, 27, 31])
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--rows", type=int, default=240, help="distinct requests in the replay ring")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--dry-run", action="store_true", help="no I/O: emit a synthetic log so the harness plumbing can be tested")
    a = p.parse_args()
    log = []
    if a.dry_run:
        Path(a.ready).write_text("ready\n")
        while not os.path.exists(a.stop):
            if os.path.exists(a.go):
                time.sleep(0.01); log.append([time.monotonic_ns(), 6 * 13_400_000])
            else:
                time.sleep(0.02)
        Path(a.out).write_text(json.dumps({"log": log, "dry_run": True})); return 0
    import bench_mirror_rows as base
    import bench_row_scheduling as bs
    import drive_conditions as dc
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    source = os.path.realpath(base.DEFAULT_SOURCE); roots = [os.path.realpath(r) for r in base.DEFAULT_ROOTS]
    layout = build_exl3_expert_layout(source)
    drives = [dc.resolve_drive(r) for r in roots]
    batches = bs.build_replay(layout.num_experts, a.layers, [a.batch], a.rows, a.seed)
    harness = bs.Harness.build(layout, source, roots, a.layers, a.batch, drives=drives)
    harness.open_everything(batches)
    spec = bs.ArmSpec("within-row 1:1", "within_row", tuple(1.0 for _ in roots))
    planner = bs.make_planner(spec, len(roots))
    before = bs.read_diskstats(roots)
    ring = itertools.cycle(batches)
    for _ in range(3):                                   # warm the files and the bounce, unlogged
        harness.run_batch(spec, planner, next(ring), -1, False)
    Path(a.ready).write_text("ready\n")
    while not os.path.exists(a.stop):
        if not os.path.exists(a.go):
            time.sleep(0.02); continue
        sample, _ = harness.run_batch(spec, planner, next(ring), -1, False)
        log.append([time.monotonic_ns(), int(sample.requested_bytes)])
    after = bs.read_diskstats(roots)
    Path(a.out).write_text(json.dumps({"log": log, "diskstats_delta": bs.diskstats_delta(before, after), "roots": roots,
                                       "cpus": sorted(os.sched_getaffinity(0)), "batch": a.batch}, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main())
