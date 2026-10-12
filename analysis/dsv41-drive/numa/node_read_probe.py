#!/usr/bin/env python3
"""Does the GPU (NUMA node 0 on divix01) read node-1 pinned host memory slower than node-0 pinned memory?

No server. Two pinned host slabs of the same size, one bound to each node before any page is touched
(expert_host_tier.allocate_host_slab with a host_numa placement, as the production tier binds), page placement
verified with host_numa.page_nodes. The kernels are copy-mechanism's (../copy-mechanism/mech_bench.cuh, unchanged):

- ce:       cudaMemcpyAsync H2D per job (mech_ce_each), at the production copy-engine sizes.
- sm:       the ld.global.cv streaming read (sm_kernel) at 16 KiB, 32 KiB (the S shape) and 512 KiB in flight.
- cw_real:  the production copy-wait read (copy_wait_read) over the four small tensors per lane, 4 lanes.
- serial:   one thread, ld.acquire.sys of one host word 1000 times (the 780 ns figure's probe).
- cold:     one ld.acquire.sys per launch at 2000 offsets spread over the slab (no reuse: DRAM, not the CPU's LLC).
- pingpong: GPU release -> host acquire+release -> GPU acquire, 256 rounds (the 1.408 us round trip's probe), the
            words on the node under test and the host thread on a node-0 core, then a node-1 core.

Every metric is measured `--reps` times, alternating nodes within each rep so drift hits both alike. `--contend`
repeats ce (8.8 MB) and the latency probes while a separate process streams O_DIRECT reads from the node-1 NVMe
mirrors into a node-0 buffer: NVMe writes crossing UPI into node 0 while the GPU reads the node under test.

    node_read_probe.py --repo <repo> --out <jsonl> [--reps 7] [--contend]
    node_read_probe.py --report <jsonl>        # the markdown tables
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MECH = HERE.parent / "copy-mechanism"
SLAB_BYTES = 2 << 30
# Production copy-engine copies (copy_engine.h: whole tensor entries per lane; with SM small copies on, the two big
# segments of a row) and their piece-stream quarters (reader_base.h kSubReads = 4).
CE_SIZES = (8_847_360, 4_423_680, 2_211_840, 1_105_920)
SMALL = (20_480, 9_216, 10_240, 4_608)  # the copy wait's four small tensors per lane, in its read order
SM_SHAPES = ((4, 1, "16KiB"), (8, 1, "32KiB"), (32, 4, "512KiB"))  # (grid, unroll) at 16 B loads: grid*256*unroll*16
KIND_CV, KIND_CW_REAL = 0, 5
BATCH_BYTES = 256 << 20
COLD_READS = 2000
PING_ROUNDS = 256
PING_CORES = (40, 60)  # node 0, node 1; both inside the GPU-work range 32-63


def emit(out, rec):
    line = json.dumps(rec)
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


def slab(node: int):
    import torch

    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
    from sglang.srt.layers.moe.host_numa import page_nodes

    s = allocate_host_slab(1, (SLAB_BYTES,), torch.uint8, register=True, placement=((node, SLAB_BYTES),)).view(-1)
    s[:: 4096].fill_(node + 1)  # already resident from registration; this makes the content node-distinct
    return s, dict(page_nodes(s, samples=1024))


def jobs(src, size: int, dst, count=None):
    """`count` jobs of `size` bytes at offsets spread evenly over the slab, into rotating VRAM slots."""
    import torch

    n = count or max(1, min(BATCH_BYTES // size, 2048))
    stride = (src.numel() - size) // max(n - 1, 1) // 4096 * 4096
    slots = dst.numel() // size
    rows = [[src.data_ptr() + i * stride, dst.data_ptr() + (i % slots) * size, size] for i in range(n)]
    return torch.tensor(rows, dtype=torch.int64)


def small_jobs(src, dst, lanes=4, rows=256):
    """The copy wait's shape: per lane its four small tensors, lanes from rows spread over the slab."""
    import torch

    lane_bytes = sum(SMALL)
    stride = (src.numel() - lane_bytes) // rows // 4096 * 4096
    table = []
    for r in range(rows):
        at, d = src.data_ptr() + r * stride, dst.data_ptr() + (r % lanes) * lane_bytes
        for b in SMALL:
            table.append([at, d, b])
            at, d = at + b, d + b
    return torch.tensor(table, dtype=torch.int64)


def timed(run):
    import torch

    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    run()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1)


def measure_bw(run, nbytes):
    run()  # warm
    ms = timed(run)
    return round(nbytes / (ms * 1e-3) / 1e9, 4)


def latency(mod, src, dev_word):
    import torch

    lat = torch.zeros(3, dtype=torch.int64, device="cuda")
    word = src[1 << 20 : (1 << 20) + 4].view(torch.int32)
    mod.mech_latency(word, dev_word, lat, 1000)
    serial = lat.cpu().tolist()[0] / 1000
    cold = []
    stride = (src.numel() - 64) // COLD_READS // 4096 * 4096
    for i in range(COLD_READS):
        at = i * stride + 2048
        mod.mech_latency(src[at : at + 4].view(torch.int32), dev_word, lat, 1)
        cold.append(lat.cpu().tolist()[0])
    return serial, cold


def pingpong(mod, src):
    import torch

    words = src[(1 << 30) : (1 << 30) + 8].view(torch.int32)
    rtt = torch.zeros(PING_ROUNDS, dtype=torch.int64, device="cuda")
    mod.mech_pingpong(words, rtt, PING_ROUNDS)
    return rtt.cpu().tolist()[16:]  # the first rounds include the kernel's start


def pct(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---- the O_DIRECT NVMe stream (a process of its own: no GIL shared with the measurement) ----

def nvme_reader(dirs: list[str], threads: int, chunk: int) -> None:
    from sglang.srt.layers.moe.host_numa import allocate_bound, page_nodes

    files = []
    for d in dirs:
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.getsize(p) >= 1 << 30:
                files.append(p)
    buf = allocate_bound(threads * chunk, [(0, 0, 1)], threads * chunk)  # all on node 0
    view = memoryview(buf.numpy())
    for t in range(threads):
        view[t * chunk] = 0  # fault each slice in on node 0
    placed = dict(page_nodes(buf, samples=64))
    stop = threading.Event()
    done = [0] * threads

    def run(t):
        mine = view[t * chunk : (t + 1) * chunk]
        f = files[t % len(files)]
        fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
        size = os.path.getsize(f) // chunk * chunk
        off = (t * 7919 * chunk) % size
        try:
            while not stop.is_set():
                n = os.preadv(fd, [mine], off)
                done[t] += n
                off = (off + chunk) % size
        finally:
            os.close(fd)

    ts = [threading.Thread(target=run, args=(t,), daemon=True) for t in range(threads)]
    t0 = time.monotonic()
    for t in ts:
        t.start()
    print(json.dumps({"reader": "started", "files": len(files), "buffer_nodes": placed}), flush=True)
    sys.stdin.read()  # the parent closes stdin to stop
    stop.set()
    for t in ts:
        t.join()
    secs = time.monotonic() - t0
    print(json.dumps({"reader": "stopped", "bytes": sum(done), "seconds": round(secs, 2),
                      "gbs": round(sum(done) / secs / 1e9, 3)}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--contend", action="store_true")
    ap.add_argument("--nvme-dirs", default="/mnt/nvme0/dsv41_flash/exl3_row_images:/mnt/nvme4/dsv41_flash/exl3_row_images")
    ap.add_argument("--reader-cores", default="32-35")  # node 1 (the drives' node), inside 32-63
    ap.add_argument("--report")
    ap.add_argument("--nvme-reader", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.report:
        print(report([json.loads(l) for l in open(a.report) if l.strip()]))
        return 0
    if a.nvme_reader:
        nvme_reader(a.nvme_dirs.split(":"), threads=16, chunk=1 << 20)
        return 0
    import torch

    import sglang

    repo = Path(a.repo).resolve()
    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    sys.path.insert(0, str(MECH))
    import mech_bench

    torch.cuda.set_stream(torch.cuda.Stream())
    probe = json.loads((MECH / "gen3/probe.json").read_text())
    mod = mech_bench.load(repo, probe)
    os.sched_setaffinity(0, {PING_CORES[0]})
    out = open(a.out, "a")
    emit(out, {"meta": True, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
               "cuda": torch.version.cuda, "sglang": sglang.__file__, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "slab_bytes": SLAB_BYTES, "reps": a.reps, "ce_sizes": CE_SIZES, "small": SMALL,
               "main_thread_core": PING_CORES[0]})
    slabs = {}
    for node in (0, 1):
        slabs[node], placed = slab(node)
        emit(out, {"kind": "placement", "node": node, "sampled_pages": placed, "bytes": SLAB_BYTES})
        if set(placed) != {node}:
            raise SystemExit(f"slab for node {node} has pages on {placed}")
    dst = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")  # 512 MiB of slots: past the 96 MiB L2
    dev_word = torch.zeros(1, dtype=torch.int32, device="cuda")
    tables = {(n, s): jobs(slabs[n], s, dst) for n in (0, 1) for s in CE_SIZES}
    dev_tables = {k: v.to("cuda") for k, v in tables.items()}
    small = {n: small_jobs(slabs[n], dst) for n in (0, 1)}
    small_dev = {n: t.to("cuda") for n, t in small.items()}

    def one_rep(rep, phase, full):
        for node in ((0, 1) if rep % 2 == 0 else (1, 0)):
            base = {"phase": phase, "rep": rep, "node": node}
            for s in CE_SIZES if full else CE_SIZES[:1]:
                nbytes = int(tables[(node, s)][:, 2].sum())
                emit(out, {**base, "kind": "ce", "size": s,
                           "gbs": measure_bw(lambda: mod.mech_ce_each(tables[(node, s)], dev_tables[(node, s)]), nbytes)})
            if full:
                for s in CE_SIZES:
                    nbytes = int(tables[(node, s)][:, 2].sum())
                    for grid, unroll, label in SM_SHAPES:
                        emit(out, {**base, "kind": "sm", "size": s, "in_flight": label,
                                   "gbs": measure_bw(lambda: mod.mech_copy(dev_tables[(node, s)], KIND_CV, grid, unroll, 16),
                                                     nbytes)})
                nbytes = int(small[node][:, 2].sum())
                emit(out, {**base, "kind": "cw_real",
                           "gbs": measure_bw(lambda: mod.mech_copy(small_dev[node], KIND_CW_REAL, 1, 0, 0), nbytes)})
            serial, cold = latency(mod, slabs[node], dev_word)
            emit(out, {**base, "kind": "serial", "ns": serial})
            emit(out, {**base, "kind": "cold", "p50": statistics.median(cold), "p99": pct(cold, 0.99), "samples": cold})
            for core in PING_CORES:
                os.sched_setaffinity(0, {core})
                r = pingpong(mod, slabs[node])
                emit(out, {**base, "kind": "pingpong", "host_core": core, "p50": statistics.median(r),
                           "p99": pct(r, 0.99), "min": min(r), "samples": r})
            os.sched_setaffinity(0, {PING_CORES[0]})

    for rep in range(a.reps):
        one_rep(rep, "idle", True)
    if a.contend:
        reader = subprocess.Popen(["taskset", "-c", a.reader_cores, sys.executable, __file__, "--nvme-reader",
                                   "--nvme-dirs", a.nvme_dirs], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        started = json.loads(reader.stdout.readline())
        emit(out, {"kind": "reader", **started})
        time.sleep(3)
        try:
            for rep in range(a.reps):
                one_rep(rep, "contended", False)
        finally:
            reader.stdin.close()
            stopped = json.loads(reader.stdout.readline())
            reader.wait()
        emit(out, {"kind": "reader", **stopped})
    out.close()
    return 0


# ---- report ----

def report(records) -> str:
    def spread(v):
        return f"{statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]"

    def row(label, v0, v1, higher_better):
        m0, m1 = statistics.median(v0), statistics.median(v1)
        diff = (m1 - m0) / m0 * 100
        worse = -diff if higher_better else diff
        return f"| {label} | {spread(v0)} | {spread(v1)} | {diff:+.2f}% | {'same' if abs(worse) < 0.005 else ('node 1 worse' if worse > 0 else 'node 1 better')} {abs(worse):.2f}% |"

    lines = []
    for phase in ("idle", "contended"):
        recs = [r for r in records if r.get("phase") == phase]
        if not recs:
            continue
        lines += [f"### {phase}", "", "| metric | node 0 median [min-max] | node 1 median [min-max] | node 1 vs 0 | verdict |",
                  "|---|---|---|---|---|"]
        keys = []
        for r in recs:
            k = (r["kind"], r.get("size"), r.get("in_flight"), r.get("host_core"))
            if k not in keys:
                keys.append(k)
        for kind, size, in_flight, core in keys:
            sel = [r for r in recs if (r["kind"], r.get("size"), r.get("in_flight"), r.get("host_core")) == (kind, size, in_flight, core)]
            by = {n: [r for r in sel if r["node"] == n] for n in (0, 1)}
            if kind in ("ce", "sm", "cw_real"):
                label = {"ce": f"CE GB/s, {size:,} B", "sm": f"SM GB/s, {size:,} B, {in_flight} in flight",
                         "cw_real": "copy-wait read GB/s, 4 small tensors x 4 lanes"}[kind]
                lines.append(row(label, [r["gbs"] for r in by[0]], [r["gbs"] for r in by[1]], True))
            elif kind == "serial":
                lines.append(row("serial acquire, same word, ns", [r["ns"] for r in by[0]], [r["ns"] for r in by[1]], False))
            elif kind == "cold":
                for q in ("p50", "p99"):
                    lines.append(row(f"cold single read {q}, ns", [r[q] for r in by[0]], [r[q] for r in by[1]], False))
            elif kind == "pingpong":
                for q in ("p50", "p99", "min"):
                    lines.append(row(f"ping-pong RTT {q}, ns, host thread on core {core}",
                                     [r[q] for r in by[0]], [r[q] for r in by[1]], False))
        lines.append("")
    for r in records:
        if r.get("kind") in ("placement", "reader"):
            lines.append("- " + json.dumps({k: v for k, v in r.items() if k != "kind"}) + f" ({r['kind']})")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
