#!/usr/bin/env python3
"""GB/s against bytes in flight for every way to move pinned host bytes into VRAM, on this host's link.

One command per host. The meta record carries the host's PCIe generation and width, so runs on different links
compare directly:

    PYTHONPATH=<repo>/python python mech_bench.py --repo <repo> --probe probe.json --out <host>.jsonl
    python3 mech_report.py <host>.jsonl

Methods (probe.json decides which exist here):
  sm_cv16 / sm_cv32   ld.global.cv, 16- or 32-byte accesses, U in flight per thread, G blocks of 256
  sm_weak16 / sm_weak32  the same with plain (weak) ld.global, valid after an acquire (not .nc: non-coherent)
                      (s_pattern = S: G8 U1 W16; cw_pattern = the copy wait: G1 U4 W16)
  ldgsts, tma         Task 4;  ce_each, ce_batch and the *_small workloads: Task 5
Every method first passes a byte check; the fresh check (kind "fresh" records) runs with the .nc control.
"""
import argparse
import collections
import json
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SEGMENTS = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)
ALL_SEGMENTS = tuple(range(len(SEGMENTS)))
ROWS = 160  # 2.1 GB of pinned source rows: no launch reuses a row within 40 launches
SLOTS = 16  # 213 MB of destination slots, rotated: past the 96 MiB L2
N_ROWS = 4
FRESH_BYTES = 64 << 10  # small enough that one block's share fits its L1, so a stale .nc read shows
KIND_CV, KIND_NC, KIND_LDGSTS, KIND_TMA, KIND_WEAK, KIND_CW_REAL = 0, 1, 2, 3, 4, 5
KIND_TMA_NOFENCE, KIND_LINE = 6, 7
LOAD_NC = 1  # sm_kernel's load template argument for the .nc control (mech_bench.cuh kLoadNc)
WRAPPERS = ["mech_copy", "mech_fresh", "mech_latency", "mech_pingpong"]
BUILD_FLAGS = [("v8_weak", "MECH_V8_WEAK")]  # (probe name, define) pairs; Task 4 and 5 append theirs
# touch: bytes of every 128 B line a cell copies (the partial-line probe); 128 = all of it.
Cell = collections.namedtuple("Cell", "method grid a b in_flight name rows which run touch", defaults=(128,))


def probe_ok(probe: dict, name: str) -> bool:
    return bool(probe["probes"].get(name, {}).get("ok"))


def load(repo: Path, probe: dict):
    from sglang.kernels.jit.utils.compile.loader import load_jit

    flags = ["-DMECH_V8"] if probe_ok(probe, "v8_cv") else []
    flags += [f"-D{define}" for name, define in BUILD_FLAGS if probe_ok(probe, name)]
    variant = "-".join(sorted(f[2:] for f in flags)) or "base"
    return load_jit("copy_mech_bench", variant, cuda_files=[str(HERE / "mech_bench.cuh")],
                    cuda_wrappers=[(n, n) for n in WRAPPERS + (["mech_ce_batch"] if probe_ok(probe, "batch") else [])],
                    extra_cuda_cflags=flags, extra_include_paths=[str(repo / "python/sglang/kernels/jit/csrc")])


def loaded_module_path(tag: str) -> str:
    """The .so of the JIT module this process loaded, from its own memory map (exact, whatever the cache holds)."""
    paths = {line.split()[-1] for line in open("/proc/self/maps") if tag in line and line.rstrip().endswith(".so")}
    if len(paths) != 1:
        raise SystemExit(f"expected one loaded module matching {tag!r}, found {sorted(paths)}")
    return paths.pop()


def sass_by_function(so: str) -> dict[str, str]:
    import os
    import shutil

    cuobjdump = shutil.which("cuobjdump") or os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin/cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", so], capture_output=True, text=True, check=True).stdout
    functions, name = {}, None
    for line in text.splitlines():
        if "Function :" in line:
            name = line.split("Function :")[1].strip()
            functions[name] = ""
        elif name is not None:
            functions[name] += line + "\n"
    return functions


def cctl_check(so: str) -> dict:
    """The fresh check is only as good as this split: the .nc control's midpoint must not invalidate L1 (no CCTL),
    and every swept copy kernel's acquire must (CCTL.IVALL). Refuses to sweep when either side is wrong."""
    control, swept, bad = [], [], []
    for name, sass in sass_by_function(so).items():
        m = re.search(r"sm_kernelILi\d+ELi\d+EL[bi](\d+)E", name)
        if m:
            is_control = int(m.group(1)) == LOAD_NC
        elif "ldgsts_kernel" in name or "tma_kernel" in name:
            is_control = False
        else:
            continue  # latency / ping-pong: not copy kernels
        has = "CCTL" in sass
        (control if is_control else swept).append(name)
        if has == is_control:
            bad.append({"function": name, "control": is_control, "cctl": has})
    if not control or not swept or bad:
        raise SystemExit(f"CCTL split wrong (control {control}, swept {len(swept)}): {bad}")
    return {"kind": "sass", "module": so, "control_no_cctl": control, "swept_with_cctl": len(swept)}


class Rows:
    """Pinned source rows (the six real EXL3 segments) and rotated VRAM destination slots."""

    def __init__(self, node: int):
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

        self.src = [allocate_host_slab(ROWS, (b,), torch.uint8, register=True,
                                       placement=((node, ROWS * b),) if node >= 0 else ()) for b in SEGMENTS]
        for s in self.src:
            for r in range(ROWS):
                s[r, :512].fill_(r % 251)
        self.dst = [torch.zeros((SLOTS, b), dtype=torch.uint8, device="cuda") for b in SEGMENTS]
        self.row = 0
        self.slot = 0

    def jobs(self, n: int, which=ALL_SEGMENTS):
        pairs = []
        for _ in range(n):
            pairs.append((self.row, self.slot))
            self.row, self.slot = (self.row + 1) % ROWS, (self.slot + 1) % SLOTS
        table = [[self.src[k][row].data_ptr(), self.dst[k][slot].data_ptr(), SEGMENTS[k]]
                 for row, slot in pairs for k in which]
        return pairs, torch.tensor(table, dtype=torch.int64)

    def check(self, pairs, which, touch: int = 128) -> bool:
        """The copied bytes match: all of each row, or the first `touch` bytes of every 128 B line."""
        return all(torch.equal(self.dst[k][slot].cpu().view(-1, 128)[:, :touch], self.src[k][row].view(-1, 128)[:, :touch])
                   for row, slot in pairs for k in which)


def sm_cells(mod, probe):
    # sm_cv*: ld.global.cv. sm_weak*: plain (weak) ld.global after the fresh acquire; contract-valid by the PTX memory
    # model, unlike .nc, which is non-coherent and is only the fresh check's control.
    flavours = [("cv", KIND_CV, [16] + ([32] if probe_ok(probe, "v8_cv") else [])),
                ("weak", KIND_WEAK, [16] + ([32] if probe_ok(probe, "v8_weak") else []))]
    for flavour, kind, widths in flavours:
        for w in widths:
            for grid in (1, 2, 4, 8, 16, 32):
                for unroll in (1, 2, 4, 8, 16):
                    if w == 32 and unroll == 16:
                        continue
                    name = {(16, 8, 1): "s_pattern", (16, 1, 4): "cw_pattern"}.get((w, grid, unroll)) if kind == KIND_CV else None
                    yield Cell(f"sm_{flavour}{w}", grid, unroll, w, grid * 256 * unroll * w, name, N_ROWS, ALL_SEGMENTS,
                               lambda cpu, dev, k=kind, g=grid, u=unroll, w=w: mod.mech_copy(dev, k, g, u, w))


CELL_GENERATORS = [sm_cells]
# (method, kind, grid, a, b, probe that must be ok, or "")
FRESH_CHECKS = [("sm_cv16", KIND_CV, 8, 1, 16, ""), ("sm_cv32", KIND_CV, 8, 1, 32, "v8_cv"),
                ("sm_weak16", KIND_WEAK, 8, 1, 16, ""), ("sm_weak32", KIND_WEAK, 8, 1, 32, "v8_weak"),
                ("nc_control", KIND_NC, 8, 1, 16, "")]

BUILD_FLAGS += [("ldgsts", "MECH_LDGSTS"), ("bulk", "MECH_BULK")]


def ldgsts_cells(mod, probe):
    if not probe_ok(probe, "ldgsts"):
        return
    for grid in (1, 2, 4, 8, 16):
        for stages in (2, 4, 8):
            yield Cell("ldgsts", grid, stages, 0, grid * stages * 4096, None, N_ROWS, ALL_SEGMENTS,
                       lambda cpu, dev, g=grid, s=stages: mod.mech_copy(dev, KIND_LDGSTS, g, s, 0))


def tma_cells(mod, probe):
    if not probe_ok(probe, "bulk"):
        return
    for grid in (1, 2, 4, 8, 16):
        for chunk, stages in ((4096, 2), (4096, 4), (4096, 8), (16384, 2), (16384, 4)):
            yield Cell("tma", grid, stages, chunk, grid * stages * chunk, None, N_ROWS, ALL_SEGMENTS,
                       lambda cpu, dev, g=grid, s=stages, c=chunk: mod.mech_copy(dev, KIND_TMA, g, s, c))


CELL_GENERATORS += [ldgsts_cells, tma_cells]
FRESH_CHECKS += [("ldgsts", KIND_LDGSTS, 8, 4, 0, "ldgsts"), ("tma", KIND_TMA, 8, 4, 4096, "bulk")]
# Informational (mech_report INFORMATIONAL): can the fresh check see a missing proxy fence at all?
FRESH_CHECKS += [("tma_nofence", KIND_TMA_NOFENCE, 8, 4, 4096, "bulk")]

WRAPPERS += ["mech_ce_each"]
BUILD_FLAGS += [("batch", "MECH_BATCH")]
SMALL = (1, 2, 4, 5)  # indices of the four small segments (20 KiB, 9 KiB, 4.5 KiB, 10 KiB)


def ce_cells(mod, probe):
    batch = probe_ok(probe, "batch")
    for n in (1, 2, 4):
        yield Cell("ce_each", 0, n, 0, 0, None, n, ALL_SEGMENTS, lambda cpu, dev: mod.mech_ce_each(cpu, dev))
        if batch:
            yield Cell("ce_batch", 0, n, 0, 0, None, n, ALL_SEGMENTS, lambda cpu, dev: mod.mech_ce_batch(cpu, dev))
    for lanes in (1, 4, 8):
        yield Cell("ce_each_small", 0, lanes, 0, 0, None, lanes, SMALL, lambda cpu, dev: mod.mech_ce_each(cpu, dev))
        if batch:
            yield Cell("ce_batch_small", 0, lanes, 0, 0, None, lanes, SMALL,
                       lambda cpu, dev: mod.mech_ce_batch(cpu, dev))
        yield Cell("sm_small", 8, 1, 16, 8 * 256 * 16, None, lanes, SMALL,
                   lambda cpu, dev: mod.mech_copy(dev, KIND_CV, 8, 1, 16))


CELL_GENERATORS += [ce_cells]
# CW's small tensors in the order its sm_table reads them per lane: w13_suh 20,480, then 9,216, 10,240, 4,608 B.
CW_ORDER = (1, 2, 5, 4)


def cw_real_cells(mod, probe):
    # The copy wait's real shape (cw_pattern is contiguous 4-deep, which CW is not). in_flight is nominal: ~4 KiB
    # (one 16 B unit per thread) outside w13_suh's single 16 KiB 4-deep pass. Read us_per_row, i.e. per lane.
    for lanes in (1, 2, 4, 8):
        yield Cell("cw_real", 1, lanes, 0, 4096, None, lanes, CW_ORDER,
                   lambda cpu, dev: mod.mech_copy(dev, KIND_CW_REAL, 1, 0, 0))


CELL_GENERATORS += [cw_real_cells]


def line_cells(mod, probe):
    # Partial-line probe at two plateau shapes (in_flight = bytes loaded, 16 B per thread per unroll step).
    for grid, unroll in ((8, 8), (32, 4)):
        for touch in (16, 32, 64, 128):
            yield Cell(f"sm_line{touch}", grid, unroll, touch, grid * 256 * unroll * 16, None, N_ROWS, ALL_SEGMENTS,
                       lambda cpu, dev, g=grid, u=unroll, t=touch: mod.mech_copy(dev, KIND_LINE, g, u, t), touch)


CELL_GENERATORS += [line_cells]


def measure(cell: Cell, rows: Rows, reps: int) -> dict:
    times, host_ns = [], []
    for i in range(reps + 3):
        pairs, table = rows.jobs(cell.rows, cell.which)
        dev = table.to("cuda")
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        h = cell.run(table, dev)
        e1.record()
        e1.synchronize()  # the side stream's end event, before any timing or byte check reads it
        torch.cuda.synchronize()
        if i == 0 and not rows.check(pairs, cell.which, cell.touch):
            raise SystemExit(f"{cell.method} g{cell.grid} a{cell.a} b{cell.b} copied the wrong bytes")
        if i >= 3:
            times.append(e0.elapsed_time(e1))
            host_ns.append(h)
    ms = statistics.median(times)
    line_bytes = cell.rows * sum(SEGMENTS[k] for k in cell.which)
    nbytes = line_bytes * cell.touch // 128  # useful bytes: what was actually loaded
    extra = {} if cell.touch == 128 and not cell.method.startswith("sm_line") else {
        "touch": cell.touch, "line_bytes": line_bytes, "line_gbs": round(line_bytes / (ms * 1e-3) / 1e9, 3)}
    return {**extra, "kind": "cell", "method": cell.method, "name": cell.name, "grid": cell.grid, "a": cell.a, "b": cell.b,
            "in_flight": cell.in_flight, "rows": cell.rows, "bytes": nbytes, "ms_p50": round(ms, 4),
            "gbs": round(nbytes / (ms * 1e-3) / 1e9, 3), "us_per_row": round(ms * 1e3 / cell.rows, 2),
            "host_ns_p50": statistics.median(host_ns) if host_ns[0] is not None else None}


def fresh(mod, kind, grid, a, b) -> bool:
    src = torch.full((FRESH_BYTES,), 0x11, dtype=torch.uint8).pin_memory()
    pattern = torch.full((FRESH_BYTES,), 0x22, dtype=torch.uint8)
    dst = torch.zeros(FRESH_BYTES, dtype=torch.uint8, device="cuda")
    words = torch.zeros(2, dtype=torch.int32).pin_memory()
    jobs = torch.tensor([[src.data_ptr(), dst.data_ptr(), FRESH_BYTES]], dtype=torch.int64, device="cuda")
    mod.mech_fresh(jobs, words, src, pattern, kind, grid, a, b)
    torch.cuda.current_stream().synchronize()
    torch.cuda.synchronize()
    return bool((dst.cpu() == 0x22).all())


def pcie(index: int) -> tuple[int, int, int, int]:
    q = "pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-i", str(index)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return tuple(int(v) for v in out.split(", "))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--node", type=int, default=0, help="NUMA node of the source rows; -1 for no binding")
    ap.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi index of the GPU under test")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--only", default="", help="comma-separated method prefixes: run just those cells and fresh "
                    "checks (nc_control always runs); for follow-up probes, into a JSONL of their own")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    probe = json.loads(Path(a.probe).read_text())
    # Everything on a side stream: cudaMemcpyBatchAsync rejects the legacy NULL stream (torch's default).
    torch.cuda.set_stream(torch.cuda.Stream())
    mod = load(repo, probe)
    sass = cctl_check(loaded_module_path("copy_mech_bench_"))
    rows = Rows(a.node)
    gen, width, gen_now, width_now = pcie(a.gpu_index)
    out = open(a.out, "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")

    emit({"meta": True, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "pcie_gen": gen,
          "pcie_width": width, "pcie_gen_at_start": gen_now, "pcie_width_at_start": width_now,
          "cuda": torch.version.cuda, "node": a.node, "sglang": sglang.__file__,
          "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "probe": {k: v["ok"] for k, v in probe["probes"].items()}})
    emit(sass)
    host_word = torch.zeros(1, dtype=torch.int32).pin_memory()
    dev_word = torch.zeros(1, dtype=torch.int32, device="cuda")
    lat = torch.zeros(3, dtype=torch.int64, device="cuda")
    n = 1000
    samples = []
    for _ in range(5):
        mod.mech_latency(host_word, dev_word, lat, n)
        samples.append(lat.cpu().tolist())
    emit({"kind": "latency", "serial_acquire_ns": statistics.median(s[0] for s in samples) / n,
          "device_acquire_ns": statistics.median(s[1] for s in samples) / n})
    words = torch.zeros(2, dtype=torch.int32).pin_memory()
    rtt = torch.zeros(256, dtype=torch.int64, device="cuda")
    mod.mech_pingpong(words, rtt, 256)
    r = rtt.cpu().tolist()[16:]  # the first rounds include the kernel's start
    emit({"kind": "pingpong", "rtt_ns_p50": int(statistics.median(r)), "rtt_ns_min": int(min(r)), "rounds": len(r)})
    only = tuple(x for x in a.only.split(",") if x)
    for method, kind, grid, ua, ub, need in FRESH_CHECKS:
        if need and not probe_ok(probe, need):
            continue
        if only and method != "nc_control" and not method.startswith(only):
            continue
        emit({"kind": "fresh", "method": method, "fresh": fresh(mod, kind, grid, ua, ub)})
    for generator in CELL_GENERATORS:
        for cell in generator(mod, probe):
            if only and not cell.method.startswith(only):
                continue
            emit(measure(cell, rows, a.reps))
    out.close()
    sys.path.insert(0, str(HERE))
    import mech_report

    records = [json.loads(line) for line in open(a.out) if line.strip()]
    # One file per run: a second meta in the file (an appended rerun) is refused by summarize.
    s = mech_report.summarize(records)
    print(json.dumps({k: s[k] for k in ("measured_ceiling_gbs", "theoretical_gbs", "bdp_bytes", "unsafe",
                                        "control_blind")}))
    return 1 if s["above_ceiling"] or s["unsafe"] or s["control_blind"] else 0


if __name__ == "__main__":
    sys.exit(main())
