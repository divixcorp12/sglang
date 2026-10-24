"""Bounded two-team CPU-engine replay, including keep-warm and OpenMP joins.

Synthetic DSV4.1 weights, target submissions only; group 0 attaches an idle draft
source to exercise the two-word hold. Does not measure GPU/PCIe or serving TPS.
Build in a pushed private worktree, then run each arm in a fresh process.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SHA = "e28fb2896a3d612b46a27d7f8ef840d34c9efbc29b70a044c42095fb94856890"
OFFSETS = dict(managed_threads=0x237040, available_cpus=0x237048,
               normal_spin_count=0x2373c8, throttled_spin_count=0x2373d8)


def runtime_probe():
    """Read-only private counters; fail closed unless this exact binary was audited."""
    path = Path(__import__("torch").__file__).parent / "lib/libgomp.so.1"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != RUNTIME_SHA:
        raise RuntimeError("runtime hash differs: re-audit counter offsets before measuring")
    library = ctypes.CDLL(str(path))

    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]

    info = DlInfo()
    dl = ctypes.CDLL(None)
    dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    if not dl.dladdr(ctypes.cast(library.GOMP_barrier, ctypes.c_void_p), ctypes.byref(info)):
        raise RuntimeError("cannot identify runtime mapping")
    addresses = [info.base + offset for offset in OFFSETS.values()]
    return dict(path=str(path), sha256=sha, provider=info.filename.decode(), offsets=OFFSETS,
                counters={name: ctypes.c_uint64.from_address(address).value
                          for name, address in zip(OFFSETS, addresses)}), addresses


def build(args):
    """Builds the engine library with the EXL3 kernel linked in: the kernel's sources and flags are the host module's
    (cpu_kernel_build), with the worker trace compiled in as the diagnostic build has it."""
    from sglang.kernels.jit.utils.compile.toolchain import host_arch_flags
    from sglang.kernels.ops.moe.cpu_kernel_build import cpu_kernel_build
    from sglang.srt.layers.moe.expert_engine.config import (
        CpuExpertsConfig, CpuWorkerTraceConfig, ExpertStreamConfig, JobTraceConfig, TraceConfig)
    args.output.mkdir(parents=True, exist_ok=True)
    config = ExpertStreamConfig(
        cpu_experts=CpuExpertsConfig(enabled=True),
        trace=TraceConfig(job=JobTraceConfig(prefix=str(args.output / "unused-job")),
                          cpu_worker=CpuWorkerTraceConfig(prefix=str(args.output / "unused-worker"))))
    kernel = cpu_kernel_build("exl3", config)
    csrc = ROOT / "python/sglang/kernels/jit/csrc"
    cxx = os.environ.get("CXX", "g++")
    objects, commands = [], []
    for source in kernel.cpp_files:
        flags = dict(kernel.source_cflags)[source]
        objects.append(str(args.output / (Path(source).stem + ".o")))
        commands.append([cxx, *flags, *host_arch_flags(), "-fPIC", "-c", str(csrc / source), "-o", objects[-1]])
    commands.append([cxx, "-std=c++20", "-O3", "-shared", "-fPIC", "-pthread", "-fopenmp",
                     "-I" + str(csrc), str(Path(__file__).with_suffix(".cpp")), *objects,
                     "-o", str(args.output / "engine-measure.so")])
    for command in commands:
        subprocess.run(command, check=True)
    result = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "engine_library": str(args.output / "engine-measure.so"),
              "compile_commands": commands,
              "engine_sha256": hashlib.sha256((args.output / "engine-measure.so").read_bytes()).hexdigest()}
    (args.output / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def arm(args):
    masks = {"one": {18}, "server": set(range(6)) | set(range(36, 42)), "broad": set(range(64))}
    os.sched_setaffinity(0, masks[args.init])
    # This must precede torch/tvm/sglang imports: libgomp caches this initial CPU budget.
    import torch
    import sglang
    import exl3_cpu_forward_ab as fixture
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait
    torch.set_num_threads(1)
    from sglang.kernels.ops.moe import expert_stream_transport as es
    provenance = json.loads(args.build.read_text())
    initial, addresses = runtime_probe()
    fixture.CAP = 12
    owners = []
    inputs, outputs, pointers, sizes = [], [], [], []
    # The plan list is the host module's (the same kernel sources and -march as the engine library's).
    host = es.cpu_kernel_options(variant="instr")
    for group, cpu in enumerate((6, 18)):
        os.sched_setaffinity(0, {cpu})
        slabs = fixture.random_slabs(torch, 5120, 2304, seed=719 + group)
        trait = Exl3CpuQuantTrait(act_limit=10.0)
        trait.attach_host(host)
        layer = trait.layer_spec(slabs, 12)
        kernel_params = (ctypes.c_int64 * len(layer.params))(*layer.params)
        pointers.extend(p for p, _ in layer.slabs)
        sizes.extend(s for _, s in layer.slabs)
        x = torch.randn((6, 5120), generator=torch.Generator().manual_seed(71 + group)).half()
        out = torch.empty((6, 5120), dtype=torch.float32)
        inputs.append(x.data_ptr()); outputs.append(out.data_ptr())
        owners.append((slabs, x, out))
    os.sched_setaffinity(0, set(range(64)))
    library = ctypes.CDLL(provenance["engine_library"])
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i64p = ctypes.POINTER(ctypes.c_int64)
    library.engine_measure.argtypes = [i64p, ctypes.c_int, u64p, u64p, u64p, u64p,
                                      *([ctypes.c_int] * 6), u64p, i64p, ctypes.c_char_p, ctypes.c_size_t]
    library.engine_measure.restype = ctypes.c_int
    as_array = lambda values: (ctypes.c_uint64 * len(values))(*values)
    jobs, cells = [], []
    combinations = [(1, 1, 200), (1, 8, 200), (6, 8, 2000), (1, 8, 0), (1, 1, 120000)]
    if args.smoke:
        combinations = [(1, 1, 200)]
    if args.ready_file:
        args.ready_file.write_text(json.dumps({"pid": os.getpid(), "runtime": initial}) + "\n")
    if args.start_file:
        deadline = time.monotonic() + 30
        while not args.start_file.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("start gate timed out")
            time.sleep(.01)
    for cell, (rows, routes, gap) in enumerate(combinations):
        reps = (3 if gap == 120000 else args.reps) + 5
        records = (ctypes.c_int64 * (2 * reps * 8))()
        error = ctypes.create_string_buffer(1024)
        start = time.monotonic_ns()
        status = library.engine_measure(kernel_params, len(kernel_params), as_array(pointers), as_array(sizes),
                                        as_array(inputs), as_array(outputs), rows, routes, reps, gap,
                                        2000, 1, as_array(addresses), records, error, len(error))
        end = time.monotonic_ns()
        if status:
            raise RuntimeError(error.value.decode())
        cells.append(dict(cell=cell, rows=rows, routes=routes, requested_gap_us=gap,
                          start_ns=start, end_ns=end, total_jobs=2 * reps))
        for group in range(2):
            for rep in range(reps):
                values = list(records[(group * reps + rep) * 8:(group * reps + rep + 1) * 8])
                jobs.append(dict(cell=cell, group=group, rep=rep, warmup=rep < 5,
                                 rows=rows, routes=routes, requested_gap_us=gap,
                                 submit_begin=values[0], submit_end=values[1], done_observed=values[2], seq=values[3],
                                 runtime=dict(zip(OFFSETS, values[4:]))))
        print(json.dumps({"cell": cell, "rows": rows, "routes": routes, "gap_us": gap, "status": 0}), flush=True)
    metadata = dict(head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    init=args.init, initial_affinity=sorted(masks[args.init]), sglang_file=sglang.__file__,
                    runtime=initial, build=provenance, cores=[list(range(6, 16)), list(range(18, 28))],
                    first_touch_cpus=[6, 18], producer_cpus=[0, 16], group0_idle_draft_source=True,
                    policy={k: os.environ.get(k) for k in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_NUM_THREADS")},
                    parity="bit-exact direct-forward comparison after every engine job")
    args.output.write_text(json.dumps(dict(metadata=metadata, cells=cells, jobs=jobs), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output", type=Path, required=True)
    # Accepted and ignored: the kernel is no longer an extension build to seed.
    b.add_argument("--seed-build", type=Path)
    a = sub.add_parser("arm")
    a.add_argument("--build", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--init", choices=("one", "server", "broad"), required=True)
    a.add_argument("--reps", type=int, default=40)
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--ready-file", type=Path)
    a.add_argument("--start-file", type=Path)
    args = parser.parse_args()
    if args.mode == "arm" and not 1 <= args.reps <= 100:
        parser.error("reps must be in [1, 100]")
    build(args) if args.mode == "build" else arm(args)


if __name__ == "__main__":
    main()
