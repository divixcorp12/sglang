"""Bounded CPU-only EXL3 route-cost and idle-gap measurements; no production tuning.

Run ``matrix --reference <server-env-actual.json> --output <new-dir>`` on divix01
from a pushed private worktree. Each arm is a separate process so OpenMP settings
and diagnostic compile flavors are fixed before the runtime is initialized.
Synthetic packed weights measure scheduling/kernel behavior, not model accuracy
or production throughput. Results include actual route patterns and call edges.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

CAPACITY = 24
HIDDEN = 5120


def shapes():
    cases = []
    for rows in range(1, 7):
        cases.append((f"one-expert-m{rows}", [[0] for _ in range(rows)]))
        cases.append((f"ten-experts-m{rows}", [list(range(10)) for _ in range(rows)]))
    # Six token rows, 11 live routes, one expert used twice and nine used once.
    cases += [
        ("mixed-11-front", [[0, 0], [1, 2], [3, 4], [5, 6], [7, 8], [9, -1]]),
        ("mixed-11-middle", [[4, 4], [0, 1], [2, 3], [5, 6], [7, 8], [9, -1]]),
        ("mixed-11-end", [[9, 9], [0, 1], [2, 3], [4, 5], [6, 7], [8, -1]]),
        ("mixed-18", [[(t + 2 * j) % 12 for j in range(3)] for t in range(6)]),
    ]
    return cases


def quantile(xs, q):
    xs = sorted(xs)
    return xs[int((len(xs) - 1) * q)]


def arm(args):
    cores = list(range(6, 16)) if args.group == 0 else list(range(18, 28))
    # Allocate/fill tensors on this group's node, before the team pins worker 0.
    os.sched_setaffinity(0, {cores[0]})
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = "bw"
    import torch
    import sglang
    import exl3_cpu_forward_ab as fixture
    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.ext import exl3_ext

    torch.set_num_threads(1)
    fixture.CAP = CAPACITY
    print(f"sglang {sglang.__file__}", flush=True)
    if args.library:
        import importlib.util
        spec = importlib.util.spec_from_file_location(args.library.stem, args.library)
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
    else:
        ext = exl3_ext()
    if not ext.exl3_moe_cpu_has_avx512_bw():
        raise RuntimeError("expected BW optimized CPU tier")
    slabs = fixture.random_slabs(torch, HIDDEN, 2304, seed=719)
    handle, trait = fixture.register_slabs(ext, slabs, 10.0)
    # Keep both owning objects alive throughout all forwards.
    assert trait is not None
    weight_bytes = sum(t.numel() * t.element_size() for t in slabs.values())
    generator = torch.Generator().manual_seed(71)
    metadata = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sglang_file": sglang.__file__, "group": args.group, "cores": cores,
        "measurement": args.measurement, "reps": args.reps, "warmups": args.warmups,
        "weight_capacity": CAPACITY, "weight_bytes": weight_bytes,
        "policy": {k: os.environ.get(k) for k in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_NUM_THREADS")},
        "trace_prefix": os.environ.get("SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX"),
        "tid": __import__("threading").get_native_id(),
        "case_order_seed": 619,
        "library": ext.__file__,
        "library_sha256": hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
        "chunk_m": torch.ops.sglang_exl3_cpu.chunk_m(),
    }
    jobs = []
    outputs = {}
    cases = shapes() if args.measurement == "cost" else [shapes()[1], shapes()[12]]
    # Randomize case blocks reproducibly; every policy sees the same ordering.
    import random
    random.Random(619).shuffle(cases)
    if args.ready_file:
        args.ready_file.write_text(json.dumps({"pid": os.getpid(), "metadata": metadata}) + "\n")
    if args.start_file:
        deadline = time.monotonic() + 30
        while not args.start_file.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("start-file gate timed out after 30s")
            time.sleep(.01)
    try:
        for name, route in cases:
            rows, k = len(route), len(route[0])
            x = torch.randn(rows, HIDDEN, generator=generator).half()
            base_sel = torch.tensor(route, dtype=torch.int32)
            weights = torch.full((rows, k), 1.0 / k, dtype=torch.float32)
            out = torch.empty((rows, HIDDEN), dtype=torch.float32)
            settings = [(reuse, 0.0) for reuse in ("repeated", "rotating")] if args.measurement == "cost" else [
                ("repeated", gap) for gap in (0.0, 0.0002, 0.002)
            ]
            for reuse, gap in settings:
                previous_end = None
                for rep in range(-args.warmups, args.reps):
                    # Cyclic slot remapping changes weights, preserves expert order within chunks.
                    offset = (max(rep, 0) * 11) % CAPACITY if reuse == "rotating" else 0
                    sel = torch.where(base_sel < 0, base_sel, (base_sel + offset) % CAPACITY).contiguous()
                    if gap:
                        time.sleep(gap)
                    start = time.monotonic_ns()
                    status, why = es.kernel_forward(handle, x, sel, weights, out, threads=10, cores=cores, variant="instr")
                    end = time.monotonic_ns()
                    if status:
                        raise RuntimeError(f"{name}: {status}: {why}")
                    if rep >= 0:
                        jobs.append({"case": name, "rep": rep, "reuse": reuse, "gap_s": gap,
                                     "actual_gap_ns": start - previous_end if previous_end else None,
                                     "start_ns": start, "end_ns": end, "wall_ms": (end - start) / 1e6,
                                     "rows": rows, "routes": sel.tolist()})
                    previous_end = end
                if not torch.isfinite(out).all():
                    raise RuntimeError(f"nonfinite output: {name}")
                # Cross-arm equality check uses a repeated-weight, identically seeded final input.
                if reuse == "repeated":
                    outputs[f"{name}/gap{gap}"] = hashlib.sha256(out.numpy().tobytes()).hexdigest()
    finally:
        es.kernel_drop(handle, variant="instr")
    bycase = collections.defaultdict(list)
    for j in jobs:
        bycase[(j['case'], j['reuse'], j['gap_s'])].append(j['wall_ms'])
    summary = [{"case": key[0], "reuse": key[1], "gap_s": key[2], "n": len(values),
                "p50_ms": statistics.median(values), "p95_ms": quantile(values, .95),
                "max_ms": max(values)} for key, values in sorted(bycase.items())]
    metadata["plan_calls"] = list(torch.ops.sglang_exl3_cpu.plan_calls())
    args.output.write_text(json.dumps({"metadata": metadata, "summary": summary, "output_hashes": outputs,
                                      "jobs": jobs}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "jobs": len(jobs), "summary": summary}), flush=True)


def matrix(args):
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text())
    # Import/build environment from the previous diagnostic arm, stripping its tracing settings.
    env = {**os.environ, **reference}
    for key in tuple(env):
        if key.startswith(("SGLANG_DSV41_EXPERT_JOB_TRACE", "SGLANG_EXL3_CPU_WORKER_TRACE")) or key == "SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE":
            env.pop(key)
    root = Path(__file__).resolve().parents[3]
    env.update(PYTHONPATH=str(root / "python"), OMP_NUM_THREADS="10", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MAX_JOBS="4", EXL3_MOE_CPU_PIN="0",
               SGLANG_EXL3_BUILD_DIR=str(args.output / "exl3-build"))
    # Reuse CUDA build artifacts, never copy a source checkout. Changed CPU sources are rebuilt by ninja.
    if not args.prebuilt_base:
        import shutil
        source = Path(reference["SGLANG_EXL3_BUILD_DIR"])
        shutil.copytree(source, args.output / "exl3-build")
    commands = []
    matrix_start = time.monotonic_ns()
    for measurement in ("cost", "wake"):
        policies = ("default",) if measurement == "cost" else ("default", "passive", "bounded-spin")
        for policy in policies:
            for trace in (False, True):
                for group in (0, 1):
                    name = f"{measurement}-{policy}-{'trace' if trace else 'plain'}-g{group}"
                    arm_env = dict(env)
                    arm_env.pop("OMP_WAIT_POLICY", None)
                    arm_env.pop("GOMP_SPINCOUNT", None)
                    if policy == "passive":
                        arm_env.update(OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
                    if policy == "bounded-spin":
                        arm_env.update(OMP_WAIT_POLICY="ACTIVE", GOMP_SPINCOUNT="100000")
                    if trace:
                        arm_env.update(SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX=str(args.output / "unused-job-events"),
                                       SGLANG_EXL3_CPU_WORKER_TRACE_PREFIX=str(args.output / name),
                                       SGLANG_EXL3_CPU_WORKER_TRACE_MIN_US="0",
                                       SGLANG_EXL3_CPU_WORKER_TRACE_CAPACITY="131072")
                    command = [sys.executable, __file__, "arm", "--measurement", measurement,
                               "--group", str(group), "--reps", str(args.reps), "--warmups", "5",
                               "--output", str(args.output / f"{name}.json")]
                    library = args.prebuilt_trace if trace else args.prebuilt_base
                    if library:
                        command += ["--library", str(library)]
                    record = {"name": name, "command": command, "start_ns": time.monotonic_ns()}
                    print(f"START {name}", flush=True)
                    with (args.output / f"{name}.log").open("w") as log:
                        result = subprocess.run(command, env=arm_env, stdout=log, stderr=subprocess.STDOUT)
                    record.update(status=result.returncode, end_ns=time.monotonic_ns())
                    commands.append(record)
                    (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
                    if result.returncode:
                        raise RuntimeError(f"{name} failed: {result.returncode}; inspect its log")
                    print(f"DONE {name}", flush=True)
    (args.output / "exit-status.json").write_text(json.dumps({"status": 0, "start_ns": matrix_start,
                                                             "end_ns": time.monotonic_ns()}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--measurement", choices=("cost", "wake"), required=True)
    a.add_argument("--group", type=int, choices=(0, 1), required=True)
    a.add_argument("--reps", type=int, default=30)
    a.add_argument("--warmups", type=int, default=5)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--library", type=Path, help="load an explicitly identified prebuilt extension without JIT")
    a.add_argument("--ready-file", type=Path)
    a.add_argument("--start-file", type=Path)
    m = sub.add_parser("matrix")
    m.add_argument("--reference", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--reps", type=int, default=30)
    m.add_argument("--prebuilt-base", type=Path)
    m.add_argument("--prebuilt-trace", type=Path)
    args = p.parse_args()
    if not 5 <= args.reps <= 100:
        p.error("reps must be between 5 and 100")
    if args.command == "matrix" and bool(args.prebuilt_base) != bool(args.prebuilt_trace):
        p.error("provide both prebuilt variants or neither")
    matrix(args) if args.command == "matrix" else arm(args)


if __name__ == "__main__":
    main()
