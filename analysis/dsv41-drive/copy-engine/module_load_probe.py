#!/usr/bin/env python3
"""Does a CUDA module load on the main thread stall the copy engine while a copy wait spins?

The real service and chain (test_exl3_piece_stream_cuda.StreamService with the copy engine), a 256 MiB ballast so
every copy job completes ~20 ms after its grant, and a 3 s request deadline. For each kind of load, the kernel is
built beforehand but never launched; the chain is launched without waiting, and while CW spins the main thread
launches that kernel for the first time, on a side stream, which loads its module then:

  triton   a fresh @triton.jit kernel (unique constant), compiled by warmup(): the launch runs cuModuleLoadData
  jit      a fresh tvm-ffi JIT CUDA module (unique source), built and dlopened: the launch lazily loads the module
  control  no load

In the "-first" variants the service is paused while the chain is launched, the load starts at once, and a helper
thread resumes the service 5 ms later: the copies are issued while the load is under way, not before it.

It reports per trial whether the request timed out, the chain's wall time and how long the load call took. A
deadlock shows as a timeout (keep 0) with the chain taking the full deadline.

    PYTHONPATH=<wt>/python python module_load_probe.py --repo <wt> --out probe.json [--trials 3]
"""

import argparse
import json
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=3)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    sys.path.insert(0, str(repo / "test" / "manual" / "dsv41"))
    from test_exl3_piece_stream_cuda import StreamService

    tmp = Path(tempfile.mkdtemp(prefix="ce-module-probe-", dir="/mnt/nvme1/pytest-tmp"))
    s = StreamService(tmp, copy_engine=True, timeout_ms=3000)
    src = torch.empty(256 << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    results = []
    try:
        s.plan([0, 1, 2])
        s.step()
        s.host.copy_engine_ballast(dst, src)

        def launch():
            s.post()
            s.hit_wait()
            s.copy1()
            s.ack1()
            s.stream()
            s.ack2()
            s.copy_wait()
            s.finalize()
            s.total()

        def prepare_triton():
            salt = uuid.uuid4().int % 1000003
            source = (
                "import triton\nimport triton.language as tl\n"
                "@triton.jit\n"
                f"def probe_kernel_{salt}(x_ptr, n, SALT: tl.constexpr):\n"
                "    i = tl.program_id(0) * 128 + tl.arange(0, 128)\n"
                "    m = i < n\n"
                "    tl.store(x_ptr + i, tl.load(x_ptr + i, mask=m) + SALT, mask=m)\n"
            )
            path = tmp / f"probe_{salt}.py"
            path.write_text(source)
            namespace = {}
            exec(compile(source, str(path), "exec"), namespace)
            kernel = namespace[f"probe_kernel_{salt}"]
            x = torch.zeros(256, dtype=torch.float32, device="cuda")
            kernel.warmup(x, 256, SALT=salt, grid=(2,))

            def run():
                with torch.cuda.stream(side):
                    kernel[(2,)](x, 256, SALT=salt)

            return run

        def prepare_jit():
            from sglang.kernels.jit.utils import load_jit as build

            salt = uuid.uuid4().hex[:12]
            path = tmp / f"probe_{salt}.cuh"
            path.write_text(
                "#include <sgl_kernel/tensor.h>\n#include <sgl_kernel/utils.h>\n#include <sgl_kernel/utils.cuh>\n"
                "#include <tvm/ffi/container/tensor.h>\n"
                f"__global__ void probe_{salt}_kernel(float* x) {{ x[threadIdx.x] += 1.0f; }}\n"
                "namespace sglang {\n"
                f"void probe_{salt}(tvm::ffi::TensorView x) {{\n"
                "  const auto stream = host::LaunchKernel::resolve_device(x.device());\n"
                f"  host::LaunchKernel(1, 32, stream)(probe_{salt}_kernel, static_cast<float*>(x.data_ptr()));\n"
                "}\n}\n"
            )
            module = build(f"probe_{salt}", cuda_files=[str(path)], cuda_wrappers=[(f"probe_{salt}", f"probe_{salt}")])
            x = torch.zeros(32, dtype=torch.float32, device="cuda")

            def run():
                with torch.cuda.stream(side):
                    getattr(module, f"probe_{salt}")(x)

            return run

        loads = {
            "control": (None, False),
            "triton": (prepare_triton, False),
            "jit": (prepare_jit, False),
            "control-first": (None, True),
            "triton-first": (prepare_triton, True),
            "jit-first": (prepare_jit, True),
        }
        for trial in range(a.trials):
            for kind, (prepare, first) in loads.items():
                load = prepare() if prepare is not None else None
                torch.cuda.synchronize()
                assert s.until(lambda: s.host.copy_engine_idle(0.0))
                timeouts = s.stats()["timeouts"]
                s.plan([0, 1, 2])
                if first:
                    s.host.pause(5.0)
                t0 = time.perf_counter()
                launch()
                if first:
                    resume = threading.Timer(0.005, s.host.resume)
                    resume.start()
                else:
                    time.sleep(0.002)  # CW is spinning: the ballast holds the copy ~20 ms
                t1 = time.perf_counter()
                if load is not None:
                    load()
                t2 = time.perf_counter()
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                if first:
                    resume.join()
                row = {
                    "trial": trial,
                    "kind": kind,
                    "keep": float(s.keep.item()),
                    "timeout": s.stats()["timeouts"] - timeouts,
                    "load_ms": round((t2 - t1) * 1e3, 2),
                    "chain_ms": round((t3 - t0) * 1e3, 2),
                    "fatal": s.host.fatal_seq(),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                if row["fatal"]:
                    raise SystemExit("the page is fatal: stopping")
    finally:
        s.host.copy_engine_ballast(None, None)
        s.close()
        Path(a.out).write_text(json.dumps({"results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
