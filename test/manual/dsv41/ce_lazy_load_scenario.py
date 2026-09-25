"""One scenario of test_exl3_copy_engine_cuda.py's module-loading test, in its own process: CUDA_MODULE_LOADING is read
once, when CUDA initialises, so each mode needs a fresh process.

The real service and chain (StreamService, copy engine armed, a ballast so each copy job takes ~20 ms, a 3 s deadline).
A fresh JIT CUDA kernel is built and loaded (tvm-ffi) but never launched. The service is paused, the chain launched,
the kernel launched for the first time on a side stream, and the service resumed 5 ms later, so the copy thread's
copies are issued while that first launch is under way. Prints one JSON line: keep, timeouts, the launch's duration.

    python ce_lazy_load_scenario.py <tmp dir>
"""

import json
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exl3_piece_stream_cuda import StreamService  # noqa: E402


def main() -> int:
    tmp = Path(sys.argv[1])
    tmp.mkdir(parents=True, exist_ok=True)
    from sglang.kernels.jit.utils import load_jit

    salt = uuid.uuid4().hex[:12]
    source = tmp / f"lazy_{salt}.cuh"
    source.write_text(
        "#include <sgl_kernel/tensor.h>\n#include <sgl_kernel/utils.h>\n#include <sgl_kernel/utils.cuh>\n"
        "#include <tvm/ffi/container/tensor.h>\n"
        f"__global__ void lazy_{salt}_kernel(float* x) {{ x[threadIdx.x] += 1.0f; }}\n"
        "namespace sglang {\n"
        f"void lazy_{salt}(tvm::ffi::TensorView x) {{\n"
        "  const auto stream = host::LaunchKernel::resolve_device(x.device());\n"
        f"  host::LaunchKernel(1, 32, stream)(lazy_{salt}_kernel, static_cast<float*>(x.data_ptr()));\n"
        "}\n}\n"
    )
    jit = load_jit(f"lazy_{salt}", cuda_files=[str(source)], cuda_wrappers=[(f"lazy_{salt}", f"lazy_{salt}")])
    kernel = getattr(jit, f"lazy_{salt}")
    x = torch.zeros(32, dtype=torch.float32, device="cuda")
    side = torch.cuda.Stream()

    s = StreamService(tmp / "svc", copy_engine=True, timeout_ms=3000)
    src = torch.empty(256 << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    try:
        s.plan([0, 1, 2])
        s.step()
        torch.cuda.synchronize()
        assert s.until(lambda: s.host.copy_engine_idle(0.0))
        s.host.copy_engine_ballast(dst, src)
        s.plan([0, 1, 2])
        timeouts = s.stats()["timeouts"]
        s.host.pause(5.0)
        s.post()
        s.hit_wait()
        s.copy1()
        s.ack1()
        s.stream()
        s.ack2()
        s.copy_wait()
        s.finalize()
        s.total()
        resume = threading.Timer(0.005, s.host.resume)
        resume.start()
        t0 = time.perf_counter()
        with torch.cuda.stream(side):
            kernel(x)  # the kernel's first launch
        launch_ms = (time.perf_counter() - t0) * 1e3
        torch.cuda.synchronize()
        resume.join()
        print(json.dumps({"keep": float(s.keep.item()), "timeouts": s.stats()["timeouts"] - timeouts,
                          "launch_ms": round(launch_ms, 2), "fatal": s.host.fatal_seq(),
                          "x": float(x[0].item())}), flush=True)
    finally:
        s.host.copy_engine_ballast(None, None)
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
