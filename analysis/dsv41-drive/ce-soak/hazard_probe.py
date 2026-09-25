#!/usr/bin/env python3
"""Which driver calls stall the copy engine while a copy wait spins? (LEASE_PROTOCOL.md 7.6, "Not guarded")

The same harness as copy-engine/module_load_probe.py: the real service and chain (test_exl3_piece_stream_cuda
.StreamService with the copy engine), a 256 MiB ballast so every copy job completes ~20 ms after its grant, and a 3 s
request deadline. The chain is launched without waiting; while CW spins, the main thread makes one call:

  control      nothing
  hostalloc    torch.empty(pin_memory=True) of a size the caching host allocator has never held: cudaHostAlloc
  hostregister cudaHostRegister of a fresh malloc'd buffer
  hostfree     cudaFreeHost of a pinned block (torch's host cache emptied)
  malloc       torch.empty on the device of a size the caching allocator cannot serve: cudaMalloc
  emptycache   torch.cuda.empty_cache() with a freed block cached: cudaFree
  h2d_after    a pinned->device copy enqueued on the chain's stream behind it (non_blocking), before the copy thread
               issues its own: a copy-engine copy that waits on the chain, as a decode step's input update does
  d2h_after    the same with a device->pinned copy (the overlap scheduler's result copies)
  d2d_after    the same with a device->device copy
  h2d_side     a pinned->device copy on a side stream that waits on the chain's stream

For every *_after/_side kind the service is paused while the chain and the copy are enqueued and resumed 5 ms
later, so the copy thread's copies are issued after the waiting copy (the order of a real decode step).

A stall shows as a timeout (keep 0, the chain taking the full deadline); a call that merely waits for the device shows
as a call time about the chain's.

    PYTHONPATH=<wt>/python python hazard_probe.py --repo <wt> --out probe.json [--trials 2] [--kinds a,b]
"""

import argparse
import ctypes
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--kinds", default="control,hostalloc,hostregister,hostfree,malloc,emptycache,h2d_after,d2h_after,d2d_after,h2d_side")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    import sglang

    if not str(Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    sys.path.insert(0, str(repo / "test" / "manual" / "dsv41"))
    from test_exl3_piece_stream_cuda import StreamService

    tmp = Path(tempfile.mkdtemp(prefix="ce-hazard-probe-", dir="/mnt/nvme1/pytest-tmp"))
    s = StreamService(tmp, copy_engine=True, timeout_ms=3000)
    src = torch.empty(256 << 20, dtype=torch.uint8).pin_memory()
    dst = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    cudart = torch.cuda.cudart()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc.restype = ctypes.c_void_p
    keep = []  # every probe allocation stays alive: only the first call of each kind is a miss
    results = []
    sizes = iter(range(1, 1000))

    def fresh_mib():
        # Distinct power-of-two classes, above anything the harness allocated, so each is an allocator miss.
        return 1 << (6 + next(sizes) % 5)  # 128 MiB .. 1 GiB, cycling; the caches keep earlier ones busy

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

        def prepare(kind):
            if kind == "control":
                return None
            if kind == "hostalloc":
                n = (fresh_mib() << 20) + 4096 * (len(keep) + 1)
                return lambda: keep.append(torch.empty(n, dtype=torch.uint8, pin_memory=True))
            if kind == "hostregister":
                n = 64 << 20
                ptr = libc.malloc(n)
                ctypes.memset(ptr, 0, n)

                def reg():
                    rc = int(cudart.cudaHostRegister(ptr, n, 0))
                    keep.append(ptr)
                    if rc:
                        raise RuntimeError(f"cudaHostRegister rc={rc}")

                return reg
            if kind == "hostfree":
                block = torch.empty((32 << 20) + 8192 * (len(keep) + 1), dtype=torch.uint8, pin_memory=True)
                del block
                return lambda: torch._C._host_emptyCache()
            if kind == "malloc":
                n = (fresh_mib() << 20) + 2 * 1024 * 1024 * (len(keep) + 1)
                return lambda: keep.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
            if kind in ("h2d_after", "d2h_after", "d2d_after", "h2d_side"):
                pinned = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
                device = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
                device2 = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
                keep.extend([pinned, device, device2])
                if kind == "h2d_after":
                    return lambda: device.copy_(pinned, non_blocking=True)
                if kind == "d2h_after":
                    return lambda: pinned.copy_(device, non_blocking=True)
                if kind == "d2d_after":
                    return lambda: device2.copy_(device, non_blocking=True)

                def side_copy():
                    side = torch.cuda.Stream()
                    keep.append(side)
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        device.copy_(pinned, non_blocking=True)

                return side_copy
            if kind == "emptycache":
                block = torch.empty(96 << 20, dtype=torch.uint8, device="cuda")
                del block
                return torch.cuda.empty_cache
            raise ValueError(kind)

        for trial in range(a.trials):
            for kind in a.kinds.split(","):
                call = prepare(kind)
                torch.cuda.synchronize()
                assert s.until(lambda: s.host.copy_engine_idle(0.0))
                timeouts = s.stats()["timeouts"]
                s.plan([0, 1, 2])
                behind = kind.endswith("_after") or kind.endswith("_side")
                if behind:
                    s.host.pause(5.0)
                t0 = time.perf_counter()
                launch()
                if behind:
                    resume = threading.Timer(0.005, s.host.resume)
                    resume.start()
                else:
                    time.sleep(0.002)  # CW is spinning: the ballast holds the copy ~20 ms
                t1 = time.perf_counter()
                if call is not None:
                    call()
                t2 = time.perf_counter()
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                if behind:
                    resume.join()
                row = {
                    "trial": trial,
                    "kind": kind,
                    "keep": float(s.keep.item()),
                    "timeout": s.stats()["timeouts"] - timeouts,
                    "call_ms": round((t2 - t1) * 1e3, 2),
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
