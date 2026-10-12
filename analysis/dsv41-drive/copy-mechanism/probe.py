#!/usr/bin/env python3
"""Which copy mechanisms exist on this host: builds and runs each probe of probe.cuh in a process of its own.

    PYTHONPATH=<repo>/python python probe.py --repo <repo> --out probe.json

A probe is `ok` when it built, ran, and copied the right bytes. `load_opcodes` are the SASS load opcodes of its
module, so a 32-byte PTX load that ptxas split into two 16-byte ones shows as such.
"""
import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PROBES = {"v8_weak": "PROBE_V8_WEAK", "v8_cv": "PROBE_V8_CV", "ldgsts": "PROBE_LDGSTS", "bulk": "PROBE_BULK",
          "batch": "PROBE_BATCH"}
BYTES = {"v8_weak": 8192, "v8_cv": 8192, "ldgsts": 4096, "bulk": 4096, "batch": 4096}
LOADS = ("LDG", "LDGSTS", "UBLKCP", "UTMA")
TIMEOUT_S = 900  # build + run; a probe that hangs on the GPU is killed and recorded, never left holding the lock


def child(repo: pathlib.Path, name: str) -> dict:
    import torch

    import sglang

    if not str(pathlib.Path(sglang.__file__).resolve()).startswith(str(repo)):
        raise SystemExit(f"INTERPRETER TRAP: sglang from {sglang.__file__}, not {repo}")
    from sglang.kernels.jit.utils.compile.loader import load_jit

    try:
        mod = load_jit(
            "copy_mech_probe", name,
            cuda_files=[str(HERE / "probe.cuh")],
            cuda_wrappers=[("probe_run", "probe_run")],
            extra_cuda_cflags=[f"-D{PROBES[name]}"],
        )
    except Exception as exc:  # a rejected instruction is a result, not a crash
        return {"build": False, "ok": False, "load_opcodes": [], "error": str(exc)[-600:]}
    n = BYTES[name]
    src = (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8).pin_memory()
    dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
    # A side stream: cudaMemcpyBatchAsync rejects the legacy NULL stream (torch's default) with invalid argument.
    with torch.cuda.stream(torch.cuda.Stream()):
        mod.probe_run(src, dst)
    torch.cuda.synchronize()
    return {"build": True, "ok": bool(torch.equal(dst.cpu(), src)), "load_opcodes": opcodes(), "error": None}


def opcodes() -> list[str]:
    cuobjdump = shutil.which("cuobjdump") or os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin/cuobjdump")
    cache = os.environ["SGLANG_JIT_CACHE_DIR"]
    found = set()
    for so in pathlib.Path(cache).rglob("*.so"):
        sass = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True).stdout
        for line in sass.splitlines():
            parts = line.split("*/")
            if len(parts) < 2:
                continue
            op = parts[1].strip().lstrip("@!P0123456789 ").split(" ")[0]
            if op.startswith(LOADS):
                found.add(op)
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out")
    ap.add_argument("--child")
    a = ap.parse_args()
    repo = pathlib.Path(a.repo).resolve()
    if a.child:
        print(json.dumps(child(repo, a.child)), flush=True)
        return 0
    import torch

    results = {}
    for name in PROBES:
        cache = tempfile.mkdtemp(prefix=f"probe-{name}-", dir=HERE)
        env = dict(os.environ, SGLANG_JIT_CACHE_DIR=cache)
        try:
            run = subprocess.run([sys.executable, __file__, "--repo", str(repo), "--child", name], env=env,
                                 capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:  # subprocess.run has already killed the child
            results[name] = {"build": None, "ok": False, "load_opcodes": [],
                             "error": f"timeout after {TIMEOUT_S} s (killed): {str(exc.stderr or '')[-400:]}"}
        else:
            lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
            if run.returncode == 0 and lines:
                results[name] = json.loads(lines[-1])
            else:
                results[name] = {"build": None, "ok": False, "load_opcodes": [],
                                 "error": f"exit {run.returncode}: {run.stderr[-600:]}"}
        shutil.rmtree(cache, ignore_errors=True)
        print(name, results[name], flush=True)
    out = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
           "probes": results}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
