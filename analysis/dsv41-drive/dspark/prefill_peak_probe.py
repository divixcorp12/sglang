"""Peak VRAM of an A/B arm's prefill: one server, a 4096-token prompt (one full chunk) then a long one, sampled by
nvidia-smi every 100 ms.

Starts an arm of both_cpu_ab.py the way spec_margin_capture.py does (its argv, server flags and env), with extra
``--env K=V`` overrides on top, and sends each prompt as fresh random token ids (no radix-cache reuse) through
/generate with one output token. Writes ``peaks.json``: the KV pool and init headroom from server.log, the used
memory when healthy, and per prompt the peak used memory and what that leaves of the card. The sampled figure is the
driver's (allocator reserve and CUDA context included), which is what an OOM is measured against.

Locks: rowimg-disk.lock, then cc-gpu.lock, through run_locks.take.

Usage: prefill_peak_probe.py OUT_DIR --arm ARM [--env K=V ...] [--prompt-tokens 4096,30000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402
import both_cpu_ab  # noqa: E402
import run_locks  # noqa: E402
from spec_margin_capture import DISK_LOCK, PORT, _healthy, stop_server  # noqa: E402

LOG_PATTERNS = {
    "kv_tokens": r"max_total_num_tokens=(\d+)",
    "init_avail_gb": r"max_total_num_tokens=.*available_gpu_mem=([\d.]+) GB",
    "pre_load_avail_gb": r"Load weight begin\. avail mem=([\d.]+) GB",
    "hot_cache_bytes": r"Expert hot cache startup .*\"allocation_bytes\": (\d+)",
}


class Sampler(threading.Thread):
    """nvidia-smi's used/total memory every 100 ms into a CSV, kept in memory as (monotonic time, used MiB)."""

    def __init__(self, path: str):
        super().__init__(daemon=True)
        self.path, self.samples, self.total_mib = path, [], None
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True,
        )

    def run(self):
        with open(self.path, "w") as f:
            for line in self.proc.stdout:
                used, total = (int(x) for x in line.split(","))
                self.total_mib = total
                self.samples.append((time.monotonic(), used))
                f.write(f"{time.time():.3f},{used},{total}\n")

    def peak(self, start: float, end: float) -> int:
        return max(used for t, used in self.samples if start <= t <= end)

    def stop(self):
        self.proc.terminate()


def _generate(tokens: int, seed: int) -> dict:
    rng = random.Random(seed)
    body = {"input_ids": [rng.randrange(1000, 100000) for _ in range(tokens)],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def _from_log(path: str) -> dict:
    with open(path, errors="replace") as f:
        text = f.read()
    found = {}
    for key, pattern in LOG_PATTERNS.items():
        m = re.findall(pattern, text)
        found[key] = float(m[0]) if m else None
    found["oom_lines"] = len(re.findall(r"out of memory|OutOfMemoryError", text, re.IGNORECASE))
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--arm", choices=tuple(both_cpu_ab.ARMS), required=True)
    p.add_argument("--env", action="append", default=[], help="K=V on top of the arm's overrides")
    p.add_argument("--prompt-tokens", default="4096,30000")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    extra = dict(kv.split("=", 1) for kv in a.env)
    overrides, _ = both_cpu_ab.ARMS[a.arm]
    env = os.environ | arm_env.arm_env({**overrides, **extra}) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = [*arm_env.ServerArgs(port=PORT, dspark=True).argv(), *both_cpu_ab.server_flags(a.arm)]
    log_path = os.path.join(a.out_dir, "server.log")
    result = {"arm": a.arm, "env": extra, "argv": argv, "prompts": []}
    with run_locks.take(DISK_LOCK), run_locks.take(arm_env.GPU_LOCK):
        sampler = Sampler(os.path.join(a.out_dir, "gpu-mem.csv"))
        sampler.start()
        with open(log_path, "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        rc = 1
        try:
            if not _healthy(PORT, time.monotonic() + arm_env.DSPARK_HEALTH_TIMEOUT_S):
                return 1
            time.sleep(5)
            result["healthy_used_mib"] = sampler.samples[-1][1]
            for i, n in enumerate(int(x) for x in a.prompt_tokens.split(",")):
                start = time.monotonic()
                try:
                    _generate(n, seed=i)
                    ok = True
                except OSError as e:
                    ok = repr(e)
                end = time.monotonic()
                time.sleep(1)
                peak = sampler.peak(start, end + 1)
                result["prompts"].append({"tokens": n, "ok": ok, "seconds": end - start, "peak_used_mib": peak,
                                          "left_mib": sampler.total_mib - peak})
                print(json.dumps(result["prompts"][-1]), flush=True)
            rc = 0 if all(r["ok"] is True for r in result["prompts"]) else 1
        finally:
            stop_server(server)
            sampler.stop()
    result["total_mib"] = sampler.total_mib
    result.update(_from_log(log_path))
    with open(os.path.join(a.out_dir, "peaks.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "argv"}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
