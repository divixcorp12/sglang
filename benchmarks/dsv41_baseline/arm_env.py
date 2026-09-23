"""The DSV4.1 EXL3 serving recipe: environment and `sglang serve` flags for every arm.

Env names and the option-C EXL3 budget (graph gather on, prefetch off, the fastest
measured DSV4.1 recipe, DSV41_REFERENCE.md section 17.6) come from
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/
env-full.sh` (layered onto phase 3a's `env.sh`). A V2 storage change under test is
layered on top via `overrides`; the merged dict is both what launches the server and
what `verify_env` checks against `/proc/<pid>/environ` afterwards.

`ServerArgs.argv()` is copied verbatim from the working launch line in
`divix01:/data/models/slang/nvfp4-work/cc-dsv41-base/analysis/baseline/smoke.sh`
(2026-09-21: server came up, served `/v1/chat/completions` 200 OK, and ran the
breakable decode CUDA graph at batch size 1 — `cuda graph: True` in the server log —
which is the path every recorded DSV4.1 number describes). `smoke.sh` needed exactly
two flags beyond the resolved `server_args`: `--expert-distribution-recorder-mode
per_pass` (the EXL3 gate refuses dynamic hot caching without it) and
`--disable-shared-experts-fusion`, both included below.

`--reasoning-parser` / `--tool-call-parser` are deliberately NOT passed: the smoke
launch's `/v1/chat/completions` request returned 200 without them, confirming the
static-code reasoning (`resolve_chat_encoding_spec` selects the `dsv41` chat encoder
from `hf_config.model_type == "deepseek_v41"`, independent of either flag).
"""

from __future__ import annotations

import msgspec

NVFP4_WORK = "/data/models/slang/nvfp4-work"
CC = f"{NVFP4_WORK}/cc-expert-prediction"

MODEL_PATH = f"{CC}/dsv41-full40"
ENGRAM_TABLE_DIR = "/mnt/nvme2/DeepSeek-V4.1-Flash"
EXPERT_DIR = "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
# Expert-row mirrors, on by default since 2026-09-22. Mirroring is a property of the
# box's storage, not of any one arm, so an arm that forgets it measures a drive layout
# nobody runs. Override to the empty string to measure the unmirrored drive.
EXPERT_MIRROR_DIRS = "/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash"
EXL3_SRC = f"{NVFP4_WORK}/exllamav3"
EXL3_BUILD_DIR = f"{CC}/exl3-build"
CUDA_HOME = "/usr/local/cuda-13.2"
PYTHON = "/data/models/slang/.venv/bin/python"
GPU_LOCK = f"{NVFP4_WORK}/cc-gpu.lock"

# smoke.sh's exact flag set (see module docstring).
CONTEXT_LENGTH = 4096
MEM_FRACTION_STATIC = 0.80
CHUNKED_PREFILL_SIZE = 512
MAX_PREFILL_TOKENS = 16384
# Decode CUDA graphs exist only at batch size 1; anything above it runs eagerly and
# skips the path under test. Sequential driver, one turn at a time (team ruling).
MAX_RUNNING_REQUESTS = 1

MAX_TOKENS = 128

# Node-0 cores only, since 2026-09-22. divix01 is two NUMA nodes of ~96 GB:
# node 0 is cpus 0-17,36-53 and node 1 is 18-35,54-71. The old "32-63" spanned both,
# so first-touch put part of the 70 GiB pinned host buffer on node 1 -- which the
# box's reth/nimbus stack keeps at ~9 GB free. Huge-page allocations there fail, and
# with THP enabled=always the allocator does not fail over, it spins in direct
# compaction: three threads at 100% system time, zero completed syscalls, GPU idle,
# and no server log line after "Load weight end" until the 900s abort. Keeping every
# server thread on node 0 keeps its memory on node 0's free ~70 GiB.
# Driver cores 8-15 are node 0 too and are excluded here so the two never overlap.
SERVER_CORES = "0-7,16-17,36-53"
DRIVER_CORES = "8-15"
FREE_CORES = "64-71"  # never touched; core 71 is production's doorbell spin core.

HEALTH_TIMEOUT_S = 900  # /health runs a real generation; slow cold. Never shorten this.


def base_env() -> dict[str, str]:
    """The option-C EXL3 recipe env, before a V2 storage-change override is applied."""
    return {
        "CUDA_HOME": CUDA_HOME,
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SGLANG_EXL3_SRC": EXL3_SRC,
        "SGLANG_EXL3_BUILD_DIR": EXL3_BUILD_DIR,
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_DSV41_ENGRAM_TABLE_DIR": ENGRAM_TABLE_DIR,
        "SGLANG_DSV41_TORCH_PREFILL_INDEXER": "1",
        "SGLANG_DSV41_ENGRAM_RAM_GIB": "5",
        "SGLANG_DSV41_EXPERT_STREAM": "1",
        "SGLANG_DSV41_EXPERT_DIR": EXPERT_DIR,
        "SGLANG_MOE_EXPERT_ROW_SOURCE": "shards",
        "SGLANG_MOE_EXPERT_MIRROR_DIRS": EXPERT_MIRROR_DIRS,
        "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
        "SGLANG_MOE_PINNED_HOST_MB": "71680",
        "SGLANG_MOE_HOT_GPU_MB": "14336",
        "SGLANG_MOE_HOT_DYNAMIC": "1",
        "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS": "256",
        "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "32",
        "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": "8",
        "SGLANG_MOE_HOT_ASYNC_PROMOTIONS": "0",
        "SGLANG_MOE_HOT_LOG_INTERVAL": "64",
        "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "0",
        "SGLANG_MOE_EXPERT_DOORBELL": "0",
        "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
        "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
        "SGLANG_DSV41_RAM_MISS_TIMEOUT_MS": "2000",
        "SGLANG_DSV41_ENABLE_EXPERT_PREFETCH": "0",
    }


def arm_env(overrides: dict[str, str]) -> dict[str, str]:
    """The base recipe with a V2 storage-change arm's overrides layered on top."""
    env = base_env()
    env.update(overrides)
    return env


class ServerArgs(msgspec.Struct, frozen=True, kw_only=True):
    port: int
    # ServerArgs default is 40. Not part of smoke.sh's flag set; exists so
    # decode_log_interval_compare.sh can override it per arm without touching the
    # base recipe. See README "One harness, not two", option 2: a genuine
    # engine-side step-latency source, with an unmeasured observer-effect risk at
    # interval=1 that this override exists to measure, not to assume.
    decode_log_interval: int | None = None

    def argv(self, *, python: str = PYTHON) -> list[str]:
        """`smoke.sh`'s exact flag set, copied rather than re-derived (team lead's
        2026-09-21 smoke confirmed this launches, serves 200s, and runs the breakable
        decode graph)."""
        argv = [
            python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_PATH,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--context-length",
            str(CONTEXT_LENGTH),
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
            "--max-prefill-tokens",
            str(MAX_PREFILL_TOKENS),
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--disable-radix-cache",
            "--cuda-graph-backend-decode",
            "breakable",
            "--cuda-graph-bs-decode",
            "1",
            "--cuda-graph-max-bs-decode",
            "1",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--expert-distribution-recorder-mode",
            "per_pass",
            "--disable-shared-experts-fusion",
        ]
        if self.decode_log_interval is not None:
            argv += ["--decode-log-interval", str(self.decode_log_interval)]
        return argv
