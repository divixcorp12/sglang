"""The DSV4.1 EXL3 serving recipe: environment and `sglang serve` flags for every arm.

Env names and the option-C EXL3 budget (graph gather on, prefetch off, the fastest
measured DSV4.1 recipe, DSV41_REFERENCE.md section 17.6) originally came from
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/
env-full.sh` (layered onto phase 3a's `env.sh`). The current default additionally
enables DIRECT insert on miss, RAM miss leases, eight RAM miss pack workers,
two-phase RAM-miss copies with piece streaming, a NUMA-placed 100 GiB pinned tier,
the fused expert graph planner, and Engram host-node io_uring lookups; it
must be measured as a new recipe, not compared as a historical phase-3b baseline.
A V2 storage change under test is
layered on top via `overrides`; the merged dict is both what launches the server and
what `verify_env` checks against `/proc/<pid>/environ` afterwards.

`ServerArgs.argv()` started from the working launch line in
`divix01:/data/models/slang/nvfp4-work/cc-dsv41-base/analysis/baseline/smoke.sh`
(2026-09-21: server came up, served `/v1/chat/completions` 200 OK, and ran the
breakable decode CUDA graph at batch size 1 — `cuda graph: True` in the server log —
which is the path every recorded DSV4.1 number describes). The current benchmark
uses production's 32,768-token context and enabled prefix cache; those differ from
the smoke launch and make historical throughput numbers non-comparable. `smoke.sh`
needed exactly two flags beyond the resolved `server_args`:
`--expert-distribution-recorder-mode per_pass` (the EXL3 gate refuses dynamic hot
caching without it) and
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

# Match the current production launcher; older benchmark arms used 4096.
CONTEXT_LENGTH = 32768
# 0.83 since 2026-09-25, with CUDA_MODULE_LOADING=EAGER (base_env): eager loading keeps every kernel resident, ~1 GiB,
# and at 0.80 the KV cache no longer fit. At 0.83 it holds 204,288 tokens (0.80 under LAZY: 209,408), with the same
# ~4.7 GB left over. An arm run at 0.80 is not comparable on memory, only on speed.
MEM_FRACTION_STATIC = 0.83
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

# Host-memory budget, resized 2026-09-22. The server's threads all sit on NUMA node 0
# (see SERVER_CORES), so the pinned buffer plus the weights must fit in node 0's free
# memory -- the whole box's free memory is the wrong number to reason with. Node 0 had
# 66619 MiB free immediately before a launch on 2026-09-22, and the previous 71680 MiB
# pinned budget alone already exceeded that, so the arm exhausted node 0 (down to
# 0.77 GB free) and spun in direct compaction instead of starting. 50 GiB leaves room
# for the ~10 GiB of weights and slack.
#
# This is a recorded constant of the measured recipe. An arm run at this size is NOT
# comparable to the 2.741 / 2.102 cells, which were measured at 71680 MiB; a valid A/B
# needs both of its arms run at the same value.
#
# Since 2026-09-24 the tier is 100 GiB, bound explicitly with SGLANG_MOE_PINNED_HOST_NUMA_MB
# rather than placed by first touch: 60 GiB on node 0, the most that fits beside the
# weights with 4 GiB of headroom, and 40 GiB on node 1. That node's share comes from reclaiming
# co-tenants' page cache (DSV41_REFERENCE.md section 24.6). Startup refuses a node that
# cannot hold its share instead of spilling and stalling.
PINNED_HOST_MB = "102400"  # 100 GiB
PINNED_HOST_NUMA_MB = "0:61440,1:40960"
NODE0_FREE_MIB = 81869  # measured 2026-09-24 14:08, production and every arm down
WEIGHTS_AND_OVERHEAD_MIB = 12288  # ~9.94 GiB of weights, plus slack

HEALTH_TIMEOUT_S = 900  # /health runs a real generation; slow cold. Never shorten this.

# Production serves the base recipe unchanged on all interfaces; launch_prod.sh is its only launcher.
PROD_PORT = 7867
PROD_HOST = "0.0.0.0"


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
        "SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING": "1",
        "SGLANG_DSV41_EXPERT_STREAM": "1",
        "SGLANG_DSV41_EXPERT_DIR": EXPERT_DIR,
        "SGLANG_MOE_EXPERT_ROW_SOURCE": "shards",
        "SGLANG_MOE_EXPERT_MIRROR_DIRS": EXPERT_MIRROR_DIRS,
        "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
        "SGLANG_MOE_PINNED_HOST_MB": PINNED_HOST_MB,
        "SGLANG_MOE_PINNED_HOST_NUMA_MB": PINNED_HOST_NUMA_MB,
        "SGLANG_MOE_HOT_GPU_MB": "14336",
        "SGLANG_MOE_HOT_DYNAMIC": "1",
        "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS": "256",
        "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "1",
        "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": "8",
        "SGLANG_MOE_HOT_ASYNC_PROMOTIONS": "0",
        "SGLANG_MOE_ASYNC_RESIDENCY_SCORES": "0",
        "SGLANG_MOE_HOT_LOG_INTERVAL": "64",
        "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1",
        "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2",
        "SGLANG_MOE_EXPERT_DOORBELL": "0",
        "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
        "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
        "SGLANG_MOE_EXPERT_FUSED_PLAN": "1",
        "SGLANG_DSV41_RAM_MISS_TIMEOUT_MS": "2000",
        "SGLANG_DSV41_ENABLE_RAM_MISS_LEASES": "1",
        "SGLANG_DSV41_RAM_MISS_PACK_WORKERS": "8",
        # Two-phase RAM-miss copies with piece streaming (DSV41_REFERENCE.md section 24).
        "SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE": "1",
        "SGLANG_DSV41_RAM_MISS_HIT_WAIT_US": "100",
        "SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM": "1",
        # Reads land straight in the pinned slabs, no pack workers (section 24.9). Needs the row images built on
        # every mirror root by scripts/dsv41/build_row_images.py; startup refuses a root without a matching set.
        "SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES": "1",
        # Three fused bookkeeping kernels replace 89 torch kernels per layer, byte-identical
        # (docs/superpowers/plans/2026-09-25-dsv41-layer-fusion.md).
        "SGLANG_DSV41_ENABLE_LAYER_FUSION": "1",
        # Engram lookups by device post/wait instead of graph host nodes: no host nodes in the decode graph,
        # byte-identical (docs/superpowers/plans/2026-09-25-dsv41-engram-no-hostnode.md).
        "SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT": "1",
        # RAM-hit rows copied by the DMA engine instead of the SM kernel C1: 112.4 vs 119.3 ms/token, byte-identical
        # (docs/superpowers/plans/2026-09-25-dsv41-final-arms.md). Needs the device wait above (no graph host nodes).
        # A kernel module first loaded mid-step after arming can still fail-stop the server (LEASE_PROTOCOL.md 7.6),
        # and graph-mode nsys must not be used with it on.
        "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": "1",
        # The copy engine requires it and the server refuses to start without it: a kernel loaded lazily after the
        # copy engine arms fail-stopped the soak deterministically, never under EAGER
        # (docs/superpowers/plans/2026-09-25-dsv41-copy-engine-soak.md). Costs ~1 GiB, hence MEM_FRACTION_STATIC.
        "CUDA_MODULE_LOADING": "EAGER",
        "SGLANG_DSV41_ENABLE_EXPERT_PREFETCH": "0",
    }


def arm_env(overrides: dict[str, str]) -> dict[str, str]:
    """The base recipe with a V2 storage-change arm's overrides layered on top."""
    env = base_env()
    env.update(overrides)
    return env


class ServerArgs(msgspec.Struct, frozen=True, kw_only=True):
    port: int
    host: str = "127.0.0.1"
    # ServerArgs default is 40. Not part of smoke.sh's flag set; exists so
    # decode_log_interval_compare.sh can override it per arm without touching the
    # base recipe. See README "One harness, not two", option 2: a genuine
    # engine-side step-latency source, with an unmeasured observer-effect risk at
    # interval=1 that this override exists to measure, not to assume.
    decode_log_interval: int | None = None

    @classmethod
    def prod(cls) -> "ServerArgs":
        return cls(port=PROD_PORT, host=PROD_HOST)

    def argv(self, *, python: str = PYTHON) -> list[str]:
        """The smoke-validated launch, updated to production's context and cache mode."""
        argv = [
            python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_PATH,
            "--host",
            self.host,
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
            # Needs the prefill CUDA graph off (it is, above) and no DP attention (deepseek_v4_hook refuses both).
            "--enable-decoder-swa-bounded-replay",
        ]
        if self.decode_log_interval is not None:
            argv += ["--decode-log-interval", str(self.decode_log_interval)]
        return argv
