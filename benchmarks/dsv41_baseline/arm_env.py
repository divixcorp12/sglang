"""The DSV4.1 EXL3 serving recipe: environment and `sglang serve` flags for every arm.

Env names and the option-C EXL3 budget (graph gather on, prefetch off, the fastest
measured DSV4.1 recipe, DSV41_REFERENCE.md section 17.6) originally came from
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/
env-full.sh` (layered onto phase 3a's `env.sh`). The current default additionally
enables DIRECT insert on miss, two-phase RAM-miss copies with piece streaming (the
RAM-miss service always reads row images, in lease mode), a NUMA-placed 100 GiB pinned tier,
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
# Three roots since 2026-09-28 (/mnt/nvme2 now x4): 101.7 vs 109.9 ms/token for two,
# byte-identical, reads split ~33% per drive (analysis/dsv41-drive/mirror3/).
EXPERT_MIRROR_DIRS = (
    "/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash:/mnt/nvme2/dsv41_flash"
)
EXL3_SRC = f"{NVFP4_WORK}/exllamav3"
EXL3_BUILD_DIR = f"{CC}/exl3-build"
# The optimized EXL3 CPU kernel is validated on GCC 15; only its build uses this (not a global CXX).
EXL3_CPU_CXX = "/opt/rh/gcc-toolset-15/root/usr/bin/g++"
CUDA_HOME = "/usr/local/cuda-13.2"
PYTHON = "/data/models/slang/.venv/bin/python"
GPU_LOCK = f"{NVFP4_WORK}/cc-gpu.lock"

# 262144 since 2026-09-26, for 250k-token prompts; the KV pool then (driver 610.57.04) held ~387k tokens. No prompt
# past 32k has been run at this recipe. Older benchmark arms used 4096, then 32768.
# 131072 since 2026-09-29: driver 615.71.09's larger EAGER footprint and MEM_FRACTION_STATIC 0.885 / hot cache 15400
# shrank the GPU KV pool to 150k-232k tokens across launches, so the context is capped below the smallest measured pool
# (analysis/dsv41-drive/recipe-mem/results.md).
CONTEXT_LENGTH = 131072
# 0.83 since 2026-09-25, with CUDA_MODULE_LOADING=EAGER (base_env): eager loading keeps every kernel resident, ~1 GiB,
# and at 0.80 the KV cache no longer fit. At 0.83 it holds 204,288 tokens (0.80 under LAZY: 209,408), with the same
# ~4.7 GB left over. An arm run at 0.80 is not comparable on memory, only on speed.
# 0.925 since 2026-09-26: the hot cache counts against this fraction, so its +3072 MiB (of 32607) raised it by the
# same amount and the KV pool is unchanged. The indexer score budget is what keeps long prefills inside the rest.
# 0.90 since 2026-09-26, with the hot cache 1 GiB smaller: ~815 MiB goes back to the 4096-token prefill chunk's
# activations, and the KV pool still grows (~387k tokens). At 0.925 a smaller hot cache only grows the KV pool, and
# 2048- and 4096-token chunks run out of memory (27.17).
# 0.885 since 2026-09-29, with the hot cache cut to 15400: NVIDIA driver 615.71.09 (from 610.57.04) takes ~0.5 GiB more
# before weights load under CUDA_MODULE_LOADING=EAGER, all of it out of the KV pool (0.80 -> 0.14-0.38 GB available vs a
# 0.23 GB SWA floor). 0.91 restored the pool but a 16k prompt OOMed. 0.895 + 15400 put the cut into the KV pool (0.71-0.88
# GB available), not prefill headroom: a chunked 16k prompt peaked 1 MiB short of the 32,202 MiB CUDA can use. 0.885
# spends ~320 MiB of that KV spare on prefill headroom (analysis/dsv41-drive/recipe-mem/diagnosis.md, results.md).
# 0.875 since 2026-09-29: at 0.885 a 419-token prompt OOMed in prefill (free fell 2.70 GB -> 37 MB with late Triton
# loads; earlier launches bottomed at 0.12 GB). The ~320 MiB comes out of the hot cache (15400 -> 15080), not the KV
# pool: that pool is only the few hundred MB left in the fraction, and 0.875 alone shrank it 161,536 -> 7,936 tokens.
MEM_FRACTION_STATIC = 0.875
# 4096 since 2026-09-26: a chunk's cost is streaming the experts it routes to, nearly the same at 512 and 4096 tokens,
# so a 16k prompt's TTFT fell 444 -> 107 s (27.17).
CHUNKED_PREFILL_SIZE = 4096
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
SERVER_CORES = "0-7,16,36-52"
# The RAM-miss service busy-polls cpu 17; its SMT sibling 53 and 17 itself are left out of SERVER_CORES so the physical
# core is the service's alone (plan 2026-10-01-expert-stream-read-record D4). Arms from here on have one physical core
# fewer: not comparable to earlier cells.
SPIN_CORE = 17
DRIVER_CORES = "8-15"
FREE_CORES = "64-71"  # never touched; NVMe completion interrupts are pinned there.

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
#
# Since 2026-10-02 node 0's share is 40 GiB (80 GiB in all): co-tenants on node 0 (QuestDB,
# a Ray cluster) left ~52 GiB free, so the 60 GiB share no longer fit beside the weights.
# Decode at 80 GiB is not comparable to cells measured at 100 GiB.
PINNED_HOST_MB = "81920"  # 80 GiB
PINNED_HOST_NUMA_MB = "0:40960,1:40960"
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
        "SGLANG_EXL3_CPU_CXX": EXL3_CPU_CXX,
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_DSV41_ENGRAM_TABLE_DIR": ENGRAM_TABLE_DIR,
        "SGLANG_DSV41_TORCH_PREFILL_INDEXER": "1",
        # Chunks the prefill indexer's score tensor: a 30k/32k prompt peaks 3.0 GiB lower with no OOM retries, and
        # that VRAM funds the larger hot cache below (DSV41_REFERENCE.md 27.7).
        "SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB": "128",
        "SGLANG_DSV41_ENGRAM_RAM_GIB": "5",
        "SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING": "1",
        "SGLANG_DSV41_EXPERT_STREAM": "1",
        "SGLANG_DSV41_EXPERT_DIR": EXPERT_DIR,
        "SGLANG_MOE_EXPERT_ROW_SOURCE": "shards",
        "SGLANG_MOE_EXPERT_MIRROR_DIRS": EXPERT_MIRROR_DIRS,
        "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
        "SGLANG_MOE_PINNED_HOST_MB": PINNED_HOST_MB,
        "SGLANG_MOE_PINNED_HOST_NUMA_MB": PINNED_HOST_NUMA_MB,
        # 14336 + 3072 on the indexer cap's freed VRAM: 110.1 -> 103.0 ms/token, byte-identical (27.7). Counts against
        # MEM_FRACTION_STATIC. Cut to 16100 for 4096-token prefill chunks and a 262144 context (27.17); the decode cost
        # of the cut is not measured (27.7's slope suggests ~2-3 ms/token). Cut to 15400 on 2026-09-29 for driver
        # 615.71.09's larger EAGER footprint, with MEM_FRACTION_STATIC 0.885 (recipe-mem/diagnosis.md, results.md).
        # 15080 since 2026-09-29, with MEM_FRACTION_STATIC 0.875: its ~320 MiB goes to prefill headroom.
        # 16080 since 2026-09-29: --language-model-only frees the ~0.97 GB vision tower (no weights in the
        # checkpoint) and the 0.10 GB multimodal reservation; 1000 MiB of that goes here.
        "SGLANG_MOE_HOT_GPU_MB": "16080",
        "SGLANG_MOE_HOT_DYNAMIC": "1",
        "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS": "256",
        "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "1",
        "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": "8",
        "SGLANG_MOE_HOT_ASYNC_PROMOTIONS": "0",
        "SGLANG_MOE_ASYNC_RESIDENCY_SCORES": "0",
        "SGLANG_MOE_HOT_LOG_INTERVAL": "64",
        "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1",
        "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2",
        "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
        "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
        "SGLANG_MOE_EXPERT_FUSED_PLAN": "1",
        "SGLANG_DSV41_RAM_MISS_TIMEOUT_MS": "2000",
        "SGLANG_DSV41_RAM_MISS_SPIN_CORE": str(SPIN_CORE),
        # The lease chain's W1 budget (the chain is two-phase with piece streaming, DSV41_REFERENCE.md section 24).
        "SGLANG_DSV41_RAM_MISS_HIT_WAIT_US": "100",
        # Engram lookups by device post/wait instead of graph host nodes: no host nodes in the decode graph,
        # byte-identical (docs/superpowers/plans/2026-09-25-dsv41-engram-no-hostnode.md).
        "SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT": "1",
        # RAM-hit rows copied by the DMA engine instead of the SM kernel C1: 112.4 vs 119.3 ms/token, byte-identical
        # (docs/superpowers/plans/2026-09-25-dsv41-final-arms.md). Needs the device wait above (no graph host nodes).
        # A kernel module first loaded mid-step after arming can still fail-stop the server (LEASE_PROTOCOL.md, "Copy engine"),
        # and graph-mode nsys must not be used with it on.
        "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": "1",
        # The copy engine requires it and the server refuses to start without it: a kernel loaded lazily after the
        # copy engine arms fail-stopped the soak deterministically, never under EAGER
        # (docs/superpowers/plans/2026-09-25-dsv41-copy-engine-soak.md). Costs ~1 GiB, hence MEM_FRACTION_STATIC.
        "CUDA_MODULE_LOADING": "EAGER",
        # CW reads each RAM-hit row's four small tensors itself, so the copy engine sends 2 copies per row, not 6:
        # outputs identical, ~1 ms/step in a node-mode trace, within noise untraced (DSV41_REFERENCE.md 27.14).
        # Needs the copy engine above.
        "SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES": "1",
        # Suffixes of 8192+ uncached tokens prefill layer-major: token 0 identical to chunked at 8k-33k
        # (DSV41_REFERENCE.md 27.19, 27.20). Not yet measured at 128k+. An arm sets "0" to force chunked.
        "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "8192",
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
            # A 4k-token conversation's SWA tail does not survive another's prefill in the 3584-slot pool, so revisits
            # reused nothing; the host copy restores them (DSV41_REFERENCE.md 27.16). Costs ~6k tokens of GPU KV.
            "--enable-hierarchical-cache",
            # 14 since 2026-09-29, for a ~10 GB host pool; DSV4 HiCache refuses --hicache-size. Ratio 2 held 1.14 GB
            # at a 195,840-token GPU pool. Each unit is ~0.25 GB of SWA plus ~1.6 KB per GPU pool token, so 14 gives
            # ~6.7-10.6 GB over the 140k-311k-token pools seen across launches.
            "--hicache-ratio",
            "14",
            "--hicache-io-backend",
            "kernel",
            "--hicache-write-policy",
            "write_through",
            # hicache_hook.resolve_layout_io_compatibility turns kernel + page_first_direct into the direct
            # backend, so KV backups are cudaMemcpy DMA copies on the copy engines the RAM-miss path also uses.
            "--hicache-mem-layout",
            "page_first_direct",
            # The checkpoint has no vision weights; without this the ViT and aligner are built empty in VRAM.
            "--language-model-only",
            # Loads each prefill Triton variant before serving, while device memory is still free (entrypoints/warmup.py).
            "--warmups",
            "dsv41_prefill_shapes",
        ]
        if self.decode_log_interval is not None:
            argv += ["--decode-log-interval", str(self.decode_log_interval)]
        return argv
