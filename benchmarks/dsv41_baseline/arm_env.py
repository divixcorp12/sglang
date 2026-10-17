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

`--reasoning-parser deepseek-v41` and `--tool-call-parser deepseekv41` are passed since
2026-10-16 (owner). The chat encoder never needed them (`resolve_chat_encoding_spec`
selects `dsv41` from `hf_config.model_type == "deepseek_v41"`). They split the thinking
into `reasoning_content` and return tool calls as `tool_calls` instead of raw DSML text.
`run_capture_sessions.py` and `long_prompt.py` read both delta fields and take token
counts from `usage`. A multi-turn session's history now carries only `content`, so
later-turn prompts differ from the pre-2026-10-16 runs: don't compare across the change.
"""

from __future__ import annotations

import os

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
EXPERT_MIRROR_DIRS = "/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash:/mnt/nvme2/dsv41_flash"
EXL3_SRC = f"{NVFP4_WORK}/exllamav3"
EXL3_BUILD_DIR = f"{CC}/exl3-build"
# GCC 15 builds the optimized EXL3 CPU kernel (validated on it) and, as CXX, every sglang JIT build: host C++ and
# nvcc's host half through -ccbin.
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
# Since 2026-10-05 the server keeps 6 physical cores (0-5 and their siblings 36-41): CPU experts run on both nodes,
# and node 0's other 12 physical cores go to the expert stream (CPU experts 6-15, copy thread 16, RAM thread 17).
# Arms from here on are not comparable to earlier cells.
SERVER_CORES = "0-5,36-41"
# The RAM-miss service busy-polls cpu 17; its SMT sibling 53 and 17 itself are left out of SERVER_CORES so the physical
# core is the service's alone (plan 2026-10-01-expert-stream-read-record D4). Arms from here on have one physical core
# fewer: not comparable to earlier cells.
SPIN_CORE = 17
# CPU expert workers per NUMA node (SGLANG_DSV41_CPU_EXPERTS_THREADS): node 0 gets 6-15, node 1 18-27.
CPU_EXPERTS_THREADS = 10
# Node 1, clear of its CPU experts (18-27) and RAM thread (35): node 0 has no physical core left for the driver.
DRIVER_CORES = "28-31"
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
#
# Since 2026-10-08 the tier is 104 GiB, 56 on node 0 and 48 on node 1 (DSV41_REFERENCE.md section 33.12). At 80 GiB,
# 35% of a DSpark verify's VRAM misses were RAM misses too, and their NVMe reads set the per-layer lease gate that was
# 61% of the step; at 104 GiB the RAM-miss share is 15% and the median decode 24% faster. The margin is thin: startup
# swaps ~12 GiB out and MemFree sits near 1 GiB while serving.
PINNED_HOST_MB = "106496"  # 104 GiB
PINNED_HOST_NUMA_MB = "0:57344,1:49152"
NODE0_FREE_MIB = 74849  # measured 2026-10-08 00:00, production and every arm down
WEIGHTS_AND_OVERHEAD_MIB = 12288  # ~9.94 GiB of weights, plus slack

HEALTH_TIMEOUT_S = 900  # /health runs a real generation; slow cold. Never shorten this.
# A DSpark server loads the target (~435 s), the draft (~870 s in), then autotunes and captures graphs (2026-10-06 smoke).
DSPARK_HEALTH_TIMEOUT_S = 2700

# Production serves the base recipe unchanged on all interfaces; launch_prod.sh is its only launcher.
PROD_PORT = 7867
PROD_HOST = "0.0.0.0"

# DSpark with both CPU-expert clients (plan 2026-10-06-dsv41-dspark-both-cpu-experts). Production serves it only once
# PROD_DSPARK is True (Owner decision 1, the A/B of that plan's Task 17).
# On since 2026-10-07, on the owner's go ("Merge + flip prod to DSpark") after DSV41_REFERENCE.md §33.11: all six
# bars pass; the median decode is 7.6% above the non-DSpark recipe (95.05 vs 88.30 ms/token, the addendum's A/B).
PROD_DSPARK = True
DSPARK_DRAFT = f"{CC}/dsv41-dspark-draft"
# Each stage's top-32 draft experts stay on the GPU, the other 96 on the CPU (§33.4).
DSPARK_RESIDENT = f"{CC}/analysis/dsv41-dspark/cpu-draft-routes/resident-top32.json"
# gamma = 5 draft tokens, a 6-token verify (speculative_hook.py).
DSPARK_ARGV = (
    "--speculative-algorithm", "DSPARK",
    "--speculative-draft-model-path", DSPARK_DRAFT,
    "--speculative-dspark-block-size", "5",
)


def base_env() -> dict[str, str]:
    """The option-C EXL3 recipe env, before a V2 storage-change override is applied."""
    return {
        "CUDA_HOME": CUDA_HOME,
        "CXX": EXL3_CPU_CXX,
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
        # CPU experts on both NUMA nodes' pinned tiers (ThreadingConfig derives each node's cores; see SERVER_CORES).
        # EXL3_MOE_CPU_PIN=0: the engine pins its own workers; the kernel's default would pin them to the first cores.
        "SGLANG_DSV41_CPU_EXPERTS": "1",
        "SGLANG_DSV41_CPU_EXPERTS_THREADS": str(CPU_EXPERTS_THREADS),
        "EXL3_MOE_CPU_PIN": "0",
        # The lease chain's W1 budget (the chain is two-phase with piece streaming, DSV41_REFERENCE.md section 24).
        "SGLANG_DSV41_RAM_MISS_HIT_WAIT_US": "100",
        # Three fused bookkeeping kernels replace 89 torch kernels per layer, byte-identical
        # (docs/superpowers/plans/2026-09-25-dsv41-layer-fusion.md).
        "SGLANG_DSV41_ENABLE_LAYER_FUSION": "1",
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
        # Prefill's pinned-tier misses read by the RAM-miss service's reader straight into the slabs, a layer's reads
        # issued up front: TTFT 21.06/17.79 -> 12.05/11.14 s, decode unchanged, byte-identical (DSV41_REFERENCE.md 27.6).
        # Reads the row images the RAM-miss service always reads (built on every mirror root by
        # scripts/dsv41/build_row_images.py; startup refuses a root without a matching set).
        "SGLANG_DSV41_ENABLE_PREFILL_FILLS": "1",
        # A prefill's pinned-tier admissions own at most 64 rows per layer, then evict their own LRU row instead of
        # decode's. Served A/B 2026-10-16 (DSpark -mergechunk-expseg arm, one pair): -3.46 ms/token, 95% CI
        # -4.91..-1.55, TTFT -0.48 s; 56-59% of decode's NVMe-cold GPU lanes had been evicted by a prefill
        # (DSV_OPTIMIZATION.md 3.8, /mnt/nvme1/share64-ab-20261016). Needs --expert-distribution-recorder-mode
        # (passed below); the service refuses to start without it. On by the owner's decision, 2026-10-16.
        "SGLANG_DSV41_ENABLE_PREFILL_SHARE": "1",
        # Fewer fp16/bf16 casts around the EXL3 gemvs at BS1 decode: 358 fewer kernels per step, bit-identical,
        # 112.4 -> 111.8 ms/token (within noise; DSV41_REFERENCE.md 27.8).
        "SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION": "1",
        # The eager streamed MoE queues each chunk's compute behind its gather instead of waiting for it: TTFT
        # 12.6 -> 9.9 s on a 260-token prompt, outputs identical (DSV41_REFERENCE.md 27.11).
        "SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN": "1",
        # A prefill chunk copies its rows already in the pinned tier before waiting for its fills: TTFT 9.9 -> 8.6 s,
        # outputs identical (DSV41_REFERENCE.md 27.12).
        "SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER": "1",
        # CW reads each RAM-hit row's four small tensors itself, so the copy engine sends 2 copies per row, not 6:
        # outputs identical, ~1 ms/step in a node-mode trace, within noise untraced (DSV41_REFERENCE.md 27.14).
        # Needs the copy engine above.
        "SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES": "1",
        # The lease chain's kernels launch with PDL: at most ~1 us/layer, outputs identical
        # (analysis/dsv41-drive/chain-pdl/results.md, on expert-stream-transfer-measurement). Needs the leases above.
        "SGLANG_DSV41_ENABLE_LEASE_PDL": "1",
        # Suffixes of 8192+ uncached tokens prefill layer-major: token 0 identical to chunked at 8k-33k
        # (DSV41_REFERENCE.md 27.19, 27.20). Not yet measured at 128k+. An arm sets "0" to force chunked.
        "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "8192",
    }
# Budget A (both-cpu/mem-budget-report.md, launch L2), for the DSpark arms only; production keeps MEM_FRACTION_STATIC
# and its 16080 MiB hot cache. The triton FlashMLA prefill merge runs in fp32 and costs about 0.9 GiB at a 2048-token
# chunk and 1.75 GiB at 4096 over production's flashinfer path, so DSpark's prefill peak sits 4,808 MiB above the end of
# init. Init headroom is set by the fraction alone: a hot-cache cut by itself only grows the KV pool (launch A3). At
# 0.78 / 10840 init has 5.21 GB of headroom, the 4096 warm-up leaves 508 MiB, and KV is 381,184 tokens (prod 288,768).
# TODO(owner, 2026-10-06): the real fix is option C — chunk the triton fp32 prefill merge over tokens (bit-identical
# math), then return the DSpark arms to MEM_FRACTION_STATIC 0.82 and the full 12040 MiB hot cache (Owner decision 4).
# flashinfer is not an alternative: it refuses the 5-token draft capture (`num_tokens > 64`).
DSPARK_MEM_FRACTION_STATIC = "0.78"
DSPARK_HOT_GPU_MB = "10840"
# Production's own DSpark memory sizes since 2026-10-16 (MERGE_PLAN_1016.md). The two above stay the OFF -mem arm's
# measured init: both_cpu_ab's fraction_keeping_kv and hot_mb_for_margin derive every arm's sizes from them. These are
# the -mergechunk-expseg arm's (both_cpu_ab.EXPSEG_HOT_GPU_MB, its ARM_MEMORY_FLAGS), the best served DSpark arm: 1,035
# hot slots, the fraction that keeps the KV pool at the pin, and the pin. With SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP
# (prod_dspark_env) the hot cache is the starting point the startup calibration grows or trims from.
PROD_DSPARK_HOT_GPU_MB = "13144"
PROD_DSPARK_MEM_FRACTION_STATIC = "0.8623"
PROD_DSPARK_MAX_TOTAL_TOKENS = 381184
# The pin01 arms' pinned tier (both_cpu_ab.PIN01_BUDGET): layers 0 and 1 whole, plus the old tier's rows elsewhere.
PROD_DSPARK_PINNED_HOST_MB = "111530"
PROD_DSPARK_PINNED_HOST_NUMA_MB = "0:59656,1:51874"
# launch_prod.sh's pre-launch floor (analysis/dsv41-drive/dspark/prelaunch_evict.py, the arms' logic): node 1 must hold
# the HiCache host pools of the pinned 381,184-token pool (12.67 GiB, as the share-64 A/B ran it), the rest of the
# server's node-1 need (arm A's 51.24 GiB) and 1 GiB of margin: 64.91 GiB. Every served -mem arm started with node 0 at
# 75.4 GiB free or more; below that the start only warns. The wait is bounded so a refusal comes in minutes.
PROD_NODE1_HICACHE_GIB = 12.67
PROD_NODE1_MARGIN_GIB = 1.0
PROD_NODE0_WARN_GIB = 75.0
PROD_FLOOR_WAIT_MIN = 10
# Other servers that read the files the pre-launch step evicts: while one listens, production checks without evicting.
PROD_PRELAUNCH_SKIP_EVICT_PORTS = (30032,)
# Where a refusal looks for the page cache that fills node 1.
PROD_PAGE_CACHE_ROOTS = ("/mnt/nvme0", "/mnt/nvme1", "/mnt/nvme2", "/mnt/nvme4", "/data/models")
# The RAM prefetch's learned L+2 T3 head (prod_dspark_env), a read-only production copy of the file the best arm
# served (nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors). launch_prod.sh refuses to start unless the
# file is there with this SHA256 (check_prod_assets).
PROD_RAM_PREFETCH_LEARNED_HEAD = "/data/models/slang/prod-assets/ram-prefetch/t3-f109-calib-b075.f0d7a1b1.safetensors"
PROD_RAM_PREFETCH_LEARNED_HEAD_SHA256 = "f0d7a1b10e029480bcb9687437d7880135d756ea57f186d9a9cd41729fbd5309"
# The DSpark confidence-head verify cut's fitted STS temperatures (docs/hx/compact-verify-exl3.draft.md), for the cut arm
# only (both_cpu_ab's -prod-confcut); production stays off and loads neither. Fitted 2026-10-16 from the -prod-stscollect
# arm's long-prompt capture (/mnt/nvme1/sts-collect-20261016, 2,048 verifies): temperatures 0.89/0.79/1.26/0.89/0.63,
# ECE 2.9-3.8% -> 2.1-3.0%. Read-only copy, 0444.
PROD_DSPARK_STS_PATH = "/data/models/slang/prod-assets/dspark-sts/dsv41-dspark-sts.json"
PROD_DSPARK_STS_SHA256 = "a0f2fc7c6acd4ca06dca213d68954b3f46e0fef2a5e6ed4ab460feda7538bcd5"


def arm_env(overrides: dict[str, str]) -> dict[str, str]:
    """The base recipe with a V2 storage-change arm's overrides layered on top."""
    env = base_env()
    env.update(overrides)
    return env


def dspark_env() -> dict[str, str]:
    """The DSpark mode's overrides on base_env: the graphed verify with the target's CPU experts and spill, and the
    draft's CPU experts. Every value's reason is in the plan's Task 15 table."""
    return {
        # The verify in the breakable decode graph with a lane per route (MISS_LANES unset: 36 lanes, a 40-lane wire),
        # the first 8 VRAM victims; the rest are CPU lanes, a miss among them read into a RAM victim (Owner decision 3).
        "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "8",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        # Layer-major prefill refuses speculative decoding (layer_major/gate.py).
        "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0",
        # The draft's CPU experts as a second job source of node 0's CPU expert team (one team per node), on the
        # optimized build (its gate names both defines).
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": "1",
        "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": DSPARK_RESIDENT,
        "SGLANG_EXL3_CPU_ACT_RESIDUAL": "1",
        "SGLANG_EXL3_CPU_ACT_BLOCK": "128",
        # Stated off: the row-weighted B arms (analysis/dsv41-drive/dspark/both_cpu_ab.py) override it, and a captured
        # shell must not carry one into an A.
        "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "",
        # Budget A, with DSPARK_MEM_FRACTION_STATIC (reasons and the owner's option-C TODO at DSPARK_MEM_FRACTION_STATIC).
        "SGLANG_MOE_HOT_GPU_MB": DSPARK_HOT_GPU_MB,
        # Every DSpark text result so far ran triton attention (§33.2, §33.8, §33.9).
        "SGLANG_SM120_FLASHMLA_BACKEND": "triton",
    }


def prod_dspark_env() -> dict[str, str]:
    """Production's DSpark settings on top of dspark_env() since 2026-10-16 (MERGE_PLAN_1016.md): the best served arm,
    both_cpu_ab's ...-pin01-splitmodel-mem-hicache1-mergechunk-expseg, with its RAM prefetch and both startup
    calibrations on. The arms state each of these themselves, so they stay out of dspark_env(), which every arm builds
    on. Evidence is DSV_OPTIMIZATION.md (DO) and DSV41_REFERENCE.md; "pair" means one served A/B pair.

    Pending, not set here: SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM (needs an A/B on the copy engine and the graphed
    verify) and SGLANG_OPT_LANE_HEAD_PATH (held)."""
    return {
        # Whole layers 0 and 1 in the pinned tier, in a tier grown to hold them: 70.5 vs 76.2 ms/token, layers 0+1
        # 20.5 -> 13.0 ms a verify (DO 3.1; the OFF run was stall-flagged).
        "SGLANG_MOE_PINNED_HOST_FULL_LAYERS": "0,1",
        "SGLANG_MOE_PINNED_HOST_MB": PROD_DSPARK_PINNED_HOST_MB,
        "SGLANG_MOE_PINNED_HOST_NUMA_MB": PROD_DSPARK_PINNED_HOST_NUMA_MB,
        # Each record's CPU/copy split from a cost model: 65.9 vs 67.9 ms/token, wait -9.4 ms a verify (DO 3.2, a pair).
        # Its costs are measured at startup whenever it is on (split_model_calibration.py; on failure the table types
        # the lanes), never refined live (owner rule: values calculated at startup; refinement had no gain, DO 3.4).
        "SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL": "1",
        "SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS": "",
        "SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES": "0",
        # The split table calibrated at startup (the library default, stated): the arms ran it off (-fixedsplit) only to
        # hold the table fixed between A and B (REF 33.36). With the model on, it types lanes only if the model is off.
        "SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION": "1",
        # The miss cut at its library default, stated: what the split model replaces when it is on.
        "SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "1",
        "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3",
        # mlockall in the scheduler: the CPU expert teams' major faults 11,877 -> 20-250 a run (DO 3.4).
        "SGLANG_ENABLE_SCHEDULER_MLOCK": "1",
        # HiCache host pools on node 1: node 0's load-time swap 11.6 GiB -> 5 MB, throughput unchanged (DO 3.4).
        "SGLANG_HICACHE_HOST_NUMA_NODE": "1",
        # The triton FlashMLA prefill merge in 128 MiB token chunks, bit-identical, 1,217 MiB of prefill peak freed for
        # the hot cache: -2.02 ms/token, 95% CI -3.7..-0.6 (DO 3.4 re-score, 3.7).
        "SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB": "128",
        # The caching allocator on expandable segments: ~1.5 GB of stranded reserve recovered at no cost (-0.26
        # ms/token, CI -0.39..-0.02), spent on 1,035 hot slots: -1.79 ms/token, CI -4.51..-0.64 (DO 3.7, expseg2).
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_MOE_HOT_GPU_MB": PROD_DSPARK_HOT_GPU_MB,
        # The hot cache sized at every launch from measured free memory (trial capture, warm-up replay, grow or trim
        # from PROD_DSPARK_HOT_GPU_MB): owner rule. Served pair neutral, +0.62 ms/token, CI -1.88..+2.60, margin held
        # under serving (DO 3.7). A failed calibration keeps the configured size. Without the A1/A2 cuts it was served
        # with: per-chunk kernels off and the indexer budget at base_env's 128, as the best arm ran.
        "SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP": "1",
        "SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS": "0",
        # The NVMe-to-RAM prefetch exactly as the best arm served it (share64-ab-20261016 resolved-env "on"): GPU
        # scorer on layer L's hidden state for layer L+2 through the learned T3 head, 2 reads a layer, 1 a token,
        # speculative reads only to an idle drive (2 pieces in flight, spread, 4 ms deadline), 2 pool slots a group,
        # landed rows promoted into the tier map and GPU misses served from the pool. No clean served win of its own
        # (DSV41_REFERENCE.md 33.14-33.37; promote "converts, does not add", 33.37). Shipped because every stacked
        # served win was measured with it on (-mergechunk, -expseg2, share64-ab-20261016), and the offline comparison
        # could not be resolved (LANE_TIER_REPLAY_HANDOFF.md): owner decision, 2026-10-16. Its 2 pool slots a group
        # are in the usable rows the prefill share is sized from (64 = 64/187 x 187 on this layout). Production only, not
        # base_env(): it was never measured without DSpark, and every DSpark arm states its prefetch keys itself.
        "SGLANG_DSV41_RAM_PREFETCH": "1",
        "SGLANG_DSV41_RAM_PREFETCH_SCORER": "gpu",
        "SGLANG_DSV41_RAM_PREFETCH_HORIZON": "2",
        "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD": PROD_RAM_PREFETCH_LEARNED_HEAD,
        "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "2",
        "SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN": "1",
        "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": "2",
        "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "0",
        "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": "",
        "SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE": "1",
        "SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES": "2",
        "SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD": "1",
        "SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US": "4000",
        "SGLANG_DSV41_RAM_PREFETCH_PROMOTE": "1",
        "SGLANG_DSV41_RAM_PREFETCH_POOL_SERVE": "1",
        "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",  # the older GPU-side expert prefetch stays off (base_env's, stated)
    }


def check_prod_assets(head: str = PROD_RAM_PREFETCH_LEARNED_HEAD,
                      sha256: str = PROD_RAM_PREFETCH_LEARNED_HEAD_SHA256) -> str:
    """launch_prod.sh's asset check: the prefetch head production loads exists and hashes to the pinned SHA256.
    Returns a line saying so; raises SystemExit with the reason otherwise."""
    import hashlib

    if not os.path.isfile(head):
        raise SystemExit(f"not starting production: the RAM prefetch head {head} is missing "
                         f"(arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD; expected sha256 {sha256})")
    digest = hashlib.sha256()
    with open(head, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"not starting production: the RAM prefetch head {head} has sha256 {digest.hexdigest()}, "
                         f"not the pinned {sha256} (arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD_SHA256)")
    return f"prefetch head verified: {head} sha256 {sha256}"


def prod_env() -> dict[str, str]:
    """Production's env: the base recipe, with the DSpark mode and production's DSpark settings once PROD_DSPARK is
    set."""
    return arm_env({**dspark_env(), **prod_dspark_env()}) if PROD_DSPARK else base_env()


class ServerArgs(msgspec.Struct, frozen=True, kw_only=True):
    port: int
    host: str = "127.0.0.1"
    # ServerArgs default is 40. Not part of smoke.sh's flag set; exists so
    # decode_log_interval_compare.sh can override it per arm without touching the
    # base recipe. See README "One harness, not two", option 2: a genuine
    # engine-side step-latency source, with an unmeasured observer-effect risk at
    # interval=1 that this override exists to measure, not to assume.
    decode_log_interval: int | None = None
    # the DSpark mode's argv (DSPARK_ARGV); its env is dspark_env()
    dspark: bool = False
    # Production's DSpark memory (PROD_DSPARK_*), unset for the arms, which append their own (both_cpu_ab.server_flags):
    # its --mem-fraction-static in place of the mode's, the pinned KV pool, and the loader dropping the shards' page
    # cache once the model is built (the -mem arms', DO 3.4).
    mem_fraction_static: str | None = None
    max_total_tokens: int | None = None
    drop_cache_after_load: bool = False
    # Production's warm-up also runs dsv41_decode_paths (~35 s); the arms keep the prefill-only one they measured with.
    decode_warmup: bool = False

    @classmethod
    def prod(cls) -> "ServerArgs":
        if not PROD_DSPARK:
            return cls(port=PROD_PORT, host=PROD_HOST, decode_warmup=True)
        return cls(
            port=PROD_PORT,
            host=PROD_HOST,
            dspark=True,
            mem_fraction_static=PROD_DSPARK_MEM_FRACTION_STATIC,
            max_total_tokens=PROD_DSPARK_MAX_TOTAL_TOKENS,
            drop_cache_after_load=True,
            decode_warmup=True,
        )

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
            self.mem_fraction_static or (DSPARK_MEM_FRACTION_STATIC if self.dspark else str(MEM_FRACTION_STATIC)),
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
            "--reasoning-parser",
            "deepseek-v41",
            # V4.1's DSML tags carry a leading space (" calls", " invoke"); this detector builds its own grammar.
            "--tool-call-parser",
            "deepseekv41",
            # Loads each prefill Triton variant before serving, while device memory is still free (entrypoints/warmup.py).
            # Production also decodes, greedy and sampled, after a two-chunk prompt: its first long request otherwise
            # loaded the SWA eviction and DSpark sampling kernels mid-serving (prod-20261016-180932/server.log).
            "--warmups",
            "dsv41_prefill_shapes,dsv41_decode_paths" if self.decode_warmup else "dsv41_prefill_shapes",
        ]
        if self.decode_log_interval is not None:
            argv += ["--decode-log-interval", str(self.decode_log_interval)]
        if self.dspark:
            argv += list(DSPARK_ARGV)
        if self.max_total_tokens is not None:
            argv += ["--max-total-tokens", str(self.max_total_tokens)]
        if self.drop_cache_after_load:
            argv += ["--weight-loader-drop-cache-after-load"]
        return argv
