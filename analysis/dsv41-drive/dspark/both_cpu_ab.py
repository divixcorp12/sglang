"""Server A/B (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 17): today's production recipe against DSpark with
both CPU-expert clients, and DSpark with the draft's only (the target's CPU experts off, §33.9's configuration in a
server) to isolate them.

Per arm: run_arm.sh's timed set over the 8 corpus sessions (ms/token, and accept length from the driver's
spec_tokens_details), then a short-lived server on the same env for logprob_probe.py (8 prompts, 128 tokens, top-5).
The text bar is dspark_text_band.py against prod. Run on divix01 from the pushed worktree holding rowimg-disk.lock
only: run_arm.sh takes cc-gpu.lock itself, and the probe phase takes it here (lock order: disk, then GPU).

    flock /data/models/slang/nvfp4-work/rowimg-disk.lock python analysis/dsv41-drive/dspark/both_cpu_ab.py OUT [ARM ...] [--sessions 0,1] [--baseline DIR]
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
sys.path.insert(0, HERE)
import arm_env  # noqa: E402
import drive_sessions  # noqa: E402
import run_locks  # noqa: E402

PORT = 30017
PROBE_PORT = 30018
SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
DRAFT_ONLY = {
    "SGLANG_DSV41_CPU_EXPERTS": "0",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": "8",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "0",
}
# Every arm but the prefetch's states the RAM prefetch off, so a shell that exports it cannot carry it into an arm.
PREFETCH_OFF = {"SGLANG_DSV41_RAM_PREFETCH": "0", "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": ""}
# Every arm but the miss cut's states the CPU split's miss cut off, likewise.
MISS_CUT_OFF = {"SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "0", "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3"}
# The GPU scorer's margin floors fit by spec_margin.py's admission sweep (the file says on what), packaged as its default.
MARGIN_FLOORS = os.path.join(REPO, "python", "sglang", "srt", "layers", "moe", "ram_prefetch_margin_floors.json")
# arm: (overrides on base_env, DSpark argv)
ARMS = {
    "prod": ({**PREFETCH_OFF, **MISS_CUT_OFF}, False),
    "dspark-draft-only": ({**arm_env.dspark_env(), **DRAFT_ONLY, **PREFETCH_OFF, **MISS_CUT_OFF}, True),
    "dspark-both": ({**arm_env.dspark_env(), **PREFETCH_OFF, **MISS_CUT_OFF}, True),
    # Experiment-only; retire with plan 2026-10-07-dsv41-row-weighted-serving-experiment. B arms, one source at a time.
    "dspark-both-rw-draft": (
        {**arm_env.dspark_env(), **PREFETCH_OFF, **MISS_CUT_OFF, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "draft"},
        True,
    ),
    "dspark-both-rw-target": (
        {**arm_env.dspark_env(), **PREFETCH_OFF, **MISS_CUT_OFF, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "target"},
        True,
    ),
    # NVMe-to-RAM prefetch (spec 2026-10-08-dsv41-ram-prefetch-design, The A/B): the replay's best arm, h=1, one
    # candidate per token, one row per layer, a pool of 2 per row and group; all pinned so a shell cannot leak.
    "dspark-both-prefetch": (
        {
            **arm_env.dspark_env(),
            "SGLANG_DSV41_RAM_PREFETCH": "1",
            "SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN": "1",
            "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "1",
            "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": "2",
            "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "0",
            "SGLANG_DSV41_RAM_PREFETCH_SCORER": "cpu",
            "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": "",
            **MISS_CUT_OFF,
        },
        True,
    ),
}
# Experiment: a miss-aware CPU split. A node with 1..3 forced CPU misses gives one of its split's lanes back to the
# copy engine or GPU, since its misses queue behind the CPU-hit job (analysis/dsv41-drive/dspark/miss_timeline.py).
ARMS["dspark-both-misscut"] = (
    {**ARMS["dspark-both"][0], "SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "1", "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3"},
    True,
)
# The cut arm with a two-stage CPU miss: w13 read first, the CPU's gate/up GEMVs overlapping the w2 read.
ARMS["dspark-both-misscut-twostage"] = (
    {**ARMS["dspark-both-misscut"][0], "SGLANG_DSV41_CPU_TWO_STAGE": "1"},
    True,
)
# The two-stage arm with the SPCC (nvme4, the middle root) on w13 alone, so it reads one contiguous piece a row: w13 at
# 6:10:6 and w2 on the Samsungs, 1:0:1. Its 10/22 of w13 (~4.0 MB) lands about when the Samsungs finish their w13
# and w2 (~4.6 MB each), so w13 + GateUp ends about with the whole row.
ARMS["dspark-both-misscut-twostage-spcc13"] = (
    {
        **ARMS["dspark-both-misscut-twostage"][0],
        "SGLANG_DSV41_CPU_TWO_STAGE_W13_WEIGHTS": "6:10:6",
        "SGLANG_DSV41_CPU_TWO_STAGE_W2_WEIGHTS": "1:0:1",
    },
    True,
)
# The cut arm with each row's parts sized by drive speed (fio QD1: 3.35 / 2.19 / 3.38 GB/s, DSV41_REFERENCE.md 33.16).
ARMS["dspark-both-weights-misscut"] = (
    {**ARMS["dspark-both-misscut"][0], "SGLANG_MOE_EXPERT_MIRROR_WEIGHTS": "16:10:16"},
    True,
)
# The cut arm with 4 tokens per CPU chunk (CHUNK_M 4, not 2): an expert shared by 3-4 verify tokens decodes its
# weights once, not twice (bench: 4 tokens -25%, 6 tokens -10%, 1-2 unchanged; DSV41_REFERENCE.md 33.20).
ARMS["dspark-both-misscut-maxm8"] = (
    {**ARMS["dspark-both-misscut"][0], "SGLANG_EXL3_CPU_MAX_M": "8"},
    True,
)
# The same, picking only among each token's predicted top_k (DSV41_REFERENCE.md 33.14's unused reads).
ARMS["dspark-both-prefetch-topk"] = (
    {**ARMS["dspark-both-prefetch"][0], "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "1"},
    True,
)
# The same arm scored on the GPU after each layer's post (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design).
ARMS["dspark-both-prefetch-gpu"] = (
    {**ARMS["dspark-both-prefetch"][0], "SGLANG_DSV41_RAM_PREFETCH_SCORER": "gpu"},
    True,
)
ARMS["dspark-both-prefetch-gpu-topk"] = (
    {**ARMS["dspark-both-prefetch-gpu"][0], "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "1"},
    True,
)
# The GPU arm reading only the candidates at or over their target row's margin floor.
ARMS["dspark-both-prefetch-gpu-floors"] = (
    {**ARMS["dspark-both-prefetch-gpu"][0], "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": MARGIN_FLOORS},
    True,
)
# The GPU-prefetch arm with the miss-aware split cut on: does the prefetch add to the cut.
ARMS["dspark-both-prefetch-gpu-misscut"] = (
    {**ARMS["dspark-both-prefetch-gpu"][0], "SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "1", "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3"},
    True,
)
# The dynamic mirror root choice (design 2026-10-09-dsv41-drive-aware-reads, change (3)) is off in every arm but the
# cap arm, pinned so a shell that exports it cannot carry it into one.
MIRROR_STATIC = {"SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": "0", "SGLANG_MOE_EXPERT_MIRROR_CAPS": ""}
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm.update(MIRROR_STATIC)
# The GPU arm with a per-drive in-flight cap: each sub-read picks its root late, skipping a root with that many
# sub-reads in flight. Roots in arm_env.EXPERT_MIRROR_DIRS order: nvme0, nvme4 (the SPCC, DSV41_REFERENCE.md 33.16),
# nvme2.
ARMS["dspark-both-prefetch-gpu-cap"] = (
    {
        **ARMS["dspark-both-prefetch-gpu"][0],
        "SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": "1",
        "SGLANG_MOE_EXPERT_MIRROR_CAPS": "4,2,4",
    },
    True,
)
# Speculative reads only to an idle drive (design 2026-10-09-dsv41-drive-aware-reads, change (2)) are off in every arm
# but the idle arm, deadline included, pinned so a shell that exports them cannot carry them into one.
IDLE_DRIVE_OFF = {
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE": "0",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US": "4000",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD": "0",
    "SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES": "1",
}
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm.update(IDLE_DRIVE_OFF)
# Promoting landed speculative rows into the tier map is off in every arm but the promote one, pinned so a shell that
# exports it cannot carry it into one.
PROMOTE = "SGLANG_DSV41_RAM_PREFETCH_PROMOTE"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[PROMOTE] = "0"
# Serving GPU misses from landed pool rows is off in every arm but the pool-serve one, likewise pinned.
POOL_SERVE = "SGLANG_DSV41_RAM_PREFETCH_POOL_SERVE"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[POOL_SERVE] = "0"
# The prefetch scores the gate of layer T+horizon on layer T's hidden state; every arm but the L+2 one states 1, so a
# shell that exports it cannot carry it into one.
HORIZON_DEFAULT = {"SGLANG_DSV41_RAM_PREFETCH_HORIZON": "1"}
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm.update(HORIZON_DEFAULT)
# The learned L+2 head file is unset in every arm but the T3 one, so a shell that exports it cannot carry it into one.
LEARNED_HEAD_OFF = {"SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD": ""}
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm.update(LEARNED_HEAD_OFF)
# The GPU-prefetch miss-cut arm with each group's speculative reads on a reader of its own, one piece at a time from a
# drive no demand reads (DSV41_REFERENCE.md 33.16 item 2: demand reads were 0.4-0.5 ms slower beside them).
ARMS["dspark-both-prefetch-gpu-misscut-idle"] = (
    {**ARMS["dspark-both-prefetch-gpu-misscut"][0], "SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE": "1"},
    True,
)
# The idle arm with each row's pieces spread over the idle drives and two in flight (DSV41_REFERENCE.md 33.18: one
# drive took most speculative bytes, and one piece at a time abandoned 35% of rows).
ARMS["dspark-both-prefetch-gpu-misscut-idle2"] = (
    {
        **ARMS["dspark-both-prefetch-gpu-misscut-idle"][0],
        "SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD": "1",
        "SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES": "2",
    },
    True,
)
# idle2 with the packaged margin floors (DSV41_REFERENCE.md 33.22): the floors halved the wasted reads but saved nothing
# while their reads fought demand reads; idle2 keeps speculative reads off a busy drive.
ARMS["dspark-both-prefetch-gpu-misscut-idle2-floors"] = (
    {**ARMS["dspark-both-prefetch-gpu-misscut-idle2"][0], "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": MARGIN_FLOORS},
    True,
)
# idle2 with the gate of layer T+2 scored on layer T's hidden state and no floors (the packaged ones were fit at L+1).
ARMS["dspark-both-prefetch-gpu-misscut-idle2-l2"] = (
    {
        **ARMS["dspark-both-prefetch-gpu-misscut-idle2"][0],
        "SGLANG_DSV41_RAM_PREFETCH_HORIZON": "2",
        "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS": "",
    },
    True,
)
# The folded T3 head exported by l2_head_export.py with the F109 calib threshold at budget 0.75.
T3_HEAD = "/data/models/slang/nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors"
# The L+2 arm with the learned T3 scorer served on the prefetch side stream; the head file is the only change.
ARMS["dspark-both-prefetch-gpu-misscut-idle2-l2-t3"] = (
    {**ARMS["dspark-both-prefetch-gpu-misscut-idle2-l2"][0], "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD": T3_HEAD},
    True,
)
# The L+2 and T3 arms at two speculative rows per layer (DSV41_REFERENCE.md 33.33: cap 1 covers ~14 of ~88 misses per
# verify); T3 keeps its cap-1 calib threshold, so its second read is any block at or above it.
for _cap1 in ("dspark-both-prefetch-gpu-misscut-idle2-l2", "dspark-both-prefetch-gpu-misscut-idle2-l2-t3"):
    ARMS[f"{_cap1}-cap2"] = ({**ARMS[_cap1][0], "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "2"}, True)
# T3 at three rows per layer, with three pool slots per row and group: with two, a layer's third read that lands in the
# same group as its other two finds no claimable slot (a job never reclaims its own pick) and is dropped.
ARMS["dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap3"] = (
    {
        **ARMS["dspark-both-prefetch-gpu-misscut-idle2-l2-t3"][0],
        "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "3",
        "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": "3",
    },
    True,
)
# The promote pair: the T3 cap-2 arm with landed rows promoted into the target layer's map, and its twin with the flag
# stated off (identical otherwise). The off twin is the on arm's reference, so the A/B reads the flag alone.
PROMOTE_OFF_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote-off"
PROMOTE_ON_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote"
ARMS[PROMOTE_OFF_ARM] = ({**ARMS["dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2"][0], PROMOTE: "0"}, True)
ARMS[PROMOTE_ON_ARM] = ({**ARMS[PROMOTE_OFF_ARM][0], PROMOTE: "1"}, True)
# The promote-off arm with GPU misses on landed pool rows copied from the pool instead of read; its reference is the
# off arm, so the A/B reads the flag alone.
POOL_SERVE_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-poolserve"
ARMS[POOL_SERVE_ARM] = ({**ARMS[PROMOTE_OFF_ARM][0], POOL_SERVE: "1"}, True)
# Both: landed rows promoted ahead of their record, and a GPU miss on a landed row the promotion left copied from the
# pool; its reference is the promote-on arm, so the A/B reads pool serve alone.
PROMOTE_POOL_SERVE_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote-poolserve"
ARMS[PROMOTE_POOL_SERVE_ARM] = ({**ARMS[PROMOTE_ON_ARM][0], POOL_SERVE: "1"}, True)
# The pair's fixed-split option: the CPU split is the off capture's table (the driver's SGLANG_DSV41_CPU_EXPERTS_SPLIT)
# with calibration off, so both arms split alike and the A/B does not read calibration drift.
FIXED_SPLIT_VAR = "SGLANG_DSV41_CPU_EXPERTS_SPLIT"
FIXED_SPLIT_CALIBRATION_OFF = {"SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION": "0"}
FIXED_SPLIT_SUFFIX = "-fixedsplit"
for _pair_arm in (PROMOTE_OFF_ARM, PROMOTE_ON_ARM, POOL_SERVE_ARM, PROMOTE_POOL_SERVE_ARM):
    ARMS[_pair_arm + FIXED_SPLIT_SUFFIX] = ({**ARMS[_pair_arm][0], **FIXED_SPLIT_CALIBRATION_OFF}, True)
# Both fixed-split with node 1's CPU expert team on 13 physical cores instead of 10: 32-34 are the node's cores no
# server, RAM, spec or driver thread uses (arm_env.DRIVER_CORES 28-31). Node 0's CPU cores are stated as derived (6-15),
# so the THREADS cap of 13 changes node 1 alone; its RAM core stays SPIN_CORE's (naming it here too is refused).
ARMS[PROMOTE_POOL_SERVE_ARM + "-cores13" + FIXED_SPLIT_SUFFIX] = (
    {
        **ARMS[PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX][0],
        "SGLANG_DSV41_CPU_EXPERTS_THREADS": "13",
        "SGLANG_EXPERT_NUMA_CORES": "0:cpu=6-15;1:ram=35,cpu=18-27,32-34",
    },
    True,
)
# DSpark's top-probability verify gate is off in every arm but the gate one, pinned so a shell that exports it cannot
# carry it into one.
TOP_PROB_GATE = "SGLANG_DSPARK_TOP_PROB_GATE"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[TOP_PROB_GATE] = "0"
# The fixed-split promote + pool-serve arm with the gate at the offline replay's best threshold: a verify stops at the
# first draft where the running product of the draft's top-1 probabilities falls below 0.4. Its reference is that arm,
# so the A/B reads the gate alone; it takes the driver's fixed split as its reference does.
TOP_PROB_GATE_ARM = PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX + "-confgate"
ARMS[TOP_PROB_GATE_ARM] = ({**ARMS[PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX][0], TOP_PROB_GATE: "0.4"}, True)
# The GPU drip (host/gpu_drip.h) is off in every arm but the drip one, pinned so a shell that exports it cannot carry it
# into one.
GPU_DRIP = "SGLANG_DSV41_GPU_DRIP_SLOTS"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[GPU_DRIP] = "0"
# The promote and pool-serve fixed-split arm with the T3 scorer's best RAM-resident, not VRAM-hot experts of layer +2
# dripped into VRAM scratch: the replay's 4 picks per scored record at its horizon 2, 24 slots (~320 MB; 12 left
# 2396 picks without a slot in the 2026-10-14 capture) and 256 KiB chunks (~19 us of link a pull or post can wait
# behind); its reference is that arm, so the A/B reads the drip alone.
GPU_DRIP_ARM = PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX + "-gpudrip"
ARMS[GPU_DRIP_ARM] = (
    {
        **ARMS[PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX][0],
        GPU_DRIP: "24",
        "SGLANG_DSV41_GPU_DRIP_PICKS": "4",
        "SGLANG_DSV41_GPU_DRIP_CHUNK_KB": "256",
    },
    True,
)
# The drip arm moving about half the bytes: the top 2 picks (4 used about 48% of landed rows) in 64 KiB chunks
# (about 5 us of link a pull or post can wait behind); same reference.
GPU_DRIP_LEAN_ARM = GPU_DRIP_ARM + "-c64p2"
ARMS[GPU_DRIP_LEAN_ARM] = (
    {**ARMS[GPU_DRIP_ARM][0], "SGLANG_DSV41_GPU_DRIP_PICKS": "2", "SGLANG_DSV41_GPU_DRIP_CHUNK_KB": "64"},
    True,
)
# Whole layers held in the pinned tier and per-layer CPU split tables are off in every arm but the pin ones, pinned so
# a shell that exports them cannot carry them into one.
FULL_LAYERS = "SGLANG_MOE_PINNED_HOST_FULL_LAYERS"
SPLIT_LAYERS = "SGLANG_DSV41_CPU_EXPERTS_SPLIT_LAYERS"
# The model-based split (and its cost overrides) likewise: off in every arm but -splitmodel.
SPLIT_MODEL = "SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL"
SPLIT_MODEL_COSTS = "SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS"
# The scheduler's mlockall, off in every arm but the -mem ones.
MLOCK = "SGLANG_ENABLE_SCHEDULER_MLOCK"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[FULL_LAYERS] = ""
    _overrides_of_arm[SPLIT_LAYERS] = ""
    _overrides_of_arm[SPLIT_MODEL] = "0"
    _overrides_of_arm[SPLIT_MODEL_COSTS] = ""
    _overrides_of_arm[MLOCK] = "0"
# Layers 0 and 1 read 6.6 and 4.5 experts a verify from NVMe, which the L+2 prefetch cannot reach
# (ram-prefetch/breakdown-20261015). Each full layer takes 204 rows a node (192 homed experts, 8 staging, 2 pool, 2
# seam), 5182 MiB a node for the pair; the other 7:6 remainder buys the 7966 rows layers 2-39 hold today (210/209).
PIN01_BUDGET = {"SGLANG_MOE_PINNED_HOST_MB": "111530", "SGLANG_MOE_PINNED_HOST_NUMA_MB": "0:59656,1:51874"}
PIN01_ARM = PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX + "-pin01"
ARMS[PIN01_ARM] = ({**ARMS[PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX][0], FULL_LAYERS: "0,1", **PIN01_BUDGET}, True)
# k*(n) with each group's forced CPU lanes counted (8.4 at layer 0, 4.9 at layer 1, the spilled lanes above), at 0.55
# ms a CPU lane and 1.1 ms a link lane shared by both groups, capped at the fixed split A: only lanes move to the link.
# At the 2-4 eligible lanes a group sees, layer 0 moves 1-2 a group and layer 1 at most 1.
PIN01_SPLIT_L0 = "0,0,0,1,2,3,3,4,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,20,21,21,22,23,23,24,24,25,26,26"
PIN01_SPLIT_L1 = "0,0,1,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,20,21,21,22,23,23,24,24,25,26,26"
PIN01_GPU01_ARM = PIN01_ARM + "-gpu01"
ARMS[PIN01_GPU01_ARM] = ({**ARMS[PIN01_ARM][0], SPLIT_LAYERS: f"0:{PIN01_SPLIT_L0};1:{PIN01_SPLIT_L1}"}, True)
# pin01 with the model-based split in place of split A and the miss cut (misscut-replay, split_replay.py: the
# token-weighted greedy, +17 ms a verify on breakdown-20261015/base at untraced CPU costs), at its default costs.
PIN01_SPLIT_MODEL_ARM = PIN01_ARM + "-splitmodel"
ARMS[PIN01_SPLIT_MODEL_ARM] = ({**ARMS[PIN01_ARM][0], SPLIT_MODEL: "1"}, True)
# Node-0 headroom (cores-ab-20261014/splitmodel-stream swapped node 0 in load and decode): -mem adds the scheduler's
# mlockall and drops each safetensors shard's page cache after loading it (server_flags), and nothing else.
MEM_SUFFIX = "-mem"
MEM_SERVER_FLAGS = ("--weight-loader-drop-cache-after-load",)
for _base in (PIN01_ARM, PIN01_SPLIT_MODEL_ARM):
    ARMS[_base + MEM_SUFFIX] = ({**ARMS[_base][0], MLOCK: "1"}, True)
# Node-0 headroom, step 2: the HiCache host KV pools (12.1 GB, node 0 by first touch) bound to node 1, then also the
# native Engram row cache off (5 GiB). The node is an EnvInt with no valid "unset" value, so only these arms name it.
HICACHE_NODE = "SGLANG_HICACHE_HOST_NUMA_NODE"
ENGRAM_NATIVE_GIB = "SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB"
HOT_DECAY = "SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY"
RESIDENCY_TIEBREAK = "SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[ENGRAM_NATIVE_GIB] = "5"
    _overrides_of_arm[HOT_DECAY] = "0.98"
    _overrides_of_arm[RESIDENCY_TIEBREAK] = "0"
PIN01_SPLIT_MODEL_HICACHE1_ARM = PIN01_SPLIT_MODEL_ARM + MEM_SUFFIX + "-hicache1"
ARMS[PIN01_SPLIT_MODEL_HICACHE1_ARM] = ({**ARMS[PIN01_SPLIT_MODEL_ARM + MEM_SUFFIX][0], HICACHE_NODE: "1"}, True)
PIN01_SPLIT_MODEL_ENGRAM0_ARM = PIN01_SPLIT_MODEL_HICACHE1_ARM + "-engram0"
ARMS[PIN01_SPLIT_MODEL_ENGRAM0_ARM] = ({**ARMS[PIN01_SPLIT_MODEL_HICACHE1_ARM][0], ENGRAM_NATIVE_GIB: "0"}, True)
# The hot cache's insert-on-miss decay at 0.995 (every other arm 0.98, the default): the cache-tiers replay puts it at
# +5.5 VRAM hits a verify, so the routed-pair hit rate is the evidence and ms/token only a regression check.
PIN01_SPLIT_MODEL_DECAY_ARM = PIN01_SPLIT_MODEL_HICACHE1_ARM + "-decay995"
ARMS[PIN01_SPLIT_MODEL_DECAY_ARM] = ({**ARMS[PIN01_SPLIT_MODEL_HICACHE1_ARM][0], HOT_DECAY: "0.995"}, True)
# The split model's residency tiebreak (equal-token lanes of a node: the CPU takes the lowest residency keys): the
# breakdown-base replay gives the same saving with 56% fewer inverted pairs; compared with -hicache1.
PIN01_SPLIT_MODEL_TIEBREAK_ARM = PIN01_SPLIT_MODEL_HICACHE1_ARM + "-tiebreak"
ARMS[PIN01_SPLIT_MODEL_TIEBREAK_ARM] = ({**ARMS[PIN01_SPLIT_MODEL_HICACHE1_ARM][0], RESIDENCY_TIEBREAK: "1"}, True)
# Since split-model-calib the model's costs are calibrated at startup, so every arm above with the model on now runs
# calibrated. -fixedcosts names every cost, the 2026-10-15 offline fits (ram_slot_map.SPLIT_MODEL_FIXTURE_MS), so it
# skips the calibration and types as -hicache1 did before it; -refine is the calibrated model with the live refinement
# of its CPU costs, compared with -fixedcosts.
SPLIT_MODEL_REFINE = "SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES"
SPLIT_MODEL_FIXED_COSTS = (
    "hit_a.0=0.016,hit_a.1=-0.050,hit_b.0=0.154,hit_b.1=0.179,hit_c.0=0.395,hit_c.1=0.390,"
    "miss_a.0=0.331,miss_a.1=0.291,miss_c.0=0.193,miss_c.1=0.245,"
    "copy_lane=1.05,first_byte=0.8,per_expert=1.6,stream_lane=0.65,stream_base=0"
)
REFINE_BATCHES = "50"
for _overrides_of_arm, _ in ARMS.values():
    _overrides_of_arm[SPLIT_MODEL_REFINE] = "0"
PIN01_SPLIT_MODEL_FIXED_COSTS_ARM = PIN01_SPLIT_MODEL_HICACHE1_ARM + "-fixedcosts"
ARMS[PIN01_SPLIT_MODEL_FIXED_COSTS_ARM] = (
    {**ARMS[PIN01_SPLIT_MODEL_HICACHE1_ARM][0], SPLIT_MODEL_COSTS: SPLIT_MODEL_FIXED_COSTS}, True
)
PIN01_SPLIT_MODEL_REFINE_ARM = PIN01_SPLIT_MODEL_HICACHE1_ARM + "-refine"
ARMS[PIN01_SPLIT_MODEL_REFINE_ARM] = (
    {**ARMS[PIN01_SPLIT_MODEL_HICACHE1_ARM][0], SPLIT_MODEL_REFINE: REFINE_BATCHES}, True
)
# The miss-cut arm with the verify's Engram lookups captured in the graph (device wait), so the verify graph has no
# breaks; prefetch stays off so the A/B reads the two Engram breaks alone (DSV41_REFERENCE.md 33.35). The flag is on by
# default since then, so this arm now equals its A unless the shell exports the flag as 0.
ARMS["dspark-both-misscut-engram-verify-graph"] = (
    {**ARMS["dspark-both-misscut"][0], "SGLANG_DSV41_ENGRAM_VERIFY_IN_GRAPH": "1"},
    True,
)
# Build caches an experiment keeps private (run protocol): passed to every arm's server when set in the driver's env.
PASSTHROUGH = ("SGLANG_JIT_CACHE_DIR", "SGLANG_EXL3_BUILD_DIR")
# An arm whose outputs are also compared with its A's, not only with prod's.
REFERENCE = {
    "dspark-both-prefetch": "dspark-both",
    "dspark-both-prefetch-topk": "dspark-both",
    "dspark-both-prefetch-gpu": "dspark-both",
    "dspark-both-prefetch-gpu-topk": "dspark-both",
    "dspark-both-prefetch-gpu-floors": "dspark-both",
    "dspark-both-misscut": "dspark-both",
    "dspark-both-weights-misscut": "dspark-both-misscut",
    "dspark-both-misscut-maxm8": "dspark-both-misscut",
    "dspark-both-misscut-twostage": "dspark-both-misscut",
    "dspark-both-misscut-twostage-spcc13": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut": "dspark-both",
    "dspark-both-prefetch-gpu-cap": "dspark-both",
    "dspark-both-prefetch-gpu-misscut-idle": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2-floors": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2-l2": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-t3": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-cap2": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2": "dspark-both-misscut",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap3": "dspark-both-misscut",
    "dspark-both-misscut-engram-verify-graph": "dspark-both-misscut",
    PROMOTE_OFF_ARM: "dspark-both-misscut",
    PROMOTE_ON_ARM: PROMOTE_OFF_ARM,
    PROMOTE_OFF_ARM + FIXED_SPLIT_SUFFIX: "dspark-both-misscut",
    PROMOTE_ON_ARM + FIXED_SPLIT_SUFFIX: PROMOTE_OFF_ARM + FIXED_SPLIT_SUFFIX,
    POOL_SERVE_ARM: PROMOTE_OFF_ARM,
    POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX: PROMOTE_OFF_ARM + FIXED_SPLIT_SUFFIX,
    PROMOTE_POOL_SERVE_ARM: PROMOTE_ON_ARM,
    PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX: PROMOTE_ON_ARM + FIXED_SPLIT_SUFFIX,
    TOP_PROB_GATE_ARM: PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX,
    GPU_DRIP_ARM: PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX,
    GPU_DRIP_LEAN_ARM: PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX,
    PIN01_ARM: PROMOTE_POOL_SERVE_ARM + FIXED_SPLIT_SUFFIX,
    PIN01_GPU01_ARM: PIN01_ARM,
    PIN01_SPLIT_MODEL_ARM: PIN01_ARM,
    PIN01_ARM + MEM_SUFFIX: PIN01_ARM,
    PIN01_SPLIT_MODEL_ARM + MEM_SUFFIX: PIN01_ARM + MEM_SUFFIX,
    PIN01_SPLIT_MODEL_HICACHE1_ARM: PIN01_SPLIT_MODEL_ARM + MEM_SUFFIX,
    PIN01_SPLIT_MODEL_ENGRAM0_ARM: PIN01_SPLIT_MODEL_HICACHE1_ARM,
    PIN01_SPLIT_MODEL_DECAY_ARM: PIN01_SPLIT_MODEL_HICACHE1_ARM,
    PIN01_SPLIT_MODEL_TIEBREAK_ARM: PIN01_SPLIT_MODEL_HICACHE1_ARM,
    PIN01_SPLIT_MODEL_FIXED_COSTS_ARM: PIN01_SPLIT_MODEL_HICACHE1_ARM,
    PIN01_SPLIT_MODEL_REFINE_ARM: PIN01_SPLIT_MODEL_FIXED_COSTS_ARM,
}


def server_flags(arm: str) -> tuple[str, ...]:
    """Server flags an arm adds to the launch argv (its env overrides cannot carry them): -mem's page-cache drop, for
    the -mem arms and those built on them."""
    return MEM_SERVER_FLAGS if MEM_SUFFIX + "-" in arm + "-" else ()


DEFAULT_SESSIONS = ",".join(str(i) for i in range(8))
COUNTER_MARKER = "exl3 RAM miss thread counters "
RAM_KEYS = (
    "rows_read",
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
    "spec_late",
    "spec_deferred",
    "spec_abandoned",
    "spec_boosted",
    "spec_moved_root",
    "spec_promoted_early",
    "spec_promote_no_delta",
    "spec_promote_no_room",
    "spec_promote_no_victim",
    "spec_promote_victim_routed",
    "spec_promote_used",
    "spec_promote_reading",
    "spec_publish_backstop",
    "spec_promote_row_busy",
    "spec_pool_served",
    "drip_issued",
    "drip_landed",
    "drip_used",
    "drip_abandoned",
    "drip_no_slot",
    "drip_paused",
    "drip_skipped_bytes",
    "drip_late",
)


# Every arm but the two-stage ones states it off and its span weights unset, so a shell that exports them cannot carry
# them into an arm.
TWO_STAGE_OFF = {
    "SGLANG_DSV41_CPU_TWO_STAGE": "0",
    "SGLANG_DSV41_CPU_TWO_STAGE_W13_WEIGHTS": "",
    "SGLANG_DSV41_CPU_TWO_STAGE_W2_WEIGHTS": "",
}


def _takes_fixed_split(arm: str) -> bool:
    return FIXED_SPLIT_SUFFIX in arm  # a -fixedsplit arm or one built on it (-confgate, -gpudrip)


def _overrides(arm: str, out: str) -> dict:
    overrides, _ = ARMS[arm]
    passed = {k: os.environ[k] for k in PASSTHROUGH if os.environ.get(k)}
    if _takes_fixed_split(arm) and os.environ.get(FIXED_SPLIT_VAR):
        passed[FIXED_SPLIT_VAR] = os.environ[FIXED_SPLIT_VAR]
    return {
        **TWO_STAGE_OFF, **overrides, **passed, "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.metrics.jsonl")
    }


def _probe_overrides(arm: str, out: str) -> dict:
    """The arm's overrides for the probe server, with a metrics file of its own: the timed server's file is appended to
    and summarize() reads its last record, which a second server would replace."""
    return {**_overrides(arm, out), "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.probe.metrics.jsonl")}


def _health_timeout_s(dspark: bool) -> int:
    return arm_env.DSPARK_HEALTH_TIMEOUT_S if dspark else arm_env.HEALTH_TIMEOUT_S


def run_timed(arm: str, out: str, sessions: str = DEFAULT_SESSIONS) -> int:
    _, dspark = ARMS[arm]
    env = os.environ | {
        "DSV41_RUN_ROOT": out,
        "DSV41_SESSION_INDICES": sessions,
        "DSV41_EXTRA_SERVER_ARGS": shlex.join((*(arm_env.DSPARK_ARGV if dspark else ()), *server_flags(arm))),
        "DSV41_HEALTH_TIMEOUT_S": str(_health_timeout_s(dspark)),
    }
    if dspark:
        # run_arm.sh builds its argv without dspark=True, so the recipe's fraction reaches it as the arm override.
        env["DSV41_MEM_FRACTION_STATIC"] = arm_env.DSPARK_MEM_FRACTION_STATIC
    cmd = [os.path.join(REPO, "benchmarks", "dsv41_baseline", "run_arm.sh"), arm, str(PORT)]
    cmd += [f"{k}={v}" for k, v in _overrides(arm, out).items()]
    print("===", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, cwd=REPO).returncode


def _healthy(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def run_probe(arm: str, out: str) -> int:
    """The arm's server on PROBE_PORT under cc-gpu.lock, for logprob_probe.py only."""
    _, dspark = ARMS[arm]
    env = os.environ | arm_env.arm_env(_probe_overrides(arm, out)) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PROBE_PORT, dspark=dspark).argv()
    with run_locks.take(arm_env.GPU_LOCK):
        with open(os.path.join(out, f"{arm}.probe-server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        try:
            if not _healthy(PROBE_PORT, time.monotonic() + _health_timeout_s(dspark)):
                return 1
            probe = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")
            return subprocess.run(
                ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, probe, "--port", str(PROBE_PORT),
                 "--sessions", SESSIONS, "--prompts", "8", "--max-tokens", "128", "--top-logprobs", "5",
                 "--out", os.path.join(out, f"{arm}.probe.json")], cwd=REPO,
            ).returncode
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def _results(out: str, arm: str) -> list[dict]:
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    with open(os.path.join(out, "servers", arm, runs[-1], "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ms_by_session(out: str, arm: str) -> dict:
    return {
        r["session_id"]: 1000.0 / r["decode_tokens_per_sec"]
        for r in _results(out, arm)
        if r.get("session_id") is not None and r.get("decode_tokens_per_sec")
    }


def _paired_gain_pct(out: str, reference: str, arm: str):
    """Per session (by session_id), the arm's ms/token gain over its reference's, in % of the reference's; None when
    the two share no session."""
    try:
        a, b = _ms_by_session(out, reference), _ms_by_session(out, arm)
    except FileNotFoundError:
        return None
    return {k: 100.0 * (a[k] - b[k]) / a[k] for k in sorted(set(a) & set(b))} or None


def _server_counters(out: str, arm: str):
    """The last counters line of the arm's timed server: the service's lifetime (warm-up and the timed set)."""
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    path = os.path.join(out, "servers", arm, runs[-1], "server.log")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        lines = [line for line in f if COUNTER_MARKER in line]
    return json.loads(lines[-1].split(COUNTER_MARKER, 1)[1]) if lines else None


def _warmup_tokens(out: str, arm: str) -> list[int]:
    """Completion tokens of each warm-up round the arm's timed server ran (run_arm.sh's results-warmup-N.jsonl)."""
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    run = os.path.join(out, "servers", arm, runs[-1])
    tokens = []
    for name in sorted(os.listdir(run)):
        if name.startswith("results-warmup-") and name.endswith(".jsonl"):
            with open(os.path.join(run, name)) as f:
                tokens.append(sum(json.loads(line).get("completion_tokens") or 0 for line in f if line.strip()))
    return tokens


def _drive_load(out: str, arm: str) -> dict:
    """The arm's per-session drive load from its timed run's boundary samples (drive_sessions.py)."""
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    path = os.path.join(out, "servers", arm, runs[-1], "boundary-samples.jsonl")
    if not os.path.exists(path):
        return {"warmup": None, "sessions": [], "notes": [f"no {path}"]}
    with open(path) as f:
        return drive_sessions.session_drive_load([json.loads(line) for line in f if line.strip()])


def _compare(base_path: str, other_path: str):
    sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
    import dspark_text_band

    with open(base_path) as f:
        base = json.load(f)
    with open(other_path) as f:
        other = json.load(f)
    try:
        return dspark_text_band.compare(base, other)
    except ValueError as error:
        return {"error": str(error)}


def summarize(out: str) -> dict:
    summary = {}
    for arm in ARMS:
        try:
            rows = _results(out, arm)
        except FileNotFoundError:
            continue
        ms = [1000.0 / r["decode_tokens_per_sec"] for r in rows if r.get("decode_tokens_per_sec")]
        spec = [r.get("spec_tokens_details") or {} for r in rows]
        verifies = sum(s.get("spec_verify_ct", 0) for s in spec)
        entry = {
            "sessions": len(rows),
            "ms_per_token_median": statistics.median(ms) if ms else None,
            "ms_per_token": ms,
            "accept_length": (sum(r["completion_tokens"] for r in rows) / verifies) if verifies else None,
        }
        metrics = os.path.join(out, f"{arm}.metrics.jsonl")
        if os.path.exists(metrics):
            with open(metrics) as f:
                last = json.loads([line for line in f if line.strip()][-1])
            graphed = last.get("counters", {}).get("graphed_verify")
            if graphed and graphed.get("graphed_verify_ct"):
                entry["reverify_rate"] = graphed["verify_overflow_ct"] / graphed["graphed_verify_ct"]
                entry["reverify_ct"] = graphed["verify_overflow_ct"]
        probes = {a: os.path.join(out, f"{a}.probe.json") for a in ("prod", arm)}
        if arm != "prod" and all(os.path.exists(p) for p in probes.values()):
            entry["text_vs_prod"] = _compare(probes["prod"], probes[arm])
        counters = _server_counters(out, arm)
        if counters:
            entry["ram"] = {k: counters.get(k) for k in RAM_KEYS}
            entry["ram"]["scope"] = "server lifetime: warm-up and the timed set"
            if counters.get("spec_used"):
                # Used reads still in flight at their demand (the GPU scorer's success criterion: under 15%).
                entry["spec_in_flight_at_use"] = counters.get("spec_promoted", 0) / counters["spec_used"]
            # The counters are server-lifetime, so the denominator is too: warm-up rounds plus the timed set.
            warmups = _warmup_tokens(out, arm)
            lifetime = sum(warmups) + sum(r["completion_tokens"] for r in rows)
            entry["warmup_rounds"] = len(warmups)
            entry["lifetime_tokens"] = lifetime
            entry["ram_rows_per_token_lifetime"] = counters["rows_read"] / lifetime if lifetime else None
            # Drive load next to the saving: every NVMe row, and the speculative rows read but never used.
            if lifetime and "spec_issued" in counters:
                entry["nvme_rows_per_token_lifetime"] = (counters["rows_read"] + counters["spec_issued"]) / lifetime
            if lifetime and "spec_landed" in counters and "spec_used" in counters:
                entry["spec_wasted_per_token_lifetime"] = (counters["spec_landed"] - counters["spec_used"]) / lifetime
        reference = REFERENCE.get(arm)
        ref_probes = {a: os.path.join(out, f"{a}.probe.json") for a in (reference, arm)} if reference else {}
        if reference and not os.path.exists(ref_probes[reference]):
            entry["text_vs_reference"] = "missing reference probe"
        elif reference and os.path.exists(ref_probes[arm]):
            entry["text_vs_reference"] = _compare(ref_probes[reference], ref_probes[arm])
        if reference:
            paired = _paired_gain_pct(out, reference, arm)
            if paired:
                entry["paired_gain_pct_vs_reference"] = paired
                entry["paired_gain_pct_median"] = statistics.median(paired.values())
        entry["drives_per_session"] = _drive_load(out, arm)
        summary[arm] = entry
    # Flagged against each drive's median over every arm's sessions, so a drive slow in one arm stands out.
    flags = drive_sessions.flag_sessions({arm: entry["drives_per_session"] for arm, entry in summary.items()})
    for arm, entry in summary.items():
        entry["drive_flags"] = flags[arm]
    baseline = os.path.join(out, "baseline.json")
    if os.path.exists(baseline):
        with open(baseline) as f:
            provenance = json.load(f)
        if provenance["arm"] in summary:
            summary[provenance["arm"]]["baseline_from"] = {"dir": provenance["dir"], "commit": provenance["commit"]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def import_baseline(source: str, arm: str, out: str) -> None:
    """Copies `arm`'s timed run, probe and metrics from an earlier A/B's directory into `out`, so the arms that name it
    as their reference compare against it without rerunning it, and records where it came from in baseline.json. The
    baseline is no longer measured in the same window as the arm: drift between the two runs reads as the arm's."""
    needed = [os.path.join("servers", arm), f"{arm}.probe.json", f"{arm}.metrics.jsonl"]
    missing = [name for name in needed if not os.path.exists(os.path.join(source, name))]
    if missing:
        sys.exit(f"baseline {source}: missing {', '.join(missing)}")
    runs = sorted(os.listdir(os.path.join(source, "servers", arm)))
    if not runs:
        sys.exit(f"baseline {source}: no timed run of {arm}")
    shutil.copytree(os.path.join(source, "servers", arm, runs[-1]), os.path.join(out, "servers", arm, runs[-1]))
    for name in needed[1:]:
        shutil.copy2(os.path.join(source, name), os.path.join(out, name))
    commit = os.path.join(source, "commit.txt")
    with open(commit) if os.path.exists(commit) else open(os.devnull) as f:
        provenance = {"arm": arm, "dir": source, "commit": f.read().strip() or None}
    with open(os.path.join(out, "baseline.json"), "w") as f:
        json.dump(provenance, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Serve A/B arms in the order given, then summarize them.")
    p.add_argument("out")
    p.add_argument("arms", nargs="*")
    p.add_argument("--baseline", help="an earlier A/B's directory whose reference arm is imported, not rerun")
    p.add_argument("--sessions", default=DEFAULT_SESSIONS, help="comma list of corpus session indices to time")
    a = p.parse_args()
    sessions = ",".join(str(int(i)) for i in a.sessions.split(","))
    out = os.path.abspath(a.out)
    arms = a.arms or list(ARMS)
    if not os.environ.get(FIXED_SPLIT_VAR) and any(_takes_fixed_split(arm) for arm in arms):
        sys.exit(f"a {FIXED_SPLIT_SUFFIX} arm needs {FIXED_SPLIT_VAR} set to the off capture's split table")
    if a.baseline:
        references = {REFERENCE[arm] for arm in arms if arm in REFERENCE}
        if len(references) != 1 or references & set(arms):
            sys.exit(f"--baseline imports one reference arm the arms share and none of them reruns; got {references}")
    os.makedirs(out, exist_ok=True)
    if a.baseline:
        import_baseline(os.path.abspath(a.baseline), references.pop(), out)
    for arm in arms:
        for name, step in (
            ("run_timed", lambda: run_timed(arm, out, sessions=sessions)),
            ("run_probe", lambda: run_probe(arm, out)),
        ):
            rc = step()
            print(f"{arm} {name}: rc={rc}", flush=True)
            if rc:
                sys.exit(rc)
    summary = summarize(out)
    loads = {arm: entry["drives_per_session"] for arm, entry in summary.items()}
    print(drive_sessions.format_table(loads, {arm: entry["drive_flags"] for arm, entry in summary.items()}), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
