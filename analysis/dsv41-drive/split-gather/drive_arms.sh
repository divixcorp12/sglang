#!/usr/bin/env bash
# Split fill gather A/B (plan 2026-09-26-dsv41-split-fill-gather): A = production recipe (route plan on), B = + the
# split, once each; then a node-mode traced B without the root PCIe session. Production must be stopped.
# Usage: drive_arms.sh SHA [ab|traced|all]
set -u
WT=/data/models/slang/nvfp4-work/wt-route-plan-arms
SHA=$1
MODE=${2:-all}
FLAG=SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER=1
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
say() { echo "$(date +%T) $*"; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do say "cc-gpu.lock held; waiting"; sleep 60; done; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
# Disk lock first, then poll the GPU lock that run_arm.sh takes itself.
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
if [ "$MODE" = ab ] || [ "$MODE" = all ]; then
    wait_gpu; say "arm A"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh split-gather-A 30021; say "A rc=$?"
    wait_gpu; say "arm B"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh split-gather-B 30021 $FLAG; say "B rc=$?"
fi
if [ "$MODE" = traced ] || [ "$MODE" = all ]; then
    # No root PCIe session: its nsys stop failed on 2026-09-25 and the orphan held cc-gpu.lock.
    wait_gpu; say "arm B traced"
    NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node NSYS_GPU_METRICS=0 EXPECT_SHA=$SHA \
        bash benchmarks/dsv41_baseline/run_arm.sh split-gather-B-node 30021 $FLAG; say "B traced rc=$?"
fi
say "DRIVER DONE"
