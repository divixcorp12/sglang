#!/usr/bin/env bash
# Route plan A/B (plan 2026-09-25-dsv41-prefill-route-plan): A = production recipe, B = + the route plan, A then B,
# once each; then one node-mode traced arm of B. Production must be stopped.
# Usage: drive_arms.sh SHA [ab|traced|all]
set -u
WT=/data/models/slang/nvfp4-work/wt-route-plan-arms
SHA=$1
MODE=${2:-all}
FLAG=SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN=1
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
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh route-plan-A 30021; say "A rc=$?"
    wait_gpu; say "arm B"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh route-plan-B 30021 $FLAG; say "B rc=$?"
fi
if [ "$MODE" = traced ] || [ "$MODE" = all ]; then
    wait_gpu; say "arm B traced"
    NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node EXPECT_SHA=$SHA \
        bash benchmarks/dsv41_baseline/run_arm.sh route-plan-B-node 30021 $FLAG; say "B traced rc=$?"
fi
say "DRIVER DONE"
