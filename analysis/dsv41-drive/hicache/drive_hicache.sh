#!/usr/bin/env bash
# The production recipe plus sglang's hierarchical KV cache (host KV pool 2x the device pool, write-through), one
# untraced arm. Production must be stopped.
# Usage: drive_hicache.sh SHA
set -u
WT=/data/models/slang/nvfp4-work/wt-payoff
SHA=$1
export DSV41_EXTRA_SERVER_ARGS="--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 --hicache-write-policy write_through"
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
say() { echo "$(date +%T) $*"; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do say "cc-gpu.lock held; waiting"; sleep 60; done; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
# Disk lock first, then poll the GPU lock that run_arm.sh takes itself.
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
wait_gpu; say "arm hicache"
EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh hicache 30025; say "hicache rc=$?"
say "DRIVER DONE"
