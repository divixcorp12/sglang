#!/usr/bin/env bash
# Traced arm of the production recipe with prefill fills and cast fusion on (003d82fb77): node-mode CUDA trace plus the
# root PCIe GPU-metrics session. Production stays stopped.
set -u
WT=/data/models/slang/nvfp4-work/wt-prod-flags
SHA=003d82fb77b7eda1757cab4c16ecbf7149306651
say() { echo "$(date +%T) $*"; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
while ! flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true; do say "cc-gpu.lock held; waiting"; sleep 60; done
say "arm start"
NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh prod-flags-node 30021; rc=$?
say "arm rc=$rc"
say "DRIVER DONE rc=$rc"
