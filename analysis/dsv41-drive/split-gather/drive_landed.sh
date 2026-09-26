#!/usr/bin/env bash
# Landed-rows split gather A/B: A = base commit, B = tip, same production recipe (split flag on in base_env), once
# each; then a node-mode traced B. Production must be stopped.
# Usage: drive_landed.sh BASE_SHA TIP_SHA [ab|traced|all]
set -u
WT=/data/models/slang/nvfp4-work/wt-split-landed-arms
REPO=/data/models/slang/sglang
PY=/data/models/slang/.venv/bin/python
BASE=$1; TIP=$2; MODE=${3:-all}
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
say() { echo "$(date +%T) $*"; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do say "cc-gpu.lock held; waiting"; sleep 60; done; }
# run_arm.sh takes cc-gpu.lock non-blocking, so another job can take it between wait_gpu and the launch: retry then.
arm() {
    while true; do
        wait_gpu; say "arm $1"
        env "${@:2}" bash benchmarks/dsv41_baseline/run_arm.sh $1 30021 ${OVERRIDES:-} 2>&1 | tee /mnt/nvme1/split-landed/arm-$1.last
        rc=${PIPESTATUS[0]}
        grep -q "cc-gpu.lock is held by another GPU job" /mnt/nvme1/split-landed/arm-$1.last || break
        say "$1 lost the GPU lock race; retrying"; sleep 60
    done
    say "$1 rc=$rc"
}
git -C $REPO fetch -q origin
[ -d $WT ] || git -C $REPO worktree add -q --detach $WT $BASE
at() {
    git -C $WT checkout -q --detach "$1" || exit 1
    [ "$(git -C $WT rev-parse HEAD)" = "$1" ] || { say "worktree not at $1"; exit 1; }
    cd $WT
    PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)"
    tree=$(git rev-parse HEAD:python)
    (cd benchmarks/dsv41_baseline && $PY -c "import generations; generations.register('$tree', '$2')")
    say "at $(git log -1 --format='%h %s'); python tree $tree registered as $2"
}
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
if [ "$MODE" = ab ] || [ "$MODE" = all ]; then
    at $BASE split-landed-base
    arm split-landed-A EXPECT_SHA=$BASE
    at $TIP split-landed-tip
    arm split-landed-B EXPECT_SHA=$TIP
fi
if [ "$MODE" = traced ] || [ "$MODE" = all ]; then
    at $TIP split-landed-tip
    OVERRIDES=SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:45056 arm split-landed-B-node EXPECT_SHA=$TIP \
        NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node NSYS_GPU_METRICS=0
fi
say "DRIVER DONE"
