#!/usr/bin/env bash
# Short node-mode traced A and B (one timed session each, TRACE_SESSION, default index 4: 128 tokens) for attribution only; never read ms/token from these.
# Usage: drive_traced.sh TAG SHA "A_OVERRIDES" "B_OVERRIDES"
set -u
TAG=$1; SHA=$2; A_OV=$3; B_OV=$4
WT=/data/models/slang/nvfp4-work/wt-compute-gap-arms
export DSV41_RUN_ROOT=/mnt/nvme1/compute-transfer-gap NSYS_OUT_DIR=/mnt/nvme1/compute-transfer-gap/nsys
OUT=$DSV41_RUN_ROOT/traced/$TAG
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
mkdir -p $OUT $NSYS_OUT_DIR /mnt/nvme1/nsys-tmp
say() { echo "$(date '+%F %T') $*" | tee -a $OUT/driver.log; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do sleep 60; done; }
arm() {  # arm NAME OVERRIDES
    local name=$1 ov=$2 rc
    while true; do
        wait_gpu; say "traced arm $name overrides=[$ov]"
        DSV41_SESSION_INDICES=${TRACE_SESSION:-4} NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node NSYS_GPU_METRICS=0 \
            EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh $name 30032 $ov \
            SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:45056 > $OUT/arm-$name.log 2>&1
        rc=$?
        grep -q "cc-gpu.lock is held by another GPU job" $OUT/arm-$name.log || break
        say "$name lost the GPU lock race; retrying"; sleep 60
    done
    say "$name rc=$rc report=$(ls -t $NSYS_OUT_DIR/$name-*.nsys-rep 2>/dev/null | grep -v pcie | head -1)"
}
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
arm $TAG-A "$A_OV"
arm $TAG-B "$B_OV"
say "DRIVER DONE"
