#!/usr/bin/env bash
# Interleaved unprofiled serving A/B on the production recipe: A B B A A B B A, one session pair per A/B pair
# (0,1 / 2,3 / 4,5 / 6,7), then concat_arms.py per arm and paired.py. Production must be stopped.
# Usage: drive_ab.sh TAG SHA "A_OVERRIDES" "B_OVERRIDES"   (overrides: space-separated KEY=VAL, may be empty)
set -u
TAG=$1; SHA=$2; A_OV=$3; B_OV=$4
WT=/data/models/slang/nvfp4-work/wt-compute-gap-arms
REPO=/data/models/slang/sglang
PY=/data/models/slang/.venv/bin/python
export DSV41_RUN_ROOT=/mnt/nvme1/compute-transfer-gap
OUT=$DSV41_RUN_ROOT/ab/$TAG
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
mkdir -p $OUT
say() { echo "$(date '+%F %T') $*" | tee -a $OUT/driver.log; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do sleep 60; done; }
arm() {  # arm NAME SESSIONS OVERRIDES
    local name=$1 sessions=$2 ov=$3 rc
    while true; do
        wait_gpu; say "arm $name sessions=$sessions overrides=[$ov]"
        DSV41_SESSION_INDICES=$sessions EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh $name 30031 $ov \
            > $OUT/arm-$name-$sessions.log 2>&1
        rc=$?
        grep -q "cc-gpu.lock is held by another GPU job" $OUT/arm-$name-$sessions.log || break
        say "$name lost the GPU lock race; retrying"; sleep 60
    done
    local dir; dir=$(ls -dt $DSV41_RUN_ROOT/servers/$name/run-* | head -1)
    say "$name sessions=$sessions rc=$rc dir=$dir"
    echo "$dir" >> $OUT/$name.dirs
    [ $rc -eq 0 ] || { say "ARM FAILED; stopping"; exit 1; }
}
git -C $REPO fetch -q origin
[ -d $WT ] || git -C $REPO worktree add -q --detach $WT $SHA
git -C $WT checkout -q --detach $SHA || exit 1
[ "$(git -C $WT rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
cd $WT
PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)" | tee -a $OUT/driver.log
tree=$(git rev-parse HEAD:python)
(cd benchmarks/dsv41_baseline && $PY -c "import generations; generations.register('$tree', 'compute-gap-$TAG')")
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held; SHA $SHA"
arm $TAG-A 0,1 "$A_OV"; arm $TAG-B 0,1 "$B_OV"
arm $TAG-B 2,3 "$B_OV"; arm $TAG-A 2,3 "$A_OV"
arm $TAG-A 4,5 "$A_OV"; arm $TAG-B 4,5 "$B_OV"
arm $TAG-B 6,7 "$B_OV"; arm $TAG-A 6,7 "$A_OV"
cd benchmarks/dsv41_baseline
rm -rf $OUT/merged-A $OUT/merged-B
$PY concat_arms.py $OUT/merged-A $(cat $OUT/$TAG-A.dirs) >> $OUT/driver.log 2>&1
$PY concat_arms.py $OUT/merged-B $(cat $OUT/$TAG-B.dirs) >> $OUT/driver.log 2>&1
$PY paired.py $OUT/merged-A $OUT/merged-B --a-name A --b-name B > $OUT/paired.txt 2>&1
say "paired rc=$?"
say "DRIVER DONE"
