#!/usr/bin/env bash
# Queue numa_h2d.py on divix01 behind cc-gpu.lock (and rowimg-disk.lock, so no smoke's disk traffic loads the socket).
# Usage: run.sh <worktree> <outdir>
set -u
WT=${1:?worktree}
OUT=${2:?outdir}
PY=/data/models/slang/.venv/bin/python
mkdir -p $OUT
LOG=$OUT/run.log
: > $LOG
say() { echo "$(date +%T) $*" >> $LOG; }
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 60; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }
say "head=$(git -C $WT rev-parse HEAD)"
cd $WT
PYTHONPATH=$WT/python OMP_NUM_THREADS=4 taskset -c 40-47 $PY -c 'import sglang; print("sglang from", sglang.__file__)' >> $LOG 2>&1
PYTHONPATH=$WT/python OMP_NUM_THREADS=4 taskset -c 40-47 $PY analysis/dsv41-drive/numa-h2d/numa_h2d.py > $OUT/results.jsonl 2>> $LOG
say "EXIT=$?"
echo DONE >> $LOG
