#!/usr/bin/env bash
# The CPU suite for engram-no-hostnode on divix01: the registered kernel tests of the RAM-miss/lease/expert work plus
# every Engram unit test, on cores 0-63, with no GPU visible (the GPU cases run under gpu_tests.sh and the lock).
# Usage: cpu_suite.sh <worktree> <log>
set -u
WT=${1:?worktree}
LOG=${2:?log}
PY=/data/models/slang/.venv/bin/python
cd $WT
git log -1 --oneline > $LOG
PYTHONPATH=$WT/python $PY -c 'import sglang;print("sglang from",sglang.__file__)' >> $LOG 2>&1
ENGRAM=$(ls test/registered/unit/layers/test_engram_*.py)
echo "cmd: CUDA_VISIBLE_DEVICES= PYTHONPATH=$WT/python OMP_NUM_THREADS=8 taskset -c 0-63 python -m pytest test/registered/unit/kernels $ENGRAM -q -p no:randomly -k 'ram_miss or lease or piece or expert or engram'" >> $LOG
CUDA_VISIBLE_DEVICES= PYTHONPATH=$WT/python OMP_NUM_THREADS=8 timeout 7200 taskset -c 0-63 $PY -m pytest test/registered/unit/kernels $ENGRAM \
  -q -p no:randomly -k 'ram_miss or lease or piece or expert or engram' 2>&1 | tail -30 >> $LOG
echo "EXIT=${PIPESTATUS[0]}" >> $LOG
