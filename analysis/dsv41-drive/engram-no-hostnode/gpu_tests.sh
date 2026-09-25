#!/usr/bin/env bash
# GPU pytest run for engram-no-hostnode on divix01, under the GPU lock and on cores 32-63 (env as copy-overlap's
# gpu_suite.sh). Waits for the GPU to be free of other compute apps. Usage: gpu_tests.sh <worktree> <log> <pytest args...>
set -u
WT=${1:?worktree}
LOG=${2:?log}
shift 2
PY=/data/models/slang/.venv/bin/python
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
export CUDA_HOME=/usr/local/cuda-13.2
export OMP_NUM_THREADS=8
cd $WT
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
git log -1 --oneline > $LOG
PYTHONPATH=$WT/python taskset -c 32-63 $PY -c 'import sglang;print("sglang from",sglang.__file__)' >> $LOG 2>&1
echo "cmd: pytest -q -p no:randomly -rfE $*" >> $LOG
PYTHONPATH=$WT/python timeout 7200 taskset -c 32-63 $PY -m pytest -q -p no:randomly -rfE "$@" >> $LOG 2>&1
echo "EXIT=$?" >> $LOG
