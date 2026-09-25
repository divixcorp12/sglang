#!/usr/bin/env bash
# Run one GPU command on divix01 under cc-gpu.lock, pinned to cores 32-63, once no other compute app is on the GPU.
# Usage: gpu_run.sh <worktree> <log> <command...>   (the command runs in <worktree> with PYTHONPATH=<worktree>/python)
set -u
WT=${1:?worktree}
LOG=${2:?log}
shift 2
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
export CUDA_HOME=/usr/local/cuda-13.2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TMPDIR=/mnt/nvme1/pytest-tmp
export PYTHONPATH=$WT/python
cd "$WT"
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
{
  git log -1 --oneline
  taskset -c 32-63 /data/models/slang/.venv/bin/python -c 'import sglang; print("sglang from", sglang.__file__)'
  echo "cmd: $*"
} > "$LOG" 2>&1
taskset -c 32-63 "$@" >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
