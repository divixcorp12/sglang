#!/usr/bin/env bash
# GPU tests for copy-overlap on divix01: the manual RAM-miss files of DSV41_REFERENCE 24.8/24.9 (six plus row images),
# the RAM-miss graph file and the side-stream file (skipped when absent, as at the base). Env as merged_suites.sh.
# Usage: gpu_suite.sh <worktree> <log>
set -u
WT=${1:?worktree}
LOG=${2:?log}
PY=/data/models/slang/.venv/bin/python
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
export CUDA_HOME=/usr/local/cuda-13.2
export OMP_NUM_THREADS=8
cd $WT
FILES="test_exl3_ram_miss_cuda.py test_exl3_lease_kernels_cuda.py test_exl3_two_phase_parity_cuda.py
  test_exl3_two_phase_timing_cuda.py test_exl3_two_phase_failure_cuda.py test_exl3_piece_stream_cuda.py
  test_exl3_piece_stream_row_images_cuda.py test_exl3_ram_miss_graph_gpu.py test_moe_side_stream_gpu.py"
ARGS=""
for f in $FILES; do [ -f test/manual/dsv41/$f ] && ARGS="$ARGS test/manual/dsv41/$f"; done
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
git log -1 --oneline > $LOG
PYTHONPATH=$WT/python taskset -c 32-63 $PY -c 'import sglang;print("sglang from",sglang.__file__)' >> $LOG 2>&1
echo "cmd: pytest -q -p no:randomly -rfE$ARGS" >> $LOG
PYTHONPATH=$WT/python timeout 7200 taskset -c 32-63 $PY -m pytest -q -p no:randomly -rfE $ARGS >> $LOG 2>&1
echo "EXIT=$?" >> $LOG
