#!/usr/bin/env bash
set -u
cd /data/models/slang/nvfp4-work/cc-exl3-cpu-bench
PY=/data/models/slang/.venv/bin/python
export EXL3_MOE_CPU_PIN=0 TORCH_EXTENSIONS_DIR=/data/models/slang/nvfp4-work/cc-exl3-cpu-bench/build OMP_NUM_THREADS=16 CUDA_HOME=/usr/local/cuda-13
echo "start $(date +%T); foreign: $(pgrep -fa 'pytest|launch_server' | grep -v pgrep | cut -c1-80 | tr '\n' ';')"
numactl --membind=1 timeout 900 taskset -c 18-35 $PY handoff.py run -1 12; echo EXIT_h12=$?
numactl --membind=1 timeout 900 taskset -c 18-35 $PY handoff.py run -1 16; echo EXIT_h16=$?
echo "end $(date +%T)"
