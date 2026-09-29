#!/usr/bin/env bash
set -u
cd /data/models/slang/nvfp4-work/cc-exl3-cpu-bench
PY=/data/models/slang/.venv/bin/python
export EXL3_MOE_CPU_PIN=0 TORCH_EXTENSIONS_DIR=/data/models/slang/nvfp4-work/cc-exl3-cpu-bench/build OMP_NUM_THREADS=16 CUDA_HOME=/usr/local/cuda-13
echo "start $(date +%T) lock acquired; foreign: $(pgrep -fa 'pytest|launch_server' | cut -c1-80 | tr '\n' ';')"
numactl --membind=1 taskset -c 18-35 $PY bench.py bw 4 1,4,8,12,18 node1-local-r2; echo EXIT_bw1=$?
numactl --membind=0 taskset -c 0-17 $PY bench.py bw 4 1,4,8,12,18 node0-local-r2; echo EXIT_bw0=$?
numactl --membind=1 taskset -c 18-35 $PY bench.py perf 384 1,4,8,12,16,18 1,2,6 node1-local-r2; echo EXIT_perf1=$?
numactl --membind=0 taskset -c 0-17 $PY bench.py perf 384 8,12,18 1,2,6 node0-local-r2; echo EXIT_perf0=$?
numactl --membind=1 timeout 900 taskset -c 18-35 $PY handoff.py run 18 12; echo EXIT_handoff=$?
echo "end $(date +%T)"
