#!/usr/bin/env bash
set -u
cd /data/models/slang/nvfp4-work/cc-exl3-cpu-bench
PY=/data/models/slang/.venv/bin/python
export EXL3_MOE_CPU_PIN=0 TORCH_EXTENSIONS_DIR=/data/models/slang/nvfp4-work/cc-exl3-cpu-bench/build OMP_NUM_THREADS=16
echo "start $(date +%T) lock acquired"
numactl --membind=1 taskset -c 18-35 $PY bench.py bw 4 1,2,4,8,12,16,18 node1-local; echo EXIT_bw=$?
numactl --membind=1 taskset -c 18-35 $PY bench.py perf 384 1,2,4,8,12,16,18 1,2,6 node1-local; echo EXIT_perf=$?
numactl --membind=1 taskset -c 0-15 $PY bench.py perf 384 8,16 2,6 node0cores-node1mem; echo EXIT_xnuma=$?
echo "end $(date +%T)"
