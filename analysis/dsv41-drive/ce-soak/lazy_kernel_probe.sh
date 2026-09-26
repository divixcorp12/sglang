#!/usr/bin/env bash
# Which kernel loads lazily on the soak's failing sequence (items 0, 11, 15 of seed 20260925)?
#
# The arm_env recipe under CUDA_MODULE_LOADING=LAZY with the copy engine OFF (so nothing can hang), traced by Nsight
# Systems in graph mode (legal: the copy engine is off) from process start, so every eager kernel launch, including
# every kernel's first launch, is in the report. The decode graph body is not (graph mode); its kernels were loaded at
# capture, long before any request, so they are not candidates. Then lazy_kernel_report.py lists the kernels whose
# first launch falls inside a soak request, with that launch's cudaLaunchKernel host time (a lazy load shows as a
# long first launch).
#
# Usage: lazy_kernel_probe.sh <tag> <worktree> [seed]
# Takes cc-gpu.lock and rowimg-disk.lock. Never starts production, never writes to / or /tmp.
set -u
TAG=${1:?tag}
WT=${2:?worktree}
SEED=${3:-20260925}
H=$WT/benchmarks/dsv41_baseline
D=$WT/analysis/dsv41-drive/ce-soak
PY=/data/models/slang/.venv/bin/python
OUT=/mnt/nvme1/ce-soak/$TAG
PORT=30021
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp TMPDIR=/mnt/nvme1/ce-soak/tmp
mkdir -p $OUT $NSYS_TMPDIR $TMPDIR
say() { echo "$(date +%T) $*" | tee -a $OUT/probe.log; }
# Disk lock, then GPU lock: the order every driver on divix01 uses (.claude/rules/divix01-run-protocol.md).
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 60; done
ss -ltn "sport = :$PORT" | grep -q LISTEN && { say "port $PORT taken"; exit 1; }
mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env({'CUDA_MODULE_LOADING': 'LAZY', 'SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE': '0'})
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
e['TMPDIR'] = '$TMPDIR'
for k, v in e.items(): print(f'{k}={v}')
")
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "import arm_env; print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')")
MODEL=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.MODEL_PATH)")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
DCORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.DRIVER_CORES)")
say "head=$(git -C $WT rev-parse HEAD) $(PYTHONPATH=$WT/python $PY -c 'import sglang; print(sglang.__file__)')"
cd $WT
taskset -c $CORES env "${ENV[@]}" nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph --sample=none \
  --cpuctxsw=none --force-overwrite=true -o $OUT/lazy "${ARGV[@]}" > $OUT/server.log 2>&1 &
NPID=$!
for i in $(seq 1 240); do sleep 5; kill -0 $NPID 2>/dev/null || break; curl -sf -m 30 localhost:$PORT/health >/dev/null && break; done
curl -sf -m 30 localhost:$PORT/health >/dev/null || { say "not healthy"; kill -INT $NPID; wait $NPID; exit 1; }
say "healthy"
# Items 0, 11 and 15 (the failing pair and the plan's first item), then two more requests.
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false taskset -c $DCORES $PY $D/soak_driver.py --port $PORT --out $OUT \
  --seed $SEED --tokenizer $MODEL --max-requests 5 --max-minutes 10 \
  --skip-items 1,2,3,4,5,6,7,8,9,10,12,13,14 > $OUT/driver.log 2>&1
say "driver rc=$?"
SPID=$(pgrep -f "sglang.launch_server.*--port $PORT" | head -1)
kill -TERM $SPID 2>/dev/null
wait $NPID
say "nsys rc=$? $(ls -la $OUT/lazy.nsys-rep 2>&1)"
for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
