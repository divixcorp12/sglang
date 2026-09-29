#!/usr/bin/env bash
# Prefill chunk-size smoke on divix01: one cold server on today's recipe (arm_env as is) with its
# --chunked-prefill-size and SGLANG_MOE_HOT_GPU_MB replaced, one short greedy warm-up request, then one
# long-prompt request (hot-cache-size/long_prompt.py), with a 50 ms nvidia-smi memory.used sampler from
# before launch to after shutdown. Derived from indexer-cap/smoke.sh.
#
# Usage: [CHUNK_NSYS=1] [LONG_MAX_NEW=n] chunk_smoke.sh <tag> <worktree> <chunk_tokens> <hot_gpu_mb> <long_tokens>
#                                                      [mem_fraction_static]
#   CHUNK_NSYS=1  captures the long prompt with Nsight Systems (node mode) to <out>/trace.nsys-rep, plus PCIe RX/TX
#                 from a root metrics session to <out>/trace-pcie.nsys-rep.
#   SMOKE_ENV="K=V ..."  further env overrides on the recipe (e.g. SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 forces
#                 chunked prefill), recorded in env.txt.
#   mem_fraction_static  replaces the recipe's (default: keep it). The hot cache counts against it, so a smaller hot
#                        cache at the same fraction grows the KV pool and leaves activation headroom unchanged.
#
# Writes env.txt, argv.txt, driver.log, server.log, stages.jsonl, warm.json, long.json, vram.csv, phases.txt and
# retries.txt to /mnt/nvme1/prefill-chunk/<tag>. Takes rowimg-disk.lock, then cc-gpu.lock (waits, never breaks
# them). Refuses when production (port 7867) is up. Never starts production.
set -u
TAG=${1:?tag}
WT=${2:?worktree path}
CHUNK=${3:?chunk_tokens}
HOT_MB=${4:?hot_gpu_mb}
LONG=${5:?long_tokens}
MFS=${6:-}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=/mnt/nvme1/prefill-chunk
LP=$WT/analysis/dsv41-drive/hot-cache-size/long_prompt.py
OUT=$T/$TAG
PORT=30013
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
: > $OUT/phases.txt
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }
phase() { echo "$(date '+%Y/%m/%d %H:%M:%S.%3N') $1" >> $OUT/phases.txt; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
for v in "$CHUNK" "$HOT_MB" "$LONG"; do [[ $v =~ ^[0-9]+$ ]] || { say "numeric arguments only: $v"; exit 2; }; done
[ -z "$MFS" ] || [[ $MFS =~ ^0\.[0-9]+$ ]] || { say "mem_fraction_static must be 0.x"; exit 2; }

exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 180; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }

rm -f $OUT/stages.jsonl
mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env({'SGLANG_MOE_HOT_GPU_MB': '$HOT_MB', **dict(kv.split('=', 1) for kv in '${SMOKE_ENV:-}'.split())})
e['SGLANG_DSV41_EXPERT_TRACE_PATH'] = '$OUT/stages.jsonl'
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
grep -q "^SGLANG_MOE_HOT_GPU_MB=$HOT_MB\$" $OUT/env.txt || { say "hot size missing from env; refusing"; exit 1; }
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
argv = arm_env.ServerArgs(port=$PORT).argv()
i = argv.index('--chunked-prefill-size')
argv[i + 1] = '$CHUNK'
if '$MFS':
    argv[argv.index('--mem-fraction-static') + 1] = '$MFS'
if int(argv[argv.index('--max-prefill-tokens') + 1]) < $CHUNK:
    raise SystemExit('max-prefill-tokens below the chunk size')
print(*argv, sep='\n')
")
[ ${#ARGV[@]} -gt 0 ] || { say "argv failed"; exit 1; }
printf '%s\n' "${ARGV[@]}" > $OUT/argv.txt
grep -A1 -x -- '--chunked-prefill-size' $OUT/argv.txt | tail -1 | grep -qx "$CHUNK" || { say "chunk size missing from argv; refusing"; exit 1; }
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
MODEL=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.MODEL_PATH)")
say "tag=$TAG chunk=$CHUNK hot_mb=$HOT_MB mfs=$(grep -A1 -x -- '--mem-fraction-static' $OUT/argv.txt | tail -1) long=$LONG wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/driver.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/driver.log || { say "sglang not imported from $WT; refusing"; exit 1; }

# The GPU is ours alone under cc-gpu.lock, so memory.used is this server plus the ~63 MiB idle floor.
taskset -c 16-17 nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 50 > $OUT/vram.csv 2>&1 &
SMI=$!
# Node free memory every 5 s: phase 0 of the layer-major plan sizes the host state store on node 1.
( while true; do echo "$(date +%s) $(numactl --hardware | awk '/free:/ {printf "%s=%s ", $2, $4}')"; sleep 5; done ) \
  > $OUT/numa.log 2>&1 &
NUMA=$!
sleep 1
phase launch
cd $WT
# CHUNK_NSYS=1: node mode, because graph-mode CUPTI tracing deadlocks the recipe's RAM-miss copy engine
# (nsys_capture.py); prefill runs eagerly, so node mode costs it nothing. Collection runs over the long prompt only.
NSYS_PREFIX=()
NSYS_SESSION=""
PCIE_SESSION=""
NSYS_SUDO=(sudo -n /usr/local/sbin/nsys-profile)
if [ "${CHUNK_NSYS:-0}" = 1 ]; then
  NSYS_SESSION=chunk-$TAG-$$
  NSYS_PREFIX=(nsys launch --session-new=$NSYS_SESSION --trace=cuda,nvtx,osrt --cuda-graph-trace=node)
fi
taskset -c $CORES "${NSYS_PREFIX[@]}" env "${ENV[@]}" "${ARGV[@]}" > $LOG 2>&1 &
LAUNCH_PID=$!
SPID=$LAUNCH_PID
if [ -n "$NSYS_SESSION" ]; then
  # nsys launch forks: find the server by its own cmdline.
  SPID=""
  for i in $(seq 1 180); do
    SPID=$(pgrep -f "sglang.launch_server.*--port $PORT" | head -1)
    [ -n "$SPID" ] && break
    sleep 1
  done
  [ -n "$SPID" ] || { say "no server under nsys"; nsys cancel --session=$NSYS_SESSION; kill $LAUNCH_PID $SMI; exit 1; }
fi
healthy=0
for i in $(seq 1 180); do
  sleep 5
  kill -0 $SPID 2>/dev/null || break
  curl -sf -m 60 localhost:$PORT/health >/dev/null && { healthy=1; break; }
done
stop_pcie() {
  [ -n "$PCIE_SESSION" ] || return 0
  "${NSYS_SUDO[@]}" $1 --session=$PCIE_SESSION >/dev/null 2>&1 || say "WARNING: nsys $1 failed for $PCIE_SESSION"
  "${NSYS_SUDO[@]}" shutdown --session=$PCIE_SESSION >/dev/null 2>&1
  PCIE_SESSION=""
}
stop_server() {
  phase stop
  # Reached with a capture still running only on an abort path: cancel, the data is not worth a report.
  [ -z "$NSYS_SESSION" ] || nsys cancel --session=$NSYS_SESSION >/dev/null 2>&1
  stop_pcie cancel
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -KILL $SPID 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  [ $LAUNCH_PID = $SPID ] || kill -TERM $LAUNCH_PID 2>/dev/null
  for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
  phase stopped
  kill $SMI 2>/dev/null
  kill $NUMA 2>/dev/null
  grep -c "memory allocation failed with OOM" $LOG > $OUT/retries.txt
  say "allocator OOM retries: $(cat $OUT/retries.txt)"
  say "stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
}
if [ $healthy != 1 ]; then
  say "server never became healthy (see $LOG)"
  phase unhealthy
  stop_server
  exit 1
fi
say "healthy"
phase healthy

# A short prompt first, so the long prompt does not also pay the first forward's one-time costs.
phase warm_start
taskset -c 8-15 $PY $LP --port $PORT --model $MODEL --text $WT/DSV41_REFERENCE.md \
  --tokens 256 --max-new 8 --out $OUT/warm.json >> $OUT/driver.log 2>&1
WRC=$?
phase warm_end
say "warm rc=$WRC"

REPORT=$OUT/trace
if [ -n "$NSYS_SESSION" ]; then
  PCIE_SESSION=pcie-$TAG-$$
  "${NSYS_SUDO[@]}" launch --session-new=$PCIE_SESSION --trace=none sleep infinity >> $OUT/driver.log 2>&1 &
  for i in $(seq 1 30); do "${NSYS_SUDO[@]}" sessions list 2>/dev/null | grep -q $PCIE_SESSION && break; sleep 1; done
  "${NSYS_SUDO[@]}" start --session=$PCIE_SESSION --output=$REPORT-pcie --sample=none --cpuctxsw=none \
    --gpu-metrics-devices=all --gpu-metrics-set=gb20x --force-overwrite=true \
    || { say "PCIe metrics session failed to start"; stop_server; exit 1; }
  nsys start --session=$NSYS_SESSION --output=$REPORT --sample=none --cpuctxsw=none --force-overwrite=true \
    || { say "nsys start failed"; stop_server; exit 1; }
  say "capture started -> $REPORT.nsys-rep"
fi
phase long_start
taskset -c 8-15 $PY $LP --port $PORT --model $MODEL --text $WT/DSV41_REFERENCE.md \
  --tokens $LONG --max-new ${LONG_MAX_NEW:-64} --out $OUT/long.json >> $OUT/driver.log 2>&1
LRC=$?
phase long_end
say "long rc=$LRC"
if [ -n "$NSYS_SESSION" ]; then
  # nsys writes the report on stop; wait for it to stop growing before the server goes.
  nsys stop --session=$NSYS_SESSION || say "WARNING: nsys stop failed"
  NSYS_SESSION=""
  stop_pcie stop
  last=-1
  for i in $(seq 1 180); do
    size=$(stat -c %s $REPORT.nsys-rep 2>/dev/null || echo -1)
    [ "$size" -gt 0 ] && [ "$size" = "$last" ] && break
    last=$size
    sleep 5
  done
  say "report: $(ls -l $REPORT.nsys-rep $REPORT-pcie.nsys-rep 2>&1 | tr '\n' ';')"
fi
curl -sf -m 60 localhost:$PORT/health >/dev/null && say "healthy after long prompt" || { say "UNHEALTHY after long prompt"; LRC=1; }

stop_server
echo DONE >> $OUT/driver.log
[ $WRC = 0 ] && exit $LRC
exit $WRC
