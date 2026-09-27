#!/usr/bin/env bash
# Prefill chunk-size smoke on divix01: one cold server on today's recipe (arm_env as is) with its
# --chunked-prefill-size and SGLANG_MOE_HOT_GPU_MB replaced, one short greedy warm-up request, then one
# long-prompt request (hot-cache-size/long_prompt.py), with a 50 ms nvidia-smi memory.used sampler from
# before launch to after shutdown. Derived from indexer-cap/smoke.sh.
#
# Usage: chunk_smoke.sh <tag> <worktree> <chunk_tokens> <hot_gpu_mb> <long_tokens> [mem_fraction_static]
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
e = arm_env.arm_env({'SGLANG_MOE_HOT_GPU_MB': '$HOT_MB'})
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
sleep 1
phase launch
cd $WT
taskset -c $CORES env "${ENV[@]}" "${ARGV[@]}" > $LOG 2>&1 &
SPID=$!
healthy=0
for i in $(seq 1 180); do
  sleep 5
  kill -0 $SPID 2>/dev/null || break
  curl -sf -m 60 localhost:$PORT/health >/dev/null && { healthy=1; break; }
done
stop_server() {
  phase stop
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -KILL $SPID 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
  phase stopped
  kill $SMI 2>/dev/null
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

phase long_start
taskset -c 8-15 $PY $LP --port $PORT --model $MODEL --text $WT/DSV41_REFERENCE.md \
  --tokens $LONG --out $OUT/long.json >> $OUT/driver.log 2>&1
LRC=$?
phase long_end
say "long rc=$LRC"
curl -sf -m 60 localhost:$PORT/health >/dev/null && say "healthy after long prompt" || { say "UNHEALTHY after long prompt"; LRC=1; }

stop_server
echo DONE >> $OUT/driver.log
[ $WRC = 0 ] && exit $LRC
exit $WRC
