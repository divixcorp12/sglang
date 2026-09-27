#!/usr/bin/env bash
# Layer-major vs chunked prefill equivalence smoke on divix01: one cold server on today's recipe
# (arm_env as is) with SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS overridden, then equiv.py's full case
# list run against it. Derived from analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh.
#
# Usage: drive_equiv.sh <arm> <worktree> <min_tokens> [out_root]
#   arm         a label, used only for output paths (e.g. chunked, layer-major, layer-major-8k).
#   min_tokens  SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS for this arm (0 disables layer-major prefill
#               entirely, i.e. the chunked-prefill baseline).
#   out_root    output root, default /mnt/nvme1/layer-major/equiv; compare only arms from one root and head.
#
# Writes env.txt, argv.txt, driver.log, server.log, vram.csv, numa.log, phases.txt and retries.txt to
# <out_root>/<arm>/, and the case results to <out_root>/<arm>.jsonl.
# Takes rowimg-disk.lock, then cc-gpu.lock (waits, never breaks them), exactly as chunk_smoke.sh.
# Refuses when production (port 7867) is up. Never starts production.
set -u
ARM=${1:?arm}
WT=${2:?worktree path}
MIN_TOKENS=${3:?min_tokens}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
ROOT=${4:-/mnt/nvme1/layer-major/equiv}
OUT=$ROOT/$ARM
PORT=30014
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
: > $OUT/phases.txt
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }
phase() { echo "$(date '+%Y/%m/%d %H:%M:%S.%3N') $1" >> $OUT/phases.txt; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
[[ $MIN_TOKENS =~ ^[0-9]+$ ]] || { say "numeric min_tokens only: $MIN_TOKENS"; exit 2; }

exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 180; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }

mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env({'SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS': '$MIN_TOKENS'})
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
grep -q "^SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=$MIN_TOKENS\$" $OUT/env.txt || { say "min_tokens missing from env; refusing"; exit 1; }
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
argv = arm_env.ServerArgs(port=$PORT).argv()
print(*argv, sep='\n')
")
[ ${#ARGV[@]} -gt 0 ] || { say "argv failed"; exit 1; }
printf '%s\n' "${ARGV[@]}" > $OUT/argv.txt
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
MODEL=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.MODEL_PATH)")
say "arm=$ARM min_tokens=$MIN_TOKENS wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/driver.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/driver.log || { say "sglang not imported from $WT; refusing"; exit 1; }

# The GPU is ours alone under cc-gpu.lock, so memory.used is this server plus the ~63 MiB idle floor.
taskset -c 16-17 nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 50 > $OUT/vram.csv 2>&1 &
SMI=$!
# Node free memory every 5 s: check (d) wants the layer-major StateStore's node-1 pinned footprint
# against this, the same sampler chunk_smoke.sh uses.
( while true; do echo "$(date +%s) $(numactl --hardware | awk '/free:/ {printf "%s=%s ", $2, $4}')"; sleep 5; done ) \
  > $OUT/numa.log 2>&1 &
NUMA=$!
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

phase run_start
taskset -c 8-15 $PY $WT/analysis/dsv41-drive/layer-major/equiv.py run --port $PORT --model $MODEL \
  --text $WT/DSV41_REFERENCE.md --out $ROOT/$ARM.jsonl >> $OUT/driver.log 2>&1
RC=$?
phase run_end
say "equiv run rc=$RC"
curl -sf -m 60 localhost:$PORT/health >/dev/null && say "healthy after run" || { say "UNHEALTHY after run"; RC=1; }

stop_server
echo DONE >> $OUT/driver.log
exit $RC
