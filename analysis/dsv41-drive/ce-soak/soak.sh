#!/usr/bin/env bash
# Copy-engine soak on divix01: one cold server on the arm_env recipe (copy engine on, its default), stage-traced, then
# soak_driver.py's seeded, varied request stream for up to SOAK_MINUTES or SOAK_REQUESTS. Derived from
# analysis/dsv41-drive/copy-engine/smoke.sh.
#
# Usage: soak.sh <tag> <worktree> <seed>
#
# Before the soak proper: one greedy priming request (the driver's determinism prompt) takes the server past the
# 16 decode forwards that arm the copy engine; the script then requires the "copy engine armed" log line and a stage
# trace whose copy counters show lanes on the DMA engine with 0 fallbacks and 0 errors, and refuses to go on otherwise.
#
# On a fail-stop (the server process dies, or its log reports a timed-out or failed request) the script captures,
# before stopping anything: py-spy (--native) and eu-stack of the scheduler if it is still alive, nvidia-smi, and the
# server log tail, under $OUT/failstop/. The driver itself captures py-spy and nvidia-smi while a stream is stalled
# (soak_driver.py CAPTURE_GAP_S), which is the only window in which the hung call is visible under a 2 s deadline.
#
# Optional environment: SOAK_OUT_ROOT (default /mnt/nvme1/ce-soak), SOAK_PORT (30021), SOAK_MINUTES (115),
# SOAK_REQUESTS (320), SOAK_DRIVER_ARGS (extra driver flags, e.g. "--only-kinds sampled"), SOAK_ENV_OVERRIDES (arm_env
# overrides as Python dict items, e.g. "'SGLANG_DSV41_RAM_MISS_TIMEOUT_MS': '60000'"), SOAK_STACK_SECONDS (sample
# the scheduler's native stacks once a second for this long after arming; diagnosis only, it perturbs timing).
#
# Takes cc-gpu.lock and rowimg-disk.lock for the whole soak. Refuses when production (port 7867) is up, or when the
# port is taken. Never starts production. Never writes to / or /tmp.
set -u
TAG=${1:?tag}
WT=${2:?worktree path}
SEED=${3:?seed}
H=$WT/benchmarks/dsv41_baseline
D=$WT/analysis/dsv41-drive/ce-soak
PY=/data/models/slang/.venv/bin/python
OUT=${SOAK_OUT_ROOT:-/mnt/nvme1/ce-soak}/$TAG
PORT=${SOAK_PORT:-30021}
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp TMPDIR=/mnt/nvme1/ce-soak/tmp
mkdir -p $OUT $NSYS_TMPDIR $TMPDIR
LOG=$OUT/server.log
: > $OUT/soak.log
say() { echo "$(date +%T) $*" | tee -a $OUT/soak.log; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
[ "$PORT" != 7867 ] || { say "port 7867 is production's; refusing"; exit 2; }

exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 180; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }
ss -ltn "sport = :$PORT" | grep -q LISTEN && { say "port $PORT taken; refusing"; exit 1; }

rm -f $OUT/stages.jsonl
mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env({${SOAK_ENV_OVERRIDES:-}})
e['SGLANG_DSV41_EXPERT_TRACE_PATH'] = '$OUT/stages.jsonl'
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
e['TMPDIR'] = '$TMPDIR'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
grep -q "^SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1$" $OUT/env.txt || { say "copy engine not on in the recipe; refusing"; exit 1; }
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
MODEL=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.MODEL_PATH)")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
DCORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.DRIVER_CORES)")
say "wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) seed=$SEED port=$PORT cores=$CORES"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/soak.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/soak.log || { say "sglang not imported from $WT; refusing"; exit 1; }

failstop_capture() {  # $1: reason. Evidence first, nothing is stopped here.
  local F=$OUT/failstop
  mkdir -p $F
  say "fail-stop capture: $1"
  local SCHED
  SCHED=$(pgrep -f "sglang::scheduler" | head -1)
  if [ -n "$SCHED" ]; then
    timeout 60 /data/models/slang/.venv/bin/py-spy dump --native --pid $SCHED > $F/py-spy-$SCHED.txt 2>&1
    timeout 60 eu-stack -p $SCHED > $F/eu-stack-$SCHED.txt 2>&1
  fi
  nvidia-smi > $F/nvidia-smi.txt 2>&1
  ps -eo pid,ppid,stat,etime,cmd | grep -E "sglang|python" | grep -v grep > $F/ps.txt
  tail -300 $LOG > $F/server-tail.log
}

cd $WT
taskset -c $CORES env "${ENV[@]}" "${ARGV[@]}" > $LOG 2>&1 &
SPID=$!
echo $SPID > $OUT/server.pid
healthy=0
for i in $(seq 1 180); do
  sleep 5
  kill -0 $SPID 2>/dev/null || break
  curl -sf -m 60 localhost:$PORT/health >/dev/null && { healthy=1; break; }
done
stop_server() {
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 180); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -KILL $SPID 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
  say "stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
}
if [ $healthy != 1 ]; then
  say "server never became healthy (see $LOG)"
  stop_server
  exit 1
fi
say "healthy"

if [ "${SOAK_STACK_SECONDS:-0}" -gt 0 ]; then
  (
    mkdir -p $OUT/stacks
    for i in $(seq 1 600); do grep -q "copy engine armed" $LOG 2>/dev/null && break; sleep 0.5; done
    SCHED=$(pgrep -f "sglang::scheduler" | head -1)
    for i in $(seq 1 ${SOAK_STACK_SECONDS}); do
      timeout 20 eu-stack -p $SCHED > $OUT/stacks/$(date +%H%M%S)-$SCHED.txt 2>&1
      kill -0 $SCHED 2>/dev/null || break
      sleep 1
    done
  ) &
fi

# Priming: the determinism prompt, greedy, 64 tokens, which takes the server past COPY_ENGINE_ARM_DECODES.
curl -sf -m 600 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"List the first ten prime numbers, then explain in two sentences why 1 is not prime."}]}' \
  > $OUT/priming.json
say "priming rc=$? $(head -c 200 $OUT/priming.json)"
for i in $(seq 1 60); do grep -q "copy engine armed" $LOG && break; sleep 1; done
grep -m1 "copy engine armed" $LOG | tee -a $OUT/soak.log || { say "copy engine never armed; refusing"; failstop_capture "not armed"; stop_server; exit 1; }
sleep 3
CE=$(taskset -c $DCORES $PY - $OUT/stages.jsonl <<'EOF'
import json, sys
last = None
for line in open(sys.argv[1]):
    if '"graph_step"' in line:
        r = json.loads(line)
        if r.get("thread"):
            last = r["thread"]
t = last or {}
print(json.dumps({k: t.get(k, 0) for k in ("copy_jobs", "copy_lanes", "copy_fallbacks", "copy_errors",
                                           "copy_generation_mismatches", "leases_copied", "leases_voided")}))
EOF
)
say "copy counters after priming: $CE"
echo "$CE" | $PY -c "
import json, sys
c = json.load(sys.stdin)
sys.exit(0 if c['copy_lanes'] > 0 and c['copy_fallbacks'] == 0 and c['copy_errors'] == 0 else 1)" \
  || { say "lanes are not flowing through the DMA engine cleanly; refusing"; failstop_capture "copy check"; stop_server; exit 1; }

# Fail-stop watch: evidence the moment the server reports a failed request or dies, while it may still be alive.
(
  while kill -0 $SPID 2>/dev/null; do
    if grep -qE "timed out or failed|fail-stop" $LOG; then failstop_capture "log: $(grep -m1 -E 'timed out or failed|fail-stop' $LOG | head -c 300)"; exit 0; fi
    sleep 0.5
  done
  failstop_capture "server process exited"
) &
WATCH=$!

say "soak: seed=$SEED minutes=${SOAK_MINUTES:-115} requests=${SOAK_REQUESTS:-320}"
OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false taskset -c $DCORES $PY $D/soak_driver.py \
  --port $PORT --out $OUT --seed $SEED --tokenizer $MODEL --server-pid $SPID \
  --max-minutes ${SOAK_MINUTES:-115} --max-requests ${SOAK_REQUESTS:-320} ${SOAK_DRIVER_ARGS:-} > $OUT/driver.log 2>&1
RC=$?
say "driver rc=$RC"
kill $WATCH 2>/dev/null
nvidia-smi > $OUT/nvidia-smi-end.txt 2>&1
sleep 3
stop_server
OMP_NUM_THREADS=4 taskset -c $DCORES $PY $D/soak_report.py $OUT --json $OUT/report.json > $OUT/report.txt 2>&1
say "report rc=$? ($OUT/report.txt)"
echo DONE >> $OUT/soak.log
exit $RC
