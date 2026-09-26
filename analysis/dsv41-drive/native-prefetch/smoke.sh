#!/usr/bin/env bash
# Native-prefetch smoke on divix01: one cold server per arm on the arm_env recipe (100 GiB tier, two-phase, piece
# stream, row images, layer fusion, Engram device wait) WITH THE COPY ENGINE ON in both arms, stage-traced, six greedy
# decode requests (prompts 0-2, twice). Writes env.txt, driver.log, server.log, stages.jsonl and responses.jsonl to
# $OUT/<tag>-<arm>. Derived from analysis/dsv41-drive/copy-engine/smoke.sh.
#
# Usage: smoke.sh <off|on> <tag> <worktree> [nsys-node|long]
#   off  arm_env defaults, SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1, SGLANG_DSV41_ENABLE_NATIVE_PREFETCH=0
#   on   arm_env defaults, SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1, SGLANG_DSV41_ENABLE_NATIVE_PREFETCH=1
#   nsys-node  wrap the server in a node-mode nsys capture (trace.nsys-rep next to the logs). Graph mode is refused:
#              graph-mode CUPTI tracing deadlocks the copy wait (LEASE_PROTOCOL.md 7.6).
#   long       instead of the six requests: one ~8k-token prompt (chunked prefill), then prompt 0
#
# Optional environment: SMOKE_OUT_ROOT (default /data/models/slang/nvfp4-work/copy-engine/smoke) holds <tag>-<arm>;
# SMOKE_ENV_OVERRIDES adds arm_env overrides as Python dict items, e.g. "'SGLANG_DSV41_RAM_MISS_TIMEOUT_MS': '60000'".
# SMOKE_EXTRA_ARGS appends server flags; SMOKE_STACK_SECONDS and SMOKE_CUDA_GDB=1 sample stacks and resident kernels.
#
# Takes cc-gpu.lock and rowimg-disk.lock. Refuses when production (port 7867) is up. Never starts production.
set -u
ARM=${1:?arm: off|on}
TAG=${2:?tag}
WT=${3:?worktree path}
MODE=${4:-}
case $MODE in
  ""|long|nsys-node) ;;
  *) echo "mode must be empty, long or nsys-node (graph-mode nsys deadlocks the copy wait)"; exit 2 ;;
esac
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=${SMOKE_OUT_ROOT:-/data/models/slang/nvfp4-work/direct-two-phase-tests/native-prefetch/smoke}
OUT=$T/$TAG-$ARM
PORT=30013
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
case $ARM in
  off) FLAG=0 ;;
  on)  FLAG=1 ;;
  *) echo "arm must be off|on"; exit 2 ;;
esac
OVR="{'SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE': '1', 'SGLANG_DSV41_ENABLE_NATIVE_PREFETCH': '$FLAG', ${SMOKE_ENV_OVERRIDES:-}}"

# Disk lock, then GPU lock: the order every driver on divix01 uses (.claude/rules/divix01-run-protocol.md).
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 180; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }
ROOT_FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
[ "$ROOT_FREE" -ge 3 ] || { say "only ${ROOT_FREE}G free on /; refusing"; exit 1; }

rm -f $OUT/stages.jsonl
mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env($OVR)
e['SGLANG_DSV41_EXPERT_TRACE_PATH'] = '$OUT/stages.jsonl'
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
grep -q "^SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1$" $OUT/env.txt || { say "copy engine not on in env; refusing"; exit 1; }
grep -q "^SGLANG_DSV41_ENABLE_NATIVE_PREFETCH=$FLAG$" $OUT/env.txt || { say "prefetch flag missing from env; refusing"; exit 1; }
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
[ -n "${SMOKE_EXTRA_ARGS:-}" ] && ARGV+=(${SMOKE_EXTRA_ARGS})  # diagnosis only, e.g. --disable-overlap-schedule
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
say "arm=$ARM wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES mode=$MODE root_free=${ROOT_FREE}G"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/driver.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/driver.log || { say "sglang not imported from $WT; refusing"; exit 1; }

cd $WT
if [ "$MODE" = nsys-node ]; then
  # Node mode is for per-kernel and per-stream attribution only; never read ms/token from it (CLAUDE.md).
  taskset -c $CORES env "${ENV[@]}" nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none \
    --cpuctxsw=none -o $OUT/trace --force-overwrite true "${ARGV[@]}" > $LOG 2>&1 &
else
  taskset -c $CORES env "${ENV[@]}" "${ARGV[@]}" > $LOG 2>&1 &
fi
SPID=$!
if [ "${SMOKE_STACK_SECONDS:-0}" -gt 0 ]; then
  # Diagnosis: once the copy engine arms, sample every thread's native stack of the scheduler process with eu-stack,
  # once a second for SMOKE_STACK_SECONDS, into stacks/. A copy thread blocked in the driver shows up here.
  (
    mkdir -p $OUT/stacks
    for i in $(seq 1 600); do grep -q "copy engine armed" $LOG 2>/dev/null && break; sleep 0.5; done
    SCHED=$(pgrep -f "sglang::scheduler" | head -1)
    [ -n "$SCHED" ] || SCHED=$(pgrep -P $SPID | head -1)
    for i in $(seq 1 ${SMOKE_STACK_SECONDS}); do
      timeout 20 eu-stack -p $SCHED > $OUT/stacks/$(date +%H%M%S)-$SCHED.txt 2>&1
      kill -0 $SCHED 2>/dev/null || break
      if [ "${SMOKE_CUDA_GDB:-0}" = 1 ] && [ $i = 15 ]; then
        # Which kernels are resident on the GPU 15 s after arming (a stalled step shows its spinning kernel).
        timeout 300 /usr/local/cuda-13.2/bin/cuda-gdb -p $SCHED -batch -ex "info cuda kernels" \
          -ex "info cuda contexts" -ex detach > $OUT/cuda-gdb.txt 2>&1
      fi
      sleep 1
    done
  ) &
fi
healthy=0
for i in $(seq 1 180); do
  sleep 5
  kill -0 $SPID 2>/dev/null || break
  curl -sf -m 60 localhost:$PORT/health >/dev/null && { healthy=1; break; }
done
if [ $healthy != 1 ]; then
  say "server never became healthy (see $LOG)"
  kill -TERM $SPID 2>/dev/null; sleep 30; kill -KILL $SPID 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  exit 1
fi
say "healthy"

# Greedy, fixed seed, fixed length. Prompts 0-2, each twice (the repeat checks the server is deterministic on its own).
# long: one ~8k-token prompt through chunked prefill, then prompt 0 once.
taskset -c 8-15 $PY - "$PORT" "$OUT/responses.jsonl" "$MODE" <<'EOF' >> $OUT/driver.log 2>&1
import json, sys, time, urllib.request
port, path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [
    "Summarise the main drivers of revenue growth for a regional bank over a decade, with figures.",
    "Explain how a refinery's crack spread affects its quarterly earnings, step by step.",
    "Write a short Python function that merges two sorted lists, then explain its complexity.",
]
runs = [(rep, i, p) for rep in range(2) for i, p in enumerate(prompts)]
if mode == "long":
    lines = [
        f"Ledger line {k}: account {k * 7919 % 10007} moved {k * 104729 % 99991} units to account "
        f"{k * 15485863 % 10009} on day {k % 365}, reference {k * 32452843 % 1000003}."
        for k in range(240)
    ]
    runs = [(0, "long", "Summarise these ledger lines, then name the three largest moves.\n" + "\n".join(lines)),
            (0, 0, prompts[0])]
with open(path, "w") as f:
    for rep, i, p in runs:
        body = {"model": "default", "max_tokens": 96, "temperature": 0, "seed": 1234, "ignore_eos": True,
                "messages": [{"role": "user", "content": p}]}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.monotonic()
        d = json.load(urllib.request.urlopen(req, timeout=1800))
        dt = time.monotonic() - t0
        text = d["choices"][0]["message"]["content"]
        f.write(json.dumps({"prompt": i, "rep": rep, "text": text, "usage": d.get("usage"), "s": round(dt, 2)}) + "\n")
        f.flush()
        print(f"prompt {i} rep {rep}: {d.get('usage')} {dt:.1f}s {text[:60]!r}", flush=True)
EOF
RC=$?
say "driver rc=$RC"

kill -TERM $SPID 2>/dev/null
for i in $(seq 1 180); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -KILL $SPID 2>/dev/null
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
say "stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
echo DONE >> $OUT/driver.log
exit $RC
