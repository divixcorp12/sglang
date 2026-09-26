#!/usr/bin/env bash
# Copy-overlap smoke on divix01: one cold server per arm on the arm_env recipe (100 GiB tier, two-phase, piece stream,
# row images), stage-traced, six greedy decode requests (prompts 0-2, twice). Writes env.txt, driver.log, server.log,
# stages.jsonl and responses.jsonl to $OUT/<tag>-<arm>. Derived from analysis/dsv41-drive/row-images/smoke.sh.
#
# Usage: smoke.sh <off|on> <tag> <worktree> [nsys-graph]
#   off  arm_env defaults with SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=0
#   on   arm_env defaults with SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=1
#   nsys-graph|nsys-node  wrap the server in an nsys capture of that graph-trace mode (trace.nsys-rep next to the logs)
#
# Takes cc-gpu.lock and rowimg-disk.lock. Refuses when production (port 7867) is up. Never starts production.
set -u
ARM=${1:?arm: off|on}
TAG=${2:?tag}
WT=${3:?worktree path}
NSYS=${4:-}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=/data/models/slang/nvfp4-work/copy-overlap/smoke
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
OVR="{'SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM': '$FLAG'}"

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
grep -q "^SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=$FLAG$" $OUT/env.txt || { say "flag missing from env; refusing"; exit 1; }
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
say "arm=$ARM wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES nsys=$NSYS"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/driver.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/driver.log || { say "sglang not imported from $WT; refusing"; exit 1; }

cd $WT
if [ "$NSYS" = nsys-graph ] || [ "$NSYS" = nsys-node ]; then
  # Node mode is for per-kernel and per-stream attribution only; never read ms/token from it (CLAUDE.md).
  taskset -c $CORES env "${ENV[@]}" nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=${NSYS#nsys-} --sample=none \
    --cpuctxsw=none -o $OUT/trace --force-overwrite true "${ARGV[@]}" > $LOG 2>&1 &
else
  taskset -c $CORES env "${ENV[@]}" "${ARGV[@]}" > $LOG 2>&1 &
fi
SPID=$!
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
if [ "${PYSPY:-0}" = 1 ]; then
  # The scheduler runs the decode loop; the request driver starts right after this, so 90 s covers its decode steps.
  SCHED=$(pgrep -f 'sglang::scheduler' | head -1)
  say "py-spy on scheduler pid=$SCHED"
  (sleep 20; /data/models/slang/.venv/bin/py-spy dump --pid $SCHED > $OUT/pyspy_dump.txt 2>&1) &
  taskset -c 16-23 /data/models/slang/.venv/bin/py-spy record --pid $SCHED --duration 90 --rate 250 --format raw \
    --output $OUT/pyspy.raw --nonblocking > $OUT/pyspy.log 2>&1 &
fi

# Greedy, fixed seed, fixed length. Prompts 0-2, each twice (the repeat checks the server is deterministic on its own).
taskset -c 8-15 $PY - "$PORT" "$OUT/responses.jsonl" <<'EOF' >> $OUT/driver.log 2>&1
import json, sys, time, urllib.request
port, path = sys.argv[1], sys.argv[2]
prompts = [
    "Summarise the main drivers of revenue growth for a regional bank over a decade, with figures.",
    "Explain how a refinery's crack spread affects its quarterly earnings, step by step.",
    "Write a short Python function that merges two sorted lists, then explain its complexity.",
]
with open(path, "w") as f:
    for rep in range(2):
        for i, p in enumerate(prompts):
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
