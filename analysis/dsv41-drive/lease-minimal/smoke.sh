#!/usr/bin/env bash
# Served smoke of the minimal lease protocol on divix01: one cold server on the production recipe (arm_env, no
# overrides: copy engine, SM small copies and lease PDL on), six greedy requests (three prompts, twice; enough captured
# decodes to arm the copy engine), then SIGTERM and a check that the service shut down in order.
#
# Usage: smoke.sh <tag> <worktree>
# Writes env.txt, argv.txt, driver.log, server.log and responses.jsonl to /mnt/nvme1/lease-minimal/<tag>.
# Takes rowimg-disk.lock, then cc-gpu.lock (waits, never breaks them). Refuses when production (port 7867) is up.
set -u
TAG=${1:?tag}
WT=${2:?worktree path}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
OUT=/mnt/nvme1/lease-minimal/$TAG
PORT=30017
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU busy; waiting"; sleep 60; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }

mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env({})
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
e['EXL3_MOE_CPU_PIN'] = '0'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
for want in SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1 SGLANG_DSV41_ENABLE_LEASE_PDL=1 \
    SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES=1 CUDA_MODULE_LOADING=EAGER; do
  grep -qx "$want" $OUT/env.txt || { say "recipe lacks $want; refusing"; exit 1; }
done
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
printf '%s\n' "${ARGV[@]}" > $OUT/argv.txt
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
say "tag=$TAG wt=$WT head=$(git -C $WT rev-parse --short HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/driver.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/driver.log || { say "sglang not imported from $WT; refusing"; exit 1; }

cd $WT
taskset -c $CORES env "${ENV[@]}" "${ARGV[@]}" > $LOG 2>&1 &
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
  exit 1
fi
say "healthy"

taskset -c 8-15 $PY - "$PORT" "$OUT/responses.jsonl" <<'EOF' >> $OUT/driver.log 2>&1
import json, sys, time, urllib.request
port, path = sys.argv[1], sys.argv[2]
prompts = [
    "Summarise the main drivers of revenue growth for a regional bank over a decade, with figures.",
    "Explain how a refinery's crack spread affects its quarterly earnings, step by step.",
    "Write a short Python function that merges two sorted lists, then explain its complexity.",
]
texts = {}
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
            texts.setdefault(i, []).append(text)
            f.write(json.dumps({"prompt": i, "rep": rep, "text": text, "usage": d.get("usage"), "s": round(dt, 2)}) + "\n")
            f.flush()
            print(f"prompt {i} rep {rep}: {d.get('usage')} {dt:.1f}s {text[:60]!r}", flush=True)
same = all(a == b for a, b in texts.values())
print(f"repeats identical: {same}", flush=True)
sys.exit(0 if same else 3)
EOF
RC=$?
say "driver rc=$RC"

kill -TERM $SPID 2>/dev/null
wait $SPID
SRC=$?
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
say "server exit status $SRC"
say "copy engine armed: $(grep -c 'copy engine armed' $LOG)"
say "orderly service stop: $(grep -c 'stopping the service thread' $LOG)"
say "FATAL lines: $(grep -c 'FATAL' $LOG); quarantine lines: $(grep -c 'quarantined' $LOG); tracebacks: $(grep -c 'Traceback' $LOG)"
say "gpu apps after: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
echo DONE >> $OUT/driver.log
exit $RC
