#!/usr/bin/env bash
# Route capture for the hot-cache policy study: analysis/dsv41-drive/row-images/smoke.sh's `rowimg` arm (the
# default recipe, row images on, stage trace on), with a parameterised prompt set so a longer run can use it too.
# Writes env.txt, driver.log, server.log, stages.jsonl (with the graph route log) and responses.jsonl to
# $T/<tag>. Takes cc-gpu.lock and rowimg-disk.lock; refuses when production (port 7867) is up; never starts it.
#
# Usage: smoke_routes.sh <tag> <worktree> [prompts.json] [max_tokens] [reps]
#   no prompts file: smoke.sh's three prompts, 96 tokens, twice (the six-prompt smoke)
# ROUTER_CAPTURE=1 also writes every graph forward's router input to $T/<tag>/router.* (RouterCapture).
set -u
ROUTER=${ROUTER_CAPTURE:-0}
TAG=${1:?tag}
WT=${2:?worktree path}
PROMPTS=${3:-}
MAXTOK=${4:-96}
REPS=${5:-2}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy
OUT=$T/$TAG
PORT=30013
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
OVR="{'SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE': '1', 'SGLANG_DSV41_RAM_MISS_HIT_WAIT_US': '100', 'SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM': '1', 'SGLANG_MOE_PINNED_HOST_MB': '102400', 'SGLANG_MOE_PINNED_HOST_NUMA_MB': '0:61440,1:40960', 'SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES': '1'}"

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

rm -f $OUT/stages.jsonl $OUT/router.json $OUT/router.x.bin $OUT/router.w.bin $OUT/router.seq.bin
mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.arm_env($OVR)
e['SGLANG_DSV41_EXPERT_TRACE_PATH'] = '$OUT/stages.jsonl'
if '$ROUTER' == '1':
    e['SGLANG_DSV41_ROUTER_CAPTURE_PATH'] = '$OUT/router'
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > $OUT/env.txt
MIRRORS=$(grep '^SGLANG_MOE_EXPERT_MIRROR_DIRS=' $OUT/env.txt | cut -d= -f2)
[ -n "$MIRRORS" ] || { say "no mirror dirs; refusing"; exit 1; }
for root in ${MIRRORS//:/ }; do
  [ -f $root/exl3_row_images/manifest.json ] || { say "no row-image manifest under $root; refusing"; exit 1; }
done
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
say "wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES prompts=${PROMPTS:-smoke} max_tokens=$MAXTOK reps=$REPS"
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
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  exit 1
fi
say "healthy"

taskset -c 8-15 $PY - "$PORT" "$OUT/responses.jsonl" "$PROMPTS" "$MAXTOK" "$REPS" <<'EOF' >> $OUT/driver.log 2>&1
import json, sys, time, urllib.request
port, path, prompts_path, max_tokens, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
prompts = json.load(open(prompts_path)) if prompts_path else [
    "Summarise the main drivers of revenue growth for a regional bank over a decade, with figures.",
    "Explain how a refinery's crack spread affects its quarterly earnings, step by step.",
    "Write a short Python function that merges two sorted lists, then explain its complexity.",
]
with open(path, "w") as f:
    for rep in range(reps):
        for i, p in enumerate(prompts):
            body = {"model": "default", "max_tokens": max_tokens, "temperature": 0, "seed": 1234, "ignore_eos": True,
                    "messages": [{"role": "user", "content": p}]}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            t0 = time.monotonic()
            d = json.load(urllib.request.urlopen(req, timeout=1800))
            dt = time.monotonic() - t0
            text = d["choices"][0]["message"]["content"]
            f.write(json.dumps({"prompt": i, "rep": rep, "id": d.get("id"), "text": text, "usage": d.get("usage"),
                                "s": round(dt, 2), "t_end": round(time.monotonic(), 6)}) + "\n")
            f.flush()
            print(f"prompt {i} rep {rep}: {d.get('usage')} {dt:.1f}s {text[:60]!r}", flush=True)
EOF
RC=$?
say "driver rc=$RC"

kill -TERM $SPID 2>/dev/null
for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -KILL $SPID 2>/dev/null
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
say "stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
echo DONE >> $OUT/driver.log
exit $RC
