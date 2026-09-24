#!/usr/bin/env bash
# Row-image smoke on divix01: one cold production-layout server per arm, stage-traced, six greedy decode requests
# (prompts 0-2, twice). Writes env.txt, driver.log, server.log, stages.jsonl and responses.jsonl to $OUT/<tag>-<arm>.
# Derived from direct-two-phase-tests/pinned-numa/smoke_100_audit.sh; the base arm is that script's `on` arm.
#
# Usage: smoke.sh <base|rowimg> <tag> <worktree>
#   base    two-phase, piece stream, 100 GiB pinned host split 60/40 GiB over NUMA nodes 0/1, mirrors (arm_env)
#   rowimg  base + SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1 (every mirror root must hold exl3_row_images/manifest.json)
#
# Takes cc-gpu.lock and rowimg-disk.lock (the row-image converter holds the latter while it writes the mirror drives;
# its writes would skew read timing). Refuses when production (port 7867) is up. Never starts production.
# The copy under version control is analysis/dsv41-drive/row-images/smoke.sh; divix01 runs
# /data/models/slang/nvfp4-work/direct-two-phase-tests/row-images/smoke.sh.
set -u
ARM=${1:?arm: base|rowimg}
TAG=${2:?tag}
WT=${3:?worktree path}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=/data/models/slang/nvfp4-work/direct-two-phase-tests/row-images
OUT=$T/$TAG-$ARM
PORT=30013
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp  # no nsys capture here; set for anyone wrapping this script in one
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
case $ARM in
  base)   EXTRA="" ;;
  rowimg) EXTRA=", 'SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES': '1'" ;;
  *) echo "arm must be base|rowimg"; exit 2 ;;
esac
OVR="{'SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE': '1', 'SGLANG_DSV41_RAM_MISS_HIT_WAIT_US': '100', 'SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM': '1', 'SGLANG_MOE_PINNED_HOST_MB': '102400', 'SGLANG_MOE_PINNED_HOST_NUMA_MB': '0:61440,1:40960'$EXTRA}"

exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"
flock 9
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"
flock 8
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
if [ $ARM = rowimg ]; then
  MIRRORS=$(grep '^SGLANG_MOE_EXPERT_MIRROR_DIRS=' $OUT/env.txt | cut -d= -f2)
  [ -n "$MIRRORS" ] || { say "rowimg arm without mirror dirs; refusing"; exit 1; }
  for root in ${MIRRORS//:/ }; do
    [ -f $root/exl3_row_images/manifest.json ] || { say "no row-image manifest under $root; refusing"; exit 1; }
  done
  grep -q '^SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1$' $OUT/env.txt || { say "flag missing from env; refusing"; exit 1; }
fi
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
say "arm=$ARM wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES"
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
for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -KILL $SPID 2>/dev/null
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
say "stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
echo DONE >> $OUT/driver.log
exit $RC
