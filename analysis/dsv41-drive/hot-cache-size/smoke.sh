#!/usr/bin/env bash
# Hot-cache size smoke on divix01: one cold default-recipe server (row images on, 100 GiB pinned tier), stage-traced,
# six greedy decode requests (prompts 0-2, twice), then one long-prompt request (long_prompt.py), with a 50 ms
# nvidia-smi memory.used sampler running from before launch to after shutdown.
# Derived from analysis/dsv41-drive/row-images/smoke.sh (its `rowimg` arm); the only additions are the hot-cache
# size and memory-fraction overrides, the VRAM sampler, phase markers and the long-prompt request.
#
# Usage: smoke.sh <tag> <worktree> <hot_gpu_mb> <mem_fraction_static> [long_tokens]
#   hot_gpu_mb           SGLANG_MOE_HOT_GPU_MB (default recipe: 14336)
#   mem_fraction_static  replaces --mem-fraction-static in arm_env.ServerArgs.argv() (default recipe: 0.80)
#   long_tokens          prompt length of the long request (default 30000; 0 skips it)
#
# Writes env.txt, argv.txt, driver.log, server.log, stages.jsonl, responses.jsonl, long.json, vram.csv and
# phases.txt to $T/<tag>. Takes cc-gpu.lock and rowimg-disk.lock (waits, never breaks them). Refuses when
# production (port 7867) is up. Never starts production.
# divix01 runs /data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-size/smoke.sh.
set -u
TAG=${1:?tag}
WT=${2:?worktree path}
HOT_MB=${3:?hot_gpu_mb}
MFS=${4:?mem_fraction_static}
LONG=${5:-30000}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-size
OUT=$T/$TAG
PORT=30013
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp  # no nsys capture here; set for anyone wrapping this script in one
mkdir -p $OUT $NSYS_TMPDIR
LOG=$OUT/server.log
: > $OUT/driver.log
: > $OUT/phases.txt
say() { echo "$(date +%T) $*" | tee -a $OUT/driver.log; }
phase() { echo "$(date '+%Y/%m/%d %H:%M:%S.%3N') $1" >> $OUT/phases.txt; }

[ -d "$WT/python/sglang" ] || { say "no sglang tree under $WT"; exit 2; }
[[ $HOT_MB =~ ^[0-9]+$ ]] || { say "hot_gpu_mb must be an integer"; exit 2; }
[[ $MFS =~ ^0\.[0-9]+$ ]] || { say "mem_fraction_static must be 0.x"; exit 2; }
OVR="{'SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE': '1', 'SGLANG_DSV41_RAM_MISS_HIT_WAIT_US': '100', 'SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM': '1', 'SGLANG_MOE_PINNED_HOST_MB': '102400', 'SGLANG_MOE_PINNED_HOST_NUMA_MB': '0:61440,1:40960', 'SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES': '1', 'SGLANG_MOE_HOT_GPU_MB': '$HOT_MB'}"

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
grep -q "^SGLANG_MOE_HOT_GPU_MB=$HOT_MB\$" $OUT/env.txt || { say "hot size missing from env; refusing"; exit 1; }
grep -q '^SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1$' $OUT/env.txt || { say "row images off in env; refusing"; exit 1; }
MIRRORS=$(grep '^SGLANG_MOE_EXPERT_MIRROR_DIRS=' $OUT/env.txt | cut -d= -f2)
[ -n "$MIRRORS" ] || { say "no mirror dirs; refusing"; exit 1; }
for root in ${MIRRORS//:/ }; do
  [ -f $root/exl3_row_images/manifest.json ] || { say "no row-image manifest under $root; refusing"; exit 1; }
done
# The memory fraction is swapped in the argv only; arm_env's MEM_FRACTION_STATIC is untouched.
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
a = arm_env.ServerArgs(port=$PORT).argv()
i = a.index('--mem-fraction-static')
a[i + 1] = '$MFS'
print(*a, sep='\n')
")
printf '%s\n' "${ARGV[@]}" > $OUT/argv.txt
grep -qx -- "$MFS" $OUT/argv.txt || { say "mem fraction missing from argv; refusing"; exit 1; }
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
MODEL=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.MODEL_PATH)")
say "tag=$TAG hot_mb=$HOT_MB mfs=$MFS long=$LONG wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) cores=$CORES"
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' 2>&1 | tee -a $OUT/driver.log
grep -q "sglang from $WT/python/sglang/__init__.py" $OUT/driver.log || { say "sglang not imported from $WT; refusing"; exit 1; }

# 50 ms VRAM sampler: the GPU is ours alone under cc-gpu.lock, so memory.used is this server plus the ~63 MiB idle floor.
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
if [ $healthy != 1 ]; then
  say "server never became healthy (see $LOG)"
  phase unhealthy
  kill -TERM $SPID 2>/dev/null; sleep 30; kill -KILL $SPID 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  kill $SMI 2>/dev/null
  exit 1
fi
say "healthy"
phase healthy

# Greedy, fixed seed, fixed length. Prompts 0-2, each twice (the repeat checks the server is deterministic on its own).
phase smoke_start
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
phase smoke_end
say "driver rc=$RC"

LRC=0
if [ "$LONG" -gt 0 ]; then
  phase long_start
  taskset -c 8-15 $PY $T/long_prompt.py --port $PORT --model $MODEL --text $WT/DSV41_REFERENCE.md \
    --tokens $LONG --out $OUT/long.json >> $OUT/driver.log 2>&1
  LRC=$?
  phase long_end
  say "long rc=$LRC"
  curl -sf -m 60 localhost:$PORT/health >/dev/null && say "healthy after long prompt" || { say "UNHEALTHY after long prompt"; LRC=1; }
fi

phase stop
kill -TERM $SPID 2>/dev/null
for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -KILL $SPID 2>/dev/null
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
phase stopped
kill $SMI 2>/dev/null
say "stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
echo DONE >> $OUT/driver.log
[ $RC = 0 ] && exit $LRC
exit $RC
