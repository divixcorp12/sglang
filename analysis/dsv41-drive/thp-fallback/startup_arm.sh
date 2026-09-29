#!/usr/bin/env bash
# Startup-only arm (results.md, "Full-size startup"): launch the production dsv41 recipe (arm_env: 100 GiB tier,
# 0:61440,1:40960) with io_uring fixed buffers on (the uring-reg R3 knobs), wait for /health, record the startup
# milestones and the reader's `expert stream io_uring:` line (register_ms), stop the server. No decode.
# Not run_arm.sh: that refuses a python tree unregistered in generations.json, which is uncommitted per arm.
#   startup_arm.sh <server worktree> <label> <out dir under /mnt/nvme1> [port]
# Takes rowimg-disk.lock, then cc-gpu.lock (blocking), on descriptors so $! stays the server.
set -uo pipefail
WT=${1:?worktree}; LABEL=${2:?label}; OUT=${3:?out dir}; PORT=${4:-30047}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$HERE/../../../benchmarks/dsv41_baseline  # arm_env from this (the analysis) worktree
PY=/data/models/slang/.venv/bin/python
case $OUT in /mnt/nvme1/*) ;; *) echo "out dir must be under /mnt/nvme1"; exit 1 ;; esac
mkdir -p "$OUT/$LABEL"; log=$OUT/$LABEL/server.log; : > "$log"
say() { echo "$(date +%T) $LABEL: $*" | tee -a "$OUT/$LABEL/driver.log"; }

exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; flock 9
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { say "GPU in use"; exit 1; }
! ss -ltn 'sport = :7867' | grep -q LISTEN || { say "production is up"; exit 1; }
! ss -ltn "sport = :$PORT" | grep -q LISTEN || { say "port $PORT busy"; exit 1; }
say "worktree $(git -C "$WT" log -1 --oneline) dirty=$(git -C "$WT" status --porcelain | wc -l)"
PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)" | tee -a "$OUT/$LABEL/driver.log"
# The launch gate falls back to the NVFP4 rules silently when the model dir cannot be read (divix01-run-protocol).
PYTHONPATH=$WT/python:$BENCH $PY -c "
import types, arm_env
from sglang.srt.arg_groups.expert_stream_requirements import expert_stream_requirements_for
cfg = types.SimpleNamespace(model_path=arm_env.MODEL_PATH, quantization=None)
label = expert_stream_requirements_for(cfg, cfg).label
raise SystemExit(0 if label == 'EXL3' else 'gate resolved ' + repr(label))" || { say "EXL3 gate failed"; exit 1; }
cores=$(PYTHONPATH=$BENCH $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
env_argv=()
while IFS='=' read -r k v; do [ -n "$k" ] && env_argv+=("$k=$v"); done < <(PYTHONPATH=$BENCH $PY -c "
import arm_env
for k, v in arm_env.arm_env({'SGLANG_EXPERT_STREAM_URING_FIXED_FILES': '1',
                             'SGLANG_EXPERT_STREAM_URING_READ_MODE': 'readv_fixed',
                             'SGLANG_EXPERT_STREAM_URING_SLAB_ARENA': '1',
                             'SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS': '1'}).items():
    print(f'{k}={v}')")
grep -E 'MemFree|AnonHuge' <(numastat -m) | tee -a "$OUT/$LABEL/driver.log"
grep Normal /proc/buddyinfo | tee -a "$OUT/$LABEL/driver.log"
t0=$(date +%s.%N)
cd "$WT"
taskset -c "$cores" env "${env_argv[@]}" PYTHONPATH="$WT/python" PYTHONUNBUFFERED=1 "$PY" -c "
import sys
sys.path.insert(0, '$BENCH')
import arm_env, os
argv = arm_env.ServerArgs(port=$PORT).argv()
os.execvp(argv[0], argv)" >> "$log" 2>&1 &
spid=$!
say "server pid $spid"
healthy=0
for _ in $(seq 1 360); do
    sleep 5
    kill -0 "$spid" 2>/dev/null || break
    curl -sf -m 60 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { healthy=1; break; }
done
t1=$(date +%s.%N)
say "healthy=$healthy startup_s=$(echo "$t1 - $t0" | bc)"
grep -E 'Load weight end|Pinned host expert cache|expert stream io_uring: mode|fired up' "$log" | cut -c1-400 \
    | tee -a "$OUT/$LABEL/driver.log"
grep -E 'MemFree|AnonHuge' <(numastat -m) | tee -a "$OUT/$LABEL/driver.log"
grep -E '^AnonHugePages' "/proc/$spid/smaps_rollup" | tee -a "$OUT/$LABEL/driver.log"
kill -TERM "$spid" 2>/dev/null
for _ in $(seq 1 120); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
kill -KILL "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
say "stopped"
[ "$healthy" = 1 ]
