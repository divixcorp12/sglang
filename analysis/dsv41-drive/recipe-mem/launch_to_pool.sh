#!/usr/bin/env bash
# Launch one DSV4.1 server from <worktree>'s own recipe (its arm_env: env + argv, fraction unchanged) and stop it once
# KV sizing and decode-graph capture are logged: "Capture target decode CUDA graph end", a RuntimeError, or 900 s. No
# request is sent. Separates code from environment for the GPU memory checkpoints (Load weight begin/end, KV
# available_bytes, Memory pool end, graph capture end) at two commits on the same driver.
# Locks: rowimg-disk.lock, then cc-gpu.lock (both blocking), held for this launch only.
# Usage: launch_to_pool.sh <worktree> <tag>   -> /mnt/nvme1/recipe-mem/pool/<tag>/{server.log,env.txt,argv.txt,summary.txt}
set -u
WT=${1:?worktree}; TAG=${2:?tag}
PY=/data/models/slang/.venv/bin/python
H=$WT/benchmarks/dsv41_baseline
OUT=/mnt/nvme1/recipe-mem/pool/$TAG
PORT=30052
mkdir -p "$OUT"
say() { echo "$(date +%T) $*" | tee -a "$OUT/driver.log"; }
[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || { say "$WT has tracked changes"; exit 1; }
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' | tee -a "$OUT/driver.log"
grep -q "sglang from $WT/python/sglang/__init__.py" "$OUT/driver.log" || { say "sglang not imported from $WT"; exit 1; }

exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock; say "waiting for rowimg-disk.lock"; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; say "waiting for cc-gpu.lock"; flock 9
say "locks held"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do say "GPU memory held; waiting"; sleep 60; done
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }

mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.base_env()
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
for k, v in e.items(): print(f'{k}={v}')
")
mapfile -t ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
[ ${#ENV[@]} -gt 0 ] && [ ${#ARGV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
printf '%s\n' "${ENV[@]}" > "$OUT/env.txt"; printf '%s\n' "${ARGV[@]}" > "$OUT/argv.txt"
say "tag=$TAG wt=$WT head=$(git -C "$WT" rev-parse HEAD) cores=$CORES gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
cd "$WT"
taskset -c "$CORES" env "${ENV[@]}" "${ARGV[@]}" > "$OUT/server.log" 2>&1 8>&- 9>&- &
SPID=$!
end="timeout"
for _ in $(seq 1 180); do
    sleep 5
    kill -0 $SPID 2>/dev/null || { end="exited"; break; }
    grep -q "Capture target decode CUDA graph end" "$OUT/server.log" && { end="captured"; break; }
    grep -q "RuntimeError" "$OUT/server.log" && { end="runtime_error"; sleep 5; break; }
done
kill -TERM $SPID 2>/dev/null
for _ in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -KILL $SPID 2>/dev/null
pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
grep -oE "(Load weight begin|Load weight end|DSV4 memory calculation|Memory pool end|Capture target decode CUDA graph end).*|RuntimeError.*" \
    "$OUT/server.log" | sed -E 's/elapsed=[0-9.]+ s, //' > "$OUT/summary.txt"
say "end=$end; gpu apps after stop: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
cat "$OUT/summary.txt" | tee -a "$OUT/driver.log"
