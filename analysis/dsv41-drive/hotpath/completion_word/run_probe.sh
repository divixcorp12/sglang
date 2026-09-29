#!/usr/bin/env bash
# Phase 2 Task P1: builds and runs probe_completion_word.cu on divix01 (RTX 5090, driver 615.71.09), from a pulled
# worktree. The shim and the probe are built into <out> (never into the worktree). The measuring run is under the
# counting shim (python/sglang/test/hotpath_shim.c, armed at load by HOTPATH_SHIM_OUT); the two --fault runs are
# separate processes, because a device fault poisons the context. GPU work runs under cc-gpu.lock on cores 32-63
# (the probe needs no disk, so it takes no rowimg-disk.lock); builds under taskset -c 0-63.
# Usage: run_probe.sh <out dir under /mnt/nvme1> [tiny jobs] [large jobs]
set -u
OUT=${1:?out dir}; SMALL=${2:-2000}; LARGE=${3:-500}
case $OUT in /mnt/nvme1/*) ;; *) echo "out dir must be under /mnt/nvme1"; exit 1 ;; esac
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(git -C "$HERE" rev-parse --show-toplevel)
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
NVCC=/usr/local/cuda-13.4/bin/nvcc
mkdir -p "$OUT"
echo "worktree $WT at $(git -C "$WT" rev-parse --short HEAD), status: $(git -C "$WT" status --porcelain | wc -l) changed paths"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
taskset -c 0-63 cc -shared -fPIC -O2 -o "$OUT/hotpath_shim.so" "$WT/python/sglang/test/hotpath_shim.c" -ldl -lpthread \
    || { echo "shim build failed"; exit 1; }
taskset -c 0-63 "$NVCC" -O2 -std=c++17 -arch=sm_120 -o "$OUT/probe" "$HERE/probe_completion_word.cu" -lcuda -ldl -lpthread \
    || { echo "probe build failed"; exit 1; }
rm -f "$OUT/probe-shim.json" "$OUT"/probe-shim.json.*
flock "$GPU_LOCK" taskset -c 32-63 env LD_PRELOAD="$OUT/hotpath_shim.so" HOTPATH_SHIM_OUT="$OUT/probe-shim.json" \
    "$OUT/probe" "$OUT/probe.json" "$SMALL" "$LARGE" > "$OUT/probe.log" 2>&1
rc=$?
cat "$OUT/probe.log"; echo "PROBE_EXIT=$rc"
echo "shim whole-process dump: $(cat "$OUT/probe-shim.json" 2>/dev/null)"
for how in wv32 kern; do
    flock "$GPU_LOCK" taskset -c 32-63 "$OUT/probe" "$OUT/fault-$how.json" "--fault=$how" > "$OUT/fault-$how.log" 2>&1
    frc=$?
    cat "$OUT/fault-$how.log"; echo "FAULT_${how}_EXIT=$frc"
done
exit "$rc"
