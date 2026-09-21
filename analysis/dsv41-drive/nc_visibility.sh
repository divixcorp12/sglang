#!/usr/bin/env bash
# Runs nc_visibility under the GPU lock and records the conditions around it (NC_VISIBILITY.md).
#   nc_visibility.sh OUTDIR [program args...]      e.g. nc_visibility.sh out --quick
# The program pins itself to cores 32-63 through gpu-run.sh; cores 64-71 stay free.
set -u
OUT=$1; shift
mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/nc_visibility
GPURUN=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh

conditions() {
  {
    echo "== $1 $(date --iso-8601=seconds)"
    echo "-- loadavg"; cat /proc/loadavg
    echo "-- busiest processes (pid %cpu core comm)"; ps -eo pid,pcpu,psr,comm --sort=-pcpu | head -12
    echo "-- gpu"; nvidia-smi --query-gpu=name,driver_version,memory.used,clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current --format=csv
    echo "-- gpu processes"; nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
  } >> "$OUT/conditions.txt"
}

conditions before
{ /usr/local/cuda/bin/nvcc --version | tail -2; git -C "$HERE" rev-parse HEAD 2>/dev/null; sha256sum "$BIN" "$HERE/nc_visibility.cu"; } > "$OUT/build.txt" 2>&1
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv -l 1 > "$OUT/gpu_samples.csv" &
SAMPLER=$!
START=$(date +%s)
"$GPURUN" env OMP_NUM_THREADS=1 "$BIN" "$@" > "$OUT/results.jsonl" 2> "$OUT/stderr.log"
RC=$?
END=$(date +%s)
kill "$SAMPLER" 2>/dev/null
conditions after
echo "rc=$RC wall_s=$((END-START))" >> "$OUT/conditions.txt"
exit "$RC"
