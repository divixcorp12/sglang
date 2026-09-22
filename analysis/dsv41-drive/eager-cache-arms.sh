#!/usr/bin/env bash
# Eager base vs eager mirror, interleaved and repeated, with cache counters.
# DSV41_REFERENCE section 19 "Unexplained: eager + mirrors is slower".
#
# Same corpus, sessions and settings as run-mirror-arms.sh e-base / e-mirror:
# graphs OFF, SGLANG_MOE_EXPERT_GRAPH_GATHER=0, sessions 0-3, 256 prompt tokens,
# 128 new tokens, 70 GiB pinned tier. New: SGLANG_DSV41_EXPERT_TRACE_PATH per arm,
# so the scheduler writes the per-forward trace and, beside it, the cache counters
# (<trace>.cache-stats); the driver brackets each session with monotonic times and
# per-drive diskstats.
#
# Usage: [START_IDX=n] eager-cache-arms.sh <label> <run-dir> <arm>...     arm = base | mirror
# NO_TRACE=1 leaves SGLANG_DSV41_EXPERT_TRACE_PATH unset (the original section 19 arms had
# none), so neither the trace nor the cache-stats file is written; only the driver json is.
# Arms are named <label>-<index>-<arm> so a repeated arm never overwrites its twin.
# Every GPU run goes through gpu-run.sh (cc-gpu.lock). Never touches port 7867.
set -uo pipefail

. /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/env-full.sh

# WT=<worktree> runs the arms from another worktree (env.sh cd's into wt-dsv41 and points
# PYTHONPATH at it); EXL3_BUILD=<dir> gives that run a private copy of the extension build dir.
if [ -n "${WT:-}" ]; then
  cd "$WT"
  export PYTHONPATH="$WT/python"
  [ -n "${EXL3_BUILD:-}" ] && export SGLANG_EXL3_BUILD_DIR="$EXL3_BUILD"
fi
echo "worktree: $(pwd)  HEAD: $(git rev-parse --short HEAD)  PYTHONPATH=$PYTHONPATH"

LABEL="$1"; shift
OUTDIR="$1"; shift
mkdir -p "$OUTDIR"
SESSIONS=/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl
MIRRORS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash
DRIVER=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-drive/eager_arm_driver.py
# Keyed by NAME, correct only while /mnt/nvme4 is nvme3n1 (there is no nvme4 device); provenance.resolve_devices() resolves by st_dev.
declare -A DEV=( [nvme0]=nvme0n1 [nvme2]=nvme2n1 [nvme4]=nvme3n1 )

sectors() { awk -v d="$1" '$3==d {print $6}' /proc/diskstats; }

idle_check() {
  echo "--- idle check, sectors read over 5 s per drive (want ~0):"
  local k a
  declare -A b
  for k in nvme0 nvme2 nvme4; do
    b[$k]=$(sectors "${DEV[$k]}")
  done
  sleep 5
  for k in nvme0 nvme2 nvme4; do
    a=$(sectors "${DEV[$k]}")
    echo "    $k: $(( (a - ${b[$k]}) * 512 / 5 )) B/s"
  done
  echo "    other processes touching the drives:"
  fuser -m /mnt/nvme0 /mnt/nvme4 2>/dev/null | head -3 || true
  uptime
}

idx=${START_IDX:-0}
for arm in "$@"; do
  name="$LABEL-$idx-$arm"
  idx=$((idx + 1))
  echo "=================================================================="
  echo "ARM $name  $(date --iso-8601=seconds)"
  echo "=================================================================="
  idle_check
  mirror_env=()
  [ "$arm" = "mirror" ] && mirror_env=(SGLANG_MOE_EXPERT_MIRROR_DIRS="$MIRRORS")
  trace_env=(SGLANG_DSV41_EXPERT_TRACE_PATH="$OUTDIR/$name.trace")
  [ -n "${NO_TRACE:-}" ] && trace_env=()
  rm -f "$OUTDIR/$name.trace" "$OUTDIR/$name.trace.cache-stats"
  t0=$(date +%s)
  $ANA/gpu-run.sh env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    SGLANG_MOE_EXPERT_GRAPH_GATHER=0 \
    "${trace_env[@]}" \
    "${mirror_env[@]}" \
    "$PY" "$DRIVER" \
    --model "$FULL" --sessions "$SESSIONS" --skip 0 --n 4 \
    --prompt-tokens 256 --new-tokens 128 --arm "$name" \
    --out "$OUTDIR/$name.json" \
    > "$OUTDIR/$name.log" 2>&1
  echo "rc=$? wall=$(( $(date +%s) - t0 ))s"
  grep -h '^{"session"' "$OUTDIR/$name.log" | cut -c1-200
done
echo "ALL ARMS DONE"
