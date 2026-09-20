#!/usr/bin/env bash
# Task 4 of the native mirror extents plan: prove in-graph decode reads the mirrors.
#
# Why this differs from run-mirror-arms.sh: that script decided routing from
# aggregate /proc/diskstats, which cannot separate expert reads from any other
# traffic on the device and is not byte-content parity. The RAM-miss service now
# emits per-request per-drive byte and extent counts of its OWN reads
# (stage timing, "drives" in each ram_miss_request trace line), so expert bytes
# are attributed from inside the service. Diskstats stays as a cross-check: it
# catches traffic the service does not know it caused.
#
# Gate (from the plan, tightened per storage-cpu-pipeline-v2):
#   - the service's own per-drive bytes show the mirrors carrying the decode
#     reads and nvme2 carrying approximately none;
#   - diskstats agrees to within a few percent, so nothing is reading the source
#     behind the service's back;
#   - total bytes match the mirrors-off arm, as they did for prefill in section 19.
# Predicted, recorded before measuring so it can be falsified: up to 1.38x on
# decode tok/s (about 3.90 from 2.823). Below ~1.1x means the NVMe share of the
# step is not what section 18.2 attributes, or within-row splitting loses at
# production queue depth (handoff section 3D).
set -uo pipefail

. /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/env-full.sh

ANALYSIS=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis
OUTDIR=$ANALYSIS/dsv41-drive/native-mirror
mkdir -p "$OUTDIR"

SESSIONS=/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl
MIRRORS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash

declare -A DEV=( [nvme0]=nvme0n1 [nvme2]=nvme2n1 [nvme4]=nvme3n1 )

snap_reads() {
  for k in nvme0 nvme2 nvme4; do
    printf "%s=%s " "$k" "$(awk -v d="${DEV[$k]}" '$3==d {print $6}' /proc/diskstats)"
  done
  echo
}

idle_check() {   # the plan forbids benchmarking a drive another job is using
  echo "--- idle check (sectors read over 2s, want ~0):"
  local before after
  before=$(snap_reads); sleep 2; after=$(snap_reads)
  for k in nvme0 nvme2 nvme4; do
    local b a
    b=$(echo "$before" | tr ' ' '\n' | awk -F= -v k="$k" '$1==k{print $2}')
    a=$(echo "$after"  | tr ' ' '\n' | awk -F= -v k="$k" '$1==k{print $2}')
    echo "    $k: $((a-b)) sectors/s"
  done
}

run_arm() {
  local name="$1"; shift
  local mirror="$1"; shift

  echo "=================================================================="
  echo "ARM $name  mirrors=${mirror:-<none>}"
  echo "=================================================================="
  idle_check

  local mirror_env=()
  [ -n "$mirror" ] && mirror_env=(SGLANG_MOE_EXPERT_MIRROR_DIRS="$mirror")

  local before after t0 t1
  before=$(snap_reads)
  t0=$(date +%s.%N)
  $ANA/gpu-run.sh env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    SGLANG_MOE_EXPERT_GRAPH_GATHER=1 \
    SGLANG_DSV41_EXPERT_TRACE_PATH="$OUTDIR/$name.trace" \
    "${mirror_env[@]}" \
    "$PY" scripts/dsv41/trace_corpus.py \
    --model "$FULL" \
    --sessions "$SESSIONS" \
    --skip 0 --n 4 \
    --prompt-tokens 256 --new-tokens 128 \
    --graphs \
    --out "$OUTDIR/$name.json" \
    > "$OUTDIR/$name.log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  after=$(snap_reads)

  echo "rc=$rc wall=$(echo "$t1 - $t0" | bc)s"
  echo "--- diskstats delta (cross-check only):"
  for k in nvme0 nvme2 nvme4; do
    local b a
    b=$(echo "$before" | tr ' ' '\n' | awk -F= -v k="$k" '$1==k{print $2}')
    a=$(echo "$after"  | tr ' ' '\n' | awk -F= -v k="$k" '$1==k{print $2}')
    awk -v k="$k" -v b="$b" -v a="$a" 'BEGIN{printf "    %-6s %10.2f GiB\n", k, (a-b)*512/1073741824}'
  done
  echo
}

run_arm base   ""
run_arm mirror "$MIRRORS"

echo "================ SERVICE-ATTRIBUTED EXPERT BYTES ================"
"$PY" "$ANALYSIS/dsv41-drive/native_mirror_report.py" \
  "$OUTDIR/base.json"  "$OUTDIR/base.trace" \
  "$OUTDIR/mirror.json" "$OUTDIR/mirror.trace"
echo "NATIVE MIRROR ARM DONE"
