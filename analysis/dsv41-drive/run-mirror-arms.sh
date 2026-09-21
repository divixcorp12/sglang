#!/usr/bin/env bash
# Task 6 Step 6, revised: end-to-end mirroring arms WITH per-drive read accounting.
#
# Why revised: exl3_ram_miss_tables() builds its path table straight from
# layout.records[].path, so the native (graph) RAM-miss reader never sees
# SGLANG_MOE_EXPERT_MIRROR_DIRS. The mirror source is only reachable on the
# eager row-source path. These arms measure that directly instead of assuming
# it: per-drive sectors-read across each run is the evidence.
#
# Arms:
#   g-mirror  graphs, GRAPH_GATHER=1, mirror dirs set
#   g-base    graphs, GRAPH_GATHER=1, no mirror dirs      <- the c32 baseline shape
#   e-mirror  eager,  GRAPH_GATHER=0, mirror dirs set
#   e-base    eager,  GRAPH_GATHER=0, no mirror dirs
# g-mirror vs g-base tests the bypass. e-mirror vs e-base shows the gain where
# mirroring is actually plumbed.
set -uo pipefail

. /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/env-full.sh

ANALYSIS=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis
OUTDIR=$ANALYSIS/dsv41-drive/e2e
mkdir -p "$OUTDIR"

SESSIONS=/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl
MIRRORS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash

# mount point -> block device, for /proc/diskstats field 3 (sectors read, 512 B)
# Keyed by NAME, correct only while /mnt/nvme4 is nvme3n1 (there is no nvme4 device); provenance.resolve_devices() resolves by st_dev.
declare -A DEV=( [nvme0]=nvme0n1 [nvme2]=nvme2n1 [nvme4]=nvme3n1 )

snap_reads() {  # prints "name=sectors" per drive
  local out=""
  for k in nvme0 nvme2 nvme4; do
    local s
    s=$(awk -v d="${DEV[$k]}" '$3==d {print $6}' /proc/diskstats)
    out="$out $k=${s:-0}"
  done
  echo "$out"
}

run_arm() {
  local name="$1"; shift
  local graphs="$1"; shift
  local gather="$1"; shift
  local mirror="$1"; shift

  echo "=================================================================="
  echo "ARM $name  graphs=$graphs gather=$gather mirror=${mirror:-<none>}"
  echo "=================================================================="

  local before after
  before=$(snap_reads)

  local extra=()
  [ "$graphs" = "1" ] && extra+=(--graphs)

  local mirror_env=()
  if [ -n "$mirror" ]; then
    mirror_env=(SGLANG_MOE_EXPERT_MIRROR_DIRS="$mirror")
  fi

  local t0 t1
  t0=$(date +%s.%N)
  $ANA/gpu-run.sh env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    SGLANG_MOE_EXPERT_GRAPH_GATHER="$gather" \
    "${mirror_env[@]}" \
    "$PY" scripts/dsv41/trace_corpus.py \
    --model "$FULL" \
    --sessions "$SESSIONS" \
    --skip 0 --n 4 \
    --prompt-tokens 256 --new-tokens 128 \
    "${extra[@]}" \
    --out "$OUTDIR/$name.json" \
    > "$OUTDIR/$name.log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)

  after=$(snap_reads)
  echo "rc=$rc wall=$(echo "$t1 - $t0" | bc)s"
  echo "reads_before:$before"
  echo "reads_after: $after"
  # delta in GiB per drive
  for k in nvme0 nvme2 nvme4; do
    local b a
    b=$(echo "$before" | tr ' ' '\n' | awk -F= -v k="$k" '$1==k{print $2}')
    a=$(echo "$after"  | tr ' ' '\n' | awk -F= -v k="$k" '$1==k{print $2}')
    awk -v k="$k" -v b="$b" -v a="$a" 'BEGIN{printf "  READ %-6s %10.2f GiB\n", k, (a-b)*512/1073741824}'
  done
  echo "--- tok/s from $name.json:"
  "$PY" - "$OUTDIR/$name.json" <<'EOF' 2>&1 | tail -5
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("no json:", e); raise SystemExit
s = d.get("sessions", d if isinstance(d, list) else [])
tps = [x.get("decode_tok_s") or x.get("tok_s") or x.get("tokens_per_s") for x in s]
tps = [t for t in tps if t]
print("per-session tok/s:", [round(t,3) for t in tps])
if tps: print("mean tok/s: %.3f" % (sum(tps)/len(tps)))
EOF
  echo
}

run_arm g-base   1 1 ""
run_arm g-mirror 1 1 "$MIRRORS"
run_arm e-base   0 0 ""
run_arm e-mirror 0 0 "$MIRRORS"

echo "ALL ARMS DONE"
