#!/usr/bin/env bash
# Mirror-scaling matrix for mirror_bench on divix01. Each rep holds rowimg-disk.lock (blocking: waits behind decode
# arms, never overlaps them) and runs every root set x QD cell, root sets interleaved, on NUMA-1 cores 18-35 (the
# drives' node, inside the permitted 0-63; 64-71 stay free for the NVMe IRQs). Within a (rep, QD) every root set
# reads the same rows (same seed).
# Usage: run_matrix.sh <bench-binary> <out.jsonl> <raw.csv> [reps]
set -u
BIN=$1; OUT=$2; RAW=$3; REPS=${4:-3}
A=/mnt/nvme0/dsv41_flash   # nvme0n1 Samsung 990 EVO Plus
B=/mnt/nvme4/dsv41_flash   # nvme2n1 SPCC (DRAM-less, HMB)
C=/mnt/nvme2/dsv41_flash   # nvme3n1 Samsung 990 EVO Plus
SETS=(
  "s-nvme0|--root $A"
  "s-spcc|--root $B"
  "s-nvme2|--root $C"
  "p-nvme0+spcc|--root $A --root $B"
  "p-nvme0+nvme2|--root $A --root $C"
  "p-spcc+nvme2|--root $B --root $C"
  "t-equal|--root $A --root $B --root $C"
  "t-spcc0.5|--root $A --root $B --root $C --weights 1:0.5:1"
)
rows_for() { case $1 in 1) echo 2000;; 2) echo 3000;; *) echo 4000;; esac; }
export OMP_NUM_THREADS=8
for rep in $(seq 1 "$REPS"); do
  (
    flock 9
    for qd in 1 2 4; do
      seed=$((1000 * rep + qd))
      for s in "${SETS[@]}"; do
        name=${s%%|*}; args=${s#*|}
        # shellcheck disable=SC2086
        taskset -c 18-35 "$BIN" --label "$name/qd$qd/r$rep" $args --qd "$qd" --rows "$(rows_for "$qd")" --seed "$seed" \
          --raw "$RAW" >>"$OUT"
        echo "rc=$? rep=$rep qd=$qd $name" >&2
      done
    done
  ) 9>/data/models/slang/nvfp4-work/rowimg-disk.lock
done
