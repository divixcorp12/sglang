#!/usr/bin/env bash
# SPCC weight sweep on the three-root set (nvme0, SPCC, nvme2), QD 1 and 2, reps interleaved, under rowimg-disk.lock on
# cores 18-35, same seeds within a (rep, QD). The fit in results.md predicts the QD1 optimum near 1:0.82:1.
# Usage: run_weights.sh <bench-binary> <out.jsonl> <raw.csv> [reps]
set -u
BIN=$1; OUT=$2; RAW=$3; REPS=${4:-3}
R="--root /mnt/nvme0/dsv41_flash --root /mnt/nvme4/dsv41_flash --root /mnt/nvme2/dsv41_flash"
export OMP_NUM_THREADS=8
for rep in $(seq 1 "$REPS"); do
  (
    flock 9
    for qd in 1 2; do
      for w in 1 0.9 0.8 0.7 0.6; do
        # shellcheck disable=SC2086
        taskset -c 18-35 "$BIN" --label "w$w/qd$qd/r$rep" $R --weights "1:$w:1" --qd "$qd" --rows 3000 \
          --seed $((5000 + 1000 * rep + qd)) --raw "$RAW" >>"$OUT"
        echo "rc=$? rep=$rep qd=$qd w=$w" >&2
      done
    done
  ) 9>/data/models/slang/nvfp4-work/rowimg-disk.lock
done
