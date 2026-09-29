#!/usr/bin/env bash
# Runs the iopoll_bench matrix on divix01 under rowimg-disk.lock (serializes against arms), on NUMA-1 cores
# 18-35 (the drives' node; inside the permitted 0-63). Output: one JSON line per variant in $OUT.
# Usage: run_matrix.sh <bench-binary> <out.jsonl> [phase...]   phases: flat row threads fixed
set -u
BIN=$1; OUT=$2; shift 2
PHASES=${*:-flat row threads fixed}
IMG=dsv41_flash/exl3_row_images
F0=/mnt/nvme0/$IMG/layer-003.rows   # nvme0n1 Samsung 990 EVO Plus, xfs, max_sectors_kb 512
F4=/mnt/nvme4/$IMG/layer-003.rows   # nvme2n1 SPCC, ext4, max_sectors_kb 256
F2=/mnt/nvme2/$IMG/layer-003.rows   # nvme3n1 Samsung 990 EVO Plus, xfs, max_sectors_kb 512
S=${SECONDS_PER:-2}
run() { flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 18-35 "$BIN" --seconds "$S" "$@" >>"$OUT"; echo "rc=$? $*" >&2; }
for ph in $PHASES; do case $ph in
flat)
  for f in $F0 $F4; do for qd in 1 8; do for sz in 4096 65536 131072 262144 524288 1048576 2228224; do
    for m in default iopoll; do run --label flat --file $f --workload flat --size $sz --qd $qd --mode $m; done
  done; done; done ;;
row)
  for qd in 1 4; do for m in default iopoll; do
    run --label row-prod --file $F0 --file $F4 --file $F2 --workload row --qd $qd --mode $m
    run --label row-cut512 --file $F0 --file $F4 --file $F2 --workload row --qd $qd --mode $m --cut 524288
    run --label row-cut256 --file $F0 --file $F4 --file $F2 --workload row --qd $qd --mode $m --cut 262144
    run --label row-read256 --file $F0 --file $F4 --file $F2 --workload row --qd $qd --mode $m --op read --cut 262144
  done; done ;;
threads)
  for t in 1 2 4; do for m in default iopoll; do
    run --label thr-flat256 --file $F0 --workload flat --size 262144 --qd 8 --threads $t --mode $m
    run --label thr-row-cut256 --file $F0 --file $F4 --file $F2 --workload row --qd 2 --threads $t --mode $m --cut 262144
  done; done ;;
fixed)
  for m in default iopoll; do
    run --label fixed-row-prod --file $F0 --file $F4 --file $F2 --workload row --qd 1 --mode $m --fixed-files
    run --label fixed-row-cut256 --file $F0 --file $F4 --file $F2 --workload row --qd 1 --mode $m --cut 262144 --fixed-files
  done ;;
esac; done
# repeat: interleaved default/iopoll pairs, QD1 rows, to average out the SPCC drive's run-to-run swings.
if [[ " $PHASES " == *" repeat "* ]]; then
  for rep in 1 2 3 4 5; do for m in default iopoll; do
    run --label rep-prod --file $F0 --file $F4 --file $F2 --workload row --qd 1 --mode $m
    run --label rep-cut256 --file $F0 --file $F4 --file $F2 --workload row --qd 1 --mode $m --cut 262144
    run --label rep-cut256-nogap --file $F0 --file $F4 --file $F2 --workload row --qd 1 --mode $m --cut 262144 --no-gap-cut
    run --label rep-cut512 --file $F0 --file $F4 --file $F2 --workload row --qd 1 --mode $m --cut 524288
  done; done
fi
# qd: rows in flight 1/2/4, prod vs cut256, interleaved.
if [[ " $PHASES " == *" qd "* ]]; then
  for rep in 1 2 3; do for qd in 2 4; do for m in default iopoll; do
    run --label qd-prod --file $F0 --file $F4 --file $F2 --workload row --qd $qd --mode $m
    run --label qd-cut256 --file $F0 --file $F4 --file $F2 --workload row --qd $qd --mode $m --cut 262144
  done; done; done
fi
