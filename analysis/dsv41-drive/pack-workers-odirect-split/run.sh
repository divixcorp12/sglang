#!/bin/bash
# usage: run.sh <tag>
# Adapted from analysis/dsv41-drive/pack-workers-odirect/run.sh: same modes/reps/rows, split mirrors
# across nvme0 and nvme4 this time (the registered run put both on nvme0).
set -u
W=/data/models/slang/nvfp4-work/cc-packrefill-task4
OUT=/data/models/slang/nvfp4-work/cc-packrefill-task4-out
tag=$1
( while true; do echo "$(date +%s.%N) $(grep -w nvme0n1 /proc/diskstats) || $(grep -w nvme3n1 /proc/diskstats)"; sleep 1; done ) > $OUT/$tag.diskstats.log &
SP=$!
uptime > $OUT/$tag.uptime.start
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/python \
 taskset -c 0-63 /data/models/slang/.venv/bin/python $W/analysis/dsv41-drive/bench_pack_workers.py \
 --direct --scenarios natural \
 --work-dir /mnt/nvme0/cc-packrefill-task4-scratch --work-dir2 /mnt/nvme4/cc-packrefill-task4-scratch \
 --modes 0:0,1:1,4:1,4:4 --reps 60 --rows 1 2 4 \
 --out $OUT/$tag.json > $OUT/$tag.log 2>&1
echo "exit $?" >> $OUT/$tag.log
kill $SP
uptime > $OUT/$tag.uptime.end
