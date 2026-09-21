#!/bin/bash
# usage: run.sh <tag> <modes>
set -u
W=/data/models/slang/nvfp4-work/cc-packworkers-odirect
OUT=$W/out
tag=$1; modes=$2
( while true; do echo "$(date +%s.%N) $(grep -w nvme0n1 /proc/diskstats)"; sleep 1; done ) > $OUT/$tag.diskstats.log &
SP=$!
uptime > $OUT/$tag.uptime.start
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$W/wt/python \
 taskset -c 0-63 /data/models/slang/.venv/bin/python $W/wt/analysis/dsv41-drive/bench_pack_workers.py \
 --direct --scenarios natural --work-dir /mnt/nvme0/cc-packworkers-scratch --modes $modes --reps 60 --rows 1 2 4 \
 --out $OUT/$tag.json > $OUT/$tag.log 2>&1
echo "exit $?" >> $OUT/$tag.log
kill $SP
uptime > $OUT/$tag.uptime.end
