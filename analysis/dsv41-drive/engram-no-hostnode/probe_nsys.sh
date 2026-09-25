#!/usr/bin/env bash
# Does a graph-mode nsys capture itself make cudaGraphLaunch block? The same ce_probe run (device-wait Engram
# lookups, no host nodes, --ahead 1 as the overlap scheduler launches) with and without nsys: replay_call_ms is
# the probe's own perf_counter around graph.replay(). Under the GPU lock, on cores 32-63.
# Usage: probe_nsys.sh <worktree> <outdir> [ce_probe args, default below]
set -u
WT=${1:?worktree}
OUT=${2:?outdir}
shift 2
PY=/data/models/slang/.venv/bin/python
export OMP_NUM_THREADS=8
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
mkdir -p $OUT $NSYS_TMPDIR
cd $WT
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
git log -1 --oneline > $OUT/probe_nsys.log
ARGS="${*:---engram-device-wait --rows 2 --ahead 1 --timeout-ms 5000 --replays 60}"
echo "args: $ARGS" >> $OUT/probe_nsys.log
PYTHONPATH=$WT/python timeout 900 taskset -c 32-63 $PY analysis/dsv41-drive/copy-overlap/ce_probe.py --repo $WT \
  --out $OUT/plain.json $ARGS > $OUT/plain.log 2>&1
echo "plain EXIT=$?" >> $OUT/probe_nsys.log
PYTHONPATH=$WT/python timeout 900 taskset -c 32-63 nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=graph \
  --sample=none --cpuctxsw=none -o $OUT/probe_graph --force-overwrite true \
  $PY analysis/dsv41-drive/copy-overlap/ce_probe.py --repo $WT --out $OUT/nsys_graph.json $ARGS > $OUT/nsys_graph.log 2>&1
echo "nsys-graph EXIT=$?" >> $OUT/probe_nsys.log
