#!/usr/bin/env bash
# The copy-overlap deadlock probe (ce_probe.py) re-run with the Engram host nodes replaced by the device-wait lookups,
# plus the host-node control in the same session. Under the GPU lock, on cores 32-63. Usage: probe.sh <worktree> <outdir>
set -u
WT=${1:?worktree}
OUT=${2:?outdir}
PY=/data/models/slang/.venv/bin/python
export OMP_NUM_THREADS=8
mkdir -p $OUT
cd $WT
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
flock 9
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
git log -1 --oneline > $OUT/probe.log
run() {
  name=$1
  shift
  echo "== $name: ce_probe.py $*" >> $OUT/probe.log
  PYTHONPATH=$WT/python timeout 900 taskset -c 32-63 $PY analysis/dsv41-drive/copy-overlap/ce_probe.py --repo $WT \
    --out $OUT/$name.json "$@" >> $OUT/$name.log 2>&1
  echo "EXIT=$?" >> $OUT/probe.log
}
run engram_rows2 --engram-device-wait --rows 2
run engram_rows2_ahead1 --engram-device-wait --rows 2 --ahead 1 --timeout-ms 200 --replays 60
run engram_rows2_r300 --engram-device-wait --rows 2 --replays 300
run engram_rows2_ahead1_r300 --engram-device-wait --rows 2 --ahead 1 --timeout-ms 200 --replays 300
run hostnodes_rows2_ahead1 --host-nodes --rows 2 --ahead 1 --timeout-ms 200 --replays 60
