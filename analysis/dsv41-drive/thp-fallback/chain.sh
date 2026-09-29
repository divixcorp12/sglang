#!/usr/bin/env bash
# Not run to completion: stopped at the pin leak (results.md). Queued after the probe sweeps: startup pairs (master, branch, master, branch), then the kernels suite at both.
set -u
A=/data/models/slang/nvfp4-work/wt-thp-fallback; B=/data/models/slang/nvfp4-work/wt-thp-fallback-base
O=/mnt/nvme1/thp-fallback/startup
until grep -q INJECT_EXIT /mnt/nvme1/thp-fallback/sweep3.log; do sleep 30; done
S=$A/analysis/dsv41-drive/thp-fallback/startup_arm.sh
$S $B before1 $O; echo "before1 EXIT=$?"
$S $A after1 $O; echo "after1 EXIT=$?"
$S $B before2 $O; echo "before2 EXIT=$?"
$S $A after2 $O; echo "after2 EXIT=$?"
for w in $B $A; do
  cd $w
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"
  echo "suite $(git log -1 --oneline) cmd: PYTHONPATH=\$PWD/python OMP_NUM_THREADS=8 flock cc-gpu.lock taskset -c 32-63 python -m pytest test/registered/unit/kernels -q -p no:randomly"
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
    /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly > /mnt/nvme1/thp-fallback/suite-$(basename $w).log 2>&1
  echo "suite $(basename $w) EXIT=$?"; tail -3 /mnt/nvme1/thp-fallback/suite-$(basename $w).log
done
echo CHAIN_DONE
