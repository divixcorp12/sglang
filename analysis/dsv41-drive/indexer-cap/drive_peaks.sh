#!/usr/bin/env bash
# The four peak/parity smokes, one after another: 30k then 32k tokens, each budget off (0) then on.
# Usage: drive_peaks.sh <worktree> <score_budget_mb>
set -u
WT=${1:?worktree}
MB=${2:?score_budget_mb}
S=$WT/analysis/dsv41-drive/indexer-cap/smoke.sh
cd /mnt/nvme1/indexer-cap
for n in 30000 32000; do
  for b in 0 $MB; do
    tag=b$b-$((n / 1000))k
    echo "$(date +%T) start $tag"
    bash $S $tag $WT 14336 $b $n > $tag.nohup 2>&1 < /dev/null
    echo "$(date +%T) done $tag rc=$?"
  done
done
