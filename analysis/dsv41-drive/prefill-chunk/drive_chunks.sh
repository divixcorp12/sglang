#!/usr/bin/env bash
# Prefill chunk-size sweep: one cold server per chunk size, each prefilling the same long prompt.
# Usage: drive_chunks.sh <worktree> <hot_gpu_mb> <long_tokens> <chunk_tokens>...
set -u
WT=${1:?worktree}
HOT_MB=${2:?hot_gpu_mb}
LONG=${3:?long_tokens}
shift 3
S=$WT/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh
P=$WT/analysis/dsv41-drive/prefill-chunk/chunk_times.py
mkdir -p /mnt/nvme1/prefill-chunk
cd /mnt/nvme1/prefill-chunk
for c in "$@"; do
  tag=c$c-$((LONG / 1000))k
  echo "$(date +%T) start $tag"
  bash $S $tag $WT $c $HOT_MB $LONG > $tag.nohup 2>&1 < /dev/null
  echo "$(date +%T) done $tag rc=$?"
  taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python $P $tag > $tag/chunks.txt 2>&1
done
echo "$(date +%T) SWEEP DONE"
