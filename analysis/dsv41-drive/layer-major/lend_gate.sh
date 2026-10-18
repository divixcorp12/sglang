#!/bin/bash
# run_lend_validation.sh's GATE: returns when no process holds the GPU (so a job such as mimo-reap is gone) and node 1 has
# at least MIN_GIB GiB available, counting clean page cache as reclaimable:
#   available = MemFree + FilePages - Shmem - Dirty - Writeback
# Read-only; signals nothing.
MIN_GIB=${MIN_GIB:-60}
INFO=/sys/devices/system/node/node1/meminfo
while true; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
  read -r free file shmem dirty wb < <(awk '/MemFree:/{f=$4} /FilePages:/{p=$4} /Shmem:/{s=$4} /Dirty:/{d=$4} /Writeback:/{w=$4}
    END{print f, p, s, d, w}' "$INFO")
  avail=$((free + file - shmem - dirty - wb))
  line="node1 MemFree $((free / 1048576)) GiB, available $((avail / 1048576)) GiB"
  if [ -z "$apps" ] && [ "$avail" -ge $((MIN_GIB * 1048576)) ]; then
    echo "gate open $(date +%T) $line"; exit 0
  fi
  echo "gate closed $(date +%T) gpu apps: $(echo $apps) $line"; sleep 60
done
