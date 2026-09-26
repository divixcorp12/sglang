#!/usr/bin/env bash
# Starts the diskstats sampler and iostat next to the prod-flags arm; both stop when drive.log says DRIVER DONE.
set -u
D=/mnt/nvme1/prod-flags
cd $D
setsid nohup taskset -c 20-23 /data/models/slang/.venv/bin/python $D/nvme_sampler.py $D/nvme_diskstats.jsonl $D/drive.log \
    > $D/sampler.log 2>&1 < /dev/null &
setsid nohup taskset -c 20-23 bash $D/iostat_until_done.sh > /dev/null 2>&1 < /dev/null &
sleep 3
pgrep -af "nvme_sampler|iostat" | grep -v pgrep
