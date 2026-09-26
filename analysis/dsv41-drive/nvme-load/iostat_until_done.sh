#!/usr/bin/env bash
# iostat -x at 1 s for the four NVMe namespaces until the arm driver logs DRIVER DONE.
D=/mnt/nvme1/prod-flags
iostat -x -t -m 1 nvme0n1 nvme1n1 nvme2n1 nvme3n1 > $D/iostat.log 2>&1 &
p=$!
until grep -q "DRIVER DONE" $D/drive.log; do sleep 10; done
kill $p
