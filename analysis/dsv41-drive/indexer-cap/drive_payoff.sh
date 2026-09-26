#!/usr/bin/env bash
# Indexer-cap payoff (DSV41_REFERENCE.md 27.7): the production recipe plus a 128 MB score budget and 3 GiB more hot
# cache (14336 -> 17408 MB), which the cap's ~3.0 GiB of freed prefill transient pays for. One untraced arm, then the
# same arm traced in node mode with the root PCIe session. Production must be stopped.
# The hot cache counts against --mem-fraction-static (at 0.83, 17408 MB leaves no KV: minimum viable 0.911), so the
# fraction rises by the same 3072 MiB of the card's 32607: 0.83 -> 0.925 keeps today's KV pool.
# Usage: drive_payoff.sh SHA [arm|traced|all]
set -u
WT=/data/models/slang/nvfp4-work/wt-payoff
SHA=$1
MODE=${2:-all}
FLAGS="SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB=128 SGLANG_MOE_HOT_GPU_MB=17408"
export DSV41_MEM_FRACTION_STATIC=0.925
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
say() { echo "$(date +%T) $*"; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do say "cc-gpu.lock held; waiting"; sleep 60; done; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
# Disk lock first, then poll the GPU lock that run_arm.sh takes itself.
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
if [ "$MODE" = arm ] || [ "$MODE" = all ]; then
    wait_gpu; say "arm payoff"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh indexer-payoff 30024 $FLAGS; say "payoff rc=$?"
fi
if [ "$MODE" = traced ] || [ "$MODE" = all ]; then
    # The root PCIe session's scratch is /tmp/nsys-root on the root volume; an orphan held cc-gpu.lock on 2026-09-25.
    avail=$(df --output=avail -B1M / | tail -1)
    [ "$avail" -ge 2048 ] || { say "root volume has ${avail} MiB free; need 2048 for the PCIe session"; exit 1; }
    # nsys adds ~17 GB of anon memory on node 0 before the pinned tier's capacity check (DSV41_REFERENCE.md 27.12).
    wait_gpu; say "arm payoff traced"
    NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node NSYS_GPU_METRICS=1 EXPECT_SHA=$SHA \
        bash benchmarks/dsv41_baseline/run_arm.sh indexer-payoff-node 30024 $FLAGS \
        SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:45056; say "payoff traced rc=$?"
    sudo -n /usr/local/sbin/nsys-profile sessions list 2>&1 | awk '/dsv41-pcie-/ {print $NF}' | while read -r name; do
        say "shutting down orphan PCIe session $name"; sudo -n /usr/local/sbin/nsys-profile shutdown --session="$name"
    done
fi
say "DRIVER DONE"
