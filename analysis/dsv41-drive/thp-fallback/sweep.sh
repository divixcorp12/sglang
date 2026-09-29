#!/usr/bin/env bash
# Scaled THP-fallback sweeps (results.md, "Scaling"). Run on divix01 from the worktree root; takes rowimg-disk.lock
# then cc-gpu.lock (lock order in .claude/rules/divix01-run-protocol.md), since some runs pin up to 16 GiB.
#   analysis/dsv41-drive/thp-fallback/sweep.sh <out.jsonl> <sweep: inject|size|natural|strategies> [extra probe args]
set -u
out=$1; sweep=$2; shift 2
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8
PY=/data/models/slang/.venv/bin/python
probe() { taskset -c 32-63 "$PY" analysis/dsv41-drive/thp-fallback/thp_probe.py "$@" >>"$out"; echo "probe $* EXIT=$?"; }
run() {
  numastat -m | grep -E 'MemFree|AnonHuge'
  case $sweep in
    inject) for k in 0 1 2 4 8 15; do
      probe --gib 16 --madvise hugepage --inject "$k" --strategies prod --label "inject$k" "$@"; done ;;
    size) for g in 4 8 12 16; do probe --gib "$g" --madvise hugepage --inject 999 --label "size$g-allmixed" "$@";
                                  probe --gib "$g" --madvise hugepage --label "size$g-clean" "$@"; done ;;
    strategies) T=0:61440,1:40960
      # The recorded run also passed clone,clone-rowsplit,clone-cap64,clone-cap256. Never again: the clone strategies
      # leak page pins on this kernel, and that run stranded ~149 GiB until a reboot (results.md).
      probe --placement $T --strategies prod,order,cap64 --label "full-none-strategies" "$@"
      probe --placement $T --madvise nohugepage --strategies prod --label "full-nohugepage" "$@" ;;
    # The production tier's split. Order matters: MADV_HUGEPAGE compacts, which changes what the next run finds.
    natural) T=0:61440,1:40960
      probe --placement $T --label "full-none" "$@"
      probe --placement $T --madvise hugepage --label "full-hugepage" "$@"
      probe --placement $T --label "full-none-again" "$@"
      probe --placement $T --repair refault --label "full-none-refault" "$@"
      probe --placement $T --repair collapse --label "full-none-collapse" "$@" ;;
  esac
}
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; flock 9
run "$@"
