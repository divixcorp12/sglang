#!/usr/bin/env bash
# Quick served check: native L+2 then T3 learned L+2, sessions 0-3, against F108's imported misscut.
# Run from a private wt-* worktree on divix01.
# Usage: run_l2_t3_quick.sh OUT_ROOT   (OUT_ROOT under /data/models/slang/nvfp4-work/ram-prefetch/)
set -euo pipefail

ROOT=/data/models/slang/nvfp4-work/ram-prefetch
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
F108_FWD=$ROOT/l2-ab-20261012/fwd

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUT_ROOT" >&2
    exit 2
fi
OUT=$(realpath -m "$1")
case "$OUT" in
    "$ROOT"/*) ;;
    *) echo "OUT_ROOT must be under $ROOT, got $OUT" >&2; exit 2 ;;
esac

HERE=$(pwd -P)
case "$HERE" in
    *dsv41-direct-prod*) echo "refusing to run in the production checkout: $HERE" >&2; exit 2 ;;
esac
case "$(basename "$HERE")" in
    wt-*) ;;
    *) echo "run from a private wt-* worktree, cwd is $HERE" >&2; exit 2 ;;
esac

export PYTHONPATH=$HERE/python
export OMP_NUM_THREADS=8

# One disk lock around the whole run; both_cpu_ab.py takes cc-gpu.lock itself (lock order: disk, then GPU).
if [[ -z "${L2_T3_QUICK_LOCKED:-}" ]]; then
    export L2_T3_QUICK_LOCKED=1
    exec flock "$DISK_LOCK" bash "$0" "$@"
fi

PY=/data/models/slang/.venv/bin/python
DSPARK=analysis/dsv41-drive/dspark
L2=dspark-both-prefetch-gpu-misscut-idle2-l2
T3=dspark-both-prefetch-gpu-misscut-idle2-l2-t3
T3_HEAD=$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import both_cpu_ab; print(both_cpu_ab.T3_HEAD)' "$HERE/$DSPARK")

run() {
    taskset -c 0-63 "$PY" "$@"
}

record_commit() {
    mkdir -p "$1"
    git -C "$HERE" rev-parse HEAD > "$1/commit.txt"
}

if [[ ! -f "$T3_HEAD" ]]; then
    echo "T3 head file missing: $T3_HEAD (export it first)" >&2
    exit 2
fi
if [[ ! -d "$F108_FWD" ]]; then
    echo "F108 baseline missing: $F108_FWD" >&2
    exit 2
fi

run "$DSPARK/l2_t3_preflight.py" || { echo "preflight failed; box is not quiet" >&2; exit 1; }

record_commit "$OUT"
record_commit "$OUT/fwd"
run "$DSPARK/both_cpu_ab.py" "$OUT/fwd" "$L2" "$T3" --sessions 0,1,2,3 --baseline "$F108_FWD"

run "$DSPARK/l2_t3_quick_report.py" "$OUT/fwd" --f108 "$F108_FWD" --out "$OUT/report"
record_commit "$OUT/report"
echo "report: $OUT/report/REPORT.md"
