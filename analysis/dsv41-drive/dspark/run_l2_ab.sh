#!/usr/bin/env bash
# Served L+2 A/B: smoke, forward order, reversed order, report. Run from a private wt-* worktree on divix01.
# Usage: run_l2_ab.sh OUT_ROOT   (OUT_ROOT under /data/models/slang/nvfp4-work/ram-prefetch/)
set -euo pipefail

ROOT=/data/models/slang/nvfp4-work/ram-prefetch
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock

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
if [[ -z "${L2_AB_LOCKED:-}" ]]; then
    export L2_AB_LOCKED=1
    exec flock "$DISK_LOCK" bash "$0" "$@"
fi

PY=/data/models/slang/.venv-upstream/bin/python
DSPARK=analysis/dsv41-drive/dspark
MISSCUT=dspark-both-misscut
FLOORS=dspark-both-prefetch-gpu-misscut-idle2-floors
IDLE2=dspark-both-prefetch-gpu-misscut-idle2
L2=dspark-both-prefetch-gpu-misscut-idle2-l2

run() {
    taskset -c 0-51,56-107 "$PY" "$@"
}

record_commit() {
    mkdir -p "$1"
    git -C "$HERE" rev-parse HEAD > "$1/commit.txt"
}

record_commit "$OUT"
record_commit "$OUT/smoke"
run "$DSPARK/both_cpu_ab.py" "$OUT/smoke" "$MISSCUT" "$L2" --sessions 0,1

run - "$OUT/smoke/summary.json" "$L2" <<'EOF'
import json
import sys

path, arm = sys.argv[1:]
with open(path) as f:
    entry = json.load(f)[arm]
ram = entry.get("ram") or {}
text = entry.get("text_vs_reference")
bad = []
if not ram.get("spec_issued"):
    bad.append(f"spec_issued={ram.get('spec_issued')}")
if not ram.get("spec_landed"):
    bad.append(f"spec_landed={ram.get('spec_landed')}")
if text is None or isinstance(text, str) or "error" in text or text.get("pass") is not True:
    bad.append(f"text_vs_reference={text}")
if bad:
    sys.exit("smoke failed: " + ", ".join(bad))
print("smoke ok", {k: ram[k] for k in ("spec_issued", "spec_landed")}, flush=True)
EOF

record_commit "$OUT/fwd"
run "$DSPARK/both_cpu_ab.py" "$OUT/fwd" "$MISSCUT" "$FLOORS" "$IDLE2" "$L2"

record_commit "$OUT/rev"
run "$DSPARK/both_cpu_ab.py" "$OUT/rev" "$L2" "$IDLE2" "$FLOORS" "$MISSCUT"

run "$DSPARK/l2_ab_report.py" "$OUT/fwd" "$OUT/rev" --out "$OUT/report"
record_commit "$OUT/report"
echo "report: $OUT/report/REPORT.md"
