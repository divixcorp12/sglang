#!/usr/bin/env bash
# F109 owner-started divix01 run: cap-1 cost of each learned checkpoint, then the full l2_cap1_eval driver.
# Inputs (env or positional RUN_DIR DATASET CORPUS B1_HEAD MODEL_PATH OUT): the R1 run dir (read only), the B0 dataset,
# corpus and F0 head, the model path. Output: OUT/{cost/cost_<name>.json,commands.txt,result.json,REPORT.md,...}.
# Usage: run from a private pushed worktree on divix01:
#   DATASET=... CORPUS=... B1_HEAD=... MODEL_PATH=... bash analysis/dsv41-drive/prefetch-replay/l2_cap1_run.sh
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-/mnt/nvme1/nvfp4-work/b1-r1-20261012}}"
DATASET="${2:-${DATASET:?DATASET is required}}"
CORPUS="${3:-${CORPUS:?CORPUS is required}}"
B1_HEAD="${4:-${B1_HEAD:?B1_HEAD is required}}"
MODEL_PATH="${5:-${MODEL_PATH:?MODEL_PATH is required}}"
OUT="${6:-${OUT:-/mnt/nvme1/nvfp4-work/f109-l2-cap1-$(date +%Y%m%d)}}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-/data/models/slang/.venv/bin/python}"
LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock

PWD_REAL="$(pwd -P)"
case "$PWD_REAL" in *dsv41-direct-prod*) echo "refusing to run inside dsv41-direct-prod" >&2; exit 2;; esac
RUN_REAL="$(realpath -m "$RUN_DIR")"
OUT_REAL="$(realpath -m "$OUT")"
case "$OUT_REAL/" in
  "$RUN_REAL"/*) echo "refusing --out inside RUN_DIR" >&2; exit 2;;
  /mnt/nvme1/nvfp4-work/*) ;;
  *) echo "OUT must be under /mnt/nvme1/nvfp4-work" >&2; exit 2;;
esac
case "$OUT_REAL" in
  *b1-run-20261012*|*b1-r1-20261012*|*b1-r2-20261012*|*l2-ab-20261012*) echo "refusing protected OUT" >&2; exit 2;;
esac

HERE="$PWD/analysis/dsv41-drive/prefetch-replay"
export PYTHONPATH="$PWD/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# The driver's own guard, before anything is written, so the two protected lists cannot drift.
PYTHONPATH="$PYTHONPATH:$HERE" "$PYTHON" - "$OUT" "$RUN_DIR" "$HERE" <<'PY'
import sys
out, run, here = sys.argv[1:4]
sys.path.insert(0, here)
import l2_cap1_eval
try:
    l2_cap1_eval.guard_out(out, run_dir=run)
except l2_cap1_eval.RefusedOut as e:
    print(f"refused: {e}", file=sys.stderr)
    sys.exit(2)
PY

mkdir -p "$OUT/cost"
CMDS="$OUT/commands.txt"

{
  echo "# F109 l2_cap1_run.sh $(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "sglang.__file__=$("$PYTHON" -c 'import sglang; print(sglang.__file__)')"
  echo "RUN_DIR=$RUN_DIR DATASET=$DATASET CORPUS=$CORPUS B1_HEAD=$B1_HEAD MODEL_PATH=$MODEL_PATH OUT=$OUT"
} >> "$CMDS"

# Same rule as r1_run._cost_layer: first biased layer of the audit manifest.
LAYER="$("$PYTHON" -I - "$RUN_DIR" <<'PY'
import json, sys
run = sys.argv[1]
audit = json.load(open(f"{run}/stages/audit/result.json"))
m = json.load(open(audit["manifest"]))
print(next(l for l, b in zip(m["layer_ids"], m["gate_has_bias"], strict=True) if b))
PY
)"
echo "LAYER=$LAYER" >> "$CMDS"

# Checkpoint paths come from learned_ckpts, the keys the driver verifies.
CKPTS="$(PYTHONPATH="$PYTHONPATH:$HERE" "$PYTHON" - "$RUN_DIR" "$HERE" <<'PY'
import json, os, sys
run, here = sys.argv[1:3]
repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
for p in (here, os.path.join(repo, "analysis/dsv41-drive/router-capture"), os.path.join(repo, "scripts/dsv41")):
    sys.path.insert(0, p)
import l2_cap1_scorers
l2 = json.load(open(f"{run}/stages/l2/result.json"))
for name, (path, _sha) in l2_cap1_scorers.learned_ckpts(l2).items():
    print(name, path)
PY
)"

# Run a command, log it and its exit status to commands.txt, and stop on failure.
_logged() {
  echo "CMD: $*" >> "$CMDS"
  local st=0
  "$@" || st=$?
  echo "EXIT=$st" >> "$CMDS"
  if [ "$st" -ne 0 ]; then echo "failed (exit $st): $*" >&2; exit "$st"; fi
}

COST_ARGS=()
while read -r NAME CKPT; do
  _logged flock "$LOCK" taskset -c 32-63 "$PYTHON" "$HERE/r1_cost.py" "$CKPT" \
    --plan "$RUN_DIR/run_plan.json" --model-path "$MODEL_PATH" --layer "$LAYER" --cap 1 \
    --out "$OUT/cost/cost_$NAME.json"
  COST_ARGS+=(--cost "$NAME=$OUT/cost/cost_$NAME.json")
done <<< "$CKPTS"

EVAL=("$PYTHON" "$HERE/l2_cap1_eval.py" "$RUN_DIR" --dataset "$DATASET" --corpus "$CORPUS" --b1-head "$B1_HEAD"
      --out "$OUT" "${COST_ARGS[@]}" --device "$DEVICE")
if [ "$DEVICE" = cuda ]; then
  _logged flock "$LOCK" taskset -c 32-63 "${EVAL[@]}"
else
  _logged taskset -c 0-63 "${EVAL[@]}"
fi
echo "done: $OUT/result.json $OUT/REPORT.md" | tee -a "$CMDS"
