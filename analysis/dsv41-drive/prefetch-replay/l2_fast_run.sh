#!/usr/bin/env bash
# F123 owner-started divix01 run: L+2 fast-path profile, cost, interference, parity, then result.json and REPORT.md.
# Inputs (env or positional RUN_DIR DATASET CORPUS B1_HEAD MODEL_PATH F109 OUT): the R1 run dir (read only), the B0
# dataset, corpus and F0 head, the model path, the F109 output dir (read only). Optional GATE_DTYPE (default bfloat16)
# is the spec_score gate-weight dtype of parity.json; parity-fp32.json is always the float32 (router_fp32) run.
# Output: OUT/{profile,cost,interference,parity,parity-fp32}.json, OUT/commands.txt, OUT/result.json, OUT/REPORT.md.
# Usage: run from a private pushed worktree on divix01:
#   DATASET=... CORPUS=... B1_HEAD=... MODEL_PATH=... bash analysis/dsv41-drive/prefetch-replay/l2_fast_run.sh
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-/mnt/nvme1/nvfp4-work/b1-r1-20261012}}"
DATASET="${2:-${DATASET:?DATASET is required}}"
CORPUS="${3:-${CORPUS:?CORPUS is required}}"
B1_HEAD="${4:-${B1_HEAD:?B1_HEAD is required}}"
MODEL_PATH="${5:-${MODEL_PATH:?MODEL_PATH is required}}"
F109="${6:-${F109:-/mnt/nvme1/nvfp4-work/f109-l2-cap1-20261013}}"
OUT="${7:-${OUT:-/mnt/nvme1/nvfp4-work/f123-l2-fast-$(date +%Y%m%d)}}"
GATE_DTYPE="${GATE_DTYPE:-bfloat16}"
PYTHON="${PYTHON:-/data/models/slang/.venv/bin/python}"
LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock

PWD_REAL="$(pwd -P)"
case "$PWD_REAL" in *dsv41-direct-prod*) echo "refusing to run inside dsv41-direct-prod" >&2; exit 2;; esac

HERE="$PWD/analysis/dsv41-drive/prefetch-replay"
export PYTHONPATH="$PWD/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# The harness's own guard, before anything is written, so the protected lists cannot drift.
PYTHONPATH="$PYTHONPATH:$HERE" "$PYTHON" - "$OUT" "$RUN_DIR" "$HERE" <<'PY'
import sys
out, run, here = sys.argv[1:4]
sys.path.insert(0, here)
import l2_fast_common
try:
    l2_fast_common.guard_out(out, run_dir=run)
except l2_fast_common.RefusedOut as e:
    print(f"refused: {e}", file=sys.stderr)
    sys.exit(2)
PY

mkdir -p "$OUT"
CMDS="$OUT/commands.txt"

{
  echo "# F123 l2_fast_run.sh $(date -u +%FT%TZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "sglang.__file__=$("$PYTHON" -c 'import sglang; print(sglang.__file__)')"
  echo "RUN_DIR=$RUN_DIR DATASET=$DATASET CORPUS=$CORPUS B1_HEAD=$B1_HEAD MODEL_PATH=$MODEL_PATH F109=$F109 OUT=$OUT GATE_DTYPE=$GATE_DTYPE"
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

# Run a command, log it and its exit status to commands.txt, and stop on failure.
_logged() {
  echo "CMD: $*" >> "$CMDS"
  local st=0
  "$@" || st=$?
  echo "EXIT=$st" >> "$CMDS"
  if [ "$st" -ne 0 ]; then echo "failed (exit $st): $*" >&2; exit "$st"; fi
}

GPU=(flock "$LOCK" taskset -c 32-63 "$PYTHON")

_logged "${GPU[@]}" "$HERE/l2_fast_profile.py" --run-dir "$RUN_DIR" --plan "$RUN_DIR/run_plan.json" \
  --model-path "$MODEL_PATH" --layer "$LAYER" --out "$OUT/profile.json"
_logged "${GPU[@]}" "$HERE/l2_fast_cost.py" --run-dir "$RUN_DIR" --model-path "$MODEL_PATH" --out "$OUT/cost.json"
_logged "${GPU[@]}" "$HERE/l2_fast_interference.py" --run-dir "$RUN_DIR" --out "$OUT/interference.json"

PARITY=("$HERE/l2_fast_parity.py" "$RUN_DIR" --dataset "$DATASET" --corpus "$CORPUS" --b1-head "$B1_HEAD"
        --model-path "$MODEL_PATH" --f109 "$F109")
_logged "${GPU[@]}" "${PARITY[@]}" --out "$OUT/parity.json" --gate-dtype "$GATE_DTYPE" --device cuda

# Q1 / K-T13 v2: the float32 gate-weight (router_fp32) spec_score case always goes to parity-fp32.json
# (report: parity_router_fp32).
_logged "${GPU[@]}" "${PARITY[@]}" --out "$OUT/parity-fp32.json" --gate-dtype float32 --device cuda

_logged taskset -c 0-63 "$PYTHON" "$HERE/l2_fast_report.py" --out "$OUT"
echo "done: $OUT/result.json $OUT/REPORT.md" | tee -a "$CMDS"
