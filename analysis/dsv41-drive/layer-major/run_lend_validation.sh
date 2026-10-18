#!/usr/bin/env bash
# Served validation of layer-major expert residency, stage 1 (SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND), branch
# layer-major-lend. One run per arm, no repeats. Run from the worktree holding the pushed commit (a dirty one is refused
# by run_capture_pair.sh), detached with its output in a log:
#
#   nohup analysis/dsv41-drive/layer-major/run_lend_validation.sh STAGE... > /mnt/nvme1/lm-lend/STAGE.log 2>&1 &
#
# DRY=1 resolves and prints every arm's env, command and outputs through run_capture_pair.sh --dry-run and launches
# nothing. Every stage takes rowimg-disk.lock then cc-gpu.lock itself (the capture and drive_equiv.sh), waits for a quiet
# box, and never starts production.
#
# Stages (each is independent; ROOT=/mnt/nvme1/lm-lend):
#   prompts      CPU only: 16k and 32k token prompts from the equivalence corpus, ROOT/prompts/p16k.md and p32k.md.
#   parity-base  base recipe, token 0 of lend vs chunked over equiv.py's whole case list (8k to 33k, flushed and
#                unflushed chains). Chunked is the recorded /mnt/nvme1/layer-major/equiv-t12fix baseline (same corpus,
#                same hash check, --allow-head-mismatch); one server run, the lend arm.
#   a32k         DSpark, the 32k prompt then 240 s of decode, with 2 more cold 32k prefills first for TTFT:
#                prod, chunked, res (stage 0), lend. Reads: token 0 parity (chunked vs res vs lend), TTFT (res vs lend),
#                and decode right after a lend (lend vs prod).
#   b16k         DSpark, the 16k prompt then 60 s of decode, 2 TTFT repeats: chunked, res, lend.
#   c-decode     DSpark, the 1,073-token SAMPLE_PROMPT for 240 s (below the 8192 threshold, so lend never fires): lend
#                only, against hot-smaller-ab-20261017c/hotfull (the same -prod arm, prompt and 240 s, the owner's 68.7
#                ms/token reference).
# Arms of a stage run in the order above. Token 0 parity in the DSpark stages is long_compare.py's divergence token
# (> 0 means token 0 matches; later divergence is documented decode drift, DSV41_REFERENCE.md 27.19).
set -u
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${ROOT:-/mnt/nvme1/lm-lend}
PY=/data/models/slang/.venv/bin/python
CAPPAIR=$WT/analysis/dsv41-drive/dspark/run_capture_pair.sh
DRYFLAG=(); [ "${DRY:-0}" = 1 ] && DRYFLAG=(--dry-run)
cd "$WT" || exit 2
PROD=$(PYTHONPATH=$WT/analysis/dsv41-drive/dspark:$WT/python $PY -c "import both_cpu_ab as a; print(a.PROD_ARM)")
RES=$PROD-layermajor-smallhot-res
LEND=$PROD-layermajor-lend
CHUNKED=$PROD-chunked
mkdir -p "$ROOT"

# One capture: pair OUTDIR BASE_ARM NAME[=K=V...] -- capture args. Returns the wrapper's status.
pair() { local out=$1 base=$2 name=$3; shift 3
  "$CAPPAIR" "${DRYFLAG[@]}" "$out" --worktree "$WT" --arm "$base" "$name" "$@"; }

prompts() {
  mkdir -p "$ROOT/prompts"
  CUDA_VISIBLE_DEVICES= taskset -c 56-63 $PY -I - "$ROOT/prompts" <<'EOF'
import sys
sys.path.insert(0, "benchmarks/dsv41_baseline")
import arm_env
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(arm_env.MODEL_PATH)
ids = tok(open("/mnt/nvme1/layer-major/equiv-corpus/corpus.txt").read())["input_ids"]
for label, n in (("p16k", 16000), ("p32k", 32000)):
    open(f"{sys.argv[1]}/{label}.md", "w").write(tok.decode(ids[:n]))
    print(label, n, "tokens of corpus ids")
EOF
}

parity_base() {
  local base=/mnt/nvme1/layer-major/equiv-t12fix
  EXTRA_ENV_JSON='{"SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS": "360", "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND": "1"}' \
    bash analysis/dsv41-drive/layer-major/drive_equiv.sh lend "$WT" 8192 "$ROOT/parity-base" || return $?
  $PY analysis/dsv41-drive/layer-major/equiv.py compare "$base/chunked.jsonl" "$ROOT/parity-base/lend.jsonl" \
    --allow-head-mismatch
  echo "parity-base compare EXIT=$?"
}

report() { # report OUT_DIRS... : the pairwise decode/TTFT/divergence read
  CUDA_VISIBLE_DEVICES= taskset -c 56-63 $PY analysis/dsv41-drive/dspark/long_compare.py "$@" --json "$ROOT/$(basename "$1")-compare.json"
}

stage_long() { # stage_long STAGE PROMPT DECODE_S ARMS...
  local stage=$1 prompt=$2 secs=$3; shift 3
  local dirs=() rc
  for arm in "$@"; do
    case $arm in
      prod)    base=$PROD ;;
      chunked) base=$CHUNKED ;;
      res)     base=$RES ;;
      lend)    base=$LEND ;;
    esac
    name=$arm=
    pair "$ROOT/$stage-$arm" "$base" "$name" -- --no-verify-split --long-prompt "$ROOT/prompts/$prompt.md" \
      --decode-seconds "$secs" --ttft-repeats 2
    rc=$?
    echo "##### $stage $arm rc=$rc $(date +%T)"
    [ $rc = 0 ] || return $rc
    dirs+=("$ROOT/$stage-$arm/${name%%=*}")
  done
  [ "${DRY:-0}" = 1 ] || report "${dirs[@]}"
}

c_decode() {
  pair "$ROOT/c-decode-lend" "$LEND" lend= -- --no-verify-split --long-prompt "$WT/SAMPLE_PROMPT_DIVIX01.md" \
    --decode-seconds 240 --ttft-repeats 0
  local rc=$?
  echo "##### c-decode lend rc=$rc"
  [ $rc = 0 ] && [ "${DRY:-0}" != 1 ] && report /mnt/nvme1/hot-smaller-ab-20261017c/hotfull "$ROOT/c-decode-lend/lend"
  return $rc
}

for stage in "$@"; do
  echo "===== $stage $(date)"
  case $stage in
    prompts)     prompts ;;
    parity-base) parity_base ;;
    a32k)        stage_long a32k p32k 240 prod chunked res lend ;;
    b16k)        stage_long b16k p16k 60 chunked res lend ;;
    c-decode)    c_decode ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
  rc=$?
  echo "===== $stage rc=$rc $(date)"
  [ $rc = 0 ] || exit $rc
done
echo VALIDATION_DONE
