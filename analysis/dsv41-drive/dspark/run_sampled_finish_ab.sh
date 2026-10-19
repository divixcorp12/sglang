#!/bin/bash
# The DSpark sampled-finish served A/B (spec F142, dspark-sampled-finish, S2.2): three arms, one sample each, in this
# order, at production sampling (temperature 0.6, top-p 0.95), each through run_capture_pair.sh on one worktree:
#   prod      -prod                            today's production
#   cut       -prod-confcut-sampled            + the confidence cut for sampled requests (post-warm-up T(w))
#   block     -prod-confcut-sampled-block      + --speculative-use-block-verification
# Arm names come from both_cpu_ab (PROD_ARM, SAMPLED_CUT_ARM, SAMPLED_CUT_BLOCK_ARM), so a rename cannot desync them.
#
# Usage: run_sampled_finish_ab.sh [--worktree DIR] [--python PY] [--out DIR] [--dry-run]
#   --worktree  the checkout to run (default: the one holding this script); a dirty one is refused by the wrapper.
#   --python    default /data/models/slang/.venv-upstream/bin/python
#   --out       default /mnt/nvme1/sampled-finish-$(date +%Y%m%d); arm X lands in OUT/X/X (server.log, long_prompt.json, ...)
#   --dry-run   resolve and print each arm's env and argv (run_capture_pair.sh --dry-run) and the preflight state; take
#               no lock, start nothing, write nothing, refuse nothing.
#
# Locks: the driver holds rowimg-disk.lock for the whole run, then waits for the GPU, which is the protocol's order
# (disk, then cc-gpu); the wrapper takes cc-gpu.lock per arm through the capture and skips its own probe for the disk lock
# (DISK_LOCK_HELD_BY_CALLER=1). It refuses to start while port 7867 is serving or the GPU has a compute app, so production
# must be stopped by the owner first. PAGE_CACHE_OPT_OUT=1. A failed arm (wrapper exit 1) keeps its directory and is run
# once more into OUT/<name>-retry; a refusal (exit 2, 3) or a failed retry stops the run. Run it detached with its output
# in a log, e.g. nohup run_sampled_finish_ab.sh > /mnt/nvme1/sampled-finish-DATE.log 2>&1 &
#
# When the arms are done it writes OUT/report.txt and OUT/report.json (sampled_finish_report.py; advisory 5% ship verdict).

die() { echo "run_sampled_finish_ab: $*" >&2; exit 2; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WT=""; PY=/data/models/slang/.venv-upstream/bin/python; OUT=""; DRY=0
PORT=7867
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
while [ $# -gt 0 ]; do
  case "$1" in
    --worktree) WT=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) die "unknown argument $1" ;;
  esac
done
WT=$(realpath "${WT:-$HERE/../../..}") || die "no worktree"
OUT=$(realpath -m "${OUT:-/mnt/nvme1/sampled-finish-$(date +%Y%m%d)}")
PROMPT=$WT/SAMPLE_PROMPT_DIVIX01.md
[ -f "$PROMPT" ] || die "no $PROMPT"
[ -x "$PY" ] || die "no python $PY"
WRAPPER=$WT/analysis/dsv41-drive/dspark/run_capture_pair.sh
[ -f "$WRAPPER" ] || die "no $WRAPPER"

arm_names() {
  CUDA_VISIBLE_DEVICES= "$PY" -c "import sys; sys.path[:0] = ['$WT/analysis/dsv41-drive/dspark', '$WT/benchmarks/dsv41_baseline', '$WT/python']
import both_cpu_ab as ab
print(ab.PROD_ARM, ab.SAMPLED_CUT_ARM, ab.SAMPLED_CUT_BLOCK_ARM)" 2>/dev/null | tail -1
}
read -r PROD_ARM CUT_ARM BLOCK_ARM <<< "$(arm_names)"
[ -n "$BLOCK_ARM" ] || die "$WT's both_cpu_ab has no SAMPLED_CUT_BLOCK_ARM (is it the sf-arms tree?)"
NAMES=(prod cut block)
ARMS=("$PROD_ARM" "$CUT_ARM" "$BLOCK_ARM")
# The 2026-10-17 confcut pair's prompt and length, now sampled. --no-verify-split as that pair and run_blong.sh.
CAPTURE=(--no-verify-split --long-prompt "$PROMPT" --decode-seconds 420 --ttft-repeats 0 --temperature 0.6 --top-p 0.95)

busy() {
  if ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .; then echo "port $PORT is serving"; return 0; fi
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  if [ -n "$apps" ]; then echo "GPU compute apps: $(echo "$apps" | tr '\n' ';')"; return 0; fi
  return 1
}

wrapper() {  # NAME ARM -> OUT/NAME/NAME, through the wrapper
  local name=$1 arm=$2; shift 2
  PAGE_CACHE_OPT_OUT=1 DISK_LOCK_HELD_BY_CALLER=1 bash "$WRAPPER" "$OUT/$name" --arm "$arm" --worktree "$WT" --python "$PY" \
    "$@" "$name=" -- "${CAPTURE[@]}"
}

if [ "$DRY" = 1 ]; then
  echo "DRY RUN worktree $WT python $PY out $OUT"
  state=$(busy) && echo "preflight (a real run would refuse): $state" || echo "preflight: port $PORT free, GPU idle"
  for i in 0 1 2; do
    echo "##### ${NAMES[$i]} -> ${ARMS[$i]}"
    echo "PAGE_CACHE_OPT_OUT=1 DISK_LOCK_HELD_BY_CALLER=1 bash $WRAPPER $OUT/${NAMES[$i]} --arm ${ARMS[$i]} --worktree $WT --python $PY ${NAMES[$i]}= -- ${CAPTURE[*]}"
    wrapper "${NAMES[$i]}" "${ARMS[$i]}" --dry-run || die "dry run of ${NAMES[$i]} failed"
  done
  echo "report: $PY $HERE/sampled_finish_report.py --prod $OUT/prod/prod --cut $OUT/cut/cut --block $OUT/block/block"
  echo "DRY RUN: nothing launched"
  exit 0
fi

why=$(busy) && { echo "REFUSED: $why; the owner must free the GPU"; exit 75; }
# The lock must be held by an ancestor of the capture (run_locks.take looks the holder's pid up in /proc/locks and trusts
# only an ancestor), so this script re-runs itself under `flock`, whose process stays the parent. A bare `flock -n 9` here
# is held by the dead flock child's pid: the capture then waits forever on its own driver. The checks above and below run
# again once the lock is ours.
if [ -z "$SF_DISK_LOCK_HELD" ]; then
  echo "taking $DISK_LOCK (waits for its holder) $(date)"
  SF_DISK_LOCK_HELD=1 exec flock "$DISK_LOCK" bash "${BASH_SOURCE[0]}" "$@"
fi
[ -e "$OUT" ] && die "$OUT exists; give --out a new directory"
echo "== sampled-finish A/B $(date) tree=$WT @ $(git -C "$WT" log -1 --format=%h) py=$PY out=$OUT"
mkdir -p "$OUT" || die "cannot create $OUT"
echo "holding $DISK_LOCK"
# Disk lock first (taken above), held to the end; then poll for a quiet GPU.
while why=$(busy) || ! flock -n "$GPU_LOCK" true; do echo "waiting for the GPU (${why:-cc-gpu.lock held}) $(date)"; sleep 60; done

rc_all=0; FINAL=()
for i in 0 1 2; do
  name=${NAMES[$i]}; arm=${ARMS[$i]}
  echo "##### $name $arm $(date +%T)"
  wrapper "$name" "$arm" > "$OUT/$name.log" 2>&1; rc=$?
  echo "##### $name rc=$rc $(date +%T) $(grep -E 'ALLDONE|REFUSED|ARM FAILED|ABORT|Traceback' "$OUT/$name.log" | tail -2 | tr '\n' ' ')"
  dir=$OUT/$name/$name
  if [ "$rc" = 1 ]; then
    echo "retrying $name once into $name-retry (the failed attempt stays in $OUT/$name)"
    wrapper "$name-retry" "$arm" > "$OUT/$name-retry.log" 2>&1; rc=$?
    echo "##### $name-retry rc=$rc $(date +%T) $(grep -E 'ALLDONE|REFUSED|ARM FAILED|ABORT|Traceback' "$OUT/$name-retry.log" | tail -2 | tr '\n' ' ')"
    dir=$OUT/$name-retry/$name-retry
  fi
  [ "$rc" = 0 ] || { echo "STOP: $name failed (rc=$rc); see $OUT/$name*.log"; exit "$rc"; }
  FINAL+=("$dir")
done

echo "== report $(date +%T)"
CUDA_VISIBLE_DEVICES= PYTHONPATH=$WT/python OMP_NUM_THREADS=8 taskset -c 0-63 "$PY" "$HERE/sampled_finish_report.py" \
  --prod "${FINAL[0]}" --cut "${FINAL[1]}" --block "${FINAL[2]}" --json "$OUT/report.json" | tee "$OUT/report.txt"
rc=${PIPESTATUS[0]}
echo "== done rc=$rc $(date +%T)"
exit "$rc"
