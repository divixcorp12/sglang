#!/usr/bin/env bash
# Task 1 matched baselines (docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md).
#
# Graph decode, GRAPH_GATHER=1, sessions 0-3, 256 prompt / 128 new tokens, 70 GiB pinned tier:
# the workload of run-native-mirror-arm.sh, from clean detached worktrees instead of wt-dsv41.
#
# Usage: [DRY=1] task1-baseline-arms.sh <label> <run-dir> <code>:<mirror>:<trace>...
#   code   old | new     old = wt-task1-old (pre two-bank reader), new = wt-task1-new
#   mirror on | off      SGLANG_MOE_EXPERT_MIRROR_DIRS set / unset
#   trace  T | U         SGLANG_DSV41_EXPERT_TRACE_PATH set (traced) / unset (untraced)
# Arms run in the order given; interleave them (ABBA) on the command line.
#
# The harness (trace_corpus.py, provenance.py) always comes from the NEW worktree, so an OLD arm
# has the same driver and the same json fields; its provenance names the tree it imported.
# Nothing here reads wt-dsv41, which holds another job's uncommitted work. Never touches 7867.
# Every arm goes through gpu-run.sh (cc-gpu.lock). DRY=1 prints what would run and takes no lock.
set -uo pipefail

CC=/data/models/slang/nvfp4-work/cc-expert-prediction
. $CC/analysis/dsv41-phase3b/env-full.sh   # defines $PY $FULL $ANA; it cd's into wt-dsv41, overridden below

WT_OLD=$CC/wt-task1-old;  HEAD_OLD=099eadba33879b705b860dad9439f2f01bdd6d06
WT_NEW=$CC/wt-task1-new;  HEAD_NEW=b57beac710b46a49b0092a5b652905d7c23e458d
SESSIONS=/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl
MIRRORS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash
EXPERT_SHARD_DIRS=(/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw /mnt/nvme0/dsv41_flash /mnt/nvme4/dsv41_flash)
VERDICT=$CC/analysis/dsv41-drive/task1_arm_verdict.py   # a copy beside this script: neither worktree holds it
declare -A DEV=( [nvme0]=nvme0n1 [nvme2]=nvme2n1 [nvme4]=nvme3n1 )

LABEL="$1"; shift
OUTDIR="$1"; shift
mkdir -p "$OUTDIR"
echo "script: $(sha1sum "$0" | cut -c1-12)  verdict: $(sha1sum "$VERDICT" | cut -c1-12)  host: $(hostname)"

sectors() { awk -v d="$1" '$3==d {print $6}' /proc/diskstats; }

# Resident page-cache bytes of the expert shards (mincore only; reads no data).
expert_resident_bytes() {
  local d
  for d in "${EXPERT_SHARD_DIRS[@]}"; do
    [ -d "$d" ] && find "$d" -type f -name '*.safetensors' -print0 2>/dev/null | xargs -0 -r fincore -b -n --raw -o RES 2>/dev/null
  done | awk '{s+=$1} END {print s+0}'
}
meminfo_cached_kb() { awk '/^Cached:/ {print $2}' /proc/meminfo; }

idle_check() {
  local k a; declare -A b
  for k in nvme0 nvme2 nvme4; do b[$k]=$(sectors "${DEV[$k]}"); done
  sleep 5
  echo "--- idle check, B/s over 5 s (the driver takes its own, stored in the json):"
  for k in nvme0 nvme2 nvme4; do
    a=$(sectors "${DEV[$k]}"); echo "    $k: $(( (a - ${b[$k]}) * 512 / 5 ))"
  done
}

preflight() {  # refuse to start unless the GPU, the port and the worktree are what the arm claims
  local wt="$1" want="$2" head dirty
  head=$(git -C "$wt" rev-parse HEAD) || return 1
  dirty=$(git -C "$wt" status --porcelain --untracked-files=no | wc -l)
  [ "$head" = "$want" ] || { echo "REFUSE: $wt is at $head, expected $want"; return 1; }
  [ "$dirty" = 0 ]      || { echo "REFUSE: $wt has $dirty tracked changes"; return 1; }
  if ss -ltn 2>/dev/null | grep -q ':7867 '; then echo "REFUSE: port 7867 is listening (production is up)"; return 1; fi
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    echo "REFUSE: a compute process holds the GPU"; return 1
  fi
}

idx=0
for spec in "$@"; do
  IFS=: read -r code mirror trace <<<"$spec"
  case "$code" in old) wt=$WT_OLD; want=$HEAD_OLD ;; new) wt=$WT_NEW; want=$HEAD_NEW ;; *) echo "bad code in $spec"; exit 2 ;; esac
  case "$mirror" in on|off) ;; *) echo "bad mirror in $spec"; exit 2 ;; esac
  case "$trace" in T|U) ;; *) echo "bad trace in $spec"; exit 2 ;; esac
  name="$LABEL-$idx-$code-$mirror-$trace"; idx=$((idx + 1))
  build=$CC/exl3-build-task1-$code
  env_args=(OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH="$wt/python" SGLANG_EXL3_BUILD_DIR="$build" SGLANG_MOE_EXPERT_GRAPH_GATHER=1)
  [ "$mirror" = on ] && env_args+=(SGLANG_MOE_EXPERT_MIRROR_DIRS="$MIRRORS")
  [ "$trace" = T ] && env_args+=(SGLANG_DSV41_EXPERT_TRACE_PATH="$OUTDIR/$name.trace")
  cmd=("$PY" scripts/dsv41/trace_corpus.py --model "$FULL" --sessions "$SESSIONS" --skip 0 --n 4
       --prompt-tokens 256 --new-tokens 128 --graphs --out "$OUTDIR/$name.json")

  echo "=================================================================="
  echo "ARM $name  code=$code@$want mirror=$mirror trace=$trace  $(date --iso-8601=seconds)"
  echo "=================================================================="
  if [ -n "${DRY:-}" ]; then
    echo "DRY cwd=$WT_NEW  taskset -c 32-63 flock \$LOCK env ${env_args[*]} ${cmd[*]}"
    continue
  fi
  preflight "$wt" "$want" || { echo "ARM $name NOT RUN"; exit 3; }
  [ -d "$build" ] || cp -r "$CC/exl3-build" "$build"
  cd "$WT_NEW"                       # harness comes from here; PYTHONPATH picks the sglang under test
  idle_check
  rm -f "$OUTDIR/$name.trace" "$OUTDIR/$name.trace.cache-stats"
  cached0=$(meminfo_cached_kb); res0=$(expert_resident_bytes)
  declare -A before; for k in nvme0 nvme2 nvme4; do before[$k]=$(sectors "${DEV[$k]}"); done
  t0=$(date +%s)
  $ANA/gpu-run.sh env "${env_args[@]}" "${cmd[@]}" > "$OUTDIR/$name.log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - t0 ))
  cached1=$(meminfo_cached_kb); res1=$(expert_resident_bytes)
  {
    printf '{"arm": "%s", "rc": %s, "wall_s": %s, "meminfo_cached_kb_before": %s, "meminfo_cached_kb_after": %s,' \
      "$name" "$rc" "$wall" "$cached0" "$cached1"
    printf ' "expert_resident_bytes_before": %s, "expert_resident_bytes_after": %s, "diskstats_read_bytes": {' "$res0" "$res1"
    sep=""; for k in nvme0 nvme2 nvme4; do
      printf '%s"%s": %s' "$sep" "$k" $(( ($(sectors "${DEV[$k]}") - ${before[$k]}) * 512 )); sep=", "
    done
    printf '}}\n'
  } > "$OUTDIR/$name.cache.json"
  echo "rc=$rc wall=${wall}s"
  "$PY" "$VERDICT" "$OUTDIR/$name.json" --root "$wt" --head "$want" --mirror "$mirror" --trace "$trace" \
    --cache "$OUTDIR/$name.cache.json" | tee "$OUTDIR/$name.verdict.txt"
done
echo "ALL ARMS DONE"
