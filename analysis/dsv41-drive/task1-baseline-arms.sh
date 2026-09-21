#!/usr/bin/env bash
# Task 1 matched baselines (docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md).
#
# Graph decode, GRAPH_GATHER=1, sessions 0-3, 256 prompt / 128 new tokens, 70 GiB pinned tier:
# the workload of run-native-mirror-arm.sh, from clean detached worktrees instead of wt-dsv41.
#
# Usage: [DRY=1] EXPECT_NEW=<sha> REFERENCE=<clean-reference.json> task1-baseline-arms.sh <label> <run-dir> <code>:<mirror>:<trace>...
# REFERENCE is REQUIRED and every arm's python/ tree must be in its "generations" (see generation_gate); else exit 5 before any arm.
# Exit 6 before any arm if wt-task1-new is not clean at EXPECT_NEW or the runtime copies of this script and the verdict differ from their blobs there (harness_gate).
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
SELF=$(readlink -f "$0")   # before env-full.sh cd's: a relative $0 stops resolving after it (task1e's header printed a blank script sha)

CC=/data/models/slang/nvfp4-work/cc-expert-prediction
. $CC/analysis/dsv41-phase3b/env-full.sh   # defines $PY $FULL $ANA; it cd's into wt-dsv41, overridden below

WT_OLD=$CC/wt-task1-old;  HEAD_OLD=099eadba33879b705b860dad9439f2f01bdd6d06
WT_NEW=$CC/wt-task1-new;  HEAD_NEW=${EXPECT_NEW:?set EXPECT_NEW to the full sha wt-task1-new must be at}  # a commit cannot name itself
SESSIONS=/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl
MIRRORS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash
EXPERT_SHARD_DIRS=(/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw /mnt/nvme0/dsv41_flash /mnt/nvme4/dsv41_flash)
VERDICT=$CC/analysis/dsv41-drive/task1_arm_verdict.py   # a copy beside this script: neither worktree holds it
# Keyed by NAME, correct only while /mnt/nvme4 is nvme3n1 (there is no nvme4 device); provenance.resolve_devices() resolves by st_dev.
declare -A DEV=( [nvme0]=nvme0n1 [nvme2]=nvme2n1 [nvme4]=nvme3n1 )

LABEL="$1"; shift
OUTDIR="$1"; shift
mkdir -p "$OUTDIR"
# A completion sentinel, so a watcher can poll for one file over short connections instead of
# holding a long-lived one that a dropped ssh takes down with it. It is written on every way out.
specs=("$@"); idx=0
SENTINEL="$OUTDIR/$LABEL.sentinel"
rm -f "$SENTINEL"
trap 'rc_exit=$?; { echo "status=$([ $rc_exit -eq 0 ] && echo DONE || echo ABORTED) exit=$rc_exit finished=$(date --iso-8601=seconds)"; echo "arms_started=$idx of ${#specs[@]}"; } > "$SENTINEL"' EXIT
trap 'exit 130' INT TERM
sha256_of() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
ENVFILE=$CC/analysis/dsv41-phase3b/env-full.sh; GPURUN=$ANA/gpu-run.sh   # in no repo: the hashes below are the only record of them
RUNTIME_SHA256=$(printf '{"arm_script": "%s", "verdict": "%s", "env_full_sh": "%s", "gpu_run_sh": "%s"}' \
  "$(sha256_of "$SELF")" "$(sha256_of "$VERDICT")" "$(sha256_of "$ENVFILE")" "$(sha256_of "$GPURUN")")
echo "script sha256:$(sha256_of "$SELF" | cut -c1-16)  verdict sha256:$(sha256_of "$VERDICT" | cut -c1-16)  env-full.sh sha256:$(sha256_of "$ENVFILE" | cut -c1-16)  gpu-run.sh sha256:$(sha256_of "$GPURUN" | cut -c1-16)  host: $(hostname)"

sectors() { awk -v d="$1" '$3==d {print $6}' /proc/diskstats; }

# Resident page-cache bytes of the expert shards (mincore only; reads no data), per directory as a
# json object {"<dir>": bytes}, so a failure names the directory that grew.
resident_by_dir_json() {
  local d sep="" res
  printf '{'
  for d in "${EXPERT_SHARD_DIRS[@]}"; do
    res=$(find "$d" -type f -name '*.safetensors' -print0 2>/dev/null | xargs -0 -r fincore -b -n --raw -o RES 2>/dev/null | awk '{s+=$1} END {print s+0}')
    printf '%s"%s": %s' "$sep" "$d" "$res"; sep=", "
  done
  printf '}'
}
RESIDENCY_DIRS=$(IFS=:; echo "${EXPERT_SHARD_DIRS[*]}")
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

# Generation gate. REFERENCE (clean-reference.json) must be set, and the python/ tree of every arm's code must be one of its
# "generations", or no arm runs. Checked for the whole list before the first arm so a bad tail cannot cost GPU time, and with DRY=1
# too. Not overridable: an unregistered tree is a new code generation that would be compared as if it were an old one.
# (Before this gate the manifest was advisory: an unknown tree printed "GENERATION unknown" and the arm was still VALID.)
# Harness gate. The harness (trace_corpus.py, provenance.py, drive_conditions.py) always runs from $WT_NEW, for old arms too, so $WT_NEW must be
# at EXPECT_NEW and clean whatever the arm's code is, and the hand-copied runtime files (this script, the verdict) must equal their blobs at
# EXPECT_NEW. Before this, an old arm's harness commit and cleanliness were neither recorded nor checked (PIPELINE_BASELINE.md section 7.4).
# Exit 6. Also under DRY=1. Not overridable.
harness_gate() {
  local head dirty pair f rel
  head=$(git -C "$WT_NEW" rev-parse HEAD 2>/dev/null)
  [ "$head" = "$HEAD_NEW" ] || { echo "REFUSE: harness worktree $WT_NEW is at ${head:-<unreadable>}, expected EXPECT_NEW=$HEAD_NEW (the harness runs from it for old arms too)"; return 1; }
  dirty=$(git -C "$WT_NEW" status --porcelain --untracked-files=no | wc -l)
  [ "$dirty" = 0 ] || { echo "REFUSE: harness worktree $WT_NEW has $dirty tracked changes"; return 1; }
  for pair in "$SELF:analysis/dsv41-drive/task1-baseline-arms.sh" "$VERDICT:analysis/dsv41-drive/task1_arm_verdict.py"; do
    f=${pair%%:*}; rel=${pair#*:}
    git -C "$WT_NEW" cat-file -e "$HEAD_NEW:$rel" 2>/dev/null || { echo "REFUSE: $rel does not exist at $HEAD_NEW, so the runtime copy $f cannot be checked"; return 1; }
    [ "$(sha256_of "$f")" = "$(git -C "$WT_NEW" show "$HEAD_NEW:$rel" | sha256sum | cut -c1-64)" ] \
      || { echo "REFUSE: runtime copy $f differs from $rel at $HEAD_NEW (a stale or edited hand copy); copy the blob from that commit"; return 1; }
  done
  echo "harness ok: $WT_NEW at $HEAD_NEW, clean; arm script and verdict equal their blobs there"
}
harness_gate || { echo "NO ARM RUN (harness gate)"; exit 6; }

generation_gate() {  # <worktree> <sha>
  local wt="$1" want="$2" tree label
  if [ -z "${REFERENCE:-}" ] || [ ! -r "$REFERENCE" ]; then
    echo "REFUSE: REFERENCE is unset or unreadable (${REFERENCE:-unset}); no arm's code generation can be checked. Set REFERENCE=<path to clean-reference.json>."
    return 1
  fi
  tree=$(git -C "$wt" rev-parse "$want:python" 2>/dev/null) || { echo "REFUSE: cannot resolve $want:python in $wt"; return 1; }
  label=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("generations", {}).get(sys.argv[2], ""))' "$REFERENCE" "$tree") || return 1
  if [ -z "$label" ]; then
    echo "REFUSE: python tree $tree of $want is in no generation of $REFERENCE. Register it first (key = git rev-parse <sha>:python)."
    return 1
  fi
  echo "generation ok: $want python tree $tree = ${label:0:70}"
}
for spec in "${specs[@]}"; do
  IFS=: read -r code _ _ <<<"$spec"
  case "$code" in old) gwt=$WT_OLD; gwant=$HEAD_OLD ;; new) gwt=$WT_NEW; gwant=$HEAD_NEW ;; *) echo "bad code in $spec"; exit 2 ;; esac
  generation_gate "$gwt" "$gwant" || { echo "NO ARM RUN (generation gate, arm spec $spec)"; exit 5; }
done

for spec in "${specs[@]}"; do
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
       --prompt-tokens 256 --new-tokens 128 --graphs --residency-dirs "$RESIDENCY_DIRS" --out "$OUTDIR/$name.json")

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
  cached0=$(meminfo_cached_kb); res0=$(resident_by_dir_json)
  declare -A before; for k in nvme0 nvme2 nvme4; do before[$k]=$(sectors "${DEV[$k]}"); done
  t0=$(date +%s)
  $ANA/gpu-run.sh env "${env_args[@]}" "${cmd[@]}" > "$OUTDIR/$name.log" 2>&1
  rc=$?
  wall=$(( $(date +%s) - t0 ))
  cached1=$(meminfo_cached_kb); res1=$(resident_by_dir_json)
  {
    printf '{"arm": "%s", "rc": %s, "wall_s": %s, "meminfo_cached_kb_before": %s, "meminfo_cached_kb_after": %s,' \
      "$name" "$rc" "$wall" "$cached0" "$cached1"
    printf ' "expert_resident_by_dir_before": %s, "expert_resident_by_dir_after": %s, "diskstats_read_bytes": {' "$res0" "$res1"
    sep=""; for k in nvme0 nvme2 nvme4; do
      printf '%s"%s": %s' "$sep" "$k" $(( ($(sectors "${DEV[$k]}") - ${before[$k]}) * 512 )); sep=", "
    done
    printf '}, "runtime_sha256": %s}\n' "$RUNTIME_SHA256"
  } > "$OUTDIR/$name.cache.json"
  echo "rc=$rc wall=${wall}s"
  "$PY" "$VERDICT" "$OUTDIR/$name.json" --root "$wt" --head "$want" --mirror "$mirror" --trace "$trace" \
    --cache "$OUTDIR/$name.cache.json" --code "$code" ${REFERENCE:+--reference "$REFERENCE"} --summary-json "$OUTDIR/$name.regime.json" > "$OUTDIR/$name.verdict.txt"
  vrc=$?
  cat "$OUTDIR/$name.verdict.txt"
  # Fail fast: an arm that cannot be a baseline poisons nothing yet, but every arm after it that
  # shares its cause would burn ~400 GiB of reads. Completed arms and this one's files stay in place.
  if [ "$rc" -ne 0 ] || [ "$vrc" -ne 0 ]; then
    echo "ABORT: arm $name failed (run rc=$rc, verdict exit=$vrc). Reasons:"
    grep -E '^(PROBLEM|UNREADABLE|INVALID)' "$OUTDIR/$name.verdict.txt" | sed 's/^/    /'
    echo "  earlier arms of this run are kept in $OUTDIR: $(ls "$OUTDIR"/"$LABEL"-*.verdict.txt 2>/dev/null | grep -v "/$name.verdict.txt" | xargs -r -n1 basename | tr '\n' ' ')"
    echo "  NOT RUN: ${specs[*]:idx}"
    exit 4
  fi
done
echo "ALL ARMS DONE"
