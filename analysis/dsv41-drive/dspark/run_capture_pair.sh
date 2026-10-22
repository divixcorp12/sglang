#!/bin/bash
# Served capture of one or more arms, one after another, each through spec_margin_capture.py with the samplers the
# reports read beside it: memsample.py (OUT/NAME.memsample.jsonl), nvme3n1 diskstats every 2 s (OUT/NAME.nvme3n1.samples)
# and nvidia-smi every 1 s (OUT/NAME.gpu-mem.csv), plus prelaunch_evict.py's floor check, memstate/diskstats around the
# run, and a node-0 watchdog during load. Generalises /mnt/nvme1/share28-ab-20261016/run_ab.sh (2026-10-16).
#
# Usage: run_capture_pair.sh OUT --arm BASE [--worktree DIR] [--env K=V ...] [--hicache-gib 12.67] [--python PY]
#                            [--dry-run] NAME=[K=V ...] [NAME=[K=V ...] ...] [-- CAPTURE_ARGS ...]
#   NAME=K=V K2=V2   an arm: the run dir OUT/NAME, served with these env assignments (--env ones apply to every arm
#                    first). One arm is a diagnostic; arms run in the order given.
#   CAPTURE_ARGS     passed to spec_margin_capture.py, e.g. --no-verify-split --long-prompt $PWD/SAMPLE_PROMPT_DIVIX01.md
#                    --decode-seconds 300 --ttft-repeats 3; none serves the default 8-prompt probe.
#   --worktree       the checkout to run (default: the one holding this script). A dirty one is refused.
#   --hicache-gib    prelaunch_evict.py's HiCache pools on node 1 (the template's -mergechunk-expseg value).
#   --dry-run        resolve and print each arm's env, command and outputs; launch and write nothing.
#
# Every SGLANG_* variable inherited from the shell is unset, except both_cpu_ab.PASSTHROUGH's build caches, and so is
# CUDA_VISIBLE_DEVICES, so an arm runs on its assignments alone. OUT/resolved-env.json records each arm's server env
# (its assignments under the capture's own) and the server argv; an assignment the capture would override is refused.
# The wrapper takes no lock: the capture takes rowimg-disk.lock, then cc-gpu.lock. Run it detached with its output in
# a log, e.g. nohup run_capture_pair.sh ... > OUT.log 2>&1 &

die() { echo "run_capture_pair: $*" >&2; exit 2; }

OUT=""; BASE=""; WT=""; PY=/data/models/slang/.venv-upstream/bin/python; HICACHE_GIB=12.67; DRY=0
# The roots prelaunch_evict.py searches for page-cache holders when evicting its listed files leaves node 1 short (the
# same as arm_env.PROD_PAGE_CACHE_ROOTS).
HOLDERS="--holders /mnt/nvme0 --holders /mnt/nvme1 --holders /mnt/nvme2 --holders /mnt/nvme4 --holders /data/models"
# Node 0's floor for the prelaunch (MemFree + reclaimable slab at the start). A served arm takes every free page of node 0
# during load: in the 13 completed arms of 2026-10-16/17 (memsample.jsonl) node-0 anonymous memory rose 81.6-83.7 GiB and
# MemFree bottomed at 0.5-0.9 GiB, because the kernel gives the page cache back. What differs between a start that loads
# and one that trips the watchdog is the start: the lowest start of a completed arm was 81.6 GiB (engram-spin, prod arm),
# so 80.6 = that less 1 GiB. A hard floor at the anonymous peak (83.7) would refuse three of those completed starts (81.6, 81.9, 83.2).
# Override with NODE0_FLOOR_GIB; 0 turns the node-0 check off.
HOLDERS="$HOLDERS --node0-floor-gib ${NODE0_FLOOR_GIB:-80.6}"
# PRELAUNCH_EVICT_HOLDERS=1 also drops the page cache of those holders at the first retry (prelaunch_evict.py
# --evict-holders; clean pages of files not open for writing). Off by default.
[ "${PRELAUNCH_EVICT_HOLDERS:-0}" = 1 ] && HOLDERS="$HOLDERS --evict-holders"
# PAGE_CACHE_OPT_OUT=1 (owner, 2026-10-18): page cache the kernel can reclaim never stops an arm. The prelaunch check
# still evicts and logs but a refusal prints PRELAUNCH FLOOR BYPASSED and the arm starts; the node-0 load watchdog logs
# WATCHDOG OPT-OUT below the min watermark instead of aborting, and aborts only when more than SWAPIN_ABORT_MIB (default
# 4096) is swapped in during the load. A real OOM still ends the server. Off by default.
PAGE_CACHE_OPT_OUT=${PAGE_CACHE_OPT_OUT:-0}
SWAPIN_ABORT_MIB=${SWAPIN_ABORT_MIB:-4096}
COMMON=(); NAMES=(); SPECS=(); CAP=()
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) BASE=$2; shift 2 ;;
    --worktree) WT=$2; shift 2 ;;
    --env) COMMON+=("$2"); shift 2 ;;
    --hicache-gib) HICACHE_GIB=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; CAP=("$@"); break ;;
    -*) die "unknown option $1" ;;
    *=*) NAMES+=("${1%%=*}"); SPECS+=("${1#*=}"); shift ;;
    *) [ -z "$OUT" ] || die "a second OUT: $1"; OUT=$1; shift ;;
  esac
done
[ -n "$OUT" ] || die "no OUT dir"
[ -n "$BASE" ] || die "no --arm"
[ "${#NAMES[@]}" -gt 0 ] || die "no arm (NAME=[K=V ...])"
OUT=$(realpath -m "$OUT")
WT=$(realpath "${WT:-$(dirname "${BASH_SOURCE[0]}")/../../..}") || die "no worktree"
ASSIGN_RE='^[A-Za-z_][A-Za-z0-9_]*='
for kv in "${COMMON[@]}"; do [[ $kv =~ $ASSIGN_RE ]] || die "--env $kv is not K=V"; done
declare -A SEEN
for i in "${!NAMES[@]}"; do
  [[ ${NAMES[$i]} =~ ^[A-Za-z0-9._-]+$ ]] || die "arm name '${NAMES[$i]}': letters, digits, . _ - only"
  [ -z "${SEEN[${NAMES[$i]}]}" ] || die "arm ${NAMES[$i]} given twice"
  SEEN[${NAMES[$i]}]=1
  for kv in ${SPECS[$i]}; do [[ $kv =~ $ASSIGN_RE ]] || die "arm ${NAMES[$i]}: $kv is not K=V"; done
done

for v in $(compgen -e); do
  case "$v" in
    SGLANG_JIT_CACHE_DIR|SGLANG_EXL3_BUILD_DIR) ;;
    SGLANG_*|CUDA_VISIBLE_DEVICES) echo "unset $v (from the shell)"; unset "$v" ;;
  esac
done
cd "$WT" || die "cannot cd to $WT"
export PYTHONPATH=$WT/python OMP_NUM_THREADS=8 TMPDIR=${OUT}-tmp
CAPTURE=analysis/dsv41-drive/dspark/spec_margin_capture.py
DIRTY=$(git status --porcelain --untracked-files=no)
echo "worktree $WT HEAD $(git rev-parse HEAD)"
git log -1 --oneline
if [ -n "$DIRTY" ]; then
  echo "$DIRTY"
  [ "$DRY" = 1 ] && echo "DIRTY worktree: a real run would refuse it" || { echo "REFUSED: dirty worktree $WT"; echo ALLDONE; exit 2; }
fi
"$PY" -c "import sglang; print(sglang.__file__)"

arm_assignments() { local i=$1; echo "${COMMON[*]} ${SPECS[$i]}"; }

RESOLVED=()
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  read -ra all <<< "$(arm_assignments "$i")"
  keys=$(for kv in "${all[@]}"; do printf '%s,' "${kv%%=*}"; done)
  res=$(env "${all[@]}" "$PY" "$CAPTURE" "$OUT/$name" --arm "$BASE" "${CAP[@]}" --resolve-env "$keys") \
    || die "arm $name: spec_margin_capture.py refused its arguments"
  RESOLVED+=("$name" "$res")
done
MERGED=$("$PY" -c '
import json, sys
out, argv, bad = {}, None, []
for name, raw in zip(sys.argv[1::2], sys.argv[2::2]):
    r = json.loads(raw)
    out[name] = r["env"]
    bad += [f"{name}: {k}" for k in r["overridden"]]
    if argv not in (None, r["argv"]):
        bad.append(f"{name}: server argv differs from the first arm")
    argv = r["argv"]
if bad:
    sys.exit("the capture overrides these arm settings, so the arm would not get them: " + "; ".join(bad))
out["argv"] = argv
print(json.dumps(out, indent=1))
' "${RESOLVED[@]}") || { echo "REFUSED: arm env"; echo ALLDONE; exit 2; }

if [ "$DRY" = 1 ]; then
  for i in "${!NAMES[@]}"; do
    name=${NAMES[$i]}
    echo "ARM $name"
    echo "  env: $(arm_assignments "$i")"
    echo "  prelaunch: taskset -c 22-25 /usr/bin/python3 $WT/analysis/dsv41-drive/dspark/prelaunch_evict.py --log $OUT/$name.prelaunch.json --hicache-gib $HICACHE_GIB $HOLDERS"
    echo "  capture: env $(arm_assignments "$i") taskset -c 0-51,56-107 $PY $CAPTURE $OUT/$name --arm $BASE ${CAP[*]}"
    echo "  outputs: $OUT/$name $OUT/$name.memsample.jsonl $OUT/$name.nvme3n1.samples $OUT/$name.gpu-mem.csv $OUT/$name.memstate.txt $OUT/$name.diskstats.before $OUT/$name.diskstats.after $OUT/$name.ready.txt"
  done
  echo "resolved-env.json: $(echo "$MERGED" | tr -d '\n')"
  echo "DRY RUN: nothing launched"
  exit 0
fi

. "$(dirname "${BASH_SOURCE[0]}")/reap_group.sh"
SERVER_PORT=30032  # spec_margin_capture.PORT
CUR_CAP=""; CUR_NAME=""
# Killed or interrupted (also on a plain exit): stop the capture by PID, give it 40 s to stop its server, then clear the
# server's group and the sampler loops it left.
stop_all() {
  if [ -n "$CUR_CAP" ] && kill -0 "$CUR_CAP" 2>/dev/null; then
    echo "stopping capture $CUR_CAP ($CUR_NAME)"; kill -TERM "$CUR_CAP"
    for ((i = 0; i < 400; i++)); do kill -0 "$CUR_CAP" 2>/dev/null || break; sleep 0.1; done
    kill -0 "$CUR_CAP" 2>/dev/null && kill -KILL "$CUR_CAP"
  fi
  [ -n "$CUR_NAME" ] && { reap_server_group "$OUT/$CUR_NAME/server.pgid" "$CUR_NAME"; touch "$OUT/$CUR_NAME.disk.stop" "$OUT/$CUR_NAME.memsample.stop"; }
}
trap 'stop_all; echo ALLDONE; exit 130' INT TERM
trap 'stop_all' EXIT
mkdir -p "$OUT" "$TMPDIR"
echo "$MERGED" > "$OUT/resolved-env.json"
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  read -ra all <<< "$(arm_assignments "$i")"
  echo "$name ${all[*]}"
  # DISK_LOCK_HELD_BY_CALLER=1: an ancestor (run_sampled_finish_ab.sh) holds rowimg-disk.lock across the arms, as the lock
  # order asks, so the probe for that lock is skipped; the capture treats the ancestor's hold as its own (run_locks.take).
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || pgrep -f "spec_margin_capture|pytest|ninja|cc1plus|nvcc|sglang.launch_server|validate_split_calib|prefill_peak_probe" > /dev/null || { [ "${DISK_LOCK_HELD_BY_CALLER:-0}" != 1 ] && ! flock -n /data/models/slang/nvfp4-work/rowimg-disk.lock true; } || ! flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true; do
    echo "waiting for a quiet box $(date)"; sleep 60; done
  # Layer-major prefill's node-1 StateStore, from the arm's resolved server env (0 when it does not prefill layer-major).
  state_gib=$(echo "$MERGED" | "$PY" -c 'import json, sys; sys.path.insert(0, sys.argv[2]); import arm_env
print(arm_env.layer_major_state_gib(json.load(sys.stdin)[sys.argv[1]]))' "$name" "$WT/benchmarks/dsv41_baseline") \
    || { echo "REFUSED $name: no layer-major StateStore size"; echo ALLDONE; exit 2; }
  if [ "$PAGE_CACHE_OPT_OUT" = 1 ]; then
    # The check still evicts our files and logs every node; a refusal only reports, and the wait is one minute.
    if ! taskset -c 22-25 /usr/bin/python3 $WT/analysis/dsv41-drive/dspark/prelaunch_evict.py --log $OUT/$name.prelaunch.json --hicache-gib $HICACHE_GIB --layer-major-state-gib "$state_gib" $HOLDERS --wait-min 1; then
      echo "PRELAUNCH FLOOR BYPASSED $name: below a floor, starting anyway (PAGE_CACHE_OPT_OUT=1) $(date)"
    fi
  elif ! taskset -c 22-25 /usr/bin/python3 $WT/analysis/dsv41-drive/dspark/prelaunch_evict.py --log $OUT/$name.prelaunch.json --hicache-gib $HICACHE_GIB --layer-major-state-gib "$state_gib" $HOLDERS; then
    echo "REFUSED $name: node 1 free below the floor $(date)"; echo ALLDONE; exit 3
  fi
  mkdir -p $OUT/$name
  { date; free -g; grep -E "^Swap" /proc/meminfo; for n in /sys/devices/system/node/node*/meminfo; do grep -E "MemFree|FilePages" $n; done; } > $OUT/$name.memstate.txt
  cat /proc/diskstats > $OUT/$name.diskstats.before
  rm -f $OUT/$name.memsample.stop $OUT/$name.disk.stop
  (taskset -c 22-25 /usr/bin/python3 $WT/analysis/dsv41-drive/dspark/memsample.py $OUT/$name $OUT/$name.memsample.jsonl $OUT/$name.memsample.stop 5 &)
  (while [ ! -e $OUT/$name.disk.stop ]; do echo "$(date +%s.%N) $(grep -w nvme3n1 /proc/diskstats)"; sleep 2; done > $OUT/$name.nvme3n1.samples &)
  taskset -c 22-25 nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader,nounits -lms 1000 > $OUT/$name.gpu-mem.csv &
  smi=$!
  start=$(date "+%Y-%m-%d %H:%M:%S")
  echo "START $name $BASE $(date)"
  env "${all[@]}" taskset -c 0-51,56-107 "$PY" "$CAPTURE" "$OUT/$name" --arm "$BASE" "${CAP[@]}" &
  cap=$!
  CUR_CAP=$cap; CUR_NAME=$name
  minfree=999999999
  swapin0=$(awk '/^pswpin /{print $2}' /proc/vmstat)
  below=0
  while kill -0 $cap 2>/dev/null && ! grep -q "fired up and ready" $OUT/$name/server.log 2>/dev/null; do
    read free min < <(awk "/^Node 0, zone +Normal/{z=1} z&&/pages free/{f=\$3} z&&/^ +min /{print f, \$2; exit}" /proc/zoneinfo)
    [ -n "$free" ] && [ "$free" -lt "$minfree" ] && minfree=$free
    swapin_mib=$(( ($(awk '/^pswpin /{print $2}' /proc/vmstat) - swapin0) * 4 / 1024 ))
    if [ "$PAGE_CACHE_OPT_OUT" = 1 ]; then
      # Below the min watermark is the kernel reclaiming page cache; swapping in is the load really short of memory.
      if [ -n "$free" ] && [ "$free" -lt "$min" ] && [ "$below" = 0 ]; then
        below=1; echo "WATCHDOG OPT-OUT $name: node 0 Normal free $free pages below min $min during load, continuing (PAGE_CACHE_OPT_OUT=1) $(date)"
      fi
      if [ "$swapin_mib" -gt "$SWAPIN_ABORT_MIB" ]; then
        echo "ABORT $name: $swapin_mib MiB swapped in during load, above $SWAPIN_ABORT_MIB $(date)"; kill $cap; break
      fi
    elif [ -n "$free" ] && [ "$free" -lt "$min" ]; then
      echo "ABORT $name: node 0 Normal free $free pages below min $min during load $(date)"; kill $cap; break
    fi
    sleep 1
  done
  echo "load watchdog $name: node 0 Normal min free seen $minfree pages (min watermark ${min:-?}), swap-in during load $(( ($(awk '/^pswpin /{print $2}' /proc/vmstat) - swapin0) * 4 / 1024 )) MiB"
  grep -m1 -n "fired up and ready" $OUT/$name/server.log | cut -c1-40 > $OUT/$name.ready.txt
  wait $cap; rc=$?
  echo "RC $name $rc $(date)"
  CUR_CAP=""
  # The capture stops its own server on every exit it can handle (spec_margin_capture); this clears what a crash or a
  # SIGKILL left, and does not go on while the port or the GPU is still held.
  cleared=1
  reap_server_group "$OUT/$name/server.pgid" "$name" || cleared=0
  server_clear "$SERVER_PORT" "$name" || cleared=0
  if [ "$cleared" = 0 ]; then
    touch $OUT/$name.disk.stop $OUT/$name.memsample.stop; kill $smi 2>/dev/null
    echo "ARM FAILED $name: its server is still up"; echo ALLDONE; exit 1
  fi
  cat /proc/diskstats > $OUT/$name.diskstats.after; touch $OUT/$name.disk.stop
  sleep 10; touch $OUT/$name.memsample.stop; kill $smi; sleep 15
  journalctl -k --since "$start" 2>/dev/null | grep -E "Killed process|oom-kill:" | tail -3
  grep -hE "Expert hot cache startup|lane head active|DSV4 pool sizes|RAM miss prefill share|RAM miss thread started" $OUT/$name/server.log | cut -c1-600
  echo "OOM lines $name: $(grep -ciE "out of memory|OutOfMemoryError|CUDA error" $OUT/$name/server.log)"
  echo "min free MiB over the run $name: $(sort -t, -k3 -n $OUT/$name.gpu-mem.csv | head -1)"
  echo "trace size $name: $(du -sh $OUT/$name/stages.jsonl 2>/dev/null)"
  [ "$rc" = 0 ] || { echo "ARM FAILED $name"; echo ALLDONE; exit 1; }
done
echo ALLDONE
