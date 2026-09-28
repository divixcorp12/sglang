#!/usr/bin/env bash
# Plan 2026-09-28-reader-crtp-uring-registration Task 5: the refactor-only decode pair. The production recipe's decode
# arm from the base reader (4fe0c37a41, python tree c2de630275) and from the CRTP split (ReaderCore + PackReader +
# RowReader), default knobs in both, each with an nvidia-smi SM-clock sampler (run_arm.sh records the clock only at
# session boundaries). Pass: byte-identical output and pooled ms/token within +-1.5.
# Template: analysis/dsv41-drive/mirror3/drive_mirror3_arms.sh (changes: two worktrees/SHAs; no mirror roots, device
# check or diskstats sampler; an EXL3 launch-gate assertion before each arm; stop on read_errors > 0).
# Lock order: rowimg-disk.lock is held across both arms; cc-gpu.lock is polled here and taken by run_arm.sh itself.
# Take the disk lock only here: launching this script under `flock rowimg-disk.lock` as well self-deadlocks, because
# the `exec 8>`/`flock 8` below opens a second open file description on the same file.
# Usage: drive_reader_crtp_pair.sh <base_worktree> <base_sha> <split_worktree> <split_sha> <out_dir under /mnt/nvme1> [port]
#
# Both arms run the SPLIT worktree's benchmarks/dsv41_baseline/run_arm.sh with DSV41_WORKTREE (and EXPECT_SHA) naming
# the arm's own worktree: run_arm.sh reads generations.json, arm_env and its helpers from its own directory and tests the
# python/ tree of DSV41_WORKTREE. So one generations.json (the split's) registers both trees and the recipe is the same
# file for both arms.
#
# Guards before anything starts: the NVIDIA driver is 615.71.09, nothing listens on production's 7867 or on the arm
# port. Before each arm: the same port check, then wait while a pytest, cc1plus or nvcc process exists or
# cc-gpu.lock is held. Processes are matched by name (pgrep -x) and, for `python -m pytest` (named "python"), by an
# interpreter exe/comm plus the exact cmdline tokens "-m" "pytest" -- never by free text over argv: a waiter doing
# that once deadlocked on its own command line. The driver's own PID and its ancestors are never counted.
# The driver records every PID it starts (sampler, run_arm.sh) in $OUT/driver-pids.txt and, on exit or on INT/TERM
# (and HUP, but only when not started under nohup, which ignores HUP before bash could trap it), kills only those.
# run_arm.sh runs as its own process group, so killing it also stops the server it started.
#
# EXL3 gate: the server's launch gate falls back to the NVFP4 rules silently when <model_path>/config.json cannot be
# read (.claude/rules/divix01-run-protocol.md), and run_arm.sh does not check which requirements resolved. Before each
# arm this driver resolves them with that arm's own python/ tree against arm_env.MODEL_PATH and stops unless the label
# is EXL3 (benchmarks/dsv41_flash/arm_config.py check_gate is the worked example).
#
# Pinned tier (as drive_mirror3_arms.sh): both arms run a tier 4096 MiB smaller on node 0 than arm_env's recipe
# (102400 = 0:61440,1:40960), passed as identical KEY=VAL overrides: SGLANG_MOE_PINNED_HOST_NUMA_MB=0:57344,1:40960
# and SGLANG_MOE_PINNED_HOST_MB=98304. Before the base arm the driver checks node 0 the way check_capacity does
# (MemFree + Active(file) + Inactive(file) from node0/meminfo) against the node-0 share + 4096 MiB headroom + the
# server's observed pre-check node-0 footprint (15360). If short, it cuts node 0 and the total by another 4096 MiB
# once, for both arms. Before the split arm it re-checks the SAME values and stops if short, rather than run mismatched
# arms.
set -u
BASE_WT=${1:?base worktree}; BASE_SHA=${2:?base commit}; SPLIT_WT=${3:?split worktree}; SPLIT_SHA=${4:?split commit}
OUT=${5:?out dir}; PORT=${6:-30031}
PY=/data/models/slang/.venv/bin/python
HERE=$SPLIT_WT/benchmarks/dsv41_baseline  # run_arm.sh, arm_env and generations.json for BOTH arms
RUN_ARM=$HERE/run_arm.sh
PROD_PORT=7867
DRIVER_VERSION=615.71.09
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
# The nine io_uring knobs at the 2026-09-28 campaign's S0 values, set explicitly in both arms (the base tree reads none).
URING_S0=(
    SGLANG_EXPERT_STREAM_URING_MODE=default SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=0
    SGLANG_EXPERT_STREAM_URING_FIXED_FILES=0 SGLANG_EXPERT_STREAM_URING_READ_MODE=normal
    SGLANG_EXPERT_STREAM_URING_WAIT_MODE=block SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS=1000
    SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU=-1 SGLANG_EXPERT_STREAM_URING_SLAB_ARENA=0
    SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS=1
)
say() { echo "$(date +%T) $*"; }
case $OUT in /mnt/nvme1/*) ;; *) say "out dir must be under /mnt/nvme1"; exit 1 ;; esac
check_worktree() {  # <worktree> <sha>: at the commit, clean, sglang imported from it, python/ tree registered
    local wt=$1 sha=$2 tree
    [ "$(git -C "$wt" rev-parse HEAD)" = "$sha" ] || { say "$wt not at $sha"; return 1; }
    [ -z "$(git -C "$wt" status --porcelain)" ] || { say "$wt dirty"; return 1; }
    PYTHONPATH=$wt/python $PY -c "import sglang; print('sglang from', sglang.__file__)" || return 1
    tree=$(git -C "$wt" rev-parse HEAD:python)
    PYTHONPATH=$HERE $PY -c "import generations; print('generation', '$tree', '=', generations.check_registered('$tree'))" \
        || { say "python tree $tree is not registered in $HERE/generations.json"; return 1; }
}
check_worktree "$BASE_WT" "$BASE_SHA" || exit 1
check_worktree "$SPLIT_WT" "$SPLIT_SHA" || exit 1
[ -f "$RUN_ARM" ] || { say "no $RUN_ARM"; exit 1; }
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[ "$driver" = "$DRIVER_VERSION" ] || { say "NVIDIA driver is '$driver', expected $DRIVER_VERSION"; exit 1; }

listening() { ss -ltn "sport = :$1" | grep -q LISTEN; }
ports_free() {
    local p
    for p in "$PROD_PORT" "$PORT"; do
        if listening "$p"; then say "something listens on $p; refusing"; return 1; fi
    done
}
ports_free || exit 1

exl3_gate() {  # <arm> <worktree>: that tree's resolution of the launch's expert-stream requirements must be EXL3
    local arm=$1 wt=$2
    PYTHONPATH=$wt/python:$HERE $PY -c "
import types, arm_env
from sglang.srt.arg_groups.expert_stream_requirements import expert_stream_requirements_for
cfg = types.SimpleNamespace(model_path=arm_env.MODEL_PATH, quantization=None)
label = expert_stream_requirements_for(cfg, cfg).label
print('$arm expert-stream requirements for', arm_env.MODEL_PATH, '=', label)
raise SystemExit(0 if label == 'EXL3' else 'the launch gate resolved ' + repr(label) + ', not EXL3')
" || { say "EXL3 gate failed for $arm ($wt)"; return 1; }
}

N0_MIB=57344; N1_MIB=40960; HEADROOM_MIB=4096; NODE0_CUT_MIB=4096; NODE0_CUTS_LEFT=1
# The server's node-0 footprint before check_capacity runs, observed (see drive_mirror3_arms.sh).
PRECHECK_FOOTPRINT_MIB=15360
node0_memory_mib() {  # "<MemFree> <Active(file)+Inactive(file)>" for node 0 in MiB: host_numa.node_memory's free, reclaimable
    awk '$3 == "MemFree:" { f += $4 } $3 == "Active(file):" || $3 == "Inactive(file):" { c += $4 }
         END { print int(f / 1024), int(c / 1024) }' /sys/devices/system/node/node0/meminfo
}
FIRST_ARM=crtp-base
# At most ONE cut (57344 -> 53248 on node 0, 98304 -> 94208 in total), settled before the first arm; never a second.
# Each check is logged and appended to $OUT/node0-gate.jsonl.
node0_gate() {  # <arm>: settle (first arm, may cut once) or re-check (second arm, never cuts) the node-0 share
    local arm=$1 free cache avail need ok
    while :; do
        read -r free cache < <(node0_memory_mib)
        avail=$((free + cache)); need=$((N0_MIB + HEADROOM_MIB + PRECHECK_FOOTPRINT_MIB))
        ok=$([ "$avail" -ge "$need" ] && echo true || echo false)
        say "node 0 before $arm: MemFree ${free} MiB + page cache ${cache} MiB = ${avail} MiB, need ${need} MiB" \
            "(share ${N0_MIB} + headroom ${HEADROOM_MIB} + pre-check footprint ${PRECHECK_FOOTPRINT_MIB}): ok=$ok"
        printf '{"arm": "%s", "utc": "%s", "memfree_mib": %d, "page_cache_mib": %d, "available_mib": %d, "need_mib": %d, "node0_share_mib": %d, "node1_share_mib": %d, "total_mib": %d, "ok": %s}\n' \
            "$arm" "$(date -u +%FT%TZ)" "$free" "$cache" "$avail" "$need" "$N0_MIB" "$N1_MIB" "$((N0_MIB + N1_MIB))" "$ok" \
            >> "$OUT/node0-gate.jsonl"
        [ "$ok" = true ] && return 0
        if [ "$arm" = "$FIRST_ARM" ] && [ "$NODE0_CUTS_LEFT" -gt 0 ]; then
            N0_MIB=$((N0_MIB - NODE0_CUT_MIB)); NODE0_CUTS_LEFT=$((NODE0_CUTS_LEFT - 1))
            say "node 0 short: cutting node 0 (and the total) by ${NODE0_CUT_MIB} MiB for BOTH arms -> 0:${N0_MIB},1:${N1_MIB} (total $((N0_MIB + N1_MIB)))"
            continue
        fi
        say "node 0 short before $arm at 0:${N0_MIB},1:${N1_MIB} (${avail} < ${need} MiB); no further cut, stopping before this arm"
        return 1
    done
}
tier_overrides() {
    echo "SGLANG_MOE_PINNED_HOST_NUMA_MB=0:${N0_MIB},1:${N1_MIB}" "SGLANG_MOE_PINNED_HOST_MB=$((N0_MIB + N1_MIB))"
}
mkdir -p "$OUT"

# PIDs this driver started, with their start times so a recycled PID is never killed.
PIDS_FILE=$OUT/driver-pids.txt
declare -A STARTED=()
starttime() { local s; s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1; s=${s##*) }; set -- $s; echo "${20}"; }
track() {  # <pid> <what>
    STARTED[$1]=$(starttime "$1")
    echo "$1 ${STARTED[$1]} $2" >> "$PIDS_FILE"
}
stop_pid() {  # <pid>: TERM it (its whole group for run_arm.sh), only if it is still the process we started
    local pid=$1 group=${2:-}
    [ -n "${STARTED[$pid]:-}" ] && [ "$(starttime "$pid")" = "${STARTED[$pid]}" ] || { unset "STARTED[$pid]"; return 0; }
    if [ -n "$group" ]; then kill -TERM -- "-$pid" 2>/dev/null; else kill -TERM "$pid" 2>/dev/null; fi
    unset "STARTED[$pid]"
}
ARM_PID=""
cleanup() {
    local pid
    [ -z "$ARM_PID" ] || { say "stopping run_arm.sh group $ARM_PID"; stop_pid "$ARM_PID" group; }
    for pid in "${!STARTED[@]}"; do stop_pid "$pid"; done
}
trap cleanup EXIT
# HUP is trappable only when the driver was not started under nohup (bash cannot trap a signal ignored on entry).
trap 'say "interrupted"; exit 130' INT TERM HUP

exec 8>"$DISK_LOCK"
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"

# This shell and every ancestor, so no process on the driver's own chain is ever counted as foreign.
MINE=" $$ $BASHPID "
p=$$
while p=$(awk '/^PPid:/ {print $2}' "/proc/$p/status" 2>/dev/null) && [ -n "$p" ] && [ "$p" != 0 ]; do MINE+="$p "; done

# PIDs of `<python> ... -m pytest ...`: the exe basename (or, when /proc/<pid>/exe is unreadable, comm) is a python
# interpreter AND /proc/<pid>/cmdline, split on NUL, holds the exact token "-m" followed by the exact token "pytest".
PYTEST_SCAN='
import os, re, sys
mine = set(sys.argv[1:]) | {str(os.getpid())}
py = re.compile(r"python[0-9.]*")
for pid in filter(str.isdigit, os.listdir("/proc")):
    if pid in mine:
        continue
    try:
        try:
            name = os.path.basename(os.readlink(f"/proc/{pid}/exe"))
        except OSError:
            name = open(f"/proc/{pid}/comm").read().strip()
        if not py.fullmatch(name):
            continue
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except OSError:
        continue
    if any(a == b"-m" and b == b"pytest" for a, b in zip(argv, argv[1:])):
        print(pid)
'
python_pytest() { $PY -c "$PYTEST_SCAN" $MINE; }

wait_for_quiet_box() {
    local busy
    while :; do
        busy=$( { pgrep -x 'pytest|cc1plus|nvcc'; python_pytest; } | while read -r p; do
                   [[ $MINE == *" $p "* ]] || echo "$p"; done | tr '\n' ' ')
        if [ -n "$busy" ]; then
            say "foreign pytest/cc1plus/nvcc running (pids $busy); waiting"
        elif ! flock -n "$GPU_LOCK" true; then
            say "cc-gpu.lock held; waiting"
        else
            return 0
        fi
        sleep 60
    done
}

run_dir_of() { sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/$1-run_arm.log" | head -1; }

read_errors_of() {  # <run_dir>: the RAM-miss service's read_errors from server.log's shutdown counters (empty if absent)
    PYTHONPATH=$SPLIT_WT/analysis/dsv41-drive/mirror3 $PY -c "
import sys, mirror3_report as m
t = m.ram_miss(sys.argv[1])['thread']
print('' if t is None or t.get('read_errors') is None else t['read_errors'])
" "$1"
}

run_one() {  # <arm> <worktree> <sha> [KEY=VAL ...]
    local arm=$1 wt=$2 sha=$3; shift 3
    local done_log=$OUT/$arm.done rc pid_c run errors
    : > "$done_log"
    ports_free || return 1
    exl3_gate "$arm" "$wt" || return 1
    wait_for_quiet_box
    node0_gate "$arm" || return 1
    set -- "$@" $(tier_overrides)
    say "arm $arm worktree $wt at $sha overrides: $*"
    # 8>&- everywhere: no child may inherit (and outlive the driver holding) rowimg-disk.lock.
    setsid nohup taskset -c 20-23 nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm --format=csv -lms 1000 \
        > "$OUT/$arm-clocks.csv" 2> "$OUT/$arm-clocks.err" < /dev/null 8>&- &
    pid_c=$!; track "$pid_c" "$arm clock sampler"
    sleep 3
    kill -0 "$pid_c" 2>/dev/null || { say "clock sampler died: $(cat "$OUT/$arm-clocks.err")"; return 1; }
    say "arm $arm start"
    # setsid execs in place (a background job is not a group leader), so $! is run_arm.sh and its process group.
    EXPECT_SHA=$sha DSV41_WORKTREE=$wt setsid bash "$RUN_ARM" "$arm" "$PORT" "$@" \
        > "$OUT/$arm-run_arm.log" 2>&1 < /dev/null 8>&- &
    ARM_PID=$!; track "$ARM_PID" "$arm run_arm.sh"
    wait "$ARM_PID"; rc=$?
    ARM_PID=""
    echo "DRIVER DONE rc=$rc" >> "$done_log"
    stop_pid "$pid_c"
    run=$(run_dir_of "$arm")
    say "arm $arm rc=$rc run: $run"
    [ "$rc" = 0 ] || return "$rc"
    errors=$(read_errors_of "$run")
    say "arm $arm read_errors=${errors:-<no counters in server.log>}"
    [ "$errors" = 0 ] || { say "arm $arm: read_errors is '${errors:-missing}', not 0; stopping"; return 1; }
}

run_one crtp-base "$BASE_WT" "$BASE_SHA" "${URING_S0[@]}" \
    || { say "base arm failed; not running the split arm (see $OUT/crtp-base-run_arm.log)"; say "DRIVER DONE rc=1"; exit 1; }
run_one crtp-split "$SPLIT_WT" "$SPLIT_SHA" "${URING_S0[@]}"
rc=$?
base=$(run_dir_of crtp-base)
split=$(run_dir_of crtp-split)
say "base run: $base"
say "split run: $split"
if [ "$rc" = 0 ]; then
    # mirror3_report.py's own functions (no diskstats here): pooled ms/token, TTFT, byte identity, counters, clocks.
    PYTHONPATH=$SPLIT_WT/analysis/dsv41-drive/mirror3 $PY -c "
import json, sys
import mirror3_report as m
def arm(run_dir, clocks_csv):
    return {'run_dir': run_dir, 'node0_gate': m.node0_gate(clocks_csv, run_dir), 'decode': m.decode(run_dir),
            'ms_per_token_median_turn': m.ms_per_token(run_dir)[1],
            'clocks': {'timed_window': m.clock_summary(clocks_csv, *m.timed_window_utc(run_dir)),
                       'session_start_end_mhz': m.session_clocks(run_dir)},
            'ram_miss': m.ram_miss(run_dir)}
out = {'base': arm('$base', '$OUT/crtp-base-clocks.csv'), 'split': arm('$split', '$OUT/crtp-split-clocks.csv'),
       'identical': m.identity('$base', '$split')}
out['all_identical'] = all(out['identical'].values())
out['delta_pooled_ms_per_token'] = out['split']['decode']['pooled_ms_per_token'] - out['base']['decode']['pooled_ms_per_token']
json.dump(out, open('$OUT/pair-report.json', 'w'), indent=2)
print('all_identical', out['all_identical'], 'base', out['base']['decode']['pooled_ms_per_token'],
      'split', out['split']['decode']['pooled_ms_per_token'], 'delta', out['delta_pooled_ms_per_token'])
" || rc=1
    say "report: $OUT/pair-report.json"
fi
say "DRIVER DONE rc=$rc"
exit "$rc"
