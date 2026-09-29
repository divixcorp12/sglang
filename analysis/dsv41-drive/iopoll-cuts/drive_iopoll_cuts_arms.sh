#!/usr/bin/env bash
# Plan 2026-09-28-iopoll-read-cuts Task 7: the IOPOLL wait fix and read-cut decode arms. One worktree, one commit, arms
# in this order: A (master default: cuts off, MODE=default), B (READ_CUTS=1, MODE=default), C (READ_CUTS=1,
# MODE=iopoll, whose WAIT_MODE=block the wait fix turns into the reap loop), D (READ_CUTS=1, MODE=sqpoll_iopoll,
# SQ_THREAD_CPU=20, the 10 s idle default) and A2 (as A: drift control). All ten SGLANG_EXPERT_STREAM_URING_* knobs are
# passed explicitly in every arm as run_arm.sh KEY=VAL overrides. ARMS="A B" (space-separated) runs a subset, e.g. the
# confirmation pass.
# Rule: an arm is promoted only if it beats mean(A, A2) by >= 1.5 ms/token pooled with byte-identical output.
# Template: analysis/dsv41-drive/uring-reg/drive_uring_reg_arms.sh (changes: five arms; check_modes reads the
# read_cuts=/effective_wait= fields and one `read cuts:` line per drive, whose cut_bytes must be 262144 on the SPCC and
# 520192 on both Samsungs; a per-thread sampler (thread_sampler.py) of the server process tree gives io-wq, SQ-thread and
# RAM-miss-thread CPU over the timed window; no fixed_reads line check). Everything else is the template's: the NVIDIA
# driver and port guards, the generations gate, the EXL3 gate before every arm, exe-name foreign-process gates, the disk
# lock taken here (never under an outer flock), the tier (full 0:61440,1:40960 if node 0 >= share+4096+15360 and
# node 1 >= share+4096+10000, else 0:51200,1:40960; settled before A, applied to every arm), PID tracking and cleanup,
# memory and startup samplers, read_errors == 0.
# Lock order: rowimg-disk.lock is held across all arms; cc-gpu.lock is polled here and taken by run_arm.sh itself.
# Usage: drive_iopoll_cuts_arms.sh <worktree> <sha> <out_dir under /mnt/nvme1> [port]
set -u
WT=${1:?worktree}; SHA=${2:?commit}; OUT=${3:?out dir}; PORT=${4:-30031}
PY=/data/models/slang/.venv/bin/python
HERE=$WT/benchmarks/dsv41_baseline  # run_arm.sh, arm_env and generations.json
RUN_ARM=$HERE/run_arm.sh
PROD_PORT=7867
DRIVER_VERSION=615.71.09
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
URING_A=(
    SGLANG_EXPERT_STREAM_URING_MODE=default SGLANG_EXPERT_STREAM_URING_READ_CUTS=0
    SGLANG_EXPERT_STREAM_URING_WAIT_MODE=block SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH=0
    SGLANG_EXPERT_STREAM_URING_FIXED_FILES=0 SGLANG_EXPERT_STREAM_URING_READ_MODE=normal
    SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS=10000 SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU=-1
    SGLANG_EXPERT_STREAM_URING_SLAB_ARENA=0 SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS=1
)
with_() {  # URING_A with the given KEY=VAL entries replaced
    local out=() kv o
    for kv in "${URING_A[@]}"; do
        for o in "$@"; do [ "${o%%=*}" = "${kv%%=*}" ] && kv=$o; done
        out+=("$kv")
    done
    echo "${out[@]}"
}
URING_B=($(with_ SGLANG_EXPERT_STREAM_URING_READ_CUTS=1))
URING_C=($(with_ SGLANG_EXPERT_STREAM_URING_READ_CUTS=1 SGLANG_EXPERT_STREAM_URING_MODE=iopoll))
URING_D=($(with_ SGLANG_EXPERT_STREAM_URING_READ_CUTS=1 SGLANG_EXPERT_STREAM_URING_MODE=sqpoll_iopoll \
                 SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU=20))
read -r -a ARMS <<< "${ARMS:-A B C D A2}"
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
check_worktree "$WT" "$SHA" || exit 1
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

# Tier: settled once before the first arm, applied to every arm,
# never cut afterwards. Gates on MemFree + page cache (Active(file) + Inactive(file)) per node:
#   full 0:61440,1:40960 / 102400 if node 0 >= 61440+4096+15360 = 80896 and node 1 >= 40960+4096+10000 = 55056 MiB
#   90g  0:51200,1:40960 /  92160 if node 0 >= 51200+4096+15360 = 70656 and node 1 >= 55056 MiB
#   otherwise stop. Before each later arm the settled tier's gates are re-checked; short means stop.
# The node-1 footprint (10000) is observed: uring-reg's first R0 (/mnt/nvme1/uring-reg/memory.jsonl) went from node-1
# MemFree 51231 MiB before launch to 1193 MiB at "ready" with a 40960 MiB node-1 share, i.e. ~9078 MiB beyond it.
TIER_NAMES=(full 90g); TIER_N0=(61440 51200); TIER_IDX=0
TIER_NAME=${TIER_NAMES[0]}; N0_MIB=${TIER_N0[0]}; N1_MIB=40960; HEADROOM_MIB=4096
# The server's node-0 footprint before check_capacity runs, observed (see drive_mirror3_arms.sh).
PRECHECK_FOOTPRINT_MIB=15360
NODE1_FOOTPRINT_MIB=10000
node_memory_mib() {  # <node>: "<MemFree> <Active(file)+Inactive(file)>" in MiB: host_numa.node_memory's free, reclaimable
    awk '$3 == "MemFree:" { f += $4 } $3 == "Active(file):" || $3 == "Inactive(file):" { c += $4 }
         END { print int(f / 1024), int(c / 1024) }' "/sys/devices/system/node/node$1/meminfo"
}
FIRST_ARM=${ARMS[0]}
node0_gate() {  # <arm>: the first arm settles the tier (full, else 90g); later arms re-check it; no other cut
    local arm=$1 free cache avail need ok free1 cache1 avail1 need1 ok1
    while :; do
        read -r free cache < <(node_memory_mib 0)
        read -r free1 cache1 < <(node_memory_mib 1)
        avail=$((free + cache)); need=$((N0_MIB + HEADROOM_MIB + PRECHECK_FOOTPRINT_MIB))
        avail1=$((free1 + cache1)); need1=$((N1_MIB + HEADROOM_MIB + NODE1_FOOTPRINT_MIB))
        ok=$([ "$avail" -ge "$need" ] && echo true || echo false)
        ok1=$([ "$avail1" -ge "$need1" ] && echo true || echo false)
        say "node 0 before $arm (tier $TIER_NAME): MemFree ${free} MiB + page cache ${cache} MiB = ${avail} MiB," \
            "need ${need} MiB (share ${N0_MIB} + headroom ${HEADROOM_MIB} + pre-check footprint ${PRECHECK_FOOTPRINT_MIB}): ok=$ok"
        say "node 1 before $arm (tier $TIER_NAME): MemFree ${free1} MiB + page cache ${cache1} MiB = ${avail1} MiB," \
            "need ${need1} MiB (share ${N1_MIB} + headroom ${HEADROOM_MIB} + footprint ${NODE1_FOOTPRINT_MIB}): ok=$ok1"
        printf '{"arm": "%s", "utc": "%s", "tier": "%s", "memfree_mib": %d, "page_cache_mib": %d, "available_mib": %d, "need_mib": %d, "node1_memfree_mib": %d, "node1_page_cache_mib": %d, "node1_available_mib": %d, "node1_need_mib": %d, "node0_share_mib": %d, "node1_share_mib": %d, "total_mib": %d, "ok": %s, "node1_ok": %s}\n' \
            "$arm" "$(date -u +%FT%TZ)" "$TIER_NAME" "$free" "$cache" "$avail" "$need" "$free1" "$cache1" "$avail1" "$need1" \
            "$N0_MIB" "$N1_MIB" "$((N0_MIB + N1_MIB))" "$ok" "$ok1" >> "$OUT/node0-gate.jsonl"
        [ "$ok" = true ] && [ "$ok1" = true ] && return 0
        if [ "$arm" = "$FIRST_ARM" ] && [ $((TIER_IDX + 1)) -lt ${#TIER_N0[@]} ]; then
            TIER_IDX=$((TIER_IDX + 1)); TIER_NAME=${TIER_NAMES[$TIER_IDX]}; N0_MIB=${TIER_N0[$TIER_IDX]}
            say "short for the full tier: using the $TIER_NAME tier for BOTH arms -> 0:${N0_MIB},1:${N1_MIB} (total $((N0_MIB + N1_MIB)))"
            continue
        fi
        say "a node is short before $arm at 0:${N0_MIB},1:${N1_MIB}; stopping before this arm"
        return 1
    done
}
tier_overrides() {
    echo "SGLANG_MOE_PINNED_HOST_NUMA_MB=0:${N0_MIB},1:${N1_MIB}" "SGLANG_MOE_PINNED_HOST_MB=$((N0_MIB + N1_MIB))"
}

memory_sample() {  # <label>: node-0/node-1 MemFree and VmallocUsed/Slab (MiB) appended to $OUT/memory.jsonl
    local label=$1 n0 n1 vm slab
    n0=$(awk '$3 == "MemFree:" { print int($4 / 1024) }' /sys/devices/system/node/node0/meminfo)
    n1=$(awk '$3 == "MemFree:" { print int($4 / 1024) }' /sys/devices/system/node/node1/meminfo)
    vm=$(awk '$1 == "VmallocUsed:" { print int($2 / 1024) }' /proc/meminfo)
    slab=$(awk '$1 == "Slab:" { print int($2 / 1024) }' /proc/meminfo)
    printf '{"label": "%s", "utc": "%s", "tier": "%s", "node0_memfree_mib": %d, "node1_memfree_mib": %d, "vmalloc_used_mib": %d, "slab_mib": %d}\n' \
        "$label" "$(date -u +%FT%TZ)" "$TIER_NAME" "$n0" "$n1" "$vm" "$slab" >> "$OUT/memory.jsonl"
    say "memory $label: node0 MemFree ${n0} MiB, node1 MemFree ${n1} MiB, VmallocUsed ${vm} MiB, Slab ${slab} MiB"
}
mkdir -p "$OUT"
# Startup sampler (both arms), from launch until "ready" or run_arm.sh's exit, every ~10 s:
#   $OUT/<arm>-startup-mem.jsonl: node-0/node-1 MemFree and /proc/vmstat's compaction, direct reclaim and THP-fallback
#   counters (allocstall_* raw and summed), to tell reclaim from compaction if startup stalls;
#   $OUT/<arm>-stacks.txt, once a minute: /proc/<server pid>/task/*/stack when readable (else one "not readable" line)
#   plus each task's state, wchan and stime, from the server pid run_arm.sh logs ("server pid=N").
startup_sampler() {  # <arm>
    local arm=$1 mem=$OUT/$1-startup-mem.jsonl stacks=$OUT/$1-stacks.txt n=0 spid="" t
    while :; do
        awk -v utc="$(date -u +%FT%TZ)" -v n0="$(awk '$3 == "MemFree:" { print int($4 / 1024) }' /sys/devices/system/node/node0/meminfo)" \
            -v n1="$(awk '$3 == "MemFree:" { print int($4 / 1024) }' /sys/devices/system/node/node1/meminfo)" '
            /^(compact_stall|compact_fail|compact_success|pgscan_direct|pgsteal_direct|thp_fault_fallback|thp_collapse_alloc_failed) / { v[$1] = $2 }
            /^allocstall_/ { v[$1] = $2; stall += $2 }
            END { printf "{\"utc\": \"%s\", \"node0_memfree_mib\": %d, \"node1_memfree_mib\": %d, \"allocstall_sum\": %d", utc, n0, n1, stall
                  for (k in v) printf ", \"%s\": %d", k, v[k]; print "}" }' /proc/vmstat >> "$mem"
        [ -n "$spid" ] || spid=$(sed -n 's/^server pid=\([0-9]*\).*/\1/p' "$OUT/$arm-run_arm.log" 2>/dev/null | head -1)
        if [ -n "$spid" ] && [ $((n % 6)) = 0 ] && [ -d "/proc/$spid" ]; then
            {
                echo "=== $(date -u +%FT%TZ) server pid $spid"
                for t in /proc/"$spid"/task/*; do
                    echo "--- task ${t##*/} $(awk '{ s = $0; sub(/.*\) /, "", s); split(s, f, " "); print "state=" f[1], "stime=" f[13] }' "$t/stat" 2>/dev/null) wchan=$(cat "$t/wchan" 2>/dev/null) comm=$(cat "$t/comm" 2>/dev/null)"
                    cat "$t/stack" 2>/dev/null || echo "    (stack not readable)"
                done
            } >> "$stacks"
        fi
        n=$((n + 1))
        sleep 10
    done
}

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
    PYTHONPATH=$WT/analysis/dsv41-drive/mirror3 $PY -c "
import sys, mirror3_report as m
t = m.ram_miss(sys.argv[1])['thread']
print('' if t is None or t.get('read_errors') is None else t['read_errors'])
" "$1"
}

# The reader's refusals: refuse() ("expert stream registering fixed buffers (regions=...): <why>"), error()
# ("expert stream <operation>: <strerror> (-<errno>)", e.g. registering fixed files), and a failed ring reset.
REFUSAL='expert stream registering fixed buffers|expert stream [^:]+: [^(]+ \(-[0-9]+\)|io_uring ring reset failed'
uring_lines() {  # <arm> <run_dir>: keep the server's io_uring lines; flag a missing fixed_reads line
    local arm=$1 run=$2 f=$OUT/$1-uring.txt
    grep -h 'expert stream io_uring:' "$run/server.log" > "$f" 2>/dev/null
    say "arm $arm io_uring lines ($f):"; sed 's/^/    /' "$f"
}
check_modes() {  # <arm> <run_dir>: the setup line and the per-drive read-cut lines must show the arm's settings
    local arm=$1 run=$2 line cuts
    line=$(grep -m1 'expert stream io_uring: mode=' "$run/server.log")
    [ -n "$line" ] || { say "arm $arm: no 'expert stream io_uring: mode=' line in server.log"; return 1; }
    cuts=$(grep -c 'expert stream io_uring: read cuts: drive=' "$run/server.log")
    case $arm in
        A|A2) [[ $line == *"mode=default "* && $line == *"read_cuts=off:0"* && $line == *"effective_wait=block"* ]] && [ "$cuts" = 0 ] ;;
        B) [[ $line == *"mode=default "* && $line == *"read_cuts=on:1"* && $line == *"effective_wait=block"* ]] && [ "$cuts" = 3 ] ;;
        C) [[ $line == *"mode=iopoll "* && $line == *"read_cuts=on:1"* && $line == *"effective_wait=reap"* ]] && [ "$cuts" = 3 ] ;;
        D) [[ $line == *"mode=sqpoll_iopoll "* && $line == *"read_cuts=on:1"* && $line == *"sq_thread_idle_ms=10000"* ]] && [ "$cuts" = 3 ] ;;
        *) false ;;
    esac || { say "arm $arm: io_uring lines do not show the arm's settings: $line (read cuts lines: $cuts)"; return 1; }
    # Each drive is cut at its own limit (plan Review Focus 5): the SPCC 262144, both Samsungs 520192.
    if [ "$cuts" = 3 ]; then
        grep 'read cuts: drive=' "$run/server.log" | grep -q 'cut_bytes=262144 ' \
          && [ "$(grep 'read cuts: drive=' "$run/server.log" | grep -c 'cut_bytes=520192 ')" = 2 ] \
          || { say "arm $arm: per-drive cut_bytes are not 262144 + 2 x 520192"; return 1; }
    fi
}

run_one() {  # <arm> [KEY=VAL ...]
    local arm=$1; shift
    local done_log=$OUT/$arm.done rc pid_c pid_s run errors sampled=0 tpid="" spid
    : > "$done_log"
    ports_free || return 1
    exl3_gate "$arm" "$WT" || return 1
    wait_for_quiet_box
    node0_gate "$arm" || return 1
    set -- "$@" $(tier_overrides)
    say "arm $arm tier $TIER_NAME worktree $WT at $SHA overrides: $*"
    [ "$arm" = "$FIRST_ARM" ] && memory_sample "baseline-before-$arm"
    # 8>&- everywhere: no child may inherit (and outlive the driver holding) rowimg-disk.lock.
    setsid nohup taskset -c 20-23 nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm --format=csv -lms 1000 \
        > "$OUT/$arm-clocks.csv" 2> "$OUT/$arm-clocks.err" < /dev/null 8>&- &
    pid_c=$!; track "$pid_c" "$arm clock sampler"
    sleep 3
    kill -0 "$pid_c" 2>/dev/null || { say "clock sampler died: $(cat "$OUT/$arm-clocks.err")"; return 1; }
    say "arm $arm start"
    # setsid execs in place (a background job is not a group leader), so $! is run_arm.sh and its process group.
    EXPECT_SHA=$SHA DSV41_WORKTREE=$WT setsid bash "$RUN_ARM" "$arm" "$PORT" "$@" \
        > "$OUT/$arm-run_arm.log" 2>&1 < /dev/null 8>&- &
    ARM_PID=$!; track "$ARM_PID" "$arm run_arm.sh"
    startup_sampler "$arm" < /dev/null 8>&- &
    pid_s=$!; track "$pid_s" "$arm startup sampler"
    # Watch for run_arm.sh's "<arm> ready:" line to take the post-ready memory sample; run_arm.sh's own timeouts
    # (health 900 s, up to 12 warm-up rounds) are untouched.
    local pinned_at="" reg_done=0 srv_log="" stopped_reg=0
    while kill -0 "$ARM_PID" 2>/dev/null; do
        # Stop rule: still registering (no `expert stream io_uring: mode=` line) 300 s after "Pinned host expert cache
        # startup" -> stop the arm (run_arm.sh's group, and with it the server). No retry.
        [ -n "$srv_log" ] || { srv_log=$(run_dir_of "$arm"); [ -z "$srv_log" ] || srv_log=$srv_log/server.log; }
        if [ -n "$srv_log" ] && [ "$reg_done" = 0 ] && [ -f "$srv_log" ]; then
            [ -n "$pinned_at" ] || ! grep -q "Pinned host expert cache startup" "$srv_log" || { pinned_at=$(date +%s); say "arm $arm: pinned host cache up; watching registration"; }
            if grep -q "expert stream io_uring: mode=" "$srv_log"; then
                reg_done=1; [ -z "$pinned_at" ] || say "arm $arm: io_uring setup line after $(( $(date +%s) - pinned_at )) s (poll 5 s)"
            elif [ -n "$pinned_at" ] && [ $(( $(date +%s) - pinned_at )) -ge 300 ]; then
                say "arm $arm: still registering 300 s after the pinned-cache line; stopping the arm (stop rule)"
                stopped_reg=1; stop_pid "$ARM_PID" group
            fi
        fi
        # Per-thread CPU of the server process tree (io-wq, SQ thread, RAM-miss service), every 2 s until it exits.
        if [ -z "$tpid" ]; then
            spid=$(sed -n 's/^server pid=\([0-9]*\).*/\1/p' "$OUT/$arm-run_arm.log" 2>/dev/null | head -1)
            if [ -n "$spid" ]; then
                taskset -c 24-27 $PY "$WT/analysis/dsv41-drive/iopoll-cuts/thread_sampler.py" "$spid" "$OUT/$arm-threads.jsonl" 2 \
                    < /dev/null 8>&- &
                tpid=$!; track "$tpid" "$arm thread sampler"
            fi
        fi
        if [ "$sampled" = 0 ] && grep -q "^$arm ready:" "$OUT/$arm-run_arm.log" 2>/dev/null; then
            stop_pid "$pid_s"
            memory_sample "$arm-ready"; sampled=1
        fi
        sleep 5
    done
    wait "$ARM_PID"; rc=$?
    ARM_PID=""
    echo "DRIVER DONE rc=$rc" >> "$done_log"
    stop_pid "$pid_c"
    stop_pid "$pid_s"
    [ -z "$tpid" ] || stop_pid "$tpid"
    run=$(run_dir_of "$arm")
    say "arm $arm rc=$rc run: $run"
    [ "$sampled" = 1 ] || say "arm $arm: never logged 'ready'; no post-ready memory sample"
    if [ -n "$run" ] && [ -f "$run/server.log" ]; then
        uring_lines "$arm" "$run"
        if grep -Eq "$REFUSAL" "$run/server.log"; then
            say "arm $arm: registration refusal in server.log:"; grep -E "$REFUSAL" "$run/server.log" | head -5 | sed 's/^/    /'
            return 1
        fi
    fi
    [ "$stopped_reg" = 0 ] || { say "arm $arm: stopped by the registration stop rule"; return 1; }
    [ "$rc" = 0 ] || return "$rc"
    check_modes "$arm" "$run" || return 1
    errors=$(read_errors_of "$run")
    say "arm $arm read_errors=${errors:-<no counters in server.log>}"
    [ "$errors" = 0 ] || { say "arm $arm: read_errors is '${errors:-missing}', not 0; stopping"; return 1; }
}

rc=0
for arm in "${ARMS[@]}"; do
    case $arm in
        A|A2) set -- "${URING_A[@]}" ;;
        B) set -- "${URING_B[@]}" ;;
        C) set -- "${URING_C[@]}" ;;
        D) set -- "${URING_D[@]}" ;;
        *) say "unknown arm $arm"; rc=1; break ;;
    esac
    run_one "$arm" "$@" || { say "arm $arm failed; stopping the pass (see $OUT/$arm-run_arm.log)"; rc=1; break; }
done
say "tier that ran: $TIER_NAME (0:${N0_MIB},1:${N1_MIB} / $((N0_MIB + N1_MIB)))"
for arm in "${ARMS[@]}"; do say "$arm run: $(run_dir_of "$arm")"; done
if [ "$rc" = 0 ]; then
    # mirror3_report.py's functions (pooled ms/token, TTFT, byte identity, counters, clocks) plus thread_sampler.report
    # over each arm's timed window.
    RUNS=""
    for arm in "${ARMS[@]}"; do RUNS+="$arm=$(run_dir_of "$arm") "; done
    PYTHONPATH=$WT/analysis/dsv41-drive/mirror3:$WT/analysis/dsv41-drive/iopoll-cuts $PY -c "
import json, statistics, sys
import mirror3_report as m
import thread_sampler as ts
out_dir = sys.argv[1]
runs = dict(kv.split('=', 1) for kv in sys.argv[2:])
def arm(name, run_dir):
    start, end = m.timed_window_utc(run_dir)
    return {'run_dir': run_dir, 'decode': m.decode(run_dir), 'ms_per_token_median_turn': m.ms_per_token(run_dir)[1],
            'clocks': {'timed_window': m.clock_summary(f'{out_dir}/{name}-clocks.csv', start, end),
                       'session_start_end_mhz': m.session_clocks(run_dir)},
            'ram_miss': m.ram_miss(run_dir), 'threads': ts.report(f'{out_dir}/{name}-threads.jsonl', start, end)}
arms = {name: arm(name, run) for name, run in runs.items()}
ref = runs.get('A') or next(iter(runs.values()))
out = {'tier': '$TIER_NAME 0:${N0_MIB},1:${N1_MIB} / $((N0_MIB + N1_MIB))', 'arms': arms,
       'identical_to_A': {n: all(m.identity(ref, r).values()) for n, r in runs.items() if r != ref}}
pooled = {n: a['decode']['pooled_ms_per_token'] for n, a in arms.items()}
base = [pooled[n] for n in ('A', 'A2') if n in pooled]
if base:
    out['baseline_mean_A_A2'] = statistics.mean(base)
    out['delta_vs_baseline'] = {n: p - out['baseline_mean_A_A2'] for n, p in pooled.items()}
if 'A' in pooled and 'A2' in pooled:
    out['drift_A2_minus_A'] = pooled['A2'] - pooled['A']
out['all_identical'] = all(out['identical_to_A'].values())
json.dump(out, open(f'{out_dir}/arms-report.json', 'w'), indent=2, default=str)
print('pooled', pooled, 'delta', out.get('delta_vs_baseline'), 'identical', out['identical_to_A'])
raise SystemExit(0 if out['all_identical'] else 'an arm\'s output differs from A')
" "$OUT" $RUNS || rc=1
    say "report: $OUT/arms-report.json"
fi
say "DRIVER DONE rc=$rc"
exit "$rc"
