#!/usr/bin/env bash
# Plan 2026-09-29-hotpath-zero-overhead Task 18: the production recipe's decode arms, master against the branch, in
# this order: A (master), B (branch), A2 (master: drift control), then C (branch, untimed) with the LD_PRELOAD counting
# shim armed from load (HOTPATH_SHIM_OUT), which writes the production server's service- and copy-thread counts for
# the whole run. C runs only if A, B and A2 passed (rc 0, read_errors 0, the build check) and B's and A2's output is
# byte-identical to A's. Pass (hotpath_report.py): identity, read_errors == 0, and B no slower than
# mean(A, A2) + max(1.5, |A2 - A|) ms/token.
# Template: analysis/dsv41-drive/reader-crtp/drive_reader_crtp_pair.sh (two worktrees; every arm runs the BRANCH's
# run_arm.sh, arm_env and generations.json with DSV41_WORKTREE/EXPECT_SHA naming the arm's own tree), with
# analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh's tier gates, memory and startup samplers and per-thread
# sampler. Changes:
#   - the arm table: A/A2 run master with SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1,
#     SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=1 and SGLANG_DSV41_RAM_MISS_PACK_WORKERS=8 (master's recipe: the branch's
#     arm_env no longer sets them); B none; C LD_PRELOAD=<shim .so built into $OUT> HOTPATH_SHIM_OUT=$OUT/C-shim.json.
#     Each arm's effective env diff against the branch's base_env is logged and written to $OUT/<arm>-env-diff.json;
#   - a build check once the server is "fired up": B's and C's "exl3 RAM miss thread started" line ends in
#     "build prod"; A's and A2's has no "build" field (master has none);
#   - perf stat (cycles:u, instructions:u, context-switches, cpu-migrations) on the exl3-ram-miss thread from "fired
#     up" to the arm's end, as the user (skipped, and said so, when perf_event_paranoid > 2), $OUT/<arm>-perf.csv.
#     Under paranoid 2 perf's context-switches and cpu-migrations read 0 (see sched_sample), so the thread's
#     voluntary/involuntary switches and migrations are also sampled from /proc, $OUT/<arm>-sched.jsonl;
#     The thread lives in the scheduler process, a child of the launch_server pid run_arm.sh logs, so it is found in
#     that pid's process tree, not in its own task list;
#   - run_arm.sh takes cc-gpu.lock non-blocking. If another lane takes the GPU between wait_for_quiet_box and that
#     flock, run_arm.sh refuses before launching anything; that refusal (and only it) is retried after the next quiet
#     box, up to 30 times.
# Everything else is the templates': the NVIDIA driver and port guards, check_worktree with the generations gate, the
# EXL3 launch-gate assertion before every arm with that arm's own python/ tree (stop unless the resolved requirements
# are EXL3, not the silent NVFP4 fallback), exe-name foreign-process gates, the disk lock taken here (exec 8>, flock 8;
# never under an outer flock), the GPU lock polled here and taken by run_arm.sh, PID tracking and cleanup, the SM-clock
# sampler, stop on read_errors > 0.
# Tier (decided once, before A, as drive_iopoll_cuts_arms.sh): full 0:61440,1:40960 / 102400 if node 0 has
# >= 61440+4096+15360 and node 1 >= 40960+4096+10000 MiB of MemFree + Active(file) + Inactive(file); else
# 0:51200,1:40960 / 92160; both passed to every arm as identical overrides; re-checked before each later arm at the
# same values, and short means stop.
# Lock order: rowimg-disk.lock is held across all arms; cc-gpu.lock is polled here and taken by run_arm.sh itself.
# Final-fix round (call-site attribution of C's counts): two more arms, run by ARMS="CS CM", never by default:
#   - CS: the branch under the shim (as C) with HOTPATH_SHIM_STACKS=$OUT/CS-callsites.txt, HOTPATH_SHIM_OUT=$OUT/CS-shim.json;
#   - CM: master with master's recipe under the same shim, $OUT/CM-callsites.txt and $OUT/CM-shim.json.
#   (Not <arm>-stacks.txt: that is the startup sampler's kernel-stack file.)
#   Neither needs A: they are not timed and have no identity gate before them; each arm's stacks are attributed by
#   final-fix/attribute_stacks.py once it ends. With REF_RUN=<a run dir> (Task 18's A), every arm's output is also
#   compared byte for byte with it; hotpath_report.py runs only when ARMS includes A.
# Usage: drive_hotpath_arms.sh <base_worktree> <base_sha> <branch_worktree> <branch_sha> <out_dir under /mnt/nvme1> [port]
set -u
BASE_WT=${1:?base worktree}; BASE_SHA=${2:?base commit}; BRANCH_WT=${3:?branch worktree}; BRANCH_SHA=${4:?branch commit}
OUT=${5:?out dir}; PORT=${6:-30031}
PY=/data/models/slang/.venv/bin/python
HERE=$BRANCH_WT/benchmarks/dsv41_baseline  # run_arm.sh, arm_env and generations.json for EVERY arm
RUN_ARM=$HERE/run_arm.sh
PROD_PORT=7867
DRIVER_VERSION=615.71.09
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
SHIM_SO=$OUT/hotpath_shim.so
MASTER_RECIPE=(
    SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1 SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=1
    SGLANG_DSV41_RAM_MISS_PACK_WORKERS=8
)
read -r -a ARMS <<< "${ARMS:-A B A2 C}"
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
check_worktree "$BRANCH_WT" "$BRANCH_SHA" || exit 1
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

# Tier: settled once before the first arm, applied to every arm, never cut afterwards (drive_iopoll_cuts_arms.sh).
TIER_NAMES=(full 90g); TIER_N0=(61440 51200); TIER_IDX=0
TIER_NAME=${TIER_NAMES[0]}; N0_MIB=${TIER_N0[0]}; N1_MIB=40960; HEADROOM_MIB=4096
PRECHECK_FOOTPRINT_MIB=15360
NODE1_FOOTPRINT_MIB=10000
node_memory_mib() {  # <node>: "<MemFree> <Active(file)+Inactive(file)>" in MiB
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
            say "short for the full tier: using the $TIER_NAME tier for EVERY arm -> 0:${N0_MIB},1:${N1_MIB} (total $((N0_MIB + N1_MIB)))"
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

# The shim for arm C, built from the branch tree into $OUT (never into a worktree: run_arm.sh refuses a dirty one).
cc -shared -fPIC -O2 -o "$SHIM_SO" "$BRANCH_WT/python/sglang/test/hotpath_shim.c" -ldl -lpthread \
    || { say "cannot build the counting shim"; exit 1; }
say "shim built: $SHIM_SO"
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
say "perf_event_paranoid=$PARANOID"

# Startup sampler (drive_iopoll_cuts_arms.sh), from launch until "ready" or run_arm.sh's exit, every ~10 s.
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
    PYTHONPATH=$BRANCH_WT/analysis/dsv41-drive/mirror3 $PY -c "
import sys, mirror3_report as m
t = m.ram_miss(sys.argv[1])['thread']
print('' if t is None or t.get('read_errors') is None else t['read_errors'])
" "$1"
}

env_diff() {  # <arm> [KEY=VAL ...]: the arm's effective env against the branch's base_env, logged and kept
    local arm=$1; shift
    PYTHONPATH=$HERE $PY -c "
import json, sys
import arm_env
overrides = dict(kv.split('=', 1) for kv in sys.argv[2:])
base, eff = arm_env.base_env(), arm_env.arm_env(overrides)
diff = {k: {'base': base.get(k), 'arm': eff.get(k)} for k in sorted(set(base) | set(eff)) if base.get(k) != eff.get(k)}
json.dump(diff, open(sys.argv[1], 'w'), indent=2)
for k, v in diff.items():
    print(f'    {k}: {v[\"base\"]!r} -> {v[\"arm\"]!r}')
" "$OUT/$arm-env-diff.json" "$@"
}

service_tid() {  # <server pid>: the exl3-ram-miss thread's tid anywhere in the server's process tree (empty if none)
    local root=$1
    ps -e -o pid=,ppid= | awk -v root="$root" '
        { parent[$1] = $2 }
        END { for (p in parent) { q = p; while (q != "" && q != 0 && q != root) q = parent[q]; if (q == root) print p } }' |
    while read -r pid; do
        for t in /proc/"$pid"/task/*; do
            [ "$(cat "$t/comm" 2>/dev/null)" = exl3-ram-miss ] && basename "$t"
        done
    done | head -1
}

PERF_PID=""; SVC_TID=""
perf_service() {  # <arm> <server pid>: perf stat of the service thread until the arm ends (no root)
    local arm=$1 spid=$2 tid
    tid=$(service_tid "$spid")
    [ -n "$tid" ] || { say "arm $arm: no exl3-ram-miss thread in server pid $spid's tree"; return 1; }
    SVC_TID=$tid
    if [ "$PARANOID" -gt 2 ]; then say "arm $arm: perf_event_paranoid=$PARANOID, perf stat skipped"; return 0; fi
    taskset -c 20-23 perf stat -x, -e cycles:u,instructions:u,context-switches,cpu-migrations \
        -t "$tid" -o "$OUT/$arm-perf.csv" < /dev/null 8>&- &
    PERF_PID=$!; track "$PERF_PID" "$arm perf stat"
    say "arm $arm: perf stat pid $PERF_PID on exl3-ram-miss tid $tid"
}

# Under perf_event_paranoid 2 perf stat adds :u to the software events too, and context-switches:u and
# cpu-migrations:u then read 0 whatever happens (measured on divix01: a thread that slept 27k times in 3 s read 0 and 0).
# So the service thread's switches and migrations are also sampled from /proc (voluntary_ctxt_switches,
# nonvoluntary_ctxt_switches, se.nr_migrations), every ~5 s while it lives, into $OUT/<arm>-sched.jsonl.
sched_sample() {  # <arm>
    local arm=$1 t=/proc/$SVC_TID/task/$SVC_TID vol nonvol mig
    [ -n "$SVC_TID" ] && [ -r "$t/status" ] || return 0
    [ "$(cat "$t/comm" 2>/dev/null)" = exl3-ram-miss ] || return 0
    vol=$(awk '$1 == "voluntary_ctxt_switches:" { print $2 }' "$t/status" 2>/dev/null)
    nonvol=$(awk '$1 == "nonvoluntary_ctxt_switches:" { print $2 }' "$t/status" 2>/dev/null)
    mig=$(awk '$1 == "se.nr_migrations" { print $3 }' "$t/sched" 2>/dev/null)
    [ -n "$vol" ] && [ -n "$nonvol" ] || return 0
    printf '{"utc": "%s", "tid": %d, "voluntary": %d, "nonvoluntary": %d, "migrations": %s}\n' \
        "$(date -u +%FT%T.%6N+00:00)" "$SVC_TID" "$vol" "$nonvol" "${mig:-null}" >> "$OUT/$arm-sched.jsonl"
}

build_check() {  # <arm> <server.log>: master's thread-started line has no build field; the branch's says "build prod"
    local arm=$1 log=$2 line
    line=$(grep -m1 'exl3 RAM miss thread started' "$log")
    [ -n "$line" ] || { say "arm $arm: no 'exl3 RAM miss thread started' line"; return 1; }
    say "arm $arm: $line"
    case $arm in
        A|A2|CM) [[ $line != *build* ]] ;;
        B|C|CS) [[ $line == *"build prod" ]] ;;
        *) false ;;
    esac || { say "arm $arm: the thread-started line does not show the arm's build"; return 1; }
}

run_one() {  # <arm> <worktree> <sha> [KEY=VAL ...]
    local arm=$1 wt=$2 sha=$3; shift 3
    local done_log=$OUT/$arm.done rc pid_c pid_s run errors sampled tpid spid fired built attempt
    : > "$done_log"
    ports_free || return 1
    exl3_gate "$arm" "$wt" || return 1
    for attempt in $(seq 1 30); do
        wait_for_quiet_box
        node0_gate "$arm" || return 1
        [ "$arm" = "$FIRST_ARM" ] && [ "$attempt" = 1 ] && memory_sample "baseline-before-$arm"
        say "arm $arm (attempt $attempt) tier $TIER_NAME worktree $wt at $sha overrides: $* $(tier_overrides)"
        say "arm $arm effective env vs the branch's base_env:"
        env_diff "$arm" "$@" $(tier_overrides) || return 1
        # 8>&- everywhere: no child may inherit (and outlive the driver holding) rowimg-disk.lock.
        setsid nohup taskset -c 20-23 nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm --format=csv -lms 1000 \
            > "$OUT/$arm-clocks.csv" 2> "$OUT/$arm-clocks.err" < /dev/null 8>&- &
        pid_c=$!; track "$pid_c" "$arm clock sampler"
        sleep 3
        kill -0 "$pid_c" 2>/dev/null || { say "clock sampler died: $(cat "$OUT/$arm-clocks.err")"; return 1; }
        say "arm $arm start"
        # setsid execs in place (a background job is not a group leader), so $! is run_arm.sh and its process group.
        EXPECT_SHA=$sha DSV41_WORKTREE=$wt setsid bash "$RUN_ARM" "$arm" "$PORT" "$@" $(tier_overrides) \
            > "$OUT/$arm-run_arm.log" 2>&1 < /dev/null 8>&- &
        ARM_PID=$!; track "$ARM_PID" "$arm run_arm.sh"
        startup_sampler "$arm" < /dev/null 8>&- &
        pid_s=$!; track "$pid_s" "$arm startup sampler"
        sampled=0; tpid=""; spid=""; fired=0; built=1; PERF_PID=""; SVC_TID=""
        local srv_log=""
        while kill -0 "$ARM_PID" 2>/dev/null; do
            [ -n "$srv_log" ] || { srv_log=$(run_dir_of "$arm"); [ -z "$srv_log" ] || srv_log=$srv_log/server.log; }
            if [ -z "$tpid" ]; then
                spid=$(sed -n 's/^server pid=\([0-9]*\).*/\1/p' "$OUT/$arm-run_arm.log" 2>/dev/null | head -1)
                if [ -n "$spid" ]; then
                    taskset -c 24-27 $PY "$BRANCH_WT/analysis/dsv41-drive/iopoll-cuts/thread_sampler.py" "$spid" \
                        "$OUT/$arm-threads.jsonl" 2 < /dev/null 8>&- &
                    tpid=$!; track "$tpid" "$arm thread sampler"
                fi
            fi
            if [ "$fired" = 0 ] && [ -n "$srv_log" ] && [ -n "$spid" ] && grep -q "fired up" "$srv_log" 2>/dev/null; then
                fired=1
                say "arm $arm: server fired up"
                if ! build_check "$arm" "$srv_log"; then
                    built=0; stop_pid "$ARM_PID" group
                else
                    perf_service "$arm" "$spid" || { built=0; stop_pid "$ARM_PID" group; }
                fi
            fi
            sched_sample "$arm"
            if [ "$sampled" = 0 ] && grep -q "^$arm ready:" "$OUT/$arm-run_arm.log" 2>/dev/null; then
                stop_pid "$pid_s"
                memory_sample "$arm-ready"; sampled=1
            fi
            sleep 5
        done
        wait "$ARM_PID"; rc=$?
        ARM_PID=""
        echo "DRIVER DONE rc=$rc attempt=$attempt" >> "$done_log"
        if [ -n "$PERF_PID" ]; then
            # perf writes its CSV on SIGINT (or on its own once the thread is gone).
            kill -INT "$PERF_PID" 2>/dev/null; wait "$PERF_PID" 2>/dev/null; unset "STARTED[$PERF_PID]"
        fi
        stop_pid "$pid_c"
        stop_pid "$pid_s"
        [ -z "$tpid" ] || stop_pid "$tpid"
        if [ "$rc" != 0 ] && [ -z "$spid" ] && grep -q "cc-gpu.lock is held by another GPU job" "$OUT/$arm-run_arm.log"; then
            say "arm $arm: another job took cc-gpu.lock before run_arm.sh did (nothing launched); retrying"
            continue
        fi
        break
    done
    run=$(run_dir_of "$arm")
    say "arm $arm rc=$rc run: $run"
    [ "$sampled" = 1 ] || say "arm $arm: never logged 'ready'; no post-ready memory sample"
    [ "$built" = 1 ] || { say "arm $arm: build check or perf start failed; stopping"; return 1; }
    [ "$fired" = 1 ] || { say "arm $arm: never saw 'fired up'"; return 1; }
    [ "$rc" = 0 ] || return "$rc"
    [ -z "$PERF_PID" ] || { say "arm $arm perf stat:"; grep -v '^#' "$OUT/$arm-perf.csv" | sed '/^$/d; s/^/    /'; }
    errors=$(read_errors_of "$run")
    say "arm $arm read_errors=${errors:-<no counters in server.log>}"
    [ "$errors" = 0 ] || { say "arm $arm: read_errors is '${errors:-missing}', not 0; stopping"; return 1; }
}

identical_so_far() {  # A2 and B must match A byte for byte before C runs
    PYTHONPATH=$BRANCH_WT/analysis/dsv41-drive/mirror3:$BRANCH_WT/analysis/dsv41-drive/hotpath $PY -c "
import sys, hotpath_report as hr
ref = sys.argv[1]
bad = {}
for name, run in (kv.split('=', 1) for kv in sys.argv[2:]):
    bad[name] = [k for k, same in hr.identity_checked(ref, run).items() if not same]
print('identity vs A:', {n: ('identical' if not b else f'{len(b)} turns differ: {b}') for n, b in bad.items()})
raise SystemExit(0 if not any(bad.values()) else 'output differs from A')
" "$(run_dir_of A)" "B=$(run_dir_of B)" "A2=$(run_dir_of A2)"
}

attribute() {  # <arm>: the arm's shim counts and stacks, attributed to call sites (CS, CM)
    PYTHONPATH=$BRANCH_WT/python $PY "$BRANCH_WT/analysis/dsv41-drive/hotpath/final-fix/attribute_stacks.py" \
        "$OUT/$1-callsites.txt" "$OUT/$1-shim.json" "$OUT/$1-attribution.json" 65536
}

same_as_ref() {  # <arm>: byte identity against REF_RUN (Task 18's A), when given
    [ -n "${REF_RUN:-}" ] || return 0
    PYTHONPATH=$BRANCH_WT/analysis/dsv41-drive/mirror3:$BRANCH_WT/analysis/dsv41-drive/hotpath $PY -c "
import sys, hotpath_report as hr
same = hr.identity_checked(sys.argv[1], sys.argv[2])
bad = [k for k, v in same.items() if not v]
print('$1 identity vs REF_RUN:', 'identical' if not bad else f'{len(bad)} turns differ: {bad}', f'({len(same)} turns)')
raise SystemExit(0 if not bad else 'output differs from REF_RUN')
" "$REF_RUN" "$(run_dir_of "$1")"
}

STACKS_ENV=(HOTPATH_SHIM_STACKS_FIRST=256 HOTPATH_SHIM_STACKS_EVERY=65536)
rc=0
for arm in "${ARMS[@]}"; do
    case $arm in
        A|A2) run_one "$arm" "$BASE_WT" "$BASE_SHA" "${MASTER_RECIPE[@]}" ;;
        B) run_one "$arm" "$BRANCH_WT" "$BRANCH_SHA" ;;
        C)
            identical_so_far || { say "B or A2 differs from A: not running C (stop-and-report)"; rc=1; break; }
            rm -f "$OUT/C-shim.json" "$OUT"/C-shim.json.*
            run_one "$arm" "$BRANCH_WT" "$BRANCH_SHA" "LD_PRELOAD=$SHIM_SO" "HOTPATH_SHIM_OUT=$OUT/C-shim.json" ;;
        CS|CM)
            rm -f "$OUT/$arm-shim.json" "$OUT/$arm-shim.json".* "$OUT/$arm-callsites.txt" "$OUT/$arm-callsites.txt".*
            if [ "$arm" = CS ]; then set -- "$BRANCH_WT" "$BRANCH_SHA"; else set -- "$BASE_WT" "$BASE_SHA" "${MASTER_RECIPE[@]}"; fi
            run_one "$arm" "$@" "LD_PRELOAD=$SHIM_SO" "HOTPATH_SHIM_OUT=$OUT/$arm-shim.json" \
                "HOTPATH_SHIM_STACKS=$OUT/$arm-callsites.txt" "${STACKS_ENV[@]}" && attribute "$arm" ;;
        *) say "unknown arm $arm"; false ;;
    esac || { say "arm $arm failed; stopping the pass (see $OUT/$arm-run_arm.log)"; rc=1; break; }
    same_as_ref "$arm" || { say "arm $arm differs from REF_RUN; stopping the pass"; rc=1; break; }
done
say "tier that ran: $TIER_NAME (0:${N0_MIB},1:${N1_MIB} / $((N0_MIB + N1_MIB)))"
RUNS=""
for arm in "${ARMS[@]}"; do
    r=$(run_dir_of "$arm" 2>/dev/null); say "$arm run: $r"; [ -z "$r" ] || RUNS+="$arm=$r "
done
if [ "$rc" = 0 ] && [[ " ${ARMS[*]} " == *" A "* ]]; then
    PYTHONPATH=$BRANCH_WT/analysis/dsv41-drive/mirror3:$BRANCH_WT/analysis/dsv41-drive/iopoll-cuts \
        $PY "$BRANCH_WT/analysis/dsv41-drive/hotpath/hotpath_report.py" "$OUT" \
        "0:${N0_MIB},1:${N1_MIB} / $((N0_MIB + N1_MIB)) ($TIER_NAME)" "$PARANOID" $RUNS || rc=1
    say "report: $OUT/arms-report.json"
fi
say "DRIVER DONE rc=$rc"
exit "$rc"
