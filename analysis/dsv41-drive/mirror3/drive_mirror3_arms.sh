#!/usr/bin/env bash
# Plan 2026-09-28-mirror3-piece-stream Task 6: the production recipe's decode arm over arm_env's 2 mirror roots
# (reference) and over 3, at one commit, each with a /proc/diskstats sampler on the three mirror drives and an
# nvidia-smi SM-clock sampler (run_arm.sh records the clock only at session boundaries).
# Lock order: rowimg-disk.lock is held across both arms; cc-gpu.lock is polled here and taken by run_arm.sh itself.
# Usage: drive_mirror3_arms.sh <worktree> <sha> <out_dir under /mnt/nvme1> [port]
#
# Guards before anything starts: the NVIDIA driver is 615.71.09, nothing listens on production's 7867 or on the arm
# port. Before each arm: the same port check, then wait while a pytest, cc1plus or nvcc process exists or
# cc-gpu.lock is held. Processes are matched by name (pgrep -x) and, for `python -m pytest` (named "python"), by an
# interpreter exe/comm plus the exact cmdline tokens "-m" "pytest" -- never by free text over argv: a waiter doing
# that once deadlocked on its own command line. The driver's own PID and its ancestors are never counted.
# The driver records every PID it starts (samplers, run_arm.sh) in $OUT/driver-pids.txt and, on exit or on INT/TERM
# (and HUP, but only when not started under nohup, which ignores HUP before bash could trap it), kills only those.
# run_arm.sh runs as its own process group, so killing it also stops the server it started.
# At the end it prints the mirror3_report.py command for the two runs.
set -u
WT=${1:?worktree}; SHA=${2:?commit}; OUT=${3:?out dir}; PORT=${4:-30031}
PY=/data/models/slang/.venv/bin/python
ROOTS3=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash:/mnt/nvme2/dsv41_flash
PROD_PORT=7867
DRIVER_VERSION=615.71.09
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
say() { echo "$(date +%T) $*"; }
case $OUT in /mnt/nvme1/*) ;; *) say "out dir must be under /mnt/nvme1"; exit 1 ;; esac
cd "$WT" || exit 1
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
[ -z "$(git status --porcelain)" ] || { say "worktree dirty"; exit 1; }
PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)"
tree=$(git rev-parse HEAD:python)
PYTHONPATH=$WT/benchmarks/dsv41_baseline $PY -c "import generations; print('generation', '$tree', '=', generations.check_registered('$tree'))" \
    || { say "python tree $tree is not registered in generations.json"; exit 1; }
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
mkdir -p "$OUT"
# The whole namespace (nvme0n1), not the partition findmnt names (nvme0n1p1): the split is per drive.
disk_of() { local src; src=$(findmnt -no SOURCE --target "$1") || return 1; lsblk -no PKNAME "$src" | grep . || basename "$src"; }
DEVS=$(for r in ${ROOTS3//:/ }; do disk_of "$r"; done | sort -u | tr '\n' ' ')
say "mirror drives: $DEVS"
[ "$(echo $DEVS | wc -w)" = 3 ] || { say "the three roots are not on three devices: $DEVS"; exit 1; }

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
# One interpreter per scan: a bash loop forking per process took ~9 s a scan on the laptop.
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

run_one() {  # <arm> [KEY=VAL ...]
    local arm=$1; shift
    local done_log=$OUT/$arm.done rc pid_s pid_c
    : > "$done_log"
    ports_free || return 1
    wait_for_quiet_box
    # 8>&- everywhere: no child may inherit (and outlive the driver holding) rowimg-disk.lock.
    setsid nohup taskset -c 20-23 $PY "$WT/analysis/dsv41-drive/nvme-load/nvme_sampler.py" \
        "$OUT/$arm-diskstats.jsonl" "$done_log" $DEVS > "$OUT/$arm-sampler.log" 2>&1 < /dev/null 8>&- &
    pid_s=$!; track "$pid_s" "$arm diskstats sampler"
    setsid nohup taskset -c 20-23 nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm --format=csv -lms 1000 \
        > "$OUT/$arm-clocks.csv" 2> "$OUT/$arm-clocks.err" < /dev/null 8>&- &
    pid_c=$!; track "$pid_c" "$arm clock sampler"
    sleep 3
    kill -0 "$pid_s" 2>/dev/null || { say "diskstats sampler died: $(cat "$OUT/$arm-sampler.log")"; return 1; }
    kill -0 "$pid_c" 2>/dev/null || { say "clock sampler died: $(cat "$OUT/$arm-clocks.err")"; return 1; }
    say "arm $arm start"
    # setsid execs in place (a background job is not a group leader), so $! is run_arm.sh and its process group.
    EXPECT_SHA=$SHA DSV41_WORKTREE=$WT setsid bash "$WT/benchmarks/dsv41_baseline/run_arm.sh" "$arm" "$PORT" "$@" \
        > "$OUT/$arm-run_arm.log" 2>&1 < /dev/null 8>&- &
    ARM_PID=$!; track "$ARM_PID" "$arm run_arm.sh"
    wait "$ARM_PID"; rc=$?
    ARM_PID=""
    echo "DRIVER DONE rc=$rc" >> "$done_log"
    # The diskstats sampler checks the done log every 50 samples (~5 s) and closes its file on the way out; let it,
    # so the EXIT trap's TERM never drops buffered samples. nvidia-smi writes each sample as it takes it.
    for _ in $(seq 1 30); do kill -0 "$pid_s" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid_s" 2>/dev/null && say "WARNING: diskstats sampler $pid_s still running 30 s after DRIVER DONE"
    stop_pid "$pid_c"
    say "arm $arm rc=$rc run: $(sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/$arm-run_arm.log" | head -1)"
    return "$rc"
}

run_one mirror2-ref || { say "reference arm failed; not running the 3-root arm (see $OUT/mirror2-ref-run_arm.log)"; exit 1; }
run_one mirror3 "SGLANG_MOE_EXPERT_MIRROR_DIRS=$ROOTS3"
rc=$?
ref=$(sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/mirror2-ref-run_arm.log" | head -1)
new=$(sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/mirror3-run_arm.log" | head -1)
say "ref run: $ref"
say "new run: $new"
say "devices: $DEVS"
say "report: $PY $WT/analysis/dsv41-drive/mirror3/mirror3_report.py $ref $OUT/mirror2-ref-diskstats.jsonl" \
    "$OUT/mirror2-ref-clocks.csv $new $OUT/mirror3-diskstats.jsonl $OUT/mirror3-clocks.csv $DEVS"
say "DRIVER DONE rc=$rc"
exit "$rc"
