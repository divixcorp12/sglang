#!/usr/bin/env bash
# Decode A/B: does down-weighting the SPCC mirror (/mnt/nvme4, the middle root) improve decode ms/token?
# Arms: W = SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=1:0.9:1, U = 1:1:1 (parse_mirror_weights maps unset to exactly (1.0,)*3,
# so 1:1:1 is the default, set explicitly so both arms' env differs in that one value only). Everything else is the
# standard recipe (arm_env.base_env: 3 mirrors, default block wait, untraced).
# Order: alternating pairs, balanced: ARMS="W1 U1 U2 W2 W3 U3" by default (add "U4 W4 W5 U5" for more pairs).
# Template: analysis/dsv41-drive/iopoll-cuts/drive_iopoll_cuts_arms.sh. Changes:
#   - both locks are taken PER ARM and released between arms (rowimg-disk.lock first, then cc-gpu.lock is polled and
#     taken by run_arm.sh itself), so other lanes can slot in; a lost race for cc-gpu.lock retries the arm;
#   - the tier is settled per PAIR at its first arm (full 0:61440,1:40960 if node 0 >= share+4096+15360 and node 1 >=
#     share+4096+10000, else fallback 0:51200,1:40960) and the pair's second arm must pass the same tier's gates;
#     a short node releases the locks and waits (up to MEM_WAIT_S), it never cuts a pair's tier;
#   - a /proc/diskstats sampler on the three mirror drives (per-drive read split over the timed window);
#   - no uring mode / read-cut / thread checks (no uring knob varies);
#   - the worktree check ignores untracked files (generations.json is registered untracked; run_arm.sh's own preflight
#     refuses tracked changes).
# Usage: drive_weight_pairs.sh <worktree> <sha> <out_dir under /mnt/nvme1> [port]
set -u
WT=${1:?worktree}; SHA=${2:?commit}; OUT=${3:?out dir}; PORT=${4:-30041}
PY=/data/models/slang/.venv/bin/python
HERE=$WT/benchmarks/dsv41_baseline
RUN_ARM=$HERE/run_arm.sh
PROD_PORT=7867
DRIVER_VERSION=615.71.09
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
DEVS="nvme0n1 nvme2n1 nvme3n1"  # /mnt/nvme0 (Samsung), /mnt/nvme4 (SPCC), /mnt/nvme2 (Samsung): the mirror roots' order
W_WEIGHTS=1:0.9:1
U_WEIGHTS=1:1:1
MEM_WAIT_S=${MEM_WAIT_S:-5400}
read -r -a ARMS <<< "${ARMS:-W1 U1 U2 W2 W3 U3}"
say() { echo "$(date +%T) $*"; }
case $OUT in /mnt/nvme1/*) ;; *) say "out dir must be under /mnt/nvme1"; exit 1 ;; esac
mkdir -p "$OUT"

check_worktree() {
    local wt=$1 sha=$2 tree
    [ "$(git -C "$wt" rev-parse HEAD)" = "$sha" ] || { say "$wt not at $sha"; return 1; }
    [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || { say "$wt has tracked changes"; return 1; }
    PYTHONPATH=$wt/python $PY -c "import sglang; print('sglang from', sglang.__file__)" || return 1
    tree=$(git -C "$wt" rev-parse HEAD:python)
    PYTHONPATH=$HERE $PY -c "import generations; print('generation', '$tree', '=', generations.check_registered('$tree'))" \
        || { say "python tree $tree is not registered in $HERE/generations.json"; return 1; }
}
check_worktree "$WT" "$SHA" || exit 1
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[ "$driver" = "$DRIVER_VERSION" ] || { say "NVIDIA driver is '$driver', expected $DRIVER_VERSION"; exit 1; }
[ "$(findmnt -no SOURCE /mnt/nvme0)" = /dev/nvme0n1p1 ] && [ "$(findmnt -no SOURCE /mnt/nvme4)" = /dev/nvme2n1p1 ] \
    && [ "$(findmnt -no SOURCE /mnt/nvme2)" = /dev/nvme3n1p1 ] || { say "mirror mounts are not nvme0n1/nvme2n1/nvme3n1"; exit 1; }

listening() { ss -ltn "sport = :$1" | grep -q LISTEN; }
ports_free() {
    local p
    for p in "$PROD_PORT" "$PORT"; do
        if listening "$p"; then say "something listens on $p"; return 1; fi
    done
}

exl3_gate() {
    local arm=$1
    PYTHONPATH=$WT/python:$HERE $PY -c "
import types, arm_env
from sglang.srt.arg_groups.expert_stream_requirements import expert_stream_requirements_for
cfg = types.SimpleNamespace(model_path=arm_env.MODEL_PATH, quantization=None)
label = expert_stream_requirements_for(cfg, cfg).label
print('$arm expert-stream requirements for', arm_env.MODEL_PATH, '=', label)
raise SystemExit(0 if label == 'EXL3' else 'the launch gate resolved ' + repr(label) + ', not EXL3')
" || { say "EXL3 gate failed for $arm"; return 1; }
}

# --- tier: per pair ---
N1_MIB=40960; HEADROOM_MIB=4096; PRECHECK_FOOTPRINT_MIB=15360; NODE1_FOOTPRINT_MIB=10000
declare -A PAIR_TIER=()
tier_n0() { case $1 in full) echo 61440 ;; fallback) echo 51200 ;; esac; }
node_memory_mib() {
    awk '$3 == "MemFree:" { f += $4 } $3 == "Active(file):" || $3 == "Inactive(file):" { c += $4 }
         END { print int(f / 1024), int(c / 1024) }' "/sys/devices/system/node/node$1/meminfo"
}
gates_pass() {  # <arm> <tier>: log both nodes against the tier's needs; 0 when both pass
    local arm=$1 tier=$2 n0 free cache avail need ok free1 cache1 avail1 need1 ok1
    n0=$(tier_n0 "$tier")
    read -r free cache < <(node_memory_mib 0)
    read -r free1 cache1 < <(node_memory_mib 1)
    avail=$((free + cache)); need=$((n0 + HEADROOM_MIB + PRECHECK_FOOTPRINT_MIB))
    avail1=$((free1 + cache1)); need1=$((N1_MIB + HEADROOM_MIB + NODE1_FOOTPRINT_MIB))
    ok=$([ "$avail" -ge "$need" ] && echo true || echo false)
    ok1=$([ "$avail1" -ge "$need1" ] && echo true || echo false)
    say "gate $arm tier $tier: node0 ${free}+${cache}=${avail} need ${need} ok=$ok; node1 ${free1}+${cache1}=${avail1} need ${need1} ok=$ok1"
    printf '{"arm": "%s", "utc": "%s", "tier": "%s", "node0_memfree_mib": %d, "node0_page_cache_mib": %d, "node0_need_mib": %d, "node1_memfree_mib": %d, "node1_page_cache_mib": %d, "node1_need_mib": %d, "ok": %s, "node1_ok": %s}\n' \
        "$arm" "$(date -u +%FT%TZ)" "$tier" "$free" "$cache" "$need" "$free1" "$cache1" "$need1" "$ok" "$ok1" >> "$OUT/tier-gate.jsonl"
    [ "$ok" = true ] && [ "$ok1" = true ]
}
choose_tier() {  # <arm> <pair>: echo the tier this arm runs at, or fail (the pair's tier is never cut)
    local arm=$1 pair=$2 t
    if [ -n "${PAIR_TIER[$pair]:-}" ]; then
        gates_pass "$arm" "${PAIR_TIER[$pair]}" >&2 && { echo "${PAIR_TIER[$pair]}"; return 0; }
        return 1
    fi
    for t in full fallback; do gates_pass "$arm" "$t" >&2 && { echo "$t"; return 0; }; done
    return 1
}

# --- PIDs this driver started ---
PIDS_FILE=$OUT/driver-pids.txt
declare -A STARTED=()
starttime() { local s; s=$(cat "/proc/$1/stat" 2>/dev/null) || return 1; s=${s##*) }; set -- $s; echo "${20}"; }
track() { STARTED[$1]=$(starttime "$1"); echo "$1 ${STARTED[$1]} $2" >> "$PIDS_FILE"; }
stop_pid() {
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
trap 'say "interrupted"; exit 130' INT TERM HUP

MINE=" $$ $BASHPID "
p=$$
while p=$(awk '/^PPid:/ {print $2}' "/proc/$p/status" 2>/dev/null) && [ -n "$p" ] && [ "$p" != 0 ]; do MINE+="$p "; done
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
gpu_holders() {  # "<pid>:<exe basename>" of every process holding GPU memory (exe name, never argv)
    local p
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
        echo "$p:$(basename "$(readlink "/proc/$p/exe" 2>/dev/null)" 2>/dev/null || cat "/proc/$p/comm" 2>/dev/null)"
    done | tr '\n' ' '
}
gpu_ready() {  # 0 when no foreign build/test, no GPU memory holder, and cc-gpu.lock is free right now
    local busy holders
    busy=$( { pgrep -x 'pytest|cc1plus|nvcc'; python_pytest; } | while read -r p; do
               [[ $MINE == *" $p "* ]] || echo "$p"; done | tr '\n' ' ')
    holders=$(gpu_holders)
    if [ -n "$busy" ]; then say "foreign pytest/cc1plus/nvcc running (pids $busy)"; return 1; fi
    if [ -n "$holders" ]; then say "GPU memory held by: $holders"; return 1; fi
    if ! flock -n "$GPU_LOCK" true; then say "cc-gpu.lock held"; return 1; fi
    ports_free
}

run_dir_of() { sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/$1-run_arm.log" | head -1; }
read_errors_of() {
    PYTHONPATH=$WT/analysis/dsv41-drive/mirror3 $PY -c "
import sys, mirror3_report as m
t = m.ram_miss(sys.argv[1])['thread']
print('' if t is None or t.get('read_errors') is None else t['read_errors'])
" "$1"
}

attempt() {  # <arm> <pair> <weights>: one locked attempt. rc 0 ok, 75 retry later (locks released), else fail
    local arm=$1 pair=$2 weights=$3 tier n0 rc pid_s pid_c run errors done_log=$OUT/$1.done
    exec 8>"$DISK_LOCK"
    say "$arm: waiting for rowimg-disk.lock"; flock 8; say "$arm: rowimg-disk.lock held"
    until gpu_ready; do sleep 60; done
    exl3_gate "$arm" || { exec 8>&-; return 1; }
    tier=$(choose_tier "$arm" "$pair") || { say "$arm: memory short; releasing locks"; exec 8>&-; return 75; }
    n0=$(tier_n0 "$tier")
    set -- SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=$weights \
        SGLANG_MOE_PINNED_HOST_NUMA_MB=0:${n0},1:${N1_MIB} SGLANG_MOE_PINNED_HOST_MB=$((n0 + N1_MIB))
    say "arm $arm pair $pair tier $tier at $SHA overrides: $*"
    echo "{\"arm\": \"$arm\", \"pair\": \"$pair\", \"tier\": \"$tier\", \"weights\": \"$weights\", \"utc\": \"$(date -u +%FT%TZ)\"}" >> "$OUT/arms.jsonl"
    : > "$done_log"; rm -f "$OUT/$arm-diskstats.jsonl"
    setsid nohup taskset -c 20-23 $PY "$WT/analysis/dsv41-drive/nvme-load/nvme_sampler.py" \
        "$OUT/$arm-diskstats.jsonl" "$done_log" $DEVS > "$OUT/$arm-sampler.log" 2>&1 < /dev/null 8>&- &
    pid_s=$!; track "$pid_s" "$arm diskstats sampler"
    setsid nohup taskset -c 20-23 nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm --format=csv -lms 1000 \
        > "$OUT/$arm-clocks.csv" 2> "$OUT/$arm-clocks.err" < /dev/null 8>&- &
    pid_c=$!; track "$pid_c" "$arm clock sampler"
    sleep 3
    kill -0 "$pid_s" 2>/dev/null && kill -0 "$pid_c" 2>/dev/null || { say "a sampler died"; exec 8>&-; return 1; }
    say "arm $arm start"
    EXPECT_SHA=$SHA DSV41_WORKTREE=$WT setsid bash "$RUN_ARM" "$arm" "$PORT" "$@" \
        > "$OUT/$arm-run_arm.log" 2>&1 < /dev/null 8>&- &
    ARM_PID=$!; track "$ARM_PID" "$arm run_arm.sh"
    wait "$ARM_PID"; rc=$?
    ARM_PID=""
    echo "DRIVER DONE rc=$rc" >> "$done_log"
    for _ in $(seq 1 30); do kill -0 "$pid_s" 2>/dev/null || break; sleep 1; done
    stop_pid "$pid_s"; stop_pid "$pid_c"
    exec 8>&-
    say "$arm: locks released"
    run=$(run_dir_of "$arm")
    say "arm $arm rc=$rc run: $run"
    if [ "$rc" != 0 ] && grep -Eq "cc-gpu.lock is held by another GPU job|GPU is in use; not starting" "$OUT/$arm-run_arm.log"; then
        say "$arm: lost the race for the GPU; retrying"; mv "$OUT/$arm-run_arm.log" "$OUT/$arm-run_arm.lostrace-$(date +%s).log"
        return 75
    fi
    [ "$rc" = 0 ] || return "$rc"
    PAIR_TIER[$pair]=$tier
    errors=$(read_errors_of "$run")
    say "arm $arm read_errors=${errors:-<none>}"
    [ "$errors" = 0 ] || { say "arm $arm: read_errors '${errors:-missing}', not 0"; return 1; }
}

run_one() {  # <arm>
    local arm=$1 kind=${1:0:1} pair=${1:1} weights deadline rc
    case $kind in W) weights=$W_WEIGHTS ;; U) weights=$U_WEIGHTS ;; *) say "unknown arm $arm"; return 1 ;; esac
    deadline=$(( $(date +%s) + MEM_WAIT_S ))
    while :; do
        attempt "$arm" "$pair" "$weights"; rc=$?
        [ "$rc" = 75 ] || return "$rc"
        [ "$(date +%s)" -lt "$deadline" ] || { say "$arm: gave up after ${MEM_WAIT_S}s of retries"; return 1; }
        sleep 120
    done
}

rc=0
for arm in "${ARMS[@]}"; do
    run_one "$arm" || { say "arm $arm failed; stopping (see $OUT/$arm-run_arm.log)"; rc=1; break; }
done
for arm in "${ARMS[@]}"; do say "$arm run: $(run_dir_of "$arm" 2>/dev/null)"; done
if [ "$rc" = 0 ]; then
    RUNS=""
    for arm in "${ARMS[@]}"; do RUNS+="$arm=$(run_dir_of "$arm") "; done
    PYTHONPATH=$WT/analysis/dsv41-drive/mirror3:$WT/analysis/dsv41-drive/mirror-scaling \
        $PY "$WT/analysis/dsv41-drive/mirror-scaling/weight_pair_report.py" "$OUT" $DEVS -- $RUNS || rc=1
fi
say "DRIVER DONE rc=$rc"
exit "$rc"
