#!/usr/bin/env bash
# Recipe memory checks (EXPECT_MFS, default 0.895; hot cache read from arm_env) on divix01, at the recipe as committed (no fraction override):
#   1. one standard decode arm (run_arm.sh), for ms/token and the KV sizing;
#   2. long-prompt prefill smokes (prefill-chunk/chunk_smoke.sh, 4096-token chunks, the recipe's hot cache):
#      16k and 64k at the recipe, each twice (-2 steps), (suffixes >= 8192 prefill layer-major), then 16k and 64k with
#      SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 (every chunk goes through the 4096-token chunked path).
# Stops at the first failure. Every launch waits for both NUMA gates of the full tier (node 0 >= 61440+4096+15360,
# node 1 >= 40960+4096+10000 MiB, MemFree + page cache). Lock order: rowimg-disk.lock, then cc-gpu.lock.
# Usage: run_checks.sh <worktree> <sha> [steps...]   steps: decode lm16k lm64k ch16k ch64k (default: all, in order)
set -u
WT=${1:?worktree}; SHA=${2:?sha}; shift 2
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(decode lm16k lm64k ch16k ch64k lm16k-2 lm64k-2 ch16k-2 ch64k-2)
PY=/data/models/slang/.venv/bin/python
OUT=/mnt/nvme1/recipe-mem
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
PORT=30051
TAGP=${TAGP:-mem0895}  # tag prefix for the arm and smoke dirs
mkdir -p "$OUT"
say() { echo "$(date +%T) $*"; }

[ "$(git -C "$WT" rev-parse HEAD)" = "$SHA" ] || { say "$WT not at $SHA"; exit 1; }
[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || { say "$WT has tracked changes"; exit 1; }
PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)" || exit 1
mfs=$(PYTHONPATH=$WT/benchmarks/dsv41_baseline $PY -c "import arm_env; print(arm_env.MEM_FRACTION_STATIC)")
[ "$mfs" = "${EXPECT_MFS:-0.895}" ] || { say "arm_env.MEM_FRACTION_STATIC is $mfs, not ${EXPECT_MFS:-0.895}"; exit 1; }
HOT=$(PYTHONPATH=$WT/benchmarks/dsv41_baseline $PY -c "import arm_env; print(arm_env.base_env()['SGLANG_MOE_HOT_GPU_MB'])")
say "recipe: MEM_FRACTION_STATIC=$mfs SGLANG_MOE_HOT_GPU_MB=$HOT"
[ -z "${DSV41_MEM_FRACTION_STATIC:-}" ] || { say "DSV41_MEM_FRACTION_STATIC is set; this check runs the constant"; exit 1; }

node_avail() {
    awk '$3 == "MemFree:" { f += $4 } $3 == "Active(file):" || $3 == "Inactive(file):" { c += $4 }
         END { print int((f + c) / 1024) }' "/sys/devices/system/node/node$1/meminfo"
}
wait_gates() {  # <step>: poll until both nodes have room for the full tier (at most 90 min)
    local step=$1 a0 a1 i
    for i in $(seq 1 90); do
        a0=$(node_avail 0); a1=$(node_avail 1)
        say "gate $step: node0 ${a0} MiB (need 80896), node1 ${a1} MiB (need 55056)"
        echo "{\"step\": \"$step\", \"utc\": \"$(date -u +%FT%TZ)\", \"node0_avail_mib\": $a0, \"node1_avail_mib\": $a1}" >> "$OUT/gates.jsonl"
        [ "$a0" -ge 80896 ] && [ "$a1" -ge 55056 ] && return 0
        sleep 60
    done
    return 1
}
gpu_free() {  # no process holds GPU memory (reported by exe name) and cc-gpu.lock is free
    local p holders=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
        holders+="$p:$(basename "$(readlink "/proc/$p/exe" 2>/dev/null)") "
    done
    [ -z "$holders" ] || { say "GPU memory held by: $holders"; return 1; }
    flock -n "$GPU_LOCK" true || { say "cc-gpu.lock held"; return 1; }
}

decode_arm() {
    local rc
    exec 8>"$DISK_LOCK"; say "decode: waiting for rowimg-disk.lock"; flock 8; say "decode: disk lock held"
    until gpu_free; do sleep 60; done
    wait_gates decode || { exec 8>&-; return 1; }
    EXPECT_SHA=$SHA DSV41_WORKTREE=$WT bash "$WT/benchmarks/dsv41_baseline/run_arm.sh" "$TAGP" "$PORT" \
        > "$OUT/decode-run_arm.log" 2>&1 < /dev/null 8>&-
    rc=$?
    exec 8>&-
    say "decode: run_arm rc=$rc ($(sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/decode-run_arm.log" | head -1))"
    return $rc
}
smoke() {  # <tag> <tokens> [SMOKE_ENV]: chunk_smoke.sh takes both locks itself, in order, and waits for them
    local tag=$1 tokens=$2 extra=${3:-} rc
    wait_gates "$tag" || return 1
    SMOKE_ENV=$extra bash "$WT/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh" "$TAGP-$tag" "$WT" 4096 "$HOT" "$tokens" \
        > "$OUT/$tag-smoke.log" 2>&1 < /dev/null
    rc=$?
    say "$tag: chunk_smoke rc=$rc ($(grep -E 'long prompt:|OOM retries|long rc' /mnt/nvme1/prefill-chunk/$TAGP-$tag/driver.log | tr '\n' ' '))"
    [ "$rc" = 0 ] || return $rc
    # Minimum free at the peak: CUDA can use 32,150 MiB of the card (DSV41_REFERENCE 27.18), so free = 32150 - peak
    # memory.used (50 ms samples). Under 64 MiB stops the pass.
    local peak free
    peak=$(awk -F', ' '$2 ~ /^[0-9]+$/ { if ($2 > m) m = $2 } END { print m + 0 }' "/mnt/nvme1/prefill-chunk/$TAGP-$tag/vram.csv")
    free=$((32150 - peak))
    say "$tag: peak memory.used ${peak} MiB, min free ${free} MiB"
    echo "{\"step\": \"$tag\", \"peak_mib\": $peak, \"min_free_mib\": $free}" >> "$OUT/peaks.jsonl"
    [ "$free" -ge 64 ] || { say "$tag: min free ${free} MiB < 64; stopping"; return 1; }
}

for step in "${STEPS[@]}"; do
    case $step in
        decode) decode_arm ;;
        lm16k) smoke lm16k 16384 ;;
        lm64k) smoke lm64k 65536 ;;
        ch16k) smoke ch16k 16384 SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 ;;
        ch64k) smoke ch64k 65536 SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 ;;
        lm16k-2) smoke lm16k-2 16384 ;;
        lm64k-2) smoke lm64k-2 65536 ;;
        ch16k-2) smoke ch16k-2 16384 SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 ;;
        ch64k-2) smoke ch64k-2 65536 SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 ;;
        *) say "unknown step $step"; false ;;
    esac || { say "step $step failed; stopping"; say "ALL DONE rc=1"; exit 1; }
done
say "ALL DONE rc=0"
