#!/usr/bin/env bash
# Recipe MEM_FRACTION_STATIC 0.91 checks on divix01, at the recipe as committed (no fraction override):
#   1. one standard decode arm (run_arm.sh), for ms/token and the KV sizing;
#   2. long-prompt prefill smokes (prefill-chunk/chunk_smoke.sh, 4096-token chunks, hot cache 16100 = recipe):
#      16k and 64k at the recipe (suffixes >= 8192 prefill layer-major), then 16k and 64k with
#      SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 (every chunk goes through the 4096-token chunked path).
# Stops at the first failure. Every launch waits for both NUMA gates of the full tier (node 0 >= 61440+4096+15360,
# node 1 >= 40960+4096+10000 MiB, MemFree + page cache). Lock order: rowimg-disk.lock, then cc-gpu.lock.
# Usage: run_checks.sh <worktree> <sha> [steps...]   steps: decode lm16k lm64k ch16k ch64k (default: all, in order)
set -u
WT=${1:?worktree}; SHA=${2:?sha}; shift 2
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(decode lm16k lm64k ch16k ch64k)
PY=/data/models/slang/.venv/bin/python
OUT=/mnt/nvme1/recipe-mem
DISK_LOCK=/data/models/slang/nvfp4-work/rowimg-disk.lock
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
PORT=30051
mkdir -p "$OUT"
say() { echo "$(date +%T) $*"; }

[ "$(git -C "$WT" rev-parse HEAD)" = "$SHA" ] || { say "$WT not at $SHA"; exit 1; }
[ -z "$(git -C "$WT" status --porcelain --untracked-files=no)" ] || { say "$WT has tracked changes"; exit 1; }
PYTHONPATH=$WT/python $PY -c "import sglang; print('sglang from', sglang.__file__)" || exit 1
mfs=$(PYTHONPATH=$WT/benchmarks/dsv41_baseline $PY -c "import arm_env; print(arm_env.MEM_FRACTION_STATIC)")
[ "$mfs" = 0.91 ] || { say "arm_env.MEM_FRACTION_STATIC is $mfs, not 0.91"; exit 1; }
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
    EXPECT_SHA=$SHA DSV41_WORKTREE=$WT bash "$WT/benchmarks/dsv41_baseline/run_arm.sh" mem091 "$PORT" \
        > "$OUT/decode-run_arm.log" 2>&1 < /dev/null 8>&-
    rc=$?
    exec 8>&-
    say "decode: run_arm rc=$rc ($(sed -n 's/^arm=.* run_dir=\([^ ]*\) .*/\1/p' "$OUT/decode-run_arm.log" | head -1))"
    return $rc
}
smoke() {  # <tag> <tokens> [SMOKE_ENV]: chunk_smoke.sh takes both locks itself, in order, and waits for them
    local tag=$1 tokens=$2 extra=${3:-} rc
    wait_gates "$tag" || return 1
    SMOKE_ENV=$extra bash "$WT/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh" "mem091-$tag" "$WT" 4096 16100 "$tokens" \
        > "$OUT/$tag-smoke.log" 2>&1 < /dev/null
    rc=$?
    say "$tag: chunk_smoke rc=$rc ($(grep -E 'long prompt:|OOM retries|long rc' /mnt/nvme1/prefill-chunk/mem091-$tag/driver.log | tr '\n' ' '))"
    return $rc
}

for step in "${STEPS[@]}"; do
    case $step in
        decode) decode_arm ;;
        lm16k) smoke lm16k 16384 ;;
        lm64k) smoke lm64k 65536 ;;
        ch16k) smoke ch16k 16384 SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 ;;
        ch64k) smoke ch64k 65536 SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0 ;;
        *) say "unknown step $step"; false ;;
    esac || { say "step $step failed; stopping"; say "ALL DONE rc=1"; exit 1; }
done
say "ALL DONE rc=0"
