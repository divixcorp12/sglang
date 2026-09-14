#!/usr/bin/env bash
# GPU accuracy session for online FP8 of the BF16 GPU weights. Requires a free GPU: stop production first.
#   fp8-session.sh A                     BF16 reference, BF16 restart noise, one run per FP8 group (KL/top-1).
#   fp8-session.sh B <groups> <hot_mb>   BF16 GSM8K, then the candidate at BASE_HOT_MB (default 14336; KL, greedy, GSM8K) and at
#                                        <hot_mb> (greedy speed + 4,096-token prefill peak). Needs phase A output.
#   fp8-session.sh diag <group>          One group with the weight-only (w8a16) scheme, to split weight from
#                                        activation error after a phase-A failure.
# Every server is stopped before the next starts. Writes STARTED, per-step EXIT lines and SESSION_EXIT to
# cc-fp8/session/<phase>.log; a missing SESSION_EXIT means the session was killed, not that it passed.
set -u

phase=${1:?phase A, B or diag}
work=/data/models/slang/nvfp4-work
root=$work/cc-fp8
harness_dir=$root/worktree/scripts/fp8_accuracy
model=/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
base_hot_mb=${BASE_HOT_MB:-14336}
port=7870
base=http://127.0.0.1:$port
out=$root/results
python=/data/models/slang/.venv/bin/python
log=$root/session/$phase-$(date +%Y%m%d-%H%M%S).log
failures=0
server_pid=
monitor_pid=

mkdir -p "$root/session" "$out"/{gen,score,gsm,compare,memory}
exec > >(tee -a "$log") 2>&1
echo "STARTED $phase $(date --iso-8601=seconds) args=$*"

harness() {
    PYTHONPATH="$root/worktree/python" taskset -c 64-71 env OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=9 \
        "$python" "$harness_dir/fp8_accuracy.py" "$@"
}

step() {
    local label=$1
    shift
    echo "== $label $(date --iso-8601=seconds)"
    "$@"
    local code=$?
    echo "EXIT[$label]=$code"
    [ "$code" -eq 0 ] || failures=$((failures + 1))
    return "$code"
}

wait_gpu_free() {
    timeout 300 bash -c 'until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 3; done'
}

start_server() {
    local name=$1 hot=$2 groups=$3 scheme=$4 layers=${5:-}
    wait_gpu_free || { echo "GPU_NOT_FREE before $name"; return 1; }
    setsid bash "$harness_dir/run-fp8-server.sh" "$name" "$port" "$hot" "$groups" "$scheme" "$layers" &
    server_pid=$!
    local peak_file=$out/memory/$name.peak
    echo 0 > "$peak_file"
    (
        while kill -0 "$server_pid" 2>/dev/null; do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            [ "$used" -gt "$(cat "$peak_file")" ] && echo "$used" > "$peak_file"
            sleep 1
        done
    ) &
    monitor_pid=$!
    timeout 1800 bash -c "until curl --fail --silent --max-time 5 $base/health >/dev/null; do kill -0 $server_pid 2>/dev/null || exit 9; sleep 5; done"
    local code=$?
    local server_log="$root/servers/$name/latest/server.log"
    grep -E "Online FP8|Load weight end|Expert hot cache startup|KV Cache is allocated|Capture target decode CUDA graph end" "$server_log" | cut -c1-320
    [ "$code" -eq 0 ] && echo "HEALTH_OK[$name]" || { echo "HEALTH_FAILED[$name]"; grep -nE "Traceback|Error|OutOfMemory" "$server_log" | tail -10 | cut -c1-300; }
    return "$code"
}

stop_server() {
    local name=$1
    [ -n "$server_pid" ] && kill -TERM -- "-$server_pid" 2>/dev/null
    wait_gpu_free || echo "GPU_NOT_RELEASED after $name"
    [ -n "$monitor_pid" ] && wait "$monitor_pid" 2>/dev/null
    local server_log="$root/servers/$name/latest/server.log"
    echo "PEAK_GPU_MIB[$name]=$(cat "$out/memory/$name.peak")"
    grep "Decode batch" "$server_log" | grep -oE "gen throughput \(token/s\): [0-9.]+" \
        | awk -v n="$name" '{s+=$4; c++} END {if (c) printf "DECODE_TOK_S[%s] logs=%d mean=%.2f\n", n, c, s/c}'
    server_pid=
    monitor_pid=
}

prefill_4096() {
    local records
    records=$(for i in $(seq 1 179); do echo "Record $i: the warehouse in district $i shipped $((i*37 % 101)) crates of copper wire."; done)
    jq -n --arg content "$records Summarize." '{text: $content, sampling_params: {max_new_tokens: 8, temperature: 0}}' \
        | curl --fail --silent --show-error --max-time 600 "$base/generate" -H "Content-Type: application/json" -d @- \
        | jq -e '.meta_info.prompt_tokens | select(. >= 4000 and . <= 4096)'
}

case "$phase" in
A)
    if start_server bf16-a "$base_hot_mb" none w8a8; then
        step gen-bf16-a harness generate --base "$base" --model-path "$model" --out "$out/gen/bf16-a.jsonl"
        step support harness support --base "$base" --gen "$out/gen/bf16-a.jsonl" --out "$out/score/support.json"
        step score-bf16-a harness score --base "$base" --gen "$out/gen/bf16-a.jsonl" --support "$out/score/support.json" --out "$out/score/bf16-a.npz"
        step score-bf16-a-repeat harness score --base "$base" --gen "$out/gen/bf16-a.jsonl" --support "$out/score/support.json" --out "$out/score/bf16-a-repeat.npz"
    else
        failures=$((failures + 1))
    fi
    stop_server bf16-a
    if start_server bf16-b "$base_hot_mb" none w8a8; then
        step score-bf16-b harness score --base "$base" --gen "$out/gen/bf16-a.jsonl" --support "$out/score/support.json" --out "$out/score/bf16-b.npz"
        step gen-bf16-b harness generate --base "$base" --model-path "$model" --out "$out/gen/bf16-b.jsonl"
    else
        failures=$((failures + 1))
    fi
    stop_server bf16-b
    for group in full_attn shared_expert gdn hc_mix lm_head; do
        if start_server "fp8-$group" "$base_hot_mb" "$group" w8a8; then
            step "score-$group" harness score --base "$base" --gen "$out/gen/bf16-a.jsonl" --support "$out/score/support.json" --out "$out/score/fp8-$group.npz"
        else
            failures=$((failures + 1))
        fi
        stop_server "fp8-$group"
    done
    step cmp-repeat harness compare-logits --ref "$out/score/bf16-a.npz" --test "$out/score/bf16-a-repeat.npz" --out "$out/compare/logits-bf16-repeat.json"
    step cmp-noise harness compare-logits --ref "$out/score/bf16-a.npz" --test "$out/score/bf16-b.npz" --out "$out/compare/logits-bf16-noise.json"
    step cmp-greedy-noise harness compare-greedy --ref "$out/gen/bf16-a.jsonl" --test "$out/gen/bf16-b.jsonl" --out "$out/compare/greedy-bf16-noise.json"
    for group in full_attn shared_expert gdn hc_mix lm_head; do
        step "cmp-$group" harness compare-logits --ref "$out/score/bf16-a.npz" --test "$out/score/fp8-$group.npz" \
            --noise "$out/compare/logits-bf16-noise.json" --out "$out/compare/logits-fp8-$group.json"
    done
    ;;
B)
    groups=${2:?groups}
    hot_mb=${3:?hot_mb}
    tag=$(echo "$groups" | tr ',' '+')
    if start_server bf16-gsm "$base_hot_mb" none w8a8; then
        step gsm-bf16 harness gsm8k --base "$base" --model-path "$model" --data "$root/data/gsm8k-test.jsonl" --out "$out/gsm/bf16.jsonl"
    else
        failures=$((failures + 1))
    fi
    stop_server bf16-gsm
    if start_server "cand-$tag" "$base_hot_mb" "$groups" w8a8; then
        step "score-cand" harness score --base "$base" --gen "$out/gen/bf16-a.jsonl" --support "$out/score/support.json" --out "$out/score/cand-$tag.npz"
        step "gen-cand" harness generate --base "$base" --model-path "$model" --out "$out/gen/cand-$tag.jsonl"
        step "gsm-cand" harness gsm8k --base "$base" --model-path "$model" --data "$root/data/gsm8k-test.jsonl" --out "$out/gsm/cand-$tag.jsonl"
    else
        failures=$((failures + 1))
    fi
    stop_server "cand-$tag"
    if start_server "cand-$tag-hot$hot_mb" "$hot_mb" "$groups" w8a8; then
        step "gen-cand-hot" harness generate --base "$base" --model-path "$model" --out "$out/gen/cand-$tag-hot$hot_mb.jsonl"
        step "prefill-4096" prefill_4096
    else
        failures=$((failures + 1))
    fi
    stop_server "cand-$tag-hot$hot_mb"
    step cmp-cand harness compare-logits --ref "$out/score/bf16-a.npz" --test "$out/score/cand-$tag.npz" \
        --noise "$out/compare/logits-bf16-noise.json" --out "$out/compare/logits-cand-$tag.json"
    step cmp-greedy-cand harness compare-greedy --ref "$out/gen/bf16-a.jsonl" --test "$out/gen/cand-$tag.jsonl" \
        --noise "$out/compare/greedy-bf16-noise.json" --out "$out/compare/greedy-cand-$tag.json"
    step cmp-gsm-cand harness compare-gsm8k --ref "$out/gsm/bf16.jsonl" --test "$out/gsm/cand-$tag.jsonl" --out "$out/compare/gsm-cand-$tag.json"
    ;;
diag)
    group=${2:?group}
    if start_server "w8a16-$group" "$base_hot_mb" "$group" w8a16; then
        step "score-w8a16-$group" harness score --base "$base" --gen "$out/gen/bf16-a.jsonl" --support "$out/score/support.json" --out "$out/score/w8a16-$group.npz"
    else
        failures=$((failures + 1))
    fi
    stop_server "w8a16-$group"
    step "cmp-w8a16-$group" harness compare-logits --ref "$out/score/bf16-a.npz" --test "$out/score/w8a16-$group.npz" \
        --noise "$out/compare/logits-bf16-noise.json" --out "$out/compare/logits-w8a16-$group.json"
    ;;
*)
    echo "unknown phase $phase"
    exit 2
    ;;
esac

echo "FAILURES=$failures"
echo "SESSION_EXIT=$([ "$failures" -eq 0 ] && echo 0 || echo 1) $(date --iso-8601=seconds)"
[ "$failures" -eq 0 ]
