#!/usr/bin/env bash
# Launches one expert-prediction shadow server with production's E16c settings from the
# cc-expert-prediction worktree on 127.0.0.1:<port>. <predictors> is a comma list or "off".
# Usage: run-shadow-server.sh <name> <port> <predictors|off> [radix]
# Refuses to start while any process holds the GPU. Runs in the foreground; the session backgrounds it.
# Env: HOT_GPU_MB (default 14336), CAPTURE=1 to record expert prediction training data.
set -euo pipefail

name=${1:?name}
port=${2:?port}
predictors=${3:?predictors or off}
radix=${4:-}
[ "$predictors" = off ] && predictors=""
hot_gpu_mb=${HOT_GPU_MB:-10240}
predictor=${PREFETCH_PREDICTOR:-off}
[ "$predictor" = off ] && predictor=""
prefetch_model_dir=${PREFETCH_MODEL_DIR:-/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630}
prefetch_budget=${PREFETCH_BUDGET:-2}
prefetch_candidates=${PREFETCH_CANDIDATES:-16}
prefetch_pull_mode=${PREFETCH_PULL_MODE:-off}
shadow_recall=${PREFETCH_SHADOW_RECALL:-0}
calibration=${PREFETCH_CALIBRATION:-0}
run_kind=${RUN_KIND:-timed}
trace_command=${PREFETCH_TRACE_COMMAND:-}
arm=${EXPERIMENT_ARM:-}
pass_id=${RUN_PASS_ID:-}
session_ids_json=${SESSION_IDS_JSON:-[]}
session_set_checksum=${SESSION_SET_CHECKSUM:-}
checkpoint_checksum=${PREFETCH_CHECKPOINT_CHECKSUM:-unknown}
model_shapes=${MODEL_SHAPES:-unknown}
model_top_k=${MODEL_TOP_K:-10}
provenance_only=${PREFETCH_PROVENANCE_ONLY:-0}
case "$run_kind" in timed|profiling|trace) ;; *) echo "RUN_KIND must be timed, profiling, or trace" >&2; exit 2 ;; esac
case "$prefetch_pull_mode" in off|count_zero|always) ;; *) echo "PREFETCH_PULL_MODE is invalid" >&2; exit 2 ;; esac
case "${calibration,,}" in
    1|true|yes|y) calibration=1 ;;
    0|false|no|n) calibration=0 ;;
    *) echo "PREFETCH_CALIBRATION must be a boolean" >&2; exit 2 ;;
esac
case "${provenance_only,,}" in
    1|true|yes|y) provenance_only=1 ;;
    0|false|no|n) provenance_only=0 ;;
    *) echo "PREFETCH_PROVENANCE_ONLY must be a boolean" >&2; exit 2 ;;
esac
if [ "$model_top_k" != 10 ]; then
    echo "MODEL_TOP_K must be the production BS1 top-k value 10" >&2
    exit 2
fi
if [ "$run_kind" = timed ] && [ "$calibration" = 1 ]; then
    echo "REFUSING_TO_START: calibration is profiling-only; timed B/C/Cr/N/D runs must disable it" >&2
    exit 2
fi
if [ "$run_kind" = trace ] && [ "$calibration" = 1 ]; then
    echo "REFUSING_TO_START: trace is diagnostic-only and must disable calibration" >&2
    exit 2
fi
if [ "$run_kind" = trace ] && [ -z "$trace_command" ]; then
    echo "PREFETCH_TRACE_COMMAND is required for RUN_KIND=trace" >&2
    exit 2
fi
trace_command_argv=()
trace_config=null
if [ "$run_kind" = trace ]; then
    # This deliberately parses simple argv tokens only; it never evaluates shell input.
    # A wrapper with an argument containing whitespace is unsupported.
    read -r -a trace_command_argv <<< "$trace_command"
    trace_command_json=$(printf '%s' "$trace_command" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')
fi
capture_dir=""
if [ "${CAPTURE:-}" = 1 ]; then
    capture_dir=/mnt/nvme2/nvfp4-work/expert-prediction-capture/$name/$(date +%Y%m%d-%H%M%S)
fi

if [ "$radix" = radix ]; then
    radix_flags=(--mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 8)
else
    radix_flags=(--disable-radix-cache --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 1)
fi

work=/data/models/slang/nvfp4-work
worktree=${PREFETCH_WORKTREE:-$work/cc-expert-prediction/worktree}
flashinfer_overlay=$work/flashinfer-0.6.18-cu130-overlay
model=/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
cache_model_path=/data/models/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
expert_cache=/mnt/nvme2/nvfp4-work/qwen38-nvfp4-expert-cache-v1
ple_cache=/mnt/nvme2/ple-cache/qwen38-nvfp4
expert_seed=/data/models/slang/slang-dev-2bit/qwen3.8-flash-next-24gb-sglang/assets/expert_freq.pt
run_dir=${PREFETCH_RUN_DIR:-$work/cc-expert-prediction/servers/$name/run-$(date +%Y%m%d-%H%M%S)}
log=$run_dir/server.log
trace_report_path=$run_dir/trace/report
if [ "$run_kind" = trace ]; then
    trace_report_path_json=$(printf '%s' "$trace_report_path" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')
    trace_config=$(printf '{"diagnostic_only":true,"command":%s,"report_path":%s}' "$trace_command_json" "$trace_report_path_json")
fi

# This object is deliberately typed rather than derived from a display string:
# the CUDA graph flags below declare BS1 and the model's routing geometry is
# fixed at top-k 10.  The offline gate calibrator rejects ambiguous metadata.
shape_provenance='{"batch_size":1,"top_k":10,"top_k_unique":true,"cuda_graph_decode_batch_size":1,"cuda_graph_max_decode_batch_size":1}'

write_provenance() {
    calibration_provenance=$(printf '{"commit":"%s","predictor":"%s","checkpoint_dir":"%s","checkpoint_checksum":"%s","cache_size":%s,"session_ids":%s,"session_set_checksum":"%s","model_shapes":"%s","shape_provenance":%s,"batch_size":1,"top_k":10,"top_k_unique":true,"bin_count":256,"bin_edges":"uniform [0,1], lower-inclusive; 1.0 is in bin 255"}' "$commit" "${predictor:-empty}" "$prefetch_model_dir" "$checkpoint_checksum" "$hot_gpu_mb" "$session_ids_json" "$session_set_checksum" "$model_shapes" "$shape_provenance")
    printf '{"arm":"%s","pass_id":"%s","commit":"%s","flags":{"fused_plan":1,"pull_mode":"%s","shadow_recall":%s,"calibration":%s,"candidates":%s,"budget":%s},"cache_size":%s,"predictor":"%s","checkpoint_dir":"%s","checkpoint_checksum":"%s","run_kind":"%s","trace":%s,"session_ids":%s,"session_set_checksum":"%s","model_shapes":"%s","shape_provenance":%s,"batch_size":1,"top_k":10,"top_k_unique":true,"calibration_provenance":{"enabled":%s,"bin_count":256,"range":"[0,1]","bin_edges":"uniform [0,1], lower-inclusive; 1.0 is in bin 255","batch_size":1,"top_k":10,"top_k_unique":true,"shape_provenance":%s},"paths":{"results":"%s/results.jsonl","prediction_metrics":"%s/expert-prediction.metrics.jsonl","hot_cache_metrics":"%s/hot-cache.metrics.jsonl","calibration":"%s/pull-calibration.json","trace_report":"%s","startup_log":"%s"}}\n' "$arm" "$pass_id" "$commit" "$prefetch_pull_mode" "$shadow_recall" "$calibration" "$prefetch_candidates" "$prefetch_budget" "$hot_gpu_mb" "${predictor:-empty}" "$prefetch_model_dir" "$checkpoint_checksum" "$run_kind" "$trace_config" "$session_ids_json" "$session_set_checksum" "$model_shapes" "$shape_provenance" "$calibration" "$shape_provenance" "$run_dir" "$run_dir" "$run_dir" "$run_dir" "$trace_report_path" "$log" > "$run_dir/run-manifest.json"
}

if [ "$provenance_only" = 1 ]; then
    mkdir -p "$run_dir"
    commit=$(git -C "$worktree" rev-parse HEAD)
    write_provenance
    printf 'wrote provenance-only manifest: %s\n' "$run_dir/run-manifest.json"
    exit 0
fi

if ss -ltn 'sport = :7867' | grep -q LISTEN; then
    echo "REFUSING_TO_START: production on 7867 is up or relaunching" >&2
    exit 1
fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING_TO_START: the GPU is in use" >&2
    exit 1
fi

mkdir -p "$run_dir/profiles" "$run_dir/trace" "$work/runtime-tmp"
ln -sfn "$run_dir" "$work/cc-expert-prediction/servers/$name/latest"
cd "$worktree"
{
    echo "cc-expert-prediction server $name port=$port predictors=${predictors:-off} radix=$([ "$radix" = radix ] && echo on || echo off) hot_gpu_mb=$hot_gpu_mb capture_dir=${capture_dir:-none} predictor=${predictor:-off} pull_mode=$prefetch_pull_mode shadow_recall=$shadow_recall calibration=$calibration run_kind=$run_kind fused_plan=1 candidates=$prefetch_candidates budget=$prefetch_budget output=$run_dir: $(date --iso-8601=seconds)"
    if [ "$run_kind" = trace ]; then
        echo "trace diagnostic-only wrapper=$trace_command report_path=$trace_report_path; wrapper uses simple whitespace-separated argv tokens only (embedded-whitespace arguments are unsupported)"
    fi
    git status --short --branch
    git log -1 --oneline
    sha256sum python/sglang/srt/model_executor/model_runner.py python/sglang/srt/layers/moe/expert_prediction/*.py
} 2>&1 | tee -a "$log"
commit=$(git rev-parse HEAD)
write_provenance

exec flock --nonblock /data/models/slang/nvfp4-work/cc-gpu.lock env \
    PYTHONPATH="$flashinfer_overlay:$worktree/python" \
    PYTHONUNBUFFERED=1 \
    TMPDIR="$work/runtime-tmp" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
    SGLANG_MOE_EXPERT_STREAM=1 \
    SGLANG_MOE_EXPERT_FILE_DIR="$expert_cache" \
    SGLANG_MOE_EXPERT_FILE_READER=uring_direct \
    SGLANG_QWEN4_PLE_FILE_READER=uring \
    SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY=1 \
    SGLANG_FILE_CACHE_MODEL_PATH="$cache_model_path" \
    SGLANG_MOE_HOT_GPU_MB="$hot_gpu_mb" \
    SGLANG_MOE_PINNED_HOST_MB=0 \
    SGLANG_MOE_EXPERT_HOST_ARENA=1 \
    SGLANG_MOE_EXPERT_GRAPH_GATHER=1 \
    SGLANG_MOE_HOT_SEED="$expert_seed" \
    SGLANG_MOE_HOT_DYNAMIC=1 \
    SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=4 \
    SGLANG_MOE_HOT_DECAY_TOKENS=1 \
    SGLANG_MOE_HOT_PROMOTION_SIGMAS=0 \
    SGLANG_MOE_HOT_BENEFIT_RATIO=2 \
    SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS=0 \
    SGLANG_MOE_HOT_LOG_INTERVAL=100 \
    SGLANG_MOE_HOT_METRICS_FILE="$run_dir/hot-cache.metrics.jsonl" \
    SGLANG_MOE_GPU_RESIDENCY_UPDATE=1 \
    SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS=64 \
    SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0 \
    SGLANG_MOE_EXPERT_COPY_BACKEND=dma \
    SGLANG_MOE_EXPERT_PREDICTOR="$predictors" \
    SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL=100 \
    SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE="$run_dir/expert-prediction.metrics.jsonl" \
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR="$capture_dir" \
    SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR="$predictor" \
    SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR="$prefetch_model_dir" \
    SGLANG_MOE_EXPERT_PREFETCH_BUDGET="$prefetch_budget" \
    SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES="$prefetch_candidates" \
    SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE="$prefetch_pull_mode" \
    SGLANG_MOE_EXPERT_PREFETCH_SHADOW_RECALL="$shadow_recall" \
    SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION="$calibration" \
    SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION_FILE="$run_dir/pull-calibration.json" \
    SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION_PROVENANCE="$calibration_provenance" \
    SGLANG_TORCH_PROFILER_DIR="$run_dir/profiles" \
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$run_dir/profiles/expert-distribution" \
    PREFETCH_TRACE_REPORT_PATH="$trace_report_path" \
    SGLANG_VLM_CACHE_SIZE_MB=0 \
    SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=4 \
    SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S=5 \
    "${trace_command_argv[@]}" /data/models/slang/.venv/bin/sglang serve --model-type llm \
        --model-path "$model" \
        --tp 1 \
        --fp4-gemm-backend flashinfer_cutlass \
        --moe-runner-backend flashinfer_cutlass \
        --moe-a2a-backend none \
        --cpu-offload-gb 80 \
        --ple-offload-embedding \
        --ple-offload-backend file \
        --ple-offload-dir "$ple_cache" \
        --page-size 64 \
        --mamba-track-interval 64 \
        --chunked-prefill-size 4096 \
        --max-prefill-tokens 4096 \
        --context-length 40000 \
        --max-total-tokens 40000 \
        --max-running-requests 1 \
        --mamba-ssm-dtype bfloat16 \
        --mem-fraction-static 0.95 \
        --disable-overlap-schedule \
        "${radix_flags[@]}" \
        --language-model-only \
        --cuda-graph-backend-decode breakable \
        --cuda-graph-bs-decode 1 \
        --cuda-graph-max-bs-decode 1 \
        --cuda-graph-backend-prefill disabled \
        --disable-flashinfer-autotune \
        --skip-server-warmup \
        --weight-loader-drop-cache-after-load \
        --expert-distribution-recorder-mode per_pass \
        --reasoning-parser auto \
        --default-chat-template-kwargs '{"enable_thinking": true}' \
        --host 127.0.0.1 \
        --port "$port" \
    >> "$log" 2>&1
