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
hot_gpu_mb=${HOT_GPU_MB:-14336}
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
worktree=$work/cc-expert-prediction/worktree
flashinfer_overlay=$work/flashinfer-0.6.18-cu130-overlay
model=/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
cache_model_path=/data/models/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
expert_cache=/mnt/nvme2/nvfp4-work/qwen38-nvfp4-expert-cache-v1
ple_cache=/mnt/nvme2/ple-cache/qwen38-nvfp4
expert_seed=/data/models/slang/slang-dev-2bit/qwen3.8-flash-next-24gb-sglang/assets/expert_freq.pt
run_dir=$work/cc-expert-prediction/servers/$name/run-$(date +%Y%m%d-%H%M%S)
log=$run_dir/server.log

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING_TO_START: the GPU is in use" >&2
    exit 1
fi

mkdir -p "$run_dir/profiles" "$work/runtime-tmp"
ln -sfn "$run_dir" "$work/cc-expert-prediction/servers/$name/latest"
cd "$worktree"
{
    echo "cc-expert-prediction server $name port=$port predictors=${predictors:-off} radix=$([ "$radix" = radix ] && echo on || echo off) hot_gpu_mb=$hot_gpu_mb capture_dir=${capture_dir:-none}: $(date --iso-8601=seconds)"
    git status --short --branch
    git log -1 --oneline
    sha256sum python/sglang/srt/model_executor/model_runner.py python/sglang/srt/layers/moe/expert_prediction/*.py
} 2>&1 | tee -a "$log"

exec env \
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
    SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0 \
    SGLANG_MOE_EXPERT_COPY_BACKEND=dma \
    SGLANG_MOE_EXPERT_PREDICTOR="$predictors" \
    SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL=100 \
    SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE="$run_dir/expert-prediction.metrics.jsonl" \
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR="$capture_dir" \
    SGLANG_TORCH_PROFILER_DIR="$run_dir/profiles" \
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$run_dir/profiles/expert-distribution" \
    SGLANG_VLM_CACHE_SIZE_MB=0 \
    SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=4 \
    SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S=5 \
    /data/models/slang/.venv/bin/sglang serve --model-type llm \
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
        --context-length 65536 \
        --max-total-tokens 65536 \
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
