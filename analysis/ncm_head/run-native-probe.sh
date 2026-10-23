#!/usr/bin/env bash
# Defaults to a dry-run description. --execute is the explicit future GPU probe.
set -euo pipefail
NCM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$NCM_ROOT/python:$NCM_ROOT"
export CUDA_HOME=/usr/local/cuda-13.2
export CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
export SGLANG_EXL3_CPU_CXX="$CXX"
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR="$NCM_ROOT/ncm-exl3-build"
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONDONTWRITEBYTECODE=1
exec /data/models/slang/.venv/bin/python -m analysis.ncm_head.probe "$@"
