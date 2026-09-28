#!/usr/bin/env bash
# Build and run wake_probe.cu on divix01. Run from the repo root of a pulled worktree:
#   flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
#     bash analysis/dsv41-drive/sleep-free-wait/run_wake_probe.sh <out_dir> [replays] [warmup]
# Echo thread on cpu 40, main thread on cpu 42: both in 32-63 and on the GPU's NUMA node 0 (0-17,36-53).
set -euo pipefail
out=${1:?out dir}; replays=${2:-20000}; warmup=${3:-2000}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
nvcc=/usr/local/cuda-13.4/bin/nvcc
mkdir -p "$out"
"$nvcc" -O2 -std=c++20 -arch=sm_120 -o "$out/wake_probe" "$here/wake_probe.cu" -lcuda
echo "built with $("$nvcc" --version | tail -1)"
"$out/wake_probe" "$replays" "$warmup" 40 42 "$out"
