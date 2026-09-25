#!/usr/bin/env bash
# Resident-first steps 1 and 2 on divix01: the split parity test, then the split-launch microbenchmark.
# Usage (from the worktree root): gpu-run.sh analysis/dsv41-drive/resident-first/run.sh <outdir>
set -u
OUT=${1:?outdir}
WT=$(pwd)
PY=/data/models/slang/.venv/bin/python
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
export CUDA_HOME=/usr/local/cuda-13.2
export OMP_NUM_THREADS=8
export PYTHONPATH=$WT/python
mkdir -p "$OUT/pytest-tmp"
git log -1 --oneline | tee "$OUT/commit.txt"
$PY -c 'import sglang; print("sglang from", sglang.__file__)' | tee "$OUT/sglang_file.txt"
echo "cmd: pytest -q -s -p no:randomly --basetemp=$OUT/pytest-tmp test/manual/dsv41/test_exl3_moe_split_parity_cuda.py" > "$OUT/parity.log"
$PY -m pytest -q -s -p no:randomly --basetemp="$OUT/pytest-tmp" test/manual/dsv41/test_exl3_moe_split_parity_cuda.py >> "$OUT/parity.log" 2>&1
echo "EXIT=$?" >> "$OUT/parity.log"
tail -3 "$OUT/parity.log"
echo "cmd: python analysis/dsv41-drive/resident-first/split_launch_bench.py --replays 300 --out $OUT/bench.json" > "$OUT/bench.log"
$PY analysis/dsv41-drive/resident-first/split_launch_bench.py --replays 300 --out "$OUT/bench.json" >> "$OUT/bench.log" 2>&1
echo "EXIT=$?" >> "$OUT/bench.log"
cat "$OUT/bench.log"
