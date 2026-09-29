#!/usr/bin/env bash
# Final-fix round (hotpath-zero-overhead): run one pytest node id N times, sequentially, in <worktree> (CPU-TEST:
# taskset -c 0-63, OMP_NUM_THREADS=8), each run's pytest status read directly (no pipe), and print the pass count.
# LOAD=<n> (default 0) also keeps n busy-loop shells on cores 0-63 for the whole loop: the "under load" condition in
# which the prefill prefix flake shows (b3-run-report.md), since an idle box lets the fill thread reap row by row.
# Usage: [LOAD=n] repeat_test.sh <worktree> <tag> <runs> <out_dir> <pytest node id>
set -u
wt=${1:?worktree}; tag=${2:?tag}; runs=${3:?runs}; out=${4:?out dir}; node=${5:?pytest node id}
PY=/data/models/slang/.venv/bin/python
mkdir -p "$out"
cd "$wt" || exit 1
export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4
echo "$tag: $(git log -1 --oneline), status: '$(git status --short | tr '\n' ' ')'"
taskset -c 0-63 $PY -c "import sglang; print(sglang.__file__)"
LOADERS=()
for _ in $(seq 1 "${LOAD:-0}"); do
    taskset -c 0-63 nice -n 5 sh -c 'while :; do :; done' &
    LOADERS+=($!)
done
trap '[ ${#LOADERS[@]} = 0 ] || kill "${LOADERS[@]}" 2>/dev/null' EXIT
echo "$tag: LOAD=${LOAD:-0} busy loops"
pass=0
for i in $(seq 1 "$runs"); do
    taskset -c 0-63 $PY -m pytest "$node" -q -p no:randomly > "$out/$tag-$i.log" 2>&1
    rc=$?
    echo "$tag run $i EXIT=$rc $(tail -1 "$out/$tag-$i.log")"
    [ "$rc" = 0 ] && pass=$((pass + 1))
done
echo "$tag: $pass/$runs passed ($node)"
