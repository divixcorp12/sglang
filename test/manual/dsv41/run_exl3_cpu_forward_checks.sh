#!/usr/bin/env bash
# Bit-exact gates for a change to the optimized EXL3 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#
#   run_exl3_cpu_forward_checks.sh baseline WORKTREE OUT
#       At the merge-base: exl3_cpu_forward_ab.py dumps per ISA tier (scalar, avx2, bw), through make_layer.
#   run_exl3_cpu_forward_checks.sh check WORKTREE OUT BASELINE_OUT [slabs]
#       (1) the same dumps, compared bitwise with BASELINE_OUT's; with `slabs`, also through the slab registration,
#       compared with the same make_layer baseline; (2) the bare-forward bench's 24 frozen DSV4.1 outputs;
#       (3) the full-stack bench's 48; (4) the CPU expert pool tests.
#
# Builds into OUT: a private copy of the extension's build directory (~670 MB) and the bench. Exits nonzero when any
# step fails; each step's log is OUT/<step>.log.
set -uo pipefail
mode=${1:?mode}
wt=$(realpath "${2:?worktree}")
out=${3:?output dir}
base=${4:-}
slabs=${5:-}
mkdir -p "$out/tmp"
out=$(realpath "$out")

if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the benches pin CPUs 16-33 and 52" >&2
  exit 2
fi

PY=/data/models/slang/.venv/bin/python
GXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
export TMPDIR=$out/tmp
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=$out/exl3-build
export SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_EXL3_CPU_CXX=$GXX CUDA_HOME=/usr/local/cuda-13.4
export CXX=$GXX  # the host module's JIT build (kernel_layer/kernel_forward from Task 6): the kernels' GCC 15
export PYTHONPATH=$wt/python OMP_NUM_THREADS=8 EXL3_MOE_CPU_PIN=0
if [[ ! -d $SGLANG_EXL3_BUILD_DIR/resid_b128_cpu_v1 ]]; then
  mkdir -p "$SGLANG_EXL3_BUILD_DIR"
  cp -a ~/.cache/sglang/exl3_ext/resid_b128_cpu_v1 "$SGLANG_EXL3_BUILD_DIR/"
fi
cd "$wt"
echo "worktree $wt at $(git rev-parse --short HEAD)"

failed=()
step() {
  local name=$1
  shift
  "$@" > "$out/$name.log" 2>&1
  local rc=$?
  echo "$name EXIT=$rc  ($(tail -1 "$out/$name.log"))"
  [[ $rc -eq 0 ]] || failed+=("$name")
}

ab=$wt/test/manual/dsv41/exl3_cpu_forward_ab.py
registrations=(table)
[[ $slabs == slabs ]] && registrations+=(slabs)
for isa in bw avx2 scalar; do
  for reg in "${registrations[@]}"; do
    [[ $mode == baseline && $reg != table ]] && continue
    step "dump-$isa-$reg" taskset -c 0-63 "$PY" "$ab" dump --isa "$isa" --registration "$reg" --out "$out/ab-$isa-$reg.pt"
    if [[ $mode == check ]]; then
      step "compare-$isa-$reg" "$PY" "$ab" compare "$base/ab-$isa-table.pt" "$out/ab-$isa-$reg.pt"
    fi
  done
done

if [[ $mode == check ]]; then
  bench=$wt/python/sglang/kernels/jit/csrc/moe/expert_stream/bench
  build=$out/bench-build
  configure=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=$GXX" "-DEXL3_TORCH_ROOT=$SITE/torch"
    -DEXL3_CXX11_ABI=1 "-DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi")
  gbench=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src
  [[ -d $gbench ]] && configure+=("-DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=$gbench")
  step bench-configure taskset -c 0-63 cmake -S "$bench" -B "$build" "${configure[@]}"
  step bench-build taskset -c 0-63 cmake --build "$build" -j16 --target exl3_cpu_optimized exl3_full_stack_prod
  export EXL3_MOE_CPU_MAX_ISA=bw OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  step bare-validate "$build/exl3_cpu_optimized" --validate-only
  mkdir -p "$out/images"
  step full-stack-validate "$build/exl3_full_stack_prod" --validate-only "--image-dir=$out/images"
  unset EXL3_MOE_CPU_MAX_ISA
  step pytest taskset -c 0-63 "$PY" -m pytest -q -p no:randomly \
    test/manual/dsv41/test_cpu_expert_pool_exl3.py test/registered/unit/kernels/test_cpu_expert_pool.py
fi

if ((${#failed[@]})); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "ALL GREEN ($mode)"
