#!/usr/bin/env bash
# Bit-exact gates for a change to the optimized EXL3 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#
#   run_exl3_cpu_forward_checks.sh baseline WORKTREE OUT
#       At the merge-base: exl3_cpu_forward_ab.py dumps per ISA tier (vnni, bw, avx2, scalar), through slab layers.
#   run_exl3_cpu_forward_checks.sh check WORKTREE OUT BASELINE_OUT
#       (1) the same dumps, compared bitwise with BASELINE_OUT's; (2) the bare-forward bench's 24 frozen DSV4.1 outputs;
#       (3) the full-stack bench's 48; (4) the CPU expert engine and service tests. (2) and (3) run twice: built for
#       the host's tier and at -march=skylake-avx512 (bw). The dumps' tiers are builds at -march too (exl3_cpu_forward_ab.py).
#
# Builds into OUT: a private copy of the extension's build directory (~670 MB) and the bench. Exits nonzero when any
# step fails; each step's log is OUT/<step>.log. CPU jobs run under taskset -c 0-51,56-107 (the bench's 22-43 sit
# inside it); the interpreter is /data/models/slang/.venv-upstream, so PYTHONPATH must be the worktree's (it is).
set -uo pipefail
mode=${1:?mode}
wt=$(realpath "${2:?worktree}")
out=${3:?output dir}
base=${4:-}
mkdir -p "$out/tmp"
out=$(realpath "$out")

if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the benches pin CPUs 22, 26, 27 and 28-43" >&2
  exit 2
fi

CPU_JOB_CORES=${CPU_JOB_CORES:-0-51,56-107}  # arm_env.CPU_JOB_CORES; narrow it (0-27) to stay off node 1 while others time there
BENCH_CORES=${BENCH_CORES:-$CPU_JOB_CORES}    # the bench validates need 22-27 and 28-43 in their mask
# A command run before each bench validate (they use node 1's 28-43 for a few seconds), e.g. one that waits until
# another job's binaries are off those cores.
NODE1_GATE=${NODE1_GATE:-true}
PY=/data/models/slang/.venv-upstream/bin/python
GXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
SITE=/data/models/slang/.venv-upstream/lib/python3.13/site-packages
export TMPDIR=$out/tmp
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=$out/exl3-build
export SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1 SGLANG_EXL3_CPU_CXX=$GXX CUDA_HOME=/usr/local/cuda-13.4
export CXX=$GXX  # the host module's JIT build (kernel_layer/kernel_forward from Task 6): the kernels' GCC 15
export PYTHONPATH=$wt/python OMP_NUM_THREADS=8 SGLANG_EXL3_MOE_CPU_PIN=0
export CUDA_VISIBLE_DEVICES=9  # CPU-only: no device 9, so no CUDA context (an empty value crashes sglang.test.test_utils)
# The default build is resid_b128_m16_cpu_v1 (SGLANG_EXL3_CPU_MAX_M=16; m8 before it); resid_b128_cpu_v1 is MAX_M=0's.
# Seed whichever the user cache holds, so a run does not rebuild them.
for flavor in resid_b128_m16_cpu_v1 resid_b128_m8_cpu_v1 resid_b128_cpu_v1; do
  if [[ ! -d $SGLANG_EXL3_BUILD_DIR/$flavor && -d ~/.cache/sglang/exl3_ext/$flavor ]]; then
    mkdir -p "$SGLANG_EXL3_BUILD_DIR"
    cp -a ~/.cache/sglang/exl3_ext/$flavor "$SGLANG_EXL3_BUILD_DIR/"
  fi
done
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
for isa in vnni bw avx2 scalar; do
  step "dump-$isa" taskset -c "$CPU_JOB_CORES" "$PY" "$ab" dump --isa "$isa" --registration slabs --out "$out/ab-$isa-slabs.pt"
  if [[ $mode == check ]]; then
    step "compare-$isa" "$PY" "$ab" compare "$base/ab-$isa-slabs.pt" "$out/ab-$isa-slabs.pt"
  fi
done

if [[ $mode == check ]]; then
  bench=$wt/python/sglang/kernels/jit/csrc/moe/expert_stream/bench
  build=$out/bench-build
  configure=(-DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_COMPILER=$GXX" "-DEXL3_TORCH_ROOT=$SITE/torch"
    -DEXL3_CXX11_ABI=1 "-DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi")
  gbench=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src
  [[ -d $gbench ]] && configure+=("-DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=$gbench")
  step bench-configure taskset -c "$CPU_JOB_CORES" cmake -S "$bench" -B "$build" "${configure[@]}"
  step bench-build taskset -c "$CPU_JOB_CORES" cmake --build "$build" -j16 --target exl3_cpu_optimized exl3_full_stack_prod
  # The bw tier is its own build, at -march=skylake-avx512.
  build_bw=$out/bench-build-bw
  step bench-configure-bw taskset -c "$CPU_JOB_CORES" cmake -S "$bench" -B "$build_bw" "${configure[@]}" -DEXL3_MARCH=skylake-avx512
  step bench-build-bw taskset -c "$CPU_JOB_CORES" cmake --build "$build_bw" -j16 --target exl3_cpu_optimized exl3_full_stack_prod
  export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
  export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  mkdir -p "$out/images"
  # The benches' default cores (bare 28-43; full stack writer 22, service 26, copy 27, workers 28-43) sit inside
  # CPU_JOB_CORES. Once at the host's tier (the default build), once at bw (its own build).
  gated() { eval "$NODE1_GATE"; "$@"; }
  step bare-validate gated taskset -c "$BENCH_CORES" "$build/exl3_cpu_optimized" --validate-only
  step full-stack-validate gated taskset -c "$BENCH_CORES" "$build/exl3_full_stack_prod" --validate-only "--image-dir=$out/images"
  step bare-validate-bw gated taskset -c "$BENCH_CORES" "$build_bw/exl3_cpu_optimized" --validate-only
  step full-stack-validate-bw gated taskset -c "$BENCH_CORES" "$build_bw/exl3_full_stack_prod" --validate-only "--image-dir=$out/images"
  step pytest taskset -c "$CPU_JOB_CORES" "$PY" -m pytest -q -p no:randomly \
    test/manual/dsv41/test_cpu_expert_engines_exl3.py test/registered/unit/kernels/test_cpu_expert_service.py \
    test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py
  # The multi-token plan test is documented to run at the host's tier and built at bw.
  step pytest-plan-bw env SGLANG_JIT_HOST_MARCH=skylake-avx512 taskset -c "$CPU_JOB_CORES" "$PY" -m pytest -q -p no:randomly \
    test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py
fi

if ((${#failed[@]})); then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "ALL GREEN ($mode)"
