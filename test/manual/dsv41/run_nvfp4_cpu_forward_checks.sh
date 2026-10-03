#!/usr/bin/env bash
# Bit-exact gate for a change to the NVFP4 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT              build every variant from WORKTREE and dump into OUT
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT BASE_OUT     ... then compare each dump bitwise with BASE_OUT's
# Variants: native (-march=native, the AVX2 dot), baseline (GGML conversion, the bench's baseline backend),
# portable (no -march=native, the scalar dot). CXX defaults to GCC 15; NVFP4_AB_CORES to 0-7 (NUMA node 0).
# NVFP4_AB_HARNESS runs another revision's harness source against WORKTREE's kernel (to re-baseline a harness change).
set -uo pipefail
wt=$(realpath "$1"); out=$2; base=${3:-}
cxx=${CXX:-/opt/rh/gcc-toolset-15/root/usr/bin/g++}
py=/data/models/slang/.venv/bin/python
cores=${NVFP4_AB_CORES:-0,1,2,3,4,5,6,7}
build=$wt/python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py
harness=${NVFP4_AB_HARNESS:-$wt/test/manual/dsv41/nvfp4_cpu_forward_ab.cpp}
mkdir -p "$out"
fail=0
for variant in native baseline portable; do
  flags=()
  case $variant in baseline) flags=(--upstream-baseline) ;; portable) flags=(--portable) ;; esac
  exe=$out/ab-$variant
  if ! taskset -c 0-63 "$py" "$build" --cxx "$cxx" --output "$exe" --main "$harness" "${flags[@]}" \
      > "$out/build-$variant.log" 2>&1; then
    echo "FAIL $variant: build (see $out/build-$variant.log)"; fail=1; continue
  fi
  if ! OMP_WAIT_POLICY=ACTIVE taskset -c "$cores" "$exe" "$out/$variant.bin" ${cores//,/ } \
      > "$out/run-$variant.log" 2>&1; then
    echo "FAIL $variant: run (see $out/run-$variant.log)"; fail=1; continue
  fi
  if [[ -z $base ]]; then
    echo "DUMPED $variant: $(cat "$out/run-$variant.log")"
  elif cmp -s "$out/$variant.bin" "$base/$variant.bin"; then
    echo "PASS $variant bit-exact"
  else
    echo "FAIL $variant differs from $base/$variant.bin"; fail=1
  fi
done
exit $fail
