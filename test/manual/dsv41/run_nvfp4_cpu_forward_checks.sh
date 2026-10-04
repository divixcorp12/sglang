#!/usr/bin/env bash
# Bit-exact gate for a change to the NVFP4 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT              build from WORKTREE, dump every variant into OUT
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT BASE_OUT     ... then compare each dump bitwise with BASE_OUT's
# Variants, both from one build: avx2 (the AVX2 tier) and scalar (NVFP4_CPU_MAX_ISA=scalar). Each run must report the
# tier it names (NVFP4_CPU_REPORT_ISA=1), so an avx2 run on a host without AVX2 fails instead of dumping scalar output.
# CXX defaults to GCC 15; NVFP4_AB_CORES to 0-7 (NUMA node 0).
# NVFP4_AB_BASE_NAMES names BASE_OUT's dumps for avx2,scalar (default avx2,scalar). Baselines from before the runtime
# tiers named them native,portable: NVFP4_AB_BASE_NAMES=native,portable.
# NVFP4_AB_HARNESS runs another revision's harness source against WORKTREE's kernel (to re-baseline a harness change).
set -uo pipefail
wt=$(realpath "$1"); out=$2; base=${3:-}
cxx=${CXX:-/opt/rh/gcc-toolset-15/root/usr/bin/g++}
py=/data/models/slang/.venv/bin/python
cores=${NVFP4_AB_CORES:-0,1,2,3,4,5,6,7}
build=$wt/python/sglang/srt/layers/quantization/nvfp4_cpu/optimized/build.py
harness=${NVFP4_AB_HARNESS:-$wt/test/manual/dsv41/nvfp4_cpu_forward_ab.cpp}
variants=(avx2 scalar)
IFS=, read -ra base_names <<< "${NVFP4_AB_BASE_NAMES:-avx2,scalar}"
if (( ${#base_names[@]} != ${#variants[@]} )); then
  echo "NVFP4_AB_BASE_NAMES must name ${#variants[@]} dumps (avx2,scalar), got '${NVFP4_AB_BASE_NAMES:-}'"; exit 2
fi
mkdir -p "$out"
exe=$out/ab
if ! taskset -c 0-63 "$py" "$build" --cxx "$cxx" --output "$exe" --main "$harness" > "$out/build.log" 2>&1; then
  echo "FAIL build (see $out/build.log)"; exit 1
fi
fail=0
for i in "${!variants[@]}"; do
  variant=${variants[$i]}; base_name=${base_names[$i]}
  if ! NVFP4_CPU_MAX_ISA=$variant NVFP4_CPU_REPORT_ISA=1 OMP_WAIT_POLICY=ACTIVE taskset -c "$cores" \
      "$exe" "$out/$variant.bin" ${cores//,/ } > "$out/run-$variant.log" 2>&1; then
    echo "FAIL $variant: run (see $out/run-$variant.log)"; fail=1; continue
  fi
  if ! grep -qx "nvfp4 isa $variant" "$out/run-$variant.log"; then
    echo "FAIL $variant: the library ran another tier (see $out/run-$variant.log)"; fail=1; continue
  fi
  if [[ -z $base ]]; then
    echo "DUMPED $variant: $(grep -vx "nvfp4 isa $variant" "$out/run-$variant.log")"
  elif cmp -s "$out/$variant.bin" "$base/$base_name.bin"; then
    echo "PASS $variant bit-exact vs $base_name"
  else
    echo "FAIL $variant differs from $base/$base_name.bin"; fail=1
  fi
done
exit $fail
