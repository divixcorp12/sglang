#!/usr/bin/env bash
# Tolerance gate for a change to the NVFP4 CPU expert kernel, on divix01 (CPU only, no GPU lock).
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT              build from WORKTREE, dump every variant into OUT
#   run_nvfp4_cpu_forward_checks.sh WORKTREE OUT BASE_OUT     ... then compare each dump with BASE_OUT's at a tolerance
# Variants: avx2 (a build at -march=x86-64-v3) and scalar (-march=x86-64), each its own build of the harness: the ISA
# tier is the build's -march. The kernel is built -Ofast, so a dump is compared with nvfp4_cpu_dump_compare.py's
# relative tolerance (NVFP4_AB_RTOL, default 1e-2: one Q8_0 quantum), not byte for byte; the harness itself still checks that a batched
# call equals its one-row calls bit for bit.
# CXX defaults to GCC 15; NVFP4_AB_CORES to 0-7 (NUMA node 0).
# NVFP4_AB_BASE_NAMES names BASE_OUT's dumps for avx2,scalar (default avx2,scalar).
# NVFP4_AB_HARNESS runs another revision's harness source against WORKTREE's kernel (to re-baseline a harness change).
set -uo pipefail
wt=$(realpath "$1"); out=$2; base=${3:-}
cxx=${CXX:-/opt/rh/gcc-toolset-15/root/usr/bin/g++}
py=/data/models/slang/.venv-upstream/bin/python
cores=${NVFP4_AB_CORES:-0,1,2,3,4,5,6,7}
rtol=${NVFP4_AB_RTOL:-1e-2}
build=$wt/python/sglang/srt/layers/quantization/nvfp4/build.py
harness=${NVFP4_AB_HARNESS:-$wt/test/manual/dsv41/nvfp4_cpu_forward_ab.cpp}
compare=$wt/test/manual/dsv41/nvfp4_cpu_dump_compare.py
variants=(avx2 scalar)
marches=(x86-64-v3 x86-64)
IFS=, read -ra base_names <<< "${NVFP4_AB_BASE_NAMES:-avx2,scalar}"
if (( ${#base_names[@]} != ${#variants[@]} )); then
  echo "NVFP4_AB_BASE_NAMES must name ${#variants[@]} dumps (avx2,scalar), got '${NVFP4_AB_BASE_NAMES:-}'"; exit 2
fi
mkdir -p "$out"
export PYTHONPATH=$wt/python
fail=0
for i in "${!variants[@]}"; do
  variant=${variants[$i]}; base_name=${base_names[$i]}
  exe=$out/ab-$variant
  if ! taskset -c 0-51,56-107 "$py" "$build" --cxx "$cxx" --output "$exe" --main "$harness" --march "${marches[$i]}" \
      > "$out/build-$variant.log" 2>&1; then
    echo "FAIL $variant: build (see $out/build-$variant.log)"; fail=1; continue
  fi
  if ! OMP_WAIT_POLICY=ACTIVE taskset -c "$cores" "$exe" "$out/$variant.bin" ${cores//,/ } > "$out/run-$variant.log" 2>&1; then
    echo "FAIL $variant: run (see $out/run-$variant.log)"; fail=1; continue
  fi
  if [[ -z $base ]]; then
    echo "DUMPED $variant: $(tail -1 "$out/run-$variant.log")"
  elif "$py" -I "$compare" "$base/$base_name.bin" "$out/$variant.bin" "$rtol" > "$out/compare-$variant.log" 2>&1; then
    echo "PASS $variant vs $base_name: $(tail -1 "$out/compare-$variant.log")"
  else
    echo "FAIL $variant differs from $base/$base_name.bin (see $out/compare-$variant.log)"; fail=1
  fi
done
exit $fail
