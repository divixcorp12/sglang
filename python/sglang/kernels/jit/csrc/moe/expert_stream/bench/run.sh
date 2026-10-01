#!/usr/bin/env bash
set -euo pipefail

# Run from the existing isolated shell. Each backend has its own process and
# fixed team; alternate process order between rounds. No Python driver.
if [[ $# -lt 2 ]]; then
  echo "Usage: bash run.sh BUILD_DIR NEW_RESULTS_DIR [benchmark options...]" >&2
  exit 2
fi
build=$(realpath "$1")
results=$2
shift 2
mkdir "$results" # Refuse to overwrite a prior record.
results=$(realpath "$results")
rounds=${EXL3_BENCH_ROUNDS:-8}
if [[ ! $rounds =~ ^[1-9][0-9]*$ ]]; then
  echo 'EXL3_BENCH_ROUNDS must be a positive integer' >&2
  exit 2
fi
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
export EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw EXL3_MOE_CPU_SMALL_WORKERS=0
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fixture=/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin
references=/data/models/exl3_exp/threading
for arg in "$@"; do
  case $arg in
    --fixture=*) fixture=${arg#--fixture=} ;;
    --reference-dir=*) references=${arg#--reference-dir=} ;;
  esac
done
if [[ -f $build/compile_commands.json ]]; then
  cp "$build/compile_commands.json" "$results/compile_commands.json"
fi
{
  date -Is
  uname -a
  cat /proc/self/cgroup
  sed -n '/Cpus_allowed_list/p;/Mems_allowed_list/p' /proc/self/status
  printf 'Arguments:'
  printf ' %q' "$@"
  printf '\n'
  sha256sum "$build/exl3_cpu_baseline" "$build/exl3_cpu_optimized"
  sha256sum "$fixture" "$references"/reference-e{1,3,5}.bin
  ldd "$build/exl3_cpu_baseline"
  ldd "$build/exl3_cpu_optimized"
} > "$results/environment.txt"
for ((round=0; round<rounds; ++round)); do
  order=(baseline optimized)
  if (( round % 2 )); then order=(optimized baseline); fi
  for backend in "${order[@]}"; do
    "$build/exl3_cpu_$backend" \
      --benchmark_min_time=512x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$backend.json" \
      --benchmark_out_format=json "$@" \
      > "$results/round-$round-$backend.log" 2>&1
    cat "$results/round-$round-$backend.log"
  done
done
echo "Results: $results"
