#!/usr/bin/env bash
# The full-stack CPU-expert bench (README.txt, "Full-stack bench"): alternating prod/instr process rounds, a results
# record like run.sh's, and each process's exit status. Under exl3bench.service, RESULTS_DIR defaults to the fresh
# directory exl3bench-run made (EXL3BENCH_RESULTS). No Python driver.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run_full_stack.sh BUILD_DIR [NEW_RESULTS_DIR] [benchmark options...]' >&2
  exit 2
fi
build=$(realpath "$1")
shift
if [[ $# -gt 0 && $1 != --* ]]; then
  results=$1
  shift
  mkdir "$results" # Refuse to overwrite a prior record.
elif [[ -n ${EXL3BENCH_RESULTS:-} ]]; then
  results=$EXL3BENCH_RESULTS
  if [[ ! -d $results || -n $(ls -A "$results") ]]; then
    echo "EXL3BENCH_RESULTS=$results is not an empty directory" >&2
    exit 2
  fi
else
  echo 'No results directory: name one, or run under exl3bench.service (EXL3BENCH_RESULTS)' >&2
  exit 2
fi
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
images=/data/models/exl3_exp/google_benchmark/full-stack-images
for arg in "$@"; do
  case $arg in
    --fixture=*) fixture=${arg#--fixture=} ;;
    --reference-dir=*) references=${arg#--reference-dir=} ;;
    --image-dir=*) images=${arg#--image-dir=} ;;
  esac
done
binaries=("$build/exl3_full_stack_prod" "$build/exl3_full_stack_instr")
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
  sha256sum "${binaries[@]}"
  sha256sum "$fixture" "$references"/reference-e{1,3,5}.bin
  echo "Images: $images"
  ls -l "$images" 2>/dev/null || true
  ldd "${binaries[@]}" || true
} > "$results/environment.txt"
failed=0
for ((round=0; round<rounds; ++round)); do
  order=(prod instr)
  if (( round % 2 )); then order=(instr prod); fi
  for variant in "${order[@]}"; do
    status=0
    "$build/exl3_full_stack_$variant" \
      --benchmark_min_time=512x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$variant.json" \
      --benchmark_out_format=json "$@" \
      > "$results/round-$round-$variant.log" 2>&1 || status=$?
    cat "$results/round-$round-$variant.log"
    echo "round $round $variant exit $status" | tee -a "$results/status.txt"
    if (( status != 0 )); then failed=1; fi
  done
done
echo "Results: $results"
exit $failed
