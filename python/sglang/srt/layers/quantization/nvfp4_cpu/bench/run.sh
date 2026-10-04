#!/usr/bin/env bash
set -euo pipefail
# Follow expert_stream/bench/run.sh: fixed teams in separate processes,
# fresh records, inherited isolation/memory policy.
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash run.sh BUILD_DIR NEW_RESULTS_DIR [benchmark options...]' >&2
  exit 2
fi
build=$(realpath "$1")
results=$2
shift 2
# The kernel's workers are one OpenMP team per forward: spin between forwards rather than sleep, never let the
# runtime shrink the team, and never let OpenMP bind threads (the kernel pins each worker to its configured core).
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-ACTIVE} GOMP_SPINCOUNT=${GOMP_SPINCOUNT:-INFINITE} OMP_DYNAMIC=FALSE
unset OMP_PROC_BIND OMP_PLACES
rounds=${NVFP4_BENCH_ROUNDS:-8}
counts=${NVFP4_BENCH_WORKERS:-16}
if [[ ! $rounds =~ ^[1-9][0-9]*$ || ! $counts =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo 'NVFP4_BENCH_ROUNDS must be positive; NVFP4_BENCH_WORKERS must be comma-separated positive integers' >&2
  exit 2
fi
IFS=, read -r -a teams <<< "$counts"
declare -A seen=()
for workers in "${teams[@]}"; do
  if [[ -n ${seen[$workers]:-} ]]; then
    echo 'NVFP4_BENCH_WORKERS must not contain duplicate counts' >&2
    exit 2
  fi
  seen[$workers]=1
done
fixture=''
for arg in "$@"; do
  case $arg in
    --fixture=*) fixture=${arg#--fixture=} ;;
    --workers=*|--write-fixture=*|--validate-only|--benchmark_out=*|--benchmark_out_format=*)
      echo 'Use NVFP4_BENCH_WORKERS for worker counts; fixture creation and validate-only run directly on a binary' >&2
      exit 2 ;;
  esac
done
test -x "$build/nvfp4_cpu_optimized"
if [[ -n $fixture ]]; then test -f "$fixture"; fi
mkdir "$results" # Never replace a prior measurement.
results=$(realpath "$results")
if [[ -f $build/compile_commands.json ]]; then
  cp "$build/compile_commands.json" "$results/compile_commands.json"
fi
{
  date -Is
  uname -a
  cat /proc/self/cgroup
  sed -n '/Cpus_allowed_list/p;/Mems_allowed_list/p' /proc/self/status
  printf 'Rounds: %s\nWorker counts: %s\nArguments:' "$rounds" "$counts"
  printf ' %q' "$@"
  printf '\n'
  printf 'OMP_WAIT_POLICY=%s GOMP_SPINCOUNT=%s OMP_DYNAMIC=%s OMP_THREAD_LIMIT=%s\n' \
    "$OMP_WAIT_POLICY" "$GOMP_SPINCOUNT" "$OMP_DYNAMIC" "${OMP_THREAD_LIMIT:-unset}"
  sha256sum "$build/nvfp4_cpu_optimized"
  if [[ -n $fixture ]]; then sha256sum "$fixture"; fi
  ldd "$build/nvfp4_cpu_optimized"
  lscpu
} > "$results/environment.txt"
for workers in "${teams[@]}"; do
  for ((round=0; round<rounds; ++round)); do
    stem="$results/workers-$workers-round-$round-optimized"
    "$build/nvfp4_cpu_optimized" --workers="$workers" \
      --benchmark_min_time=512x --benchmark_repetitions=1 \
      --benchmark_out="$stem.json" --benchmark_out_format=json "$@" \
      > "$stem.log" 2>&1
    cat "$stem.log"
  done
done
echo "Results: $results"
