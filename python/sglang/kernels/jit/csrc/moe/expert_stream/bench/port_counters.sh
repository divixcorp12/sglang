#!/usr/bin/env bash
# Per-forward hardware counters of the bare-forward bench (exl3_cpu_optimized): for each (bench, workers, event group),
# two runs at N1 and N2 iterations; port_counters.py divides their difference by N2 - N1, the steady loop alone (setup
# and validation cancel). Event groups hold at most four programmable counters, so nothing multiplexes. User-space
# counts only (perf_event_paranoid 2). Refuses while a server runs: the bench pins 28-43.
#
#   bash port_counters.sh BUILD_DIR NEW_RESULTS_DIR
#   python3 port_counters.py NEW_RESULTS_DIR
set -uo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash port_counters.sh BUILD_DIR NEW_RESULTS_DIR" >&2
  exit 2
fi
bin=$(realpath "$1")/exl3_cpu_optimized
out=$2
if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the bench pins CPUs 28-43" >&2
  exit 2
fi
mkdir "$out"  # Refuse to overwrite a prior record.
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
fixed=cycles:u,instructions:u,ref-cycles:u
groups=(
  uops_dispatched_port.port_0:u,uops_dispatched_port.port_1:u,uops_dispatched_port.port_5:u,uops_dispatched_port.port_6:u
  core_power.lvl0_turbo_license:u,core_power.lvl1_turbo_license:u,core_power.lvl2_turbo_license:u,cycle_activity.stalls_mem_any:u
  uops_issued.any:u,cycle_activity.stalls_total:u,uops_dispatched_port.port_2:u,uops_dispatched_port.port_3:u
)
n1=${N1:-256}
n2=${N2:-4352}
echo "$n1 $n2" > "$out/iterations.txt"
sha256sum "$bin" > "$out/environment.txt"
for workers in 1 16; do
  for bench in experts:1 experts:5 routed; do
    if [[ $bench == routed ]]; then
      args=(--routed=6:6:random "--benchmark_filter=rows:6/k:6/random")
      tag=r6k6
    else
      args=("--benchmark_filter=$bench/")
      tag=${bench/:/}
    fi
    for g in 0 1 2; do
      for n in "$n1" "$n2"; do
        perf stat -x, -e "$fixed,${groups[$g]}" -o "$out/$tag-w$workers-G$((g + 1))-$n.csv" \
          "$bin" --workers="$workers" --cpus=28-43 --benchmark_min_time="${n}x" --benchmark_repetitions=1 "${args[@]}" \
          > "$out/$tag-w$workers-G$((g + 1))-$n.log" 2>&1 || echo "FAILED $tag w$workers G$((g + 1)) $n" >&2
      done
    done
  done
done
