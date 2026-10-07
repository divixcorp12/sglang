#!/usr/bin/env bash
# A/B of two builds of the bare-forward bench (exl3_cpu_optimized): BASE_BUILD and HEAD_BUILD run as separate processes,
# alternating order each round, for EXL3_BENCH_ROUNDS rounds (default 8), each with the same flags. Writes every round's
# Google JSON and log, and summary.tsv: per benchmark, the median over rounds of p50_us for each build, head / base, and
# the plan each build ran (from the dsv41_calls / generic_calls counters). Refuses while a server runs: it pins 18-33.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash ab.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [benchmark options...]" >&2
  exit 2
fi
base=$(realpath "$1")
head=$(realpath "$2")
results=$3
shift 3
if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the bench pins CPUs 18-33" >&2
  exit 2
fi
mkdir "$results"  # Refuse to overwrite a prior record.
results=$(realpath "$results")
rounds=${EXL3_BENCH_ROUNDS:-8}
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
export EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw EXL3_MOE_CPU_SMALL_WORKERS=0
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
  date -Is
  printf 'Arguments:'
  printf ' %q' "$@"
  printf '\n'
  sha256sum "$base/exl3_cpu_optimized" "$head/exl3_cpu_optimized"
} > "$results/environment.txt"
for ((round = 0; round < rounds; ++round)); do
  order=(base head)
  if ((round % 2)); then order=(head base); fi
  for side in "${order[@]}"; do
    build=$base
    [[ $side == head ]] && build=$head
    "$build/exl3_cpu_optimized" --benchmark_min_time=256x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$side.json" --benchmark_out_format=json "$@" \
      > "$results/round-$round-$side.log" 2>&1
  done
done
"${PYTHON:-python3}" - "$results" "$rounds" <<'PY'
import json, statistics, sys

results, rounds = sys.argv[1], int(sys.argv[2])
rows = {}
for side in ("base", "head"):
    for r in range(rounds):
        for b in json.load(open(f"{results}/round-{r}-{side}.json"))["benchmarks"]:
            name = b["name"].split("/", 1)[1].removesuffix("/manual_time")
            entry = rows.setdefault(name, {}).setdefault(side, {"p50": [], "plan": set()})
            entry["p50"].append(b["p50_us"])
            entry["plan"].add("dsv41" if b["generic_calls"] == 0 else "generic" if b["dsv41_calls"] == 0 else "mixed")
with open(f"{results}/summary.tsv", "w") as out:
    out.write("benchmark\tbase_p50_us\thead_p50_us\thead/base\tbase_plan\thead_plan\n")
    for name, sides in rows.items():
        b, h = statistics.median(sides["base"]["p50"]), statistics.median(sides["head"]["p50"])
        line = f"{name}\t{b:.1f}\t{h:.1f}\t{h / b:.3f}\t{'/'.join(sorted(sides['base']['plan']))}\t{'/'.join(sorted(sides['head']['plan']))}"
        out.write(line + "\n")
        print(line)
PY
echo "Results: $results"
