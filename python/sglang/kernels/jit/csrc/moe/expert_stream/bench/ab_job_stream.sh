#!/usr/bin/env bash
# A/B of two builds of the full-stack bench (exl3_full_stack_prod) on BM_jobstream, the job-stream mode: BASE_BUILD and
# HEAD_BUILD run as separate processes, alternating order each round, for EXL3_BENCH_ROUNDS rounds (default 8), each with
# the same flags. Writes every round's Google JSON and log, and summary.tsv: per case, the median over rounds of the
# record completion time p50 and p95 (us) and of records/s for each build, head / base. Refuses while a server runs (the
# bench pins 28-43 and 22) and holds cc-gpu.lock for the whole run, as a timed result must (EXL3_BENCH_NO_LOCK=1 skips
# it, for a smoke run whose numbers are not results).
#
# The base build is the engine without the job window with this bench's sources taken over it (README.txt, "Job-stream
# bench"); both run the same driver.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash ab_job_stream.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [benchmark options...]" >&2
  exit 2
fi
base=$(realpath "$1")
head=$(realpath "$2")
results=$3
shift 3
if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the bench pins CPUs 22 and 28-43" >&2
  exit 2
fi
if [[ -z ${EXL3_BENCH_NO_LOCK:-} ]]; then
  exec 9> /data/models/slang/nvfp4-work/cc-gpu.lock
  flock 9
fi
mkdir "$results"  # Refuse to overwrite a prior record.
results=$(realpath "$results")
rounds=${EXL3_BENCH_ROUNDS:-8}
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCUNT=INFINITE
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
  date -Is
  printf 'Arguments:'
  printf ' %q' "$@"
  printf '\n'
  sha256sum "$base/exl3_full_stack_prod" "$head/exl3_full_stack_prod"
} > "$results/environment.txt"
for ((round = 0; round < rounds; ++round)); do
  order=(base head)
  if ((round % 2)); then order=(head base); fi
  for side in "${order[@]}"; do
    build=$base
    [[ $side == head ]] && build=$head
    "$build/exl3_full_stack_prod" --benchmark_filter=BM_jobstream --benchmark_min_time=16x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$side.json" --benchmark_out_format=json "$@" \
      > "$results/round-$round-$side.log" 2>&1
  done
done
{
  grep -H '^isa tier ' "$results"/round-*.log || true
} > "$results/isa.txt"
"${PYTHON:-python3}" - "$results" "$rounds" <<'PY'
import json, statistics, sys

results, rounds = sys.argv[1], int(sys.argv[2])
rows = {}
for side in ("base", "head"):
    for r in range(rounds):
        for b in json.load(open(f"{results}/round-{r}-{side}.json"))["benchmarks"]:
            if b.get("error_occurred"):
                sys.exit(f"round {r} {side}: {b['name']}: {b.get('error_message')}")
            name = b["name"].removeprefix("BM_jobstream/").removesuffix("/manual_time")
            entry = rows.setdefault(name, {}).setdefault(side, {"p50_us": [], "p95_us": [], "records_per_s": []})
            for key in entry:
                entry[key].append(b[key])
cols = ("p50_us", "p95_us", "records_per_s")
with open(f"{results}/summary.tsv", "w") as out:
    header = ["case"] + [f"{c}_{s}" for c in cols for s in ("base", "head", "head/base")]
    out.write("\t".join(header) + "\n")
    print("\t".join(header))
    for name, sides in rows.items():
        cells = [name]
        for c in cols:
            b, h = statistics.median(sides["base"][c]), statistics.median(sides["head"][c])
            cells += [f"{b:.1f}", f"{h:.1f}", f"{h / b:.3f}"]
        line = "\t".join(cells)
        out.write(line + "\n")
        print(line)
PY
echo "Results: $results"
