# DeepSeek-V4.1-Flash: RAM-miss leases on vs off, end to end

Serving-level A/B of `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` through a real `sglang.Engine`: tokens/s, and
min/p50/p95/p99 of per-token decode latency. The closest
existing thing is `analysis/dsv41-drive/open11/open11_serving_path.py`, which drives the real backend and the
real switch but no Engine; this drives the whole server path.

**Nothing here has been run against the model.** A real run loads the model, holds the whole card and ~80 GiB
of host RAM for many minutes per process, and needs explicit approval. What was verified without loading it is in the hand-off
report: `--dry-run`, the import-path assertion, the paths, and `test_harness.py`.

## The GRAPH_GATHER trap

The phase3a `env.sh` sets `SGLANG_MOE_EXPERT_GRAPH_GATHER=0` and decodes eagerly. Lease mode lives in the
in-graph RAM-miss path (`Exl3RamMissRowBackend.post` runs it only from a captured decode graph). Launch with
that recipe and both arms take the eager path: they report the same number, and the delta is zero because the
switch reached nothing. This harness uses `GRAPH_GATHER=1` (the phase3b `env-full.sh` override) with a
breakable decode graph, and `arm_config.check_recipe` refuses anything else.

Do not read "both arms gave the same number" as a finding. Read `lease_check` in each result instead.

## Files

| file | job |
|---|---|
| `arm_config.py` | the two arms' environment and Engine kwargs; import-path assertion; recipe check; runs the real EXL3 gate on a launch without loading anything |
| `bench_arm.py` | one arm in one process: Engine, warmup, load, counters, `.json.gz`; `--dry-run` prints both resolved launches |
| `workload.py`, `metrics.py` | prompts from the corpus, closed-loop driver over `Engine.async_generate`, percentiles |
| `counters.py` | reads the service counters out of the stream trace and decides whether the lease path ran |
| `run_ab.py` | both arms, N reps, separate processes, each under `gpu-run.sh`; writes `comparison.md` |
| `compare.py` | the comparison table over a directory of results |
| `test_harness.py` | CPU tests, no Engine (`PYTHONPATH=$PWD/python python -m pytest benchmarks/dsv41_flash/test_harness.py`) |

## Cost model: two loads, nothing more

The model load (~70 GiB pinned host cache) dominates; decode tokens are cheap (~67 ms each). The switch is read once at
service start, so two process launches is the floor, and it is the default: one `lease_off`, one `lease_on`, one
repetition. Nothing is pre-planned beyond that. `--reps` (default 1) counts pairs.

## Launch

On divix01, in `wt-dsv41` after `git pull --ff-only shared dsv41`:

```bash
cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41
# rehearsal: prints every command and both resolved launches, asserts the EXL3 gate; no lock, no GPU
taskset -c 0-63 env OMP_NUM_THREADS=16 PYTHONPATH=$PWD/python \
  /data/models/slang/.venv/bin/python benchmarks/dsv41_flash/run_ab.py --dry-run
# the real thing (needs approval); each arm runs under gpu-run.sh, which takes cc-gpu.lock
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python benchmarks/dsv41_flash/run_ab.py \
  --out-dir /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-flash-bench/$(date -u +%Y%m%d)
```

`run_ab.py` passes unknown flags to `bench_arm.py` (`--requests`, `--output-tokens`, ...). `PYTHONPATH` must name the
tree under test: `/data/models/slang/.venv/bin/python` otherwise imports sglang from `main-port-probe-7bc4eb`, so
`bench_arm.py` asserts `sglang.__file__` resolves under this repo and exits 2 (`REFUSED: ImportPathError`) if not.
The per-arm command is `gpu-run.sh <python> bench_arm.py --arm {lease_off,lease_on} --rep N --position P --out ...`;
the recipe environment is set inside `bench_arm.py`, so it cannot drift between arms.

Each `.json.gz` holds the arm, the env that differs between arms (only the lease switch), the whole recipe env, the
workload, all service counters, `rows_read`, the measured-window mix, and every request's raw record including every
per-token step latency, so any percentile can be recomputed. `--position` records whether the process was first or
second of its pair.

## What is measured

- **Statistic: per-token decode latency**, the time between consecutive streamed chunks. Per-request percentiles are
  not computed (4 requests); e2e and TTFT appear as means only.
- **Warmup discarded, and recorded**: one warmup request (`--warmup-requests 1`, 64 tokens) before the window, then
  the first `--discard-steps` (30) steps of every measured request are dropped. The count is in
  `summary.discarded_steps_per_request`.
- **Default window: 4 requests x 512 tokens**, four different corpus prompts of 256 tokens, greedy, `ignore_eos`:
  4 x (512 - 1 - 30) = 1924 per-token samples per arm, ~2 min at 67 ms. Both arms see the same prompts in the same
  order. Decode batch size is pinned at 1 by the EXL3 gate, so requests cannot overlap and each one's TTFT (prefill,
  possibly tens of seconds of cold RAM misses) is pure overhead that adds no decode samples. Hence few requests with
  many tokens each; 4 rather than 2 so the window spans four routes, not one pathological one. Nothing about the
  count is sacred: `--requests 2 --output-tokens 1024` gives 2000 samples for two TTFTs, `--requests 8
  --output-tokens 256` for eight.
- **Projected wall clock**: right after warmup the process prints `[projection] ...` from the warmup request's own TTFT
  and decode rate: the measured window and this process's total. The project's figures disagree (~67 ms/token in-graph
  vs 300-450 ms in the Sept 19 corpus runs), so read the projection in the first minutes and abort (Ctrl-C) if it is far
  off; `--max-projected-s N` makes the process abort itself if the projected window exceeds N seconds. The warmup is
  the coldest request, so the projection leans slow.
- Reported: min, p50, p95, p99 and mean of step latency; tokens/s over the whole window (prefill included).
- **Within-arm spread**: the samples are cut into `--blocks` (5) consecutive slices; `p50s`/`mins` of each slice and
  their relative range are recorded. Consecutive, so a drift within the window shows up as spread.
- **Concurrency stays 1**: the EXL3 gate allows decode graphs at batch size 1 only, so a batch of two runs eagerly and
  skips the lease path. `--concurrency 2` is refused unless `--allow-eager-batches`.

## The effect to resolve, and why ~1900 tokens

The lease cost is ~0.68 ms per 40-layer step on ~66.8 ms: about 1%. The standard error of a p50 over n samples is
about 1.25 x sd / sqrt(n), so 1924 samples resolve 0.67 ms if the per-step sd is below roughly 8 ms; a run whose
steps stall on RAM misses (sd of tens of ms) will not, and the block spread will say so. The main threat is
cross-process variance (two processes, two loads), which is why `min` is reported beside p50: on the OPEN 11 re-take
min-to-min and p50 agreed to 0.4 us on a quiet card and disagreed six-fold on a busy one. A p50 delta whose min-to-min
delta disagrees in sign is box noise.

## Decision rule: do not reflexively run three of everything

Run the pair once. Then read the last lines of `comparison.md` (also `compare.py RUN_DIR`):

- **Done** when `|p50 delta| >= 2 x within-arm spread` and the min-to-min delta has the same sign: the table prints
  `decision: RESOLVED`. Two loads were enough.
- **Otherwise** run one more pair with the order flipped, into the same directory:
  `run_ab.py --rep-start 1 --out-dir <same dir>` (odd reps put `lease_on` first, so the order effect cancels), or a
  longer window (`--output-tokens 1024 --requests 4`; the context length allows 256 + 3800) if the block spread itself is large. Then decide again.
- **Invalid** whatever the deltas, if the lease check failed (below). An identical number from both arms is the
  symptom of a lease path that never ran, not a result.

The 2x factor and the 5 blocks are judgment, not a statistical test; they are constants at the top of `compare.py`.

## Hit/miss mix

OPEN 11's arming cost falls on all-hit layers; the lease machinery exists for misses. So the window's mix is recorded:
deltas of `served`, `touch_only`, `rows_read`, `evictions` over the measured window (`window.counters_delta` has every
counter), `mix.layer_miss_fraction` (layer-steps with a demand RAM-miss row, over all layer-steps), and a label:
`all-hit` (< 1%), `all-miss` (> 90%), else `mixed`. The cut-offs are arbitrary; the fraction is what to read. An
all-hit or all-miss window measures a different regime from steady-state serving, and `compare.py` says so. If the two
arms' fractions differ by more than 5 points it warns, because then they did different work.

## How the lease path is proven to have run

The service lives in the scheduler subprocess, so the driver cannot call `service.host.counters()` as
`open11_serving_path.py` does. The channel that exists is the stream trace: `bench_arm.py` sets
`SGLANG_DSV41_EXPERT_TRACE_PATH`, each decode graph step appends the service's cumulative counters, and
`counters.verify_lease` reads them (line-buffered, so the scheduler's SIGKILL at shutdown loses nothing).

- both arms: at least one decode graph step reached the trace, the service served requests, `late_after_fatal == 0`,
  `read_errors == 0`, and the run finished (the scheduler fail-stops on a fatal, so a completed run has none)
- `lease_on`: `leases_granted > 0`, and `leases_granted - leases_acked` returns to 0 in one of the last 8 snapshots
- `lease_off`: `leases_granted == leases_acked == 0` (proves the switch reached the service)

A failed check sets `lease_check.ok = false`, exits 3 and puts `INVALID` at the top of the table. The final gap is also
recorded: the trace snapshots per batch, not per step boundary, so a nonzero final gap is expected to be in flight.
`fatal_seq` itself is a request-page word not present in the snapshot; this is where the brief's `fatal == 0` is
weaker than in `open11`.

**The trace is on in both arms**, so the service also takes stage timestamps and the scheduler writes a JSON line per
decode step. That is symmetric between the arms so it cancels in the delta, but its cost per step is unmeasured, and it adds
to the step time the 1% is taken against. `--no-trace` removes it and
the verification with it.

## Fixed choices, and where they come from

Model `dsv41-full40`, experts `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw`, `SGLANG_MOE_HOT_GPU_MB=14336`,
`--mem-fraction-static 0.80`, no mirror dirs (`--mirror-dirs` adds `SGLANG_MOE_EXPERT_MIRROR_DIRS`): phase3a `env.sh`
and phase3b `prof-graph.sh`. `SGLANG_MOE_HOT_GPU_MB=16384` failed on this card
(`analysis/dsv41-phase3a/smoke-attempt1-hot16384.log`). Engine kwargs are `scripts/dsv41/trace_corpus.engine_kwargs`
with `--graphs`, the launch the EXL3 gate was written for.
