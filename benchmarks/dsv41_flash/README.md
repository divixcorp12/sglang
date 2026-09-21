# DeepSeek-V4.1-Flash: RAM-miss leases on vs off, end to end

Serving-level A/B of `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES` through a real `sglang.Engine`: tokens/s, and
p50/p95/p99 of per-token decode latency, time to first token and end-to-end request latency. The closest
existing thing is `analysis/dsv41-drive/open11/open11_serving_path.py`, which drives the real backend and the
real switch but no Engine; this drives the whole server path.

**Nothing here has been run against the model.** A real run loads the model, holds the whole card and ~80 GiB
of host RAM for hours, and needs explicit approval. What was verified without loading it is in the hand-off
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

## Launch

On divix01, in `wt-dsv41` after `git pull --ff-only shared dsv41`:

```bash
cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41
# rehearsal: prints every command and both resolved launches, asserts the EXL3 gate; no lock, no GPU
taskset -c 0-63 env OMP_NUM_THREADS=16 PYTHONPATH=$PWD/python \
  /data/models/slang/.venv/bin/python benchmarks/dsv41_flash/run_ab.py --dry-run
# the real thing (needs approval); every arm runs under gpu-run.sh, which takes cc-gpu.lock
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python benchmarks/dsv41_flash/run_ab.py \
  --reps 3 --out-dir /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-flash-bench/$(date -u +%Y%m%d)
```

`run_ab.py` passes unknown flags to `bench_arm.py` (`--requests`, `--output-tokens`, ...). `PYTHONPATH` must name the
tree under test: `/data/models/slang/.venv/bin/python` otherwise imports sglang from `main-port-probe-7bc4eb`, so
`bench_arm.py` asserts `sglang.__file__` resolves under this repo and exits 2 (`REFUSED: ImportPathError`) if not.
The per-arm command is `gpu-run.sh <python> bench_arm.py --arm {lease_off,lease_on} --rep N --out ...`; the recipe
environment is set inside `bench_arm.py`, so it cannot drift between arms.

Arm order alternates (off,on then on,off) across reps so host drift does not favour one arm. Every `.json.gz` holds
the arm, the env that differs between arms (only the lease switch), the whole recipe env, the workload, the counters,
`rows_read`, and every request's raw record including per-token step latencies, so any percentile can be recomputed.

## What is measured

- **Load**: closed loop, `--concurrency` requests in flight (default 1), `--warmup-requests` (default 4, not measured)
  then `--requests` (default 16). Prompts are first turns of `sessions.jsonl` cut to exactly `--input-tokens` (256),
  greedy, `ignore_eos`, `--output-tokens` (128). Both arms see the same prompts in the same order.
- **Concurrency stays 1.** The EXL3 gate allows decode graphs at batch size 1 only, so a batch of two runs eagerly and
  skips the lease path. `--concurrency 2` is refused unless `--allow-eager-batches`.
- **tokens/s**: completion tokens over the wall time of the window (first submit to last finish), so prefill counts.
  `decode_tok_s_mean` is the per-request rate after the first token.
- **Decode step latency**: time between consecutive streamed chunks, per token, pooled over all measured requests
  (`provenance.step_latency`). **e2e**: submit to last chunk. **TTFT**: submit to first chunk. Percentiles are
  nearest-rank; with 16 requests the request-level p99 is the maximum, so trust the per-token p99 (~2000 samples).

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
decode step. That is symmetric, small next to a step of hundreds of ms, and unmeasured. `--no-trace` removes it and
the verification with it.

## Effect size

`open11_serving_path.py` measured about 8 us per all-hit layer for lease mode, ~0.32 ms per 40-layer step. Streamed
decode in the Sept 19 corpus runs took 300-450 ms per token, so an all-hit overhead is ~0.1% of a step and will not
show in tokens/s. Anything larger has to come from misses. `compare.py` prints each metric's rep-to-rep spread next
to the delta and says whether the delta is inside it; believe the delta only where it is not.

## Fixed choices, and where they come from

Model `dsv41-full40`, experts `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw`, `SGLANG_MOE_HOT_GPU_MB=14336`,
`--mem-fraction-static 0.80`, no mirror dirs (`--mirror-dirs` adds `SGLANG_MOE_EXPERT_MIRROR_DIRS`): phase3a `env.sh`
and phase3b `prof-graph.sh`. `SGLANG_MOE_HOT_GPU_MB=16384` failed on this card
(`analysis/dsv41-phase3a/smoke-attempt1-hot16384.log`). Engine kwargs are `scripts/dsv41/trace_corpus.engine_kwargs`
with `--graphs`, the launch the EXL3 gate was written for.
