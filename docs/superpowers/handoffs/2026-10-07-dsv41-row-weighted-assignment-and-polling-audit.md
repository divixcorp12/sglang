# DSV4.1 row-weighted assignment experiment and polling audit

Row weighting improved isolated native CPU forward latency by about 22–23% on
the reconstructed mixed-row shape that previously exposed a 2:1 worker load.
The requested Astra audit separately reproduced two polling defects. Neither
result establishes the cause of the observed 15 ms steady-decode hold.

## Experiment and implementation

Implementation commit: `d68fc51fd9`; ownership-test correction and benchmark
HEAD: `33261d8880d87c362473b9a48575b9c845edc74d`.

`exl3/optimized/tile_assignment.hpp` partitions aligned gate/up and down output
tile ranges by `chunk.m * tiles`, rather than tile count alone. Adjacent workers
share the same rounded boundary, preserving exact ownership and the existing
2-/8-tile alignment requirements. Uniform chunk sizes retain the original
mapping, including whole-GEMV striding for large batches. Arithmetic and route
accumulation order are unchanged.

The compile switch `EXL3_MOE_CPU_ROW_WEIGHTED_ASSIGNMENT` defaults to 0. The bare
native benchmark exposes `-DEXL3_ROW_WEIGHTED_ASSIGNMENT=ON`; no production
configuration, CPU/GPU split, memory split, polling code, or live server was
changed. The full-stack benchmark targets are unaffected by this CMake option.

The runner is `benchmarks/dsv41_baseline/run_row_weighted_experiment.py`.
It ran 16 serial processes: four alternating AB/BA pairs on each NUMA node,
10 pinned workers on CPUs 6–15 / 18–27, with matching `numactl --membind`.
Each case had 64 warm-up forwards and 128 timed forwards per process. All
optional tracing counters were compiled out. The timed run took 141 seconds,
excluding builds and the polling audit.

Eight real 3-bit EXL3 fixture layers (H=5120, I=2304), with 12 separately
allocated expert slots per layer, provide a working set larger than cache.
Inputs are synthetic and expert weights repeat the fixture's five identities
at distinct addresses. The `counts` cases reconstruct chunk multiplicities,
not exact captured activations or routing. Both CPU groups ran separately;
there was no GPU work or simultaneous serving load.

Fixture SHA256:
`55de978d5eb3c9d1bc3ac38b5a7937ea724a47fea2c0cf87c04bd1616e188a8c`.

## Results

Times are medians of four per-process mean forward latencies, in milliseconds.
Each process contributes 128 samples per case. A is the original equal-tile
mapping; B is row weighted. These are complete CPU forwards, not per-expert
latencies or decode token times.

| Routing case | Node 0 A → B | Change | Node 1 A → B | Change |
|---|---:|---:|---:|---:|
| 6 rows, 10 chunks, first chunk has 2 rows | 5.893 → 4.583 | −22.2% | 5.603 → 4.337 | −22.6% |
| Same, doubled chunk in middle | 5.869 → 4.566 | −22.2% | 5.537 → 4.299 | −22.4% |
| Same, doubled chunk last | 5.878 → 4.532 | −22.9% | 5.570 → 4.337 | −22.1% |
| 5 rows, k=3, seeded random routes | 6.276 → 5.999 | −4.4% | 6.016 → 5.754 | −4.4% |
| 6 rows, k=2, all chunk sizes 1 | 5.099 → 5.095 | −0.1% | 4.816 → 4.796 | −0.4% |
| 6 rows, k=2, all chunk sizes 2 | 3.463 → 3.455 | −0.2% | 3.431 → 3.327 | −3.0% |
| 1 row, k=3 | 1.300 → 1.303 | +0.3% | 1.202 → 1.205 | +0.3% |

For the first mixed case, median per-process p95 improved 6.101 → 4.746 ms
on node 0 and 5.849 → 4.509 ms on node 1. All four A/B mean ranges are
well separated. Uniform controls' p50 changes stayed within 1%; their noisier
means/tails should not be interpreted as a row-weighting gain. In particular,
the one-row control's median p95 rose about 1.3–1.5%, and p99 about 1.5–3.1%.
This short run cannot establish equivalence of all tail behavior.

The assignment routine itself gives the following tile-row units for the first
mixed case (static ownership calculation, not a new instrumented phase trace):

| Phase | Equal tiles | Row weighted |
|---|---|---|
| Gate/up | worker 0: 576; others: 288 | workers: 316–318 |
| Down | worker 0: 640; others: 320 | all workers: 352 |

The mixed-case benefit is robust to the heavier chunk's position. Row count
remains a cost approximation: this does not prove perfect runtime balance or
optimal assignment under contention, larger mixed prefill batches, other ISAs,
or live target/draft scheduling.

## Correctness and artifacts

The focused native ownership test passed with GCC 15 on divix01. It checks
exactly-once coverage, bounds/alignment, uniform ownership preservation,
empty and large chunk sets, worker counts up to 64, and the captured load skew
for both split granularities. The original partition permits empty callbacks
when workers exceed groups; the test preserves that existing behavior.

Every benchmark process passed the original 24 frozen output checks. A wrote
seven routed output reference files; every subsequent A/B process matched
them bit-exactly before and after timing, with finite and repeatable outputs.
No frozen references were modified. Python syntax checks and git diff checks
also passed. No broad test suite or live-model throughput benchmark was run.

Remote results:
`/data/models/slang/nvfp4-work/row-weighted-20261007/results-01/`.
The two retained build directories are alongside it. Results include all raw
Google Benchmark JSON/logs, reference bytes, exact commands, compiler flags,
library linkage, binary/fixture hashes, and hardware topology.

Laptop copy:
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/row-weighted-experiment/results-01/`.

## Astra polling audit

Audited commit `1756407149393e7493921ebe452771c7b378c0a8`. The polling headers
are identical at the benchmark HEAD. The agent used bounded CPU-only probes
with fake engines, then stopped them before assignment timing. It made no
repository edits.

1. **Confirmed control-notification race.** `cpu_experts.h::run()` checks stop,
   draft attachment and detach before sampling the doorbell. A control update
   can publish and ring between those checks and the snapshot, leaving the hold
   with stale control state and the new notification value. Forced interleavings
   delayed shared detach by 100.288 ms with a 100 ms hold, versus 1.145 ms in
   the ordinary case. Draft-only stop similarly delayed about 100 ms, excluding
   its normal watchdog join. Both infinite-hold cases stayed blocked at 1.5 s
   and were killed by the probe. Attach has the same source-level race but was
   not runtime-reproduced. Fix direction: sample the notification before all
   associated state checks, including stop.
2. **Confirmed avoidable draft sleep.** The 50 µs sleep predicate checks jobs
   and stop, but omits draft readiness. Using the actual Doorbell header and
   this predicate, a head ready before the wait still slept 83–121 µs in all
   12 trials. A comparison predicate checking head returned in 67–113 ns.
   This is a native readiness probe, not an end-to-end GPU timing result.
3. **Unmeasured queue-publication latency window.** Target queue publication
   precedes its doorbell ring. Producer preemption in between can leave work
   queued while a held team sees unchanged notification words. No occurrence
   was established in the observed 15 ms interval.

> 🧠 **From Hindsight memory (Conventions and patterns)** — the shared-team
> design intentionally prioritizes waiting target jobs and runs target/draft
> work sequentially.

The current run loop matches that policy, with no fairness budget; a target
backlog can delay draft work. It should be evaluated separately from the two
confirmed notification/readiness defects. Hold and forward also enter separate
OpenMP parallel regions; cached worker pinning avoids repeated affinity calls.
A fast final join does not bound first-notification-to-all-workers-ready time.

The full audit, sources, results, setup-failure notes, and replay commands are
on the laptop at:
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/row-weighted-experiment/polling-audit.md`
and `polling-probes/`. Remote originals are under
`/data/models/slang/nvfp4-work/astra-poll-audit-20261007/`.

## Next decision

The isolated test supports carrying row weighting into a serving experiment.
Keep its production default off until live workload validation. Fix the two
confirmed polling races with deterministic regression tests as a separate
change. To attribute the 15 ms hold, measure actual GPU publication through
CPU head observation and all-worker release; the present probes and earlier
fast joins do not supply that causal interval.
