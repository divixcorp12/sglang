# Test row-weighted CPU expert assignment in DSpark serving

Status: proposed execution plan, 2026-10-07. The standalone experiment is
complete; the work below validates its integration and benefit with the current
persistent CPU team and real serving workload. Implementation and measurements
remain to be run.

## Objective and hypothesis

Determine whether balancing EXL3 matrix output tiles by chunk token-row count
reduces complete CPU forwards, complete layer latency, and steady decode time
without changing numerical outputs or CPU/GPU work placement.

The hypothesis is specific: mixed chunk row counts make equal-tile assignment
uneven. Row weighting should shorten the busiest worker's gate/up and down
computation, allowing dependent phases to start sooner. It does not address
request detection, GPU publication, target priority, or every scheduler stall.

The first serving measurement is B only, respecting the owner's earlier
request to use an existing A. Reuse A only after checking its compatibility.
Do not silently treat an old OpenMP run or a different calibrated split as A.

## Evidence already available

> 🧠 **From Hindsight memory (Row-weighted CPU expert assignment experiment)** —
> the opt-in experiment changes aligned tile ownership, preserves arithmetic
> and accumulation order, and is separate from CPU/GPU split optimization.
> Its isolated native benefit has not established live decode throughput.

Verified against source, the committed handoff, and saved result files:

- Measured implementation HEAD:
  `33261d8880d87c362473b9a48575b9c845edc74d`.
- Six rows, ten chunks, eleven routes, with one two-row chunk: native complete
  forward 5.8926 -> 4.5833 ms on node 0 and 5.6032 -> 4.3371 ms on node 1.
  Heavy-first/middle/last cases all improved about 22–23%.
- Random routing improved about 4.4%; uniform controls' p50 changes were <1%.
  Single-row p99 rose about 1.5–3.1% in the short run, so tail equivalence is
  not established.
- Sixteen processes passed frozen and cross-build bit-exact output checks.
  The experiment ran nodes separately, with synthetic activations and a
  rotating real-weight fixture; there was no GPU or simultaneous serving.
- A captured mixed forward gave one worker 576 gate/up tile-row units versus
  288 for peers, and 640 down units versus 320. The heavy worker's wall time
  was almost entirely computation. Weighted static assignment gives 316–318
  gate/up and 352 down units per worker for that shape.

Laptop artifacts:
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/row-weighted-experiment/results-01`.
Remote artifacts:
`/data/models/slang/nvfp4-work/row-weighted-20261007/results-01`.
Full investigation:
`docs/superpowers/handoffs/2026-10-07-dsv41-cpu-stalls-complete-session-handoff.md`.

### Interpret the repeated-expert counters correctly

Owner-supplied snapshot, printed twice identically; use it once:

```text
jobs=3894, rows=19470, forward_ns=9252299977
collided_jobs=2304, shared_routes=5798, collided_forward_ns=7503968516
```

`count_shared_routes` sorts live CPU slots within a single draft job and counts
occurrences beyond each slot's first. `collided_jobs` increments if that count
is positive. This measures expert reuse within a draft job, not target/draft
execution overlap or queue contention.

| Derived observation | Value |
|---|---:|
| Repeated-slot jobs / all draft jobs | 59.17% |
| Forward time in repeated-slot jobs / all draft forward time | 81.10% |
| Mean repeated-slot job forward | 3.2569 ms |
| Mean other job forward | 1.0996 ms |
| Mean rows/job | 5 |
| Mean repeats per repeated-slot job | 2.5165 |

`collided_forward_ns` includes the entire forward, not additional penalty.
Route count, expert identity, stage and chunk shape may explain the different
means. Reuse may improve weight locality. Some repeated-slot jobs have uniform
chunk sizes and therefore get no different assignment. Group results by actual
post-chunking row counts, not just this binary counter.

## Source baseline and experiment boundaries

At plan-writing time, the persistent-team checkout was:
`/Users/dnikolaidis/.codex/worktrees/persistent-team/sglang-nvfp4`, HEAD
`12c13f64ee662376fe13ecf042213498bb5b7519`.

It contains a persistent pthread `Team`, the default-off weighted assignment,
and a newer calibration change that routes approximately 1.25 tokens per lane.
The historical handoff describes earlier calibration and OpenMP states. Do not
apply those historical descriptions to the current source.

Before implementation, choose and record one immutable source commit containing
the intended team and calibration. If development has advanced, record the new
commit and inspect the difference; do not move the baseline during measurement.
Apply weighting integration on top of that commit in an isolated experiment
checkout. Preserve other sessions' work and their servers.

| Control | Requirement |
|---|---|
| Threading | Same persistent team, barriers, idle work and priority in A/B |
| CPU workers | Same ten per group and same core lists, expected 6–15 /18–27 |
| NUMA expert allocation | Preserve established 40 GiB /40 GiB; verify actual launch |
| CPU/GPU split | Explicit identical table on every relevant row/group; no retuning |
| Quantization/kernel | Same weights, ISA, compiler, activation/residual flags, MAX_M and CHUNK_M |
| GPU configuration | Same hot cache, memory fraction, graph mode and resident map |
| Draft configuration | Same model, block size, CPU resident experts and routing policy |
| Requests | Same prompt bytes/tokenization, order, output lengths, seed and sampling |
| Measurement | Same warmup admission and decode timing formula |
| Diagnostics | Same build mode for a diagnostic comparison; optional collectors off for throughput |

Historical expected launch values are 10840 MiB hot cache, memory fraction .78,
DSpark block 5 and eight victim lanes. Obtain the actual baseline manifest
rather than silently substituting these if the latest run differs.

Freeze the explicit split table from the intended A. If A calibrated dynamically,
recover its effective table from logs and inject that table into B. If it cannot
be recovered, B can provide a smoke result but cannot isolate weighting's effect.
Automatic calibration/retuning in B would confound assignment with placement,
especially given the recent 1.25-routes/lane calibration change.

Do not change CPU/GPU share, memory split, team implementation, CHUNK_M, phase
fusion, middle/input preparation assignment, quantization, or scheduler policy
in this experiment. Row weighting applies to EXL3 gate/up and down output tiles
for both target and draft calls using that kernel. Separate target-only or
draft-only activation is a follow-up experiment if attribution needs it.

## Phase 0: validate the existing A and record provenance

- [ ] Locate A's exact result directory, launch manifest and request outputs.
- [ ] Record A's source commit, kernel/host library paths and hashes, compiler,
  effective split tables, team configuration, environment and warmup verdict.
- [ ] Confirm A uses the intended persistent-team source and matching kernel
  arithmetic; a commit difference is acceptable only if audited as irrelevant
  to measured behavior or solely the weighting option.
- [ ] Confirm raw session timings, output counts and acceptance data exist;
  a cumulative printed rate alone is insufficient for a matched comparison.
- [ ] Identify whether A is an instrumented capture or counters-off serving.
  Never compare those rates as an isolated assignment effect.
- [ ] Freeze a run manifest with explicit per-field values, not inherited shell
  defaults. Save model/fixture identities, hardware topology and visible load.

If compatible A exists, do not rerun it automatically. If it does not, continue
with B correctness/smoke and mark comparative conclusions unavailable. A fresh
matched control is then the minimum additional experiment needed for a causal
performance claim, rather than an automatic full A/B campaign.

## Phase 1: wire the option into the actual serving kernel

Important integration detail: the CPU kernel function pointer is provided by
the **EXL3 Torch extension**. Defining a macro only on the expert-stream host
JIT module does not necessarily change the matrix kernel it calls.

Relevant sources, relative to the pinned implementation checkout:

| File | Planned work |
|---|---|
| `python/sglang/kernels/jit/csrc/exl3/optimized/tile_assignment.hpp` | Existing original/weighted partition; retain algorithm and default 0 |
| `python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp` | Existing weighted gate/up/down call sites; inspect coverage, no arithmetic change |
| `python/sglang/srt/layers/quantization/exl3/ext.py` | Add experimental CPU define and distinct extension flavor/cache directory/name |
| `python/sglang/srt/environ.py` | Declare option if following existing environment configuration conventions |
| `python/sglang/srt/layers/quantization/exl3/schemes/exl3_cpu_experts.py` | Verify kernel-address provider is the loaded weighted extension |
| `benchmarks/dsv41_baseline/arm_env.py` and selected run driver | Pass through option explicitly, log effective state; avoid stale captured environments |

Proposed new option: `SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT=0|1`.
**This is proposed; it is not an existing supported serving flag at plan-writing
time.** Its implementation must validate values, default to 0 and produce
`-DEXL3_MOE_CPU_ROW_WEIGHTED_ASSIGNMENT=1` in the translation unit compiling the
optimized CPU forward. Reject incompatible/unoptimized use rather than logging
enabled while silently executing an unaffected kernel.

- [ ] Include the setting in extension flavor, module identity and build path,
  as well as compiler flags. Existing `exl3_ext()` uses process-local caching;
  use a fresh process for each variant rather than toggling after import.
- [ ] Keep weighted and original libraries in distinct directories. Verify
  trace-on/off and weighting-on/off combinations also remain distinct.
- [ ] Provide read-only build metadata or an equivalent authoritative check
  exposing effective assignment, CHUNK_M and kernel identity.
- [ ] Log the loaded EXL3 extension path/hash and resolved kernel provider.
  Inspect the build commands; checking the environment variable alone fails
  the integration gate.
- [ ] Preserve the unchanged default path. No host queue/polling changes.

The bare benchmark's `-DEXL3_ROW_WEIGHTED_ASSIGNMENT=ON` is not a serving
configuration and does not propagate automatically to this extension.

## Phase 2: focused correctness and build-selection checks

Use existing coverage rather than repeating a broad test suite.

- [ ] Run `test/registered/unit/kernels/test_exl3_row_weighted_assignment.py`
  at the pinned implementation source: exact ownership, bounds, alignment,
  empty work, uniform fallback, mixed sizes and worker counts.
- [ ] Add focused build-selection checks for the new option: default/off,
  on, invalid values, independent trace combination and distinct cache identity.
  Check final compile definitions and flavor construction, not just env parsing.
- [ ] Warm exact modules before bounded subprocess checks; compare original
  and weighted kernels in separate fresh processes using identical saved input.
- [ ] Compare frozen references and actual extension outputs bit-for-bit for
  single-row, uniform-two-row, mixed-one/two-row, random and chunk-boundary
  cases, including counts above CHUNK_M. Preserve existing references.
- [ ] Include actual five-row draft shapes and six-row verify shapes, rather
  than assuming draft and target have the same row count.
- [ ] Check finite/repeatable outputs and supported split alignments. Preserve
  valid empty callbacks when workers exceed assigned groups.
- [ ] Smoke the real target/draft kernel registration and completion paths,
  bounded and with clean shutdown. No arithmetic or ABI mismatch is acceptable.

Stop on any numerical mismatch, ownership violation, wrong loaded library,
unexpected fallback or completion failure. Diagnose before timing; do not
regenerate golden outputs to accommodate a mismatch.

Existing focused test invocation on divix01, from the private checkout:

```bash
taskset -c 0-63 env OMP_NUM_THREADS=4 \
  CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ PYTHONPATH="$PWD/python" \
  /data/models/slang/.venv/bin/python -m pytest -q \
  test/registered/unit/kernels/test_exl3_row_weighted_assignment.py
```

This is a recipe, not a test run performed while writing the plan. Select any
new integration tests narrowly and record their exact invocation and exit code.

## Phase 3: short native sanity check on the new source

The original 141-second native experiment already answers whether weighting
can help the old reconstructed shape. Do not repeat its full matrix by default.

- [ ] Build a counters-off B at the pinned persistent-team source.
- [ ] Verify original frozen/cross-build references remain applicable to its
  arithmetic; record their provenance and any unrelated source changes.
- [ ] Check heavy-first/middle/last, one actual five-row draft shape, uniform
  row counts and single-row controls on each node.
- [ ] Use ten workers and matching NUMA binding, rotating weights, warmup and
  enough repetitions to distinguish a failure from one slow sample.
- [ ] Run B only against a compatible existing native A. The old benchmark
  executable remains historical evidence if its team/build differs.

Do not link an old EXL3 library against a new team ABI merely to reuse A.
The serving extension parity gate matters more than re-running a bare binary.
If the short check has no compatible native A, report B time and correctness;
use the diagnostic phase below to inspect balance without claiming a speedup.

## Phase 4: one bounded steady-decode diagnostic B

Purpose: check whether the mechanism survives real routed work, with both CPU
groups and GPU/copy activity present. This run is for attribution, not the
primary throughput result. Start with one request, capture only after formal
warmup and steady-decode admission, and bound capture to roughly 10–20 seconds.

Use existing compile-time job/worker collectors where available. Add only
missing fields, bounded and compiled out of normal builds; do not enable
Nsight or magic-trace for the first assignment experiment.

For each retained job, record/join:

| Field | Reason |
|---|---|
| Source, group, stage/layer, epoch/sequence | Separate draft/target, nodes and dependent jobs |
| Rows, live CPU routes, distinct CPU experts | Describe actual computation |
| Ordered chunk row-count vector and CHUNK_M | Reconstruct assignment and locality |
| Per-worker gate/up/down tile-row units | Confirm assignment is changed as intended |
| Phase begin/end and worker arrivals | Measure critical worker time and waiting |
| Complete CPU forward and queue timestamps | Distinguish compute from occupied-engine delay |
| Worker thread CPU time, where supported | Separate math cost from off-CPU tails |
| CPU and DMA completion, correlated layer identity | Determine whether CPU savings shorten layer completion |
| Drop counts and trace gate/bracket | Establish which joins are valid |

Store exact ordered shapes for replay. A sorted chunk-size histogram is useful
for summaries but insufficient to reproduce tile boundaries and weight locality.
If storage is constrained, intern ordered signatures and use bounded IDs.

Required analysis:

1. Partition by draft/target, NUMA group, stage/layer, rows, live routes and
   ordered chunk shape; optionally label uniform versus mixed chunk sizes.
2. Compute per phase `max(worker units) / mean(worker units)`, maximum useful
   compute duration, phase wall time, and complete forward time.
3. Separate early workers' wait for useful work from the tail after all useful
   work completed. Faster final barrier alone is not the assignment mechanism.
4. Compare to matching diagnostic A only if one exists. Do not compare
   instrumented B with counters-off A. Without matching A, compare observed
   weighted units to original mapping reconstructed offline, and label that
   comparison static rather than a measured phase speedup.
5. Report common shapes and their frequency-weighted contribution. Also
   standardize A/B forward results using A's shape frequencies where support
   overlaps, to avoid mistaking routing-distribution changes for faster math.

Provisional mechanism gate: targeted skewed shapes move from roughly 1.8 max/
mean units toward 1.0, with exact expected values computed for each shape;
complete phase time should fall in matching measured comparisons. Uniform
shapes must keep their original ownership. The prior 22% forward gain is a
motivation, not a required serving result for every job.

No dropped records are allowed in a claimed complete job/phase/layer join.
Report incomplete groups instead of treating missing records as zero latency.
GPU-publication arrival timing is not needed to evaluate tile ownership;
cross-clock claims require separately validated clock anchors.

Scheduling contention can be measured separately if existing events support
it: draft ready while target busy, target ready while draft busy, and associated
wait duration. Do not repurpose `collided_jobs` as that metric or add scheduler
changes while testing assignment.

## Phase 5: counters-off serving B and comparison to existing A

- [ ] Restart a fresh B process with optional job/worker/resource/PT/Nsight
  collectors compiled out. Keep ordinary aggregate statistics identical to A.
- [ ] Verify loaded weighted extension, pinned commit, team, all effective split
  tables and manifest. Record readiness and warmup gates before measurement.
- [ ] Use the same request suite and decode formula as compatible A. Prefer
  four saved prompts, two repeats each, serial BS=1: eight sessions total.
  Match A's output length exactly (128 or 256 generated tokens); do not change
  length merely to fit this suggested suite.
- [ ] Use deterministic sampling/seeds and identical prompt order. Preserve
  request/response bytes, generated-token counts, first-token/last-token timing,
  verify-step counts and acceptance per session.
- [ ] Snapshot aggregate counters at quiescent bracket boundaries and take
  deltas; the repeated log snapshot above is lifetime cumulative and includes
  unknown warmup. Do not double-count duplicated log messages.
- [ ] Record CPU frequency/temperature, GPU clocks, node free memory and swap/
  reclaim deltas using lightweight sampling. Do not change other services'
  affinity or priority for this run.

Primary outcome: session decode ms/generated-token, using exactly A's harness
definition, with TTFT reported separately. Tokens/s is its corresponding rate;
do not mix overall request latency, streaming intervals and decode-only rates.
Report per-session/per-prompt results, median, range and aggregate totals.

Secondary outcomes: complete layer/forward latency from compatible diagnostics,
draft and target execution cost by shape, verify-step count, accepted tokens
per verify, output length, memory behavior and tail observations. Cheap
cumulative means remain supporting data; they cannot supply phase p95.

Eight sessions are a short decision sample. Treat sessions/prompts, not thousands
of correlated layer jobs, as independent units for throughput uncertainty.
Report descriptive paired changes when A matches, plus their variability;
do not manufacture statistical certainty from job count. If results are close,
conclude inconclusive rather than expanding the run indefinitely.

## Decision rules

These thresholds are proposed practical experiment gates, not previously
approved production release criteria.

| Outcome | Action |
|---|---|
| Any numerical/ownership/completion failure | Stop; weighting remains default off |
| Incorrect build, split mismatch, unvalidated warmup or incompatible A | Correct provenance or report B-only; no causal speedup claim |
| Balance improves, but matching phase time does not | Inspect locality, extra assignment overhead and equal-unit cost differences |
| Phase/forward improves but decode does not | Inspect CPU-versus-DMA/GPU critical path and target/draft scheduling |
| Clear repeated decode gain, proposed >=5% lower median ms/token | Candidate for a broader serving validation, if correctness and tails hold |
| Gain smaller than variability or inconsistent across prompts | Inconclusive; keep default off and report evidence |
| Uniform/single-row p50 systematically >2% worse, or recurring new large tails | Investigate before promotion; do not hide regressions in pooled mixed cases |
| Acceptance/workload changes materially | Report actual end-to-end result, but use matched-shape CPU evidence for attribution |

Evaluate acceptance changes explicitly; an approximately >1% relative change
is a review flag for this short test, not proof of broken arithmetic. Exact
kernel parity is the correctness gate. Do not normalize token-rate gains by
acceptance using an invented correction formula.

One abnormal scheduling sample is not enough to reject weighting; recurring
shape-conditioned regressions or a repeatable tail increase are. If quantiles
have few samples, state sample counts and show maxima rather than implying a
stable p99 estimate.

If promising, the next experiment can separate draft-only and target-only
weighting, use more prompts, and evaluate changed calibration. None is part of
the first fixed-placement B. Keep production default off until that decision.

## Execution discipline, outputs and estimated cost

Read `.claude/rules/divix01-run-protocol.md` in the implementation checkout.
Laptop edits go through explicit-path commits and origin push/fetch into a
private divix01 checkout; never copy a code tree or run in production checkout.
Print `sglang.__file__` with private `$PWD/python` first. Cap CPU jobs to 0–63;
64–71 are NVMe IRQ cores. Preserve GPU/disk lock order; do not double-lock a
driver whose child `run_arm.sh` acquires the GPU lock. Respect actual serving
main/worker affinity from the baseline, rather than wrapping it in an arbitrary
GPU-only mask. No concurrent GPU benchmark or other experiment arm.

Store results on remote disk under a new
`/data/models/slang/nvfp4-work/row-weighted-serving-<run-id>` directory. Copy only
artifacts to a new laptop directory under
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/row-weighted-serving-<run-id>`.
Retain hashes and verify the copy. Do not put large analysis caches on RAM-backed
temporary storage. Preserve failures and actual exit status.

Required deliverables:

- `manifest.json`: pinned source, builds/hashes, model/fixture, topology,
  effective options/split tables, request suite and A compatibility verdict.
- `correctness/`: exact focused commands, raw outputs, reference hashes and
  extension parity/build-selection results.
- `diagnostic/`: compile definitions, gated raw job/worker/shape records,
  drop counts, joins and shape-conditioned analysis.
- `serving/`: counters-off build identity, warmup verdict, raw per-session
  timing/acceptance/output records and counter deltas.
- `comparison.json`: per-prompt A/B results or explicit unavailable fields,
  mechanism evidence, control regressions and sample counts.
- `findings.md`: result, limits, decision and the next bounded experiment.
- Remote/local SHA256 manifests; full shell commands and real exit codes.

Budget estimates, not measured completion promises: focused warmed tests and
native sanity are minutes; steady diagnostic measurement is 10–20 seconds;
eight short decode sessions are roughly a few minutes after warmup. Model
launch/warmup and compilation dominate: prior warm launch ~5.5 minutes, cold
launch ~10–15 minutes, with roughly 50–100 seconds per cold host module.
Two serving launches can therefore cost 20–45 minutes depending on cache and
gates, excluding implementation/debugging and lock queues. A cold EXL3
extension rebuild may add more. Check build progress before calling a timeout
a runtime hang.

Stop after the short correctness/diagnostic/serving sequence and deliver the
findings. Do not automatically sweep CPU shares, migrate NUMA memory, tune
CHUNK_M, add tracing tools, or start a long benchmark campaign.
