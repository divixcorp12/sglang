# Experiment: routing-aware CPU expert calibration for DSV4.1 DSpark

Date: 2026-10-07. Status: experiment design; no instrumentation or policy changes implemented.

## Decision to make

Determine whether the target's current calibration leaves useful CPU capacity unused, and choose the simplest calibration calculation that reduces **measured decode time per accepted token**. Optimize the existing execution order and legal CPU/GPU assignments. Do not change draft residency, victim capacity, CPU math, token acceptance, or scheduling overlap in this experiment.

“Optimal” has two bounded meanings here:

1. The best legal suffix split for a replayed layer at its recorded cache state: an empirical, local oracle.
2. The best deployable policy among the tested candidates on unseen sessions, including its own decision overhead and subsequent cache effects.

The local oracle is not a claim of globally optimal cache management or serving performance. More CPU utilization is a diagnostic, not the objective. A finding that the current split is already adequate is a valid outcome.

> 🧠 **From Hindsight memory (Key decisions and rationale)** — Prior project decisions require reproducible measurements before speculative-verify or device changes. This is historical guidance; the current execution and benchmark limitations below were checked in source.

## Verified starting point

- Source baseline: `74bb721843`; the both-CPU production configuration has a five-position draft and a six-position target verify, top-k 6, V=8, a 40-lane wire, two NUMA groups, and ten CPU workers per node. Freeze and record the actual resolved configuration at execution time.
- `CpuExpertService.calibrate()` passes the verify token count to `calibrate_cpu_split()`. `CpuExpertEngine::write_calibration_table()` routes every selected calibration expert to every token. The report omits the token count.
- Live target routing writes each expert's actual token mask. `type_lanes()` chooses a suffix of each node's eligible, unforced lanes using `split[node][n]`. Forced lanes remain on the CPU regardless of that table. The current calibration submits no representative forced-lane background work.
- In the target routed-expert path, `Exl3RamMissRowBackend.post()` queues a copy wait before `Exl3MoEMethod._apply_graph()` runs the fused GPU expert pass. CPU work overlaps transfers; it does not currently overlap that following GPU expert pass. The draft path separately posts CPU work, runs GPU experts, then joins. Node 0's CPU team serves both clients.
- The 2026-10-07 native benchmark, ten workers on node 1, measured post-merge one-token/one-expert p50 at 0.415 ms and six-token/shared-expert p50 at 1.679 ms. These are bare forwards, not loaded layer timings. A sixteen-worker identical-binary control varied by 3.3% for one expert, so historical 1.2% noise is not today's assumed floor.
- Read-only corpus inspection found 2,674 sessions at `divix01:/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`, with `session_id`, `domain`, `source`, `split`, `context_chars`, and `turns` fields. Freeze its hash and a selected-session manifest before collecting measurements.

Relevant source:

- `python/sglang/srt/layers/moe/cpu_experts/{service,policy}.py`
- `python/sglang/kernels/jit/csrc/moe/expert_stream/host/{split_calibration,cpu_experts,copy_engine}.h`
- `python/sglang/kernels/jit/csrc/moe/expert_stream/{lease_device,lease_kernels}.cuh`
- `python/sglang/srt/layers/moe/exl3_ram_miss.py`
- `python/sglang/srt/layers/quantization/exl3/exl3.py`
- `benchmarks/dsv41_baseline/{arm_env.py,run_arm.sh,launch_prod.sh}`

Raw native benchmark evidence: `divix01:/data/models/slang/nvfp4-work/cpu-postmerge-20261007/` (`ab-10-workers`, `ab-16-workers`, `noise-16-workers`).

## Hypotheses and controls

| Hypothesis | Measurement that decides it |
|---|---|
| All-token calibration overprices sparsely routed experts | Joint split sweeps on identical states with actual token masks, compared with the all-routed masks |
| Forced CPU work changes the useful discretionary split | Repeat at representative forced-hit and forced-miss loads; retain their real arrival times |
| Separate node tables miss shared PCIe/DRAM effects | Measure node 0 and node 1 together, including asymmetric work and both-node traffic |
| A better fixed table is sufficient | Compare a fitted per-node static table with the local oracle on held-out records |
| Routing-aware decisions add useful improvement | Compare a small measured lookup policy with the static table, including lookup cost |
| Local savings survive changes in residency and acceptance | Untraced full-server A/B on unseen sessions, including cache, misses, and acceptance counters |

Do not assume all hypotheses are true. In particular, mandatory CPU work can make assigning additional optional experts counterproductive.

## Fixed conditions and dataset

Keep both CPU clients on, block size 5, V=8, hot-GPU budget 10,840 MiB, static memory fraction 0.78, current draft resident set, MAX_M=4 / CHUNK_M=2, compiler/ISA, ten workers per NUMA node, and existing keep-warm/spin settings fixed. Record resolved core lists, NUMA bindings, GPU/driver, CPU frequency/thermal state, environment, kernel/build digests, model hashes, and cache capacities. Do not run builds, tests, or unrelated workloads during timings. Reserve CPU workers and their SMT siblings from other work where possible; record any remaining background activity.

Use a deterministic manifest, seed `20261007`, of **24 sessions in three sets of eight**: fit, validation, and final test. Stratify using available domain and context-length metadata; group sessions from the same document/conversation/source lineage so related prompts cannot cross sets. Audit historical use and keep the first eight previously benchmarked sessions out of the final test. Do not select by policy performance or realized acceptance. Record session IDs, corpus indices, grouping rationale, and exclusions.

Fit costs and tables only on fit sessions. Validation chooses one finalist and freezes its settings. Final test compares that frozen finalist with the current policy; do not retune after seeing it. This is a hardware-serving benchmark on the selected corpus, not proof of domain-general gains. Preserve an additional batch-1 regression panel outside the DSpark policy fit.

The current `run_arm.sh` selects indices from its fixed eight-session subset and constructs prompts truncated to 256 tokens. It cannot implement this new dataset by setting indices 8..23. Add an experiment-only manifest adapter while reusing the existing launch/lifecycle controls. Hash the tokenized inputs actually submitted, preserve the intended context-length strata after preprocessing, and state all truncation rules. Use the old 256-token panel only as a separate reproducibility control. The main panel must retain short, medium, and long contexts represented in the selected corpus, within the pinned server's capacity.

## Phase 0: reproducible baseline and instrumentation boundary

**Output:** `manifest.json`, `environment.json`, `baseline.json`, and an instrumentation-overhead report.

1. Pin the pushed source commit and model artifacts. Use a private divix01 worktree; verify `sglang.__file__` resolves there. Register a new Python tree for `run_arm.sh` in its existing generation registry.
2. Reproduce the current `dspark-both` arm, excluding startup, prefill warm-up, and pre-arming verifies from steady-state metrics. Record their costs separately.
3. Run an identical-policy A/A control with the same session/reset protocol. Measure current variation rather than borrowing old noise estimates.
4. Add experiment-only, bounded observation to an instrumented build, with no changed lane choices. Compare observation-on/off on the fit panel. Prefer a preallocated native/device ring and batch drain after the measured interval; no per-layer Python readback, formatting, allocation, or file writes.
5. Require 95% paired confidence bounds within +/-2% overhead for the chosen observation mode. If that cannot be established, use it only to collect workload descriptions and replay attribution. Performance admission remains on an observation-disabled build.

An implementation checklist must explicitly identify new instrumentation and entry points. Existing cumulative `jobs/lanes/forward_ns` counters cannot supply token multiplicity or layer critical-path attribution.

## Phase 1: capture real workload shapes

**Output:** bounded `records` and `cycles` datasets with a schema, hashes, coverage counts, dropped-record counts, and a sampling manifest.

For each sampled target layer record, retain:

- Session/turn, verify-cycle ID, layer ID, sequence/generation, live token count, top-k, and lane-plan order.
- All routed expert IDs and weights, per-expert token mask and popcount, and actual token activation rows for a small replay subset.
- GPU-resident routes separately from the streamed plan. Forced count is derived from `forced_from` in that plan, **not** `all_unique_routed_experts - V`.
- Per-node eligible optional lanes and their order; forced lanes; RAM-hit/miss kind; home node; source/destination slots; relevant cache/slot-map state and read-completion order.
- CPU queue/claim/start/end/completion, DMA issue/completion, forced-miss read completion, copy-wait release, GPU routed-MoE start/end, and combined output-ready events.
- Adjacent draft-stage timings, draft/target job transitions, keep-warm state, and idle intervals. Do not assume simultaneous draft and target computation merely because they share an engine.

Keep host intervals on a common monotonic clock and GPU intervals on a common CUDA clock. Align clocks only with a validated mapping; never subtract a host timestamp directly from a CUDA-event timestamp. Full post-to-output latency must use endpoints on the same validated timeline. Distinguish host queueing, CPU active work, waiting for NVMe, DMA, and exposed layer wait.

Start with 8,192 bounded metadata records sampled deterministically across all target layers and fit sessions, plus at most 128 representative full activation/state snapshots. Stratify snapshots by token multiplicity, node balance, optional-lane count, forced-hit/miss load, and observed latency tail. Include natural record frequencies so oversampled tails can be reweighted. Increase limits only to fill a documented coverage gap. Missing/overwritten records invalidate affected timing joins; they are not zero-cost samples.

Collect a separately identified validation replay set using the same sampler after fitting, without updating costs or candidate parameters from it. Final-test sessions remain unused until the finalist is frozen. Any additional sampling or model revision prompted by validation stays outside the final-test set.

## Phase 2: measure costs and establish a layer oracle

**Output:** raw paired timings for each replay state/action, a per-state oracle with uncertainty, and a predicted serving headroom report.

First extend the native benchmark only as needed to run actual multi-token route tables through the existing CPU engines. Measure token masks with popcounts 1..6, their mixtures, and forced work at observed quantiles on both nodes. Use isolated bare forwards to explain chunk reuse and bandwidth. Then run both-node CPU jobs concurrently with the real DMA path, real slabs/NUMA placement, and the recorded job ordering.

The existing bare fixture has five experts per layer. Repeating those five weights across many slots does not establish a realistic 36-route working set. The existing full-stack `DeviceSim` uses a simulated device and a host copy backend; it cannot establish real PCIe, GPU waits, or verify latency. Reuse its lifecycle utilities, but implement the missing multi-token, forced-lane, real-CUDA replay support before claiming a full-layer result. Synthetic states are labeled stress tests and never mixed into empirical serving weights. Policy-fitting profiles must satisfy actual top-k routing constraints: six tokens at top-k 6 cannot all route eight experts each, even though the current synthetic calibration can time that shape.

For state `s`, let `E_g(s)` be node g's eligible optional lanes in the current plan order, `n_g = |E_g|`, and `F_g(s)` its fixed forced lanes. A legal action is:

```text
a = (k0, k1), where 0 <= kg <= ng
CPU_g(a) = F_g + last kg lanes of E_g
GPU(a)   = unchanged resident routes + remaining legal GPU-served lanes
```

Keep expert homes, mandatory lanes, plan order, token masks/weights, victim assignments, and accumulation rules unchanged. With V=8 globally and `n0+n1 <= 8`, there are at most `(n0+1)(n1+1) <= 25` legal suffix actions. Assert this from each captured plan. This is a tractable joint sweep, not a search over arbitrary expert subsets.

For every replay state, enumerate all legal actions in randomized, balanced order. Start with eight alternating rounds of ten timed repeats after a discarded warm-up; compare identical-action controls and add repetitions only when the uncertainty can change the chosen action. Rotate real layer/weight working sets and include observed idle gaps; do not infer serving cost solely from tight-loop, cache-hot forwards. Readiness and lease ordering remain those of real jobs.

Reset each action to an equivalent, verified initial snapshot using a purpose-built private replay tier/device state. Restore host and GPU maps, cache contents, staging occupancy, output rows, and generation state without breaking monotonic job sequences or reusing live records. Verify a state checksum before each action. Separate warm-RAM and recorded NVMe-miss experiments; record disk/page-cache preparation and actual read times rather than inventing deterministic miss latency. No action may benefit from a previous action's cache mutation.

Measure:

```text
L(s,a) = time from layer post to combined routed-MoE output ready
a_oracle(s) = argmin_a E[L(s,a)]
```

The useful decomposition is approximately readiness of all CPU groups and transfers, followed by the GPU expert pass and combine. CPU readiness includes queueing and late CPU jobs after reads; DMA readiness includes read dependencies and shared-link contention. Measure the timeline directly rather than summing unrelated isolated medians or assuming CPU and GPU expert compute overlap on the target.

Report mean, p50, p95, uncertainty, and the set of statistically indistinguishable actions. Retain the current action when its apparent saving is below the measured uncertainty; use a documented bandwidth-preserving tie rule only within that indistinguishable set. Do not call a noisy per-cell minimum a reproducible oracle win.

Report local headroom relative to the current action and group it by forced work, token multiplicity, node, and layer. Aggregate predicted savings over complete captured verify cycles and divide by their total accepted tokens. Mark this as a fixed-state projection: future cache effects and changes in draft/acceptance can invalidate its serving prediction.

## Phase 3: derive calibration calculations

**Output:** versioned candidate tables/models, their feature definitions and fallback, fit diagnostics, and held-out oracle regret.

Compare these policies on the same captured states:

| Policy | Calculation | Purpose |
|---|---|---|
| A: current | Current all-token measured grid and per-node `split[n]` | Serving control |
| B: no optional CPU | `kg=0`; forced CPU lanes remain | Bounds the optional split's contribution |
| C: representative static table | Per-node `split_g[n]` fitted against joint replay latency, weighted by observed record frequency and mandatory work | Lowest-complexity candidate |
| D: small routing-aware lookup | Joint `(k0,k1)` choice from measured state features, with current suffix semantics | Tests the value of adapting to each record |
| Oracle | Best measured legal action per frozen state | Local comparison bound, never a production policy |

For C, a node's choice depends only on its `n`, but fit both tables against **joint** measured latency. Optimizing each node's isolated CPU/DMA curve is not equivalent. Use a deterministic bounded search and retain its search log; report the best-found table, not an unproved global optimum over all tables.

For D, candidate features available at decision time are optional-lane counts, per-expert token-mask popcounts/chunk counts, forced CPU token work and hit/miss mix per node, and GPU/transfer demand. With CHUNK_M=2, `sum(ceil(popcount(mask)/2))` is an explanatory feature, not an assumed exact cost formula. Test whether these features improve prediction over just unique-expert counts. Do not use future read completion, realized latency, or acceptance as policy inputs.

Bucket only to the resolution supported by observations. Treat the two nodes' workloads jointly because DMA and host bandwidth are shared. Fit mean layer latency first; report tail behavior separately. Unsupported or out-of-range states fall back to the current split with unchanged mandatory lanes.

The current typing pass runs before the later token-table write. D therefore requires an explicitly measured experiment-only path to make routing features available before selection. Do not describe it as a drop-in change to `split[n]`. Include feature extraction, table access, code size/register effects, and any graph changes in its measured overhead.

Promote C if it captures at least 80% of statistically resolvable local-oracle savings on validation and satisfies tail constraints. Evaluate D only if C leaves meaningful headroom; select D only when its additional measured benefit exceeds its overhead and uncertainty. If neither policy has resolvable headroom, stop with a no-change result. Freeze one finalist before final serving evaluation.

### Proposed startup calculation to evaluate

The offline sweep establishes the workload profiles and candidate model, not an unrestricted production-time search. Select at most 24 representative fit-state profiles that cover the observed routing/forced-work distribution, with frozen empirical weights. Benchmark a bounded shortlist of actions on these profiles at startup, on the normal CPU engines with concurrent real DMA. Fit/update costs from those observations and derive C's joint-fitted static tables or D's small lookup. Evaluate shortlist regret against the full offline action sweep on validation; prune further if startup exceeds its time/scratch budget. Measure the incremental scratch and cleanup, and label any GPU-expert cost retained from an offline profile as such.

Require logs to distinguish tokens per record, unique experts, live token-expert evaluations, forced work, node, and whole-job latency. Install a complete validated policy only after calibration succeeds; keep the prior policy on failure. Version the profile set, weights, cost model, resolved hardware/runtime configuration, and resulting table. A cached offline table is a distinct arm: it must have a matching configuration key and is not evidence that fresh startup calibration fits the budget. If bounded startup cannot reproduce the offline finalist's quality, report that limitation instead of presenting the full offline oracle as a deployable calibrator.

## Phase 4: full-server validation

**Output:** untraced paired session results, correctness results, acceptance/cache/read counters, confidence intervals, and a ship/no-ship recommendation.

Use the same benchmark-capable source build for A and the finalist, selected by an experiment-only policy switch. Keep both CPU clients, VRAM/host budgets, graph shapes, resident sets, model weights, and request parameters identical. Apply the policy consistently across warm-up and measurement, and preserve the same deterministic session order and cache initialization procedure in each arm. Warm-up and pre-arming verifies are excluded from the timed interval. Record resulting cache populations, KV capacities, and accepted lengths rather than assuming equality.

Use validation for screening; on final-test sessions run three ABBA blocks (six server runs per arm), with the initial label assignment randomized and then frozen. Use fresh result paths for every run so the resumable session driver cannot silently skip work. Predeclare maximum generated tokens and request sampling parameters in the manifest; use identical values in all arms. Time a pilot to estimate the run budget before launching the full matrix; warm server starts have historically taken minutes.

Primary endpoint: paired per-session steady-state decode ms per accepted/output token. Also report token-weighted total decode time divided by total output tokens, actual throughput, per-request p95 decode latency, draft time, verify time, accept length, CPU work, GPU waits, bytes transferred, NVMe reads, cache hit rates, and steady-state overflow/reverify counts. Batch-1 gets its own matched control; its policy must retain the existing one-token behavior.

Bootstrap at the session/lineage and run-block levels, preserving pairing. Thousands of correlated layer records are not thousands of independent serving trials. Report 95% intervals and all sessions, including outliers. Do not drop an unfavorable session after seeing its result. Text/acceptance differences are analyzed as possible effects of accumulation order; do not silently divide them out of throughput.

Correctness and performance gates:

- Replay outputs pass the established CPU/GPU numerical contracts for every action, including sparse masks, forced misses, asymmetric nodes, zero optional lanes, and all-token cases. Exactness is required where the existing path promises it; cross-device reassociation uses the existing documented text/numerical bar rather than inventing a tolerance.
- Serving passes the existing `scripts/dsv41/dspark_text_band.py` bar (1.4 nats), with divergences and gaps reported. No new steady-state overflow/reverify growth, lease failure, stale partial output, or missing read.
- A deployment recommendation requires at least **5% lower paired median ms/token**, a 95% paired interval establishing improvement, and no established p95 degradation above 5%. If intervals or tails are unresolved, collect more independent sessions/runs before deciding; do not lower the bar after seeing results.
- Instrumentation is disabled for admission, and D's actual decision overhead is included. No host-RAM/VRAM budget reduction or KV-capacity change may explain the result unnoticed.
- Startup calibration completes within a proposed 10-second incremental budget and 256 MiB incremental scratch ceiling, with measured breakdown. These are acceptance targets, not current implementation facts. Failure/timeout/out-of-range states retain the existing valid split and cannot partially install a table.

## Execution, scope, and handoff

This change is **design only**. No experiment code has been written, no performance gate has run, and no production change is authorized by a positive microbenchmark alone.

Implementation order:

1. Add bounded observational schema and overhead control; collect and audit fit data.
2. Add missing multi-token/forced-lane replay support and validate equivalent snapshots and output correctness.
3. Sweep joint actions with real transfers and GPU execution; quantify headroom before writing a new serving selector.
4. Fit and validate C; build D only if the remaining headroom justifies it.
5. Freeze the finalist, execute held-out untraced A/B, and issue the recommendation.

Follow `.claude/rules/divix01-run-protocol.md`: pushed commits in private worktrees, explicit `PYTHONPATH`, cores 0–63 with capped threads, no competing benchmark lanes, and real process exit statuses. Take `rowimg-disk.lock` before `cc-gpu.lock`; let `run_arm.sh` own its GPU lock rather than nesting it under a second lock acquisition. Production stop/restart remains subject to the owner's explicit approval; inspect live state at execution time.

Reuse existing native/serving tools where they support the required shape. New wrappers, routing-feature hooks, multi-token real-CUDA replay, per-group experimental table control, and instrumentation described here are **to be implemented**, not existing flags or capabilities. Keep kernel math, V, draft residency, and MAX_M fixed. Returning VRAM through the Triton prefill-merge work or overlapping target GPU experts with CPU execution is a separate experiment.

Store results under a unique `divix01:/data/models/slang/nvfp4-work/cpu-calibration-<run-id>/` directory. Required artifacts: manifest/environment/build hashes; instrumentation overhead; captured records and sampling weights; raw action timings and identical-action controls; oracle uncertainty/headroom; candidate parameters and fallback; validation/regret; untraced final A/B; correctness; startup budget; and a decision report documenting which hypotheses survived. Preserve commands and actual exit codes next to every reported result.
