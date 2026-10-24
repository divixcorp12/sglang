# Handoff for the T3 fixed split comparison

Written 2026-10-13. This handoff carries the session-level findings and a concrete diagnostic plan for comparing prefetch off with T3 while holding CPU allocation constant. The immediate question is why fewer demand NVMe reads do not consistently shorten verification. Do not start with another predictor training run.

**Current authorization is to write this handoff. No implementation, capture, server launch, benchmark or production change was authorized by that request.** The next session should inspect the state below and prepare the experiment through the repository's hx workflow before execution is authorized. This file is a plan, not evidence that its new checks or harness additions exist.

## Findings that motivate the comparison

The existing captures are on divix01 at `/data/models/slang/nvfp4-work/ram-prefetch/t3-cap2-20261013/{off,t3,t3cap2}/`. They contain one instrumented run per arm, eight sequential prompts each. `t3` is cap 1. The underlying read limit is per layer record, not per generated token.

The later session analysis is saved locally at `/Users/dnikolaidis/Documents/Codex/t3-cap2-session-analysis-20261013/`:

- [Detailed report](/Users/dnikolaidis/Documents/Codex/t3-cap2-session-analysis-20261013/REPORT.txt)
- [Session CSV](/Users/dnikolaidis/Documents/Codex/t3-cap2-session-analysis-20261013/session_summary.csv)
- [Joined layer and forward data](/Users/dnikolaidis/Documents/Codex/t3-cap2-session-analysis-20261013/analysis.json)
- [Source hashes and output comparisons](/Users/dnikolaidis/Documents/Codex/t3-cap2-session-analysis-20261013/provenance.json)
- [Read-only remote analyzer](/Users/dnikolaidis/Documents/Codex/t3-cap2-session-analysis-20261013/analyze_remote.py)

Mean milliseconds per verification window, with startup forwards excluded:

| Session | Off | T3 cap 1 | T3 cap 2 |
|---|---:|---:|---:|
| CDW | 329.5 | 343.3 | 324.6 |
| ETR | 266.8 | 276.6 | 261.5 |
| TSCO | 279.4 | 300.5 | 283.1 |
| BKR | 260.8 | 260.8 | 248.6 |
| K | 533.3 | 287.1 | 299.2 |
| DISCA | 275.9 | 274.0 | 276.0 |
| WRK | 252.4 | 256.7 | 253.4 |
| VLO | 308.6 | 331.6 | 322.1 |

These windows run from first-layer copy submission to last-layer gate opening. They are not the entire draft plus verification cycle and are not end-to-end ms/token.

The original headline included 7/8/7 complete startup forwards. Session-only windows number 122/123/122. Gate accounting uses 122/123/121 complete forwards, totaling 14,640 layers. One cap-2 DISCA forward has endpoints but no row-25 host record: include its 228.771-ms window in session timing, exclude it from complete-layer attribution. Its missing layer fits within a 2.308-ms interval. Do not silently drop partial forwards or invent their last finisher.

### Read savings can be offset inside the same gate

In ETR, cap 1 changes the mean per verification as follows:

| Quantity | Off | Cap 1 | Change |
|---|---:|---:|---:|
| Demand NVMe rows remaining | 78.63 | 63.89 | -14.74 |
| RAM-hit CPU and DMA completion envelope | 153.08 ms | 168.03 ms | +14.95 ms |
| Miss completion beyond that envelope | 54.49 ms | 47.97 ms | -6.52 ms |
| Work outside layer gate windows | 59.18 ms | 60.54 ms | +1.37 ms |
| Verification window | 266.79 ms | 276.59 ms | +9.80 ms |

The saved reads help, but the other dependencies take longer. This is an accounting result, not a proof that T3 caused the CPU slowdown.

Last observed dependency across both NUMA groups, on complete session forwards:

| Dependency | Off | Cap 1 | Cap 2 |
|---|---:|---:|---:|
| CPU job for RAM-miss experts | 61.3% | 56.0% | 55.3% |
| CPU job for RAM-resident experts | 28.6% | 27.7% | 30.6% |
| DMA | 10.1% | 16.3% | 14.1% |

The miss-CPU category includes prefetched experts: prefetch removes a demand read, not expert computation. About 76% of cap-1 layers using prefetched rows still have an uncovered miss. Gate release after the last observed dependency is about 1 us at the median; its p99 is only a few us. The gate-release mechanism is not the multi-ms loss.

### CPU execution and split calibration varied

Cap-1 hit jobs run slower in several sessions even after matching session, layer, NUMA group, expert count and live token-expert count: ETR approximately +0.364 ms/job, CDW +0.463, TSCO +0.453 and VLO +0.481. BKR, DISCA and WRK differences are much smaller. Both NUMA groups show the slowdown. Matching these coarse shapes does not control expert identity, cache locality or the token-work distribution among experts.

Cap 1 often executes fewer live token-expert pairs while taking more CPU time. ETR has about 351 versus 378 hit pairs per verification, but summed hit-job execution rises from 203 to 218 ms. Those sums include parallel groups and are not wall time.

The startup-calibrated split tables also differ between arms. Off group 0 begins `[0,1,2,2,3,3,4,5,5,...]`; cap 1 group 0 begins `[0,1,1,2,3,3,3,4,5,...]`. ETR's RAM-to-GPU DMA grows from about 1.287 to 1.399 GB/verification. Routes and residency differ too; do not attribute all extra DMA to the table without a controlled comparison.

### The baseline has a large arrival stall

Off/session K ends with verification windows of 559, 567, 949, 1,534 and 1,008 ms. In one layer, the final miss CPU job is submitted after 87.42 ms and executes in 0.43 ms. The delay is before the job becomes runnable, consistent with a demand-row arrival/service stall. These logs cannot identify a particular SSD or distinguish device delay from reader scheduling.

Excluding K as a diagnostic sensitivity check changes pooled mean timing to off 275.92, cap 1 283.59 and cap 2 273.86 ms. This is not permission to discard K from the benchmark. K's cap-1 output differs after five tokens and is 63 tokens long versus 42 off. Cap 2 produces off's K output. Other text differences are recorded in `provenance.json`; identical final text still does not guarantee identical draft blocks.

The old claims that layer-0 stability excludes drift, that a no-miss slowdown can be multiplied by all 40 layers, and that approximately 0.5% bounds learned prefetch gains are not supported. Keep CPU, DMA, arrival delays and session composition separate.

## Repository state to reconcile before implementation

At handoff creation, local HEAD was `7413b974de0f725d9c87895f90558e6e523cb0a5`. There were existing changes in `DSV41_REFERENCE.md`, `analysis/dsv41-drive/dspark/both_cpu_ab.py`, `docs/docs/references/environment_variables.mdx`, `python/sglang/srt/dsv41_config.py`, and `python/sglang/srt/environ.py`, plus untracked `RAM_PREFETCH_L2_HEADS_NEXT_HANDOFF.md`. These are not this handoff's changes. Recheck status and preserve them.

The current reference working tree also contains section 33.35: another cap-2 capture, a cap-3 experiment, and Engram verification graph changes. Those newer captures were not included in the session analysis above. Do not say cap 2 has only ever been captured once. Their raw directories are `ram-prefetch/t3-cap3-20261013/` and `ram-prefetch/engram-verify-20261013/`; inspect separately if needed, without pooling their headline numbers into this study.

The working tree changes `SGLANG_DSV41_ENGRAM_VERIFY_IN_GRAPH` to default on. For this diagnostic series, explicitly set it to **0 in every arm**, matching the execution mode of the captures being explained. This is a temporary experiment override, not a request to change the owner's default. A follow-up on the current Engram recipe is a separate series with the flag explicitly 1 throughout.

## Exact fixed configuration

Use the prefetch-off capture's final calibrated table. The two NUMA groups used the same complete 41-entry table, verified from their stats lines in the hashed source log. Entry n selects the CPU share for n eligible lanes; the existing forced-miss and miss-cut rules still apply. It is not a promise of identical CPU job sizes when routing changes.

```bash
export SGLANG_DSV41_CPU_EXPERTS_SPLIT='0,1,2,2,3,3,4,5,5,6,7,7,8,9,9,10,11,11,12,13,13,14,15,15,16,17,17,18,19,19,20,21,21,22,23,23,24,24,25,26,26'
export SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION=0
export SGLANG_DSV41_CPU_EXPERTS_RETUNE_BATCHES=0
export SGLANG_DSV41_CPU_SPLIT_MISS_CUT=1
export SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX=3
export SGLANG_DSV41_ENGRAM_VERIFY_IN_GRAPH=0
```

An explicit split already bypasses calibration in `CpuExpertService.calibrate`; the explicit calibration flag makes intent auditable. Retuning is separately disabled. Simply switching calibration off without providing the table would instead use the cost-model fallback and would not reproduce the reference split.

Verify the table length against the actual wire (`lanes + 1`, currently 41), `0 <= split[n] <= n`, and exact equality on both NUMA groups before timed requests and after the final request. Record the effective table, not only the intended environment. Keep the kernel and all other settings constant:

- CPU teams: ten workers each, node 0 cores 6–15, node 1 cores 18–27; verify live TIDs and allowed CPU masks.
- Identical target/draft EXL3 extension paths and hashes, tile assignment, thread counts and keep-warm settings. The old captures logged extension hash prefix `477e18ba32b2`; verify the chosen new build rather than assuming an unchanged path implies unchanged code.
- RAM tier: 104 GiB, `SGLANG_MOE_PINNED_HOST_MB=106496`, NUMA split `0:57344,1:49152`; same VRAM cache and prefetch pool allocation. Record effective pool settings because they can reduce demand residency.
- DSpark five draft tokens, six-token static verification, same draft model, seed, sampling, prompt list/order, maximum 128 output tokens and warmup protocol.
- Identical idle-drive prefetch policy and horizon 2 for T3 arms. Do not tune thresholds while testing CPU stability.

T3 artifact on divix01:

```text
/data/models/slang/nvfp4-work/ram-prefetch/l2-t3-head/t3-f109-calib-b075.safetensors
exported file SHA256: f0d7a1b10e029480bcb9687437d7880135d756ea57f186d9a9cd41729fbd5309
source checkpoint SHA256 in metadata: e675e2bdea5b9fae15c0b5b0373f291281c9d2775acd2a5d2bc3544c3a62cb82
threshold: 0.1301608383655548
```

The exported-file and source-checkpoint hashes are different identities. Both were checked while writing this handoff. Verify both again and retain an immutable artifact reference in the run manifest.

## Preparation before the first capture

Prepare a bounded diagnostic driver, using the existing capture and sampler code. No new training or predictor implementation is needed. Required harness additions should follow hx planning and focused validation:

1. Generate a manifest containing commit, effective configuration, both split tables, model/head/library hashes, prompt hashes, warmup policy, arm order, telemetry configuration and output directories. Refuse to overwrite a non-empty output directory.
2. Record explicit per-request start/end timestamps on CLOCK_MONOTONIC and epoch clocks, session IDs, response tokens and output hashes. Record timed-window boundaries after warmup. The old `logprob_probe.py` stores text but not these timestamps. Do not rely on HTTP timestamps rounded to seconds for new joins.
3. Record per-verification request ID, ordinal and accepted-token count if a low-overhead path is available. Otherwise identify this limitation and keep full-cycle client timing separate. The existing optional route/accept capture is available through `spec_margin_capture.py` with verify-split enabled, but adds overhead: enable it identically in every arm of a separate diagnostic series, not selectively.
4. Start and stop bounded telemetry with each experiment server; collect return codes and complete trace/sampler footers. Validate zero dropped trace events. The existing capture driver sets job tracing; resource tracing must be enabled explicitly for the diagnostic series.
5. Verify how environment overrides reach the child. `spec_margin_capture.py` uses `os.environ | server_env(...)`, so arm/base overrides take precedence over shell exports. The current fixed-split keys pass through, but future wrappers may whitelist or override them. Assert effective values in the launched process and actual group tables. Do not assume `both_cpu_ab.py` passes every custom variable through.
6. Use a fresh server and identical warmup for each arm. Keep filesystem and cache preparation consistent. Do not drop system caches or change governors, BIOS settings, SMT, IRQ affinity or power limits as part of this experiment.

## CPU execution stability measurements

> 🧠 **From Hindsight memory (Reusable CPU expert and copy completion tracing)** — bounded CPU/copy events and optional resource snapshots already exist. Current source confirms their scope; the resource snapshots belong to the engine's owning thread, not the whole worker team.

Collect the same telemetry in all diagnostic arms:

| Measurement | Existing support or preparation | Interpretation |
|---|---|---|
| CPU job duration and queue delay | `cpu_submit`, `cpu_start`, `cpu_shape`, `cpu_end` | Separate hit and miss jobs, target and draft; match actual shapes rather than lifetime ms/lane |
| Owner-thread faults, switches and CPU time | `SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE=1`, with job tracing | Difference start/end cumulative counters; not a sum over ten workers |
| All expert-worker execution and runqueue time | `benchmarks/dsv41_baseline/stall_sampler.py`, default 50 ms | Samples `schedstat`, current CPU, faults, switches and wait channels for `exl3-*` TIDs |
| NUMA/process memory and reclaim | Same sampler, memory counters every 250 ms | Correlate major faults, swapping and reclaim with job/arrival stalls |
| Effective CPU frequency and throttling | Check available read-only host tooling; preferably effective-frequency counters at about 1 s | Log each team/core or socket; a requested governor frequency is not actual execution frequency |
| Competing processes and core occupancy | Bounded process/thread samples, approximately 1 s | Include experiment and non-experiment load; a GPU lock does not exclude CPU jobs or owner GPU jobs |
| Arrival and DMA timing | Existing tier, CPU and copy events; drive counters | Keep arrival delays distinct from CPU math and gate-release overhead |

The sampler selects a scheduler using the exact unique job-trace prefix, inspecting launch ancestors if necessary. `Team` names workers like the owner, so the filter should see the two ten-thread teams. Verify **all 20 expert-team TIDs**, including the owners, plus their group/core mapping before accepting telemetry. Also identify tier, copy and speculative-reader threads. Do not silently treat missing worker samples as idle workers.

Record `/proc` tick frequency, TID start times and CPU topology, including SMT sibling relationships, so tick deltas and reused TIDs are interpreted correctly. Sampling at 50 ms cannot resolve every sub-ms interruption; inaccessible/zero wait-channel data is not proof of no stall. Persistent workers spin, so high thread CPU usage alone is not useful-work throughput.

If effective-frequency tooling is unavailable, record that limitation. Do not enable kernel modules, alter hardware policy or change privileges automatically. Scheduling, memory and shape-matched duration evidence remain useful, but a frequency explanation cannot then be confirmed or ruled out.

### Analysis of CPU work

For RAM-hit target jobs, form cells by `(session, layer, NUMA group, call.rows, expert count, live token-expert count)`. Require at least five jobs in each compared arm for a primary cell; report the share of jobs represented and unmatched shapes. Compare job duration medians and p90 values using the **same baseline cell weights** across arms. Report groups separately. Do not sum two parallel groups to predict wall time.

Coarse shapes omit expert identity, per-expert token multiplicity and cache locality. If a slowdown persists without a scheduling/frequency explanation, the next diagnostic should record those shapes or use a fixed-input CPU replay. That replay isolates CPU behavior but cannot establish a serving gain by itself.

For miss jobs, report arrival-to-submit, submit-to-start and execution separately wherever timestamps support them. CPU job submission is a row-ready proxy; without a recorded I/O start, do not label the whole pre-submit interval pure NVMe service time. Correlate long queues with preceding hit jobs, other miss jobs and draft intervals. A runnable wait on a busy worker differs from a delayed read.

## Bounded experiment sequence

All stages below are diagnostic until a later counters-off serving comparison. Each capture uses the same eight sessions and maximum 128 tokens. Record one output directory per arm, for example under `/data/models/slang/nvfp4-work/ram-prefetch/fixed-split-<date>-<run>/`.

| Stage | Order | Purpose and progression |
|---|---|---|
| Baseline repeatability | Off A, Off B | Determine whether fixed allocation and CPU execution repeat before changing prefetch |
| Cap 1 diagnosis | T3 cap 1, Off C | Compare cap 1 against the surrounding Off B/C; retain Off A as an independent baseline check |
| Cap 2 follow-up | T3 cap 2, Off D | Proceed after cap-1 attribution is interpretable; reuse stable Off C as the preceding control |

Arm names are `dspark-both-misscut`, `dspark-both-prefetch-gpu-misscut-idle2-l2-t3`, and `dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2`. Preserve each arm's explicit prefetch/head configuration. The fixed settings and instrumentation are common overlays. Cap 2 keeps the cap-1 threshold for this study; threshold optimization is a separate question.

This is a maximum six-capture initial sequence, not an unbounded sweep. If Off A/B are unstable, do not run the T3 captures yet. Diagnose the recorded evidence; at most one additional off capture may check a specifically identified transient condition after its resolution, within the approved run scope. Do not rerun repeatedly until numbers look favorable.

Suggested diagnostic command shape, **not executed by this handoff**:

```bash
# On divix01 in the approved private worktree, with manifest/telemetry wrapper prepared.
# The driver takes disk and GPU locks; do not take them again in the wrong order.
PYTHONPATH="$PWD/python" OMP_NUM_THREADS=1 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -B \
  analysis/dsv41-drive/dspark/spec_margin_capture.py \
  "$FIXED_SPLIT_RUN_DIR/off-a" \
  --arm dspark-both-misscut --prompts 8 --max-tokens 128 --no-verify-split
```

The driver applies the established child-server and request-driver affinities and thread settings; the parent command above is bounded separately. The wrapper must arrange request markers, check the effective configuration and start/stop telemetry. The command alone does not provide the proposed manifest, markers or stability gate. Verify the repository's configured interpreter before execution; the path above is the current `arm_env.PYTHON`.

### Baseline stability criteria

Freeze these proposed operational tolerances in the manifest before runs. They are **diagnostic flags**, not statistical proof and not measured historical noise bounds:

1. Exact agreement for both effective split tables, model/kernel/head identities where applicable, thread placement, graph mode and measurement configuration. A mismatch invalidates the comparison.
2. For each CPU group, the baseline-weighted median of matched hit-job duration ratios Off B/Off A should be within **0.97–1.03**. Flag a median of matched-cell p90 ratios outside **0.90–1.10**. Require matched cells to represent at least **70% of hit jobs in each arm**; otherwise call the CPU check inconclusive, not passed.
3. For per-session mean verification times, flag a median Off B/Off A ratio outside **0.97–1.03**, or any session outside **0.90–1.10**. Inspect flagged sessions even if the median passes. Output/acceptance changes make this a mixed-workload check rather than a direct CPU-speed comparison.
4. Inspect effective-frequency shifts, runqueue wait, faults, reclaim and competing workloads alongside any timing shift. Flag group effective-frequency changes greater than roughly **5%** when measured with comparable active intervals. Preserve actual data and availability rather than fabricating a pass for missing telemetry.
5. All accepted traces need complete footers, zero reported event drops and explicit session attribution. Count missing layer records and unmatched CPU jobs; retain endpoint-only forwards separately as in DISCA above.

A stable Off A/B pair permits diagnosis; it does not establish enough resolution to declare a 1% speedup. Apply the same checks to Off B/C and Off C/D. Interpolate between surrounding controls only if their drift is small and consistent; otherwise show both comparisons separately.

**Do not reject a T3 run merely because its CPU jobs slow down.** With stable surrounding off controls, a reproducible T3-only slowdown is the effect being investigated. Separate invalid configuration, documented external contamination, unexplained control drift and treatment-associated overhead. Keep every attempted run and explain all exclusions.

## Gate accounting and reporting

At each layer, join both NUMA groups. Let `t0` be first copy submission, `B` the latest RAM-hit CPU/DMA completion (at least t0), `M` the latest miss-CPU completion (at least t0), and `G` gate open:

```text
gate wait = (B - t0) + max(0, M - B) + (G - max(B, M))
```

Validate this identity per layer. The first term is a measured completion envelope, not a fixed floor. DMA observations bound completion through polling. A final miss CPU job does not itself prove the disk was last: prefetched experts still compute. Within the miss extension, distinguish observed winning-group CPU execution from residual waiting without calling the latter pure SSD time.

For each session and arm, report:

- Output hash, generated tokens, verification count and accepted tokens per verify where available; text-divergent sessions stay visible.
- Median, mean and tail verification windows; CPU/DMA envelope, miss extension, release overhead and inter-layer gaps. Separately report the full decode interval and ms/accepted token if the harness records their endpoints reliably.
- Forced misses, prefetched rows used, remaining demand reads, timely/in-flight use, and DMA bytes, all for timed decode windows. Lifetime counters mix prefill/warmup and cannot substitute.
- Last-finisher shares and matched hit-job duration/queue distributions for each NUMA group; CPU frequency, runqueue, fault and memory evidence around outliers.
- Pool size/residency effects and distinct target-verification versus draft-generator CPU work. Anchor-only, draft-only and shared target-expert demand require route/accept data; mark unavailable rather than infer from host events alone.

Summarize session-paired ratios, not just the pooled median of all forwards. Treat sessions and repeated arm cycles as the experimental units; hundreds of forwards in one capture are not hundreds of independent treatment repetitions. Show sensitivity to anomalies alongside, not instead of, the complete results. Do not exclude K merely because it is inconvenient.

## Decisions after the diagnostic series

| Finding | Next action |
|---|---|
| Off/off CPU timing or arrival stalls do not repeat | Diagnose host scheduling, frequency, memory and reader conditions before claiming a prefetch effect |
| Fixed allocation removes the earlier offset | Attribute the new comparison only; varying allocation was a plausible contributor, not retrospectively proven to explain all old differences |
| Off controls are stable and T3 alone slows matched CPU work | Investigate speculative-reader/core placement and memory traffic; distinguish frequency, runqueue and execution effects before changing thread counts |
| CPU is stable but gates still barely improve | Inspect uncovered final misses, late prefetch arrival, CPU execution of prefetched experts and DMA becoming last |
| Cap 2 advances gates consistently under stable controls | Prepare a repeated session-paired serving comparison with heavy tracing disabled; then add native L+2 cap 2 to separate head quality from read budget |

After diagnosis, an order-balanced repeated comparison can quantify performance. This does not require resurrecting the owner's deferred F108 reverse-order experiment. It is a new comparison with fixed controls. Cap 3, new heads, threshold tuning, different CPU kernels and split optimization are outside this initial series.

## Source map and operational boundaries

Read these before changing the harness:

- [Run protocol](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/.claude/rules/divix01-run-protocol.md)
- [Existing L2 handoff](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/RAM_PREFETCH_L2_HEADS_NEXT_HANDOFF.md) and [reference sections 33.30 through 33.35](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/DSV41_REFERENCE.md)
- [Split configuration and calibration](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/srt/layers/moe/cpu_experts/service.py), [split policy](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/srt/layers/moe/cpu_experts/policy.py)
- [Capture driver](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/analysis/dsv41-drive/dspark/spec_margin_capture.py), [probe](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/scripts/expert_prediction/prefetch/logprob_probe.py), [arm definitions](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/analysis/dsv41-drive/dspark/both_cpu_ab.py), [shared environment](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/benchmarks/dsv41_baseline/arm_env.py)
- [Worker sampler](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/benchmarks/dsv41_baseline/stall_sampler.py), [resource trace](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/kernels/jit/csrc/moe/expert_stream/host/job_trace.h), [worker team](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/team.hpp)
- [CPU trace emission](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h), [completion gate](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/kernels/jit/csrc/moe/expert_stream/host/copy_engine.h), [miss timeline](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/analysis/dsv41-drive/dspark/miss_timeline.py), [layer records](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/analysis/dsv41-drive/dspark/layer_misses.py)

Code changes move by laptop commit/push to `origin`, then a private divix01 worktree; no copied code trees. Register the served generation and verify `sglang.__file__` with `PYTHONPATH=$PWD/python`. Use the configured test runner and only affected tests; no laptop pip/uv installs. Keep the existing user worktree and stash untouched.

CPU analysis stays within cores 0–63 with bounded threads. Cores 64–71 remain free for NVMe interrupts. GPU experiments take `rowimg-disk.lock` before `cc-gpu.lock`, with the established server/driver affinities. Avoid double-locking a driver that already owns them. Check live processes and GPU memory before launch and throughout the run: owner jobs may not honor these locks. Do not stop unrelated jobs. Outputs belong on `/mnt/nvme1` or `/data/models/slang/nvfp4-work/ram-prefetch/`, never the expert mirror drives. Use new directories and explicit return-code handling.

**Production on divix01 is stopped. Never start, stop, restart or execute in `cc-expert-prediction/dsv41-direct-prod`, or use production port 7867. Only the owner can restart production or change its defaults.** Fixed split and graph-mode overrides belong only to explicitly authorized experiment servers.

## Completion checklist for the next session

- Preserve this evidence and reconcile current source/default changes before selecting the experiment commit.
- Prepare and review the bounded driver, effective-config checks, request markers and telemetry validation.
- Run only the authorized stages; stop expansion when baseline stability is unresolved.
- Produce a manifest, per-session CSV, CPU shape stability report, gate decomposition, anomaly timeline and clearly bounded conclusion.
- State what remains unexplained and whether a counters-off served comparison is justified. Do not mark prediction accuracy, CPU optimization or production performance as solved solely from this diagnostic series.
