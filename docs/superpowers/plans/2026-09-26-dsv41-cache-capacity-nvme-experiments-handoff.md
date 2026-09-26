# DSV4.1 cache capacity and NVMe tail: experiment handoff

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` when executing this handoff. Complete the measurement gates before changing the serving recipe. This document requests experiments, not an unconditional production rollout.

**Goal:** Reduce decode latency by avoiding expert transfers first, then reducing the NVMe wait that remains exposed after RAM-hit transfers.

**Architecture:** Resume the existing prefill indexer score-cap implementation. Measure its memory and TTFT effects at fixed cache capacity, then allocate verified spare VRAM to the GPU expert cache. After choosing a capacity, attribute the remaining storage tail and test one evidence-selected storage change.

**Tech stack:** Python/PyTorch, existing DSV4.1 benchmark drivers, EXL3 graph-based DIRECT residency, pinned row images, io_uring direct reads, CUDA copy engine, Nsight Systems node-mode traces, diskstats.

**Spec:** The user's 2026-09-26 request for the next experiments following the RAM-miss frontend no-go. The intended order is indexer cap → larger GPU cache → exposed NVMe tail. No changes to W1, the lease protocol, resident-first compute, or generic prefetch are included.

**Status:** Handoff written; none of the experiments below has been run as part of this handoff.

## 1. Read this before resuming

The workspace used to write this file is `/home/dimitri/data/divix/sglang-nvfp4`. Its `master` is behind the newer findings. Do not assume this checkout contains the current production implementation.

The newer reference material was read in `/home/dimitri/data/divix/sglang-nvfp4-worktrees/direct-two-phase`, at `91f4f801add4699ba120cc9ae3366fe3081539fd` when inspected:

- `DSV41_REFERENCE.md` §27.7: indexer-cap implementation and outstanding measurements.
- `DSV41_REFERENCE.md` §26.2: historical cache-capacity estimates; later sections supersede its older copy/coalescing proposals.
- `DSV41_REFERENCE.md` §27.3–27.5: exposed NVMe waits and copy-thread scheduling evidence.
- `DSV41_REFERENCE.md` §27.14: link utilization, SM small copies, async-promotion and prefetch no-go results.
- `docs/superpowers/plans/2026-09-26-dsv41-ram-miss-frontend.md`: completed frontend sizing and real-transfer resident-first study.

Read the latest available versions before running anything. The paths below are repository-relative unless explicitly absolute. The score-cap branch was `cc/indexer-cap`, reported head `b9ae131af0`; it is not merged in the inspected reference. Re-resolve its current commit and inspect its diff. Do not replace the current serving stack with that old branch wholesale.

### Evidence that determines priority

| Finding | Consequence |
|---|---|
| Frontend estimates: 0.324 ms/step for W1 budget zero, 0.411 ms/step for reset-only | Those A/Bs and the reset implementation were skipped. Do not repeat them without new evidence. |
| Real-copy resident-first: 12.81 µs/layer best at 4+2, about 0.512 ms over 40 layers | Below the prior 1 ms gate; leave the split shelved. |
| CE carries about 995 MB/step in the frontend trace | Bytes avoided are a more promising target than frontend overhead. |
| Prior active-copy rate about 13.6 GB/s; CE and SM overlap already approaches measured link capacity | A lower CE-only rate during SM activity is not by itself unused PCIe bandwidth. |
| Historical extra GPU-cache GiB saved about 2–2.7 transfers/token | A hypothesis for the local capacity curve, not a guaranteed linear extrapolation. |
| Indexer 128 MiB score budget estimated to save about 2.5 GB of transient VRAM | Not measured end-to-end. Peak savings do not automatically equal reusable persistent capacity. |
| Existing NVMe analysis found about 13 ms/step exposed after RAM-hit copies | Re-measure after increasing the cache; the route/miss population may change. |

The six tasks in the frontend document were gated tasks: frontend sizing and real-transfer benchmarking ran; W1 A/B, reset implementation and its lifecycle tests did not. Do not describe them as six completed GPU experiments.

## 2. Global constraints and run discipline

- This handoff authorizes no production interruption. For execution, use an agreed maintenance window and private checkout on `divix01`. Do not edit or launch experiments in `dsv41-direct-prod` or `dsv41-direct-live`.
- Do not change `benchmarks/dsv41_baseline/arm_env.py::base_env()` during experiments. Pass per-arm overrides. A production recipe update is a separate decision after results are reviewable.
- Recheck cwd, branch, dirty state, relevant handoffs and currently running jobs. Preserve unrelated edits. Integrate the score-cap change onto the agreed current baseline in an isolated checkout; record the resulting full SHA.
- Run A and B from the **same integrated commit**; budget zero is the control. Preserve all other recipe choices, especially EAGER, CE, SM small copies if active, graph gathering, row images, DIRECT stage 2, residency updates, pinned capacity/NUMA split, mirrors, model, context length and memory fraction.
- Verify effective live-server environment, arguments and `sglang.__file__`. Record requested hot-cache MiB **and actual allocated per-layer slots/bytes**, since allocation rounding can change the effective increment.
- On divix01, use `/data/models/slang/.venv/bin/python`. Scratch/results go under `/mnt/nvme1/`, not the root filesystem. Preserve existing result directories; every arm gets a unique tag.
- Lock order: `rowimg-disk.lock` before `cc-gpu.lock`, under `/data/models/slang/nvfp4-work/`. Existing indexer smoke takes both itself; do not wrap it in the same locks. Audit locking before wrapping other drivers. Never break a lock to start a run.
- Keep background CPU/memory and storage load consistent. Record interference, CPU affinity, GPU clocks and link state. Do not change unrelated services without authorization. Do not interpret contaminated runs as an optimization result.
- Use unprofiled runs for latency decisions. Use separate node-mode nsys captures for attribution; graph-mode capture is refused with CE. Trace overrides, including NUMA capacity changes, must be recorded and matched between traced arms.
- Compare the same prompt corpus, seeds, decoding settings, warmup and session order. Separate cold post-prefill decode from later steady decode. Repeat short performance arms in A–B–B–A order to detect drift; long-context peak validation need not repeat that entire expensive matrix unless unstable.
- The old peak driver does not stop on every failed smoke: inspect each arm's exit code, health, completed long response and logs. Do not treat the driver reaching its end as success.
- Keep per-step CE bytes, SM-transferred bytes where available, GPU misses, RAM misses and NVMe rows separate. A counter change can reflect ownership/accounting rather than fewer physical transfers. Use counter deltas over matching decode windows, excluding startup/prefill/warmup.
- No hardware run is assumed available locally. The raw frontend evidence is on `divix01:/mnt/nvme1/frontend/`; do not claim it was reproduced by reading the write-up.

## 3. Review focus

1. **Wrong generation or recipe:** old indexer branch loses newer optimizations. Task 1 checks integration and effective recipe; A/B uses one SHA.
2. **Peak sampling misses a transient:** the existing sampler polls at 50 ms. Task 2 treats sampled headroom as incomplete and checks allocator peaks/retries where available; Task 3 validates the actual larger-cache configuration at maximum context.
3. **Chunking changes routing:** Task 1 runs indexer parity including ties/tail chunks; Tasks 2–3 compare greedy outputs. A mismatch blocks a performance verdict until explained and fixed.
4. **Capacity consumes KV/context headroom:** Task 3 preserves supported context/concurrency and the memory fraction; Task 4 validates the selected capacity under the largest supported workload.
5. **Tail attribution double-counts overlap or mixes clocks:** Task 5 uses timestamped dependencies and disjoint intervals, not a sum of kernel durations or raw host/GPU timestamps.

## 4. Experiment 1 — establish one reproducible baseline

**Purpose:** Make the existing score-cap implementation runnable against today's stack, without changing policy.

**Existing files:**
- Branch `cc/indexer-cap`: `analysis/dsv41-drive/indexer-cap/smoke.sh`, `drive_peaks.sh`.
- `test/registered/unit/kernels/test_dsv41_torch_indexer_chunking.py`.
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py` and config wiring identified by the branch diff.
- `benchmarks/dsv41_baseline/run_arm.sh`, `arm_env.py`, `paired.py`, `generations.py`.

- [ ] Resolve current baseline and cap commits; inspect the full cap diff and integrate only its required code/tests/harness into a private checkout.
- [ ] Verify `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB=0` retains the built-in 1 GiB score budget; `128` chunks rows without truncating context or changing precision/top-k.
- [ ] Run the existing CPU/CUDA indexer parity tests appropriate to the integrated diff. Historical result was GPU 7 passed; record the actual current result, not that expected count as proof.
- [ ] Pin and register the integrated Python generation using the existing harness. Do not weaken provenance or clean-tree gates to start an arm.
- [ ] Run a short cap-off/cap-on smoke at `HOT_GPU_MB=14336`, with `long_tokens=0`, confirming startup, graph capture, greedy parity and effective flags.
- [ ] Save a manifest with SHAs, corpus identity, env/argv, memory fraction, context/concurrency, slot counts, affinity, hardware and output paths.

**Deliverable:** A validated candidate checkout and baseline manifest. If integration/parity fails, resolve it before peak or speed measurements.

## 5. Experiment 2 — score-cap memory and TTFT at fixed cache capacity

**Question:** Does the cap actually free long-context peak memory, at an acceptable prefill cost?

| Arm | Score budget MiB | GPU hot cache MiB | Prompt tokens |
|---|---:|---:|---:|
| A30 | 0 | 14336 | 30000 |
| B30 | 128 | 14336 | 30000 |
| A32 | 0 | 14336 | 32000 |
| B32 | 128 | 14336 | 32000 |

Use 32k only if it is within the actual supported context including generated tokens. Validate prompt token counts from responses. If the supported maximum differs, test that maximum and state the deviation.

The existing verified command interface is below. Set `EXP_WT` to the executor's verified integrated private checkout, not the old branch assumed current. Set `EXP_RUN` to a fresh unique run prefix; confirm neither result directory already exists.

```bash
bash "$EXP_WT/analysis/dsv41-drive/indexer-cap/smoke.sh" "${EXP_RUN}-a30" "$EXP_WT" 14336 0 30000
bash "$EXP_WT/analysis/dsv41-drive/indexer-cap/smoke.sh" "${EXP_RUN}-b30" "$EXP_WT" 14336 128 30000
```

Run these individually with an explicit exit-code and result check between them. Repeat with 32000 and suffixes `-a32`/`-b32` for A32/B32. Prefer this over blindly launching the old four-run driver.

- [ ] Complete the 30k pair before paying for the maximum-context pair; stop on crash, incomplete response, parity mismatch or OOM.
- [ ] Record startup/graph, prefill and decode memory separately, TTFT, allocator retries, token counts, greedy output parity, and run duration.
- [ ] Use `analysis/dsv41-drive/hot-cache-size/vram_peaks.py` on each result directory. Its total-memory constant is hardware-specific: verify against the actual GPU. Its 50 ms samples are sampled peaks, not exact maxima.
- [ ] Record allocator max allocated/reserved and OOM/retry information if the harness exposes them. If absent, explicitly mark unavailable; any added peak telemetry belongs in a separate instrumented validation run with clear phase boundaries.
- [ ] Preserve `env.txt`, `argv.txt`, `driver.log`, `server.log`, `responses.jsonl`, `long.json`, `vram.csv`, `phases.txt` and `retries.txt`.

**Gate A:** Continue to cache expansion only if parity holds, memory reduction is repeatable enough to justify a concrete increment, and the workload remains within its existing memory envelope. Use a provisional **5% maximum TTFT regression** for the cap-only comparison; this is a proposed experiment gate, not an existing product SLA. If exceeded, try a 256 MiB budget at fixed cache size once and compare the memory/TTFT tradeoff. If neither budget offers useful headroom without unacceptable cost, report no-go and proceed to Task 5 at the baseline cache size.

Do not equate “2.5 GB estimated transient reduction” with permission to add 2.5 GiB of persistent cache. KV preallocation, allocator reservations, graph allocations and fragmentation must be included.

## 6. Experiment 3 — convert headroom into fewer transfers

**Question:** Does cap-enabled extra capacity reduce bytes and end-to-end latency?

Keep a cap-only control so the effect of the cap is separated from the effect of the cache.

| Arm | Score cap | Hot cache | Role |
|---|---|---:|---|
| A | 0 | 14336 MiB | Current recipe control |
| B | Passing cap from Task 2 | 14336 MiB | Cap-only control |
| C | Same cap | 15360 MiB | First +1 GiB candidate, only if headroom allows |
| D | Same cap | 16384 MiB | Optional second increment, only after C passes |

If +1 GiB cannot fit within the validated envelope, test +512 MiB (`14848`) instead. If no safe increment exists, stop this track. Never reduce supported context, concurrency or change memory fraction to make a capacity arm pass silently.

- [ ] Before each increment, document the memory budget: observed baseline peak, cap peak, proposed persistent bytes and retained headroom. Preserve at least the baseline configuration's observed headroom at the same workload; uncertain sampled headroom requires a conservative increment and actual long-context validation.
- [ ] Run the existing smoke with the candidate `hot_gpu_mb` and score cap at the largest supported prompt. Require completion, output parity and no new allocation failures/retries before performance runs.
- [ ] Use `run_arm.sh <unique-arm-name> 30021 KEY=VALUE ...` with `EXPECT_SHA` pinned and `DSV41_RUN_ROOT=/mnt/nvme1/cache-capacity-nvme` explicitly set; otherwise the driver defaults to a different result root. Override only `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB` and `SGLANG_MOE_HOT_GPU_MB` between the relevant arms.
- [ ] Run A/B once to quantify the cap-only effect on the regular corpus. Run B/C in A–B–B–A order using fresh names for each repetition. Screen D only if C is useful and memory remains available; repeat the eventual winner against A.
- [ ] Use `paired.py <control-run-dir> <candidate-run-dir>` for matched outputs and client latency. Supplement its aggregate with request-level decode p50/p95, cold/steady segments and per-step transfer/miss deltas. Use available telemetry; record any missing metric rather than substituting a differently defined one.
- [ ] Report the sample count beside every percentile. The default two-session harness supports an initial comparison, not a reliable request-level p95 verdict. Use repeated and held-out requests for tail validation; if observations are insufficient, mark tail confidence insufficient and do not claim that gate passed. Do not count correlated decode tokens as independent requests.
- [ ] Use `analysis/dsv41-drive/hot-cache-size/compare_arms.py NAME=DIR ...` for compatible smoke outputs (`stages.jsonl`/`responses.jsonl`), not as a replacement for client latency. Its lagged graph timestamps are diagnostic.

**Gate B:** A useful candidate must reduce actual GPU misses/H2D traffic and improve unprofiled decode latency by at least **1 ms/token**, consistently across paired repetitions and beyond observed drift. Require output parity and no repeatable decode-p95 regression above 5%; ambiguous results get another paired repetition, not a success claim. Report TTFT separately and apply the Task 2 tolerance unless the user chooses a different tradeoff.

Report effects per actual extra GiB and per avoided transfer. The historical 2–2.7 misses/GiB slope is a prediction to check, not a passing criterion. If misses fall but bytes/latency do not, inspect accounting and whether another bottleneck absorbed the benefit.

## 7. Experiment 4 — confirm the useful capacity, not merely the largest

- [ ] Retain the best passing capacity from Task 3; do not consume all spare VRAM just because a single smoke succeeded.
- [ ] Repeat the winning configuration's maximum-context case and at least one different representative prompt/seed. Include the existing production concurrency requirement; a single-request smoke does not validate concurrent serving.
- [ ] Compare the winner with the original A recipe on the same held-out session subset, with matched startup and warmup, using the harness's `DSV41_SESSION_INDICES` support where applicable.
- [ ] Check output parity, OOM/retry counts, peak memory, TTFT, cold/steady decode latency and transfer bytes. Confirm the gain is not confined to a fitted route trace.
- [ ] Produce a proposed recipe diff and measured tradeoff table. Do not apply it to production as part of writing results.

**Gate C:** Recommend the winner only if the combined memory, parity and performance criteria hold on the held-out workload. Otherwise retain the baseline and record why. Keep indexer-cap correctness and cache-capacity performance conclusions separate.

## 8. Experiment 5 — attribute the remaining exposed NVMe tail

**Run on:** The selected candidate and one matched baseline, or baseline alone if the capacity track failed. Offline analysis can begin earlier; hardware runs remain serialized.

**Existing analysis to inspect/reuse:**
- `analysis/dsv41-drive/pcie-trace/tail_one.py`, `more.py`.
- `analysis/dsv41-drive/nvme-load/nvme_sampler.py`, `nvme_windows.py`, `nvme_busy_rate.py`.
- `analysis/dsv41-drive/hot-cache-size/compare_arms.py` for host request/completion/publication timing.
- `analysis/dsv41-drive/frontend/frontend_bound.py` for chain grouping and CE completion summaries.

Some scripts hardcode report names, streams or output directories. Inspect and parameterize only what is needed in the experiment checkout; do not execute the old `start_nvme.sh` against its hardcoded `prod-flags` directory.

- [ ] Collect an unprofiled baseline first. Separately capture node-mode kernels/memcopies and diskstats for the same representative requests. Add PCIe RX only if required to distinguish idle link from SM traffic; capture and align the root session using the established harness.
- [ ] Rank layers by **exposed delay after RAM-hit delivery**, with mean and p95 plus miss counts. Historical hotspots L0/L19/L39/L23/L13 are starting points, not assumed current winners.
- [ ] Attribute request-visible → admission → first submission, submission → CQE, CQE → piece publication, and piece-ready → GPU delivery where actual timestamps exist. Record the final row/read and mirror that determine the layer's completion.
- [ ] Keep host monotonic, device and nsys clocks separate until calibrated. In the existing stage schema `pieces[].publish` is a sequence number, not a timestamp; use real row publication clocks or add narrowly scoped diagnostic timestamps if needed.
- [ ] Distinguish total NVMe service time from its **unhidden tail**. Do not add S duration, CW duration and CE copy duration when they overlap. Similarly, CE duration sums are not proof of continuous PCIe occupancy; use interval unions/PCIe samples for utilization.
- [ ] Inspect rows per demand, queue depth, request sizes, per-mirror completion imbalance, and host publication delay. Never infer SSD saturation from `%util` alone.
- [ ] Produce an attribution table naming the largest actionable cause and a conservative saving estimate under explicit scheduling assumptions.

**Gate D:** Proceed to a storage code experiment only if the identified component plausibly exposes at least **2 ms/token** on the representative workload. This is a screening estimate; only an unprofiled A/B establishes a gain. If no cause meets the gate, report that result rather than implementing generic queue/thread tuning.

## 9. Experiment 6 — one targeted storage intervention

Choose exactly one intervention from Task 5's evidence. Define a separate bounded implementation plan with the responsible files, baseline, new flag if needed, correctness tests and rollback before changing production logic.

| Observed dominant cause | Candidate experiment | Required evidence |
|---|---|---|
| Submission gaps with idle mirrors | Reduce admission/submission delay or batch existing demand reads | Earlier useful submissions, unchanged bytes and no starvation |
| One mirror consistently determines last-row completion | Adjust demand assignment using measured mirror behavior | Lower completion tail, same data/parity and no extra speculative traffic |
| CQEs complete but publication is delayed | Reduce owner-thread publication delay | Shorter completion-to-publication interval with lease/generation ordering preserved |
| Many avoidable repeated RAM misses | Revisit admission/capacity using prefill **and** decode traces | Held-out reduction in physical reads, without greater H2D traffic or TTFT cost |
| Storage latency itself dominates at adequate queue depth | Diagnose device/topology/contention before scheduler changes | Device-level evidence tied to exposed layer stalls |

- [ ] Preserve generation validation, source lease lifetime, destination protection, piece visibility and failure cleanup. Do not remove fences or waits solely because they appear in a long-running kernel.
- [ ] For any changed reader/publication path, exercise all-hit, all-miss, mixed, slot-reuse, failed-read and shutdown cases through its existing tests, adding a regression that fails before the specific fix.
- [ ] Measure the single change against the same selected baseline in repeated unprofiled pairs. Keep cap/cache configuration fixed.
- [ ] Require at least **1 ms/token repeatable gain**, parity and no material p95/TTFT regression. Use traces to explain the gain; do not use profiled wall time as the acceptance metric.

Do not reopen generic next-layer prefetch, async hot promotions, per-layer RAM weighting, larger thread pools or row-layout coalescing merely because this experiment fails. Their existing no-go results must be addressed by a new hypothesis.

## 10. Results and completion handoff

Keep raw runs under `/mnt/nvme1/cache-capacity-nvme/` where configurable; existing indexer smokes write under `/mnt/nvme1/indexer-cap/<unique-tag>`. Maintain an index linking every arm to its actual directory. Do not move evidence while processes still write it.

For every arm record:

| Provenance | Configuration | Correctness/resources | Performance/mechanism |
|---|---|---|---|
| SHA, generation, env, corpus, seed, order, output path | score budget, requested/actual cache, context, concurrency, NUMA, mirrors | parity, exit code, peak allocated/reserved/sampled memory, retries, retained headroom | TTFT, client decode latency/p95, cold/steady misses, CE/SM bytes, NVMe reads and exposed tail |

Missing measurements must say **unavailable**, with the reason. Mark each experiment completed, skipped by its gate, or blocked by a named prerequisite. Never populate proposed arm rows with historical measurements.

- [ ] Write a results document alongside this handoff and update the current `DSV41_REFERENCE.md` only after evidence is available.
- [ ] State the chosen score budget/cache capacity, rejected alternatives, observed benefit, confidence and workload limits.
- [ ] Include exact reproduction commands, raw evidence locations, test outcomes, any harness fixes and a concrete proposed recipe diff.
- [ ] Have an independent reviewer check attribution, parity coverage, memory headroom and A/B comparability before recommending rollout.

**First action for the next executor:** inspect the current `cc/indexer-cap` diff and serving baseline, then complete Experiment 1. Do not begin by rewriting the transfer kernels.
