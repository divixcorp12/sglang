# DSV4.1 threading and lease overhead: experiment handoff

> **For agentic workers:** Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` when executing this handoff. These are gated experiments. Measure the active path before changing code, preserve the lease protocol, and separate experimental results from a production rollout decision.

**Goal:** Determine whether host placement, retirement cadence, request publication or DIRECT ranking has enough exposed cost to justify optimization.

**Architecture:** Establish a current diagnostic baseline; test thread placement without changing memory placement; optimize hot-bitmap construction while retaining single-writer publication. Advance bounded retirement waits, partial victim selection and lane-parallel validation only when their respective measurements justify them.

**Tech stack:** C++ threads/pthreads, Linux affinity and IRQ telemetry, io_uring, CUDA graph/JIT kernels, PyTorch GPU residency updates, existing lease/piece-stream/copy-engine tests, node-mode Nsight Systems and unprofiled serving arms.

**Spec:** The user's request to explore C++ threading and lease-protocol overhead, followed by a request to save the explicit experiment recommendations in a handoff. This is a companion to [the cache-capacity/NVMe handoff](2026-09-26-dsv41-cache-capacity-nvme-experiments-handoff.md). The cache-capacity track retains priority for decode speed; this track investigates narrower mechanisms.

**Status:** Document only. No experiment below has been run or implemented as part of this handoff.

## 1. Source baseline and existing evidence

This file is written in `/home/dimitri/data/divix/sglang-nvfp4`, whose checkout is older than the reviewed implementation. The preceding read-only review used `/home/dimitri/data/divix/sglang-nvfp4-worktrees/direct-two-phase` at `c08f5484c9933f70b43d7f68081125099642f273`. That worktree is actively advancing: resolve the current intended experiment commit and read its applicable instructions before execution. Do not assume this document's line numbers or prior HEAD are still current.

Repository-relative sources:

| Source | Relevant code/evidence |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` | `RowReader::set_pack/read/reap/submit/drain`; `RamThread::run`; `CopyEngine::run`; `retire_leases`, `release_copied` and SmAck processing |
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` | Post hot bitmap/request publication; W1 validation; S admission/final validation; CW small-tensor reads and SmAck |
| `python/sglang/kernels/ops/moe/exl3_ram_miss.py` | `start_thread(cpu_core=-1, spin_us=5000)` and `enable_copy_engine(spin_us=5000)` APIs |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | Production callers of those APIs and the captured transfer chain |
| `python/sglang/srt/layers/moe/expert_residency_gpu.py` | `_apply`, `_rank_victims`, `_init_insert_direct`, gather/commit |
| `python/sglang/srt/layers/moe/expert_residency.py` | Exact integer `residency_rank_keys` semantics |
| `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh` | Already-fused warp-ballot destination filtering |
| `DSV41_REFERENCE.md` §25.4, §27.5, §27.14–27.15 | NUMA interference, scheduling, link utilization and frontend no-go |
| `analysis/dsv41-drive/LEASE_PROTOCOL.md` | Ownership/publication/failure contracts, especially current CE/SmAck requirements |
| `docs/superpowers/plans/2026-09-26-dsv41-ram-miss-frontend.md` | Executed frontend sizing and resident-first study |

### Facts the experiments must not misinterpret

- `ROW_IMAGES=1` forces the native packing-worker count to zero, avoids bounce packing and creates no packing pool. `PACK_WORKERS=8` is ineffective for this active path. Verify the runtime trace reports zero; do not sweep pack-worker counts.
- The production RAM-service call supplies no `cpu_core` override, so it inherits affinity. With no pack pool, excluding packing CPUs excludes nothing. CE also inherits affinity. The reader's `owner_core` pin scaffold is test-only; it is not a production environment knob.
- Service defaults to spinning for 5 ms after recent work, then sleeping 50 µs. CE polls while copies or acknowledgements are outstanding; when idle it can wait on a notified condition variable for up to 1 ms. That timeout is not a mandatory 1 ms delay per request.
- Reader progress is checked at the top of its loop at a nominal 200 µs cadence. Normal blocking occurs in `io_uring_submit_and_wait(..., 1)`; `io_uring_wait_cqe` is used in failure/exception draining. The callback cannot run while those waits block.
- I/O runs outside the tier mutex. Ordinary retirement, reservations and CE release take that mutex; contention is plausible but unmeasured.
- Ordinary retirement skips CE-owned lanes. The CE thread handles them independently, and with SM small copies enabled release requires DMA completion **and** generation-matched SmAck.
- `pump_demand()` retires ordinary acknowledgements before serving another demand. A long callback gap alone does not establish blocked useful work.
- Existing scheduling evidence found CE run-queue wait 0.01%, service 0.03%. Do not begin from an assumed starvation problem.
- Historical H2D measurements were similar from either NUMA node when unloaded; CPU memory load on the GPU's node reduced bandwidth by 27–43%. Moving thread affinity does not move pinned slab pages.
- S's normal lane admission is already parallel. W1 and S's final post-completion validation retain serial work.
- Historical all-hit lease arming around 17 µs/layer is not a measurement of current CE/row-image/piece-stream overhead. The frontend study estimated only 0.324/0.411 ms/step for W1-zero/reset variants; those variants were not run.
- The reported 0.35–0.4 ms/step for DIRECT gather/commit is not a measurement of the whole score-update/ranking boundary. Measure that boundary separately and count its actual invocation frequency.

## 2. Global run discipline

- Use a private divix01 checkout of an agreed current baseline, preserving unrelated edits. A/B arms must share an integrated SHA, with the experimental behavior explicitly selected. No production checkout changes or interruption are authorized by writing this document.
- Keep model, corpus, seed, warmup, cache capacity, actual slot allocation, pinned-page split, mirrors, EAGER, CE, SM small copies, graph gathering and DIRECT stage 2 constant within a comparison. Do not combine cache-size work with a threading/kernel arm.
- Preserve `HOT_UPDATE_DECODE_FORWARDS=1`. Neither ranking cadence nor eviction policy changes belong in these implementation-only experiments.
- Use the existing serving harness's generation registration, clean-tree and live-env checks. Record the effective thread masks and experiment control values; reject silently ignored controls. Add experimental wiring only where the needed control is absent, rather than inventing an env flag and assuming it works.
- Put results under `/mnt/nvme1/threading-lease/`, with unique directories. Set `DSV41_RUN_ROOT` explicitly for `benchmarks/dsv41_baseline/run_arm.sh`; otherwise its default root is different. Inspect each exit code and completed response, not merely the driver's final line.
- Acquire `rowimg-disk.lock` before `cc-gpu.lock` when both are needed. Audit which driver already acquires them; do not acquire the same lock twice. Use an agreed production maintenance window and never break an existing lock.
- Use unprofiled A–B–B–A serving runs for acceptance and separate short diagnostic runs for attribution. Use node-mode nsys when CE is active. Do not infer serving latency from the sum of profiled kernel durations.
- Record request count, cold/steady windows, clocks, CPU/memory load and disk load. A default two-session run is a screening comparison, not adequate proof of request-p95 improvement. Report tail confidence as insufficient when there are too few observations; correlated tokens are not independent requests.
- Treat host monotonic clocks, GPU timers and nsys timestamps as different domains until calibrated. Label observed-ack-to-release latency separately from true device-ack-to-release latency.
- No broad scheduler-priority changes, global SMT disable, generic thread-pool expansion or unrelated service termination. IRQ changes affect the host: save original state, obtain applicable host-change authorization at execution, and restore it after the isolated arm.

**Proposed acceptance rules (freeze before running arms):**

- Speed: at least 1 ms/token repeatable unprofiled improvement beyond drift.
- Tail: at least 5% reduction in request-level decode p95, with at least 100 requests per arm across multiple independent sessions and a session-clustered 95% confidence interval excluding zero improvement. Do not bootstrap individual correlated tokens. If this evidence is unavailable, mark the tail gate inconclusive.
- Efficiency: at least 20% less combined service-plus-CE CPU time per decoded token across repeated pairs, without shifting an offsetting cost into kernel/IRQ work. Allow at most 1% regression in median client decode latency and 3% in request p95; use the same tail-evidence requirement before declaring this gate passed.
- Speed/tail candidates also must not exceed those 1% median/3% p95 regression limits on the representative held-out workload. Report TTFT separately and do not accept a repeatable regression above 5% without an explicit workload-tradeoff decision.

These are experiment criteria, not pre-existing product SLAs. Record any agreed changes before collecting acceptance results. Kernel microbenchmark gains alone do not qualify for rollout.

## 3. Experiment 0 — baseline topology and overhead attribution

**Deliverable:** One topology manifest and timing table that determine which later experiments run.

- [ ] Record CPU/core/socket/NUMA/SMT topology, GPU and NVMe PCIe locality, relevant TID names/masks, task migrations and per-IRQ effective masks/counter deltas. Example read-only entry points:

```bash
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
ps -L -p "$SERVER_PID" -o tid,comm,psr
cat /proc/interrupts
```

Read `/proc/$SERVER_PID/task/<tid>/status` and `schedstat` for identified TIDs; `/sys/devices/system/cpu/cpu<id>/topology/thread_siblings_list` for siblings; and `/proc/irq/<irq>/effective_affinity_list` for relevant NVMe vectors. Resolve IDs from actual topology rather than copying CPU numbers from an old run. A TID's last CPU (`psr`) is not its affinity mask.

- [ ] Confirm packing workers are zero and inventory the service, CE, prefill helper, watchdog, scheduler and other active pools. Separate prefill behavior from steady decode.
- [ ] Measure service demand-observation delay, CE queue-to-issue delay, first/last CQE, callback gaps, ordinary retirement backlog, CE retirement backlog, admission/reuse deferrals and tier-mutex acquisition/hold time. Add sampled or bounded diagnostic counters if unavailable; quantify instrumentation overhead.
- [ ] Measure Post entry-to-demand-publication and total Post duration. Read `analysis/dsv41-drive/MOE_SERVICE_TRACE_PLAN.md` before adding timer stamps; preserve publication ordering.
- [ ] Measure the complete captured residency boundary, then identify score/key construction, sort and output work. Do not attribute every generic torch sort in the graph to residency. Benchmark with production shapes and meaningful replayed state transitions.
- [ ] Record both region cost and whole-graph latency. A large wait duration is not automatically removable work.

**Screening:** If placement shows no interference/locality problem, keep its A/B small. If retirement backlog causes no useful-work delay, defer Experiment C. If a kernel region has negligible total exposed cost, retain only a microbenchmark result and do not develop a larger rewrite.

## 4. Experiment A — thread placement, pinned pages unchanged

**Files if harness wiring is needed:** `python/sglang/srt/layers/moe/exl3_ram_miss.py`, `python/sglang/kernels/ops/moe/exl3_ram_miss.py`, and the native thread entry points in `exl3_ram_miss_host.cpp`. Prefer a narrowly scoped experimental launch harness where possible.

| Arm | Placement |
|---|---|
| A0 | Current inherited affinity |
| A1 | Service and CE on separate quiet physical cores on the current socket |
| A2 | Keep CE fixed; move only service to a quiet physical core on the other socket |
| A3 | Keep the best service placement; move only CE between sockets |

- [ ] Choose allowed CPUs from topology. Explicit service CPUs 64–71 are rejected in the reviewed API; do not bypass its reserved-core checks. Verify masks after startup and for every fresh server.
- [ ] Keep SMT sibling occupancy, scheduler placement, background load and IRQ masks matched. Do not pin all process threads to the two experiment cores. Do not change the pinned NUMA allocation split.
- [ ] Record queue/observation latency, CQE tails, piece publication, CPU/migrations/run-queue wait, H2D rate and client latency.
- [ ] Compare A0/A1, then A1/A2, then the best fixed service placement with the two CE placements. Avoid a full factorial sweep without evidence.

**Decision:** Retain only repeatable useful placement gains. A different last-observed CPU or fewer migrations alone is not evidence of better serving performance.

## 5. Experiment B — IRQ interference, conditional on topology evidence

- [ ] Run only if relevant completion IRQs overlap service/CE physical cores or siblings and diagnostics implicate them. Preserve application pins from Experiment A.
- [ ] A: observed current IRQ placement. B: move only relevant completion IRQs onto other suitable cores on the NVMe socket, preserving housekeeping capacity.
- [ ] Save requested/effective masks and irqbalance policy, verify actual counter distribution, and restore the prior configuration after the arm. Managed IRQs may not obey an ordinary affinity write; stop rather than claiming an uncontrolled comparison succeeded.
- [ ] Compare CQE p95/p99, IRQ/softirq CPU time, queue depth, service/CE timing and token tails. Hold all storage settings and bytes fixed.

**Decision:** No gain means no IRQ change. Do not generalize this experiment into disabling SMT or moving every interrupt.

## 6. Experiment C — bounded reader waits, only for demonstrated retirement stalls

**Entry condition:** Experiment 0 shows delayed ordinary lease retirement causes admission, ring-reuse or pause delays. A long blocking read with no retirement opportunity does not meet the condition.

**Source responsibility:** `RowReader::read/reap/submit/drain` and the caller's progress callback in `exl3_ram_miss_host.cpp`. This requires code; no existing production timeout flag was verified.

| Arm | Normal reader wait |
|---|---|
| C0 | Current blocking `io_uring_submit_and_wait(..., 1)` |
| C1 | Timeout-bounded wait, initially 200 µs |
| C2 | Timeout-bounded wait, 1 ms |

- [ ] First construct a delayed-completion/ordinary-ack case demonstrating the backlog and its release, using the existing reader and lease fault harness. Verify the test actually exercises non-CE retirement.
- [ ] Design the timeout-capable submission/completion integration against the installed liburing/kernel APIs. Timeout returns to progress; it does not fail the read, cancel it, duplicate submission or make a destination reusable. Keep prepared-vs-in-kernel accounting intact.
- [ ] Preserve failure/exception draining until all kernel references are quiescent. Do not turn `drain()` into an early-return timeout or introduce a new busy-poll/reaper thread in this experiment.
- [ ] Test completion-before-timeout, repeated timeouts, partial submissions, EINTR, short/error reads, stop/pause, stale completion and destination reuse. Apply the existing relevant fault tests and add a failing regression for the demonstrated delay.
- [ ] Compare actual callback gaps, ordinary ack backlog, CE CopyDone/SmAck backlog, deferrals, mutex time, syscalls/wakeups, CPU and unprofiled latency. A requested 200 µs timeout is not a hard real-time guarantee.

**Decision:** Keep only if useful blocking falls without lifecycle regressions or excessive wakeup cost. If CE retirement dominates, this mechanism is the wrong target; do not merge it merely to make callback intervals prettier.

**Optional separate efficiency arm:** If diagnostics show costly idle spinning, compare service `spin_us=5000` versus `500`, keeping CE at 5000 and placement fixed. Measure idle-to-demand latency, CPU and tail behavior. This changes idle polling only, not blocking I/O. Test CE spin separately only if queue-to-issue evidence implicates idle wakeup; its outstanding-work polling remains active regardless of the idle budget.

## 7. Experiment D — hot bitmap before request publication

**Why this differs from W1 trimming:** The bitmap is constructed before LaneRequest and demand-head publication. It can delay storage/DMA starting, rather than merely overlap their completion.

**Source responsibility:** `exl3_ram_miss_post_kernel` in `exl3_ram_miss.cuh`; the ops launcher only if arguments/layout require it. Confirm the active GPU-hot path supplies `hot_page`.

| Arm | Bitmap construction | Host publication |
|---|---|---|
| D0 | Thread 0 scans all hot slots for every bitmap byte | Existing thread-0 writes/releases |
| D1 | Threads compute independent bytes into shared memory | Barrier, then existing thread-0 writes/releases |
| D2 | Threads scan disjoint slots and atomic-OR a shared word bitmap | Shared zeroing/barrier, construction/barrier, then thread-0 writes/releases |

- [ ] Keep D1 and D2 independent candidates. D1 reduces serial work without changing total comparison count; D2 reduces construction from approximately `ceil(E/8)*H` comparisons to a slot scan plus bitmap initialization/output.
- [ ] Restructure the current thread-zero guard carefully: every thread required by a barrier must reach it. Preserve uniform failure/empty exits and existing request resets. Do not move demand publication ahead of required data.
- [ ] Keep pinned-host publication single-writer in the first experiment. Thread zero's release store does not by itself prove ordering for stores performed by other threads. No fence removal belongs in this change.
- [ ] Compare exact bitmap bytes against the baseline for zero/full occupancy, empty slots, non-byte-aligned expert counts, boundary IDs, changing membership and ring wrap. Exercise the protocol's existing failure/arming cases.
- [ ] Benchmark production capacities and the candidate larger-cache capacities from the companion handoff. Measure Post entry→publication, Post duration, first useful host action/transfer, graph and client latency.

**Decision:** A faster construction microbenchmark must translate to earlier useful service or graph completion. Historical Post/arming measurements cannot substitute for this current result. If the whole measured Post budget is below the adoption threshold, report the small result without escalating scope.

## 8. Experiment E — W1 lane-parallel validation, lowest priority

**Source responsibility:** `lease_hit_wait_body` and normal/stream W1 wrappers in `exl3_ram_miss.cuh`. S admission is already lane-parallel; do not reimplement it.

- [ ] Benchmark D0/current W1 separately before combining any changes. First variant assigns one warp lane per request lane for metadata read, fence, ready reread and judgment; thread zero combines results and performs stable compaction.
- [ ] Each participating reader preserves its payload-read→system-fence→ready-reread sequence. W1 currently batches payload reads before one fence per pass; do not assume one fence per lane was the old bottleneck or that parallelism reduces total PCIe bytes.
- [ ] Preserve bounds checks before reads; generation/tag/expert/slot identity; READY/LOADING/COPYING semantics; `claimed=2` for CE; absolute deadline and wait budget; complete rollback on invalid identity; final single `go_1` commit.
- [ ] Reset `go_1`, `violated`, `claimed[]` and `host_rows_1` before any reader proceeds. Streaming W1 also resets `go_2`, `stream_count`, `stream_abort`, on every replay. Use uniform barriers/early exits.
- [ ] Exercise zero through maximum supported lane count, delayed publication, mixed ownership, stale generation, count overflow, malformed metadata, ring wrap and failure followed by healthy replay.
- [ ] Measure one-pass and repeated-pass cost plus Post→F and serving latency. Leave S's final completion reread unchanged initially; it must not be replaced by cached pre-completion masks.

**Decision:** Existing frontend sizing places this below the main capacity work. Advance beyond a microbenchmark only if fresh current evidence contradicts the low-payoff estimate.

## 9. Experiment F — exact partial victim selection

**Source responsibility:** `_apply` and `_rank_victims` in `expert_residency_gpu.py`, with `residency_rank_keys` semantics from `expert_residency.py`. A custom selector, if justified, needs a separately scoped ops/JIT implementation plan; it is not required for the first comparison.

| Arm | Selection |
|---|---|
| F0 | Current key construction, full `torch.sort`, shortlist extraction |
| F1 | Same keys, ordered smallest-k selection with the required shortlist width |
| F2 | Only after F1/evidence justify it: fused per-layer key construction and selection |

- [ ] Benchmark the complete captured score/ranking boundary at actual shapes and invocation frequency, not only an isolated sort. Keep state updates meaningful across replay; restore/replay identical initial states and route histories in paired arms.
- [ ] Require identical ordered **valid** shortlist entries and validity masks. Normalize invalid padding when comparing unused entries; invalid ties must not become usable destinations.
- [ ] Preserve free-slot priority, routed-resident penalty, integer rank keys and score/expert tie-breaking. Preserve the shortlist/capacity guarantee and per-gather protection of currently routed resident slots.
- [ ] Test free/full cache, equal/zero scores, every resident recently routed, different layer capacities, padding, minimum supported capacity and repeated insertions. Verify scores, mappings, valid victims, truncation counters and resulting bytes, not only final generated text.
- [ ] Do not change `HOT_UPDATE_DECODE_FORWARDS`, shortlist width, minimum capacity, insertion policy or numerical score updates to make a faster arm. The existing fused gather already uses ballot compaction; do not count that as a missing optimization.
- [ ] Compare boundary time, full graph, serving latency, GPU misses and physical transfer bytes. Confirm no CPU readback/synchronization was introduced.

**Decision:** A policy-preserving win is useful. A selector that changes victims is a different cache-policy experiment and must not be presented as equivalent acceleration. Stop if full-boundary cost is too small to matter.

## 10. Correctness suite and invariants shared by code experiments

Read the existing tests and select those affected by the change; add a regression for each new mechanism. Relevant files include:

- `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`: request arming, publication and wrap.
- `test/manual/dsv41/test_exl3_piece_stream_cuda.py`: W1 budget/LOADING exit, replay resets, final masks, block abort, repeated replay and reader/slot lifetimes.
- `test/manual/dsv41/test_exl3_copy_engine_cuda.py`: CE ownership, read/commit mask equality, all-thread SmAck ordering, failure cleanup and slot reuse.
- Existing registered residency/DIRECT tests discovered by references to `_rank_victims`, `gather_destinations` and `insertion_truncated` in the current checkout.

For all code arms: no premature source reuse, lost/double lease signals, stale generation acceptance, unpublished data reads, new fatal/timeout failures or leaked outstanding work. Keep S's completion ordering and failure handling. With SM small copies enabled, CW must finish every thread's reads before SmAck; even an armed failed request must acknowledge its SM phase as specified. Keep exact read-mask/commit-mask validation and DMA-plus-SmAck retirement.

Use the existing delayed-thread/slot-overwrite tests where relevant. Do not remove acknowledgements, release/acquire operations or visibility fences as a shortcut. A timeout never permits freeing memory still referenced by DMA or NVMe.

## 11. Execution order and final evidence

Recommended order: **0 → A → D**, then **B** if IRQ evidence supports it, **C** if retirement stalls are demonstrated, **F** if ranking is material, and **E** last. A skipped gate is a valid outcome. Do not start all hardware arms concurrently or combine improvements before attribution.

For each experiment record:

| Provenance/configuration | Mechanism | Outcome |
|---|---|---|
| SHA, generation, corpus/seed, arm order, effective flags, thread/page/IRQ placement, cache slots, clocks/load | Region timings, relevant backlog/deferrals, copy/read bytes, mutex/CPU/wakeup costs | Exit/test/parity results, client latency with sample counts, tail confidence, adopted/rejected/deferred and reason |

- [ ] Separate diagnostic-build results from uninstrumented performance results. Quantify perturbation before using a diagnostic number for a gate.
- [ ] Save raw logs/traces, analysis commands and schemas under the result root. A missing metric is **unavailable**, not zero.
- [ ] Write a result document next to this handoff; link exact source commits and evidence paths. Update `DSV41_REFERENCE.md` only after measurements exist.
- [ ] Obtain an independent review of memory ordering, lifecycle behavior and A/B comparability for any candidate change.
- [ ] Present a concrete proposed serving/configuration diff for any winning arm. Do not modify `base_env()` or production as part of merely reporting results.

**First action:** resolve the current serving baseline and collect Experiment 0's manifest/timing table. The eight configured packing workers are not the first tuning target.
