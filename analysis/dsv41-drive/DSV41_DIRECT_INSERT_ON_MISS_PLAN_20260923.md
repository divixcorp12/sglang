# DSV4.1 EXL3 Direct Insert-on-Miss Implementation Plan

> **For implementers:** Execute the checked tasks in order. Review the native
> request protocol and the GPU cache ownership change independently before a
> served benchmark. Work on `codex/nvfp4-expert-stream-main`; do not silently
> change the default serving recipe.

**Goal:** Reuse Qwen's DIRECT insert-on-miss policy for DSV4.1 so a routed
EXL3 miss is copied once, from its NVMe-backed pinned slot directly into a GPU
resident slot, inside the captured decode graph.

**Architecture:** Keep the existing graph route planner, EXL3 `io_uring`
service, pinned-slot translation, row-copy kernel, and Qwen victim selection.
Add an EXL3-specific source-lifetime and native-hot-set bridge. The GPU owns
residency during decode; the native service receives a graph-published snapshot
before it can evict pinned rows. Eager prefill refreshes a host snapshot at its
existing stream synchronization boundary. EXL3 prefill promotions from dense
expert-ID host rows are disabled in this first implementation; prefill still
updates scores and directly serves its routed misses.

**Design basis:** `MOE_EXPERT_TRANSFER.md` (Qwen DIRECT),
`analysis/dsv41-drive/MOE_SERVICE_TRACE_PLAN.md` (EXL3 graph path),
`analysis/dsv41-drive/LEASE_PROTOCOL.md` (pinned-source lifetime), and the
current DSV4.1 recipe in `benchmarks/dsv41_baseline/arm_env.py`.
The divix01 checkpoint's `text_config.n_routed_experts` is **384** (verified
read-only from its `config.json` on 2026-09-23).
For the pinned sidecar's cross-device publication, check the aligned-access
and system-fence rules in NVIDIA's [mapped-memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
and [C++ language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
against the existing `exl3_ram_miss.cuh` request protocol. Do not assume a
general GPU atomic on mapped host memory is host-atomic.

## Why this is worth testing

The current 14 GiB hot-cache allocation contains 888 resident rows and 240
scratch rows across 40 layers (`artifacts/preforward-readiness-20260923/
run-20260923-155330/server.log`). DIRECT returns the 240 scratch rows,
2.98 GiB, to the same allocation for potential residency. The actual gain
depends on per-layer limits and route locality. Each EXL3 row is 13,315,584
bytes, so SCRATCH insertion would add a large device-to-device copy per
inserted miss. DIRECT is the first port target.

## Invariants and scope

1. A GPU `expert_to_slot[e] >= 0` is published only after the same stream has
   issued a successful copy of expert `e` into that slot. A routed GPU hit is
   never chosen as a victim for its own forward.
2. Every pinned source slot read by a graph copy remains leased until the
   copy's acknowledgement. If a request fails, its GPU destination is not
   made resident and the existing fatal path prevents a served result.
3. Before the native service selects a pinned victim for a demand, its hot set
   reflects the latest GPU residency for that layer, plus that request's
   protected routes. Eager host uses get the same protection from a snapshot
   taken at their existing synchronization point.
4. No new `.cpu()`, event wait, Python callback, or device allocation enters
   a decode graph replay. The request's pinned-memory publication is ordered
   before `demand_head` exactly as today's request payload is.
5. Initially support the DSV4.1 shape: EXL3 native RAM-miss backend, single
   graph stream and decode batch size 1, RAM-miss leases on, single-phase copy,
   no expert advisory/prefetch, no side pull, no expert doorbell. Refuse other
   combinations explicitly. Keep the dense Qwen path unchanged.

## Files and interfaces

| Area | Files | Contract |
|---|---|---|
| Feature gates and allocation | `python/sglang/srt/layers/moe/expert_format.py`, `expert_hot_cache.py`, `expert_residency_gpu.py`, `offload_presets.py` | Allow pinned-tier GPU residency only for EXL3 + DIRECT + native leases; preserve the per-layer `capacity >= 2 * gather_rows` floor and the pinned-tier slot limit. |
| Graph copy and commit | `python/sglang/srt/layers/moe/expert_stream.py`, `exl3_ram_miss.py`, `expert_residency_gpu.py` | The native backend exposes the copied count (`go_count` in the supported lease mode); DIRECT commits only those rows after the copy/ack sequence. |
| Native hot publication | `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`, `exl3_ram_miss_host.cpp`, `python/sglang/kernels/ops/moe/exl3_ram_miss.py` | The post kernel copies a GPU hot bitmap into a small pinned sidecar ring, fences, then publishes the demand record and head; the service applies the matching bitmap under its tier lock before eviction. |
| Eager host use | `python/sglang/srt/layers/moe/exl3_ram_miss.py`, `exl3_expert_format.py`, `expert_residency_gpu.py` | Refresh one pinned CPU snapshot of GPU `slot_to_expert` before the existing outer stream synchronization; native `set_hot` and `is_pinned` use it until the next eager handoff. |
| Verification | `test/registered/unit/layers/moe/test_expert_residency_gpu.py`, `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`, `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`, `test_exl3_ram_miss_tier.py`, `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py` | Match row bytes, outputs, mapping, lifetime, and failure behavior under capture/replay. |

## Task 1 — Gate the mode and lock its allocation contract

- [ ] Add a configuration test showing that today's EXL3 GPU-update rejection
  remains for stage OFF/SCRATCH and doorbell, while EXL3 + DIRECT + leases is
  accepted. Include explicit refusals for leases off, two-phase on, expert
  advisory/prefetch on, side pull on, and a non-native pinned backend. The
  rejection must happen before CUDA graph capture.
- [ ] Change `require_graph_gather_support`/`ExpertHotCacheManager.from_model`
  only enough to admit this EXL3 combination. Because `from_model` constructs
  the updater before `_attach_formats()` installs `Exl3RamMissRowBackend`, run
  the final backend check immediately after attachment, then call
  `check_miss_plans`. Do not weaken dense-format validation.
- [ ] Keep `gather_rows = batch_size * top_k` for route width but allocate
  `scratch_rows = 0` for DIRECT. If the inclusive pinned-tier limit cannot
  provide `2 * gather_rows` resident slots for any layer, fail startup with
  that layer and both capacities, rather than hitting the current floor-pass
  assertion. Assert the startup log reports zero scratch bytes and a resident
  count within the requested GPU budget.
- [ ] Run the focused configuration/allocation tests before and after the
  change. Commit this gate separately; it must not enable DIRECT by default.

## Task 2 — Publish GPU residency to the native service without readback

- [ ] Add a separate pinned `hot_page` ring with 16 records, matching the
  demand ring. Each record has a 32-bit sequence, 32-bit expert count, then
  `ceil(num_experts / 8)` bitmap bytes, rounded to a 64-byte stride. For the
  verified 384-expert checkpoint this is 16 × 64 = 1,024 bytes. Derive the
  stride at startup and validate it in the host/device wrappers; do not
  change the existing 128-byte request record or baseline page. Hold the
  sidecar through native-service shutdown/quarantine. Add an ABI parity test
  for Python, device, and host layout constants.
- [ ] Pass the GPU updater's per-layer `slot_to_expert` row and its true
  capacity through `Exl3RamMissRowBackend` to `Exl3RamMissDevice.post`. In
  `exl3_ram_miss_post_kernel`, build the bitmap from valid resident slots and
  write its sidecar slot as `seq=0 → fence → payload → fence → release(seq)`.
  Then publish the ordinary demand record and finally `demand_head`, using
  their existing release order. Do not add a separate kernel or host callback.
- [ ] Have native `read_record` copy the sidecar bitmap only when both its
  sequence and the demand-record sequence match the expected request before
  and after the copy. For an armed DIRECT demand, replace `tier.hot` from that
  bitmap under `mutex_` in `pump_demand()`, **before `defers(request)`**;
  `defers` itself calls `census_locked` to choose whether the request has a
  victim. Reapply the snapshot on a deferred request's retry. Retain the
  current request's `protect`/`need` exclusion in both the census and
  `serve()`. Since advisories are rejected in Task 1, no same-layer eviction
  occurs between this demand and its next post. A missing,
  lapped, or malformed bitmap must fail closed, not silently use a stale set.
  Keep the ordinary `set_hot` path for stage OFF.
- [ ] Test an already-RAM-resident GPU miss (`need_count == 0`): lease mode
  must arm it, service must observe the bitmap, and the pinned source must
  remain protected until GPU ack. Test ring wrap, lapped record, stale bitmap,
  and eviction pressure. A protected resident cannot be the victim.

## Task 3 — Make eager prefill use the GPU-owned hot set

- [ ] Keep CPU hot-list protection during startup seeding:
  `ExpertHotCacheManager.from_model` calls `cache.reassign` before it creates
  `GpuResidencyUpdater`, so `NativePinnedSlotTable.before_host_use` may run
  before any GPU snapshot exists. After the updater is constructed, seed its
  pinned host snapshot from the now-initialized GPU slot bank and switch the
  service/format to GPU-owned protection before graph capture or serving.
  Test both sides of this handoff; never make `_push_hot` expect an updater
  during startup seeding.
- [ ] Allocate one reusable pinned host snapshot for the updater's
  `slot_to_expert` bank before capture. At the outermost
  `Exl3RamMissService.before_host_use`, enqueue its device-to-host copy on the
  current stream **before** the synchronization already done there; after the
  synchronization and native-thread pause, turn it into per-layer hot ID lists.
  Nested host uses reuse that snapshot. This adds no decode replay sync.
- [ ] In DIRECT mode, replace `NativePinnedSlotTable._push_hot` and the
  `Exl3ExpertFormat.is_pinned` closure's read of the stale Python
  `hot.slot_to_expert` with those snapshot lists. Push them to native
  `set_hot` before any eager admission. Ensure a prefill that follows the
  final decode insertion protects that insertion even when no next same-layer
  demand occurred.
- [ ] Skip the dense expert-ID `_promote` copy for EXL3 prefill boundaries;
  still advance scores, clear the boundary, and rank DIRECT victims for the
  next decode. Large eager prefills continue to read current GPU hot hits and
  fetch misses through the pinned tier. Test this explicitly across
  decode → eager prefill → decode, including a pinned-tier eviction attempt.
- [ ] Do not let a CPU `stage_reassign` or an old residency listener overwrite
  the GPU-owned map. EXL3 attachment should register its fail-stop check, but
  use the bitmap/snapshot protocol instead of the CPU residency listener in
  this mode. Keep the listener for stage OFF.

## Task 4 — Bind DIRECT commit to delivered bytes

- [ ] Reuse Qwen's `gather_destinations` victim filter and same-stream
  `commit_gather` order. Make the EXL3 backend expose the successful copy's
  device count and the **post-ack `keep` word** to `commit_gather`; in the
  supported single-phase lease mode, these are `go_count` and `keep`.
  The active commit mask is `(lane < go_count) & (keep > 0)`, never merely
  `_graph_miss_count`. The ack kernel can detect a slot-generation violation
  after `go_count` became nonzero and clear `keep`; that path must publish no
  GPU resident. A zero/refused delivery likewise publishes none.
- [ ] Check that the graph's row-copy node is followed by lease ack, then
  residency commit, then the MoE read, all on the same stream. Preserve
  the route planner's hit slots while choosing victims. Require
  `insertion_truncated == 0` on successfully delivered gathers; a positive
  count there is a victim-shortlist correctness failure. Do not count a
  fail-stopped delivery with `go_count == 0` as shortlist truncation;
  report it through the native delivery/fatal counters instead.
- [ ] Extend the captured-graph test with miss → next-replay hit → eviction →
  refetch, duplicate routed IDs, all-hit, all-miss, a hit on the top-ranked
  victim, capture reset, and forced read timeout. Compare actual EXL3 row
  bytes and MoE output to the fresh source; assert no response is served after
  a fatal delivery. Exercise both the fused and generic route planners.

## Task 5 — Measure the net effect on divix01

- [ ] Use a clean, commit-pinned divix01 worktree and the existing
  `benchmarks/dsv41_baseline/run_arm.sh`/`paired.py` eight-session harness.
  Keep model, corpus, 50 GiB pinned tier, 14 GiB GPU budget, `uring_direct`,
  CUDA graph shape, CPU placement, and clock/compile gates fixed. Run three
  arms: **A** current leases-off recipe; **B** current recipe with
  `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=1`; **C** B plus
  `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`,
  `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`, and
  `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1`. Leave prefetch/advisory off.
- [ ] Record actual server environment and startup cache sizes, graph replay
  count, resident slots, hot-hit and pinned-hit rates, NVMe rows/bytes, H2D
  rows/bytes, insertion/eviction/successful-delivery truncation counters,
  native delivery failures, and decode throughput.
  Use B→C to isolate the DIRECT mode from lease overhead; use A→C for the
  practical result against today's launch. Note that C also disables EXL3
  prefill promotions, so report prefill behavior and TTFT separately.
- [ ] Ship the opt-in only if all correctness and fail-stop tests pass, the
  eight-session pairing passes the harness validity gates, C improves net
  served decode over A beyond its measured noise, and NVMe demand or H2D
  bytes do not regress enough to erase the gain. If it loses, keep the mode
  guarded off and use the counters to decide whether native bitmap posting,
  lost prefill promotions, or cache policy caused it. Do not infer benefit
  from the Qwen result or a microbenchmark alone.

## Review focus

- A graph capture or warm-up DIRECT insertion changes cache contents before
  first replay: `reset_after_capture` must rank victims from the resulting
  mapping, and the native hot snapshot must match before the first eviction.
- A demand with no NVMe read still needs a lease and an acknowledged copy.
- A request timeout or cancellation must neither publish a GPU resident nor
  allow a later copy to overwrite a reused GPU slot. Include an ack-time
  generation violation after a nonzero `go_count`.
- Eager prefill must not consult the stale Python `slot_to_expert` list.
- A model whose expert count/bitmap stride exceeds the allocated sidecar,
  insufficient per-layer capacity, or an unsupported service mode must fail
  at startup with a precise error.

No GPU measurement is part of writing this plan. The benchmark work is on
divix01; the local checkout is used only for code and review.
