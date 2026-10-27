# Layer-major prefill expert residency, stage 1: lend late-layer hot-cache slots

## Goal
Layer-major prefill keeps stage 0's TTFT win from per-layer expert residency (16k 103.2 -> 56.7 s, 32k 207.6 -> 93.5 s
vs chunked) without shrinking the VRAM expert hot cache, so decode stays at full-hot-cache speed.

## Acceptance criteria
- Decode ms/token with lending on is within run-to-run noise (about ±3 ms/token) of the full hot cache
  (68.7 ms/token, 65.6% routed hot hits, 418 VRAM-miss rows/verify; hot-smaller-ab-20261017c arm A) in a served A/B.
- The decode A/B includes a 32k-prompt-then-decode run, so decode right after a lend is measured.
- TTFT at 16k and 32k with lending on is within about 2 s of stage 0's (56.7 s, 93.5 s).
- Token-0 parity with chunked prefill at 16k and 32k (base recipe and DSpark).
- After every pass (success, chunk failure, restore failure), every lent hot-cache slot either holds its
  original expert's bytes with an unchanged slot map, or is logically evicted (mapping -1, slot FREE); no path leaves
  a mapped slot with wrong bytes.
- GPU test: lend -> overwrite -> reload leaves the lent slots byte-identical to the originals.

## Decisions
- Owner approved starting stage 1 on 2026-10-18, after the hot-smaller A/B showed stage 0's 360-slot hot-cache
  reduction costs +9.7 ms/token (+14%) decode.
- Off by default: `SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND` (EnvBool False). With
  `SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS` > 0 it lends late-layer hot slots instead of allocating a buffer.
- Lend in place: residency rows are the late layers' existing hot-cache rows (expert_hot_cache.py:226-236); only
  their bytes change, for one forward. VMM granule lending is rejected (re-map can fail on OOM after graph capture,
  leaving mapped-but-unbacked slots; hot_cache_retire.check_retire_allowed refuses capacity changes after capture).
- Donors are layers 21-39 (late_layer_start..39), which the layer-major pass loop never runs; their MoE runs only in
  finish_pass, after restore(). 493 slots (6.3 GiB) are lendable against ~360 needed.
- Reload from the pinned RAM tier, which is inclusive of the hot cache for EXL3 (exl3_expert_format.py:185), so it
  is H2D only: ~355 rows x 13.3 MB = 4.7 GB at 12.3 GB/s, ~0.4 s, paid in TTFT before the tail.
- A per-layer reload overlapped into forward_late_tail is deferred (invasive).
- Decode bar (owner, 2026-10-18): decode must match the full hot cache within run-to-run noise; the reload cost is
  paid in TTFT, not decode.
- Lent slots are taken from layer 39 downward in slot order; no residency-score ordering (it only matters on the
  restore-failure eviction path).

## Rules
- KV stays pinned at 381,184 tokens; HiCache stays on.
- Nothing new is on by default.
- The slot map, slot states, generations, policy scores, RAM-miss hot lists and lease/victim tables are never
  modified by lending; only restore-failure eviction may change them.
- Launch refuses lending when SGLANG prefetch pull mode is not off or GPU drip slots > 0 (they could write or read
  late slots mid-pass).

## Non-goals
- CPU experts in prefill (measured not worth it in stage 0).
- Lending from finished early layers (493 >= 360 makes it unnecessary).
- Overlapping the reload with the tail.

## Open questions

## Design

### Lend plan
An ordered list of (layer, slot) taken from layer 39 downward, built at begin(). adopt takes the next unused entry
lazily; only written rows are reloaded. If lendable capacity < ROWS, the buffer is smaller and later experts stream
per chunk as stage 0 does when full.

### Lifecycle
- begin() (driver.py:27): quiesce with manager.finish_promotions() and gpu_residency.flush(), then stream sync (the
  flush is the pending decode boundary that would otherwise promote into lent slots at layer 0's first gather);
  snapshot lent rows' slot_to_expert and slot_generations from the device bank; build the lend plan.
- make_resident(L) (driver.py:29): as stage 0, PrefillResidency.activate; early layers overwrite lent rows.
- adopt (prefill_residency.py:86) copies staged misses into lent rows grouped per donor layer; scatter (:103) copies
  resident rows back into staging via a new pointer-table Triton scatter (one launch per tensor name, not per donor).
- restore() (driver.py:35, also the failure path :47-48): deactivate; reload every written lent row with its
  original expert through the donor streamer's pinned tier (pinned_host_cache.gather_rows, expert_stream.py:698)
  into staging, then scatter into the slots; count any NVMe fallback; verify the snapshot; sync before finish_pass.
- end() (driver.py:52): log rows lent, reloaded, pinned hits/misses, reload ms.

### Consistency
No decode, verify, draft MoE or insert-on-miss can run during the pass: layer-major requires max-running-requests 1
(gate.py:38), the pass is one blocking forward (tp_worker.py:682-684), DSpark KV injection runs after it returns,
and the draft does no MoE in prefill. A fail-stop check (expert_hot_cache.py:2044/:2067) raises if a graph gather or
promotion runs while a lend is active; the begin/restore snapshot catches any generation change.

### Failure handling
- Chunk or tail raises: the driver always calls restore() (restored flag prevents double calls), then release_pass.
- Restore fails (CUDA error, NVMe fallback error) or the snapshot mismatches: logically evict the affected rows in
  the device bank (mapping -1, slot FREE, as hot_cache_retire._evict_tail, :231) and refresh the service hot lists;
  log and count. The cache runs short until boundary promotions refill.
- Lending allocates nothing and the reload reuses the eager staging buffer, so no new OOM path.

### Plug-in points
- prefill_residency.py: PrefillResidency lent-row backend (list of (cache, slot), adopt/scatter via the pointer
  kernel, reload_written(), snapshot()/verify()); Exl3LayerMajorResidency.begin/restore/end (:144-166).
- expert_hot_cache.py: ExpertHotCacheManager.quiesce_for_prefill_lend(); the fail-stop check.
- worker_entry._residency (:61) chooses the backend. Adapter and driver ordering unchanged.

## Testing
- CPU fakes: lend plan, adopt/scatter round trip, reload + verify, mismatch -> evict, failure path.
- GPU: pointer-scatter kernel; lend -> overwrite -> reload byte-identical.
- Served: token-0 parity 16k/32k; TTFT pair vs stage 0; decode A/B vs -prod including a 32k-prompt-then-decode run.
