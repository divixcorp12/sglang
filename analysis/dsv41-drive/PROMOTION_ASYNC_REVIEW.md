# Independent review of `PROMOTION_ASYNC.md` (Task 8 design), 2026-09-21

Reviewed: `analysis/dsv41-drive/PROMOTION_ASYNC.md` at `09dcb79564`, against the source at `HEAD` (`git show HEAD:<path>`,
line numbers below are HEAD's). Read-only: nothing under `python/` was touched, no GPU, no drive load. The reviewer is not the
author of the design. Serena was unavailable, so this is grep and reading, not symbolic navigation.

Legend: **CONFIRMED** claim matches the source; **CORRECTED** real but misread or mislocated; **LESS SEVERE** real, and less
than stated; **GAP** the design leaves something unassigned; **ANSWERED** an open question the source settles.
Severity is mine, with the evidence beside it.

## Verdict

No claimed defect is false. One claimed safeguard is nominal (F1). The two hazards the document found by reading are both real
by mechanism and both weaker than presented. The deadlock is real for the *proposed* design and cannot form on today's code. The
round trip is confirmed, and the "M0 saves almost nothing" claim is confirmed. Two facts about capacity change the document's
arithmetic for the measured configuration (F9).

## Findings

### F1. MEDIUM, a claimed existing safeguard is nominal: `retire(..., consumer_complete)`

Section 4 lists, under "What already works (reused, not replaced)", "`retire` (which requires `consumer_complete`)".
`expert_hot_cache.py` `ExpertHotCache.retire` (`:398`) returns False unless the caller passes `consumer_complete=True`. **Every
production caller passes `True` unconditionally:** `:344` (`_reserve`'s victim retire), `:677` (`stage_reassign`), `:686`
(`stage_reassign`'s `reserve`), `:774` (`assign_prefetch`). Nothing anywhere computes whether a consumer has finished. The
parameter is a caller's assertion, and what actually protects a recycled slot today is temporal exclusion: `before_host_use`
synchronises the stream and pauses the service before any promotion runs. An implementer who reads "requires
`consumer_complete`" may assume the lifecycle already enforces consumer retirement and add nothing; M1 needs a real value (the
document's DRAINING rule, sections 2.1 and 7.4, is that value). Suggested fix: say in section 4 that `consumer_complete` is a
caller assertion today.

### F2. LOW, hazard 1 (partial enqueue): real, mislocated, and weaker than stated

- **Mislocated.** The promotion path does not issue six operations in `_submit_operations`' loop. `submit_hot_cache_promotions`
  (`expert_hot_cache.py:809`) calls `submit_expert_row_copy_batch` (`expert_transfer.py:563`), which calls
  `executor.submit_batch(plans, copy_all)` with **one** callback; `copy_all` issues the six copies through
  `ExpertRowCopyRoutes.copy_rows`. A raise partway leaves earlier copies queued and no event recorded, as the document says,
  but inside `copy_all`.
- **The abort does not go where the document says.** `submit_hot_cache_promotions` has its own `except BaseException` that calls
  `abort_promotion` for every promotion (`:841-843`) and re-raises. By the time `_load_reserved_in_chunks` sees the exception
  `promotion_in_flight` is already `None`, so its `submitted` logic (`:498`) never acts. "Covers this only when `submitted` is
  True" is true and beside the point: the abort has already happened without a drain.
- **Consequence is weaker.** The destinations are LOADING, never mapped; after the abort they are FREE. Any later copy into
  them goes through the same single executor stream (`AsyncExpertTransferExecutor.stream`), so it runs after the stale
  kernels; nothing published reads them. What can raise between launches on the EXL3 GPU route is a launch-time CUDA error,
  which leaves the context poisoned anyway. The exception then propagates uncaught (no `try` around `_update_residency` at
  `expert_hot_cache.py:2734`) out of the forward-pass epilogue.
- **A variant the document did not name.** The copy kernels read the plan's device buffers (`_rows64`, `_rows32`) at run time,
  and `FixedRowTransferPlan.set_rows` uploads them on the *current* stream (`expert_transfer.py:139-140`) with no ordering
  against stale kernels already on the executor stream. A stale kernel could read a half-updated plan. The window is
  microseconds and needs the same launch-time failure; I list it for completeness, not as a reason to act.
- **Recommendation.** Keep the "record an event in a `finally`, quarantine until it completes" rule for the new `try_submit`
  (cheap and sound), but do not describe today's code as having a reachable slot-reuse corruption.

### F3. LOW, hazard 2 (full ring raises): real, unreachable on today's EXL3 path

`_acquire_slot` (`expert_transfer.py:388-397`) raises `RuntimeError("expert transfer ticket ring is full")` after scanning all 8
slots: **CONFIRMED**. On today's EXL3 path it cannot happen: every ticket is waited before the next submit, and a whole update
takes one ring slot. Correct as a requirement for an asynchronous submitter. Two corrections to OPEN 4: `expert_prefetch.py:107`
uses the shared `for_device` singleton unless a `copy_stream` or `ready_event` is supplied (the `max_inflight=1` private
executor is the exception, `:108-116`); and `for_device` raises `ValueError` when a caller asks for a different ring size than
the registered one (`expert_transfer.py:250-252`), so an async design cannot simply ask for a larger ring.

### F4. CONFIRMED, real for the proposal only: the section 9.2 deadlock

Mechanism verified: `_submit_operations` does `self.stream.wait_stream(producer_stream)` (`expert_transfer.py:376`), which
orders the executor stream after everything already enqueued on the producer at that call. The observer runs in the `finally` of
`with_forward_pass` (`eplb/expert_distribution.py:198-205`, `_on_forward_pass_end`), i.e. after the forward's graph replay is
enqueued, so an armed `exl3_ram_miss_wait_kernel` can already be in the producer's queue. The cycle needs a promotion source
lease that makes a demand undeliverable, and **no lease exists today**, so it cannot form on current code. The fix (plan
upload on the executor stream, `producer_stream=None`, destination FREE) removes the dependency and is sufficient for the cycle
described. Residual: `set_rows`' host block on `_upload_event.synchronize()` (B6) remains.

### F5. CORRECTED, wrong symbol name (LOW)

`Exl3RamMissTable` (`PROMOTION_ASYNC.md` lines 105 and 849) does not exist. The class is `NativePinnedSlotTable`
(`exl3_ram_miss.py:156`; `before_host_use` `:219`; the `expert_to_slot` property rebuilds an `OrderedDict` when
`host.version()` moves, `:189-199`, which is the behaviour B9 and R8 describe). `Exl3RamMissService.before_host_use` (`:401`)
is correct. An implementer grepping the wrong name finds nothing.

### F6. CORRECTED, "the manager already batches a whole update behind one ticket" is dense-only (LOW)

Section 6.1. `_update_residency` batches promotions across layers only for formats where `stage_reassign` returns a promotion
(dense). For EXL3, `stage_reassign` calls `_load_reserved` inside its slot batch (`:692`) and returns no promotion; each layer
then submits its own ticket per chunk synchronously (`_load_reserved_in_chunks` -> `submit_hot_cache_promotions([promotion])`).
One ticket per update would be new behaviour for EXL3, not an existing one.

### F7. GAP (LOW-MEDIUM), layer-local failure isolation has no precedent in the code

Sections 4 and 10 say a failed promotion quarantines its layer and the process "continues serving what it can", citing
`_drain_device` as the precedent. Today a failure raises out of `_update_residency` (`:2734`, uncaught) into the forward pass;
the sticky-error case additionally leaves `promotion_in_flight` set so later updates are skipped and counted. The precedent
covers "refuse further updates", not "keep serving". The poll step will need its own catch.

### F8. GAP (LOW), B10 has no owner

`ExpertPinnedHostCache._refresh_mapping` (`expert_stream.py:281`) is a pageable H2D that synchronises, called at the end of
`ensure_rows` (`:346`) and from `before_host_use` when the native version moved. It is inventoried as B10 but no milestone
removes it: M0 only changes `ensure_rows`' `.tolist()`. It stays on the M1/M2 path only if `ensure_rows` does. The section 11.1
`set_sync_debug_mode` test would catch it (not verified that the mode flags a pageable H2D; PyTorch behaviour I did not check).

### F9. ANSWERED and CORRECTED, capacity (OPEN 6)

- **Arithmetic CONFIRMED:** 15,019,978,752 / 1,128 = 13,315,584 exactly; 40 x 384 = 15,360; 1,128 / 15,360 = 7.3 %.
- **The measured configuration is not 1,128 slots.** The Task 1 arms' startup line (`task1c-0-new-on-T.log`, divix01) reads
  `slots: 888`, `allocation_bytes: 15,019,978,752`, `scratch_bytes: 3,195,740,160`, `residency_bytes: 11,824,238,592`,
  `prefetch_pull_bytes: 0`. The scratch is 240 rows (3,195,740,160 / 13,315,584), i.e. 40 layers x 6 gather rows; the
  888 resident slots are what a promotion competes for. 1,128 is the eager run's figure. The document mentions 888 as the
  "option-C shape" but computes K against 1,128.
- **Per-layer split, by reading `ExpertHotCacheManager` construction (`:1112-1200`):** candidates are sorted by
  `(-score x bytes, expert_id, layer_id)` and taken greedily until the budget is spent. With no seed
  (`SGLANG_MOE_HOT_SEED` resolved to `''` in the arms' provenance) every score is 1.0, so the order is expert 0 of every
  layer, then expert 1, and so on: **22 or 23 slots per layer** (888 = 22 x 40 + 8, layers 0-7 get 23), uniform. With a seed
  the split follows the seed and is not uniform. **K = 1 costs about 4.5 % and K = 2 about 9.0 % of the resident slots in that
  configuration**, against the document's 7-9 % from a 22-28 range. This is read from the code and consistent with the logged
  total; the per-layer counts themselves were not logged.

### F10. ANSWERED (partly), OPEN 2

The Python eager reader (`UringFileReaderObj`, `uring_file_reader.cpp`) and the native `RowReader` are **separate io_uring
instances**: neither shares a ring or a registered-buffer table with the other (`ring_` is per object; the EXL3 eager path
registers no buffers, `TOPOLOGY.md` section 8.1). They contend only for the drives. One constraint the document should carry:
the general reader enforces creator-thread ownership (`check_owner_`, `uring_file_reader.cpp:586-593`), so any milestone that
moves Python-reader calls off the scheduler thread will throw `"io_uring file reader belongs to thread ..."`. Section 7.1 does
not propose that; it is worth stating.

## Checked, no finding

- **B1-B9** each match the source: `decide_residency_policies` (`expert_residency.py:450-471`, `.cpu()` on the stacked scores);
  `advance_residency_policies` (three `torch._foreach_*` launches); `before_host_use` (`exl3_ram_miss.py:401-408`, synchronise at
  `_pause_depth == 0`, then `pause(2 x timeout + 1 s)`); `set_rows` (`expert_transfer.py:125`) and `_publish_slots` (`expert_hot_cache.py:235`) event
  synchronisation; `wait_for_slot_publication` (`:253`).
- **Why `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` is inert for EXL3:** `stage_reassign` routes spec-only formats to `_load_reserved` and
  returns no promotion; `exl3_reqs._check` refuses the flag (`:55`).
- **The round trip (item 4), CONFIRMED:** `_prepare_promotion` builds `torch.tensor(expert_rows, device=self.device)`;
  `ExpertPinnedHostCache.ensure_rows` (`expert_stream.py:307`) does `source_ids.tolist()` and otherwise reads only
  `source_ids.numel()`, then builds two CPU tensors. **"M0 saves almost no wall time" CONFIRMED:** `_load_reserved_in_chunks`
  opens the outer `host_use`, so `_prepare_promotion`'s is nested and does not resynchronise (`before_host_use` syncs only at
  depth 0). The `torch.tensor(list, device=cuda)` is itself a blocking pageable copy of a few bytes on an idle stream.
- **Section 8.1's mutex table:** every Python-facing `RamTier` method (`has`, `touch`, `assign`, `release`, `mapping`,
  `slot_to_expert`, `lru_order`, `set_hot`) takes `mutex_` (`exl3_ram_miss_host.cpp:1585-1666`, HEAD blob).
- **Section 2.2:** the copy kernels and wrapper never mention `generation` (`git grep` empty on `expert_cache_transfer.cuh` and
  `ops/moe/expert_cache_transfer.py`); `slot_state`/`slot_generations` are consumed outside `expert_hot_cache.py` only by
  `expert_residency_gpu.py`, which EXL3 cannot use (`pinned_tier_ok=not gpu_residency_update`).
- **Section 7.2 step 1:** `set_rows` uploads on the current stream. **Section 7.1:** the observer is synchronous inside the
  forward pass's `finally`. **`_drain_device` quotation** is verbatim.
- **The §18.6 figures** (19.5 / 15.9 / 13.8 / 9.6 ms; 115 ms excess over 112 boundary steps; 1.055 ms/row; 10.16 ms) are in
  `DSV41_REFERENCE.md` at the lines the document cites.

## Not checked

The section 5 state machine against any model; the test specifications in 11.2-11.4 and the arms in 12; the proposed native
API (R1-R8, 8.2), which does not exist; `read_host_rows` and the EXL3 row source internals; the DMA route's internals; whether
the forward stream is the replay stream in every scheduler mode (OPEN 1); the per-layer RAM row counts (141 per layer);
PyTorch's sync-debug behaviour on pageable copies; the §18.6 measurements themselves (I read only that they are recorded).
Nothing here was run.
