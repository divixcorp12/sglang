# Non-blocking expert promotions: design (Task 8)

Status: design only. No code was written or run for this document, the GPU was not
used, and no test was executed. Everything under "Tests" and "Measurements" is a
specification for someone else to run. It was written against the `dsv41` working tree on
2026-09-21. Another session is editing the native host service (`exl3_ram_miss_host.cpp`
and `ops/moe/exl3_ram_miss.py` are modified and uncommitted), so line numbers drift;
every citation gives a symbol so it can be found again.

Plan reference: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, "Task 8"
(this document's charter), "Task 5" (the lease contract it builds on), "Global
constraints". Companion document: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (Task 5),
written by another agent and uncommitted at the time of writing. Where this design needs
something from it, section 9 states it as a requirement rather than an assumption.

Legend:

- **[E]** exists in the tree today; the citation names the symbol.
- **[P]** proposed by this document; it does not exist.
- **[OPEN n]** something I could not determine; section 14 lists what must be true.
- **[DECIDE n]** a choice I made that the owner may overturn; section 14 lists them.

Files cited (all under `python/sglang/`):

| Short name | Path |
|---|---|
| `hot_cache.py` | `srt/layers/moe/expert_hot_cache.py` |
| `stream.py` | `srt/layers/moe/expert_stream.py` |
| `transfer.py` | `srt/layers/moe/expert_transfer.py` |
| `residency.py` | `srt/layers/moe/expert_residency.py` |
| `srt_ram_miss.py` | `srt/layers/moe/exl3_ram_miss.py` |
| `ops_ram_miss.py` | `kernels/ops/moe/exl3_ram_miss.py` |
| `host.cpp` | `kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` |
| `device.cuh` | `kernels/jit/csrc/moe/exl3_ram_miss.cuh` |
| `exl3_reqs.py` | `srt/arg_groups/expert_stream_requirements_exl3.py` |

---

## 0. Summary, and what this is and is not for

1. **The problem is destroy-then-copy.** Today a promotion reuses the slot indices it
   just retired. Under any asynchronous publish, the evicted experts become misses
   immediately while their replacements are still loading. The plan's item 2 requires
   the opposite: old mappings stay readable until the new bytes have landed and the old
   consumers have retired. Resolving that one fact drives everything else: it needs
   spare destination slots, a new "draining" state, and a source lease on the pinned RAM
   row for the life of the copy (section 2).
2. **The avoidable round trip is real and small.** The residency decision yields Python
   ints. `_prepare_promotion` turns that list into a CUDA tensor; `ensure_rows` then
   calls `.tolist()` on it (a blocking device-to-host copy) to rebuild a list, and then
   builds CPU tensors from that list. list -> CUDA tensor -> list -> CPU tensor, with a
   hidden device sync, for data that never left the host (section 3). It is milestone M0.
3. **A lot already works.** The dedicated transfer stream, the 8-event ring, non-blocking
   completion queries, stale-ticket safety, generation-qualified slot tickets, and above
   all the failed-copy quarantine in `_load_reserved_in_chunks` are correct and are
   reused, not replaced (section 4).
4. **The throughput case is small, and this document does not pretend otherwise.**
   `DSV41_REFERENCE.md` section 18.6 measured a mean boundary excess of about 115 ms
   (about 3.6 ms/token, about 1% of throughput), one 2.1 s burst on one session, and
   ruled on 2026-09-19 that an async promotion path is not worth taking for the gain.
   Task 8 is required for a different reason: the plan forbids describing the pipeline
   as fully asynchronous while promotions block. This design exists so that claim can
   later be made honestly or withheld honestly. Its measurable claims are safety and
   boundary tail, not tokens per second.
5. **Scope.** Promotions only. The eager-gather `host_use` pauses (prefill
   `gather_rows`, seeding, direct calls) are a separate broad pause that this design
   leaves in place and that Task 5's leases must eventually replace (section 9.3).

Milestones (each has its own gate, section 12):

| | What | Depends on |
|---|---|---|
| M0 | Preserve CPU expert ids through admission (item 1). CPU-testable. | nothing |
| M1 | Spare slots, draining state, ticketed async copy and publication, for rows already resident in the pinned RAM tier | Task 5 host-lease API |
| M2 | Asynchronous RAM admission through the native service at promotion priority (no NVMe read on the scheduler thread; no pause) | M1 |
| M3 | Asynchronous decision readback, backpressure, tuning, and the gate measurement | M2 |

Until M2's gate passes, `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` stays refused for EXL3
(`exl3_reqs._check`), and the new mode is a separate knob (section 13).

---

## 1. What runs today, and where it blocks

### 1.1 The path

On every forward, the expert-distribution recorder calls its registered observer
(`ExpertHotCacheManager.on_expert_distribution`, registered in `model_runner.py`
via `register_forward_observer`). At a residency boundary (a large prefill, or every
`SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` decode forwards) it calls `_update_residency`
[E].

For EXL3 the whole promotion then runs synchronously on the scheduler thread:

```text
on_expert_distribution
  _update_residency                                        hot_cache.py
    advance_residency_policies      (3 fused launches)     residency.py
    decide_residency_policies       torch.stack(scores).cpu()   BLOCKING D2H
    per layer: stage_reassign                              hot_cache.py
      retire evicted READY slots (map -> -1)
      reserve promoted experts into the SAME slot indices
      _load_reserved  (EXL3: no dense source -> has_spec_only_tensors)
        _load_reserved_in_chunks                           chunk = evictable_rows() tickets
          pinned_cache.host_use()                          -> NativePinnedSlotTable.before_host_use
            current_stream().synchronize()                 BLOCKING host sync
            Exl3RamMissHost.pause(2*timeout+1 s)           BLOCKING, all layers, waits for the
                                                           thread to finish its current request
          _prepare_promotion
            ensure_rows  -> streamer.read_host_rows        BLOCKING NVMe reads, scheduler thread
            FixedRowTransferPlan.set_rows                  may block on _upload_event.synchronize()
          submit_hot_cache_promotions -> executor.submit
          executor.wait(ticket, current_stream)
          current_stream.synchronize()                     BLOCKING host sync, every chunk
          complete_promotion -> publish_ready
      _publish_slots (pinned upload + device index_copy_)  may block on _slot_upload_event.synchronize()
      wait_for_slot_publication                            BLOCKING event synchronize
```

Why `SGLANG_MOE_HOT_ASYNC_PROMOTIONS=1` did nothing for EXL3 [E]: `stage_reassign`
returns a staged promotion to defer only when the streamer has six tensors and every one
of them has a dense source. EXL3's six streamed tensors have none
(`ExpertStreamer.has_spec_only_tensors` is true), so the path is `_load_reserved`, which returns
nothing to defer. `_update_residency`'s `async_promotions` branch therefore never sees a
promotion. `exl3_reqs._check` now refuses the flag rather than ignoring it, and the
paired arms in `DSV41_REFERENCE.md` section 18.6 (`casync` = `c32` within 0.3%) confirm
it was inert. "Existing generic async flags are not evidence of EXL3 support" is a
statement about exactly this.

`_load_reserved` has three callers: `stage_reassign` (boundary promotions, this
document), `reassign` (startup seeding, not serving, stays synchronous), and
`assign_prefetch` (the dense-NVFP4 speculative VRAM prefetch; out of scope).

### 1.2 The blocking inventory

Every site the promotion path can block. "Normal path" means a boundary with no fault.

| # | Site (symbol) | What blocks | Who waits | Measured |
|---|---|---|---|---|
| B1 | `decide_residency_policies`: `.cpu()` on the stacked scores | D2H on the current stream | scheduler thread (inside the forward) | 9.6 ms CPU incl. decision, section 18.6 py-spy |
| B2 | `Exl3RamMissService.before_host_use`: `current_stream().synchronize()` | drains the forward stream | scheduler thread | not isolated |
| B3 | same: `Exl3RamMissHost.pause` (`RamThread::pause` in `host.cpp`) | sleep-polls at 20 us until the single service thread is between two requests; pauses every layer at once; waits behind any demand read in flight | scheduler thread; and every demand for the length of the pause | not isolated |
| B4 | `ExpertPinnedHostCache.ensure_rows` -> `read_host_rows` | synchronous io_uring NVMe reads | scheduler thread | 19.5 ms `_prepare_promotion`, of which io_uring 15.9 ms, section 18.6 |
| B5 | `_load_reserved_in_chunks`: `current_stream.synchronize()` per chunk | waits for the copies | scheduler thread | 13.8 ms, section 18.6 |
| B6 | `FixedRowTransferPlan.set_rows`: `_upload_event.synchronize()` | previous plan upload not yet run | scheduler thread | not isolated |
| B7 | `ExpertHotCache._publish_slots`: `_slot_upload_event.synchronize()` | previous slot upload not yet run | scheduler thread | not isolated |
| B8 | `wait_for_slot_publication` (end of `_load_reserved_in_chunks`, `reassign`) | event synchronize | scheduler thread | not isolated |
| B9 | `ExpertPinnedHostCache.evictable_rows`, `expert_to_slot` property | O(residents) Python per chunk; rebuilds an `OrderedDict` from `host.mapping` and `host.lru_order` when the C++ version moved | scheduler thread | not isolated |
| B10 | `_refresh_mapping` (end of `ensure_rows`; and from `before_host_use` when the native version moved) | `torch.tensor(list, device=cuda)`: a pageable H2D that synchronizes | scheduler thread | not isolated. **No milestone edits it; it leaves the promotion path when the promotion stops calling `ensure_rows` and `host_use`** (M1 for RAM-resident rows, M2 for admitted rows), and stays on the eager path (section 9.3). M0 does not remove it (review F8). |

Only B4 and B5 have measured cost; the rest are enumerated from reading, not timed.

The measured total, from `DSV41_REFERENCE.md` section 18.6's py-spy attribution on the
burst session: `_update_residency` 45 ms per step averaged over 77 decode steps, of which
19.5 ms `_prepare_promotion`, 13.8 ms `synchronize`, 9.6 ms `decide_residency_policies`.
The typical boundary is cheaper: the burst-frequency run measured about 115 ms excess
over 112 boundary steps (mean 417 ms against 359 ms), about 3.6 ms/token.

---

## 2. The core problem: destroy-then-copy (D1)

`ExpertHotCache.stage_reassign` [E]:

1. computes `existing` = READY experts, `promoted` = desired - existing, `evicted` =
   existing - desired;
2. `retire`s every evicted READY slot (`slot_to_expert = -1`, state FREE);
3. takes `free_slots`, which now include those just retired;
4. `reserve`s the promoted experts into them, bumping each slot's generation.

So the promoted expert is written into the same physical slot the evicted expert
occupied. The order is unavoidable in place: the slot cannot hold both.

Consequence under any asynchronous publish. Step 2 is published immediately when
`publish=self.async_promotions` (`_update_residency` passes it): the device
`expert_to_slot` sends the evicted experts to -1. From that moment until the copy
completes and a later forward publishes READY, every route to an evicted expert is a
miss (a RAM or NVMe gather), and the replacement is not yet servable either. The cache
temporarily holds fewer hot experts than its capacity. This is a self-inflicted miss
window whose length is the copy plus the wait for a later forward.

Item 2 of Task 8 forbids exactly this: "Old mappings stay readable until new payload
copies complete and old consumers retire; account for extra capacity or defer admission
when no safe replacement exists."

Today's synchronous mode does not show the window only because the whole thing happens
between two forwards (the copies are ordered behind the serving stream by
`stream.wait_stream(producer_stream)` in `AsyncExpertTransferExecutor._submit_operations`
and the host waits for them). It is correct by temporal exclusion, not by ownership; it
stops being correct the moment the host stops waiting.

### 2.1 The resolution

A promotion must never write into a slot that any published mapping can reach. That
needs a destination that is free while the victim stays mapped, so:

1. **Spare slots.** Each layer keeps `K` slots that are FREE by construction between
   boundaries (section 6). Promotions write only into FREE slots.
2. **Old stays readable.** The victim keeps its mapping until the new rows are copied.
3. **Atomic swap at publication.** One publication step, on the serving stream, maps the
   new experts and unmaps the victims. The victims' slots become **DRAINING**, not FREE:
   a replay queued before the publication may still be reading them.
4. **Drain, then recycle.** A DRAINING slot becomes FREE only when the serving stream has
   passed the publication (section 7.4); it is then the next wave's spare.

Where no FREE slot exists, the promotion is **deferred**, never forced (section 6.3).

### 2.2 A field that is maintained and never consulted

`FixedRowTransferPlan` uploads a `generations` row to the device
(`rows64[3]`, set in `set_rows` from the tickets' `generation`) but the copy kernel never
reads it: no occurrence of "generation" in `expert_cache_transfer.cuh` or
`ops/moe/expert_cache_transfer.py`. Generation qualification exists on the host only
(`HotCacheSlotTicket`, `ExpertHotCache._ticket_matches`). This is worse than no field: a
reader who sees a generation travelling to the device will assume the device enforces it.
It does not. The design below either makes the device consult it or documents it as
host-only; section 7.5 chooses the latter and keeps the field out of any safety argument.

Likewise `slot_state` and `slot_generations` are published to the device
(`_publish_slots`) and are read only by the GPU-residency updater
(`expert_residency_gpu.py`), not by any graph gather or copy kernel [E, by grep of the
non-test tree]. The graph consumes only `expert_to_slot`.

---

## 3. Item 1: the CPU -> GPU -> CPU round trip

The residency decision already produces Python integers: `ResidencyDecision.desired_experts`
(from `ExpertResidencyPolicy._decide_from_host`), consumed by `stage_reassign` as
`expert_ids`. `reserve()` and `HotCacheSlotTicket.expert_id` hold Python ints. The ids
never leave the host.

Then, in `ExpertHotCache._prepare_promotion` [E]:

```python
expert_rows = [ticket.expert_id for ticket in tickets]                 # Python ints
...
pinned_cache.ensure_rows(torch.tensor(expert_rows, device=self.device))  # list -> CUDA tensor
```

`torch.tensor(list, device=cuda)` is a pageable host-to-device copy (a synchronizing
one). Then `ExpertPinnedHostCache.ensure_rows` [E]:

```python
requested = list(dict.fromkeys(int(value) for value in source_ids.tolist()))   # CUDA tensor -> list
...
source_ids_cpu = torch.tensor(list(final_slots.values()), dtype=torch.long)    # list -> CPU tensor
slots_cpu = torch.tensor(list(final_slots), dtype=torch.long)
self.streamer.read_host_rows(source_ids_cpu, {...}, slots_cpu)
```

`.tolist()` on a CUDA tensor is a blocking device-to-host copy that waits for the current
stream. The result is used only to build an insertion-ordered unique list; the CUDA
tensor is otherwise unused (`ensure_rows` reads `source_ids.numel()` and nothing else
from it). Path: **list -> CUDA tensor -> list -> CPU tensor**, with a hidden device sync,
for data that never needed to leave the host.

Where the sync hides: inside `host_use()`, whose entry already ran
`current_stream().synchronize()`. In today's design the stream is idle at that point, so
the round trip's *wall-time* cost is small (microseconds of transfer plus a sync that
returns at once). That does not make it harmless. In M1 and later the promotion path
must not synchronize at all, and an `ensure_rows` that takes a CUDA tensor and calls
`.tolist()` on it would reintroduce a device sync inside a path whose whole point is to
have none. It is the kind of dependency that a "no synchronization" proof (section 11.1)
would have to find; better to remove it now.

`ExpertPinnedHostCache.gather_rows` legitimately receives CUDA ids (they are real routed
experts from the device) and its `lookup` / `.tolist()` / `.item()` calls are the demand
path, not the promotion path. Do not change those.

### 3.1 M0: the change

- Give `ensure_rows` a host-ids entry: it accepts a `Sequence[int]` (or a CPU integer
  tensor) and does no CUDA operation. Keep the existing tensor entry, delegating to the
  new one after its own `.tolist()`, so `gather_rows` and other callers are unchanged.
- `_prepare_promotion` passes `expert_rows` as a list.
- In `ensure_rows`, build `source_ids_cpu` / `slots_cpu` without an intermediate CUDA
  tensor (they already are CPU tensors; only the input changes).
- Also in `_prepare_promotion`: `cached_rows` is read from `pinned_cache._expert_to_slot`
  after `ensure_rows`. That is host-only already; no change in M0.

`ExpertPinnedHostCache` supports a CPU-device tier ("a CPU `device` keeps the tier on the
host with unregistered slabs, so it runs without a GPU", its class docstring [E]), so M0
is unit-testable with no GPU (section 11).

M0 does not change any byte, any slot, or any ordering. Its gate is byte-for-byte
identical admissions and copies, and a CUDA-API trace showing one fewer
`cudaMemcpy`-family call and one fewer stream synchronization inside `ensure_rows` per
promotion chunk. The measured wall-time gain is expected to be near zero in the current
synchronous design; its value is that M1 and M2 need it.

---

## 4. What already works (reused, not replaced)

Say this plainly, because a design that rebuilt these would be worse than the code:

- **A dedicated stream and a bounded event ring.** `AsyncExpertTransferExecutor` owns one
  stream and `max_inflight = 8` events; `for_device` returns a per-device singleton.
- **Non-blocking completion queries.** `is_complete` / `has_completed` use
  `event.query()`. `has_completed` treats a ticket whose ring slot was reused as
  complete, "a ring slot is reused only after its event completed".
- **Stale-ticket safety.** `ExpertTransferTicket` carries `(slot, sequence)`;
  `_event_for` raises on a stale ticket.
- **A plan claimed while in flight.** `FixedRowTransferPlan._claim` / `_release` /
  `set_rows` ("cannot rewrite a transfer plan while it is in flight").
- **Generation-qualified slot tickets** with FREE / RESERVED / LOADING / READY:
  `HotCacheSlotTicket`, `_ticket_matches`, `reserve`, `begin_loading`, `publish_ready`,
  `cancel`, `retire`. **Correction (review finding F1): `retire`'s `consumer_complete`
  parameter is a caller assertion, not a check.** `retire` returns False unless the caller
  passes `consumer_complete=True`, but every production caller passes `True`
  unconditionally (`_reserve`'s victim retire, `stage_reassign` twice, `assign_prefetch`),
  and nothing anywhere computes whether a consumer has finished. What actually protects a
  recycled slot today is **temporal exclusion**: `before_host_use` synchronizes the stream
  and pauses the service before any promotion runs, and the synchronous path waits for its
  copies. The generation-qualified tickets protect against *stale tickets* (a host-side
  bookkeeping error), not against a *live reader*. So the lifecycle has no notion of "the
  last consumer has retired"; DRAINING with a real `retire_event` (sections 2.1 and 7.4) is
  the first place that value would be computed. This strengthens the case for leases: the
  parameter's name promises an ownership check the code does not perform.
- **`stage_reassign` refuses to run while a promotion is in flight**
  (`promotion_in_flight`), and `_update_residency` counts such layers in
  `deferred_residency_updates`.
- **A manager-level async skeleton for dense formats.** `_inflight_promotions`,
  `_publish_completed_promotions(wait=False)` polled from `on_expert_distribution`,
  `finish_promotions`. This is the shape M1 generalizes.
- **A cancellable, low-priority read class in the native service.** `RamTier::serve(...,
  advisory=true)` gives up when a demand is pending, a pause or stop is requested, has at
  most one row of I/O outstanding, and publishes only rows that completed whole
  (`host.cpp`). A promotion-admission class (section 8) has this shape.
- **A priority vocabulary.** `TransferPriority.EXACT_DEMAND` /
  `BACKGROUND_PROMOTION` and `ExpertResidencyPolicy.schedule_transfers` [E]. Today
  `_update_residency` discards the result (it exists for its metrics); the classes are
  there for M1 to use.
- **An async-shaped read interface.** `ExpertRowSource.submit` returns a `ReadTicket`
  with `done()` / `wait()` (`expert_row_source.py`); the EXL3 sources use
  `SynchronousSubmit`, which reads immediately. The interface is not the obstacle.
- **The correct failure rule, in Python, today.** In `_load_reserved_in_chunks`:

  ```text
  except BaseException:
      if promotion is not None and self.promotion_in_flight is promotion:
          if not submitted or self._drain_device():
              self.abort_promotion(promotion)
  ```

  and `_drain_device` returns False when `torch.cuda.synchronize` raises, with the
  docstring "If the device cannot drain (a sticky CUDA error), the promotion stays in
  flight with its slots LOADING. `stage_reassign` then refuses further updates, so no
  later reservation can reuse a slot a copy may still write."

  **This is the precedent for what Task 5 must do in C++.** Never recycle storage whose
  last reader cannot be shown to have finished; quarantine it and refuse new work.
  `LEASE_PROTOCOL.md` section 14 states the same rule for the native side. The codebase
  already knows the pattern; this design reuses it as the failure semantics for every state
  in section 5. **Limit of the precedent (review finding F7):** it covers "refuse further
  updates" (`stage_reassign` raises while `promotion_in_flight` is set), not "keep serving".
  Today a failed promotion raises uncaught out of `_update_residency` into the forward-pass
  epilogue. Layer-local isolation, where one layer's failure leaves the others promoting, is
  new behaviour: the poll step needs its own catch and its own per-layer quarantine flag.

### 4.1 Where the existing code falls short of its own rule

Two places, both by reading, neither run:

- **A partial enqueue is not followed by an event (D2), hygiene rather than a live bug.**
  The promotion path submits **one** callback: `submit_hot_cache_promotions` ->
  `submit_expert_row_copy_batch` -> `executor.submit_batch(plans, copy_all)`, where
  `copy_all` issues the six copies through `ExpertRowCopyRoutes.copy_rows`. If a launch
  raises partway, earlier kernels are queued, no completion event is recorded, and
  `_submit_operations` releases the plan claims and re-raises. `submit_hot_cache_promotions`
  has its own `except BaseException` that calls `abort_promotion` for every promotion, so
  by the time `_load_reserved_in_chunks` sees the exception `promotion_in_flight` is already
  `None` and its `submitted` logic never acts: the abort has happened without a drain.
  **The consequence is weak today** (review finding F2): the destinations are LOADING and
  never mapped, a later write into them goes through the same single executor stream and so
  runs after the stale kernels, and what can raise between launches on the GPU route is a
  launch-time CUDA error, after which the context is poisoned and the exception propagates
  uncaught anyway. I do not claim reachable slot-reuse corruption in today's code. The rule
  is still worth adopting for the new `try_submit`, where destinations and leases outlive the
  call: record an event in a `finally` and quarantine the wave until it completes. A related
  variant: `FixedRowTransferPlan.set_rows` uploads the plan on the *current* stream with no
  ordering against stale kernels already on the executor stream, so a stale kernel could read
  a half-updated plan; the window is microseconds and needs the same failure. Section 7.2's
  plan upload on the executor stream removes it as a side effect.
- **A full ring raises (D3).** `_acquire_slot` raises `RuntimeError("expert transfer
  ticket ring is full")` after scanning all 8 slots. For a synchronous caller that is a
  can't-happen. For an asynchronous submitter a full ring is *normal backpressure* and
  must become a deferral (section 6.3), not an exception. Note also that `for_device`
  returns one executor shared by everything on the device, so ring pressure is not
  private to promotions [OPEN 4]. On today's EXL3 path it is unreachable: every ticket is
  waited before the next submit. It is a requirement on the asynchronous design, not a
  present defect.

### 4.2 The timeout hole, as it bears on promotions

`LEASE_PROTOCOL.md` D1: on timeout the wait kernel raises fatal but the gather still runs
against `host_rows`, while the service may be evicting. Promotions do not use that path.
They matter here only because a promotion source lease (section 9) must not make that
hole worse: a leased slot is never evicted, so a promotion cannot be the reason a timed-out
gather reads recycled bytes.

---

## 5. The ticket and its state machine [P]

### 5.1 Objects

- **VRAM slot states**: FREE, RESERVED, LOADING, READY [E] plus **DRAINING** [P]:
  unmapped from the published map, bytes still owned because a replay queued before the
  publication may read them. `HotCacheSlotState` gains one value.
  A DRAINING slot has `slot_to_expert = -1` (so it is not in the published map, not in
  `resident_experts()`, and its expert's RAM row loses `hot` protection, which is correct:
  no GPU consumer of the old VRAM slot reads RAM) but is not FREE, so `reserve` will not
  hand it out. Re-promoting the same expert into a spare while its old slot drains must be
  allowed; `_reserve`'s "nonresident expert" check keys on `slot_to_expert`, so it is.
- **PromotionTicket** [P], host-only, one per (layer, wave). Never reused; a monotone
  `ticket_id` (64-bit) identifies it. Fields: `layer`; `rows[]`, each
  `{expert, dst: HotCacheSlotTicket, victim: HotCacheSlotTicket | None, lease:
  LeaseRef | None, admit_state}`; `copy: ExpertTransferTicket | None`; `retire_event`;
  `state`; `bytes`; `created_boundary`; `deadline_boundary`.
- **LeaseRef** [P, from LEASE_PROTOCOL section 17.1]: `{row, slot, slot_generation,
  lease_id}`.

`ticket_id` is independent of the executor's ring `sequence` (which is 32-bit-safe only
per ring slot) and of `HotCacheSlotTicket.generation`.

### 5.2 States

```text
                 no FREE dst / no lease / cap /                                    (terminal)
                 ring full / plan busy                                          +------------+
   PLANNED ------------------------------------------------------------------->| DEFERRED   |
      | reserve dst slots (FREE -> RESERVED); set_hot;                          +------------+
      | rows in RAM? -- yes ------------------------------+
      | no: submit admit to service (M2)                  |
      v                                                   |
   ADMITTING --admit ok, lease_on_ready--> LEASED <-------+   (RAM-resident rows lease directly)
      |                                      |
      | cancel / TTL / service failed        | enqueue copies on the owned stream, record event
      v                                      v
   CANCELLED                              COPYING --event complete--> COPIED
      (release lease; dst -> FREE)           |                          | release source lease
                                             | cancel: ABANDONED        | wait_event + publish on the
                                             |  (keep everything until  | serving stream
                                             |   the event completes;   v
                                             |   never publish)      PUBLISHED  (victims -> DRAINING)
                                             |                          | retire_event complete
                                             | event error /            v
                                             | enqueue failure       DONE  (victims -> FREE)
                                             v
                                          QUARANTINED  (leases and dst stay; no new tickets on this
                                                        layer; section 10)
```

Each arrow is taken by exactly one actor (the scheduler thread, section 7.1) except the
ADMITTING -> LEASED edge, which the service thread takes under its mutex; the scheduler
thread only observes it.

Rules that make the machine safe (each is tested, section 11):

- **R-A. Nothing a copy can touch is released before its event completes.** In COPYING and
  ABANDONED the destination slots stay LOADING and the source leases stay held. Cancelling
  a ticket that has been enqueued is not a cancellation of the copy; it is a decision not
  to publish.
- **R-B. No publication without a completed event and a serving-stream dependency**
  (section 7.3).
- **R-C. A stale ticket cannot act.** Every transition re-checks `_ticket_matches` for the
  destination and victim, and the lease generation for the source. A stale one is a
  counted no-op, never a state change.
- **R-D. Failure quarantines; it does not recycle** (section 10).
- **R-E. The source lease is released at COPIED, not at PUBLISHED.** Between COPIED and
  PUBLISHED the RAM row is protected by the `hot` flag (the destination slot is LOADING
  and in `slot_to_expert`, pushed to the service with `set_hot`), and after PUBLISHED by
  the same flag through the inclusive hierarchy (`DSV41_REFERENCE.md` section 9.1:
  promotion keeps the RAM copy). The lease covers exactly the interval in which a GPU
  reader exists.

---

## 6. Item 2: tickets that reserve RAM and replacement slots

### 6.1 What a ticket reserves

| Resource | How | Released |
|---|---|---|
| VRAM destination slot | `reserve()` of a FREE slot: RESERVED -> LOADING | at PUBLISHED it becomes READY; at CANCELLED it returns FREE |
| Pinned RAM source row | a lease on `(row, host_slot, slot_generation)`, taken atomically with the row becoming READY (M2), or `acquire_host_lease` on a resident row (M1) | at COPIED |
| Transfer ring entry | one executor ticket per layer-wave; one executor ticket per layer-wave. (Correction, review F6: the manager batches a whole update behind one ticket only for dense formats, where `stage_reassign` returns a promotion. EXL3 submits one ticket per layer per chunk today, so one ticket per layer-wave is new behaviour for EXL3, not existing.) | at COPIED (`release_plans`) |
| Plan buffer | the layer's `FixedRowTransferPlan`, claimed while in flight | at COPIED |
| In-flight byte budget | section 8.3 | at COPIED |

One layer has at most one ticket in flight, because its plan is claimed while in flight
and `stage_reassign` refuses to overlap ([E]; kept). Waves for one layer are therefore
sequential; waves for different layers run concurrently up to the ring and the byte
budget.

### 6.2 Extra capacity: K spare slots per layer, matched by default [DECIDE 1]

The plan says "account for extra capacity or defer admission when no safe replacement
exists". The two are the same decision viewed from two sides: with `K = 0` there is no
safe replacement in a full cache, so every promotion is deferred and the cache never
changes. Some spare is unavoidable.

Design: `ExpertHotCache` keeps physical `capacity` slots (tensors, pointers and
graph-captured addresses unchanged), but the residency policy is built with
`live_capacity = capacity - K` (`ExpertResidencyPolicy(num_experts, cache.capacity, ...)`
in `ExpertHotCacheManager` construction). The policy never asks for more than
`live_capacity` residents, so at least `K` slots are FREE or DRAINING between waves.

Matched capacity, as the gate requires: the VRAM budget (`SGLANG_MOE_HOT_GPU_MB`) is
unchanged. The async arm has `C_total = C` physical slots of which `C - K` are live. The
baseline synchronous arm has `C` live. The async arm therefore serves from a smaller live
cache. **That is a real cost and the design does not hide it**: section 12 reports live
capacity and hit-rate change as first-class results, and an unmatched arm (`C` live plus
`K` spare) is a labelled experiment that isolates the effect of the spare slots from the
effect of the async protocol.

What K costs, in numbers I can derive and numbers I cannot:

- Derived: one hot slot is 13,315,584 B: 15,019,978,752 B / 1,128 slots
  (`DSV41_REFERENCE.md` section 16.2 config table), about 13.3 MB, matching "a 13.3 MB row" in
  section 18.2. 14,336 MB gives 1,128 slots in the eager run and 888 in the option-C shape
  (section 17.6).
- Derived: the model has 40 streamed layers of 384 experts (15,360 rows;
  1,128 / 15,360 = 7.3% as the reference states).
- **1,128 is the eager run's slot count; the measured async configuration has 888.** The
  Task 1 arms' startup log reads `slots: 888`, `allocation_bytes` 15,019,978,752,
  `scratch_bytes` 3,195,740,160 (240 rows = 40 layers x 6 gather rows) and
  `residency_bytes` 11,824,238,592 (review F9). K must be costed against 888.
- **Per-layer split (answers OPEN 6 for the unseeded case).** By reading the manager
  construction, candidates are sorted by `(-score x bytes, expert_id, layer_id)` and taken
  greedily until the budget is spent. With no seed (`SGLANG_MOE_HOT_SEED` resolved to `''`
  in those arms) every score is equal, so the order is expert 0 of every layer, then expert 1,
  and so on: **22 or 23 slots per layer** (888 = 22 x 40 + 8), uniform. With a seed the
  split follows the seed and is **not** uniform. Read from code and consistent with the
  logged total; the per-layer counts themselves were not logged.
- **What K costs there:** `K = 1` per layer takes 40 slots, about 4.5% of 888; `K = 2` takes
  80, about 9.0%. That is not small, which is why the default `K` should be the smallest that
  keeps the deferral rate acceptable, chosen from the per-layer promotion distribution. Against
  a seeded configuration the fraction per layer differs and must be recomputed.
- **Not determined [OPEN 7]:** that distribution. The reference gives whole-model
  promotions per 32-forward window (median 25, max 832 over 48 windows), not per layer.
  The per-layer counters exist (`_counters[phase][layer_id].promotions`); the analysis has
  not been done. If the median layer promotes under one expert per boundary, `K = 1`
  covers the typical window and bursts are served in waves.

Why not a cross-layer spare pool: the slot tensors are per layer
(`ExpertHotCache.tensors`) and the captured graph-gather addresses are per layer, so a
pool would need a global slot indirection in the graph. Out of scope. Why not use the
existing extra rows: `scratch_rows` receive graph-gather misses on every replay and
`reserves_prefetch_pull_row` is the prefetch puller's; neither is idle.

### 6.3 Deferral: no safe replacement means wait, and say why

A promotion that cannot be admitted this boundary is **deferred**, not failed and not
forced. Each reason is a counter (section 12):

| Reason | Condition |
|---|---|
| `no_free_vram_slot` | fewer FREE slots than promotions wanted; the surplus (lowest rank in `decision.promotions`) waits for the next boundary. The policy re-decides each boundary, so nothing is queued. |
| `no_ram_victim_leases` | a RAM row is needed but every candidate is held only because of leases (the tri-state of `LEASE_PROTOCOL.md` section 8: {slot, deferred, none}) |
| `ring_full` | `try_submit` finds no free event slot |
| `byte_cap` | section 8.3 |
| `plan_busy` | the layer's plan is claimed |
| `demand_pending` | the service has a demand or advisory waiting (section 8.2) |
| `eager_host_use` | an eager pause is in progress (section 9.3) |
| `quarantine` | the layer is quarantined |
| `shutdown` | admission closed |

Deferral consumes no resources: nothing is reserved until the whole wave's needs are
known (all-or-nothing per layer-wave; `LEASE_PROTOCOL.md` A2 is the same rule for
demand). Partial admission would let one row hold a slot while waiting on another.

`_update_residency` today skips a layer whose promotion is in flight and bumps
`deferred_residency_updates` [E]. That stays, and is refined to the reasons above.

### 6.4 RAM capacity

Leased RAM rows are unevictable, so an in-flight promotion shrinks the RAM tier's usable
capacity for demand and advisory admissions for as long as it is held. The design bounds
that three ways: leases exist only for rows of a wave that is being enqueued (RAM-resident
rows lease and enqueue in the same step; M2 leases from admission completion until the
observer next enqueues it, bounded by one poll interval); the byte cap bounds the number
of concurrently leased rows; and a promotion lease **must never be the cause of a demand
failure** (section 9, R4). If a demand is deferred because of promotion leases, promotions
in LEASED (not yet enqueued) release their leases and go to DEFERRED; those already
COPYING finish, and their hold time is bounded by copy time (section 8.3).

---

## 7. Item 3: owned stream, off-hot-path completion, safe publication

### 7.1 Who does what

The scheduler thread owns the promotion state machine. `AsyncExpertTransferExecutor` is
not documented as thread-safe (it mutates `_next_slot`, `_tickets`, `_next_sequence`), so
this design does not share it across threads [DECIDE 3]. The native service thread owns
its own metadata and is reached only through the mutex-guarded API (section 9).

`on_expert_distribution` runs inside the forward call (`with_forward_pass` in
`model_runner.py` wraps `_forward_raw` and then invokes the observers in its
`finally`). In overlap mode the forward runs inside `with self.forward_stream_ctx:` (in
`scheduler.py`, the `run_batch` overlap branch), so by reading `torch.cuda.current_stream()`
there is the forward stream. I did not verify that CUDA-graph replays are launched on that
same stream in every mode (non-overlap, spec v2 draft-extend, breakable decode graphs
with eager breaks) [OPEN 1]. Section 7.3's rule depends on it.

Progress is driven by a bounded **poll step** at the end of the observer, the same slot
where the manager already polls completed trace buffers ("A query-only poll gives
completed trace buffers to their writer while every later forward continues without
waiting", `on_expert_distribution`) and already calls
`_publish_completed_promotions(wait=False)` [E]. The poll step is O(in-flight tickets)
calls to `event.query()` and O(1) service calls, all non-blocking. "Outside the serving
hot path" here means: not inside any per-layer or per-token path, not on the gather, and
never blocking; it does run on the scheduler thread once per forward. Its measured cost
is a metric (section 12). An optional dedicated progress thread is possible later but
would put the executor behind a lock and the GIL in contention; it is not proposed
[DECIDE 3].

### 7.2 Enqueue

For a wave in LEASED:

1. Fill the layer's plan on the **executor stream**: the plan upload
   (`_rows64.copy_(host, non_blocking=True)`) must be issued with `self.stream` current,
   not the serving stream, so the copy does not depend on the serving stream at all. Today
   `set_rows` uploads on the *current* stream and `submit(..., producer_stream)` makes the
   executor stream wait for it; that dependency is what section 9's deadlock argument
   forbids once a lease is held.
2. Enqueue the six row copies (`ExpertRowCopyRoutes.copy_rows`) and record the event.
   `producer_stream` is `None`: a destination is a FREE slot no published mapping can reach
   (section 2.1), and the sources are pinned rows whose bytes the CPU finished writing
   before the enqueue (program order plus the service's release/acquire).
3. `try_submit` variant [P]: returns `None` instead of raising when the ring is full
   (D3), records the event in a `finally` (D2).

Hazard check on the destination: the wave's destination is FREE. It is FREE because it was
never mapped, or because it was a DRAINING victim whose `retire_event` has been observed
complete on the host (7.4). No device-side wait on the serving stream is needed, and none
is issued.

Double-buffering: `FixedRowTransferPlan.set_rows` currently blocks on
`_upload_event.synchronize()` if the previous upload is recorded (B6); `_publish_slots`
does the same with `_slot_upload_event` (B7). Both are "the pinned staging buffer may not
be rewritten until its previous upload has run". Replace the block with a **poll**: if
`event.query()` is False the wave/publication is deferred (`plan_busy` / `publish_busy`) to
the next poll. Alternatively two pinned buffers alternate. Either removes the host block;
choosing is an implementation detail [DECIDE 4].

### 7.3 Publication: the safe graph boundary

Publication is one enqueue batch **on the serving stream**, from the scheduler thread, at
the end of the observer (after the forward whose kernels are already queued):

```text
for each COPIED ticket, batched:
    serving_stream.wait_event(ticket.copy_event)          # CUDA-visible dependency
    for row: publish_ready(dst)                            # LOADING -> READY (host)
             unpublish(victim)                             # READY -> DRAINING (host)
    _publish_slots()                                       # pinned upload + index_copy_ + copy_
    record retire_event on serving_stream                  # section 7.4
```

Requirements, each of which is a test (section 11):

- **P1. On the serving stream, never the executor stream.** `_publish_slots` is several
  kernels (`copy_` of the pinned upload, `_mapping_scratch.fill_`, `index_copy_`,
  `expert_to_slot.copy_`). A replay on another stream could interleave with them and see a
  torn map. Stream order between replays makes them atomic with respect to replays on
  that stream.
- **P2. `wait_event` is issued even when the host has seen the event complete.** The host
  poll (`event.query()`) decides *when to enqueue* publication; the `wait_event` makes
  correctness independent of the poll's accuracy. It costs a stream dependency on an
  already-complete event.
- **P3. Never enqueue publication whose event is incomplete.** `wait_event` on an
  incomplete copy would stall the serving stream on the device (and, by section 9.2, could
  close a cycle). The poll gates it.
- **P4. Not during capture.** Assert `not torch.cuda.is_current_stream_capturing()`.
  Publication is eager, between replays; nothing in the captured graph changes. The graph's
  `expert_to_slot` pointer is stable (`ExpertHotCache.data_ptrs`); only its contents change,
  between replays, in stream order.
- **P5. Bounded work per forward.** One publication batch per poll.

"Safe graph boundary" therefore means: a point in the serving stream's enqueue order that
lies between two replays (or eager forwards), reached from the thread that owns that
order, after the copies' event has been observed complete and waited on. On a single serving
stream that is any enqueue point outside capture. It is unsafe only if a replay that reads
the map can be running concurrently with the publication kernels, i.e. if replays were on
a different stream from the publication, or overlapped each other. `LEASE_PROTOCOL.md`
[OPEN 9] notes the guard against concurrent graphs is not found in the tree. This design
adds one dependence on the same assumption (A1, section 14) and asserts it at attach.

### 7.4 Old consumers retire: the DRAINING rule

After publication, the victims' old mapping is gone from the published map, but replays
enqueued *before* the publication may not have run yet; they read the old map and the old
slot bytes. Because the serving stream executes in order, every such replay has finished
by the time the publication kernels run. So:

- Record `retire_event` on the serving stream immediately after the publication batch.
- The host polls `retire_event.query()`. When True, every consumer of the old mapping has
  retired (the stream passed the publication), and the victim slots go DRAINING -> FREE.

They are then the next wave's spares. Until then no copy targets them: the destination
rule in 7.2 admits only FREE slots.

This is exact for **one serving stream with serialized replays** (the plan's Global
constraint: "No concurrent graph replays or concurrent fused compute in this phase"). It is
not valid for multi-in-flight consumers (Task 9's "Hit/miss or multi-request compute
overlap"); that follow-up would need per-consumer retirement, and this design marks the
place (section 14, A1).

### 7.5 The device does not check generations

Copy kernels and graph gathers do not read `generations`, `slot_state` or
`slot_generations` (2.2). Safety therefore rests on host-side state transitions, ordered
device work and the DRAINING rule, not on a device check. A **detector** in the spirit of
`LEASE_PROTOCOL.md` section 6.5 would need a device word the copy kernel reads; that
is not proposed here (it would change a captured-adjacent kernel). Instead the test plan
(section 11.4) uses byte-pattern poisoning to catch a violation in a test, and the host
counters catch stale-ticket attempts.

### 7.6 Decision readback (B1), M3

`decide_residency_policies` reads the scores back with a blocking `.cpu()`. That D2H is
legitimate (the scores live on the device) but it blocks. M3: copy the stacked scores into
a pinned buffer with `non_blocking=True`, record an event, and run the decision on a later
poll when the event query is True. The decision is then one to a few forwards stale. The
scores are exponentially-decayed counts (`advance_residency_policies`), so a one-forward lag
is small against `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=32`; but it is a semantic change,
and its parity check is: the decisions equal the synchronous decisions on the same
score snapshot (they are computed by the same pure function), and boundary-to-boundary
hit-rate behaviour is unchanged within noise. This is the only part of the design that
changes *when* decisions are made; it is last for that reason [DECIDE 5].

---

## 8. Item 4 (part): bounded and below demand

### 8.1 Removing the pause: what the pause protected, and what replaces it

`before_host_use` exists because eager Python touches the pinned tier. Enumerating what
the pause protects, and whether the promotion path needs it [E, by reading `host.cpp`]:

| The pause protects | Needed by the promotion path? | Replacement |
|---|---|---|
| The slot **metadata** (`expert_slot`, `slot_to_expert`, `state`, `stamp`, `hot`) from a concurrent service update | No. Every Python-facing method takes `RamTier::mutex_` (`has`, `touch`, `assign`, `release`, `mapping`, `slot_to_expert`, `lru_order`, `set_hot`). | the same mutex |
| The slab **bytes** of a `kLoading` slot the service is filling | No. Promotions read only `kReady` rows. | leases on `kReady` rows |
| The bytes of a `kReady` slot the promotion reads, against eviction/overwrite by the service | Yes. Today held by the pause plus the per-chunk `synchronize`. | **the source lease** (section 9) |
| The D12 argument that "the device is waiting on this record, so no gather is in flight" when the service evicts for an armed demand | Not by promotions, but the argument no longer covers a promotion's GPU reader. | **leased slots are excluded from eviction** |
| A second reader on the same io_uring rings / slabs: the eager Python `Exl3ShardRowSource` versus the service's `RowReader` | Avoided: from M2 promotions never use the Python reader. | promotions read only through the service |

The last row is an open question in its own right: whether the Python eager reader may
run concurrently with the native reader [OPEN 2, now answered in part]. Review F10: they are
**separate io_uring instances** sharing only the drives (each reader object owns its own ring;
the EXL3 eager path registers no buffers). Today they also never overlap in time, because the
pause serializes them. One constraint follows: the general reader enforces creator-thread
ownership (`check_owner_` in `uring_file_reader.cpp`), so nothing may move its calls off the
scheduler thread. Section 7.1 does not propose that. And `ExpertRowSource.register_destinations`
"must run on the reader's owner thread" (`expert_row_source.py`). **What must be true**
for M2 to be safe: promotion admissions are performed only by the service thread; the
Python eager reader is used only inside an eager `host_use` (which still pauses); and no
promotion state lives in slots the eager reader may assign. If the two readers can share
rings safely, that constraint could be relaxed later; it is not needed.

### 8.2 Promotion-admission class in the service (M2) [P]

A third request class beside demand (device-posted, armed) and advisory
(`pump_advice`): **promotion admit**.

- Submitted from Python: `submit_admit(row, experts[], ticket_id)`, into a
  mutex-guarded queue. No device involvement, so nothing is added to the mapped request
  page or the lease block; the single-writer rule of `LEASE_PROTOCOL.md` section 5 concerns
  mapped memory only.
- Served only when no demand is pending and no advisory is pending: strictly below both.
  A promotion admit is a long-horizon investment; an advisory hides a latency that is
  about to be paid.
- **Cancellable at row granularity**, exactly as `serve(advisory=true)`: at most one row
  of I/O outstanding, gives up on `demand_pending()`, `pause_requested_`,
  `stop_requested_`; publishes the rows that completed whole. The priority-inversion bound
  is therefore **one row read**: a demand posted during a promotion row read waits for that
  row (mean 10.16 ms per row, `DSV41_REFERENCE.md` section 18.2; tail unmeasured). This is
  the same bound advisories have today.
- On a row becoming READY the service takes the lease in the same critical section
  (`lease_on_ready`, requirement R3) and records `{ticket_id, expert, slot,
  slot_generation, lease_id, status}` in a result queue the poll step drains
  (`poll_admits()`, non-blocking, under the mutex).
- Failure is per row: an unreadable row fails its ticket row (ADMITTING -> CANCELLED),
  not the layer, not the process. A demand failure is the existing fail-stop; a promotion
  is optional, so its failure is a counter and a log line.

### 8.3 Bounding in-flight bytes

Three caps, all [P], env-configurable (section 13), all initial values to be tuned:

- **`max_inflight_bytes`**: total bytes across COPYING waves. One hot row is about 13.3 MB
  (derived above), and `DSV41_REFERENCE.md` section 16.12 gives about 1.1 ms of H2D per
  RAM-tier promotion row; a 32-row cap is about 425 MB and about 35 ms of copy time, so the
  median window (25 promotions) fits one wave and the bound on the *hold time of a source
  lease* is about tens of milliseconds, far below the 2 s demand timeout
  (`SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`). The 1.1 ms figure is from one measurement; the
  copy rate under concurrent forward traffic is unmeasured [OPEN 5].
- **`max_admit_rows`**: rows in ADMITTING / LEASED, which bounds the RAM tier's leased
  fraction (5,644 rows total; about 141 per layer if uniform [OPEN 6]; not derived).
- **`max_waves_per_boundary`** and one publication batch per poll (P5), so a burst
  (832 promotions in the worst measured window) is spread over boundaries instead of
  landing on one forward.

### 8.4 "Below demand" on the device and the link

The service ordering above is real priority. On the GPU and PCIe there is none to be
had cheaply:

- The executor stream is created with the default priority
  (`device_module.Stream(device=self.device)`), so it competes equally with the forward
  stream. CUDA stream priorities exist, but whether a stream's priority reaches the kernel
  nodes of a *captured graph* replayed on another stream I did not determine [OPEN 8]. Do
  not rely on it.
- The promotion copy on the GPU route is an SM kernel (`copy_expert_rows_gpu`) reading
  pinned host memory, i.e. it competes for **SMs and for the PCIe link** with the graph's
  own demand gathers (the in-graph `copy_expert_row_segments_gpu` of missed rows,
  1.055 ms/row of G, section 18.2). A DMA route exists (`_ROUTE_DMA`,
  `ExpertDMARowRoute`, host-issued `cudaMemcpyAsync`) that uses the copy engines and no
  SMs; it is not graph-capturable, which is irrelevant for the executor stream. Whether it
  is better under load is an experiment (arm C-dma, section 12), not an assumption.
- So "priority below demand" on the device is achieved by **pacing**: the byte cap, rows
  per enqueue, and one wave per layer. The metric that tests it is the demand gather's
  ms/row inside promotion-copy windows against outside them (section 12). That measurement
  may show the pacing is insufficient; the design's response would then be a smaller
  `max_inflight_bytes` or a duty-cycle limit, not a claim of priority.

---

## 9. Item 4 (part): the lease contract this design needs

### 9.1 Requirements on LEASE_PROTOCOL.md

`LEASE_PROTOCOL.md` section 17.1 already sketches the hook: `acquire_host_lease(row,
expert)` and `release_host_lease(LeaseRef)`, "I do not specify the promotion protocol
(that is Task 8's own design)". This is that design's list of what it needs from the lease
owner. These are **requirements**, not assumptions; if any is refused, the corresponding
milestone is not implementable as written.

- **R1. A host-lease API callable without a pause.** `acquire_host_lease` /
  `release_host_lease` take only `RamTier::mutex_`, are callable from the scheduler
  thread, and never call `pause`. Release is exactly-once, from any thread, and a stale or
  repeated `LeaseRef` (slot generation or `lease_id` mismatch) is a **counted error**, not a
  silent no-op and not a second decrement.
- **R2. The pause-time split (LEASE_PROTOCOL table F10).** F10 says the pause
  acknowledgement "requires `outstanding == 0`, otherwise the pause fails and eager use is
  refused". **`outstanding` must count graph-lane leases only.** Host leases (promotions)
  must not block an eager pause. Reasoning:
  - Otherwise one in-flight promotion, whose lease is held for the copy's duration, makes
    every eager `before_host_use` fail for that duration, and a prefill gather would be
    refused because a background promotion exists.
  - `before_host_use` cannot wait for the promotion to finish: it synchronizes only
    `torch.cuda.current_stream()` (`Exl3RamMissService.before_host_use`), not the executor
    stream, so it **cannot establish that a promotion copy on the executor stream is done**.
    Draining the executor stream from `before_host_use` would add a second synchronization
    to the eager path and is not proposed.
  - The split is safe because a host-leased slot is excluded from eviction (R6); the pause's
    remaining job for it, protecting slab bytes from a concurrent eviction, is done by the
    lease, not by the absence of the thread.
  So: graph-lane leases block the pause; host leases do not, because their slots are
  protected by `leases > 0`.
- **R3. `lease_on_ready` for service-initiated admission (M2).** A promotion-admit row
  becomes `kReady` and is leased in the same `mutex_` critical section. Section 17.1 says
  the slot "must be kReady" at `acquire_host_lease`; that leaves a window between
  `kReady` and the lease in which a demand or advisory eviction could take the row unless it
  is `hot`. This design also pushes `set_hot` at reservation (R-E), but `hot` is a boolean
  keyed by expert that the next `set_hot` overwrites, so it is not a substitute.
- **R4. Promotion leases never cause a demand failure.** With `LEASE_PROTOCOL.md` A3, a
  promotion's release must not depend on a demand being served. Add the bound: the hold
  time is at most enqueue-to-completion of a capped copy (section 8.3), and if a demand is
  deferred for lack of a victim *only because of promotion leases*, the poll step releases
  LEASED-not-yet-COPYING leases immediately (section 6.4). The tri-state
  {slot, deferred, none} must report *whose* leases caused the deferral.
- **R5. The eviction predicate covers every path.** `take_slot_locked`, `assign` and
  `release` (the Python-facing eager calls) all refuse a slot with `leases > 0`; `release`
  of a leased slot throws exactly as it does for `kLoading` today (LEASE_PROTOCOL section 8).
- **R6. Shutdown covers host leases.** Outstanding host leases at teardown are part of the
  quarantine set of `LEASE_PROTOCOL.md` section 14: never recycle a pinned slab while a
  promotion copy may still read it. Note this design adds one more GPU reader class to
  D5: the executor stream, and `ExpertPinnedHostCache._release_slabs` is a
  `weakref.finalize(self, release_host_slabs, registered)` that runs at exit
  without a device barrier [E, `stream.py`, `ExpertPinnedHostCache.__init__`].
- **R7. Counters** `host_leases_outstanding`, lease hold time (histogram or
  p50/p95/p99), `pauses`, `pause_wait_ns`, `defer_reason` counts. Section 11's "no
  service-wide pause in promotion submission" test reads `pauses`.
- **R8. Non-interference with `kVersion` consumers.** `NativePinnedSlotTable.expert_to_slot`
  rebuilds an `OrderedDict` when `host.version()` moves. The lease API must not bump
  `kVersion` (it changes no mapping), or the promotion path pays B9 for every lease.

### 9.2 A deadlock the current copy ordering would create (for this proposal only)

**Scope: this cycle cannot form on today's code**, because no source lease exists yet
(confirmed by the review, F4). It is a property of *this design if it kept today's copy
ordering*. The mechanism itself is real: `_submit_operations` does
`stream.wait_stream(producer_stream)`, and the observer runs in the `finally` of
`with_forward_pass`, after the forward's replay has been enqueued, so an armed wait kernel can
already be in the producer's queue when a promotion enqueues.

`LEASE_PROTOCOL.md` A3 forbids a non-graph holder's release from depending on a demand.
Today's copy path has that dependence, hidden. `_submit_operations` does
`self.stream.wait_stream(producer_stream)`. Suppose a promotion holds a source lease on
slot S and its copy waits on the serving stream. The serving stream is inside an armed
`exl3_ram_miss_wait_kernel`, waiting for a demand the service cannot serve because its
only victim is S. Cycle: demand wait -> service deferral (lease on S) -> lease release
needs copy completion -> copy waits for the serving stream -> serving stream is in the
wait. The 2 s device timeout would break it, as a fatal.

This is why section 7.2 forbids any dependency of the copy stream on the serving stream
once a lease is held: the plan upload goes on the executor stream and `producer_stream` is
`None`. It is also why the destination must be a FREE slot (no retire ordering needed).

### 9.3 What is left of the broad pause

After M2, `before_host_use` is still called by every eager pinned-tier use:
`ExpertPinnedHostCache.lookup`, `ensure_rows`, `copy_rows`, `gather_rows` (each opens a
`host_use`), seeding, and direct row reads (`stream.py`, "Row-source reads outside eager
gathers (promotions, seeding, direct calls)"). Those still synchronize the stream and
pause the whole service. **That is the remaining broad pause; this design does not remove
it.** Removing it requires those eager consumers to hold leases (a Task 5 extension: an
eager copy is a "non-graph holder" in section 17.1's terms) and is named here so the
"fully asynchronous" claim cannot be made while it stands. Prefill gathers are not
bounded to boundaries, so they pause on every eager prefill batch, outside Task 8's
scope.

---

## 10. Failure, cancellation, shutdown

All follow one rule, taken from `_drain_device`'s precedent: **when the last reader of a
resource cannot be shown to have finished, keep the resource and stop using the path.**

| Event | Action |
|---|---|
| Cancel in PLANNED / ADMITTING / LEASED | release lease; destination -> FREE; ticket CANCELLED. Nothing was enqueued. |
| Cancel in COPYING | ABANDONED: keep destination LOADING and the lease; when the event completes, release both and discard; never publish. |
| Cancel in COPIED (before publication) | release destination -> FREE; lease already released. The victim is untouched (still mapped). |
| Cancel after PUBLISHED | not a cancel; the promotion happened. A later boundary may evict. |
| Event query raises / `synchronize` raises (sticky CUDA error) | QUARANTINED: destinations stay LOADING, leases stay held, the layer takes no further tickets; other layers continue; log once. **Layer-local isolation is new behaviour** (review F7): today a failed promotion raises uncaught out of `_update_residency`, so the poll step needs its own catch and a per-layer quarantine flag. A sticky CUDA error will also break demand, which is the existing fail-stop. |
| Enqueue fails after a partial launch | as above: record an event in a `finally` (D2); quarantine until it completes; if the event cannot be recorded, quarantine permanently. |
| Ring full / plan busy / cap | DEFERRED with a reason; no exception. |
| Service admit failure (row unreadable) | that row ADMITTING -> CANCELLED; the wave proceeds with the rows it has, or defers if none. Counter + log. |
| TTL exceeded (`deadline_boundary`) | cancel by the state's row above. |
| Layer/manager teardown | `close_admission`; cancel PLANNED..LEASED; ABANDON COPYING; then an *ordered* drain: synchronize the executor stream (allowed off the normal path) before any slab is unregistered; if it fails, quarantine and skip unregistering. |
| Process exit through `_stop_live` (atexit) | `LEASE_PROTOCOL.md` DECIDE 2: unconditional quarantine, no ordered sequence. This design adds the executor stream to what that path must not free under. See R6. |

`Exl3RamMissService.shutdown()` has no production caller (its callers are two tests);
the real teardown is `_stop_live`. A promotion design that only worked under `shutdown()`
would be untested in production; hence the last two rows. **A production caller of an
orderly shutdown is a requirement of Task 5 (`LEASE_PROTOCOL.md` section 14.6), and
promotions depend on it** for the drain-then-unregister sequence.

---

## 11. Item 5: proofs and tests, specified for someone else to run

I cannot run any of these. Each is specified to the level of "what to instrument, what to
assert, what would make it fail".

### 11.1 Proving no normal-path sync or pause in promotion submission

Define "promotion submission" as the call graph from the poll step and `_update_residency`
to the return of the last enqueue, on a boundary with no fault.

**Static (CPU, no GPU).** A test that monkeypatches, with a call-counting spy, each of:
`torch.cuda.synchronize`, `torch.cuda.Stream.synchronize`, `torch.cuda.Event.synchronize`,
`ExpertHotCache.wait_for_slot_publication`, `ExpertHotCache._drain_device`,
`Exl3RamMissHost.pause`, `Exl3RamMissService.before_host_use`,
`ExpertPinnedHostCache.host_use`. Drive one full ticket life (PLANNED to DONE) with fake
executor, stream, event and service objects (`AsyncExpertTransferExecutor.__init__`
accepts `stream`, `event_factory` and `stream_context`, which is why `test_expert_transfer.py`
can run without a GPU [E]). Assert **zero** calls. The spy for `_drain_device` must
not fire on the normal path; it is the failure path only.

**Dynamic, sync-causing tensor ops.** Run the same drive on a CUDA build under
`torch.cuda.set_sync_debug_mode("error")` so any `.item()`, `.tolist()`, `.cpu()`,
`.numpy()` on a CUDA tensor, or `torch.tensor(list, device=cuda)` inside submission raises.
(This mode does not catch an explicit `Stream.synchronize()` call; the spies above do.)

**Dynamic, CUDA API trace.** In an Nsight Systems run, trace CUDA API on the scheduler
thread and assert **zero** rows of `cudaStreamSynchronize`, `cudaEventSynchronize`,
`cudaDeviceSynchronize`, and non-async `cudaMemcpy` inside every window between "boundary
start" and "boundary end" markers (NVTX ranges added around `_update_residency` and the
poll step [P]). Use `--cuda-graph-trace=graph` for decode traces (project
`CLAUDE.md`, "Nsight Systems traces": `node` mode inflates host time per graph by about
0.77 us per node and makes step tails look like idle GPU); do not rank kernels from a
graph-mode trace, which omits the graph body (same note). Run it under `taskset -c 0-63`,
threads capped, per the directive; do not copy a report to `/tmp`.

**Service-wide pause.** Assert the service's `pauses` counter (R7) is **unchanged**
across a promotion-only interval (a run with `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` off and
no eager prefill), and that `pause_wait_ns` is unchanged. Separately assert the counter
does move on an eager prefill, to prove the counter works.

A failure of any of these is a blocking finding for the milestone, not a warning.

### 11.2 Unit tests, CPU only (no GPU)

Use the fake executor and a fake or `NativePinnedSlotTable`-shaped service, in
`test/registered/unit/layers/moe/` beside `test_expert_hot_cache.py`,
`test_expert_hot_cache_publication.py`, `test_expert_transfer.py` and
`test_exl3_ram_miss_service.py`.

- **M0**: `ensure_rows(list)` on a CPU-tier `ExpertPinnedHostCache` performs no CUDA call
  (a spy on `torch.tensor` with a `device` argument and on `Tensor.tolist` of CUDA tensors);
  admissions equal those of the tensor entry for the same ids, including duplicates and
  more misses than slots ("More misses than slots reassign a slot within this call").
- **State machine**: every arrow of section 5.2, and every *illegal* transition raises or
  counts: publish before COPIED; publish twice; release lease twice; cancel after publish;
  stale destination ticket; stale lease generation.
- **Cancellation** (each row of the 10 table): after cancel in each state, assert (a) no
  resource that an enqueued copy can touch was released before the event completed (fake
  event held incomplete), (b) the victim is still mapped after a CANCELLED and COPIED cancel,
  (c) the destination is FREE only in the states that allow it, (d) no publication ever
  occurs for an ABANDONED ticket.
- **Copy failure**: fake event whose `query()` raises; fake enqueue that raises on the
  third of six operations. Assert: the layer QUARANTINED, destinations LOADING, leases
  held, no new ticket on that layer, other layers unaffected, `deferred_reason=quarantine`.
  Assert a partial enqueue records an event (D2).
- **Insufficient capacity**: (i) `K = 0` full cache: every promotion deferred with
  `no_free_vram_slot`, none forced; (ii) fewer FREE slots than promotions: the highest-rank
  subset is admitted and the rest deferred, all-or-nothing per layer-wave; (iii) ring full:
  `try_submit` returns `None`, deferral counted, no exception; (iv) all RAM candidates
  leased: `no_ram_victim_leases`, and a *demand* in the same state defers rather than fails
  (needs the tri-state); (v) byte cap: in-flight bytes never exceed `max_inflight_bytes`
  over a 832-promotion burst; assert the maximum of the gauge, not just the final value.
- **Old and new map consumers** (logic level): a model of the serving stream as an ordered
  list of `{replay(reads map), publish, record}` entries; assert (a) a replay ordered before a
  publication reads the old map and its victim's bytes are unchanged, (b) a replay after
  reads the new map and the new bytes, (c) the victim is not written until the
  `retire_event` after the publication has completed, (d) a replay ordered *between* the
  copy event and the publication is legal and reads the old map.
- **Priority**: a service model in which a demand arrives during a promotion admit
  cancels the admit at the next row; assert the rows already complete are published whole
  and the rest released; assert the promotion class is never served while a demand or
  advisory is pending.

### 11.3 Native tests (`test/registered/unit/kernels/`, CPU with the C++ thread)

Beside the existing `test_exl3_ram_miss_split.py` / `test_exl3_ram_miss_thread.py`.

- A leased slot is never chosen by `take_slot_locked`, `assign` or `release`, under
  demand pressure that would otherwise evict it; the deferral is reported with the cause.
- `acquire_host_lease` then eviction pressure then `release_host_lease`: the slot becomes
  evictable exactly after the release.
- A stale or duplicate `LeaseRef` release is counted as an error and does not decrement.
- `lease_on_ready`: an admitted row cannot be evicted between READY and the lease (drive
  eviction from a second thread at every interleaving the harness can force).
- The F10 split: an outstanding host lease does not make `before_host_use` fail; an
  outstanding graph-lane lease does.
- A promotion-admit is cancelled by a posted demand within one row's read time, using the
  existing fault injector (`inject(delay_s=..., delay_after_demands=...)`) to make rows slow.
- `pauses` does not move across `submit_admit` / `poll_admits` / lease calls.

### 11.4 GPU tests (manual, under the GPU lock; do not run without scheduling)

`test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py` is the existing home. GPU time on
divix01 is scheduled through the coordination the project directives require; none of this
is to be started ad hoc.

- **Byte-pattern poisoning, old/new consumers.** Fill the victim slot with pattern A and
  the promoted row's pinned source with pattern B, both with per-row checksums. Launch a
  consumer kernel that reads via `expert_to_slot` (a stand-in for the graph gather plus MoE
  read) *before* the publication, slow the copy with a spin kernel on the executor stream
  (`torch.cuda._sleep`) so the copy overlaps the consumer, and a second consumer after.
  Assert the first consumer sees only A, complete, the second only B, complete, and no row
  ever mixes. Repeat with the victim slot **recycled as the next wave's destination**
  immediately after `retire_event` completes: assert a consumer enqueued before the
  publication never sees the next wave's bytes. This is the test that fails if DRAINING is
  wrong.
- **Publication atomicity** (P1): run consumers on a *different* stream from the publication
  (the wrong configuration) and assert the test can detect a torn map; then the correct
  configuration passes. It shows the assumption is testable, and not vacuous.
- **`wait_event` necessity** (P2): assert publication enqueued behind a copy that is still
  running does not publish early (delayed copy; the consumer after the publication sees B
  complete, not partial).
- **Lease under delayed GPU consumption**: hold a promotion copy with a spin kernel while
  forcing RAM admission pressure through demand and advisory traffic; assert the leased
  pinned slot's bytes are unchanged for the whole interval. Lease-pressure test of Task 5's
  gate, promotion side.
- **No deadlock**: a promotion holding a lease while the serving stream is in an armed
  wait for a demand whose only victim is that slot must complete without the 2 s timeout
  (section 9.2). Assert the demand is served after the copy completes and no fatal is
  raised. This test fails if `producer_stream` is used.
- **Copy failure on a device**: inject a launch failure (an invalid destination pointer in
  one of six operations under a debug build) and assert the D2 behaviour.
- **Shutdown with a promotion in flight**: through the real `_stop_live` path, not
  `shutdown()`; assert no `cudaHostUnregister` of a slab under a running copy (or that the
  unregister is skipped and the slab quarantined).

### 11.5 Regression checks the existing suite must keep passing

Run as the plan's verification section lists, plus the promotion and publication tests
above: `test_exl3_ram_miss_split.py`, `test_exl3_ram_miss_thread.py`,
`test_expert_hot_cache.py`, `test_expert_hot_cache_publication.py`,
`test_expert_transfer.py`. Do not report skipped hardware tests as passed evidence (plan,
verification procedure).

---

## 12. Measurements and the gate

### 12.1 The gate, as the plan states it

"Safe asynchronous publication and reduced boundary stalls at matched cache capacity.
Existing generic async flags are not evidence of EXL3 support."

Two claims, each with its own evidence:

- **Safe publication**: every test in 11 passes; the no-sync and no-pause proofs in 11.1
  hold; no violation of R-A..R-E in any run (a counter, section 12.3).
- **Reduced boundary stalls at matched capacity**: section 12.2.

No throughput claim is part of the gate. Section 0 says why.

### 12.2 Arms

Matched conditions follow the plan's Global constraints and `DSV41_REFERENCE.md` section
18.6's `c32` shape: same corpus and sessions, same seed, same `SGLANG_MOE_HOT_GPU_MB`, same
pinned RAM budget, unprofiled runs for tokens/s and latency, traced runs only for
attribution, cache state recorded.

| Arm | Live slots | Spare | Promotion mode | Purpose |
|---|---|---|---|---|
| A0 | `C` | 0 | synchronous (today) | baseline; the `c32` arm |
| A1 | `C` | 0 | synchronous + M0 | isolates M0 |
| A2 | `C - K` | `K` | async (M1..M3) | **the gate arm, matched VRAM** |
| A3 | `C` | `K` extra | async | **unmatched, labelled**: isolates the spare-slot cost from the protocol cost; more VRAM |
| A4 | `C - K` | `K` | synchronous | isolates the capacity cost of `K` alone (`K` slots unused) |
| A5 | `C` | 0 | none (`c0`) | reference, exists (`c0`) |
| C-dma | as A2 | | async, `_ROUTE_DMA` copies | experiment, section 8.4 |

Reading them: A2 against A0 is the gate. A2 against A4 is the protocol's effect at equal
capacity. A3 against A2 is the price of the matched budget. A4 against A0 is the price of
the spare slots themselves; if that price dominates, the design is not worth taking at
that `K`, and the result should say so.

### 12.3 What to record

Boundary and step tails (the gate's primary):

- Step latency for boundary steps (the step at `k % 32 == 0` and the next, "for overlap
  lag") against non-boundary steps: p50, p95, p99, p99.9, max, and the boundary excess over
  the non-boundary median (baseline: mean 417 vs 359 ms, ~115 ms excess, 112 boundary
  steps, from `DSV41_REFERENCE.md` section 18.6's burst run).
- Host time inside `_update_residency` and the poll step, per boundary and per forward:
  p50/p95/p99 (`time.perf_counter_ns` around the observer, NVTX ranges for traces).
- The count and length of scheduler-thread stalls attributable to promotion: from the API
  trace of 11.1, the total `cudaStreamSynchronize` time inside boundary windows (target: 0).

Miss behaviour (the check that the spare slots did not just move the cost):

- RAM misses/token (`c32` baseline 18.52), G rows/token (126.9 in `c32`, 151.8 in `c0`),
  hot-hit rate, `unique_miss_rows`, per layer where possible; live capacity per layer.
- Pinned admissions and evictions (`counters.pinned_admissions`, `pinned_evictions`).
- Promotions issued, deferred (by reason), cancelled, abandoned, quarantined; time from
  decision to publication (staleness of a promotion's benefit).

Safety and cost of the mechanism:

- Source-lease hold time p50/p95/p99, in-flight bytes gauge and its maximum,
  `host_leases_outstanding` maximum, ring occupancy.
- Demand gather ms/row and demand wait time **inside promotion-copy windows against
  outside** (section 8.4). Demand deferrals caused by promotion leases.
- Service `pauses` and `pause_wait_ns` (11.1).
- Whole-run tokens/s, p50/p90/p99 step latency (baseline 344 / 540 / 776 ms), with
  session-to-session spread, so a 1% change is not read as signal.

### 12.4 Sample size, stated honestly

The existing burst data is 1,983 steps and 112 boundary steps. A p99 over 112 samples is
close to the maximum and is not a stable statistic; a p95 is the second or third largest
few. To report boundary p99 with a meaningful interval, plan for at least several
hundred boundary steps per arm and report a bootstrap interval, not a point. At 32
forwards per boundary and about 0.35 s per step, 400 boundaries is about 12,800 steps, on
the order of 75 minutes of decode per arm. That is a large GPU request on a contended box;
it must be scheduled per the project directives and is not a task for this document. If the
budget allows fewer, report p95 and the maximum and **say the p99 is not resolved**.

The run-to-run spread is small where it has been measured (`c32` 2.823 against the
burst run's 2.831 tok/s), but a change of 1% is inside the session-to-session variation of
sessions 1 to 3 (within +-1.3% of each other). Do not claim a throughput change of that
size.

### 12.5 Proposed pass criteria [DECIDE 6]

The plan gives the gate qualitatively. Thresholds below are proposals, chosen from the
measured noise, for the owner to accept or replace:

- Boundary-step excess over the non-boundary median: reduced at p95 and p99 (interval
  excludes zero change), and no boundary step over the arm's non-boundary p99.9.
- RAM misses/token and G: within the run-to-run spread of A0, or worse only by an amount
  that A4 alone explains (i.e. attributable to `K`, disclosed).
- Whole-run tokens/s: not worse than A0 by more than 1%. (There is at most about 1% to
  gain and 3.4% of G-reduction benefit to lose.)
- Zero safety-counter violations, zero forbidden calls in 11.1, all tests passing.

A design that passes the tails and fails the miss-rate check is not a success: it has
moved the cost into the cache. The reason the gate says "matched cache capacity" is that.

---

## 13. Knobs, and what happens to the existing flag

Proposed names, to be checked against the environment-variable conventions
(`.claude/skills/env-var-conventions/SKILL.md`: define in `Envs`, `EnvBool`/`EnvInt`,
prefer one ordered integer over fighting booleans) when implemented. This document does not
edit `environ.py`.

- `SGLANG_MOE_HOT_ASYNC_PROMOTIONS` [E]: unchanged; stays refused for EXL3 until the M2 gate
  passes. It describes the dense-format behaviour ("publish the promoted slots on a later
  forward"). It must not be reused for EXL3 by silently making it "work", because that is
  what the plan's "generic async flags are not evidence" warns against.
- `SGLANG_DSV41_PROMOTION_MODE` [P]: `IntEnum` OFF-of-async (0, today) / ASYNC (1). One
  ordered knob.
- `SGLANG_DSV41_PROMOTION_SPARE_SLOTS` [P]: `K`, default 0 (which, with the async mode,
  defers everything; a nonzero `K` is required for the async mode to promote).
- `SGLANG_DSV41_PROMOTION_MAX_INFLIGHT_MB`, `..._MAX_ADMIT_ROWS`, `..._TTL_BOUNDARIES` [P].
- `exl3_reqs._check` [E]: validates the combination, refuses `PROMOTION_MODE=ASYNC` with
  `SPARE_SLOTS=0`, and keeps refusing `SGLANG_MOE_HOT_ASYNC_PROMOTIONS`.

Because K is carved from `SGLANG_MOE_HOT_GPU_MB`, the capacity computation
(`ExpertHotCache.capacity_for_budget`, and the manager's budget split) must subtract `K`
per layer before computing live capacity, so that "matched" is by construction and the
startup log line (`Expert hot cache startup`) reports physical, live and spare slots so an
arm's capacity can be verified from its log instead of trusted.

---

## 14. Assumptions, open questions, decisions

### Assumptions this design rests on

- **A1. One serving stream, replays serialized.** The plan's Global constraint. Sections 7.3
  and 7.4 are exact under it and wrong without it. `LEASE_PROTOCOL.md` [OPEN 9] found no
  guard in the tree; this design asserts stream identity at attach and treats concurrent
  graphs as unsupported.
- **A2. The pinned rows a copy reads are final when it is enqueued.** True for
  RAM-resident rows (nothing writes a `kReady` slot except an eviction the lease forbids)
  and for admitted rows (the service's release store plus lease grant precede the enqueue).
- **A3. Lease API semantics** as required in 9.1. If the lease owner declines R2 or R3, M1
  or M2 must be redesigned; nothing here can substitute.

### Open (I could not determine)

- **[OPEN 1] Which stream is current in `on_expert_distribution`, and whether a graph
  replay for the next forward can already be queued.** By reading, in overlap mode the
  observer runs inside `forward_stream_ctx`, so it is the forward stream. Unverified: that
  replays launch on that stream in every mode; the non-overlap path; speculative v2's
  draft-extend running after the observer; breakable decode graphs' eager breaks. **What
  must be true:** publication kernels and every replay that reads the map are on one
  stream. Test: at attach, record the replay stream and the observer's `current_stream()`;
  refuse the async mode if they differ; log both.
- **[OPEN 2] Whether the Python eager reader can run concurrently with the service's io_uring
  reader.** Answered in part by the review (F10): they are separate io_uring instances sharing
  only the drives, so there is no shared ring or registered-buffer state; O_DIRECT alignment
  state and drive contention were not checked, and the pause still serializes them today.
  **What must be true:** nothing new for this design, because promotions never use the Python
  reader from M2 and eager use still pauses. The Python reader's creator-thread ownership
  forbids moving its calls off the scheduler thread.
- **[OPEN 3] Whether `hot` protects a row across a cancel/retire in every path.** `set_hot`
  is keyed by expert and overwritten by the next push; I argued it is not a lease (R3) but
  did not enumerate every push site. Test: cancel a ticket while its copy is in flight, push
  `set_hot` without it, apply eviction pressure; the leased slot must survive.
- **[OPEN 4] Who else submits to the shared per-device executor.** `AsyncExpertTransferExecutor.for_device`
  is a singleton. Correction (review F3): `expert_prefetch.py` uses that **shared** singleton by
  default; its private `max_inflight=1` executor is the exception (taken only when a
  `copy_stream` or `ready_event` is supplied). Other `for_device` users were not enumerated.
  `for_device` raises `ValueError` when asked for a ring size different from the registered
  one, so an async design **cannot simply request a larger ring**; it must share the 8 slots or
  own a separate executor object. Ring pressure and the `try_submit` behaviour depend on it.
- **[OPEN 5] The copy rate under concurrent forward traffic.** 1.1 ms per row is one
  idle-conditions figure. The SM-gather copy competes with the graph for SMs and the link;
  lease hold time and the byte cap's meaning depend on it.
- **[OPEN 6] Per-layer slot split under a seed, and the per-layer RAM row counts.**
  Unseeded it is uniform, 22 or 23 per layer (section 6.2, review F9); a seeded run follows the
  seed. Per-layer RAM row counts (5,644 rows total) were not derived. Determines what `K` costs
  per layer and how many leased rows a cap allows.
- **[OPEN 7] The per-layer promotions-per-boundary distribution.** Determines the smallest
  useful `K`. The counters exist; the analysis has not been done.
- **[OPEN 8] Whether CUDA stream priority reaches the kernel nodes of a captured graph
  replayed on another stream.** Determines whether device-side priority is available at all.
- **[OPEN 9] Whether the score-readback lag (M3, 7.6) changes hit rate.** Argued small; not
  measured.
- **[OPEN 10] Whether removing the per-chunk sync changes anything about the `evictable_rows`
  chunking.** The chunk size existed to keep pinned rows from being evicted by the next
  chunk; leases make it unnecessary, but the interaction with the inclusive `is_pinned`
  callback (`ExpertPinnedHostCache.evictable_rows`) was not traced through in full.

### Decisions I made that the owner may overturn

- **[DECIDE 1]** Matched capacity by default: `K` spare slots per layer carved from the
  existing budget; unmatched only as a labelled arm. Alternative: extra VRAM for spares
  (simplest, hides the cost, fails the plan's gate wording).
- **[DECIDE 2]** Promotion admission goes through the native service at a priority strictly
  below advisories, reusing the advisory cancellation shape, rather than a second Python
  reader. Alternative: restrict promotions to RAM-resident rows (M1 only). The reference's
  2026-09-19 ruling rejected "promoting only experts already in RAM" as a cheaper substitute
  because it gives back part of the 16% G reduction; this design therefore does not stop at
  M1 for the gate.
- **[DECIDE 3]** One owner thread for the executor and the ticket table; polling in the
  observer; no progress thread.
- **[DECIDE 4]** Non-blocking staging-buffer reuse by polling and deferring, not by adding
  buffers. Double buffering is equivalent and simpler to reason about if the deferral rate
  proves annoying.
- **[DECIDE 5]** Asynchronous decision readback is the last milestone and is optional for
  the safety gate (B1 blocks the scheduler thread for the score copy). It is required for
  the claim "no blocking in the boundary at all".
- **[DECIDE 6]** The pass thresholds in 12.5.
- **[DECIDE 7]** DRAINING is recycled by host-polled `retire_event`, not by a device-side
  wait, so the copy stream never depends on the serving stream (9.2).
- **[DECIDE 8]** The lease is released at COPIED, with `hot` covering the interval to
  publication and beyond; alternative is holding the lease to PUBLISHED, which is simpler and
  costs a longer hold and more demand-deferral risk.

---

## 15. Milestones and gates

Each milestone has a gate that can fail. Do not merge a milestone on its checklist.

**M0: CPU ids.** Files: `expert_hot_cache.py` (`_prepare_promotion`), `expert_stream.py`
(`ensure_rows`), tests. Gate: identical admissions and copies (byte parity, same slot
assignment, same stats) on the CPU-tier unit tests and one eager EXL3 run; API trace shows
the `torch.tensor(..., device=cuda)` H2D and the `.tolist()` D2H gone from `ensure_rows`. No
performance claim.

**M1: spare slots, draining, async copy and publication, RAM-resident rows.** Requires the
host-lease API (R1, R2, R5, R6, R7, R8). Files: `expert_hot_cache.py` (states, ticket, poll,
publication), `expert_transfer.py` (`try_submit`, plan upload on the executor stream, D2, D3),
`exl3_reqs.py`, `srt_ram_miss.py` (lease calls), tests. The poll step has its own exception handling and a per-layer quarantine flag (review F7). Gate: all of 11.2 and 11.4's
consumer, atomicity, lease and deadlock tests pass; 11.1's static and API-trace proofs hold
for RAM-resident rows; no throughput claim; the synchronous path is unchanged with the mode
off.

**M2: asynchronous RAM admission through the service.** Requires R3, R4. Files: `host.cpp`
(promotion class, `submit_admit`, `poll_admits`, `lease_on_ready`, counters),
`ops_ram_miss.py`, `srt_ram_miss.py`, tests. Gate: 11.3 passes; `_refresh_mapping` and `ensure_rows` are no longer on the promotion path (B10); NVMe reads are off the
scheduler thread (API/CPU trace: no io_uring submit from the scheduler thread in a
boundary); `pauses` unchanged across promotions; demand latency during promotion admission
within the one-row bound. This is the milestone after which `SGLANG_DSV41_PROMOTION_MODE`
may be described as non-blocking for promotions.

**M3: async decision readback, backpressure, tuning, gate measurement.** Gate: section 12
in full at matched capacity, with the sample-size statement in 12.4 honoured.

Final wording rule: even after M3, the pipeline is **not** to be called fully asynchronous
while the eager-gather `host_use` pause (9.3) remains. State what remains.

---

## 16. References

- Plan: Task 8, Task 5, Global constraints, Task 9 (multi-in-flight overlap) in
  `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`.
- `analysis/dsv41-drive/LEASE_PROTOCOL.md`: sections 1.2 (D1, D5), 8, 14, 16.2 (A1..A4), 17.1,
  table F10.
- `DSV41_REFERENCE.md` section 18.6 (the residency boundary, paired arms, burst frequency,
  the 2026-09-19 ruling), section 9.1 (inclusive hierarchy), section 18.2 (per-row NVMe cost).
- Symbols: `ExpertHotCache.stage_reassign`, `_prepare_promotion`, `_load_reserved`,
  `_load_reserved_in_chunks`, `_drain_device`, `_publish_slots`, `wait_for_slot_publication`,
  `HotCacheSlotTicket`; `ExpertHotCacheManager._update_residency`,
  `_publish_completed_promotions`, `on_expert_distribution`; `ExpertPinnedHostCache.ensure_rows`,
  `host_use`, `evictable_rows`; `AsyncExpertTransferExecutor._submit_operations`,
  `_acquire_slot`, `has_completed`; `FixedRowTransferPlan.set_rows`;
  `Exl3RamMissService.before_host_use`, `Exl3RamMissHost.pause`;
  `RamTier::serve`, `take_slot_locked`, `set_hot`, `RamThread::pause` (`host.cpp`);
  `exl3_ram_miss_wait_kernel` (`device.cuh`); `exl3_reqs._check`.
