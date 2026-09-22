# Second review of `PROMOTION_ASYNC.md`, sections 5 and 11 (Task 8 design), 2026-09-21

The first review (`PROMOTION_ASYNC_REVIEW.md`) covered the document's claims about existing
code and explicitly did not cover section 5 (the slot state machine, including DRAINING) or
section 11 (the test specifications). Those two carry the safety claim. This review covers
only them, and does not repeat the first review's ten items.

Reviewed: `analysis/dsv41-drive/PROMOTION_ASYNC.md` at `d9d5622466`. Commit `3b80318c5c` changed
it afterwards (section 9 only: R2/R3 text and a note under 9.2 citing `lease_model.py`); sections
5 and 11 are byte-identical across those commits. Source read at the `dsv41` working tree (the
executor and hot-cache files are unmodified there) and at `HEAD` blobs for the two files another
session is editing (`exl3_ram_miss_host.cpp`, `ops/moe/exl3_ram_miss.py`). Read-only: nothing under
`python/` was touched, no GPU, no drive load, nothing run except one CPU-only check (torch calls
with `CUDA_VISIBLE_DEVICES` empty). Serena was unavailable, so this is grep and reading. The
reviewer is not the author of the design.

Sections read: 5 with every section it leans on (2, 4, 6, 7, 8, 9, 10, 14 A1 and OPEN 1), all of 11.

**Severity.** HIGH: can produce a wrong result or a fatal in the design as written, or a safety
test that cannot detect its own bug. MEDIUM: the machine or the spec is incomplete or contradicts
itself in a way an implementer will settle by guessing. LOW: precision.

## Verdict

The DRAINING ordering itself is sound under assumption A1 (one serving stream, serialized
replays). The safety claim as a whole is not carried by the state machine and the test list as
written:

- the test the document names "the test that fails if DRAINING is wrong" cannot fail on that
  bug (B1);
- the source lease is released by a host poll, which reopens the section 9.2 cycle through the
  host instead of the stream (A1);
- publication waits on a ring event that the ring may already have re-recorded (A2);
- one state, QUARANTINED, can never be exited (A3);
- the section 11.2 tests labelled "CPU only" need a GPU to construct their subject (B2).

## Requirement on every safety test in section 11: a negative control

**A byte test that has never been shown to fail is not evidence.** It converts an unchecked
assumption into a checked one in the reader's mind, which is worse than having no test. Every
safety rule this design states (R-A, R-B with P2 and P3, the DRAINING rule, R-C, the lease rules)
needs a **mutation control** specified beside its test: a deliberately broken implementation
(the rule removed or shortcut) that the test must fail, on the intended assertion. The test is
accepted only when its mutant fails it and the correct implementation passes. The design already
does this once, for P1 (run the consumers on the wrong stream and show the test can see it); it
must do it for the rest. Section 11 as written specifies none for R-A, R-B, DRAINING or the lease.

## Findings in section 11 (the tests)

### B1. HIGH: the DRAINING poisoning test cannot fail on the bug it is for

Section 11.4 calls the byte-pattern test "the test that fails if DRAINING is wrong". As specified
it does not.

- It slows the **copy** with a spin kernel on the executor stream. That shows the copy does not
  disturb the victim, that is, that the destination is not the victim. That is the destroy-then-
  copy fix (section 2), not the recycle rule.
- Its recycle half recycles the victim "immediately after `retire_event` completes". A correct
  implementation cannot recycle earlier, and the test itself waits for that event, so "a consumer
  enqueued before the publication never sees the next wave's bytes" is true by stream order
  whether or not the host logic is right.
- To fail on a wrong DRAINING rule the **consumer** must be delayed (a `torch.cuda._sleep`
  kernel on the serving stream ahead of consumer 1), the driver must attempt the next wave into
  the drained slot before the retire event completes (`K = 1`, so the drained slot is the only
  spare), and the read must be ordered after the corrupting copy. None of it is specified. There is
  no third pattern C for the next wave (only A and B are defined), and no mutation control: run it
  once with DRAINING -> FREE shortcut past the `retire_event` and assert the test fails.
- The `_sleep` precedent exists in this repo (`test/registered/unit/layers/moe/test_expert_gpu_pull.py:165`),
  so this is buildable.

### B2. HIGH (spec precision): "CPU only (no GPU)" tests need a GPU to build their subject

Section 11.2 is headed "Unit tests, CPU only (no GPU)" and its state-machine tests drive
`ExpertHotCache`. `ExpertHotCache.__init__` calls `torch.cuda.current_device()` and
`.pin_memory()` (`expert_hot_cache.py:148` and `:192`). With `CUDA_VISIBLE_DEVICES` empty, both
raise "No CUDA GPUs are available" (`torch.cuda.Event()` alone succeeds). The sibling files the
document points at, `test_expert_hot_cache.py` and `test_expert_hot_cache_publication.py`, are
`skipUnless(torch.cuda.is_available())` under `register_cuda_ci`.

What is true: `AsyncExpertTransferExecutor` is fakeable (it accepts `stream`, `event_factory`,
`stream_context`), and `TestAsyncExpertTransferExecutor` runs without CUDA, so the document's
claim about that file is right for the executor. The DRAINING state, victim ticket checks,
publication, and the "no forbidden call" spy on real `ExpertHotCache` methods are not testable
CPU-only. That makes the section 11.1 "Static (CPU, no GPU)" test, the state-machine tests and the
"old and new map consumers (logic level)" test unimplementable as labelled. Either extract the slot
lifecycle into a pure class that the machine drives (then they are CPU-only for real), or label them
CUDA-runner tests. The document must say which.

### B3. MEDIUM: the zero-call assertions can pass vacuously, and their counters do not exist

- Section 11.1 asserts zero synchronizations, unchanged `pauses` and `pause_wait_ns`, and zero
  CUDA sync rows in boundary windows. None requires evidence that a promotion was submitted in the
  window, so an interval with no promotion passes. Each needs a positive control (at least N
  enqueues and N publications counted in the same window). The document does include a control for
  the counter itself ("moves on an eager prefill"); it needs the same for "promotions happened".
- `pauses` and `pause_wait_ns` do not exist. At `HEAD` the host has counters `kServedRequests`
  through `kSpinCpu` (`exl3_ram_miss_host.cpp`, `enum Counter`), mirrored by name order in
  `_COUNTER_NAMES` (`ops/moe/exl3_ram_miss.py:334`); nothing counts pauses. R7 is marked [P], so
  this is not a false claim, but adding a counter is a lockstep edit of the C++ enum, the Python
  names and the `counters()` dict, which the test specification does not say. The same applies to
  the deferral-reason counters, `host_leases_outstanding`, the lease-error counter and the in-flight
  bytes gauge asserted in 11.2 and 11.3: all [P], and meaningful only if exposed and read.
- Existing analogue: `overruns` is exposed through `counters()` and logged at exit, and nothing
  asserts on it, so an assertion on it would prove less than it appears to.

### B4. MEDIUM: failure modes the state machine admits that no test exercises

- Publication failure (A4); `retire_event` failure; a DRAINING slot left DRAINING on failure.
- ABANDONED whose event errors; a QUARANTINED exit (there is none, A3); a test that asserts every
  resource is eventually released after each quarantine cause.
- A stale executor ticket at publication and ring-slot reuse between COPIED and PUBLISHED (A2).
- Two overlapping waves on one layer (A7); the same expert re-promoted while its old slot drains
  (section 5.1 states it as a rule; no test).
- The K invariant: at least K slots FREE or DRAINING between waves, and the policy never asks for
  more than `capacity - K`.
- Deferral reasons `plan_busy`, `demand_pending`, `eager_host_use`, `shutdown`. Only
  `no_free_vram_slot`, `ring_full`, `no_ram_victim_leases`, `byte_cap` and `quarantine` are tested.
- The M2 lease grant racing a cancel (A6), and the all-or-nothing versus partial-wave contradiction.
- R6, outstanding host leases at teardown, has no native test. The teardown ordering (close
  admission, cancel PLANNED..LEASED, abandon COPYING, synchronize the executor stream, skip
  unregister on failure) has only a GPU test through `_stop_live`, none for the ordering logic.
- The A1 liveness case (a host that cannot poll while the GPU is stalled).

### B5. LOW-MEDIUM: precision in section 11.4

- P1 ("run consumers on a different stream and assert the test can detect a torn map") needs a
  deterministic construction (for example a `_sleep` between the map-updating kernels). As written
  it is a race, so the negative control can flake.
- The P2 test ("publication enqueued behind a copy still running does not publish early") enqueues
  a publication before the copy completes, which P3 forbids on the real path. It needs a test hook
  that bypasses the poll gate; none is specified.
- `torch.cuda.set_sync_debug_mode("error")` catching `torch.tensor(list, device=cuda)` is
  unverified here and by the first review (its F8). I could not test it without a GPU. Open.

## Findings in section 5 (the state machine)

### A1. HIGH: the source lease is released by the host poll, not by copy completion

**Status: a well-founded argument, not a verified fact.** The non-overlap half rests on reasoning
about the scheduler, not on reading `scheduler.py`.

- Section 5.2 takes COPYING -> COPIED on "event complete" and releases the lease there. Section 7.1
  says progress is driven only by the poll step at the end of the observer (scheduler thread, once
  per forward). R-E says the lease "covers exactly the interval in which a GPU reader exists". It
  does not: it is held until the next poll after the copy finished.
- Section 9.2 removes the copy-waits-on-serving-stream dependency (`producer_stream=None`, plan
  upload on the executor stream). That removes one edge of the cycle. The release still needs the
  scheduler thread to reach the next observer. Cycle: armed wait kernel (GPU) -> the service defers
  the demand because its only victim is leased -> lease release needs a host poll -> the host reaches
  the next observer only after reading the current forward's results. In a non-overlap scheduler the
  host blocks on forward N's outputs, never gets there, and the demand times out (fatal). In overlap
  mode the host can launch N+1 and poll, so it usually resolves. The document does not condition the
  design on overlap mode, and neither arm harness records whether overlap is on
  (`trace_corpus.py` and `eager_arm_driver.py` set no overlap flag).
- This is the dependency LEASE_PROTOCOL A3 and the document's own R4 forbid ("a promotion's release
  must not depend on a demand being served"), reached through the host instead of the stream. R4's
  mitigation ("the poll step releases LEASED-not-yet-COPYING leases immediately") is also host-driven.
- The hold-time bounds are unsupported. Section 8.3: "tens of milliseconds, far below the 2 s
  demand timeout"; section 6.4: "bounded by one poll interval". The real hold is copy time plus time
  to the next poll: up to one forward (about 250 ms per decode step at 3.9 tok/s), and across an
  eager prefill forward (TTFT about 55 s in the eager arms) no poll runs at all, so leases taken at
  the boundary before it persist through the whole prefill.
- The lease model (`lease_model.py`, cited from the section 9.2 note) does not cover this: by the
  document's own account it models neither the hold-time bound nor R4's release nor executor-stream
  events. I did not read `lease_model.py` or `LEASE_PROTOCOL.md`.
- **Required change (decision recorded by the team lead, 2026-09-21):** the lease must not be released by
  the host poll. Write a completion flag from the device after the six copies, into mapped memory,
  read by the service thread, which releases the lease itself with no host progress required. This
  matches the principle the rest of the design follows. Conditioning the design on overlap scheduling
  was considered and rejected: the safety of an async mechanism should not depend on a scheduler mode
  that the arm harnesses do not record.
- Test gap: the section 11.4 "No deadlock" test passes if the harness polls from its own thread,
  which production does not. It must run where the host cannot poll while the GPU is stalled, and
  needs its mutation control (B1's rule).

### A2. HIGH: publication waits on a copy event that the ring may have re-recorded

- Section 6.1 releases the transfer ring entry at COPIED. Section 7.3 issues
  `serving_stream.wait_event(ticket.copy_event)` at PUBLISHED, possibly several polls later
  (`publish_busy` deferral; one publication batch per poll).
- Code: `AsyncExpertTransferExecutor._acquire_slot` (`expert_transfer.py`) reuses any ring slot whose
  event `query()` is True. The ring is 8 events per device and shared (`for_device` singleton). A
  slot is re-recorded after at most 8 further submits, and section 8.3 allows a burst of waves in one
  boundary, so reuse before publication is reachable.
- After reuse, `_event_for(ticket)` (used by `wait`) raises "expert transfer ticket is stale", so
  publication raises with no edge for it in the state machine. If an implementer bypasses that and
  waits on the raw event handle, the serving stream waits on the newer wave's copy: a P3 violation
  (waiting on an incomplete copy) that feeds the section 9.2 cycle.
- **Two semantics for one ticket.** `has_completed` treats a reused slot as complete, which is right
  for the host poll. `_event_for` raises on the same condition. The same document uses both. An
  implementer will settle this by guessing.
- P2's stated reason ("issued even when the host has seen the event complete") is weaker than
  written: `event.query()` True has no false positives, so host-observed completion already orders
  the copy before any later-launched kernel. The `wait_event` is redundant safety, and with reuse it
  is the harmful part. Give each ticket its own event (not one from the shared ring), or drop the
  wait when the ticket is stale (complete by construction) and say so.

### A3. MEDIUM-HIGH: QUARANTINED is never exited, which is a leak by design

The diagram reaches it from COPYING with no outgoing edge. Section 10 says "record an event in a
`finally`... quarantine until it completes" for a partial enqueue, which implies an exit that the
diagram lacks. Both cannot hold.

Leaked: the wave's destination VRAM slots stay LOADING for good (they are the K spare slots; with
K = 1 or 2 that is the layer's entire spare capacity), the RAM rows stay leased (unevictable, so the
tier shrinks permanently, against R4's promise), and, until its event completes, a ring slot of the
shared ring. A sticky CUDA error is fatal anyway. A non-sticky partial-enqueue failure is recoverable
(the event completes) and is exactly the case where the leak is pointless.

### A4. MEDIUM: the publication path has no failure edges, and host mutations precede the busy check

Section 7.3 does `publish_ready(dst)`, `unpublish(victim)`, `_publish_slots()`, then records
`retire_event`. There is no edge for: `wait_event` raising (A2), `_publish_slots` raising partway,
the `retire_event` record failing, or `retire_event.query()` raising. Section 10's table covers
COPYING failures only. If `_publish_slots` is deferred for `publish_busy` after the host lists
already say READY and DRAINING, host and device disagree and the DRAINING victims have no
`retire_event` to wait for: the busy check must precede the host mutations, and that is not stated.

A safe rule, which should be written down: a DRAINING slot is freed only after a recorded and
completed `retire_event`; on any publication failure it stays DRAINING.

### A5. MEDIUM: which transitions publish and synchronize is unspecified, and the naive one stalls

In `expert_hot_cache.py`, `reserve`, `begin_loading`, `publish_ready`, `cancel` and `retire` each
run inside `_slot_batch()`, whose exit calls `_publish_slots()` when dirty. `_publish_slots` does
`_slot_upload_event.synchronize()` whenever the previous upload was recorded, and
`_slot_upload_recorded` is set True at `:246` and never reset. Within one observer call (which runs
after the forward's replay is enqueued), a second publish waits on an event recorded after the first
publish's upload, which sits behind that replay in stream order: a host block until the GPU finishes
the current forward. That is the stall Task 8 removes.

The existing `stage_reassign` avoids it with `_slot_batch(publish=False)`; the document never
mentions `_slot_batch` (no occurrence). The section 11.1 spies would catch it, but only in a
GPU-capable run (B2). Section 5 should say that transitions before PUBLISHED do not publish (the
device consumes only READY: `ready = rows[0] == READY`, `:237`) and that PUBLISHED publishes once.

### A6. MEDIUM: LEASED hold time, undrawn edges, and a contradiction with sections 6.3 and 10

- Section 6.3: "all-or-nothing per layer-wave... partial admission would let one row hold a slot
  while waiting on another". Section 10: "that row ADMITTING -> CANCELLED; the wave proceeds with the
  rows it has". Section 11.2's Priority test: "rows already complete are published whole and the
  rest released". The first says all-or-nothing; the other two say partial waves. `admit_state` is
  per row and `state` is per ticket, with no rule for when a ticket leaves ADMITTING.
- Enqueue can fail after LEASED: `ring_full`, `plan_busy`, `byte_cap` and `quarantine` are
  enqueue-time conditions, and in M2 PLANNED and enqueue are polls apart. The diagram has no
  LEASED -> DEFERRED or CANCELLED edge (only section 6.4's prose does), and section 6.3's "deferral
  consumes no resources" holds only if they are pre-checked at PLANNED, which is stale by enqueue.
  So section 6.4's "bounded by one poll interval" has no basis.
- M2 race, unaddressed: the service thread grants ADMITTING -> LEASED while the scheduler cancels on
  TTL. A lease granted after the cancel lands in the `poll_admits()` results of a CANCELLED ticket,
  and nothing in section 5 says to release it. It needs a rule (drain results for cancelled ids and
  release them); otherwise the lease is unowned.

### A7. LOW: diagram versus prose, and smaller gaps

- ABANDONED has no outgoing edge in the diagram (section 10: release and discard on event
  completion; an erroring event is not drawn).
- COPIED -> CANCELLED (section 10, TTL and teardown) and LEASED -> CANCELLED are not drawn, so
  section 11.2's "every arrow of section 5.2" never tests them.
- "Cancel in PLANNED" is unreachable by the document's own threading rule (one thread takes every
  arrow in one step, so PLANNED never survives to be cancelled) unless PLANNED is a resting state,
  which the diagram does not say.
- Edges that return a destination to FREE do not say `set_hot` is re-pushed. `cancel` clears
  `slot_to_expert`, but hot is pushed only by the residency listener
  (`_notify_residency_listeners`, after a residency update), and `take_slot_locked` skips hot rows
  with no fallback (`exl3_ram_miss_host.cpp`), so a stale hot flag keeps a RAM row unevictable until
  the next push. Bounded by a wave; the same family as A3.
- "One ticket in flight per layer" is derived from the plan claim, which ends at COPIED
  (`is_complete` -> `release_plans`), not from the state machine. A second wave for the same layer
  can start while the first is COPIED, PUBLISHED or awaiting DONE. Not prohibited, not modelled, not
  tested.
- The victim ticket: `_ticket_matches` requires `slot_to_expert[slot] == ticket.expert_id`, and
  section 5.1 sets DRAINING `slot_to_expert = -1`. The victim's original ticket therefore cannot
  authorise DRAINING -> FREE; R-C's "re-check `_ticket_matches` for the victim" needs a second
  ticket `(slot, -1, generation)` from `ticket_for_slot` (which works for non-FREE slots).
  Workable, unstated.

### A8. LOW-MEDIUM: A1's verification cannot be done as specified

OPEN 1 says "at attach, record the replay stream and the observer's `current_stream()`; refuse the
async mode if they differ". A graph replay runs on the current stream at `replay()` time, which is
not known at attach (manager construction precedes any replay). The check has to be per replay or per
forward. This matters more than its severity suggests: DRAINING and `retire_event` correctness depend
on it (a `retire_event` on the wrong stream frees slots with consumers still queued), so the load-
bearing assumption's only guard is unimplementable as worded.

## Checked, no finding (section 5)

- **The DRAINING ordering itself.** Publication and `retire_event` are recorded on the serving stream
  after the enqueue of every replay that can read the old map. The victim reaches FREE only after
  `retire_event` completes. A replay enqueued afterwards reads the new map. The destination rule
  admits FREE slots only. Under A1 that is exact.
- **The published map excludes DRAINING.** Not for the reason section 5.1 gives (`slot_to_expert =
  -1`): `_publish_slots` sends every non-READY slot to a dump index (`expert_hot_cache.py:237`).
  `free_slots` in `stage_reassign` is by state FREE, so DRAINING is not handed out. `_reserve`'s
  "nonresident expert" test uses `slot_to_expert` of non-FREE slots, so a re-promoted expert in a
  spare while its old slot drains is permitted, as claimed.
- **`consumer_complete` is not relied on by section 5.** The document mentions it only in section 4
  as a correction. `retire` accepts only READY (`_ticket_matches(ticket, READY)`), so it cannot serve
  DRAINING -> FREE. The first review's finding does not leak into section 5. The risk runs the other
  way: an implementer who reuses `retire(..., consumer_complete=True)` for a new step passes a fresh
  nominal True. Say that DRAINING -> FREE is a new method that takes a completed event.
- **R-A and R-C's stale-destination handling** hold given the ring semantics, apart from A2.
- **Scope of the claim.** The invariant "the destination is FREE, unreachable by any published
  mapping" is per ticket. `assign_prefetch` (dense NVFP4, declared out of scope in section 1) still
  retires READY slots with `consumer_complete=True` (`expert_hot_cache.py:774`; `plan_prefetch` offers
  READY slots as writable). It is off in the EXL3 arms; the async mode should refuse it explicitly
  rather than assume it.

## Claims in the two sections that are real

Not everything is a finding. These hold: the executor is fakeable; `test_exl3_ram_miss_graph_gpu.py`
exists in `test/manual/dsv41`; the "More misses than slots" comment quoted in the M0 test exists
(`expert_stream.py:331`); `torch.cuda._sleep` is already used in this repo; `Exl3RamMissHost.pause`,
`host_use`, `_drain_device` and `wait_for_slot_publication` exist under the names given; the fault
injector's `delay_s` and `delay_after_demands` exist (`ops/moe/exl3_ram_miss.py:533-537`). No claimed
defect in these sections was found to be false.

## Not checked

Section 8 and 9's proposed native API (R1-R8) beyond what section 5 leans on; `LEASE_PROTOCOL.md`
(uncommitted when this was written, not read); `lease_model.py` (its coverage is taken from the
document's own account); `scheduler.py` and the overlap and non-overlap scheduler behaviour (A1
rests on reasoning); whether CUDA graph replays share the observer's stream in every mode (OPEN 1);
the M3 readback; the section 12 arms; the lease and eviction interplay in `take_slot_locked` beyond
the `hot` skip; PyTorch's sync-debug behaviour on pageable copies. No test, build or GPU was run.

## Suggested order of fixes

A2 and A3 first (each is one design edit and needs no decision). Then A1 (decided above). Then B1 and
B2, which change what the tests are, and the negative-control requirement for every safety test.
A4, A5 and A6 are edits to section 5 and the section 10 table; B3 to B5 and the rest of section 5
follow.
