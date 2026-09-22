# Source leases and device acknowledgements: protocol design (Task 5)

Status: design only. No code was written or run for this document, the GPU was not
used, and nothing here has been model-checked (section 19 lists that as an open
item). It was written against `dsv41` HEAD `bea06e789c`. Another session is editing the
native host service, so line numbers below will drift; every citation gives a symbol
so it can be found again.

Plan reference: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`,
"Task 5", "Task 6", "Global constraints", "Proposed interfaces and ownership".

Legend used throughout:

- **[E]** exists in the tree today; the citation names the symbol.
- **[P]** proposed by this document; it does not exist.
- **[OPEN n]** something I could not determine or could not decide; listed in section 19.
- **[DECIDE n]** a choice I made that the owner may want to overturn; listed in section 19.

Files cited (all under `python/sglang/`):

| Short name | Path |
|---|---|
| `host.cpp` | `kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` |
| `device.cuh` | `kernels/jit/csrc/moe/exl3_ram_miss.cuh` |
| `transfer.cuh` | `kernels/jit/csrc/moe/expert_cache_transfer.cuh` |
| `ops.py` | `kernels/ops/moe/exl3_ram_miss.py` |
| `srt_ram_miss.py` | `srt/layers/moe/exl3_ram_miss.py` |
| `row_plan.py` | `srt/layers/moe/expert_row_plan.py` |
| `host_tier.py` | `srt/layers/moe/expert_host_tier.py` |

---

## 1. Why this task exists: today's safety is temporal exclusion, and it already fails

### 1.1 What actually protects a pinned RAM slot today

There is no ownership of a pinned slot. The service's per-slot state is
`Tier::state` in `host.cpp` (`kFree`, `kLoading`, `kReady`), plus `stamp` (LRU),
`hot` and `slot_to_expert`. There is no reference count and no slot generation. The
only counter that moves when the map changes is `kVersion`, which serves Python cache
invalidation, not the device.

A slot is not recycled under a GPU reader today only because four unwritten
assumptions hold together:

1. **Graph replays are serialized.** One graph producer, one stream. The post kernel
   updates `state[kPosted]` with a plain read-modify-write (`exl3_ram_miss_post_kernel`),
   which is only correct with one replay at a time.
2. **The device sits in an armed wait before it gathers.** `exl3_ram_miss_wait_kernel`
   polls `demand_done`, and only then translates experts to slots.
3. **The service is one thread serving one request synchronously.** `RamTier::serve`
   returns only after `RowReader::read` has emptied its ring; nothing else evicts
   meanwhile.
4. **Each layer has its own tier**, so an eviction for row R+1 (an advisory) cannot
   touch row R's slots.

A fifth rule closes the remaining gap: an *unarmed* record is touch-only
(`touch_request`); it never evicts, because "the device may already be gathering any
mapped slot". That rule is sufficient only because of a premise that is stated nowhere
near it: advisories exist only when `advise != 0`, and `advise != 0` arms *every* record
(`armed = need_count > 0 || advise != 0`, `device.cuh`). So unarmed records occur only
with advisories off, and then nothing else evicts on a row the device is gathering
(section 15).

I could not construct a non-fault interleaving in today's code that violates this
exclusion. It is real. But it is implicit, it is nowhere stated as an invariant, and
Task 6 (rows delivered to the GPU while later reads are still outstanding) and Task 8
(promotions on their own stream) each remove one of the four assumptions.

### 1.2 Where the exclusion already fails

These are defects in the code as it stands, each with the symbol to check.

**D1. The timeout path gathers anyway.** `exl3_ram_miss_wait_kernel` (`device.cuh`)
on timeout sets `ok = false` and calls `raise_fatal`, then falls through to the
translate loop (`host_rows[i] = slot`), and finally `keep[0] = 0`. Nothing stops the
copy. `Exl3RamMissRowBackend.translate` (`srt_ram_miss.py`) issues post and wait; it
inherits `PinnedTierRowBackend.post` (`row_plan.py`), which then calls
`copy_expert_row_segments_gpu(self.segments[tag], self.host_rows, plan.slots, plan.count)`
with the full `plan.count`. The copy kernel therefore reads pinned slots for lanes whose
request failed, while the service may be inside `serve()` evicting one of those very slots
(`take_slot_locked` unmaps, then overwrites). `keep = 0` drops the layer's output; the
watchdog aborts the process. That is containment by two side effects, not a design
guarantee. The plan's Task 6 says it in one sentence: "setting the fused MoE's `keep` to
zero alone does not protect an earlier gather."

**D2. The device resolves through the mutable slot map.** The wait kernel reads
`slot_map` with `ld_volatile` into device `host_rows`, and the copy then indexes the
slabs by `host_rows`. The map is written by the service (`publish_map`) at any time. What
breaks is specified in section 10; in one line, an entry that changes between the
wait's read and the gather's use makes the copy read another expert's bytes.
**Severity today: latent, not live.** Today the map cannot change in that window without a
fault (the temporal exclusion above), and after a timeout `ok` is false and `keep` is 0, so
the counters do not report success. D2 becomes a live defect when Task 6 lets the service
evict while an earlier lane's copy runs, or when a non-graph reader (Task 8) shares the
slabs. The model (section 18.3) shows the same: without the exclusion, wrong bytes are
accepted.

**D3. No GPU-to-CPU acknowledgement exists.** The service cannot learn that a gather
has finished. The only "acknowledgement" in the code base is `demand_done` for an armed
record, which is CPU-to-GPU. (The handoff calls it Option F's acknowledgement,
`EXL3_COPY_PIPELINE_HANDOFF.md`, "the armed condition requires a CPU acknowledgement".)

**D4. The wait kernel cannot be interrupted.** In `exl3_ram_miss_wait_kernel` the page's
fatal word is checked once, on entry; the poll loop checks only `demand_done` and the
clock. After a fatal is raised elsewhere, or after a shutdown, a spinning wait holds its
stream until `timeout_ns`, which defaults to 2000 ms
(`SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`, `environ.py`). Shutdown, or a fatal raised elsewhere, is
therefore bounded by the wait timeout rather than prompt; the design's `Header.shutdown`
poll (section 7.3) makes it one poll interval.

**D5. Shutdown does not establish that GPU readers are done, and does the opposite of
quarantine.**

- `Exl3RamMissService.shutdown()` (`srt_ram_miss.py`) calls `host.stop()` and then
  `tier.close()`, which reaches `release_host_slabs` (`host_tier.py`) and
  `_cuda_host_unregister` (`mem_cache/pool_host/common.py`). There is no device
  synchronization and no check that any kernel has finished reading.
- When `cudaHostUnregister` fails, `_cuda_host_unregister_ranges` logs a warning,
  records the range as failed, and returns. The slab tensor is then dropped and torch
  frees the storage. Memory that is still registered, and possibly still being read,
  goes back to the allocator.
- `shutdown()` is a **test-only entry point**: its only callers are
  `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py` and
  `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`. The production teardown path is
  the module-level `@atexit.register _stop_live` in `ops.py`, which calls
  `Exl3RamMissHost.stop()` for every live host (thread stop, then the finalizer
  `exl3_ram_miss_close`; `self._close.atexit = False` because `_stop_live` owns exit).
  That path stops the service thread and closes the C++ handle. It does not synchronize
  the device. The slab tensors survive only because `Exl3RamMissHost.tables` keeps them
  alive, and interpreter teardown then frees them with no device barrier.
- **A second exit-time actor unregisters the slabs, and it runs in every process.**
  `ExpertPinnedHostCache.__init__` (`expert_stream.py`) creates
  `self._release_slabs = weakref.finalize(self, release_host_slabs, registered)` with the
  default `atexit=True` ("Unregisters the slabs when the cache is collected, at exit, or on
  close()"). So at ordinary process exit `cudaHostUnregister` is called on every slab whether
  or not anyone called `shutdown()`, and it is not ordered against `_stop_live` or against
  GPU work by anything I could find **[OPEN 13]** (the relative order of the two atexit
  callbacks depends on registration order: `weakref.finalize` installs its exit hook when
  the first finalizer is created, `_stop_live` when `ops.py` is imported; I did not
  determine which is first). Any quarantine has to *detach this finalizer*; otherwise it
  still fires.
  An independent review answered the ordering: `weakref.finalize`'s atexit hook is
  registered before `_stop_live`, and `atexit` is last-in first-out, so `_stop_live` runs
  **first** and the slab-unregister finalizers run **second**, with no device barrier between
  them. Which exit path production takes: corrected by reading the launcher (see the finding at the top of section
  14). The scheduler is **not** unconditionally SIGKILLed: on a graceful shutdown (`ShutdownReq` sets
  `gracefully_exit`) `run_scheduler_process`'s `finally` runs `Scheduler.release_host_resources()`, and only the
  exception path can SIGKILL. An earlier version of this bullet cited a docstring in `exl3_stream_trace.py` for the
  opposite and is withdrawn. So the exit-time unregister is a live hazard on a normal exit, and whether every production
  teardown reaches the graceful path (`Engine.shutdown()` kills its children's tree) is untraced (OPEN 18).
- Design consequence: this document designs for the `_stop_live` path. If the
  design needs a production caller of an orderly shutdown to exist, that is a
  **requirement** (section 14.6), not an assumption.

**D6. The demand and advisory sequence wraps are handled differently by the two sides.**
The post kernel and `sim_post` skip sequence 0 on wrap (`if (seq == 0) seq = 1`), and
`RamTier::open` skips it when it seeds `next_demand_`. But `pump_demand` and `pump_advice`
advance with a bare `next_demand_ += 1u` / `next_advice_ += 1u`. At the wrap the service
expects seq 0 after 0xFFFFFFFF, the device posts seq 1, the service finds `head` reached,
reads the record slot `(0-1) % 16 = 15`, fails the seqlock read, counts one `kOverruns`
(or, on the advisory ring, one `kAdvisoriesSkipped`), begins a stage record, and stores
`demand_done = 0` (or `advise_done = 0`). It then carries on with seq 1.

Confirmed by execution by the independent reviewer (heads seeded at 0xFFFFFFFD, requests
posted through the existing `sim_post` and pump: one phantom iteration at expected seq 0,
`demand_done` stepping 0xFFFFFFFF, 0, 1). The signature depends on the page: on a used page
the phantom read fails and `overruns` is 1, as above; on a never-used page record slot 15
holds seq 0, so the phantom read *succeeds* as an empty touch record and `overruns` stays 0.
Same phantom iteration, different signature, so a test must assert on the number of requests
handled, not only on `overruns` (the wrap test does both). The device's
`reached(0, 0xFFFFFFFF)` is true under the signed compare, which is why waiters are
unaffected.

Severity, by reading: **a spurious counter and a spurious trace record once per 2^32
requests, not a failed request.** Nothing can be waiting on sequence 0, because the device
never posts it, and I found no consumer of `kOverruns` outside the counter list in
`ops.py` and tests (a grep over `python/`, `test/`, `analysis/` and `scripts/`). The
comment in `pump_demand` that a pending status "makes a waiting layer fail stop" describes
the *lapped-record* case, where a real waiter exists, not this one. The model
(section 18.3) agrees: with a service that does not skip 0 it finds a phantom sequence
and no safety violation.

**The lap-skip resume can also land on zero.** Found by execution by the instrumentation
agent, not by me: `next = head - kDemandRecords + 2u` evaluates to 0 when `head - next` equals
the ring size (seed 0xFFFFFFFD, post 16 demand or 64 advisory records), and it failed before
the fix. The fix that landed (`85cdbf9382`) is a `skip_zero()` helper at four sites: both
increments and both lap-skip resumes. My model reproduces this shape independently: in a
world with three unarmed records posted across the wrap, only a service whose *increment*
skips 0 still finds a phantom through the lap resume (`skip0_lap=False`), and the fixed service
is clean. It is the same pattern as the epoch hole the model found in this document's own
draft (11.3): a lap skip crossing the wrap without passing through the value being skipped,
once in today's code and once in the proposed protocol, found by different methods. (The
model had missed the fresh-page signature until it was made to flag *any* handling of seq 0,
not only a failed record read; the same lesson as the test that asserted on `overruns` alone.)

The four signatures are now observed, not argued (by the instrumentation agent, running the
merged wrap test against the code before the fix): demand ring on a used page, `overruns = 1`;
demand ring on a fresh page, `touch_only = 1` with `overruns = 0`; advisory ring on a used
page, `advisories_skipped = 1`; advisory ring on a fresh page, `advisories = 5` with
`advisories_skipped = 0`. The last is why a test that asserts on the skip or overrun counter
alone passes on a fresh page: my own first draft did exactly that, which is why the merged
test asserts on the done word and on what was served instead.

The transferable lesson, stated once: **a ring with a reserved value must skip that value at
every place the counter can move, and a resume-after-lap is such a place.** The increment
was fixed on one side; the lap resume jumped over the increment and could still land on the
reserved value. This document's own draft had the same shape with epochs (11.3): a lap skip
crossing the wrap without passing through the value being tracked. Anyone writing a ring
with a sentinel should add "every jump, not just every step" to their checklist.

It is still a defect: the two sides disagree on one value, and any code that keys
something by `next_demand_` inherits the disagreement. This document's own design avoids
that dependency (section 11.3). The test that drives both rings through the wrap, on both
a used and a fresh page, is `test/registered/unit/kernels/test_exl3_ram_miss_wrap.py`. That
file is the reviewer's rewrite of my first draft (which asserted only on `overruns` and so
would have passed on a fresh page for the advisory ring); I could not run either version
myself (the laptop's interpreter has no `transformers`, and the project interpreter is on
divix01, where I was told not to touch the worktrees). **[OPEN 1]** (now narrowed: the
phantom itself is confirmed by execution): confirm that the committed test fails on today's
`pump_demand`/`pump_advice` and passes with the one-line fix.

**D7. Lane capacity is silently 8.** The post kernel reads `min(count, kMaxIds)` lanes
with `kMaxIds = 8`; the wait kernel translates up to `host_rows.numel()` lanes. A plan
with more than 8 lanes has lanes that are never requested from the service, and the wait
kernel then finds them missing and raises a fatal via `unserved_misses`. DSV4.1 is top-6
(per the independent review), so bs 1 gives at most 6 lanes and today's decode is safe. But
`graph_gather_rows` is `tokens * top_k`, and `Exl3RamMissService.attach` passes it to
`Exl3RamMissRowBackend` as the capacity with **no `<= kMaxIds` check**, so any multi-token
graph exceeds 8. **Required change (not an open question): enforce `graph_gather_rows <=
kMaxIds` (the lease block's `lanes`) at attach and refuse otherwise.**

### 1.3 The shape of the failure this design exists to prevent

This codebase already shipped a bug in which the reader published bytes that were never
read, while every observable agreed with success. The failure shape here is the same:

> A GPU reader copies from pinned slot `s` after slot `s` has been recycled for a
> different expert, and `keep`, `ram_miss`, the status word, byte counters and all
> tests in the current suite report a correct, served request.

Every design choice below is judged against that shape. The invariant that prevents it
is stated in section 2 and enforced by named mechanisms; and, because a proof by
construction can still be wrong, section 6.5 adds a *detector* that turns a violation
into a fatal error instead of a silent success.

---

## 2. The invariant

**I1 (no recycle under a reader).** Let a *lane* be one planned expert of one request
(section 3). If a GPU copy kernel reads pinned slot `s` of row `r` on behalf of lane
`k`, then the bytes of slot `s` are unchanged from the moment the service published
readiness for that lane until the service has observed either that lane's
acknowledgement, or a terminal record that names that lane in its skipped mask.

It holds because of six mechanisms; each is specified below and each has a test in
section 18:

| # | Mechanism | Section |
|---|---|---|
| E1 | The service never evicts, reassigns, releases or rewrites a slot with `leases > 0`. The counter is service-private and single-threaded. | 8 |
| E2 | A lease is decremented only on (a) the lane's device acknowledgement, (b) a device terminal record whose skipped mask names the lane, or (c) an explicit host release by a non-graph holder. Never on a timer, never on the service's own guess. | 8, 13 |
| E3 | The device publishes a lane's acknowledgement only from a kernel that starts after the copy kernel that read the slot has completed. | 6.4 |
| E4 | The device publishes a terminal skip for a lane only when no copy for that lane can start or is running. | 13 |
| E5 | If completion cannot be established (CUDA error, synchronization timeout), leases are never decremented and the memory is quarantined until process teardown. | 14 |
| E6 | The acknowledgement kernel re-checks the slot generation after the copy. A mismatch makes the request fail and is counted; it cannot end as a success. | 6.5 |

**I2 (identity).** The bytes a lane copies belong to the expert the lane requested:
the device compares the row result's expert with the lane's own planned expert before
copying, and the service compares the lane's expert with `slot_to_expert[slot]` before
publishing.

**I3 (exactly one release).** Each lease is released exactly once. One acknowledgement
releases one lease; it cannot release another lane's source.

**I4 (fail closed).** A skipped copy cannot produce a `CONSUMED` acknowledgement. The
copy and the acknowledgement are driven by one device word (`go_count`), written once
per replay by the kernel that decides, and zero at entry to that kernel.

**I5 (the worker does not wait).** The service never blocks on an acknowledgement.
Retirement is a polled step; a request that cannot get slots because of outstanding
leases is *deferred*, not failed and not waited on.

---

## 3. Vocabulary and sizes

- **Request**: one demand record posted by the device (one streamed layer, one token).
  Identified by a **request generation** `G` (section 11). Occupies **request slot**
  `idx = (seq - 1) % R` with `R = kDemandRecords = 16` [E].
- **Lane**: index `i` in `[0, count)` into the request's planned experts; `count <= L`
  with `L = 8 = kMaxIds` [E]. Lane `i` owns its own GPU destination row and its own copy.
- **Lease**: one `(row, host_slot, slot_generation)` reference held by one lane of one
  request. One lease per consuming lane (section 9).
- **Row result**: the immutable record the service publishes for a lane: which expert,
  which slot, which slot generation, which status.
- **Acknowledgement**: the device's release-store that says "lane `i` of request `G`
  finished reading its source".
- **Terminal record**: the device's release-store that says "no lane named in this mask
  of request `G` will read a source, ever".
- **Slot generation**: a per-`(row, slot)` counter bumped every time the slot's bytes are
  about to change (section 6.5). Used only as a detector.

---

## 4. Control block layout

### 4.1 A separate pinned region, page ABI untouched [DECIDE 1]

The lease block is a **new pinned host region beside the existing 10304-byte request
page**. The page (`PAGE_BYTES`, `kDemandRing`, `kAdviseRing`, record offsets) is not
changed by one byte. Its layout is currently held in agreement between `host.cpp`,
`device.cuh` and `ops.py` by `test_exl3_ram_miss_device_args`; that test is one of the
few guards against silent ABI drift across the language boundary, and this design does
not put it at risk to save an allocation. The new block's constants join the **same**
test (section 18), so agreement checking scales instead of forking.

Allocation: pinned, zero-filled, base address 4096-aligned. The constructor **adds** a
check that refuses a block whose `data_ptr() % 4096 != 0`; there is no alignment check today
(`Exl3RamMissDevice.__init__` checks numel, dtype, host, contiguous and pinned, and nothing
about alignment), so an implementer must not assume one exists. It also refuses a block that
is not pinned for a CUDA device, as `__init__` does for the page. **[OPEN 3]**: `torch.zeros(...,
pin_memory=True)` (what `new_page` uses) does not obviously guarantee 4096 alignment. If
not guaranteed, allocate `bytes + 4096` and slice, as `allocate_host_slab` does for its
slabs.

Cache-line rule: **every 128-byte line has exactly one writer.** Service-written and
device-written words never share a line. (128 B, not 64 B: it is twice the PCIe/CPU
line, so no adjacent-line prefetch pairs a service line with a device line.)

### 4.2 Word encoding

All words are naturally aligned. Multi-byte words are little-endian, as on the x86 host
and the GPU.

- **G56**: a request generation in the low 56 bits of a 64-bit word
  (`epoch << 32 | seq32`, `epoch` 24 bits). Zero is never a valid `G56`, because `seq32`
  is never 0 (the existing skip-0 rule, [E] `if (seq == 0) seq = 1`).
- **Tagged word**: `u64 = (tag << 56) | G56`. The tag is 8 bits: a status, an outcome, or
  a reason. A tagged word is *the* publication point of its record; everything else in
  the record is written before it.

A tagged word with tag 0 is "not published". The 24-bit epoch wraps after
`2^24 * 2^32 = 7.2e16` requests; I treat that as never (section 11.3).

### 4.3 Region map

Offsets are from the block base; each area starts on a 4096-byte boundary.

**Area H: header (service-written), offset `0x0000`**

| Offset | Size | Field | Writer / lifetime |
|---|---|---|---|
| 0x00 | u32 | `magic = 0x4C534531` ("LSE1") | service, at open, immutable |
| 0x04 | u32 | `abi_version = 1` | service, at open, immutable |
| 0x08 | u32 | `ring = 16` (`R`) | service, at open, immutable |
| 0x0C | u32 | `lanes = 8` (`L`) | service, at open, immutable |
| 0x10 | u32 | `rows` (streamed layers) | service, at open, immutable |
| 0x14 | u32 | `shutdown` (0 running, 1 closed to admission) | service, release-store, sticky |
| 0x18 | u32 | reserved, zero (an earlier draft held a service-maintained epoch here; see 11.3) | |
| 0x1C | u32 | reserved, zero | |
| 0x20 | u32 | `slot_gen_offset` (bytes from base to the first slot-generation word) | service, immutable |
| 0x24..0x7F | | reserved, zero | |
| 0x80 | `rows * 8` B | `RowTable[rows]`: `{u32 slot_gen_base; u32 capacity}` | service, immutable |

Immutable words are written before the device is constructed, so the device may read
them without ordering. The one mutable word is `shutdown`.

**Area S: service-written, offset `0x1000`**

`RowResult[R][L]`, 32 bytes each, `R * L * 32 = 4096` bytes:

| Offset in record | Size | Field |
|---|---|---|
| 0x00 | u64 | `ready`: tagged word. Tag 1 = READY, 2 = FAILED. **Stored last, with release.** |
| 0x08 | u32 | `slot_generation` at grant time |
| 0x0C | i32 | `host_slot` (pinned slot index in the row's tier; -1 when FAILED) |
| 0x10 | i32 | `expert` |
| 0x14 | u16 | `row` |
| 0x16 | u16 | `lane` |
| 0x18 | u32 | reserved, zero |
| 0x1C | u32 | reserved, zero |

Record address: `0x1000 + (idx * L + lane) * 32`. Four records per 128-byte line; a
request's 8 lanes fill two lines.

`SlotGen[]`: at `slot_gen_offset` (the next 4096 boundary after `RowResult`, i.e.
`0x2000`): `u32` words, one per `(row, slot)` at `slot_gen_base[row] + slot`, the whole
array padded up to a multiple of 4096. Service-written, release-stored. Size: the sum of
row capacities times 4 bytes. I do not know the production capacity **[OPEN 4]**; it is
tens of KiB at most for any plausible tier.

**Area D: device-written, starts at the next 4096 boundary after `SlotGen[]`
(call it `D0`)**

| Offset from `D0` | Size | Structure | Writer |
|---|---|---|---|
| 0x000 | 16 x 64 B | `LaneRequest[R]`: `{u64 gen (tagged word, tag 1); u32 count; u32 row; i32 expert[8]; u32 reserved[4]}` = 8+4+4+32+16 = 64 B | device post kernel |
| 0x400 | 16 x 8 x 8 B | `LaneAck[R][L]`: one tagged u64 per lane; tag 1 = CONSUMED, 2 = VIOLATED | device ack kernel |
| 0x800 | 16 x 16 B | `Terminal[R]`: `{u32 skipped_mask; u32 reason; u64 gen (tagged word)}`; the tagged word is stored last (offset 8) | device wait/finalize kernel |

`LaneAck[idx]` is exactly one 64-byte line (8 lanes x 8 bytes); two request slots share a
128-byte line, which is fine because both halves have the same single writer (the
device). Total block size is `D0 + 0x900` rounded up to 4096.

### 4.4 Which of `{request_generation, slot_generation, host_slot, status}` lives where

The plan's proposed control contract publishes these four per lane. Placement:

| Field | Where | Notes |
|---|---|---|
| `request_generation` | low 56 bits of `RowResult.ready` (and of every other tagged word) | it is the validity key: a `ready` word whose generation is not the one the device expects is "not for me" |
| `status` | tag byte of `RowResult.ready` | published atomically with the generation, so readiness and status can never disagree |
| `slot_generation` | `RowResult.slot_generation`, plain word, before `ready` | also mirrored live in `SlotGen[]` for the detector |
| `host_slot` | `RowResult.host_slot`, plain word, before `ready` | -1 when FAILED |

Readiness is published **last**: the tagged `ready` word is the final store of the
record. It is a `st.release`-class store on the service side and an acquire load on the
device side (section 6).

---

## 5. Single-writer ownership

"Reader may assume" is the reader's licence: what it can rely on after the acquire that
makes the word visible, and nothing else.

### 5.1 Words in mapped memory

| Word / structure | Single writer | Written when | Reader | Reader may assume |
|---|---|---|---|---|
| Header immutable fields | service | before device construction | device, Python checks | constant for the block's life |
| `Header.shutdown` | service | once, at stop | device wait/post kernels | after seeing 1, the service issues no new leases and will not publish further `RowResult`s |
| `RowResult[idx][lane]` payload | service | after the previous generation of that request slot retired, before `ready` | device | nothing, until it has acquired `ready` with the expected generation |
| `RowResult[idx][lane].ready` | service | last store of the record | device (wait kernel) | if `gen == G` and tag == READY: payload is complete, the lease exists, and the slot's bytes are final and immutable until this lane's retirement |
| `SlotGen[row, slot]` | service | before the first byte store that changes the slot | device (ack kernel) | a value different from the leased generation means the slot's bytes were, or are being, rewritten |
| `LaneRequest[idx]` | device post kernel | before `demand_head` is stored | service | after acquiring `demand_head` and passing the seqlock re-check, lane `i`'s expert is `expert[i]` for request `G` |
| `LaneAck[idx][lane]` | device ack kernel | after the copy kernel for that lane completed | service | tag CONSUMED / VIOLATED with `gen == G`: that lane's copy finished; nothing else. Not an ordering statement about any other word |
| `Terminal[idx]` | device wait/finalize kernel | once per request, only if some lane is skipped | service | `skipped_mask` lanes will never read a source. Lanes not in the mask may still be copying, and will acknowledge |

Nothing in the lease block is ever written by Python, and nothing is written by two
actors. **Words are write-once per generation.** They are never cleared; validity is the
generation compare, not a flag.

The existing page words keep their owners **[E]**: `demand_head`, `fatal` and the
demand-ring records by the device; `demand_done`, `advise_done`, `busy_seq`,
`heartbeat` and each record's status by the service.

### 5.2 Service-private state (not mapped, not visible to the device)

Under `RamTier::mutex_`, added to `Tier` **[P]**:

- per slot: `leases` (u32), `slot_generation` (u32, mirrored to `SlotGen[]`);
- per request slot `idx`: an `Outstanding` entry `{G, row, count, lane[8] {slot, slot_generation, state}}`, with `state` in `{NONE, GRANTED, ACKED, VOID}`; this is the "generation-qualified per-request record" of the plan's "Proposed interfaces and ownership" section.

The lease counters are the eviction gate. The device never reads or writes them. That is
deliberate: the eviction gate is single-threaded and needs no cross-agent atomics.

---

## 6. The release/acquire mechanism

"Ordinary Python stores are not its implementation" (plan). Nothing in Python touches
the lease block. Every publication below is a C++ atomic or a PTX instruction.

### 6.1 Host side (service thread) [E] primitives

`host.cpp` already has the validated pair:

- `load_acquire(addr)` is `__atomic_load_n(..., __ATOMIC_ACQUIRE)`;
- `store_release(addr, v)` is `__atomic_store_n(..., __ATOMIC_RELEASE)`;
- `_mm_sfence()` precedes publication after packing stores, because the split's `memcpy`
  stores are not assumed to be ordered by a release store alone (see the `_mm_sfence()` calls
  in `serve` and `pump_demand`).

New 64-bit forms **[P]**: `load_acquire64` and `store_release64` on `uint64_t*`, same
builtins. The block is naturally aligned so 64-bit accesses are single atomic accesses on
x86-64.

Publication order for one lane, service side **[P]**:

1. bump `SlotGen` if the slot was (re)loaded (done earlier, section 6.5);
2. `leases[slot]++` (private, under `mutex_`);
3. plain stores: `slot_generation`, `host_slot`, `expert`, `row`, `lane`;
4. `_mm_sfence()`;
5. `store_release64(&ready, tag | G56)`.

Only after all lanes of a Task 5 request are published does the service do the existing
tail: `_mm_sfence()`, `set_status(record, kServed)`, `store_release(demand_done, seq)`
(`pump_demand`, `handle_demand`).

### 6.2 Device side [E] primitives, [P] additions

`device.cuh` already has `ld_acquire_sys` (`ld.acquire.sys.global.u32`) and
`st_release_sys` (`st.release.sys.global.u32`). **[P]** 64-bit twins:
`ld.acquire.sys.global.u64` and `st.release.sys.global.u64`. No other primitives are
introduced.

### 6.3 Device-side publication order

*Post kernel* (`exl3_ram_miss_post_kernel`), for a request with `count > 0` **[P]**:

1. write `LaneRequest[idx]` as the existing `write_record` does: `gen = 0` (invalidate),
   `__threadfence_system()`, payload `{count, row, expert[]}`, `__threadfence_system()`,
   tagged `gen` word last;
2. the existing `write_record` for the demand record and the `st_release_sys(demand_head)`.

The `LaneRequest` seqlock is the same shape as `write_record` and is read by the same
pattern as `read_record`: acquire `gen`, read payload, `atomic_thread_fence(acquire)`,
re-read `gen`, and treat a change as a lapped record. Cost: two more
`__threadfence_system()` per post. I have not measured it **[OPEN 5]**; the existing post
already pays three or four.

*Wait kernel*, described in section 7.3. *Acknowledgement kernel*, section 7.4.

### 6.4 Why the acknowledgement is a separate kernel, and why that is enough (E3)

The dangerous reordering is: host sees the acknowledgement, rewrites the slot, and the
GPU load of the *old* bytes is still outstanding or is satisfied later from the new
bytes. So the acknowledgement store must be ordered after every load of the source
bytes has completed.

The mechanism: a **separate, single-block kernel launched after the copy kernel in the
same stream (or as its successor node in the captured graph)**. By CUDA stream and graph
semantics the successor does not start until the predecessor has completed, and a
thread's loads have returned their data by the time the thread retires (the destination
store depends on the loaded value). So every source load of the copy kernel has
completed before any instruction of the acknowledgement kernel executes. The
acknowledgement kernel then does `st.release.sys.u64` on the `LaneAck` word.

Why not fuse it into the copy kernel's tail: a fused ack would need a grid-wide "all
loads of this lane completed" condition (a last-block-done counter plus a system fence
per lane). That is more machinery, harder to argue, and buys only one kernel launch. Task
6 may revisit it after measurement; if it does, this section's argument has to be redone
for the fused form.

Limit of this argument: it rests on the CUDA stream/graph ordering guarantee and on
loads having returned at thread exit. It does not depend on any property of `ld.global.nc`.
The `nc` question is a different question (6.6).

### 6.5 The detector (E6) and what it can and cannot catch

Purpose: convert "the slot was recycled under a reader" from a silent success into a
fatal error. This is a seqlock reader on the GPU.

- **Service (writer)**, when a slot is about to hold different bytes (reassign to a new
  expert, or reload after eviction): `SlotGen[row, slot] += 1` (release store), then
  `_mm_sfence()`, *then* the first byte store. The fence matters: x86 store-to-store
  order is preserved for ordinary stores, but the row packing can use non-temporal stores
  that a release store does not order (the reason `_mm_sfence()` already appears in
  `serve` before `publish_map`).
- **Device (reader)**: the acknowledgement kernel, after the copy kernel completed
  (so the copy's loads are complete and ordered before it), does `ld.acquire.sys` of
  `SlotGen[row, slot]` and compares with the lane's leased generation.
  - Equal: outcome CONSUMED.
  - Different: outcome VIOLATED. The ack kernel also sets `keep[0] = 0` (it precedes the
    fused MoE in stream order), and raises the page fatal word.

What it catches: any rewrite that began before the copy finished and whose generation
bump became visible before its first byte store, i.e. every rewrite by a correct writer,
regardless of timing.

What it does not catch: a rewrite by a service bug that skips the bump. It is a detector
for lease-protocol bugs, not a second protection. Also, the counter is 32-bit; a wrap of
one slot's counter inside a single lease window needs 2^32 reloads of one slot within one
GPU copy, which I treat as impossible.

Effect if VIOLATED ever happens: the request fails closed, the process fail-stops
through the existing fatal path, and the counter `lease_violations` is non-zero. It can
never end as a served request.

### 6.6 The `ld.global.nc` audit: the argument, then its limit

The plan (Task 5 gate) asks to "audit repeated mutable-slot reads through `ld.global.nc`
on the deployed GPU; treat this as a visibility test, not a presumption of corruption."
The copy path reads host memory with `ld.global.nc` (`copy_expert_host_unit16` and
`load_expert_host_word_noncoherent` in `transfer.cuh`). The wrapper name says
"noncoherent".

**The argument.** As I recall the PTX ISA text (I did not re-verify the wording against
the deployed PTX ISA version, **[OPEN 6]**), `ld.global.nc` is valid for data that is
not modified during the kernel's lifetime. The lease makes that true *within* the copy
kernel: from readiness until acknowledgement the slot's bytes are immutable (E1). So the
intra-kernel contract of `.nc` is met by construction, and the protocol never asks any
kernel to observe a host write that lands while it runs.

**What the argument does not cover.** The cross-kernel case: the same slot address is
read by kernel K1 (old expert), rewritten by the host (new expert), then read by kernel
K2. That K2 sees the new bytes needs (a) the per-SM non-coherent cache to be invalidated
at kernel start, and (b) any L2 residency of the host line not to be stale. I do not
know from documentation whether the L2 caches system-memory lines across kernel
launches on the target part, and I did not find a test in the repo that reloads a slot
under `.nc` and checks the second read. The existing byte-parity results on decode
(`DSV41_REFERENCE.md` section 19: exact parity with mirrors off) are adjacent evidence:
slots are evicted and reloaded constantly in those runs. They are not a test of this
claim. I am not asserting the read is safe or unsafe.

**The experiment that would settle it. It was run afterwards (NC_VISIBILITY.md, see the note
below this list); the plan as written here is kept for the record.** On the RTX 5090, under the GPU lock and with crypto-c9's scheduling:

1. Allocate a registered host slab larger than L2 (size past ~128 MiB per the project's
   microbenchmark rule) and fill row `r` with pattern A.
2. K1 reads row `r` with the copy kernel's exact `.nc` loads into a device buffer.
3. Host rewrites row `r` to pattern B (with the same `_mm_sfence()` the service uses),
   release-stores a word.
4. K2, launched *after* step 3, reads row `r` and checks it is B, for at least 10^6
   iterations, with and without another kernel touching the same lines between.
5. Repeat with `ld.global.cv` (or `ld.relaxed.sys`) in place of `.nc`.

If step 4 ever returns A with `.nc` and never with the non-`nc` load, the copy in lease
mode uses the non-`nc` load (a compile-time variant of `copy_expert_host_unit16`); the
lease protocol itself is unchanged. If it never fails, record the count and the
GPU/driver versions, and say "not observed", not "safe". The intra-kernel variant (host
writes during K2) is **not** a supported use and must not be tested as if it were.

**Result (run afterwards; full write-up and its pre-registration in `NC_VISIBILITY.md`).** The
experiment above was pre-registered, then run on divix01 (RTX 5090, PCIe Gen3 x16, driver
610.57.04). By the pre-registered rule the verdict is **keep `ld.global.nc`**: stale or mixed
words were **not observed** in 1.3e10 words per cell across the eight `nc` cells (host store
regular or non-temporal; boundary, graph, L2-thrashed and GPU-busy shapes), and the non-`nc`
`cv` load was equally clean. The harness can tell the variants apart: in a control, a `.nc`
load of a host-written flag inside a kernel never saw it in 20 ms (0 of 100) while `cv` saw it in
8.6 us (100 of 100), and a device-memory control returned the old value 100,000 of 100,000 times
under `.nc`. Two consequences for the design. (1) Step 4 keeps `.nc` for the copy; the `cv`
variant stays a compile-time switch, at no measured bandwidth cost (12.34 GB/s both, against
13.79 for `cudaMemcpyAsync`, Gen3 ceiling 15.75). (2) **A `.nc` load of host memory is stale for
the life of the kernel, so any kernel that polls host memory (the wait kernel, and Task 6's
per-lane readiness poll) must use `ld.acquire.sys`, never `.nc`.** This is "not observed", not
"safe", for one GPU, driver and host; OPEN 6 is narrowed, not closed.

---

## 7. One request, end to end (Task 5 scope: whole-request copy)

Task 5's second item: exercise acknowledgements "after the existing whole-request copy,
without changing its scheduling." So the flow keeps one wait kernel, one batched copy
kernel and one fused MoE per layer, and adds an acknowledgement kernel after the copy.

```text
device (captured graph, per layer)          service thread (RamThread::run)
--------------------------------------      ----------------------------------------
post:  LaneRequest[idx] (seqlock)
       demand record; st.release demand_head --> pump_demand sees head
                                              read_record + read LaneRequest (re-check)
                                              retire_leases() [non-blocking]
                                              idx retired? else DEFER
                                              serve(): reserve slots, read rows, pack
                                              per lane: grant lease, publish RowResult
                                              set_status(kServed); store_release demand_done
wait:  poll demand_done (+fatal, +shutdown)
       acquire RowResult.ready per lane
       validate generation, tag, expert
       commit? -> host_rows, go_count=count
       else    -> go_count=0, Terminal, fatal, keep=0
copy:  copy_expert_row_segments_gpu(..., host_rows, slots, go_count)
ack:   per lane < go_count: acquire SlotGen; st.release LaneAck  --> retire_leases() sees it
fused MoE (keep)                                                     leases--, idx free
```

### 7.1 Service: admission of a request

**A deferral must not touch the stage trace.** `pump_demand` calls `begin_stage` before it
decides anything, and its stage ring holds 8192 records. A deferral that returns without
advancing `next_demand_` and re-enters on every loop turn would push a stage record per
poll and flood the ring within milliseconds. So: a deferral pushes nothing (it resets
`cur_` without `ring_->push`), and a deferred request is re-attempted only when
`retire_leases()` changed something or a `Terminal` appeared, not on every poll. The stage
record eventually written for that request should carry the request's *first* observation
time, so that the time spent deferred shows up in the trace instead of vanishing.
How that interacts with `prev_done` and the backlog field is left to the code task
**[OPEN 15]**.

Before `serve()` grants anything, in the order:

1. `idx = (seq - 1) % 16`. If the `Outstanding` entry for `idx` holds an earlier
   generation that is not fully retired, **defer**: do not advance `next_demand_`, return
   to the loop. Section 11.4 says why this cannot deadlock or lap.
2. If `Terminal[idx].gen == G`, the device has already given up on this request. Do not
   serve it as a demand (no lease is granted); advance and count `late_after_terminal`.
   This check has no ordering requirement: if it races with a terminal published just
   after, the terminal retires whatever was granted (section 13).
3. Read `LaneRequest[idx]` with the seqlock re-check. Its low 32 bits must equal `seq`;
   its full `gen` is the request's `G` for everything that follows (11.3). Its
   `count`/`expert[i]` define the lanes. A mismatch means a later request has already
   overwritten the slot (a lap): count an overrun and skip. For an *armed* record that
   must not happen (the device is blocked in its wait and cannot post the next request into
   this slot; 11.4), so it is a protocol error there. `count == 0` means no lanes, no leases (the record is touch-only or an Option F
   all-GPU-hit handshake).
4. Add every `expert[i]` to the request's `wanted` set (protect, need, and now the lane
   experts), so that a lane's own expert can never be chosen as a victim by the same
   request's later slot choices. I did not verify that planned experts are always a subset
   of the routed experts the post kernel puts in `protect` (a prefetch producer may fill a
   plan with other experts) **[OPEN 12]**, so the design does not rely on it. A lane
   expert outside `[0, experts)` fails the request. Also fail it if `count` on the device
   exceeded `L` (the post kernel clamps to `kMaxIds`, so lanes past 8 would silently
   never be requested).

### 7.2 Service: serving, granting, publishing

The existing `serve()` steps stay as they are: dedupe `wanted`, find `missing`, take
slots (`take_slot_locked`), read and pack rows, publish each newly read row to `kReady`
and to the slot map. Additions **[P]**:

- `take_slot_locked` gains one eligibility condition: `leases == 0` (section 8).
- When a slot is assigned for a new expert (state to `kLoading`), bump `SlotGen` with the
  fence of section 6.5 before any byte is written.
- After the rows are ready and before `set_status`, for lane `i` in `[0, count)`:
  find `slot = expert_slot[expert[i]]`; assert `state[slot] == kReady` and
  `slot_to_expert[slot] == expert[i]`; then run the publication order of section 6.1.
  **Hits are leased exactly like newly read rows** (plan: "Include RAM hits as well as
  newly read rows in source ownership").
- If anything throws between the lease increment and the `ready` store, an RAII guard
  undoes the increments of that request. Nothing was published, so nothing can be
  reading.
- A demand that fails (I/O error, no victim, invalid) publishes no `RowResult` at all and
  sets status FAILED as today; no lease is ever granted for it. The device, which reads
  the status before any row result, never inspects the lanes.

### 7.3 Device: the wait kernel **[P]** (replaces the translate half of `exl3_ram_miss_wait_kernel`)

```text
go_count[0] = 0                               // fail closed: zero at entry
ok = !sticky && fatal == 0 && header.shutdown == 0
if count[0] > lanes (8): ok = false           // the post kernel clamps at 8; never serve a lane it dropped
if pending seq:
    poll: done = ld.acquire.sys(demand_done)
          while !reached(done, seq) and now - start < timeout:
              if fatal != 0 or header.shutdown != 0: break      // [P] fixes D4
              nanosleep; re-poll
    if !reached(done, seq):       reason = timeout or aborted; ok = false
    else if record.status != kServed: reason = failed;          ok = false
if ok and count > 0:
    for lane i < count:
        r = ld.acquire.sys.u64(RowResult[idx][i].ready)
        valid_i = (gen(r) == G) and tag(r) == READY
        also require RowResult.expert == planned[i]            // I2
        also require 0 <= host_slot                            // range
        if !valid_i: ok = false; reason = identity_or_stale
if ok:
    for i < count: host_rows[i] = RowResult[idx][i].host_slot
                   lane_ctx[i] = {G, slot_generation, row, host_slot}
    go_count[0] = count                       // the single commit point
else:
    publish Terminal[idx]: skipped_mask = all lanes (< count), reason;
                           tagged gen word stored last with st.release.sys.u64
    raise_fatal(page, seq); keep[0] = 0; sticky = 1
```

Notes:

- **`host_rows` is filled from `RowResult`, never from `slot_map`.** The wait kernel no
  longer reads `slot_map` at all for lease-mode rows. (`ram_miss`, the miss counter, is
  now the number of lanes whose status was not READY, for parity with today's counter
  meaning; **[OPEN 7]**: I did not determine every consumer of `ram_miss` and
  `unserved_misses` and whether their meaning must be preserved bit for bit.)
- **The commit decision is made once, before the copy.** In Task 5 the request is
  all-or-nothing, so the copy either runs for every lane or for none. This is what
  stops D1: on the timeout path the copy kernel gets `go_count == 0` and reads nothing.
- The copy kernel is unchanged. `copy_expert_row_segments_gpu` already takes the active
  count as a tensor (`row_plan.py`: `(segments, source_rows, destination_slots, count)`);
  the lease-mode backend passes `go_count` where it today passes `plan.count`. The
  in-graph kernels see the same captured tensor addresses every replay, which is what
  capture requires.
- `raise_fatal` is the existing `if (ld_acquire_sys(fatal) == 0) st_release_sys(...)`.
  It is not an atomic compare-and-set, which is fine only because one wait kernel runs
  at a time (assumption 1 of section 1.1).

### 7.4 Device: the acknowledgement kernel **[P]**

One block, `L` threads, launched after the copy kernel:

```text
for lane i < go_count[0]:                     // exactly the copy's active set (I4)
    ctx = lane_ctx[i]
    gen = ld.acquire.sys(SlotGen[ctx.row, ctx.slot])
    outcome = (gen == ctx.slot_generation) ? CONSUMED : VIOLATED
    st.release.sys.u64(LaneAck[idx][i], outcome_tag | G56)
if any VIOLATED: keep[0] = 0; raise_fatal
```

If `go_count[0] == 0` the kernel does nothing: **a skipped copy emits no
acknowledgement**, satisfying "a skipped GPU copy must not accidentally emit a
successful source-consumption acknowledgement" by construction: one word drives both the
copy set and the ack set, and it is written once, by the deciding kernel.

Inactive lanes (`i >= count`) are never in the set.

### 7.5 Service: retirement

`retire_leases()` **[P]**, called from three places and never blocking:

1. the top of every `RamThread::run()` iteration (before `pump_demand`);
2. the between-batches callback of `RowReader::read` (the lambda passed in `serve()`
   that currently evaluates `demand_pending() || pause_requested_ || ...`), so leases keep
   retiring while a long read is in flight;
3. once when a pause is being acknowledged (section 12, F10) and once in the shutdown
   drain.

Work per call is bounded: at most `R * L = 128` lane entries, and it returns immediately
when nothing is outstanding. For each `Outstanding` entry (all lanes not yet ACKED/VOID):

- `a = load_acquire64(LaneAck[idx][i])`; if `gen(a) == G`: CONSUMED or VIOLATED ->
  `leases[slot]--`, lane state ACKED (VIOLATED also increments `lease_violations`).
- `t = load_acquire64(Terminal[idx].gen)`; if `gen(t) == G`: read `skipped_mask`;
  for each lane in the mask not yet ACKED: `leases[slot]--`, lane state VOID.
- When every lane of the entry is ACKED or VOID the entry is freed and `idx` is reusable.

A decrement that would underflow, or a lane retired twice, is an internal error: fatal.
The per-lane state machine makes a double release impossible even if the device
misbehaves.

---

## 8. Lease lifecycle and the eviction predicate

Lane state machine (service-private, per `Outstanding` lane):

```text
NONE --grant+publish--> GRANTED --ack CONSUMED|VIOLATED--> ACKED
                            \----terminal, lane in mask---> VOID
```

- `GRANTED` is entered and `ready` published in one critical step (6.1). There is no
  granted-but-unpublished state that survives the step (the RAII guard removes it if the
  step throws).
- Transitions are monotone. `ACKED` and `VOID` are terminal.

Eviction predicate. `take_slot_locked` today (`host.cpp`) requires: `state == kReady`,
`!hot[expert]`, and the expert not in the `protect` list unless falling back. **[P]** adds
`leases == 0` (and, for the future asynchronous reader, no I/O or packing reference).
`assign`/`release`, the Python-facing eager calls, get the same rule; `release` of a
leased slot throws exactly as it does today for a `kLoading` slot.

When no victim exists *only because* of leases (there is a candidate that would qualify
if leases were zero), a **demand** is deferred, not failed (section 16). When no
candidate exists even ignoring leases, it is the existing `kNoVictim` failure. The
distinction must be computed, not guessed: `take_slot_locked` returns a tri-state
{slot, deferred, none}, and it must report *whose* leases caused a deferral (graph-lane or
host), because Task 8 needs to act on the second kind (`PROMOTION_ASYNC.md` R4).

An advisory can run while a demand is deferred. `pump_advice` starts one whenever it is not
stale, and `demand_pending()` is consulted only in the give-up lambda *after* reservation,
so it does **not** stop an advisory from reserving. An advisory posted *before* the deferred
demand is stale by the `after` rule (the head has reached `after + 1`); one posted by the
same post kernel *after* the demand, for the next row, is not stale (its `after` is the
demand's own seq) and may start. It takes slots in the *next* row's tier, not the deferred
demand's, and gives up after one row because `demand_pending()` is then true, so it delays
the retry by at most one row's read. An implementer must not rely on `demand_pending()` to
keep advisories off a deferred demand's row.

Slot generation lifecycle: bumped by the service under `mutex_` when a slot is assigned
to a new expert, before its first byte write. Hits do not bump.

---

## 9. The deduplication problem: one lease per consuming lane

**Chosen: count one source lease per consuming lane. Do not deduplicate.**

Justification.

1. **Lane ownership is already per lane.** Each lane has its own GPU destination row
   (`plan.slots[i]`), its own copy work and, in Task 6, its own copy kernel and
   acknowledgement. The natural unit that finishes reading a source is the lane. A lane
   count keeps "one acknowledgement releases exactly one lease" a local statement.
2. **Dedup makes one lane's ack authority depend on another lane's copy.** With dedup, if
   lane 3 and lane 5 name expert E, one lease covers both; then either the lease is
   released by whichever lane finishes last (a cross-lane dependency the device would have
   to track) or by the canonical lane alone (lane 5 may read E after lane 3 released it).
   The plan says the same: "one acknowledgement cannot release another lane's source."
3. **The plan's distinctness is convention, not enforcement.** The `ExpertRowPlan`
   docstring in `row_plan.py` says "Any producer may fill a plan"; only the router-miss
   producer is described as distinct misses, and `plan_candidates` dedupes by its own
   choice. The post kernel dedups `need` (`listed`) for the service's benefit, but the wait
   kernel does not dedup: it maps every lane through the map. `supports_fused_graph_routes`
   in `expert_route_plan.py` records that a multi-token call "may repeat an expert across
   tokens" and that its fused kernel has "no duplicate-ID handling"; it is a different
   kernel from this path, but it shows repeats are a live possibility in this code base,
   guarded there by a predicate rather than by structure. A protocol that is correct only
   because every producer is distinct fails in the one case nobody tested.
4. **The cost is small and bounded.** A slot's lease count is at most `R * L = 128`
   (16 requests x 8 lanes); it is a `u32`.

What dedup would have saved and what it costs here: it would save one `RowResult` record
and one 8-byte ack word per duplicate lane, and nothing else, since the source bytes are
read once per lane regardless (each lane has its own destination). The extra work is one
32-byte record.

Mechanics: two lanes naming E get two `RowResult` records with the same `host_slot` and
the same `slot_generation`, two lease increments, two acks, two decrements. The service's
existing `wanted` dedup is untouched: the *reads* are still deduplicated (one slot per
expert), only the *leases* are per lane.

Rejected alternative, for the record: "deduplicate expert requests before granting
lanes" would need a lane-to-canonical-lane table written by the service and read by the
device, and the device would have to redirect lane 5's copy from its own row result to
lane 3's. That is more shared state with two writers of its meaning, for no saving.

---

## 10. Expert identity in the row result, and what breaks if the GPU reads the map

### 10.1 What the protocol does

`RowResult.expert` is immutable once `ready` is published. The wait kernel compares it
with `planned[i]`, the expert *that lane* was planned for (device memory, written by the
routing plan for this replay). A mismatch, a stale generation or a non-READY tag fails
the request closed (terminal record, fatal, `keep = 0`). The service also checks
`slot_to_expert[slot] == expert[i]` before publishing (7.2), so an identity error inside
the service is caught before the device ever sees it.

### 10.2 Why "never a subsequently mutable global expert-to-slot map"

Take today's design (D2): the wait kernel reads `slot_map[row][expert]` into `host_rows[i]`
after `demand_done`; the copy kernel later indexes the slab with `host_rows[i]`. Two
distinct failures, both silent:

- **Stale slot index (time of check to time of use).** Lane 2 asks for expert E; at wait
  time the map says E is in slot 5, so `host_rows[2] = 5`. Before the copy reads slot 5,
  something evicts E: `take_slot_locked` calls `publish_map(row, E, -1)`, hands slot 5 to
  expert F, and starts writing F's bytes. The copy reads slot 5 and delivers **F's bytes
  (or a torn mixture) as expert E's weights**. `keep` is 1, `ram_miss` is 0, the request
  status is `kServed`, the byte counters agree. Fused MoE runs on the wrong weights. This
  is the shipped-bug shape of section 1.3. Today it is prevented only by the temporal
  exclusion, and it is reachable after a timeout (D1), and it would become reachable in
  ordinary operation when Task 6 lets the service evict while an earlier lane's copy runs.
- **Re-reading instead of caching.** If a design re-read the map at copy time, an entry
  that turned to -1 would be clamped to slot 0 (the wait kernel's `slot = 0` for a miss,
  a "valid address"), so the lane silently copies whichever expert occupies slot 0.
- **ABA.** E evicted and reloaded into slot 9 between the wait and the copy: the map says
  9, `host_rows` says 5. Either value alone is wrong for one of the two readers, and the
  map cannot say which reader saw which generation.

A per-lane immutable row result has none of these: the slot number is stated once, by
the writer, together with the generation and the expert, and the lease means it stays true
until acknowledged.

The slot map is not removed: it remains the service's and Python's lookup structure, and
the post kernel still uses it to decide `need` (a hint about what to ask for). It stops
being an input to *where the copy reads from*.

---

## 11. Generations, wrap, and request-slot reuse

### 11.1 The generations

- **Request generation `G56 = epoch << 32 | seq32`.** `seq32` is the existing device
  counter (`state[kPosted]`, skip-0). `epoch` counts how many times `seq32` wrapped.
- **Slot generation**: u32 per `(row, slot)`, service-owned, bumped before a slot's bytes
  change (6.5).
- The existing `seq32` and its 16-record ring are untouched.

### 11.2 Why 64 bits: the aliasing hole with 32

A lane that has been idle for `2^32` requests still holds an old `LaneAck` word from
generation `G - 2^32` (nothing ever clears it; clearing would be a second writer). A
32-bit generation compare cannot tell it from a fresh acknowledgement of `G`. At an
assumed 1200 posts per second (about 60 streamed layers at 20 tokens per second; the
layer count and rate are my assumption, not a measurement) `2^32` requests is 41 days.
Production runs longer than that. So the word carries the epoch.

### 11.3 Who owns the epoch: the device, and the service only echoes it

- **Device owns it.** New device-state words `kEpoch` and `kPendingEpoch`, appended to
  `STATE_WORDS` (indices 9 and 10; appended, not reordered, so existing indices keep their
  meaning). The post kernel increments `kEpoch` when `seq32` wraps past 0xFFFFFFFF to 1 and
  writes the full `G56` into `LaneRequest.gen`. The wait and acknowledgement kernels use the
  pending epoch stored by the request's own post.
- **The service does not count epochs.** It takes `G` from `LaneRequest[idx].gen`, after
  checking that the low 32 bits equal the sequence it is serving. Every lease, `RowResult`
  and terminal check for that request is keyed by that echoed `G`.
- A device built over a used block starts at epoch 0; the block is created fresh with the
  device (one device incarnation per block, **[OPEN 14]**).

Why the service must not count epochs itself, which is what an earlier draft of this
section said (a service-side `epoch_` incremented whenever `next_demand_` wraps): the
explicit-state model (section 18.3, `lease_model.py`, mutant `echo_gen=False`) found a
counterexample. The service can move `next_demand_` across the wrap **without stepping
through it**: `pump_demand` resumes after a lap at `head - 14` (`next_demand_ = head -
kDemandRecords + 2u`). If the device has posted unarmed touch-only records across the wrap
while the service lagged, the service resumes on the far side of the wrap, its counted epoch
is one behind the device's, the armed request that follows is keyed with the wrong epoch, its
`LaneRequest` looks stale, and the request is dropped as lapped while the device waits for
it: a healthy system fails stop. In the model's ring of 2 the trace is nine device steps
and two service steps. With the real ring of 16 it needs the service to lag the device by
16 or more records across the wrap, which unarmed records make possible (the device posts
them without waiting) whenever the service is inside a long read. That is rare (once per
2^32 requests, times the lag), but it is a healthy-system fail-stop, which is why it is
worth removing by construction. Echoing the device's `G` removes the service's epoch state
and with it the whole class.

Consequences: `Header.demand_epoch` is gone (4.3); fixing D6 is no longer a prerequisite of
this protocol, though it should still be fixed (section 1.2).

The 24-bit epoch overflows after `2^24` wraps, `7.2e16` requests, about 1.9e6 years at
1200/s. Treated as never; **not** handled.

### 11.4 Request-slot reuse

A request slot `idx` (record, `LaneRequest`, `RowResult`, `LaneAck`, `Terminal`, and the
service's `Outstanding`) is reused for generation `G + 16`. Rule:

> The service serves `G + 16` only when the `Outstanding` entry for `idx` is fully
> retired (every lane ACKED or VOID). Until then it defers, without advancing
> `next_demand_`.

Why this is safe and cannot deadlock or lap:

- The device posts `G + 16` only after its wait for `G` returned and the stream ran G's
  copy and ack kernels (one stream, serialized graphs; assumption 1). So G's
  acknowledgements have been *issued* before `G + 16` can be posted; the service merely
  has to *observe* them, which is finite latency and is done by `retire_leases()` in the
  polling loop. Deferral is therefore a transient wait for visibility, not a wait for
  progress.
- While `G + 16` is deferred, the device is blocked in its armed wait for `G + 16` and
  posts nothing further; the only posts that do not wait are unarmed touch-only records
  with `count == 0`, which take no lease and so cannot exhaust the ring of *lease*
  entries. Lapping of the demand ring by `head - next_demand_ >= 16` is therefore not
  reachable through a deferred armed request. **[OPEN 8]**: I argued this and the model
  supports it within its bounds (unarmed records piling past the ring in a ring of 2 and of
  3, across the wrap, with deferral reached: no armed request was ever lapped once the
  epoch is echoed). Not tested at ring 16 or against the real lap-skip code; the interplay
  of `pump_demand`'s lap skip with deferral needs a targeted test (post 16 unarmed records
  behind a deferred armed one).
- If G's acknowledgements never arrive (CUDA error, dead stream) `G + 16` is never
  posted either; nothing is ever served into a reused slot. The failure is a hang of a
  dead stream, which the device timeouts and watchdog already bound.
- The device-side reader of `RowResult[idx]` for `G` cannot see `G + 16`'s payload,
  because the service writes it only after G retired, and the tagged `ready` is stored
  last. As a belt, the wait kernel re-reads `ready` after reading the payload and treats a
  changed generation as a protocol violation (fatal), the same seqlock check as 6.3.

### 11.5 Guard for concurrent graphs [OPEN 9]

The plan says concurrent graph execution "remains unsupported and guarded". I did not
find where it is guarded today (`state[kPosted]` is a plain read-modify-write in the post
kernel, so two concurrent replays would race on it). The design assumes one replay at a
time; a guard, if wanted, is that the post kernel asserts `seq == previous posted + 1` and
raises a fatal otherwise. I did not specify it further.

---

## 12. Failure matrix

`V` = leases held on the affected request when the event happens. "Leases" is what retires
them; "Slots" is what happens to the pinned slots; "GPU" is what the GPU reads.

| # | Event | What the device does | What the service does | Leases | GPU reads a source? | Notes |
|---|---|---|---|---|---|---|
| F1 | **Failure before readiness** (I/O error, short read, no victim, invalid request) | wait sees status != kServed: `go_count = 0`, publishes `Terminal` (reason `failed`), raises fatal, `keep = 0` | publishes no `RowResult`, sets status FAILED, releases its unpublished slots (existing) | none ever granted (`V = 0`) | no | the device never inspects lanes when status is FAILED |
| F2 | **Timeout before copy** (device gives up while the service is slow or dead) | wait loop exits at `timeout_ns` with `demand_done` not reached: `go_count = 0`, publishes `Terminal` (reason `timeout`) **before** raising fatal | if it later serves the request: sees `Terminal[idx].gen == G`, does not serve; or if it granted first, `retire_leases()` voids the lanes named in the mask | retire via terminal (VOID) | no (go_count 0) | the D1 hole, closed |
| F3 | **Failure after a subset copied** (Task 6 only) | lanes `< j` copied and acknowledged; lane `j` times out or fails; remaining copies skip on their predicate; a *finalize* kernel, stream-ordered after every lane's copy and ack kernels, publishes `Terminal` with `skipped_mask` = the skipped lanes; compute is suppressed by the final request-success check | copied lanes retire by their acks; skipped lanes retire by the mask | mixed ACKED / VOID | copied lanes only, and those were leased | the mask names lanes; the service never infers "everything else" |
| F4 | **Terminal cancellation handshake** | see section 13 | see section 13 | VOID by mask | none after the mask is published | |
| F5 | **CUDA error during copy or ack** | kernel aborts; no ack, no terminal; the error is sticky | nothing observes anything | **never decremented** (uncertain) | unknown | leases are pinned forever; shutdown quarantines (section 14) |
| F6 | **Device hang** (spinning wait) | bounded by `timeout_ns` **and** by fatal/shutdown polled in the loop (D4 fix) | none | as F2 | no | copy kernels are finite work |
| F7 | **Service thread hang or death** | wait times out (F2) | existing watchdog aborts on stuck `busy_since` or held fatal | irrelevant, process ends | irrelevant | existing behaviour [E] `RamThread::watch` |
| F8 | **Request lapped by the device** | existing: status stays pending, `kOverruns` counted, device wait times out | existing lap skip | none granted for a skipped generation | no | section 11.4 shows a deferred armed request cannot be lapped |
| F9 | **Shutdown / process exit** | pollers exit promptly on `Header.shutdown`; committed copies finish and ack | stop admission, drain, then Python establishes completion (section 14) | see section 14 | no new reads | |
| F10 | **Eager pinned-tier use** (`before_host_use`) | none | the caller synchronizes the current stream (existing `before_host_use`), then pause; the pause acknowledgement runs `retire_leases()` and requires `outstanding == 0` **counting graph-lane leases only** (host leases do not block it; 17.1 R2), otherwise the pause fails and eager use is refused | expected 0; nonzero means a leaked lease and is reported loudly | no | today's eager path calls `current_stream().synchronize()` then `host.pause(...)`; a leak here is exactly a protocol anomaly |
| F11 | **Duplicate lanes** (two lanes, one expert) | two acks | two leases on one slot | `leases == 2`, each retired once | yes, each lane | section 9 |
| F12 | **Identity mismatch or stale/foreign row result** | wait compares expert/generation/tag; fails closed as F1 with reason `identity` | counter `identity_errors` from the service's own pre-publish check | none or VOID by mask | no | should be unreachable; it is the alarm, not a path |
| F13 | **Recycle detected after copy** (E6) | ack outcome VIOLATED, `keep = 0`, fatal | counts `lease_violations`, decrements (the reader is done) | ACKED | yes, and detected | a protocol bug, not an expected path |

The property to check in review: in every row, either the GPU reads no source, or every
source it reads is leased and its lease is retired only by that lane's ack or by a
terminal mask that does not name it.

---

## 13. The terminal cancellation handshake

Problem: once the service has published readiness and granted a lease, it cannot tell
whether the GPU has started reading. It therefore cannot reclaim a granted lease
unilaterally, on a timer or on its own fault. Only the device can say "I will not read".

Protocol **[P]**:

1. The deciding kernel (the wait kernel in Task 5; the finalize kernel in Task 6) writes
   `Terminal[idx]` **only** for lanes it has irrevocably decided to skip, in the order:
   `skipped_mask`, `reason`, then the tagged `gen` word with `st.release.sys.u64`.
2. **A lane in `skipped_mask` never starts a copy afterwards.** In Task 5 it is enforced
   by `go_count = 0`, which the wait kernel writes before the terminal and which is
   stream-ordered before the copy kernel. In Task 6 it is enforced by the per-lane
   predicate (`copy_one_row_if_valid`) reading device lane state that the finalize kernel
   does not change, and the finalize kernel is stream-ordered after every lane's copy and
   ack kernels, so a lane is never both copying and named in the mask.
3. The service retires exactly the lanes named in the mask (7.5). It does not retire
   "every lane not yet acknowledged", because an acknowledgement and a terminal are
   published by different kernels and I do not rely on cross-kernel visibility order
   between two different addresses; the mask removes the ambiguity ("unacked" could mean
   "skipped" or "copied but the ack is not yet visible").
4. A lane is in exactly one of {mask, acknowledged}. If the device ever produced both, the
   service's per-lane state machine catches the second retirement as an internal error.

No new leases after a terminal: a terminal for `G` marks the request closed (7.1 step 2).
A lease granted concurrently is retired by the terminal at the next `retire_leases()`;
its late `ready` is harmless because `go_count` is 0.

---

## 14. Shutdown and quarantine

Requirement from the plan: "stop admission, drain storage/packing, then establish
completion of all GPU readers before freeing their memory. If a CUDA error prevents
establishing completion, retain/quarantine allocations until process teardown; never
recycle uncertain storage." Today (D5) the code frees first and never checks.

**FINDING, before anything else in this section: `Exl3RamMissService.shutdown()` has no production caller today, but
the place to call it already exists.** Its only callers are tests and the exit hook the wiring adds. So the orderly
sequence below (S0-S5) is correct, tested and mutation-checked as a *wiring*, and changes nothing about what a
production teardown does until it is called. What is reachable in production today is **only the exit-hook quarantine,
and only on a normal interpreter exit**.

**The caller is an extension point that already exists: `Scheduler.release_host_resources()`**
(`managers/scheduler.py`, "Release pinned host buffers in userspace on graceful shutdown ... Called from
`run_scheduler_process`'s finally"). It is reached only on the graceful path (`if scheduler.gracefully_exit:` in that
`finally`, set by a `ShutdownReq`), and the code there already makes the decision this design makes: the comment says
the device barrier is not attempted on the exception path "because the GPU may be wedged and the synchronize() could
itself hang". It already stops the expert doorbell (`stop_doorbell()`, inside a try/except that logs and continues),
destroys the hisparse coordinator, and releases the tree cache's and the decode-offload manager's host resources: the
same class of work with the same failure discipline. A call to `Exl3RamMissService.shutdown()` belongs there. **Not
wired**, and not to be wired without reading `large-class-style` (`Scheduler` is a frozen class under
`.claude/rules/modify-component-must-read.md`) and a review of the placement: whether it goes before or after
`stop_doorbell()` (no other reader of the slabs may still run when the slabs are freed) is undecided.

An earlier version of this paragraph said the scheduler is SIGKILLed at shutdown, sourced to a docstring in
`exl3_stream_trace.py`. **That is wrong for the graceful path** and is withdrawn: `run_event_loop()` blocks until a
`ShutdownReq` sets `gracefully_exit`, the only `os.killpg(SIGKILL)` in `scheduler.py` is on the exception path behind
`SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION`, and the existence of `stop_doorbell()` and the other host-buffer releases is
itself evidence that the graceful path runs. **Not established:** whether *every* production teardown takes it.
`Engine.shutdown()` kills its children's process tree (`kill_process_tree`, SIGKILL); whether a `ShutdownReq` always
reaches the scheduler and drains before that is untraced. That is the residue of OPEN 18.

**Do not read any test or mutation result on this step as protection of a production path, and no Task 5 box that
mentions shutdown is to be ticked on step 6's strength**, until the call is wired and a graceful-shutdown test drives
it through `release_host_resources()`.

### 14.1 What is quarantined, precisely

Everything a still-running or possibly-running GPU kernel may read *or write*:

- the pinned slabs (every tier's `pinned_host_cache.tensors`, i.e. `tables.slabs` and
  their `keepalive`) and their `cudaHostRegister` registrations;
- the request page, `slot_map` and the lease block (pinned tensors);
- device buffers used by captured kernels: `state`, `last_routes`, `host_rows`,
  `go_count`, `lane_ctx`, `keep`, `ram_miss`, the GPU scratch destination rows, and the
  captured graph's memory pool.

Not quarantined: the C++ `RamTier` and its io_uring, file descriptors and bounce buffer.
Nothing GPU-side touches them. (Whether io_uring teardown waits for an in-flight O_DIRECT
read into a slab is a *storage-side* question this document does not settle: **[OPEN 10]**.)

### 14.2 How

1. **Do not unregister, do not free.** In the quarantine path, `release_host_slabs` /
   `_cuda_host_unregister` are not called, and the slab tensors are not dropped. This
   means calling `detach()` on each `ExpertPinnedHostCache._release_slabs` finalizer
   (`expert_stream.py`), because that finalizer runs `release_host_slabs` at exit by
   default (D5); skipping only `tier.close()` would not be enough.
2. **A module-level list is not enough.** Interpreter finalization clears module globals
   and would free the tensors, with no device barrier. The quarantine takes an extra
   strong reference that is never released, for example `ctypes.pythonapi.Py_IncRef` on
   each object, so the refcount cannot reach zero during finalization. State this in the
   code where it is implemented; a plain list would look right and be wrong.
3. The C++ handle is still closed: the service thread is joined (it must not touch
   anything after this) and `exl3_ram_miss_close` runs. It holds only raw pointers into
   the quarantined Python-owned memory, so leaving that memory in place is exactly what it
   needs.
4. Leases are **not** decremented in bulk. They stay as they are; the tier is dead.
5. Log once, loudly: which condition forced quarantine, and the count of outstanding
   leases.

### 14.3 The orderly sequence

```text
S0  Python: refuse new work (the graph launch path checks Service._shut_down; existing)
S1  service: store_release(Header.shutdown, 1)  -> post kernels post nothing, wait kernels
                                                    exit promptly (D4 fix), pause is refused
S2  service: keep polling retire_leases(); serve nothing new; storage/packing drain
             (existing stop semantics: request_stop, advisory gives up at its next row)
S3  Python: establish GPU completion:
        run torch.cuda.synchronize(device) in a helper thread, join(timeout = T)
        T = SGLANG_DSV41_RAM_MISS_TIMEOUT_MS + margin   (a kernel bounded by that timeout
                                                        or the shutdown word must be done)
        success                    -> every kernel queued before this point has completed
        raises (any CUDA error)    -> cannot establish; quarantine (14.2)
        does not return by T       -> cannot establish; quarantine (14.2)
S4  success only: stop and join the service thread; retire_leases() one last time;
    any lease still outstanding is a *protocol anomaly*: log it, count it, but memory
    safety was established by device completion, not by acks, so it may proceed
S5  success only: unregister and free (release_host_slabs and the device buffers)
```

Two points a reviewer should challenge:

- **Safety at S5 rests on device completion (S3), not on the acknowledgements.** Acks are
  the fine-grained mechanism during operation; shutdown does not trust them, because a
  missing ack is exactly the uncertain case.
- **The helper thread is required** because `cudaDeviceSynchronize` is not
  interruptible; a hung kernel would otherwise hang shutdown with no way to declare
  "uncertain".

### 14.4 The `_stop_live` (atexit) path

`_stop_live` in `ops.py` stops hosts without a device barrier and without a hook to the
Python service (it iterates `_LIVE` hosts, not `Exl3RamMissService`). Separately, the
`ExpertPinnedHostCache` finalizer unregisters the slabs at exit (D5). Two options:

- **(a)** Make the atexit hook run S1..S5 above for the service that owns the host, and
  have `Exl3RamMissHost.stop()` refuse to run when leases are outstanding and completion
  is unproven, choosing quarantine instead.
- **(b)** Leave `_stop_live` as the last-resort path but make it quarantine
  unconditionally: at exit the process is ending, so leaking is free and safe; do the
  ordered sequence only in `Exl3RamMissService.shutdown()`.

I recommend **(b)** for the first implementation [DECIDE 2]: at process exit nothing is
gained by freeing, and (b) needs no device synchronization inside an atexit handler
(where CUDA may already be tearing down). The cost is that the orderly sequence is
exercised only by tests until a production caller exists.

### 14.5 What this changes about the D5 contrast

| | Today | This design |
|---|---|---|
| Establish GPU readers done | never | `torch.cuda.synchronize` bounded by a helper thread, before any free |
| CUDA error at teardown | not looked for | forces quarantine |
| `cudaHostUnregister` failure | warn, forget the range, free anyway | retain and never free the storage |
| Interpreter-finalization free | happens | prevented by an unreleased extra reference |
| Spinning wait kernel at shutdown | holds the stream up to `timeout_ns` | exits on `Header.shutdown` within one poll |

### 14.6 Requirement: a production caller of an orderly shutdown

There is none today. S0-S5 will run in production only if something calls
`Exl3RamMissService.shutdown()` (or the atexit path is upgraded per 14.4 (a)). If (b) is
chosen, S0-S5 are exercised by the tests and by a future scheduler-teardown hook.
Declaring that hook is out of scope for Task 5, but it is a **requirement** of the
"shutdown" acceptance line, not something this design assumes exists.

---

## 15. Option F: the acknowledgement and eviction ordering that must survive

What Option F does today **[E]**: with advisories on (`SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`,
`prefetch_enabled()`), the post kernel arms *every* record (`armed = need_count > 0 ||
advise != 0`, `device.cuh`), so the device waits for `demand_done` even when every planned
row is already in RAM. It also posts an advisory record for `next_row` built from the
stored previous-token routes. The service serves demands first; an advisory is stale if a
later demand has been posted (`reached(demand_head, request.after + 1)`), gives up between
rows, protects only its own ids, and evicts only unprotected, non-hot `kReady` rows.

Orderings to preserve, each with its status under leases:

| Option F ordering | Status |
|---|---|
| `demand_done(G)` is stored after every service state change made for `G` (touches, evictions, map publishes) | **kept**, and extended: leases are granted and `RowResult`s published before `set_status` and `demand_done` (6.1) |
| The all-hit handshake: with advise on, the device waits for the service even when nothing is read, so the service's recency/eviction decisions for the row precede the gather | **kept, and now mandatory whenever `count > 0`**, with or without advise: the handshake is where the lease is granted. See below |
| Demands before advisories; an advisory yields at the next row when a demand is posted | kept unchanged |
| An advisory for row R+1 cannot evict slots the demand for R+1 needs | kept (`protect`), and strengthened: an advisory also cannot evict any leased slot |
| An unarmed record never evicts (touch-only) | kept, but reachable only for `count == 0` in lease mode |

**Consequence to state plainly.** Without advise, today's *unarmed* records (nothing
missing) avoid the service round trip. In lease mode a request with `count > 0` cannot
skip the handshake, because the GPU may only read a source it holds a lease on and only
the service grants leases. So lease mode arms every record with `count > 0`, exactly as
Option F already does when advise is on. That is a per-layer round trip added for the
no-advise configuration; **measured 2026-09-21 at about 8 us per all-hit layer, ~0.32 ms per step at 40 layers
(OPEN 11, `open11/results.md`, kernel-and-service level, eager). Re-taken through the real backend in a CUDA graph the
same day: about 17 us per all-hit layer, ~0.68 ms per step, and the two are not reconciled (section 20.2m).** The plan anticipates this:
"Removing its all-hit handshake is a separate optimization after equivalent protection is
proven." What an equivalent protection would be, and why I did not design it here: a
device-side lease taken by an atomic on a mapped per-slot counter would have to be
ordered against the service's eviction test with a Dekker-style argument in both
directions (device increments then re-checks the slot's generation; service sets a
"evicting" state then re-checks the count). That needs its own proof and its own model
check, and it is Task 5's non-goal.

---

## 16. The worker never waits for an acknowledgement, and why it cannot deadlock


**Status of that requirement.** It is **unwired into an existing extension point**, not unowned and not blocked:
`Scheduler.release_host_resources()` on the graceful-shutdown path (see the finding at the top of this section). Wiring
it is one call inside a function of a frozen class and needs its own review; it is not part of Task 5's file list.
Consequence for Task 6, still true until that call is wired: its "stop during SM transfer" case (a shutdown while lane
copies are in flight) can only be exercised by tests that call `shutdown()` directly.

### 16.1 The rule

The service thread never blocks on an acknowledgement or a terminal. Retirement is a
polled, bounded step (`retire_leases()`, section 7.5). No code path in `RamThread::run`,
`pump_demand`, `pump_advice`, `serve` or `RowReader::read` waits on the lease block.
Submitting, reaping and cancellation handling (`pause_requested_`, `stop_requested_`,
`demand_pending()`) are evaluated on every loop and every batch boundary and do not
depend on any lease count. `stop()` exits the loop without waiting for outstanding leases
(they do not affect service liveness).

Today's `serve()` is synchronous per request and returns only when its ring is empty
(plan: the blocking reader contract stays). Task 5 does not change that. The requirement
"keep submitting/reaping ... while acknowledgements are outstanding" is satisfied in Task
5 by three things: acks are never waited for; `retire_leases()` runs between batches
inside a long read (call site 2); and no read is ever *gated* on an ack (a gate would be a
wait). It becomes non-trivial in Task 6, where the loop is an asynchronous progress
function; the same rule carries over.

### 16.2 Why the resource dependency cannot deadlock

Deadlock needs a cycle. The only resource with a lease dependency is a pinned slot in one
row's tier. The waits:

- The **device** waits for the service: wait kernel on `demand_done` / `ready`, bounded
  by `timeout_ns`, fatal and shutdown.
- The **service**, when a demand cannot get a slot because every candidate is leased,
  **defers** the request (section 8: `deferred`, no state change, slots taken are
  released) and keeps polling. It waits for acks *by returning to its loop*, not by
  blocking. A deferred request is dropped when the device gives up (Terminal, section 13),
  so the device timeout bounds the deferral.
- The **acks** the deferred request needs belong to earlier requests of the same row.
  They are published by ack kernels that are *stream-ordered before* this request's post
  kernel (one stream, serialized graphs). An ack kernel depends only on its own copy
  kernel and the commit decision that preceded it, never on a later request.

So the wait-for graph is: (later request's wait) -> (service deferral) -> (earlier
request's ack) -> (earlier request's copy) -> (earlier request's readiness, already
published). It is a chain into the past; there is no edge from an earlier request to a
later one, hence no cycle.

Assumptions this argument needs, all of them explicit:

- **A1**: one stream, one graph replay at a time (assumption 1 of 1.1; guard: [OPEN 9]).
- **A2**: the service reserves *all* the slots a request needs before it starts I/O, and
  releases them all if it must defer. `serve()` already takes its slots in one locked pass
  before any read (the `take_slot_locked` loop); Task 5 keeps that. A request holding some
  slots while waiting for others would make later lanes wait on earlier lanes'
  acknowledgements of *this* request, which are gated on this request's readiness: a real
  cycle. All-or-nothing reservation is what removes it.
- **A3**: every non-graph acquirer (Task 8) releases its lease without waiting on a
  demand, and **once a lease is held, the acquirer's copy stream must not depend on the
  serving stream** (no `wait_stream(producer_stream)` after acquisition; a dependency the
  copy needs must be satisfied *before* the lease is taken, or the copy must not have one).
  The cycle it forbids is concrete: a promotion holds a source lease on slot `S` and its
  copy waits on the serving stream; the serving stream is inside an armed wait whose only
  victim is `S`; the service defers the demand for lack of a victim; the lease is released
  only when the copy completes; the copy waits for the serving stream. Nothing breaks it
  but the 2 s device timeout, as a fatal (`PROMOTION_ASYNC.md` 9.2, found by Task 8's
  design; today's `_submit_operations` does `self.stream.wait_stream(producer_stream)`).
  Section 17.
- **A4**: for Task 6, lanes are consumed in a fixed order and lane `j`'s acknowledgement
  is issued before the GPU waits for lane `j + 1`'s readiness. Lane `j + 1`'s readiness
  then never waits on an ack of lane `>= j + 1`. Task 6 must re-establish this; I have
  not designed Task 6.

If A2 is violated the failure mode is a bounded fail-stop (device timeout), not a hang.
That is the design's fallback for every assumption above: the device timeout and the
watchdog turn an unforeseen cycle into a reported failure rather than a silent stall.

---

## 17. Non-graph acquirers and Task 6

### 17.1 The contract must not assume every acquirer is a captured graph lane

The lease is `(row, slot, slot_generation)` with a count. Its release channel is the only
thing that differs by acquirer:

- **graph lane**: released by the device acknowledgement or a terminal mask (this
  document);
- **non-graph holder** (a promotion or an eager copy on some other stream) **[P]**:

  ```text
  optional<LeaseRef> acquire_host_lease(row, expert)   // service mutex; slot must be kReady;
                                                       // leases++; LeaseRef = {row, slot,
                                                       // slot_generation, lease_id}
  LeaseRef lease_on_ready(row, slot)                   // service-initiated admission: the lease
                                                       // is taken in the SAME mutex_ section
                                                       // that moves kLoading to kReady
  void release_host_lease(LeaseRef)                    // exactly once; leases--; any thread
  ```

  The holder decides when its consumer (for example a `cudaMemcpyAsync` and its event) is
  complete and calls `release_host_lease` itself. The service never infers it.

Requirements filed on this contract by Task 8 (`analysis/dsv41-drive/PROMOTION_ASYNC.md`
9.1, R1-R8), accepted here as requirements of the lease owner:

- **R1** The host-lease calls take only `RamTier::mutex_`, are callable from the scheduler
  thread, and never pause the service. A stale or repeated `LeaseRef` (slot generation or
  `lease_id` mismatch) is a **counted error**, not a silent no-op and not a second decrement.
- **R1a (added by review finding F1, `LEASE_MODEL_REVIEW.md`): a host lease must be releasable
  from a context that is not the scheduler thread, and this is a requirement, not a permission.**
  In the Task 8 design the promoter and the eager caller are the *same* thread: the poll step
  that releases a promotion lease runs on the scheduler thread, and eager `before_host_use`
  blocks that same thread in `torch.cuda.current_stream().synchronize()`. The cycle: a decode
  replay is in flight and its armed demand is deferred because the only victim carries a
  promotion lease; the copy has finished but the lease is released only by the poll step; the
  scheduler thread has entered `before_host_use` and is blocked waiting for that replay; the
  replay waits for the service; the service defers behind the lease. The model shows it (the
  Task 8 world with `blocking_eager` and `poll_release` is a Deadlock in 4,802 states, an 11-step
  trace) and shows it gone when the release is not on the scheduler thread. It is stronger than
  A3: not merely "not the host poll", but **not the scheduler thread at all**, because
  `before_host_use` blocks it.
  **Chosen mechanism: a host callback enqueued on the executor stream right after the copy**
  (`cudaLaunchHostFunc`), whose only action is `release_host_lease` (a `mutex_` section and a
  counter decrement, no CUDA call, no Python). Why this and not the alternative of making
  `before_host_use` poll with `stream.query()` and release completed leases: the callback removes
  the dependency for every path, whereas polling fixes only the one blocking call we know of and
  leaves the next `synchronize()` anyone adds as the same hazard; and it makes the release happen
  at copy completion rather than at the next poll, which is what the lease's hold time (R4)
  should be measured against. Its costs, stated: one callback per copy; a stalled or
  unscheduled driver thread delays the release (the deferral is then bounded by the device
  timeout as for any other lease); the callback takes `mutex_`, which the service holds only in
  short sections and never across I/O; and the poll step must then **not** also release (a
  double release is a counted error, R1), only reclaim bookkeeping. **[OPEN 16]**: `host.cpp`
  makes no CUDA calls today, so the enqueue belongs in the device module (`exl3_ram_miss.cuh`,
  which already has CUDA) and reaches the release function of the host module by an address
  exported as an integer; that cross-module call and its lifetime at shutdown (a callback still
  queued when the tier closes) are not designed here. If they prove awkward, the fallback is
  `before_host_use` polling, accepting its weaker guarantee.
- **R2 (the F10 split)** The pause acknowledgement's `outstanding == 0` counts **graph-lane
  leases only**. Host leases must not block an eager pause: otherwise one in-flight
  promotion makes every eager `before_host_use` fail for the copy's duration, and
  `before_host_use` cannot wait for it (it synchronizes the current stream only, not the
  executor stream). The split is safe because a host-leased slot is excluded from eviction
  and from `assign`/`release` (below), so the lease, not the absence of the thread, protects
  its bytes. Table row F10 in section 12 is read with this split.
- **R3 `lease_on_ready`** Section 17.1's original `acquire_host_lease` requires the slot to
  be `kReady` first, which leaves a window between `kReady` and the lease in which a demand
  or advisory eviction could take the row unless it is `hot`. `hot` is a boolean the next
  `set_hot` overwrites, so it is not a substitute. Service-initiated admission therefore
  takes its lease inside the section that publishes the row.
- **R4** A promotion lease never causes a demand failure. A deferral caused *only* by host
  leases must be reported as such (section 8's tri-state) so the poll step can act on it;
  the hold time is bounded by the capped copy's enqueue-to-completion.
- **R5** The eviction predicate covers `take_slot_locked`, `assign` and `release`; `release`
  of a leased slot throws exactly as it does for `kLoading` today (section 8).
- **R6** Outstanding host leases are part of the shutdown quarantine set (section 14). The
  executor stream is one more GPU reader class: the device-wide synchronization in S3 covers
  it only if S3 is `torch.cuda.synchronize(device)` and not a single-stream synchronize, so
  S3 must be the device-wide form.
- **R7** Counters: `host_leases_outstanding`, lease hold time, `pauses`, `pause_wait_ns`,
  `defer_reason`.
- **R8** The lease API must **not** bump `kVersion` (it changes no mapping); otherwise every
  lease pays an `expert_to_slot` rebuild in `NativePinnedSlotTable`.

Both channels decrement the same `leases` counter and are subject to the same eviction
predicate. **I do not specify the promotion protocol** (that is Task 8's own design), only
that this contract does not preclude it, and constraint A3 above binds it: a release must
never depend on a demand being served.

### 17.2 What Task 6 inherits, and what it has to redo

Inherited unchanged: the block layout, single-writer ownership, generations and wrap,
`RowResult`/`LaneAck`/`Terminal`, the eviction predicate, the terminal mask semantics,
deferral, the quarantine rules.

Task 6 also inherits OPEN 11. Lease mode arms every record with `count > 0`, so each layer
pays a service round trip even when nothing is read, and that cost lands on exactly the
path Task 6 is trying to shorten. Task 6's benefit has to be measured net of it, and the
"removing the all-hit handshake" optimization of section 15 is the thing that would give it
back. **That cost is now measured, and re-measured: about 17 us per all-hit layer and ~0.68 ms per step at
40 layers, which is about 61% of the 1.114 ms `G*` Task 6 is trying to win.** The first measurement gave ~8 us /
~0.32 ms / 28-30% from a hand-written step; the re-take through `Exl3RamMissRowBackend` on an exclusively held card
roughly doubled both, and the x40 is now measured in one graph rather than extrapolated. See `open11/results.md`.
**This is the paragraph Task 6's baseline requirement is stated in, so a reader arriving here must not take the
superseded number.**
Material rather than fatal, and it is the figure Task 6's net benefit must be reported
against. See `open11/results.md` for the limits, chiefly that it is an upper bound attained
only when every layer is all-hit, and that it is not a serving-path number.

Task 6 must add, and I have not designed: per-lane wait/copy/ack kernels with a lane
predicate that reads the lane's own readiness (the wait becomes per lane); a finalize
kernel that publishes the terminal mask after all lane kernels; a single request timeout
budget; A4 above; and the launch-count cost. The `LaneRequest` mechanism already gives
the service the lane list before the request is served, which per-lane early publication
needs.

**Hard requirement on Task 6's readiness poll (from the `ld.global.nc` experiment,
`NC_VISIBILITY.md`):** a `.nc` load of host memory is stale for the life of the kernel, so the
per-lane readiness poll must use `ld.acquire.sys` (or `cv`), never `.nc`, however natural `.nc`
looks by symmetry with the copy path, which keeps it.

### 17.3 The executor-stream release callback: ownership, lifetime and close (OPEN 16)

Requirement (17.1 R1a): a host lease is released by a host function enqueued on the executor
stream after the copy, not by the scheduler thread. This section designs the mechanism far enough
that no step of it can be a use-after-free, because the failure it prevents would present as a rare
crash in unrelated code. Nothing here is implemented, and **step 7 must not get ahead of it.**

**Design rule: a callback dereferences nothing that can be freed.** It carries a handle and a lease
id, never a pointer to the tier, its mutex or its counters, and it reaches the tier only through a
process-lifetime registry. Everything below follows from that.

1. **What crosses the module boundary is one plain C function, once.** `host.cpp` exports
   `extern "C" int exl3_ram_miss_release_by_handle(int64_t handle, int64_t lease_id)` and an
   `exl3_ram_miss_release_abi()` returning a version integer. The device module (which already has
   CUDA, and enqueues with `cudaLaunchHostFunc`) receives the function's address as an integer,
   together with the ABI version it was built against, and refuses to enqueue when the versions differ.
   A shared header declares the signature as a `typedef`, and both modules `static_assert` it, so a
   change to one side breaks the build of the other rather than the process.
2. **Who owns the address, and when it is valid.** The address is the entry point of code in the host
   module. It becomes valid when that module is loaded and stays valid for the life of the process,
   because `_host_module()` is a `cache_once` `load_jit` and nothing in this tree drops the reference.
   That claim is the one this design rests on and I have **not** verified that `tvm_ffi` never unloads
   a JIT module: **[OPEN 17]**. It must be checked, and a test must assert `_host_module()` returns the
   same object throughout a process. If the host module were torn down first, the device module's
   stored address would dangle; the design makes that impossible by never tearing the host module
   down, not by detecting it.
3. **The ticket.** The enqueuer heap-allocates `{fn, handle, lease_id, cookie}`, passes it as the
   host function's user data, and the callback frees it; a failed enqueue frees it on the enqueuer's
   side. The cookie is checked in the callback so a stray pointer is refused, not followed.
4. **The registry is a leaked singleton.** `handle -> shared_ptr<RamTier>` is created once with
   `new` and never destroyed, so the static-destruction order at process exit cannot free it while a
   callback runs on a driver thread. (The existing `registry()` and `thread_registry()` in `host.cpp`
   are function-local statics, destroyed at exit; the release path must not use them as they are.)
5. **The callback body.** It takes a shared lock on the registry, copies the `shared_ptr`, drops the
   lock, calls `release_host_lease` (a `mutex_` section and a decrement; no CUDA call, no Python), and
   drops the `shared_ptr`. It is `noexcept`: every exception is caught and counted, because an
   exception leaving a driver thread terminates the process.
6. **A callback racing with, or after, tier close.** Close erases the registry entry. A callback
   either finds the entry and holds the tier alive for its own duration (the destructor then runs on
   the last `shared_ptr` drop, possibly on the driver thread, and must not call CUDA or Python; today's
   `~RowReader` does neither) or finds nothing and counts `releases_after_close` and returns. **No
   ordering has to be won for memory safety**: there is no pointer to dangle. "Draining first" is
   therefore not what makes it safe, and it does not need to be proven complete for that purpose.
7. **What the drain establishes, and how: the slabs, not the tier object.** The shutdown sequence's
   S3 (14.3) is a device-wide `torch.cuda.synchronize`. CUDA orders a host function after the stream
   work enqueued before it, and a device-wide synchronize does not return until every stream's work,
   host functions included, has completed. So when S3 succeeds, **no host function is queued or
   running on any stream**, and freeing the slabs is safe. Admission closing (S1) is what stops new
   enqueues: `enqueue_release` checks a `closing` flag under the registry lock and refuses, so nothing
   can be enqueued between S3 and the free. If S3 cannot succeed (a CUDA error, a timeout), the slabs
   are quarantined (14.2) and any callback that later fires only touches the registry, which is
   safe by rule 6.
8. **A failed enqueue leaves an uncertain lease.** If `cudaLaunchHostFunc` returns an error (a sticky
   CUDA error, say), the release will never run. The lease is not released by anything else (R1a: the
   poll must not double-release), so the tier is marked `poisoned`, the counter
   `release_enqueue_failures` is bumped, and the shutdown treats a poisoned tier as "completion cannot
   be established": it quarantines rather than frees. This is the same rule as any CUDA error in S3.
9. **No blocking under `mutex_`.** The callback takes the tier's `mutex_`; the service holds it only
   in short sections and never across I/O or while waiting for anything (it must stay so: state it as
   an invariant of `RamTier`). A callback that waited for `mutex_` would stall the executor stream's
   later copies, so the critical sections it contends with are bounded and short.
10. **Stream lifetime.** The executor stream that carries the callbacks belongs to the promotion
    executor and lives for the process (Task 8's design); a callback is not enqueued on a stream that
    may be destroyed first.

Tests this design requires before step 7 lands: a released lease through the callback path; a
callback after `close` is a counted no-op and touches nothing; a stress test in which a thread fires
callbacks at random times while the tier is closed and re-opened (must not crash under a sanitizer
build, and must never decrement a different tier's lease: the handle must not be reused while
callbacks are outstanding, so handles are never recycled); an enqueue after `closing` is refused; a
failed enqueue poisons the tier and shutdown quarantines; the callback survives an exception thrown
by `release_host_lease`. The `_host_module()` identity assertion covers rule 2.

If any of this proves awkward, the fallback of 17.1 (poll in `before_host_use` with
`stream.query()`) removes the callback and the whole cross-module question, at the cost of the
weaker guarantee stated there.

---

## 18. What has to exist before this can be accepted

This is a list of required *code changes and tests*, so the implementation task has a
checklist. None has been done.

### 18.1 Symbols that change (no code written)

| File | Symbol | Change |
|---|---|---|
| `host.cpp` | `Tier`, `take_slot_locked`, `assign`, `release`, `serve`, `pump_demand`, `RamThread::run`, the `RowReader::read` give-up lambda | leases and slot generations; tri-state slot choice; `retire_leases()`; skip-0 in `pump_demand` and `pump_advice` (D6; not required by this protocol); `Header.shutdown` |
| `host.cpp` | `exl3_ram_miss_sim_post`, `_sim_wait` | extend the host-side simulated device with `sim_ack`, `sim_terminal` and the lane request, so the whole protocol is testable on CPU with no GPU |
| `device.cuh` | `exl3_ram_miss_post_kernel`, `exl3_ram_miss_wait_kernel` | `LaneRequest`; row-result acquire/validate; `go_count`; terminal; poll on fatal/shutdown (D4); 64-bit acquire/release |
| `device.cuh` | new `exl3_ram_miss_ack_kernel` | section 7.4 |
| `ops.py` | `PAGE_BYTES`-style constants for the block; `STATE_WORDS`; `new_lease_block`; `Exl3RamMissDevice` | append `epoch`, `pending_epoch`; allocate/validate the block; ack wrapper |
| `srt_ram_miss.py` | `Exl3RamMissRowBackend.translate` and its copy | pass `go_count` to the copy; launch the ack kernel after the copy |
| `srt_ram_miss.py`, `ops.py`, `host_tier.py`, `expert_stream.py` | `Exl3RamMissService.shutdown`, `_stop_live`, `release_host_slabs`, `ExpertPinnedHostCache._release_slabs` | sections 14.2-14.4 (`expert_stream.py` is outside the four files the plan lists for Task 5; flagging it) |
| `test_exl3_ram_miss_device_args.py` | the layout agreement test | extend to the lease block constants (three-way agreement stays one test); see the constraints below |
| `host.cpp` / `attach` | `Exl3RamMissService.attach` | enforce `graph_gather_rows <= kMaxIds` (D7); today there is no such check |

Constraints from the existing tests that the implementation must respect (found by review,
not by me):

- `test_exl3_ram_miss_device_args._constants` parses **every** `constexpr <type> kName = <expr>;`
  line in `host.cpp` and `device.cuh`; expressions may use only integers, `+`, `-`, `*` and
  already known names, and a duplicate name in one file breaks it. So the block's
  `constexpr`s must stay `+ - *`, and the tag encoding (`tag << 56 | G56`) must live **in
  code, not in a `k` constexpr** (no `<<`, `|`, `/` or `sizeof` in a constexpr line).
- The device-state check compares `STATE_WORDS` to a hard-coded dict exactly. Appending
  `epoch` and `pending_epoch` means editing `device.cuh`, `STATE_WORDS` and that dict together.
- The copy kernel's count argument must be an int32 CUDA tensor of shape `[1]`
  (`_validate_plan`), so `go_count` is that, not a scalar or an int64.

### 18.2 Tests required (each maps to a Task 5 item or a defect)

**Audited against five ways a check can be unable to fail** (`#audit` below, 2026-09-21): a fault
injected where it short-circuits before the code under test; an assertion over a scan or
collection that can be empty; a test whose only failure on old code is a missing name; a stub
whose default makes the interesting state unreachable; and a test of a model or simulator where
the claim is about the service or the kernel. The first version of this list had all of them.
Each test below therefore states its **class** (what it actually exercises), its **precondition**
(what must be asserted first so that it is not vacuous) and the **mutation it must fail under**
(named, and to be demonstrated by applying it, not assumed).

Classes: **[service]** the real C++ service driven through the CPU simulated device; **[sim]** the
simulated device only, which is code written from the same spec and is *not* evidence about the
CUDA kernels; **[python]** the Python shutdown/quarantine wiring; **[kernel]** the CUDA kernels on
a GPU. A [sim] test is a consistency check of the scaffolding, never a claim about the kernel.

Every [service] test needs test-only introspection (`slot_leases(row, slot)`, `slot_generation`,
and counters `deferred`, `deferred_reuse`, `lease_double_signal`, `late_after_terminal`). **On
today's code each of them fails with a missing name, which is not evidence of anything**: the
evidence is the mutation column, applied to the finished service.

| # | Test | Class | Precondition to assert first | Must fail under |
|---|---|---|---|---|
| 1 | **Lease pressure.** A lease held with its ack withheld, on a tier whose capacity is exhausted so that the leased slot is the *only* possible victim of the next request; the request must defer (not fail), then be served after the ack is delivered. Include a RAM *hit* lane. | service | `deferred > 0` and `demand_done` has not advanced while the ack is withheld; `slot_leases == 1` for the hit slot and for the loaded slot | `take_slot_locked` ignores `leases` (with spare capacity the pressure takes another victim and a test without the exhausted-capacity setup passes on this mutant); a demand fails instead of deferring |
| 2 | **Delayed ack, newer request present.** A sentinel is written into the leased slot's slab bytes *after* publication; a newer request or advisory that reserves and reads runs; the sentinel and `slot_generation` are unchanged. Use the existing `_sentinel` and `_exact_or_untouched` helpers of `test_exl3_ram_miss_split.py`, requiring *untouched* (the helper also accepts a whole exact row, which is wrong for a leased slot). | service | the newer request's reservation ran (an eviction or `advisories` counter moved for another slot) | eviction that reloads the *same* expert into the leased slot with identical bytes (a byte comparison cannot see it; the sentinel and the generation can). The original wording cited the existing poisoned-recycle tests as the technique; they poison reader-owned descriptors and slots the reader is about to write (`test_poisoned_descriptors_and_slots_are_recycled...`), and detect nothing about a rewrite of a slot nobody is reading |
| 3 | **Duplicate lanes.** Two lanes, one expert. | service | `slot_leases == 2` **before any ack**; after one ack `slot_leases == 1` and the slot is still not a victim (repeat item 1's pressure); after the second, 0 | leases deduplicated per expert (a test that only checks the end state passes); an ack that releases every lease on its slot |
| 4 | **The rows of section 12.** One test per row *that is observable*, naming its row. **Service-observable:** F1 (a read fault injected *after* reservation, so it is not short-circuited; assert `slot_leases == 0` and no `RowResult`); F2 in *both* orders (terminal delivered before the service serves: `late_after_terminal`, no lease; terminal delivered after grant: retired by the mask); F8 (a lapped request); F10 (item 9); F11 (item 3). **Not observable on the CPU:** F5 (CUDA error), F6 (device hang), F7 (service hang, existing watchdog), and the *device halves* of F1, F2, F12, F13, which are behaviours of the kernels ([kernel], items 12, 5b). F13 exists only if a protocol bug exists; an injected recycle tests the simulator's ack, so it is [sim]. | service / sim / kernel | per row as stated | F1: leases granted at reservation and not voided on failure; F2: a lease granted after a terminal that nothing retires |
| 5 | **Skipped copy emits no ack.** With `go_count == 0` no `LaneAck` word is written. | **sim only** here; **[kernel] on the GPU** | assert the words are all zero *and* that the simulated wait did run its abort path | On the CPU this checks the simulator against the same author's spec. A kernel that acknowledges `[0, count)` instead of `[0, go_count)` passes every CPU test; only a GPU test that runs `exl3_ram_miss_ack_kernel` catches it. Do not cite the CPU version as evidence for the kernel |
| 6 | **Terminal mask retires exactly its lanes; a lane signalled by both an ack and a mask is counted, not decremented twice.** | service | inject both signals for one lane and deliver both *before* `retire_leases()` runs (pump once after both are visible), else the first retirement empties the lane and the second is a trivial no-op; a partial mask to exercise the per-lane state machine (the real device publishes only full masks until Task 6) | retirement decrements per signal instead of per lane state. **The old wording, "double-retire is an internal error", disagrees with the model and the design, where the second signal is ignored by the lane state machine; specify the counter `lease_double_signal` and assert exact `leases`** |
| 7 | **Generation wrap.** (a) the demand/advisory wrap, written (`test_exl3_ram_miss_wrap.py`), fails on the old `pump_demand`/`pump_advice` and passes on the fix; (b) leases retire across the wrap; (c) the stale-acknowledgement case; (d) a lap that crosses the wrap. | (a) service; (b)-(d) service + sim | (c) the stale word must have the **same low 32 bits and a different epoch** as the awaited generation, else it is rejected for the wrong reason and the test passes with a 32-bit compare; assert the lease is *still held* while the stale word is present, and released by the real ack. (d) assert a lap occurred (`overruns`/resume counter) with the service held back by `pump()` stepping, and that the armed request after it is served | (c) a service that compares only the low 32 bits of the generation; (d) a service that counts epochs itself (the model's counterexample) |
| 8 | **Request-slot reuse.** A request slot is reused only after its lease row retired, and a deferral does not flood the stage ring. | service | acknowledgements withheld so request `G + 16` actually arrives with `G` unretired: `deferred_reuse > 0`; with instant acks it never defers. Assert the stage ring gained **one** record for the deferred request, not one per poll | `defer_reuse=False` (a request slot overwritten with a granted lane unretired: leaked lease); a deferral that re-enters `begin_stage` every poll |
| 9 | **Pause and shutdown.** (a) a *graph-lane* lease outstanding: the pause is refused; (b) a *host* lease outstanding: the pause is **granted** (R2; the first version tested only the refusal, which passes for an implementation that refuses whenever any lease exists); (c) a fake CUDA error and a fake `synchronize` that outlasts its deadline each select quarantine. | (a),(b) service; (c) python | (c) patch `release_host_slabs` **before** the cache is built (the finalizer binds the function at creation, which the first version of `TestQuarantine` had to work around); assert the fake `synchronize` was actually called; release the blocked helper thread at the end; assert the slab is alive via a weak reference and no unregister was recorded | (a)/(b): a pause that counts host leases; (c): a shutdown that frees when the sync did not complete |
| 10 | **The worker never waits.** With every acknowledgement withheld the service still serves *another row's* request, and `pause`, `stop` and cancellation complete within a bound. | service | a demand is **deferred** (`deferred > 0`) when `pause`/`stop` is issued, otherwise an idle worker passes; a watchdog as the thread tests already have | a `serve()` that spins on the acknowledgement (hangs `pause`/`stop`; the watchdog fires) |
| 11 | The `ld.global.nc` experiment of section 6.6. | kernel | done: `NC_VISIBILITY.md` | see there |
| 12 | **Graph parity and fail-closed.** Byte-exact output with leases on versus off; on an injected timeout `go_count` is 0 and the copy reads nothing. | kernel | the timeout must be injected **after the post and before the demand is served** (service paused), else there is nothing to copy and the test cannot fail; the destination is pre-filled with a sentinel and must be unchanged; a poisoned source slab | a copy that runs after a failed wait (D1); an ack kernel that acknowledges `[0, count)`; an ack published before the copy completes |
| 13 | Cost of the added fences, the ack kernel, and arming every `count > 0` record (OPEN 5, OPEN 11). | kernel | a measurement, not a test | n/a |
| 14 | **The stage ring is not flooded by a deferral** (7.1). | service | as item 8 | a per-poll `begin_stage` |
| 15 | **Stale or repeated `LeaseRef` is a counted error, not a second decrement** (R1); **the lease API does not bump `kVersion`** (R8). | service | assert the counter value and `version()` unchanged around a lease | a release that decrements twice; a lease that bumps the version |
| 16 | **The release-callback tests of 17.3** (a callback after close is a counted no-op; a callback racing with close and re-open never crashes and never touches another tier). | service | the callback must be fired **from another thread while the tier closes**, not called inline (an inline call cannot race) | a callback that dereferences the tier by pointer |
| 17 | **The model's mutants replayed on the service.** | service | see the ledger below | see the ledger below |

**Mutation ledger** (section 20.0 says each model mutant becomes a regression test on the real
code; this says which can and which cannot). Applied to the finished service and shown to fail:

| Model mutant | Service test | Note |
|---|---|---|
| `leases=False` | items 1, 2, 3 | with the exhausted-capacity precondition |
| `defer_reuse=False` | item 8 | needs withheld acks |
| `gen64=False` + stale ack | item 7(c) | same low 32 bits, different epoch |
| `echo_gen=False` | item 7(d) | a lap that crosses the wrap |
| `defer_leased=False` | item 1 (defers rather than fails) | |
| `retire=False` | item 10 | |
| `free_needs_sync=False`, `free_waits_executor=False`, `host_admission_closed=False` | item 9(c), the shutdown tests | python |
| `host_guard=False`, `eager_host_guard=False`, `pause_counts_host` | item 9(a)/(b), step 7 tests | |
| **`ack_after_copy=False`, `fail_closed=False` (fail-open), `detector=False`** | **none on the CPU** | **kernel properties: only items 5(kernel) and 12 can catch them.** The simulator would only re-implement the rule under test |
| `copy_waits_on_serving` | none | a property of the Task 8 copy path, not of the service |

**Mutation run status** (naming a mutation is necessary but not sufficient: it must be *run*
against finished code). Runs were made on divix01 with `/data/models/slang/.venv/bin/python`
(from `t1-instrument`), from a plain `git archive HEAD` export (`48da151fb0`), CPU only, cores 0-63,
a private JIT cache directory. Status today:

| Test or check | Mutation | Status |
|---|---|---|
| Item 7(a): the wrap tests (`test_exl3_ram_miss_wrap.py`, 10 cases) | remove `skip_zero` at each of the four sites in `host.cpp`, one at a time | **RUN, all four caught; unmutated 10 pass.** Demand increment (line 1538): 4 tests fail (both waiter tests and both page states). Advisory increment (line 1574): 2 fail. Demand lap resume (line 1526): 1 fails. Advisory lap resume (line 1549): 1 fails. Each lap resume is guarded by exactly one test, so those two are thin |
| `TestQuarantine` (step 6, part) | `quarantine()` without `detach()`; no extra reference (`Py_IncRef`) | **RUN, both caught** (the second only after I removed a strong reference that made the first version of that test unable to fail) |
| Lease layout agreement (step 1) | change a constant on the Python side (`LANE_ACK_BYTES` 8 to 16); change one in the `.cuh` (`kLeaseTerminalBytes`) | **RUN, both caught** (2 tests fail on the Python-side change: the layout test and the agreement test) |
| Lease block allocator (step 1) | skip the alignment check; allow generation 0 | **RUN, both caught** (one test each) |
| Host-lease agreement guard | rename, partial, drifted and unmirrored constants | run as **input mutations** on synthetic sources (`test_the_host_lease_guard_can_fail`); not a mutation of `host.cpp`, which has no lease code |
| Items 1, 2(part), 3, 6, 7(b)-(d), 8, 10, 14 and the service half of 9 (steps 3a-3c) | 17 mutants of grant/publish/retire (3a), 12 of deferral (3b), 5 of pause and wrap (3c), on divix01 against the landed service | **RUN, all killed after two survivors were fixed.** 3a: "grant ignores whether the request succeeded" survived (its test left no lane resident, so the grant could not happen anyway); 3b: "every refused retry counts as a new deferral" survived (no test retried). Both now die to a test that reaches the path. Test-first did not prevent either. Not testable on the CPU: the write order of a row result's payload versus its ready word, and that the pause's own retirement pass matters (the running thread retires first) |
| Item 2 (delayed ack with a newer request present) | as item 1 | **PARTIAL**: with injected leases (step 2, run) the leased slot is never chosen and never rewritten; with service-granted leases the same predicate is exercised by the deferral tests. **Superseded by R4 (section 20.2f): a test now fires a newer demand and an advisory at a served request's leased slot and checks a sentinel, so item 2 is demonstrated as written on the CPU service.** |
| Item 4 F1/F2 service-observable rows | grant on failure; a terminal already seen | **RUN** (3a) |
| Items 15 (stale `LeaseRef`, `kVersion`) and 16 (callback race) | | **NOT RUNNABLE**: the host-lease API is step 7 |
| Item 9(c) shutdown wiring (`Exl3RamMissService.shutdown`) | frees when the sync failed | **NOT RUNNABLE**: needs the file another change holds. Only the slab half is run (row 2) |
| Items 5(kernel), 12; `ack_after_copy`, fail-open, detector | see the ledger | **NOT RUN**: need step 4 and the GPU |

The nearest the existing tree comes to demonstrating a *service-side* lease mutation today is
none: the closest is the wrap tests above, which mutate existing service code and are the only
service-level mutation evidence in this list.

#### Audit

Findings against the first version of this list, each shown at the source or by a named
mutation the test would miss: (A1) item 2 cited a technique that does not detect a rewrite of an
unread slot (`test_exl3_ram_miss_split.py`, `test_poisoned_descriptors_and_slots...`); (A2) items 1
and 2 pass on the mutant "eviction ignores leases" without the exhausted-capacity setup; (A3) item
3 passes on lease deduplication if only the end state is asserted; (A4) item 4 was an instruction,
not a test, and covered rows that cannot be observed on a CPU; (A5) item 5 tested the simulator
against the same author's spec and could not fail on the kernel; (A6) item 6 asserted "internal
error" where the design ignores the second signal, and needed both signals visible before
retirement; (A7) item 7's stale-word case passes with a 32-bit compare unless the low 32 bits
match; (A8) item 8 never defers with instant acknowledgements, and the stage-ring flood test was
absent; (A9) item 9 tested only the refusal, which an "always refuse" pause passes; (A10) item 10
passes on an idle worker; (A11) item 12's timeout must fire after the post; (A12) every service
test fails today with a missing name, which is not evidence, so the mutation column is the
acceptance criterion; (A13) three model mutants have no CPU analogue.

### 18.2a Tests still owed after step 3, each with a path witness and a property

Rule from the mutation work (two survivors of mine in test-first steps, six of `t2-scheduling`'s): a test needs
**both** a *path witness* (something that proves the intended path actually ran, so a shortcut that produces the
same outcome fails it) **and** the *property* asserted on that path. Naming the outcome is not enough, and
naming the path is not enough. Each entry says what the shortcut would be.

| # | Test | Path witness | Property | The shortcut the outcome alone would accept |
|---|---|---|---|---|
| R1 | **A long deferral does not trip the watchdog's stuck rule** (thread mode, real watchdog, in a subprocess because an abort kills the interpreter). This corrects the request "abort after a long deferral": the requirement is that a *deferral* never aborts, while a hung read still does (existing test). | `deferred == 1`, the deferral observed to be older than `fatal_wait` on the test's own clock (start the thread with `fatal_wait_s` well below the wait), `busy_since_ns() == 0` and `busy_seq == 0` sampled throughout | the process is alive at the end; the demand is served after the lease retires | a `fatal_wait` larger than the wait (nothing could abort); a deferral that is short. Mutation: the deferral marks itself busy (already killed in pump mode by the `busy_since` assertion; this is the real-watchdog version) |
| R2 | **An advisory arriving while a demand is deferred is processed, not skipped, and touches nothing the deferred demand needs.** Replaces the weaker check in `test_exl3_ram_miss_lease_thread.py`, which asserts only that `advise_done` advanced (a stale-skip advances it too). | posted **after** `deferred == 1` is observed; `advisories` counter +1 and `advisories_skipped` unchanged (it entered `serve`, it was not skipped as stale); `demand_pending()` true at that moment | it gives up before reading (`advisory_rows == 0`); no leased slot is taken; the deferred demand's `demand_done` is unchanged; an advisory for **another row** reserves in that row's own tier | an advisory marked stale and dropped (advances `advise_done`, does nothing); an advisory that reads while a demand is deferred (priority violation); an advisory that takes a leased slot |
| R3 | **Retirement between the reader's batches** (needs the missing call site 2 of 7.5 first; `t4-packworker` has confirmed the give-up lambda survives their refactor untouched and is evaluated on more turns in worker mode, so the call must be **idempotent per batch**). | a delayed read (`inject(delay_s)`) so the request is provably in `serve`'s read phase when the acknowledgement is delivered; `demand_done` not yet advanced | `leases_acked` increments **during** the read, not after it | retirement only at the top of `pump_demand` (today's behaviour): the counter moves only after the read returns |
| R4 | **Item 2 of 18.2 as written: a served request's leased slot keeps its bytes under a newer request and an advisory.** | the tier is exhausted so the leased slot is the only possible victim; the newer request's reservation ran (`deferred` moved, or `evictions` moved on another slot) | a sentinel written into the leased slot's slab bytes after publication is intact, and its `slot_generation` and `SlotGen` word are unchanged | eviction that reloads the same expert into the leased slot with identical bytes (a state-and-generation check would not see a missed bump either) |
| R5 | **CI runs what pytest runs, and the collected count is asserted, not the exit status.** Run each lease file as CI does (`python3 <file> -f`) and compare the number of tests it ran with `pytest --collect-only -q`. Use `CUDA_VISIBLE_DEVICES=9` (a nonexistent index: `torch.cuda.is_available()` stays False and `sglang/test/test_utils.py:246` parses it), never an empty value (an empty value raises `IndexError` at import and the file errors at collection); run each invocation in its own session (at least one test in this corpus kills its process group). The class-order hazard (CI runs classes alphabetically) is tested by the same run. | the two counts, side by side, each nonzero | equal counts, and no file that passes under pytest and fails under CI | **two** ways to look green while running nothing: a missing `__main__` (exits 0, zero tests) and an import error under an empty `CUDA_VISIBLE_DEVICES` (errors at collection). Both are invocation-level, so the exit status lies under either. Also a `-k` selector that matches only a new test, which reports a mutant caught by a test that never ran |

**A general rule that R1 illustrates: a test that asserts something does NOT happen is vacuous by default.** "The
watchdog does not abort on a long deferral" passes on any build in which the deferral was short, or in which
`fatal_wait` exceeded the wait, so nothing could have aborted. Such a test needs a witness that the *conditions for
the event were present* (here: the deferral measurably older than `fatal_wait` on the test's own clock, with the
rule's input `busy_since` observed at 0), or it proves nothing about the event's absence. R1's first wording in this
document said the opposite ("abort after a long deferral"); the requirement is that a long deferral does not trip the
stuck rule.

**Claims about the service that no CPU test defends** (code-reading claims; a device-side harness would be the way
to defend them, and the list is now long enough to argue for one): the write order of a row result's payload versus
its ready word (`grant_lanes_locked`); `kBusySeq` cleared before `demand_done` (this one is known to be undefendable
by CPU tests, not merely undefended: a mutant that clears it after the done store is not caught, because a host
poller cannot land in a sub-microsecond gap, and the plan labels it READ FROM SOURCE, NOT EXECUTED); that the pause's own retirement pass
matters (the running thread retires first); and that the grant precedes `set_status` (only the ordering before
`demand_done` is observed, through `inject_done_stall`).

### 18.3 The model check (done, bounded)

`analysis/dsv41-drive/lease_model.py` is a self-contained explicit-state model of this
protocol; `test_lease_model.py` (36 tests, about five minutes, pure Python) pins what it
finds. It has two threads (the device's post, wait, copy, acknowledge, consume; the service's
observe, reserve, load, grant and publish, retire), pinned slots that an advisory may evict
and reload whenever the service is idle, injectable timeouts and read failures, an optional
shutdown with a CUDA error, and a 32-bit sequence shrunk to a handful of values that starts
next to its wrap. Device stores to acknowledgement and terminal words reach the service in
any order across different words. It checks I1 directly (ghost state: which slot each device
lane holds), the shipped-bug shape (a request accepted with bytes that are not the
requested experts'), lease underflow and leak, deadlock, and "a run with no injected fault
must not fail stop".

Results, all exhaustive over the bounds (up to about 2.4 million states each, complete):

| World | Result |
|---|---|
| Designed protocol; 3 requests; timeouts and read failures on; ring 2 | no violation |
| Same, no timeouts or faults (liveness) | no violation, no deadlock, no leak, no spurious fatal |
| Same with a victim-forcing request mix | no violation; the deferral on leased victims and on an unretired request slot were both reached |
| All request shapes, 2 requests; ring 3 across the wrap; shutdown with a CUDA error | no violation |
| **Design as an earlier draft had it** (service counts epochs) | **counterexample**, section 11.3; fixed by echoing the device's `G` |
| Mutants: eviction ignores leases; ack published at commit; 32-bit generations with a stale ack; request slot reused before retirement; a demand fails instead of deferring; the service never retires; Python frees without a sync | each found, with the violation named in the test |
| Mutant: leases removed, detector kept vs removed | the mechanism fails either way; only without the detector are wrong bytes **accepted** (E6 does what section 6.5 says) |
| Today's protocol, no faults, temporal exclusion | no violation (the model does not cry wolf) |
| Today's protocol, one timeout | I1 violated (D1); wrong bytes not accepted, because `keep = 0` contains it |
| Today's protocol without the exclusion | wrong bytes accepted (D2) |
| Service that does not skip 0 (D6), on the increment or only on the lap resume | a phantom sequence, nothing else; the fixed service is clean, including with three unarmed records posted across the wrap so that the lap resume lands on 0 |
| **Task 8 extension.** Designed protocol plus a promoter (host lease, copy, release) and an eager caller (pause, assign, resume); 2 requests; with and without timeouts and read failures | no violation, no deadlock, no leak; the eager pause was granted **while a host lease was held**, and a demand was deferred for want of an unleased victim (both reached) |
| Mutant: the service evicts a host-leased slot; the eager assign takes a host-leased slot | each is I1 violated (the two paths are checked separately: R5 covers `take_slot_locked`, `assign` and `release`) |
| Mutant: the eager pause also waits for host leases (R2 broken) | `PauseBlockedByHostLease`: the pause is refused with graph leases at zero and only a promotion's lease in the way |
| Mutant: the promotion copy waits on the serving stream once it holds a lease (**A3 broken**) | **Deadlock**, found by search: the demand waits for the service, the service defers because its only victim is the host-leased slot, the lease is released only when the copy finishes, and the copy waits for the serving stream. This is the cycle `PROMOTION_ASYNC.md` 9.2 found by reading, so A3 is now shown necessary, not asserted |
| Mutant: the promoter never releases its lease | Deadlock (a demand deferred behind a lease that never retires; with the device timeout this becomes a fatal, which is why R4 bounds the hold time) |

What this does **not** establish, stated so a pass is not over-read:

- It is sequentially consistent apart from the one relaxation above. A missing fence, the
  `ld.global.nc` question (6.6), or PCIe reordering of the *service's* stores are outside it.
- It is Task 5's whole-request protocol only. Per-lane copy, the finalize kernel and the
  terminal mask with a partly copied request (Task 6) are not modelled, so the
  "terminal names lanes" argument of section 13 is exercised only with all lanes named.
- The bounds are small: 2 lanes, 2 slots, 3 experts, at most 3 requests, one advisory, and
  request shapes restricted in the 3-request runs. A hole that needs more of any of them is
  not excluded. Visited states are kept as 64-bit hashes; a collision could hide a state
  (about 1e-5 at these sizes).
- It models one stream and serialized replays (assumption 1 of 1.1). Concurrent graphs are
  outside it.
- It does not model the lease-mode arming cost, or the watchdog beyond "fatal is followed by
  abort". The Task 8 side (host leases, the eager pause and its graph-lane/host split) *is*
  modelled, coarsely, as the next bullets say.
- **The Task 8 extension is thinner than the graph side.** One promotion, one eager cycle,
  2 requests and four request shapes (against 3 requests for the graph-only runs), so its
  world is smaller by one request. It models the *contract* (lease, copy, release; pause and
  assign), not Task 8's protocol: no promotion admission, no `lease_on_ready` window (a
  mutant for R3 would need the promoter to hold a slot number across a gap, which my
  `acquire_host_lease` lookup-by-expert does not do, so the model has no such window), no
  capped copies or executor-stream events, no bound on hold time (R4's "release
  leased-not-yet-COPYING leases immediately" is not modelled), no stale or repeated
  `LeaseRef` (R1), no counters (R7) and no `kVersion` effect (R8). The stream dependency is a
  coarse "the copy is queued behind the serving stream's tail at acquisition time". The
  eager pause's synchronization is idealized (device idle, its stores landed). R2's safety
  claim is therefore checked *as an eviction-predicate property over the modelled
  interleavings*, not against Task 8's real copy path.
- Its advisory pressure is applied only while no demand is visible to the service, and to
  the single modelled row. In the real service an advisory for the *next* row can start while
  a demand is deferred (section 8); that is a different tier and cannot take the deferred
  demand's victims, so I do not expect it to change the result, but the model does not show
  it.

**Independent review, and what changed after it.** `LEASE_MODEL_REVIEW.md` reviewed the
model at `845a54994e` and found the transcription faithful where it can be checked, with no
headline result contradicted. Its findings are recorded here so they are not lost:

- **F1 (real design hazard the model could not see).** The promoter and the eager caller were
  independent actors; in the design they are one thread. Now modelled (`blocking_eager`,
  `poll_release`): the cycle is a Deadlock, and it is gone with a callback release. Applied to
  17.1 as R1a.
- **F2.** R6 (shutdown with host leases) was unmodelled and 18.3 contradicted itself about it.
  Now modelled: `host_admission_closed` and `free_waits_executor`, each with a mutant
  (`HostLeaseAfterFree`, `FreedWhileReading`). The contradiction is removed.
- **F3.** Terminal retirement cannot be shown *necessary* by the search: after a fatal the model
  lets the process die and a leak after a fatal is invisible, so the "retired by terminal"
  counter shows the step is reachable, not that the design needs it. E2(b), E4 and section 13
  for that step rest on the argument. (In the real system the watchdog also aborts the process
  after a fatal, so the rule matters for runs that continue past a failed wait, such as a
  shutdown in progress.) Not changed.
- **F4.** The D1 fix (fail closed, `go_count = 0`) had no mutant because an `assert` forbade
  one. Removed: the fail-open mutant finds `RecycledUnderReader` in 41,047 states and is a test.
- **F5.** Four mutants (`leases`, `host_guard`, `eager_host_guard`, `pause_counts_host`) fire on
  the removed rule at the first byte store or on reachability, before any reader has begun, so
  their tests show the rule matters, not the harm. Only the `leases=False` detector test shows
  harm.
- **F6.** The stream-dependency model is sound for **presence** and optimistic for **absence**.
  The modelled copy is released as soon as the device posts the next request, but a real
  `wait_stream(producer)` orders after everything already enqueued, several requests of a
  replay. So the deadlock found is real (the real dependency contains the modelled one), and
  "A3 is necessary" stands, but a cycle at request k + 1 is invisible: **a clean run with
  `copy_waits_on_serving=True` would not show a weaker dependency safe.**
- **F7.** `Config.map_read` was a dead knob (the device resolves through the map exactly when
  `protocol=False`). Removed. "Today" is `protocol=False`, and the D2 result is "today without
  the exclusion", not a separate map-read configuration.
- **F8.** Ring 2 or 3 against 16, and a sequence range of 8-11 against 2^32. At ring 2 the wrap
  makes seq 7 and seq 1 share request slot 0, where the real ring goes from index 14 to 0: harsher
  than reality, so it shows the reuse rule matters at ring 2 but does not establish it at ring
  16 (OPEN 8). Slot generations and content versions are modulo 4 (four reassignments inside one
  lease window would be needed to alias, which the lease forbids; only mutants reach it).
- **F9. Blind spots the limits above did not name:** (1) thread identity (F1, now partly
  modelled for the eager caller only); (2) post-fatal behaviour: the model stops the device after
  the first fault, whereas the real wait and copy kernels of the rest of the replay still run
  (host_rows from the map, copy executed, `keep = 0`) until the watchdog, so D1 is wider in
  reality than in the model; (3) multi-word atomicity: the record and the lane request are single
  atomic writes and are read in one step, so a torn read between them (the real seqlock writes
  seq 0, a fence, payload, a fence, seq) is not representable; (4) copy concurrency: lanes are
  copied one at a time in the model and together in the kernel, which cannot hide an I1 hole
  (the ack follows all lanes) but cannot show an intra-kernel one; (5) shutdown with Task 8
  (now modelled, F2); (6) no hot set, and `wanted` is the lane list, whereas the real `protect` is
  routes plus need, so the victim pool is a superset (conservative for the designed protocol,
  optimistic for "today": if planned experts are not a subset of routed ones, OPEN 12, today can
  fail stop where the model cannot); (7) one tier and one row, with no cross-row ordering;
  (8) a CUDA error only during copy or acknowledgement.
- **Changes made after the review (mine, so not covered by it):** the F1/F2/F4 additions above
  and the removal of `map_read`. The reviewed version is `845a54994e`.

The model's own faithfulness is the thing to review first: `lease_model.py` restates the
service and device steps by hand from this document and the code, so a step I mis-transcribed
would make a pass meaningless. The mutants and the "today" runs are the guard against that
(they reproduce D1, D2 and D6 from the code as read), but they cannot prove the designed
side is transcribed faithfully.

---

## 19. Open questions and decisions

### Open (I could not determine)

- **[OPEN 1]** D6: does the demand sequence wrap really produce the phantom seq-0
  iteration? By reading, yes; not run. Settled by test 7.
- **[OPEN 2]** (resolved: required change, D7) `lanes <= 8` must be enforced at attach.
- **[OPEN 3]** Whether `torch.zeros(..., pin_memory=True)` gives 4096-byte alignment.
  Allocate with slack and slice if not guaranteed.
- **[OPEN 4]** Production tier capacities, so the size of `SlotGen[]`.
- **[OPEN 5]** Cost of two extra `__threadfence_system()` per post and of the ack kernel.
- **[OPEN 6]** (narrowed by the run in 6.6: cross-kernel `.nc` staleness not observed; PTX wording still unverified) The exact PTX ISA wording for `ld.global.nc` on the deployed toolkit, and
  whether L2 keeps system-memory lines across kernel launches. Section 6.6 gives the
  experiment.
- **[OPEN 7]** Every consumer of `ram_miss` and `unserved_misses`, and whether their
  meaning must be preserved exactly when the wait kernel stops consulting the map.
- **[OPEN 8]** The interplay of `pump_demand`'s lap skipping with deferral: argued in
  section 11.4, not tested.
- **[OPEN 9]** Where "concurrent graph execution ... is guarded" is implemented today. I
  did not find a guard; `state[kPosted]` is a non-atomic read-modify-write.
- **[OPEN 10]** Whether `io_uring_queue_exit` in `~RowReader` waits for an in-flight
  O_DIRECT read into a slab. Storage-side shutdown, not GPU-side; relevant to "drain
  storage" before freeing slabs.
- **[OPEN 11]** ~~The per-layer cost of arming every `count > 0` record when advise is off.~~ **MEASURED
  2026-09-21, superseded: re-measured at about 17 us per all-hit layer, ~0.68 ms per step at 40 layers (re-measured 2026-09-21 on an exclusively held card; the earlier ~8 us / ~0.32 ms figures are superseded, see `open11/results.md`)** (the original three runs agreed to 6%;
  `open11/results.md`). The exposed cost is the wait alone (+11 us); the acknowledgement kernel costs ~14 us in
  isolation but is largely hidden in the stream. Upper bound: a layer with a miss was already armed and pays
  nothing. Not the serving path, which cannot be measured until step 5 lands.
- **[OPEN 12]** Whether planned experts are always a subset of the routed experts in the
  post kernel's `protect` set. The design does not rely on it (7.1 step 4).
- **[OPEN 16]** (designed in 17.3; the design's own assumption is OPEN 17) The mechanics of the
  executor-stream release callback (17.1 R1a).
- **[OPEN 18]** (narrowed) Whether every production teardown reaches `Scheduler.release_host_resources()`:
  `Engine.shutdown()` kills its children's process tree, and whether a `ShutdownReq` always drains the scheduler first
  is untraced. The caller itself is identified (14).
- **[OPEN 17]** That `tvm_ffi` never unloads a JIT module, so the function address the device module
  holds stays valid for the process. Unchecked; 17.3 rule 2 rests on it.
- **[OPEN 15]** How a deferral interacts with the stage record's `observed`, `prev_done` and
  `backlog` fields (7.1).
- **[OPEN 14]** Epoch seeding for a second device incarnation over one lease block. The
  design is one incarnation per block (11.3); a device re-created over a used block would
  restart at epoch 0 while old acknowledgement words still carry epoch 0 generations.
- **[OPEN 13]** (ordering answered by review, production path still open: see D5) The order of the exit-time callbacks: `_stop_live` (`ops.py`), the
  `ExpertPinnedHostCache._release_slabs` finalizer (`expert_stream.py`), and the
  `Exl3RamMissHost._close` finalizer (`atexit = False`). If the slab finalizer runs while a
  GPU kernel can still read, the D5 hazard occurs in every normal exit today. Determine the
  actual order before choosing between 14.4 options (a) and (b); (b) is safe under any order
  only if it detaches the slab finalizer.

### Decisions I made that the owner may overturn

- **[DECIDE 1]** The lane list travels in a device-written `LaneRequest` in the lease
  block, with its own seqlock, so the request page stays untouched (as directed). The
  alternative, using the record's 44 spare bytes (offsets 84..127) for `count` and
  `expert[8]`, would get the seqlock for free but changes the validated page ABI; I did
  not choose it.
- **[DECIDE 2]** `_stop_live` (atexit) quarantines unconditionally; only
  `Exl3RamMissService.shutdown()` runs the ordered sequence (14.4 option b).
- **[DECIDE 3]** Lease at publication, not at slot reservation, for Task 5. Task 6 may
  want reservation-time leases; nothing here prevents it.
- **[DECIDE 4]** The acknowledgement is a separate kernel, not fused into the copy tail
  (6.4).
- **[DECIDE 5]** The service echoes the request generation from the device's `LaneRequest`
  instead of counting epochs (11.3). Forced by the model's counterexample, not a preference.

### Plan correction

The plan's Task 5 file list is Native service/pipeline header, `exl3_ram_miss.cuh`,
`ops/moe/exl3_ram_miss.py`, `srt/layers/moe/exl3_ram_miss.py` and the thread/GPU-graph tests.
The shutdown requirement cannot be met inside those files: `ExpertPinnedHostCache.__init__`
in `srt/layers/moe/expert_stream.py` creates the `weakref.finalize(... release_host_slabs ...)`
that unregisters the slabs at every process exit, and a quarantine that leaves it attached is
silently undone at exit. `expert_stream.py` (and `expert_host_tier.py`, where
`release_host_slabs` lives) belong in Task 5's file list, and the plan should be amended.

---

## 20. Implementation order (Task 5)

Design only, as everywhere in this document: this section says what to change in what order,
and none of it has been executed. It is a plan for the person who implements after the model's
transcription has been reviewed independently (if that review finds a mis-transcribed step,
this order may change).

### 20.0 Ground rules

- **Lease mode is a switch that defaults off, and off is today's protocol bit for bit.** Every
  step below leaves the tree passing all existing tests with the switch off. The switch is a
  new `SGLANG_*` environment variable, so it needs the env-var conventions
  (`python/sglang/srt/environ.py`; read `.claude/skills/env-var-conventions` first).
- **Every model counterexample and mutant becomes a regression test on the real code**, run
  through the CPU simulated device (`sim_post`, `sim_wait`, extended in step 3). The model's
  traces are the test scenarios; do not invent new ones and lose the ones already found.
- **Each structural milestone gets an independent review before the next starts** (plan:
  "no approval based solely on checklist completion"). Steps 3, 4 and 6 are the milestones.
- **The native service is shared.** Steps 2 and 3 edit `exl3_ram_miss_host.cpp`, which other
  work is editing; check `git status` and the freeze state before starting, and do not begin
  while a GPU arm series depends on the code.
- Already landed and to be **built on, not redone**: `skip_zero` at all four sites
  (`85cdbf9382`) and the refusal to attach a layer whose gather is wider than the service's
  lanes (`8c78749b35`, D7).

### 20.1 The ordered steps

| # | Change (symbol, file) | Test that covers it | Constraint that bites | Needs GPU |
|---|---|---|---|---|
| 1 | **Constants and block allocator, no behaviour.** `ops.py`: lease-block constants (`LEASE_RING`, `LEASE_LANES`, offsets of area H/S/D from 4.3) and `new_lease_block(rows, capacities, pin)`, allocating `bytes + 4096` and slicing to a 4096-aligned view (there is no alignment check today; add it). Mirror the same names in `host.cpp` and `device.cuh`. | Extend `test_exl3_ram_miss_device_args` so the lease constants join the existing three-way agreement test; a test that an unaligned or unpinned block is refused. | The `_constants` parser accepts only `constexpr <type> kName = <expr>;` with integers, `+ - *` and known names, and fails on a duplicate name. No `<<`, `|`, `/`, `sizeof` in a `k` constexpr; the tag encoding stays in code. | no |
| 2 | **Slot generations and the eviction predicate.** `host.cpp`: `Tier` gains `leases` and `slot_generation`; `SlotGen[]` mapped writes with the `_mm_sfence()` of 6.5; `take_slot_locked` returns the tri-state {slot, deferred, none} with the cause (graph or host); `assign`/`release` refuse a leased slot; new counters. `exl3_ram_miss_open` takes the block tensor and writes the immutable header and `RowTable` (the service is the only writer); `ops.py` `Exl3RamMissHost.__init__` creates the block itself when none is passed, so no test call site changes. Nothing grants a lease yet. | All existing `test_exl3_ram_miss_tier` / `_split` / `_thread` / `_advisory` unchanged and green; new CPU tests for the tri-state, for `SlotGen` bumping before the first byte store, for `release` of a leased slot throwing. | Append the new counters at the end of `enum Counter` before `kCounterCount` **and** the same position in `COUNTERS` (`ops.py`), which is index-aligned by convention. `kVersion` must not move on a lease change (R8). | no |
| 3 | **Admission, grant, publish, retire, terminal; the CPU device.** `host.cpp`: read `LaneRequest` with the seqlock re-check; take `G` from it (11.3); `Outstanding` ring; in `serve()` grant and publish per lane (6.1 order, RAII undo); `retire_leases()` called from the top of `RamThread::run` and from the give-up lambda of `RowReader::read`; the pause acknowledgement's graph-lane `outstanding == 0`; the terminal check and `late_after_terminal`; deferral. Extend `exl3_ram_miss_sim_post` / `_sim_wait` and add `sim_ack`, `sim_terminal` so the whole protocol runs with no GPU. | Section 18.2 items 1-4, 6, 7(b)-(d), 8, 10, 14 and 15, each with its precondition asserted and its mutation demonstrated (the ledger in 18.2; item 5 is sim-only on the CPU and is not evidence for the kernel), plus the model counterexamples that have a service analogue, replayed at scenario level (20.2b). The wrap tests (`test_exl3_ram_miss_wrap.py`) still pass. | **A deferral must return before `begin_stage`, or push nothing**, and retry only when `retire_leases()` changed something or a `Terminal` appeared, or the 8192-slot stage ring floods per poll (7.1, OPEN 15). Weak ordering between ack and terminal words: the sim must be able to deliver them in either order, as the model does. | no |
| 4 | **The device kernels.** `device.cuh`: `LaneRequest` in the post kernel; the wait kernel with `go_count`, the `RowResult` validate (generation, tag, expert), the terminal, the `fatal`/`Header.shutdown` poll inside the loop (D4), and 64-bit `ld.acquire.sys` / `st.release.sys`; new `exl3_ram_miss_ack_kernel`. `ops.py`: `Exl3RamMissDevice.post/wait/ack`, `lane_ctx`, `go_count`; append `epoch` and `pending_epoch` to `STATE_WORDS`. | CPU: argument validation in `test_exl3_ram_miss_device_args`; the state-word agreement test. GPU (manual, under the lock, after crypto-c9 schedules): a wait kernel timeout leaves `go_count == 0` and a poisoned slab is not read; an acknowledgement is emitted only for copied lanes; the `ld.global.nc` experiment (6.6), independent of this step. | **`STATE_WORDS` is a triple edit**: the device-state enum in `device.cuh`, `STATE_WORDS` in `ops.py`, and the hard-coded dict in the device-args test, all together or the existing test fails. **`go_count` must be an int32 CUDA tensor of shape `[1]`** (`_validate_plan`), not a scalar, not int64. Constexprs as in step 1. | yes |
| 5 | **The backend and the arming rule.** `srt_ram_miss.py`: `Exl3RamMissRowBackend` overrides `post` (today only `translate` is overridden and `post` is inherited from `PinnedTierRowBackend`) to pass `go_count` to `copy_expert_row_segments_gpu` and to launch the acknowledgement kernel after it; `Exl3RamMissService.ensure_started` / `attach` allocate the block and hand it to the host and the device; lease mode arms every record with `count > 0` (15). The new environment switch is read here. | Existing service tests green with the switch off; with it on, the GPU graph parity test (`test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`): byte-exact output versus off, an injected timeout, and the arming cost measurement (OPEN 11) reported, not assumed. | The switch defaults off. `advise` and the arming rule must agree between the device (`armed = need_count > 0 || advise != 0`) and the service's `touch_request` path, or a record is waited on that the service treats as touch-only. | yes |
| 6 | **Shutdown and quarantine. (Production reachability: only the exit-hook quarantine, only on normal exit; `shutdown()` has no production caller. See the finding at the top of section 14.)** `srt_ram_miss.py` `Exl3RamMissService.shutdown` runs S0-S5 (14.3); `Header.shutdown` is set by the service; the device-wide `torch.cuda.synchronize` in a helper thread with a deadline; `_stop_live` in `ops.py` quarantines unconditionally (DECIDE 2). **`expert_stream.py`: `ExpertPinnedHostCache._release_slabs.detach()`**, and `expert_host_tier.py`: the unregister-skipping path; the quarantine takes an unreleased extra reference (`Py_IncRef`), not a module-level list. | Section 18.2 item 9(c) (a fake CUDA error and a fake sync timeout select quarantine; `release_host_slabs` patched before the cache is built, the fake sync asserted to have been called, no unregister, the slab alive by weak reference); an atexit-order test that pins OPEN 13's answer; the model's shutdown mutant as a regression. | **The two files the plan did not list** (`expert_stream.py`, `expert_host_tier.py`): the finalizer that unregisters slabs at exit lives in the first. Without `detach()` the quarantine is silently undone. | no (helper-thread and fake-error paths are CPU) |
| 7 | **The host-lease API for Task 8.** `host.cpp` and the export table: `acquire_host_lease`, `lease_on_ready`, `release_host_lease` (17.1, R1-R8), never bumping `kVersion`, releasable by an executor-stream host callback (R1a, OPEN 16) and not only by the scheduler thread's poll, a stale or repeated `LeaseRef` counted as an error; the pause counts graph-lane leases only; **a host lease is released by an executor-stream host callback, not the scheduler thread's poll (R1a, OPEN 16)**. | R1-R8 tests; the model's Task 8 mutants replayed against the real API through the sim. | R8 (`kVersion`), R2 (the pause split), R3 (`lease_on_ready`), and A3 for whoever calls it. | no |
| 8 | **Enable and gate.** Flip nothing by default. Record the cost of the added fences and of arming every `count > 0` record (OPEN 5, OPEN 11); run the `ld.global.nc` experiment if not already done (OPEN 6); state the result of the wrap tests against the real code (OPEN 1). | The plan's Task 5 gate: lease-pressure and fault tests prove no reuse before consumption; concurrent graph execution stays guarded (OPEN 9). | Do not report skipped hardware tests as passed. | yes |

### 20.1a What "GPU-free" means for step 4

`nvcc` 13.4 is installed on divix01 at `/usr/local/cuda/bin/nvcc` (not on `PATH`); a
standalone CUDA program compiled for `sm_120` there in 4.5 s of CPU time with no GPU touched
(`nc_visibility.cu`). So the kernels of step 4 can be **written and compile-checked** without
the GPU, and the argument-validation and state-word agreement tests are CPU-only. Whether the
project's JIT path (`load_jit`) finds that `nvcc` from a non-interactive environment is not
checked; do not assume it. What only a GPU can verify is that a timed-out wait leaves
`go_count == 0`, that acknowledgements are emitted only for copied lanes, and that the 64-bit
acquire/release words behave as designed. **Until step 4's GPU test has run, the model's
device-side steps are transcribed but unexecuted**, which is a second reason (beside the
independent review) that lease mode defaults off.

### 20.2 What each step must not do

- Step 2 must not grant a lease or change a request's outcome; if any existing test changes
  its result, the step is wrong.
- Step 3 must not touch `device.cuh` or the request page's layout; the page ABI is untouched
  by the whole task (DECIDE 1).
- Step 4 must not change `copy_expert_row_segments_gpu`'s kernel; the copy kernel already
  takes the active count as a tensor, and a zero `go_count` is the whole fail-closed
  mechanism.
- Step 5 must not enable lease mode by default, and must not remove the un-armed touch-only
  path for `count == 0` records.
- Step 6 must not free anything on any path where completion was not established.

### 20.2a Deviations from this order, recorded as they happen

- **Step 1 deviated.** The order above put the block constants and allocator in `ops.py` and
  `host.cpp`. They were built instead as a **new module** (`ops/moe/exl3_lease_block.py`), the
  constants in `exl3_ram_miss.cuh`, and new tests, because `host.cpp`, `ops.py` and
  `srt/layers/moe/exl3_ram_miss.py` were carrying another change's uncommitted diff (the Task 4
  packing-worker pool) and editing them in step 1 would have collided with it. The host-side
  constants therefore move to step 2, and the agreement test for them (see below) is a guarded
  skip until then. Nothing else in the order changes.
- **The host-source agreement test is guarded so it cannot silently match nothing.** It skips
  with a reason while `host.cpp` has no lease code, and fails hard if `host.cpp` mentions lease
  concepts (`LaneRequest`, `SlotGen`, `Outstanding`, `retire_leases`, `leases`...) without
  defining the `kLease*` constants, or defines only some of them, or defines them with other
  values. The guard has its own test that each of those failure modes is refused, because a
  check that cannot fail is the failure this project keeps finding.
- **Step 6 was started early** for its part that touches no contested file:
  `ExpertPinnedHostCache.quarantine()` and `quarantine_host_slabs()`. The `shutdown` wiring that
  calls it is still to do.

- **DEFERRED AND WHY: retirement between the reader's batches** (7.5 call site 2, the give-up lambda of
  `RowReader::read`) is **not implemented**, on purpose, pending `t4-packworker`'s refactor of the code around that
  lambda: two sessions editing it from opposite directions would produce a merge nobody can review. Until it lands,
  `retire_leases()` runs only at the top of `pump_demand`, so during one long read the acknowledgements of earlier
  requests wait until it returns (a hold-time cost; nothing waits on an acknowledgement). It is why box 5 ("keeps
  reaping") is not ticked. The test it needs: a delayed read (`inject(delay_s)`), a withheld-then-delivered
  acknowledgement, and the witness that `leases_acked` moves **during** the read (R3 in 18.2a). If the refactor keeps
  the lambda recognisably, the change is one call; if it dissolves it, where retirement belongs in the new shape is to
  be decided before anyone writes it.
- **A test-first suite's first run failed for the wrong reason, which is normal:** the first run of the 3c thread
  tests failed because a raw `sim_post` writes no lane request, so the service saw an overrun; the test was wrong,
  not the service.

- **The device kernels (7.3, 7.4) landed with three post-kernel additions the spec did not spell out, and they
  are decisions, not readings of the text.** (1) The post kernel had none of its [P] half: it wrote no `LaneRequest`,
  kept no `kEpoch`/`kPendingEpoch` (11.3), so the wait kernel had no `G56` to name. Both are now written, in lease
  mode only (`lease_address != 0`; zero leaves the post byte-for-byte as before). (2) **Arming — and this one is NOT a decision; correcting the
  sentence above for this item.** 7.1 step 3 indeed says nothing about `count > 0` with every planned expert already
  in RAM, which today's post leaves unarmed, and the service reads a request's lanes only when the record is armed, so
  such a request would be copied from slots nobody leased. But **section 15 settles it and mandates exactly what was
  implemented**: "The all-hit handshake ... kept, and now mandatory whenever `count > 0`, with or without advise: the
  handshake is where the lease is granted", and "lease mode arms every record with `count > 0`, exactly as Option F
  already does when advise is on", with `count == 0` staying unarmed and touch-only. So the post arming every request
  with planned lanes is the registered rule, not an implementer's choice, and **it is not open to being optimised away
  here**: section 15 records that removing the handshake needs an equivalent protection (a device-side lease by mapped
  atomic) whose Dekker-style argument in both directions and model check are an explicit Task 5 non-goal. Cost: a wait
  per streamed layer that today would have been skipped, which is section 15's already-registered **OPEN 11**,
  **measured 2026-09-21, and re-measured at about 17 us per all-hit layer, ~0.68 ms per step at 40 layers (re-measured 2026-09-21 on an exclusively held card; the earlier ~8 us / ~0.32 ms figures are superseded, see `open11/results.md`)**. The implementation matching the spec is a stronger result than a defensible choice would have been. (3) **Where the wait kernel refuses
  with no generation to name** (sticky at entry, or lanes with nothing armed) it publishes no `Terminal`: nothing was
  posted, so nothing was leased. Terminal `reason` values (`timeout 1, aborted 2, failed 3, identity 4, count 5`) are
  mine; the service does not interpret them. The wait also refuses a `host_slot` at or above the row's capacity
  (read from the immutable `RowTable`), which bounds the ack kernel's `SlotGen` read; 7.3's "range" comment did not
  say against what. An abort on the fatal word or `Header.shutdown` raises no fatal word of its own. `ram_miss` counts
  every planned lane of a refused request (OPEN 7 remains open). The legacy `exl3_ram_miss_wait` is kept for lease
  mode off; `exl3_ram_miss_lease_wait` is a separate kernel, so nothing of today's path moved.

### 20.2b Step 3's CPU simulated device: what the scaffolding must provide

The simulated device of step 3 extends `exl3_ram_miss_sim_post` / `_sim_wait`. To be faithful to
the model rather than decorative it must:

- write a `LaneRequest` with the request, and take the lease block, so a test can post lanes;
- run the wait kernel's decisions (row-result validation, `go_count`, terminal) and a copy that
  reads the leased slot's bytes into a test buffer and compares them with the expected expert,
  so a test observes the shipped-bug shape (wrong bytes accepted), not just a counter;
- run the acknowledgement kernel, including the `SlotGen` re-check and its `VIOLATED` outcome;
- hold the device's stores to `LaneAck` and `Terminal` in an outbox and let the test deliver them
  in any order across words (same word in order), as the model does;
- be able to withhold acknowledgements entirely (the "never retires" and lease-pressure tests) and
  to time out (`go_count = 0`, terminal published).

**Model traces are replayed at the scenario level, not step for step.** The model's service steps
(reserve, load, grant, status, done) are finer than the real `serve()`, which is one call per
request; so a counterexample becomes a scenario whose *outcome* is asserted (for example: an
advisory pressure applied while a lease is held leaves the slot's bytes and generation unchanged;
a request whose device gave up leaves no lease after the terminal is delivered; a deferred demand
is served once the acknowledgement is delivered and its stage ring holds one record, not one per
poll), not a sequence of internal calls. Say which model trace each test corresponds to in its
docstring.

### 20.2c Notes for whoever lands step 2, from reading `host.cpp` at `48da151fb0`

Written while the details were fresh, by reading (not by running anything), so treat each as a
claim to check against the file as it stands when step 2 starts. Only things that bear on the
lease design are listed.

1. **`take_slot_locked` mutates as it chooses, so a deferral cannot be built on it alone.** The
   reservation loop in `serve()` calls it once per missing expert, and each call unmaps its victim
   and marks the slot free before the next. The failure path even says so: "each slot taken may
   have evicted a row, and that eviction stays". A *deferral* that ran the loop and then backed
   out would evict a row on every retry, which is data loss driven by a poll. The design's
   "tri-state {slot, deferred, none}" (section 8) is not enough for an all-or-nothing request:
   step 2 needs a **dry-run** that counts free, evictable and lease-blocked slots for the whole
   request first, and only then either commits the takes or defers with no state change. This
   corrects the sketch in section 8.
2. **`pump_demand`'s structure fights an early return.** After `begin_stage` it reads the record,
   calls `handle_demand`, then unconditionally does `_mm_sfence()`, `store_release(demand_done)`
   and `next_demand_ = skip_zero(next_demand_ + 1u)`. A deferral must return **between**
   `read_record` and `handle_demand` without running that tail, and must clear `cur_` without
   pushing (7.1). It must also not reach `busy_since_` or `kBusySeq`: `handle_demand` sets them at
   entry, and the watchdog's "stuck" rule (`max(30 s, 3 x timeout)`) would count a long deferral
   as a hung read and abort the process.
3. **Lease at publication (DECIDE 3) is safe only while the service is the only actor that can
   evict.** In `serve()` the residency check and reservation happen in one `mutex_` section, the
   read happens outside it, and the publication happens in a third. Today nothing else evicts in
   between: the hit slots are protected only by this request's own `wanted` list, and other
   actors (Python's `assign`, `touch`) run only while the thread is paused. **Task 8 breaks that**:
   a promotion admission on the scheduler thread will call `take_slot_locked` on the same row
   while the service is mid-read. A READY hit slot then has no protection from reservation to
   grant. So when step 7 lands, hit and loaded lanes need a reservation-time hold (a per-request
   reserved count checked by the eviction predicate, or the lease itself taken at reservation),
   and DECIDE 3 has to be reopened. Step 2's predicate should be written so that this is a
   one-line addition, not a rewrite.
4. **Eager `assign()` publishes before the bytes exist.** It marks the slot `kReady` and publishes
   the map entry immediately; Python fills the bytes afterwards, with the device idle and the
   thread paused. The `SlotGen` bump therefore belongs at `assign` time (before Python writes),
   not at "slot becomes ready", or the detector of 6.5 would compare against a generation taken
   after the bytes changed. Serving reads bump at the move to `kLoading`.
5. **`Request` carries `need` and `protect` only.** The lane list arrives in a separate
   `LaneRequest` with its own seqlock, so `serve()`'s `wanted` must be built from
   protect, need **and** the lane experts (7.1 step 4), and `Request` grows a `lanes` field.
6. **`counters_` is index-aligned with Python `COUNTERS`.** New lease counters go at the end of
   `enum Counter` and the same position in `ops.py`; `exl3_ram_miss_open` gains a block argument
   and the many test call sites are spared by the constructor allocating one when none is given
   (step 2 as written above).

### 20.3 Dependencies

`1 -> 2 -> 3`, `3 -> 4`, `4 -> 5`, `2 -> 6`, `3 -> 7`, `5 + 6 -> 8`. Steps 1-3, 6 and 7 are
GPU-free, so they can run while the GPU is scheduled elsewhere; 4, 5 and 8 need the
scheduled GPU time.

### 20.2d Step 6 (shutdown) landed as `fa22865ac6`: reachability, boundary, mutation ledger

**Reachability, first.** Nothing here is wired into `Scheduler.release_host_resources()`. What reaches production
is the exit-hook quarantine (`atexit`, registered inside `ensure_started`) on a normal interpreter exit,
subject to OPEN 18's residue. The barrier in the tests is faked (`_cuda_active`, `_synchronize`): the tests show
the wiring and the order, not that a real device barrier orders GPU work. No shutdown box is ticked by this step.

**Boundary in `exl3_ram_miss_host.cpp`.** `pump_demand` is touched (one added line after the `retire_leases()`
call), `pump_advice` gets one added line, and `close_admission` plus its export are new. `handle_demand`, `serve`,
`pack_one` and `RowReader` are not touched.

**Mutation ledger.** Driver: a mutant counts as KILLED only if `passed + failed == collected` and at least one
named test failed; baseline asserted clean first (20 collected, 20 passed). Files: `test_exl3_ram_miss_shutdown.py`
and `test_exl3_ram_miss_lease_service.py`.

| Mutant | Result | Killed by |
|---|---|---|
| S1 barrier skipped | KILLED 3 | order test, cuda-error test, barrier-timeout test |
| S2 barrier before admission closes | KILLED 3 | same three |
| S4 CUDA error at the barrier treated as success | KILLED 1 | cuda-error test |
| S5 non-returning barrier treated as success | KILLED 1 | barrier-timeout test |
| S6 exit path attempts the barrier | KILLED 1 | exit-quarantine test |
| S7 exit path frees | KILLED 1 | exit-quarantine test |
| S8 quarantine keeps no host tensors | KILLED 1 | cuda-error test |
| S9 exit-time finalizer left attached | KILLED 1 | cuda-error test |
| S10 uncertain shutdown frees anyway | KILLED 3 | cuda-error, barrier-timeout, exit-quarantine |
| S11 barrier joined without the deadline | KILLED 1 | barrier-timeout test (47 s wall: the mutant waits 30 s by construction) |
| S12 `close_admission` does not set the header word | KILLED 2 | order test, header-word test |
| S13 request served after admission closes | KILLED 1 | header-word test |
| S14 retirement stops when admission closes | KILLED 1 | header-word test |

13 of 13 killed; no survivors. Not established: S3 was not written (the numbering skips it), and S6 and S7 are
killed by the same single test, so it alone distinguishes the two.

**S8 was first INVALID, and that stays in the record.** As first written S8 removed the line and left an empty
`if` body: an `IndentationError`, so the run produced 0 passed and 0 failed of 20 collected. A driver that
looked for a `FAILED` line would have read that as *survived*, or, with an equally broken baseline, as *killed*.
The driver reported `INVALID (not every collected test ran)`. The mutant was rewritten to replace the line with
`pass`, and it then died to a named test. Evidence that the collected-equals-executed rule works on a case
that was not in the three it was derived from.

### 20.2e R2 (an advisory during a deferral): tests and mutation ledger

Two tests in `test_exl3_ram_miss_lease_thread.py` replace the `advise_done`-only check that lived in the
third test of that file. That check passed when the advisory was dropped as stale (a stale skip also advances
`advise_done`). Shown, not argued: the old file run against mutants A1 and A4 below gives SURVIVED both times
(3 passed of 3 collected, 0 failed); the new file kills both.

Path witness in both: a demand for two experts is deferred (`deferred == 1`, `demand_done` not advanced, so
`demand_pending()` is true), the advisory is posted **after** that, with `after` equal to the demand's seq (not stale
by the `after` rule), and the advisory is observed to have been *served*: `advisories` +1 and
`advisories_skipped` unchanged. Property, row 1 test: `advisory_rows == 0`, `rows_read` unchanged, `deferred`
still 1, the deferred row's slot table unchanged, the demand still not done. Property, own-row test: no eviction,
slot table and `SlotGen` words unchanged, `deferred` still 1.

| Mutant | Result | Killed by |
|---|---|---|
| A1 advisory dropped as stale while a demand is pending | KILLED 2 of 5 | both R2 tests |
| A2 advisory reads while a demand is deferred (`demand_pending()` removed from the give-up test) | KILLED 1 of 5 | the other-row test |
| A3 an advisory that cannot be served counts as a deferral | KILLED 1 of 5 | the own-row test |
| A4 an advisory ignores leases when choosing a victim (census guard and lease skip both bypassed for advisories) | KILLED 1 of 5 | the own-row test |
| A5 an advisory is not consumed at all while a demand is pending | KILLED 2 of 5 | both R2 tests (fail on "the advisory was consumed") |

Baseline 5 of 5 collected and passed before each run; every kill has passed + failed == collected. The A5 kill is
a wait that times out on a positive assertion, not an absence read as a pass.

Not established: an advisory for a row whose tier is only *partly* leased; the advisory's interaction with a
deferral that a retirement is about to end (no test overlaps the two in time).

### 20.2f R4 (item 2 of 18.2 as written): test and mutation ledger

`test_a_served_requests_leased_slot_keeps_its_bytes_under_a_newer_demand_and_an_advisory` in
`test_exl3_ram_miss_lease_service.py`, capacity-2 row. A request leases expert 2 and its acknowledgement is
withheld. A newer unleased neighbour (expert 5) is made resident **after** it, so the leased slot is the
least recently used one: exactly the victim an unguarded eviction chooses. A sentinel (0xAB) is written into the
leased slot's slab bytes after publication. A newer demand (expert 4) is then served and retires normally, and an
advisory (expert 1) is served and reads.

Path witness (asserted before the property): both newer requests reserved and read (`evictions` +2, `rows_read`
+2, `advisory_rows` +1, `deferred` unchanged), and the advisory's expert landed in the unleased slot. Property:
the sentinel is intact in every streamed tensor of the leased slot; its `slot_info` entry (state, expert, lease
count, generation) and its `SlotGen` word are unchanged. Control: after the late acknowledgement retires the lease,
a further demand takes that same slot and the sentinel is gone, so the lease was the only thing protecting it.

The first version failed at the witness (`evictions` +1, `advisory_rows` 0): the newer demand's own lane request
had leased the second slot, so the advisory gave up with nothing to take. Fixed by having the newer request
acknowledge normally; recorded because a version without the witness would have passed the property trivially.

| Mutant | Result | Killed by |
|---|---|---|
| B1 the eviction ignores leases | KILLED 1 of 16 | this test (the other 15 service tests do not catch it) |
| B2 a demand ignores leases, an advisory respects them | KILLED 1 of 16 | this test |
| B3 an advisory ignores leases, a demand respects them | KILLED 1 of 16 | this test |
| B4 the slot generation is not bumped on reassignment | KILLED 1 of 16 | `test_each_lane_gets_a_row_result...`, **not** this test: R4's generation check is meaningful only once a slot is rewritten, which a guarded service never does |

Baseline 16 of 16 collected and passed before the run. Not established: the sentinel check covers the CPU
service's slab writes, not the device kernels' reads (item 5 and 12 stay GPU-only); a slot leased by two lanes was
not the subject here.

### 20.2g R1 (a long deferral does not trip the watchdog): test and mutation ledger

`test_a_long_deferral_does_not_trip_the_watchdog_and_the_demand_is_served_after_the_lease_retires` in
`test_exl3_ram_miss_lease_thread.py`. A subprocess (the watchdog's abort kills the interpreter), thread mode with
the real watchdog and `fatal_wait_s = 0.3`. Row 0 is full of host leases, a two-expert demand is deferred, and the
script samples every 10 ms for 5 x `fatal_wait` on its own clock: `busy_since_ns()`, the `busy_seq` word and the
`fatal` word, and that `demand_done` has not moved and `deferred` is still 1. It then retires the leases and
requires the demand to be served and the process alive.

Path witness, asserted by the parent from what the script printed: the deferral was observed for at least
5 x `fatal_wait` with at least 20 samples (a shorter or unobserved deferral proves nothing about an abort).
Property: `busy_since_ns`, `busy_seq` and `fatal` were 0 at every sample, return code 0, `SERVED`, `alive`, and
no `ERROR exl3` on stderr. The hung-read abort itself is still tested by the existing thread-file tests.

| Mutant | Result | How it died |
|---|---|---|
| C1 the deferral marks itself busy at its first observation | KILLED 1 of 6 | the process aborted (-6): "a request stayed in service for 0.3 s" |
| C2 the same, on every poll | KILLED 1 of 6 | same abort. Not distinct from C1: a polled deferral returns before the marking line once observed, so a mutant placed there runs once |
| C3 the deferral publishes the busy word | KILLED 1 of 6 | the property: `busy_seq` was 1 |
| C4 the deferral raises the fatal word | KILLED 3 of 6 | R1 and both R2 tests (an advisory is skipped while the fatal word is up) |

Baseline 6 of 6 collected and passed; every run executed all 6. Not established: the test does not vary
`fatal_wait` against the deferral (5x is a fixed ratio, on a 0.3 s watchdog); a deferral of a request-slot reuse
(`kDeferredReuse`) rather than a victim shortage was not run through the watchdog.

### 20.2h Scope and caveats of the step 6, R5, R2, R4 and R1 numbers

**Archive scope.** The step 6 count (348 collected = 348 passed, 20 files) and the R5 table (13/13, 13/13, 8/8,
15/15, 14/14, 3/3, 2/2, 5/5) were taken on exports of `python test` (the 348 run added `scripts/dsv41` after the first
red). That scope is too narrow for the corpus: 16 test files reach `scripts/` or `analysis/` by relative path, and
the one red in the first run (`test_graph_steps_are_traced_and_read_back_by_tier_sim`) is one of them, not an
undeclared `PYTHONPATH` entry as first reported. None of the lease or shutdown files reaches outside `python/` or
`test/`. The numbers are correct for that scope and are not whole-tree numbers.

**Re-taken on the full scope, on a named commit.** A fresh `git archive b022c8ee6b python test scripts analysis`
(divix01, `CUDA_VISIBLE_DEVICES=9`, `taskset -c 0-63`, every run executing all collected tests):
- 20 files, 352 collected = 352 passed; each lease file and `test_exl3_ram_miss_shutdown.py` run as CI runs it
  (`python3 <file> -f`): 13, 13, 8, 16, 14, 6, 2, 5 passed, equal to pytest's counts.
- R2 mutants A1-A5, R1 mutants C1-C4, R4 mutants B1-B4: **the same verdicts and the same killing tests as in
  20.2e-20.2g.** (Those sections' first numbers came from an export whose host files matched the landed ones; this
  run supersedes them as the citable one.)

**"13 of 13" (20.2d) is fewer independent pieces of evidence than it reads.** The thirteen shutdown mutants were
killed by **five distinct tests** (order, CUDA-error, barrier-timeout, exit-quarantine, header-word) out of 20
collected. S6 and S7 die to the same single test; S4, S8 and S9 to the same single test; S13 and S14 to the same
single test. The evidence is the set of tests that fired, not the tally.

**Box 3 stays open.** R4 supersedes item 2 of 18.2 (a byte-level immutability check with service-granted leases,
uniquely detecting B1-B3 in its file). The box's first clause is "inject delayed GPU consumption", and there is no
GPU consumer: the delay is injected into `LeaseSim`, a stand-in written from the same specification. R4 does not
change that.

**Class order.** Eight of the nine test files I added or changed (lease_block, device_args, leases, lease_service, lease_defer, lease_thread, lease_wrap and shutdown) have no test classes, so the class-order divergence between
`unittest` and pytest cannot apply to them (closed by construction, not swept). The exception is `TestQuarantine`
in `test_expert_host_tier.py`, a 12-class file that is not mine. Not established for any of them: isolation between
functions (a reverse-order run was not done) and that no production path writes an environment variable the tests
read (established by reading, not demonstrated).

**R1's subprocess.** `subprocess.run(..., capture_output=True, timeout=60)` reads both pipes to the end, so a full
pipe cannot block the child, and the timeout bounds a hang. Neither was tested by making the child chatty.

### 20.2i PROPOSAL (not implemented): placement of the orderly shutdown call in `Scheduler.release_host_resources()`

Read first: `.claude/skills/large-class-style/SKILL.md`. Its frozen list is `model_runner.py` only; `scheduler.py` is
covered for `__init__` (section 2), and this change is not in `__init__`. Section 1.3's test is still the right one
for the shape: the added statements must construct, wire, delegate or order, and never compute.

**Where.** `Scheduler.release_host_resources()` (`scheduler.py:1865`, called only under `if scheduler.gracefully_exit`
from `run_scheduler_process`'s `finally`, `scheduler.py:6099`). There is already a block for the same kind of
collaborator: `expert_hot_cache_manager.stop_doorbell()` inside a `try/except Exception` that logs and goes on.
Proposed: a second block of the same form **immediately after it and before** `hisparse_coordinator.destroy()` and
`tree_cache.release_host_resources()`, delegating to one function in `exl3_ram_miss.py`:
`shutdown_exl3_ram_miss_service()`, which does nothing unless `Exl3RamMissService._instance` exists (it must not
construct the singleton at shutdown) and otherwise calls `shutdown()`. All logic stays in that module; the
scheduler gets one delegating `try` block, no computation.

**Why that order.**
1. After `stop_doorbell()`: the doorbell copier is joined before the RAM-miss service closes admission and
   synchronizes the device, so no host-side producer is left running against slabs about to be freed.
2. Before the other host releases and `destroy_global_*`: the barrier needs a working CUDA context, and the docstring
   there already says the graceful path exists because the exception path may have a wedged GPU.
3. Before `abort_distributed_environment()` (the last step of the `finally`): nothing in `shutdown()` uses a communicator.
4. The exit hook stays registered and becomes a no-op after an orderly shutdown (`_shut_down` is set), so the two
   paths cannot both free.

**Import.** `exl3_ram_miss.py` pulls in the MoE stack; a top-level import in `scheduler.py` would load it for every
run. Proposed: import inside the `try` block. Its cost was not measured; a run without EXL3 would pay the import at
shutdown only, which is the trade being made.

**What would show it works, and what does not exist.** A unit test with a stub scheduler object (the four test
files that mention `release_host_resources` show the pattern) that records call order: the RAM-miss shutdown is
called once, after the doorbell stop and before the tree-cache release, and an exception in it does not stop the
remaining releases. Mutations to run: the block is omitted; it is placed before the doorbell stop; it is placed
after the tree-cache release; the `except` is removed. What no CPU test can show: that the real device barrier
orders GPU work (the tests fake `_synchronize`), and that every teardown path reaches `gracefully_exit` (OPEN 18's
residue: a SIGTERM or an engine shutdown that never sets it still ends in the quarantine at exit, not the orderly
path). A GPU run needs crypto-c9's scheduling.

**Not decided here.** Whether `scheduler.py` review wants the block behind a config gate; the function is a no-op
without a live service, so this proposal has none.

**20.2i addendum: the test drives the real method, and the ordering is what it asserts.** `test_expert_doorbell_copier.py`
(`_scheduler_stub`, `_release_scheduler_host_resources`, lines ~1500-1553) already calls the *unbound real*
`Scheduler.release_host_resources(stub)` with the module's capturers patched, and has one test that the doorbell is
stopped through the model runner and one that a failing `stop_doorbell` still releases the rest. The wiring test
follows that pattern and no other: it must run the real method, not a copy of its order. Two tests. (1) A recorder
stub: the doorbell stop, the RAM-miss shutdown and `tree_cache.release_host_resources` append to one list; assert
the list equals `[doorbell, ram_miss, tree_cache]` (the relative order to `stop_doorbell` being the point). (2) The
same with a live `Exl3RamMissService` whose barrier is the faked `_synchronize` of `test_exl3_ram_miss_shutdown.py`:
the copier reports `running == 0` at the moment the service closes admission, and afterwards the tiers are freed
(barrier clean) or quarantined (barrier errored); a failing RAM-miss shutdown must still reach the tree-cache
release. Mutations, each to be run and its killing test read: block omitted; before the doorbell stop; after the
tree-cache release; `except` removed; the function constructs the singleton when none exists (a run without EXL3
would then create a service at shutdown). The fake barrier is still a fake: the second test shows the wiring, not
that the real barrier orders GPU work.

**20.2i revision after the independent review (`SHUTDOWN_WIRING_REVIEW.md`): the placement above is superseded.**
The block goes **last** in `Scheduler.release_host_resources()`, after hisparse, the tree cache, the decode-offload
manager, both capturers and `rank_consensus_checker` (review F3, decided). Reason 2 above ("the barrier needs a
working CUDA context, so run first") was not a reason: nothing before it frees a context. Cost of a failing barrier:
up to `RAM_MISS_TIMEOUT + 5 s`, plus an unbounded `host.stop()` (a hung service thread is ended by the watchdog's
abort after `fatal_wait`, as at the exit hook today); `abort_distributed_environment()` in the caller waits behind it.
The block reads `sys.modules` rather than importing (F4). The claim "the worst case is what the exit hook does today"
was false until `shutdown()` treated a failed `stop()` as uncertain (F1), chose the barrier's device on the calling
thread (F2) and recorded completion separately from start (F5). The code and its ledger are in a patch awaiting the
reviewer's second pass; nothing is landed. The mutation ledger goes with the code (20.2j).

### 20.2j The scheduler wiring: what landed, after two independent review passes (`SHUTDOWN_WIRING_REVIEW.md`)

The change (proposal and revision in 20.2i): one block, **last** in `Scheduler.release_host_resources()`, that reads
`sys.modules` and, when `exl3_ram_miss` is loaded, calls `shutdown_exl3_ram_miss_service()` inside a `try/except` that
logs. The function is a no-op unless a service exists. Fixes in `shutdown()`: a failed or interrupted `stop()` makes it
uncertain (F1); the barrier's devices are chosen on the calling thread, each with an explicit index and de-duplicated,
from the service's device buffers and the tiers (F2, S2); completion is tracked apart from start (F5); an interrupt in
the barrier is re-raised after the quarantine like one in `stop()` (S1). The block must stay last (a comment says so).

**The earlier claim "the worst case is what the exit hook does today" was false until F1, F2 and F5 were fixed.**

**Accepted risk, ruled on by the reviewer (read from the C++, not run).** (a) A graceful shutdown with the service thread
hung in a read ends in SIGABRT after up to `max(30 s, 3 x RAM_MISS_TIMEOUT)`, before `abort_distributed_environment()`:
`RamThread::stop` joins the service thread before it stops the watchdog, and the watchdog aborts on `busy_since != 0`
for longer than that, for a demand or an advisory. This is the exit hook's outcome reached earlier, and other ranks see a
dead peer. (b) A hang outside a request (`busy_since == 0`, for example a lock deadlock in the thread) is not covered by
the watchdog and would hang the join without bound; that already holds at the exit hook. (c) `stop()` is not bounded and
this is not tested: bounding it would put two threads in one `std::thread` join with the exit hook, which is worse.
Revisit if a stop hang is ever observed. A log line before `stop()` names the abort deadline.

**Tests** (`test_exl3_ram_miss_shutdown.py`, 5 to 22). The wiring tests drive the real unbound
`Scheduler.release_host_resources` on a stub recording `doorbell, hisparse, tree_cache, decode_offload, both capturers,
rank_consensus`, so the block's position against every neighbour is asserted. The barrier is a fake; F2 and S2 are shown
against a recorder of the `device` argument, which is the only instrument available on a one-GPU box and not a weaker
substitute for a hardware check.

**Mutation ledger.** Baseline 22 of 22 collected and passed; every run executed all 22; the failing assertion line was
read for every kill (`--tb=line`) and none was an error at a call site, a hang or a fixture. Killers: 25 kills rest on
**17 distinct tests**; twelve mutants each depend on a single test (W4, W4c, W5, W7a, F1b, F4, F5a, F5b, S1, S2a, S2b, S2c).

| Mutant | Result | Fired on |
|---|---|---|
| W1 block omitted | KILLED 6 | the `==` on the recorded order |
| W2 / W3a / W3b / W3c block before the doorbell / before hisparse / before the tree cache / before the last cheap release | KILLED 5 each | the same `==` |
| W4 try/except removed | KILLED 1 | "the release raised RuntimeError(...)" (the test's own assertion) |
| W4b `except Exception: return` | **SURVIVED, equivalent, only because the block is last** | (no failure: nothing follows the block; if it stops being last the mutant is live again) |
| W4c nested under the hot-cache-manager check | KILLED 1 | `'close_admission' in order` |
| W5 the function constructs the singleton | KILLED 1 | the singleton is not None |
| W6 the function passes `at_exit=True` | KILLED 4 | free expected, quarantine seen |
| W7a exit hook ignores completion | KILLED 1 | the recorded order after the hook |
| W7b completion never recorded | KILLED 4 | `_completed` asserts |
| W7c `shutdown()` has no idempotence guard | KILLED 2 | the exit-hook order and the second-shutdown order |
| F1a a `stop()` failure does not make it uncertain | KILLED 2 | free seen where quarantine expected |
| F1b an interrupt during `stop()` is swallowed | KILLED 1 | DID NOT RAISE KeyboardInterrupt |
| F2a device-less synchronize | KILLED 4 | `[None]` against `cuda:3` and `cuda:2` |
| F2b devices looked up inside the helper thread | KILLED 2 | `cuda:0` against `cuda:2` |
| F4 the block imports unconditionally | KILLED 1 | "the scheduler imported the module" |
| F5a exit hook returns once a shutdown started | KILLED 1 | `_completed` assert |
| F5b completion recorded on entry | KILLED 1 | `assert not True` |
| S1 an interrupt in the barrier is swallowed | KILLED 1 | DID NOT RAISE KeyboardInterrupt (raised in the calling thread) |
| S1b no interrupt is ever re-raised | KILLED 2 | the same, both interrupt tests |
| S2a the service's device is ignored | KILLED 1 | `[cuda:3, cuda:2]` against `[cuda:1, cuda:3, cuda:2]` |
| S2b only the first device is synced | KILLED 1 | `[cuda:1]` against three devices |
| S2c an index-less device is passed on | KILLED 1 | `device(type='cuda')` in the recorded list |
| S2d devices are not de-duplicated | KILLED 2 | the tiers-only and the shared-device lists |

The S2 test sets `device_side.state` on `cuda:1` while the tiers claim `cuda:3` and an index-less `cuda`, so a test in
which both agreed (which would pass the mutant) is not the one that ran. The "imports nothing" test was also run alone,
in a fresh process, and passed. Not covered: that the service thread is gone after a normal `stop()` returns (the orderly
test asserts `host.threaded` is False, which reflects the Python wrapper, not the thread).

**Not established, and not credited to any test here.** The real barrier ordering GPU work, on a real second GPU;
real `cudaHostUnregister` (the tier tests use `device="cpu"`); that every teardown path sets `gracefully_exit` (OPEN 18,
and the gate sits outside the method the tests drive); the two GPU-only doorbell tests that call the real method
(`test_expert_doorbell_copier.py`, lines 1519 and 1543) skip without a GPU and were not run with the block present; a
multi-rank teardown.

### 20.2k R3: retirement between the reader's batches (7.5, call site 2): HELD, with its mutation ledger

> **Resolved 2026-09-22 in 20.2n below, in a different shape.** The held diff is superseded, not
> applied, and the "no meaningful test is possible" blocker recorded here has been removed. Read
> this subsection as the record of why the abandon-callback form was rejected; it is still the
> evidence for that.

The change: `serve()`'s abandon callback calls `retire_leases()` before it decides whether to give up. One added call in
`exl3_ram_miss_host.cpp`; `serve`, `pump_demand`, `pump_advice`, `handle_demand` and `RowReader` are otherwise untouched.
The call is idempotent (a per-lane state machine), never blocks, and cannot touch the request in service (its own
leases are granted after publication). Cost: a null check and a relaxed atomic load of `lanes_outstanding_` come before
any lock, so a turn with no lease outstanding adds one load and a branch; `mutex_` (which `serve` does not hold while
reading) is taken only when a lease is outstanding.

**Test** (`test_exl3_ram_miss_lease_thread.py`, run inline and with two pack workers): request 1's lease is
acknowledged while request 2 is asleep in the injected delay before its read; request 2 then reads and stalls before
`demand_done`. Nothing else retires in that stall, so `leases_acked` reaching 1 while `demand_done` has not advanced,
after `rows_read` moved, can only come from inside the read. The counter is read before `demand_done`, so a mutant's late
move cannot pass by racing. A second test (`test_exl3_ram_miss_lease_service.py`) repeats retirement with another lease
still outstanding, because with nothing outstanding retirement returns at once and would hide a lane released twice.

Baseline 25 of 25 collected and passed before each run; every reported run executed all collected tests.

| Mutant | Result | Killed by |
|---|---|---|
| D1 no retirement inside the read (only at the top of `pump_demand`) | KILLED 2 | the in-read test, both modes, at the assertion "the lease was retired only after the request had finished" |
| D2 the callback retires only for advisories | KILLED 2 | the same, both modes |
| D3 the callback retires only from the second batch on | KILLED 2 | the same, both modes (a one-row read calls the callback once, at batch 0) |
| D4 a released lane keeps its state | KILLED 6 of 17 (service file alone) | five earlier tests and the new idempotence test |

**D4 is not a clean number.** Run together with the thread tests, D4 makes the service thread throw `lease underflow`,
which aborts the process: the run has no summary and the driver correctly scored it INVALID. It was scored on the
service file alone (17 collected), where the failures are exceptions from `pump()`. The new idempotence test was first
written without a second outstanding lease and did NOT kill D4: the early return on `lanes_outstanding_ == 0` hid a
lane released twice. **The optimisation that makes the per-turn call cheap and the blind spot of the first test are the
same line of code.** It was rewritten with another lease held and then killed D4.

**Not established.** The worker-mode behaviour is exercised only through `pack_workers=2` on a one-row read: the callback
was not shown to be evaluated on packing turns, only that the counter moves during the read in both modes. The fault
that slows packing (`pack_delay_ns`) is not reachable from the service, so a multi-batch read with retirement between
batches was not built. The cost of the added call while a lease IS outstanding was not measured: each such evaluation takes `mutex_` and
scans up to 16 entries, on a loop that polls in worker mode. With none outstanding it is one relaxed load and a branch
(read from the code, not measured). Box 5 ("keeps reaping") is not ticked here.

**Second-reviewer findings (`RETIRE_IN_READ_REVIEW.md`), accepted.** The change has no consumer today (see the commit
body): the test proves the call exists and runs at read start, not that it runs between batches, because a one-batch read
has no between. D1-D3 are one test's worth of assurance (the same test in two modes), not three. The reviewer predicted a
flake from asserting `rows_read` after polling for the counter; the assertion was redundant (the done stall already
proves nothing else retires) and was removed. A replacement, `busy_since_ns() != 0`, was tried
and FAILED every run: `busy_since` is cleared at the end of `handle_demand`, before the stall. The test then passed 8
of 8 repeated runs and D1-D3 were killed again on the same assertion. Advisory-path retirement between rows has no test:
a delay before the read can only place an acknowledgement before the first call, which the top-of-loop retire also
covers, and the fault that slows individual rows is not reachable from the service.

**Decision: HELD, not landed (team-lead, on the second reviewer's finding).** Recorded here so it is not rediscovered.
(Superseded 2026-09-22; see 20.2n. Blocker 2 below is gone, blocker 1 stands.)
1. **The Task 6 dependency.** No consumer exists today. A demand is one batch, so the callback runs once at batch 0,
   microseconds after `pump_demand`'s own retire; the only observable effect is that `leases_acked` moves earlier. The
   rationale (a long read holding an already-acknowledged lease) is what V2's per-row transfer creates, and it is not built.
2. **The blocker for a test that would make it meaningful.** It needs a multi-batch read, and the fault that slows
   individual rows (`pack_delay_ns`, in `ReadFault`) is not reachable from the service's own reader: `RamTier::inject`
   takes only a delay before the read, a read failure, a demand count and an abandon count. Exposing a per-row delay on the
   service's reader is the first thing to do when Task 6 picks this up, and it is worth more than the one-line diff.
3. **The flake, fixed in the held package.** The first test asserted `rows_read == rows_before + 1` after a 2 ms poll saw
   the acknowledgement; the retire happens at read start and `rows_read` increments at publication, so a poll landing
   inside the read fails it. The assertion was redundant and is removed. The suggested replacement,
   `busy_since_ns() != 0`, is WRONG and must not be reintroduced: `busy_since` is cleared at the end of `handle_demand`,
   before the done stall, and the test failed 6 of 6 runs with it. Without either, 8 of 8 repeated runs pass and D1-D3 are
   killed again on the same assertion.
The patch is kept at `analysis/dsv41-drive/held/R3_retire_in_read.diff` (applies on `bcfe378f0d`).

**20.2k addendum, from the reviewer's third pass.**
1. **The idempotence test adds nothing for D4 and does not touch the new call.** D4 (a released lane keeps its state) is
   already killed by five earlier tests, and the test drives `pump()`, which exercises the existing top-of-function
   retire, not the call in the abandon callback. It closes a gap in the retire function, not in R3. (Its first version was
   vacuous for the reason given above; the second is merely redundant.)
2. **A mutant on `lanes_outstanding_` itself, run.** E1: `lanes_outstanding_` is not decremented on release, so the
   fast path is never taken and an eager pause would be refused forever. **KILLED by exactly one test**, in
   `test_exl3_ram_miss_lease_thread.py` (`test_a_pause_is_refused_promptly_while_a_graph_lane_lease_is_outstanding_and_
   granted_once_it_retires`: 1 of 8 failed), and it SURVIVES the whole service file (17), the leases file (14) and the
   defer file (8). The kill fires as the `RuntimeError` "not paused, a GPU reader still holds a graph-lane lease" raised
   by the `pause()` call the test expects to succeed: the behaviour under test, not a call-site accident, but a single
   test's worth of assurance for the counter that every fast path and every pause depends on.
3. **A correction to the constraint this change was written to.** The requirement passed on as binding ("idempotent because
   worker mode evaluates the callback on more turns") was false for demands: `admit()` evaluates the callback only inside
   `while (next_batch < batches)`, so for a demand it is never evaluated after batch 0. Idempotence is still cheap and
   correct, but it was not the constraint it was presented as.

### 20.2l The device kernels (7.3, 7.4) landed as `70847da92f`: verification ledger, and what it does not establish

**Independently verified by the lead**, on a fresh detached worktree built at the branch tip, not the implementer's
tree, so none of it depends on that environment. GPU work ran under `cc-gpu.lock` through the wrapper, on cores
0-63, production not started.

| Check | Result |
|---|---|
| `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (RTX 5090) | **31 of 31 pass**, 42 s |
| Full `exl3` CPU suite (`test/registered/unit/kernels/ -k exl3`) | **694 passed, 0 failed**, 424 deselected |
| Layout sync (`test_exl3_lease_block.py`, `test_exl3_ram_miss_device_args.py`) | **26 pass** |
| Pushed to `origin` | no; `shared` only |

**The lease-off path.** `armed` became `need_count > 0 || advise != 0 || (lease != nullptr && planned_count > 0)`,
which reduces to the old expression exactly when the lease pointer is null. A hazard worth recording because it was
handled rather than hit: `state[kPendingEpoch] = state[kEpoch]` is written **unconditionally**, including on the
lease-off path, into two state words that did not exist before, so any caller still sizing that array at the old
length would have taken an out-of-bounds write in code unrelated to leases. `STATE_WORDS` carries `epoch: 9` and
`pending_epoch: 10`, the allocation uses `len(STATE_WORDS)`, and no hardcoded size survives.

**Mutation ledger: 14 applied one at a time to `exl3_ram_miss.cuh`, all 14 KILLED.** Run with `pytest -x`, so the
named test is the *first* failure only; others may also kill a given mutant and that was not checked.

| # | Mutant | Killed by |
|---|---|---|
| M1 | ack kernel drops the `lane < n` guard (acks all 8 lanes) | `..._acknowledges_exactly_the_committed_lanes` |
| M2 | ack kernel uses `lane <= n` | same as M1 |
| M3 | ack kernel sets `consumed = true`, never VIOLATED | `..._a_slot_generation_that_moved_is_acknowledged_violated...` |
| M4 | wait kernel removes `go_count[0] = 0` at entry | `..._go_count_is_zero_on_entry_even_when_the_previous_request_committed` |
| M5 | wait kernel fails **open** (`go_count = planned_count` on the failure path) | `..._a_plan_of_more_than_eight_lanes_is_clamped_and_the_wait_refuses_it` |
| M6 | wait kernel never publishes `Terminal` | same as M5 |
| M7 | wait kernel drops `host_slot < capacity` | `..._refuses_the_whole_request_and_publishes_a_terminal[slot_past_capacity]` |
| M8 | wait kernel drops `expert == planned[i]` | same test, `[wrong_expert]` |
| M9 | ack kernel: VIOLATED raises fatal but no longer sets `keep = 0` | same as M3 |
| M10 | post kernel does not arm a request with lanes in lease mode | `..._post_writes_the_lane_request_and_arms_a_request_with_lanes_only_in_lease_mode` |
| M11 | wait kernel does not abort on the fatal word or `Header.shutdown` while polling | `..._shutdown_ends_the_wait_promptly_without_a_fatal_word` |
| M12 | ack kernel reads `SlotGen` at `4*slot`, ignoring the row's `slot_gen_base` | `..._the_slot_generation_is_read_from_the_lanes_own_row` |
| M13 | wait kernel drops the request-generation comparison on `ready` | same as M7, `[stale_generation]` |
| M14 | post kernel does not bump the epoch when the sequence wraps | `..._the_epoch_advances_when_the_sequence_wraps_and_names_the_generation` |

**Of M1-M14 every killing test is in `TestHandDriven`; the end-to-end class killed none of them** (it kills one of
the two later mutants below). Those end-to-end cases (graph capture and replay, the real C++ service thread with the
real copy kernel) are mostly evidence that the pieces compose rather than evidence that the tests would notice a
broken handshake. Mutation strength rests almost entirely on the hand-driven cases, and a reader quoting these
numbers should know that.

**M15 and M16, run afterwards to close the gap this ledger first reported as outstanding. Both KILLED.** Run in
full, without `-x`, so every failing test is listed.

| # | Mutant | Failing tests |
|---|---|---|
| M15 | ack kernel uses `n = (go_count == 0) ? 8 : go_count`: a refused request acks all lanes from the stale `lane_ctx` | `..._a_refused_request_emits_no_acknowledgement_even_with_a_stale_lane_context` at all three parameters `[identity]`, `[timeout]`, `[sticky]`, **and** `TestServiceEndToEnd::..._a_timeout_gives_the_copy_nothing_to_read_and_no_acknowledgement...` |
| M16 | `n = (go_count == 0) ? 1 : go_count`: the **minimal** case, one spurious acknowledgement | the same four tests |

M16 is the one that matters: a single spurious acknowledgement, the smallest possible violation, is caught. The
stale-lane-context test kills both on its own, on every refusal path it covers. **This is also the one place an
end-to-end test did some killing**: the timeout case asserts the acknowledgement area is all zero afterwards.

On the host-chain consequence this ledger describes: no test checks `retire_leases`'s branch order directly, but the
timeout test asserts `leases_acked == 0` and `leases_granted == leases_voided`, so it would have caught the masking
outcome too; it simply fails on the acknowledgement-area assertion first.

**Not established, listed so nobody mistakes this ledger for more than it is.**
1. ~~7.4's central "by construction" property is the one thing no mutant probed.~~ **RESOLVED by M15 and M16
   above**, which were run after this ledger first recorded the gap. The property that a skipped copy emits no
   acknowledgement is now backed by mutation evidence, in the minimal single-ack form. The reason it was worth
   closing rather than arguing: in `retire_leases` the `acknowledged` branch is tested **before** `voided`, so on a
   refused request (whose `Terminal` marks the lanes skipped) a spurious ack would win the if-chain, retire the lane
   as `kLeasesAcked`, record that the GPU consumed a source it never read, and hide the refusal from the counters.
2. **Ordering is not covered and mostly cannot be.** No mutant touches the 11.4 re-read of `ready`, and none swaps
   the `Terminal` and fatal stores (F2). Memory-ordering defects are not deterministically observable in a single
   run; they need stress or a formal argument. **The ordering guarantee rests on section 6.4's argument, not on any
   test**, and that is a stated limit rather than a gap left unfilled. Chasing it with mutants would produce passes
   for the wrong reason.
3. `keep` left unset on a refused wait, the `count > 8` reason value, and the unarmed-with-lanes branch that clears
   `pending` are untested. None is a safety property.
4. **"Passes now" is demonstrated; "unchanged from before" was not demonstrated by the implementer**, who did not run
   a parent-commit baseline. The lead's 694-test run is the evidence for that claim, not the implementer's runs.
5. Nothing here is a LeaseSim result: **no LeaseSim tests were written.** Per 20.2b and the repeated ruling in this
   document, a LeaseSim pass would not have been GPU evidence anyway.

**No plan checkbox is ticked, and none should be on this evidence.** Task 5's remaining boxes name service-side
behaviours (the shutdown drain, the terminal cancellation handshake, admission pressure under a delayed GPU
consumer) that these two kernels now make *possible* and do not themselves demonstrate.

### 20.2m Step 5 (the backend and the switch) landed: what was wired, and what was and was not verified

Wired (`exl3_ram_miss.py`, `environ.py`; the file 20.1 row 5 calls `srt_ram_miss.py` is `srt/layers/moe/exl3_ram_miss.py`):

- `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES = EnvBool(False)`, under the option C / option F block of `Envs`, read **once**
  in `Exl3RamMissService.ensure_started` into `service.lease_mode`. That one field enables the host
  (`enable_lease_mode()`, before `start_thread`) and, in `attach`, hands the device the host's own `lease_block` and
  `lease_layout`. There is no second read and no second notion of arming: the device's `armed` is written into the
  record and the service reads it back, so the hazard row 5 names (a record the device waits on that the service
  treats as touch-only) cannot arise from this wiring.
- `Exl3RamMissRowBackend.post`: without a lease block on the device it is the inherited `post`; with one it is
  translate, copy with `device_side.go_count` in place of `plan.count`, then `device_side.ack(keep)`, in that stream.
- The quarantine also keeps the device's `go_count` and `lane_ctx` alive.
- Nothing allocates a block in the service: `Exl3RamMissHost` already allocates one (step 1); row 5's "allocate the
  block" is satisfied by passing that one on.

Verification, by kind (none of it ticks a plan box):

- **GPU (RTX 5090, divix01, under `gpu-run.sh`):** `test_exl3_ram_miss_graph_gpu.py`, five tests, all passing: the
  served-replay and forced-timeout tests with the switch off and on, and a new test that replays five routes (misses,
  an all-hit repeat) with the switch off and on and requires the output bytes to be identical. The timeout case with
  leases on also requires `go_count == 0` and the scratch rows byte-for-byte untouched. `test_exl3_lease_kernels_cuda.py`
  32 of 32, including a new case with advise and leases both on. Three backend mutants (no ack; copy with `plan.count`;
  host lease mode never enabled) were each killed. The OPEN 11 serving-path figure above.
- **CPU only:** the full `exl3` suite `test/registered/unit/kernels/ -k exl3`, **694 passed, 0 failed**, the same as
  before the change; and `layers/moe` exl3 tests. New CPU tests (switch default, host-before-thread order, one read,
  the backend's call order against a fake device) prove the wiring's shape and nothing about the device.
  `test_exl3_stream_trace.py::test_ram_miss_requests_are_traced_and_skipped_by_tier_sim` fails with a `KeyError:
  'pack_workers'` **at the parent commit as well**, unrelated to this change.
- **Not verified:** a real serving run with the switch on; lease mode in a graph with `advise` on (only the
  kernel-and-service level case above covers that combination); a model with more than one streamed layer through the
  graph path; eviction pressure with leases on through the backend; and everything section 20.2l lists as not
  established (the ordering argument of 6.4).

### 20.2n R3 resolved: superseded by the phase-1 progress callback (2026-09-22)

R3 is no longer held. It did not land in the held form, and the held diff
(`analysis/dsv41-drive/held/R3_retire_in_read.diff`) is superseded rather than applied:
hooking `serve()`'s abandon callback gave a call that ran once at batch 0, which is why
20.2k could not build a test that meant anything. The landed shape instead passes an
optional `progress` callback into `RowReader::read()` and calls it from the drain loop on
a time gate (`kProgressIntervalNs`, 200 us), so retirement happens **between batches of a
read**, which is what R3 was always for. Commits `d24c681fc3` (the call), `7b40535f91`
(constant and test), `e5082832fe` (a race in that test).

**Blocker 2 of the held decision is gone.** 20.2k recorded that a meaningful test needed a
multi-batch read, and that `pack_delay_ns` was unreachable from the service's own reader,
calling that "worth more than the one-line diff". `c8a1309cf7` carries a full `ReadFault`
through to the tier's reader, so the test now injects `pack_delay_ns=50_000_000` on a
three-row request: the read spans ~150 ms, far above both the 200 us progress gate and the
test's 0.1 s poll deadline, and the acknowledged lease is observed retiring while that read
is still running. That is the between-batches evidence 20.2k said did not exist.

**Mutant, demonstrated 2026-09-22** in a private detached worktree at `38e981fc83`, cores
0-31 (a GPU arm held 32-63): baseline **18 passed**, exit 0 -> `progress();` removed from
the drain loop -> **1 failed, 17 passed**, exit 1, failing only
`test_a_lease_acknowledged_mid_read_retires_before_that_read_returns` at its predicted
assertion `retired_mid_flight` -> reverted -> **18 passed**, exit 0. This is D1 of 20.2k's
ledger re-run against a test that can actually see the difference.

**Blocker 1 stands.** There is still no consumer: V2's per-row transfer, the long read that
holds an already-acknowledged lease, is not built. What changed is that the mechanism is now
demonstrated rather than argued, so the question is whether to carry it, not whether it works.

**Cost: no measurable throughput effect.** Served-path arms with lease mode on and off differ
by 1.8% (2.141 vs 2.102 token-weighted) against a within-arm session spread of 1.67-2.28 at
n=1 -- i.e. nothing resolvable. See `DSV41_REFERENCE.md` section 20, which also records why
throughput is the wrong instrument for a stall/progress property.

**One thing tried and reverted.** `de055eda15` throttled phase 1's clock read on idle spin
turns; reverted in `134d1cd0b2`. The `now_ns()` per drain turn is not known to cost anything
measurable, and the throttle added a second time source to reason about.

**20.2j addendum: a known limit of the S1 test.** The interrupt test replaces `_establish_gpu_completion` with a function that
raises `KeyboardInterrupt` in the calling thread, so it bypasses the real helper-thread `join`. It is accepted because a real
interrupt during the join lands in the same `except BaseException`, but the `join` itself is not exercised by it. The
post-landing review of `bcfe378f0d` found nothing to fix forward.
