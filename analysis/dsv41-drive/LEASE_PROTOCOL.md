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
  them. Which exit path production actually takes is a separate question: the docstring of
  `record_graph_step` in `exl3_stream_trace.py` says the scheduler "is SIGKILLed at
  shutdown", in which case none of these hooks run. I have not confirmed that from the
  launcher, so I state the hazard's scope as: **a normal-exit and test-teardown hazard;
  under SIGKILL it does not occur, and nothing in this design should be read as fixing a
  production defect that SIGKILL already hides.**
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

**The experiment that would settle it (needs the GPU; not run; not runnable in this
phase).** On the RTX 5090, under the GPU lock and with crypto-c9's scheduling:

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
no-advise configuration; its cost is unmeasured **[OPEN 11]**. The plan anticipates this:
"Removing its all-hit handshake is a separate optimization after equivalent protection is
proven." What an equivalent protection would be, and why I did not design it here: a
device-side lease taken by an atomic on a mapped per-slot counter would have to be
ordered against the service's eviction test with a Dekker-style argument in both
directions (device increments then re-checks the slot's generation; service sets a
"evicting" state then re-checks the count). That needs its own proof and its own model
check, and it is Task 5's non-goal.

---

## 16. The worker never waits for an acknowledgement, and why it cannot deadlock

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
back.

Task 6 must add, and I have not designed: per-lane wait/copy/ack kernels with a lane
predicate that reads the lane's own readiness (the wait becomes per lane); a finalize
kernel that publishes the terminal mask after all lane kernels; a single request timeout
budget; A4 above; and the launch-count cost. The `LaneRequest` mechanism already gives
the service the lane list before the request is served, which per-lane early publication
needs.

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

CPU-only (simulated device), most of them:

1. Lease-pressure: hold a lease (no ack) and force admission pressure on the same row;
   assert the leased slot is never chosen as a victim, the demand defers rather than
   fails, and it proceeds after the ack. Include a RAM *hit* lane.
2. Delayed acknowledgement with a newer request present: the leased source stays
   byte-identical (poison the slot on any illegal rewrite, as the existing poisoned-recycle
   tests do).
3. Duplicate lanes: two lanes, one expert, two leases, one ack releases one.
4. F1, F2, F3-shaped (Task 6 later), F12, F13: each row of section 12 has a test naming its
   row.
5. Skipped copy emits no ack: with `go_count == 0` no `LaneAck` word is written (assert the
   words stay zero).
6. Terminal mask retires exactly its lanes; an ack and a mask never both retire a lane
   (double-retire is an internal error).
7. Generation wrap: seed the page head and the device near `0xFFFFFFF0`, drive through the
   wrap; no `kOverruns`, leases retire across the wrap, the `LaneAck` aliasing case (a
   stale word from `G - 2^32`) is rejected, and a lap that crosses the wrap (unarmed
   records posted past it while the service lags) still serves the armed request that
   follows. The first part is written (`test_exl3_ram_miss_wrap.py`, both rings) and should
   fail on today's `pump_demand` (D6); the rest needs the lease code.
8. Request-slot reuse: retire-before-reuse, deferral without lapping (OPEN 8's test).
9. Pause with a leaked lease is refused (F10); shutdown ordering including the
   helper-thread timeout path selecting quarantine; a fake CUDA error selecting
   quarantine and asserting no `unregister` and no free call.
10. Worker never waits: a fault-injected reader with 100% of acks withheld still submits,
    reaps, honours `pause`, `stop` and cancellation (a watchdog in the test, as the thread
    tests already do for hangs).

GPU (under the lock; not run here):

11. The `ld.global.nc` visibility experiment of section 6.6.
12. Graph-parity: byte-exact output with leases on versus off; `go_count` zero on an
    injected timeout and no source bytes read (compare against a poisoned slab).
13. Cost of the added `LaneRequest` fences and the ack kernel per layer, and of arming every
    `count > 0` record without advise (OPEN 5, OPEN 11).

### 18.3 The model check (done, bounded)

`analysis/dsv41-drive/lease_model.py` is a self-contained explicit-state model of this
protocol; `test_lease_model.py` (23 tests, about two minutes, pure Python) pins what it
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
| Service that does not skip 0 (D6) | a phantom sequence, nothing else |

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
- It did not cover the `pause`/eager path (F10) or its graph-lane/host split, host leases
  at all (Task 8), the lease-mode arming cost, or the watchdog beyond "fatal is followed by
  abort".
- Its advisory pressure is applied only while no demand is visible to the service, and to
  the single modelled row. In the real service an advisory for the *next* row can start while
  a demand is deferred (section 8); that is a different tier and cannot take the deferred
  demand's victims, so I do not expect it to change the result, but the model does not show
  it.

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
- **[OPEN 6]** The exact PTX ISA wording for `ld.global.nc` on the deployed toolkit, and
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
- **[OPEN 11]** The per-layer cost of arming every `count > 0` record when advise is off.
- **[OPEN 12]** Whether planned experts are always a subset of the routed experts in the
  post kernel's `protect` set. The design does not rely on it (7.1 step 4).
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
