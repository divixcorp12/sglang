# Review of `PER_ROW_TRANSFER.md` (Task 6 design), 2026-09-21

Reviewed: `analysis/dsv41-drive/PER_ROW_TRANSFER.md` at `116533cb61`, against the `dsv41` working tree (the native
service and `ops/moe/exl3_ram_miss.py` are being edited by others; symbols cited, not line numbers). Reviewer: the
author of `PROMOTION_ASYNC.md`, who therefore knows the lease contract and the transfer path; that is also a bias,
because the two designs make requests of each other (section 7 of the document). Read-only: nothing under `python/`
was touched, no GPU, nothing run except `sha256sum` and `git`. Serena unavailable; grep and reading.

**Severity.** HIGH: can produce a wrong result or a safety failure in the design as written. MEDIUM: a claim the
document's own text or the source contradicts, or a gap an implementer will settle by guessing. LOW: precision.

## Verdict

No HIGH finding. The two simplifications the document makes to other documents (A4 not needed as written; DECIDE 3
stands) are **both correct against the source**, with one condition each that the document should state (section
"Arguments checked"). The document does lead with the redirect. It has four real problems: the early-signal cost is
not in the summary next to the number it buys (F1), the plan's "at most 2.55 ms / 1.0%" is contradicted by the
document's own 5.06 ms / 2.0% (F2), the failure-cleanup path of `serve()` is not addressed for published-and-leased
rows (F3), and one lease-hold claim contradicts the design's own stage order (F4).

## The questions asked

**1. Does it lead with the redirect?** Yes, with a caveat. Section 0 item 1's first sentence is "The idea is worth
building, but not as the plan specifies it"; item 2 recommends V1 and expects V2 REJECTED. "SUPPORT" first appears in
section 1.2, after the ceiling is framed. The registered verdict line ("SUPPORT; SPREAD-IRRELEVANT") is not the first
thing a reader meets, so the trap is avoided. The caveat: item 1 states 7.5% to 15% before it says what the saving
requires (F1).

**2. A4 and DECIDE 3.** Both arguments hold; see "Arguments checked".

**3. The failed-demand publication change.** Noted with a safety reason, not justified as a design choice, and the
service-side consequence is missing (F3).

**4. Existing safeguards and bounds.** No claimed existing safeguard is false. Bounds at the end.

## Findings

### F1. MEDIUM: the early signal is specified, but not where the number is, and V1 and V2 are not compared on it

Your point from the precheck files, checked: today nothing reaches the device until `read()` returns. `serve()` sets
`kReady` and calls `publish_map` for every row after `RowReader::read` returns, and only `handle_demand` stores
`demand_done`; the wait kernel polls `demand_done` and then translates every lane. So the RAM-hit saving (87% of the
modelled figure) needs a **new** service-to-device readiness publication made at reservation time.

What the document does: section 3 states the fact plainly ("Nothing reaches the device until `read()` returns"); section
4.3 specifies hit lanes "published (and leased) in the reservation step, before `read()`"; section 1.4 says A2 "needs
the service to publish hit lanes before it reads; it does not today". So it is absorbed, and not waved at.

What is missing:

- **Section 0 does not say it.** Item 1: hit lanes "need no read at all and can be copied the moment the service has
  reserved the request". A reader who stops there takes that as true of today's service. It is true only of a service
  that does not exist yet. One sentence belongs beside the number.
- **Section 1.4's cross-reference is wrong.** "It does not today (section 6)": section 6 is the ceiling's limits and
  contains no such discussion. The content is in sections 3 and 4.3.
- **The cost is not itemised.** Section 8's table has "Post kernel: ord, h, k, deadline" and "Per-row publication under
  `mutex_`" but no row for "arm every record with hits and publish hit lanes before the read". The lease-mode
  round-trip row is close but is Task 5's; the early publication itself, and the device's poll of it (`__nanosleep(256)`
  plus a PCIe read per stage), are the new costs of the 87%.
- **V1 as designed uses per-lane readiness words, not one phase flag.** Sections 4.1 and 5.5 give V1's hit stage the
  same `RowResult.ready` acquire per lane as V2. So the comparison "V2 costs 4 more stages than V1" hides that V1 is
  built on the full Task 5 lane protocol, and a cheaper V1' (one service-written "hits leased for request G" word) is
  not compared. OPEN 5 half-asks the question ("whether V1's hit group needs to wait on `RowResult` at all"). If both
  mechanisms need an early signal, the marginal cost of the *signal* (phase-level against per-lane) belongs in the
  V1-against-V2 comparison of section 5.6, not only the marginal kernel count.
- **V1's predicted upper bound is V2's number.** Section 5.6 predicts V1 "20.8 ms/step to 38.6 ms/step (BEST)". BEST
  is hits first *and* miss rows in pack order, which is V2's figure. V1's own ceiling in the document's arithmetic is
  38.6 - 5.06 = 33.5 ms (13.0%).

### F2. MEDIUM: the marginal contribution of per-row is quoted two ways, and the plan quotes the smaller as "at most"

The document's section 0 item 2: at most **5.06 ms/step (2.0%)** with the best order, and 2.55 ms (1.0%) if lane order
is random. The plan's Task 6 blockquote says per-row's own contribution "is at most 2.55 ms/step, about 1.0%, which
is below what this plan's own measurement design can resolve." Checked:

- 5.06 is right: `sum(m-1) * c / steps` = (1407 + 2*352 + 3*70 + 4*11 + 5*2) * 1.055 / 495 = 2375 * 1.055 / 495 = 5.06.
  Per request, per-row over a two-phase copy saves `(m-1)*c` when the miss rows land more than `c` apart, so this
  is the correct upper bound for V2 over V1. 2.55 is the registered `RANDOM-miss-only`, which is per-row against
  *batched*, miss rows only, random order. It is not V2 minus V1.
- **The plan's "at most 2.55 / 1.0% / below resolution" is therefore not a bound.** The gross bound is 2.0%, which is
  *above* the 1.5% the plan's design resolves (`task1e-PREDICTIONS.txt` line 10: 80% power at about 1.5%, conditional on
  within-cell sd). The conclusion "below resolution" holds only **net** of the stage cost: the document's section 5.6
  gets 1.1% to 1.5% net for the best order, which is at the resolution, not clearly below it. The redirect is still right
  (V1 first; V2 must beat V1 to be admitted), but the plan's sentence overstates it. **The plan blockquote should say
  "at most 5.06 ms (2.0%) gross; about 1% net of launch cost; at or below the design's resolution."**
- **The random-order figures do not describe V2.** V2 in the document has the order array (section 4.2), so its lane
  order is not random. The random figure describes the plan's original per-row with a fixed lane order and no `ord`,
  and that mechanism is not "slightly better than V1": by the document's own numbers it is much worse. V1 does not
  depend on lane order (hits first by construction), so V1 is about 33.5 ms; the plan's fixed-order per-row is 19.2 ms,
  about 14 ms *below* V1. That is unregistered arithmetic (38.6 - 5.06 - 19.2), and I have not seen it stated. It is the
  strongest single argument for the redirect, and it makes section 5.6's "random order 0.1% to 0.5%" for V2 over V1
  a mis-scoped comparison.

### F3. MEDIUM: per-row publication meets `serve()`'s failure cleanup, and the document does not say how

Section 7: "A failed demand today publishes nothing unless every row landed. With per-row publication, rows that packed
whole before a failure stay published as ordinary `kReady` rows ... a behaviour change and any test pinning the old
behaviour needs updating." Checked against the source:

- The existing guarantee is real: the `serve()` comment says so, and `serve()`'s final loop publishes a slot only if
  `ok`, or if the request was cancelled and `packed[i] != 0`; otherwise `release_locked`. The documented reason to keep
  publishing gated on whole rows is the repository's own history (`LEASE_PROTOCOL.md` section 1.3: a bug shipped in which
  the reader published bytes never read).
- The document's safety claim is supported: `pack_one` checks `filled < needed` and sets `c.failed` rather than pack
  bytes no drive delivered, and the `packed[]` gate is already used to publish whole rows of a cancelled advisory. That
  is a precedent for publishing whole-packed rows on an incomplete request, and the document should cite it as the
  justification; it cites only the negative.
- **What is not addressed:** with a per-row hook, a row is `kReady`, mapped, **and leased** (Task 5 leases at
  publication) by the time a later row fails. `serve()`'s cleanup for `!ok && !cancelled` then reaches `release_locked`
  for every slot in `slots`, including the published and leased ones, and Task 5 makes `release` of a leased slot throw
  (`LEASE_PROTOCOL.md` section 8). Either the cleanup must skip published lanes, or the service thread throws on the
  first mid-request failure. The document's "they stay published" implies the former without saying it changes the
  cleanup loop. `LEASE_PROTOCOL.md` section 7.2's "a failed demand publishes no `RowResult` at all" also stops being true,
  which is REQ 1's territory.
- **Justification of the new behaviour, not just its safety:** the document never says *why* early miss-row publication is
  worth touching the most bug-prone site in the reader. It is only needed for **V2** (per-row miss publication). V1's hit
  lanes are rows already `kReady`, so V1 changes nothing about failed-demand publication. That is a further reason for
  V1 first, and the document's section 7 does not say so.
- **A test the design owes:** a mutation control (the standard used tonight) for the hook: publish before the row's
  `memcpy` and `_mm_sfence`, or for a row with `filled < needed`, and assert a test fails.

### F4. LOW-MEDIUM: "longer-held hit leases" contradicts the design's own stage order

Section 5.2 and the section 8 table: "hit leases are now held from reservation to acknowledgement, which can span the
whole read wait (p50 5.3 ms, p90 8.9 ms)". In the design, `A_s` acknowledges the stage's lanes after `C_s`, and the
chain is `W_1 -> C_1 -> A_1 -> W_2`: the hit stage copies early and acknowledges early, so a hit lease is held from
reservation to the end of the *hit* copy (a few milliseconds), not through the read wait. In Task 5's whole-request
protocol the hit lease is taken at publication, after the read, and acknowledged after the batched copy. The Task 6 hold
is not obviously longer. If the claim is true for some reason (for example `W_1` cannot start until the request is
posted and the device is behind other work), the document should say what; otherwise the "new contention" row and the
remark about Task 8's R4 are unsupported.

Also stale: section 7's "Its `lease_on_ready` / R3 window is unaffected." `PROMOTION_ASYNC.md` withdrew `lease_on_ready`
at `6b0b2b41f6` (eight minutes before this document's commit), and its R4 hold is now "enqueue to ack", not "the copy".
Cosmetic, but it cites a mechanism that no longer exists.

### F5. LOW: precision

- Section 1.4, "does not today (section 6)": wrong cross-reference (F1).
- Section 3 says `pack_one` "refuses to publish a row whose bytes a drive did not deliver": it refuses to *pack* (sets
  `c.failed`); publication is `serve()`'s `packed[]` gate. Same effect, different place.
- Section 2's caution that pack order may be "forced by the reader rather than by the drives" is right and can be
  stated exactly: `pack_one` takes the lowest ordinal among rows that are `Ready`, and the precheck's own note says all
  extents of a request completed within 2.9 ms of each other against about 2.7 ms to pack one row, so by the time the
  first pack finishes every row is ready and the lowest ordinal wins. Ordinal order is a property of serial packing, as the
  plan blockquote says. It is not evidence about drive order.

## Arguments checked

**A4 is not needed as written; the requirement is all-or-nothing reservation. CORRECT, with one condition.**
`LEASE_PROTOCOL.md` A4 is about lane `j+1`'s readiness not waiting on an ack of a lane `>= j+1` of the same request.
In the source, `serve()` reserves every slot for `wanted = protect + need` in one `mutex_` pass before any I/O, with
`take_slot_locked(request.row, wanted, false, ...)`: `wanted` protects every hit from being a victim, `fallback` is
false, and on failure every slot taken is released. Nothing in the request's readiness therefore waits on the request's
own leases, and the "S -/-> A_s" edge does not exist. The stream-order half (`A_s` before `W_{s+1}` on one captured
chain) is plausible and the document makes it a graph-shape test; I did not verify the capture behaviour (no GPU).
*Condition to state:* the argument needs the hit **lease taken inside the same critical section as the reservation**.
The document says "published (and leased) in the reservation step"; it should say "same `mutex_` section", because the
current `serve()` releases `mutex_` between reservation and `read()`.

**DECIDE 3 stands for the current synchronous service. CORRECT, same condition.** During `RowReader::read` nothing else in
that tier evicts: one service thread; advisories run between requests; eager `assign` needs a pause, which
`RamThread::pause` acknowledges only between requests. For a hit lane, publication and lease coincide with the
reservation, so "lease at publication" and "lease at reservation" are the same instant. The document's REQ 3 (a
constraint on any future asynchronous service) is the right scoping. The retraction of the earlier "must overturn DECIDE
3" claim is warranted.

**The closed form.** Re-derived: with `e_j = max(e_{j-1}, r_j) + c`, `T_p = max_j (r_j + (k-j+1)c)` and
`S = T_b - T_p = min_j [(M - r_j) + (j-1)c] >= 0`; one slow lane at a uniform position gives `E[S] = (k-1)c/2`. Correct,
under uniform `c`.

**Numbers checked against source or files:** the two SHA-256 values (`0c145def...`, `e81b8b60...`) match the files;
`sum(m-1)` and 5.06; `676 of 2,022` and the 3.20 ms gather per layer (`DSV41_REFERENCE.md` 18.2); MIRROR_ROWS p50 4.164 /
p99 7.387; the 1.5% resolution; `graph_gather_rows > MAX_IDS` is refused at `attach` (so `LEASE_PROTOCOL` D7 is stale, as
stated); `_validate_plan` accepts a one-element `int32` slice as `count` and equal-length 1-D views (so no new copy kernel
is needed); `route_tables` scales the route weights by `keep` and zeroes `expert_count` when `keep` is 0; `_apply_graph`
copies the live `topk_ids` into `backend.routes`; `pack_one` picks the lowest ordinal among `Ready` rows.

## Not checked

`RowReader::admit_batch` and `refill` internals (the "one batch, extents in row order" claim); the copy kernel's
warp-to-row mapping (the "`count == 1` gets all 64 warps" claim, which OPEN 1 turns on); `exl3_stream_trace.py` and the
stamp semantics beyond `row_pack_ns` being packed in `pack_one`; `per_row_precheck.py` itself and the two post-hoc scripts
(only their printed output); `task1e_analyze.py`; `LEASE_PROTOCOL.md` sections 6, 13, 15 in full; whether a captured
graph gives the linear `A_s -> W_{s+1}` edge; every GPU-side test in the section 5.4 matrix and whether it is
implementable; the claim that 94% of requests wait at least `h*c` (post-hoc, unregistered); A1 (hit lanes spread evenly
over layers), which the document itself marks as the caveat that carries the result. Nothing was run.

---

# Second pass, 2026-09-21: the early-signal question, the all-or-nothing search, and the document at `6f91e24683`

The document changed after the review above (`53bc8c7229`, `c5c572bd92`, `4808935c70`, `6f91e24683`). This pass answers the
team lead's two factual questions from the code, then re-reads the revised document. Same limits as above: read-only, no
GPU, nothing run.

## A. Is there an early device-visible signal? Enumeration from the source

Words the device could observe (the mapped request page, plus `slot_map`), who writes each, and when relative to `read()`.
Sources: `exl3_ram_miss.cuh` (device constants and kernels), `exl3_ram_miss_host.cpp` (`RamTier`, `pump_demand`,
`handle_demand`, `serve`).

| Word | Writer | When, relative to `RowReader::read()` in `serve()` | Read by a device kernel today? |
|---|---|---|---|
| `kDemandHead` (0) | device post kernel | before the service sees the request | n/a (device-written) |
| record `kRecSeq`, `kRecRow`, need / protect ids, `kRecArmed` | device post kernel | before the service sees the request | n/a |
| **`kDemandDone` (4)** | service, **one place only**: `RamTier::pump_demand`, after `handle_demand` returns | **after** `read()`, after every row is published, after the record status | **yes, the wait kernel polls only this word** |
| record status (`kRecStatus`) | service, `set_status` in `handle_demand` | after `serve()` returns | yes, read after `demand_done` |
| `kFatal` (8) | device (`raise_fatal`) or host | on failure | yes |
| **`kBusySeq` (24)** | service, `handle_demand` | **`store_release(kBusySeq, request.seq)` is the first thing `handle_demand` does, before `serve()`, so before reservation and before `read()`; cleared to 0 after `set_status`, before `pump_demand` stores `demand_done`** | **no: `exl3_ram_miss.cuh` defines only `kDemandHead`, `kDemandDone`, `kFatal`, `kAdviseHead`; `kBusySeq` is read by the host watchdog only** |
| `kHeartbeat` (28) | service | once per 1024 idle iterations, and per request served | no |
| `kAdviseDone` (20) | service | after an advisory | no (different ring) |
| `slot_map[row][expert]` | service, `publish_map` | a **newly read** row: after `read()`. An **evicted** victim: `-1` at the reservation pass, before its bytes are overwritten. A **hit**: unchanged, already `>= 0` | yes, `ld_volatile`, in the post kernel and after `demand_done` |

**Answers to the specific questions.**

- *Is there a device-visible signal that a lane is a RAM hit whose row is resident, written before the read completes?*
  **No.** Nothing per lane and nothing per phase. `slot_map` cannot carry it: a hit's entry is already `>= 0` before the request
  exists, and reservation only ever *removes* entries. A waiter cannot tell "hit, protected by this request" from "hit, not yet
  evaluated".
- *Are hit lanes resolved at plan time, before any read is submitted; is that visible to the device or host-side only?* Yes, and
  it is host-side only. `RamTier::serve`, in its first `mutex_` pass and before any I/O, treats `tier.expert_slot[expert] >= 0` as
  a hit (touching its LRU stamp) and everything else as `missing` (assigned a `kLoading` slot). That resolution lives in the
  C++-private `Tier` and reaches the device only through the map after `read()`. The device's own resolution is the post kernel's
  `need`, decided from `slot_map` at post time: a racy hint. Confirmed as the document states it (section 3.1).

**Verdict on the three branches: Confirmed (no early readiness signal), with one fact the document does not have.** `kBusySeq` is
a device-observable, service-written word stored *before* reservation and *before* `read()`. It is not a readiness signal and
it is not a lease. But it is an early "the service has taken this request" signal, and combined with invariants the current
service already has, it may be sufficient to start the hit phase **without any new service code**:

1. `handle_demand` runs on the single service thread, after any advisory has returned. So once the device sees
   `kBusySeq == seq`, no advisory is in progress and none can start until this request finishes.
2. `serve()`'s reservation pass takes victims with `take_slot_locked(row, wanted, false, ...)`: `wanted = protect + need` is
   never a victim, and there is no fallback. For the router-miss producer every planned lane is in `protect`
   (`LEASE_PROTOCOL` OPEN 12, answered in the document), so a lane that is a hit **after** `kBusySeq == seq` is stable until the
   request ends.
3. Eager `assign` needs a pause, which `RamThread::pause` acknowledges only between requests, and `before_host_use`
   synchronizes the stream first.
4. A hit that an earlier advisory evicted before `kBusySeq` reads as `-1` in the map and simply joins the second phase.

If that holds, a hit-phase gate of "poll `kBusySeq == seq` or `demand_done`, then read `slot_map` for the planned lanes, copy
those `>= 0`" is **safe for the current synchronous service and costs no service change**. What that would cost: a change to
the wait kernel (poll two words), a new stage kernel, and no `RowResult`, no lease, no per-lane words. It is *not* Task 5
compliant (there is no lease; the plan wants one), and it is **valid only while the service stays synchronous**: any
asynchronous `progress()` service, and Task 8's promotions on the same tier, would break invariant 1 or 2. That is a
maintenance hazard, and the reason the plan wants leases, but it changes the cost picture the document gives: "V1 needs one
early publication of the hit lanes" is an upper bound on the cost of a *safe* V1, not the cheapest possible V1.

I have not run anything. Two things would need checking before anyone relies on this: (a) that the device really observes
`kBusySeq == seq` in time (it is set and cleared within one request; a device that polls late sees `0` and falls through to
`demand_done`, which is safe but gives no early start; the fraction of read waits during which it is visible is unmeasured);
(b) invariant 2's "every planned lane is in `protect`" for every producer in production, not only the router-miss one.

## B. The ungated case: the document's section 3.1 and 3.2 safety claim

**Claim (the lead's task B):** for a RAM hit the `slot_map` entry carries no ownership, so gathering hits early would rely on
the temporal exclusion that early copying itself removes, and an advisory on the same tier can evict unprotected rows.

**Verified at the source, with two precisions.**

- **True as stated for an early copy that is not gated on the service.** The eviction mechanism is real: `serve()` for an
  advisory builds `wanted` from the *advisory's own* `protect` and `need`, and `take_slot_locked(row, wanted, false, ...)` evicts
  any `kReady`, non-`hot` row outside it, publishing `-1` before the bytes are overwritten. A device that took its hit lanes
  from the post-time map and copied them while such an advisory ran would read a slot being rewritten. Nothing would flag it: the
  wait kernel's check after `demand_done` re-reads the map and counts only planned experts with `slot < 0`, so it detects a *missing*
  row (fail-stop, loud) and not an early copy that used a slot since reassigned (silent wrong bytes).
- **Precision 1: the eviction happens at the advisory's reservation pass, not throughout its read.** An advisory reserves all its
  slots in one `mutex_` pass and only then does I/O. So the exposed window is not "an advisory in progress" but "an advisory whose
  reservation pass runs after the device's map read and before the service takes the demand". The service takes demands ahead
  of advisories (`pump_demand() || pump_advice()`), but it does not interrupt an advisory mid-serve: the window is up to one
  advisory serve (reservation plus at most one row of I/O, about 10 ms), not the few microseconds the document gives for the
  *armed-record* race in 3.2. The two are different races and 3.2's "few microseconds" does not bound the early-copy one.
- **Precision 2: it is conditional on advisories.** With `advise` off no advisory exists and nothing evicts in the tier between
  post and wait. That is exactly today's unarmed all-hit path (`touch_request`, `7526d4cdc3`): an existing device gather of hits with
  no service handshake, safe by temporal exclusion. The document says this (3.2) and I agree with it. It also means the hazard
  is latent for anyone who enables prefetch (`advise != 0`) while adding an early hit copy.

So: **the claim is right and load-bearing, and it is not overstated for the ungated case.** It is *overstated* only if read as
"any early hit copy needs Task 5's lease": section A above shows a `kBusySeq`-gated copy preserves the temporal exclusion in the
current service. The precise statement is "an early hit copy needs *either* a service-side ownership grant (Task 5's lease) *or*
a gate that keeps the copy inside the interval in which the single service thread is provably inside this request".

## C. Search for consumers of the all-or-nothing property

**Question:** does anything depend on "if any row of this demand failed, no row of it is visible", as opposed to per-row
correctness? The property is `serve()`'s publication gate: a row publishes only if `ok`, or (advisory, cancelled) `packed[i] != 0`;
otherwise `release_locked`.

**Result: a confirmed negative for production code, and two positive findings about tests and documents that the author's
sentence does not cover.**

*Production code (negative, agreeing with the author).* I looked at every reader of published rows:

- Device: after a failed armed demand the wait kernel sees `status != kServed`, raises fatal and sets `keep = 0`; it does not
  branch on how many rows are visible. `unserved_misses` is checked only when `ok` (served), so it depends on "served implies
  every planned row mapped", the positive direction. Not the all-or-nothing property.
- Python: `NativePinnedSlotTable.expert_to_slot` / `lru_order` / `mapping` read only `kReady` rows, and are used under a pause;
  `fail_stop_check` raises out of the scheduler on a fatal, so the process stops. **There is no recovery path that continues
  serving after a failed armed demand and infers state from the tier.** (The service thread does keep running until the process
  dies: `kLateAfterFatal`.)
- The eager Python `ensure_rows` has its own transaction ("on any failure every slot this call assigned is freed") and it
  publishes at `assign`, before the read, under a pause: the opposite ordering, not a consumer.
- Counters and stage records: `rows_demand` and `kRowsRead` add `published`, and the stage record's `rows` is `published`, even on
  a failure (`published` is `0` today). A per-row hook would make a failed demand add rows to `layer_rows()`, which
  `_trace_step` labels "demand rows only: advisory reads are not misses". Moot after a fatal; noted for the trace consumers.
- `kVersion` bumps on any request that changed the map, including failures (a test pins it).

*Tests (positive finding: none would break, and the two cited or named pins do not pin it).*

- **The test the document cites as the pin does not pin the tier property.**
  `test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one` (`test_exl3_ram_miss_split.py`) calls the standalone reader
  (`ops.read_rows_with_fault`, which builds its own `RowReader`) and asserts the *reader's* property: the failing row is untouched and
  every other slot is whole or untouched. "The tier releases those slots and publishes nothing" appears only in a **comment**.
  It survives per-row publication unchanged, because it never involves publication.
- **The test named for the property is vacuous for multi-row failures.**
  `test_a_failed_read_publishes_none_of_the_rows_it_had_already_packed` (`test_exl3_ram_miss_thread.py`) injects
  `fail_reads=True`. In `serve()` that flag short-circuits *before* `reader_.read()` (`if (fail_reads_.load()) { ... ok = false;
  status = kStatusFailed; } else { reader_.read(...) }`), so no row is ever read or packed. Despite its name, it never has a packed
  row to fail to publish. It passes today and would pass under per-row publication.
- The remaining tier tests that pin a failed demand are single-row or fail before any read
  (`test_a_failed_read_frees_its_slots_and_reports_failed`, `test_a_file_cut_short_after_open_fails_the_read`,
  `test_a_request_that_evicts_and_then_fails_bumps_the_version`, `test_a_full_cache_serves_...fails_one_that_cannot_fit`):
  with one row, or no read, there is no earlier packed row for per-row publication to leave behind.
- **No test exercises "a multi-row demand, some rows packed, then a failure" at the service level.** `ReadFault` (`hold_ordinal`,
  `part_error`, `cqe_error`, ...) is reachable only through the standalone entry points (`set_fault` is called only from
  `exl3_ram_miss_read_rows_faulted` and its two siblings); `RamTier::inject` exposes `fail_reads`, delay and abandon only. So the
  guarantee the author is reversing is **unpinned by any test that could fail if it were reversed, in either direction**. That is
  the finding: changing it is not caught, and neither is keeping it. Before per-row publication, the tier needs a service-level
  mid-read fault injector (a `ReadFault` on the tier's own `reader_`), and the new behaviour needs its own test.

*Documents and the model (the consumers that do encode it).*

- `lease_model.py`: the model's read failure fires **after reservation and before any row loads** (`if c.io_faults and plan:` frees every
  reserved slot and goes straight to `S_STATUS`), so the model has no state in which a failed request has published a row or granted a
  lease. `LEASE_PROTOCOL.md` section 7.2 ("A demand that fails publishes no `RowResult` at all ... no lease is ever granted for it")
  and failure-matrix row F1 say the same. These are the consumers the design invalidates; the document's REQ 1 and REQ 2 point at them.

*Not searched, so this is not proof of absence:* analysis scripts that read stage-record `rows` for failed requests
(`native_mirror_report.py`, `task1_arm_verdict.py`: I saw that the precheck filters to `served`, not the others); manual GPU tests
under `test/manual/dsv41/` (grep found no assertion on the property, but I did not read them); any external consumer outside the
tree.

## D. The document at `6f91e24683` against the first-pass findings

- **F1 (early signal not in the summary): fixed.** The summary now says the saving "first requires a new early, device-visible,
  generation-tagged readiness word", states the 87% as an upper bound, and section 3.1 is the enumeration above, correctly. Two residuals:
  it says "no early signal exists" without the `kBusySeq` fact (section A here), and the "(section 6)" cross-reference at the A2 assumption
  is still wrong.
- **F2 (2.55 quoted as "at most"): not fixed.** The summary still reads "at most 5.06 ms, 2.0%, and 2.55 ms, 1.0%, if lane order is
  random: below the ~1.5%", which calls a 2.0% figure below 1.5%. The plan blockquote has the same shape.
- **F3 (`serve()`'s failure cleanup for published-and-leased rows): not addressed.** Section 7 now asks that "anything that relied on
  [the guarantee] needs an explicit look", and section C above is that look for the guarantee itself, but the cleanup loop
  (`release_locked` on every slot of a failed request, which Task 5 makes throw on a leased slot) is not mentioned.
- **F4 (longer-held hit leases; stale `lease_on_ready`): not fixed.** Both lines are still in section 7 and the cost table.
- **New and good:** the document now states that its pack-order 100% is a packer property "not tested with the tie-break changed", puts
  the `kDemandDone` finding first, cites the test that pins the coverage check with its negative controls (reported, not run), and carries
  the "I could not find another consumer" sentence honestly.

## E. The four adversarial items

1. **A1 and the 37% floor: the arithmetic is right; the wording is ambiguous.** The bar `(P1 - L) / T >= 3%` needs
   P1 >= 0.03 x 257.5 + 1.0 = 8.7 ms; P1 = 19.2 with 2.55 miss-only leaves a 16.7 ms hit part, so 8.7 - 2.55 = 6.2 ms is needed:
   37% of it. But "37% of the modelled hit lanes" means 37% of those **in read layers** (about 2.2 per read request, 31.8 per step by my
   arithmetic from the post-hoc k histogram and 14.4 read requests per step), not of the 112 hit lanes per step across all layers. The
   two are different denominators; the document uses 112 in the same paragraph as the 26 non-read layers x 6 lanes = 156, which is the
   right comparison for "could all hit lanes sit outside read layers". The floor holds. What I cannot say is whether A1 is *likely*: a
   crude independence check from the document's own numbers (per-lane RAM-miss rate 19.1 / 131 = 14.6%, k about 3.3, so a layer reads with
   probability 1 - 0.854^3.3 = 41%) predicts a read-layer fraction of about 41% against 33% to 36% observed (`676 / 2,022`; 14.4 / 40): mild
   concentration of misses, in the direction that puts more hit lanes in read layers, not fewer. That is consistent with A1 being
   conservative, and it is not evidence, because lanes per layer are not recorded. REQ 4 stays the fix.
2. **Pack order 1,888 / 1,888: agree with the document that it is the tie-break.** `pack_one` picks the lowest ordinal among `Ready`
   rows, one row per loop turn, and the precheck's own note says all extents of a request complete within 2.9 ms of each other against
   about 2.7 ms to pack one row. By the time the first pack ends every row is ready, and the lowest ordinal wins. It is a property of
   serial packing. It says nothing about drive completion order, and the document now says so.
3. **c = 1.055 ms/row: plausibly link-limited, and the document should say why.** 13.3 MB per row over 1.055 ms is about 12.6 GB/s. The
   reference states the whole host is Gen3 and measures the link at 12.02 GB/s (`DSV41_REFERENCE.md` line 186, "Link time / miss row at
   12.02 GB/s"), and 18.2 says the gather is "at the PCIe line rate". So `c` is the link's time per row, not a kernel latency, which
   supports the model and has one consequence the document only half draws: early copies do not make rows cheaper, they move link time
   into the read wait, and the whole saving is bounded by link idle time during waits (the 93 ms/step of read-wait against 138 ms of
   modelled gather). OPEN 1's alternative (an 8-SM latency limit) is what a Gen3 link at 12 GB/s makes unlikely, but a `count = 1` copy has
   not been measured on this path.
4. **The baseline: correctly stated, not yet enforced.** Section 5.6 names Task 5 in lease mode, batched (V0'), as the fair baseline and
   lists arms A0 today, A1 V0', A2 V1, A3 V2. The ceiling (7.5% to 15%) is computed against *today's* batched path, so it is V1 against V0'
   that it predicts (the same overlap, both paying lease overhead) and V1 against today is that minus the lease-mode cost X on all 40 layers.
   X is unmeasured (OPEN 11), and the document says so. It is netted if and only if the arms are run as listed; a report that quotes only
   A2 against A0 would flatter Task 6 by exactly the cost it inherits from Task 5.

## F. Not checked in this pass

Whether any producer other than the router-miss path can put a planned lane outside `protect` (invariant 2 of section A);
`kBusySeq` visibility timing on the device; analysis scripts that read `rows` for failed requests; manual GPU tests; whether the wait
kernel could be split without changing capture stability; `RowReader::admit_batch`/`refill`; every claim in the first pass's "not
checked" list that is not repeated above. Nothing was run.
