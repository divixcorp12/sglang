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
