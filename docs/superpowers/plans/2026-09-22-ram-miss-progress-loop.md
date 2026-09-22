# The RAM-miss service thread's progress loop

**Status:** not started. Written 2026-09-22.

## Read this before proposing it on the queueing rationale

The usual argument for a progress loop is that draining one entire request before returning makes
other work wait. **On this workload that is false, and it was measured, not assumed.**

From `/mnt/nvme1/dsv41-stagetrace-20260921-220109.jsonl`, schema 5, the whole trace:

| measure | value |
|---|---|
| records examined (`demand` 48,732 + `touch` 64,668) | 113,400 |
| whose `observed` falls inside another record's `pack_start`..`pack_end` | **0 (0.00%)** |
| whose `observed` falls inside another record's **full service window** `observed`..`done` | **0 (0.00%)** |
| `observed` -> `reserved` | p50 **6 us**, p90 8 us, p99 **11 us** |
| `backlog` at observation | p50 0, p90 0, **max 0** |

`backlog` never exceeds zero anywhere in the trace. Nothing ever queues behind the owner.

The second row matters more than the first and was added after the pack-window measure was found too narrow: the pump is blocked for the **whole** request (`submit` -> `last_cqe` -> `pack_end`), not merely while packing, so a request arriving during the read but outside the pack window would have been missed. Measured over the full `observed`..`done` interval by a sweep line, the service windows **never overlap at all** -- not demand against demand, and not `touch` against demand either. There is nothing for a yielded pump to service, including the 64,668 touches.

There is a structural reason. **EXL3 launches are gated to batch size 1**
(`expert_stream_requirements.py`; see `.claude/rules/divix01-run-protocol.md`), and layers run in
sequence, so the model issues one RAM-miss request and blocks on it before issuing the next. One
outstanding request at a time, by construction. A loop that services *other requests* during a read
would service nothing.

So do not justify this work by queueing, concurrency or backlog. Those are zero and will stay zero
until something generates concurrent work.

## The drive is idle three quarters of the time, and the plumbing to fill it exists unused

Measured over the same trace, per decode forward, merging every request's `submit`..`last_cqe`:

| measure | value |
|---|---|
| fraction of the forward span with a read in flight | p10 0.172, p50 **0.246**, p90 0.328 |
| idle, no read in flight | p50 **206.14 ms** per forward (p90 255.42 ms) |
| forward span, first `observed` -> last `done` | p50 270.10 ms |
| `batches` per request | **1**, for all 48,732 |

So the NVMe is busy about a quarter of the time. That is the real saturation opportunity, and it is
large.

Note where it is **not**: `batches == 1` for every request in the trace, so a request already hands
the queue all of its rows in one submission. Intra-request saturation is fine. **The idle sits
between requests**, while the model computes a layer and has not yet asked for the next one's rows.

The *plumbing* for that gap is already written and switched off -- note the distinction, because the policy that would drive it does not exist (see phase 0). `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`
(`exl3_expert_format.py:374`, "Option F advisories") gates a full advisory path: advisory records on
their own ring, `serve(request, advisory=true)` (`:2599`), a per-row abandon callback that gives up
the moment a demand is posted or a pause or stop is requested (`:2720`), `kStatusCancelled` (`:99`)
and `kAdvisoryRows` (`:2789`). The traced run had it off: zero advisory records among 113,400.

## Phase 0: turn the existing prefetch on -- BLOCKED, and not for a reason the code shows

**Do not start this.** The advisory *plumbing* above is real and complete, but there is no expert
predictor trained on the DSV4.1 network, so nothing can decide which experts an advisory should
fetch. An advisory that guesses wrong spends the idle drive on rows nobody wants and is then
cancelled when the real demand arrives -- worse than idle, because the bandwidth is consumed and
the bounce slots are occupied.

This is recorded because the env var is discoverable and the C++ path looks ready. It is not. The
prerequisite is a trained predictor for this network, which is its own body of work
(`scripts/expert_prediction/`), not a flag flip.

If a predictor does land, the measurement below is what phase 0 becomes. Measure the drive-busy fraction with advisories enabled, and count how many are
cancelled by `demand_pending()` before finishing.

That cancellation count is the number that decides whether phase 2 is worth building. Today an
advisory **abandons** when demand arrives, because the pump can only service one request at a time.
If advisories mostly complete, the existing design already fills the gap and a re-entrant loop buys
little. If they are mostly cancelled, the I/O is being started and thrown away, and *that* is the
concrete argument for letting an advisory stay in flight alongside a demand -- which is phase 2.

## `--max-running-requests` is a throughput lever, not a single-prompt one

The gate at `expert_stream_requirements.py:235` pins it to 1, and unlike its neighbours it carries
no rationale comment. The documented reason sits in `MOE_EXPERT_TRANSFER.md:339`, under **"Caveats
before shipping"**: "Measured only at BS1 ... Concurrency changes the memory split between KV and
the hot cache." That is a measurement scope and a sizing interaction, not a correctness invariant.
`DSV41_REFERENCE.md:231` gives the mechanism: the pool configurator reserves SWA slots from
`max_running_requests` before sizing the full pool, and `compare_oracle.py` needed 4 to avoid a
starved pool.

**But be clear about what raising it would and would not do.** It caps how many *inference
requests* -- separate prompts, separate HTTP calls -- decode concurrently. It is the decode batch
size. It is not parallelism within one prompt.

- **With concurrent traffic:** more sequences means the router's expert union per layer grows, so a
  layer's RAM-miss request carries more rows. That deepens the queue per submission, which is the
  direct answer to the single-row shape (`rows_asked == 1` for 67.9% of requests). It does **not**
  create concurrent requests: a forward still takes the whole batch through each layer once, so
  there is still one expert gather per layer. Bigger requests, not more of them -- so it does not
  give a re-entrant loop anything to service either.
- **With one prompt:** it changes nothing. One sequence is one running request whatever the flag
  says, and the 206 ms of idle drive stays exactly as measured.

So it is a throughput lever for a multi-user workload, and irrelevant to the latency of a single
interactive generation. Raising it costs a re-derived KV-versus-hot-cache budget and more bytes
read per step; better drive utilisation is not automatically faster decode.

## What is actually wrong

`retire_leases()` is called from exactly one place in the normal path: the top of `pump()`
(`exl3_ram_miss_host.cpp:1868`). The chain `pump -> handle_demand (:2795) -> serve -> reader_.read
(:2714)` runs on one thread, and `RowReader::read()` contains a `while (true)` drain loop
(`:660` onward) that does not return until every row of the request is read and packed.

**So for the whole duration of a demand read, no lease is retired.** That window is not small: the
summed `reserved` -> `mapped` span is **118.36 ms at p50 per decode forward** (p90 187.44 ms), and
a single request's `submit` -> `last_cqe` runs tens of milliseconds.

Three consequences, none of which need a second request to exist:

1. **Slots stay unusable.** A lease the device acknowledged early in the read is not released until
   the read ends, so its slot cannot be taken by the next request even though the GPU is done.
2. **An eager pause is refused while `lanes_outstanding_` is non-zero** (`:2794` and its comment).
   That counter cannot fall during a read, because only `retire_leases` lowers it.
3. **The effective `retire_leases` poll interval is the whole request.** This is the mechanism
   behind T5's retired double-signal claim: the detector needs a lane's ack and the request's
   terminal visible to a *single* poll, and with the poll interval this long they never are. See
   `task6-v1-checklist.md` T5.

## Phase 1: run the periodic work inside the read loop

Do this first. It is small, it is the change that captures the three effects above, and it needs no
re-entrancy, no state machine and no concurrency.

Call `retire_leases()` from inside `read()`'s drain loop, bounded so it does not become a spin.

**Verified safe:** `read()` is called with `mutex_` **not** held. The only `lock_guard` in `serve()`
around it is the scoped S2 reservation hold, which closes before the read begins. `retire_leases()`
takes `mutex_` itself, so this is deadlock-free -- but that is a property of the current code, and
any future yield point introduced *under* a lock would not be. State the invariant in the code, not
just here.

**Do not call it every turn of the loop.** The loop's turns are as short as a `_mm_pause()`, and
`retire_leases` takes `mutex_` and walks `kDemandRecords` entries. Gate it on elapsed time or on a
completion having been reaped, and record which was chosen and why.

**Falsifiable prediction, and the reason to run T5 again after this lands.** If the poll interval is
what makes T5's `kLeaseDoubleSignal` detector unreachable, then shortening it should make the
detector fire under T5's mutant, where today it stays 0 with the mask provably wrong. Either
outcome is informative: firing confirms the mechanism recorded in the T5 row, and staying silent
means the entry-closing gate (`:2216`, `:2244-2246`) is the binding constraint on its own, as
`t6t-failure` argued. Run it; do not assume.

## Phase 2: the re-entrant loop

Only after phase 1, and only with a consumer that generates concurrent work -- otherwise this
measures zero, which the table above already established.

Turning the drain loop into something that can yield means the following stop being true, and each
one is load-bearing:

- **`Call c` is a single member** (`:636`, `c = Call{}`). One read at a time is currently an
  invariant of the type, not a coincidence.
- **`Quiesce` guarantees no packing worker is still copying when `read()` returns** (`:657`),
  because the caller releases the slots on return and the next read reuses the bounce. A yield that
  returns to the pump is not a return to the caller -- do not let the destructor fire on a yield.
- **`read()` publishes nothing; the caller publishes** (`:469-472`). The all-or-nothing boundary is
  deliberate: a failure leaves fully packed rows in unpublished slots, never a partial row in a
  published one.
- **The watchdog counts a wait as a hung read** via `busy_since_`/`kBusySeq` (see the deferral
  comment at `:1888`). A yielded request must not read as hung.
- **`next_demand_` must not advance while a request is mid-flight**, or the demand ring's ordering
  and the lapped-record accounting (`kDemandRecords`, `kOverruns`) stop meaning what they say.
- **`entry.active` is per ring entry** and is cleared once no lane is in state 1. Interleaving
  changes when that happens relative to the terminal.

## Testing

The campaign rule applies: **a test without a demonstrated killing mutant does not count as
written** (see `2026-09-21-task6-device-tests.md`). For phase 1 the test is that a lease
acknowledged partway through a read is retired before the read returns; the mutant is removing the
in-loop `retire_leases()` call, and it must go red for that reason. Record the restored baseline.

## What would make any of this measurable

Phase 1's effects are observable without new workloads: slot reuse latency, and whether an eager
pause can be granted during a read. **Phase 1 is the only item here that is not blocked on
something else**, which is why it is the one to build.

Everything aimed at the idle drive needs a producer, and each candidate producer is blocked on its
own prerequisite:

| lever | what it needs first |
|---|---|
| prefetch advisories | an expert predictor trained on the DSV4.1 network |
| a larger decode batch | concurrent traffic; irrelevant to one prompt |
| phase 2, the re-entrant loop | one of the two above, since neither today's workload nor a bigger batch produces overlapping requests |

Be honest that this leaves the single-prompt case open. The 206 ms of idle drive per forward is
real and none of the levers here reach it: one sequence asks for one layer's rows at a time, waits
for them, and computes. Closing that gap means predicting the next layer's experts, which is the
predictor problem, not a service-thread problem.
