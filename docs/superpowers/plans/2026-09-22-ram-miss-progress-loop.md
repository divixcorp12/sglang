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
pause can be granted during a read. Phase 2's are not. Before building phase 2, establish that a
consumer exists which issues speculative or prefetch work concurrently with a demand read -- and
size it, because at batch size 1 with sequential layers nothing does today.
