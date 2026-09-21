# Review: `retire_leases()` inside the reader's abandon callback (R3, LEASE_PROTOCOL 7.5 call site 2), 2026-09-21

Reviewed: `t5-patches/r3.diff`, `r3.msg`, `r3_ledger.md` (not landed), against `exl3_ram_miss_host.cpp` at `bcfe378f0d`/HEAD (`RamTier::serve`, `retire_leases`,
`release_lease_locked`, `pump_demand`, `pump_advice`, `handle_demand`, `RowReader::read`/`admit`, `RamThread::stop`/`watch`) and the two tests in the diff. Read-only; nothing run.
The package's own 718/718 and mutation numbers are taken from their report, not reproduced. Same reviewer as the shutdown-wiring reviews; no stake in this lane.

## Verdict

**Safe as far as I can see, and I found nothing that makes it worse; but I could find no consumer of the thing it does, so I recommend holding it, not landing it.** The four
questions asked (locking, harm to the request in service, `lease_changes_`, cost) all come out clean (section 1). The problem is upstream of them: the added call releases leases
nobody can use until the read ends, so the change is a production edit to the read loop whose only observable effect today is a counter (section 2). The test proves the call happens;
it cannot show that it matters, because nothing does yet.

## 1. The four questions

**(a) Locking: no path calls `read()` holding `mutex_`.** In `serve()` the reservation `lock_guard` is scoped to an `if (ok) { ... }` block that ends before the read, and the publication
`lock_guard` starts after it. `serve()`'s two callers, `handle_demand` and `pump_advice`, take no lock (the only other guard in `pump_advice` is `trace_mutex_`, and it is not held across
`serve`). The abandon callback is invoked only from `RowReader::admit`, which `read()` calls on the owner thread, in worker mode too: `admit(abandon)` is the first statement of each loop turn and the
`PackPool` workers only pack (I found no `lock_guard<std::mutex>` anywhere in the `RowReader` region; the first is at line ~1904). `mutex_` is a plain `std::mutex`, so the check that matters is that
`retire_leases()` is never reached under it, and it is not. The one other taker of `mutex_` that is not paused is `set_hot` (from `on_residency` on the scheduler thread): a brief wait, not a cycle.

**(b) Can retirement during a read hurt the request in service? No.** Its slots are `kLoading` and unleased; `release_lease_locked` decrements a lease count and changes no slot state or map.
Its own leases are granted after publication (`grant_lanes_locked`), so its `outstanding_[idx]` is not yet active. Retiring an earlier request's entry only makes that entry inactive sooner, which
is what `grant_lanes_locked`'s `entry.active` refusal wants.

**(c) `lease_changes_` waking a deferral mid-read: correct, and never earlier than the existing call would wake it.** A deferral exists only for `next_demand_`, and while it is refused the loop runs
`pump_advice`. A retirement inside an advisory read bumps `lease_changes_`; the next `pump_demand` sees `lease_changes_ != deferred_stamp_` and retries, exactly as it would after its own top-of-function
`retire_leases()`. The advisory abandons at the next row boundary when a demand is pending, so the demand's retry is not delayed by the advisory either way.

**(d) Cost while a lease is outstanding.** Lease-outstanding is the *normal* state in lease mode (each request holds its leases until the device acknowledges), so this is the common case, not an exotic
one: `mutex_` plus a scan of up to 16 entries per callback. But the callback runs only while `next_batch < batches` (it is inside `admit`'s `while`): once per one-batch demand, and per turn for an advisory
(step 1, one row reading at a time). That is a handful of calls per request at well under a microsecond each against reads of milliseconds. Negligible; unmeasured, as they say. The lease-mode-off case is one
load and a branch.

**Exception surface (not asked).** `release_lease_locked` throws on underflow. Previously that happened between requests; now it can happen inside `read()`, with reads in flight. `read()` has a
`Quiesce` guard for the packing workers, so the unwind is designed for, and the service thread's `run()` has no handler, so the outcome is the same `std::terminate` as before. I found no new outcome.

## 2. The finding: the change has no consumer today

**The callback is called far less often than "between batches" suggests.** `admit()` calls `abandon(next_batch)` only inside `while (... c.next_batch < c.batches)`. A demand reads at most 8
rows (`kMaxIds`, and `wanted` is bounded by it), step is `kBounceRows` = 8, so **a demand is one batch: the callback runs once, at batch 0, before any I/O** (the ledger's own D3 comment says so: "a one-row
read calls the callback once, at batch 0"). That is microseconds after `pump_demand`'s `retire_leases()` at its top, so for demands the new call can only retire an acknowledgement that arrived in that gap.
For advisories (step 1) it runs per turn while rows remain, which is the "between batches" case, and advisories grant no leases.

**Nothing can use a lease released mid-read before the read ends.** The candidates: (i) the request in service: reserved before the read, cannot use it; (ii) a deferred demand: only `next_demand_` can be
deferred, it is not looked at until the loop returns to `pump_demand`, which retires first; (iii) an eager pause: acknowledged only between requests, then it retires itself; (iv) promotions (Task 8): host
leases are separate and protect their own slots. So the only effect is that `leases_acked` moves earlier and the lease-hold interval statistic shrinks. The design's 7.5 call site 2 anticipates long reads holding
acknowledgements; that is a Task 6 situation (early acknowledgements of hit lanes arriving while the same request's misses are still reading), which does not exist yet.

**What the test does and does not show.** The test delivers the acknowledgement during the injected delay, which sleeps *before* `reader_.read()`; the first callback call is then the first thing in the read.
It kills D1 (no call), D2 (advisories only) and D3 (second batch onward), which is real: the call exists and runs at batch 0. It cannot distinguish "retires at the start of a read" from "retires between
batches", because a one-batch read has no between, and the ledger says multi-batch reads are unreachable from the service. The claim in the message, "a long read no longer holds leases the device has
acknowledged", is therefore untested in the sense that matters: no test has a long read, and no code path today has a long demand read.

**A probable flake in that test.** The retirement happens at read start; `rows_read` is incremented at publication, after the read. The test polls `leases_acked` every 2 ms, then reads `demand_done`, then
asserts `rows_read == rows_before + 1`. If the first poll that sees `acked == 1` lands inside the read (about the row's read time out of a 2 ms period), `rows_read` has not moved yet and the assertion fails.
Estimated at a few percent per run (read time over 2 ms, worker hand-off making it longer in the `pack_workers=2` case); not run. The assertion is also unnecessary: the ordering argument
(nothing else retires during the 0.8 s stall) already proves the call was in the read. Replace it with `host.busy_since_ns() != 0` (the request is still in service), which is deterministic and states the same thing.

## 3. Recommendation

1. **Hold it.** A production edit to the read loop should have a consumer. There is none until Task 6 lands early acknowledgements; hold it with that dependency named, or land it with the message corrected to say
   it is a no-op for one-batch demands today and for every lease-mode request's own read.
2. If it lands, **fix the flake** (`busy_since_ns() != 0` in place of the `rows_read` assertion) and **add one advisory-path test** (an advisory of several rows with an acknowledgement delivered between its rows), since
   advisories are where the call actually runs repeatedly and nothing tests it.
3. **Do not describe D1 to D3 as three mutants of one behaviour**: all three die to the same single test in two modes, so they are one test's worth of assurance.
4. **Box 5 stays open**, as they say; I would add that this change would not move it, because the requirement it names (the worker keeps submitting, reaping and processing cancellations while acknowledgements
   are outstanding, and never waits on one) is met by polled retirement at the top of `pump_demand`, not by retiring mid-read.

## 4. Not checked

Whether any *other* consumer of a mid-read retirement exists outside the C++ (Python side reads `leases_acked`? I found none); the worker-mode packing turns (as they say); the actual read duration of the fake rows
(the flake estimate depends on it); the plan's "box 5" wording (taken from their description); multi-rank behaviour. Nothing was run.
