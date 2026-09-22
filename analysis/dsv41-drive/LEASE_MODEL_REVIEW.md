# Transcription review of `lease_model.py` (and `test_lease_model.py`), 2026-09-21

Reviewed at `845a54994e` (the model is 950 lines, the test file 32 tests; the brief said 31 and `LEASE_PROTOCOL.md` 18.3 says 31, the extra
test came with the lap-resume commit). Read-only: `lease_model.py`, `test_lease_model.py` and `LEASE_PROTOCOL.md` were not edited; the probes
below are scratch scripts that import the model and subclass or reconfigure it. Nothing under `python/`, no GPU. Code references are to `HEAD`
blobs (`git show HEAD:<path>`); Serena was unavailable, so this is reading plus running the model.

What was run: the whole test file (32 passed, 217 s, under a 6 GiB `systemd-run` cap); all 13 mutants with their shortest traces read; nine
probes of my own (Appendix A). Nothing was run on divix01.

## Verdict

**The transcription is faithful where it can be checked, the mutants fail for the reasons claimed, and the "today" calibration matches the
code with two stated caveats.** None of the headline results is contradicted by the model as written. Three things need saying before anyone
writes C++ against it:

1. **F1 (medium): the model cannot see a cycle that the Task 8 design contains.** It models the promoter and the eager caller as independent
   actors; in the design both are the scheduler thread, and the eager `before_host_use` blocks that thread in `synchronize()` while the poll
   step that releases promotion leases cannot run. Adding one coupling to the model turns a clean run into a deadlock (probe 6). It is not a
   bug in today's code, since no promotion lease exists there, and it resolves as a fatal after the 2 s device timeout, but it is a design hazard the
   "R2 checked" result does not cover.
2. **F3/F4: two designed rules cannot be shown necessary by the model** (terminal retirement, and fail-closed after a failed wait), one because of how
   the model ends a run after a fatal, one because an `assert` forbids removing the rule. The second is easy to add; my probe shows the model then
   finds the D1 violation at once.
3. **The limits list in 18.3 is honest about what it names and silent on about eight other blind spots** (F9), including the one behind F1.

## Findings

### F1. MEDIUM, a blind spot that hides a real cycle: promoter and eager caller are one thread in the design

- **Model.** `Model.host` (the promoter) and `Model.eager` are independent transition sources. `eager`'s first step requires `dpc in (IDLE, HALT)`,
  no pending stores and no outstanding entries (`lease_model.py` `eager`, the `E_IDLE` branch), i.e. it starts only once the device is idle. The
  comment says why: "The current stream was synchronized". The blocking synchronize itself is not a state, so nothing can be waiting in it.
- **Design.** Lease acquire and release are on the scheduler thread: the poll step at the end of the observer (`PROMOTION_ASYNC.md` 609-613;
  release "at COPIED", R-E at 472; `acquire_host_lease` "callable from the scheduler thread", 846). Eager use calls
  `Exl3RamMissService.before_host_use` (`exl3_ram_miss.py:401-407`): `torch.cuda.current_stream().synchronize()`, then `host.pause(...)`. Same thread.
- **The cycle.** A decode replay is in flight; its armed demand is deferred because the only victim carries a promotion lease; the copy has
  finished on the executor stream but the lease is released only by the poll step; the scheduler thread has just entered `before_host_use` and is
  blocked in `synchronize()` waiting for that replay; the replay is waiting for the service; the service is deferring behind the lease.
- **Probe 6 (Appendix A).** Subclass the model so that acquire and release are disabled while the eager caller is in a new "blocked in
  synchronize" state, everything else unchanged, `requests=2, menu=EVICTING, promotions=1, eager_uses=1`, no timeouts or faults: **Deadlock**,
  4,802 states, an 11-step trace (advisory takes slot 0 for expert 2; device posts a two-lane demand; promoter leases slot 0 and its copy ends;
  scheduler enters `before_host_use`). With timeouts and read failures on it is clean: the device timeout turns it into a fatal, which is what the
  design's own A3 text says a cycle costs.
- **Consequence.** Rare (needs a deferred demand and an eager entry while a replay is in flight, and EXL3 runs at batch size 1), but the outcome is a
  process abort. It also bounds what "R2 is safe" means: R2 was checked as an eviction-predicate property (as 18.3 says), and the model has no
  thread identity to check it against.
- **Suggested fix (design, not model).** Release host leases from something other than the scheduler thread (R1 already says "from any thread"; a CUDA
  host callback that only decrements the lease needs no CUDA call), or have `before_host_use` poll (`stream.query()` and release completed leases) instead of
  a blocking `synchronize()`. Either belongs in `PROMOTION_ASYNC.md` and `LEASE_PROTOCOL.md` 17.1.

### F2. LOW-MEDIUM, R6 (shutdown and host leases) is not modelled, and the limits list contradicts itself

- `environment()` flags `FreedWhileReading` only for the device's hold or copy state (`lease_model.py` "python frees memory": `dhold` and `dpc in (COPY_B, COPY_E, ACK)`).
  Probe 5 (`requests=1, promotions=1, shutdown=True`, 120,668 states, zero violations) reaches "python frees memory" **followed by** "promoter leases slot 0" and its copy:
  the promoter neither stops at shutdown nor is checked after the free. PROMOTION_ASYNC R6 (close admission, drain the executor stream, then unregister) is exactly the rule that
  would make this state unreachable, so the model cannot confirm or deny it.
- `LEASE_PROTOCOL.md:1508-1509` still says the model does "not cover ... host leases at all (Task 8)", which the Task 8 rows above it (1491) contradict; and "31 tests" is now 32.

### F3. LOW, terminal retirement cannot be shown necessary by this model

- Probe 3b: remove retirement by terminal (`Model.retire` filtered) in the 3-request, faults-on world: **no violation**, 779,445 states, complete. So the coverage counter
  "retired by terminal" proves reachability of the step, not that the design needs it.
- Reason: after a fatal the model lets the process die (`environment` "watchdog aborts the process"), `quiescent_problem` returns `None` for a dead run, and the leak check needs
  `not st["fatal"]`. A lease that is never retired after a fatal is therefore invisible. The safety half (I1) is still checked across a fatal (F4 shows it firing); the liveness half is not.
- Not a transcription error; it means E2(b), E4 and section 13 rest on the argument, not on the search.

### F4. LOW-MEDIUM, the D1 fix (fail closed, `go_count = 0`) has no mutant, because an assertion forbids one

`abort()` begins `assert c.fail_closed or not c.protocol` (`lease_model.py` `abort`). The rule is the design's answer to D1 (I4/E4), and `today()` reproduces the *bug*, but there is no
designed-protocol run with the rule removed. Probe 3a (a subclass whose `abort` publishes the terminal and then lets the copy run, protocol on, timeouts on, `requests=3`, `menu=MENU`) finds
`RecycledUnderReader` in 41,047 states with a seventeen-step trace: the device gives up, the service grants and publishes, the terminal retires the lease, an advisory evicts the slot the device is about
to copy. That is the E4 rationale, and the model can show it; it should be a test.

### F5. LOW, four of the mutants fire on the removed rule rather than on a consequence

`leases=False`, `host_guard=False`, `eager_host_guard=False` are flagged by `check_evict` at the first byte store whenever a lease of either kind exists, before any reader has necessarily begun
(shortest trace for `leases=False`: the request is served, then an advisory evicts, the device never starts its wait). `pause_counts_host` fires on reachability of "pause wanted while a host lease is held"
(a five-step trace ending in the flag the mutant itself raises). The reasons are correct and I1 is defined from publication, so none of this is incidental; but the tests do not show harm for these four.
(`test_the_detector_turns_a_recycled_slot_into_a_failure...` does show it for `leases=False`.)

### F6. LOW, the stream-dependency coarseness is sound for presence and optimistic for absence

`hneed = dk` at acquisition and `waits = hneed >= 0 and not (device idle or dk > hneed)` (`lease_model.py` `host`, `H_IDLE`/`H_COPY_B`): the copy is released as soon as the device posts the *next* request.
A real `wait_stream(producer)` (`expert_transfer.py:376`) orders after everything already enqueued, and the observer runs after the whole replay's work is enqueued, so the real copy waits for the tail of the
graph, several requests. Consequences: **the deadlock the model finds is real** (the real dependency contains the modelled one), so "A3 is necessary" stands; **a cycle at request k+1 is invisible**, so a clean
run with `copy_waits_on_serving=True` would not show a weaker dependency safe. Only the first use is made in the tests.

### F7. LOW, `map_read` is a dead knob

`Config.map_read` (`lease_model.py:79`) is set by `today()` and never read: the device resolves through the map exactly when `protocol=False` (the `RD` branch). So there is no "designed protocol plus map read"
run, and "today" is `protocol=False`, not "map_read". Harmless, but the 18.3 claim that the mutants cover D2 is really "today without the exclusion".

### F8. LOW, ring and counter geometry

- Ring 2 or 3 against 16, sequence range 8-11 against 2^32. At ring 2 the wrap skip makes seq 7 and seq 1 share request slot 0 (`ring_index`); in the real ring 14 is followed by 0. That is harsher, not softer, and
  the `defer_reuse=False` counterexample uses it, so the model shows the reuse rule *matters* at ring 2 but does not establish it at ring 16 (OPEN 8, which 18.3 names).
- Slot generations and slot content versions are modulo 4 (`% 4` in reserve, load and advisory). A slot would need four reassignments between a lease's grant and its ack to alias, which the lease forbids;
  only the mutants could reach it. Not a hazard here, unnamed.

### F9. Blind spots the limits list does not name

Each is a place where the model is coarser than the design, so a pass is silent about it:

1. **Thread identity** (F1).
2. **Post-fatal behaviour** (F3): nothing after a fatal is checked for leaks or liveness; the device stops requesting after the first fault (`dstop`), where the real post kernel returns early once the sticky flag is set but the wait and copy kernels of the rest of the replay still run
   (host_rows filled from the map, copy executed, `keep = 0`) until the watchdog (`exl3_ram_miss.cuh` wait kernel: `ok=false`, host_rows still filled, copy still runs). D1 is wider in reality than in the model.
3. **Multi-word atomicity.** The record and the lane request are single atomic writes (`P_REC`, `P_LREQ`) and the service reads both in one step (`observe`); the real seqlock writes seq 0, a fence, the payload,
   a fence, seq (`write_record`, `exl3_ram_miss.cuh`), and reads them separately. A torn read between the two reads is not representable.
4. **Copy concurrency.** The copy of lanes is serialised (`COPY_B`/`COPY_E` per lane); the real kernel copies lanes together. The acknowledgement comes after all lanes, as designed, so this cannot hide an I1 hole,
   but it cannot show an intra-kernel one.
5. **Shutdown with Task 8** (F2).
6. **No hot set, and `wanted` = lane experts** (the service protects lanes; the real `protect` is routes plus need, `serve()`, `exl3_ram_miss_host.cpp`). This makes the model's victim pool a superset, which is
   conservative for the designed protocol, but see the caveat under "Today" below.
7. **One tier, one row.** Advisories always target the demand's tier (named in 18.3), and there is no cross-row service ordering.
8. **CUDA error only during copy and acknowledgement** (`cuda_error` at `COPY_B`/`COPY_E`/`ACK`).

## Checked, no finding

### Fidelity of the steps

- **Service, against `LEASE_PROTOCOL.md` 7.1-7.2, 7.5 and the code.** `observe` follows the document's order (record seqlock, echoed generation, defer on an unretired request slot, terminal check, lane-request check, `wanted` from lanes);
  the lap resume `nd = head - (ring - 2)` (now with `skip0_lap`) matches `pump_demand`'s `next_demand_ = skip_zero(head - kDemandRecords + 2u)` (`exl3_ram_miss_host.cpp` at HEAD); `next_seq_service` matches the
  `skip_zero(next + 1u)` fix (`85cdbf9382`); `open()` starting at `done + 1` with a skip is modelled in `initial()`. Victim choice is nondeterministic over all subsets and orders (a superset of the LRU `take_slot_locked`), reservation is one atomic
  step (the real one is under `mutex_`), slot generations are bumped at reservation before any byte (7.2), leases are granted after the rows are ready and before the status (7.2), `retire` matches 7.5 (ack by generation, terminal by mask,
  entry freed only when every lane is ACKED or VOID, with `final` added so a partly granted entry is not freed early, which is what the real single-threaded loop guarantees).
- **Device, against 7.3-7.4 and `exl3_ram_miss.cuh`.** Post order lane request, record, head; wait on `demand_done` with the signed compare (`reached` is the model of `static_cast<int32_t>(a - b) >= 0`, `exl3_ram_miss_host.cpp:1347`); status check;
  per-lane row-result validation (generation, tag, expert); the single commit that sets the copy count; acknowledgement per lane after the copy with the slot-generation re-check; on failure zero the copy count, publish the terminal mask, raise fatal. Timeout is
  offered at any time while waiting (a superset of the real timeout).
- **Sequence arithmetic.** `ring_index` reproduces `(seq - 1) % 16` in unsigned arithmetic including the skipped index at the wrap; `reached`, the lap test `head - nd >= ring`, and the overrun/`done` behaviour on a failed record read match `pump_demand`.

### Today's code

- **Arming.** `armed = need_count > 0 || advise != 0` (`exl3_ram_miss.cuh` post kernel). With advisories on, which is the only configuration in which advisories exist, every request is armed, which is what the model does for `count > 0`; with advise off there is no advisory pressure,
  so the unarmed all-hit gather has nothing to race. The model has no unarmed-gather path and does not need one for these claims.
- **The exclusion.** The real rule is the advisory staleness test `reached(demand_head, request.after + 1)` at the start of `pump_advice`, a single service thread, and one tier per row, so an advisory can begin only before the row's demand is posted and finishes before the
  demand is served. The model's "no advisory eviction starts between post and consume" (`in_device_window`) is that rule; an advisory already running when the demand posts is still allowed to finish, as in the code.
- **D1.** After a timeout the real wait kernel raises fatal, fills `host_rows` from the map (misses become slot 0) and the copy still runs with `keep = 0` (`exl3_ram_miss.cuh` wait kernel); the model's shortest trace reads slot 0 while the service writes it, and flags no wrong bytes accepted. Matches the D1 text and the
  reviewed severity.
- **D2.** Without the exclusion the model accepts wrong bytes (an advisory evicts between the map read and the copy). It is a statement about the mechanism if the exclusion did not hold, not about today's code, consistent with D2 being a latent design defect.
- **Two caveats to the calibration, not errors.** (a) `wanted` is the lane list, so the model's service protects every planned lane; the real protect set is routes plus need, and whether planned is a subset of routes is OPEN 12. If it is not, today can fail stop where the model cannot. (b) After the first fault the
  model stops the device (F9.2).

### The mutants

All 13 shortest traces were read. Each differs from a clean control by one knob, and each violation is the removed rule's mechanism: `ack_after_copy=False` (the ack retires the lease, an advisory evicts while the device still holds the slot), `gen64=False` with a stale ack (the alias retires a fresh lease at grant),
`defer_reuse=False` (a request slot is overwritten with a granted lane unretired, the old lease is never released), `echo_gen=False` (the service's epoch is a lap behind after a lap resume, the armed request looks stale: the design change of 11.3), `defer_leased=False` (a demand needing two slots fails because one is
held by an unretired ack), `retire=False` (deferral on the unretired slot never ends), `free_needs_sync=False` (freed while a lane is held), `copy_waits_on_serving=True` (device waiting, service deferring behind a host-leased slot, copy queued behind the device: the cycle found by reading), `host_release=False` (deferred behind a lease never released). F5 and F6 qualify how much three
of them show.

### Larger worlds, designed protocol (my probes, all complete unless stated)

3 lanes, 3 slots, 4 experts, 2 requests, timeouts and read failures on: clean, 795,337 states. Task 8 actors with two promotions and two eager cycles, 2 requests, no faults: clean, 2,727,541 states. Two advisories, 2 requests, faults on: clean, 356,451 states.
Task 8, 3 requests, faults on: hit the 5,000,000-state cap with no violation (**incomplete**, so not a result).

### The lap-resume fix

The first version I read flagged the phantom only on the failed-read path; commit `845a54994e` moved the flag to any handling of sequence 0 and added `skip0_lap`, after which a lap that lands on 0 is flagged without the fix and clean with it. (My probe of that world before I noticed the commit found the same
gap independently: a fresh page reads the empty slot as a request and served it silently.)

## Not checked

`LEASE_PROTOCOL.md` sections 6, 9, 10, 12 (rows other than F10), 13 beyond the terminal mask, 14-16 were not transcribed against the model line by line, and Task 6's per-lane copy and finalize kernel are outside it (as 18.3 says). The design's own claims about `ld.global.nc`,
PCIe ordering of the service's stores and missing fences are outside a sequentially consistent model and were not assessed. I did not check the designed protocol against code because none exists. The hash-collision probability quoted in 18.3 was not recomputed. Model size sensitivity was probed only at the
sizes above.

## Appendix A: the probes (scratch, not committed as code; each imports the model unmodified)

Probe 3a (F4), a subclass that lets the copy run after an abort with the protocol on:

```python
class FailOpen(lm.Model):
    def abort(self, st, why):
        idx, gen, seq, count, experts, armed = st["dreq"]
        n = dict(st); n["dok"] = False; n["fatal"] = 1; n["dstop"] = 1
        n["pend"] = n["pend"] + (("term", idx, 0, (gen, tuple(range(count)))),)
        n["dpc"], n["di"] = (lm.RD if count > 0 else lm.COMMIT), 0
        yield (f"device aborts (fail open): {why}", self.emit(n))
# lm.Model = FailOpen; lm.explore(lm.Config(requests=3, menu=((),(0,),(0,1),(1,1))))  -> RecycledUnderReader, 41,047 states
```

Probe 3b (F3): a subclass of `Model.retire` that drops the transitions whose label contains "by terminal"; same config: no violation, 779,445 states.

Probe 6 (F1), the scheduler-thread coupling:

```python
E_SYNC = 5
class OneThread(lm.Model):
    def host(self, st):
        if st["epc"] == E_SYNC and st["hpc"] in (lm.H_IDLE, lm.H_REL):
            return                       # the scheduler thread is inside synchronize()
        yield from super().host(st)
    def eager(self, st):
        pc = st["epc"]
        if pc == lm.E_IDLE:
            if st["eleft"] <= 0 or st["paused"] or st["fatal"]: return
            n = dict(st); n["epc"] = E_SYNC
            yield ("scheduler enters before_host_use: blocks in synchronize", self.emit(n))
        elif pc == E_SYNC:
            base = (st["dpc"] in (lm.IDLE, lm.HALT) and not st["pend"]
                    and all(o is None for o in st["out"]) and st["spc"] == lm.S_IDLE)
            if not base: return
            n = dict(st); n["paused"], n["epc"] = 1, lm.E_TAKE
            yield ("eager caller pauses the service", self.emit(n))
        else:
            yield from super().eager(st)
# Config(requests=2, menu=EVICTING, promotions=1, eager_uses=1, timeouts=False, io_faults=False) -> Deadlock, 4,802 states
```

Probe 5 (F2): BFS over `Config(requests=1, menu=EVICTING, promotions=1, shutdown=True, timeouts=False, io_faults=False)` looking for a state with `freed` set and `hpc` in a copy state:
reachable, no violation flagged, 120,668 states.
