# Independent review: wiring `Exl3RamMissService.shutdown()` into `Scheduler.release_host_resources()` (2026-09-21)

Reviewed: the proposal `LEASE_PROTOCOL.md` section 20.2i and its addendum (`3d702eb1c2`, `88c126a655`), against `scheduler.py`
(`release_host_resources`, `run_scheduler_process`'s `finally`), `exl3_ram_miss.py` (`Exl3RamMissService.shutdown`,
`_establish_gpu_completion`, `_quarantine`, `_quarantine_service_at_exit`), `ops/moe/exl3_ram_miss.py` (`Exl3RamMissHost.stop`,
`close_admission`), `expert_stream.py` (`ExpertPinnedHostCache.close`, `quarantine`), `expert_host_tier.py`
(`quarantine_host_slabs`), and the tests `test_exl3_ram_miss_shutdown.py` and the `release_host_resources` tests in
`test_expert_doorbell_copier.py`. Read-only: nothing under `python/` was touched, no GPU, nothing run. Reviewer: the author of
`PROMOTION_ASYNC.md`; I have no stake in this lane, and did not write or approve any of it.

**Severity.** HIGH: the change can leave production worse than not making it. MEDIUM: a claim the code or tests contradict, or a
defect a test cannot see. LOW: precision.

## Verdict

**Do not land it as proposed.** The load-bearing claim ("worst case equals today's behaviour") is **false in two ways I can show from
the code**, both cheap to fix, and the proposed tests cannot see either. The placement is defensible and I found no missed ordering
that makes it incorrect; the double-free claim holds by reading but is untested. The mutation set is not sufficient.

## 1. The failure mode: is the worst case equal to today's? No.

The claim: if the barrier raises or times out, the service quarantines rather than frees, so the worst case is today's behaviour
(retained allocations until teardown). True for the barrier itself, and I verified it (below). It is not true of the whole
function, because `shutdown()` has two other steps that today's exit path never reaches in a state where a *free* is possible.

**What I verified is true.** `shutdown()` (non-exit) sets `uncertain` from `_establish_gpu_completion()`, which runs the device
synchronize on a helper thread with a deadline (`SGLANG_DSV41_RAM_MISS_TIMEOUT_MS` / 1000 + 5 s) and returns a reason on an error or
a deadline miss. Any `BaseException` at `close_admission` or the barrier is caught and becomes `uncertain`. A non-None `uncertain`
goes to `_quarantine(reason)`, which detaches each tier's exit-time unregister finalizer and takes a never-released reference on the
slabs, page, slot map and lease block. The exit-hook path (`at_exit=True`) sets `uncertain` unconditionally and never frees. So a
barrier failure quarantines exactly as the exit hook does today.

### F1. MEDIUM-HIGH: a failing `host.stop()` with a clean barrier frees the slabs under a possibly live service thread

```python
finally:
    try:
        if self.host is not None:
            self.host.stop()
    finally:
        if uncertain is None:
            ... tier.close()          # frees
        else:
            self._quarantine(uncertain)
```

`uncertain` is decided **before** `stop()` runs and is never updated by it. If the barrier succeeds (`uncertain is None`) and
`Exl3RamMissHost.stop()` raises (`exl3_ram_miss_stop_thread` is a TVM FFI call that can raise; its `finally: close()` then destroys the
native handle regardless), the inner `finally` runs the *free* branch. The service thread, which writes into the slabs through raw
addresses (the comment two lines above says so), may still be running. Today the same failure at exit is harmless, because the exit
path quarantines unconditionally. **After wiring, this is a path strictly worse than not wiring.** The same holds for a
`KeyboardInterrupt`/`SystemExit` delivered during `stop()`.

*Fix:* a failure or a not-joined thread from `stop()` must make the shutdown uncertain, so the quarantine branch runs. *Test the
proposal lacks:* a fake `stop` that raises with a clean fake barrier; assert the tiers are quarantined and not freed. *Mutation:*
"a `stop()` failure does not change `uncertain`" must fail that test. None of the 13 step-6 mutants or the four proposed ones covers it.

### F2. MEDIUM: the barrier runs on a helper thread, and a new thread does not inherit the scheduler's CUDA device

`_establish_gpu_completion` runs `self._synchronize()` (`torch.cuda.synchronize()` with no argument) in a new `threading.Thread`.
`torch.cuda.synchronize(device=None)` synchronizes the *calling thread's* current device, and a thread that never called
`set_device` starts on device 0. The scheduler main thread selected its device with `set_device` (thread-local). In a process that
serves a GPU other than device 0 (any tensor-parallel rank above 0 without a per-rank `CUDA_VISIBLE_DEVICES`), the barrier
synchronizes the wrong device, reports "completion established", and the slabs are freed under kernels still running on the right one.
This is from PyTorch/CUDA semantics as I know them, **not run**; on divix01's single GPU it cannot show. The tests cannot see it
either: they replace `_synchronize` with a fake, and their tiers are on `cpu`.

*Fix:* synchronize an explicit device (the tiers' `cache.device`; the service already holds `device_side` with it), or call
`torch.cuda.set_device` in the helper first. *Test:* patch `torch.cuda.synchronize` to record its `device` argument and assert it is
the tiers' device and not `None`. *Mutation:* revert to the argument-less call.

### F3. MEDIUM: "worst case equals today" ignores time and ordering

Today the exit hook's `stop()` and quarantine run **last**, after the whole graceful path has finished. After wiring, a failing barrier
costs up to `timeout + 5 s` (7 s at the 2 s default; 65 s if the timeout is set to 60 s) **and** `host.stop()` has no deadline at all (it
joins the service thread, which may be inside an NVMe read), and both now run **before** `hisparse_coordinator.destroy()`,
`tree_cache.release_host_resources()`, `decode_offload_manager.release_host_resources()`, the capturer teardowns,
`rank_consensus_checker.shutdown()` and `abort_distributed_environment()`. A supervisor's kill timeout during that wait skips all of
them, which today's ordering does not; a hung `stop()` withholds the distributed abort from the other ranks. The proposal should state
the added worst-case delay and its consequence, and choose deliberately. Two options: run the block **after** the cheap releases and
before the `destroy_global_*` calls (the ordering reason the proposal gives, "the barrier needs a working CUDA context", does not
depend on it running first: nothing above it frees a CUDA context), or bound `stop()` with the same deadline and quarantine on a miss.

### F4. LOW-MEDIUM: the import at shutdown

The proposal imports `exl3_ram_miss` inside the `try` to reach `Exl3RamMissService._instance`, so every graceful shutdown of every
non-EXL3 run imports the MoE stack (unmeasured, as it says) and, if that import fails, logs a spurious exception. A guard costs nothing:
`sys.modules.get("sglang.srt.layers.moe.exl3_ram_miss")` is `None` if no service could exist, so skip without importing. Test:
with the module absent from `sys.modules`, the delegating function must not import it (mutation: import unconditionally).

### F5. LOW: `_shut_down` is set on entry, so a shutdown that fails partway disables the exit hook

`shutdown()` sets `self._shut_down = True` first; `_quarantine_service_at_exit` then returns immediately. If the orderly path is
interrupted after that (an exception out of `_quarantine` itself, or a `BaseException` in the free loop), tiers whose finalizers are
still attached are unregistered by the interpreter's exit hook with no barrier, which is the outcome the quarantine exists to prevent.
Unlikely, and after a clean barrier harmless, but it is a state the exit hook cannot recover from. Track completion (`_completed` /
`_quarantined`) separately from "started", and let the exit hook re-attempt the quarantine when a started shutdown did not complete.

## 2. The placement

After `stop_doorbell()`, before the other host releases, before `abort_distributed_environment()`, inside the graceful gate.

- **The gate is real and the only caller.** `release_host_resources` has one production caller, `run_scheduler_process`'s `finally`,
  under `if scheduler.gracefully_exit` (the other hits are tests and `tree_cache`/`decode_offload_manager` methods of the same name).
  The comment there ("Graceful path only: on the exception path the GPU may be wedged and the synchronize() ... could itself hang")
  is the same convention the proposal follows. Note the gate is **outside** `release_host_resources`, so the wiring test that drives
  `release_host_resources` cannot test it; that constraint is enforced by call structure, not by any test.
- **Ordering constraints I looked for and did not find violated:** other producers against the slabs (the service thread is stopped by
  `shutdown()` itself; the doorbell copier is stopped first; I found no other `threading.Thread` in the expert modules besides the
  trace-drain thread, which reads GPU trace buffers, not slabs; promotion and prefetch copies are covered by the device-wide
  barrier); a communicator (none used); and later users of the service (`_refuse_if_shut_down` makes any later call raise loudly, and
  nothing after this block in the function touches EXL3).
- **What the proposal's reason 2 gets wrong or thin:** "before the other host releases: the barrier needs a working CUDA context" is
  not a constraint on order (see F3). A reason that does hold, and is not stated: `hisparse_coordinator.destroy()` and the host KV
  release do their own `synchronize()` per the comment in `run_scheduler_process`, so a hang there is also possible; the RAM-miss
  block is not more exposed than they are.
- **An ordering the test cannot assert:** the proposal says "before `hisparse_coordinator.destroy()`", but the test stub sets
  `hisparse_coordinator=None`, so the block's position relative to it is untested (see 4).

## 3. The double-free claim

"The two paths cannot both free, because `_shut_down` makes the exit hook a no-op." **True by reading; not exercised by any test.**

- After an orderly success: `tier.close()` calls the tier's `weakref.finalize` (`_release_slabs`) once, and a finalizer is callable
  once. The exit hook `_quarantine_service_at_exit` calls `shutdown(at_exit=True)`, which returns at `if self._shut_down`. `_stop_live`
  (ops) calls `Exl3RamMissHost.stop()`, which is a no-op once the handle's finalizer is no longer alive. So no second unregister, no
  second native close.
- **What is missing is a test of that sequence.** The five tests in `test_exl3_ram_miss_shutdown.py` call `shutdown()` or
  `shutdown(at_exit=True)` separately; none runs an orderly shutdown *then* the exit hook. Add it: orderly shutdown, then
  `_quarantine_service_at_exit(weakref.ref(service))`, assert no further `free`/`quarantine` events, no exception, and every tier's
  finalizer dead. Mutation: the exit hook ignores `_shut_down` (or `shutdown` does not set it).
- The residual hole is F5, not a double free: a half-finished orderly shutdown followed by a no-op exit hook.

## 4. The proposed mutations: not sufficient

The proposed set: block omitted; before the doorbell stop; after the tree-cache release; `except` removed (plus, in the addendum, the
function constructing the singleton).

- **Three of the four die to one assertion.** Omitted, before-doorbell and after-tree-cache are all killed by the single
  `assert order == [doorbell, ram_miss, tree_cache]` (and "omitted" also by test 2). That is one test's worth of assurance for three
  mutants. Acceptable only if the driver records that the failing line is that `==` and not a fixture error, and if the list contains the
  hisparse position (below).
- **"`except` removed" would be killed by an exception escaping the call, not by the assertion that states the requirement.** The
  existing doorbell-failure test has the same shape: with the `except` gone the call raises and pytest reports an error at the call site,
  masking the assertion (`tree_cache.release_host_resources.assert_called_once()`) that says "later releases still run". This is the recorded
  defect class of a guard detecting the mutant while masking the assertion. The test should call the method inside `try/except`, assert it
  returned normally *and* that the later releases ran, so the mutant fails on the named assertion.
- **Missing mutants, each a plausible wrong edit:**
  1. **Nested under `if expert_hot_cache_manager is not None`** (the natural place to paste it): the RAM-miss shutdown then never runs
     when the manager is absent. Needs a stub with `manager=None` and a live service.
  2. **`except Exception: return`** (early return): later releases skipped. Killed by the failing-shutdown test, but a different mutant
     from "except removed" and should be listed.
  3. **Wrong hisparse position:** add a recorder for `hisparse_coordinator.destroy()` and assert `[doorbell, ram_miss, hisparse, tree_cache]`.
  4. **`stop()` failure treated as clean** (F1), **argument-less synchronize** (F2), **unconditional import** (F4), **exit hook ignoring
     `_shut_down`** (section 3).
  5. **`shutdown(at_exit=True)` passed by the delegating function** (never frees): test 2's freed-versus-quarantined assertion kills it; list it.
- **What no test can show, stated so no one credits it:** that the real barrier orders GPU work (F2's thread/device question is part of
  this), that `cudaHostUnregister` succeeds on real slabs (the tier tests use `device="cpu"`, so `_release_slabs` unregisters an empty list),
  that every teardown path sets `gracefully_exit`, and the `abort_distributed_environment()` ordering (outside `release_host_resources`).

## 5. What I did not check

The real CUDA barrier and the real `cudaHostUnregister` (no GPU); PyTorch's thread-device behaviour is from knowledge, not run; the
service thread's join behaviour under a stuck I/O (`RamThread::stop` was not read for a timeout); whether any other repository path
constructs `Exl3RamMissService` outside the manager; the mlx scheduler mixin's `release_host_resources` test; the doorbell copier's own
stop; anything after `run_scheduler_process` returns beyond the atexit hooks named above.

## 6. Recommendation

1. Fix F1 and F2 in `shutdown()` / `_establish_gpu_completion` first (small, in the module the proposal already owns), each with the
   test and mutation above.
2. Decide F3 explicitly and write the added worst-case delay into the proposal.
3. Add the sys.modules guard (F4) and the completion state (F5) if cheap.
4. Replace the mutation list with the one in section 4 and require the driver to log the failing assertion line for each, so a fixture error
   cannot count as a kill; report kills by distinct killing test, not by count.
5. Land the wiring after that. It is a small change and the direction is right: an orderly release is better than an unconditional quarantine,
   provided the failure modes cannot make it worse.

## 7. Addendum: F1 and F2 are defects in landed code, not only in a proposal

F1 and F2 are properties of `Exl3RamMissService.shutdown()` and `_establish_gpu_completion` **as landed in `fa22865ac6`**; the proposal only makes them
reachable. They are harmless today for a specific reason: `shutdown()` has no production caller except the exit hook, and the exit hook passes `at_exit=True`, which
sets `uncertain` unconditionally, so the free branch is reachable only from tests. That is "unreachable", which is not "correct". The step-6 ledger (13 mutants, 20
tests) never asked what happens when a *late* step (`stop()`) fails after an *early* decision (`uncertain`) has been taken, so neither defect could have surfaced there.
Any fix should be reviewed as a change to landed code, and the wiring must not be the first thing to exercise the free branch outside the tests.

F2 in particular has no instrument on divix01 (one GPU, device 0 only): the unit test recording the `device` argument passed to `torch.cuda.synchronize` is the only
thing that can see it, and a hardware run would return a green result that means nothing about it.

**The barrier test that a GPU window can settle (queued behind F1 and F2; not before, since it would test code known to be wrong).** A long spin kernel on a side
stream reading a registered slab; the real `shutdown()` with the real `torch.cuda.synchronize`; assert the tiers are freed only after the kernel completes and the slab
stays readable until then. The mutant is "skip the barrier". The slab must be **poisoned with a detectable pattern** so the mutant fails on wrong bytes: reading
freed-but-not-reused memory usually succeeds, and without the poison the mutant would most likely pass. It also exercises the real `cudaHostUnregister` on a real slab,
which the CPU tiers (`device="cpu"`, an empty unregister list) never do.
