# Handoff: performance review of `ram_thread.h`

Written 2026-10-22. This carries a read-only review of the RAM-miss service thread's host code and a ranked list of
optimization candidates. **Nothing here is implemented, measured or authorized.** Every item is a hypothesis from
reading the source; the file states what the existing tests will police and what a measurement would have to show.

Target: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h` (345 lines). Laptop HEAD at review time:
`d82ae69d03`. Reviews the file as of the commit series ending `d6467ba884`; `git log --oneline -- <path>` first.

## Prompt (paste this to start the next session)

> Review-and-optimize session for the RAM-miss service thread. Read `RAM_THREAD_OPTIMIZATION_NEXT_HANDOFF.md`
> (repo root, untracked) first: it carries six ranked candidates, the tests that will police each one, and the
> measurement that would justify it. Start by re-verifying the file and the tests named there against current source —
> the line numbers are from `d82ae69d03`. Then propose which candidates to take through hx planning. Do **not**
> implement, capture, launch a server, or run on divix01 from this handoff alone; production on divix01 is stopped and
> only its owner restarts it. The cheapest first result is a pause-latency measurement at the FFI (candidate 2), on
> the laptop, before any code change.

## What the file does, in one paragraph

One `RamThread<Tier>` per tier owns `Wire::kNodes` service threads — one per NUMA group — plus one watchdog. Each
service thread spins on `pump_demand(g)`; when a pause is requested it drains its remaining records, publishes
`parked_epoch_[g]`, and waits for the epoch to clear. `pause()`/`resume()` are the ownership handoff under the tier's
`caller_mutex()`. The design is already tight in the ways the earlier audit asked for: the spin is an iteration
budget calibrated once in `start()` (`idle_budget`, `reader_base.h:112`), so the serving loop reads **no clock**;
`run()` and `watch()` are named steps; the watchdog aborts rather than hanging decode. The remaining candidates below
are therefore second-order — handshake latency, cache-line sharing, loop-invariant branches — not redesigns.

## Ranked candidates

| # | Where | Change | Why | Risk | Policed by |
|---|---|---|---|---|---|
| 1 | `:341` `parked_epoch_` | Pad each group's word to its own cache line (`alignas(64)`, or a 64-byte struct in the array) | With `kNodes == 2` both 8-byte atomics land on one line: each service thread writes its own at `:243` while the pauser reads both in `all_parked` (`:174-178`) | None mechanically; the array grows from 8 to 64 bytes per group | no test reads the type; re-run the thread suite |
| 2 | `pause()` `:156` and `park()` `:248` | In `park()`, spin a **short bounded** budget (a few hundred `_mm_pause`) before falling back to the 20 us sleep, the way `run()` already does at `:225-226` | The handshake is quantized twice: the service checks its epoch once per 20 us sleep, the pauser once per 20 us poll (`:156`). Both are `nanosleep` syscalls. This is the eager host-use path (`before_host_use` → `pause()`), so a lower floor is a lower floor on eager latency | A parked thread burning a core for a long pause; must stay bounded and must still sleep when `spin_ns_ >= 0` | `test_exl3_ram_miss_thread.py::test_an_idle_service_sleeps_only_after_its_spin` asserts parked CPU **< 0.1 s per 0.5 s park** for `spin_us=1000` — a bounded spin passes, spinning like `-1` fails |
| 3 | `pause()` `:144,152` | Read the clock every N iterations, not every turn | `clock_gettime` per 20 us poll, including the iteration that succeeds. The watchdog accepts 20 ms granularity; the pause deadline is seconds-scale | Changes the `NON_TRACE_CLOCK_READS` count in the trace-gate test | `test_exl3_ram_miss_stage_trace_causal.py::test_no_clock_read_bypasses_the_trace_gate` asserts that dict **exactly** — update it with the new line and its comment |
| 4 | `park()` `:244` | `stop_.load(std::memory_order_relaxed)` | `run()` already uses relaxed (`:217`); the wait loop's seq_cst is inconsistent. Plain load on x86 either way | None | none |
| 5 | `run()` `:226`, `park()` `:245` | Template the thread on `busy_poll_` and on `spin_ns_ < 0` (both fixed at construction) | Two loop-invariant branches in the innermost loop of a spinning core | Must not disturb `busy_poll`'s core check | `test_a_busy_polling_service_needs_a_core`, `..._refuses_a_physical_core_it_would_share` (`:424`, `:435`) |
| 6 | `pause()` `:144` vs `:160` | Share one deadline across the park wait and `wait_copy_idle_owned` | Today the second wait gets a **fresh** full `timeout_ns`, so `pause()` can take 2x the caller's budget. Python already accounts for that bound (`exl3_ram_miss.py:1598-1621`, `watchdog_wait_s` `3 x timeout`) | Changes what a timed-out pause means when a copy is outstanding, and the caller's contract | `test_expert_stream_ownership.py::test_a_pause_waits_for_an_outstanding_copy`; confirm nothing relies on the second full budget |

## Not worth touching

- **The watchdog** (`:266-275`): wakes 50x/s, does ~2x groups acquire loads plus one `copy_gate()` read per wake. Nothing.
- **`all_parked`'s N loads and `start()`/`stop()`'s futures and joins**: setup and teardown, once per server.
- **The 50 us idle sleep at `:228`** (audit row L12): real, but it only delays the first miss after an idle gap, and
  `SGLANG_DSV41_RAM_MISS_IDLE_SPIN_US` (default 5000, `environ.py:1911`) is the dial. Measure before touching it.

## Doc/code mismatches found while reading (cosmetic, but worth deciding)

- The class comment says a busy-poll thread spins "with no PAUSE" (`:24-25`), but `park()` `:246` executes `_mm_pause()`
  regardless of `busy_poll_`. The test agrees with the code ("busy_poll ... only drops the PAUSE", in the parked case
  too). So the comment is the thing that is wrong — or the PAUSE in `park()` is a deliberate omission not yet made.
- `pause()`'s comment at `:151` explains a line-based constraint from the clock-read test, so item 3 must keep that
  line's identity or update the dict deliberately.

## Measurement before implementation

The audit's instrument already exists and counts exactly these things per thread role: `python/sglang/test/hotpath_shim.py`
(LD_PRELOAD child) reports `sleep`, `futex` and `clock` counts for the `service` and `copy` roles, with
`service_spin_us` as a knob. That is enough to show whether a bounded spin removes the parked sleeps and whether it
adds unscheduled clock reads.

What it does **not** give is a latency. The candidate-2 question is "how long does `pause()` block an eager caller",
so the cheapest honest measurement is a small FFI-level benchmark: park an idle tier, then time `pause()` → `resume()`
at the Python boundary, repeated, against the current build and against a build with a bounded spin in `park()`.
There is no wall-clock number to quote from today's evidence — the trace-gate test counts pause's clock reads, it does
not time them — so produce the before-number rather than assuming the 20 us floor is the observed cost.

Record for any candidate that gets implemented: the laptop test command (narrow selection per
`.claude/rules/divix01-run-protocol.md`: `test/registered/unit/kernels/test_*exl3*.py test/registered/unit/kernels/test_*expert*.py`,
plus the two named files above), the JIT build-key note (`ram_thread.h` is header-only and included by the host JIT
modules, so **any** edit rebuilds every expert-stream host variant — 50-100 s per variant, serialized behind the build
lock), and whether the change is measured or assumed.

## Where this file sits relative to the earlier work

- `docs/superpowers/specs/2026-09-29-hotpath-audit.md` is the parent: §4.4 asked for the spin budget, and rows L12 and
  M8 are the idle sleep and the old clock-based pacing. Candidate 2 is the same family, applied to the pause path
  rather than the idle path.
- `17d0b3ebca` restored the idle spin budget; `f8f1783416` had removed sleeping between polls; `91d4fd4d21` split
  `pause()`/`run()`/`watch()` into named steps and gave `pause()` a `PauseResult`. Re-read those three commits before
  changing the handshake so a reverted decision is not re-litigated by accident.
- If the handshake spin turns out not to be the cost, the next suspects on this path are **tier-side, not in this
  file**: `pause()` calls `quiesce_spec()` (`:141`), `flush_pending()` (`:159`) and `wait_copy_idle_owned()` (`:160`).
  Read `ram_tier.h` for those before concluding the pause latency lives in `ram_thread.h`.

## Constraints that bind the next session

- Production on divix01 is stopped; never start, stop or execute in `cc-expert-prediction/dsv41-direct-prod` or on port
  7867. Only the owner changes its defaults. (`.claude/rules/divix01-run-protocol.md`, `RAM_PREFETCH_L2_HEADS_NEXT_HANDOFF.md`.)
- Code moves by laptop commit/push to `origin` then a private divix01 worktree; no ad-hoc copies. CPU jobs under
  `taskset -c 0-63`; cores 64-71 stay free; a CPU-only job sets `CUDA_VISIBLE_DEVICES=` (empty).
- Planning and execution go through hx, not superpowers planning or plan mode. Never amend or force-push.
- The working tree is dirty with other sessions' files (`RAM_PREFETCH_FIXED_SPLIT_NEXT_HANDOFF.md`,
  `RAM_PREFETCH_L2_HEADS_NEXT_HANDOFF.md`, `analysis/mimo-v8-source-review/`, `analysis/ncm_head/`, `docs/research/`,
  a modified `.claude/rules/divix01-run-protocol.md`); preserve them, stage by name.

## Completion checklist for the next session

- Re-verify the six candidates against the current file; drop any that the intervening commits already fixed.
- Choose the measurement for candidate 2 and produce its before-number before writing the change.
- For each implemented candidate, name the test file that polices it and run exactly that selection; state the JIT
  rebuild cost. Keep a rejected candidate's evidence rather than deleting the reasoning.
- State plainly which candidates are measured wins, which are mechanically safe no-ops in practice, and which remain
  unmeasured. Do not report a serving gain from a laptop microbenchmark.
