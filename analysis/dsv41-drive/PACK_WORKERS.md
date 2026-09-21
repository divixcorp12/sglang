# Packing workers against the bounded single-owner loop (Task 4, the last open item)

Plan item: "compare a packing worker against the bounded single-owner loop". Built and compared on CPU
only; no GPU, no drive contention. Code: branch `t4-packworker` (worker pool, tests, benchmark). Flag:
`SGLANG_DSV41_RAM_MISS_PACK_WORKERS=N` (default 0 = today's behaviour, no thread exists).

**Gate on the flag.** `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` defaults to 0 and is enabled nowhere. It must not be
enabled in any run that produces stage traces until the StageRecord carries the mode (the word can ride
with the schema bump that adds the per-extent submit stamp): `overlap_timeline.py` silently misreads
worker-mode traces (see the WARNING under "What changed for the other safety arguments", which also names
the metrics that stay valid).

## Result

**"Add a packing worker" is the wrong lever on its own.** A worker that copies whole rows does not
shrink the exposed tail, because the last row's copy is still one serial memcpy after its last
completion. Two things move, and they are different knobs:

| what | knob | effect on the copy after the last completion |
|---|---|---|
| one row's copy, exposed after the last read | **byte-range split of each row** across workers (`c` below) | p50 1.7x-2.7x shorter at c=4 (last-row tail), saturating by c=4-8 |
| several rows ready at the same instant (the row spread Task 6 wants) | **row-level parallelism** (`W` workers, `c=1`) | 1.8x-3.5x shorter for 4-8 rows |
| a single row, whole-row worker (`W1/c1`, `W2/c1`, `W4/c1`) | worker count | no change (3 of 3 runs) |

The exposed tail of every request is one row's packing (p50 2.73-2.76 ms in the production traces), so
the split is what touches it: `N` workers, each row cut into `N` chunks (the production knob sets both).
No claim about production speed-up follows from these CPU-only numbers; see "What this cannot say".

## Safety properties, and the mutation that breaks each

Every rule below is enforced by the owner thread; a worker only copies bytes the owner has vetted.
Each mutation was applied to the reader, the named tests were run (taskset 0-63, divix01), and the
reader restored. `M*` is the mutation; all fail the test that states the rule.

| rule | mutation | fails |
|---|---|---|
| `pack_one`'s coverage check (`filled >= needed`) still refuses a row the drives did not fully deliver | check deleted | `test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing` (all modes), `test_a_row_short_of_its_segments_is_never_handed_to_a_worker` |
| the check runs before dispatch: a refused row is never copied | dispatch first, check after | the same two |
| a bank is reused only after its copies are done | bank released at dispatch | killed by guards, not by an ordering assertion: the admit guard (`assert (0 == 1)`, the original test in 15 worker modes and the dedicated test in 3); with it removed the re-arm guard throws (dedicated test); with both removed the reader deadlocks and both tests hit the 120 s hang guard. No run has failed on the ordering assertion, and none can: it runs after the read returns, which this mutant prevents. It is shown correct only on crafted records (`test_the_ordering_assertion_fires_on_a_record_that_violates_it`, which fails when `>` is weakened to `>=`) |
| same, on the finish path | row finished before its job reads done | that test and 4 more |
| `read()` returns only when no copy is running | no quiesce | `test_a_failure_returns_only_when_no_copy_is_still_running` |
| the loop does not end while a row is packing | exit condition ignores it | 22 failures across 7 tests |
| workers never run on cores 64-71 | reserved cores not removed | `test_a_worker_may_not_use_cores_64_to_71` |
| chunks are disjoint and complete | 1-byte gap between chunks / wrong last chunk | the chunk tests and the concurrency test |

`test_exl3_ram_miss_pack_workers.py` re-runs every faulted/traced/host test of the split and thread suites with workers
in three modes, so promises those suites make (short reads, reversed CQEs, poisoned recycled descriptors,
generation wrap, hard errors with both banks in flight, cancellation, pause/stop) are checked on the worker
path without a second copy of each test. Evidence (divix01, taskset 0-63, CUDA_VISIBLE_DEVICES=9, one session per
pytest, load1 24-27), on dsv41 `b022c8ee6b` plus this branch: the new file 362 passed (0 skipped; it has since gained the crafted-record ordering test, 363, which passed on its own), and 362 passed
run as CI runs it, `python3 <file> -f` (the entry point ignores `-f`; the count matching pytest's is what shows the
entry point runs the file); `pytest test/registered/unit/kernels/ -k "ram_miss or lease"` 679 passed, 2 skipped, 0
error or Interrupted lines, collected ids identical (681) with CUDA_VISIBLE_DEVICES empty and 9. The 2 skips are other
files' O_DIRECT-support skips. The flag-off refactor alone is parity-checked on its own commit: 313 passed, 2 skipped
before and after. The mutation runs (each passed + failed == 51 collected, no hang, no error line) and their observed
failure output are in the threading commit message; the bank-reuse mutant is killed by the admit guard's refusal, not
by the ordering assertions of the tests it fails, and that assertion has not been seen to fire end to end.

**Not covered by a test, argued instead.** Each worker executes `_mm_sfence()` after its copy and before
it signals done. glibc uses non-temporal stores for large copies; the sfence the service thread already
executes does not order another core's stores. No test can observe this, so there is no mutation for it.
Also untested: the scope guard that quiesces on an *exception* (nothing in the reader throws there today;
the guard is what makes a future throw safe). A job re-armed while a worker holds it now throws (found by
the bank mutation, which otherwise hung under the test's hang guard).

**Found on the way.** `pack_one`'s coverage check had no test at all (committed separately on `dsv41`,
`test(exl3): pin pack_one's coverage check`, with deletion and one-page-weakening mutations that fail it).
The fake-checkpoint tests leak 6-8 file descriptors each until process exit; the pack-workers file lifts
its own soft limit rather than change every fixture.

## What changed for the other safety arguments

* **Ring empty on return** is kept and extended: no copy in flight on return, on every exit.
* **One request at a time** holds at the request level. Inside a request workers copy concurrently but
  touch no tier state (mutex, slot states, map).
* **Pause / eager use** is acknowledged between requests, when the pool is idle by the rule above.
* **Watchdog** (`busy_since_`) still covers the service thread's request; a worker that never finished
  would leave the owner polling inside it, so the same stuck rule fires.
* **WARNING for trace analysis: worker-mode traces are not comparable with inline traces.** `pack_ns`
  is still the sum of the rows' spans, which now overlap (it can exceed first-start to last-end);
  `pack_start` is now the earliest start, not the first row to finish; and a row's `start` is its first
  chunk's start, after a worker woke. The trace record does not say which mode produced it. Concretely,
  `overlap_timeline.py` reads them as inline stamps: `ready_to_pack_ns` (start minus the last extent's
  reap) becomes the worker wake-up delay, so `rows_queued_behind_the_packer` (threshold 50 us) counts wake
  latency, not a busy packer; `serial = window + pack_total` and `saved_fraction` over-count because
  `pack_total` sums overlapping spans; `hidden_fraction_of_pack` inherits the same. `coverage` and `tail`
  use the union of intervals and the last end, and are still meaningful. Run that script only on
  inline-mode traces, or record the mode next to the trace file.
* **While packing is in flight the owner polls** the completion queue instead of blocking in
  `submit_and_wait` (a worker cannot wake it). Measured extra process CPU per read: 0-3 ms at W<=4,
  about +14 ms at W8/c8 against ~85 ms of setup and kernel copy (run 3; noisy).

## Comparison

`bench_pack_workers.py`: production-size rows (13.3 MB, hidden 4096 x inter 2880), mirrored 1:1, page-cached
files under /dev/shm, 30 repetitions per cell with every mode in every round, per-request stamps from the
reader's own trace record. `tail` = packing left after the last completion. All numbers p50/p90 in ms.
Buffered reads, **not O_DIRECT** (production is O_DIRECT); the bounce slots are 13 MB against a 24.75 MiB
L3 per socket, so most of each row is evicted before its copy, but the warm-source effect is not removed.

Conditions (the box was heavily loaded by other agents' CPU jobs in cores 0-63 throughout; the foreign
service processes nimbus ~90% and reth/op-reth ~40-45% were also running):

| run | load1 start / end | cores used (12 least busy of 0-63) | notes |
|---|---|---|---|
| 1 | 27.7 / 28.7 | 1 6 7 13 20 29 48 50 54 55 59 62 | earlier script version; no burst scenario |
| 2 | 34.9 / 35.6 | 23 25 26 27 28 29 34 49 51 55 58 60 | |
| 3 | 37.0 / 28.4 | 8 26 29 33 42 51 52 53 55 58 59 63 | |

Build: `c++ -std=c++20 -O3`, liburing, kernel 6.12.0-211.51.1.el10_2, Xeon Gold 6154 (2 sockets, cores 0-63
span both NUMA nodes), commit of the branch at run time in the JSON. Raw JSON and logs:
`divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/pack-workers-bench/run{1,2,3}*.json` (not committed).

**One row's copy left exposed (last row of a 4-row request withheld until the others have packed):**

| mode | run 1 | run 2 | run 3 |
|---|---|---|---|
| inline | 3.4 / 5.8 | 6.0 / 9.1 | 5.5 / 7.9 |
| W1 c1 | 3.3 / 3.5 | 6.6 / 11.7 | 3.4 / 5.2 |
| W2 c1 | 3.3 / 4.0 | 6.8 / 9.6 | 3.4 / 6.0 |
| W4 c1 | 3.3 / 3.7 | 6.0 / 7.9 | 3.4 / 7.3 |
| W2 c2 | 2.0 / 2.3 | 4.2 / 6.2 | 1.8 / 3.3 |
| W4 c4 | 1.4 / 2.0 | 3.5 / 4.8 | 2.0 / 2.8 |
| W8 c8 | 1.4 / 2.1 | 3.3 / 5.4 | 1.8 / 2.8 |

**Four rows ready at the same instant (`burst`, all four withheld and released together):**

| mode | run 2 | run 3 |
|---|---|---|
| inline | 26.0 / 35.5 | 23.0 / 28.1 |
| W1 c1 | 27.6 / 40.6 | 13.5 / 15.7 |
| W2 c1 | 17.2 / 23.7 | 7.5 / 10.9 |
| W4 c1 | 14.2 / 18.9 | 6.6 / 10.8 |
| W4 c4 | 13.2 / 17.0 | 6.4 / 8.6 |
| W8 c8 | 12.0 / 17.3 | 6.0 / 8.4 |

Eight rows: inline 41.4 / 43.0, W4 c1 21.2 / 13.6, W8 c8 18.1 / 12.2 (runs 2 / 3).

Run-to-run spread is large (inline's own tail is 3.4, 6.0 and 5.5 ms across the three runs, under load 28-37),
so read ratios within a run, not across runs. The natural-overlap scenario (cached reads a few ms apart) gives
the same picture as `last`: here the read window (8.5 ms per row) is longer than a row's copy, so the packer
is never on the critical path except after the last read.

## Predictions against outcome

| stated before building | outcome |
|---|---|
| a whole-row worker leaves the single-row tail unchanged | **confirmed**, 3 of 3 runs (W1/W2/W4 with c1 vs inline) |
| W=1 may be *worse* than inline under buffered reads (cross-core pull of a warm bounce) | **not observed.** Within 10% of inline in runs 1 and 2; **much better** in run 3 (5.5 -> 3.4 ms one row). That reversal is unexplained; the owner thread is not pinned in this harness, so a noisy core under the owner is the likely cause. Treat W1 vs inline as unresolved, not as a win |
| row-level workers shrink bursts | **confirmed**, 1.8x (run 2) to 3.5x (run 3) at 4 rows |
| splitting a row shrinks its exposed tail, saturating with bandwidth | **confirmed**, 1.7x-2.7x at c=4, little more at c=8 |
| bank reuse and credit accounting unchanged | **confirmed** by test: rows-in-flight, `pending_max`, batches, extents, bytes equal to the inline reader's for the same read |

## What this cannot say

* **Real NVMe timing.** Cached reads complete a few ms apart because of the page-cache copy and the kernel's
  io workers, not because of a drive. Production's read window with mirrors is 6.0 ms at p50 against
  5.6-5.7 ms of packing per request; that regime (a drive-paced overlap, rows sometimes waiting on the packer)
  is not reproduced here.
* **O_DIRECT.** Direct reads leave the bounce cold in DRAM; buffered reads leave part of it warm. The
  direction of the difference for a worker on another core is not known from these runs.
* **Production contention.** N extra runnable threads compete with the scheduler and tokenizer threads, and
  the copy crosses NUMA nodes when workers land on the other socket. Neither is priced here.
* **Small differences.** No difference under ~10% between two modes should be believed at this load.

## The run that would settle it (not run: it reads the production drives)

Same script with `--direct --scenarios natural --work-dir <a directory on the drive under test>`: O_DIRECT
reads of real mirror-split rows, modes `0:0,1:1,4:1,4:4` (drop the others), 60 repetitions, 1/2/4 rows.
(`--direct` on /dev/shm is accepted by tmpfs and behaves as buffered, so it only means something on a real drive.)
Preconditions from the plan's global constraints: coordinate with crypto-c9 (it reads a production drive),
record `/proc/diskstats` before and during (the script stores the device's line at start and end), run under
`taskset -c 0-63` with OMP/MKL capped, cores 64-71 free, state the file age and temperature, and do it when
the box is quiet enough that inline's own tail reproduces between two runs (it did not here). Success test
written before the run: at c=4 the p50 `tail` is at most 0.6x inline's and its p90 is not above inline's, in
two consecutive runs.
