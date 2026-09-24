# Packing workers against the bounded single-owner loop (Task 4, the last open item)

Plan item: "compare a packing worker against the bounded single-owner loop". Built and compared on CPU
only; no GPU, no drive contention. Code: branch `t4-packworker` (worker pool, tests, benchmark). Flag:
`SGLANG_DSV41_RAM_MISS_PACK_WORKERS=N` (default 0 = today's behaviour, no thread exists).

**Gate on the flag, updated 2026-09-21.** `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` still defaults to 0 (off,
inline) and is enabled nowhere in production; setting it does not change the default. The gate this note used to
state -- "must not be enabled in any run that produces stage traces until the StageRecord carries the mode" -- is
now **closed**: the stage record's `pack_workers`/`pack_split` fields (STAGE_FIELDS schema 5) carry the mode
through to the exported JSONL (`test_the_mode_reaches_the_jsonl_a_trace_writes`) and `overlap_timeline.py`
refuses a worker-mode trace's overlap-derived metrics while keeping the ones that stay valid
(`test_the_analysis_refuses_the_metrics_of_a_trace_a_worker_host_wrote_and_keeps_those_of_an_inline_one`). So
enabling the flag in a traced run is no longer a silent-misread hazard, just something to remember when reading
`overlap_timeline.py`'s output (see the WARNING under "What changed for the other safety arguments" for which
metrics that affects).

**The measured configuration is already what a single env var selects.** `RowReader::set_pack(workers, split)`
(`exl3_ram_miss_host.cpp`) defaults `split` to `workers` when the caller passes 0, and the only entry point wired
into production (`exl3_ram_miss_open` -> `RamTier` -> `RowReader`, via `Dsv41Config.ram_miss_pack_workers` ->
`Exl3RamMissRowBackend`) passes `pack_workers` alone -- there is no `SGLANG_DSV41_RAM_MISS_PACK_SPLIT`. So
`SGLANG_DSV41_RAM_MISS_PACK_WORKERS=4` already gives exactly the measured winning mode, `4:4` (c=4), not `4:1`
(which the registered run above found no better than inline) or any other split. **No code change was needed
to land this**; the wiring already existed and already matches the measured result. Not exposing `pack_split`
independently is deliberate, not an oversight: the registered data gives no reason to pick any ratio other than
`N:N`, and a second env var would only add a way to accidentally select `4:1`. **The default stays 0** -- this
document recommends measuring under production contention (see "What this cannot say") before enabling it, not
changing what ships.

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

## Evidence for "refill submitted I/O before performing bounded packing work"

The Task 4 plan bullet has a second clause distinct from the packing-worker comparison above, and the plan
recorded it as having no evidence of its own. The claim: the reader's main loop (`exl3_ram_miss_host.cpp:642-655`)
calls `refill()` before `pack_one()` on every turn, so a read that credit frees up is submitted before the CPU
spends that turn packing, not queued behind it.

`test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack`
(`test/registered/unit/kernels/test_exl3_ram_miss_split.py`) pins this down directly, in the inline reader (no
packing workers, where "before packing" means before the owner blocks on a memcpy, not merely before a
non-blocking dispatch). Three single-extent rows, credit for two: rows 0 and 1 submit and retire together, then
row 0 packs (150 ms, inline, blocks the owner thread). Row 2 was never submitted -- credit was exhausted by rows
0-1 -- so it can only submit once a slot frees, which happens when row 0's turn ends and control returns to the
loop, **before** row 1 (the next row in packing order) packs. The test asserts exactly that: row 2's extent-submit
timestamp precedes row 1's `pack_start`.

**Mutation, applied and reverted on a private worktree (`nvfp4-work/cc-packrefill-task4`, `e61c505731`), not
committed:** moving the `refill()` call from before `pack_one()` to after it (keeping admit/ready-check/reap in
their original positions) makes the read fail outright. `c.pending` and `has_ready()` both start at 0 for a
freshly admitted batch, so the loop's own termination check (`c.pending == 0 && !ready && ...`) fires on the very
first turn, before anything was ever submitted; the post-loop clean-state check then sees rows still marked
reading and fails the read. The test's `assert result == 1` catches this immediately (confirmed: the mutated
build returns 0 and the test fails at that assertion, not at the finer timing check). Reverted with
`git checkout --` after the run; `pytest -k refill_submits_a_credit_freed` and the full split/thread/pack_workers
suites (539 passed) were re-run clean afterward.

This is real evidence for the clause, not a restatement of the already-cited packing-worker result: it is a new
scenario (credit-gated multi-row submission racing a blocking pack), not the "one upfront submit, then packing"
scenario the timeline gate and `test_a_row_packs_while_another_rows_read_is_still_outstanding` already cover.

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
| W=1 may be *worse* than inline under buffered reads (cross-core pull of a warm bounce) | **not observed.** Within 10% of inline in runs 1 and 2; **much better** in run 3 (5.5 -> 3.4 ms one row). **Resolved 2026-09-22, but not by the explanation offered here.** The owner-pinning scaffold (`read_rows_traced(..., owner_core=)`, `--owner-core`) was built and measured at both row populations. Pinning changed nothing: spread stayed 2-6% whether the owner was pinned or not, at `rows=4/last` and at `rows=1/natural`, and the rare p90 outliers (4.6-4.8 ms) appeared in pinned runs too. W1 beat inline in 1 of 6 runs at rows=1, by a hair. **But the pinning hypothesis was not actually falsified** -- the box sat at load1 3.0-8.6 against the 27.7-37.0 of the original runs, and every new number fell in a tight 2.4-2.6 ms band. The wide spread this row is about never occurred in the *unpinned* baseline either, so there was nothing for pinning to narrow. A null result under conditions where the phenomenon is absent is not a refutation. What does settle it is this table's own data: **inline's tail swung 3.4 / 6.0 / 5.5 ms across the original three runs, and inline has no worker thread to leave unpinned.** Whatever drives that swing hits the thread doing the copy regardless of whether a worker exists, so "the owner is not pinned" cannot be a W1-specific explanation. The run-3 reversal is better read as inline being anomalously slow that run than as W1 being fast. Combined with the production breakdown -- `rows_asked == 1` is 67.9% of served decode demand requests (32,977 of 48,559), where a single worker has no intra-request overlap to exploit by construction -- W1 has no mechanism for the majority case. **The default stays 0.** The heavy-load regime remains untested and would need a genuinely busy window; manufacturing contention was declined, because production runs on this box. |
| row-level workers shrink bursts | **confirmed**, 1.8x (run 2) to 3.5x (run 3) at 4 rows |
| splitting a row shrinks its exposed tail, saturating with bandwidth | **confirmed**, 1.7x-2.7x at c=4, little more at c=8 |
| bank reuse and credit accounting unchanged | **confirmed** by test: rows-in-flight, `pending_max`, batches, extents, bytes equal to the inline reader's for the same read |

## What this cannot say

* **Read completion timing of production.** The registered run (below) used a real NVMe with O_DIRECT, so
  completions are drive-paced, but both mirror roots sit on the same drive; production splits them across
  drives. That degrades the realism of the read completion timing (how far apart the rows finish), not the
  mechanism under test: whether a cold DRAM bounce changes the packing tail is a DRAM-state effect and does not
  depend on which drive the bytes came from. The earlier buffered runs on /dev/shm did not have drive timing at
  all. Production's read window with mirrors is 6.0 ms at p50 against 5.6-5.7 ms of packing per request; the
  read window here is 5.0-5.3 ms per row and ~16-27 ms for 4 rows (one drive serving all of it).
* **O_DIRECT: answered for the registered comparison, see the section below.** No cold-bounce penalty of 10% or
  more was measured for a worker on another core (one-row `1:1` against inline: 2.50 and 2.55 ms against 2.73
  and 2.73). That is a statement about this load and this drive, not about production.
* **Production contention.** N extra runnable threads compete with the scheduler and tokenizer threads, and
  the copy crosses NUMA nodes when workers land on the other socket. Neither is priced here.
* **Small differences.** No difference under ~10% between two modes should be believed at this load.

## The registered O_DIRECT run (run 2026-09-21)

Same script, `--direct --scenarios natural --work-dir /mnt/nvme0/cc-packworkers-scratch`, modes
`0:0,1:1,4:1,4:4`, 60 repetitions, 1/2/4 rows, unchanged. Each mode ran in every round, so drift hits all modes.
Code at `36b512f515` (the `dsv41` tip on `shared`), a fresh detached worktree on divix01, run with
`PYTHONPATH=<that worktree>/python`. Under `taskset -c 0-63`, OMP/MKL/OpenBLAS = 1, cores 64-71 untouched
(`cores_used` in each JSON), GPU hidden, production not running (nothing on :7867, GPU 63 MiB), no GPU lock.
Raw output, driver script and the drive's /proc/diskstats sampled every second:
`pack-workers-odirect/{gate1,gate2,run1,run2}.json`, `*.diskstats.txt`, `run.sh`.

**Drive: `nvme0n1` (Samsung 990 EVO Plus 2 TB, `/mnt/nvme0`, 811 GB free), approved by the lead.** It is a
production-class drive, chosen because no non-production NVMe exists on the box (`nvme1` forbidden, `nvme2` 89%
full, `nvme3`/`/mnt/nvme4` carries a live Ray session). Working set ~1.3 GB (ckpt plus two mirror copies, 32 rows of
13.3 MB each), a fresh directory next to nothing of production's. `--direct` was real: the drive's read counter
rose by 5,385 MiB per gate run and 21,388 MiB per full run, which is the bytes the reps read (60 reps x 7 rows x
12.7 MiB x modes) and not the ~zero a buffered read of a warm file would give.
**File age and temperature.** The files were written by the run itself, under a minute before their first read
and about 20 s before their last (each run builds a new set and deletes it). They were dirty at the start: the
drive shows one 205 MiB write burst (416 write ops, identical in every run) in the first seconds, which is the
setup's writeback flushed by the first direct reads, so the first few repetitions overlap it. Every read then went
to the drive (O_DIRECT bypasses the file's own page-cache pages, which are warm from the write). The bounce buffer is
cold in DRAM by construction.
**Foreign traffic on `nvme0n1`:** none of substance. Whole-device counters over each window equal the
partition's own (writes: 416 to 439 ops = 204.8 to 205.0 MiB in total, all of it the setup burst; 23 extra write
ops in run 2 and 6 in gate 1, at most ~0.2 MiB; reads exceed the partition's by 14 to 23 ops in
~12,000 to ~45,000, with the byte totals equal). The rest of the box was busy: load1 3.1-3.5 at each start, 3.4-4.5 at each end; foreign top
processes java (QuestDB, 239%), nimbus (92%), reth (46%), op-reth (26%).

### Precondition: does inline's own tail reproduce?

Fixed before any data, and applied to p50 and p90 at every row count: two runs "reproduce" if they differ by at most
10% of the smaller. (The registered text gives no number; ~10% is the document's own no-belief threshold.)

| inline `0:0`, tail p50 / p90 ms | rows 1 | rows 2 | rows 4 |
|---|---|---|---|
| gate 1 (inline only) | 2.549 / 2.694 | 2.693 / 2.879 | 5.566 / 8.938 |
| gate 2 (inline only, run straight after) | 2.751 / 2.885 | 2.785 / 2.983 | 5.620 / 8.999 |
| gate difference | 7.9% / 7.1% | 3.4% / 3.6% | 1.0% / 0.7% |
| run 1 (inline in the four-mode run) | 2.732 / 2.876 | 2.745 / 2.922 | **2.758** / 5.884 |
| run 2 (inline in the four-mode run, straight after run 1) | 2.727 / 2.771 | 2.749 / 2.829 | **5.538** / 6.097 |
| run 1 vs run 2 difference | 0.2% / 3.8% | 0.1% / 3.2% | **101% / 3.6%** |

* The gate pair (inline alone, two consecutive runs) **passed** everywhere.
* **The comparison pair did not reproduce inline at 4 rows.** Inline's p50 tail at 4 rows was 2.76 ms in run 1 and
  5.54 ms in run 2; three of the four inline observations at 4 rows (5.57, 5.62, 5.54) agree and one (2.76) does not.
  The 4-row inline tail is bimodal: one row's packing (~2.7 ms) when the last read finishes after the row before it
  has packed, two rows' packing (~5.5 ms) when the last two rows finish within one pack of each other. The p50 sits
  on the boundary between the modes, so a small change in the read gaps flips it (the 4-row inline read window was
  26.7, 18.9, 18.4 and 20.7 ms in the four runs). Its p90 (5.9-9.0 ms) is stable to 3.6% in the two full runs but not
  between the gate and the full runs (8.9-9.0 against 5.9-6.1). At 1 and 2 rows inline reproduces.
* So the precondition the registered text asks for held for the gate and for 1 and 2 rows, and **failed for the
  4-row p50 of the pair the success test is evaluated on**. That means inline's 4-row p50 in this run is a coin
  between ~2.7 and ~5.5 ms, and any ratio to it is uncertain by 2x. The verdict below is computed as registered and
  is also checked against the more favourable-to-inline value.

### Result (4 rows = the "c=4" of the registered test; `4:4` is c=4; tail = `pack_end - last_cqe`, p50 / p90 ms, n=60)

| mode | rows | run 1 | run 2 | vs inline p50, run 1 / run 2 |
|---|---|---|---|---|
| inline `0:0` | 4 | 2.758 / 5.884 | 5.538 / 6.097 | 1 |
| `1:1` | 4 | 4.990 / 6.467 | 7.661 / 8.235 | 1.81x / 1.38x |
| `4:1` | 4 | 2.741 / 2.954 | 2.816 / 3.350 | 0.99x / 0.51x |
| **`4:4` (c=4)** | 4 | **1.119 / 1.233** | **1.082 / 1.279** | **0.41x / 0.20x** |
| inline `0:0` | 2 | 2.745 / 2.922 | 2.749 / 2.829 | 1 |
| `4:4` | 2 | 0.950 / 1.086 | 0.993 / 1.185 | 0.35x / 0.36x |
| inline `0:0` | 1 | 2.732 / 2.876 | 2.727 / 2.771 | 1 |
| `1:1` | 1 | 2.504 / 2.851 | 2.550 / 2.732 | 0.92x / 0.94x |
| `4:1` | 1 | 2.498 / 2.886 | 2.542 / 2.739 | 0.91x / 0.93x |
| `4:4` | 1 | 0.791 / 1.098 | 0.818 / 0.868 | 0.29x / 0.30x |

(Rows 2 for `1:1` and `4:1` are within 3% of inline, not believed.) Whole-process CPU per request p50, 4 rows:
inline 22.0 / 28.5 ms, `4:4` 34.7 / 38.8 ms, `1:1` 33.3 / 39.0 ms (workers spin; the rows are packed in the same time
but the box pays for it).

### Verdict on the pre-registered test

Test, fixed in advance: at c=4 the p50 `tail` is at most 0.6x inline's and its p90 is not above inline's, in two
consecutive runs.

**Passed, in both consecutive runs, as registered:** run 1 p50 1.119 <= 0.6 x 2.758 = 1.655 and p90 1.233 <= 5.884;
run 2 p50 1.082 <= 0.6 x 5.538 = 3.323 and p90 1.279 <= 6.097. It also holds at 1 and 2 rows (0.29x-0.36x).
It does not depend on the unstable inline value: run 1's inline 4-row p50 of 2.758 ms is the *lowest* inline value
seen at 4 rows in any of the four runs, and `4:4` still clears 0.6x against it in both runs; and against the
smallest inline p90 seen (5.884) it is 4.6x under. **Caveat, stated plainly: the registered precondition (inline
reproduces between the two runs) failed for the 4-row p50, so the size of the ratio at 4 rows (0.20x-0.41x) is not
a stable number; what is stable is the sign and a factor of at least 2.4x.** The 1- and 2-row ratios are on inline
values that did reproduce.

What this says and does not say:

* **Splitting each row across four workers (`4:4`) shortens the exposed tail under O_DIRECT reads on a real NVMe**,
  to about 0.8-1.1 ms from 2.7-5.5 ms. This confirms the buffered result (1.7x-2.7x) under the condition it could not
  cover. A cold bounce buffer does not turn a cross-core copy into a loss.
* **No cold-bounce penalty was found for a whole-row worker on another core.** One row: `1:1` 2.50 / 2.55 ms against
  inline 2.73 / 2.73 (0.91x-0.94x, under the 10% the document does not believe, so "no worse", not "better").
* **A single worker is not a win at 4 rows and is worse than inline in both runs** (`1:1` 4.99 and 7.66 ms against 2.76
  and 5.54; 1.4x-1.8x, more than 10% in both). The likely reason, not tested here, is that one serial worker packs 4
  rows back to back while the owner has stopped packing. `4:1` (four workers, no split) is at or below inline at 4 rows.
* What the run cannot say still stands: production's two-drive completion timing, and production contention (the
  packing workers here ran on idle cores; the scheduler/tokenizer threads were not competing).

Deviations from the brief, named: (1) the divix01 `dsv41` worktrees were stale, so a new detached worktree at
`36b512f515` was made (`nvfp4-work/cc-packworkers-odirect/wt`); (2) the reproduction gate was defined as at most 10% of
the smaller on p50 and p90 at every row count, because the registered text gave no number; (3) the driver `run.sh`
also samples `nvme0n1` every second, which the script alone does not. Cleanup: the script's `rmtree` ran in all four
runs (scratch directory was empty afterwards and then removed; `df` used bytes on `/mnt/nvme0` are identical before
the first and after the last run, 1,187,494,809,600).

## The split-drive rerun (2026-09-21), mirror roots on separate drives

The run above put both mirror roots under one `--work-dir` (`/mnt/nvme0`), so the two mirrors' read-completion
timing was less realistic than production's, which splits them across drives. `bench_pack_workers.py` gained a
`--work-dir2` argument (mirror 0 stays under `--work-dir`, mirror 1 moves under `--work-dir2`; omitting it keeps
the old one-drive behaviour, so every existing invocation is unaffected) and reran the **same** pre-registered
test, unchanged, with mirror 0 on `/mnt/nvme0` and mirror 1 on `/mnt/nvme4` (`nvme3n1p1`), the same two roots
production splits its mirrors across.

**Box condition, checked before the run:** load1 4.5 at start / 7.5 (run 1) and 6.5 / 6.2 (run 2) at end, similar
to the single-drive run's 3.1-4.5; foreign top processes were the same standing services (nimbus, reth, op-reth,
a QuestDB java process) and no other job held `cc-gpu.lock` (63 MiB used, nothing on :7867 -- this item is CPU
and drive only and never touched the GPU). `--direct` was real on both drives: `nvme0n1p1` and `nvme3n1p1` each
show ~10.7 GiB of additional sector reads per run (10696.0/10693.2 MiB run 1, 10696.0/10692.6 MiB run 2), matching
each other to within 0.03% -- the StaticSplitPolicy((1.0, 1.0)) split each row's bytes evenly across the two
drives, as production does, and both drives actually served their half.

Code: the two changed files copied onto a fresh worktree at `e61c505731` (`nvfp4-work/cc-packrefill-task4`,
removed after the run). Same script, same modes (`0:0,1:1,4:1,4:4`), same 60 reps, rows 1/2/4, natural scenario,
`taskset -c 0-63`, OMP/MKL/OpenBLAS = 1, cores 64-71 untouched. Raw output and `run.sh`:
`pack-workers-odirect-split/{run1,run2}.json`, `*.diskstats.log`.

### Result

| mode | rows | run 1 | run 2 | vs inline p50, run 1 / run 2 |
|---|---|---|---|---|
| inline `0:0` | 4 | 5.487 / 5.598 | 5.502 / 5.549 | 1 |
| `1:1` | 4 | 7.573 / 7.878 | 7.503 / 7.856 | 1.38x / 1.36x |
| `4:1` | 4 | 2.853 / 3.004 | 2.848 / 2.966 | 0.52x / 0.52x |
| **`4:4` (c=4)** | 4 | **1.343 / 1.685** | **1.182 / 1.437** | **0.24x / 0.21x** |
| inline `0:0` | 2 | 2.718 / 2.773 | 2.739 / 2.783 | 1 |
| `4:4` | 2 | 0.908 / 1.439 | 0.895 / 0.955 | 0.33x / 0.33x |
| inline `0:0` | 1 | 2.721 / 2.885 | 2.717 / 2.864 | 1 |
| `4:4` | 1 | 0.787 / 0.879 | 0.768 / 0.904 | 0.29x / 0.27x |

(p50 / p90 ms, n=60.) **Unlike the single-drive run, inline's own 4-row p50 reproduces here**: 5.487 ms and
5.502 ms, 0.27% apart -- well inside the document's 10% no-belief threshold, at every row count. The single-drive
run's bimodal straddle (one row's packing vs two rows') did not reappear; splitting the mirrors across drives
gave the two rows a wider, more consistent completion gap, so the 4-row read consistently lands in the
two-rows-packed regime. This removes the single-drive run's own caveat about its precondition failing at 4 rows.

**Verdict on the pre-registered test, unchanged and unloosened: at c=4, p50 `tail` at most 0.6x inline's and p90
not above inline's, in two consecutive runs.** Run 1: 1.343 <= 0.6 x 5.487 = 3.292 and 1.685 <= 5.598. Run 2:
1.182 <= 0.6 x 5.502 = 3.301 and 1.437 <= 5.549. **PASSED, in both runs, with more margin than the single-drive
run and no reproducibility caveat this time.**

Whole-process CPU per request p50, 4 rows: inline 21.5 / 22.2 ms, `4:4` 36.1 / 36.5 ms (runs 1 / 2) -- about
1.65x-1.68x, slightly higher than the single-drive run's 1.4x-1.6x but the same order of magnitude and the same
mechanism (the workers spin).

**What this closes and what it still does not say.** It closes the single-drive run's "both mirror roots sat on
one drive" caveat: the mechanism (a cold DRAM bounce shrinks under a split copy) now holds under production's own
drive topology, not just a plausibility argument that the source drive doesn't matter. It does not cover
production contention (the packing workers here ran on otherwise-idle cores among `taskset -c 0-63`; the
scheduler and tokenizer threads were not competing for those cores), and it does not touch the GPU or H2D path.

## Per-piece packing on divix01: wake-up, copy bandwidth, and a spinning pool (2026-09-24)

Under piece streaming a pack job is one piece: 1.66 MB, cut into 8 chunks of about 208 KB. The stage trace stamps
only a row's earliest chunk start, so it could not tell the slowest worker's wake from the copy.
`analysis/dsv41-drive/pack-pool-bench/pack_pool_bench.cpp` stamps every chunk (`ChunkStamp`). It drives the real
`PackPool` the way a read does: 8 pieces 250 µs apart, then a 30 ms gap. It ran under the server's CPU set
`0-7,16-17,36-53`, with the destination on node 0.

| µs from post | Old pool (parked, unpinned) | Pinned, parked | Pinned, spinning |
|---|---:|---:|---:|
| Slowest worker starts, first piece of a read | 106 | 95 | 8 |
| Pack tail, first piece of a read | 173 | 163 | 120 |
| Slowest worker starts, later pieces | 14 | 13 | 8 |
| Pack tail, later pieces | 126 | 110 | 117 |
| Per-chunk rate | 2.4 GB/s | 2.4 GB/s | 2.0 GB/s |

**Findings.**
- The C6 wake the handoff predicted is real, but only after the gap between reads. That is
  the first piece of a read, about 100 µs.
- Later pieces are bound by copy bandwidth. With the source and destination both on node 0,
  cold copies plateau at 13.7 GB/s from four threads on. One core alone does 4.9 GB/s.
- So 2–2.5 GB/s per worker is that plateau shared eight ways. It is the socket's DRAM, with
  4 RDIMMs on a 6-channel CPU, not a pool defect.
- Copying into node-1 rows from node-0 cores plateaus at 7.8 GB/s. From node-1 cores it
  reaches 24.6 GB/s.

**Serving, 100 GiB tier, untraced smokes.** Figures are pack tail p50 (last piece landing → last chunk end) and
multi-row demands whose reads stalled more than 10 ms:

| Arm | Pack tail p50 | Stalled multi-row demands |
|---|---:|---:|
| Base `9183637f51`, two runs | 212 / 236 µs | 0 / 3 of ~2,000 |
| Spinning, service thread kept off the workers' CPUs | 194 µs | **303 of 1,982** |
| Spinning, service thread unrestricted | 196 µs | 1 of 1,975 |
| Parked and pinned, service thread kept off (shipped, `92e588b20e`) | 185 / 184 µs | 3 / 0 |

The stalls were SPCC-mirror sub-reads completing 10–25 ms late while the Samsung mirror's were on time. Only the
combination produced them, most likely by crowding the CPUs that take the SPCC's completion interrupts (36-71)
with the busy-polling service thread. Spinning was removed; pinning and the service thread's exclusion stay.
Details and the next options are in `DSV41_REFERENCE.md` §24.8.
