# Mutation results for the host-facing tests, base 1 (2026-09-21)

`TESTS_THAT_CANNOT_FAIL.md` argued from source that some tests cannot fail. This document is the same
question answered by running: apply one small change to production code, run the tests, and see whether
anything fails. Nothing here was committed to `exl3_ram_miss_host.cpp`; every mutant was applied to a copy
in a `git archive` export on divix01 and reverted from a pristine base before the next.

## Base, method and what is not yet in this document

- **Base 1 = `a01f9347d6`.** `exl3_ram_miss_host.cpp` md5 `8d7ca11951d2ec49b984e1340f30d2df`. **This base does
  NOT contain Task 5 step 2 (`6ea3b6a4f9`)**: that commit is not an ancestor of `a01f9347d6`, and it changes
  `take_slot_locked`, `assign`, `release` and `serve` and adds a tensor argument to `exl3_ram_miss_open`.
  Everything below is a statement about the tree before it.
- **Base 2 = `457e44e036`** (contains `6ea3b6a4f9`; host.cpp md5 `99cca0912c8bd837746b88fac50c8521`) is exported
  and every mutant patches it text-exactly once, **but no base-2 result exists yet.** A first attempt to run it
  was launched twice by mistake and its results are discarded as contaminated (see "What went wrong"). It has
  to be rerun in a single pass, and the question it answers (does the publish gate's coverage differ after
  Task 5 step 2, given that step 2 touched `serve`?) is open.
- Tests: 14 files, the host-facing set (`test_exl3_ram_miss_{split,thread,tier,advisory,attach_lanes,wrap,
  device_args,stage_trace,stage_trace_causal,stage_trace_lanes,trace_export}.py`, `test_exl3_lease_block.py`,
  `test_exl3_ram_miss_service.py`, `test_exl3_ram_miss_tables.py`). Baseline: **297 passed, 1 skipped** (the
  `kLease*` agreement stub). Files-specific runs for the verifier (`test_exl3_verify_expert_mirror.py`, 37
  tests), the layout (`test_exl3_expert_layout.py`, 8) and the uring reader (`test_uring_file_reader.py`,
  `test_exl3_mirror_row_source.py`, `test_exl3_row_reader.py`; 98 tests).
- Run on divix01 under `taskset -c 0-63`, `OMP_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=` (no GPU), a private
  `SGLANG_JIT_CACHE_DIR`, the box loaded by other work throughout. A mutant counts as KILLED only if a
  pytest FAILED/ERROR line names a test; SURVIVED means the whole run passed with the same counts as the
  baseline. I did not read every failure message, so I cannot say no kill was timing-driven; kills are
  pytest FAILED lines naming a test.
- A control was run first (H02, below) to show the machinery kills.

## Survivors first (6 of 22)

Each carries the classification asked for: which test was supposed to catch it, and whether that test is
**unfailable in general** (nothing it could be given would make it fail on this defect) or **aimed elsewhere**
(it can fail, and does on other defects, but its fixture or assertion does not reach this one).

### H01: publish gate widened (the edit Task 6 V2 intends) `SURVIVED`

`exl3_ram_miss_host.cpp` `serve()`: `if (ok || (cancelled && i < packed.size() && packed[i] != 0))` ->
`if (ok || (i < packed.size() && packed[i] != 0))`. That is exactly "keep rows that packed whole when the
demand failed" (Task 6 V2, section 7). 297 passed, 1 skipped.

- **Supposed to catch it:** `test_exl3_ram_miss_thread.py::test_a_failed_read_publishes_none_of_the_rows_it_had_already_packed`.
- **Classification: aimed elsewhere, not unfailable in general.** It injects `fail_reads`, which sets
  `ok = false` *instead of* calling `reader_.read(...)`, so no row ever packs, `packed` is empty, and the
  widened gate has nothing to publish. But the test is not inert: it **fails under H03** (unpublished slots
  no longer released, one of the nine kills), because that defect acts on a failed request whether or not
  anything packed. Whether it also fails under H19 (`if (true)`, publish everything on any failure) is my
  prediction and has NOT been run. What it cannot reach is "a failure AFTER some rows packed",
  and nothing in the suite can: `RamTier::inject` offers `fail_reads`, delay and abandon, none of which is
  "some rows packed, then a failure". The reviewer's alternative, truncating the source file inside expert
  2's superset so rows 0 and 1 pack and row 2 fails, was not tried.
- **What it means for Task 6:** the edit V2 intends to make deliberately would leave the suite green. Someone
  could implement early publication, get it subtly wrong in the other direction, and see nothing.
  **Established on base 1 only.** Whether it holds after `6ea3b6a4f9` is the open base-2 question.

### H15: a resubmitted extent overwrites its first submit stamp `SURVIVED`

`extent_submit[slot]` is set unconditionally and `extent_attempts` incremented when it was already set,
instead of keeping the first stamp. 297 passed.

- **Supposed to catch it:** `test_exl3_ram_miss_stage_trace_causal.py::test_a_resubmitted_extent_counts_an_attempt_and_keeps_its_first_submit`.
- **Classification: half real, half unfailable for this defect.** The fixture is right (it asserts
  `retried_bytes > 0`, so the fault demonstrably fired) and the "counts an attempt" half is asserted
  (`sum(attempts) >= 1`; H15 preserves that count). The "keeps its first submit" half is not asserted at
  all: the only chain assertion is `admit <= submit <= cqe <= start`, which a later resubmit stamp also
  satisfies, and nothing compares the extent's submit to the request's. Fix: one comparison (the fault-free
  test at `causal.py:90` already compares them).
- **What it does and does not touch:** the FIFO analysis (`PER_ROW_PRECHECK_REVIEW` R3) used the completion
  stamp `extent_cqe_ns[].cqe` on schema-2 traces, which carry no per-extent submit at all, and none of the
  8 traces under `task1-results` contains a single retried byte (`byte_split.retried == 0` in all 20,800
  requests of each; `attempts == 0` on all 19,310 extents of the one schema-4 file). So this survivor
  changes no analysed stamp. It leaves per-extent submit-to-completion latency under retries unpinned.

### H18: a failed thread start leaves the tier marked threaded `SURVIVED`

`RamThread::start()`: delete `tier_->set_threaded(false)` before the throw on a pin failure. 297 passed.

- **Supposed to catch it:** `test_a_reserved_or_unusable_core_is_refused` (the second case, core 1000).
- **Classification: aimed elsewhere.** Its assertion `assert not host.threaded` reads a Python attribute set
  to `False` in `__init__` and to `True` only after `start_thread` returns, and the call raises, so it is
  `False` whatever the C++ does. The C++ state IS observable: `enable_trace` throws "enable the stage trace
  before the service thread starts" when `threaded_` is set, so calling `host.enable_trace()` after the
  failed start would fail under the mutant. Untried; stated from source.

### P01 and P02: the verifier reads buffered and reports O_DIRECT `SURVIVED` (both)

P01: `verify_expert_mirror.py` mirror sources built with `direct=False` (was `direct=direct`). P02: source
rows read with `direct=False`. Each is run separately; each leaves all **37** verifier tests passing.

- **Supposed to catch them:** `test_a_complete_verification_is_exit_zero` and the verifier suite.
- **Classification: unfailable in general against this class of defect.** No verifier test observes the read
  mode at all: every fixture passes `direct=False`, and the one `direct=True` run asserts the exit code and
  that the string `"O_DIRECT"` appears, which is printed from the flag. It is not aimed elsewhere; there is
  nothing aimed at it. The verifier's byte tests remain sound for content.
- Production threads `direct` correctly (read and traced in `TESTS_THAT_CANNOT_FAIL.md` finding 3), so this
  is a survivor about tests over correct code, on the instrument that certifies bytes.

### P03: the layout ignores its prefix `SURVIVED`

`exl3_expert_layout.py`: the pattern's `re.escape(prefix)` becomes the literal `layers`. All **8** layout
tests pass.

- **Supposed to catch it:** `test_draft_prefix_selects_mtp_experts`.
- **Classification: unfailable in general given its fixture, fixable by the fixture.** The mtp tensors are
  copies at the same `(0, 0)` key, so `list(layout.records) == [(0, 0)]` holds either way. With mtp tensors
  at distinct expert ids or distinct content plus an offset assertion, it would fail. Only a manual test on
  the real checkpoint (`test/manual/dsv41/test_exl3_checkpoint_layout.py`, shapes only) would catch it, and
  it was not run.

## Killed (16 of 22), one line each

| id | mutation | killed by |
|---|---|---|
| H02 (control) | gate `false && ...`: a cancelled advisory keeps none of its rows | 2: `test_a_cancelled_advisory_keeps_the_rows_that_completed...`, `test_rows_kept_from_a_cancelled_advisory_follow_the_usual_eviction_rules` |
| H03 | unpublished slots not released | 9, including `test_a_failed_read_publishes_none_of_the_rows_it_had_already_packed` |
| H04 | demand lap resume without `skip_zero` | 1: `test_a_lap_that_would_resume_at_sequence_zero_skips_it[demand]` |
| H05 | demand advance without `skip_zero` | 4 (`test_exl3_ram_miss_wrap.py`) |
| H06 | advisory lap resume without `skip_zero` | 1: `...skips_it[advisory]` |
| H07 | advisory advance without `skip_zero` | 2 (`...at_the_wrap[advisory-*]`) |
| H08 | demand lap resumes one record early | 2 |
| H09 | advisory lap resumes one record early | 1 (the same single `[advisory]` test) |
| H10 | lap overrun counted as 1, not the skipped count | 1: `test_a_lapped_demand_ring_counts_every_skipped_record` |
| H11 | `take_slot_locked` evicts hot rows | 4 |
| H12 | eviction does not unmap the victim (D11) | 4 |
| H13 | eviction leaves `expert_slot` pointing at the reused slot | 8 |
| H14 | `pump_demand` tail skipped for an unreadable record | 1: `test_a_record_whose_seq_does_not_match_is_an_overrun` |
| H16 | every file attributed to drive 0 | 1: `test_exl3_ram_miss_split.py::test_mirrored_reads_are_accounted_per_drive` |
| H17 | per-drive bytes all on drive 0 (two sites) | the same 1 |
| U01 | the reader never opens `O_DIRECT` | 1, **incidentally**: `test_uring_file_reader.py::test_failed_read_raises_and_reader_stays_usable` |

Things these kills say, beyond "killed":
- **The four `skip_zero` sites** (lap resume and advance, demand and advisory) are each killed. The two
  lap-resume sites are each guarded by exactly one test (the `[demand]` and `[advisory]` parametrization of
  one function); H09 shows the advisory lap-resume margin is a single test. Thin, but it fires.
- **`pump_demand`'s tail** is guarded by one test.
- **`take_slot_locked`'s three protections (hot, unmap, `expert_slot`) are each guarded by 4-8 tests.**
- **U01 is the cleanest result in the run.** Direct I/O is enforced by exactly one test, and only because
  that test passes an unaligned destination, which `O_DIRECT` rejects and a buffered read accepts. No test of
  an aligned read can tell direct from buffered. Combined with P01/P02, the direct-I/O path is guarded by an
  accident of alignment, and the verifier's `reads: O_DIRECT` line carries no information.
- **H16/H17 refute a claim in `TESTS_THAT_CANNOT_FAIL.md`.** The single-drive test I flagged still cannot
  see a per-drive attribution bug, but another test does. The sweep corrected accordingly.

## How the sweep's source arguments held up

Of the sweep findings I could map to a mutant I was able to run (no GPU): **5 of 6 findings were survivors
as the sweep said** (the gate; the verifier, whose two sites are two mutants; the layout prefix; the resubmit
stamp; the threaded flag), and **1 was refuted** (per-drive attribution, two mutants, guarded in another
file). Counted by mutant that is 6 survivors and 2 kills of the 8 mapped. So a sweep claim "this test cannot
fail" is a test-level statement; the stronger suite-level statement "nothing fails" held for five of six.
Against the running tally of 21 proven-unable-to-fail in roughly 1,140 swept, that is a check on the 21's
precision (about 5 of 6 where checkable), not a new count. Two sweep findings could not be tested (the
`_ticket_matches` generation gate and the insert-on-miss victim; both need a GPU) and remain source
arguments.

## What went wrong, and what is not established

- **Double launch, base 2.** I started the base-2 queue twice (a shell quoting error in the first launch
  masked that it had already begun). Two `mutate2.py` chains patched the same tree at once, so every base-2
  result from that run is untrustworthy and none is reported; the file is kept as
  `results2.CONTAMINATED-double-launch.jsonl` on divix01. The processes were killed by PID, the tree recopied
  from the pristine export, and the trees' md5s checked. Base 1's queue was a single launch with unique ids
  and is unaffected.
- **Base 2 has not been run.** Whether H01 and the other gate mutants survive after `6ea3b6a4f9` is open.
  If the result differs between bases, that is a finding about Task 5's landing changing the coverage of code
  Task 6 intends to change.
- **H19 (`if (true)`) and H20 (`if (ok || cancelled)`), the other directions of the same gate,** are defined
  and patch cleanly but have not been run on either base. The prediction that
  `test_a_failed_read_publishes_none...` kills H19 is a prediction until they run.
- **Time-sensitive tests** ran on a box loaded to about 35 by other work. I did not re-run any mutant and
  did not read every failure message, so a load-induced kill cannot be excluded for the sparse ones (H04,
  H06, H09, H10, H14, H16, H17, U01: one test each). No survivor can be a load artifact: a survivor is a
  pass.
- Nothing here changes any code. The mutants are in `/data/models/slang/nvfp4-work/t2-mutants` on divix01,
  outside every worktree.
