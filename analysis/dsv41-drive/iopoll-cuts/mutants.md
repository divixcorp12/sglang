# Read-cut mutants and verification counts, 2026-09-28

Plan `docs/superpowers/plans/2026-09-28-iopoll-read-cuts.md` Task 6 Step 3, re-run after the final-review fix round.
Runner: `analysis/dsv41-drive/iopoll-cuts/mutants.sh`, divix01 only.
- It applies each mutant in the private worktree `wt-cuts-mutant`, runs its detecting tests under `rowimg-disk.lock`, records the result, and reverts with `git checkout --`.
- It takes the disk lock itself; `IOPOLL_CUTS_DIR` sits on the SPCC root, `/mnt/nvme4`.
- The worktree is created before a run and removed after it.

**Protocol deviation.** The first run (at `4738822c9b`) used a copy of the runner that was scp'd to
`divix01:/mnt/nvme1/iopoll-cuts-logs/mutants.sh`. That breaks the run protocol's no-scp rule. It was a throwaway
harness, not code under test: the mutated code was always the pushed commit, checked out in `wt-cuts-mutant`.
The runner is committed here, with its anchors updated for the review-fix head. The second run (at `3684a70d96`)
executed the committed copy from inside the mutant worktree.

## Mutants

| # | Mutant | First run @ `4738822c9b` | Review-fix run @ `3684a70d96` (failing tests) |
|---|---|---|---|
| M1 | `cut_legs`: `gap = false && ...` (skip the gap cut) | killed (2) | **killed**: `test_read_cuts_planner_and_limits`, `test_gap_cuts_at_slab_rows_off_the_page` |
| M2 | `cut_legs`: leg room `cut_bytes + 512` (one block past the device limit) | killed (6) | **killed**: planner, 4 × `test_cut_reads_are_byte_identical_and_within_the_cut`, and the punt check `test_iopoll_punts_uncut_reads_and_never_cut_ones` (1 io-wq worker at legs of 262656 B on the SPCC) |
| M3 | `blocking_wait()` = `wait_mode == Block` (the IOPOLL wait trap restored) | killed (2) | **killed**: `test_uring_reader_driver_contract[False/True]` |
| M4 | `queue_depth()`: `per_read = 1` (credit not scaled by legs) | killed (1) | **killed**: `test_credit_scales_with_the_leg_bound` |
| M5 | `size_legs`: no leg bound with cuts (storage not sized) | killed (2) | **killed**: 4 × `test_cut_reads_are_byte_identical_and_within_the_cut` (logic_error "more legs than open() sized") |
| M6 | `device_limits`: on failure return `{INT64_MAX/2, 0}` (cuts silently off) | killed (1) | **killed**: `test_read_cuts_planner_and_limits` |
| M7 | `process`: retire once no leg is in flight, without every leg Done | killed (1) | **killed**: `test_one_short_cut_leg_resubmits_only_that_leg` |
| M8 | `cut_legs`: a virtually contiguous join counts as a gap (review fix #5 reverted) | — | **killed**: `test_read_cuts_planner_and_limits` |
| M9 | `size_legs`: cuts off gives stride `max_iovecs` (review fix #1 reverted) | — | **killed**: `test_cuts_off_keeps_todays_credit_and_legs` |

**Survivors: none.** Restored green after each run (the worktree clean, detecting files re-run):
`flock rowimg-disk.lock taskset -c 0-63 python -m pytest test_expert_stream_read_cuts.py test_expert_stream_uring_options.py
test/manual/dsv41/test_expert_stream_read_cuts_nvme.py -q -p no:randomly --basetemp=/mnt/nvme2/...`:
- **28 passed** at `4738822c9b`;
- **28 passed** at `3684a70d96`.

## Verification counts (divix01, `PYTHONPATH=$PWD/python`, `sglang.__file__` under the worktree)

| What | Command | Base `966962a5c2` | `4738822c9b` (Task 6) | `3684a70d96` (review fixes) |
|---|---|---|---|---|
| Registered kernels suite | `flock cc-gpu.lock taskset -c 32-63 python -m pytest test/registered/unit/kernels -q -p no:randomly` | 1911 passed, 23 skipped | 1933 passed, 23 skipped | 1933 passed, 23 skipped |
| Reader and uring files | `taskset -c 0-63 python -m pytest test_expert_stream_{read_cuts,reader_golden,fixed_buffers,uring_options,uring_native,uring_integration}.py -q -rs -p no:randomly` | 76 passed, 22 skipped (without read_cuts) | — | 98 passed, 22 skipped |
| NVMe set | `flock rowimg-disk.lock taskset -c 0-63 python -m pytest test/manual/dsv41/test_expert_stream_read_cuts_nvme.py test_expert_stream_{uring_native,uring_integration,read_cuts}.py -q -rs -p no:randomly --basetemp=/mnt/nvme2/nvfp4-work/iopoll-cuts-native` with `IOPOLL_CUTS_DIR=/mnt/nvme4/nvfp4-work/iopoll-cuts-tests` | — | 70 passed, 0 skipped | 70 passed, 0 skipped |

- **The 22 skips.** In the reader/uring set they are the IOPOLL and READV_FIXED capability skips on `/tmp`'s filesystem. On NVMe (`--basetemp`) the same cases run and pass.
- **The punt check's phase numbers.** Measured at `4f6fa71637`, MODE=iopoll on the SPCC, 20 reads × 8 rows, twice each:
  - cut: 0 io-wq workers, SQEs ≤ 262144 B;
  - uncut: 1 worker, SQEs up to 3,560,448 B.
