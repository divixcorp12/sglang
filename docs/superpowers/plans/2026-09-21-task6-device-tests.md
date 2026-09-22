# Task 6 V1: the device-chain tests (T3, T4, T5-T11)

**Status:** not started. Written 2026-09-21 after V1 landed at `2916c544e4`.

## Why this exists

Task 6 V1 is merged, off by default, and its host path is tested. Its **device chain is
not**. The `.cuh` compiles and the Task 5 lease kernels still pass, but no test drives
`post -> W1 -> C1 -> A1 -> W2 -> C2 -> A2 -> F`. Every claim about that chain currently
rests on code reading.

The checklist (`task6-v1-checklist.md` §5) specifies eleven tests. Four exist:

| written | missing (this plan) |
|---|---|
| T1, T1b, T4b, T4c | **T3, T4** (host only), **T5-T11** (GPU) |

T2 has no falsifying mutant by construction (§7 O3) and is folded into T1 as a labelled
observation. It is not in scope here and must not be "fixed".

## The one rule that matters

**A test without a demonstrated killing mutant does not count as written.**

This is not a style preference. On 2026-09-21 four two-phase tests passed on their first
execution and told us almost nothing, because they had never been seen to fail. When their
mutants were finally run, they killed — but that was luck, not evidence. Separately, a
warm-up loop in `benchmarks/dsv41_baseline` could not fail at all and nobody noticed until
the loop was read line by line (`TESTS_THAT_CANNOT_FAIL.md`, pattern C).

So for every test below:

1. Write the test. Run it. It must pass.
2. Apply **the mutant named in the checklist row**, not one you invented because it was
   easier. Run it. It must fail, and fail for the stated reason.
3. Revert the mutant. Re-run. It must pass again. **A mutation result is only meaningful
   next to the restored baseline.**
4. Record all three outcomes, with the mutant's actual diff, in your report.

If a mutant leaves the test green, **say so and stop**. That is a real finding about the
test, and it is worth more than a green tick. Do not quietly strengthen the mutant until
it kills; if the checklist's mutant does not kill, the test is not testing what the
checklist claims.

Where a row names **two** mutants (T3, T5), both are required.

## Work split

Three agents, disjoint files, no shared edits.

### Agent A - failure and release semantics
- **T3** (host) - a hit lane's slot is never its own request's victim. Two mutants, both
  required; mutant (a) may legitimately stay green through recency, and finding that out
  *is* the result.
- **T4** (host) - a failed request voids the hit leases and releases no slot.
- **T5** (GPU) - the partial terminal mask names only unacknowledged lanes. Two mutants.
- Files: `test/registered/unit/kernels/test_exl3_ram_miss_two_phase_victim.py` (T3, T4),
  `test/manual/dsv41/test_exl3_two_phase_failure_cuda.py` (T5)
- Branch: `t6t-failure`

### Agent B - timing and degenerate shapes
- **T6** (GPU) - `keep` has exactly one writer.
- **T7** (GPU) - one request deadline, not one per stage.
- **T10** (GPU) - an all-miss request does not pay the read wait twice.
- **T11** (GPU) - the batched path survives as stage 1 covering every lane.
- File: `test/manual/dsv41/test_exl3_two_phase_timing_cuda.py`
- Branch: `t6t-timing`

### Agent C - graph topology and parity
- **T8** (GPU) - the captured graph is the linear chain D7 builds, via `cudaGraphGetEdges`.
- **T9** (GPU) - output parity against **M1**, the Task 5 lease-mode batched arm, eager and
  graph, fixed routes and seeds.
- File: `test/manual/dsv41/test_exl3_two_phase_parity_cuda.py`
- Branch: `t6t-parity`

T9 is the hardest and the most valuable: its mutant (swap two lanes' destination slots in
stage 1's compaction) is the failure mode compaction introduces, and it is the one the
implementation already hit once by accident.

## Constraints

**Copy the working recipe; do not derive one.** Every time an invocation was built from
first principles in this campaign instead of copied from disk, it cost a cycle. Five times.

- GPU harness to copy: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py`. It shows the
  `pytestmark` skip, the hand-driven lease-block writes, and the real-service end-to-end
  shape. Do not invent a new harness.
- Host harness to copy: `test/registered/unit/kernels/test_exl3_ram_miss_two_phase.py`
  (T1/T1b/T4b/T4c) and `sglang.test.dsv41_ram_miss_fixtures.ram_miss_setup`.

**Two-phase is off by default.** Tests must enable
`SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE` and lease mode; two-phase is refused without lease
mode. `SGLANG_DSV41_RAM_MISS_HIT_POLL_BOUND` defaults to 64 and is unmeasured (§7 O4) - if a
test depends on its value, set it explicitly rather than relying on the default.

**divix01 protocol** (`.claude/rules/divix01-run-protocol.md`):
- Code is written on the laptop, committed, pushed to `shared`, pulled on divix01. Never
  rsync/scp a tree. Push **only** to `shared`; `origin` is the upstream sgl-project repo.
- Run from a git-fetched worktree with `PYTHONPATH=$PWD/python`. Print `sglang.__file__` and
  read it: without PYTHONPATH the venv imports from `main-port-probe-7bc4eb` and reports
  green on unrelated code.
- Every CPU job under `taskset -c 0-63`, threads capped. **Cores 64-71 stay free** - core 71
  is production's doorbell spin core.
- All GPU work through `analysis/dsv41-phase3b/gpu-run.sh`, which takes `cc-gpu.lock`.
  Never a bare command. **The lock is exclusive and three agents share it** - keep each hold
  short (these are kernel tests, ~1 min, not server launches), and expect to queue.
- Mutants go in a **private** worktree (`git worktree add --detach
  /data/models/slang/nvfp4-work/wt-<name> <commit>`), never in a shared one, and are never
  committed. Remove the worktree when done.

**Do not** touch another agent's file, the plan files, `CLAUDE.md`, or anything under
`.omc/`. Do not merge another agent's branch. The lead integrates.

## What to report

What you wrote, and for each test: the mutant's diff, whether it killed, and the restored
baseline. Name anything the checklist specified that you could not do, and why. An honestly
incomplete report beats a tidy one - it will be audited, and both prior agents' reports were
checked line by line against the tree.

If a checklist row turns out to be wrong about the code (this has already happened twice -
D7 cited a symbol that does not exist, and D2's instruction contradicted §6), report the
discrepancy rather than implementing around it silently.

## Acceptance

- T3, T4, T5-T11 written, each passing, each with its checklist mutant shown to kill and the
  baseline restored.
- Full registered suite still green on divix01. **The command, recorded 2026-09-22 because the
  original number was not reproducible without it:**

  ```
  PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
    /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly
  ```

  Do **not** point pytest at `test/registered` or `test/registered/unit`: both sweep in directories
  (`xpu/`, `layers/moe/`, others) that fail collection on this box with `AttributeError: module
  'pyarrow'` under pyarrow 25.0.1. That breakage is environmental and pre-existing -- it reproduces
  identically at `2916c544e4` with none of this work present -- but it buries the signal in
  hundreds of errors. Check the pytest exit status, not a pipeline's: `... | tail -2` reports
  `tail`'s status, which is 0 even when pytest exits 2.

  Measured: baseline `2916c544e4` **1133 passed**, merged `8c0613cc56` **1135 passed**, both exit 0.
  The +2 is T3 and T4, the only new tests under `test/registered/`; the other three files are
  `test/manual/` and are not collected. The earlier "724 passed / 409 skipped" figure was this same
  target with 409 GPU tests skipping at the time.
- No change to production behaviour: two-phase stays **off by default**.
