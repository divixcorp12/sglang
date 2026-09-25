# Running code on divix01

Code is written on the laptop, committed, pushed to GitHub, and pulled on
divix01. Never copy a working tree to divix01 by other means.

There is one remote, `origin` = `git@github.com:divixcorp12/sglang.git`, and one
main branch, `master` (`upstream` is sgl-project, for syncing only). The old divix01
bare repo (`remotes/sglang-nvfp4.git`, remote `shared`) is retired: do not push to it.

```bash
# laptop, in the worktree for the branch
git push origin <branch>

# divix01: a private worktree of the divix01 clone, at the pushed commit
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-<name> origin/<branch> \
  && git -C /data/models/slang/nvfp4-work/wt-<name> log -1 --oneline'
```

Then run in that worktree, with `PYTHONPATH=$PWD/python`. Never run tests in the
production checkout (`cc-expert-prediction/dsv41-direct-prod`); it tracks `origin/master`.

## No ad-hoc copies

Do not `rsync`, `scp`, or `git archive` a tree into a scratch directory to run
it. A hand-picked copy omits directories the tests import from, and the result
is a red test that looks like a regression:

- `test_exl3_ram_miss_service.py` imports `scripts/dsv41/tier_sim.py`.
- `test_exl3_ram_miss_pack_workers.py` imports `analysis/dsv41-drive/overlap_timeline.py`.

Both were reported as failures on 2026-09-21 by a copy holding only `python/`
and `test/`. Neither was a real failure. A pulled worktree has every path by
construction, so the failure mode does not exist.

## The interpreter trap

`/data/models/slang/.venv/bin/python` imports sglang from
`main-port-probe-7bc4eb` unless `PYTHONPATH` points at the tree under test.
Without it a run exercises unrelated code and reports green. Print
`sglang.__file__` and read it before trusting any result.

## A pipeline reports the last command's status, not pytest's

`... | tail -2` exits with `tail`'s status. `tail` succeeds at printing the
output of a suite that failed, so the pipeline exits 0 while pytest exited 2,
and a background runner reports "completed (exit code 0)" for a red suite.

```bash
# wrong: always 0
pytest test/... -q 2>&1 | tail -2

# right: read pytest's own status
pytest test/... -q 2>&1 | tail -2; echo "EXIT=${PIPESTATUS[0]}"
```

On 2026-09-22 a run of the registered suite reported exit 0 while pytest had
exited 2 with 1401 collection errors. Piping to `tail` is the normal way to keep
a long run's output readable, so this is not a rare shape -- assume any suite
result that came through a pipe is unverified until its `PIPESTATUS` is read.
The same applies to `| grep`, which exits non-zero when it matches nothing.

## Point the registered suite at `unit/kernels`, not the whole tree

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly
```

`test/registered` and `test/registered/unit` sweep in directories (`xpu/`,
`layers/moe/`, others) that fail collection on this box with `AttributeError:
module 'pyarrow'` under pyarrow 25.0.1. The breakage is environmental and
pre-existing -- it reproduces at any commit, with none of your work present --
but pytest aborts the whole run on collection errors, so a wider target gives no
signal at all rather than a partial one.

Establish the comparison before reading a result as a regression: run the same
target at the merge-base and diff the counts. On 2026-09-22 that turned a
1401-error scare into a clean `1133 -> 1135`, the delta being exactly the two
tests added under `test/registered/`.

**Record the command next to any suite number you quote.** A plan that asserted
"724 passed / 409 skipped" without its invocation could not be checked at all
until the target was recovered by arithmetic: 724 + 409 is the 1133 that
`test/registered/unit/kernels` collects, with the GPU tests skipping that day.

## What this costs

Running unverified code needs a commit first. That is the intended trade: the
branch carries a fix-up commit rather than the box carrying an untracked state
nobody can reconstruct. Do not amend or rebase to tidy it.

## Mutants and concurrent lanes

A mutant is deliberately throwaway, so it is the one edit that is made on
divix01 rather than pushed: apply it in the worktree, run, then
`git checkout --` the file. Never commit one.

Apply it in a **private** worktree, not `wt-dsv41`, whenever anything else may
be working the branch:

```bash
git -C <repo> worktree add --detach /data/models/slang/nvfp4-work/wt-<name> <commit>
# ... mutate, run, revert ...
git -C <repo> worktree remove /data/models/slang/nvfp4-work/wt-<name>
```

On 2026-09-21 mutants were run in `wt-dsv41` while an agent worked the same
branch; it found the tree dirty and had to build its own worktree to proceed.
Nothing was lost, but a concurrent `git checkout --` would have destroyed its
uncommitted work.

After reverting, re-run the suite and record that it is green again. A mutation
result is only meaningful next to the restored baseline.

## CPU and cores

Every CPU job runs under `taskset -c 0-63` with threads capped
(`OMP_NUM_THREADS`). Cores 64-71 stay free: core 71 is production's doorbell
spin core. GPU work goes through `gpu-run.sh` (takes `cc-gpu.lock`, pins cores
32-63), never a bare command.

## The launch gate can validate the wrong format and pass

`expert_stream_requirements_for` (`python/sglang/srt/arg_groups/expert_stream_requirements.py:160`) asks
`expert_quant_method` for the launch's format. That returns `None` when `model_path` is not a local directory
(`:134-136`), and `None` falls back to `NVFP4_EXPERT_STREAM_REQUIREMENTS` **silently** (`:164-166`). So a gate run
against a mistyped or nonexistent model directory checks the NVFP4 rules, reports that it accepted the launch, and
never applies any EXL3 rule -- not the breakable decode graph, not batch size 1, not the refusals of
`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` and `SGLANG_MOE_HOT_ASYNC_PROMOTIONS`.

Note the asymmetry that makes it easy to miss: an *unsupported* method raises a clear `ValueError` naming the
supported methods (`:171-177`), so the one branch that is silent is the one that looks like success.

**This is not a bug to fix in the gate.** Returning `None` for a non-directory is deliberate: `model_path` may be a
remote HuggingFace repo id, and raising there would refuse legitimate launches. The defect is that "could not
determine the format" and "there is genuinely no expert format" are the same value.

**What a caller must do instead:** after running the gate, assert the resolved requirements are the format you meant
to test. A harness that merely checks the gate did not raise has proved nothing about an EXL3 launch. Found by the
`benchmarks/dsv41_flash` work, which now asserts exactly this; `check_gate` there is the worked example.
