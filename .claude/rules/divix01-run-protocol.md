# Running code on divix01

Code is written on the laptop, committed, pushed to `shared`, and pulled on
divix01. Never copy a working tree to divix01 by other means.

```bash
# laptop, in the worktree for the branch
git push shared <branch>

# divix01
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41 \
  && git pull --ff-only shared dsv41 && git log -1 --oneline'
```

Then run in that worktree, with `PYTHONPATH=$PWD/python`.

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
