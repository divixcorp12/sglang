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

The serving venv is `/data/models/slang/.venv-upstream` (torch 2.14.1,
flashinfer 0.7.2; `arm_env.PYTHON`) since the upstream merge of 2026-10-18. Its
sglang is an editable install of `wt-upstream-merge`, so it imports that tree
unless `PYTHONPATH` points at the tree under test. The old venv
`/data/models/slang/.venv` (torch 2.13) imports `main-port-probe-7bc4eb` the
same way; keep it for rollback and for re-running dated records made on it.
Without `PYTHONPATH` a run exercises unrelated code and reports green. Print
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

## Run only the tests your change affects

Run only the test files that exercise the code you changed, and name them as narrowly as you can (files, then `-k`).
Do not run a whole suite to check a change it mostly does not touch: it costs minutes per run and adds load-sensitive
noise. A change to the expert-stream / RAM-miss host or device code, for example, runs
`test/registered/unit/kernels/test_*exl3*.py test/registered/unit/kernels/test_*expert*.py` plus the GPU files in
`test/manual/dsv41`, not all of `unit/kernels`. Record the exact selection next to the result.

## Point the registered suite at `unit/kernels`, not the whole tree

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-51,56-107 \
  /data/models/slang/.venv-upstream/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly
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

## Cold JIT builds and child-process timeouts

Any C++, compiler or flag change gives every JIT module a new build key, so the first run afterwards rebuilds all of
them: 50-100 s per expert-stream host variant, serialized by the build lock in `kernels/jit/utils/compile/loader.py`,
so under `pytest -n 8` the queue makes it longer still.

A test whose child process would build a module must warm it in the parent first, where nothing times the wait.
`run_host_script` (`python/sglang/test/dsv41_ram_miss_fixtures.py`) is the worked example: it calls
`warm_host_modules` with the variant and lane count its child script constructs, then spawns. Start every such child
with `spawn_child` in the same file, which does both steps. The child has no conftest, so it also loads the default
build through `host_layout()`, and a child that serves a miss builds `uring_file_reader`; the helper warms both.
Children that test a build itself (a build failure, a fresh build dir) are left cold on purpose. `hotpath_shim.run_child`
goes through `spawn_child` too, with `LD_PRELOAD` on the child only.

An abort-expecting child must run with core dumps off (`spawn_child` does it, through `NO_CORE_DUMP`): each dump is about
443 MB through systemd-coredump, which can outlast a 60 s timeout under load. Leave pytest's own limit alone, so a real
crash of the test process still leaves a dump.

A `TimeoutExpired` on a cold cache is the compiler, not a hang. Check the module's build-dir mtimes before calling it a
regression. On 2026-10-04, after the GCC 15 / `-march` change, nine `run_host_script` callers failed at 60 s under
`-n 8` and all passed serially once the module was cached.

A suite runner should load the modules once before `-n 8` (the session runner does; no repo script does). To prove a
fix on a cold cache without touching anyone's cache, set `SGLANG_JIT_HOST_MARCH` to a name not used before (for example
`x86-64-v2`), which gives every module a new key.

## Run GPU tests with `-n 8`

GPU test files (`register_cuda_ci` files, `test/manual/dsv41`, `test/registered/unit/kernels`) run under
`pytest -n 8`, like the CPU tests. Run serially, they leave the RTX 5090 almost idle: on 2026-10-22 a serial
`unit/kernels` run took 19m48s with the GPU at ~0% utilization and ~1.8 GiB used, and the expert-engine GPU selection
took ~2 min under `-n 8` against ~15 min serially. Tests share the card without `cc-gpu.lock` (see "CPU and cores");
A/Bs, timed benchmarks and nsys captures still take the lock and never run with `-n`.

- Warm the JIT modules first, serially, as above: a cold cache under `-n 8` queues builds behind the build lock and
  times out child processes.
- Compare against a base run made the same way. Load-sensitive tests (idle-spin, keep-warm, timing windows) flake under
  `-n 8`; rerun a branch-only failure serially before calling it a regression.
- A serving process holding most of the card's memory makes GPU tests fail with out-of-memory on both sides; that is
  not a result. Check `nvidia-smi` before reading one.

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

divix01 has two Xeon Platinum 8280s (28 physical cores per socket, SMT 2, 112 CPUs; cpu N's SMT sibling is N + 56):

| | node 0 (the GPU's) | node 1 (all four NVMe drives) |
|---|---|---|
| CPUs | 0-27, 56-83 | 28-55, 84-111 |
| Production roles | server 0-5,56-61; CPU experts 6-15; RAM thread 26; copy thread 27; benchmark client 18-21; helpers 22-25 | CPU experts 28-37; RAM thread 51; RAM-prefetch thread 50 |
| Kept free | | 52-55,108-111 (`arm_env.FREE_CORES`) |

(The roles are `benchmarks/dsv41_baseline/arm_env.py`'s; the 18-core CPUs this box had before 2026-10-21 used 0-17,36-53 and
18-35,54-71 with the same roles. Older notes and result files quote those numbers.)

- **CPU-only jobs** run under `taskset -c 0-51,56-107` (everything but the free cores; `arm_env.CPU_JOB_CORES`) with
  threads capped (`OMP_NUM_THREADS`).
- **GPU jobs** run on node 0, `taskset -c 0-27,56-83` (`arm_env.GPU_JOB_CORES`). Several can share the card: GPU unit
  and correctness tests, mutants and offline GPU analysis run alongside each other without `cc-gpu.lock`.
- **Only A/B tests and performance-sensitive benchmarks take `cc-gpu.lock`**, for exclusive use of the card: served
  A/Bs and arms (`run_arm.sh` takes it itself), timed benchmarks and anything whose numbers another GPU job would
  distort. Owner, 2026-10-22: a 30 s test file had waited tens of minutes behind another agent's test round.
- **52-55,108-111 stay free**: NVMe completion interrupts belong there, and the RAM-miss path reads rows from NVMe.
  Nothing pins the interrupts yet: irqbalance spreads them over node 1 and node 0's siblings. A root step does (write the
  cores to each nvme queue IRQ's `/proc/irq/N/smp_affinity_list`, and set `IRQBALANCE_BANNED_CPULIST=0-51,56-107`). The
  code refuses these cores already (`RESERVED_CORES` in `threading_config.py`, `is_reserved_core` in `core_topology.h`).

Two ways to take the GPU lock (A/B tests and performance-sensitive benchmarks only), both on divix01:

- `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh <cmd>`
  (not in any repo). It retries the lock every 60 s and **gives up after 30 min
  with exit 75**, so it fails behind a long arm or soak; check for 75.
- To wait as long as it takes, lock directly:
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-27,56-83 <cmd>`,
  or `exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; flock 9` in a script
  (see the lock order below).

### A CPU-only job must hide the GPU

Launch offline analysis, replays and training that need no GPU with
`CUDA_VISIBLE_DEVICES=` (empty). Never use `env -u CUDA_VISIBLE_DEVICES` for
them: that form is for pytest's GPU tests, and it leaves the card visible.
Anything that imports torch can then open a CUDA context, about 500 MiB,
without holding `cc-gpu.lock`.

That is enough to break a serving launch. Its KV pool is sized from free VRAM
at startup, so a stray context shrinks it below the pinned 381,184 tokens, and
the server dies with `KvPinMissed`. On 2026-10-17 a CPU-only head-retrain job
launched with `env -u` did this to another agent's capture twice (KV pool
210,944). Check with `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv` that none of your PIDs are listed.

## Lock order: `rowimg-disk.lock`, then `cc-gpu.lock`

A job that needs both locks (an A/B or a performance-sensitive benchmark) takes the disk lock first. `run_arm.sh` takes
`cc-gpu.lock` itself (non-blocking), so an arm driver must already hold
`rowimg-disk.lock` and poll for the GPU; a script that held the GPU while
waiting for the disk would deadlock against it, both waiting forever.
Every committed script under `analysis/dsv41-drive/` follows this order.

## The launch gate can validate the wrong format and pass

`expert_stream_requirements_for` (`python/sglang/srt/arg_groups/expert_stream_requirements.py`) asks
`expert_quant_method` for the launch's format. That returns `None` when `model_path` is not a local directory, and
`None` falls back to `NVFP4_EXPERT_STREAM_REQUIREMENTS` **silently** (the first `return` in
`expert_stream_requirements_for`). So a gate run
against a mistyped or nonexistent model directory checks the NVFP4 rules, reports that it accepted the launch, and
never applies any EXL3 rule -- not the breakable decode graph, not batch size 1, not the refusals of
`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` and `SGLANG_MOE_HOT_ASYNC_PROMOTIONS`.

Note the asymmetry that makes it easy to miss: an *unsupported* method raises a clear `ValueError` naming the
supported methods (the registry miss in `expert_stream_requirements_for`), so the one branch that is silent is the one that looks like success.

**This is not a bug to fix in the gate.** Returning `None` for a non-directory is deliberate: `model_path` may be a
remote HuggingFace repo id, and raising there would refuse legitimate launches. The defect is that "could not
determine the format" and "there is genuinely no expert format" are the same value.

**What a caller must do instead:** after running the gate, assert the resolved requirements are the format you meant
to test. A harness that merely checks the gate did not raise has proved nothing about an EXL3 launch. Found by the
`benchmarks/dsv41_flash` work, which now asserts exactly this; `check_gate` there is the worked example.
