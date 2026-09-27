# Doorbell side-thread copier removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the doorbell side-thread expert copier (kernel, Python wrapper, runtime wiring, env vars, offload
preset, tests, benchmarks), because it is dead code. Keep the NVFP4 graph-gather path and the DSV4.1 EXL3 path
behaving exactly as they do today with the doorbell off, which is the only supported state.

**Architecture:** The doorbell has four surfaces, and each is removed as one reviewable unit:
1. **Runtime** (Task 1): `ExpertDoorbellCopier` and its `.cuh`, `DoorbellRowBackend`, the hot-cache manager's
   `_start_doorbell`, quiesce, resume, stop, log and stats code, `ModelRunner`'s startup checks and capture
   quiesce wrappers, and `Scheduler`'s shutdown stop. One piece of shared state survives under a new name.
   `ExpertHotCacheManager.doorbell_fail_stop_check` is also the only caller of the EXL3 RAM-miss service's
   per-batch fail-stop check (`register_fail_stop_check`), so it becomes `run_fail_stop_checks()` and is not
   deleted.
2. **Launch surface** (Task 2): the eight `SGLANG_MOE_EXPERT_DOORBELL*` descriptors move into the
   deprecated-env registry (`_DEPRECATED_ENVS`), which warns when a variable is set. The `doorbell` preset, its
   rules and its CLI choice go, and so do the gate refusals and the benchmark pins.
3. **Docs** (Task 3): living docs get a dated "removed" note, and history stays as written.
4. **Proof** (Task 4): a whole-repo grep classification, the suite against the merge-base baseline, and a
   Qwen3.8 NVFP4 graph-gather server smoke on divix01.

**Tech Stack:** Python 3 + PyTorch, pytest/unittest, CUDA JIT kernels (only deleted here), msgspec, sglang
`Envs`/`EnvField`. The tests and the server smoke run on divix01 (RTX 5090).

**Spec:** No spec file. The spec is the user's 2026-09-27 request ("remove the doorbell side-thread copier, which is
dead code", a decision the user made). It is restated under Design below. Read it together with the two project
skills it binds: `.claude/skills/env-var-conventions/SKILL.md` and `.claude/skills/large-class-style/SKILL.md`.
Also read `.claude/rules/modify-component-must-read.md` and `.claude/rules/divix01-run-protocol.md`.

## Design (the spec, restated)

- **Why it is dead:**
  - The production launch sets `SGLANG_MOE_EXPERT_DOORBELL=0` (`run_server.md:444`; that line is in DSV4.1's
    "Current args").
  - It cannot run with the overlap scheduler that production uses (`MOE_EXPERT_TRANSFER.md:214-220`).
  - Stage-2 insert-on-miss refuses it (`offload_presets.py:291-295`, `expert_hot_cache.py:1126`).
  - EXL3/DSV4.1 refuses it (`expert_stream_requirements.py:327-331`).
  - The DSV4.1 benchmarks pin it to 0.
- **The full env-var list** (verified at `environ.py:436-453`, and nowhere else defined):
  - `SGLANG_MOE_EXPERT_DOORBELL`
  - `SGLANG_MOE_EXPERT_DOORBELL_CPU`
  - `SGLANG_MOE_EXPERT_DOORBELL_TIMEOUT_POLLS`
  - `SGLANG_MOE_EXPERT_DOORBELL_DEGRADED_POLLS`
  - `SGLANG_MOE_EXPERT_DOORBELL_DRAIN_POLLS`
  - `SGLANG_MOE_EXPERT_DOORBELL_MODE`
  - `SGLANG_MOE_EXPERT_DOORBELL_FATAL_WAIT_S`
  - `SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY`

  The raw, test-only `DOORBELL_SPIN_CORE` / `DOORBELL_TEST_UNPRIMED` (not `SGLANG_*`) disappear with the tests
  that read them.
- **What a user who still sets a removed variable gets (decision): a warning, not a refusal.** All eight names go
  into `_DEPRECATED_ENVS` as `_DeprecatedEnv(note=...)`. That is `environ.py`'s single registry for removed
  variables, and env-var-conventions Rule 5's mechanism for "a full removal where the env var is going away"
  (`_print_deprecated_env("SGLANG_OLD_NAME")` in the skill's words; this tree's equivalent is the registry). Any set
  value warns at import, in every process, including `0`. The reasons for warning rather than refusing:
  - Every production and benchmark launch pins the variable to `0` today (`run_server.md:444`, the divix01
    production script). A refusal would stop those launches for asking for the only supported state.
  - Setting `1` has no correctness contract left to break: every row the doorbell failed to deliver was already
    copied in-graph by the residual. So the behaviour the user now gets, the in-graph copy, is the fallback they
    had already accepted.
  - A warning names the variable, so a stale `1` is not silent.

  The same precedent covers `SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2` and six more in the "Removed without
  replacement" group.
- **What `--moe-offload-preset doorbell` gets:**
  - On the CLI, argparse refuses it (`invalid choice`), because the choice is gone.
  - Through the Python API (`Engine(moe_offload_preset="doorbell")`, which bypasses argparse),
    `handle_moe_offload_preset` raises `KeyError: 'doorbell'`. That is loud and names the value; this plan pins it
    rather than adding a new validation path.
- **Shared state that must keep working:**
  - `InGraphRowBackend`, `PinnedTierRowBackend`, `ExpertRowPlan`, `ExpertRowPlanner`, `ExpertRowDelivery`,
    `plan_residual_routes`, `live_slot_map` (`expert_row_plan.py`).
  - `expert_stream.py:1388`'s `row_backend.copy_residual`.
  - `copy_expert_row_segments_gpu` / `ExpertRowSegments` (`expert_cache_transfer`).
  - `register_fail_stop_check`, and EXL3's registration at `exl3_ram_miss.py:919`.
  - `Scheduler.release_host_resources`' EXL3 shutdown ordering (LEASE_PROTOCOL.md 20.2i).
- **Behaviour with the doorbell off is unchanged:**
  - With `doorbell is None`, `doorbell_fail_stop_check(synchronize=True)` only ran the registered checks and
    returned 0.0 without touching the device. `run_fail_stop_checks()` does exactly that.
  - `_expert_doorbell_quiesced()` yielded without side effects.
  - `stop_doorbell()` was a no-op.
  - `require_graph_gather_support` received `pinned_tier_ok=not gpu_residency_update` and
    `exl3_direct_ok=(gpu_residency_update and stage == DIRECT)`.

  Removing these changes no executed statement on either production path.

## Global Constraints

- **Branch and worktree:** branch `doorbell-removal`, created from `master` at `29c7d4e2b1`. It gets its own
  laptop worktree at `/home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal`. Never edit in the main
  checkout `/home/dimitri/data/divix/sglang-nvfp4`. Every line number in this plan refers to `29c7d4e2b1`, and
  Task 0 checks the anchors.
- **Local `master` is 52 commits ahead of `origin/master` (`c377185c16`).** Pushing `doorbell-removal` publishes
  those commits as branch history. It does not move `origin/master`. Do not push `master`.
- **Concurrent plan:** a separate plan splits `csrc/moe/dsv41_layer_fusion.cuh` at the same time, touching:
  - `python/sglang/srt/layers/moe/expert_residency_gpu.py`;
  - `python/sglang/srt/layers/quantization/exl3_fused_moe.py`;
  - `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py`;
  - `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh`;
  - `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`.

  **This plan edits none of those five files.** `expert_residency_gpu.py` keeps its two doorbell strings (the
  `check_miss_plans` docstring bullet at :365-368 and the message "leave SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY
  at 0" at :404) until both branches have merged; see Follow-ups. Possible overlap to watch:
  `test/registered/unit/layers/moe/test_expert_residency_gpu.py` (this plan edits :171-356) is the other plan's
  natural test file. If both touch it, rebase whichever lands second; the regions do not overlap today.
- **Frozen file:** `model_runner.py` is frozen (large-class-style §1). This plan only *deletes* from it: the
  doorbell's inlined validation block (domain logic that §1.4 forbids anyway), seven keyword arguments, and a
  context manager. It adds no statement. `Scheduler` edits are one deleted delegating block and one renamed
  delegating method (§1.3 "delegate"); no `__init__` changes.
- **Env vars:** follow env-var-conventions. The descriptors are deleted from `Envs`, the names go to
  `_DEPRECATED_ENVS` (Rule 5), and no `os.getenv("SGLANG_MOE_EXPERT_DOORBELL...")` is added anywhere. Tests set the
  removed names through `os.environ` / `monkeypatch.setenv`, because no descriptor exists to `override`.
- **Out of scope** (do not touch):
  - the cores 64-71 reservation and its "71 is production's doorbell core" wording in
    `expert_stream_transport.py:800`, `expert_stream/host/ffi_exports.h:1033`, `expert_stream/host/pack_pool.h:96`,
    `.claude/rules/divix01-run-protocol.md`, `benchmarks/dsv41_baseline/{arm_env.py:86,test_dsv41_baseline.py:1112,README.md:313}`,
    `analysis/**`;
  - the offline pricing model `expert_prediction/prefetch_pricing.py`, `scripts/expert_prediction/prefetch/price_prefetch.py`
    and `test_expert_prefetch_pricing.py`: it models the design's timing with measured constants, imports nothing
    removed, and dropping it is a separate decision;
  - historical `*.md` (experiment logs, old plans and specs, `analysis/**/*.md` except `LEASE_PROTOCOL.md`).
- **divix01 protocol** (`.claude/rules/divix01-run-protocol.md`):
  - Commit, push, and run only in a pulled private worktree: `/data/models/slang/nvfp4-work/wt-doorbell-removal`
    (head) and `/data/models/slang/nvfp4-work/wt-doorbell-base` (merge-base).
  - Always set `PYTHONPATH=$PWD/python TMPDIR=/mnt/nvme1/tmp-ests OMP_NUM_THREADS=8`, and read `sglang.__file__`.
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`, and CPU-only work under
    `taskset -c 0-63`.
  - Read pytest's own exit status: redirect to a file and `echo EXIT=$?`, never `| tail`.
  - Red commits are real commits (`test(...): ... (red)`), as this repo does. Never amend or rebase; fix-ups are
    new commits.
- **Commits:** end each message with the attribution trailer your session's instructions require.
- **Remote run recipe.** Tasks name these blocks by label. Each is complete as written here.

  **R1: push (laptop)**
  ```bash
  git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal push origin doorbell-removal
  ```

  **R2: move the divix01 head worktree to the pushed branch**
  ```bash
  ssh divix01 'git -C /data/models/slang/sglang fetch origin doorbell-removal \
    && git -C /data/models/slang/nvfp4-work/wt-doorbell-removal checkout --detach origin/doorbell-removal \
    && git -C /data/models/slang/nvfp4-work/wt-doorbell-removal log -1 --oneline'
  ```
  Expected: the sha that `git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal log -1 --oneline`
  prints.

  **R3-cpu `<label>` `<files...>`: CPU-only test files**
  ```bash
  ssh divix01 'cd /data/models/slang/nvfp4-work/wt-doorbell-removal \
    && export PYTHONPATH=$PWD/python TMPDIR=/mnt/nvme1/tmp-ests OMP_NUM_THREADS=8 \
    && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
    && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly -rfE <files...> \
       > /mnt/nvme1/tmp-ests/doorbell-removal/<label>.txt 2>&1; echo EXIT=$?; tail -5 /mnt/nvme1/tmp-ests/doorbell-removal/<label>.txt'
  ```
  The first printed line must be `/data/models/slang/nvfp4-work/wt-doorbell-removal/python/sglang/__init__.py`.
  If it is not, stop: the run exercised other code.

  **R3-gpu `<label>` `<files...>`**: the same as R3-cpu, with `taskset -c 0-63` replaced by
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`.

  **SUITE**: the test set that Task 0 and Task 4 run, as one shell word list. `test_exl3_*.py` expands on divix01.
  ```
  test/registered/unit/kernels
  test/registered/unit/layers/moe/test_expert_graph_gather.py
  test/registered/unit/layers/moe/test_expert_residency_gpu.py
  test/registered/unit/layers/moe/test_expert_fail_stop_checks.py
  test/registered/unit/layers/moe/test_offload_presets.py
  test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py
  test/registered/unit/layers/moe/test_expert_format.py
  test/registered/unit/layers/moe/test_expert_stream.py
  test/registered/unit/layers/moe/test_exl3_*.py
  test/registered/unit/test_expert_stream_requirements.py
  test/registered/unit/test_expert_stream_requirements_exl3.py
  test/registered/unit/test_environ.py
  test/registered/unit/test_ci_registration_ratchet.py
  benchmarks/dsv41_flash/test_harness.py
  benchmarks/dsv41_baseline/test_dsv41_baseline.py
  ```
  List files explicitly. Do not target `test/registered/unit/layers/moe/` as a directory: it sweeps in files that
  fail collection on divix01 under pyarrow 25.0.1 (divix01-run-protocol).

## Review Focus

1. **EXL3's RAM-miss fail-stop must still run after every batch.** Its only caller was
   `doorbell_fail_stop_check`. A rename that drops the call, or a scheduler that stops calling the hook, lets a
   timed-out RAM-miss request serve a dropped layer instead of stopping the process. Pinned by Task 1 Step 1:
   the real unbound `Scheduler._run_expert_fail_stop_checks` on a stub, plus a wiring check on
   `Scheduler.process_batch_result`'s source.
2. **A launch that still exports `SGLANG_MOE_EXPERT_DOORBELL=0`** (the divix01 DSV4.1 production script) must
   start and only warn. Pinned by Task 2 Step 1: a subprocess import test with the value `0`, and an EXL3
   Window-C gate test with the variable set.
3. **The dsv41_flash harness refuses an arm that names an undefined `SGLANG_*` variable** (`bench_arm.py:268`).
   A doorbell pin left in `arm_config.py` would make every A/B refuse to start. Pinned by Task 2 Step 1: a
   harness test that every arm names only defined variables.
4. **`--moe-offload-preset doorbell`** must fail loudly on both entry paths, not fall through to `off`. Pinned by
   Task 2 Step 1: the argparse refusal and the API `KeyError`.
5. **NVFP4 graph-gather gathers stay byte-identical** once the manager loses its doorbell parameters and the
   capture loses its quiesce wrapper. Pinned by existing tests that Task 1 keeps, not deletes:
   - `test_flag_off_gather_matches_the_pre_doorbell_path`;
   - `test_gathers_match_the_pre_doorbell_path_across_residency_updates`;
   - `..._across_insert_on_miss_updates`.

   Their reference is `7de955329a`'s `_gather_graph`. Task 4's server smoke checks the same end to end.

---

### Task 0: Branch, worktrees and the merge-base baseline

**Files:** none changed.

**Interfaces:**
- Consumes: nothing.
- Produces:
  - the branch `doorbell-removal` (at `29c7d4e2b1`);
  - the worktrees `wt-doorbell-base` and `wt-doorbell-removal` on divix01;
  - `/mnt/nvme1/tmp-ests/doorbell-removal/base-collect.txt`, `base-run.txt` and `base-summary.txt`, which Task 4
    diffs against.

- [ ] **Step 1: Check the base and the anchors**

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4 rev-parse --short master
git -C /home/dimitri/data/divix/sglang-nvfp4 worktree add -b doorbell-removal \
  /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal 29c7d4e2b1
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
grep -n "SGLANG_MOE_EXPERT_DOORBELL = EnvBool" python/sglang/srt/environ.py
grep -n "class DoorbellRowBackend" python/sglang/srt/layers/moe/expert_row_plan.py
grep -n "def _start_doorbell\|def doorbell_fail_stop_check\|def _log_doorbell" python/sglang/srt/layers/moe/expert_hot_cache.py
grep -n "if envs.SGLANG_MOE_EXPERT_DOORBELL.get():" python/sglang/srt/model_executor/model_runner.py
grep -n "def _expert_doorbell_fail_stop_check\|stop_doorbell()" python/sglang/srt/managers/scheduler.py
```
Expected:
- `29c7d4e2b1`, then the worktree is created.
- `436:`, `306:`, `1842:`/`2048:`/`2068:`, `766:`, and `4806:`/`1876:`.

If any anchor differs, re-derive this plan's line numbers before editing.

- [ ] **Step 2: Publish the branch and build both divix01 worktrees**

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal push -u origin doorbell-removal
ssh divix01 'git -C /data/models/slang/sglang fetch origin doorbell-removal \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-doorbell-base 29c7d4e2b1 \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-doorbell-removal origin/doorbell-removal \
  && git -C /data/models/slang/nvfp4-work/wt-doorbell-base log -1 --oneline \
  && git -C /data/models/slang/nvfp4-work/wt-doorbell-removal log -1 --oneline \
  && mkdir -p /mnt/nvme1/tmp-ests/doorbell-removal'
```
Expected: both worktrees print `29c7d4e2b1 merge(expert-stream): T3 no longer double-signals a served lane ...`.

- [ ] **Step 3: Collect the baseline's node ids**

In the command below, `SUITE` is the literal word list from Global Constraints.
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-doorbell-base \
  && export PYTHONPATH=$PWD/python TMPDIR=/mnt/nvme1/tmp-ests OMP_NUM_THREADS=8 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest --collect-only -q -p no:randomly SUITE \
     > /mnt/nvme1/tmp-ests/doorbell-removal/base-collect.txt 2>&1; echo EXIT=$?; \
  tail -3 /mnt/nvme1/tmp-ests/doorbell-removal/base-collect.txt; \
  grep -c "^test/registered/unit/kernels/test_expert_doorbell_copier.py::" /mnt/nvme1/tmp-ests/doorbell-removal/base-collect.txt'
```
Expected:
- the first line is `.../wt-doorbell-base/python/sglang/__init__.py` and `EXIT=0`;
- a "N tests collected" line;
- a count, `N_COPIER`. Record it in the plan's execution notes, because Task 4 subtracts it.

If `EXIT` is not 0, read the errors. A collection error that is environmental (pyarrow) is recorded and must
reproduce identically in Task 4.

- [ ] **Step 4: Run the baseline under the GPU lock** (use `run_in_background`; `flock` waits behind other GPU users)

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-doorbell-base \
  && export PYTHONPATH=$PWD/python TMPDIR=/mnt/nvme1/tmp-ests OMP_NUM_THREADS=8 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly -rfE SUITE \
     > /mnt/nvme1/tmp-ests/doorbell-removal/base-run.txt 2>&1; echo EXIT=$?; \
  tail -1 /mnt/nvme1/tmp-ests/doorbell-removal/base-run.txt > /mnt/nvme1/tmp-ests/doorbell-removal/base-summary.txt; \
  grep -E "^(FAILED|ERROR) " /mnt/nvme1/tmp-ests/doorbell-removal/base-run.txt | sort > /mnt/nvme1/tmp-ests/doorbell-removal/base-failures.txt; \
  cat /mnt/nvme1/tmp-ests/doorbell-removal/base-summary.txt; wc -l < /mnt/nvme1/tmp-ests/doorbell-removal/base-failures.txt'
```
Expected: a summary line such as `X passed, Y skipped[, Z failed] in T s`. Record it next to the command in the
execution notes (divix01-run-protocol: "Record the command next to any suite number you quote"). Failures at the
base are pre-existing, not this plan's to fix.

---

### Task 1: Remove the copier and its runtime wiring; keep the fail-stop hook

**Files:**
- Delete: `python/sglang/kernels/jit/csrc/moe/expert_doorbell.cuh`
- Delete: `python/sglang/kernels/ops/moe/expert_doorbell.py`
- Delete: `test/registered/unit/kernels/test_expert_doorbell_copier.py`
- Delete: `benchmark/expert_doorbell/bench_doorbell.py`, `benchmark/expert_doorbell/probe_hold.py` (they import the
  deleted module)
- Modify: `python/sglang/srt/layers/moe/expert_row_plan.py:28-49,65,67-68,306-330`
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py:1006-1012,1119-1129,1422-1433,1842-1992,1994-1995,2048-2111,2497-2499,2755-2756,2942-2944,3068`
- Modify: `python/sglang/srt/model_executor/model_runner.py:766-812,847-853,1343-1346,1726-1737,1741-1742,1802-1807`
- Modify: `python/sglang/srt/managers/scheduler.py:1872-1878,4790,4806-4819`
- Modify: `python/sglang/srt/layers/moe/expert_format.py:279-281,300`
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py:1057` (docstring only)
- Modify: `test/registered/unit/test_ci_registration_ratchet.py:58`
- Test: `test/registered/unit/layers/moe/test_expert_fail_stop_checks.py`,
  `test/registered/unit/layers/moe/test_expert_graph_gather.py`,
  `test/registered/unit/layers/moe/test_expert_residency_gpu.py`,
  `test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py`,
  `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`,
  `test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py`,
  `test/registered/unit/layers/moe/test_expert_format.py`

**Interfaces:**
- Consumes: Task 0's branch and worktrees.
- Produces:
  - `ExpertHotCacheManager.run_fail_stop_checks(self) -> None`;
  - `Scheduler._run_expert_fail_stop_checks(self) -> None`;
  - `ExpertHotCacheManager.from_model(...)` without the parameters `expert_doorbell`, `doorbell_cpu_core`,
    `doorbell_timeout_polls`, `doorbell_degraded_polls`, `doorbell_drain_polls`, `doorbell_plan_capacity` and
    `doorbell_fatal_wait_s`.

  Unchanged: `register_fail_stop_check(check: Callable[[], None]) -> None`, `InGraphRowBackend`,
  `PinnedTierRowBackend`, `ExpertRowDelivery`, `ExpertRowPlan`, `ExpertRowPlanner`, `plan_residual_routes`.
  Task 2 relies on `model_runner.py` no longer reading any `SGLANG_MOE_EXPERT_DOORBELL*` descriptor.

- [ ] **Step 1: Write the failing fail-stop tests**

In `test_expert_fail_stop_checks.py`, replace `_manager` and the four functions from
`test_registered_checks_run_before_the_doorbell_check` through `test_a_manager_without_checks_still_returns`
(lines 23-68) with:

```python
def _manager():
    return ExpertHotCacheManager.__new__(ExpertHotCacheManager)


def test_registered_checks_run_in_registration_order():
    manager = _manager()
    calls = []
    manager.register_fail_stop_check(lambda: calls.append("a"))
    manager.register_fail_stop_check(lambda: calls.append("b"))
    manager.run_fail_stop_checks()
    assert calls == ["a", "b"]


def test_a_failing_check_raises_through_the_scheduler_hook():
    manager = _manager()

    def fail():
        raise RuntimeError("fail-stop")

    manager.register_fail_stop_check(fail)
    with pytest.raises(RuntimeError, match="fail-stop"):
        manager.run_fail_stop_checks()


def test_a_manager_without_checks_still_returns():
    assert _manager().run_fail_stop_checks() is None


def _scheduler_stub(manager):
    return SimpleNamespace(
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(expert_hot_cache_manager=manager)
        )
    )


def test_the_scheduler_hook_runs_the_managers_checks():
    """The real, unbound Scheduler method on a stub. EXL3's RAM-miss service registers
    its per-batch fail-stop here (exl3_ram_miss.py), and this hook is its only caller."""
    from sglang.srt.managers.scheduler import Scheduler

    manager = _manager()
    calls = []
    manager.register_fail_stop_check(lambda: calls.append("exl3"))
    Scheduler._run_expert_fail_stop_checks(_scheduler_stub(manager))
    assert calls == ["exl3"]


def test_the_scheduler_hook_does_nothing_without_a_manager():
    from sglang.srt.managers.scheduler import Scheduler

    Scheduler._run_expert_fail_stop_checks(_scheduler_stub(None))
    Scheduler._run_expert_fail_stop_checks(SimpleNamespace())


def test_every_batch_result_reaches_the_fail_stop_hook():
    """Wiring: process_batch_result must call the hook for every forward mode (Review Focus 1)."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    assert "self._run_expert_fail_stop_checks()" in inspect.getsource(
        Scheduler.process_batch_result
    )
```

In `_AttachingFormat.attach_hot_cache_manager` (lines 164-168), replace the comment and the `events.append(...)`
with:

```python
        # The manager is finished: every cache exists.
        self.events.append(("attach", streamer.layer_id, sorted(manager.caches)))
```
and in `test_from_model_attaches_formats_then_pushes_residency` replace the first assertion with:
```python
    assert events[:2] == [("attach", 0, [0, 1]), ("attach", 1, [0, 1])]
```

- [ ] **Step 2: Commit the red tests and run them to verify they fail**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
git add test/registered/unit/layers/moe/test_expert_fail_stop_checks.py
git commit -m "test(moe): the fail-stop hook is the manager's, not the doorbell's (red)"
```
Run R1, R2, then `R3-cpu t1-red test/registered/unit/layers/moe/test_expert_fail_stop_checks.py`.
Expected: `EXIT=1`, and the failures are `AttributeError: ... 'run_fail_stop_checks'` and
`'_run_expert_fail_stop_checks'`, plus the attach test's tuple length mismatch.

- [ ] **Step 3: Rename the hook in the manager and the scheduler**

In `expert_hot_cache.py`, replace `doorbell_fail_stop_check` (lines 2048-2066) with:

```python
    def run_fail_stop_checks(self) -> None:
        """Run every registered fail-stop check, in registration order.

        The scheduler calls this after each forward's results are processed and
        before the next forward, in every forward mode. A check raises to stop the
        process and must not synchronize the device.
        """
        for check in getattr(self, "fail_stop_checks", ()):
            check()
```
and change `register_fail_stop_check`'s first docstring line (1995) to:
```python
        """Run ``check`` after every batch result (see ``run_fail_stop_checks``).
```

In `scheduler.py`, change the call at line 4790 to `self._run_expert_fail_stop_checks()`, and replace
`_expert_doorbell_fail_stop_check` (4806-4819) with:

```python
    def _run_expert_fail_stop_checks(self) -> None:
        """Run the expert hot cache's fail-stop checks after every forward's results.

        Runs in every forward mode, before the next forward. A check raises to stop the
        process (the EXL3 RAM-miss service registers one) and never synchronizes the device.
        """
        model_runner = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        manager = getattr(model_runner, "expert_hot_cache_manager", None)
        if manager is not None:
            manager.run_fail_stop_checks()
```

In `release_host_resources`, delete lines 1872-1878: the `model_runner` / `expert_hot_cache_manager` lookups and
the `try: expert_hot_cache_manager.stop_doorbell() ... except` block. Keep the two comment lines above them.

In `exl3_ram_miss.py:1057`, change `(the scheduler's doorbell hook)` to
`(the scheduler's fail-stop hook, ``run_fail_stop_checks``)`.

- [ ] **Step 4: Delete the copier's runtime wiring from the hot-cache manager**

In `expert_hot_cache.py`:
- In `from_model`'s signature, delete the seven parameters at lines 1006-1012 (`expert_doorbell` through
  `doorbell_fatal_wait_s`).
- Replace lines 1119-1129 with:
  ```python
          if index(graph_gather_batch_size) or gpu_residency_update:
              require_graph_gather_support(
                  streamers.values(),
                  pinned_tier_ok=not gpu_residency_update,
                  exl3_direct_ok=(
                      gpu_residency_update
                      and insert_on_miss == InsertOnMissStage.DIRECT
                  ),
              )
  ```
- Delete the `manager.doorbell = (...)` statement (1422-1433).
- Delete these methods whole: `_start_doorbell` (1842-1970), `quiesce_doorbell`, `resume_doorbell`, `stop_doorbell`
  (1972-1992), and `_log_doorbell` (2068 through the closing `)` of its `logger.info` at 2111).
- Delete the three stats blocks: 2497-2499 (`doorbell = getattr(...)` / `metadata["doorbell"] = ...`), 2755-2756
  (`if "doorbell" in metadata: ...`) and 2942-2944.
- Delete the `self._log_doorbell()` call at 3068.

Then:
```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
grep -n -i "doorbell" python/sglang/srt/layers/moe/expert_hot_cache.py
pyflakes python/sglang/srt/layers/moe/expert_hot_cache.py | grep -v "imported but unused.*TYPE_CHECKING" || true
```
Expected: no `doorbell` line. Pyflakes reports no new unused name. Compare against
`git show 29c7d4e2b1:python/sglang/srt/layers/moe/expert_hot_cache.py | pyflakes -`, and remove only names that
became unused.

- [ ] **Step 5: Delete the row backend, the frozen-file checks and the capture quiesce**

In `expert_row_plan.py`:
- Delete `BACKEND_DOORBELL` (65), `MODE_CURRENT` and `MODE_NEXT_LAYER` (67-68). They have no reader outside this
  module and the docstring.
- Delete the class `DoorbellRowBackend` (306-330).
- In the module docstring, drop the sentence "The copier's ring holds at least two outstanding requests." from
  the **Tags** bullet, and replace the **Backends** bullet (43-49) with:
  ```
  * **Backends**: ``in_graph`` (the existing ``copy_expert_row_segments_gpu``
    serves the plan at ``post``; resolve reports everything delivered) and
    ``pinned_tier`` (the same kernel, sourcing rows from a partial pinned host
    tier; see ``PinnedTierRowBackend``). A doorbell side-thread backend was
    removed unused on 2026-09-27.
  ```

In `model_runner.py` (frozen; delete only):
- Delete the whole `if envs.SGLANG_MOE_EXPERT_DOORBELL.get():` block (766-812), from that line through the
  `raise ValueError(...)` about `--disable-overlap-schedule`.
- Delete the seven `expert_doorbell=` / `doorbell_*=` keyword arguments (847-853) from the `from_model` call.
- Delete the method `_expert_doorbell_quiesced` (1726-1737, its `@contextlib.contextmanager` included).
- At its three call sites, drop the `with self._expert_doorbell_quiesced():` line and dedent the body one level:
  ```python
          capture = capture_cuda_graphs(
              model_runner=self, capture_decode_cuda_graph=capture_decode_cuda_graph
          )
  ```
  ```python
          capture = capture_decode_graph(model_runner=self)
  ```
  ```python
          capture = capture_prefill_graph(
              model_runner=self,
              eager_runner=self.eager_runner,
              force_for_draft_worker=force_for_draft_worker,
          )
  ```

In `expert_format.py`, change the docstring's `(``pinned_tier_ok``), not the GPU residency update or the
doorbell, whose` to `(``pinned_tier_ok``), not the GPU residency update, whose`. Change the message at 299-300 to:
```python
                "graph gather; unset SGLANG_MOE_EXPERT_GRAPH_GATHER and "
                "SGLANG_MOE_GPU_RESIDENCY_UPDATE"
```

- [ ] **Step 6: Delete the copier, its tests and its benchmarks, and drop the ratchet entry**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
git rm python/sglang/kernels/jit/csrc/moe/expert_doorbell.cuh \
       python/sglang/kernels/ops/moe/expert_doorbell.py \
       test/registered/unit/kernels/test_expert_doorbell_copier.py \
       benchmark/expert_doorbell/bench_doorbell.py benchmark/expert_doorbell/probe_hold.py
```
Then delete the line `"unit/kernels/test_expert_doorbell_copier.py",` (58) from `KNOWN_NO_REGISTRY` in
`test/registered/unit/test_ci_registration_ratchet.py`. `test_the_known_lists_carry_no_fixed_files` fails on a
listed file that no longer exists.

- [ ] **Step 7: Adapt the tests that built a doorbell**

`test_expert_graph_gather.py`:
- Delete these five methods of `TestExpertGraphGather` whole:
  - `test_doorbell_gather_writes_the_same_cache_rows_as_the_in_graph_copy`
  - `test_doorbell_gather_replays_after_a_quiesced_capture`
  - `test_doorbell_drain_on_this_path_ends_at_its_budget_and_never_recovers_the_copy`
  - `test_doorbell_disables_after_a_drain_runs_out_and_no_late_copy_lands`
  - `test_doorbell_gather_falls_back_to_correct_rows_when_the_thread_stalls`
- Delete the helpers only they used: `DOORBELL_SPIN_CORE`, `_streamed_model`, `_manager`, `_cache_rows`,
  `_FRESH_PROCESS_ENV`, `_run_in_fresh_process`, and the method `_host_scratch_rows`. Delete the now-unused imports
  `os`, `subprocess`, `sys` and `time`.
- **Keep** `test_flag_off_gather_matches_the_pre_doorbell_path` unchanged; it is Review Focus 5's pin.
- Replace `test_same_plan_through_both_backends_writes_identical_rows_and_masks` with:

```python
    def test_in_graph_backend_delivers_every_planned_row_byte_exact(self):
        from sglang.srt.layers.moe.expert_row_plan import (
            ExpertRowPlan,
            ExpertRowPlanner,
            InGraphRowBackend,
        )

        layer = _layer(seed=41)
        streamer, cache = self._graph_streamer(layer)
        for tensor in cache.tensors.values():
            tensor[cache.capacity :].view(torch.uint8).zero_()
        backend = InGraphRowBackend({0: streamer._graph_row_segments})
        planner = ExpertRowPlanner(cache, cache.capacity, TOP_K)
        plan = ExpertRowPlan.for_scratch(TOP_K, cache.capacity, TOP_K, cache.device)
        for candidates in ([0, 2, 3, 5], [7, 5], [], [2, 0, 7, 3]):
            planner.plan_candidates(
                torch.tensor(candidates, dtype=torch.int64, device="cuda"), plan
            )
            backend.post(0, plan)
            delivery = backend.resolve(0, plan)
            backend.copy_residual(0, delivery)
            mask = delivery.mask()
            torch.cuda.synchronize()
            self.assertEqual(sum(mask.tolist()), len(candidates), candidates)
            for row, expert in enumerate(candidates):
                for name in NVFP4_STREAM_TENSORS[:4]:
                    self.assertTrue(
                        torch.equal(
                            cache.tensors[name][cache.capacity + row].view(torch.uint8).cpu(),
                            _source_bytes(layer, name, torch.tensor([expert]))[0],
                        ),
                        f"{name} candidates={candidates}",
                    )
```
- Replace `test_residual_after_a_partly_wrong_plan_serves_actual_routes_byte_exact` with:

```python
    def test_residual_after_a_partly_wrong_plan_serves_actual_routes_byte_exact(self):
        """A plan that predicted some routed experts and some unrouted ones: routes to delivered
        experts read their scratch rows, the other misses are the residual copied in-graph into
        rows no needed delivered expert occupies."""
        from sglang.kernels.ops.moe.expert_cache_transfer import (
            copy_expert_row_segments_gpu,
        )
        from sglang.srt.layers.moe.expert_row_plan import (
            ExpertRowPlan,
            ExpertRowPlanner,
            InGraphRowBackend,
            plan_residual_routes,
        )

        layer = _layer(seed=33)
        streamer, cache = self._graph_streamer(layer)
        segments = streamer._graph_row_segments
        backend = InGraphRowBackend({0: segments})
        planner = ExpertRowPlanner(cache, cache.capacity, TOP_K)
        plan = ExpertRowPlan.for_scratch(TOP_K, cache.capacity, TOP_K, cache.device)
        residual = ExpertRowPlan.for_scratch(TOP_K, cache.capacity, TOP_K, cache.device)
        for predicted, routes in (
            ([0, 2, 5], [2, 3, 6, 7]),
            ([7, 3], [3, 7, 0, 2]),
            ([5, 0, 2, 3], [1, 4, 6, 1]),
            ([], [0, 2, 3, 5]),
        ):
            planner.plan_candidates(
                torch.tensor(predicted, dtype=torch.int64, device="cuda"), plan
            )
            backend.post(0, plan)
            delivery = backend.resolve(0, plan)
            backend.copy_residual(0, delivery)
            ids = torch.tensor([routes], dtype=torch.int32, device="cuda")
            remap = plan_residual_routes(
                ids.reshape(-1).long(), cache, cache.capacity, delivery, residual
            )
            copy_expert_row_segments_gpu(
                segments, residual.expert_ids, residual.slots, residual.count
            )
            torch.cuda.synchronize()
            expected_residual = len(
                {e for e in routes if e not in (1, 4, 6) and e not in predicted}
            )
            self.assertEqual(int(residual.count.item()), expected_residual)
            self._assert_host_rows(layer, ids, remap.reshape(ids.shape), cache.tensors)
```

`test_expert_residency_gpu.py` (the concurrent plan may also edit this file; keep edits inside 171-356):
- Delete `DOORBELL_SPIN_CORE` (171). Keep `import os`, which line 760 uses.
- In `_captured`, replace the `manager.quiesce_doorbell()` / `try` / `finally: manager.resume_doorbell()` with:
  ```python
          graph = torch.cuda.CUDAGraph()
          with torch.cuda.graph(graph):
              forward()
          manager.discard_graph_capture_routes()
          return graph
  ```
- In `test_gathers_match_the_pre_doorbell_path_across_residency_updates`, change the docstring phrase
  `with SGLANG_MOE_GPU_RESIDENCY_UPDATE, the doorbell off and on.` to `and with SGLANG_MOE_GPU_RESIDENCY_UPDATE.`,
  and replace the loops with:
  ```python
          for gpu in (False, True):
              with self.subTest(gpu_residency_update=gpu):
                  self._run_mode(gpu)
  ```
- In `..._across_insert_on_miss_updates`, change `planner, doorbell plan or remap` to `planner or remap` and replace
  the loop with `self._run_mode(True, insert_on_miss=True)`.
- `_run_mode(self, gpu, insert_on_miss=False)`:
  - build `current = _manager(current_model, gpu, **mode)`;
  - set `context = f"gpu={gpu} step {step}"`;
  - give the final message as `f"gpu={gpu}: no residency update happened"`;
  - delete the `try:` / `finally: if current.doorbell is not None: current.doorbell.stop()` wrapper, dedenting its
    body one level.
- Leave lines 1268-1275 (`assertRaisesRegex(ValueError, "DOORBELL_PLAN_CAPACITY")`) unchanged. They match
  `expert_residency_gpu.py:404`, which this plan does not edit (Follow-ups).

`test_exl3_ram_miss_shutdown.py`:
- Line 304-305's comment: replace `as test_expert_doorbell_copier.py does for the doorbell` with
  `as the removed doorbell tests did`.
- In `_release_scheduler_host_resources`, replace `recorder` and the stub's manager with:
  ```python
      def recorder(name):
          mock = MagicMock()
          getattr(mock, "destroy" if name == "hisparse" else "release_host_resources").side_effect = (
              lambda: order.append(name)
          )
          return mock

      # A bare namespace: release_host_resources calls nothing on the manager, and any call would raise.
      stub = SimpleNamespace(
          tp_worker=SimpleNamespace(model_runner=SimpleNamespace(expert_hot_cache_manager=SimpleNamespace() if manager else None)),
  ```
- Set `CHEAP = ["hisparse", "tree_cache", "decode_offload", "experts_capturer", "indexer_capturer", "rank_consensus"]`.
- In the first test's docstring, change `it runs before the doorbell stop, before hisparse` to
  `it runs before hisparse`.
- Delete the line `assert "doorbell" not in order` (375).

`test_exl3_ram_miss_service.py:209`: change the comment `OFF/SCRATCH and a doorbell configuration retain` to
`OFF/SCRATCH retain`.

`test_expert_format.py:381`: delete the tuple entry `dict(expert_doorbell=True),`.

`test_expert_pinned_graph_gather.py`: rename
`test_the_hot_cache_manager_refuses_a_pinned_tier_under_residency_update_or_doorbell` to
`test_the_hot_cache_manager_refuses_a_pinned_tier_under_residency_update`, and replace its final loop with:
```python
    with pytest.raises(ValueError, match="does not support graph gather"):
        ExpertHotCacheManager.from_model(
            torch.nn.Sequential(layer), **common, gpu_residency_update=True
        )
```

- [ ] **Step 8: Check the tree has no stale reference, then commit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
git grep -n "expert_doorbell\|DoorbellRowBackend\|ExpertDoorbellCopier\|quiesce_doorbell\|resume_doorbell\|stop_doorbell\|doorbell_fail_stop_check\|_expert_doorbell_quiesced\|_expert_doorbell_fail_stop_check\|BACKEND_DOORBELL\|DOORBELL_SPIN_CORE" -- python test benchmark benchmarks scripts
pyflakes test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_residency_gpu.py \
  test/registered/unit/layers/moe/test_expert_fail_stop_checks.py test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py \
  python/sglang/srt/layers/moe/expert_row_plan.py python/sglang/srt/model_executor/model_runner.py
git add -A python test benchmark
git commit -m "refactor(moe): remove the doorbell side-thread copier; the fail-stop hook becomes the manager's"
```
Expected:
- The grep prints nothing.
- Pyflakes prints nothing new relative to the same files at `29c7d4e2b1`. `model_runner.py` may carry
  pre-existing warnings; compare with `git show 29c7d4e2b1:<path> | pyflakes -`.

- [ ] **Step 9: Run the touched tests on divix01 and verify they pass**

Run R1 and R2. Then run the command below, where `$MOE` is `test/registered/unit/layers/moe`:
```
R3-gpu t1 $MOE/test_expert_fail_stop_checks.py $MOE/test_expert_graph_gather.py $MOE/test_expert_residency_gpu.py $MOE/test_exl3_ram_miss_shutdown.py $MOE/test_exl3_ram_miss_service.py $MOE/test_expert_pinned_graph_gather.py $MOE/test_expert_format.py test/registered/unit/test_ci_registration_ratchet.py test/registered/unit/kernels
```
Expected:
- `EXIT=0`, except for any `FAILED` line that also appears in Task 0's `base-failures.txt`.
- The first printed line is the `wt-doorbell-removal` path.
- `test_flag_off_gather_matches_the_pre_doorbell_path`, `test_gathers_match_the_pre_doorbell_path_across_*` and
  every `test_exl3_ram_miss_shutdown.py` test pass.

If one of those fails, the NVFP4 or EXL3 path changed. Stop and debug; do not adapt the assertion.

---

### Task 2: Remove the launch surface: env vars, preset, CLI choice, gates and benchmark pins

**Files:**
- Modify: `python/sglang/srt/environ.py:433-453` (descriptors) and the `_DEPRECATED_ENVS` block (~2310-2396)
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements.py:327-331`
- Modify: `python/sglang/srt/layers/moe/offload_presets.py:13,61,65-69,95-97,134-149,236-237,254,259-264,279-303`
- Modify: `python/sglang/srt/arg_groups/moe_offload_hook.py:87`
- Modify: `python/sglang/srt/arg_groups/fields/exec_.py:944-951`
- Modify: `benchmarks/dsv41_baseline/arm_env.py:154`, `benchmarks/dsv41_flash/arm_config.py:114`
- Test: `test/registered/unit/test_environ.py`, `test/registered/unit/test_expert_stream_requirements.py`,
  `test/registered/unit/test_expert_stream_requirements_exl3.py`,
  `test/registered/unit/layers/moe/test_offload_presets.py`, `benchmarks/dsv41_flash/test_harness.py`

**Interfaces:**
- Consumes: Task 1, after which no runtime code reads a doorbell descriptor.
- Produces:
  - `offload_presets.check_offload_config(values, *, speculative, decode_graphs_disabled, decode_max_bs, tp_size,
    pp_size, dp_size, dp_attention, nvfp4_hot_cache) -> None` (`allowed_cpus` removed);
  - `offload_presets.PRESETS == {"off": None, "graph-gather": GRAPH_GATHER_PRESET}`;
  - `environ._DEPRECATED_ENVS[name]` for each of the eight names, with `replacement is None`;
  - the module constant `environ._DOORBELL_REMOVED_NOTE: str`.

- [ ] **Step 1: Write the failing launch-surface tests**

`test/registered/unit/test_environ.py`: add after the imports:

```python
DOORBELL_ENVS = (
    "SGLANG_MOE_EXPERT_DOORBELL",
    "SGLANG_MOE_EXPERT_DOORBELL_CPU",
    "SGLANG_MOE_EXPERT_DOORBELL_TIMEOUT_POLLS",
    "SGLANG_MOE_EXPERT_DOORBELL_DEGRADED_POLLS",
    "SGLANG_MOE_EXPERT_DOORBELL_DRAIN_POLLS",
    "SGLANG_MOE_EXPERT_DOORBELL_MODE",
    "SGLANG_MOE_EXPERT_DOORBELL_FATAL_WAIT_S",
    "SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY",
)
```
and add to `TestDeprecatedEnvRegistry`:
```python
    def test_the_doorbell_envs_warn_that_the_copier_was_removed(self):
        for old_name in DOORBELL_ENVS:
            with self.subTest(old_name=old_name):
                os.environ[old_name] = "1"
                self.addCleanup(os.environ.pop, old_name, None)
                caught = self._apply(old_name, _DEPRECATED_ENVS[old_name])
                self.assertTrue(caught, old_name)
                message = str(caught[0].message)
                self.assertIn(f"{old_name} is deprecated", message)
                self.assertIn("doorbell side-thread expert copier was removed", message)
                self.assertIsNone(_DEPRECATED_ENVS[old_name].replacement)
                self.assertFalse(hasattr(envs, old_name))

    def test_a_pinned_off_doorbell_env_warns_when_sglang_starts_and_nothing_fails(self):
        # Production launches pin it to 0 today: they must start, and learn the variable is gone.
        env = {**os.environ, "SGLANG_MOE_EXPERT_DOORBELL": "0"}
        result = subprocess.run(
            [sys.executable, "-W", "always", "-c", "import sglang.srt.environ"],
            env=env, capture_output=True, text=True, check=True,
        )
        self.assertIn(
            "Environment variable SGLANG_MOE_EXPERT_DOORBELL is deprecated", result.stderr
        )
```

`test/registered/unit/test_expert_stream_requirements.py`: replace `test_doorbell_is_refused` and
`test_it_accepts_a_launch_with_all_three_settings_off` with:
```python
    def test_the_removed_doorbell_variable_is_not_a_gate_setting(self):
        # Removed 2026-09-27: set, it only warns at import (sglang.srt.environ); the gate ignores it.
        os.environ["SGLANG_MOE_EXPERT_DOORBELL"] = "1"
        memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))

    def test_it_accepts_a_launch_with_both_settings_off(self):
        os.environ.update(
            SGLANG_MOE_PREFETCH_MAX_CANDIDATES="0",
            SGLANG_MOE_GPU_RESIDENCY_UPDATE="0",
        )
        memory_hook.handle_offload_compatibility(_launch(quantization="eager_test"))
```

`test/registered/unit/test_expert_stream_requirements_exl3.py`: delete `"SGLANG_MOE_EXPERT_DOORBELL": False,` from
`WINDOW_C_ENV`, because `_under_window_c_env` calls `getattr(envs, name)`. Then add:
```python
def test_the_removed_doorbell_variable_does_not_gate_an_exl3_launch(model_dir, monkeypatch):
    # Assert the resolved format first: a gate that silently fell back to NVFP4 proves nothing (divix01-run-protocol).
    monkeypatch.setenv("SGLANG_MOE_EXPERT_DOORBELL", "1")
    args = _launch(model_dir)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    _gate(args)
```

`test/registered/unit/layers/moe/test_offload_presets.py`:
- Delete `ALL_CPUS` and the `allowed_cpus=ALL_CPUS,` context entry in `check`.
- Delete `test_doorbell_preset_departs_from_graph_gather_only_where_the_doorbell_forces_it`.
- In `test_every_preset_field_names_an_envs_descriptor`, change the preset set to `{"off", "graph-gather"}`.
- Add to `TestPresetValues`:
```python
    def test_the_doorbell_preset_is_refused_by_the_cli(self):
        import contextlib
        import io

        from sglang.srt.server_args import prepare_server_args

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            prepare_server_args(["--model-path", "dummy", "--moe-offload-preset", "doorbell"])
        self.assertIn("invalid choice", stderr.getvalue())
        self.assertIn("doorbell", stderr.getvalue())
```
- Replace `TestOverlapRule`'s methods with:
```python
    def test_graph_gather_keeps_overlap(self):
        graph = presets.preset_env(presets.GRAPH_GATHER_PRESET)
        self.assertFalse(presets.needs_overlap_off(graph, nvfp4_hot_cache=True))

    def test_a_hot_cache_without_graph_gather_and_the_residency_update_turns_overlap_off(self):
        self.assertTrue(presets.needs_overlap_off({"SGLANG_MOE_HOT_GPU_MB": "1024"}, nvfp4_hot_cache=True))
        self.assertFalse(presets.needs_overlap_off({}, nvfp4_hot_cache=True))

    def test_the_hot_cache_rule_is_nvfp4s_only(self):
        self.assertFalse(presets.needs_overlap_off({"SGLANG_MOE_HOT_GPU_MB": "1024"}, nvfp4_hot_cache=False))
```
- Replace `TestValidation.test_both_presets_pass_their_intended_setups` and
  `test_invalid_combinations_are_refused_before_the_weight_load` with:
```python
    def test_the_graph_gather_preset_passes_its_intended_setup(self):
        check(presets.preset_env(presets.GRAPH_GATHER_PRESET), speculative=True)

    def test_invalid_combinations_are_refused_before_the_weight_load(self):
        graph = presets.preset_env(presets.GRAPH_GATHER_PRESET)
        cases = (
            (graph, dict(decode_graphs_disabled=True), "decode CUDA graphs"),
            (graph, dict(decode_max_bs=2), "cuda-graph-max-bs-decode 1"),
            (dict(graph, SGLANG_MOE_EXPERT_STREAM="0"), {}, "SGLANG_MOE_EXPERT_STREAM=1"),
        )
        for values, context, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check(values, **context)
```
- In `TestPresetHook.setUp`, delete the three `affinity = patch.object(moe_offload_hook.os, "sched_getaffinity", ...)`
  lines.
- Replace `test_doorbell_turns_overlap_off` and `test_a_refusal_names_the_preset` with:
```python
    def test_an_api_launch_naming_the_doorbell_preset_fails_loudly(self):
        # Engine(moe_offload_preset=...) bypasses argparse's choices; the unknown name must not fall through to "off".
        with self.assertRaises(KeyError):
            moe_offload_hook.handle_moe_offload_preset(self.args("doorbell"))

    def test_a_refusal_names_the_preset(self):
        args = self.args(
            "graph-gather",
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="breakable", max_bs=2),
                prefill=PhaseConfig(backend="disabled"),
            ),
        )
        moe_offload_hook.handle_moe_offload_preset(args)
        with self.assertRaisesRegex(
            ValueError, "--moe-offload-preset graph-gather: .*cuda-graph-max-bs-decode 1"
        ):
            moe_offload_hook.check_moe_offload_config(args)
```

`benchmarks/dsv41_flash/test_harness.py`: add after `test_the_arms_differ_only_in_the_lease_switch_and_both_graph_gather`:
```python
def test_every_arm_names_only_sglang_variables_this_tree_defines():
    # bench_arm refuses an arm naming an undefined SGLANG_* variable; the doorbell pin went with the copier.
    from sglang.srt.environ import envs

    for arm in ac.ARMS:
        env = ac.arm_env(arm, paths=PATHS, res=RES)
        unknown = [n for n in env if n.startswith("SGLANG_") and n not in vars(type(envs))]
        assert unknown == [], arm
```
This harness test passes at the base, where the descriptor still exists. It is a guard that turns red if Step 3
removes the descriptors but leaves the pin.

- [ ] **Step 2: Commit the red tests and run them to verify they fail**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
git add test/registered/unit/test_environ.py test/registered/unit/test_expert_stream_requirements.py \
  test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_offload_presets.py \
  benchmarks/dsv41_flash/test_harness.py
git commit -m "test(moe): removed doorbell variables warn, the preset is refused, gates ignore it (red)"
```
Run R1 and R2. Then run:
```
R3-cpu t2-red test/registered/unit/test_environ.py test/registered/unit/test_expert_stream_requirements.py test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_offload_presets.py benchmarks/dsv41_flash/test_harness.py
```
Expected: `EXIT=1`, with these failures:
- `test_the_doorbell_envs_warn_that_the_copier_was_removed` (`KeyError` on `_DEPRECATED_ENVS`);
- `test_a_pinned_off_doorbell_env_warns...` (no warning);
- `test_the_removed_doorbell_variable_is_not_a_gate_setting` ("set it to 0");
- `test_the_removed_doorbell_variable_does_not_gate_an_exl3_launch`;
- `test_the_doorbell_preset_is_refused_by_the_cli` (no `SystemExit`);
- `test_an_api_launch_naming_the_doorbell_preset_fails_loudly` (no `KeyError`);
- `test_invalid_combinations_are_refused_before_the_weight_load` passes, and `test_harness.py` passes.

The `check()` calls, which no longer pass `allowed_cpus`, fail with `TypeError: missing ... 'allowed_cpus'`, which
also counts as red.

- [ ] **Step 3: Move the eight variables into the deprecated-env registry**

In `environ.py`, delete lines 433-453: the comment `# Serve graph-gather miss copies through the doorbell copier
thread...` through `SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY = EnvInt(0)`. Keep `SGLANG_MOE_HOT_LOG_INTERVAL`
(454). Add just above `_DEPRECATED_ENVS: Dict[str, _DeprecatedEnv] = {`:

```python
_DOORBELL_REMOVED_NOTE = (
    "The doorbell side-thread expert copier was removed on 2026-09-27; graph-gather "
    "misses are always copied in-graph. Unset this env."
)
```
and add at the end of the "# Removed without replacement." group, after the
`"SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK": _DeprecatedEnv(...)` entry:

```python
    # The doorbell copier and all its knobs (removed 2026-09-27). Production launches
    # still pin SGLANG_MOE_EXPERT_DOORBELL=0, so a set value warns rather than refuses.
    **{
        name: _DeprecatedEnv(note=_DOORBELL_REMOVED_NOTE)
        for name in (
            "SGLANG_MOE_EXPERT_DOORBELL",
            "SGLANG_MOE_EXPERT_DOORBELL_CPU",
            "SGLANG_MOE_EXPERT_DOORBELL_TIMEOUT_POLLS",
            "SGLANG_MOE_EXPERT_DOORBELL_DEGRADED_POLLS",
            "SGLANG_MOE_EXPERT_DOORBELL_DRAIN_POLLS",
            "SGLANG_MOE_EXPERT_DOORBELL_MODE",
            "SGLANG_MOE_EXPERT_DOORBELL_FATAL_WAIT_S",
            "SGLANG_MOE_EXPERT_DOORBELL_PLAN_CAPACITY",
        )
    },
```

- [ ] **Step 4: Remove the preset, its rules, the CLI choice, the gate refusal and the benchmark pins**

`offload_presets.py`:
- Line 13: `from collections.abc import Mapping`, because `Collection` loses its only user.
- Line 61's comment: `# 2 (DIRECT) copies misses straight into victim slots, +4.4% over 1.`
- Delete the fields `expert_doorbell`, `expert_doorbell_mode` and `expert_doorbell_cpu`, with their two comment
  lines (65-69).
- Delete the three `ENV_NAMES` entries (95-97).
- Delete `DOORBELL_PRESET` with its comment (134-144), and `"doorbell": DOORBELL_PRESET,` (149).
- In `needs_overlap_off`, delete `if _value(values, "SGLANG_MOE_EXPERT_DOORBELL"): return True` (236-237).
- In `check_offload_config`:
  - delete the parameter `allowed_cpus: Collection[int],`;
  - delete the docstring's `allowed_cpus` paragraph (259-264). Keep the `nvfp4_hot_cache` sentence, reworded as
    ```
        ``nvfp4_hot_cache`` is as in ``needs_overlap_off``.
    ```
  - delete everything from `if not _value(values, "SGLANG_MOE_EXPERT_DOORBELL"):` (279) to the end of the
    function (303). The function then ends with the `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1 requires
    --cuda-graph-max-bs-decode 1` check.

`moe_offload_hook.py:87`: delete `allowed_cpus=os.sched_getaffinity(0),`. `os` is still used for `os.environ`.

`fields/exec_.py:944-951`:
```python
            help="Named MoE expert-offload configuration that fills every offload "
            "SGLANG_MOE_* / SGLANG_QWEN4_* variable left unset. 'graph-gather' is the "
            "current best (in-graph gather, insert-on-miss stage 2, fused planner; "
            "tuned for a 32 GB RTX 5090). 'off' sets nothing. Explicitly set variables "
            "win. See sglang.srt.layers.moe.offload_presets.",
            choices=["off", "graph-gather"],
```

`expert_stream_requirements.py`: delete the block at 327-331 (`if envs.SGLANG_MOE_EXPERT_DOORBELL.get(): raise
ValueError(... "SGLANG_MOE_EXPERT_DOORBELL; set it to 0")`).

`benchmarks/dsv41_baseline/arm_env.py:154` and `benchmarks/dsv41_flash/arm_config.py:114`: delete the line
`"SGLANG_MOE_EXPERT_DOORBELL": "0",`.

- [ ] **Step 5: Check the tree has no reader of a removed variable, then commit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
git grep -n "SGLANG_MOE_EXPERT_DOORBELL\|DOORBELL_PRESET\|expert_doorbell" -- python benchmarks benchmark scripts test \
  | grep -v "^python/sglang/srt/environ.py:\|^python/sglang/srt/layers/moe/expert_residency_gpu.py:404:\|^test/registered/unit/test_environ.py:\|^test/registered/unit/test_expert_stream_requirements.py:\|^test/registered/unit/test_expert_stream_requirements_exl3.py:"
pyflakes python/sglang/srt/layers/moe/offload_presets.py python/sglang/srt/arg_groups/moe_offload_hook.py \
  test/registered/unit/layers/moe/test_offload_presets.py test/registered/unit/test_environ.py
git add -A python benchmarks test
git commit -m "refactor(moe): drop the doorbell env vars, preset and gate rules; set variables now warn"
```
Expected:
- The grep prints nothing. The excluded lines are the registry, the deferred message and the new tests.
- Pyflakes prints nothing new relative to `29c7d4e2b1`.

- [ ] **Step 6: Run the launch-surface tests and verify they pass**

Run R1 and R2. Then run:
```
R3-cpu t2 test/registered/unit/test_environ.py test/registered/unit/test_expert_stream_requirements.py test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_offload_presets.py benchmarks/dsv41_flash/test_harness.py benchmarks/dsv41_baseline/test_dsv41_baseline.py
```
Expected: `EXIT=0`, apart from failures that `base-failures.txt` already lists, and the `wt-doorbell-removal`
path printed first.

---

### Task 3: Mark the living docs, and reword the benchmark that cited the removed one

**Files:**
- Modify: `MOE_EXPERT_TRANSFER.md` (under line 214's paragraph, line 276's paragraph and the `## 2. Doorbell
  side-thread copier` heading at 1283)
- Modify: `run_server.md:444`
- Modify: `MERGE_BRIEF_DOORBELL.md` (top)
- Modify: `DSV41_REFERENCE.md:712`
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (under `## 14. Shutdown and quarantine` at 1339, and under
  `### 20.2i PROPOSAL ...` at 2535)
- Modify: `benchmark/expert_delivery/benchmark_ingraph_delivery_cost.py:5-9,20-22,41-42`

**Interfaces:**
- Consumes: Task 1's removal commit, which the notes cite.
- Produces: nothing code-facing.

- [ ] **Step 1: Resolve the commit to cite**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
SHA=$(git log -1 --format=%h -- python/sglang/kernels/ops/moe/expert_doorbell.py); echo "$SHA"
```
Expected: Task 1's `refactor(moe): remove the doorbell side-thread copier ...` commit. Every `$SHA` below is that
literal value.

- [ ] **Step 2: Add the notes (history above them stays as written)**

Insert the note below, as its own paragraph, directly under each of these places:
- `MOE_EXPERT_TRANSFER.md`: the `## 2. Doorbell side-thread copier` heading, the paragraph that begins
  `**The doorbell and overlap scheduling are mutually exclusive.**`, and the paragraph that contains
  `` `--moe-offload-preset doorbell` is the doorbell copier ``;
- `MERGE_BRIEF_DOORBELL.md`: the `# Doorbell copier: what you would be deciding` title;
- `LEASE_PROTOCOL.md`: the `## 14. Shutdown and quarantine` heading and the `### 20.2i PROPOSAL` heading.

The note:
```markdown
> **Removed on 2026-09-27** (`$SHA`, branch `doorbell-removal`): the doorbell side-thread copier, its
> `SGLANG_MOE_EXPERT_DOORBELL*` variables and `--moe-offload-preset doorbell` no longer exist. A set variable only
> warns at startup, and `Scheduler.release_host_resources` no longer calls a doorbell stop. The per-batch fail-stop
> hook is now `ExpertHotCacheManager.run_fail_stop_checks`. The text below is kept as history.
```
`run_server.md:444`: replace `- SGLANG_MOE_EXPERT_DOORBELL=0` with
`- SGLANG_MOE_EXPERT_DOORBELL=0 (removed on 2026-09-27, $SHA: ignored, warns at startup; drop it from the launch script at the next restart)`.
`DSV41_REFERENCE.md:712`: prefix the row's second cell with `**Removed 2026-09-27 ($SHA).** `.

`benchmark_ingraph_delivery_cost.py`:
- Lines 5-9 become:
  ```
  expert_row_plan.py): ``post`` issues ``copy_expert_row_segments_gpu``
  synchronously on the current stream, so it serializes with whatever compute
  shares that stream.
  ```
- Line 21: `real per-expert-row byte layout used by benchmark/expert_doorbell/bench_doorbell.py` becomes
  `real per-expert-row byte layout of the removed benchmark/expert_doorbell/bench_doorbell.py (last at 29c7d4e2b1)`.
- Line 42's comment: `# benchmark/expert_doorbell's cc-pcie-bench Setup (verified on divix01).` becomes
  `# the cc-pcie-bench Setup of the removed benchmark/expert_doorbell (verified on divix01).`

- [ ] **Step 3: Verify and commit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
grep -c "Removed on 2026-09-27" MOE_EXPERT_TRANSFER.md MERGE_BRIEF_DOORBELL.md analysis/dsv41-drive/LEASE_PROTOCOL.md
grep -n "removed on 2026-09-27" run_server.md; grep -n "Removed 2026-09-27" DSV41_REFERENCE.md
grep -n '\$SHA' MOE_EXPERT_TRANSFER.md MERGE_BRIEF_DOORBELL.md analysis/dsv41-drive/LEASE_PROTOCOL.md run_server.md DSV41_REFERENCE.md
python3 -m py_compile benchmark/expert_delivery/benchmark_ingraph_delivery_cost.py
git add MOE_EXPERT_TRANSFER.md MERGE_BRIEF_DOORBELL.md analysis/dsv41-drive/LEASE_PROTOCOL.md run_server.md DSV41_REFERENCE.md \
  benchmark/expert_delivery/benchmark_ingraph_delivery_cost.py
git commit -m "docs(moe): mark the doorbell copier removed in the living docs"
```
Expected:
- counts `3`, `1` and `2`, then one line each from `run_server.md` and `DSV41_REFERENCE.md`;
- the `$SHA` grep prints nothing, because every placeholder holds the literal sha;
- `py_compile` succeeds.

---

### Task 4: Prove it: grep classification, the suite against the baseline, and the server smoke

**Files:** none changed. If a check fails, fix it in the task that owns the file, as a new commit.

**Interfaces:**
- Consumes: Tasks 0-3, with `base-collect.txt`, `base-run.txt` and `base-failures.txt` from Task 0.
- Produces: the evidence recorded in the execution notes.

- [ ] **Step 1: Classify every remaining `doorbell` hit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/doorbell-removal
git grep -il doorbell -- ':!*.md' | sort
```
Expected: exactly these files, each for the reason given:
```
.claude/rules/divix01-run-protocol.md                          # core-71 reservation rationale (out of scope)
analysis/dsv41-drive/c_measurement/proposed_amendment6/quiet_check.py   # comment, history
analysis/dsv41-drive/c_measurement/quiet_check.py              # comment, history
analysis/dsv41-drive/moe_service_node_analysis.py              # comment, history
benchmark/expert_delivery/benchmark_ingraph_delivery_cost.py   # cites the removed benchmark (Task 3)
benchmarks/dsv41_baseline/arm_env.py                           # core-71 comment (line 86)
benchmarks/dsv41_baseline/test_dsv41_baseline.py               # core-71 comment (line 1112)
benchmarks/dsv41_flash/test_harness.py                         # the Task 2 guard's comment
python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h  # core-71 reservation message
python/sglang/kernels/jit/csrc/moe/expert_stream/host/pack_pool.h    # core-71 reservation comment
python/sglang/kernels/ops/moe/expert_stream_transport.py       # core-71 reservation message
python/sglang/srt/environ.py                                   # the deprecated-env registry
python/sglang/srt/layers/moe/expert_prediction/prefetch_pricing.py   # offline pricing model (kept)
python/sglang/srt/layers/moe/expert_residency_gpu.py           # deferred: concurrent plan's file
python/sglang/srt/layers/moe/expert_row_plan.py                # "removed unused on 2026-09-27" docstring
scripts/dsv41/provenance.py                                    # comment about spin-thread CPU accounting
scripts/expert_prediction/prefetch/price_prefetch.py           # offline pricing model (kept)
test/registered/unit/layers/moe/test_exl3_ram_miss_shutdown.py      # "the removed doorbell tests" comment
test/registered/unit/layers/moe/test_expert_graph_gather.py    # test_flag_off_gather_matches_the_pre_doorbell_path
test/registered/unit/layers/moe/test_expert_prefetch_pricing.py      # offline pricing model's test (kept)
test/registered/unit/layers/moe/test_expert_residency_gpu.py   # *_pre_doorbell_path names + deferred assertion
test/registered/unit/test_environ.py                           # removed-variable tests
test/registered/unit/test_expert_stream_requirements.py        # removed-variable gate test
test/registered/unit/test_expert_stream_requirements_exl3.py   # removed-variable gate test
test/registered/unit/layers/moe/test_offload_presets.py        # removed-preset tests
```
Any other file is a missed removal: fix it in its owning task's file set, as a new commit. For `*.md`, run
`git grep -il doorbell -- '*.md'`. Every hit is history except the five living docs that Task 3 marked.

- [ ] **Step 2: Collect at the head and diff the node ids against the base**

Run R1 and R2. Then run the command below, where `SUITE` is the literal list:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-doorbell-removal \
  && export PYTHONPATH=$PWD/python TMPDIR=/mnt/nvme1/tmp-ests OMP_NUM_THREADS=8 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest --collect-only -q -p no:randomly SUITE \
     > /mnt/nvme1/tmp-ests/doorbell-removal/head-collect.txt 2>&1; echo EXIT=$?; \
  cd /mnt/nvme1/tmp-ests/doorbell-removal && diff <(grep "::" base-collect.txt | sort) <(grep "::" head-collect.txt | sort) \
  | grep -v "test_expert_doorbell_copier.py::"; \
  grep -c "^test/registered/unit/kernels/test_expert_doorbell_copier.py::" head-collect.txt'
```
Expected: `EXIT=0`, the `wt-doorbell-removal` path, a final count of `0`, and a diff of exactly these lines, in
any order. Removed (`<`):
```
< ...test_expert_graph_gather.py::TestExpertGraphGather::test_doorbell_gather_writes_the_same_cache_rows_as_the_in_graph_copy
< ...test_expert_graph_gather.py::TestExpertGraphGather::test_doorbell_gather_replays_after_a_quiesced_capture
< ...test_expert_graph_gather.py::TestExpertGraphGather::test_doorbell_drain_on_this_path_ends_at_its_budget_and_never_recovers_the_copy
< ...test_expert_graph_gather.py::TestExpertGraphGather::test_doorbell_disables_after_a_drain_runs_out_and_no_late_copy_lands
< ...test_expert_graph_gather.py::TestExpertGraphGather::test_doorbell_gather_falls_back_to_correct_rows_when_the_thread_stalls
< ...test_expert_graph_gather.py::TestExpertGraphGather::test_same_plan_through_both_backends_writes_identical_rows_and_masks
< ...test_expert_fail_stop_checks.py::test_registered_checks_run_before_the_doorbell_check
< ...test_expert_fail_stop_checks.py::test_the_doorbell_check_still_runs_without_registered_checks
< ...test_expert_pinned_graph_gather.py::test_the_hot_cache_manager_refuses_a_pinned_tier_under_residency_update_or_doorbell
< ...test_offload_presets.py::TestPresetValues::test_doorbell_preset_departs_from_graph_gather_only_where_the_doorbell_forces_it
< ...test_offload_presets.py::TestOverlapRule::test_graph_gather_keeps_overlap_and_doorbell_turns_it_off
< ...test_offload_presets.py::TestOverlapRule::test_the_hot_cache_rule_is_nvfp4s_but_the_doorbell_rule_is_every_formats
< ...test_offload_presets.py::TestValidation::test_both_presets_pass_their_intended_setups
< ...test_offload_presets.py::TestPresetHook::test_doorbell_turns_overlap_off
< ...test_expert_stream_requirements.py::TestEagerFormatRequirements::test_doorbell_is_refused
< ...test_expert_stream_requirements.py::TestEagerFormatRequirements::test_it_accepts_a_launch_with_all_three_settings_off
```
Added (`>`):
```
> ...test_expert_graph_gather.py::TestExpertGraphGather::test_in_graph_backend_delivers_every_planned_row_byte_exact
> ...test_expert_fail_stop_checks.py::test_registered_checks_run_in_registration_order
> ...test_expert_fail_stop_checks.py::test_the_scheduler_hook_runs_the_managers_checks
> ...test_expert_fail_stop_checks.py::test_the_scheduler_hook_does_nothing_without_a_manager
> ...test_expert_fail_stop_checks.py::test_every_batch_result_reaches_the_fail_stop_hook
> ...test_expert_pinned_graph_gather.py::test_the_hot_cache_manager_refuses_a_pinned_tier_under_residency_update
> ...test_offload_presets.py::TestPresetValues::test_the_doorbell_preset_is_refused_by_the_cli
> ...test_offload_presets.py::TestOverlapRule::test_graph_gather_keeps_overlap
> ...test_offload_presets.py::TestOverlapRule::test_the_hot_cache_rule_is_nvfp4s_only
> ...test_offload_presets.py::TestValidation::test_the_graph_gather_preset_passes_its_intended_setup
> ...test_offload_presets.py::TestPresetHook::test_an_api_launch_naming_the_doorbell_preset_fails_loudly
> ...test_expert_stream_requirements.py::TestEagerFormatRequirements::test_the_removed_doorbell_variable_is_not_a_gate_setting
> ...test_expert_stream_requirements.py::TestEagerFormatRequirements::test_it_accepts_a_launch_with_both_settings_off
> ...test_expert_stream_requirements_exl3.py::test_the_removed_doorbell_variable_does_not_gate_an_exl3_launch
> ...test_environ.py::TestDeprecatedEnvRegistry::test_the_doorbell_envs_warn_that_the_copier_was_removed
> ...test_environ.py::TestDeprecatedEnvRegistry::test_a_pinned_off_doorbell_env_warns_when_sglang_starts_and_nothing_fails
> ...dsv41_flash/test_harness.py::test_every_arm_names_only_sglang_variables_this_tree_defines
```
The collected total must equal
`base − N_COPIER − 16 + 17`, with `N_COPIER` from Task 0 Step 3. Any other line in the diff is an unplanned test
change: find it before going on.

- [ ] **Step 3: Run the suite at the head under the GPU lock and compare** (`run_in_background`)

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-doorbell-removal \
  && export PYTHONPATH=$PWD/python TMPDIR=/mnt/nvme1/tmp-ests OMP_NUM_THREADS=8 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly -rfE SUITE \
     > /mnt/nvme1/tmp-ests/doorbell-removal/head-run.txt 2>&1; echo EXIT=$?; \
  cd /mnt/nvme1/tmp-ests/doorbell-removal; tail -1 head-run.txt; cat base-summary.txt; \
  grep -E "^(FAILED|ERROR) " head-run.txt | sort > head-failures.txt; \
  comm -13 <(sed "s/ - .*//" base-failures.txt) <(sed "s/ - .*//" head-failures.txt)'
```
Expected:
- `comm` prints nothing: no failure is new at the head.
- `passed(head) = passed(base) − P_REMOVED + 17`. `P_REMOVED` is the number of the 16 removed node ids plus
  `N_COPIER` copier ids that passed at the base (read them from `base-run.txt`; `-rfE` lists only failures, so
  passed = collected − failed − skipped per id).
- The skip count changes only by removed tests that skipped at the base.

This comparison is the EXL3 proof. The EXL3 path's code changes are two:
- the fail-stop hook rename, pinned by Review Focus 1;
- the deleted `stop_doorbell` call before the RAM-miss shutdown, pinned by `test_exl3_ram_miss_shutdown.py`'s
  whole-order assertions.

Neither changes a kernel, a graph or an env value the EXL3 gate reads. So `test/registered/unit/kernels`
(its EXL3 RAM-miss kernel tests) and `test_exl3_*.py` passing identically to the base prove the path is
unaffected. **A DSV4.1 A/B is skipped** for that reason: the doorbell was refused on EXL3 before this change, so no
EXL3 launch ever executed a statement this plan removes.

- [ ] **Step 4: Smoke the NVFP4 graph-gather preset on a Qwen3.8 server** (`run_in_background`; needs the whole GPU)

Availability, checked 2026-09-27:
- the Qwen3.8 NVFP4 snapshot at
  `/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47`;
- its expert and PLE cache manifests;
- the seed `expert_freq.pt`;
- the flashinfer overlay;
- the launch script `/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh` (the production graph-gather
  preset: `--moe-offload-preset graph-gather`, NEXTN, breakable decode graph at bs 1).

Re-check them first:
```bash
ssh divix01 'ls -d /mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  /mnt/nvme2/nvfp4-work/qwen38-nvfp4-expert-cache-v1 /mnt/nvme2/ple-cache/qwen38-nvfp4 \
  /data/models/slang/slang-dev-2bit/qwen3.8-flash-next-24gb-sglang/assets/expert_freq.pt \
  /data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh'
```
Build a private copy of the launch script that points at this branch's worktree, on 127.0.0.1:7877 (never
production's 7867), with its run directory under `/mnt/nvme1/tmp-ests`:
```bash
ssh divix01 'set -e; smoke=/mnt/nvme1/tmp-ests/doorbell-removal-smoke; mkdir -p "$smoke/runtime-tmp"
sed -e "s#^worktree=.*#worktree=/data/models/slang/nvfp4-work/wt-doorbell-removal#" \
    -e "s#^work_root=.*#work_root=$smoke#" \
    -e "s#^runtime_tmp=.*#runtime_tmp=$smoke/runtime-tmp#" \
    -e "s#--host 0.0.0.0#--host 127.0.0.1#" -e "s#--port 7867#--port 7877#" \
    /data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh > "$smoke/run.sh"
grep -nE "^worktree=|^work_root=|^runtime_tmp=|--host|--port|moe-offload-preset" "$smoke/run.sh"
cat > "$smoke/smoke.sh" <<'"'"'EOF'"'"'
#!/usr/bin/env bash
# Runs under: flock cc-gpu.lock taskset -c 32-63. Never stops a process it did not start.
set -uo pipefail
smoke=/mnt/nvme1/tmp-ests/doorbell-removal-smoke
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo GPU_BUSY_OUTSIDE_LOCK; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 75
fi
export SGLANG_MOE_EXPERT_DOORBELL=1   # the removed variable, set on purpose: it must only warn
setsid bash "$smoke/run.sh" > "$smoke/driver.out" 2>&1 &
server=$!
ready=0
for _ in $(seq 180); do
  curl --fail --silent --max-time 5 http://127.0.0.1:7877/health >/dev/null && { ready=1; break; }
  kill -0 "$server" 2>/dev/null || break
  sleep 10
done
status=1
if [ "$ready" = 1 ]; then
  model=$(curl -fsS --max-time 10 http://127.0.0.1:7877/v1/models | jq -r ".data[0].id")
  curl -fsS --max-time 300 http://127.0.0.1:7877/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: NVFP4 OK\"}],\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    > "$smoke/reply.json" && status=0
fi
kill -TERM -- "-$server" 2>/dev/null
for _ in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
echo "READY=$ready STATUS=$status"; exit $status
EOF
chmod +x "$smoke/smoke.sh"'
ssh divix01 'flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /mnt/nvme1/tmp-ests/doorbell-removal-smoke/smoke.sh; echo EXIT=$?'
```
Expected:
- the `grep` shows the four rewritten lines and `--moe-offload-preset graph-gather`;
- the last command prints `READY=1 STATUS=0` and `EXIT=0`.

`EXIT=75` means a process holding the GPU outside the lock (for example production). Stop and ask the user for a
window. Do not stop that process.

Then read the evidence:
```bash
ssh divix01 'smoke=/mnt/nvme1/tmp-ests/doorbell-removal-smoke; log=$(ls -t $smoke/run-*/server.log | head -1); echo "$log"
jq -c "{content: .choices[0].message.content, finish: .choices[0].finish_reason}" $smoke/reply.json
sed -n 1,8p "$log" | grep -E "^[0-9a-f]{7,} "
grep -c "Environment variable SGLANG_MOE_EXPERT_DOORBELL is deprecated" "$log" $smoke/driver.out
grep -i doorbell "$log" | grep -v "SGLANG_MOE_EXPERT_DOORBELL is deprecated" | cut -c1-200
grep -m1 "Expert hot cache startup" "$log" | grep -o "\"insert_on_miss_stage\": \"[A-Z_]*\""
grep "MoE offload preset graph-gather sets" "$log" | grep -ci doorbell
grep -E "Traceback|CUDA out of memory|OutOfMemory|Killed" "$log" | head -3'
```
Expected:
- `"content":"NVFP4 OK"`;
- the logged `git log -1` sha equals the branch head;
- at least one deprecation line;
- no other `doorbell` line;
- `"insert_on_miss_stage": "DIRECT"`;
- `0` preset lines naming doorbell;
- no `Traceback` or OOM line.

Clean up with `ssh divix01 'rm -rf /mnt/nvme1/tmp-ests/doorbell-removal-smoke/runtime-tmp'`, and keep the logs.

**Fallback if the smoke cannot run** (the model or caches are gone, or the GPU stays held outside the lock for the
whole window): say so in the report; never claim the smoke. The remaining evidence for "the graph-gather path is
unchanged" is then:
- Step 3's GPU tests that compare gathers byte-for-byte against `7de955329a`'s reference
  (`test_flag_off_gather_matches_the_pre_doorbell_path`, `test_gathers_match_the_pre_doorbell_path_across_*`), which
  build a real `ExpertHotCacheManager.from_model` with graph gather, GPU residency and insert-on-miss, and capture
  and replay CUDA graphs;
- Task 2's `test_offload_presets.py::TestPipelineWiring`, which resolves the graph-gather preset through the real
  argument pipeline.

- [ ] **Step 5: Tidy the divix01 worktrees**

```bash
ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-doorbell-base'
```
Keep `wt-doorbell-removal` until the branch is merged; it is where review re-runs happen.

---

## Follow-ups outside this plan

- **`expert_residency_gpu.py:365-368,404`**: two doorbell strings, deferred to avoid colliding with the concurrent
  layer-fusion split. After both branches merge:
  - reword the `check_miss_plans` bullet to "a backend that posts its own plan buffers";
  - change the message to "each graph gather's own miss plan" without the variable;
  - change `test_expert_residency_gpu.py:1255`'s regex from `DOORBELL_PLAN_CAPACITY` to `own miss plan`;
  - reword that same test's comment at `test_expert_residency_gpu.py:1248-1249` ("the doorbell backend posts one,
    and then its thread owns those slots on another stream") to match, alongside the regex.
- **The divix01 DSV4.1 production launch script** still exports `SGLANG_MOE_EXPERT_DOORBELL=0`. It will warn once
  production runs this code. Remove the line at the next planned restart; this plan does not touch production.
- **The cores 64-71 reservation** keeps "71 is production's doorbell spin core" as its rationale in code messages
  and `.claude/rules/divix01-run-protocol.md`. Whether the reservation should shrink now that nothing spins there is
  the user's decision.
- **The offline pricing model** (`prefetch_pricing.py` `DOORBELL_*`, `doorbell_saving_ms`) models the removed design.
  Keep it or delete it; that is a separate call.

## Self-review record

1. **Spec coverage:**

   | Spec item | Where |
   |---|---|
   | Delete the `.cuh`/`.py` | Task 1 Step 6 |
   | The eight env vars, list verified | Design; Task 2 Step 3 |
   | `DoorbellRowBackend` and backend constants | Task 1 Step 5 |
   | Hot-cache `_start_doorbell`, quiesce/resume/stop, `doorbell_fail_stop_check`, `_log_doorbell`, stats | Task 1 Steps 3-4 |
   | `model_runner.py:766-812` checks | Task 1 Step 5 |
   | Scheduler callers | Task 1 Step 3 |
   | Preset `expert_doorbell` fields, `ENV_NAMES`, rules, and `--moe-offload-preset doorbell` (it exists) | Task 2 Step 4 |
   | `expert_format.py:300`, `expert_stream_requirements.py:327` | Task 1 Step 5; Task 2 Step 4 |
   | Removed-var behaviour decided per the skill | Design; Task 2 |
   | Tests: copier file, graph-gather doorbell tests and others | Task 1 Steps 6-7 |
   | `benchmark/expert_doorbell/` and `benchmark_ingraph_delivery_cost.py` | Task 1 Step 6; Task 3 |
   | Arm pins | Task 2 Step 4 |
   | Whole-repo grep classification | Task 4 Step 1 |
   | Living-doc notes | Task 3 |
   | Shared state: `InGraphRowBackend`, residual path | Task 1 Step 7's rewritten tests, and Review Focus 5 |
   | Verification: divix01 protocol, merge-base baseline, gate tests, Qwen smoke, EXL3 | Tasks 0, 2 and 4 |
   | Execution context: branch/worktree, concurrent-plan overlap | Global Constraints |

   Two spec facts were corrected against the tree:
   - `run_server.md:444` is DSV4.1's current args, not Qwen's.
   - The Qwen launch that uses the graph-gather preset is `run-nvfp4-e16c-public.sh`, not the `run_server.md`
     script `run-nvfp4-expert-dynamic-hot10g.sh`, which sets the variables by hand.
2. **Placeholder scan:** `$SHA` is computed in Task 3 Step 1 and checked absent in Step 3. `SUITE` and
   `R1`-`R3` are defined in Global Constraints. `N_COPIER` is measured in Task 0 Step 3. The node ids name real
   classes (`TestEagerFormatRequirements` holds the two gate tests in `test_expert_stream_requirements.py`).
3. **Type consistency:** `run_fail_stop_checks() -> None` and `_run_expert_fail_stop_checks() -> None` are used
   identically in Task 1 Steps 1 and 3 and in Task 3's note. `check_offload_config` loses `allowed_cpus` in Task 2
   Step 4, and every caller is updated in the same step (`moe_offload_hook.py:87`, `test_offload_presets.check`).
4. **Review Focus:** each of the five lines has its pinning test in the owning task's Step 1, or is an existing test
   Task 1 keeps (item 5).
