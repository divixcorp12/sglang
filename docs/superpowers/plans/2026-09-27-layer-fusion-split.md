# Layer-fusion split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the misnamed `csrc/moe/dsv41_layer_fusion.cuh` and its Python wrapper by owner. The two generic
GPU-residency kernels go to `expert_residency/`, and the EXL3 route-table kernel goes to `exl3/`. Every C++ launcher
checks its own inputs. In the same branch, split `csrc/moe/dsv41_cast_fusion.cuh` (added scope, approved by the
user) into one `exl3/` file per kernel, because its two kernels need opposite fast-math flags. Kernel behaviour and
performance are unchanged.

**Architecture:** The work runs in three phases.
1. **Safety in place (Task 1).** The three launchers verify every tensor with `host::TensorMatcher` plus
   `expert_stream::verify_named`, and enforce the 1-32 width bound in C++. The Python `_check`s are deleted, except
   the three dtype checks that choose a template instantiation.
2. **Moves proven byte for byte.** The C++ split (Task 2) is a comment-only prepare commit followed by a move commit
   that `cxx_move_proof.py` certifies. The move leaves an include-only shim at the old path, so every commit builds.
   The Python split (Task 3) is one `Repro`-certified move that also repoints every importer.
3. **Postpare, which also renames to the new home (Task 4).**
   - The JIT modules build the new headers under owner-named module names.
   - The code moves into the owner namespaces `sglang::expert_residency` and `sglang::exl3`.
   - Two file-local helpers are renamed.
   - The shim is deleted.

   All renames live in this `non_mechanical_provable` commit, never in the proven move.
4. **Cast-fusion split (Task 5).** A pure relocation with no behaviour change, done as prepare, proven C++ move, proven
   Python move, and postpare.
   - `exl3_silu_mul_clamp_half` goes to `exl3/exl3_silu_mul_clamp_half.cuh`, which must build with `-use_fast_math`.
   - `exl3_scale_to_bf16` goes to `exl3/exl3_scale_to_bf16.cuh`, which must never build with fast math.
   - Each file's header comment states its required flag. Today the only record of it is the loaders in
     `ops/moe/dsv41_cast_fusion.py`.
   - The wrapper becomes `ops/moe/exl3_cast_fusion.py`, beside `exl3_route_tables.py`.
   - The cast postpare renames its JIT modules and moves its code into `sglang::exl3`.

Four decisions hold throughout:
- **No route-table adapter template.** There is one expert format today (YAGNI). A second format adds its own
  `csrc/moe/<format>/..._route_tables.cuh` beside the EXL3 one.
- **Names follow the new home (user decision, 2026-09-27).** Every rename lands in a postpare commit (Task 4, Task 5
  Step 7), so each move commit stays a pure, byte-proven relocation.

  | Old | New |
  |---|---|
  | JIT module `dsv41_direct_gather_destinations` | `expert_residency_direct_gather_destinations` |
  | JIT module `dsv41_direct_commit_gather` | `expert_residency_direct_commit_gather` |
  | JIT module `dsv41_exl3_moe_route_tables` | `exl3_moe_route_tables` |
  | JIT module `dsv41_exl3_silu_mul_clamp_half` | `exl3_silu_mul_clamp_half` |
  | JIT module `dsv41_exl3_scale_to_bf16` | `exl3_scale_to_bf16` |
  | namespace `sglang` (the two residency kernels, their launchers, `verify_bool_named`) | `sglang::expert_residency` |
  | namespace `sglang` (route tables, silu-mul-clamp-half, scale-to-bf16) | `sglang::exl3` |
  | `kLayerFusionWarp` | `kDirectGatherWarp` |
  | `layer_fusion_to_float` | `route_tables_to_float` |
  | wrapper `direct_gather_destinations_gpu<...>` / `direct_commit_gather_gpu` | `expert_residency::direct_gather_destinations_gpu<...>` / `expert_residency::direct_commit_gather_gpu` |
  | wrapper `exl3_moe_route_tables_gpu<...>` / `exl3_silu_mul_clamp_half<...>` / `exl3_scale_to_bf16` | the same names, prefixed `exl3::` |

  - **Kernel base names are kept.** None of `direct_gather_destinations_kernel`, `direct_commit_gather_kernel`,
    `exl3_moe_route_tables_kernel`, `exl3_silu_mul_clamp_half_kernel` or `exl3_scale_to_bf16_kernel` misstates its
    owner. The two DIRECT kernels are the generic residency updater's (DIRECT is the residency mode, not a model),
    and the other three carry the correct `exl3_` prefix. What misstated the owner was the file, the module name and
    the namespace. Those change, so a demangled Nsight name moves from `sglang::direct_commit_gather_kernel` to
    `sglang::expert_residency::direct_commit_gather_kernel`.
  - **Tooling.** Nothing in `analysis/`, `benchmark/`, `benchmarks/` or `scripts/` names a JIT module, a helper, or a
    namespace-qualified kernel. The grep in Task 0 Step 2 shows this.
    - The only kernel-name keys are the substring tests in `analysis/dsv41-drive/decode-fusion-2/kernel_groups.py`:
      `:25` tests `"scale_to_bf16"`, and `:35` tests `"direct_gather_destinations"`, `"direct_commit_gather"` and
      `"route_tables"`.
    - A substring test matches a traced name with or without the new namespace prefix, so the one script already
      matches both already-recorded traces (old names) and new ones. It needs no edit, and keeping it unedited
      preserves its reading of old reports.
    - `analysis/dsv41-drive/resident-first/split_launch_bench.py` calls `Exl3FusedMoE._fused_route_tables`, a method
      that does not change.
    - Historical `.md` reports (`DSV41_REFERENCE.md`, the dated plans) stay as they are.
  - **Cold compile.** The renamed JIT modules have no cached build anywhere, so the first load on every machine
    compiles, once per module and per template instantiation.
    - The `cuda_files` changes alone already change every build key, so the renames add no compile that the split
      would not have caused anyway.
    - In production these loads happen in the warmup forward, before CUDA-graph capture, so the one-time cost lands
      in startup, not in decode. Steady-state decode is unchanged.
    - A launch that sets `SGLANG_CRASH_ON_JIT_COMPILE` refuses to compile at runtime, so it needs its JIT cache seeded
      with the five new modules first (Task 6 Step 6 reports this).
- **No compatibility shim at the old Python path.** Every importer is repointed in Task 3. The whole list is 2 production
  files and 2 test files; nothing in `analysis/`, `scripts/` or `benchmarks/` imports the module.
- **`topk_ids` indexes `expert_to_slot` unchecked; this is a documented precondition.**
  - The host cannot read device ids without a sync, and a CUDA-graph capture refuses a sync.
  - The same forward's route planner has already indexed `expert_to_slot` with the same ids
    (`expert_route_plan.cuh:47-48` on the fused path, `expert_stream.py:1352` `index_select` otherwise). A bad id has
    therefore already read out of bounds before this kernel runs, and a check here would guard nothing.
  - A device clamp would make the fused path silently differ from the torch reference, which is what the parity tests
    are built on.

**Cast-fusion device check (controller ruling, fixed in Task 5 Steps 9-11):**
- Both cast-fusion launchers call `device.set_options<kDLCUDA>()` and then the untemplated `.with_device(device)`
  (`dsv41_cast_fusion.cuh:59-61` and `:94-96`). That resets the allowed-device options to empty, so today the
  launchers accept a tensor on any device (the `bdcf769eb8` footgun).
- The moves carry this code verbatim. The fix is its own TDD pair after the cast postpare: a red test commit, then a
  `non_mechanical_provable` fix commit. It never rides with a move.
- The correct idiom is `.with_device<kDLCUDA>(device)`, as in `expert_stream/lease_kernels.cuh:906-920`. Do **not**
  copy `route_quant_fused.cuh`: its `:84-96` (and `route_radix.cuh:604-615`) use the same broken
  `set_options` + untemplated `.with_device(device)` form. Both files are outside this plan; see Follow-ups.
- The launchers this plan writes in Task 1, which reach `expert_residency/direct_gather.cuh` and
  `exl3/exl3_route_tables.cuh` through the move, all use `.with_device<kDLCUDA>(device)`. Task 1 Step 5 and Task 4
  Step 3 grep that no untemplated `with_device(` remains in them.

**Tech Stack:** CUDA C++20 (nvcc, JIT via `load_jit`), TVM FFI, `sgl_kernel/tensor.h` (`TensorMatcher`, `SymbolicSize`,
`SymbolicDevice`), Python 3 + PyTorch, pytest, pre-commit (clang-format, ruff-format, isort), the
`mechanical-refactor-verify` skill's `Repro`, and `analysis/expert-stream-split/cxx_move_proof.py`.

**Spec:** No spec file. The spec is the controller's 2026-09-27 request ("`dsv41_layer_fusion.cuh` is misnamed ...
Agreed direction ..."), restated in the Architecture section above and the Global Constraints below. Read
`.claude/skills/mechanical-refactor-verify/SKILL.md` and `guide-split.md`, and `.claude/rules/divix01-run-protocol.md`,
before starting.

## Global Constraints

- **Base commit:** `29c7d4e2b1` (master, 2026-09-27). Every line number in this plan refers to it. Task 0 checks that
  nothing has moved.
- **Branch and worktrees:**
  - Branch `layer-fusion-split`, created from master.
  - Laptop worktree `/home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split`.
  - divix01 worktree `/data/models/slang/nvfp4-work/wt-layer-fusion-split` (detached, updated by fetch + checkout).
- **Concurrent plan.** The doorbell-copier removal runs at the same time. It owns `expert_hot_cache.py`,
  `expert_row_plan.py`, `model_runner.py`, `scheduler.py`, `offload_presets.py`, `environ.py`,
  `ops/moe/expert_doorbell.py`, `expert_doorbell.cuh` and their tests. Do not touch any of them.
  - **No file is edited by both plans.** The one file of this plan that touches the doorbell's area is
    `python/sglang/srt/layers/moe/expert_residency_gpu.py`, where this plan changes only the two function-local import
    lines `:793` and `:872`. The doorbell plan (`docs/superpowers/plans/2026-09-27-doorbell-removal.md`) explicitly
    defers its own edits there (`:365-368`, `:404`) to avoid this plan, so the file is changed only here.
  - **A test file edited by one plan and run by the other:**
    `test/registered/unit/layers/moe/test_expert_residency_gpu.py`. The doorbell plan edits its `:171-356`; this plan
    only runs it in `SUITE_UNIT`. Compare this plan's counts with its own merge-base baseline (Task 0), never with a
    tree that includes the doorbell branch.
  - **`python/sglang/srt/layers/quantization/exl3.py`** (Task 5 edits its import lines `:230` and `:434` only) is not
    in the doorbell plan's file list. `grep -n "quantization/exl3.py" docs/superpowers/plans/2026-09-27-doorbell-removal.md`
    finds no match; its only `exl3` hits are `exl3_fused_moe.py` (listed as this plan's) and
    `test_expert_stream_requirements_exl3.py`, which this plan does not touch.
- **Kernel bodies are frozen.** No `__global__` function body, launch geometry (`LaunchKernel(...)` grid, block,
  stream) or kernel argument list changes in any task. Task 1 adds checks *before* each launch. Task 2 changes two
  comment lines and relocates text.
- **FFI surface frozen, apart from the listed renames:**
  - the export name `run` and each launcher's parameter list and order. The wrapper strings change only as the
    Architecture rename table lists, and only in the postpares;
  - the template parameters `IdT, RemapInT, RemapOutT` and `RemapT, WeightT, XT`;
  - `SGLANG_DSV41_ENABLE_LAYER_FUSION` and `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION`, with no env var added or renamed;
  - the cast-fusion build flags: `-use_fast_math` for the silu module only, and none for the scale module.
- **Error messages:**
  - No test matches any message of the old wrapper. `git grep` over `test/` finds none of "must be a contiguous CUDA
    tensor", "must hold", "1-32", "go together".
  - The three dtype messages that stay in Python keep their exact text: `f"{name} must be {dtypes}, got {tensor.dtype}"`.
  - Four messages move to C++ byte-identical: `the shortlist and the routes must hold 1-32 entries`,
    `the commit must cover 1-32 lanes`, `delivered and keep go together`, `remap must hold 1-32 routes`.
- **`TensorMatcher::with_device`:** always call it with template codes, as `.with_device<kDLCUDA>(device)`. The
  untemplated `.with_device(device)` calls `set_options<>()` with no codes, which clobbers the device options and
  accepts any device (fixed once already in `bdcf769eb8`). `route_quant_fused.cuh` and `route_radix.cuh` show the
  broken form; `expert_stream/lease_kernels.cuh` shows the correct one.
- **Commit classification** (`guide-split.md` section 1.1):
  - Subject format: `layer-fusion-split(<commit-id>,<kind>): <message>`.
  - `<kind>` is exactly `mechanical_provable` (pure relocations, each with a PASSing proof) or
    `non_mechanical_provable` (everything else).
  - Never amend or rebase a pushed commit. A move commit whose proof fails before it is pushed is redone with
    `git reset --soft HEAD~1`.
- **Commit trailer:** end every commit message with these two lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH
  ```
- **Repo rules:**
  - `.claude/rules/comment-style.md`: comment lines of at most 120 columns, because clang-format's Google style
    reflows longer ones; Doxygen `\brief` on each launcher.
  - `.claude/rules/unit-test-admission.md`: each new test case names the diff that turns it red.
  - `.claude/rules/general-code-style.md`: keyword arguments in Python. The FFI `.run(...)` calls stay positional,
    because TVM FFI takes positional arguments.
- **Stage files by name.** Never `git add -A`: the untracked proof folder must not ride into a code commit.
- **Out of scope** (do not edit):
  - `DSV41_REFERENCE.md` and `docs/superpowers/plans/2026-09-2*.md`, which are dated records of their day;
  - `analysis/dsv41-drive/decode-fusion-2/kernel_groups.py`. Its substring keys (`direct_gather_destinations`,
    `direct_commit_gather` and `route_tables` at `:35`, `scale_to_bf16` at `:25`) match both the old and the
    namespace-qualified kernel names, so it reads old and new traces alike;
  - `test/registered/unit/layers/quantization/test_exl3_cast_fusion_cpu.py`. Its docstring (`:5`) names the test
    file `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py`, which keeps its name because it tests the
    `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION` feature; it never names the moved module, so the reference stays true;
  - `csrc/moe/expert_stream/tensor_checks.h`, which is used as it is.
- **Running code** (`.claude/rules/divix01-run-protocol.md`):
  - Commit, push, and run in the divix01 worktree. Never rsync or scp.
  - Set `PYTHONPATH=$PWD/python`, and print `sglang.__file__` once.
  - Read `${PIPESTATUS[0]}` after every piped pytest.
  - GPU work runs as `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 ...` with
    `TMPDIR=/mnt/nvme1/tmp-ests`.
  - Never run anything in `cc-expert-prediction/dsv41-direct-prod`.
- **Suites**, defined once and referenced by name. Run them on divix01 from the worktree root, after these assignments:

  ```bash
  cd /data/models/slang/nvfp4-work/wt-layer-fusion-split
  PY=/data/models/slang/.venv/bin/python
  LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
  EXL3_ENV="SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build CUDA_HOME=/usr/local/cuda-13.2"

  # SUITE_UNIT: the registered kernels suite plus the registered tests of the calling modules
  env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests flock $LOCK taskset -c 32-63 \
    $PY -m pytest -q -p no:randomly \
    test/registered/unit/kernels \
    test/registered/unit/layers/moe/test_expert_residency_gpu.py \
    test/registered/unit/layers/quantization/test_exl3_fused_moe.py \
    test/registered/unit/layers/quantization/test_exl3_cast_fusion_cpu.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"

  # SUITE_GPU: the manual device tests of the five kernels (parity, CUDA graph, production Exl3FusedMoE path)
  env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests $EXL3_ENV flock $LOCK taskset -c 32-63 \
    $PY -m pytest -q -p no:randomly \
    test/manual/dsv41/test_dsv41_layer_fusion_gpu.py \
    test/manual/dsv41/test_exl3_moe_split_parity_cuda.py \
    test/manual/dsv41/test_exl3_graph_apply_gpu.py \
    test/manual/dsv41/test_dsv41_cast_fusion_gpu.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
  ```

  From Task 1 Step 6 on, `SUITE_GPU` also includes `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py`,
  appended to its file list. A suite passes when `EXIT=0` and its counts equal Task 0's baseline plus exactly the
  tests the task adds. Record the command and the counts in the commit body of the task's last commit, or, for a task
  whose last commit is already pushed, in the next commit.
- **Updating the divix01 worktree** after a push:

  ```bash
  git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split push origin layer-fusion-split
  ssh divix01 'git -C /data/models/slang/sglang fetch origin && \
    git -C /data/models/slang/nvfp4-work/wt-layer-fusion-split checkout --detach origin/layer-fusion-split && \
    git -C /data/models/slang/nvfp4-work/wt-layer-fusion-split log -1 --oneline'
  ```

## Review Focus

These inputs follow from the spec, but no existing test exercises them. Each reaches a launcher only when a caller
bypasses the Python wrapper, which the old code trusted to check. Task 1 owns all five tests, in
`test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py`.

1. **Width or route count above 32, passed directly.**
   - Expected: `direct_gather_destinations_gpu`, `direct_commit_gather_gpu` and `exl3_moe_route_tables_gpu` raise
     before launch.
   - The risk: the commit kernel's `bool live[32]` / `evicted[32]` / `int64_t old_expert[32]` would be overrun.
2. **A tensor on the CPU where the kernel dereferences a device pointer.**
   - Expected: refused, with the tensor's name first in the message (`^victims: `, `^destinations: `, `^x: `).
3. **A dtype that does not match the loaded instantiation**, for example int64 ids given to an `int32_t` module, or
   fp16 weights given to a `bf16_t` module.
   - Expected: refused by name. The risk: the kernel would read the ids' bytes as the wrong type.
4. **Companion tensors whose sizes disagree:**
   - `remap_in` shorter than `topk_ids`;
   - `slot_state` sized unlike `slot_to_expert`;
   - a `num_experts` that is not `mapping.size(0) - 1`;
   - `det` passed flat instead of `[3, slots + 1]`;
   - `x16_out` with the wrong hidden size.

   Expected: refused by name, or by the consistency message.
5. **`delivered` without `keep`** (the leased pair half-present). Expected: `delivered and keep go together`, as the
   Python wrapper said before.

---

## File structure (end state)

| Path | Responsibility |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` | **new.** In `sglang::expert_residency` (from Task 4): `kDirectGatherWarp`, `direct_gather_destinations_kernel`, `direct_commit_gather_kernel`, `verify_bool_named`, and their two checked launchers. Generic GPU-residency bookkeeping. |
| `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh` | **new.** In `sglang::exl3` (from Task 4): `route_tables_to_float`, `exl3_moe_route_tables_kernel` and its checked launcher. EXL3-specific. |
| `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` | **new.** `direct_gather_destinations`, `direct_commit_gather` and their JIT loaders. |
| `python/sglang/kernels/ops/moe/exl3_route_tables.py` | **new.** `exl3_moe_route_tables` and its JIT loader. |
| `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` | **new.** Direct-call refusal tests (the Review Focus). |
| `analysis/layer-fusion-split/proof/` | **new.** The two C++ manifests, the four proof scripts, and the copied `mechanical_refactor_reproduction_utils.py`. |
| `python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh` | **new (T5).** `exl3_silu_mul_clamp_half_kernel` and its launcher. Must build with `-use_fast_math`. |
| `python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh` | **new (T5).** `exl3_scale_to_bf16_kernel` and its launcher. Must never build with fast math. |
| `test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py` | **new (T5).** Direct-call CPU-tensor refusal tests for the two cast-fusion launchers. |
| `python/sglang/kernels/ops/moe/exl3_cast_fusion.py` | **new (T5).** `exl3_silu_mul_clamp_half`, `exl3_scale_to_bf16` and their loaders (`_silu_module` with `-use_fast_math`, `_scale_module` without). |
| `python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh` | **deleted** (T5). An include-only shim between the cast move and the cast postpare. |
| `python/sglang/kernels/ops/moe/dsv41_cast_fusion.py` | **deleted** (T5). |
| `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh` | **deleted** (Task 4). Between Tasks 2 and 4 it is an include-only shim. |
| `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py` | **deleted** (Task 3). |

Commit chain (13 commits; every code commit is covered by a suite run before the next task starts):

| # | Task | Subject |
|---|---|---|
| 1 | T1 | `layer-fusion-split(launcher-checks-red,non_mechanical_provable): the layer-fusion launchers must refuse bad inputs on their own (red)` |
| 2 | T1 | `layer-fusion-split(launcher-checks,non_mechanical_provable): the layer-fusion launchers check every tensor in C++` |
| 3 | T2 | `layer-fusion-split(cxx-prepare,non_mechanical_provable): drop the DSV4.1 file comment and the stale width note` |
| 4 | T2 | `layer-fusion-split(cxx-move,mechanical_provable): split dsv41_layer_fusion.cuh into expert_residency/ and exl3/` |
| 5 | T3 | `layer-fusion-split(python-move,mechanical_provable): split the layer-fusion ops module by owner and repoint its importers` |
| 6 | T4 | `layer-fusion-split(postpare,non_mechanical_provable): rename the layer-fusion code to its new home; drop the include shim` |
| 7 | T5 | `layer-fusion-split(cast-prepare,non_mechanical_provable): drop the cast-fusion file comment` |
| 8 | T5 | `layer-fusion-split(cast-cxx-move,mechanical_provable): split dsv41_cast_fusion.cuh into one exl3/ file per kernel` |
| 9 | T5 | `layer-fusion-split(cast-python-move,mechanical_provable): move the cast-fusion ops module to exl3_cast_fusion.py and repoint its importers` |
| 10 | T5 | `layer-fusion-split(cast-postpare,non_mechanical_provable): rename the cast-fusion code to its new home; drop the include shim` |
| 11 | T5 | `layer-fusion-split(cast-device-check-red,non_mechanical_provable): the cast-fusion launchers must refuse a CPU tensor (red)` |
| 12 | T5 | `layer-fusion-split(cast-device-check,non_mechanical_provable): the cast-fusion launchers restrict tensors to CUDA` |
| 13 | T6 | `layer-fusion-split(proofs,non_mechanical_provable): move proofs for the layer-fusion and cast-fusion splits` |

---

### Task 0: Branch, worktrees, baseline

**Files:** none.

**Interfaces:**
- Produces: the branch `layer-fusion-split`, the two worktrees, and `/mnt/nvme1/layer-fusion-split/baseline.txt` on
  divix01 (the `SUITE_UNIT` and `SUITE_GPU` counts at the merge-base, with their commands).

- [ ] **Step 1: Create the branch and the laptop worktree** (superpowers:using-git-worktrees)

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4 worktree add -b layer-fusion-split \
  /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split master
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split log -1 --oneline
```
Expected: `29c7d4e2b1 merge(expert-stream): T3 no longer double-signals a served lane (test-only race fix)`, or a
later master.

- [ ] **Step 2: Prove the line numbers still hold**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
git diff --stat 29c7d4e2b1 HEAD -- \
  python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh python/sglang/kernels/ops/moe/dsv41_layer_fusion.py \
  python/sglang/srt/layers/moe/expert_residency_gpu.py python/sglang/srt/layers/quantization/exl3_fused_moe.py \
  test/manual/dsv41/test_dsv41_layer_fusion_gpu.py python/sglang/kernels/jit/csrc/moe/expert_stream/tensor_checks.h \
  python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh python/sglang/kernels/ops/moe/dsv41_cast_fusion.py \
  python/sglang/srt/layers/quantization/exl3.py test/manual/dsv41/test_dsv41_cast_fusion_gpu.py
git grep -ln "dsv41_layer_fusion" -- python test scripts benchmarks analysis
git grep -ln "dsv41_cast_fusion" -- python test scripts benchmarks analysis
git grep -nE "dsv41_direct_|dsv41_exl3_|layer_fusion_to_float|kLayerFusionWarp|sglang::(direct_|exl3_)|direct_gather_destinations|direct_commit_gather|route_tables|silu_mul_clamp_half|scale_to_bf16" \
  -- analysis benchmark benchmarks scripts ':!*.md'
```
Expected:
- The diff is empty. If it is not, re-derive the line references in Tasks 1-4 from the quoted anchors before going on.
- The grep lists exactly these four files:
  - `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py` (the module itself, whose `cuda_files` name the `.cuh`)
  - `python/sglang/srt/layers/moe/expert_residency_gpu.py`
  - `python/sglang/srt/layers/quantization/exl3_fused_moe.py`
  - `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`

  Any fifth file is an importer this plan missed: add it to Task 3's repoint table and to its `Repro`.
- The second grep lists exactly these four files, and any other is an importer Task 5 missed:
  - `python/sglang/kernels/ops/moe/dsv41_cast_fusion.py`
  - `python/sglang/srt/layers/quantization/exl3.py`
  - `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py`
  - `test/registered/unit/layers/quantization/test_exl3_cast_fusion_cpu.py` (a docstring naming the GPU *test file*,
    which is not renamed; no edit is needed)
- The third grep prints exactly the seven lines checked on 2026-09-27:
  - `analysis/dsv41-drive/decode-fusion-2/kernel_groups.py:25` and `:35`, the substring keys, which need no edit (see
    Architecture, "Tooling");
  - `analysis/dsv41-drive/resident-first/split_launch_bench.py:12,128,129,164,167,168`, which name
    `split_route_tables` and `_fused_route_tables`, Python methods this plan does not rename.

  Any other hit is tooling keyed on an old name. Add it to Task 4, or to Task 5 Step 7, with a match on both the old
  and the new name.

- [ ] **Step 3: Create the divix01 worktree and check the interpreter**

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split push -u origin layer-fusion-split
ssh divix01 'git -C /data/models/slang/sglang fetch origin && \
  git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-layer-fusion-split \
    origin/layer-fusion-split && \
  mkdir -p /mnt/nvme1/tmp-ests /mnt/nvme1/layer-fusion-split && \
  cd /data/models/slang/nvfp4-work/wt-layer-fusion-split && git log -1 --oneline && \
  PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"'
```
Expected: the printed path lies under `/data/models/slang/nvfp4-work/wt-layer-fusion-split/python/`.

- [ ] **Step 4: Baseline at the merge-base**

On divix01, in the worktree, run `SUITE_UNIT` and then `SUITE_GPU` (Global Constraints). Write both summary lines,
both `EXIT=` values, both commands and `git log -1 --oneline` into `/mnt/nvme1/layer-fusion-split/baseline.txt`.

Expected: both `EXIT=0`. If `test_exl3_moe_split_parity_cuda.py` skips or errors for an environmental reason (for
example, a missing checkpoint under `/mnt/nvme2`), record that as part of the baseline; later tasks must reproduce the
same outcome, not fix it. A non-zero exit whose cause is not environmental stops the plan: ask the user.

---

### Task 1: Checked launchers (in place)

A semantic commit reviewed for correctness, placed before the moves so that the moves carry already-checked code
(`guide-split.md` section 2.2: "a large but honestly-labeled reshape placed before the move is legitimate"). The
Python `_check` helper disappears here, which leaves the Python move in Task 3 no shared function to duplicate.

**Files:**
- Create: `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py`
- Modify: `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh` (includes at `:1-12`; the three launchers at
  `:215-322`; a new helper inserted after `:213`)
- Modify: `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py` (delete `:25` `MAX_WIDTH`, delete `:61-68` `_check`,
  and replace the checks at `:89-101`, `:141-162` and `:202-216`)

**Interfaces:**
- Consumes: `sglang::expert_stream::verify_named(std::string_view, host::TensorMatcher&&, tvm::ffi::TensorView)` from
  `csrc/moe/expert_stream/tensor_checks.h`.
- Produces:
  - `sglang::verify_bool_named(const char* name, host::TensorMatcher&& matcher, tvm::ffi::TensorView view)`.
  - The three launchers keep their signatures and now raise before launch on bad input.
  - The Python module keeps `_gather_module(id_dtype, remap_in, remap_out)`, `_commit_module()`,
    `_route_tables_module(remap, weight, x)` and the three public wrappers with unchanged signatures.
  - `_ID_DTYPES` and `_FLOAT_DTYPES` stay at module level. `MAX_WIDTH` and `_check` are gone.

- [ ] **Step 1: Write the failing tests**

Create `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py`:

```python
"""The layer-fusion launchers refuse bad inputs on their own, without the Python wrappers' help.

Each case calls a JIT module's ``run`` directly -- what a caller that skips the wrapper does -- with one input wrong and
every other input valid, and expects the C++ launcher to raise before it launches. Deleting the check that raises the
case's ``match`` from the launcher turns that case red: the kernel would launch on the bad input instead (a device
read of a host pointer, a read of mistyped bytes, or, for a width past 32, an overrun of the commit kernel's 32-entry
lane arrays).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

CUDA = "cuda"


def _gather_args(width: int = 6, routes: int = 6, experts: int = 16) -> dict:
    """Valid inputs for ``direct_gather_destinations_gpu<int32_t, int32_t, int32_t>``, in FFI parameter order."""
    return {
        "topk_ids": (torch.arange(routes) % experts).to(device=CUDA, dtype=torch.int32),
        "expert_to_slot": torch.full((experts,), -1, dtype=torch.int64, device=CUDA),
        "victims": torch.arange(width, dtype=torch.int64, device=CUDA),
        "victim_valid": torch.ones(width, dtype=torch.bool, device=CUDA),
        "miss_count": torch.zeros(1, dtype=torch.int32, device=CUDA),
        "remap_in": torch.arange(routes, dtype=torch.int32, device=CUDA),
        "scratch_base": 64,
        "destination_slots_out": torch.zeros(width, dtype=torch.int32, device=CUDA),
        "destinations_out": torch.zeros(width, dtype=torch.int64, device=CUDA),
        "live_out": torch.zeros(width, dtype=torch.bool, device=CUDA),
        "remap_out": torch.zeros(routes, dtype=torch.int32, device=CUDA),
    }


def _commit_args(width: int = 6, experts: int = 16, slots: int = 8) -> dict:
    """Valid inputs for ``direct_commit_gather_gpu`` (leased form), in FFI parameter order."""
    return {
        "destinations": torch.arange(width, dtype=torch.int64, device=CUDA) % slots,
        "live": torch.zeros(width, dtype=torch.bool, device=CUDA),
        "new_experts": torch.arange(width, dtype=torch.int64, device=CUDA),
        "num_experts": experts,
        "slot_dump": slots,
        "mapping": torch.full((experts + 1,), -1, dtype=torch.int64, device=CUDA),
        "slot_to_expert": torch.full((slots + 1,), -1, dtype=torch.int64, device=CUDA),
        "slot_state": torch.zeros(slots + 1, dtype=torch.uint8, device=CUDA),
        "slot_generations": torch.zeros(slots + 1, dtype=torch.int64, device=CUDA),
        "gather_insertions": torch.zeros(1, dtype=torch.int64, device=CUDA),
        "gather_evictions": torch.zeros(1, dtype=torch.int64, device=CUDA),
        "insertion_truncated": torch.zeros(1, dtype=torch.int64, device=CUDA),
        "delivered": torch.zeros(1, dtype=torch.int32, device=CUDA),
        "keep": torch.ones(1, dtype=torch.float32, device=CUDA),
        "miss_count": torch.zeros(1, dtype=torch.int32, device=CUDA),
        "ready": 3,
        "free_state": 0,
    }


def _route_args(routes: int = 6, slots: int = 12, hidden: int = 64) -> dict:
    """Valid inputs for ``exl3_moe_route_tables_gpu<int32_t, bf16_t, bf16_t>``, in FFI parameter order."""
    return {
        "remap": torch.arange(routes, dtype=torch.int32, device=CUDA),
        "weights": torch.ones(routes, dtype=torch.bfloat16, device=CUDA),
        "keep": torch.ones(1, dtype=torch.float32, device=CUDA),
        "x": torch.zeros(1, hidden, dtype=torch.bfloat16, device=CUDA),
        "remap64_out": torch.zeros(routes, dtype=torch.int64, device=CUDA),
        "x16_out": torch.zeros(1, hidden, dtype=torch.float16, device=CUDA),
        "out_zero": torch.zeros(1, hidden, dtype=torch.float32, device=CUDA),
        "expert_count": torch.zeros(slots + 1, dtype=torch.int64, device=CUDA),
        "inv_order": torch.zeros(routes, dtype=torch.int64, device=CUDA),
        "weight_sorted": torch.zeros(routes, dtype=torch.float16, device=CUDA),
        "det": torch.zeros(3, slots + 1, dtype=torch.int64, device=CUDA),
    }


def _run_gather(args: dict) -> None:
    from sglang.kernels.ops.moe.dsv41_layer_fusion import _gather_module

    _gather_module(torch.int32, torch.int32, torch.int32).run(*args.values())


def _run_commit(args: dict) -> None:
    from sglang.kernels.ops.moe.dsv41_layer_fusion import _commit_module

    _commit_module().run(*args.values())


def _run_route(args: dict) -> None:
    from sglang.kernels.ops.moe.dsv41_layer_fusion import _route_tables_module

    _route_tables_module(torch.int32, torch.bfloat16, torch.bfloat16).run(*args.values())


GATHER_REFUSALS = {
    "width_past_32": (lambda: _gather_args(width=33), "the shortlist and the routes must hold 1-32 entries"),
    "routes_past_32": (lambda: _gather_args(routes=33), "the shortlist and the routes must hold 1-32 entries"),
    "victims_on_cpu": (lambda: {**_gather_args(), "victims": torch.arange(6, dtype=torch.int64)}, "^victims: "),
    "topk_ids_int64_into_int32": (
        lambda: {**_gather_args(), "topk_ids": torch.arange(6, dtype=torch.int64, device=CUDA)},
        "^topk_ids: ",
    ),
    "remap_in_shorter_than_routes": (
        lambda: {**_gather_args(), "remap_in": torch.arange(5, dtype=torch.int32, device=CUDA)},
        "^remap_in: ",
    ),
    "victim_valid_not_bool": (
        lambda: {**_gather_args(), "victim_valid": torch.ones(6, dtype=torch.uint8, device=CUDA)},
        "victim_valid: must be a bool tensor",
    ),
}

COMMIT_REFUSALS = {
    "width_past_32": (lambda: _commit_args(width=33), "the commit must cover 1-32 lanes"),
    "delivered_without_keep": (lambda: {**_commit_args(), "keep": None}, "delivered and keep go together"),
    "slot_state_sized_unlike_slot_to_expert": (
        lambda: {**_commit_args(), "slot_state": torch.zeros(8, dtype=torch.uint8, device=CUDA)},
        "^slot_state: ",
    ),
    "num_experts_not_mapping_minus_dump": (
        lambda: {**_commit_args(), "num_experts": 17},
        "num_experts must be mapping's size minus the dump column",
    ),
    "destinations_on_cpu": (
        lambda: {**_commit_args(), "destinations": torch.arange(6, dtype=torch.int64)},
        "^destinations: ",
    ),
    "keep_not_float32": (
        lambda: {**_commit_args(), "keep": torch.ones(1, dtype=torch.int32, device=CUDA)},
        "^keep: ",
    ),
}

ROUTE_REFUSALS = {
    "routes_past_32": (lambda: _route_args(routes=33), "remap must hold 1-32 routes"),
    "x_on_cpu": (lambda: {**_route_args(), "x": torch.zeros(1, 64, dtype=torch.bfloat16)}, "^x: "),
    "det_flattened": (
        lambda: {**_route_args(), "det": torch.zeros(3 * 13, dtype=torch.int64, device=CUDA)},
        "^det: ",
    ),
    "weights_fp16_into_bf16": (
        lambda: {**_route_args(), "weights": torch.ones(6, dtype=torch.float16, device=CUDA)},
        "^weights: ",
    ),
    "x16_out_wrong_hidden": (
        lambda: {**_route_args(), "x16_out": torch.zeros(1, 65, dtype=torch.float16, device=CUDA)},
        "^x16_out: ",
    ),
}


@pytest.mark.parametrize("case", list(GATHER_REFUSALS))
def test_gather_launcher_refuses(case):
    make, match = GATHER_REFUSALS[case]
    with pytest.raises(Exception, match=match):
        _run_gather(make())


@pytest.mark.parametrize("case", list(COMMIT_REFUSALS))
def test_commit_launcher_refuses(case):
    make, match = COMMIT_REFUSALS[case]
    with pytest.raises(Exception, match=match):
        _run_commit(make())


@pytest.mark.parametrize("case", list(ROUTE_REFUSALS))
def test_route_tables_launcher_refuses(case):
    make, match = ROUTE_REFUSALS[case]
    with pytest.raises(Exception, match=match):
        _run_route(make())
```

The JIT imports are function-scoped on purpose: Task 3's `repath_import` primitive repaths only nested imports.

- [ ] **Step 2: Commit the red tests and run the safe subset red on divix01**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
git add test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py
git commit -m "layer-fusion-split(launcher-checks-red,non_mechanical_provable): the layer-fusion launchers must refuse bad inputs on their own (red)

Direct calls into the three JIT modules, one bad input each (Review Focus 1-5 of
docs/superpowers/plans/2026-09-27-layer-fusion-split.md). Red until the next commit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
Push and update the divix01 worktree (Global Constraints). Then, on divix01, in the worktree:
```bash
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests flock /data/models/slang/nvfp4-work/cc-gpu.lock \
  taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
  test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py -k "past_32 and not commit" 2>&1 | tail -5; \
  echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `3 failed`, each `Failed: DID NOT RAISE`, and `EXIT=1`.

Only these three cases run red. Their unchecked launches stay in bounds: 32 lanes read the first 32 entries, and the
remap stays below `scratch_base`. The others, unchecked, would dereference a host pointer or overrun the commit
kernel's local arrays, which leaves a sticky CUDA error that fails every later case in the process.

- [ ] **Step 3: Check the launchers in C++**

In `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh`:

(a) After `#include <stdint.h>` (`:12`), add:

```cpp

#include <type_traits>
#include <utility>

#include "expert_stream/tensor_checks.h"
```

(b) Between the end of `exl3_moe_route_tables_kernel` (`:213`, its closing `}`) and
`template <typename IdT, typename RemapInT, typename RemapOutT>` (`:215`), insert the helper. Its first line is Task 2's
anchor, so keep it verbatim:

```cpp

/// \brief `verify_named` for a tensor the kernel reads as `bool` (`kDLBool` has no dtype trait).
inline void verify_bool_named(const char* name, host::TensorMatcher&& matcher, tvm::ffi::TensorView view) {
  expert_stream::verify_named(name, std::move(matcher), view);
  host::RuntimeCheck(view.dtype().code == kDLBool && view.dtype().bits == 8, name, ": must be a bool tensor");
}
```

(c) Put this doc block directly above the gather launcher's `template <typename IdT, typename RemapInT, typename
RemapOutT>` line (`:215`); the template line itself stays. The parameter list (`:216-227`) and everything from
`const auto stream = host::LaunchKernel::resolve_device(topk_ids.device());` (`:228`) to the closing `}` (`:244`)
stay unchanged.

```cpp
/// \brief Checked launcher for `direct_gather_destinations_kernel`: one layer's DIRECT gather destinations.
///
/// Precondition, not checked: every `topk_ids` entry indexes `expert_to_slot`. Reading the ids would need a device
/// sync, which a graph capture refuses, and the route planner indexes `expert_to_slot` with the same ids earlier in
/// the forward (`expert_route_plan.cuh`); a bad id has already read out of bounds by the time this launch runs.
```
Then insert these as the first statements of the body, before `const auto stream = ...` (`:228`):
```cpp
  using namespace host;
  static_assert(
      (std::is_same_v<IdT, int32_t> || std::is_same_v<IdT, int64_t>) &&
          (std::is_same_v<RemapInT, int32_t> || std::is_same_v<RemapInT, int64_t>) &&
          (std::is_same_v<RemapOutT, int32_t> || std::is_same_v<RemapOutT, int64_t>),
      "direct_gather_destinations: ids and remaps are int32 or int64");
  auto K_ = SymbolicSize{"routes"};
  auto W_ = SymbolicSize{"width"};
  auto device = SymbolicDevice{};
  expert_stream::verify_named(
      "topk_ids", TensorMatcher({K_}).with_dtype<IdT>().with_device<kDLCUDA>(device), topk_ids);
  expert_stream::verify_named(
      "expert_to_slot", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), expert_to_slot);
  expert_stream::verify_named(
      "victims", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), victims);
  verify_bool_named("victim_valid", TensorMatcher({W_}).with_device<kDLCUDA>(device), victim_valid);
  expert_stream::verify_named(
      "miss_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), miss_count);
  expert_stream::verify_named(
      "remap_in", TensorMatcher({K_}).with_dtype<RemapInT>().with_device<kDLCUDA>(device), remap_in);
  expert_stream::verify_named(
      "destination_slots_out",
      TensorMatcher({W_}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
      destination_slots_out);
  expert_stream::verify_named(
      "destinations_out", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), destinations_out);
  verify_bool_named("live_out", TensorMatcher({W_}).with_device<kDLCUDA>(device), live_out);
  expert_stream::verify_named(
      "remap_out", TensorMatcher({K_}).with_dtype<RemapOutT>().with_device<kDLCUDA>(device), remap_out);
  RuntimeCheck(
      0 < W_.unwrap() && W_.unwrap() <= kLayerFusionWarp && 0 < K_.unwrap() && K_.unwrap() <= kLayerFusionWarp,
      "the shortlist and the routes must hold 1-32 entries");
```

(d) Put this doc block directly above `void direct_commit_gather_gpu(` (`:246`). Its first line is an anchor:

```cpp
/// \brief Checked launcher for `direct_commit_gather_kernel`: one layer's DIRECT residency commit.
///
/// The width bound keeps the kernel's 32-entry lane arrays in range. Precondition, not checked (it would need a
/// device sync): every `destinations` entry indexes `slot_to_expert`, and every live lane's `new_experts` entry
/// indexes `mapping`; the gather kernel and the planner produce both.
```
and insert these as the first statements of its body (before `const auto stream = ...` at `:264`):
```cpp
  using namespace host;
  auto W_ = SymbolicSize{"width"};
  auto E_ = SymbolicSize{"mapping_columns"};
  auto S_ = SymbolicSize{"slot_columns"};
  auto device = SymbolicDevice{};
  expert_stream::verify_named(
      "destinations", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), destinations);
  verify_bool_named("live", TensorMatcher({W_}).with_device<kDLCUDA>(device), live);
  expert_stream::verify_named(
      "new_experts", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), new_experts);
  expert_stream::verify_named(
      "mapping", TensorMatcher({E_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), mapping);
  expert_stream::verify_named(
      "slot_to_expert", TensorMatcher({S_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), slot_to_expert);
  expert_stream::verify_named(
      "slot_state", TensorMatcher({S_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device), slot_state);
  expert_stream::verify_named(
      "slot_generations", TensorMatcher({S_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), slot_generations);
  expert_stream::verify_named(
      "gather_insertions", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), gather_insertions);
  expert_stream::verify_named(
      "gather_evictions", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), gather_evictions);
  expert_stream::verify_named(
      "insertion_truncated",
      TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device),
      insertion_truncated);
  RuntimeCheck(delivered.has_value() == keep.has_value(), "delivered and keep go together");
  if (delivered.has_value()) {
    expert_stream::verify_named(
        "delivered", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), delivered.value());
    expert_stream::verify_named(
        "keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep.value());
  }
  expert_stream::verify_named(
      "miss_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), miss_count);
  RuntimeCheck(0 < W_.unwrap() && W_.unwrap() <= kLayerFusionWarp, "the commit must cover 1-32 lanes");
  RuntimeCheck(num_experts == E_.unwrap() - 1, "num_experts must be mapping's size minus the dump column");
  RuntimeCheck(slot_dump == S_.unwrap() - 1, "slot_dump must be slot_to_expert's last column");
```

(e) Put this doc block directly above the route-tables launcher's
`template <typename RemapT, typename WeightT, typename XT>` (`:287`). Its first line is an anchor:

```cpp
/// \brief Checked launcher for `exl3_moe_route_tables_kernel`: the fused MoE's route tables and input staging.
///
/// `x`, `x16_out` and `out_zero` are the one decode token's `[1, hidden]` rows; `det` is the `[3, slots + 1]` stack.
```
and insert these as the first statements of its body (before `const auto stream = ...` at `:300`):
```cpp
  using namespace host;
  static_assert(std::is_same_v<RemapT, int32_t> || std::is_same_v<RemapT, int64_t>, "remap is int32 or int64");
  static_assert(
      (std::is_same_v<WeightT, fp32_t> || std::is_same_v<WeightT, fp16_t> || std::is_same_v<WeightT, bf16_t>) &&
          (std::is_same_v<XT, fp32_t> || std::is_same_v<XT, fp16_t> || std::is_same_v<XT, bf16_t>),
      "weights and x are fp32, fp16 or bf16");
  constexpr int64_t kMaxRoutes = 32;
  auto K_ = SymbolicSize{"routes"};
  auto H_ = SymbolicSize{"hidden"};
  auto C_ = SymbolicSize{"columns"};
  auto device = SymbolicDevice{};
  expert_stream::verify_named("remap", TensorMatcher({K_}).with_dtype<RemapT>().with_device<kDLCUDA>(device), remap);
  expert_stream::verify_named(
      "weights", TensorMatcher({K_}).with_dtype<WeightT>().with_device<kDLCUDA>(device), weights);
  expert_stream::verify_named("keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep);
  expert_stream::verify_named("x", TensorMatcher({1, H_}).with_dtype<XT>().with_device<kDLCUDA>(device), x);
  expert_stream::verify_named(
      "remap64_out", TensorMatcher({K_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), remap64_out);
  expert_stream::verify_named(
      "x16_out", TensorMatcher({1, H_}).with_dtype<fp16_t>().with_device<kDLCUDA>(device), x16_out);
  expert_stream::verify_named(
      "out_zero", TensorMatcher({1, H_}).with_dtype<float>().with_device<kDLCUDA>(device), out_zero);
  expert_stream::verify_named(
      "expert_count", TensorMatcher({C_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), expert_count);
  expert_stream::verify_named(
      "inv_order", TensorMatcher({K_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), inv_order);
  expert_stream::verify_named(
      "weight_sorted", TensorMatcher({K_}).with_dtype<fp16_t>().with_device<kDLCUDA>(device), weight_sorted);
  expert_stream::verify_named("det", TensorMatcher({3, C_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), det);
  RuntimeCheck(0 < K_.unwrap() && K_.unwrap() <= kMaxRoutes, "remap must hold 1-32 routes");
```

The route launcher uses its own `kMaxRoutes`, not `kLayerFusionWarp`, so that after Task 2 the EXL3 header does not
depend on the residency header.

**The `[1, hidden]` and `[3, slots + 1]` shapes are a tightening** of the old numel-only checks. Every caller already
passes them:
- `exl3_fused_moe.py:105-106` allocates `self.out` and `self.x16` as `(1, hidden)`, and `:118` allocates `self.det`
  as `(3, slots + 1)`;
- `Exl3FusedMoE.run` refuses any `x` whose `shape[0] != 1` (`:146`);
- the parity test at `test_dsv41_layer_fusion_gpu.py:225-238` uses the same shapes.

- [ ] **Step 4: Thin the Python wrapper to the instantiation dtype checks**

In `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py`:
- Delete `MAX_WIDTH = 32` (`:25`).
- Delete the whole `_check` function (`:61-68`).
- In `direct_gather_destinations`, replace `:89-101` (from `width = victims.numel()` through
  `_check("live_out", ...)`) with:

```python
    for name, tensor in (("topk_ids", topk_ids), ("remap", remap), ("remap_out", remap_out)):
        if tensor.dtype not in _ID_DTYPES:
            raise ValueError(f"{name} must be {_ID_DTYPES}, got {tensor.dtype}")
```
  and append one sentence to its docstring's last paragraph: `The launcher checks every tensor; this refuses only a
  dtype that has no instantiation.`
- In `direct_commit_gather`, delete `:141-162` (from `width = destinations.numel()` through
  `_check("miss_count", miss_count, torch.int32, 1)`). The launcher now checks all of it, including
  `delivered and keep go together`.
- In `exl3_moe_route_tables`, replace `:202-216` (from `top_k = remap.numel()` through `_check("det", ...)`) with:

```python
    for name, tensor, dtypes in (("remap", remap, _ID_DTYPES), ("weights", weights, _FLOAT_DTYPES), ("x", x, _FLOAT_DTYPES)):
        if tensor.dtype not in dtypes:
            raise ValueError(f"{name} must be {dtypes}, got {tensor.dtype}")
```
  and append the same sentence to its docstring.

These dtype checks stay in Python because they choose the template instantiation. Without them, an unsupported dtype
would compile a fresh module only to fail its `static_assert` (minutes of nvcc for an error), or would raise a
`KeyError` in `make_cpp_args`. Their message is byte-identical to the old `_check` dtype message.

- [ ] **Step 5: Format, self-check, commit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
pre-commit run --files python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh python/sglang/kernels/ops/moe/dsv41_layer_fusion.py
git grep -nE "with_device\(" -- python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh
git diff -U0 -- python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh | grep -nE "^[-+].*(__global__|LaunchKernel\(|<<<)"
```
Expected:
- pre-commit passes; if it reformatted a file, re-run it until it passes.
- Both greps print nothing. The first means no untemplated `with_device`; the second means no kernel signature or
  launch line changed.

Then commit:
```bash
git add python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh python/sglang/kernels/ops/moe/dsv41_layer_fusion.py
git commit -m "layer-fusion-split(launcher-checks,non_mechanical_provable): the layer-fusion launchers check every tensor in C++

Each launcher verifies every tensor (TensorMatcher + verify_named, CUDA only, one
shared device) and the 1-32 width bound before launching; the Python wrapper keeps
only the dtype checks that choose the instantiation. topk_ids -> expert_to_slot stays
an unchecked, documented precondition. Kernel bodies and launch lines unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```

- [ ] **Step 6: Run everything green on divix01**

Push and update the divix01 worktree. Then run `SUITE_GPU`, with the checks file now appended, and `SUITE_UNIT`.

Expected:
- `SUITE_GPU`: `EXIT=0`, at the Task 0 count plus `17 passed` (the 6 + 6 + 5 refusal cases).
- `SUITE_UNIT`: `EXIT=0`, at the Task 0 counts exactly.

Record both commands and results. This task's last commit is already pushed, so they go in the body of Task 2's
first commit.

---

### Task 2: C++ split (prepare + proven move)

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh` (prepare; then it becomes the include-only shim)
- Create: `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh`
- Create: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh`
- Create (untracked until Task 6): `analysis/layer-fusion-split/proof/manifest-cxx.json`

**Interfaces:**
- Consumes: the post-Task-1 file with its anchor lines. Four exist since base: `constexpr int kLayerFusionWarp = 32;`,
  the `layer_fusion_to_float(float v)` line, and the first comment lines of the gather and route-tables kernels.
  Four more were added in Task 1: the three launcher `\brief` lines and the `verify_bool_named` `\brief` line.
- Produces: the two headers with exactly the moved text, both still in `namespace sglang` (Task 4 renames the namespaces) and both `#pragma once`.
  `dsv41_layer_fusion.cuh` includes both, so every JIT module still builds unchanged until Task 4.

- [ ] **Step 1: Prepare: two comment edits in place**

In `dsv41_layer_fusion.cuh`, delete these three lines and the blank line after them:
```cpp
// Three single-launch replacements for the per-layer torch op chains of the DSV4.1 EXL3 graph decode
// (SGLANG_DSV41_ENABLE_LAYER_FUSION). Each one reproduces its chain's results bit for bit: the chains are integer
// bookkeeping plus two exact float conversions, so nothing here reorders a floating-point sum.
```
Their facts reappear in the new headers' authored leading comments (Step 3). A file comment cannot be moved into two
files, and the proof exempts only a new file's leading comment.

Then replace the stale width sentence of the commit-kernel comment. This line:
```cpp
// running it serially is what keeps every final value, dump columns included, identical. width is top_k (6).
```
becomes:
```cpp
// running it serially is what keeps every final value, dump columns included, identical.
```
The width is the gather's miss-lane count, not top_k, and its 1-32 bound now lives in the launcher.

- [ ] **Step 2: Commit the prepare**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
pre-commit run --files python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh
git add python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh
git commit -m "layer-fusion-split(cxx-prepare,non_mechanical_provable): drop the DSV4.1 file comment and the stale width note

Comment-only. The file comment cannot follow its code into two files; the new headers
restate it. 'width is top_k (6)' was wrong (width is the miss-lane count, bounded by the
launcher since the previous commit).

Task 1 suites on divix01 (commands in docs/superpowers/plans/2026-09-27-layer-fusion-split.md):
SUITE_GPU <paste summary line> EXIT=0; SUITE_UNIT <paste summary line> EXIT=0.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
Replace the two `<paste summary line>` markers with pytest's actual summary lines from Task 1 Step 6 before running
the command.

- [ ] **Step 3: Build the two headers, the shim and the manifest by line ranges**

Run from the worktree root. The script copies whole line ranges, and authors only the two leading comments, the
`#pragma once` and include lines, and the namespace lines:

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
mkdir -p analysis/layer-fusion-split/proof
python3 - <<'EOF'
import json
import subprocess
from pathlib import Path

CSRC = "python/sglang/kernels/jit/csrc/moe"
OLD = f"{CSRC}/dsv41_layer_fusion.cuh"
RES = f"{CSRC}/expert_residency/direct_gather.cuh"
EXL3 = f"{CSRC}/exl3/exl3_route_tables.cuh"
BASE = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

ANCHORS = {
    "warp": "constexpr int kLayerFusionWarp = 32;",
    "float": "__device__ __forceinline__ float layer_fusion_to_float(float v) {",
    "gather_kernel": "// GpuResidencyUpdater.gather_destinations, for one layer. One warp; lane j owns shortlist entry j (width <= 32).",
    "route_kernel": "// exl3_fused_moe.route_tables plus the copies around it in Exl3FusedMoE.run: the int64 remap, x -> fp16, the zeroed",
    "bool": "/// \\brief `verify_named` for a tensor the kernel reads as `bool` (`kDLBool` has no dtype trait).",
    "route_launcher": "/// \\brief Checked launcher for `exl3_moe_route_tables_kernel`: the fused MoE's route tables and input staging.",
    "end": "}  // namespace sglang",
}

RES_HEAD = """// DIRECT residency bookkeeping for the GPU hot cache (GpuResidencyUpdater), one launch per layer each: the gather's
// destinations and the residency commit. Both reproduce their torch chains bit for bit (integer bookkeeping only).
// Nothing here depends on the expert format or the model.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <stdint.h>

#include <type_traits>
#include <utility>

#include "../expert_stream/tensor_checks.h"

namespace sglang {

"""

EXL3_HEAD = """// The EXL3 fused MoE's route tables and input staging (exl3_fused_moe.route_tables and the copies around it) in one
// launch per layer, bit for bit: integer bookkeeping plus two exact float conversions, so nothing reorders a sum.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <type_traits>

#include "../expert_stream/tensor_checks.h"

namespace sglang {

"""

lines = Path(OLD).read_text().split("\n")


def at(anchor: str) -> int:
    hits = [i + 1 for i, line in enumerate(lines) if line.rstrip() == anchor]
    assert len(hits) == 1, (anchor, hits)
    return hits[0]


a = {name: at(text) for name, text in ANCHORS.items()}
blocks = {
    RES: [(a["warp"], a["warp"]), (a["gather_kernel"], a["route_kernel"] - 1), (a["bool"], a["route_launcher"] - 1)],
    EXL3: [(a["float"], a["gather_kernel"] - 1), (a["route_kernel"], a["bool"] - 1), (a["route_launcher"], a["end"] - 1)],
}
for dst, head in ((RES, RES_HEAD), (EXL3, EXL3_HEAD)):
    parts = ["\n".join(lines[first - 1 : last]).strip("\n") for first, last in blocks[dst]]
    Path(dst).parent.mkdir(exist_ok=True)
    Path(dst).write_text(head + "\n\n".join(parts) + "\n\n}  // namespace sglang\n")
Path(OLD).write_text('#include "exl3/exl3_route_tables.cuh"\n#include "expert_residency/direct_gather.cuh"\n')
manifest = {
    "base": BASE,
    "sources": [OLD],
    "files": {
        RES: [[OLD, first, last] for first, last in blocks[RES]],
        EXL3: [[OLD, first, last] for first, last in blocks[EXL3]],
        OLD: [],
    },
}
Path("analysis/layer-fusion-split/proof/manifest-cxx.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest, indent=1))
EOF
```
Expected:
- The script prints the manifest: six blocks, and `"base"` equal to the Step 2 commit.
- Every line of the old file outside the blocks is an include, the namespace line, a blank line, or the closing
  namespace line (the proof in Step 4 checks this).

- [ ] **Step 4: Format, prove, commit**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
F="python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh"
git add $F && pre-commit run --files $F; git add $F
python3 analysis/expert-stream-split/cxx_move_proof.py analysis/layer-fusion-split/proof/manifest-cxx.json
```
Expected: `PASS`.

A `FAIL` naming a body line means clang-format reflowed a moved line. Compare that line with the base. If the
reflow is in moved code, the Task 1 code had a line over 120 columns: fix it with a fix-up commit on top of Task 1's
code, *before* this step, and rebuild from Step 3. Do not hand-edit the new files.

```bash
git commit -m "layer-fusion-split(cxx-move,mechanical_provable): split dsv41_layer_fusion.cuh into expert_residency/ and exl3/

Pure relocation. expert_residency/direct_gather.cuh gets kLayerFusionWarp, both DIRECT
kernels, verify_bool_named and their two launchers; exl3/exl3_route_tables.cuh gets
layer_fusion_to_float, the route-tables kernel and its launcher. dsv41_layer_fusion.cuh
keeps only two includes of the new headers, so every JIT module builds unchanged.

Proof: python3 analysis/expert-stream-split/cxx_move_proof.py \\
  analysis/layer-fusion-split/proof/manifest-cxx.json -> PASS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
The `manifest-cxx.json` file stays untracked until Task 6.

- [ ] **Step 5: The shim builds everything on divix01**

Push and update the divix01 worktree. Run `SUITE_GPU`, which includes the checks file.

Expected: `EXIT=0`, with the same counts as Task 1 Step 6. This compiles all three JIT modules through the shim, so it
also proves the new headers' include lists are complete.

---

### Task 3: Python split (one proven move)

**Files:**
- Create: `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py`
- Create: `python/sglang/kernels/ops/moe/exl3_route_tables.py`
- Delete: `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py`
- Modify: `python/sglang/srt/layers/moe/expert_residency_gpu.py:793` and `:872` (import lines only)
- Modify: `python/sglang/srt/layers/quantization/exl3_fused_moe.py:128` (import line only)
- Modify: `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py:216` (import line only)
- Modify: `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` (its three function-scoped import lines)

**Interfaces:**
- Consumes: the post-Task-1 module, whose top-level symbols in order are `_ID_DTYPES`, `_FLOAT_DTYPES`,
  `_gather_module`, `_commit_module`, `_route_tables_module`, `direct_gather_destinations`, `direct_commit_gather` and
  `exl3_moe_route_tables`.
- Produces:
  - `sglang.kernels.ops.moe.expert_residency_direct_gather` holds `_ID_DTYPES`, `_gather_module`, `_commit_module`,
    `direct_gather_destinations` and `direct_commit_gather`.
  - `sglang.kernels.ops.moe.exl3_route_tables` holds `_ID_DTYPES`, `_FLOAT_DTYPES`, `_route_tables_module` and
    `exl3_moe_route_tables`.
  - Signatures are unchanged. The `load_jit` names and `cuda_files` strings inside the moved bodies are still the old
    ones until Task 4.

- [ ] **Step 1: Write the two modules**

Create `python/sglang/kernels/ops/moe/exl3_route_tables.py` as exactly this header, followed by `_route_tables_module`
and `exl3_moe_route_tables` cut verbatim, decorators included, from `dsv41_layer_fusion.py`, in that order:

```python
"""JIT wrapper for the EXL3 fused MoE's route tables (SGLANG_DSV41_ENABLE_LAYER_FUSION).

``exl3_moe_route_tables`` launches one kernel in place of ``exl3_fused_moe.route_tables`` and the copies around it in
``Exl3FusedMoE.run`` (22 kernels per layer), with bit-identical results; the flag-off path runs the torch chain, and
the parity test compares against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
```

Create `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` as exactly this header, followed by
`_gather_module`, `_commit_module`, `direct_gather_destinations` and `direct_commit_gather` cut verbatim, in that
order:

```python
"""JIT wrappers for the DIRECT residency kernels of ``GpuResidencyUpdater`` (SGLANG_DSV41_ENABLE_LAYER_FUSION).

Each wrapper launches one kernel in place of a per-layer chain of small torch ops, with bit-identical results:

* ``direct_gather_destinations`` -- ``GpuResidencyUpdater.gather_destinations`` (26 kernels per layer), with the
  route-slot lookup ``expert_to_slot.index_select(0, flat.long())`` folded in;
* ``direct_commit_gather`` -- ``GpuResidencyUpdater.commit_gather`` (41 kernels per layer).

The torch chains stay the reference: the flag-off path runs them, and the parity tests compare against them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
```

Then delete `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py` with `git rm`.

- [ ] **Step 2: Repoint the importers**

Change the module path, and nothing else, in each function-scoped import:

| File:line | Import becomes |
|---|---|
| `python/sglang/srt/layers/moe/expert_residency_gpu.py:793` | `from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_gather_destinations` |
| `python/sglang/srt/layers/moe/expert_residency_gpu.py:872` | `from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_commit_gather` |
| `python/sglang/srt/layers/quantization/exl3_fused_moe.py:128` | `from sglang.kernels.ops.moe.exl3_route_tables import exl3_moe_route_tables` |
| `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py:216` | `from sglang.kernels.ops.moe.exl3_route_tables import exl3_moe_route_tables` |
| `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` (`_run_gather`) | `from sglang.kernels.ops.moe.expert_residency_direct_gather import _gather_module` |
| `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` (`_run_commit`) | `from sglang.kernels.ops.moe.expert_residency_direct_gather import _commit_module` |
| `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` (`_run_route`) | `from sglang.kernels.ops.moe.exl3_route_tables import _route_tables_module` |

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
git grep -n "ops.moe.dsv41_layer_fusion" -- python test scripts benchmarks analysis/dsv41-drive
```
Expected: no output.

- [ ] **Step 3: Format and commit (local only)**

```bash
F="python/sglang/kernels/ops/moe/expert_residency_direct_gather.py python/sglang/kernels/ops/moe/exl3_route_tables.py python/sglang/srt/layers/moe/expert_residency_gpu.py python/sglang/srt/layers/quantization/exl3_fused_moe.py test/manual/dsv41/test_dsv41_layer_fusion_gpu.py test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py"
git add $F && pre-commit run --files $F; git add $F
git status --short   # expect exactly the 6 files above plus 'D  python/sglang/kernels/ops/moe/dsv41_layer_fusion.py'
git commit -m "layer-fusion-split(python-move,mechanical_provable): split the layer-fusion ops module by owner and repoint its importers

Pure relocation. expert_residency_direct_gather.py gets _gather_module, _commit_module,
direct_gather_destinations and direct_commit_gather; exl3_route_tables.py gets
_route_tables_module and exl3_moe_route_tables. dsv41_layer_fusion.py is deleted and its
five function-scoped importers (two in expert_residency_gpu.py, one in exl3_fused_moe.py,
two test files) are repathed. JIT names and cuda_files strings are untouched (next commit).

Proof: analysis/layer-fusion-split/proof/<this sha>.py (Repro) -> PASS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
The `<this sha>` in the message is literal text, because a commit cannot name its own sha. Task 6's proof file name
supplies the sha.

- [ ] **Step 4: Write and run the proof before pushing**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
cp .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_utils.py analysis/layer-fusion-split/proof/
SHA=$(git rev-parse HEAD); PARENT=$(git rev-parse HEAD~1)
cat > analysis/layer-fusion-split/proof/${SHA:0:10}.py <<EOF
"""Proof: the python-move commit is a pure relocation (Repro, mechanical-refactor-verify)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanical_refactor_reproduction_utils import Repro

BASE = "${PARENT}"
TARGET = "${SHA}"
EOF
cat >> analysis/layer-fusion-split/proof/${SHA:0:10}.py <<'EOF'
OLD = "python/sglang/kernels/ops/moe/dsv41_layer_fusion.py"
RES = "python/sglang/kernels/ops/moe/expert_residency_direct_gather.py"
EXL3 = "python/sglang/kernels/ops/moe/exl3_route_tables.py"
OLD_MOD = "sglang.kernels.ops.moe.dsv41_layer_fusion"
RES_MOD = "sglang.kernels.ops.moe.expert_residency_direct_gather"
EXL3_MOD = "sglang.kernels.ops.moe.exl3_route_tables"
CHECKS = "test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py"

EXL3_HEADER = '''"""JIT wrapper for the EXL3 fused MoE's route tables (SGLANG_DSV41_ENABLE_LAYER_FUSION).

``exl3_moe_route_tables`` launches one kernel in place of ``exl3_fused_moe.route_tables`` and the copies around it in
``Exl3FusedMoE.run`` (22 kernels per layer), with bit-identical results; the flag-off path runs the torch chain, and
the parity test compares against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
'''

RES_HEADER = '''"""JIT wrappers for the DIRECT residency kernels of ``GpuResidencyUpdater`` (SGLANG_DSV41_ENABLE_LAYER_FUSION).

Each wrapper launches one kernel in place of a per-layer chain of small torch ops, with bit-identical results:

* ``direct_gather_destinations`` -- ``GpuResidencyUpdater.gather_destinations`` (26 kernels per layer), with the
  route-slot lookup ``expert_to_slot.index_select(0, flat.long())`` folded in;
* ``direct_commit_gather`` -- ``GpuResidencyUpdater.commit_gather`` (41 kernels per layer).

The torch chains stay the reference: the flag-off path runs them, and the parity tests compare against them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
'''

EXL3_SYMBOLS = ["_route_tables_module", "exl3_moe_route_tables"]
RES_SYMBOLS = ["_gather_module", "_commit_module", "direct_gather_destinations", "direct_commit_gather"]

repro = (
    Repro(BASE, TARGET)
    # _ID_DTYPES survives in the source here (re-derived copy); _FLOAT_DTYPES leaves with the route tables.
    .extract_symbols_to_new_module(
        OLD, EXL3, symbols=EXL3_SYMBOLS, header=EXL3_HEADER, order=EXL3_SYMBOLS, drop_assigns=["_FLOAT_DTYPES"]
    )
    .extract_symbols_to_new_module(
        OLD, RES, symbols=RES_SYMBOLS, header=RES_HEADER, order=RES_SYMBOLS, drop_assigns=["_ID_DTYPES"]
    )
    .delete_file(OLD)
    .repath_import(
        "python/sglang/srt/layers/moe/expert_residency_gpu.py",
        old_module=OLD_MOD,
        new_module=RES_MOD,
        name="direct_gather_destinations",
    )
    .repath_import(
        "python/sglang/srt/layers/moe/expert_residency_gpu.py",
        old_module=OLD_MOD,
        new_module=RES_MOD,
        name="direct_commit_gather",
    )
    .repath_import(
        "python/sglang/srt/layers/quantization/exl3_fused_moe.py",
        old_module=OLD_MOD,
        new_module=EXL3_MOD,
        name="exl3_moe_route_tables",
    )
    .repath_import(
        "test/manual/dsv41/test_dsv41_layer_fusion_gpu.py",
        old_module=OLD_MOD,
        new_module=EXL3_MOD,
        name="exl3_moe_route_tables",
    )
    .repath_import(CHECKS, old_module=OLD_MOD, new_module=RES_MOD, name="_gather_module")
    .repath_import(CHECKS, old_module=OLD_MOD, new_module=RES_MOD, name="_commit_module")
    .repath_import(CHECKS, old_module=OLD_MOD, new_module=EXL3_MOD, name="_route_tables_module")
)
sys.exit(1 if repro.run() else 0)
EOF
python3 analysis/layer-fusion-split/proof/${SHA:0:10}.py; echo "EXIT=$?"
```
Expected: `PASS: reproduces the commit byte-for-byte.` and `EXIT=0`.

On `RESIDUAL`, the commit bundled something the relocation does not account for; usually a header differs from
Step 1's text. The commit is not pushed yet, so run `git reset --soft HEAD~1`, fix the working tree, recommit with the
same message, and regenerate the proof (its sha changed).

- [ ] **Step 5: Run everything on divix01**

Push and update the divix01 worktree. Run `SUITE_UNIT` and `SUITE_GPU`.

Expected: both `EXIT=0`, with the same counts as Task 1 Step 6.

---

### Task 4: Postpare: rename to the new home and point the JIT modules at the split headers

**Files:**
- Modify: `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` (string literals in `_gather_module` and
  `_commit_module`)
- Modify: `python/sglang/kernels/ops/moe/exl3_route_tables.py` (string literals in `_route_tables_module`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` (namespace lines and
  `kLayerFusionWarp`)
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh` (namespace lines and
  `layer_fusion_to_float`)
- Delete: `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh`

**Interfaces:**
- Consumes: the Task 3 modules and the Task 2 headers.
- Produces:
  - `_gather_module` builds JIT module `"expert_residency_direct_gather_destinations"` and `_commit_module` builds
    `"expert_residency_direct_commit_gather"`, both from `moe/expert_residency/direct_gather.cuh`, with the wrappers
    `expert_residency::direct_gather_destinations_gpu<{args}>` and `expert_residency::direct_commit_gather_gpu`.
  - `_route_tables_module` builds `"exl3_moe_route_tables"` from `moe/exl3/exl3_route_tables.cuh`, with the wrapper
    `exl3::exl3_moe_route_tables_gpu<{args}>`.
  - The C++ code lives in `namespace sglang::expert_residency` and `namespace sglang::exl3`. The generated wrapper
    opens `namespace sglang {`, so the wrapper strings are qualified relative to `sglang`.

- [ ] **Step 1: Rename in the two headers**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
RES=python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh
RT=python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh
sed -i 's/^namespace sglang {$/namespace sglang::expert_residency {/; s|^}  // namespace sglang$|}  // namespace sglang::expert_residency|; s/\bkLayerFusionWarp\b/kDirectGatherWarp/g' $RES
sed -i 's/^namespace sglang {$/namespace sglang::exl3 {/; s|^}  // namespace sglang$|}  // namespace sglang::exl3|; s/\blayer_fusion_to_float\b/route_tables_to_float/g' $RT
git diff --stat -- $RES $RT
```
Expected: `git diff --stat` shows only these changed lines:
- in `direct_gather.cuh`, the two namespace lines plus every line naming `kLayerFusionWarp`: the definition, the
  kernels' `__launch_bounds__` and shared arrays, the commit kernel's lane arrays, and the two launchers'
  `LaunchKernel` and width checks;
- in `exl3_route_tables.cuh`, the two namespace lines plus the three overload definitions and the two call sites.

Lookups of `expert_stream::verify_named`, `host::...` and `device::...` still resolve from the nested namespaces,
because unqualified lookup searches the enclosing `sglang` as well.

- [ ] **Step 2: Edit the Python string literals**

In `expert_residency_direct_gather.py`:
- `"dsv41_direct_gather_destinations",` becomes `"expert_residency_direct_gather_destinations",`.
- `"dsv41_direct_commit_gather",` becomes `"expert_residency_direct_commit_gather",`.
- Both `cuda_files=["moe/dsv41_layer_fusion.cuh"],` become `cuda_files=["moe/expert_residency/direct_gather.cuh"],`.
- `f"direct_gather_destinations_gpu<{args}>"` becomes `f"expert_residency::direct_gather_destinations_gpu<{args}>"`.
- `"direct_commit_gather_gpu"` becomes `"expert_residency::direct_commit_gather_gpu"`.

In `exl3_route_tables.py`:
- `"dsv41_exl3_moe_route_tables",` becomes `"exl3_moe_route_tables",`.
- `cuda_files=["moe/dsv41_layer_fusion.cuh"],` becomes `cuda_files=["moe/exl3/exl3_route_tables.cuh"],`.
- `f"exl3_moe_route_tables_gpu<{args}>"` becomes `f"exl3::exl3_moe_route_tables_gpu<{args}>"`.

Then `git rm python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh`.

- [ ] **Step 3: Check nothing still names the old files or names**

```bash
git grep -nE "dsv41_layer_fusion|dsv41_direct_|dsv41_exl3_moe_route_tables|kLayerFusionWarp|layer_fusion_to_float" -- python test scripts benchmark benchmarks analysis/dsv41-drive
git grep -nE "with_device\(" -- python/sglang/kernels/jit/csrc/moe/expert_residency python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh
```
Expected: both greps print nothing. `git grep` matches file contents, not paths, so the test file
`test_dsv41_layer_fusion_gpu.py` does not match. It keeps its name because it tests the
`SGLANG_DSV41_ENABLE_LAYER_FUSION` feature, whose name is unchanged.

- [ ] **Step 4: Commit, push, run everything**

```bash
F="python/sglang/kernels/ops/moe/expert_residency_direct_gather.py python/sglang/kernels/ops/moe/exl3_route_tables.py $RES $RT"
git add $F && pre-commit run --files $F; git add $F
git commit -m "layer-fusion-split(postpare,non_mechanical_provable): rename the layer-fusion code to its new home; drop the include shim

Renames only, no logic: the JIT modules become expert_residency_direct_gather_destinations,
expert_residency_direct_commit_gather and exl3_moe_route_tables and build
expert_residency/direct_gather.cuh and exl3/exl3_route_tables.cuh; the code moves into
sglang::expert_residency and sglang::exl3 (wrapper strings qualified to match);
kLayerFusionWarp -> kDirectGatherWarp, layer_fusion_to_float -> route_tables_to_float. Kernel
base names are unchanged, so kernel_groups.py's substring keys match old and new traces.
The include-only dsv41_layer_fusion.cuh is deleted. First load compiles cold.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
Push and update the divix01 worktree. Run `SUITE_UNIT` and `SUITE_GPU`.

Expected: both `EXIT=0`, with the same counts as Task 1 Step 6. The first `SUITE_GPU` compiles the three renamed
modules cold, so it runs longer than earlier runs. This is the first run that compiles each header alone, which
proves that neither header depends on the other.

---

### Task 5: Cast-fusion split (prepare, proven C++ move, proven Python move, postpare)

This is the added scope, approved by the user. Steps 1-8 are a pure relocation plus renames, with no behaviour
change. They move the launchers verbatim, including the untemplated `.with_device(device)`. Steps 9-11 then fix that
device check as a separate TDD pair (controller ruling).

**Files:**
- Modify, then delete: `python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh`. The prepare deletes `:15-18`;
  the move leaves an include-only shim; the postpare deletes the file.
- Create: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh`
- Create: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh`
- Create: `python/sglang/kernels/ops/moe/exl3_cast_fusion.py`
- Delete: `python/sglang/kernels/ops/moe/dsv41_cast_fusion.py`
- Modify: `python/sglang/srt/layers/quantization/exl3.py:230` and `:434` (function-scoped import lines only)
- Modify: `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py:15` (module-level import line only)
- Create: `test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py` (Step 9)
- Modify again (Step 10): the two new `exl3/` cast headers' device checks
- Create (untracked until Task 6): `analysis/layer-fusion-split/proof/manifest-cast.json`,
  `analysis/layer-fusion-split/proof/<cast-python-move sha10>.py`

`test/registered/unit/layers/quantization/test_exl3_cast_fusion_cpu.py` is not edited. Its only reference (`:5`)
names the GPU test file, which keeps its path, and never names the moved module.

**Interfaces:**
- Consumes the base file's two anchors:
  - `:21` `// The shared expert's y.to(bf16) -> silu_mul_clamp -> x.to(fp16): gate_up [rows, 2 * inter] fp16 in, the down`
  - `:79` `// Exl3MoEMethod's out.to(bf16) * routed_scaling_factor on the fused MoE's fp32 output. The factor is the float`
- Produces:
  - `sglang.kernels.ops.moe.exl3_cast_fusion`, with `exl3_silu_mul_clamp_half(gate_up, swiglu_limit) -> Tensor`,
    `exl3_scale_to_bf16(routed, factor) -> Tensor`, and the loaders `_silu_module()` and `_scale_module()`.
    Signatures are unchanged.
  - After the postpare, `_silu_module` loads `"exl3_silu_mul_clamp_half"` from `moe/exl3/exl3_silu_mul_clamp_half.cuh`
    with `extra_cuda_cflags=["-use_fast_math"]`.
  - `_scale_module` loads `"exl3_scale_to_bf16"` from `moe/exl3/exl3_scale_to_bf16.cuh` with no extra flags.
  - The C++ launchers `exl3_silu_mul_clamp_half<kUsePDL>` and `exl3_scale_to_bf16`, and the kernels
    `exl3_silu_mul_clamp_half_kernel` and `exl3_scale_to_bf16_kernel`, keep their base names. They stay in
    `namespace sglang` through the move, and the postpare moves them to `namespace sglang::exl3`. The wrappers
    become `exl3::exl3_silu_mul_clamp_half<{args}>` and `exl3::exl3_scale_to_bf16`.

- [ ] **Step 1: Prepare: drop the file comment, and commit**

In `dsv41_cast_fusion.cuh`, delete these three lines (`:15-17`) and the blank line after them (`:18`):
```cpp
// Kernels of the DSV4.1 EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION). exl3_gemm reads and writes
// fp16 while the model runs in bf16, so every EXL3 linear sits between two casts. These kernels keep each rounding
// of the unfused chain, in the same order, and only drop the round trips through memory.
```
The facts reappear in the two new files' authored leading comments. A file comment cannot follow its code into two
files, and the proof exempts only a new file's leading comment.

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
pre-commit run --files python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh
git add python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh
git commit -m "layer-fusion-split(cast-prepare,non_mechanical_provable): drop the cast-fusion file comment

Comment-only. The file comment cannot follow its two kernels into two files; each new
header restates it together with the fast-math flag that kernel requires.

Task 4 suites on divix01: SUITE_UNIT <paste summary line> EXIT=0; SUITE_GPU <paste summary line> EXIT=0.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
Before running the command, replace the two `<paste summary line>` markers with the actual summary lines from Task 4
Step 3.

- [ ] **Step 2: Build the two headers, the shim and the manifest by line ranges**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
mkdir -p analysis/layer-fusion-split/proof
python3 - <<'EOF'
import json
import subprocess
from pathlib import Path

CSRC = "python/sglang/kernels/jit/csrc/moe"
OLD = f"{CSRC}/dsv41_cast_fusion.cuh"
SILU = f"{CSRC}/exl3/exl3_silu_mul_clamp_half.cuh"
SCALE = f"{CSRC}/exl3/exl3_scale_to_bf16.cuh"
BASE = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

ANCHORS = {
    "silu": "// The shared expert's y.to(bf16) -> silu_mul_clamp -> x.to(fp16): gate_up [rows, 2 * inter] fp16 in, the down",
    "scale": "// Exl3MoEMethod's out.to(bf16) * routed_scaling_factor on the fused MoE's fp32 output. The factor is the float",
    "end": "}  // namespace sglang",
}

SILU_HEAD = """// REQUIRED BUILD FLAG: -use_fast_math. silu_and_mul (deepseek_v4/silu_and_mul_masked_post_quant.cuh) must compile
// to the same instructions as in silu_and_mul_clamp's module, which builds with it; bit parity depends on that.
// Part of the EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION): it keeps each rounding of the unfused
// chain, in the same order, and only drops the round trips through memory.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "../../deepseek_v4/silu_and_mul_masked_post_quant.cuh"

namespace sglang {

"""

SCALE_HEAD = """// REQUIRED BUILD FLAG: none; never -use_fast_math. Fast math flushes subnormal fp32 products to zero, which
// torch's bf16 multiply keeps, and bit parity with the unfused chain breaks. Part of the EXL3 decode cast fusion
// (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION): it keeps each rounding of the unfused chain and only drops a memory trip.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <stdint.h>

namespace sglang {

"""

lines = Path(OLD).read_text().split("\n")


def at(anchor: str) -> int:
    hits = [i + 1 for i, line in enumerate(lines) if line.rstrip() == anchor]
    assert len(hits) == 1, (anchor, hits)
    return hits[0]


a = {name: at(text) for name, text in ANCHORS.items()}
blocks = {SILU: [(a["silu"], a["scale"] - 1)], SCALE: [(a["scale"], a["end"] - 1)]}
for dst, head in ((SILU, SILU_HEAD), (SCALE, SCALE_HEAD)):
    parts = ["\n".join(lines[first - 1 : last]).strip("\n") for first, last in blocks[dst]]
    Path(dst).write_text(head + "\n\n".join(parts) + "\n\n}  // namespace sglang\n")
Path(OLD).write_text('#include "exl3/exl3_scale_to_bf16.cuh"\n#include "exl3/exl3_silu_mul_clamp_half.cuh"\n')
manifest = {
    "base": BASE,
    "sources": [OLD],
    "files": {
        SILU: [[OLD, first, last] for first, last in blocks[SILU]],
        SCALE: [[OLD, first, last] for first, last in blocks[SCALE]],
        OLD: [],
    },
}
Path("analysis/layer-fusion-split/proof/manifest-cast.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest, indent=1))
EOF
```
Expected: the script prints a manifest with two blocks, and `"base"` equal to the Step 1 commit.

The shim includes both headers into each JIT module, as the single file did before: the silu module still compiles
the unused scale kernel under fast math, and the scale module still compiles the unused silu kernel without it. The
kernels that actually run keep their flags until the postpare separates the sources.

- [ ] **Step 3: Format, prove, commit the C++ move**

```bash
F="python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh"
git add $F && pre-commit run --files $F; git add $F
python3 analysis/expert-stream-split/cxx_move_proof.py analysis/layer-fusion-split/proof/manifest-cast.json
```
Expected: `PASS`. On a `FAIL`, do not hand-edit the new files: rebuild them from Step 2.

```bash
git commit -m "layer-fusion-split(cast-cxx-move,mechanical_provable): split dsv41_cast_fusion.cuh into one exl3/ file per kernel

Pure relocation. exl3/exl3_silu_mul_clamp_half.cuh (build with -use_fast_math) gets the
shared expert's silu kernel and launcher; exl3/exl3_scale_to_bf16.cuh (never fast math)
gets the routed output's scale kernel and launcher. Each header's leading comment states
its required flag. dsv41_cast_fusion.cuh keeps only two includes, so both JIT modules
build unchanged.

Proof: python3 analysis/expert-stream-split/cxx_move_proof.py \\
  analysis/layer-fusion-split/proof/manifest-cast.json -> PASS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```

- [ ] **Step 4: Move the Python module and repoint its importers**

Create `python/sglang/kernels/ops/moe/exl3_cast_fusion.py` as exactly this header, followed by `_silu_module`,
`_scale_module`, `exl3_silu_mul_clamp_half` and `exl3_scale_to_bf16` cut verbatim, decorators and comments included,
from `dsv41_cast_fusion.py`, in that order:

```python
"""JIT wrappers for the EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION).

* ``exl3_silu_mul_clamp_half`` -- the shared expert's ``gate_up.to(bf16)``, ``silu_and_mul_clamp`` and the down
  projection's ``.to(fp16)`` as one kernel on exl3_gemm's fp16 output;
* ``exl3_scale_to_bf16`` -- the routed MoE output's ``out.to(bf16) * routed_scaling_factor``.

Both are bit-identical to the torch chains; the flag-off path runs those chains, and the parity tests compare against
them. The two kernels need opposite fast-math flags, and each loader carries its own: ``_silu_module`` builds with
``-use_fast_math``, ``_scale_module`` without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module
```
Then `git rm python/sglang/kernels/ops/moe/dsv41_cast_fusion.py`, and change these import lines' module path and
nothing else:

| File:line | Import becomes |
|---|---|
| `python/sglang/srt/layers/quantization/exl3.py:230` | `from sglang.kernels.ops.moe.exl3_cast_fusion import exl3_silu_mul_clamp_half` |
| `python/sglang/srt/layers/quantization/exl3.py:434` | `from sglang.kernels.ops.moe.exl3_cast_fusion import exl3_scale_to_bf16` |
| `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py:15` | `from sglang.kernels.ops.moe.exl3_cast_fusion import exl3_scale_to_bf16, exl3_silu_mul_clamp_half` |

```bash
F="python/sglang/kernels/ops/moe/exl3_cast_fusion.py python/sglang/srt/layers/quantization/exl3.py test/manual/dsv41/test_dsv41_cast_fusion_gpu.py"
git add $F && pre-commit run --files $F; git add $F
git grep -n "ops.moe.dsv41_cast_fusion" -- python test scripts benchmarks analysis/dsv41-drive   # expect no output
git status --short   # expect the 3 files above plus 'D  python/sglang/kernels/ops/moe/dsv41_cast_fusion.py'
git commit -m "layer-fusion-split(cast-python-move,mechanical_provable): move the cast-fusion ops module to exl3_cast_fusion.py and repoint its importers

Pure relocation. exl3_cast_fusion.py gets _silu_module, _scale_module,
exl3_silu_mul_clamp_half and exl3_scale_to_bf16; dsv41_cast_fusion.py is deleted; exl3.py's
two function-scoped imports and the GPU test's module-level import are repathed. JIT names,
cuda_files and flags are untouched (the postpare).

Proof: analysis/layer-fusion-split/proof/<this sha>.py (Repro) -> PASS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
The `<this sha>` in the message is literal text; Task 6's proof file name supplies the sha.

- [ ] **Step 5: Write and run the Python proof before pushing**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
SHA=$(git rev-parse HEAD); PARENT=$(git rev-parse HEAD~1)
cat > analysis/layer-fusion-split/proof/${SHA:0:10}.py <<EOF
"""Proof: the cast-python-move commit is a pure relocation (Repro, mechanical-refactor-verify)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanical_refactor_reproduction_utils import Repro

BASE = "${PARENT}"
TARGET = "${SHA}"
EOF
cat >> analysis/layer-fusion-split/proof/${SHA:0:10}.py <<'EOF'
OLD = "python/sglang/kernels/ops/moe/dsv41_cast_fusion.py"
NEW = "python/sglang/kernels/ops/moe/exl3_cast_fusion.py"
OLD_MOD = "sglang.kernels.ops.moe.dsv41_cast_fusion"
NEW_MOD = "sglang.kernels.ops.moe.exl3_cast_fusion"
EXL3 = "python/sglang/srt/layers/quantization/exl3.py"
TEST = "test/manual/dsv41/test_dsv41_cast_fusion_gpu.py"

HEADER = '''"""JIT wrappers for the EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION).

* ``exl3_silu_mul_clamp_half`` -- the shared expert's ``gate_up.to(bf16)``, ``silu_and_mul_clamp`` and the down
  projection's ``.to(fp16)`` as one kernel on exl3_gemm's fp16 output;
* ``exl3_scale_to_bf16`` -- the routed MoE output's ``out.to(bf16) * routed_scaling_factor``.

Both are bit-identical to the torch chains; the flag-off path runs those chains, and the parity tests compare against
them. The two kernels need opposite fast-math flags, and each loader carries its own: ``_silu_module`` builds with
``-use_fast_math``, ``_scale_module`` without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module
'''

SYMBOLS = ["_silu_module", "_scale_module", "exl3_silu_mul_clamp_half", "exl3_scale_to_bf16"]

repro = (
    Repro(BASE, TARGET)
    .extract_symbols_to_new_module(OLD, NEW, symbols=SYMBOLS, header=HEADER, order=SYMBOLS)
    .delete_file(OLD)
    .repath_import(EXL3, old_module=OLD_MOD, new_module=NEW_MOD, name="exl3_silu_mul_clamp_half")
    .repath_import(EXL3, old_module=OLD_MOD, new_module=NEW_MOD, name="exl3_scale_to_bf16")
    .remove_import(TEST, "from sglang.kernels.ops.moe.dsv41_cast_fusion import")
    .add_import(TEST, "from sglang.kernels.ops.moe.exl3_cast_fusion import exl3_scale_to_bf16, exl3_silu_mul_clamp_half")
)
sys.exit(1 if repro.run() else 0)
EOF
python3 analysis/layer-fusion-split/proof/${SHA:0:10}.py; echo "EXIT=$?"
```
Expected: `PASS: reproduces the commit byte-for-byte.` and `EXIT=0`.

On `RESIDUAL`, the commit is not pushed yet, so:
1. Run `git reset --soft HEAD~1`.
2. Fix the working tree.
3. Recommit with the same message.
4. Regenerate this proof, because the sha changed.

- [ ] **Step 6: Run both moves on divix01**

Push and update the divix01 worktree. Run `SUITE_UNIT` and `SUITE_GPU`.

Expected: both `EXIT=0`, with the same counts as Task 4. `test_dsv41_cast_fusion_gpu.py` passes unchanged, which
confirms that both kernels built through the shim with their old flags.

- [ ] **Step 7: Postpare: rename to the new home, and each JIT module builds its own header**

Move both headers into `sglang::exl3`:
```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
for H in python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh; do
  sed -i 's/^namespace sglang {$/namespace sglang::exl3 {/; s|^}  // namespace sglang$|}  // namespace sglang::exl3|' $H
done
git diff --stat -- python/sglang/kernels/jit/csrc/moe/exl3/
```
Expected: 2 lines changed in each of the two headers. The silu kernel's `using namespace device;` and its unqualified
`SiluAndMulClampParams`, `silu_and_mul` and `to_bf16x2` still resolve: unqualified lookup from `sglang::exl3`
searches the enclosing `sglang` and the global namespace, where they resolved before.

In `python/sglang/kernels/ops/moe/exl3_cast_fusion.py`, change the wrappers:
- `f"exl3_silu_mul_clamp_half<{args}>"` becomes `f"exl3::exl3_silu_mul_clamp_half<{args}>"`;
- `"exl3_scale_to_bf16"` in `cuda_wrappers` becomes `"exl3::exl3_scale_to_bf16"`.

Then, in the same file:
- In `_silu_module`: `"dsv41_exl3_silu_mul_clamp_half",` becomes `"exl3_silu_mul_clamp_half",`, and
  `cuda_files=["moe/dsv41_cast_fusion.cuh"],` becomes `cuda_files=["moe/exl3/exl3_silu_mul_clamp_half.cuh"],`. The
  `extra_cuda_cflags=["-use_fast_math"]` line and its comment stay.
- In `_scale_module`: `"dsv41_exl3_scale_to_bf16",` becomes `"exl3_scale_to_bf16",`, and
  `cuda_files=["moe/dsv41_cast_fusion.cuh"],` becomes `cuda_files=["moe/exl3/exl3_scale_to_bf16.cuh"],`. No
  `extra_cuda_cflags` is added.

Then `git rm python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh`.

Renaming the module names breaks no tooling: Task 0 Step 2 found nothing in `analysis/`, `benchmark/`,
`benchmarks/` or `scripts/` keyed on them. The kernel base names are unchanged, so `kernel_groups.py:25`'s
`"scale_to_bf16"` substring matches the old name and the namespaced one. The new names do not collide with any other
`load_jit` name. Both modules compile cold on first load.

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
git grep -n "dsv41_cast_fusion\|dsv41_exl3_silu_mul_clamp_half\|dsv41_exl3_scale_to_bf16" -- python scripts benchmarks analysis/dsv41-drive
git grep -c "use_fast_math" -- python/sglang/kernels/ops/moe/exl3_cast_fusion.py
git grep -n "exl3_scale_to_bf16.cuh" -A3 -- python/sglang/kernels/ops/moe/exl3_cast_fusion.py | grep -c fast_math
```
Expected:
- The first grep prints nothing. `test/` is excluded on purpose: the file `test_dsv41_cast_fusion_gpu.py` keeps its
  name, and `test_exl3_cast_fusion_cpu.py:5` names that file.
- The second grep prints `python/sglang/kernels/ops/moe/exl3_cast_fusion.py:3`: the docstring twice plus the
  `_silu_module` flag once.
- The third grep prints `0`: the scale module has no fast-math flag.

```bash
F="python/sglang/kernels/ops/moe/exl3_cast_fusion.py python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh"
git add $F && pre-commit run --files $F; git add $F
git commit -m "layer-fusion-split(cast-postpare,non_mechanical_provable): rename the cast-fusion code to its new home; drop the include shim

Renames only, no logic: _silu_module builds exl3/exl3_silu_mul_clamp_half.cuh (still with
-use_fast_math) as JIT module exl3_silu_mul_clamp_half; _scale_module builds
exl3/exl3_scale_to_bf16.cuh (no fast math) as exl3_scale_to_bf16; both headers move into
sglang::exl3 (wrapper strings qualified to match). Nothing keys on the old dsv41_ module
names; kernel base names are unchanged. The include-only dsv41_cast_fusion.cuh is
deleted. First load compiles cold.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```

- [ ] **Step 8: Run everything on divix01**

Push and update the divix01 worktree. Run `SUITE_UNIT` and `SUITE_GPU`. The first `SUITE_GPU` compiles the two
renamed modules cold.

Expected: both `EXIT=0`, with the same counts as Task 4. The flags are guarded by the tests that already exist:
- `test_scale_to_bf16_matches_cast_then_multiply` feeds `logspace(-30, 30)` magnitudes, whose subnormal products fast
  math would flush, so a fast-math scale build turns it red.
- `test_silu_mul_clamp_half_matches_the_cast_silu_cast_chain` compares bits against `silu_and_mul_clamp`'s
  fast-math build.

- [ ] **Step 9: Red test: each cast-fusion launcher must refuse a CPU tensor**

Create `test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py`:

```python
"""The cast-fusion launchers refuse a tensor that is not on a CUDA device, even when called without the wrapper.

Both launchers dereference ``input.data_ptr()`` in a kernel. A CPU input would be read as a device address. Each case
passes a CPU input straight to the JIT module's ``run``, with a valid CUDA output, and expects the launcher's
``TensorMatcher`` to raise before it launches. Writing ``.with_device(device)`` without ``<kDLCUDA>`` in either
launcher turns its case red: the untemplated call resets the allowed devices to "any" (the ``bdcf769eb8`` footgun), so
the CPU tensor is accepted.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_silu_mul_clamp_half_launcher_refuses_a_cpu_input():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _silu_module

    gate_up = torch.zeros(1, 16, dtype=torch.float16)
    out = torch.empty(1, 8, dtype=torch.float16, device="cuda")
    with pytest.raises(Exception, match="not in the allowed options"):
        _silu_module().run(gate_up, out, 7.0)


def test_scale_to_bf16_launcher_refuses_a_cpu_input():
    from sglang.kernels.ops.moe.exl3_cast_fusion import _scale_module

    routed = torch.zeros(8, dtype=torch.float32)
    out = torch.empty(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception, match="not in the allowed options"):
        _scale_module().run(routed, out, 1.5)
```

The match text comes from `SymbolicDevice::set_value` (`sgl_kernel/tensor.h:338-345`):
`Device value [...] not in the allowed options: ...`. The input is verified first, so the device is still unbound
when the check runs.

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
git add test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py
git commit -m "layer-fusion-split(cast-device-check-red,non_mechanical_provable): the cast-fusion launchers must refuse a CPU tensor (red)

Direct calls into both cast-fusion JIT modules with a CPU input. Red: the launchers'
untemplated .with_device(device) resets the allowed devices to any.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
Push and update the divix01 worktree. Then run each case red in **its own process**. The unchecked launch reads a
host pointer on the device, which leaves a sticky CUDA error behind, so run them one per process:
```bash
for K in silu_mul_clamp_half scale_to_bf16; do
  env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests flock /data/models/slang/nvfp4-work/cc-gpu.lock \
    taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
    test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py -k "$K" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
done
```
Expected: each run shows `1 failed` with `Failed: DID NOT RAISE` and `EXIT=1`. A CUDA illegal-address error at
teardown is also acceptable red evidence, because it shows that the kernel ran on the host pointer.

- [ ] **Step 10: Fix: the templated device form in both launchers**

In `python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh`, in `exl3_silu_mul_clamp_half`:
- delete the line `  device.set_options<kDLCUDA>();`;
- change `TensorMatcher({M, D}).with_dtype<fp16_t>().with_device(device).verify(input);` to
  `TensorMatcher({M, D}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(input);`;
- change `TensorMatcher({M, H}).with_dtype<fp16_t>().with_device(device).verify(output);` to
  `TensorMatcher({M, H}).with_dtype<fp16_t>().with_device<kDLCUDA>(device).verify(output);`.

In `python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh`, in `exl3_scale_to_bf16`:
- delete the line `  device.set_options<kDLCUDA>();`;
- change `TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(input);` to
  `TensorMatcher({N}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(input);`;
- change `TensorMatcher({N}).with_dtype<bf16_t>().with_device(device).verify(output);` to
  `TensorMatcher({N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(output);`.

`with_device<kDLCUDA>(device)` sets the options on the shared `SymbolicDevice` itself, so the separate
`set_options` line is redundant. It also binds both tensors to one device, as before: `device.unwrap()` still feeds
`LaunchKernel`.

```bash
F="python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh"
git grep -nE "with_device\(" -- $F   # expect no output
git diff -U0 -- $F | grep -E '^[-+][^-+]' | grep -vE 'with_device|set_options'   # expect no output
git add $F && pre-commit run --files $F; git add $F
git commit -m "layer-fusion-split(cast-device-check,non_mechanical_provable): the cast-fusion launchers restrict tensors to CUDA

.with_device(device) without template codes reset the allowed devices to any, undoing the
set_options<kDLCUDA>() before it (the bdcf769eb8 footgun), so a CPU tensor reached the
kernel. Both launchers now use .with_device<kDLCUDA>(device), the idiom of
expert_stream/lease_kernels.cuh. Kernels and launch lines unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```

- [ ] **Step 11: Green on divix01**

Push and update the divix01 worktree. From now on, `SUITE_GPU` also includes
`test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py`, appended to its file list. Run `SUITE_UNIT` and
`SUITE_GPU`.

Expected: both `EXIT=0`. `SUITE_UNIT` matches the Task 4 counts. `SUITE_GPU` is the Task 4 count plus `2 passed`,
and `test_dsv41_cast_fusion_gpu.py` still passes, which shows that every real CUDA call is still accepted.

---

### Task 6: Proofs, chain verification, final comparison

**Files:**
- Create: `analysis/layer-fusion-split/proof/<cxx-move sha10>.py` and `<cast-cxx-move sha10>.py`, the
  chain-verifier wrappers.
- Commit the files created earlier:
  - `analysis/layer-fusion-split/proof/manifest-cxx.json` (Task 2)
  - `analysis/layer-fusion-split/proof/manifest-cast.json` (Task 5)
  - `analysis/layer-fusion-split/proof/<python-move sha10>.py` (Task 3)
  - `analysis/layer-fusion-split/proof/<cast-python-move sha10>.py` (Task 5)
  - `analysis/layer-fusion-split/proof/mechanical_refactor_reproduction_utils.py` (Task 3)

**Interfaces:**
- Consumes: the chain `$(git merge-base master layer-fusion-split)..layer-fusion-split` and Task 0's baseline file.
- Produces: a chain report with verdict PASS, and the final suite counts next to the baseline.

- [ ] **Step 1: Wrap each C++ proof for the chain verifier**

The chain verifier runs `python3 <sha-prefix>.py` and requires both exit code 0 and a `PASS:` line. Each wrapper runs
`cxx_move_proof.py` at its move commit, and additionally refuses a commit that touches a file outside its manifest,
which `cxx_move_proof.py` alone does not check.

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
wrap() {  # $1 = commit-id in the subject, $2 = manifest file name
  MOVE=$(git log --format=%H --grep="($1,mechanical_provable)" -1)
  OUT=analysis/layer-fusion-split/proof/${MOVE:0:10}.py
  cat > $OUT <<EOF
"""Proof: the $1 commit is a pure relocation of line blocks (cxx_move_proof.py) and touches nothing else."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET = "${MOVE}"
MANIFEST = Path(__file__).resolve().parent / "$2"
EOF
  cat >> $OUT <<'EOF'
CHECKER = Path("analysis/expert-stream-split/cxx_move_proof.py").resolve()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    allowed = set(manifest["sources"]) | set(manifest["files"])
    touched = subprocess.run(
        ["git", "diff", "--name-only", manifest["base"], TARGET], check=True, capture_output=True, text=True
    ).stdout.split()
    extra = sorted(set(touched) - allowed)
    if extra:
        print(f"FAIL: the commit touches files outside the manifest: {extra}")
        return 1
    worktree = tempfile.mkdtemp(prefix="cxx-move-proof-")
    subprocess.run(["git", "worktree", "add", "--detach", worktree, TARGET], check=True, capture_output=True)
    try:
        result = subprocess.run(
            [sys.executable, str(CHECKER), str(MANIFEST)], cwd=worktree, capture_output=True, text=True
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            return 1
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", worktree], check=False)
    print("PASS: reproduces the commit byte-for-byte (cxx_move_proof).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF
  python3 $OUT; echo "EXIT=$?"
}
wrap cxx-move manifest-cxx.json
wrap cast-cxx-move manifest-cast.json
```
Expected: for each of the two wrappers, the checker's `PASS`, then
`PASS: reproduces the commit byte-for-byte (cxx_move_proof).`, then `EXIT=0`.

The `--grep` patterns do not collide: `(cxx-move,` does not occur inside `(cast-cxx-move,`, because the opening
parenthesis precedes the id.

- [ ] **Step 2: Commit the proof folder**

```bash
git add analysis/layer-fusion-split/proof/
git commit -m "layer-fusion-split(proofs,non_mechanical_provable): move proofs for the layer-fusion and cast-fusion splits

manifest-cxx.json and manifest-cast.json with their chain-verifier wrappers for the two
C++ move commits, the Repros for the two Python move commits, and the reproduction utils
they import.

Task 5 Step 11 suites on divix01: SUITE_UNIT <paste summary line> EXIT=0; SUITE_GPU <paste summary line> EXIT=0.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019mB5q8wVhMMYiPPWtNuwcH"
```
Before running the command, replace the two `<paste summary line>` markers with the actual summary lines from Task 5
Step 11.

- [ ] **Step 3: Verify the whole chain**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_cli.py \
  --base $(git merge-base master layer-fusion-split) --branch layer-fusion-split \
  --proof analysis/layer-fusion-split/proof \
  --report /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split-chain-report.md; echo "EXIT=$?"
```
Expected:
- `EXIT=0`, and the report's chain verdict is PASS.
- 4 commits are `mechanical_provable` with PASS: cxx-move, python-move, cast-cxx-move and cast-python-move.
- 9 commits are `non_mechanical_provable`.
- There is no `UNCLASSIFIED`, `MISSING_PROOF` or `AMBIGUOUS_*` row.

Then do the reviewer audit that `guide-verify-proof.md` section 1 requires for the nine `non_mechanical_provable`
commits. `git show --stat <sha>` for each is enough to confirm its file set:
- launcher-checks-red adds tests only;
- launcher-checks changes launchers and wrapper checks only;
- cxx-prepare and cast-prepare change comments only;
- postpare changes only string literals, namespace lines and the two helper names, and deletes its shim. Check with
  `git show <sha> | grep -E '^[-+]' | grep -vE 'namespace|kLayerFusionWarp|kDirectGatherWarp|layer_fusion_to_float|route_tables_to_float|dsv41_|cuda_files|_gpu|^(\+\+\+|---)'`,
  which must print nothing but the deleted shim's two include lines;
- cast-postpare changes only string literals and namespace lines, and deletes its shim;
- cast-device-check-red adds one test file only;
- cast-device-check changes only the four `with_device` calls and deletes the two `set_options` lines;
- proofs adds `analysis/` only.

- [ ] **Step 4: Final run and baseline comparison on divix01**

Push and update the divix01 worktree to the tip. Run `SUITE_UNIT` and `SUITE_GPU`, then compare with
`/mnt/nvme1/layer-fusion-split/baseline.txt`.

Expected:
- `SUITE_UNIT`: counts identical to the baseline, including `test_exl3_cast_fusion_cpu.py`.
- `SUITE_GPU`: the baseline plus exactly `19 passed` (17 layer-fusion refusal cases and 2 cast-fusion ones). The skips must be the same, and so must
  any environmental outcome for `test_exl3_moe_split_parity_cuda.py`.

Any other delta is a regression. Bisect it over the 13 commits; each one builds on its own.

- [ ] **Step 5: Decide the A/B arm (none needed) and record why**

No performance arm is run. Report this reasoning to the user with the final counts:
1. **The GPU code is unchanged.**
   - Every `__global__` body and every launch line of all five kernels is unchanged. Task 1's diff guard found no
     hunk touching a `__global__`, `LaunchKernel(` or `<<<` line, the two C++ moves are byte-proven, and Task 5
     Step 10 changes only host-side device checks.
   - Each kernel that runs keeps its compile flags: the layer-fusion modules and the scale module build without
     extra flags, and the silu module builds with `-use_fast_math`. The same source under the same flags gives the
     same SASS.
2. **Decode replays a CUDA graph.** In production the launchers run only at capture time, so Task 1's host-side
   checks cost nothing per decode step. They are metadata reads with no device sync, as the captured-graph test in
   `test_dsv41_layer_fusion_gpu.py` shows: a sync during capture would fail it.
3. **Kernel base names are unchanged.** Only the namespace prefix is new, so `kernel_groups.py`'s substring keys
   classify a new trace the same way as an old one, and new traces stay comparable with old ones.
4. **Cold compile is a startup cost, not a decode cost.** The first load of each renamed module compiles on every
   machine. It happens in the warmup forward, before capture, so it adds nothing to traced or steady-state
   ms/token.

An A/B arm is warranted only if Step 4 shows a parity or count regression, and by construction it shows neither.

- [ ] **Step 6: Hand off**

Leave the branch unmerged. Report to the controller:
- the branch tip and the chain report path;
- the Step 4 counts next to the baseline;
- that `expert_residency_gpu.py` changed only at `:793` and `:872`, and `exl3.py` only at `:230` and `:434`;
- that `test_expert_residency_gpu.py`, which the doorbell plan edits, was run here but not edited;
- the Follow-up below (the same device footgun in `route_quant_fused.cuh` and `route_radix.cuh`);
- that the five renamed JIT modules compile cold on the first load on every machine, and that production must seed
  its JIT cache first if it runs with `SGLANG_CRASH_ON_JIT_COMPILE`.

Merging is the controller's decision (superpowers:finishing-a-development-branch). After the merge, remove both
worktrees:

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4 worktree remove /home/dimitri/data/divix/sglang-nvfp4-worktrees/layer-fusion-split
ssh divix01 'git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-layer-fusion-split'
```

---

## Follow-ups (not in this plan)

- **The same device footgun outside this plan's files.** `route_quant_fused.cuh:84-96` and `route_radix.cuh:604-615`
  (and `:675-682`) call `device.set_options<kDLCUDA>()` and then an untemplated `.with_device(device)`, so they
  accept any device. The fix is Task 5 Steps 9-11's pattern: a CPU-input red test, then `.with_device<kDLCUDA>(device)`.
  Neither file is touched by this plan or by the doorbell plan.

---

## Files touched by this plan (for the overlap check against the doorbell-removal plan)

| Action | Path |
|---|---|
| Create | `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` |
| Create | `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh` |
| Create | `python/sglang/kernels/jit/csrc/moe/exl3/exl3_silu_mul_clamp_half.cuh` |
| Create | `python/sglang/kernels/jit/csrc/moe/exl3/exl3_scale_to_bf16.cuh` |
| Create | `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` |
| Create | `python/sglang/kernels/ops/moe/exl3_route_tables.py` |
| Create | `python/sglang/kernels/ops/moe/exl3_cast_fusion.py` |
| Create | `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` |
| Create | `test/manual/dsv41/test_exl3_cast_fusion_launcher_checks_gpu.py` |
| Create | `analysis/layer-fusion-split/proof/manifest-cxx.json` |
| Create | `analysis/layer-fusion-split/proof/manifest-cast.json` |
| Create | `analysis/layer-fusion-split/proof/<cxx-move sha10>.py` |
| Create | `analysis/layer-fusion-split/proof/<python-move sha10>.py` |
| Create | `analysis/layer-fusion-split/proof/<cast-cxx-move sha10>.py` |
| Create | `analysis/layer-fusion-split/proof/<cast-python-move sha10>.py` |
| Create | `analysis/layer-fusion-split/proof/mechanical_refactor_reproduction_utils.py` (verbatim copy) |
| Modify, then delete | `python/sglang/kernels/jit/csrc/moe/dsv41_layer_fusion.cuh` |
| Modify, then delete | `python/sglang/kernels/ops/moe/dsv41_layer_fusion.py` |
| Modify, then delete | `python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh` |
| Delete | `python/sglang/kernels/ops/moe/dsv41_cast_fusion.py` |
| Modify | `python/sglang/srt/layers/moe/expert_residency_gpu.py` (only the import lines `:793` and `:872`; the doorbell plan defers its own edits to this file) |
| Modify | `python/sglang/srt/layers/quantization/exl3_fused_moe.py` (only the import line `:128`) |
| Modify | `python/sglang/srt/layers/quantization/exl3.py` (only the import lines `:230` and `:434`; not in the doorbell plan) |
| Modify | `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py` (only the import line `:216`) |
| Modify | `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py` (only the import line `:15`) |

Nothing else changes. This plan touches none of the doorbell plan's files: `expert_hot_cache.py`,
`expert_row_plan.py`, `model_runner.py`, `scheduler.py`, `offload_presets.py`, `environ.py`,
`ops/moe/expert_doorbell.py`, `expert_doorbell.cuh`, or their tests. It only runs, and never edits,
`test_expert_residency_gpu.py`, which the doorbell plan edits.

---

## Self-review

1. **Spec coverage.**
   - Split by owner: Tasks 2-3.
   - Generic residency file and matching ops module: `expert_residency/direct_gather.cuh` and
     `expert_residency_direct_gather.py`.
   - EXL3 file and matching ops module: `exl3/exl3_route_tables.cuh` and `exl3_route_tables.py`.
   - No adapter template: stated in Architecture.
   - `TensorMatcher` + `verify_named` with `with_device<kDLCUDA>` and width <= 32 in C++: Task 1.
   - The `with_device` footgun: Global Constraints and the greps in Tasks 1 and 4. The pre-existing instance in the
     cast-fusion launchers is fixed TDD in Task 5 Steps 9-11, after the pure move. The instances in
     `route_quant_fused.cuh` and `route_radix.cuh` are Follow-ups.
   - `topk_ids` -> `expert_to_slot`: a documented precondition, justified in Architecture and in the launcher's
     `\brief`.
   - Byte-identical messages: Global Constraints and Task 0's test grep.
   - Prepare/move/postpare labels: the commit table.
   - `cxx_move_proof.py` reused: Tasks 2, 5 and 6.
   - All importers updated, no shim: Tasks 3 and 5, with the lists confirmed by Task 0's greps.
   - Renames to the new home, which supersede the earlier "keep names" direction: the Architecture rename table,
     applied in the Task 4 and Task 5 Step 7 postpares, never inside a proven move.
   - Tooling grep over `analysis/`, `benchmark/`, `benchmarks/` and `scripts/`: Task 0 Step 2. The one hit keyed on
     kernel names matches old and new names by substring (the reason is in Architecture); historical `.md` files
     stay untouched.
   - Cold-compile note: Architecture, Task 4 Step 4, Task 5 Step 8, and Task 6 Steps 5-6.
   - divix01 protocol and all test lists: Global Constraints `SUITE_*`.
   - New direct-call refusal tests: Task 1.
   - Merge-base comparison and the A/B decision: Task 6 Steps 4-5.
   - Branch, worktree, concurrency (including `exl3.py`) and the full file list: Global Constraints, Task 0 and the
     files table.
   - **Added scope** (cast fusion): one file per kernel under `exl3/`, with the flag stated in each header comment;
     the relative include fixed to `../../deepseek_v4/...`; `-use_fast_math` on the silu module only; the wrapper
     moved to `exl3_cast_fusion.py`; `exl3.py:230,434` and both tests handled (the CPU test's docstring needs no
     edit, and the reason is stated); JIT names checked and then renamed; prepare + proven move with no behaviour
     change in the moves. All of it is in Task 5.
2. **Placeholder scan.** The only angle-bracket markers left are:
   - shas that do not exist until their commit is made (`<... sha10>`, `<this sha>`), each with the command that
     produces it;
   - the `<paste summary line>` markers, each with an instruction to replace it before running the command.
3. **Type and name consistency.**
   - `verify_bool_named`, `_ID_DTYPES`/`_FLOAT_DTYPES`, `_silu_module` and `_scale_module` are spelled the same in
     every task and in the anchor tables.
   - `kLayerFusionWarp` and `layer_fusion_to_float` keep their old names through Tasks 1-3, so that the anchors and
     the byte proofs hold. Task 4 alone renames them to `kDirectGatherWarp` and `route_tables_to_float`.
   - The FFI parameter orders in Task 1's test dicts match the C++ signatures at `dsv41_layer_fusion.cuh:216-227`,
     `:247-263` and `:289-299`.
   - The `repath_import`, `remove_import` and `add_import` targets match the import lines the tasks edit.
4. **Review Focus.** All five lines have tests in Task 1's file:
   - width or routes over 32: 4 cases;
   - wrong device: 3 cases;
   - dtype/instantiation mismatch: 3 cases;
   - companion sizes: 5 cases;
   - delivered without keep: 1 case;
   - plus `keep_not_float32`.

   Task 5 Steps 9-11 add the cast-fusion CPU-input refusal: 2 cases, red first.
