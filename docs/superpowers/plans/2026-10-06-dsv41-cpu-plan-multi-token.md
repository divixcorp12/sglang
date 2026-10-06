# The DSV4.1 CPU plan takes chunks of two tokens: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A forward whose routes put two tokens on one expert runs the EXL3 CPU kernel's DSV4.1 plan
(`ForwardPlan<Dsv41Shape, Isa::Bw>`) instead of falling back, for the whole call, to `ForwardPlan<GenericShape, Bw>`,
with bit-identical output.

**Architecture:**
- `Exl3Quant::dispatch` groups routes into chunks of at most `CHUNK_M` tokens. `CHUNK_M = MAX_M / ACT_ROWS = 4 / 2 = 2`
  in this build, so "m ≥ 2" means exactly **m = 2**. One new case, not a family.
- The register kernels (`register_rows`, `register_band`, `math_avx512.hpp`) get a template parameter `M` (token rows,
  default 1). `M = 2` reads the four quantized rows that `compact_quantize_block` already writes for a two-token chunk,
  and shares each decoded weight tile between both tokens. A new `register_tiles_m2` drives them; `run_tiles` calls it
  for compact input at `m == 2`. Only the DSV4.1 plan hands in compact input, so the generic plan is unchanged.
- The compact scratch is laid out by rows (each chunk's `ACT_ROWS * m` rows follow the previous chunk's) instead of
  one fixed stride per chunk. A call whose chunks are all one token keeps today's offsets exactly.
- `Dsv41Shape::accepts` takes chunks of one or two tokens.
- The grouped multi-band traversal and the 512-wide quantization stay one-token only (see the table in "What assumes
  m == 1").
- Two counters come first, so the change can be measured and tested:
  - the draft CPU thread counts the jobs whose routes share a slot (Task 1);
  - the kernel counts the forwards each plan took (Task 2, `sglang_exl3_cpu::plan_calls`).

**Tech Stack:** C++20 (GCC 15, AVX-512BW intrinsics, OpenMP), the EXL3 torch extension and the expert-stream host module
(JIT), Python/pytest, Google Benchmark (the native bench under `expert_stream/bench`). Every run is on divix01, CPU only.

**Spec:**
- The team lead's brief for this plan (2026-10-05), items 1-5.
- `DSV41_REFERENCE.md` §33.3 item 5 ("The tuned path turns off for chunks with more than one token"), §33.5's v2 note,
  §33.8 ("v2 ... not worth building on this evidence").
- `python/sglang/kernels/jit/csrc/exl3/optimized/README.txt`, "Selection": "Invalid/duplicate routes and multi-token
  chunks retain the generic path."
- `.claude/rules/divix01-run-protocol.md`: how every run below is done.

## Open questions for the owner

Each has a default the plan follows unless the owner says otherwise.

1. **Ship gate (Task 6).**
   - Default proposal:
     - every one-token control (`experts:1/3/5`) has a median p50 within ±2% of BASE;
     - the new plan is no slower than the generic plan on any routed pattern;
     - it is ≥5% faster on `16:3:random` or `16:3:pairs`.
   - If it fails, does the kernel change (Task 5) still merge, or do only the counters and harness (Tasks 1-4) merge?
   - The plan's default is to stop and ask.
2. **Is this worth doing now?**
   - §33.8 shelved graphed DSpark, and this change helps only multi-row callers.
   - Today that is the DSpark draft's CPU share: `DraftCpuThread`, M ≤ 16, the eager and the graphed draft both.
   - The target's `CpuExpertEngine` calls `rows = 1` per job (`host/cpu_experts.h`, `run_job`), so target decode and
     prefill gain nothing until D2-4 exists.
   - Default: land Task 1 first, read the draft's collision rate from one DSpark arm, then decide on Tasks 5-6.
3. **Plan-selection observability.**
   - Task 2 adds two process-wide relaxed counters to the production kernel (one RMW per forward) and a torch op.
   - Alternative: an env knob forcing the generic plan. It would let one build A/B itself, but it is a new
     `EXL3_MOE_*` variable and more code.
   - Default: counters.
4. **Follow-ups.** Grouped traversal and wide (512) quantization at m = 2 are numerically identical, but their
   performance is untested there. Default: out of scope here, listed under Out of scope.

## Global Constraints

- **Branch** `dsv41-cpu-plan-m2`, off `dsv41-dspark-graph`.
  - Laptop worktree: `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2`.
  - Never edit, stage, check out or stash anything in `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-graph`.
    Another agent commits there.
  ```bash
  git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-graph fetch origin
  git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-graph worktree add -b dsv41-cpu-plan-m2 \
    /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 origin/dsv41-dspark-graph
  ```
- **Running code (`.claude/rules/divix01-run-protocol.md`).**
  - Commit, push `origin dsv41-cpu-plan-m2`, then run in the divix01 worktree at the pushed commit. No rsync or scp.
    Fix-up commits are fine; never amend or rebase.
  ```bash
  # laptop
  git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 push origin dsv41-cpu-plan-m2
  # divix01, first time
  ssh divix01 'git -C /data/models/slang/sglang fetch origin \
    && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cpu-plan-m2 origin/dsv41-cpu-plan-m2 \
    && git -C /data/models/slang/nvfp4-work/wt-cpu-plan-m2 log -1 --oneline'
  # divix01, every later sync ("SYNC" below)
  ssh divix01 'git -C /data/models/slang/sglang fetch origin \
    && git -C /data/models/slang/nvfp4-work/wt-cpu-plan-m2 checkout --detach origin/dsv41-cpu-plan-m2 \
    && git -C /data/models/slang/nvfp4-work/wt-cpu-plan-m2 log -1 --oneline'
  ```
- **CPU only, no GPU lock, no production.**
  - CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS` capped.
  - The benches pin cores 18-33 themselves. `run_exl3_cpu_forward_checks.sh` and `ab.sh` refuse to run while
    `sglang.launch_server` is running. Never stop a server to make room; wait, or ask the owner.
- **Read `PIPESTATUS`.** Every pytest piped to `tail` is followed by `echo "EXIT=${PIPESTATUS[0]}"`. Record the exact
  command next to every number you quote.
- **The interpreter trap.** Every Python run sets `PYTHONPATH=$PWD/python` in the worktree and checks
  `sglang.__file__` once per session (Task 2 Step 0).
- **Kernel environment ("KENV").** Create this file once on divix01, in Task 2 Step 0. It is outside the repo.
  `/data/models/slang/nvfp4-work/cpu-plan-m2/env.sh`:
  ```bash
  W=/data/models/slang/nvfp4-work/cpu-plan-m2
  GXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++
  export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
  export SGLANG_EXL3_BUILD_DIR=$W/exl3-build TMPDIR=$W/tmp
  export SGLANG_DSV41_CPU_EXPERTS=1 SGLANG_EXL3_CPU_CXX=$GXX CXX=$GXX CUDA_HOME=/usr/local/cuda-13.4
  export EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw OMP_NUM_THREADS=8
  mkdir -p "$TMPDIR" "$SGLANG_EXL3_BUILD_DIR"
  [[ -d $SGLANG_EXL3_BUILD_DIR/resid_b128_cpu_v1 ]] || cp -a ~/.cache/sglang/exl3_ext/resid_b128_cpu_v1 "$SGLANG_EXL3_BUILD_DIR/"
  ```
- **Cold JIT.**
  - A header change rebuilds the EXL3 extension (minutes) and the host module (50-100 s per variant).
  - The first run after one may hit a child-process timeout. Check the build directory's mtimes before calling that
    a regression.
  - Run these suites serially, not `-n 8`.
- **Numerics contract (the plan's acceptance condition).**
  - **m = 1:** byte-identical to today. A call whose chunks are all one token runs the same code at the same scratch
    offsets.
    - Proven by the 24 frozen bare-forward outputs, the 48 full-stack outputs, and the A/B dumps against BASE.
  - **m = 2:** byte-identical to the generic plan.
    - Both keep exact per-row int32 block sums, one FMA scale per (k-block, row), fp32 sums in increasing-k-block
      order, and base plus residual added once.
    - Both also use the dispatch's accumulation order (expert ascending, then token, then route; FP16-rounded
      weights), which this plan does not touch.
    - Proven three ways:
      - the A/B dumps against BASE, where those cases ran the generic plan;
      - the swizzled layer, which the DSV4.1 plan refuses, so it runs the generic plan in the same process;
      - each token run alone.
- **No new environment variables.** No change to `CHUNK_M`, `MAX_M`, the dispatch's grouping or the generic plan's
  kernels.
- **BASE** is the commit that ends Task 4. It has every counter and harness change and no kernel change. Record its
  sha in Task 4's last step. Every A/B compares against it.

## Review Focus

1. **A call that mixes one- and two-token chunks, in any order.** The compact scratch offsets are now a running row
   sum. Expect bits equal to the generic plan's. Pinned by `mixed-sizes` and `three-tokens-one-expert` (Task 2's test,
   flipped in Task 5) and AB case `(3, 1)` (Task 3).
2. **One token routed twice to one expert.** This makes a chunk of two holding the same token twice; both rows are added
   into its out row in route order. Expect the generic plan's bits. Pinned by `one-token-twice` (Task 2) and AB case
   `(1, 2)` (Task 3). The draft counter counts it as a collision (Task 1's test, job 3).
3. **A one-expert call of two tokens.** Here `nc == 1`, so `grouped` is passed as true while `wide` is false. The m = 2
   path must ignore `grouped`. Pinned by `two-tokens-one-expert` (Task 2) and AB case `(2, 1)` (Task 3).
4. **Tile-pair ranges that split a 128-output block across workers at odd team sizes.** The down phase's per-block
   atomic transform now finishes two token rows. Expect bits independent of the team. Pinned by
   `test_dsv41_plan_bits_do_not_depend_on_the_team` at 1, 3 and 16 threads (Task 2, plan check flipped in Task 5).
5. **Two forwards with two-token chunks running at once on different core groups.** Each has its own thread-local
   arena, which now grows by rows. Expect each group's output to equal the one-group output. Pinned by Task 5 Step 9:
   `exl3_cpu_forward_ab.py dump --registration cores` over Task 3's routes.

---

## What assumes m == 1, and what this plan does with each piece

| Piece | Where | m == 1 assumption | Decision |
|---|---|---|---|
| `Dsv41Shape::accepts` | `shapes.hpp:42-50` | refuses any chunk with `m != 1` | **Generalise**: refuse only `m < 1 \|\| m > 2` (Task 5) |
| Compact scratch sizing and offsets | `forward_plan.hpp:547-551, 570-574` | `nc * ACT_ROWS * H` int16, chunk `j` at `j * ACT_ROWS * H` | **Generalise**: running row offset (`ACT_ROWS * m` rows per chunk); all-one-token calls keep today's offsets (Task 5) |
| Compact quantization `compact_quantize_block` | `math.hpp:543-556` | none: already lays out `m` rows (`block * ACT_ROWS * m * 128 + row * 128`, residual rows at `r + m`) | **Unchanged** |
| `prepare_gu_blocks`, `middle_blocks` | `forward_plan.hpp:380-444` | none: loop `r < ch.m`, residual at `r + ch.m` | **Unchanged** |
| `register_rows` / `register_band` | `math_avx512.hpp:929-1001` | two quantized rows (`i < 2`), k-block stride 256, stores row 0 = base + residual, row 1 = residual | **Generalise**: template `M` (default 1): `2 * M` rows, stride `2 * M * 128`, stores per token; at `M = 1` the same code (Task 5) |
| `register_tiles` (compact, unswizzled) | `math_avx512.hpp:1106-1121` | called only at `m == 1` (`run_tiles` :201) | **Keep m == 1**. New `register_tiles_m2` for `m == 2`, compact and unswizzled only (Task 5) |
| Grouped multi-band traversal `traversal_tiles` / `traversal_kblock` / `traversal_group` | `math_avx512.hpp:1003-1093` | hard-wired four bands × two rows (`IntegerAccum [4][2][2]`, 16 ZMM) | **Keep m == 1**. At m = 2 four bands would need 32 integer accumulators. `register_tiles_m2` ignores `grouped` |
| Wide 512 quantization | `forward_plan.hpp:524`, `math.hpp:522-541` | `wide = m_total == 1 && nc == 1` | **Keep m == 1** (out of scope). It is numerically identical (exact max and int sums), so widening it is a perf-only follow-up |
| Down phase, `kSplitTiles = 2` | `forward_plan.hpp:696-717` | none: `transform_owned_blocks` loops `ch.m` rows, and the block counter counts tiles, not rows | **Unchanged**; Review Focus 4 tests it |
| Accumulate | `forward_plan.hpp:718-734` | none | **Unchanged** |
| `run_tiles` gate | `forward_plan.hpp:199-206` | register path only at `m == 1` | **Add** `m == 2 && in.compact` → `register_tiles_m2` (Task 5). Generic input never has `compact` set |

History (`git log -p --follow` on `forward_plan.hpp` and its predecessor `moe_mul1.cpp`): the m == 1 limit came with
the imported "selected" kernels (`e5e5e8cec2`, 2026-10-01), measured and validated for one-token decode only. In
`00bdf9d5e5^`, `single_expert_quant512` was "Compact scratch already guarantees the DSV4.1 shape, unswizzled 3-bit
weights and one row per chunk". `00bdf9d5e5` turned that comment into `Dsv41Shape::accepts`. `58668eedfc` added
`kSplitTiles = 2`. None of them records a correctness reason beyond the two-row register kernels.

## File map

| File | Change | Task |
|---|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h` | collision counters | 1 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` | `draft_cpu_stats` returns 7 values | 1 |
| `python/sglang/kernels/ops/moe/dspark_draft_cpu.py` | `stats()` keys | 1 |
| `test/registered/unit/kernels/test_dspark_draft_cpu_thread.py` | collision test | 1 |
| `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h`, `kernel.cpp`, `forward_plan.hpp`, `torch_ops.cpp` | plan counters and op | 2 |
| `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` (new) | parity and plan tests | 2, 5 |
| `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` | runs the new test | 2 |
| `test/manual/dsv41/exl3_cpu_forward_ab.py` | more routes | 3, 5 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/cpu_forward.cpp`, `bench/ab.sh` (new), `bench/README.txt` | routed workloads, A/B driver | 4 |
| `python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp`, `forward_plan.hpp`, `shapes.hpp`, `README.txt` | the kernel change | 5 |
| `DSV41_REFERENCE.md` | results | 7 |

---

### Task 1: The draft CPU thread counts jobs whose routes share a slot

The measurement to take before (or alongside) the kernel work: how often a real draft call has a chunk of two.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h` (`serve` :175-222, getters :116-127, members :267)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h:616-627`
- Modify: `python/sglang/kernels/ops/moe/dspark_draft_cpu.py:193-197`
- Test: `test/registered/unit/kernels/test_dspark_draft_cpu_thread.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `DraftCpuThread::collided_jobs()`, `shared_routes()` and `collided_forward_ns()` (all `int64_t`).
  - `expert_stream_draft_cpu_stats(handle, out)` with `out` an int64 tensor of shape `[7]`.
  - `DraftCpuHost.stats()` gains the keys `"collided_jobs"`, `"shared_routes"` and `"collided_forward_ns"`.
  - `DraftCpuExperts.stats()` (`srt/layers/moe/cpu_experts/draft.py:142`) passes them through. Its close-time log line
    `DSpark CPU experts: {...}` carries them with no change.

- [ ] **Step 1: Write the failing test.** Append to `test/registered/unit/kernels/test_dspark_draft_cpu_thread.py`,
  after `test_one_stage_serves_m_rows`:

```python
def test_stats_count_the_jobs_whose_routes_share_a_slot(request):
    """The EXL3 kernel groups a call's live routes by slot (two tokens per chunk), so a job whose routes name one slot
    twice runs a chunk of two: collided_jobs counts those jobs, shared_routes the routes past each slot's first, and
    collided_forward_ns their forward time. -1 is not a route."""
    areas, host = _host(request, ns_per_expert=1000)
    k = 3
    jobs = [
        [[0, 1, 2], [3, 4, 5]],  # every slot once
        [[0, 1, 2], [2, -1, 5]],  # slot 2 twice
        [[7, 7, -1]],  # one token routed twice to slot 7
    ]
    for seq, slots in enumerate(jobs, start=1):
        rows = len(slots)
        _stage(areas, 0, rows, k, seed=seq)
        areas.slots[0, :rows, :k] = torch.tensor(slots, dtype=areas.slots.dtype)
        _post(areas, 0, rows, k, seq=seq)
        _finish(areas, seq)
    stats = host.stats()
    assert (stats["jobs"], stats["collided_jobs"], stats["shared_routes"]) == (3, 2, 2)
    assert 0 < stats["collided_forward_ns"] < stats["forward_ns"]
```

- [ ] **Step 2: Commit, push, SYNC, and run it to see it fail.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add test/registered/unit/kernels/test_dspark_draft_cpu_thread.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(dspark): the draft CPU thread counts the jobs whose routes share a slot (failing)"
```
Then push and SYNC (Global Constraints), and run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_dspark_draft_cpu_thread.py -q -p no:randomly \
  -k share_a_slot 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `1 failed` with `KeyError: 'collided_jobs'`, and `EXIT=1`.

- [ ] **Step 3: Implement the counters in `draft_cpu_thread.h`.**

  1. Add `#include <algorithm>` to the includes, in sorted position before `<atomic>`.
  2. After `holds()` (:125-127), add:
  ```cpp
  /// Jobs whose live routes name one slot more than once: the EXL3 kernel runs each with a chunk of two tokens.
  int64_t collided_jobs() const {
    return collided_jobs_.load(std::memory_order_relaxed);
  }
  /// Live routes past each slot's first, summed over jobs.
  int64_t shared_routes() const {
    return shared_routes_.load(std::memory_order_relaxed);
  }
  /// Forward time of the collided jobs, in ns.
  int64_t collided_forward_ns() const {
    return collided_forward_ns_.load(std::memory_order_relaxed);
  }
  ```
  3. In `serve`, after the compaction loop (:193-197), add:
  ```cpp
    const int shared = count_shared_routes(slots, rows * k);
  ```
  4. Then, after `add(rows_, rows);` (:218):
  ```cpp
    if (shared > 0) {
      add(collided_jobs_, 1);
      add(shared_routes_, shared);
      add(collided_forward_ns_, end - start);
    }
  ```
  5. Before `add` (:257), add the helper:
  ```cpp
  /// The call's live routes (slot >= 0) past each slot's first. Outside the timed forward; at most kMaxRows * kMaxK.
  static int count_shared_routes(const int32_t* slots, int n) {
    int32_t live[kMaxRows * kMaxK];
    int count = 0;
    for (int i = 0; i < n; ++i)
      if (slots[i] >= 0) live[count++] = slots[i];
    std::sort(live, live + count);
    int shared = 0;
    for (int i = 1; i < count; ++i) shared += live[i] == live[i - 1];
    return shared;
  }
  ```
  6. Change the member line (:267) to:
  ```cpp
  std::atomic<int64_t> jobs_{0}, rows_{0}, forward_ns_{0}, holds_{0}, collided_jobs_{0}, shared_routes_{0},
      collided_forward_ns_{0};
  ```

- [ ] **Step 4: Export the counters.**

  1. In `ffi_exports.h`, replace `draft_cpu_stats` (:616-627) with:
  ```cpp
  // int64 [7]: jobs, rows, forward ns, keep-warm holds, collided jobs, shared routes, collided forward ns.
  static void draft_cpu_stats(int64_t handle, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    expert_stream::verify_named("out", TensorMatcher({7}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const std::shared_ptr<draft::DraftCpuThread> thread = find_draft(handle);
    auto* o = static_cast<int64_t*>(out.data_ptr());
    o[0] = thread->jobs();
    o[1] = thread->rows();
    o[2] = thread->forward_ns();
    o[3] = thread->holds();
    o[4] = thread->collided_jobs();
    o[5] = thread->shared_routes();
    o[6] = thread->collided_forward_ns();
  }
  ```
  2. In `dspark_draft_cpu.py`, replace `stats` (:193-197) with:
  ```python
    def stats(self) -> dict:
        out = torch.zeros(7, dtype=torch.int64)
        self._module.expert_stream_draft_cpu_stats(self.handle, out)
        jobs, rows, forward_ns, holds, collided_jobs, shared_routes, collided_forward_ns = (int(v) for v in out.tolist())
        return {
            "jobs": jobs,
            "rows": rows,
            "forward_ns": forward_ns,
            "keep_warm_calls": holds,
            "collided_jobs": collided_jobs,
            "shared_routes": shared_routes,
            "collided_forward_ns": collided_forward_ns,
        }
  ```

- [ ] **Step 5: Commit, push, SYNC, and run the whole file.** The host module rebuilds cold here.

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/ops/moe/dspark_draft_cpu.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(dspark): the draft CPU thread counts collided jobs, shared routes and their forward time"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_dspark_draft_cpu_thread.py \
  test/registered/unit/kernels/test_dspark_draft_cpu_experts.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: all pass, `EXIT=0`. If a child times out on the first run, check the host module's build-dir mtimes (cold
JIT), then rerun.

- [ ] **Step 6: Ask the owner for one DSpark arm (GPU; the owner schedules it).** That arm's close-time log line
  `DSpark CPU experts: {...}` gives `collided_jobs / jobs` and `collided_forward_ns / forward_ns`. Record both in
  Task 7. This step does not block Tasks 2-5.

---

### Task 2: The kernel counts the forwards each plan took; the parity test

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h`
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.cpp:24-30`
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp:738-754` (`run_plan`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/torch_ops.cpp`
- Create: `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py`
- Modify: `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` (the pytest step)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `struct SglangExl3CpuPlanCalls { int64_t dsv41; int64_t generic; };` in `kernel.h`.
  - `sglang::exl3_cpu::exl3_cpu_plan_calls()`, hidden visibility, returning `SglangExl3CpuPlanCalls`.
  - The torch op `sglang_exl3_cpu::plan_calls() -> int[]`, returning `[dsv41, generic]`.
  - In the test file: `_expected_plan(routes) -> str` (`"dsv41"` or `"generic"`). Task 5 changes exactly this
    function.

- [ ] **Step 0: Set up the divix01 environment, once.**
  1. Create `/data/models/slang/nvfp4-work/cpu-plan-m2/env.sh` with the KENV content from Global Constraints.
  2. Check the interpreter:
  ```bash
  ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
    && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"'
  ```
  Expected: `/data/models/slang/nvfp4-work/wt-cpu-plan-m2/python/sglang/__init__.py`.

- [ ] **Step 1: Write the test file.** Create `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py`:

```python
"""The EXL3 CPU kernel's DSV4.1 plan on calls whose routes share an expert (csrc/exl3/optimized/forward_plan.hpp).

Exl3Quant::dispatch groups a call's routes by expert into chunks of up to CHUNK_M (2 in this build) tokens. Each case
runs one call on a DeepSeek V4.1-shaped layer:
  - unswizzled (native), which Dsv41Shape::accepts may take on AVX-512BW;
  - swizzled, which it always refuses, so it runs the generic plan;
  - token by token, through the native layer.
The three must agree bit for bit. The plan counter (sglang_exl3_cpu::plan_calls) shows which plan each call took.

Needs SGLANG_EXL3_SRC, SGLANG_DSV41_CPU_EXPERTS=1 and the bw tier (EXL3_MOE_CPU_MAX_ISA=bw on a host above it). Run on
divix01 under taskset -c 0-63.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

H, I, CAP, LIMIT, THREADS = 5120, 2304, 12, 10.0, 4


def _swizzle(t):
    """The swizzled trellis layout over a slab's last three dims [k/16, n/16, 48] (as test_exl3_cpu_act_quant)."""
    tk, tn, ps = t.shape[-3:]
    return t.reshape(-1, tk, tn // 8, 8, ps).permute(0, 2, 1, 3, 4).contiguous().view(t.shape)


def _slabs(seed):
    """The pinned tier's six slabs, CAP slots of random 3-bit codes and signs."""
    g = torch.Generator().manual_seed(seed)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=g) * 2 - 1).half()

    def codes(*shape):
        return torch.randint(-32768, 32767, shape, generator=g, dtype=torch.int16)

    return {
        "w13_trellis": codes(CAP, 2, H // 16, I // 16, 48),
        "w13_suh": signs(CAP, 2, H),
        "w13_svh": signs(CAP, 2, I),
        "w2_trellis": codes(CAP, 1, I // 16, H // 16, 48),
        "w2_suh": signs(CAP, 1, I),
        "w2_svh": signs(CAP, 1, H),
    }


def _random_routes(seed, rows, k):
    g = torch.Generator().manual_seed(seed)
    return [torch.randperm(CAP, generator=g)[:k].tolist() for _ in range(rows)]


# (name, routes [rows][k]; -1 is no route). The comment gives the chunks dispatch makes, in expert order.
CASES = [
    ("one-token", [[4, 0, 5]]),  # 1, 1, 1
    ("two-tokens-one-expert", [[3], [3]]),  # 2: a single-expert call (grouped on, wide off)
    ("two-tokens-three-experts", [[4, 0, 5], [0, 4, 5]]),  # 2, 2, 2
    ("three-tokens-one-expert", [[2], [2], [2]]),  # 2 then 1, one expert
    ("mixed-sizes", [[0, 1, 2], [1, 3, 5]]),  # 1, 2, 1, 1, 1
    ("one-token-twice", [[1, 1, 2]]),  # 2 (the same token twice), 1
    ("dead-routes", [[1, -1, 2], [1, 2, -1]]),  # 2, 2; token 1's route to expert 1 weighs 0
    ("draft-2x3", _random_routes(2, 2, 3)),
    ("draft-4x3", _random_routes(4, 4, 3)),
    ("draft-8x3", _random_routes(8, 8, 3)),
    ("draft-16x3", _random_routes(16, 16, 3)),
    ("prefill-64x6", _random_routes(64, 64, 6)),
]
ZERO_WEIGHT = {"dead-routes": (1, 0)}


def _shares_an_expert(routes):
    live = [s for row in routes for s in row if s >= 0]
    return len(live) != len(set(live))


def _expected_plan(routes):
    """The plan a call on the native layer takes: the DSV4.1 plan unless two of its routes share an expert."""
    return "generic" if _shares_an_expert(routes) else "dsv41"


def _inputs(name, routes):
    g = torch.Generator().manual_seed(sum(map(ord, name)))
    rows, k = len(routes), len(routes[0])
    x = (torch.randn(rows, H, generator=g) * 4.0).half()
    weights = torch.rand(rows, k, generator=g) + 0.05
    if name in ZERO_WEIGHT:
        weights[ZERO_WEIGHT[name]] = 0.0
    return x, torch.tensor(routes, dtype=torch.int32), weights


@pytest.fixture(scope="module")
def layers():
    """{"native": handle, "swizzled": handle} over the same experts."""
    os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")
    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the kernel under test is the optimized one: set SGLANG_DSV41_CPU_EXPERTS=1")
    ext = exl3_ext()
    if not ext.exl3_moe_cpu_has_avx512_bw():
        pytest.skip("the DSV4.1 plan needs AVX-512BW")
    slabs = _slabs(20261006)
    handles = {}
    for layout in ("native", "swizzled"):
        swizzled = layout == "swizzled"
        trait = Exl3CpuQuantTrait(ext, act_limit=LIMIT, swizzled=swizzled)
        held = {n: _swizzle(t) if swizzled and n.endswith("_trellis") else t for n, t in slabs.items()}
        handles[layout] = es.kernel_layer(trait.kernel_address(), trait.layer_spec(held, CAP), variant="instr")
    yield handles
    for handle in handles.values():
        es.kernel_drop(handle, variant="instr")


def _forward(layer, x, slots, weights, threads=THREADS):
    """One call, unpinned; returns (out, the plan it took) from the plan counter's step."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    before = torch.ops.sglang_exl3_cpu.plan_calls()
    out = torch.full((x.shape[0], H), float("nan"))
    status, why = es.kernel_forward(layer, x, slots, weights, out, threads=threads, variant="instr")
    assert (status, why) == (0, "")
    after = torch.ops.sglang_exl3_cpu.plan_calls()
    step = (after[0] - before[0], after[1] - before[1])
    assert step in ((1, 0), (0, 1)), step
    return out, "dsv41" if step == (1, 0) else "generic"


@pytest.mark.parametrize("name,routes", CASES, ids=[c[0] for c in CASES])
def test_dsv41_plan_matches_the_generic_plan_and_one_token_runs(layers, name, routes):
    x, slots, weights = _inputs(name, routes)
    got, plan = _forward(layers["native"], x, slots, weights)
    want, generic = _forward(layers["swizzled"], x, slots, weights)
    assert (plan, generic) == (_expected_plan(routes), "generic")
    assert torch.isfinite(got).all()
    assert torch.equal(got, want)
    singles = torch.cat(
        [_forward(layers["native"], x[t : t + 1], slots[t : t + 1], weights[t : t + 1])[0] for t in range(len(routes))]
    )
    assert torch.equal(got, singles)


@pytest.mark.parametrize("threads", [1, 3, 16])
def test_dsv41_plan_bits_do_not_depend_on_the_team(layers, threads):
    routes = dict(CASES)["draft-16x3"]
    x, slots, weights = _inputs("draft-16x3", routes)
    want, _ = _forward(layers["swizzled"], x, slots, weights)
    got, plan = _forward(layers["native"], x, slots, weights, threads=threads)
    assert plan == _expected_plan(routes)
    assert torch.equal(got, want)
```

- [ ] **Step 2: Commit, push, SYNC, and run it to see it fail.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(exl3-cpu): the DSV4.1 plan against the generic plan and one-token runs, with the plan each call took (failing)"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `15 failed`, each on a missing `plan_calls` op (`AttributeError`/`RuntimeError` naming
`sglang_exl3_cpu.plan_calls`), and `EXIT=1`.

- [ ] **Step 3: Add the counters.**

  1. In `kernel.h`, after `struct SglangExl3CpuParams { ... };`:
  ```cpp
  // Forwards since load by the plan they took (forward_plan.hpp's run_plan): dsv41, ForwardPlan<Dsv41Shape, Bw>;
  // generic, ForwardPlan<GenericShape, *>. Read by tests and benches to see which plan a call took.
  struct SglangExl3CpuPlanCalls {
      int64_t dsv41;
      int64_t generic;
  };
  ```
  2. Inside `namespace sglang::exl3_cpu { ... }`, after `exl3_cpu_kernel()`:
  ```cpp
  __attribute__((visibility("hidden"))) SglangExl3CpuPlanCalls exl3_cpu_plan_calls();
  ```
  3. In `forward_plan.hpp`, just above `run_plan` (:738):
  ```cpp
  // Forwards by plan since load: [0] the DSV4.1 plan, [1] the generic plan. Relaxed: counts only, read through
  // exl3_cpu_plan_calls (kernel.cpp, the one file that includes this header).
  std::atomic<int64_t> g_plan_calls[2];
  ```
  4. In `run_plan`, put `g_plan_calls[0].fetch_add(1, std::memory_order_relaxed);` as the first statement inside the
     `if (isa == Isa::Bw && Dsv41Shape::accepts(...))` block. Put
     `g_plan_calls[1].fetch_add(1, std::memory_order_relaxed);` right after that block, before
     `const Experts<GenericShape> E{&l, p};`.
  5. In `kernel.cpp`, inside `namespace sglang::exl3_cpu`, after `exl3_cpu_kernel()`:
  ```cpp
  SglangExl3CpuPlanCalls exl3_cpu_plan_calls()
  {
      return {g_plan_calls[0].load(std::memory_order_relaxed), g_plan_calls[1].load(std::memory_order_relaxed)};
  }
  ```
  6. In `torch_ops.cpp`:
     - Add `#include <vector>`.
     - Inside the anonymous namespace:
       ```cpp
       std::vector<int64_t> plan_calls()
       {
           const SglangExl3CpuPlanCalls c = ::sglang::exl3_cpu::exl3_cpu_plan_calls();
           return {c.dsv41, c.generic};
       }
       ```
     - In `TORCH_LIBRARY`: `m.def("plan_calls() -> int[]", &plan_calls);`.
     - Extend the header comment's first sentence with: ", and sglang_exl3_cpu::plan_calls, the forwards each plan
       took ([dsv41, generic])".
  7. In `run_exl3_cpu_forward_checks.sh`, add `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` to the
     final `step pytest` line's file list.

- [ ] **Step 4: Commit, push, SYNC, and run.** The extension rebuilds cold.

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h \
  python/sglang/kernels/jit/csrc/exl3/optimized/kernel.cpp python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp \
  python/sglang/kernels/jit/csrc/exl3/optimized/torch_ops.cpp test/manual/dsv41/run_exl3_cpu_forward_checks.sh
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(exl3-cpu): the kernel counts the forwards each plan took (sglang_exl3_cpu::plan_calls)"
```
Then run the same command as Step 2.

Expected: `15 passed`, `EXIT=0`. Today a call whose routes share an expert runs the generic plan on both layers, so
"generic" is expected there. Generic output already equals the one-token DSV4.1 runs bitwise (as
`test_exl3_cpu_act_quant.py::test_batch_matches_single_token_runs` pins). If `one-token` reports `generic`, the tier
is not bw: check `EXL3_MOE_CPU_MAX_ISA`.

---

### Task 3: The A/B harness covers two-token chunks

The A/B dumps of `exl3_cpu_forward_ab.py` at BASE ran the generic plan for these cases. The same dumps at the head
run the DSV4.1 plan. The cases must exist before BASE, or the two dumps hold different case sets.

**Files:**
- Modify: `test/manual/dsv41/exl3_cpu_forward_ab.py:26-33` (`ROUTES`)

**Interfaces:**
- Consumes: nothing.
- Produces: dump case names `{shape}/l{limit}/t{tokens}k{k}/s{scale}` for the new `(tokens, k)` keys. Task 5 compares
  them against BASE.

- [ ] **Step 1: Replace `ROUTES` and its comment:**

```python
# (tokens, experts per token) -> routes. Two tokens on one expert make a chunk of two (CHUNK_M): (2, 3) shares experts
# 0 and 4; (2, 1) is one expert's chunk of two; (3, 1) one expert's chunks of two and one; (1, 2) routes one token
# twice to expert 1; (6, 3) and (16, 3) mix both sizes, as a DSpark draft call does. The DSV4.1 plan refuses chunks of
# two: at the DSV4.1 shape those cases run the generic plan.
ROUTES = {
    (1, 1): [[2]],
    (1, 3): [[4, 0, 5]],
    (1, 5): [[1, 3, 0, 5, 2]],
    (2, 3): [[4, 0, 5], [0, 2, 4]],
    (1, 2): [[1, 1]],
    (2, 1): [[3], [3]],
    (3, 1): [[2], [2], [2]],
    (6, 3): [[(t + 2 * j) % CAP for j in range(3)] for t in range(6)],
    (16, 3): [[(5 * t + j) % CAP for j in range(3)] for t in range(16)],
}
```

- [ ] **Step 2: Commit, push, SYNC, and run one bw dump and the two-core-group mode.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add test/manual/dsv41/exl3_cpu_forward_ab.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(exl3-cpu): the A/B harness routes two tokens to one expert"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && export PYTHONPATH=$PWD/python \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa bw \
     --registration slabs --out $W/t3-bw.pt 2>&1 | tail -1; echo "EXIT=${PIPESTATUS[0]}"; \
  taskset -c 0-63 /data/models/slang/.venv/bin/python test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa bw \
     --registration cores --cores 0-7 --out $W/t3-cores.pt 2>&1 | tail -1; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected:
- `72 outputs (bw, slabs) -> .../t3-bw.pt` with `EXIT=0`. That is 9 routes × 2 scales × 2 shapes × 2 limits.
- `72 outputs (bw, cores) -> .../t3-cores.pt` with `EXIT=0`.

---

### Task 4: The bench runs routed multi-token workloads; an A/B driver

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/cpu_forward.cpp`
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/README.txt` (a "Routed workloads and A/B" section)

**Interfaces:**
- Consumes:
  - `SglangExl3CpuPlanCalls` and `sglang::exl3_cpu::exl3_cpu_plan_calls()` (Task 2, `kernel.h`).
- Produces:
  - Flags `--routed=M:k:pattern[,...]`, with pattern one of `distinct`, `pairs` or `random`; `--routed-slots=N`
    (default 48); and `--routed-layers=N` (default 2).
  - Benchmarks named `optimized/rows:M/k:K/PATTERN`.
  - Counters `dsv41_calls` and `generic_calls` on every optimized benchmark.
  - `bash ab.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [flags]`, which writes `summary.tsv`.

- [ ] **Step 1: Run the bench with `--routed` to see it fail.** Use the bench build at the Task 3 commit:

```bash
ssh divix01 'bash -s' <<'EOF'
cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
taskset -c 0-63 cmake -S python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $W/bench-head -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src > /dev/null
taskset -c 0-63 cmake --build $W/bench-head -j16 --target exl3_cpu_optimized 2>&1 | tail -1
export EXL3_MOE_CPU_MAX_ISA=bw OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16
$W/bench-head/exl3_cpu_optimized --validate-only --routed=2:3:pairs; echo "EXIT=$?"
EOF
```
Expected: Google Benchmark reports `unrecognized command-line flag: --routed=2:3:pairs`, and `EXIT=1`.

- [ ] **Step 2: Implement routed workloads in `cpu_forward.cpp`.**

  1. **Includes.** Add `#include <numeric>`, `#include <random>`, `#include <string>` and `#include <vector>` to the
     include block, in sorted position.
  2. **`Options`.** After `bool validate_only = false;`:
  ```cpp
  std::vector<std::string> routed;  // --routed=M:k:pattern,...: routed workloads (optimized build only)
  int routed_slots = 48;
  int routed_layers = 2;
  ```
  3. **`parse_options`.** Before `else if (arg == "--validate-only")`:
  ```cpp
    else if (arg.starts_with("--routed=")) {
      std::stringstream list(value("--routed="));
      for (std::string item; std::getline(list, item, ',');)
        opt.routed.push_back(item);
    } else if (arg.starts_with("--routed-slots="))
      opt.routed_slots = number(value("--routed-slots="));
    else if (arg.starts_with("--routed-layers="))
      opt.routed_layers = number(value("--routed-layers="));
  ```
     Then append to the `--help` text:
     `"--routed=M:k:distinct|pairs|random,... --routed-slots=48 --routed-layers=2 (optimized build)\n"`.
  4. **`SlabLayer`.** Fill `slots` slots, slot `s` holding fixture expert `s % experts`. The existing workloads pass
     `slots <= 5`, so they copy exactly what they copied before:
  ```cpp
  SlabLayer(const LayerFixture& f, int hidden, int slots) {
    const int experts = static_cast<int>(f.matrices[0].size());
    ::sglang::cpu_experts::ExpertLayer shape;
    shape.capacity = slots;
  ```
     In the copy loop, change `for (int e = 0; e < experts; ++e)` to `for (int s = 0; s < slots; ++s)`,
     `dst = slabs[n].get() + e * row` to `dst = slabs[n].get() + s * row`, `(*part)[e]` to `(*part)[s % experts]`,
     and the allocation size `experts * row` to `slots * row`. Update the comment to: "One fixture layer as the
     pinned tier holds it: `slots` slots, slot s holding fixture expert s % 5, in six slabs ...".
  5. **`Workload`.** Add the following and delete `state.counters["experts"] = workload.experts;` from
     `full_forward`:
  ```cpp
  void label(benchmark::State& state) const {
    state.counters["experts"] = experts;
  }
  ```
  6. **`RoutedWorkload`.** Add after `struct Workload { ... };`:
  ```cpp
  #ifndef EXL3_BENCH_BASELINE
  // The slab layers every routed workload shares: the first --routed-layers fixture layers, --routed-slots slots each.
  using RoutedLayers = std::vector<std::unique_ptr<SlabLayer>>;

  // One --routed=M:k:pattern workload: M token rows of k routes each, rotating over the routed layers. Token t's input
  // is its layer's fixture input rotated by 641 * t elements. Patterns: distinct (no slot shared: every chunk holds one
  // token), pairs (tokens 2i and 2i + 1 share all k slots: every chunk holds two), random (each token k distinct slots,
  // seeded). No frozen reference: validate checks the outputs are finite and that a second run repeats them bit for bit.
  struct RoutedWorkload {
    const Fixture& fixture;
    const Options& options;
    const RoutedLayers& layers;
    int rows = 0, k = 0;
    std::string pattern;
    std::vector<std::vector<at::Half>> inputs;  // per layer, [rows][hidden]
    std::vector<int32_t> slots;
    std::vector<float> weights;
    std::vector<float> output;

    RoutedWorkload(const Fixture& f, const Options& opt, const RoutedLayers& l, const std::string& spec)
        : fixture(f), options(opt), layers(l) {
      std::stringstream fields(spec);
      std::string m, kk;
      if (!std::getline(fields, m, ':') || !std::getline(fields, kk, ':') || !std::getline(fields, pattern))
        throw std::runtime_error("--routed takes M:k:pattern, not " + spec);
      rows = number(m);
      k = number(kk);
      const int capacity = opt.routed_slots;
      if (pattern != "distinct" && pattern != "pairs" && pattern != "random")
        throw std::runtime_error("Unknown routed pattern: " + pattern);
      const int needed = pattern == "distinct" ? rows * k : pattern == "pairs" ? (rows + 1) / 2 * k : k;
      if (rows < 1 || k < 1 || needed > capacity)
        throw std::runtime_error(spec + " needs " + std::to_string(needed) + " slots of " + std::to_string(capacity));
      std::mt19937 rng(20261006u + 1000u * rows + k);
      std::vector<int32_t> pool(capacity);
      for (int t = 0; t < rows; ++t) {
        std::iota(pool.begin(), pool.end(), 0);
        for (int j = 0; j < k; ++j) {
          int32_t slot;
          if (pattern == "distinct")
            slot = t * k + j;
          else if (pattern == "pairs")
            slot = t / 2 * k + j;
          else {
            std::swap(pool[j], pool[j + rng() % (capacity - j)]);
            slot = pool[j];
          }
          slots.push_back(slot);
          weights.push_back(0.071234f + (k == 1 ? 0.0f : 0.23f * j / (k - 1)));
        }
      }
      for (size_t layer = 0; layer < layers.size(); ++layer) {
        const auto* x = static_cast<const at::Half*>(fixture.layers[layer].input.data_ptr());
        std::vector<at::Half> rows_x(static_cast<size_t>(rows) * f.hidden);
        for (int t = 0; t < rows; ++t)
          for (int i = 0; i < f.hidden; ++i)
            rows_x[static_cast<size_t>(t) * f.hidden + i] = x[(i + 641 * t) % f.hidden];
        inputs.push_back(std::move(rows_x));
      }
      output.resize(static_cast<size_t>(rows) * f.hidden);
    }

    size_t layer_count() const {
      return layers.size();
    }

    void forward(size_t layer) {
      ::sglang::cpu_experts::ForwardCall call;
      call.rows = rows;
      call.k = k;
      call.threads = options.workers;
      call.x = inputs[layer].data();
      call.slots = slots.data();
      call.weights = weights.data();
      call.out = output.data();
      call.cores = g_cores;
      ::sglang::exl3_cpu::exl3_cpu_kernel().forward(layers[layer]->layer, call);
    }

    void validate() {
      for (size_t layer = 0; layer < layers.size(); ++layer) {
        forward(layer);
        const std::vector<float> first = output;
        for (float value : first)
          if (!std::isfinite(value)) throw std::runtime_error("Non-finite routed output");
        forward(layer);
        if (std::memcmp(first.data(), output.data(), first.size() * sizeof(float)))
          throw std::runtime_error("A routed forward did not repeat bit for bit");
      }
    }

    void label(benchmark::State& state) const {
      state.counters["rows"] = rows;
      state.counters["k"] = k;
    }
  };
  #endif
  ```
  7. **`full_forward`.**
     - Make it `template <class W> void full_forward(benchmark::State& state, W& workload)`.
     - Immediately before `std::vector<double> samples;`:
       ```cpp
       #ifndef EXL3_BENCH_BASELINE
           const SglangExl3CpuPlanCalls before = ::sglang::exl3_cpu::exl3_cpu_plan_calls();
       #endif
       ```
     - Immediately after the timed `for` loop:
       ```cpp
       #ifndef EXL3_BENCH_BASELINE
           const SglangExl3CpuPlanCalls after = ::sglang::exl3_cpu::exl3_cpu_plan_calls();
           state.counters["dsv41_calls"] = static_cast<double>(after.dsv41 - before.dsv41);
           state.counters["generic_calls"] = static_cast<double>(after.generic - before.generic);
       #endif
       ```
     - Replace the deleted `experts` counter line with `workload.label(state);`.
  8. **`main`.** After the `for (int experts : {1, 3, 5})` loop:
  ```cpp
  #ifndef EXL3_BENCH_BASELINE
      RoutedLayers routed_layers;
      std::vector<std::unique_ptr<RoutedWorkload>> routed;
      if (!options.routed.empty()) {
        if (options.routed_layers < 1 || options.routed_layers > int(fixture.layers.size()) || options.routed_slots < 1)
          throw std::runtime_error("Invalid --routed-layers or --routed-slots");
        for (int layer = 0; layer < options.routed_layers; ++layer)
          routed_layers.push_back(std::make_unique<SlabLayer>(fixture.layers[layer], fixture.hidden, options.routed_slots));
        for (const auto& spec : options.routed) {
          routed.push_back(std::make_unique<RoutedWorkload>(fixture, options, routed_layers, spec));
          routed.back()->validate();
        }
        std::cerr << "Verified " << routed.size() << " routed workloads repeat bit for bit\n";
      }
  #else
      if (!options.routed.empty()) throw std::runtime_error("--routed runs on the optimized build only");
  #endif
  ```
     Then, in the registration block, after the `workloads` loop:
  ```cpp
  #ifndef EXL3_BENCH_BASELINE
        for (auto& workload : routed) {
          auto* w = workload.get();
          benchmark::RegisterBenchmark((std::string(EXL3_BENCH_BACKEND) + "/rows:" + std::to_string(w->rows) + "/k:" +
                                        std::to_string(w->k) + "/" + w->pattern).c_str(),
                                       [w](benchmark::State& state) { full_forward(state, *w); })
              ->UseManualTime()
              ->Unit(benchmark::kMicrosecond);
        }
  #endif
  ```
  9. **File comment.** Update the header comment's map:
     `Workload / RoutedWorkload   one expert count's layers (frozen references) / one routed M-row workload`.

- [ ] **Step 3: Write `bench/ab.sh`.**

```bash
#!/usr/bin/env bash
# A/B of two builds of the bare-forward bench (exl3_cpu_optimized): BASE_BUILD and HEAD_BUILD run as separate processes,
# alternating order each round, for EXL3_BENCH_ROUNDS rounds (default 8), each with the same flags. Writes every round's
# Google JSON and log, and summary.tsv: per benchmark, the median over rounds of p50_us for each build, head / base, and
# the plan each build ran (from the dsv41_calls / generic_calls counters). Refuses while a server runs: it pins 18-33.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: bash ab.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [benchmark options...]" >&2
  exit 2
fi
base=$(realpath "$1")
head=$(realpath "$2")
results=$3
shift 3
if pgrep -f sglang.launch_server > /dev/null; then
  echo "a server is running: the bench pins CPUs 18-33" >&2
  exit 2
fi
mkdir "$results"  # Refuse to overwrite a prior record.
results=$(realpath "$results")
rounds=${EXL3_BENCH_ROUNDS:-8}
export OMP_DYNAMIC=FALSE OMP_THREAD_LIMIT=16 OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=INFINITE
export EXL3_MOE_CPU_PIN=0 EXL3_MOE_CPU_MAX_ISA=bw EXL3_MOE_CPU_SMALL_WORKERS=0
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
{
  date -Is
  printf 'Arguments:'
  printf ' %q' "$@"
  printf '\n'
  sha256sum "$base/exl3_cpu_optimized" "$head/exl3_cpu_optimized"
} > "$results/environment.txt"
for ((round = 0; round < rounds; ++round)); do
  order=(base head)
  if ((round % 2)); then order=(head base); fi
  for side in "${order[@]}"; do
    build=$base
    [[ $side == head ]] && build=$head
    "$build/exl3_cpu_optimized" --benchmark_min_time=256x --benchmark_repetitions=1 \
      --benchmark_out="$results/round-$round-$side.json" --benchmark_out_format=json "$@" \
      > "$results/round-$round-$side.log" 2>&1
  done
done
"${PYTHON:-python3}" - "$results" "$rounds" <<'PY'
import json, statistics, sys

results, rounds = sys.argv[1], int(sys.argv[2])
rows = {}
for side in ("base", "head"):
    for r in range(rounds):
        for b in json.load(open(f"{results}/round-{r}-{side}.json"))["benchmarks"]:
            name = b["name"].split("/", 1)[1].removesuffix("/manual_time")
            entry = rows.setdefault(name, {}).setdefault(side, {"p50": [], "plan": set()})
            entry["p50"].append(b["p50_us"])
            entry["plan"].add("dsv41" if b["generic_calls"] == 0 else "generic" if b["dsv41_calls"] == 0 else "mixed")
with open(f"{results}/summary.tsv", "w") as out:
    out.write("benchmark\tbase_p50_us\thead_p50_us\thead/base\tbase_plan\thead_plan\n")
    for name, sides in rows.items():
        b, h = statistics.median(sides["base"]["p50"]), statistics.median(sides["head"]["p50"])
        line = f"{name}\t{b:.1f}\t{h:.1f}\t{h / b:.3f}\t{'/'.join(sorted(sides['base']['plan']))}\t{'/'.join(sorted(sides['head']['plan']))}"
        out.write(line + "\n")
        print(line)
PY
echo "Results: $results"
```
Make it executable: `chmod +x python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh`.

- [ ] **Step 4: Document both in `bench/README.txt`.** Add this section after "Run":

```text
Routed workloads and A/B
------------------------
The optimized build also runs M-row calls, as the DSpark draft's CPU thread does:
  --routed=M:k:pattern[,...]  pattern distinct (no slot shared), pairs (tokens 2i, 2i+1 share all k slots) or random
  --routed-slots=48           slab slots per routed layer; slot s holds fixture expert s % 5
  --routed-layers=2           fixture layers the routed workloads rotate over (shared by all of them)
Token t's input is the layer's fixture input rotated by 641 * t elements. There is no frozen reference: validation
checks the outputs are finite and repeat bit for bit. Every optimized benchmark reports dsv41_calls and generic_calls,
the forwards each plan ran (exl3_cpu_plan_calls).

ab.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [flags] runs two optimized builds as alternating processes for
EXL3_BENCH_ROUNDS rounds (default 8) and writes summary.tsv (median p50 per build, head/base, the plan each ran).
```

- [ ] **Step 5: Commit, push, SYNC, rebuild, and validate.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/kernels/jit/csrc/moe/expert_stream/bench
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(cpu-bench): routed M-row workloads, plan counters and an A/B driver"
```
Then run Step 1's command again, replacing its last line with:
```bash
$W/bench-head/exl3_cpu_optimized --validate-only \
  --routed=2:3:pairs,16:3:pairs,16:3:distinct,6:3:random,16:3:random,64:6:random; echo "EXIT=$?"
```
Expected on stderr:
- `Verified 24 bit-exact layer outputs; ...`
- `Verified 6 routed workloads repeat bit for bit`
- `EXIT=0`

- [ ] **Step 6: Record BASE.** Run
  `git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 rev-parse --short HEAD`. That sha is **BASE**.
  Create the base worktree on divix01:
```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cpu-plan-m2-base <BASE> \
  && git -C /data/models/slang/nvfp4-work/wt-cpu-plan-m2-base log -1 --oneline'
```

---

### Task 5: The DSV4.1 plan takes chunks of two tokens

**Files:**
- Modify: `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` (`_expected_plan`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp:929-1001` (`register_rows`, `register_band`), plus a new `register_tiles_m2` after `register_tiles` (:1121)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp:199-206` (`run_tiles`), `:529-596` (`prepare_scratch`), `:491-509` (PlanTraits comments)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/shapes.hpp:39-50` (`accepts`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/README.txt` ("Selection", "Code layout")
- Modify: `test/manual/dsv41/exl3_cpu_forward_ab.py` (the `ROUTES` comment)

**Interfaces:**
- Consumes: `torch.ops.sglang_exl3_cpu.plan_calls` (Task 2); the A/B cases (Task 3); BASE (Task 4).
- Produces:
  - `register_rows<J, Pairs, Compact, M = 1>` and `register_band<Pairs, FixedK, FixedN, Compact, M = 1>`;
    `M` is the number of token rows.
  - `constexpr int kTwoTokenPairs = 2;` and
    `void register_tiles_m2(const Exl3Projection&, const PreparedIn&, float* tout, int t0, int t1)`.
  - `Dsv41Shape::accepts` takes `m ∈ {1, 2}`.

- [ ] **Step 0: Start the BASE baseline dumps in the background.** They need no change from this task:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && nohup bash wt-cpu-plan-m2-base/test/manual/dsv41/run_exl3_cpu_forward_checks.sh \
  baseline wt-cpu-plan-m2-base cpu-plan-m2/checks-base > cpu-plan-m2/checks-base.out 2>&1 &'
```
Expected at the end of `cpu-plan-m2/checks-base.out`: `ALL GREEN (baseline)`.

- [ ] **Step 1: Make the test expect the DSV4.1 plan for every native call.** Replace `_expected_plan` with:

```python
def _expected_plan(routes):
    """The plan a call on the native layer takes: the DSV4.1 plan, whose chunks hold one or two tokens."""
    return "dsv41"
```

- [ ] **Step 2: Commit, push, SYNC, and run to see it fail.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(exl3-cpu): a call whose routes share an expert takes the DSV4.1 plan (failing)"
```
Then run Task 2 Step 2's command.

Expected: `EXIT=1`, with failures of the form `assert ('generic', 'generic') == ('dsv41', 'generic')`:
- every hand-written case except `one-token`;
- `draft-16x3`, `prefill-64x6`, and all three `threads` cases;
- any random draft case whose routes collide.

`one-token` passes.

- [ ] **Step 3: Generalise the register kernels to M token rows (`math_avx512.hpp`).**

**3a.** `register_rows`: replace its template line, parameter list and inner row loop. Keep the body between them as
it is.

```cpp
template<int J,int Pairs,bool Compact=false,int M=1>
M1_TARGET_BW M1_ALWAYS_INLINE void register_rows(__m512i prev,__m512i a,__m512i b,__m512i c,
    const std::conditional_t<Compact,int16_t,int32_t>* splat,__m512i (&acc)[Pairs][2][2*M],int band) {
```
```cpp
        // 2 * M quantized rows, 128 apart: M token rows, then their M residual rows (compact_quantize_block).
        for(int i=0;i<2*M;++i) {
            uint32_t pair;std::memcpy(&pair,reinterpret_cast<const char*>(splat+i*128+row)+(Compact?0:2),4);
            acc[band][half][i]=_mm512_add_epi32(acc[band][half][i],_mm512_madd_epi16(sum,_mm512_set1_epi32(int32_t(pair))));
        }
        register_rows<J+1,Pairs,Compact,M>(prev,a,b,c,splat,acc,band);
```

**3b.** `register_band`: replace the whole function with this. At `M = 1` it is the same code: `R = 2`, stride
`kb * 256`, and the same stores in the same order.

```cpp
// M token rows (1, or 2 for a two-token chunk of the DSV4.1 plan) share each decoded tile. Per output: the int32 sum of
// each (k-block, quantized row), one FMA scale each, fp32 sums in increasing k-block order, then token + residual once:
// the generic path's rounding (bw3_blocked_band).
template<int Pairs,int FixedK=0,int FixedN=0,bool Compact=false,int M=1>
M1_TARGET_BW void register_band(const Exl3Projection& mat,const PreparedIn& in,float* tout,int n0) {
    constexpr int R=2*M;  // quantized rows: M token rows, then their M residual rows
    const int k=FixedK?FixedK:mat.k,n=FixedN?FixedN:mat.n;
    const bool swz=FixedK?false:mat.swz;
    const int tiles_k=k/16,tiles_n=n/16;
    const size_t step=size_t(swz?8:tiles_n)*96;
    __m512 sums[Pairs][2][R];
    alignas(64) static constexpr int previous[16]={7,0,1,2,3,4,5,6,15,8,9,10,11,12,13,14};
    for(int kb=0;kb<k/128;++kb) {
        __m512i acc[Pairs][2][R];
        for(int b=0;b<Pairs;++b)for(int h=0;h<2;++h)for(int i=0;i<R;++i)acc[b][h][i]=_mm512_setzero_si512();
        for(int kt=0;kt<8;++kt) {
            const auto* splat=[&]() {
                    if constexpr(Compact) return in.compact+size_t(kb)*R*128+kt*16;
                    else return in.splat_dup+size_t(kb)*R*128+kt*16;
                }();
            for(int band=0;band<Pairs;++band) {
                const int nt=n0+2*band,tile_k=kb*8+kt;
                const size_t tile=swz?size_t(nt/8)*tiles_k*8+size_t(tile_k)*8+nt%8:size_t(tile_k)*tiles_n+nt;
                const uint32_t* packed=reinterpret_cast<const uint32_t*>(mat.trellis+tile*48);
                const uintptr_t future=reinterpret_cast<uintptr_t>(packed)+step*2;
                #pragma GCC unroll 3
                for(int line=0;line<3;++line)_mm_prefetch(reinterpret_cast<const char*>(future+line*64),_MM_HINT_T0);
                const __m512i p0=_mm512_loadu_si512(packed),p1=_mm512_loadu_si512(packed+16),p2=_mm512_loadu_si512(packed+32);
                const __m512i a=register_word<0>(p0,p1,p2),b=register_word<1>(p0,p1,p2),c=register_word<2>(p0,p1,p2);
                const __m512i prev=_mm512_permutexvar_epi32(_mm512_load_si512(previous),c);
                register_rows<0,Pairs,Compact,M>(prev,a,b,c,splat,acc,band);
            }
        }
        __m512 scales[R],corrections[R];
        for(int i=0;i<R;++i) {
            const float scale=0x1.bb8p-8f*in.bq[kb*MAX_M+i];
            scales[i]=_mm512_set1_ps(scale);
            corrections[i]=_mm512_set1_ps(-510.0f*float(in.bsum[kb*MAX_M+i])*scale);
        }
        for(int b=0;b<Pairs;++b)for(int h=0;h<2;++h)for(int i=0;i<R;++i) {
            const __m512 v=_mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][h][i]),scales[i],corrections[i]);
            if(kb==0)sums[b][h][i]=v;else sums[b][h][i]=_mm512_add_ps(sums[b][h][i],v);
        }
    }
    // Token row t at tout + t * n gets token + residual; its residual row stays at tout + (M + t) * n, as the generic
    // path leaves it.
    for(int b=0;b<Pairs;++b)for(int t=0;t<M;++t) {
        float* row=tout+size_t(t)*n;
        float* res=tout+size_t(M+t)*n;
        const __m512 lo=_mm512_add_ps(sums[b][0][t],sums[b][0][M+t]);
        const __m512 hi=_mm512_add_ps(sums[b][1][t],sums[b][1][M+t]);
        _mm512_storeu_ps(row+(n0+2*b)*16,_mm512_shuffle_f32x4(lo,hi,0x44));
        _mm512_storeu_ps(row+(n0+2*b+1)*16,_mm512_shuffle_f32x4(lo,hi,0xee));
        _mm512_storeu_ps(res+(n0+2*b)*16,_mm512_shuffle_f32x4(sums[b][0][M+t],sums[b][1][M+t],0x44));
        _mm512_storeu_ps(res+(n0+2*b+1)*16,_mm512_shuffle_f32x4(sums[b][0][M+t],sums[b][1][M+t],0xee));
    }
}
```

**3c.** After `register_tiles` (:1121), add:

```cpp
// Compact input of a two-token chunk (the DSV4.1 plan, unswizzled): tile pairs through register_band at M = 2, at most
// kTwoTokenPairs pairs per call (each pair holds eight integer accumulators). No grouped traversal: that is one token's.
constexpr int kTwoTokenPairs = 2;

M1_TARGET_BW void register_tiles_m2(const Exl3Projection& mat,const PreparedIn& in,float* tout,int t0,int t1) {
    TORCH_CHECK(in.compact && !mat.swz && t0%2==0 && t1%2==0,"two-token input must be compact, unswizzled tile pairs");
    for(int n0=t0;n0<t1;) {
        const int pairs=std::min(kTwoTokenPairs,(t1-n0)/2);
        switch(pairs) {
            case 1:register_band<1,0,0,true,2>(mat,in,tout,n0);break;
            case 2:register_band<2,0,0,true,2>(mat,in,tout,n0);break;
            case 3:register_band<3,0,0,true,2>(mat,in,tout,n0);break;
            default:register_band<4,0,0,true,2>(mat,in,tout,n0);break;
        }
        n0+=pairs*2;
    }
}
```

**3d.** In the file header comment, change "register_tiles and its traversal" to "register_tiles, register_tiles_m2
and the traversal".

- [ ] **Step 4: Route compact two-token chunks to it, and lay out compact scratch by rows (`forward_plan.hpp`).**

**4a.** In `run_tiles`, inside `if constexpr (I == Isa::Bw && ACT_ROWS == 2 && EXL3_MOE_CPU_ACT_BLOCK == 128)` and after
the `m == 1` block:

```cpp
        // Only the DSV4.1 plan hands in compact input (PlanTraits::kCompactScratch); its chunks hold one or two tokens.
        if (m == 2 && in.compact)
        {
            register_tiles_m2(mat,in,tout,tn0,tn1);
            return;
        }
```

**4b.** In `prepare_scratch`:
1. Replace the `} else { grow(ar.compact_g, ...` branch with:
   ```cpp
        } else {
            // Compact: each chunk's ACT_ROWS * m rows follow the previous chunk's (a call of one-token chunks keeps chunk
            // j at j * ACT_ROWS rows).
            size_t rows = 0;
            for (const Chunk& ch : ctx.chunks) rows += size_t(ACT_ROWS) * ch.m;
            grow(ar.compact_g,rows*H);
            grow(ar.compact_u,rows*H);
            grow(ar.compact_d,rows*I_);
        }
   ```
2. Before the `for (int j = 0; j < nc; ++j)` loop that fills `ctx.prep_*`, declare `size_t row0 = 0;`.
3. Replace its `if(compact) { ... }` block with:
   ```cpp
            if(compact) {
                ctx.prep_g[j].compact=ar.compact_g.data()+row0*H;
                ctx.prep_u[j].compact=ar.compact_u.data()+row0*H;
                ctx.prep_d[j].compact=ar.compact_d.data()+row0*I_;
                row0+=size_t(ACT_ROWS)*ctx.chunks[j].m;
            }
   ```

**4c.** In `PlanTraits`, change these two comments to:
- `kGroupedTraversal`: `// range-aware multi-band traversal when one expert is routed (one-token chunks only)`
- `kWideSingleExpert`: `// 512-wide quantization for one token through one expert`, which stays as is.

- [ ] **Step 5: Accept chunks of two (`shapes.hpp`).** Replace the comment and the loop in `accepts`:

```cpp
    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // every fact above (shape, limit 10, unswizzled 3-bit), and every chunk holds one or two tokens (the register
    // kernels' M; CHUNK_M is 2 in this build).
    static bool accepts(const ExpertLayer& l, const Exl3Quant::Params& p, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        if (l.hidden != kHidden || l.intermediate != kIntermediate || l.act_limit != kActLimit) return false;
        if (p.bits != kBits || p.swizzled) return false;
        for (const auto& ch : chunks)
            if (ch.m < 1 || ch.m > 2) return false;
        return true;
    }
```

- [ ] **Step 6: Update the documentation the change falsifies.**

  1. In `README.txt` "Selection", replace the first paragraph (through "...other ISA/bit-width fallbacks are
     retained.") with:
  ```text
  For a call whose chunks each hold one or two tokens, with H=5120, I=2304, residual activation rows,
  128-element quantization blocks, 3-bit unswizzled matrices and AVX512BW:
    one expert, one token: range3_unroll (range-aware neighboring-band traversal);
    other one-token chunks: unroll_small (one-band traversal);
    two-token chunks: register_band at M = 2 (kTwoTokenPairs pairs per call), both tokens
    sharing each decoded tile.
  The measured multiple-expert counts are 3 and 5; counts 2, 4 and above 5
  use the same default but have no measured performance claim. Two-token chunks
  are bit-identical to the generic plan (test_exl3_cpu_dsv41_plan_multitoken.py).
  Invalid routes retain the generic path. Existing scalar, AVX2 and other
  ISA/bit-width fallbacks are retained.
  ```
  2. In "Code layout", change "compact scratch, grouped traversal, wide single-expert quantization" to "compact
     scratch (one- and two-token chunks), grouped traversal and wide single-expert quantization (one token)".
  3. In `exl3_cpu_forward_ab.py`, replace the `ROUTES` comment's last sentence with: "The DSV4.1 plan takes chunks of
     one or two tokens, so at the DSV4.1 shape these cases run it: a dump made before 2026-10-06 ran them on the
     generic plan, and the two must agree bit for bit."

- [ ] **Step 7: Commit, push, SYNC, and run the parity test and the existing multi-row tests.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/kernels/jit/csrc/exl3/optimized \
  test/manual/dsv41/exl3_cpu_forward_ab.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(exl3-cpu): the DSV4.1 plan takes chunks of two tokens, bit-identical to the generic plan"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py test/manual/dsv41/test_exl3_cpu_act_quant.py \
  "test/manual/dsv41/test_cpu_expert_engines_exl3.py::test_a_multi_row_forward_on_prod_matches_one_row_forwards" \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `19 passed` (15 + 3 + 1), `EXIT=0`.

`test_batch_matches_single_token_runs` now compares the DSV4.1 plan's two-token chunks with its one-token runs.
`test_swizzled_layout_matches_native` now compares them with the generic plan.

- [ ] **Step 8: Run the full bit-exact gate against BASE.** First confirm Step 0's `checks-base.out` ends
  `ALL GREEN (baseline)`. Then:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && bash wt-cpu-plan-m2/test/manual/dsv41/run_exl3_cpu_forward_checks.sh \
  check wt-cpu-plan-m2 cpu-plan-m2/checks-head cpu-plan-m2/checks-base 2>&1 | tail -16; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: every step `EXIT=0` and `ALL GREEN (check)`:
- `compare-bw`, `compare-avx2` and `compare-scalar` print `72/72 bit-exact`. This is the m = 2 parity against the
  generic plan at BASE and the m = 1 identity, in one comparison.
- `bare-validate` prints `Verified 24 bit-exact layer outputs`.
- `full-stack-validate` passes (48 frozen outputs).
- `pytest` passes.

On a mismatch, `compare-bw` names the case (`MISMATCH dsv41/l10/t16k3/s8.0: ...`). Stop and debug with
superpowers:systematic-debugging. Do not adjust a reference.

- [ ] **Step 9: Run two core groups at once with two-token chunks (Review Focus 5).**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python test/manual/dsv41/exl3_cpu_forward_ab.py \
  dump --isa bw --registration cores --cores 0-7 --out $W/t5-cores.pt 2>&1 | tail -1; echo "EXIT=${PIPESTATUS[0]}"; \
  /data/models/slang/.venv/bin/python test/manual/dsv41/exl3_cpu_forward_ab.py compare $W/checks-base/ab-bw-slabs.pt $W/t5-cores.pt \
  2>&1 | tail -1; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `72 outputs (bw, cores)` and `EXIT=0`, then `72/72 bit-exact: bw/slabs vs bw/cores` and `EXIT=0`.

- [ ] **Step 10: Check the one-token path's machine code.** This is informative, not a gate. Build the bench at BASE
  and at head (Task 6 Step 1 builds both; run it first if needed), then:

```bash
ssh divix01 'bash -s' <<'EOF'
W=/data/models/slang/nvfp4-work/cpu-plan-m2
for side in base head; do
  objdump -d --no-show-raw-insn -C $W/bench-$side/exl3_cpu_optimized \
    | awk '/<\(anonymous namespace\)::(traversal_kblock|compact_pairs|register_tiles)\(/,/^$/' \
    | sed -E 's/^ *[0-9a-f]+:[[:space:]]*//; s/\b[0-9a-f]{6,}\b/ADDR/g' > $W/disasm-$side.txt
done
wc -l $W/disasm-base.txt $W/disasm-head.txt; diff -q $W/disasm-base.txt $W/disasm-head.txt && echo IDENTICAL
EOF
```
Expected: `IDENTICAL`. If the code differs, record it with Task 6's one-token controls. Only those timings decide
whether the difference matters.

---

### Task 6: Measure the new plan against the generic plan, and decide

**Files:** none in the repo. Results go under `/data/models/slang/nvfp4-work/cpu-plan-m2/` on divix01 and into Task 7.

**Interfaces:**
- Consumes: `ab.sh` and `--routed` (Task 4), BASE and the base worktree (Task 4), the head (Task 5).
- Produces: `ab-m1-*/summary.tsv`, `ab-routed-*/summary.tsv`, the `kTwoTokenPairs` choice, and the gate decision.

- [ ] **Step 1: Build both benches.** Run this once per side, with `side=base wt=wt-cpu-plan-m2-base` and with
  `side=head wt=wt-cpu-plan-m2`:

```bash
ssh divix01 'bash -s' <<'EOF'
side=base; wt=/data/models/slang/nvfp4-work/wt-cpu-plan-m2-base   # then: side=head; wt=.../wt-cpu-plan-m2
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
taskset -c 0-63 cmake -S $wt/python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $W/bench-$side -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src > /dev/null
taskset -c 0-63 cmake --build $W/bench-$side -j16 --target exl3_cpu_optimized 2>&1 | tail -1
EOF
```
Expected: `[100%] Built target exl3_cpu_optimized` for each side.

- [ ] **Step 2: Run the one-token controls, which must not regress.** Run with no server on the box:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-base cpu-plan-m2/bench-head \
  cpu-plan-m2/ab-m1-$(date +%Y%m%d-%H%M%S) --benchmark_filter=experts: 2>&1 | tail -5'
```
Expected: three rows (`experts:1`, `experts:3`, `experts:5`), each with `dsv41` on both sides. The gate for each
(open question 1): head/base in [0.98, 1.02].

- [ ] **Step 3: Run the routed workloads, the comparison itself:**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-base cpu-plan-m2/bench-head \
  cpu-plan-m2/ab-routed-$(date +%Y%m%d-%H%M%S) \
  --routed=2:3:pairs,16:3:pairs,16:3:distinct,6:3:random,16:3:random,64:6:random --benchmark_filter=rows: 2>&1 | tail -8'
```
Expected: six rows.
- `16:3:distinct` is `dsv41`/`dsv41`, a control that should read ≈1.00.
- Every other row is `generic` on base and `dsv41` on head. The head/base column is the answer.

- [ ] **Step 4: Tune `kTwoTokenPairs` with mutants.** This follows the run protocol: a private worktree, never
  committed. For P in 1, 3 and 4:

```bash
ssh divix01 'bash -s' <<'EOF'
P=4   # then 1, then 3
R=/data/models/slang/sglang; M=/data/models/slang/nvfp4-work/wt-cpu-plan-m2-p$P
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh; SITE=/data/models/slang/.venv/lib/python3.13/site-packages
git -C $R worktree add --detach $M origin/dsv41-cpu-plan-m2
sed -i "s/constexpr int kTwoTokenPairs = 2;/constexpr int kTwoTokenPairs = $P;/" $M/python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp
grep -n "kTwoTokenPairs = " $M/python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp
taskset -c 0-63 cmake -S $M/python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $W/bench-p$P -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src > /dev/null
taskset -c 0-63 cmake --build $W/bench-p$P -j16 --target exl3_cpu_optimized 2>&1 | tail -1
PYTHON=/data/models/slang/.venv/bin/python bash $M/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh \
  $W/bench-head $W/bench-p$P $W/ab-p$P-$(date +%Y%m%d-%H%M%S) \
  --routed=2:3:pairs,16:3:pairs,16:3:random,64:6:random --benchmark_filter=rows: 2>&1 | tail -5
git -C $M checkout -- . && git -C $R worktree remove $M
EOF
```
Expected: four rows per P; "head" in this run means the mutant.
- If a P is ≥3% faster than 2 on `16:3:random` and `16:3:pairs` and slower on none, set `kTwoTokenPairs` to it in a
  commit (`perf(exl3-cpu): two-token chunks take P tile pairs per call`).
- Then rerun Task 5 Steps 7 and 8, and this task's Step 3. Expected: green again, and the bench numbers in that
  record.

- [ ] **Step 5: Decide by the gate (open question 1) and report to the owner.** The report gives both `summary.tsv`
  tables, the Task 5 Step 10 verdict and the Task 1 collision rate if it is in.
  - If the gate passes: proceed to Task 7.
  - If it fails: stop. Report to the owner with the numbers, and do not revert anything without their decision.

---

### Task 7: Record the result

**Files:**
- Modify: `DSV41_REFERENCE.md`. Add §33.9 after §33.8 ("What D2-3 does not do" ends its section).

**Interfaces:**
- Consumes: Task 1 Step 6's collision rate, Task 5 Steps 8-10, and Task 6's tables and decision.
- Produces: nothing code reads.

- [ ] **Step 1: Write §33.9.** Copy every number from the named `summary.tsv` or log, with its command. Use this
  structure:

```markdown
### 33.9 The DSV4.1 CPU plan takes chunks of two tokens (2026-10-06)

Plan `docs/superpowers/plans/2026-10-06-dsv41-cpu-plan-multi-token.md`, branch `dsv41-cpu-plan-m2`. This covers §33.3
item 5's "the tuned path turns off for chunks with more than one token".

**What changed.**
- `Dsv41Shape::accepts` takes chunks of one or two tokens (`CHUNK_M` is 2).
- Two-token chunks run `register_band` at `M = 2`: both tokens share each decoded tile, `kTwoTokenPairs` = <P>.
  The grouped traversal and the wide quantization stay one-token only.
- The kernel counts the forwards per plan (`sglang_exl3_cpu::plan_calls`).
- The draft CPU thread counts collided jobs (`collided_jobs`, `shared_routes`, `collided_forward_ns`).

**Bits.**
- `run_exl3_cpu_forward_checks.sh check` against BASE `<sha>`: 72/72 per tier, 24 + 48 frozen outputs, ALL GREEN.
- `test_exl3_cpu_dsv41_plan_multitoken.py`: 15 passed.
- The one-token path's machine code: <IDENTICAL | differs, see the controls>.

**Time** (`ab.sh`, 8 rounds, 16 workers on 18-33, median p50 µs; <dir>):

| workload | generic (BASE) | DSV4.1 (head) | head/base |
|---|---|---|---|
| experts:1 / 3 / 5 (controls) | | | |
| rows:2/k:3/pairs | | | |
| rows:16/k:3/pairs | | | |
| rows:16/k:3/distinct (control) | | | |
| rows:6/k:3/random | | | |
| rows:16/k:3/random | | | |
| rows:64/k:6/random | | | |

**How often the draft hits it** (Task 1, one DSpark arm, <log path>): collided_jobs / jobs = <x>;
collided_forward_ns / forward_ns = <y>.

**What it does not do:** the target's `CpuExpertEngine` (one token per job; D2-4); grouped traversal or wide quantization
at m = 2; swizzled compact input.
```

- [ ] **Step 2: Commit and push.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add DSV41_REFERENCE.md
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "docs(dsv41): §33.9, the DSV4.1 CPU plan on chunks of two tokens"
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 push origin dsv41-cpu-plan-m2
```

---

## Risks

- **The M = 2 kernel may not beat the generic plan.**
  - `register_band<2,…,M=2>` holds 16 integer accumulators plus 16 fp32 partial sums. GCC will spill the sums, as it
    already does at `M = 1, Pairs = 4`.
  - The generic `bw_tiles<3, 4>` also shares each decode across four rows.
  - Mitigation: Task 6's A/B, the `kTwoTokenPairs` mutants, and the stop-and-ask gate.
- **Templating perturbs the one-token path.**
  - `forward_plan.hpp`'s header warns that inlining and cloning follow definition order.
  - Mitigation: the frozen references (bits), Task 5 Step 10 (machine code), and Task 6 Step 2 (time).
- **Scratch offsets.** A wrong running row offset corrupts only calls that mix chunk sizes, silently.
  Mitigation: `mixed-sizes`, `three-tokens-one-expert`, AB `(3, 1)` / `(6, 3)` / `(16, 3)`.
- **The in-process generic oracle depends on layout independence.**
  - The generic oracle is the swizzled layer. It is valid only while the generic plan gives the same bits for both
    layouts, which `test_swizzled_layout_matches_native` pins today.
  - The A/B dumps against BASE are the independent oracle.
- **Counter contention.**
  - The target engine and the draft thread both touch `g_plan_calls` once per forward: about 100 ns of cache-line
    transfer at worst, against forwards of 300 µs or more.
  - Task 6 Step 2 would show it.
- **Production and shared caches.**
  - The benches pin 18-33 and the scripts refuse while a server runs.
  - The extension builds into the private `SGLANG_EXL3_BUILD_DIR`, never into `~/.cache`.
- **A one-token payoff, measured synthetically.**
  - The routed patterns span the extremes. Only Task 1's real collision rate says how much a DSpark arm gains.

## Out of scope

- Multi-row jobs in the target's `CpuExpertEngine` (`rows = 1`) and the rest of D2-4 (§33.3 item 5's record, wire and
  summed lane weight).
- Grouped multi-band traversal and wide (512) quantization for two-token chunks.
- Swizzled compact input at m = 2: the DSV4.1 plan refuses swizzled layers.
- The generic plan, the AVX2, VNNI and VBMI tiers, and `ACT_ROWS == 1` builds (where `CHUNK_M` is 4).
- Changing `CHUNK_M`, `MAX_M`, the dispatch's grouping or its accumulation order.
- Running a DSpark arm. The owner schedules it; Task 1 Step 6 only reads its log.
- An env knob forcing the generic plan (open question 3).

## Self-review

1. **Spec coverage.**
   - Brief item 1 (which pieces assume m == 1, and what to do with each): the table under "What assumes m == 1".
   - Item 2 (the numerics contract and its proofs): Global Constraints, then Task 5 Steps 7-9.
   - Item 3 (tests):
     - parity at m ∈ {1, 2} (`CHUNK_M` = 2), mixed sizes, draft M = 2..16 with k = 3, prefill 64×6: Tasks 2 and 5;
     - `accepts` takes m = 2, through the plan counter: Task 5 Step 1;
     - the m == 1 tests stay byte-exact: Task 5 Step 8.
   - Item 4 (measurement): the bench A/B in Tasks 4 and 6; the draft counter in Task 1, ordered first.
   - Item 5 (risks, out of scope): the sections above.
   - The branch name, the plan path and the open questions are at the top.
2. **Placeholders.**
   - Only Task 7's result table has blanks, by design: results not yet measured, each tied to the file it is copied
     from.
   - No TBD or TODO, and no step without its code or command.
3. **Type consistency.** These names are used the same way in every task:
   - `SglangExl3CpuPlanCalls{dsv41, generic}`, `exl3_cpu_plan_calls()`, `sglang_exl3_cpu::plan_calls`, `g_plan_calls`;
   - `register_tiles_m2`, `kTwoTokenPairs`, `register_band<…, M>`, `_expected_plan`;
   - `RoutedWorkload`, `RoutedLayers`, `ab.sh` and its `summary.tsv`;
   - `collided_jobs`, `shared_routes`, `collided_forward_ns`, `count_shared_routes`.
4. **Review Focus.** Each of the five lines names its test and the task that owns it.
