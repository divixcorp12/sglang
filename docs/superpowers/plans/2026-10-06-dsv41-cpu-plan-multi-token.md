# The DSV4.1 CPU plan takes chunks of any size up to CHUNK_M: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A forward whose routes put several tokens on one expert runs the EXL3 CPU kernel's DSV4.1 plan
(`ForwardPlan<Dsv41Shape, Isa::Bw>`), with bit-identical output. Today it falls back, for the whole call, to
`ForwardPlan<GenericShape, Bw>`. The token count is generic, so raising `MAX_M` later is a one-constant change.

**Architecture:**
- **What a chunk is.** `Exl3Quant::dispatch` groups routes into chunks of at most `CHUNK_M = MAX_M / ACT_ROWS` tokens.
  In this build `CHUNK_M` is 4 / 2 = 2.
- **One build-time constant.** `MAX_M` becomes `EXL3_MOE_CPU_MAX_M` (default 4), the way `EXL3_MOE_CPU_ACT_BLOCK` is
  done. A test build raises it with `SGLANG_EXL3_CPU_MAX_M`, which gets its own extension build key (Task 4b).
  - Every array, stride and dispatch derives from it. That includes the generic tiers' hand-written
    `bits * 4 + rows - 1` switches, which become tables generated from `MAX_M`.
  - The generic plan's output does not depend on `MAX_M`. Its per-row arithmetic is row-independent and its
    accumulation order is set by the dispatch. Task 4b proves this at `MAX_M = 8` against BASE, on every tier
    divix01 runs.
- **The register kernels, generic in M.** `register_rows`, `register_band` and `register_tiles` (`math_avx512.hpp`)
  are templated on `M`, the number of tokens in a chunk.
  - `run_tiles` dispatches `m = 1..CHUNK_M` through a direct-call sequence generated from `CHUNK_M`.
  - One table, `kRegisterBudget`, gives each M its tile pairs per call and whether it uses the grouped traversal.
  - A `static_assert` proves every entry fits 32 zmm. A `CHUNK_M` without an entry fails to compile, and the message
    names the table.
  - `M = 1` is today's code, byte for byte (Task 5's objdump check).
- **Compact scratch, laid out by rows.** Each chunk's `ACT_ROWS * m` rows follow the previous chunk's. A call of
  one-token chunks keeps today's offsets.
- **What the DSV4.1 plan accepts.** `Dsv41Shape::accepts` takes `1 <= m <= CHUNK_M`. The grouped traversal and the
  512-wide quantization stay one-token only (`kRegisterBudget[0]`, `wide`).
- **Counters come first:**
  - jobs whose routes share a slot, counted by the draft CPU thread (Task 1);
  - forwards per plan and the build's `CHUNK_M`, counted by the kernel (Task 2: `sglang_exl3_cpu::plan_calls` and
    `chunk_m`).
- **Task 5b (conditional).** It raises the default `MAX_M` to 8 (`CHUNK_M` 4) if and only if the m = 2 measurement
  beat the generic plan.

**Tech Stack:** C++20 (GCC 15, AVX-512BW intrinsics, OpenMP, `std::integer_sequence`), the EXL3 torch extension and
the expert-stream host module (JIT), Python/pytest, Google Benchmark (the native bench under `expert_stream/bench`).
Every run is on divix01, CPU only.

**Spec:**
- The team lead's brief (2026-10-05), items 1-5.
- The owner's revision (2026-10-06), relayed by the team lead:
  - the token count is generic and `MAX_M` is one constant with a build-time override;
  - a register-budget table with a static_assert;
  - an audit of every `MAX_M` and `CHUNK_M` use;
  - `accepts` takes `1 <= m <= CHUNK_M`;
  - `CHUNK_M` is exposed to Python, and the parity tests widen with it, including one `MAX_M = 8` run;
  - the conditional Task 5b;
  - results in `DSV41_REFERENCE.md` §33.10.
- `DSV41_REFERENCE.md` §33.3 item 5 ("The tuned path turns off for chunks with more than one token"), §33.5's v2 note,
  and §33.8.
- `python/sglang/kernels/jit/csrc/exl3/optimized/README.txt`, "Selection": "Invalid/duplicate routes and multi-token
  chunks retain the generic path."
- `.claude/rules/divix01-run-protocol.md`, which governs every run below.

## Owner decisions (2026-10-06)

1. **Scope.** Do the whole plan, Tasks 1-7, kernel included.
2. **Ship gate.** It applies to Task 6, and again to Task 5b.
   - Every one-token control (`experts:1/3/5`) has a median p50 within ±2% of the reference build.
   - The new build is no slower than the reference on any routed pattern.
   - It is ≥5% faster on at least one of the named patterns.
   - On a miss, stop and ask. Do not revert without the owner.
3. **Observability.** Counters (`plan_calls`, `chunk_m`), not an env knob that forces the generic plan.
4. **Follow-ups.** The grouped traversal and wide quantization at m ≥ 2 stay a follow-up.

## Global Constraints

- **Branch** `dsv41-cpu-plan-m2`, already created off `dsv41-dspark-graph`.
  - Laptop worktree: `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2`.
  - Never edit, stage, check out or stash anything in `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-graph`.
    Another agent commits there.
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
  - `SGLANG_EXL3_CPU_MAX_M=8` builds a separate extension, flavor `resid_b128_m8_cpu_v1`, under the private
    `SGLANG_EXL3_BUILD_DIR`. It is a fresh build key that touches nobody's cache, and its first build is cold.
  - Warm each new build once with a serial one-line import (Task 4b Step 6) before any test times out on it.
  - Check the build directory's mtimes before calling a timeout a regression. Run these suites serially, not `-n 8`.
- **Numerics contract (the plan's acceptance condition).**
  - **m = 1:** byte-identical to today. A call whose chunks are all one token runs the same register code at the
    same scratch offsets.
    - Proven by the 24 frozen bare-forward outputs, the 48 full-stack outputs, the A/B dumps against BASE, and the
      objdump check.
  - **1 < m <= CHUNK_M:** byte-identical to the generic plan.
    - Both keep exact per-row int32 block sums, one FMA scale per (k-block, quantized row), fp32 sums in
      increasing-k-block order, and token plus residual added once.
    - Both also keep the dispatch's accumulation order (expert ascending, then token, then route; FP16-rounded
      weights), which this plan does not touch.
    - Proven three ways:
      - the A/B dumps against BASE, where those cases ran the generic plan;
      - the swizzled layer, which the DSV4.1 plan refuses, so it runs the generic plan in the same process;
      - each token run alone.
  - **The generic plan at any `MAX_M`:** byte-identical to `MAX_M = 4`. Proven by the A/B dumps of a
    `SGLANG_EXL3_CPU_MAX_M=8` build against BASE's on the scalar, avx2 and bw tiers.
    - The VNNI and VBMI tiers compile at every `MAX_M` but cannot run on divix01 (Xeon Gold 6154, AVX-512BW only).
- **Environment variables.**
  - One new build-only variable, `SGLANG_EXL3_CPU_MAX_M` (Task 4b), defined in `Envs` beside
    `SGLANG_EXL3_CPU_ACT_BLOCK`.
  - Read the `env-var-conventions` skill before adding it (`.claude/rules/modify-component-must-read.md`).
  - No change to the dispatch's grouping or its accumulation order.
  - The default `MAX_M` stays 4 unless Task 5b's gate passes.
- **BASE** is the commit that ends Task 4. It has every counter and harness change and no kernel change.
  - Record its sha in Task 4's last step.
  - Every A/B and every bit-exact comparison is made against BASE's dumps.

## Review Focus

1. **A `MAX_M` raise that misses an index.**
   - The failure: any array, stride or dispatch still sized for 4 rows.
   - Expect a `SGLANG_EXL3_CPU_MAX_M=8` build to give BASE's bits on every tier divix01 runs, and the DSV4.1 plan to
     take chunks of 1-4 with the generic plan's bits.
   - Pinned by Task 4b Step 7 (A/B dumps at `MAX_M = 8` against BASE) and Task 5 Step 12 (the parity file at
     `MAX_M = 8`). That file's `test_every_chunk_size_matches_the_generic_plan` loops `m = 1..CHUNK_M` read from the
     build.
   - The "Every use of MAX_M and CHUNK_M" audit lists each site.
2. **Register spills at a higher M.**
   - The failure: an M whose accumulators spill inside the k-tile loop. That is bit-correct but slow, and invisible
     to every bit test.
   - The `kRegisterBudget` static_assert (Task 5 Step 4) refuses any entry whose accumulators plus decode
     temporaries exceed 32 zmm.
   - Task 5 Step 14 counts the zmm stack accesses of every instantiated `register_band<…, M>` / `register_tiles<M>`
     and gates them against BASE's M = 1 code.
3. **Generic-plan drift at a raised `MAX_M`.**
   - The failure: the generated tile tables, the row-index table or the VBMI band rule changing a bit at
     `MAX_M = 8`.
   - Pinned by Task 4b Step 7, which compares scalar, avx2 and bw against BASE at 72/72 each. It also includes
     swizzled cases through `test_exl3_cpu_act_quant.py` (Task 4b Step 8).
4. **Mixed chunk sizes, and one token routed twice.**
   - The compact scratch offsets are a running row sum. A token routed twice to one expert fills one chunk with the
     same token twice.
   - Expect the generic plan's bits.
   - Pinned by `mixed-sizes`, `three-tokens-one-expert`, `one-token-twice`, the generated `m{CHUNK_M+1}-overflow`
     case (Tasks 2 and 5), and AB cases `(3, 1)` and `(1, 2)` (Task 3). The draft counter counts the second as a
     collision (Task 1).
5. **Odd team sizes and concurrent core groups.**
   - Tile-pair ranges split a 128-output block across workers. The down phase's per-block atomic transform finishes
     m rows. Each thread-local arena grows by rows.
   - Expect bits independent of the team, and two groups at once equal to one.
   - Pinned by `test_dsv41_plan_bits_do_not_depend_on_the_team` at 1, 3 and 16 threads, and by Task 5 Step 11
     (`exl3_cpu_forward_ab.py dump --registration cores`).

---

## What assumes one token per chunk, and what this plan does with each piece

| Piece | Where | Assumption today | Decision |
|---|---|---|---|
| `Dsv41Shape::accepts` | `shapes.hpp:42-50` | refuses any chunk with `m != 1` | **Generalise**: `1 <= m <= CHUNK_M` (Task 5) |
| Compact scratch sizing and offsets | `forward_plan.hpp:547-551, 570-574` | `nc * ACT_ROWS * H` int16; chunk `j` at `j * ACT_ROWS * H` | **Generalise**: running row offset, `ACT_ROWS * m` rows per chunk. All-one-token calls keep today's offsets (Task 5) |
| `compact_quantize_block` | `math.hpp:543-556` | none: lays out `m` rows (`block * ACT_ROWS * m * 128 + row * 128`, residual rows at `r + m`) | **Unchanged** |
| `prepare_gu_blocks`, `middle_blocks` | `forward_plan.hpp:380-444` | none: loop `r < ch.m`, residual at `r + ch.m` | **Unchanged** |
| `register_rows` / `register_band` | `math_avx512.hpp:912-1001` | two quantized rows (`i < 2`), k-block stride 256, stores row 0 = token + residual and row 1 = residual | **Generalise**: template `M`. `2 * M` rows, stride `2 * M * 128`, stores per token. At `M = 1` the same code (Task 5) |
| `register_tiles` | `math_avx512.hpp:1106-1133` | one token (`run_tiles` :201) | **Generalise**: `register_tiles<M>`. `M = 1` keeps today's body (non-compact and swizzled callers, grouped traversal); `M > 1` takes compact, unswizzled tile pairs, `kRegisterBudget[M-1].pairs` per call (Task 5) |
| `run_tiles` register gate | `forward_plan.hpp:199-206` | `m == 1` only | **Generalise**: `m == 1 \|\| in.compact`, dispatched to `register_tiles<1..CHUNK_M>` by a sequence generated from `CHUNK_M` (Task 5). Generic input never has `compact` set |
| Grouped traversal (`traversal_tiles` / `_kblock` / `_group`) | `math_avx512.hpp:1003-1093` | four bands × two rows (`IntegerAccum [4][2][2]`, 16 zmm) | **One token only** (`kRegisterBudget[0].traversal`). Follow-up |
| Wide 512 quantization | `forward_plan.hpp:524`, `math.hpp:522-541` | `wide = m_total == 1 && nc == 1` | **One token only**. Numerically identical, so widening it is a perf-only follow-up |
| Down phase, `kSplitTiles = 2` | `forward_plan.hpp:696-717` | none: `transform_owned_blocks` loops `ch.m` rows; the counter counts tiles | **Unchanged**; Review Focus 5 |
| Accumulate | `forward_plan.hpp:718-734` | none | **Unchanged** |

History (`git log -p --follow` on `forward_plan.hpp` and its predecessor `moe_mul1.cpp`):
- The one-token limit came with the imported "selected" kernels (`e5e5e8cec2`, 2026-10-01), which were measured and
  validated for one-token decode only.
- `00bdf9d5e5^`'s `single_expert_quant512` said "Compact scratch already guarantees the DSV4.1 shape, unswizzled
  3-bit weights and one row per chunk". `00bdf9d5e5` turned that into `Dsv41Shape::accepts`.
- `58668eedfc` added `kSplitTiles = 2`.
- None of them records a correctness reason beyond the two-row register kernels.

## Every use of MAX_M and CHUNK_M (the audit; line numbers at `fbfb799`)

After this plan, every entry derives from the single `MAX_M` (`EXL3_MOE_CPU_MAX_M`). The "fix" rows are the ones that
are hard-coded today.

| Site | File:line | Today | After |
|---|---|---|---|
| `MAX_M` | `math.hpp:46` | `constexpr int MAX_M = 4;` | **fix**: `= EXL3_MOE_CPU_MAX_M` (default 4; static_assert even, 2..8) (Task 4b) |
| `ACT_ROWS`, `CHUNK_M` | `math.hpp:57-58` | derived | unchanged |
| `PreparedIn::q`, `sum_x8` | `math.hpp:340-341` | `[MAX_M]`, one per quantized row | unchanged |
| `PreparedIn::bq`, `bsum` | `math.hpp:342-344`; written `math.hpp:412, 551, 554`, `forward_plan.hpp:408, 438`; read `forward_plan.hpp:240-241`, `math_avx512.hpp:844-845, 982-984, 1045-1047` | `[k / B][MAX_M]`, index `b * MAX_M + row` | unchanged (row < `ACT_ROWS * m` <= `MAX_M`) |
| `quantize_act` layout | `math.hpp:390-417` | `rows = ACT_ROWS * m`, block offset `b * rows * B` | unchanged |
| `compact_quantize_block` | `math.hpp:544-556` | block offset `block * ACT_ROWS * m * 128` | unchanged |
| `Chunk::token`, `weight` | `shapes.hpp:18-19` | `[MAX_M]`; holds <= `CHUNK_M` | unchanged; comment `shapes.hpp:4` says "up to MAX_M token rows", **fix** to CHUNK_M (Task 4b) |
| Dispatch chunk cap | `forward_plan.hpp:790` | `ch.m < CHUNK_M` | unchanged |
| Generic raw tile switches | `forward_plan.hpp:44, 85, 124` | `switch (mat.bits * 4 + m - 1)`: **4 rows hard-coded**, 32 cases per tier | **fix**: `kTiles<I>[bits - 1][rows - 1]`, generated from `MAX_M` (Task 4b) |
| `run_tiles` rows and `part` | `forward_plan.hpp:212-261` | `rows = ACT_ROWS * m`, `part` sized `rows * n`, `sub_in.q[i] = bq[b * MAX_M + i]` | unchanged |
| `run_tiles` register gate | `forward_plan.hpp:199-206` | `m == 1` | **fix**: dispatch sequence over `1..CHUNK_M` (Task 5) |
| Down-input row index | `forward_plan.hpp:692` | `static const int idx4[MAX_M] = {0, 1, 2, 3};` (**4 hard-coded**) | **fix**: `kRowIndex`, `std::array<int, MAX_M>` 0..MAX_M-1 (Task 4b) |
| Arena per-chunk strides | `forward_plan.hpp:536-569, 586-593` (`tin`, `splat`, `splat_dup`, `tout`, `bq`, `bsum`) | `j * MAX_M * {H, I_}`, `MAX_M * (k / 16)` | unchanged |
| `tout` per-chunk strides | `forward_plan.hpp:388-389, 673, 687-688, 702, 725` | `j * MAX_M * {I_, H}` | unchanged (the register stores write rows < `2 * M` <= `MAX_M`) |
| Compact arena | `forward_plan.hpp:547-551, 570-574` | `nc * ACT_ROWS * {H, I_}` (**one token hard-coded**) | **fix**: running rows (Task 5) |
| Scalar tile accumulators | `math_scalar.hpp:27` | `float acc[MAX_M][16]` | unchanged |
| AVX2 row accumulate | `math_avx2.hpp:108-139` (`avx2_accum_row`) | **hand-unrolled `ACC_ROW(0..3)`**: rows 4+ silently dropped at `MAX_M = 8` (found by Task 4b Step 7: 48/72 on avx2) | **fixed** (commit `6d5245771c`, then `4cf19a3f0d`): `avx2_accum_rows<I>` template recursion to `MAX_M`; gate: default check ALL GREEN, bw/avx2/scalar 72/72 against BASE, `MAX_M = 8` dumps 72/72 on all three; the avx2 tile machine code at `MAX_M = 4` differs from the macro version (9817 -> 6504 disassembled lines), outputs identical |
| AVX2 tile accumulators | `math_avx2.hpp:109, 138, 201` | `__m256i acc[MAX_M][2]`, runtime `m` | unchanged (16 ymm at `MAX_M = 8`: correct, may spill; Risks) |
| Sweep for other fixed-row unrolls (2026-10-06) | `math_scalar.hpp`, `math_avx2.hpp`, `math_avx512.hpp` (VNNI, BW, VBMI), `math.hpp`, `forward_plan.hpp`, `shapes.hpp`, `kernel.cpp` | grepped for `[2-8]` array extents, `{0, 1, 2, 3}`, `case N:` on rows, `rows/m` compared to a literal, `ACC_ROW`-style macros, and literals standing for `MAX_M`/`CHUNK_M`/`ACT_ROWS` | no other site: every AVX-512 row loop is `for (i < rows)` with `rows` a template parameter; band widths are formulas of `rows` (`12 / rows`, `rows <= 3 ? 4 : 2`, the VBMI `rows > 4` rule); the `bits` switches (1..8) are bit widths; `IntegerAccum lanes[4]` and `traversal_group` are bands of the one-token traversal (`static_assert(Groups <= 4)`); `register_band` is bounded by `kRegisterBudget`'s `static_assert`s. No new `static_assert` needed. |
| AVX-512 band accumulators | `math_avx512.hpp:101, 165, 300, 394, 659, 709, 814` | `__m512i acc[band][MAX_M]`, `rows` a template parameter | unchanged |
| AVX-512 band widths | `math_avx512.hpp:242-243` (vnni), `518-519` (bw), `778-779` (vbmi) | by `rows`; VBMI swizzled gives `rows > 4` four bands (**32 accumulators at rows 8**) | **fix**: VBMI swizzled `rows > 4` → 2 bands. Every rule keeps `rows * band <= 16` (Task 4b) |
| `bw3_blocked_band` | `math_avx512.hpp:806-857` | `rows = 2` constant (the generic m == 1 odd-tile path) | unchanged |
| Register kernels | `math_avx512.hpp:912-1001, 1106-1121` | two rows, stride 256 | **fix**: `M`, `kRegisterBudget` (Task 5) |
| Traversal | `math_avx512.hpp:1006-1093` | `[4][2][2]` | unchanged; M = 1 only |
| `Dsv41Shape::accepts` | `shapes.hpp:47-48` | `ch.m != 1` | **fix**: `1..CHUNK_M` (Task 5) |
| Python | `test/…`, `exl3_cpu_forward_ab.py` | none | `chunk_m` op (Task 2); the parity tests read it |

Out of scope: the vendored baseline kernel (`csrc/exl3/moe_mul1.cpp`), which has its own `MAX_M`.

## How the plan checks register spills, per M

- **At compile time.** `kRegisterBudget` (`math_avx512.hpp`, Task 5 Step 4) holds one entry per M.
  - A `static_assert` over the whole table checks `4 * M * pairs + kRegisterDecodeZmm <= 32` for every entry. The
    integer accumulators are `[pairs][2 halves][2 * M rows]`. `kRegisterDecodeZmm = 12` covers the decode
    temporaries of `register_rows`: prev, a, b, c, state, sum, two products, the multiplier pair, ones, and the
    broadcast pair.
  - A second `static_assert` refuses a `CHUNK_M` with no entry.
  - So M = 2 can have 1 or 2 pairs (3 would need 36), and M = 3 and 4 can have 1.
- **In the binary.** Task 5 Step 14, and Task 5b Step 3 at `MAX_M = 8`.
  - For each instantiated `register_band<P, 0, 0, true, M>` and `register_tiles<M>`, count the zmm loads and stores
    against `%rsp`/`%rbp` in `objdump -d`, and report `-fstack-usage` frame sizes.
  - The fp32 partial sums `sums[P][2][2M]` live across the k-block loop and may sit in memory by design (the
    validated M = 1, P = 4 code already does that).
  - The gate is per fp32 partial sum: (zmm stack accesses) / (4 · M · P summed over the instantiated P) must not
    exceed BASE's ratio for M = 1 (`register_band<P, 0, 0, true>`, or `compact_pairs` where it is inlined) plus 1.
  - An accumulator spill inside the k-tile loop breaks that ratio. A miss is reported, and the measured timings
    decide.

## File map

| File | Change | Task |
|---|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h` | collision counters | 1 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` | `draft_cpu_stats` returns 7 values | 1 |
| `python/sglang/kernels/ops/moe/dspark_draft_cpu.py` | `stats()` keys | 1 |
| `test/registered/unit/kernels/test_dspark_draft_cpu_thread.py` | collision test | 1 |
| `python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h`, `kernel.cpp`, `forward_plan.hpp`, `torch_ops.cpp` | plan counters, `chunk_m` | 2 |
| `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` (new) | parity, plan and chunk-size tests | 2, 5, 5b |
| `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` | runs the new test | 2 |
| `test/manual/dsv41/exl3_cpu_forward_ab.py` | more routes | 3, 5 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/src/cpu_forward.cpp`, `bench/ab.sh` (new), `bench/README.txt` | routed workloads, A/B driver | 4 |
| `python/sglang/kernels/jit/csrc/exl3/optimized/math.hpp`, `forward_plan.hpp`, `math_avx512.hpp`, `shapes.hpp` | `MAX_M` override, generated tile tables, row index, VBMI band rule | 4b |
| `python/sglang/srt/environ.py`, `python/sglang/srt/layers/quantization/exl3/ext.py`, `test/registered/unit/layers/quantization/test_exl3_ext.py`, `bench/CMakeLists.txt` | `SGLANG_EXL3_CPU_MAX_M` → define and flavor; bench `EXL3_MAX_M` | 4b |
| `python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp`, `forward_plan.hpp`, `shapes.hpp`, `README.txt` | register kernels in M, budget table, dispatch, scratch, `accepts` | 5 |
| `python/sglang/kernels/jit/csrc/exl3/optimized/math.hpp` | default `MAX_M` 8 (only if 5b's gate passes) | 5b |
| `DSV41_REFERENCE.md` | §33.10 | 7 |

---

### Task 1: The draft CPU thread counts jobs whose routes share a slot

The measurement to take before (or alongside) the kernel work: how often a real draft call has a chunk of more than one token.

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
    """The EXL3 kernel groups a call's live routes by slot (up to CHUNK_M tokens per chunk), so a job whose routes name
    one slot twice runs a chunk of several tokens: collided_jobs counts those jobs, shared_routes the routes past each slot's first, and
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
  /// Jobs whose live routes name one slot more than once: the EXL3 kernel runs each with a chunk of several tokens.
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

### Task 2: The kernel counts the forwards each plan took and reports CHUNK_M; the parity test

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
  - In `kernel.h`:
    - `struct SglangExl3CpuPlanCalls { int64_t dsv41; int64_t generic; };`;
    - `sglang::exl3_cpu::exl3_cpu_plan_calls()` (hidden), returning it;
    - `int32_t sglang::exl3_cpu::exl3_cpu_chunk_m()` (hidden), returning the build's `CHUNK_M`.
  - Torch ops:
    - `sglang_exl3_cpu::plan_calls() -> int[]`, returning `[dsv41, generic]`;
    - `sglang_exl3_cpu::chunk_m() -> int`.
  - In the test file:
    - `_expected_plan(routes) -> str` (`"dsv41"` or `"generic"`), which Task 5 changes;
    - `DEFAULT_CHUNK_M = 2`, which Task 5b changes only if its gate passes;
    - `_chunk_m()`, read by every chunk-size case.

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

Exl3Quant::dispatch groups a call's routes by expert into chunks of up to CHUNK_M tokens (sglang_exl3_cpu::chunk_m:
MAX_M / 2, so 2 by default and 4 in a SGLANG_EXL3_CPU_MAX_M=8 build). Each case runs one call on a DeepSeek
V4.1-shaped layer:
  - unswizzled (native), which Dsv41Shape::accepts may take on AVX-512BW;
  - swizzled, which it always refuses, so it runs the generic plan;
  - token by token, through the native layer.
The three must agree bit for bit. The plan counter (sglang_exl3_cpu::plan_calls) shows which plan each call took. The
chunk-size cases are generated from the build's CHUNK_M, so a larger MAX_M widens them with no edit here.

Needs SGLANG_EXL3_SRC, SGLANG_DSV41_CPU_EXPERTS=1 and the bw tier (EXL3_MOE_CPU_MAX_ISA=bw on a host above it). Run on
divix01 under taskset -c 0-63.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

H, I, CAP, LIMIT, THREADS = 5120, 2304, 12, 10.0, 4
DEFAULT_CHUNK_M = 2  # CHUNK_M at the source's default MAX_M (math.hpp's EXL3_MOE_CPU_MAX_M)


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


# (name, routes [rows][k]; -1 is no route). The comment gives the chunks dispatch makes at CHUNK_M = 2, in expert order.
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


def _chunk_m():
    return int(torch.ops.sglang_exl3_cpu.chunk_m())


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


def _check(layers, name, routes):
    """One call on both layers and token by token: the same bits everywhere, and the plan _expected_plan names."""
    x, slots, weights = _inputs(name, routes)
    got, plan = _forward(layers["native"], x, slots, weights)
    want, generic = _forward(layers["swizzled"], x, slots, weights)
    assert (plan, generic) == (_expected_plan(routes), "generic"), name
    assert torch.isfinite(got).all(), name
    assert torch.equal(got, want), name
    singles = torch.cat(
        [_forward(layers["native"], x[t : t + 1], slots[t : t + 1], weights[t : t + 1])[0] for t in range(len(routes))]
    )
    assert torch.equal(got, singles), name


def test_the_build_reports_its_chunk_size(layers):
    """CHUNK_M is half of MAX_M: SGLANG_EXL3_CPU_MAX_M's half when the build sets it, else the source default's."""
    max_m = int(os.environ.get("SGLANG_EXL3_CPU_MAX_M") or 0)
    assert _chunk_m() == (max_m // 2 if max_m else DEFAULT_CHUNK_M)


@pytest.mark.parametrize("name,routes", CASES, ids=[c[0] for c in CASES])
def test_dsv41_plan_matches_the_generic_plan_and_one_token_runs(layers, name, routes):
    _check(layers, name, routes)


def test_every_chunk_size_matches_the_generic_plan(layers):
    """For m = 1..CHUNK_M, read from the build: m tokens on one expert and m tokens sharing three experts (chunks of
    exactly m), then CHUNK_M + 1 tokens on one expert (a full chunk, then a chunk of one)."""
    chunk_m = _chunk_m()
    for m in range(1, chunk_m + 1):
        _check(layers, f"m{m}-one-expert", [[3]] * m)
        _check(layers, f"m{m}-three-experts", [[4, 0, 5]] * m)
    _check(layers, f"m{chunk_m + 1}-overflow", [[2]] * (chunk_m + 1))


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
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(exl3-cpu): the DSV4.1 plan against the generic plan and one-token runs, at every chunk size (failing)"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `17 failed`, and `EXIT=1`. The 17 are 1 chunk-size test, 12 cases, 1 every-chunk-size test and 3 team
sizes. Each fails on a missing op: an `AttributeError` or `RuntimeError` naming `sglang_exl3_cpu.plan_calls` or
`sglang_exl3_cpu.chunk_m`.

- [ ] **Step 3: Add the counters and `chunk_m`.**

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
// The most tokens a chunk holds in this build (math.hpp's CHUNK_M = MAX_M / ACT_ROWS).
__attribute__((visibility("hidden"))) int32_t exl3_cpu_chunk_m();
```
3. In `forward_plan.hpp`, just above `run_plan` (:738):
```cpp
// Forwards by plan since load: [0] the DSV4.1 plan, [1] the generic plan. Relaxed: counts only, read through
// exl3_cpu_plan_calls (kernel.cpp, the one file that includes this header).
std::atomic<int64_t> g_plan_calls[2];
```
4. In `run_plan`:
   - Put `g_plan_calls[0].fetch_add(1, std::memory_order_relaxed);` as the first statement inside the
     `if (isa == Isa::Bw && Dsv41Shape::accepts(...))` block.
   - Put `g_plan_calls[1].fetch_add(1, std::memory_order_relaxed);` right after that block, before
     `const Experts<GenericShape> E{&l, p};`.
5. In `kernel.cpp`, inside `namespace sglang::exl3_cpu`, after `exl3_cpu_kernel()`:
```cpp
SglangExl3CpuPlanCalls exl3_cpu_plan_calls()
{
    return {g_plan_calls[0].load(std::memory_order_relaxed), g_plan_calls[1].load(std::memory_order_relaxed)};
}

int32_t exl3_cpu_chunk_m() { return CHUNK_M; }
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

int64_t chunk_m() { return ::sglang::exl3_cpu::exl3_cpu_chunk_m(); }
```
   - In `TORCH_LIBRARY`, add `m.def("plan_calls() -> int[]", &plan_calls);` and
     `m.def("chunk_m() -> int", &chunk_m);`.
   - Extend the header comment's first sentence with ", sglang_exl3_cpu::plan_calls, the forwards each plan took
     ([dsv41, generic]), and sglang_exl3_cpu::chunk_m, the build's CHUNK_M".
7. In `run_exl3_cpu_forward_checks.sh`, add `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` to the
   final `step pytest` line's file list.

- [ ] **Step 4: Commit, push, SYNC, and run.** The extension rebuilds cold.

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h \
  python/sglang/kernels/jit/csrc/exl3/optimized/kernel.cpp python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp \
  python/sglang/kernels/jit/csrc/exl3/optimized/torch_ops.cpp test/manual/dsv41/run_exl3_cpu_forward_checks.sh
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(exl3-cpu): the kernel counts the forwards each plan took and reports CHUNK_M (sglang_exl3_cpu::plan_calls, chunk_m)"
```
Then run the same command as Step 2.

Expected: `17 passed`, `EXIT=0`.
- Today a call whose routes share an expert runs the generic plan on both layers, so "generic" is expected there.
- Generic output already equals the one-token DSV4.1 runs bitwise, as
  `test_exl3_cpu_act_quant.py::test_batch_matches_single_token_runs` pins.
- If `one-token` reports `generic`, the tier is not bw: check `EXL3_MOE_CPU_MAX_ISA`.

---

### Task 3: The A/B harness covers multi-token chunks

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
# (tokens, experts per token) -> routes. Tokens sharing an expert make chunks of up to CHUNK_M tokens: (2, 3) shares
# experts 0 and 4; (2, 1) is one expert's chunk of two; (3, 1) one expert's three tokens; (1, 2) routes one token twice
# to expert 1; (6, 3) and (16, 3) mix chunk sizes, as a DSpark draft call does ((16, 3) fills chunks of 4 at MAX_M 8).
# The DSV4.1 plan refuses chunks of more than one token: at the DSV4.1 shape those cases run the generic plan.
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
  - `SglangExl3CpuPlanCalls`, `sglang::exl3_cpu::exl3_cpu_plan_calls()` and `sglang::exl3_cpu::exl3_cpu_chunk_m()`
    (Task 2, `kernel.h`).
- Produces:
  - Flags `--routed=M:k:pattern[,...]`, with pattern `shared<G>` (tokens in groups of G share all k slots, so every
    chunk holds min(G, CHUNK_M) tokens; `shared1` shares nothing) or `random`; `--routed-slots=N` (default 48); and
    `--routed-layers=N` (default 2).
  - Benchmark context `chunk_m`: the build's CHUNK_M.
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
$W/bench-head/exl3_cpu_optimized --validate-only --routed=2:3:shared2; echo "EXIT=$?"
EOF
```
Expected: Google Benchmark reports `unrecognized command-line flag: --routed=2:3:shared2`, and `EXIT=1`.

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
     `"--routed=M:k:shared<G>|random,... --routed-slots=48 --routed-layers=2 (optimized build)\n"`.
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
  // is its layer's fixture input rotated by 641 * t elements. Patterns: shared<G> (tokens in groups of G share all k
  // slots, so every chunk holds min(G, CHUNK_M) tokens; shared1 shares nothing), random (each token k distinct slots,
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
      int group = 0;  // shared<G>: G; random: 0
      if (pattern.starts_with("shared"))
        group = number(pattern.substr(6));
      else if (pattern != "random")
        throw std::runtime_error("Unknown routed pattern: " + pattern);
      if (pattern.starts_with("shared") && group < 1) throw std::runtime_error("shared<G> needs G >= 1: " + spec);
      const int needed = group ? (rows + group - 1) / group * k : k;
      if (rows < 1 || k < 1 || needed > capacity)
        throw std::runtime_error(spec + " needs " + std::to_string(needed) + " slots of " + std::to_string(capacity));
      std::mt19937 rng(20261006u + 1000u * rows + k);
      std::vector<int32_t> pool(capacity);
      for (int t = 0; t < rows; ++t) {
        std::iota(pool.begin(), pool.end(), 0);
        for (int j = 0; j < k; ++j) {
          int32_t slot;
          if (group)
            slot = t / group * k + j;
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
     Then, in the registration block, after `benchmark::AddCustomContext("compiler", __VERSION__);`, and again after the
     `workloads` loop:
  ```cpp
  #ifndef EXL3_BENCH_BASELINE
        benchmark::AddCustomContext("chunk_m", std::to_string(::sglang::exl3_cpu::exl3_cpu_chunk_m()));
  #endif
  ```
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
  --routed=M:k:pattern[,...]  pattern shared<G> (tokens in groups of G share all k slots; shared1 shares none) or
                              random (each token k distinct slots, seeded)
  --routed-slots=48           slab slots per routed layer; slot s holds fixture expert s % 5
  --routed-layers=2           fixture layers the routed workloads rotate over (shared by all of them)
Token t's input is the layer's fixture input rotated by 641 * t elements. There is no frozen reference: validation
checks the outputs are finite and repeat bit for bit. Every optimized benchmark reports dsv41_calls and generic_calls,
the forwards each plan ran (exl3_cpu_plan_calls), and the context records the build's chunk_m (exl3_cpu_chunk_m).

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
  --routed=2:3:shared2,16:3:shared2,16:3:shared4,16:3:shared1,6:3:random,16:3:random,64:6:random; echo "EXIT=$?"
```
Expected on stderr:
- `Verified 24 bit-exact layer outputs; ...`
- `Verified 7 routed workloads repeat bit for bit`
- `EXIT=0`

- [ ] **Step 6: Record BASE.** Run
  `git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 rev-parse --short HEAD`. That sha is **BASE**.
  Create the base worktree on divix01:
```bash
ssh divix01 'git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cpu-plan-m2-base <BASE> \
  && git -C /data/models/slang/nvfp4-work/wt-cpu-plan-m2-base log -1 --oneline'
```

---

### Task 4b: MAX_M is one build-time constant; the generic plan is the same at any MAX_M

This task makes every `MAX_M` use derive from one overridable constant. The DSV4.1 plan still takes one-token chunks
only (Task 5 widens it), so at `MAX_M = 8` every shared-expert call runs the generic plan with chunks of up to four.
That is exactly the generic-drift test.

**Files:**
- Modify: `python/sglang/srt/environ.py:1283-1284` (beside `SGLANG_EXL3_CPU_ACT_BLOCK`)
- Modify: `python/sglang/srt/layers/quantization/exl3/ext.py:66-101` (`cpu_act_defines`, `build_flavor`)
- Test: `test/registered/unit/layers/quantization/test_exl3_ext.py` (`_cpu_experts_defines` :56-67, new tests)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/CMakeLists.txt` (`EXL3_MAX_M`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/math.hpp:46`
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp:38-189` (`run_tiles_raw`), `:692-693` (`idx4`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp:771-779` (the VBMI band rule)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/shapes.hpp:1-4` (comment)

**Interfaces:**
- Consumes: BASE (Task 4); `exl3_cpu_forward_ab.py`'s routes (Task 3); `chunk_m` (Task 2).
- Produces:
  - `EXL3_MOE_CPU_MAX_M` (C++ macro, default 4) and `constexpr int MAX_M`.
  - `SGLANG_EXL3_CPU_MAX_M`: an `EnvInt`, 0 for the default, else even in [2, 8], with the optimized kernel only. It
    adds the define `-DEXL3_MOE_CPU_MAX_M=N` and the flavor `_resid_b128_mN_cpu_v1`.
  - The bench's CMake cache variable `EXL3_MAX_M` (empty means the source default).
  - `kTiles<I>[bits - 1][rows - 1]` and `kRowIndex` (`forward_plan.hpp`).

- [ ] **Step 1: Read the `env-var-conventions` skill** (`.claude/skills/env-var-conventions/SKILL.md`), as
  `.claude/rules/modify-component-must-read.md` requires before adding an `SGLANG_*` variable.

- [ ] **Step 2: Write the failing tests.** In `test/registered/unit/layers/quantization/test_exl3_ext.py`:

  1. In `_cpu_experts_defines`, change the name tuple to
     `("SGLANG_EXL3_CPU_ACT_RESIDUAL", "SGLANG_EXL3_CPU_ACT_BLOCK", "SGLANG_EXL3_CPU_MAX_M")`. A `MAX_M` in the
     caller's environment then cannot leak into the existing cases.
  2. After `test_cpu_experts_refuse_another_accuracy_flavor`, add:
```python
def test_cpu_max_m_maps_to_a_define_and_its_own_flavor():
    defines = _cpu_experts_defines(SGLANG_EXL3_CPU_MAX_M=8)
    assert defines == ["-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128", "-DEXL3_MOE_CPU_MAX_M=8"]
    assert exl3_ext.optimized_cpu(defines) and exl3_ext.build_flavor(defines) == "_resid_b128_m8_cpu_v1"


@pytest.mark.parametrize("max_m", [1, 3, 10, -2])
def test_cpu_max_m_must_be_even_in_2_to_8(max_m):
    with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_MAX_M"):
        _cpu_experts_defines(SGLANG_EXL3_CPU_MAX_M=max_m)


def test_cpu_max_m_needs_the_optimized_kernel():
    """The vendored kernel (moe_mul1.cpp) keeps its own MAX_M: setting the variable without CPU experts is refused."""
    with exl3_ext.envs.SGLANG_DSV41_CPU_EXPERTS.override(False), exl3_ext.envs.SGLANG_EXL3_CPU_MAX_M.override(8):
        with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_MAX_M"):
            exl3_ext.cpu_act_defines()
```

- [ ] **Step 3: Commit, push, SYNC, and run them to see them fail.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add test/registered/unit/layers/quantization/test_exl3_ext.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(exl3-ext): SGLANG_EXL3_CPU_MAX_M sizes the optimized CPU kernel in its own build (failing)"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/layers/quantization/test_exl3_ext.py -q -p no:randomly \
  -k max_m 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `6 failed` (an `AttributeError` naming `SGLANG_EXL3_CPU_MAX_M` on `envs`), and `EXIT=1`.

- [ ] **Step 4: Add the variable and the bench option.**

1. In `environ.py`, after `SGLANG_EXL3_CPU_ACT_BLOCK = EnvInt(0)`:
```python
    # Build-time: the optimized EXL3 CPU kernel's quantized activation rows per chunk (EXL3_MOE_CPU_MAX_M, even in
    # [2, 8]; a chunk holds half as many tokens). 0 keeps the kernel's default. Any other value builds into its own
    # extension (flavor _m<N>). SGLANG_DSV41_CPU_EXPERTS only: the vendored kernel keeps its own.
    SGLANG_EXL3_CPU_MAX_M = EnvInt(0)
```
2. In `ext.py`, add above `cpu_act_defines`:
```python
def _max_m_defines() -> list[str]:
    """-DEXL3_MOE_CPU_MAX_M from SGLANG_EXL3_CPU_MAX_M; none at 0, the kernel's default."""
    max_m = envs.SGLANG_EXL3_CPU_MAX_M.get()
    if not max_m:
        return []
    if max_m < 2 or max_m > 8 or max_m % 2:
        raise ValueError(f"SGLANG_EXL3_CPU_MAX_M must be 0 or even in [2, 8], got {max_m}")
    return [f"-DEXL3_MOE_CPU_MAX_M={max_m}"]
```
3. In `cpu_act_defines`:
   - Change `return list(OPTIMIZED_CPU_DEFINES)` to `return list(OPTIMIZED_CPU_DEFINES) + _max_m_defines()`.
   - Right after the `if envs.SGLANG_DSV41_CPU_EXPERTS.get(): ...` block, add:
```python
    if envs.SGLANG_EXL3_CPU_MAX_M.get():
        raise ValueError(
            "SGLANG_EXL3_CPU_MAX_M sizes the optimized EXL3 CPU kernel, which SGLANG_DSV41_CPU_EXPERTS=1 builds; "
            "the vendored kernel keeps its own"
        )
```
4. In `build_flavor`'s loop, after the `-DEXL3_MOE_CPU_ACT_BLOCK=` branch:
```python
        elif d.startswith("-DEXL3_MOE_CPU_MAX_M="):
            flavor += "_m" + d.split("=", 1)[1]
```
   Also update the docstring example to `"_resid_b128"` or `"_resid_b128_m8_cpu_v1"`.
5. In `bench/CMakeLists.txt`:
   - After the `EXPERT_STREAM_NODES` cache line:
```cmake
set(EXL3_MAX_M "" CACHE STRING "EXL3_MOE_CPU_MAX_M for the optimized kernel (even, 2..8); empty: the source default")
```
   - After the `target_compile_definitions(exl3_cpu_${BACKEND} ...)` call:
```cmake
  if(EXL3_MAX_M AND NOT BACKEND STREQUAL "baseline")
    target_compile_definitions(exl3_cpu_${BACKEND} PRIVATE EXL3_MOE_CPU_MAX_M=${EXL3_MAX_M})
  endif()
```
   - After the full-stack target's `target_compile_definitions(${TARGET} ...)`:
```cmake
    if(EXL3_MAX_M)
      target_compile_definitions(${TARGET} PRIVATE EXL3_MOE_CPU_MAX_M=${EXL3_MAX_M})
    endif()
```

- [ ] **Step 5: Derive every use from one MAX_M** (the "fix" rows of the audit that belong to this task).

1. `math.hpp:46`. Replace `constexpr int MAX_M = 4;` with:
```cpp
#ifndef EXL3_MOE_CPU_MAX_M
#define EXL3_MOE_CPU_MAX_M 4
#endif
// Quantized activation rows per chunk: CHUNK_M tokens times ACT_ROWS (below). Every per-chunk array, stride and tile
// table derives from it. A build raises it with -DEXL3_MOE_CPU_MAX_M (SGLANG_EXL3_CPU_MAX_M; the bench's EXL3_MAX_M).
constexpr int MAX_M = EXL3_MOE_CPU_MAX_M;
static_assert(MAX_M >= 2 && MAX_M <= 8 && MAX_M % 2 == 0, "EXL3_MOE_CPU_MAX_M must be even, in [2, 8]");
```
2. `shapes.hpp:4`. Change "(up to MAX_M token rows)" to "(up to CHUNK_M tokens)".
3. `forward_plan.hpp`, `run_tiles_raw` (:38-189).
   - Add `#include <array>` to the includes, in sorted position.
   - Replace the function with the tables below and this `run_tiles_raw`:
```cpp
using TilesFn = void (*)(const Exl3Projection&, const PreparedIn&, float*, int, int);

// One AVX-512 tier's GEMV tiles for `bits` and `rows` quantized rows. VBMI at 8 bits runs the VNNI tiles: byte pairing
// is impossible there (shift % 8 == 0) and the byte windows straddle the register pairs, measured slower than the
// dword scheme.
template <Isa I, int bits, int rows>
void tiles_for(const Exl3Projection& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    if constexpr (I == Isa::Vbmi && bits != 8) vbmi_tiles<bits, rows>(mat, in, tout, tn0, tn1);
    else if constexpr (I == Isa::Vbmi || I == Isa::Vnni) vnni_tiles<bits, rows>(mat, in, tout, tn0, tn1);
    else bw_tiles<bits, rows>(mat, in, tout, tn0, tn1);
}

template <Isa I, int bits, int... R>
constexpr std::array<TilesFn, sizeof...(R)> tiles_rows(std::integer_sequence<int, R...>)
{
    return {&tiles_for<I, bits, R + 1>...};
}

template <Isa I, int... B>
constexpr std::array<std::array<TilesFn, MAX_M>, sizeof...(B)> tiles_table(std::integer_sequence<int, B...>)
{
    return {tiles_rows<I, B + 1>(std::make_integer_sequence<int, MAX_M>{})...};
}

// [bits - 1][rows - 1] for bits 1..8 and rows 1..MAX_M, generated from MAX_M: a larger MAX_M instantiates its rows
// here with no hand-written case.
template <Isa I>
constexpr auto kTiles = tiles_table<I>(std::make_integer_sequence<int, 8>{});

template <Isa I>
void run_tiles_raw(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Vbmi || I == Isa::Vnni || I == Isa::Bw)
    {
        kTiles<I>[mat.bits - 1][m - 1](mat, in, tout, tn0, tn1);
    }
    else if constexpr (I == Isa::Avx2)
    {
        switch (mat.bits)
        {
            case 1: avx2_tiles<1>(mat, in, tout, m, tn0, tn1); return;
            case 2: avx2_tiles<2>(mat, in, tout, m, tn0, tn1); return;
            case 3: avx2_tiles<3>(mat, in, tout, m, tn0, tn1); return;
            case 4: avx2_tiles<4>(mat, in, tout, m, tn0, tn1); return;
            case 5: avx2_tiles<5>(mat, in, tout, m, tn0, tn1); return;
            case 6: avx2_tiles<6>(mat, in, tout, m, tn0, tn1); return;
            case 7: avx2_tiles<7>(mat, in, tout, m, tn0, tn1); return;
            default: avx2_tiles<8>(mat, in, tout, m, tn0, tn1); return;
        }
    }
    else
    {
        switch (mat.bits)
        {
            case 1: scalar_tiles<1>(mat, in, tout, m, tn0, tn1); return;
            case 2: scalar_tiles<2>(mat, in, tout, m, tn0, tn1); return;
            case 3: scalar_tiles<3>(mat, in, tout, m, tn0, tn1); return;
            case 4: scalar_tiles<4>(mat, in, tout, m, tn0, tn1); return;
            case 5: scalar_tiles<5>(mat, in, tout, m, tn0, tn1); return;
            case 6: scalar_tiles<6>(mat, in, tout, m, tn0, tn1); return;
            case 7: scalar_tiles<7>(mat, in, tout, m, tn0, tn1); return;
            default: scalar_tiles<8>(mat, in, tout, m, tn0, tn1); return;
        }
    }
}
```
   The AVX2 and scalar branches are today's (:161-188), unchanged: their tiles take `m` at run time.
4. `forward_plan.hpp:692-693`.
   - Delete `static const int idx4[MAX_M] = {0, 1, 2, 3};`.
   - Pass `kRowIndex.data()` instead of `idx4`.
   - Define, above `enum class Phase`:
```cpp
// A chunk's down input row r is its gate output row r: the identity over MAX_M rows (prepare_rows' token_idx).
constexpr std::array<int, MAX_M> kRowIndex = [] {
    std::array<int, MAX_M> rows{};
    for (int i = 0; i < MAX_M; ++i) rows[i] = i;
    return rows;
}();
```
5. `math_avx512.hpp:777-779` (`vbmi_tiles`). Replace the `max_band` expression with the one below. Up to 4 rows the
   values are today's:
```cpp
    // rows > 4 (a raised MAX_M): two bands, at most 16 accumulators.
    const int max_band = mat.swz
        ? (rows <= 2 ? 8 : rows > 4 ? 2 : (rows == 4 && bits == 2 ? 2 : 4))
        : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
```
   The other tiers already keep `rows * band <= 16` for rows 5..8:
   - unswizzled, `12 / rows` gives 2 or 1;
   - BW and VNNI swizzled, `rows <= 3 ? 4 : 2` gives 2.

- [ ] **Step 6: Commit, push, SYNC; run the ext tests, and warm both extension builds.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/srt/environ.py \
  python/sglang/srt/layers/quantization/exl3/ext.py python/sglang/kernels/jit/csrc/moe/expert_stream/bench/CMakeLists.txt \
  python/sglang/kernels/jit/csrc/exl3/optimized
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(exl3-cpu): MAX_M is one build-time constant (EXL3_MOE_CPU_MAX_M, SGLANG_EXL3_CPU_MAX_M); the generic tiles are generated from it"
```
Then run:
```bash
ssh divix01 'bash -s' <<'EOF'
cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
export PYTHONPATH=$PWD/python PY=/data/models/slang/.venv/bin/python
taskset -c 0-63 $PY -m pytest test/registered/unit/layers/quantization/test_exl3_ext.py -q -p no:randomly 2>&1 | tail -2
echo "EXIT=${PIPESTATUS[0]}"
WARM='import torch; from sglang.srt.layers.quantization.exl3.ext import exl3_ext; e = exl3_ext(); print(e.__file__, torch.ops.sglang_exl3_cpu.chunk_m())'
taskset -c 0-63 $PY -c "$WARM" 2>&1 | tail -1
SGLANG_EXL3_CPU_MAX_M=8 taskset -c 0-63 $PY -c "$WARM" 2>&1 | tail -1
EOF
```
Expected:
- The ext tests all pass with `EXIT=0`.
- The two warm lines read `.../resid_b128_cpu_v1/sglang_exl3_ext_resid_b128_cpu_v1.so 2` and
  `.../resid_b128_m8_cpu_v1/sglang_exl3_ext_resid_b128_m8_cpu_v1.so 4`. The second is a cold build, several minutes.

- [ ] **Step 7: The bit-exact gate at both MAX_M values, against BASE.**

1. Take BASE's dumps, once, if `cpu-plan-m2/checks-base.out` does not end `ALL GREEN (baseline)` yet:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && bash wt-cpu-plan-m2-base/test/manual/dsv41/run_exl3_cpu_forward_checks.sh \
  baseline wt-cpu-plan-m2-base cpu-plan-m2/checks-base > cpu-plan-m2/checks-base.out 2>&1; tail -1 cpu-plan-m2/checks-base.out'
```
   Expected: `ALL GREEN (baseline)`.
2. Run the default build's full gate:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && bash wt-cpu-plan-m2/test/manual/dsv41/run_exl3_cpu_forward_checks.sh \
  check wt-cpu-plan-m2 cpu-plan-m2/checks-4b cpu-plan-m2/checks-base 2>&1 | tail -16; echo "EXIT=${PIPESTATUS[0]}"'
```
   Expected: `ALL GREEN (check)`, with `compare-{bw,avx2,scalar}` each `72/72 bit-exact`. This shows the generated
   tables change no bit at `MAX_M = 4`.
3. Compare the `MAX_M = 8` build's dumps on every tier divix01 runs (generic drift):
```bash
ssh divix01 'bash -s' <<'EOF'
cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
export PYTHONPATH=$PWD/python SGLANG_EXL3_CPU_MAX_M=8 PY=/data/models/slang/.venv/bin/python
for isa in bw avx2 scalar; do
  taskset -c 0-63 $PY test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa $isa --registration slabs \
    --out $W/m8-4b-$isa.pt 2>&1 | tail -1
  $PY test/manual/dsv41/exl3_cpu_forward_ab.py compare $W/checks-base/ab-$isa-slabs.pt $W/m8-4b-$isa.pt 2>&1 | tail -1
  echo "EXIT=${PIPESTATUS[0]}"
done
EOF
```
   Expected, for each of bw, avx2 and scalar:
   - `72 outputs (<isa>, slabs) -> .../m8-4b-<isa>.pt`;
   - `72/72 bit-exact: <isa>/slabs vs <isa>/slabs`;
   - `EXIT=0`.

   A `MISMATCH` names the case. Find the site in the audit table that still assumes 4 rows. Do not touch a reference.

- [ ] **Step 8: The swizzled generic plan at MAX_M = 8.**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && SGLANG_EXL3_CPU_MAX_M=8 PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_cpu_act_quant.py -q -p no:randomly 2>&1 | tail -2; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `3 passed`, `EXIT=0`. Its five-token batch now runs chunks of up to four through the generic plan, and must
equal the one-token runs and the swizzled layout bit for bit.

---

### Task 5: The DSV4.1 plan takes chunks of 1..CHUNK_M tokens

**Files:**
- Modify: `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` (`_expected_plan`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp`:
  - `:912-1001` (`register_rows`, `register_band`);
  - `:1106-1133` (`register_tiles`);
  - new `kRegisterBudget` and `register_band_for`.
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp`:
  - `run_tiles` and its new `register_tiles_for`;
  - `prepare_scratch` (:529-596);
  - the PlanTraits comments (:491-509).
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/shapes.hpp:39-50` (`accepts`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/README.txt` ("Selection", "Code layout")
- Modify: `test/manual/dsv41/exl3_cpu_forward_ab.py` (the `ROUTES` comment)

**Interfaces:**
- Consumes:
  - `chunk_m` and `plan_calls` (Task 2), the A/B cases (Task 3), and BASE with its dumps (Tasks 4 and 4b);
  - `MAX_M` and `SGLANG_EXL3_CPU_MAX_M` (Task 4b).
- Produces:
  - `struct RegisterBudget { int pairs; bool traversal; }` and `constexpr RegisterBudget kRegisterBudget[]`, one entry
    per M. Also `constexpr int kRegisterDecodeZmm = 12;`.
  - `register_rows<J, Pairs, Compact, M = 1>` and `register_band<Pairs, FixedK, FixedN, Compact, M = 1>`.
  - `template <int M> void register_tiles(const Exl3Projection&, const PreparedIn&, float* tout, int t0, int t1, bool grouped)`,
    where `M = 1` is today's `register_tiles`.
  - `register_tiles_for(std::integer_sequence<int, Ms...>, ...)` (`forward_plan.hpp`).
  - `Dsv41Shape::accepts`, which takes `1 <= m <= CHUNK_M`.

- [ ] **Step 1: Make the test expect the DSV4.1 plan for every native call.** Replace `_expected_plan` with:

```python
def _expected_plan(routes):
    """The plan a call on the native layer takes: the DSV4.1 plan, whose chunks hold 1..CHUNK_M tokens."""
    return "dsv41"
```

- [ ] **Step 2: Commit, push, SYNC, and run to see it fail.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "test(exl3-cpu): a call whose routes share an expert takes the DSV4.1 plan (failing)"
```
Then run Task 2 Step 2's command.

Expected: `EXIT=1`, with failures of the form `AssertionError: <case>`, where `('generic', 'generic') != ('dsv41', 'generic')`.
- These fail: every hand-written case except `one-token`, `draft-16x3`, `prefill-64x6`, any random draft case whose
  routes collide, `test_every_chunk_size_matches_the_generic_plan` (at `m2-one-expert`), and all three team-size
  cases.
- These pass: `one-token` and `test_the_build_reports_its_chunk_size`.

- [ ] **Step 3: Make the register kernels generic in M (`math_avx512.hpp`).**

  1. Add `#include <iterator>` and `#include <utility>` to the includes, in sorted position.
  2. `register_rows`: replace its template line, parameter list and inner row loop. Keep the body between them.
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
  3. `register_band`: replace the whole function with the version below. At `M = 1` it is the same code: `R = 2`, the
     same `kb * 256` stride, and the same stores in the same order.
```cpp
// M token rows (the tokens of one chunk) share each decoded tile. Per output: the int32 sum of each (k-block,
// quantized row), one FMA scale each, fp32 sums in increasing k-block order, then token + residual once: the generic
// path's rounding (bw3_blocked_band).
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

- [ ] **Step 4: The register budget table, and `register_tiles<M>` (`math_avx512.hpp`).**

  1. Insert the table just above `register_rows` (:912):
```cpp
// The compact register kernels' register budget, one entry per tokens-per-chunk M = 1..CHUNK_M (register_tiles<M>):
// pairs, the tile pairs one register_band call holds; traversal, whether whole 128-output groups take the grouped
// traversal (its IntegerAccum holds one token's rows). Raising EXL3_MOE_CPU_MAX_M past twice the entries fails to
// compile below until an entry is added.
struct RegisterBudget
{
    int pairs;
    bool traversal;
};
constexpr RegisterBudget kRegisterBudget[] = {
    {4, true},   // M = 1: today's kernels (register_band<1..4>, traversal_kblock's four bands)
    {2, false},  // M = 2
    {1, false},  // M = 3
    {1, false},  // M = 4
};
// zmm registers register_rows keeps live besides the integer accumulators: prev, a, b, c, the state, its byte sum, two
// products, the multiplier halves, the ones vector and the broadcast activation pair.
constexpr int kRegisterDecodeZmm = 12;

constexpr bool register_budget_fits()
{
    for (int m = 1; m <= int(std::size(kRegisterBudget)); ++m)
        if (kRegisterBudget[m - 1].pairs < 1 || 4 * m * kRegisterBudget[m - 1].pairs + kRegisterDecodeZmm > 32)
            return false;
    return true;
}
static_assert(register_budget_fits(), "kRegisterBudget (math_avx512.hpp): an entry's integer accumulators "
                                      "(4 * M * pairs) plus kRegisterDecodeZmm exceed the 32 zmm registers");
static_assert(ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128 || CHUNK_M <= int(std::size(kRegisterBudget)),
              "kRegisterBudget (math_avx512.hpp) has no entry for this CHUNK_M: add one before raising "
              "EXL3_MOE_CPU_MAX_M");
```
  2. Just after `register_band`, add:
```cpp
// register_band<P, 0, 0, true, M> for a runtime P in 1..sizeof...(P): one direct call per P.
template<int M,int... P>
M1_TARGET_BW M1_ALWAYS_INLINE void register_band_for(std::integer_sequence<int,P...>,const Exl3Projection& mat,
                                                     const PreparedIn& in,float* tout,int n0,int pairs) {
    (void)((pairs==P+1 && (register_band<P+1,0,0,true,M>(mat,in,tout,n0),true)) || ...);
}
```
  3. Make `register_tiles` (:1106) a template on M. Its `M == 1` branch is today's body, unchanged:
```cpp
// The tokens of one chunk, M = 1..CHUNK_M. M = 1 is the one-token kernel either plan calls (compact or not, swizzled
// or not; the grouped traversal: kRegisterBudget[0]). M > 1 is the DSV4.1 plan's compact, unswizzled input, in runs of
// up to kRegisterBudget[M - 1].pairs tile pairs.
template<int M>
M1_TARGET_BW void register_tiles(const Exl3Projection& mat,const PreparedIn& in,float* tout,int t0,int t1,
                                 [[maybe_unused]] bool grouped) {
    [[maybe_unused]] constexpr RegisterBudget budget=kRegisterBudget[M-1];
    if constexpr (M==1) {
        if(in.compact) {
            if(mat.swz) {
                // The swizzled layout stores a 128-output group's eight tiles together.
                TORCH_CHECK(t0%8==0 && t1%8==0,"compact swizzled input requires whole output blocks");
                for(int t=t0;t<t1;t+=8)register_band<4,0,0,true>(mat,in,tout,t);
                return;
            }
            // Unswizzled: whole groups through the traversal, a partial group at either end as tile pairs.
            TORCH_CHECK(t0%2==0 && t1%2==0,"compact input requires whole tile pairs");
            const int a0=std::min(t1,(t0+7)/8*8), a1=std::max(a0,t1/8*8);
            compact_pairs(mat,in,tout,t0,a0);
            if(a1>a0)traversal_tiles(mat,in,tout,a0,a1,grouped);
            compact_pairs(mat,in,tout,a1,t1);
            return;
        }
        for(int n0=t0;n0<t1;) {
            if((n0%2)||t1-n0<2){bw3_blocked_band<1>(mat,in,tout,n0++);continue;}
            const int pairs=std::min({4,(t1-n0)/2,(8-n0%8)/2});
            switch(pairs) {
                case 1:register_band<1>(mat,in,tout,n0);break;
                case 2:register_band<2>(mat,in,tout,n0);break;
                case 3:register_band<3>(mat,in,tout,n0);break;
                case 4:register_band<4>(mat,in,tout,n0);break;
            }
            n0+=pairs*2;
        }
    } else {
        static_assert(!budget.traversal,"the grouped traversal holds one token's rows (IntegerAccum)");
        TORCH_CHECK(in.compact && !mat.swz && t0%2==0 && t1%2==0,
                    "a chunk of several tokens needs compact, unswizzled tile pairs");
        for(int n0=t0;n0<t1;) {
            const int pairs=std::min(budget.pairs,(t1-n0)/2);
            register_band_for<M>(std::make_integer_sequence<int,budget.pairs>{},mat,in,tout,n0,pairs);
            n0+=pairs*2;
        }
    }
}
```
  4. In the file header comment, change "register_tiles and its traversal" to "register_tiles<M> (M tokens per chunk,
     kRegisterBudget) and the one-token traversal".

- [ ] **Step 5: Dispatch 1..CHUNK_M, and lay out compact scratch by rows (`forward_plan.hpp`).**

  1. Above `run_tiles`, add:
```cpp
// register_tiles<M> for M = 1..CHUNK_M, generated from CHUNK_M: one direct call per M.
template <int... Ms>
M1_ALWAYS_INLINE void register_tiles_for(std::integer_sequence<int, Ms...>, const Exl3Projection& mat,
                                         const PreparedIn& in, float* tout, int m, int tn0, int tn1, bool grouped)
{
    (void)((m == Ms + 1 && (register_tiles<Ms + 1>(mat, in, tout, tn0, tn1, grouped), true)) || ...);
}
```
  2. In `run_tiles`, replace the body of `if constexpr (I == Isa::Bw && ACT_ROWS == 2 && EXL3_MOE_CPU_ACT_BLOCK == 128)`
     with:
```cpp
        // One token from either plan; several only as the DSV4.1 plan's compact input (PlanTraits::kCompactScratch).
        if ((m == 1 || in.compact) && mat.bits == 3 && act_blocked(mat.k))
        {
            register_tiles_for(std::make_integer_sequence<int, CHUNK_M>{}, mat, in, tout, m, tn0, tn1, grouped);
            return;
        }
```
  3. In `prepare_scratch`:
     - Replace the `} else { grow(ar.compact_g, ...` branch with:
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
     - Declare `size_t row0 = 0;` before the `for (int j = 0; j < nc; ++j)` loop that fills `ctx.prep_*`.
     - Replace its `if(compact) { ... }` block with:
```cpp
            if(compact) {
                ctx.prep_g[j].compact=ar.compact_g.data()+row0*H;
                ctx.prep_u[j].compact=ar.compact_u.data()+row0*H;
                ctx.prep_d[j].compact=ar.compact_d.data()+row0*I_;
                row0+=size_t(ACT_ROWS)*ctx.chunks[j].m;
            }
```
  4. In `PlanTraits`, change the `kGroupedTraversal` comment to
     `// range-aware multi-band traversal when one expert is routed (one-token chunks: kRegisterBudget[0])`.

- [ ] **Step 6: Accept 1..CHUNK_M (`shapes.hpp`).** Replace the comment and the loop in `accepts`:

```cpp
    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // every fact above (shape, limit 10, unswizzled 3-bit), and every chunk holds 1..CHUNK_M tokens (register_tiles<M>,
    // kRegisterBudget).
    static bool accepts(const ExpertLayer& l, const Exl3Quant::Params& p, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        if (l.hidden != kHidden || l.intermediate != kIntermediate || l.act_limit != kActLimit) return false;
        if (p.bits != kBits || p.swizzled) return false;
        for (const auto& ch : chunks)
            if (ch.m < 1 || ch.m > CHUNK_M) return false;
        return true;
    }
```

- [ ] **Step 7: Update the documentation the change falsifies.**

  1. In `README.txt` "Selection", replace the first paragraph (through "...other ISA/bit-width fallbacks are
     retained.") with:
  ```text
  For a call whose chunks each hold 1..CHUNK_M tokens (CHUNK_M = MAX_M / 2; MAX_M is
  EXL3_MOE_CPU_MAX_M, default 4), with H=5120, I=2304, residual activation rows,
  128-element quantization blocks, 3-bit unswizzled matrices and AVX512BW:
    one expert, one token: range3_unroll (range-aware neighboring-band traversal);
    other one-token chunks: unroll_small (one-band traversal);
    chunks of M > 1 tokens: register_tiles<M>, all M tokens sharing each decoded tile,
    kRegisterBudget[M - 1].pairs tile pairs per call.
  kRegisterBudget (math_avx512.hpp) holds one entry per M and a static_assert keeps each
  within 32 zmm; raising MAX_M needs an entry per new M. The measured multiple-expert
  counts are 3 and 5; counts 2, 4 and above 5 use the same default but have no
  measured performance claim. Every chunk size is bit-identical to the generic plan
  (test_exl3_cpu_dsv41_plan_multitoken.py). Invalid routes retain the generic path.
  Existing scalar, AVX2 and other ISA/bit-width fallbacks are retained.
  ```
  2. In "Code layout":
     - Change "compact scratch, grouped traversal, wide single-expert quantization" to "compact scratch (chunks of
       1..CHUNK_M tokens), grouped traversal and wide single-expert quantization (one token)".
     - Add the sentence "MAX_M (math.hpp) is one build-time constant, EXL3_MOE_CPU_MAX_M; the generic tiers' tile
       tables (kTiles) and the register dispatch (register_tiles_for) are generated from it."
  3. In `exl3_cpu_forward_ab.py`, replace the `ROUTES` comment's last sentence with:
     > The DSV4.1 plan takes chunks of 1..CHUNK_M tokens, so at the DSV4.1 shape these cases run it. A dump made
     > before 2026-10-06 ran them on the generic plan, and the two must agree bit for bit.

- [ ] **Step 8: Commit, push, SYNC, and run the parity test and the existing multi-row tests.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/kernels/jit/csrc/exl3/optimized \
  test/manual/dsv41/exl3_cpu_forward_ab.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "feat(exl3-cpu): the DSV4.1 plan takes chunks of 1..CHUNK_M tokens (register_tiles<M>, kRegisterBudget), bit-identical to the generic plan"
```
Then run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py test/manual/dsv41/test_exl3_cpu_act_quant.py \
  "test/manual/dsv41/test_cpu_expert_engines_exl3.py::test_a_multi_row_forward_on_prod_matches_one_row_forwards" \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `21 passed` (17 + 3 + 1), `EXIT=0`.

- [ ] **Step 9: Run the full bit-exact gate against BASE.**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && bash wt-cpu-plan-m2/test/manual/dsv41/run_exl3_cpu_forward_checks.sh \
  check wt-cpu-plan-m2 cpu-plan-m2/checks-head cpu-plan-m2/checks-base 2>&1 | tail -16; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: every step `EXIT=0` and `ALL GREEN (check)`:
- `compare-{bw,avx2,scalar}` print `72/72 bit-exact`. This covers parity for chunks of two against BASE's generic
  plan and the m = 1 identity, in one comparison.
- `bare-validate` verifies 24 outputs; `full-stack-validate` verifies 48.
- `pytest` passes.

On a mismatch, stop and use superpowers:systematic-debugging. Never adjust a reference.

- [ ] **Step 10: Rebuild the `MAX_M = 8` extension.** It is cold again: the kernel changed. Run Task 4b Step 6's
  `SGLANG_EXL3_CPU_MAX_M=8` warm line.
  Expected: `.../sglang_exl3_ext_resid_b128_m8_cpu_v1.so 4`.

- [ ] **Step 11: Run two core groups at once with multi-token chunks (Review Focus 5).**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && export PYTHONPATH=$PWD/python && PY=/data/models/slang/.venv/bin/python \
  && taskset -c 0-63 $PY test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa bw --registration cores --cores 0-7 \
     --out $W/t5-cores.pt 2>&1 | tail -1; echo "EXIT=${PIPESTATUS[0]}"; \
  $PY test/manual/dsv41/exl3_cpu_forward_ab.py compare $W/checks-base/ab-bw-slabs.pt $W/t5-cores.pt 2>&1 | tail -1; \
  echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `72 outputs (bw, cores)` and `EXIT=0`, then `72/72 bit-exact: bw/slabs vs bw/cores` and `EXIT=0`.

- [ ] **Step 12: Check parity at MAX_M = 8 (chunks of 1..4; Review Focus 1).**

```bash
ssh divix01 'bash -s' <<'EOF'
cd /data/models/slang/nvfp4-work/wt-cpu-plan-m2 && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
export PYTHONPATH=$PWD/python SGLANG_EXL3_CPU_MAX_M=8 PY=/data/models/slang/.venv/bin/python
taskset -c 0-63 $PY -m pytest test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py test/manual/dsv41/test_exl3_cpu_act_quant.py \
  -q -p no:randomly 2>&1 | tail -2; echo "EXIT=${PIPESTATUS[0]}"
taskset -c 0-63 $PY test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa bw --registration slabs --out $W/m8-t5-bw.pt 2>&1 | tail -1
$PY test/manual/dsv41/exl3_cpu_forward_ab.py compare $W/checks-base/ab-bw-slabs.pt $W/m8-t5-bw.pt 2>&1 | tail -1
echo "EXIT=${PIPESTATUS[0]}"
EOF
```
Expected:
- `20 passed`, `EXIT=0`. `test_the_build_reports_its_chunk_size` sees 4, and
  `test_every_chunk_size_matches_the_generic_plan` runs m = 1..4 and 5.
- `72/72 bit-exact: bw/slabs vs bw/slabs`, `EXIT=0`. Chunks of up to four through the DSV4.1 plan equal BASE's
  generic plan.

- [ ] **Step 13: Check the one-token machine code (byte-identical to BASE).** Build the default benches first
  (Task 6 Step 1). Then:

```bash
ssh divix01 'bash -s' <<'EOF'
W=/data/models/slang/nvfp4-work/cpu-plan-m2
for side in base head; do
  objdump -d --no-show-raw-insn -C $W/bench-$side/exl3_cpu_optimized \
    | awk '/^[0-9a-f]+ <\(anonymous namespace\)::(traversal_kblock|compact_pairs|register_tiles(<1>)?)\(/,/^$/' \
    | sed -E 's/^ *[0-9a-f]+:[[:space:]]*//; s/\b[0-9a-f]{6,}\b/ADDR/g; s/register_tiles<1>/register_tiles/g' \
    > $W/disasm-$side.txt
done
wc -l $W/disasm-base.txt $W/disasm-head.txt; diff -q $W/disasm-base.txt $W/disasm-head.txt && echo IDENTICAL
EOF
```
Expected: `IDENTICAL`, from two non-empty files.
- `run_tiles` itself is expected to differ (the generated dispatch), and so are the generic tiers (Task 4b's tables).
  Task 6 Step 2's one-token controls time those.
- If these three functions differ, report the diff with Task 6's controls. Do not paper over it.

- [ ] **Step 14: Check spills for every instantiated M** (see "How the plan checks register spills").

```bash
ssh divix01 'bash -s' <<'EOF'
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
B=/data/models/slang/nvfp4-work/wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench
GB=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src
for max_m in "" 8; do
  build=$W/bench-head-su${max_m:+-m$max_m}
  taskset -c 0-63 cmake -S $B -B $build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch \
    -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=$GB \
    -DEXL3_MAX_M=$max_m -DCMAKE_CXX_FLAGS=-fstack-usage > /dev/null
  taskset -c 0-63 cmake --build $build -j16 --target exl3_cpu_optimized 2>&1 | tail -1
done
for build in bench-base bench-head-su bench-head-su-m8; do
  echo "== $build: zmm stack accesses per function"
  objdump -d --no-show-raw-insn -C $W/$build/exl3_cpu_optimized | awk '
    /^[0-9a-f]+ <.*>:$/ { name = $0; keep = name ~ /register_band<|register_tiles|compact_pairs/; next }
    keep && /zmm/ && /\(%r[sb]p\)/ { count[name]++ }
    END { for (f in count) print count[f] "\t" f }' | sort -t$'\t' -k2
  echo "== $build: stack frames"
  find $W/$build -name '*.su' -exec grep -hE 'register_band|register_tiles|compact_pairs' {} + | sort
done
EOF
```
Expected:
- **Builds.** Both builds end `[100%] Built target exl3_cpu_optimized`. Compiling at all is the `kRegisterBudget`
  static_assert passing for M = 1..2 and 1..4.
- **Rows.** One row per instantiated function. A function that is absent has no zmm stack access, or was inlined into
  its caller (read the caller's row).
- **Gate.** For each M ≥ 2, divide its rows' total count by the fp32 partial sums of the bands it instantiates,
  `4 · M · Σ_{P=1..pairs} P`. M = 2 gives 4·2·(1+2) = 24; M = 3 and M = 4 give 12 and 16.
  - That ratio must not exceed BASE's, computed the same way over `register_band<1..3, 0, 0, true>` /
    `compact_pairs` (M = 1, Σ P = 6, 24 partial sums), plus 1.
  - Record the table and the ratios in Task 7.
- **On a miss.** It is an accumulator spill in the k-tile loop. Report it with Task 6's timings, and with Task 5b's
  for M = 3 and 4. Lowering that entry's `pairs` is the fix to try.

---

### Task 5b (conditional): Raise the default MAX_M to 8 (CHUNK_M 4), only if chunks of two beat the generic plan

This task runs after Task 5's gate and before Task 6's decision. It changes the source default only if both of these
hold:
- chunks of two beat the generic plan (the Task 6 measurement);
- chunks of up to four beat chunks of two, under the same ship gate.

Every measurement here uses the build-time override, so nothing is committed unless the gate passes.

**Files (only if Step 6 is reached):**
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/math.hpp` (the `EXL3_MOE_CPU_MAX_M` default)
- Modify: `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` (`DEFAULT_CHUNK_M`)
- Modify: `python/sglang/kernels/jit/csrc/exl3/optimized/README.txt` ("Selection": "default 4")

**Interfaces:**
- Consumes: Task 6 Steps 1-4 (`ab-routed-*/summary.tsv`); Task 5 Step 14's `bench-head-su-m8` spill table; `ab.sh`;
  `--routed=…:sharedG`.
- Produces: the decision "5b applied" or "5b skipped: <reason>", for Task 6 Step 5 and Task 7. If applied: default
  `MAX_M = 8` and `DEFAULT_CHUNK_M = 4`.

- [ ] **Step 1: Run Task 6 Steps 1-4 now** (benches, one-token controls, routed m = 2 against the generic plan, the
  M = 2 mutant).
  - Read `ab-routed-*/summary.tsv`. Chunks of two "beat the generic plan" when Task 6's ship gate passes on it:
    - no routed row has head/base > 1.00;
    - `16:3:random` or `16:3:shared2` has head/base ≤ 0.95;
    - every `experts:` control is in [0.98, 1.02].
  - If it does not pass: skip this task. Write "5b skipped: chunks of two did not beat the generic plan" plus the
    three numbers into Task 7's notes, and go to Task 6 Step 5.

- [ ] **Step 2: Build the `MAX_M = 8` bench without `-fstack-usage`** (timing build; same sources as Task 5 Step 14):

```bash
ssh divix01 'bash -s' <<'EOF'
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
taskset -c 0-63 cmake -S /data/models/slang/nvfp4-work/wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench \
  -B $W/bench-head-m8 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 \
  -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src \
  -DEXL3_MAX_M=8 > /dev/null
taskset -c 0-63 cmake --build $W/bench-head-m8 -j16 --target exl3_cpu_optimized 2>&1 | tail -1
EOF
```
Expected: `[100%] Built target exl3_cpu_optimized`.

- [ ] **Step 3: Check the M = 3 and M = 4 spill gate** from Task 5 Step 14's `bench-head-su-m8` table. Expected: both
  ratios within the gate. On a miss, stop and ask. Do not lower `pairs` below 1: there is nothing lower.

- [ ] **Step 4: Run the one-token controls, `MAX_M` 4 against 8:**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-head cpu-plan-m2/bench-head-m8 \
  cpu-plan-m2/ab-m8-m1-$(date +%Y%m%d-%H%M%S) --benchmark_filter=experts: 2>&1 | tail -5'
```
Expected: three rows, `dsv41` on both sides, and head/base (here 8 against 4) in [0.98, 1.02].

- [ ] **Step 5: Run chunks of two and four on the routed 16:3 patterns:**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-head cpu-plan-m2/bench-head-m8 \
  cpu-plan-m2/ab-m8-routed-$(date +%Y%m%d-%H%M%S) \
  --routed=16:3:shared1,16:3:shared2,16:3:shared4,16:3:random,6:3:random,64:6:random --benchmark_filter=rows: 2>&1 | tail -8'
```
Expected: six rows, `dsv41`/`dsv41`.
- `16:3:shared1` and `16:3:shared2` are controls. They make the same chunks in both builds and should read ≈1.00.
- `16:3:shared4` is the first pattern that runs chunks of four.
- The gate: no row above 1.00, and `16:3:shared4` or `16:3:random` at ≤ 0.95.
- For the record (not gating), run the same command with `cpu-plan-m2/bench-base` in place of `cpu-plan-m2/bench-head`.
  That gives chunks of four against the generic plan.

- [ ] **Step 6: If Steps 3-5 pass, raise the default.** Otherwise stop and ask the owner, with both tables, and commit
  nothing.

  1. `math.hpp`: `#define EXL3_MOE_CPU_MAX_M 4` → `#define EXL3_MOE_CPU_MAX_M 8`.
  2. The test: `DEFAULT_CHUNK_M = 2` → `DEFAULT_CHUNK_M = 4`.
  3. `README.txt` "Selection": "default 4" → "default 8".
  4. Commit and push:
```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add python/sglang/kernels/jit/csrc/exl3/optimized \
  test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "perf(exl3-cpu): chunks hold up to four tokens by default (EXL3_MOE_CPU_MAX_M 8)"
```
  5. SYNC, then rerun:
     - Task 5 Step 8 (expected `21 passed`, now at `CHUNK_M` 4);
     - Step 9 (`ALL GREEN (check)`, 72/72 against BASE);
     - Step 11 (cores, 72/72).

     The extension under `resid_b128_cpu_v1` rebuilds cold (Task 4b Step 6's default warm line now prints `4`).
  6. Rerun Task 5 Step 13. Expected: the three one-token functions now differ from BASE only in the
     `bq`/`bsum` stride immediates (`kb * MAX_M`, 16 → 32 bytes). Record the diff.
  7. Record "5b applied" for Task 6 Step 5 and Task 7.

---

### Task 6: Measure the new plan against the generic plan, and decide

**Files:** none in the repo. Results go under `/data/models/slang/nvfp4-work/cpu-plan-m2/` on divix01, and into
Task 7.

**Interfaces:**
- Consumes: `ab.sh` and `--routed` (Task 4); BASE and the base worktree (Task 4); the head (Task 5); Task 5b's outcome.
- Produces: `ab-m1-*/summary.tsv`, `ab-routed-*/summary.tsv`, the `kRegisterBudget[1].pairs` choice, and the gate
  decision.

- [ ] **Step 1: Build both default benches.** Run this once per side: first `side=base` with
  `wt=…/wt-cpu-plan-m2-base`, then `side=head` with `wt=…/wt-cpu-plan-m2`.

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
Expected: three rows (`experts:1`, `:3`, `:5`), `dsv41` on both sides, each head/base in [0.98, 1.02].

- [ ] **Step 3: Run the routed workloads: chunks of two against the generic plan.**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-cpu-plan-m2/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-base cpu-plan-m2/bench-head \
  cpu-plan-m2/ab-routed-$(date +%Y%m%d-%H%M%S) \
  --routed=2:3:shared2,16:3:shared2,16:3:shared1,6:3:random,16:3:random,64:6:random --benchmark_filter=rows: 2>&1 | tail -8'
```
Expected: six rows.
- `16:3:shared1` is `dsv41`/`dsv41`, a control at ≈1.00.
- Every other row is `generic` on base and `dsv41` on head.
- The gate: no row above 1.00, and `16:3:random` or `16:3:shared2` at ≤ 0.95.

- [ ] **Step 4: Run the M = 2 mutant.**
  - The budget table allows `pairs` 1 or 2 for M = 2: 3 needs 36 zmm, which the static_assert refuses.
  - Try 1 by the run protocol: a private worktree, never committed.

```bash
ssh divix01 'bash -s' <<'EOF'
R=/data/models/slang/sglang; M=/data/models/slang/nvfp4-work/wt-cpu-plan-m2-p1
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh; SITE=/data/models/slang/.venv/lib/python3.13/site-packages
git -C $R worktree add --detach $M origin/dsv41-cpu-plan-m2
sed -i 's|    {2, false},  // M = 2|    {1, false},  // M = 2|' $M/python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp
grep -n "// M = 2" $M/python/sglang/kernels/jit/csrc/exl3/optimized/math_avx512.hpp
taskset -c 0-63 cmake -S $M/python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $W/bench-p1 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src > /dev/null
taskset -c 0-63 cmake --build $W/bench-p1 -j16 --target exl3_cpu_optimized 2>&1 | tail -1
PYTHON=/data/models/slang/.venv/bin/python bash $M/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh \
  $W/bench-head $W/bench-p1 $W/ab-p1-$(date +%Y%m%d-%H%M%S) \
  --routed=2:3:shared2,16:3:shared2,16:3:random,64:6:random --benchmark_filter=rows: 2>&1 | tail -5
git -C $M checkout -- . && git -C $R worktree remove $M
EOF
```
Expected: the `grep` shows `{1, false},  // M = 2`, then four rows, where "head" means the mutant.
- If the mutant is ≥3% faster on `16:3:random` and `16:3:shared2` and slower on none: set `{1, false}` in a commit
  (`perf(exl3-cpu): chunks of two take one tile pair per call`).
- Then rerun Task 5 Steps 8, 9 and 14, and this task's Step 3.

- [ ] **Step 5: Decide by the ship gate and report to the owner.**
  - The report covers:
    - Steps 2-4's `summary.tsv` tables;
    - Task 5 Steps 13-14's verdicts;
    - Task 5b's outcome (applied with its tables, or skipped with its reason);
    - Task 1's collision rate, if it is in.
  - If the gate passes, proceed to Task 7.
  - If it fails, stop. Report to the owner with the numbers, and revert nothing without their decision.

---

### Task 7: Record the result

**Files:**
- Modify: `DSV41_REFERENCE.md`. Add **§33.10**, after §33.9. The dspark-graph branch's Task 7 writes §33.9; if it is
  not on this branch yet, put §33.10 after §33.8 and keep its number.

**Interfaces:**
- Consumes: Task 1 Step 6's collision rate; Task 4b Steps 7-8; Task 5 Steps 9-14; Task 5b's outcome; Task 6's tables and
  decision.
- Produces: nothing code reads.

- [ ] **Step 1: Write §33.10.** Copy every number, with its command, from the `summary.tsv` or log it came from. Use this
  structure:

```markdown
### 33.10 The DSV4.1 CPU plan takes chunks of 1..CHUNK_M tokens (2026-10-06)

Plan `docs/superpowers/plans/2026-10-06-dsv41-cpu-plan-multi-token.md`, branch `dsv41-cpu-plan-m2`. It covers §33.3 item
5's "the tuned path turns off for chunks with more than one token".

**What changed.**
- `MAX_M` is one build-time constant, `EXL3_MOE_CPU_MAX_M`, default <4 | 8 (Task 5b)>.
  - A test build sets it through `SGLANG_EXL3_CPU_MAX_M`.
  - The generic tiers' tile tables and the register dispatch are generated from it.
- `Dsv41Shape::accepts` takes chunks of 1..CHUNK_M tokens.
- `register_tiles<M>` shares each decoded tile among a chunk's M tokens.
  - `kRegisterBudget`: M = 2 → <pairs>, M = 3 and 4 → 1. A static_assert keeps every entry within 32 zmm.
  - The grouped traversal and wide quantization stay one-token only.
- The kernel counts forwards per plan and reports CHUNK_M (`sglang_exl3_cpu::plan_calls`, `chunk_m`).
- The draft CPU thread counts collided jobs.

**Bits.**
- `run_exl3_cpu_forward_checks.sh check` against BASE `<sha>`: 72/72 per tier, 24 + 48 frozen outputs, ALL GREEN.
- At `SGLANG_EXL3_CPU_MAX_M=8`: scalar, avx2 and bw dumps 72/72 against BASE (the generic plan before Task 5, the
  DSV4.1 plan after it). The parity file passes, 17 tests at CHUNK_M 4.
- One-token machine code: <IDENTICAL | the diff>.
- Spill ratios per M (zmm stack accesses per fp32 partial sum): M = 1 (BASE) <r1>; M = 2 <r2>; M = 3 <r3>;
  M = 4 <r4>.

**Time** (`ab.sh`, 8 rounds, 16 workers on 18-33, median p50 µs; <dirs>):

| workload | generic (BASE) | chunks ≤ 2 (head) | head/base | chunks ≤ 4 (MAX_M 8) | 8/4 |
|---|---|---|---|---|---|
| experts:1 / 3 / 5 (controls) | | | | | |
| rows:2/k:3/shared2 | | | | | |
| rows:16/k:3/shared1 (control) | | | | | |
| rows:16/k:3/shared2 | | | | | |
| rows:16/k:3/shared4 | | | | | |
| rows:6/k:3/random | | | | | |
| rows:16/k:3/random | | | | | |
| rows:64/k:6/random | | | | | |

**Task 5b:** <applied: default MAX_M 8 | skipped: reason and numbers>.

**How often the draft hits it** (Task 1, one DSpark arm, <log path>): collided_jobs / jobs = <x>;
collided_forward_ns / forward_ns = <y>.

**What it does not do:**
- multi-row jobs in the target's `CpuExpertEngine` (D2-4);
- the grouped traversal or wide quantization at m ≥ 2;
- swizzled compact input;
- running the VNNI and VBMI tiers at a raised `MAX_M` (compiled only; no such CPU here).
```

- [ ] **Step 2: Commit and push.**

```bash
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 add DSV41_REFERENCE.md
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 commit -m "docs(dsv41): §33.10, the DSV4.1 CPU plan on chunks of 1..CHUNK_M tokens"
git -C /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-plan-m2 push origin dsv41-cpu-plan-m2
```

---

## Risks

- **Chunks of M > 1 may not beat the generic plan.**
  - `register_band<…, M>` holds up to 16 integer accumulators plus `4 · M · P` fp32 partial sums, which may sit in
    memory.
  - The generic `bw_tiles<3, 2M>` also shares each decode across the chunk's rows.
  - Mitigation: Task 6's A/B, the M = 2 mutant, Task 5b's own gate, and stop-and-ask.
- **Templating perturbs the one-token path.**
  - `forward_plan.hpp`'s header warns that inlining and cloning follow definition order.
  - Mitigation: the frozen references (bits), Task 5 Step 13 (machine code), and Task 6 Step 2 (time).
- **The generated dispatches.**
  - The generic tiers now call through `kTiles` function pointers instead of a switch of direct calls. The register
    dispatch is a fold of direct calls.
  - Bits are unaffected (Task 4b Step 7). The cost is one indirect call per GEMV range, against microseconds of work.
  - Task 6 Step 2 times the DSV4.1 path; the generic plan is not timed.
- **The generic tiers at a raised `MAX_M` are correct, not tuned.**
  - AVX2's `acc[MAX_M][2]` is 16 ymm at `MAX_M = 8`.
  - The VNNI and VBMI rows 5..8 compile but cannot run on divix01, a Skylake-SP: no bit test covers them there.
  - The default stays 4 unless Task 5b passes, and 5b gates on the DSV4.1 path only.
- **Compile time.** The tile tables instantiate bits × `MAX_M` rows × up to 8 bands per AVX-512 tier, so
  `MAX_M = 8` roughly doubles the AVX-512 tile instantiations of a cold extension build.
- **`kRegisterDecodeZmm` is an estimate.** It is counted by hand from `register_rows`. The binary check (Task 5 Step 14)
  is the backstop, and timing is the arbiter.
- **Scratch offsets.** A wrong running row offset corrupts only calls that mix chunk sizes, and it does so silently.
  Mitigation: `mixed-sizes`, `m{CHUNK_M+1}-overflow`, and AB `(3, 1)` / `(6, 3)` / `(16, 3)`.
- **The in-process generic oracle is the swizzled layer.**
  - It is valid while the generic plan is layout-independent bitwise, which `test_swizzled_layout_matches_native`
    pins.
  - The A/B dumps against BASE are the independent oracle.
- **Counter contention.** The target engine and the draft thread touch `g_plan_calls` once per forward each: about
  100 ns at worst, against forwards of 300 µs or more.
- **Production and shared caches.**
  - The benches pin 18-33, and the scripts refuse while a server runs.
  - Every extension build lives under the private `SGLANG_EXL3_BUILD_DIR`, the `MAX_M = 8` one with its own flavor.
- **The payoff is measured synthetically.** Only Task 1's real collision rate says how much a DSpark arm gains.

## Out of scope

- Multi-row jobs in the target's `CpuExpertEngine` (`rows = 1`), and the rest of D2-4.
- The grouped multi-band traversal and wide (512) quantization at m ≥ 2 (owner: follow-up).
- Swizzled compact input at m > 1: the DSV4.1 plan refuses swizzled layers.
- `CHUNK_M` above 4.
  - `kRegisterBudget` would need entries for M ≥ 5. M = 5 at one pair fills all 32 zmm exactly, and M ≥ 6 does not
    fit, so it needs a different register kernel.
  - `math.hpp`'s static_assert caps `MAX_M` at 8, which also bounds the generic tile tables.
- Tuning the generic tiers (AVX2, VNNI, VBMI) for rows 5..8, and the vendored baseline kernel's `MAX_M`.
- Changing the dispatch's grouping or its accumulation order.
- Running a DSpark arm. The owner schedules it; Task 1 Step 6 only reads its log.
- An env knob forcing the generic plan (owner: counters).

## Self-review

1. **Spec coverage.**
   - Brief item 1 (which pieces assume one token): the "What assumes one token per chunk" table.
   - Item 2 (the numerics): Global Constraints, then Task 4b Step 7 and Task 5 Steps 8-12.
   - Item 3 (tests): Tasks 2, 3 and 5. Item 4 (measurement): Tasks 1, 4, 6 and 5b. Item 5: Risks and Out of scope.
   - The revision, point by point:
     1. no m == 2 special case: `register_tiles<M>` and the `register_tiles_for` fold over `CHUNK_M`, with M = 1
        identical by objdump (Task 5);
     2. one `kRegisterBudget` table with static_asserts, and a per-M spill check (Task 5 Steps 4 and 14);
     3. the audit table, with every site derived from `MAX_M` and generic drift tested at `MAX_M = 8` (Task 4b);
     4. `accepts` takes `1..CHUNK_M` (Task 5 Step 6);
     5. `EXL3_MOE_CPU_MAX_M` / `SGLANG_EXL3_CPU_MAX_M` / `EXL3_MAX_M`, `chunk_m` to Python, tests generated from
        `chunk_m`, and a `MAX_M = 8` run (Tasks 2, 4b and 5 Step 12);
     6. Task 5b;
     7. §33.10 (Task 7).
2. **Placeholders.**
   - Only Task 7's result table has blanks, by design: values not yet measured, each tied to its source file.
   - No TBD or TODO, and no step without its code or command.
3. **Type consistency.** These names are used the same way in every task:
   - `SglangExl3CpuPlanCalls{dsv41, generic}`, `exl3_cpu_plan_calls()`, `exl3_cpu_chunk_m()`,
     `sglang_exl3_cpu::plan_calls`, `sglang_exl3_cpu::chunk_m`, `g_plan_calls`;
   - `EXL3_MOE_CPU_MAX_M`, `SGLANG_EXL3_CPU_MAX_M`, `EXL3_MAX_M`, `kTiles`, `tiles_for`, `kRowIndex`;
   - `RegisterBudget`, `kRegisterBudget`, `kRegisterDecodeZmm`, `register_tiles<M>`, `register_tiles_for`,
     `register_band_for`;
   - `_expected_plan`, `DEFAULT_CHUNK_M`, `_chunk_m`, `_check`;
   - `RoutedWorkload`, `sharedG`, `ab.sh`, `summary.tsv`;
   - `collided_jobs`, `shared_routes`, `collided_forward_ns`, `count_shared_routes`.
4. **Review Focus.** Each of the five lines names its test and the step that owns it.
