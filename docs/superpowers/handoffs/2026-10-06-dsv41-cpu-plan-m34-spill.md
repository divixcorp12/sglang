# Handoff: the DSV4.1 CPU plan's M = 3 and M = 4 register kernels spill

Written 2026-10-06 at commit `da2d8d9dbf` (branch `dsv41-cpu-plan-m2`, code last changed at `4cf19a3f0d`). Self-contained: you
need no prior context. Everything runs on divix01, CPU only.

## Background in five lines

- The EXL3 CPU expert kernel (`python/sglang/kernels/jit/csrc/exl3/optimized/`) groups a call's routes by expert into
  **chunks** of up to `CHUNK_M` tokens. `CHUNK_M = MAX_M / ACT_ROWS`; `ACT_ROWS = 2` (token row + residual row).
- `MAX_M` is one build-time constant, `EXL3_MOE_CPU_MAX_M` (`math.hpp:46-52`, default **4**, so `CHUNK_M = 2`; even, 2..8).
  Overrides: Python `SGLANG_EXL3_CPU_MAX_M` (`srt/environ.py`, `srt/layers/quantization/exl3/ext.py:66-73`, own extension
  flavor `_m<N>`); bench `-DEXL3_MAX_M=<N>` (`bench/CMakeLists.txt:41,87`).
- The DSV4.1 plan (`ForwardPlan<Dsv41Shape, Isa::Bw>`) now accepts chunks of 1..`CHUNK_M` tokens
  (`shapes.hpp:42-50`, `Dsv41Shape::accepts`), bit-identical to the generic plan. At the default `MAX_M = 4` that is
  chunks of 1 and 2 tokens, and it wins 24-42% over the generic plan on routed patterns.
- Chunks of 3 and 4 tokens exist only in a `MAX_M = 8` build. They work (bit-exact) but the kernels for M = 3 and 4 spill
  registers, and `MAX_M = 8` loses on some patterns. The owner kept the default at 4.
- **Your job:** make M = 3 and M = 4 not spill, so raising the default `MAX_M` to 8 becomes a win. Raising the default
  is a separate decision, taken only after your pass criteria hold.

## The problem, with numbers

### Spill table (zmm accesses to `%rsp`/`%rbp` per function, and `-fstack-usage` frame bytes)

Measured at `229aa33ba5` (kernel code unchanged since). Raw output: `/data/models/slang/nvfp4-work/cpu-plan-m2/t5-s14.txt`.

| function | zmm stack accesses | frame bytes | ratio = accesses / (4 * M * sum of P) |
|---|---|---|---|
| BASE `register_band<1,0,0,true>` (M=1) | 112 | n/a (no -su at BASE) | |
| BASE `register_band<2,0,0,true>` | 88 | | |
| BASE `register_band<3,0,0,true>` | 249 | | |
| BASE M=1 total P=1..3 | 449 | | 449 / 24 = **18.7** (limit 19.7) |
| head `register_band<1..4,0,0,true,1>` (M=1) | 112 / 88 / 249 / 309 (same as BASE) | 1032 / 1352 / 2248 / 2760 | same |
| head `register_band<1,0,0,true,2>` (M=2, P=1) | 408 | 3336 | |
| head `register_band<2,0,0,true,2>` (M=2, P=2) | 285 | 4104 | (408+285) / 24 = **28.9** |
| head `register_band<1,0,0,true,3>` (M=3, P=1) | 722 | 5704 | 722 / 12 = **60.2** |
| head `register_band<1,0,0,true,4>` (M=4, P=1) | 1064 | 8008 | 1064 / 16 = **66.5** |

The plan's gate was: each M >= 2 ratio must not exceed BASE's M = 1 ratio plus 1 (19.7). Every M >= 2 misses; M = 3 and 4
miss by 3x. M = 2 misses too but its timings win, and the owner overruled that miss.

### Timings (median p50 in us over 8 alternating rounds, `ab.sh`; "base" is the first build, "head" the second)

**MAX_M = 4 head vs the generic plan (BASE `b750c6d96f`)** (`ab-routed-20261005-225531`):

| pattern | generic | head | head/base |
|---|---|---|---|
| `2:3:shared2` | 1929.4 | 1126.0 | 0.584 |
| `16:3:shared2` | 14684.2 | 9743.6 | 0.664 |
| `16:3:shared1` (control, same plan) | 16687.7 | 16711.3 | 1.001 |
| `6:3:random` | 8712.5 | 6465.9 | 0.742 |
| `16:3:random` | 19267.4 | 14594.2 | 0.757 |
| `64:6:random` | 124366.1 | 75229.0 | 0.605 |

**One-token controls, BASE vs MAX_M = 4 head** (`ab-m1-20261005-225448`): `experts:1` 0.977, `experts:3` 1.000,
`experts:5` 1.016.

**MAX_M = 8 build vs MAX_M = 4 build, same head `4cf19a3f0d`** (the 5b measurement; base = MAX_M 4, head = MAX_M 8):

| pattern | MAX_M 4 | MAX_M 8 | head/base | note |
|---|---|---|---|---|
| `experts:1` / `:3` / `:5` | 346.3 / 1001.5 / 1705.6 | 344.8 / 995.1 / 1708.8 | 0.996 / 0.994 / 1.002 | controls, pass |
| `16:3:shared1` | 17270.5 | 17390.8 | 1.007 | noise control (one-token chunks) |
| `16:3:shared2` | 9701.1 | 9787.5 | 1.009 | noise control (chunks of two) |
| `16:3:shared4` | 9690.4 | 6434.5 | **0.664** | win (chunks of four) |
| `64:6:random` | 75534.5 | 61571.4 | **0.815** | win |
| `16:3:random` | 14850.7 | 15553.0 | **1.047** | **regression** |
| `6:3:random` | 6589.2 | 6724.5 | **1.021** | **regression (borderline)** |

(`ab-m8-m1-20261005-234648`, `ab-m8-routed-20261005-234723`.) MAX_M 8 against the generic plan: `16:3:shared4` 0.438,
`16:3:random` 0.793, `64:6:random` 0.497 (`ab-m8-vsbase-20261006-000101`). So 8 is far better than the generic plan; it
is worse than 4 only on mixed random patterns, where a chunk's size is 1-4 and the M = 3/4 kernels are slower per token
than two M = 2 calls would be.

**Noise floor** (same binary on both sides, 8 rounds, `ab-noise-20261005-232648`): `experts:1` 1.012, `16:3:shared1`
1.007. So about +/-1.2%; the shared1/shared2 controls above agree (1.007, 1.009). Absolute `experts:1` p50 moves 345-380 us
between runs, so always compare within one `ab.sh` run, never across runs.

## Where the code is

All paths are under `python/sglang/kernels/jit/csrc/exl3/optimized/` unless noted; line numbers at `da2d8d9dbf`.

| what | where |
|---|---|
| `MAX_M`, `CHUNK_M`, `ACT_ROWS` | `math.hpp:46-52` (override macro + range `static_assert`), `:63-64` |
| register budget table `RegisterBudget`, `kRegisterBudget[]` | `math_avx512.hpp:919-931` (entries: M=1 `{4,true}`, M=2 `{2,false}`, M=3 `{1,false}`, M=4 `{1,false}`) |
| `kRegisterDecodeZmm = 12`, `register_budget_fits`, the two `static_assert`s | `math_avx512.hpp:932-946` |
| `register_rows<J,Pairs,Compact,M>` (decode one tile row, `madd` into `acc[band][half][2*M]`) | `math_avx512.hpp:947-985` |
| `register_band<Pairs,FixedK,FixedN,Compact,M>` (the k-block loop; `sums[Pairs][2][R]`, `acc[Pairs][2][R]`, `R = 2*M`) | `math_avx512.hpp:990-1043` |
| `register_band_for<M>` (runtime P to a direct call) | `math_avx512.hpp:1046-1050` |
| `register_tiles<M>` (M=1: today's code incl. `compact_pairs`, `traversal_tiles`; M>1: runs of `kRegisterBudget[M-1].pairs` pairs) | `math_avx512.hpp:1157-1203` |
| `register_tiles_for` (one direct call per M = 1..`CHUNK_M`) and `run_tiles` (the gate `(m == 1 \|\| in.compact) && bits == 3`) | `forward_plan.hpp:107-130` |
| compact scratch laid out by rows (`row0`) | `forward_plan.hpp:461-515` (`prepare_scratch`) |
| `PlanTraits<Dsv41Shape, Isa::Bw>` | `forward_plan.hpp:435-442` |
| `Dsv41Shape::accepts` (`1 <= m <= CHUNK_M`) | `shapes.hpp:38-50` |
| dispatch chunking (`ch.m < CHUNK_M`) | `forward_plan.hpp:733` |
| README "Selection" / "Code layout" (update if behavior changes) | `README.txt` |
| bench, routed workloads, `ab.sh` | `python/sglang/kernels/jit/csrc/moe/expert_stream/bench/` (`src/cpu_forward.cpp`, `ab.sh`, `README.txt`) |
| parity and plan tests | `test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` (`DEFAULT_CHUNK_M = 2`) |

## Why it spills

`register_band<Pairs,...,M>` processes `Pairs` tile pairs (each pair = 2 output tiles = 2 halves of 16 outputs) for `M`
tokens. Per k-block (128 inputs, 8 k-tiles) it keeps:

- **integer accumulators** `acc[Pairs][2][2M]` zmm: `R = 2M` quantized rows (M token rows + M residual rows) x 2 halves x
  Pairs. That is `4 * M * Pairs` zmm, live across the 8-k-tile inner loop;
- **decode temporaries** in `register_rows`: prev, a, b, c, the decoded state, its byte-sum, two products, the multiplier
  halves, the ones vector, the broadcast activation pair: about 12 zmm (`kRegisterDecodeZmm`);
- **fp32 partial sums** `sums[Pairs][2][2M]`: another `4 * M * Pairs` zmm, live across the *outer* k-block loop (16 k-blocks
  at H = 5120), converted from `acc` at the end of each k-block (`fmadd` with scale/correction, then `add` in increasing
  k-block order, which is the generic plan's rounding order).

The compile-time budget only counts the integer part: `4 * M * Pairs + 12 <= 32`. At M = 4, Pairs = 1 that is 16 + 12 = 28,
but the 16 fp32 partial sums are *also* live, so the real demand is 44 zmm; at M = 3 it is 12 + 12 + 12 = 36; at M = 2,
Pairs = 2 it is 16 + 12 + 16 = 44; at M = 2, Pairs = 1 it is 8 + 12 + 8 = 28 (fits, yet the table shows 408 accesses: the
compiler still keeps some partial sums in memory and reloads them per k-block). The validated M = 1, Pairs = 4 code already
keeps fp32 partial sums in memory by design (309 accesses), so "some partial-sum traffic" is normal; the question is
whether the k-tile loop spills *integer* accumulators or decode temporaries, which is what costs time. The per-partial-sum
ratio mixes both, which is why it reads so high. Step 1 of the work below is to separate them.

## Fix approaches, ranked

1. **Sub-tile M into register-resident groups of <= 2 tokens that share one decode (most likely to work).**
   Decode a tile row once into `prev,a,b,c`/`state`/`sum` (the expensive part), then loop `for sub in groups of 2 tokens`
   accumulating each group's `acc[Pairs][2][4]` into its own fp32 `sums` slot, keeping only one group's integer
   accumulators live at a time. For M = 3 use groups {2,1}, for M = 4 groups {2,2}. Integer accumulators stay at the M = 2
   footprint (8 per pair), decode is shared (same as today), and the M = 2 kernel is the measured winner. Cost: the decoded
   `sum` (one zmm) must stay live or be recomputed per group, and `sums` must still hold `2M` rows per pair, so keep
   `Pairs = 1` and keep the partial sums memory-backed deliberately. This needs a restructured inner loop in
   `register_rows`/`register_band` (loop the `acc` update over groups, one `madd` per row as today); bit-exactness holds
   because each (row, k-block) integer sum and each FMA is unchanged.
2. **Flush partial sums per k-block into a small fp32 scratch row instead of keeping `sums` live** (reorder to
   k-block-outer, store `v = fmadd(...)` to a per-band scratch, accumulate in memory). That makes the spill explicit and
   cheap (one store + one load per output vector per k-block) instead of compiler-chosen, and frees 4 * M * Pairs zmm for
   the inner loop. Keep the increasing-k-block add order so the result stays bit-identical. Expect to win where approach 1
   leaves `sums` register-resident; the M = 1 traversal already does this (`traversal_kblock` writes fp32 partials to
   aligned scratch).
3. **Reduce live accumulators by splitting the M rows across calls** (call `register_band<1,...,2>` twice for M = 4, with
   half the tokens each). Simplest to write (a loop in `register_tiles<M>`, no kernel change), but it decodes every tile
   twice, so it only helps if decode is cheaper than the spill traffic. Treat as the baseline to beat, and as a fallback
   if 1 and 2 stall. This amounts to making M = 3 and 4 behave like M = 2 twice.
4. **Cap `CHUNK_M` at the winning size in the dispatch** (`forward_plan.hpp:733`: `ch.m < min(CHUNK_M, 2)`), so a
   `MAX_M = 8` build never makes chunks of 3-4 for the DSV4.1 plan while the generic tiers still use the larger chunks.
   Not a fix; it makes `MAX_M = 8` equal `MAX_M = 4` on the DSV4.1 plan, so it only exists to unblock a default raise if
   nothing else works. Do not do this unless the owner agrees.

Tune `kRegisterBudget[M-1].pairs` last: M = 3 and 4 already have the minimum (1), and the `static_assert` forbids 3 pairs
at M = 2.

## How to validate

### Run rules (`.claude/rules/divix01-run-protocol.md`; follow exactly)

- Code is written on the laptop, committed, pushed (`git push origin <branch>`), and pulled on divix01 into a **private
  worktree** `/data/models/slang/nvfp4-work/wt-<name>`:
  `ssh divix01 'git -C /data/models/slang/sglang fetch origin && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-<name> origin/<branch>'`.
  Never rsync/scp a tree. Never run in `cc-expert-prediction/dsv41-direct-prod`, never start or stop a server.
- Every Python run: `PYTHONPATH=$PWD/python`, and print `sglang.__file__` once (the interpreter trap: the venv otherwise
  imports another tree).
- CPU jobs under `taskset -c 0-63` with `OMP_NUM_THREADS` capped (8). Cores 64-71 stay free. Set
  `SGLANG_EXL3_CPU_CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++`.
- Read pytest's status, not a pipe's: `cmd > log 2>&1; echo EXIT=$?` or `... | tail -2; echo EXIT=${PIPESTATUS[0]}`.
- **Timing benches (`ab.sh`) pin cores 18-33, and the box is shared.** Before one: no `launch_server`, `trace_corpus` or
  `graphed_verify` process (`pgrep -af`), and ask the team lead for a "bench clear" while GPU arms may be running. If a
  `cc-gpu.lock` holder overlaps a bench (`flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true || echo held`), re-run it.
  GPU work is not part of this task.
- Cold JIT: a header change rebuilds the extension (minutes) and each host variant (50-100 s). Warm builds serially before
  anything times out; a `TimeoutExpired` on a cold cache is the compiler, not a hang.
- Mutants (a throwaway edit, e.g. changing a `kRegisterBudget` entry) only in their own private worktree, never committed:
  `git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-<name> <commit>`, edit, build,
  run, `git checkout -- .`, `git worktree remove`. Then re-run the suite and record it green.
- Ship code with a commit per step (fix-up commits are fine; never amend or rebase).

### Environment file

Create `/data/models/slang/nvfp4-work/cpu-plan-m2/env.sh` if it is missing (it is outside every repo). `source` it for the
commands below (it sets `W`, `GXX`, build dir, `SGLANG_DSV41_CPU_EXPERTS=1` and friends):

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

Note: `SGLANG_DSV41_CPU_EXPERTS=1` in that file breaks `test_exl3_ext.py` (it tests the unset case); run that file with
`env -u SGLANG_DSV41_CPU_EXPERTS`.

### 1. Bit-exact gates (outputs must not change by one bit)

BASE's dumps already exist at `$W/checks-base` (made at `b750c6d96f`, the commit before any kernel change). If they are
gone, rebuild them: create a worktree at `b750c6d96f` and run
`bash <wt-base>/test/manual/dsv41/run_exl3_cpu_forward_checks.sh baseline <wt-base> $W/checks-base` (must end
`ALL GREEN (baseline)`). Never regenerate a reference to make a mismatch go away.

**MAX_M = 4 (default build), the full gate.** `BASELINE_OUT` must be an **absolute** path (the script `cd`s into the
worktree first):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work && bash wt-<name>/test/manual/dsv41/run_exl3_cpu_forward_checks.sh \
  check wt-<name> cpu-plan-m2/checks-<name> /data/models/slang/nvfp4-work/cpu-plan-m2/checks-base > cpu-plan-m2/checks-<name>.out 2>&1; \
  echo EXIT=$?; tail -14 cpu-plan-m2/checks-<name>.out'
```
Expected: `ALL GREEN (check)`; `compare-{bw,avx2,scalar}` each `72/72 bit-exact`; `bare-validate` 24 outputs;
`full-stack-validate` 48 outputs; `pytest` passes (65 tests: `test_cpu_expert_engines_exl3.py`, `test_cpu_expert_service.py`,
`test_exl3_cpu_dsv41_plan_multitoken.py`).

**The parity tests and multi-row tests, MAX_M = 4:**

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-<name> && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh \
  && PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py test/manual/dsv41/test_exl3_cpu_act_quant.py \
  "test/manual/dsv41/test_cpu_expert_engines_exl3.py::test_a_multi_row_forward_on_prod_matches_one_row_forwards" \
  -q -p no:randomly > /tmp/gate.log 2>&1; echo EXIT=$?; tail -1 /tmp/gate.log'
```
Expected: `21 passed`.

**MAX_M = 8 (this is where M = 3 and 4 run), the gates that matter for your change.** Cold extension build, several
minutes, then:

```bash
ssh divix01 'bash -s' <<'EOF'
cd /data/models/slang/nvfp4-work/wt-<name> && source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
export PYTHONPATH=$PWD/python SGLANG_EXL3_CPU_MAX_M=8 PY=/data/models/slang/.venv/bin/python
taskset -c 0-63 $PY -m pytest test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py test/manual/dsv41/test_exl3_cpu_act_quant.py \
  -q -p no:randomly > $W/m8-pytest.log 2>&1; echo "pytest EXIT=$?"; tail -1 $W/m8-pytest.log     # expect 20 passed
for isa in bw avx2 scalar; do
  taskset -c 0-63 $PY test/manual/dsv41/exl3_cpu_forward_ab.py dump --isa $isa --registration slabs --out $W/m8-$isa.pt > $W/m8-$isa.log 2>&1
  echo "dump $isa EXIT=$?"
  $PY test/manual/dsv41/exl3_cpu_forward_ab.py compare $W/checks-base/ab-$isa-slabs.pt $W/m8-$isa.pt 2>&1 | tail -1   # expect 72/72 bit-exact
done
EOF
```
The dump cases include chunks of 1-4 tokens (`t3k1`, `t6k3`, `t16k3` etc. in `exl3_cpu_forward_ab.py` `ROUTES`), so this
is the parity check for M = 3 and 4 against BASE's generic plan. `test_every_chunk_size_matches_the_generic_plan` runs
m = 1..4 and 5 at MAX_M 8.

The A/B harness also has a two-core-group mode: `exl3_cpu_forward_ab.py dump --isa bw --registration cores --cores 0-7`
then `compare ... bw-slabs.pt ...-cores.pt` must be `72/72`.

### 2. Spill table and machine-code check

Build both an `-fstack-usage` bench at `MAX_M` default and at 8 from your worktree, count zmm stack accesses per
function, and compare with BASE's table. The script below is self-contained (replace `<name>`); it needs a base bench
built without the flag only for the access counts (`bench-base`, see "Benches" for how to build it).

```bash
#!/usr/bin/env bash
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
B=/data/models/slang/nvfp4-work/wt-<name>/python/sglang/kernels/jit/csrc/moe/expert_stream/bench
GB=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src
for max_m in "" 8; do
  build=$W/bench-<name>-su${max_m:+-m$max_m}
  taskset -c 0-63 cmake -S $B -B $build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch \
    -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=$GB \
    -DEXL3_MAX_M=$max_m -DCMAKE_CXX_FLAGS=-fstack-usage > /dev/null 2>&1
  taskset -c 0-63 cmake --build $build -j16 --target exl3_cpu_optimized 2>&1 | tail -1
done
for build in bench-base bench-<name>-su bench-<name>-su-m8; do
  echo "== $build: zmm stack accesses per function"
  objdump -d --no-show-raw-insn -C $W/$build/exl3_cpu_optimized | awk '
    /^[0-9a-f]+ <.*>:$/ { name = $0; keep = name ~ /register_band<|register_tiles|compact_pairs/; next }
    keep && /zmm/ && /\(%r[sb]p\)/ { count[name]++ }
    END { for (f in count) print count[f] "\t" f }' | sort -t$'\t' -k2 | sed -E 's/\(Exl3Projection.*//; s/sglang::exl3_cpu::\(anonymous namespace\):://; s/^([0-9]+)\t[0-9a-f]+ </\1\t</'
  echo "== $build: stack frames (bytes, symbol)"
  find $W/$build -name '*.su' -exec grep -hE 'register_band|register_tiles|compact_pairs' {} + | awk -F'\t' '{print $2, $1}' | cut -c1-140 | sort -u
done
```

Read it as: the access count per `register_band<P,0,0,true,M>` row; the ratio is the M's total accesses divided by
`4 * M * (sum of P instantiated)` (M = 2: sum P = 1+2 = 3 so denominator 24; M = 3 and 4: P = 1 so 12 and 16; BASE M = 1:
P = 1..3, denominator 24).

**One-token machine-code check (informational, not a gate).** The M = 1 `register_band` functions already differ from BASE
in register allocation (same spill counts); `traversal_kblock` and `traversal_group<2..4>` are byte-identical to BASE. To
re-check after your change, build the default benches of base and head (below) and run:

```bash
W=/data/models/slang/nvfp4-work/cpu-plan-m2
for side in base head; do
  rm -rf $W/disasm-$side; mkdir -p $W/disasm-$side
  objdump -d --no-show-raw-insn -C $W/bench-$side/exl3_cpu_optimized | sed -E 's/, 1>/>/g' | awk -v dir=$W/disasm-$side '
    /^[0-9a-f]+ <.*>:$/ { keep=0; if ($0 ~ /\(anonymous namespace\)::(traversal_kblock|traversal_group<[0-9]>|register_band<[0-9], 0, 0, (true|false)>)\(/) { n=$0; sub(/^[0-9a-f]+ </,"",n); gsub(/[^A-Za-z0-9_<>,]/,"_",n); n=substr(n,1,80); f=dir "/" n ".txt"; keep=1; next } }
    keep { print > f }'
  for f in $W/disasm-$side/*.txt; do
    sed -E -i 's/^ *[0-9a-f]+:[[:space:]]*//; s/ +#.*$//; s/\b[0-9a-f]{6,}\b/ADDR/g; s/0x[0-9a-f]+\(%rip\)/RIP/; s/<[^>]*>/<T>/g' $f
  done
done
for f in $W/disasm-base/*.txt; do b=$(basename $f); h=$W/disasm-head/$b
  if [ -f $h ]; then cmp -s $f $h && echo "IDENTICAL $b" || echo "DIFFERS   $b"; else echo "ABSENT-IN-HEAD $b"; fi; done
```
(`register_tiles` and `compact_pairs` are inlined into other symbols, so the plan's first awk pattern matched nothing; use
the symbols above.)

### 3. Benches

Build the default benches of the base and head (once per side; `side=base` with the BASE worktree at `b750c6d96f`, `side=head`
with yours):

```bash
ssh divix01 'bash -s' <<'EOF'
side=head; wt=/data/models/slang/nvfp4-work/wt-<name>     # then: side=base; wt=<worktree at b750c6d96f>
source /data/models/slang/nvfp4-work/cpu-plan-m2/env.sh
SITE=/data/models/slang/.venv/lib/python3.13/site-packages
taskset -c 0-63 cmake -S $wt/python/sglang/kernels/jit/csrc/moe/expert_stream/bench -B $W/bench-$side -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$GXX -DEXL3_TORCH_ROOT=$SITE/torch -DEXL3_CXX11_ABI=1 -DEXL3_TVM_FFI_ROOT=$SITE/tvm_ffi \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src > /dev/null
taskset -c 0-63 cmake --build $W/bench-$side -j16 --target exl3_cpu_optimized 2>&1 | tail -1
# the MAX_M = 8 timing build (no -fstack-usage): add -DEXL3_MAX_M=8 and use -B $W/bench-<name>-m8
EOF
```

Then the A/B runs (`ab.sh BASE_BUILD HEAD_BUILD NEW_RESULTS_DIR [flags]` alternates processes for `EXL3_BENCH_ROUNDS` rounds,
default 8, and writes `summary.tsv` with the median p50 of each side, head/base and which plan ran):

```bash
# Controls: MAX_M 4 build against your MAX_M 8 build (expect dsv41/dsv41, each head/base in [0.98, 1.02])
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-<name>/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-<name> cpu-plan-m2/bench-<name>-m8 \
  cpu-plan-m2/ab-<name>-m1-$(date +%Y%m%d-%H%M%S) --benchmark_filter=experts: 2>&1 | tail -5'
# Routed: the patterns where 8 lost (random) and won (shared4, 64:6:random), plus two noise controls
ssh divix01 'cd /data/models/slang/nvfp4-work && PYTHON=/data/models/slang/.venv/bin/python bash \
  wt-<name>/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/ab.sh cpu-plan-m2/bench-<name> cpu-plan-m2/bench-<name>-m8 \
  cpu-plan-m2/ab-<name>-routed-$(date +%Y%m%d-%H%M%S) \
  --routed=16:3:shared1,16:3:shared2,16:3:shared4,16:3:random,6:3:random,64:6:random --benchmark_filter=rows: 2>&1 | tail -8'
# For the record: your MAX_M 8 against the generic plan (BASE build)
#   same command with cpu-plan-m2/bench-base as the first build and --routed=16:3:shared4,16:3:random,64:6:random
```
Flags: `--routed=M:k:pattern[,...]` (pattern `shared<G>`: tokens in groups of G share all k slots, so chunks hold
min(G, `CHUNK_M`) tokens; or `random`), `--routed-slots=48`, `--routed-layers=2`. Each optimized benchmark reports
`dsv41_calls`/`generic_calls` counters and the context `chunk_m`.

## Pass criteria

All of these, measured in one session against the `MAX_M = 4` build of the same commit:

1. **Bit-exact:** every gate in "How to validate" passes at `MAX_M` 4 and 8 (72/72 x 3 tiers vs BASE at both, 24, 48, pytest).
   One bit of difference fails the change.
2. **Spill target:** no integer-accumulator or decode-temporary spill in the k-tile loop of `register_band<P,0,0,true,M>` for
   M = 3 and 4; as the plan's numeric gate, the ratio (accesses / (4 * M * sum P)) at most BASE's M = 1 ratio + 1, i.e.
   **<= 19.7** (BASE 18.7; today M=2: 28.9, M=3: 60.2, M=4: 66.5). If the metric cannot be met because fp32 partial sums are
   memory-backed by design, say so and replace it with a loop-level argument (objdump of the k-tile loop body showing no
   `zmm` `%rsp` accesses), agreed with the owner.
3. **Timings, `MAX_M` 8 build vs `MAX_M` 4 build:** no routed row above the noise floor (**head/base <= 1.012**, i.e. no row
   slower than +1.2%) on `16:3:shared1`, `16:3:shared2`, `16:3:shared4`, `16:3:random`, `6:3:random`, `64:6:random`; today
   `16:3:random` is 1.047 and `6:3:random` 1.021. At least one row below 0.95 (today `16:3:shared4` 0.664 and `64:6:random`
   0.815 already are).
4. **Controls:** `experts:1/3/5` of the `MAX_M` 8 build against the `MAX_M` 4 build within +/-2% (today 0.996, 0.994, 1.002;
   noise floor 1.012 for `experts:1`).
5. Noise floor (reference): same binary against itself gives `experts:1` 1.012 and `16:3:shared1` 1.007. A reading inside
   that band is not a regression or a win.

When all five hold, **raising the default `MAX_M` to 8 is the follow-up decision**, taken by the owner: change
`EXL3_MOE_CPU_MAX_M`'s default in `math.hpp:46-47` to 8, `DEFAULT_CHUNK_M` in
`test/manual/dsv41/test_exl3_cpu_dsv41_plan_multitoken.py` to 4, the README "Selection" text ("default 4" to "default 8"),
and re-run the full gate and Step 11 (core groups) at the new default. That is not part of this handoff.

## Pointers

- The plan and its ledger (rulings, every number above with its command): `docs/superpowers/plans/2026-10-06-dsv41-cpu-plan-multi-token.md`
  and `.superpowers/sdd/2026-10-06-dsv41-cpu-plan-multi-token/progress.md` in the `dsv41-cpu-plan-m2` worktree (the ledger is
  git-ignored scratch, so treat the plan's audit section as the durable record).
- Raw measurement files on divix01: `/data/models/slang/nvfp4-work/cpu-plan-m2/` (`t5-s14.txt`, `ab-*/summary.tsv`).
- Known repo facts: `test_exl3_ext.py` expects the optimized build's sources to be `[kernel.cpp, torch_ops.cpp]`; the
  A/B harness at `MAX_M = 8` exposed a hand-unrolled four-row AVX2 accumulate (now a template recursion), so look for the
  same shape in any new fixed-row code.
