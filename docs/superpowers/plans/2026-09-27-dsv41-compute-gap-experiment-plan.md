# DSV4.1 compute-to-transfer gap: experiment plan and log

Executes `2026-09-26-dsv41-compute-transfer-gap-experiments-handoff.md` (the handoff). Section numbers
C0–C5 / R1–R6 refer to it.

## Baseline

- Source: laptop `master` at `91173977c2` (origin/master `c377185c16` + the expert-stream transport
  split merge). Branch `dsv41-compute-gap`, laptop worktree `sglang-nvfp4-worktrees/compute-gap`,
  divix01 worktree `/data/models/slang/nvfp4-work/wt-compute-gap`.
- Recipe: `benchmarks/dsv41_baseline/arm_env.py` at that commit, unchanged (27.17: hot cache
  16100 MB, static fraction 0.90, 4096-token prefill chunks, context 262144, HiCache ratio 2).
  **This is not the recipe the handoff's trace was taken with** (17,408 MiB / 0.925). Trace-derived
  budgets are therefore context, and every comparison below is matched within this recipe.
- External EXL3: divix01 `/data/models/slang/nvfp4-work/exllamav3` at pinned `02aef45cd6`, clean.
- Run root: `DSV41_RUN_ROOT=/mnt/nvme1/compute-transfer-gap`.

## Byte-rate framing, from the checkpoint's `tensor_storage` (all attention/shared linears are 5 bpw)

| Tensor | Stored bytes/layer | Trace µs/call | Implied rate |
|---|---:|---:|---:|
| wq_b 1280→32768 | 26.2 MB | 22.8 | ~1.15 TB/s |
| wo_b 8192→5120 | 26.2 MB | 24.3 | ~1.08 TB/s |
| wo_a BF16 (dequantized at load) [8,1024,4096] | 64 MiB | 43.6 | ~1.54 TB/s |
| shared w1+w3+w2 | 22.1 MB | ~32 (3 calls) | ~0.69 TB/s |

RTX 5090 peak DRAM is ~1.79 TB/s. wo_a is already close to the floor (a ≥5 µs/call gain needs
≥1.72 TB/s effective), so C2a is expected to be a quick no-go; the dense INT8 GEMVs have the most
measured data-movement headroom. Rates are sanity checks, not predictions.

## Experiments, in execution order

| ID | What | Kind | Gate to continue |
|---|---|---|---|
| E0 | Registered suite `test/registered/unit/kernels` at the baseline worktree | GPU test | Green, or failures identical to known environmental ones |
| E1 | C2a: einsum vs `wo_a_bf16_gemv` (BN=1) and BN∈{2,4} variants, 40 rotating layers, CUDA-graph replay | microbench | ≥5 µs/call stable win and numerical report, else no-go |
| E2 | C2b+R1: dense EXL3 INT8 SQ GEMV at wq_b / wo_b / wq_a / wkv / shared shapes with real layer weights, 40 rotating layers, graph replay; private builds varying `GEMV_STAGE_D` ∈ {2,3,4} and staged vs direct extraction | microbench (private exl3 builds) | ≥10% aggregate wq_b+wo_b saving, bit-identical outputs (stage depth must not change arithmetic) |
| E3 | C3a+R2: routed MoE group width {8,16,28} and `EXL3_MOE_TILE_N` 128 vs 256, 40 layers of real rows (`split_launch_bench.py` pattern) | microbench (private exl3 build) | ≥10 µs/call stable win; parity per the probe's tolerances and a bitwise-change report |
| E4 | C1: `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM` 0 vs 1, unprofiled A B B A A B B A, one session pair per A/B pair (0,1 / 2,3 / 4,5 / 6,7) | serving A/B | See acceptance; keep only with a repeatable net win and byte-identical outputs |
| E5 | Serving A/B for any E1–E3 candidate that passes its gate, integrated behind an opt-in flag | serving A/B | Same as E4 |
| E6 | C0b: identify graph node `8589934607` (the 2.047 ms prefix hole) from graph topology, no production change | diagnostic | Report only |

Deferred, stated not run: C3b static scheduling (needs E3 evidence of control overhead), C2c output
cast fusions (~0.1 ms each historically), R3/R4 (need R1/R2 stall evidence), R5 low precision
(numerical track), and one new candidate the byte table surfaces: wo_a is stored as 21 MB of 5-bpw
EXL3 but held as 64 MiB of BF16; running it from EXL3 would free ~1.7 GiB VRAM for expert cache at a
compute and numerical cost. Record it; do not start it here.

## Method rules

- GPU work only under `cc-gpu.lock` (blocking `flock`, never broken); CPU work under
  `taskset -c 0-63`, cores 64–71 untouched. Arm drivers take `rowimg-disk.lock` first, then poll the
  GPU lock (run_arm.sh takes it non-blocking itself). Production is not started or stopped by this work.
- Private exllamav3 builds: a separate detached worktree of the divix01 exllamav3 clone per variant
  at the pinned commit, a patch committed in this repo under `analysis/dsv41-compute-gap/exl3-patches/`,
  loaded by the benchmark through `torch.utils.cpp_extension.load` with a distinct module name and
  build dir, recording pinned HEAD + patch sha256. The production extension and its build dir are never
  touched. Builds run on CPU only, never while a serving arm is timing.
- Microbenchmarks: working set past L2 (40 layers of weights), timed as CUDA-graph replays, median of
  repeated trials, sanity-checked against DRAM peak; bitwise comparison against the stock kernel on
  the same inputs.
- Serving arms: `run_arm.sh` with `EXPECT_SHA`, `DSV41_SESSION_INDICES`, per-arm `KEY=VAL`; merged
  with `concat_arms.py`, compared with `paired.py`. Read every pytest / arm status from the command
  itself (`PIPESTATUS`), never through a pipe.
- Acceptance for promotion (handoff §6): repeatable ≥1 ms/token unprofiled improvement or a gain
  clearly above measured arm-to-arm noise, byte-identical completions (greedy), comparable NVMe bytes,
  no compile contamination. Otherwise no-go, recorded.

## Results log

### E6 / C0b — the 2.047 ms graph-prefix hole is node-mode launch overhead (measured, closed)

Evidence: `analysis/dsv41-compute-gap/c0b_prefix_launch.txt`, from the handoff's own trace and 90 windows.

- In every replay the hole sits between nodes `8589934606` and `8589934608`. Its median is 2041.8 µs.
- `cudaGraphLaunch` returns a median 2068.7 µs after the graph's first kernel starts. Across replays this
  interval and the hole correlate at r = 0.9995.
- The first kernel after the hole starts a median 8.3 µs *before* `cudaGraphLaunch` returns (range −15.3 to −5.4 µs).
  The GPU runs the first 14 nodes, then waits for the host to finish submitting the rest of the graph.
- The graph has 2,795 nodes. 2.04 ms / 2,795 is about 0.73 µs per node, the per-node cost that node-mode tracing
  adds to `cudaGraphLaunch` (CLAUDE.md measured 0.77 µs).
- Graph-node creation records predate the captured API window, so they cannot give node 15's type. The type does
  not matter for the conclusion: the hole's end is set by host submission, not by that node.
- Decision: exclude the hole from every optimization budget. Traced step spans from node-mode reports carry about
  2 ms of this artifact. Leave the external event record and its write-after-read contract unchanged.
