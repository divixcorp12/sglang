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

### E0 — baseline suite (measured, green)

- Command: `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 32-63 python -m pytest test/registered/unit/kernels -q -p no:randomly`.
- Run in divix01 `wt-compute-gap` at `8d599afe9c`, under `cc-gpu.lock`.
- Result: **1748 passed, 1 skipped, EXIT=0**. The exit status was read directly, not through a pipe.

### E4 — C1 shared-expert side stream with CE (measured: passes the promotion bar)

Setup:
- `analysis/dsv41-compute-gap/drive_ab.sh c1-sidestream 8d599afe9c "SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=0" "...=1"`.
- The production recipe unchanged, with `DSV41_RUN_ROOT=/mnt/nvme1/compute-transfer-gap`.
- Arm order A B B A A B B A, with session pairs 0,1 / 2,3 / 4,5 / 6,7.
- Run 2026-09-27 04:36–05:19. Every arm exited 0.
- Outputs are in `ab/c1-sidestream/` (`paired.txt`, `compare.json`).

| | A (off) | B (on) |
|---|---:|---:|
| decode ms/token, pooled over 486 tokens | 107.40 | 106.20 |
| median per-session tok/s | 9.221 | 9.312 |
| NVMe bytes, ready→end | 300.43 GB | 300.27 GB |
| RAM-miss rows read (4 runs) | 17,631 | 17,702 |

- **B wins 8/8 sessions, sign test p = 0.0039.**
- The median per-session change is **−1.16 ms/token (−1.1%)**, ranging from −0.79 to −1.50.
- TTFT is unchanged (7.7–8.7 s in both arms).
- Every completion (content and reasoning, temperature 0) is **byte-identical** between arms.
- Every B server logged `MoE side stream enabled` and a first fork of `DeepseekV2MoE._forward_shared_experts`. No A server logged either, so the path really forks.
- The copy engine was armed in all 8 arms. The combination of side stream and CE wait kernel is therefore exercised in serving.
- Read errors, overruns and fatal counters were 0 everywhere, and there were no compile events.
- `test/manual/dsv41/test_moe_side_stream_gpu.py` passed 4/4 (EXIT=0).
- paired.py's clock, tenancy and compile gates all passed.
- Attribution comes from a node-mode traced pair.
  - Setup: `drive_traced.sh c1t4`, one session (index 4) per arm, analysed with `trace_chain.py --skip 5`, 28 replays
    per arm. The results are in `results/c1-traced/`.
  - Validation: on the handoff's own trace the script reproduces F→nextPost 13.39 ms/step, against the handoff's 13.46.
  - Node-mode times are inflated, so only the A-vs-B differences below mean anything.

  | per step (mean) | A (off) | B (on) | Δ |
  |---|---:|---:|---:|
  | F-end → next Post-start, summed over 39 transitions | 13.45 ms | 12.21 ms | **−1.24 ms** |
  | graph span | 120.7 ms | 118.8 ms | −1.9 ms |
  | CE bytes / copies | 1032.8 MB / 155.6 | 1030.9 MB / 155.4 | same |
  | CE interval union | 83.1 ms | 81.9 ms | −1.2 ms |
  | Post-start → first CE copy (per layer) | 45.5 µs | 49.7 µs | +4.2 µs |

  - The mechanism is the one the handoff hypothesised. Moving the shared expert off the main stream shortens the
    dependent chain from one layer's transfer boundary to the next layer's request by about 1.2 ms/step, at
    unchanged transfer bytes. That matches the untraced −1.16 ms/token.
  - The cost is a small Post→first-copy slowdown, about 0.17 ms over 40 layers, probably contention with the
    side-stream GEMVs. The net win survives it.
  - A first traced attempt (`c1t`, session 0, 7 tokens) gave only 2 replays and pointed the same way
    (16.28 → 12.12 ms).
- Recommendation: promote `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM=1` into `arm_env.base_env()`. This is a production recipe change, left for the owner to apply.

### E1 — C2a wo_a (measured, no-go)

`results/micro-1/wo_a.log`. The run uses 40 distinct 64 MiB layers, timed by graph replay over two interleaved rounds that agree to 0.01 µs.

| Candidate | µs/call | TB/s |
|---|---:|---:|
| einsum (production fallback) | 42.43 | 1.58 |
| `wo_a_bf16_gemv` (BN=1, 4 warps) | 40.74 | 1.65 |
| best BN/warp variant | 40.74 | 1.65 |

- The saving is 1.7 µs/call, about 0.07 ms/token over 40 layers. That is below the 5 µs gate.
- Both kernels already run at the DRAM floor.
- The GEMV is not bitwise equal to the einsum: 0.27% of elements differ by one BF16 ulp.
- Decision: do not integrate.

### E2 — C2b/R1 dense INT8 SQ GEMV stage depth (measured, no-go)

`results/micro-1/gemv-*.{json,log}`. Setup:

- Real 5-bpw weights from all 40 layers, one CUDA graph per kind, 400 round-robin replays.
- `prod` is the production build. `knobs` is the patched source with no defines. Every variant was compared bitwise against `prod` on fixed inputs.
- The `knobs` build equals `prod` in both bits and time, so the patch alone is inert.
- Repeat runs of `prod` and `knobs` agree within 0.03 µs.

| µs/call | wq_a | wq_b | wkv | wo_b | sh_w1 | sh_w3 | sh_w2 | bitwise |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| prod (D=4, staged 3/5/7) | 8.56 | 21.05 | 7.84 | 21.99 | 10.46 | 10.45 | 9.52 | — |
| D=2 | 11.96 | 21.31 | 11.81 | 23.22 | 12.64 | 12.63 | 11.17 | yes |
| D=3 | 9.15 | 20.63 | 8.61 | 21.82 | 10.58 | 10.57 | 9.53 | yes |
| D=6 | 8.43 | 21.54 | 7.71 | 22.42 | 11.15 | 11.17 | 10.21 | yes |
| no staging | 9.57 | 20.17 | 9.12 | 21.16 | 10.63 | 10.65 | 9.59 | yes |

- The best per-shape choice (no staging for wq_b/wo_b, D=6 for wq_a/wkv) saves about 2.0 µs per layer, about 0.08 ms/token.
- On wq_b + wo_b together that is 4.0%, below the 10% gate.
- These two kernels run at 1.19–1.25 TB/s against the 1.65 TB/s that E1 shows is reachable. Stage depth does not close that gap.
- The decomposition changes that might close it alter per-slice activation quantization, which is a numerical change and outside this pass.
- Decision: stop R1 at stage depth.

### E3 — C3a/R2 routed MoE (measured; one bitwise-identical win)

`results/micro-1/moe-*`. Setup:

- The production `Exl3FusedMoE.run` over 40 layers of real rows, about 80 MB touched per layer.
- 300 graph replays.
- Each configuration ran in its own process.

| Configuration | µs/layer | vs prod | bitwise |
|---|---:|---:|---|
| prod (6 groups × 28, N=256, SH=3, FS=3) | 101.85 | — | — |
| width 24 / 16 / 8 | 110.5 / 144.5 / 254.5 | slower | no (rel ≤5.7e-3) |
| `EXL3_MOE_TILE_N=128` | 133.1 | slower | no |
| `MOE_SH_STAGES=2` | 148.6 | slower | yes |
| **`MOE_SH_STAGES=4`** | **95.88** | **−5.97 (−5.9%)** | **yes** |
| `MOE_FRAG_STAGES=2` | 102.3 | +0.5 | yes |

- A fourth shared-memory pipeline stage gives the same bits and saves 6.0 µs per layer, about 0.24 ms/token.
- Its p10–p90 spread is 0.5 µs.
- It misses the handoff's 10 µs screening gate, but it is exact and needs no code change beyond one constant.
- Follow-up sweep (`results/micro-2`):

  | Configuration | µs/layer | bitwise |
  |---|---:|---|
  | prod | 101.87 (repeat 101.83) | — |
  | SH=4 | **95.85 (repeat 95.90)** | yes |
  | SH=5 | 98.90 | yes |
  | SH=6 | 98.90 | yes |
  | SH=4 FS=4 | 100.67 | yes |
  | SH=4 FS=2 | 99.75 | yes |

  SH=4 is the best depth, reproducibly.
- `MOE_SH_STAGES` feeds only `exl3_moe_kernel`. That covers the decode tile and the 32/64-row prefill tiles. The dense
  GEMM/GEMV paths do not use it.
- A serving A/B (`ab/c1moe-sh4`) is queued: A = side stream on, B = side stream on + a private SH=4 exllamav3 tree
  and build dir passed through `SGLANG_EXL3_SRC`/`SGLANG_EXL3_BUILD_DIR`. Its provenance is recorded in
  `exl3-serve-sh4.provenance`. Production's extension and pin are untouched.
- Integration is blocked on a decision, not on evidence. The constant lives in the pinned upstream exllamav3 (`turboderp-org/exllamav3`, no fork remote), so production would need either a fork commit with a pin bump, or sglang carrying the patch and applying it at build time.

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
