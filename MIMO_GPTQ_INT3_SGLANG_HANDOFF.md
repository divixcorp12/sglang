# Handoff: try the experimental MiMo GPTQ INT3 checkpoint with our SGLang fork

Prepared 2026-10-30. Documentation only: no builds, tests, remote probes, quantization, or serving processes were run while preparing this handoff. Remote paths and export metadata below were last checked on 2026-10-29; refresh them before execution. Historical validation is explicitly separated from work still required.

## Goal and current boundary

Try the existing experimental MiMo REAP288 W3 checkpoint through our SGLang fork, preserving its exact quantized weights. Begin with the implemented C++ CPU expert backend and establish actual runtime integration before attempting a bounded full-model generation smoke.

The checkpoint and the native CPU kernel exist. **A complete SGLang loader/serving integration has not been established.** There is no verified one-line server launch recipe for this checkpoint. Do not treat the model directory as a stock GPTQ model or invent a `--quantization gptq_int3` option.

Work directly in Codex, not through hx. This handoff requests a subsequent experimental tryout, not a production deployment, a new quantization run, or a rewrite of existing numerical policy.

> 🧠 **From Hindsight memory (GPTQ INT3 CPU expert engine using BesTLA primitives)** — The approved design preserves checkpoint codes, signed FP16 scales and FP16 dequantization rounding, uses floating-point activations, and reuses the typed CPU kernel/JobWindow infrastructure. The current repository README confirms those boundaries; memory alone is not serving qualification.

## Repository and commits

Local fork:

```text
/Users/dnikolaidis/Desktop/divix/sglang-nvfp4
```

At handoff preparation, this checkout is on `master`, HEAD `162392ac84e2d4fda5f01d9989f1eba8f32ebfe8`, with a clean working tree. Its ancestry includes:

| Commit | Meaning |
|---|---|
| `4f2768e385` | INT3 feature commit on the isolated master-based feature branch |
| `f649876181` | Merge of the INT3 feature into local master |
| `4d58112bce` | Original equivalent feature commit on the earlier `job-state-slim` branch |

Check branch, dirty state, ancestry, and remote refs again. Do not assume these commits were pushed or that the remote serving checkout contains them. Preserve any new unrelated edits. Use a separate experimental branch/worktree for integration changes; do not reset a shared checkout.

The full SGLang checkout and serving virtualenv on divix01 must be located and verified in the next session. The isolated CPU validation directories listed below are **not** complete SGLang installations.

## Model and experiment paths on divix01

Experimental output, to read without modifying:

```text
/mnt/nvme2/mimo-autoround-candidates/mimo-v26-pro-reap288-w3-rtn-experimental-v1
```

Protected original source:

```text
/data/models/mimo-reap-candidates/mimo-v26-pro-reap288-v1
```

Quantization workspace, historical runbooks, scripts, receipts and logs:

```text
/mnt/nvme4/mimo-reap/autoround-reap288-w3-v1
```

Read the latest `RUNBOOK` and the export completion/verification receipts there before relying on historical state. Old quantization, capture, scoring and diagnostic runs are terminal; do not restart them to try serving.

Important files under the experimental model root:

```text
export-manifest.json
export-seal.json
plan.json
state.json
model.safetensors.index.json
config.json
portable_gptq.py
experts/layer-01/expert-000/weights.safetensors
experts/layer-01/expert-000/commit.json
nonexpert/
provenance/
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

The expert path pattern is `experts/layer-{layer:02d}/expert-{expert:03d}/weights.safetensors`, layers 1 through 69, experts 0 through 287. Each payload contains gate/up/down projections with fully qualified model tensor names.

Trusted manifest SHA256 from the verified export:

```text
7b1445ed9cd90d94cfc98e3cde385c618d4a133dc8adcd0cf031f890aa6b5c65
```

Original export-plan SHA256:

```text
47fded1787e6de67d4adb270d4863367128eff51e68d97efff702b2dcc036416
```

Verify hashes against these trusted receipts, rather than accepting a rewritten adjacent seal. The CPU reader requires an explicit trusted manifest hash and checks payload hash/size, exact tensor ownership, policy and path confinement.

## Exact quantization contract

| Property | Existing artifact |
|---|---|
| Export format | `mimo-hybrid-gptq-v1` |
| Expert policy | GPTQ INT3, symmetric, group size 128, `desc_act=False` |
| Producer | AutoRound optimized RTN, `iters=0`, `enable_neuqi=False`, seed 42 |
| Scales | FP16, finite and nonzero; negative scales are valid |
| Groups | Contiguous `g_idx = arange(K)//128`, no act-order permutation |
| Packed zeros | Unpacked value 3; GPTQ offset +1 gives symmetric zero 4 |
| Expert coverage | 69 layers × 288 experts × 3 projections = 59,616 projections |
| Nonexpert coverage | 1,064 original tensors, raw storage unchanged |
| Tensor payload | 330,957,148,032 bytes: 330.96 GB / approximately 308.23 GiB, excluding file/metadata overhead |

For a projection with weight shape `[N,K]`, canonical storage is `qweight: int32[K/32*3,N]`, `qzeros: int32[K/128,N/32*3]`, `scales: FP16[K/128,N]`, and `g_idx: int32[K]`. INT3 codes cross word boundaries at positions 10 and 21 within each group of 32 codes.

The runtime adapter rearranges codes into BesTLA low2/high1 planes without requantization. Each expert has six slabs: gate/up low2, gate/up high1, gate/up FP16 scales, then down low2, down high1, down FP16 scales. Gate rows precede up rows. Keep scale bit patterns, including their signs.

The CPU backend rounds each dequantized weight to FP16 **before** its FP32 dot accumulation. It retains explicit FP16 gate/up, clamp, SiLU/product and down boundaries, then performs ordered FP32 route-weight reduction. It does not quantize activations to INT8. SIMD/scalar/GPU reduction outputs are not claimed bit-exact.

Dense, attention, router, normalization, embedding and head tensors retain their original storage, including FP8/BF16 where present. Inspect the actual configuration/index before selecting loaders: a global FP8 or GPTQ config must not accidentally send every tensor through the wrong quantization path.

## Existing implementation map

All repository-relative paths below are under the local fork above; locate their matching revision in the remote experimental checkout.

| Path | Responsibility |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/gptq_int3/README.md` | Detailed numerical contract, usage, provenance and validation |
| `python/sglang/kernels/jit/csrc/moe/gptq_int3/kernel.cpp` | Native expert arithmetic, validation, job-owned scratch and job phases |
| `python/sglang/kernels/jit/csrc/moe/gptq_int3/entry.h` | Concrete `GptqInt3CpuKernel` interface |
| `python/sglang/kernels/jit/csrc/moe/gptq_int3/bestla_int3.h` | Pinned/adapted scalar, AVX2 and AVX512 bit-plane primitives |
| `python/sglang/kernels/jit/csrc/moe/gptq_int3/half.h` | Explicit IEEE FP16 boundaries |
| `python/sglang/kernels/jit/csrc/moe/formats/gptq_int3/` | Native `F::CpuKernel` binding and row layout |
| `python/sglang/srt/layers/moe/cpu_experts/gptq_int3.py` | `load_exported_expert`, lossless adapters, `GptqInt3CpuTrait` |
| `python/sglang/srt/layers/moe/expert_engine/formats/gptq_int3.py` | Explicit CPU-only format and capability declarations |
| `python/sglang/srt/layers/moe/expert_engine/formats/__init__.py` | Lazy format registry |
| `python/sglang/kernels/ops/moe/cpu_kernel_build.py` | CPU source/compiler flag registration |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | `GPTQ_INT3_BUILD = ("gptq_int3", "sigmoid")` and host layer registration |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/` | Existing typed kernel, Team, JobWindow, stage-gate and completion contracts |
| `python/sglang/srt/layers/moe/cpu_experts/service.py` | Existing CPU service integration seam |
| `test/registered/kernels/ops/moe/test_gptq_int3_cpu.py` | CPU feature suite and optional real-expert verification |
| `test/registered/kernels/ops/moe/gptq_int3_cpu_bridge.cpp` | Test-only bridge into the real typed kernel/scheduler; not a production Python API |

BesTLA unpack primitives are adapted from `intel/neural-speed` commit `bfd5d3c17dee18a20e2768f855f2d8fe132fc579`; Apache licensing is preserved in `LICENSE.bestla`. No upstream scheduler or activation-quantized GEMV path is imported.

The backend reuses Team/JobWindow, preserves route order and output dependencies, and reads down slabs only after their stage gate. It uses bounded job-owned scratch rather than a full-layer decoded FP16 cache. Keep `CpuExpertLayerSpec.keep` and slab owners alive until all readers/leases finish.

## What is proven, and what is not

Historical validation on divix01 reported 74 feature cases passing, including scalar/AVX2/AVX512 builds, exact code/scale adaptation, malformed inputs, manifest/path/owner rejection, transactional output errors, protected staged-down buffers, four concurrent JobWindow jobs, and Clang ASan/UBSan. Real exported experts `(1,0)`, `(32,17)` and `(69,0)` were compared with an independent canonical decoder. The maximum reported relative L2 error was `9.20486e-5`, below that test's `.002` bound. The master-based feature tree also passed 74 cases before merge. These results have not been rerun for this handoff.

The reported AVX512 component timings were approximately 3.3–3.6 ms for two synthetic token rows, eight unpinned workers and one expert, including test Team construction/destruction. These numbers do not predict full-model latency or throughput.

Remaining gaps are material:

1. The Python format advertises `supports_cpu_experts=True` but `engine_served=False`; GPU apply/streamer and draft paths are deliberately unavailable. `get_format("gptq_int3")` is explicit opt-in; generic GPTQ config resolution does not select it.
2. A bounded expert reader and typed CPU binding are not a full model-loader adapter. Validate tensor dispatch, mixed nonexpert formats, layer construction and actual calls from the SGLang model path.
3. The CPU bridge tests do not prove the complete production host-module FFI, lease/transport lifecycle, model routing, attention/KV or end-to-end generation.
4. No GPU expert serving backend has been integrated for this checkpoint. If using CPU experts initially, prove that the actual routing path can send every routed expert to CPU; do not assume a split knob bypasses an unavailable GPU apply method.
5. Whole-model quality remains unverified. The export flags `serving_loader_validated`, `native_precision_verified` and `objective_quality_verified` were false at the last inspection. Do not flip them to bypass startup checks.

The earliest useful success is a real fork model/host call using the selected exported expert and the real CPU service/FFI path. A later success is a bounded full-model deterministic generation smoke. Report these as different milestones.

## Host, environments and capacity

Execution host: `divix01`, x86-64 Linux. Historical CPU validation used two Xeon Platinum 8280 CPUs (112 hardware threads), GCC 15.2 and Clang 21 sanitizer builds. Refresh actual CPU ISA, toolchain and resources before use. AVX512BF16/FP16 hardware is not required by this backend.

Existing separate environments:

```text
/mnt/nvme4/mimo-reap/autoround-reap288-w3-v1/.venv
    AutoRound export environment; historically Transformers 5.5

/mnt/nvme4/mimo-reap/.venv-compressor
    Reference evaluation environment; historically Transformers 5.18
```

Do not mutate these environments, vendor code, or model files to install SGLang. Identify a compatible fork runtime or create an isolated experimental environment after inspecting requirements. Record Python/Torch/CUDA/Transformers/FFI/compiler versions. Keep `HF_HUB_OFFLINE=1` for local artifact use.

Historical numerical compiler:

```text
/opt/rh/gcc-toolset-15/root/usr/bin/g++
```

GCC sanitizer runtime linking was incomplete on that host; `GPTQ_INT3_SANITIZER_CXX=clang++` was used successfully without changing numerical `CXX`.

Historical isolated test directories:

```text
/mnt/nvme4/mimo-reap/gptq-int3-cpu-dev-v1
/mnt/nvme4/mimo-reap/gptq-int3-cpu-master-validation-v1
```

These were partial CPU test checkouts, not full serving environments. Verify or recreate an exact-revision isolated test tree rather than treating them as installed SGLang.

Before loading the model, inventory RAM per NUMA node, GPU free/total memory, KV/workspace budgets and disk. Quantized expert storage alone is hundreds of GB; do not decode all experts to FP16 for residency. Unchanged nonexpert tensors account for **34,595,360,640 bytes (about 32.22 GiB)** before runtime overhead: do not assume they all fit on a single 32 GiB GPU, especially with KV cache and concurrent jobs. A workable placement/offload plan is a prerequisite, not a reason to silently change tensor precision.

Honor the current GPU hold and shared benchmark lock:

```text
/mnt/nvme4/mimo-reap/performance-v1/GPU_HOLD.json
/mnt/nvme4/mimo-reap/performance-v1/benchmark.lock
```

GPU sharing with tests was explicitly allowed previously; mere presence of another context is not a blocker. Check actual capacity and preserve the existing 6 GiB runtime GPU reserve unless the user explicitly changes that policy. Never terminate unrelated jobs. Do not infer permission to stop a current production server from an old experiment's server-stop authorization; inspect current ownership and scope first.

## Suggested next-session sequence

1. **Preflight without starting a model.** Read this handoff, the INT3 README, latest experiment runbook/receipts and current repository instructions. Refresh Git/remote state, model seals, process ownership, holds, installed versions and capacity. Follow the required LSP workflows before code inspection/edits. Record every fact as current, historical or unverified.
2. **Reconfirm bounded CPU correctness in an isolated checkout/environment.** Use the command below, with the actual verified checkout as the working directory. Preserve the `.002` test bound and all exact storage/provenance checks. This is a prerequisite, not serving qualification.
3. **Trace the fork's real integration seams.** Inspect the MiMo model profile (`mimo_v26`), sigmoid router, hybrid tensor loading, quantization dispatch, CPU trait registration, host FFI and forward dispatch. Produce an exact list of missing paths. Do not simply set `engine_served=True` or route the hybrid checkpoint through a global stock GPTQ loader.
4. **Implement the smallest explicit experimental integration if needed.** Keep it on a separate branch, preserving the artifact and current CPU numerical policy. Prefer an initial CPU-expert route if feasible with the existing engine; validate the real host/service path with selected expert(s) and fixed activations before attempting the whole model. If an absent GPU apply path is structurally required, address that explicitly rather than falling back silently or disguising the standalone bridge as serving.
5. **Validate real model routing and placement.** Check tensor names/shapes/dtypes, gate/up/down ownership, expert IDs and slot maps, duplicate/skipped routes, routing weights, SwiGLU clamp, stage-two readiness, output accumulation and lifetimes. Distinguish lossless storage parity from arithmetic tolerance. Compare intermediate outputs with the preserved reference at meaningful boundaries.
6. **Only after compatibility and capacity are demonstrated, run a bounded full-model smoke.** Use an isolated process, logs and free port. Start with one short fixed prompt, greedy decoding and a small output-token limit; initially avoid speculative decoding and graph capture. Verify current CLI flags from the fork and record the exact command/environment. No launch command is supplied here because none has been validated for this hybrid format.
7. **Report measured scope honestly.** Save commit/checkout/env/model pins, loader decisions, placement, log paths, resource peaks, outputs and comparison results. Keep cold preparation separate from warm expert execution and end-to-end decode. Only expand into existing fixed-cohort quality evaluation once the evaluator and full path are validated; no new calibration, held-out fitting or data selection.

Candidate CPU-only verification command for the next session, after confirming environment and checkout:

```sh
HF_HUB_OFFLINE=1 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
GPTQ_INT3_SANITIZER_CXX=clang++ \
GPTQ_INT3_REAL_ROOT=/mnt/nvme2/mimo-autoround-candidates/mimo-v26-pro-reap288-w3-rtn-experimental-v1 \
python -m pytest -qs test/registered/kernels/ops/moe/test_gptq_int3_cpu.py
```

Use the verified environment's Python and documented `PYTHONPATH`/installation setup. Do not run the reference interpreter against a guessed fork path. This command was **not run while writing this handoff**.

Preserve the original 128 calibration / 32 held-out lineage, `capture-inputs-v1/windows.json`, its seal, imatrix, prior failures and diagnostics. Imatrix is not REAP saliency. The earlier BF16 Humming `.002` failure and exact-FP16 calibration-input failure remain separate preserved evidence; FP16 deployment is not a waiver. Original-source decoded-path controls and bounded quantization diagnostics do not certify native serving or task quality.

## Ready-to-paste continuation prompt

> Continue the experimental MiMo GPTQ INT3 tryout with our SGLang fork. First read `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/MIMO_GPTQ_INT3_SGLANG_HANDOFF.md` and the INT3 kernel README. Work natively in Codex, not hx. The completed export is on divix01 at `/mnt/nvme2/mimo-autoround-candidates/mimo-v26-pro-reap288-w3-rtn-experimental-v1`; preserve its exact codes, signed FP16 scales, nonexpert tensors and manifest. Feature commit `4f2768e385` was merged into local master as `f649876181`; refresh actual checkout/remote state. No model/test process was launched while preparing the handoff. Recheck processes, holds, capacities and environments, then revalidate the bounded CPU backend and trace the real MiMo loader/host/forward path. Implement only the necessary explicit experimental integration on an isolated branch, establish actual CPU service/FFI and routing correctness, and attempt a short isolated full-model generation smoke only after compatibility and placement are demonstrated. The current `engine_served=False` and unavailable GPU apply are real integration gaps, not flags to bypass. Do not requantize, restart completed experiments, change precision/data/gates, terminate unrelated jobs or deploy to production. Record exact commands, pins, logs, resource peaks and comparison evidence, distinguishing CPU component success from full-model serving and quality.
