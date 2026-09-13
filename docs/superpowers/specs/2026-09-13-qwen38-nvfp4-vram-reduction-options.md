# Qwen3.8-Flash-Next-NVFP4 on one RTX 5090: VRAM reduction options

Status: vision-tower removal adopted 2026-09-13; everything else is a candidate, not implemented.

Scope: the bs=1 expert-offload serving setup (`run-nvfp4-expert-dynamic-hot10g.sh`,
checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54`). Every GiB freed here is worth about
390 extra hot-cache slots (2.64 MiB per NVFP4 expert row), or headroom for the NEXTN
speculative-decoding experiment.

## Where the GPU memory goes today

Source: production log `nvfp4-expert-dynamic-hot10g-20260913-141752.log` plus safetensors header sizes.

| Component | GiB | Notes |
|---|---|---|
| CUDA context | ~1.3 | `avail mem` 30.53 of 31.84 GiB before weights load |
| Non-expert weights | 10.89 | `Load weight end ... mem usage` — broken down below |
| Expert hot cache | 8.76 | 3,403 slots |
| Graph-gather scratch | 1.24 | 10 rows per layer |
| KV cache (12 full-attention layers, 8,192 tokens) | 0.18 | BF16 |
| Linear-attention state (36 layers, 1 request) | 0.11 | |
| Free after graph capture | 9.17 | consumed by prefill activations for 4,096-token chunks |

Non-expert weights on the GPU (checkpoint tensor bytes):

| Module group | GiB | dtype |
|---|---|---|
| Linear attention (GatedDeltaNet), 36 layers | 4.15 | BF16 |
| `embed_tokens` | 1.18 | BF16 |
| `lm_head` | 1.18 | BF16 |
| Hyper-connection `input_mix_weight_{up,down}` | 1.18 | BF16 |
| Full attention, 12 layers | 1.15 | BF16 |
| Vision tower | 0.57 | BF16 |
| Shared experts | 0.44 | BF16 |
| Routers, PLE key/value projections, block-inject | 0.19 | BF16 |
| Unattributed (staging buffers, rope caches, workspace) | ~0.8 | |

Already off the GPU: routed experts (63.3 GiB, host arena) and PLE n-gram tables (47.7 GiB FP8, file
backend). `--ple-offload-embedding` moves only the PLE n-gram tables; `embed_tokens` stays on the GPU.

## Options

### 1. Skip the vision tower — ADOPTED

- Change: launch flag `--language-model-only`. The tower is never built and its weights never loaded.
- Requires `Qwen4ExpForConditionalGeneration` in `ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES`
  (`server_args.py`). Without it `handle_language_model_only` raises during startup, and the launcher
  kills the server with no traceback in the log, only `kill_process_tree called` right after the
  attention-backend line.
- Saves: 0.57 GiB.
- Cost: image and video requests are rejected.
- Verify: the startup log has no multimodal attention backend line, and `Load weight end ... mem usage`
  drops by about 0.57 GiB from 10.89.

### 2. Serve `embed_tokens` from host memory — lossless

- Change: put `embed_tokens` on the host and stage the looked-up rows the way PLE rows are staged
  (`Qwen4ExpPinnedHostEmbedding` / the uring row stager).
- Saves: 1.18 GiB.
- Cost: one 2,560-wide BF16 row per decode token (5 KiB); at most 4,096 rows (~40 MiB) per prefill chunk.
  In the captured decode graph the lookup must happen before replay, exactly like PLE staging.
- Done when: a decode step shows no new host sync in the profiler, and outputs match token-for-token at
  temperature 0 against the current build.

### 3. FP8 weights for the BF16 linear layers — changes weights, needs a quality check

- Candidates: linear attention (4.15), full attention (1.15), hyper-connection mix projections (1.18),
  shared experts (0.44). Together 6.9 GiB in BF16.
- Saves: about 3.5 GiB with per-channel FP8 E4M3 weights.
- Why it is not free: NVIDIA's MIXED_PRECISION recipe deliberately kept these layers in BF16, so there is
  no published accuracy evidence for quantizing them.
- Work: quantize selected BF16 `nn.Linear` modules at load time (per-output-channel weight scales), on top
  of the existing `modelopt_mixed` loader, and confirm an FP8 matmul kernel runs on SM120 (RTX 5090).
  Decode compute is only ~13 ms of a ~48 ms step, so a slower kernel costs little, but measure it.
- Quality gate: same server, same prompts, BF16 vs FP8 — GSM8K subset and a long-generation sample;
  anything beyond noise on GSM8K means stop.
- Measure the hit-rate value first: replay the route trace with `sweep.py --budget-mb` raised by the freed
  amount before building anything.

### 4. FP8 `lm_head` — changes weights

- Saves: 0.59 GiB.
- Risk: logit precision on near-tied tokens; check with a greedy token-match rate against BF16.

### 5. Investigate the ~0.8 GiB unattributed allocation

The difference between summed weight bytes and `mem usage` after load. Check with
`torch.cuda.memory_snapshot()` right after `Load weight end` before choosing an option.

### 6. Allocate the NEXTN draft's `embed_tokens` and `lm_head` on `meta` — lossless, speculative runs only

- Observed 2026-09-13: `Qwen4ExpForCausalLMMTP` loads at 4.86 GB, i.e. its FP8 experts (2.34), other MTP
  weights (0.17) plus its own `embed_tokens` (1.18) and `lm_head` (1.18). The checkpoint's `mtp.*`
  tensors contain neither of the last two.
- Why they cost memory: `Scheduler.init_model_worker` sizes the target KV pool (`init_target_memory_pool`)
  while those copies are still allocated; only afterwards does `EAGLEWorkerV2.alloc_memory_pool` call
  `init_lm_head`, which replaces them with the target's tensors via `set_embed_and_head`. With the full
  10 GiB hot cache this made startup fail with "Loaded weights leave no GPU memory for the KV cache".
- Change: build the two modules under `torch.device("meta")` when the draft will share them (as the PLE
  n-gram tables already do), or share them before the target pool is sized.
- Saves: 2.36 GB at startup for speculative runs; likely enough to keep the 10 GiB hot cache with NEXTN.
- Done when: `Load weight end ... type=Qwen4ExpForCausalLMMTP` reports about 2.5 GB, and a NEXTN run with
  `SGLANG_MOE_HOT_GPU_MB=10240` starts and matches the 8 GiB run's accept length.
- Status: implemented in `0f311510ce` (Phase 2 of the NEXTN graph-capture plan), not yet checked on GPU.
  The draft binds the target's tensors at construction (`speculative/draft_shared_weights.py`) instead of
  staying on `meta`, because post-load staging raises on any `meta` tensor (`post_load.py:139-142`).

## Not worth doing

- KV-cache quantization: the whole KV cache is 0.18 GiB, so FP8 KV saves 0.09 GiB.
- Linear-attention state quantization: 0.11 GiB, with quality risk.
- The CUDA context: fixed by the CUDA runtime and libraries.

## Checkpoint alternative: `RadixArk/Qwen3.8-Flash-Next-NVFP4`

Not a VRAM win for this setup. Both checkpoints quantize only the routed experts to NVFP4 W4A4 and keep
attention, GatedDeltaNet, hyper-connections, shared experts, routers, embeddings and `lm_head` in BF16, so
the non-expert GPU footprint is the same. Differences that matter here:

| | nvidia (in use) | RadixArk |
|---|---|---|
| Recipe | ModelOpt MIXED_PRECISION (0.46.0 dev) | ModelOpt NVFP4, experts only (0.46.0 @ `87c9f8cf`) |
| MTP head | experts FP8 W8A8 (2.34 GiB) | BF16, byte-identical to source (~4.7 GiB) |
| PLE n-gram tables | FP8 | FP8, taken from the updated `Qwen3.8-Flash-Next-FP8` revision |
| Calibration | cnn_dailymail + Nemotron-Post-Training-Dataset-v2 | 128 cnn_dailymail articles, MoE inputs captured from SGLang prefill |
| File layout | 10 shards + `model-fp8-mtp-ple.safetensors` | one file per layer per 128 experts |
| Published evals | 9 benchmarks vs Qwen FP8 (GPQA 91.5 vs 92.0, etc.) | GSM8K 97.27, AIME26 98.75 pass@1 (earlier revision) |
| License / status | NVIDIA Open Model License, commercial use | "private candidate release", Qwen terms |

Switching would double the MTP head's GPU cost for the NEXTN experiment and require rebuilding the
115 GB expert and PLE file caches, whose manifests are keyed to the NVIDIA checkpoint path.
