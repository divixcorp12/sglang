# FP8 for the BF16 GPU weights of Qwen3.8-Flash-Next-NVFP4

Status: implemented behind an environment flag and off by default. CPU tests pass. No GPU accuracy data yet, so
nothing ships until the protocol in §6 passes.

Goal: free about 3.3 GiB of VRAM for the expert hot cache. E17 predicts +12–14% production decode tok/s from
that capacity.

## 1. What is on the GPU today

Byte counts come from the checkpoint safetensors headers (`fc694b54`, all BF16). Production `Load weight end`
reports 10.05 GB with `--language-model-only`.

| Group | Tensors (per layer) | Layers | BF16 GiB |
|---|---|---|---|
| GDN projections | `in_proj_qkv` [10240, 2560] + `in_proj_z` [6144, 2560] (loaded fused as `in_proj_qkvz`), `out_proj` [2560, 6144] | 36 | 3.867 |
| GDN small tensors | `in_proj_a/b` [48, 2560], `conv1d`, `A_log`, `dt_bias`, norm | 36 | 0.019 |
| Full attention | `q_proj` [12288, 2560] (gated), `k/v_proj` [512, 2560] (loaded as `qkv_proj`), `o_proj` [2560, 6144] | 12 | 1.113 |
| QSA indexer `index_qk_proj` | [640, 2560] | 12 | 0.037 |
| Hyper-connection mix | `input_mix_weight_down` [320, 10240] + `_up` [10240, 320], ×2 per layer, plus the final mixer | 48 (+1) | 1.184 |
| `block_inject_weight`, `hc_norm` | [4, 10240], [10240] | 48 | 0.010 |
| Shared expert | `gate_up_proj` [1280, 2560], `down_proj` [2560, 640] | 48 | 0.439 |
| Router `mlp.gate` | [512, 2560] | 48 | 0.117 |
| `lm_head` | [248320, 2560] | 1 | 1.184 |
| `embed_tokens` | [248320, 2560] | 1 | 1.184 |

The embedding is not offloaded. `--ple-offload-embedding` moves only the PLE n-gram tables (47.7 GiB of FP8);
`embed_tokens` stays on the GPU in BF16. The VRAM-options spec, option 2, covers moving it losslessly, so it is
out of scope here. The 4.15 GiB GDN figure in that spec includes the small tensors and rounding; the table above
is exact.

## 2. What converts, in order of risk

"Risk" means the expected logit error per GiB freed, judged from each module's position in the network.

1. **Shared expert** (0.219 GiB freed). An MLP that runs beside 10 routed experts, which are already NVFP4 and
   so carry far coarser quantization. Its output is scaled by `shared_expert_gate`. FP8 W8A8 on dense MLPs is
   the best-documented near-lossless case (Qwen's own FP8 releases).
2. **Full attention q/k/v/o** (0.556 GiB). Standard FP8 targets. Only 12 layers. `q_proj` includes the sigmoid
   output gate. QK-norm follows the projection, which absorbs per-channel scale error on q/k.
3. **GDN `in_proj_qkvz` + `out_proj`** (1.931 GiB). The largest group. Errors in q/k/v feed a recurrent state
   (the delta rule), so they can accumulate over the sequence, unlike attention. That is why GDN ranks below
   attention despite the size. Kept BF16: `in_proj_a/b` (they set decay `dt` and write strength `β`, and weigh
   only 7 MiB), `conv1d`, `A_log`, `dt_bias` and the gated norm.
4. **Hyper-connection mix down/up** (0.588 GiB). They produce the sigmoid mixing weights for the 4 residual
   streams in every layer, twice per layer, 97 modules in total. An error here scales the whole residual input
   of every block. There is no published evidence for quantizing hyper-connections. `down` also sees the widest,
   most outlier-prone activation (the 10,240-wide normalized residual). Kept BF16: `block_inject_weight` and
   `hc_norm` (4 MiB).
5. **`lm_head`** (0.591 GiB). The error lands directly in the logits and flips near-tied tokens. It is its own
   group so it can be dropped without affecting anything else.

Never converted:
- Norms, `conv1d`, `A_log`/`dt_bias`, `in_proj_a/b`.
- Router `mlp.gate`: routing errors change which NVFP4 experts run and are discontinuous. The saving would be
  0.06 GiB.
- `shared_expert_gate`, the QSA indexer, PLE key/value projections, `embed_tokens`.
- Every `mtp.*` module: the draft path is unused. Draft modules are excluded by prefix.

## 3. Scheme

**Default `w8a8`:**
- Weights: symmetric E4M3, one float32 scale per output row (`amax/448`).
- Activations: dynamic per-token E4M3 (`sglang_per_token_quant_fp8`).
- GEMM: `apply_fp8_linear(..., use_per_token_if_dynamic=True)`.

On SM120 (RTX 5090) this dispatches as follows:
1. `cutlass_fp8_supported()` returns True (capability 12, CUDA 13).
2. Channelwise CUTLASS path: `weight_scale.numel() == out_features`.
3. `fp8_scaled_mm`, whose SM12x registration is "streaming GEMV plus CUTLASS skinny GEMM".
4. A tuned Triton tile replaces it wherever `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE` has one.

This is the same kernel stack that `Fp8LinearMethod` uses for online FP8 on CUDA. Every converted shape satisfies
the CUTLASS divisible-by-16 condition, including `lm_head` (248320) and the 320-wide HC projections.

**Why not the alternatives:**
- **Block-128 W8A8** (`w8a8_block_fp8_linear`, FlashInfer groupwise CUTLASS on SM120): cannot represent the
  320-wide hyper-connection projections (320 % 128 ≠ 0). Its per-token-group activation quant only helps if
  per-token activation error turns out to be the problem, and the `w8a16` diagnostic below tells us whether it
  is.
- **Per-tensor W8A8** (the fastest SM120 path for checkpoint FP8): one scale per matrix is visibly worse on
  outlier rows, with no memory advantage.
- **`--fp8-gemm-backend`** stays at the production value `auto` (→ `cutlass`) and is independent of
  `--fp4-gemm-backend flashinfer_cutlass`, which serves only the NVFP4 experts.

**Diagnostic `w8a16`:** the same FP8 weights, dequantized for a BF16 matmul, in chunks of ≤ 32,768 rows. It
isolates weight error from activation error. It is slower and has larger transients (up to 320 MiB of float32
for `lm_head`), so it is not a ship candidate.

## 4. Enabling it

Environment variables are read when the model is built. Unset means bit-identical behaviour: every call site
returns a fresh `UnquantizedLinearMethod()` exactly as before, and `GatedResidual` takes its original branches.

| Variable | Values |
|---|---|
| `SGLANG_ONLINE_FP8_GROUPS` | comma list of `full_attn`, `gdn`, `shared_expert`, `hc_mix`, `lm_head`, or `all` |
| `SGLANG_ONLINE_FP8_SCHEME` | `w8a8` (default), `w8a16` |
| `SGLANG_ONLINE_FP8_LAYERS` | optional decoder-layer ids, e.g. `0-11,40`. It restricts layer-scoped modules; the final mixer and `lm_head` stay BF16 while it is set. |

Unknown groups, schemes or layer syntax raise at model build.

**Code:**
- **`layers/quantization/online_fp8.py`** (new): selection parsing, prefix → group matching,
  `quantize_fp8_per_channel` (4,096-row chunks, so the float32 transient is ≤ 157 MiB), and
  `OnlineFp8LinearMethod`.
  - Creates the same BF16 parameter as the unquantized method, so checkpoint loading is unchanged.
  - Quantizes in `process_weights_after_loading`, which runs after `load_weights` and before
    `maybe_init_expert_hot_cache`.
  - Deliberately not an `UnquantizedLinearMethod` subclass. `Qwen3_5GatedDeltaNet.finalize_fused_in_proj`
    fuses `in_proj_qkvz` and `in_proj_ba` into one BF16 tensor when that isinstance check passes. That would
    alias the BF16 storage, so no memory would be freed and the FP8 path would be skipped.
- **`modelopt_quant.py`**: `ModelOptMixedPrecisionConfig.get_quant_method` returns
  `online_fp8_or_unquantized(prefix)` in place of the two BF16 `UnquantizedLinearMethod()` returns. NVFP4, FP8
  and MXFP8 layers are unaffected.
- **`hyperconnection.py`**: `GatedResidual(online_fp8_scheme=...)` attaches `OnlineFp8LinearMethod` to the two
  mix `nn.Linear`s. The loader's generic `quant_method` loop quantizes them. `mix()` then runs
  `_mix_online_fp8` (the math of `_mix_compute`) instead of the BF16 Triton or compiled kernels.
- **`qwen4_exp.py`**: passes `hc_mix_online_fp8_scheme(prefix)` to the layer and final mixers.
- `lm_head` needs no model change: `ParallelLMHead` already takes its method from the quant config, and
  `LogitsProcessor._compute_lm_head` calls `quant_method.apply` for any method outside
  `_UNQUANTIZED_LM_HEAD_METHODS`.

**Not supported:**
- Speculative decoding with `lm_head` selected. `speculative/draft_shared_weights.build_with_target_weight`
  shares the target head only when shape and dtype match. An FP8 head matches neither, so the draft would
  allocate its own 1.18 GiB BF16 head, cancelling the saving. It is unverified whether that head is ever loaded
  when sharing was expected; if not, it holds uninitialized memory. Do not combine `lm_head` with NEXTN.
- Weight reload (`update_weights`): the BF16 parameter and its `weight_loader` are replaced after loading.

## 5. Memory freed and slot count

Each converted matrix of shape [r, c] frees `r·c` bytes (BF16 → FP8) and adds `4·r` bytes of float32 scales.

| Group | Bytes freed | GiB | MiB |
|---|---|---|---|
| `gdn` | 2,073,452,544 | 1.931 | 1,977.4 |
| `full_attn` | 596,926,464 | 0.556 | 569.3 |
| `hc_mix` | 631,601,920 | 0.588 | 602.3 |
| `shared_expert` | 235,192,320 | 0.219 | 224.3 |
| **All but `lm_head`** | **3,537,173,248** | **3.294** | **3,373.3** |
| `lm_head` | 634,705,920 | 0.591 | 605.3 |
| **All** | **4,171,879,168** | **3.885** | **3,978.6** |

**How the freed memory reaches the hot cache.** `SGLANG_MOE_HOT_GPU_MB` is a fixed budget and does not grow on
its own. Production's startup record gives:
- `residency_bytes` 9,408,641,624 = 3,403 slots × 2,764,808 B per expert row;
- `scratch_bytes` 1,327,107,840 = 480 rows (10 per layer × 48).

So: slots = ⌊HOT_MB · 2²⁰ / 2,764,808⌋ − 480.

**Baseline changed 09-13 ~21:20.** Production now runs `SGLANG_MOE_HOT_GPU_MB=14336`, `--context-length 65536`,
`--max-total-tokens 65536` and thinking on by default. A headroom check on whether 14 GiB fits is in progress. FP8's
freed memory goes on top of 14,336 MiB. To keep peak GPU memory equal to the 14,336 MiB BF16 server, raise the
budget by the freed MiB:

| Candidate | `SGLANG_MOE_HOT_GPU_MB` | Slots | Δ slots vs 14,336 |
|---|---|---|---|
| BF16 (production now) | 14,336 | 4,957 | — |
| All but `lm_head` | 17,709 | 6,236 | +1,279 |
| All | 18,315 | 6,466 | +1,509 |

The earlier 10,240 MiB baseline had 3,403 slots, where the same groups meant +1,279 and +1,508.

**No speed prediction exists for the new baseline.** E17 replayed only 3,403–4,762 slots, so 4,957 is already
past its range. Its curve flattens: +1,165 slots bought +12.4%, and +194 more bought only +1.8 points. The
return on +1,279 slots above 4,957 will therefore be well below E17's +13%. Rerun the E17 replay
(`cc-phase1/e13-replay/fp8-slots`) at 4,957 / 6,236 / 6,466 slots before committing GPU time to phase B's speed
runs. The accuracy phases do not depend on that answer.

- **Why no budget is freed automatically:** the freed memory is released during load (`Load weight end` should
  drop from 10.05 to ≈ 6.76 GB, or ≈ 6.17 GB with `lm_head`), before `maybe_init_expert_hot_cache` sizes the
  cache. A larger budget can therefore use it without any code change. Nothing sizes the budget from free memory
  automatically; the flag and the budget are set together in the launch script.
- **Why it should fit:** at equal peak, FP8 trades weight bytes for hot-cache bytes one for one. The KV cache is
  unchanged: 65,536 tokens × 24,576 B across 12 full-attention layers = 1.50 GiB, up from 0.19 GiB at 8,192.
- **Tight memory:** a rough BF16 estimate at 14,336 MiB is E19's 26.5 GiB prefill peak + 4.0 GiB of extra hot
  cache + 1.3 GiB of extra KV ≈ 31.8 GiB, against 31.84 GiB total. Take the measured peak from the lead's
  headroom check, not this estimate.
- **What to verify:** the raised-budget run must show peak ≤ (measured 14,336 MiB BF16 peak) + 256 MiB (the
  allowance covers CUTLASS workspace and per-token scale tensors).

## 6. Accuracy protocol

All comparisons use one server per configuration, the production launch settings (`run-fp8-server.sh`) at
temperature 0 with thinking off, and a BF16 reference built from the same worktree with the flag unset.

### (a) Teacher-forced logits: KL and top-1 agreement

- **Prompts:** 24 fixed prompts:
  - the Stage 3 workload, with `count_again` dropped as a duplicate of `count`;
  - the E1 route-trace workload (6);
  - 12 broader prompts: Chinese, German, JSON, SQL, Rust, debugging, arithmetic, logic, a poem, science,
    networking, ops.
- **Reference sequences:** BF16 greedy replies of up to 256 tokens (`generate`). Every configuration scores
  exactly these token ids.
- **Scored positions:** the last 64 prompt tokens plus every reply token. That is about 6,000–7,000 positions.
  The longest sequence (the 2,800-token records prompt + reply) stays in one 4,096-token prefill chunk.
- **Support:** the BF16 server's top-20 ids at each position, unioned per sequence (`support`).
- **Scoring:** each server returns logprobs for exactly that support (`token_ids_logprob`) plus its full-vocab
  top-2 (`score`). SGLang's input logprobs are computed over the full vocabulary, so both are exact.
- **KL** is `KL(P_BF16 ‖ Q_FP8)` over the support plus one tail bucket (1 − support mass).
  - Merging tokens can only reduce KL (data-processing inequality), so this is a lower bound on the full KL. It
    is tight when the tail mass is small; the report prints mean and p99 tail mass.
  - The CPU smoke test checks the bound numerically.
  - Full-vocab logits are not transferred: 248,320 floats per position would be about 6 GB of JSON.
- **Top-1** compares the full-vocab argmax of each server.
- **Noise floors:** the same BF16 server scored twice, and a second BF16 launch.

### (b) Greedy-match length

- Each configuration's greedy replies to the 24 prompts are compared with the BF16 reference. The metric is the
  first differing token index, capped at the reply length, taken as the median over prompts.
- A second BF16 launch gives the noise floor. E7 found only 3 of 7 production replies reproducible across runs,
  so an absolute token-identity gate is unusable.

### (c) GSM8K

- 200 fixed questions: indices ⌊i · 1319/200⌋, i = 0..199, from the official test set. The copy on divix01 has
  sha256 `3730d312…`.
- Zero-shot chat prompt ending "Answer: <number>", 512 max tokens. BF16 vs candidate are paired per question.

### Pass thresholds, and where each number comes from

| Gate | Threshold | Derivation |
|---|---|---|
| Mean KL, per group and for the candidate | ≤ 0.005 nats | Anchor: llama.cpp's published full-model table for Llama-3-8B vs BF16. q8_0 = 0.00136 nats, PPL ratio 1.0004; q6_K = 0.00545 nats, PPL ratio 1.0035 (the level usually treated as indistinguishable in practice); q4_K_M = 0.0313 nats, PPL ratio 1.028 (visibly degraded). 0.005 is the q6_K level. It is a strict bar for a change that touches only part of the weights, and our number is a lower bound measured on the model's own low-entropy continuations. |
| Top-1 agreement | ≥ min(0.977, BF16-noise agreement − 0.01) | 0.977 is llama.cpp q8_0 on Llama-3-8B (97.67%), measured on wikitext tokens, which are harder than self-generated replies; FP8 per-channel should do no worse than int8 block-32. The noise term keeps the gate from demanding more than BF16 achieves against itself. |
| Greedy-match median | ≥ 0.5 × BF16-vs-BF16 median | With per-token divergence rate d, the expected match length is about 1/d. Allowing FP8 to add divergence at most equal to run-to-run noise (d → 2d) halves the median. |
| GSM8K | drop ≤ 2.0 points **and** no McNemar-significant drop (exact, p < 0.05) | At accuracy near 0.95, the paired standard deviation of the difference is √(discordance/n): about 1.6 points at 5% discordance and n = 200. A true drop of about 4.5 points is detected with roughly 80% power; smaller drops cannot be certified with 200 questions. GSM8K is therefore a gross-failure gate and KL is the fine gate. The 2-point budget is 4× NVIDIA's published NVFP4-vs-FP8 GPQA gap (0.5), because 200 questions cannot resolve less. |

- p99 KL, the flips not explained by the Pinsker bound, and per-request KL are reported, not gated. No sourced
  threshold exists for them.
- **The number I trust least is the 0.005 KL anchor.** It comes from a different model family and text
  distribution. If the BF16 restart noise itself exceeds 0.001, reconsider the gate before running the groups.

**Decision rule:**
1. Every group must pass (a) alone in phase A.
2. If a group fails with `w8a8`, run the `w8a16` diagnostic for it:
   - `w8a16` passes: activation quantization is the cause; drop the group or try block-128 as a follow-up.
   - `w8a16` also fails: weight precision is the cause; drop the group.
   - To localize a failing group, bisect its layers with `SGLANG_ONLINE_FP8_LAYERS`.
3. The phase-B candidate is the union of passing groups; `lm_head` joins only if it passes too. The candidate
   must pass (a), (b) and (c).
4. Speed must not regress at equal budget (decode tok/s within 3% of BF16), and must improve at the raised
   budget.
5. Anything else: stop.

## 7. GPU session plan

**Scripts:** `scripts/fp8_accuracy/` (synced to `work/cc-fp8/worktree`).
- `run-fp8-server.sh <name> <port> <hot_mb> <groups|none> <scheme> [layers]`
- `fp8-session.sh A | B <groups> <hot_mb> | diag <group>`

Results are written to `work/cc-fp8/results/`, and logs to `work/cc-fp8/session/`. Each server takes about
3 minutes to start (load 108 s, hot cache, graph capture).

Every server uses production's current flags: 65,536-token context, thinking-on default, E16c policy. The base
budget is `BASE_HOT_MB` (default 14,336). The harness sends token ids built with thinking off, for
determinism and GSM8K run time; the server default does not change what the harness measures. Budget does not
change the numerics, because expert bytes are identical. If the headroom check shows 14,336 MiB does not fit
for BF16, run phases A and B with `BASE_HOT_MB=10240`; the accuracy results stay valid.

| Phase | Servers | Work | Time |
|---|---|---|---|
| A | bf16-a, bf16-b, fp8-{full_attn, shared_expert, gdn, hc_mix, lm_head} (all at `BASE_HOT_MB`) | 2 × 24-prompt greedy (~11 min each at ~10 tok/s), 4 scoring passes on BF16, 1 on each group (~1 min each) | ≈ 60–70 min |
| B | bf16-gsm, candidate at 10,240, candidate at raised budget | GSM8K ×2 (200 × ~250 tokens ≈ 85–90 min each), candidate scoring + greedy, raised-budget greedy + 4,096-token prefill | ≈ 3.5–4 h |
| diag | 1 server per failing group | scoring | ≈ 5 min each |

**Peak memory:**

| Run | Peak GPU |
|---|---|
| BF16 servers at 14,336 MiB | the lead's headroom-check peak (rough estimate ≈ 31.8 GiB during a 4,096-token prefill; at 10,240 MiB E19 measured 26.5 GiB) |
| FP8 servers at `BASE_HOT_MB` | BF16 peak minus the freed amount (−3.29 GiB all-but-`lm_head`, −3.89 GiB all) |
| Raised-budget run (17,709 or 18,315 MiB) | ≤ BF16 peak + 256 MiB (gate) |

Scoring adds a transient of at most about 0.7 GiB: ≤ 320 scored positions × 248,320 float32 logits plus
log-softmax.

## 8. Open risks

- **Kernel coverage on SM120 is unmeasured.** It is untested whether CUTLASS channelwise FP8 at m = 1 runs
  inside the breakable decode graph, and whether it is slower than the BF16 path. `hc_mix` currently uses a
  fused Triton kernel at decode (≤ 16 rows); FP8 replaces it with 2 GEMMs plus eager pointwise ops per mix
  (194 mixes per step), still inside the graph. Decode compute is about 13 ms of a 48 ms step, so this probably
  costs little, but phase B measures it.
- **GDN recurrence.** Teacher-forced prefill scoring runs the chunked GDN kernels, while decode runs the
  recurrent path. Error accumulation over long decodes is covered only by greedy-match and GSM8K, not by KL.
- **Tail mass.** If the reported tail mass is large, the KL lower bound is loose. Raise `--top-k` for `support`.
- **Graph gather and PLE staging are untouched,** but any new host sync would show up as `breaks>0` in the
  graph log. Check the `Capture target decode CUDA graph` line in every FP8 run.
- **lm_head and speculative decoding** are incompatible (§4): the draft silently allocates a BF16 head. NEXTN is
  off the roadmap after E19.
- **A malformed `SGLANG_ONLINE_FP8_LAYERS`** raises only when the first BF16 layer is built, deep in model
  construction, not at argument parsing.
- **The HC-mix FP8 path** replaces the fused Triton mix kernel with 2 GEMMs + per-token quant + eager pointwise
  ops, 194 times per decode step. This is the most likely source of a speed regression; phase B measures it.
