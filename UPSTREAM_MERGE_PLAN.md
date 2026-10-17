# Finishing the upstream merge (flashinfer 0.7.2 / torch 2.14.1)

Nothing here has been executed on divix01. Every step marked OWNER GO waits for the owner.
Branch: `upstream-merge-20261017` (worktree `/Users/dnikolaidis/Desktop/divix/wt-upstream-merge`).

## 1. Status

- Merge commit `55db88f833` (`--no-ff`): parents `f111be253d` (origin/master at merge time) and `14c4dda24e` (upstream/main).
- Upstream range: `993d1fccba..14c4dda24e`, 1,734 commits. Fork side: 3,964 commits.
- 36 files conflicted: 27 combined, 9 took upstream with our changes re-applied (7 files plus 2 deletions: `sharded_greedy.py`, `candidate_indexer.py`), 0 ours-only.
- Only static checks have run: compileall, import and name AST checks, no conflict markers. No test, no import of torch.
- `python/uv.lock`, `.python-version`, `*_NEXT_HANDOFF.md` are not committed. `uv.lock` is stale against the new pins.

### NEEDS REVIEW

1. **DSpark top-prob gate** (`speculative/dspark_components/dspark_draft.py:247` `draft_markov_block`, `dspark_draft_sampler.py`, `dspark_draft_top_prob_log.py:28`).
   Ours: probability from `sharded_greedy_step(top_prob_out)`, no logits kept. Upstream: `MarkovSampleStep` replaced that kernel.
   Now: probability from the kept step logits (`top_prob_out`), so a greedy draft with the gate on builds `[bs, gamma, vocab]` logits in the draft graph and skips the one-row `MarkovGreedyStep` fast path. `sharded_greedy.py` and `sample_block_greedy_fused` are removed. The gate still requires tp_size 1.
2. **Layer-major late tail** (`models/deepseek_v4.py:4649` `forward_late_tail`, `_forward_layers_hc_pre_from_prev`).
   Ours: `prev_pre`/`layer_ids`/`hash_ids_in` threaded through the old hand-written loop. Upstream: `HcState` plus `residual_batch`/`hc_boundary`.
   Now: `HcState(hidden_states, prev_pre_in)` and a `layer_ids` loop. The CUDA `residual_batch.start/set_written/clear` path when resuming mid-stack is unverified.
3. **EXL3 cast fusion and stats-stream refork** (`models/deepseek_v4_mhc.py:408` `_combine`, `:258` `mix_stats`).
   Ours lived in the deleted decoder forward (`hc_combine_norm_half` + `EXL3_HALF_INPUT.publish`; `_refork_stats_stream`). Ported into upstream's mhc module. Upstream's `fork_stats_stream` may already cover the refork.
4. **Candidate indexer** (`layers/attention/dsv4/v41_indexer/__init__.py`, `scoring.py`, `deepseek_v4_backend.py:1000,3503`).
   Ours: `candidate_indexer.py` with tail-only publish, topk_v2-off mask path, torch score budget. Upstream: `v41_indexer` package, tail publish built in.
   Carried: sm_120 DeepGEMM metadata for c1/c2 pools and the prefill-graph indexer, fp32 weights on sm_120, topk_v2-off falls back to dense blocks (decode and prefill), no candidate blocks when `candidate_source_layer_id < 0`, `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB`, layer-major skip-candidates (calls `full_topk_indexer.topk_prefill` instead of publishing).
   Deleted tests: `test_dsv4_candidate_indexer.py`, `test_dsv41_torch_indexer_chunking.py` (tail-only publish mechanics are not tested any more).
5. **Layer-major launch gate** (`arg_groups/deepseek_v4_hook.py:263`). Upstream auto-resolves decoder SWA replay. The layer-major refusal now runs after that resolution in all three branches (explicit on, explicit off, auto).
6. **`use_vision_topk`** (`models/deepseek_v2.py:1048`). Upstream computes it from the VL bias; ours always routes V4.1 through `vision_topk`. Now OR-ed with `self.v41_router`.
7. **Qwen4-Exp** (`models/qwen4_exp.py`). Our PLE file cache, staged prefetch, host token embedding and route trace sit on top of upstream's meta-device PLE build and PP missing layers. Best effort; outside the fork's DSV4.1 scope.
8. **Offloader** (`utils/offloader.py`). `create_offloader_from_server_args` lost `dp_rank` (V2 reads it from `get_parallel()`); callers and tests updated.

## 2. Re-sync, after the owner's batches land on master

1. In the worktree: `git fetch origin upstream && git merge origin/master` (merge, never rebase). If upstream moved: `git merge upstream/main` as a separate merge.
2. Follow `.claude/rules/merge-rounds.md`: fix the merge set first, one message, owner's yes, then test.
3. Edit with the Edit/Write tools only. Never commit `python/uv.lock`, `.python-version` or handoff files.
4. Expected conflicts (branch: what it touches, what upstream structure to port onto):

| Branch | Touches | Port onto |
|---|---|---|
| `cut-sampled` | `dspark_confidence_cut.py`, `dspark_worker_v2.py`, minimal AcceptSampling (confidence cut in sampled mode, STS recalibration, draft time in T(w), online calibration) | Worker: `_gate_verify_lens`, `proposal.confidence_raw` (not `confidence_tap`), `accept_and_finalize(..., simulate_bonus_sampling_info, gate_verify_lens)`. Draft time must come from the new `DraftBlockProposer.propose`. |
| `sampled-accept` | `dspark_accept.py`, `dspark_draft.py` around `SampleStepTokens`; greedy drafting for sampled requests; block verification; fused greedy draft | `SampleStepTokens`, the folded sampler and the fused greedy draft no longer exist. Use `draft_markov_block(..., greedy_draft=)`, `MarkovSampleStep` (`kernels/ops/speculative/dspark/markov_sample.py`), `DraftBlockResult.greedy_draft`, `DraftNoise`. Upstream already has `block_verification` in `dspark_verify.accept_draft_tokens`. |
| `warmup-preload`, `tw-calib` | warm-up end signal, copy-engine arming after a quiet period, decode warm-up, cut-kernel preload (`http_server`, `scheduler.py`, `warmup.py`, `dspark_accept` preload) | `Scheduler(server_args, port_args)` has no rank arguments; `init_all_cuda_graphs`/`init_model_worker` already carry our `startup_memory_probe` and hot-cache calibration; the `DSparkWorkerV2.init_cuda_graphs` early return `if not self._hosts_draft` and `_draft_sampler_built` guard. |
| `prelaunch-cache` | analysis scripts only | Should merge clean. |
| `engram-spin-log` | Engram logging | `layers/engram.py`; the Engram call in `deepseek_v4.py` `_forward_layers_hc_pre_from_prev` now uses `post_engram_device_lookups`. |
| `engram-relayout` (prototype) | Engram file table / native store | Same Engram call sites; `hc_boundary.materialize` is called before an Engram layer (`deepseek_v4.py` loop). |

5. Deprecated envs, now no-ops with a warning (no code reads them):
   - `SGLANG_DSPARK_FOLDED_SAMPLING`: a folded proposal always samples. Use `SGLANG_DSPARK_USE_GREEDY_DRAFT=1` for argmax-only, or `SGLANG_DSPARK_FOLDED_PROPOSAL=0` to keep it out of the graph.
   - `SGLANG_DSPARK_FAST_SAMPLING`: draft sampling is always Gumbel-max inside `MarkovSampleStep`.
   - `SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV`: a one-row step of `SGLANG_DSPARK_USE_GREEDY_DRAFT` on a vanilla markov head fuses the bias GEMV (`MarkovGreedyStep`).
   - `git grep` the arm scripts (`benchmarks/dsv41_baseline/arm_env.py`, `launch_prod.sh`) for these three before the first served run.

## 3. Environment on divix01 (OWNER GO)

- New venv: `/data/models/slang/.venv-upstream`, Python >= 3.11 (the merged `requires-python`). Never touch `/data/models/slang/.venv` or its `main-port-probe-*` import path.
- Install from the merged `python/pyproject.toml`: torch==2.14.1 (cu13), flashinfer_python[cu13]==0.7.2, **flashinfer-cubin==0.7.2 and flashinfer-jit-cache==0.7.2+cu130** (the Dockerfile installs both from `https://flashinfer.ai/whl` and `.../whl/cu130`; confirm the wheels exist for the box's CUDA), sgl-deep-gemm==0.2.2, sgl-deep-ep==0.1.2.post1, sglang-kernel==0.4.10, nvidia-cutlass-dsl[cu13]==4.8.0, quack-kernels==0.6.5, transformers==5.19.0, tokenizers==0.23.2, xgrammar==0.2.7, torchcodec==0.17.0, sentencepiece==0.2.1, numba==0.65.1, tilelang==0.1.12, plus new `regex`.
- `torchaudio==2.11.0` is pinned upstream and `torchvision` is unpinned. Verify against torch 2.14.1 with `pip install --dry-run` first; if torchaudio has no matching wheel, ask the owner before overriding the pin.
- `uv.lock`: do not regenerate or commit unless the owner says so. Install with `uv pip install -e python` against the pyproject, not `uv sync`.
- CUDA/driver check before installing: `nvidia-smi` (driver must support CUDA 13), `nvcc --version`, and that the 5090 reports sm_120.
- Caches, so the old venv is untouched:
  - `export SGLANG_JIT_CACHE_DIR` (or whatever `kernels/jit/utils/compile` reads in this tree: verify with `grep -rn "cache" python/sglang/kernels/jit/utils/compile/loader.py`) to a new dir, e.g. `/data/models/slang/nvfp4-work/jit-upstream`.
  - `SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/exl3-upstream-build`.
  - `TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE` likewise under `.../upstream-cache/`.
  - `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp` for any traced arm; `/tmp` on the root volume is ~88% full.
- Use a private worktree: `git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-upstream origin/upstream-merge-20261017`. Never run in `dsv41-direct-prod`.

## 4. Rebuilds (all under the new venv and the new cache dirs)

| Item | How | Time (estimate) |
|---|---|---|
| EXL3 extension | `SGLANG_EXL3_SRC=<pinned exllamav3 checkout>` (see `exl3_ext.EXLLAMAV3_COMMIT`), build with the gcc-toolset-15 compiler used for the current build (same toolchain flags as master) | 10-20 min |
| sglang-kernel | prefer the 0.4.10 wheel from the pin. Local build only if the wheel lacks sm_120 (check `python -c "import sgl_kernel"` loads and a sm_120 op runs) | wheel: seconds; local: 30-60 min |
| Expert-stream host variants and other JIT modules | any C++/flag change gives every module a new key. Warm in the parent before tests: `run_host_script`/`warm_host_modules`, serialized by the build lock | 50-100 s per variant, 20-40 min total |
| sgl-deep-gemm 0.2.2 vs our sm_120 DeepGEMM fork | check that `deep_gemm` exports `fp8_fp4_paged_mqa_logits`, `fp8_fp4_mqa_logits` and the paged sparse MQA logits (`DEEPGEMM_PAGED_SPARSE_MQA_LOGITS` in `layers/deep_gemm_wrapper/configurer.py`). If the stock wheel lacks sm_120 support, the fork must be rebuilt against 2.14.1 | wheel: seconds; fork: 15-30 min |
| TileLang 0.1.12 and Triton JITs (indexer, mhc, hc_mix, markov_sample) | first run compiles them | 5-15 min |

## 5. Test plan, in order

Common prefix (CPU): `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv-upstream/bin/python -m pytest <target> -q -p no:randomly; echo EXIT=${PIPESTATUS[0]}`. Print `sglang.__file__` first. Do not point pytest at `test/registered` as a whole (pyarrow collection errors).

0. **Baseline once:** the same commands on current master with the old venv, record counts (pass/skip/fail) next to the command. Reuse the base for later rounds.
1. CPU unit groups:
   - `benchmarks/dsv41_baseline` (its tests dir)
   - `test/registered/unit/kernels/test_*exl3*.py test/registered/unit/kernels/test_*expert*.py test/registered/unit/kernels/test_dsv41_torch_score_budget.py`
   - `test/registered/unit/spec test/registered/unit/speculative test/registered/spec/dspark`
   - `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py test/registered/unit/layers/quantization/test_modelopt_nvfp4.py test/registered/unit/layer_major test/registered/unit/test_nvfp4_expert_offload.py test/registered/unit/models/test_mtp_shared_embed_head.py test/registered/unit/test_environ.py`
2. GPU groups (take `rowimg-disk.lock` first if the group reads row images, then): `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh <cmd>`, or `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 <cmd>` (gpu-run.sh exits 75 after 30 min):
   - DSpark kernel parity: `test/registered/kernels/ops/speculative`
   - `test/manual/dsv41/test_dspark_top_prob_gate_gpu.py` (rewritten on `MarkovSampleStep`)
   - `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py`
   - flash_mla sm120 tests: `git grep -ln flash_mla_sm120 -- test` and run those files
   - the `test/manual/dsv41` graph-parity tests (`test_exl3_ram_miss_graph_gpu.py` and neighbours)
3. Compare counts to step 0. Any new failure is a candidate regression until reproduced on the base.

## 6. Served validation (OWNER GO)

Plain A/B, `-prod` argv and env unchanged, non-7867 port, both arms under the same locks:
- A: current master, old venv. B: the merged branch, new venv.
- Compare: greedy token parity for the first N tokens (N >= 256) on fixed prompts, ms/token, tokens per verify, TTFT, peak VRAM, late Triton loads, `copy_engine_module_loads`.
- NEEDS REVIEW checks:
  - Item 1: run B once with `SGLANG_DSPARK_TOP_PROB_GATE` set and once without. Report ms/token and VRAM delta of the full-vocab logits in the draft graph; gate must still cut as before (compare `gate_verify_len` log).
  - Item 2: one long prompt with `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS` set; output must match the chunked path; no OOM or failure result.
  - Item 3: decode with the EXL3 cast fusion env on, batch size 1, then a breakable-prefill capture; no stream hang, token parity.
  - Item 4: a long prompt (candidate blocks on) in sampled and greedy mode; sm_120 must run the dense-block path (topk_v2 off).

## 7. Optional follow-up arm

flashinfer sparse MLA on sm_120 under DSpark: `SGLANG_SM120_FLASHMLA_BACKEND=flashinfer` (the env default in this tree reads `"flashinfer"`; confirm what production sets). Separate arm only after section 6 passes. Check that the `num_tokens > 64` refusal is gone, peak VRAM and speed against the triton FlashMLA backend.

## 8. Rollback

Production keeps the old venv and master until the owner switches. The merge branch is separate; nothing is pushed to master. To abandon: delete the worktree and the `.venv-upstream` / upstream cache dirs.

## 9. Owner-go checkpoints

1. Start the re-sync merge (after the current batches are on master), and confirm the merge set (merge-rounds rule).
2. Create `.venv-upstream` and install the packages (including any override of the torchaudio pin).
3. Rebuild EXL3 / sgl-kernel / DeepGEMM fork and warm the JIT caches (long, uses the box).
4. Run the GPU groups (takes `cc-gpu.lock`).
5. Run the served A/B (takes the GPU and disk locks).
6. Switch production to the new venv (only after all of the above).

## 10. All conflicts and how each was resolved

Source of truth: `git merge-tree --write-tree --name-only origin/master upstream/main` lists exactly 36 conflicted paths (re-run to reproduce; it writes only objects, no worktree). Line refs are in the merged tree at the commit that adds this section. "NR" = NEEDS REVIEW item from section 1. Resolution key: **C** combined (hunks resolved by hand), **T+** upstream's file taken, our changes re-applied by hand, **D** deleted. No file was taken as ours alone. All conflicts are content conflicts unless stated.

### 10.1 The 36 conflicted files

1. `.codespellrc` (content). Both sides edited the ignore-words line. C: union of both word lists (`...,valu,ser,suh`). Check: `codespell` config parses; no test. Not NR.
2. `kernels/ops/speculative/dspark/dspark_draft_model.py` (content, `_block_quant_stack_applies`, :304-322). Ours: return False when a linear has no dense `.weight` (EXL3). Upstream: gfx950 aiter MXFP8 gate. C: both guards, ours first. Check: `test/registered/spec/dspark` unit tests; on box, EXL3 DSpark draft loads. Not NR.
3. `kernels/ops/speculative/dspark/sharded_greedy.py` (modify/delete). Ours added `top_prob_out`/sum-exp kernel (gate). Upstream deleted it for `MarkovSampleStep`. D (briefly kept, then removed once the gate moved to step logits). Dropped: the fused sharded top-prob kernel. Check: `test_dspark_top_prob_gate_gpu.py::test_the_markov_step_logits_give_the_softmax_top_probability_with_the_argmax`. **NR 1**.
4. `srt/arg_groups/deepseek_v4_hook.py` (content, :237, :263, :439-458). Ours: `dsv41_layer_major_refusal` + layer-major launch checks after the explicit-replay branch. Upstream: `validate_deepseek_v41_encoder_replay_budget` and auto-resolution of decoder SWA replay. C: both functions kept; new `validate_deepseek_v41_layer_major_prefill` called in all three branches (explicit on/off, resolved). Uses `attn_dp_enabled_of(cfg)` (renamed field). Check: `test/registered/unit/layer_major/test_gate.py` plus a launch with layer-major env set and replay off. **NR 5**.
5. `srt/environ.py` (content, 4 hunks: :310, :1056-1059, :2353-2356, :2715-2743). Ours: `InsertOnMissStage`, `SGLANG_HICACHE_HOST_NUMA_NODE/WRITE_BACK_STAGING_MAX_MB/DEVICE_RESERVE_MB`, `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB`, our deprecation notes. Upstream: removed `DsparkFoldedSampling`, added `SGLANG_HICACHE_TMA_TRANSFER`, `SGLANG_OPT_DSV41_BF16_INDEX_LOGITS`, deprecations. C: both sets kept; dropped only `DsparkFoldedSampling` (no users). Check: no duplicate env names (checked by script), `test/registered/unit/test_environ.py`. Not NR.
6. `srt/eplb/expert_distribution.py` (content, :173-185, :300-330). Ours: forward/pre-forward observers (hot cache). Upstream: `get_current_pass_count_buffer`, `_record_current_pass`/`should_record_forward_pass`. C: observers kept; recording gate now `recording or observers` and `_record_current_pass` applies to both. Check: hot-cache observer test under `test/registered/unit/layers/moe`; on box, hot cache sees every forward. Not NR (megamoe-only filter is off by default).
7. `srt/layers/attention/deepseek_v4_backend.py` (content, 9 hunks). Ours: `candidate_indexer` calls, tail-only masks, layer-major skip, sm_120 fixes. Upstream: `v41_indexer` package, rewrite. T+. Carried over: sm_120 `force_deep_gemm_metadata` for c1/c2 and the prefill-graph metadata (:1374, :1567); `_low_ratio_in_prefill_graph(forward_mode)` (:3177); `install_forward_metadata` (:2619); `layer_major_skip_candidates` field (:1000) and branch (:3503); source-layer < 0 means no candidate blocks (:1182). Dropped (upstream native or replaced): `candidate_publish_rows`, `candidate_tail_only_of`, `keep_row_slice`, `_torch_indexer_rows_per_chunk` (moved to scoring.py), `_selects_candidates_through_masks` (moved to the factory). Check: `test_dsv4_nonpaged_indexer.py`; box: sampled and greedy long prompt on sm_120. **NR 4**.
8. `srt/layers/attention/dsv4/candidate_indexer.py` (modify/delete). Ours: `source_layer_id` param, topk_v2-off returns None, `keep_row_slice`. Upstream deleted it for `v41_indexer`. D. Carried: source/topk_v2 behaviour (see 7 and 30); dropped: `keep_row_slice`, mask-based tail publish. Check: `TestCandidateIndexerGating`. **NR 4**.
9. `srt/layers/attention/dsv4/indexer.py` (content, :457 old). Ours: 3-line sm_120 fp32 weights fix in `deep_gemm_fp4_paged_mqa_logits`. Upstream moved that function to `kernels/ops/attention/dsv4/index_logits.py`. T+: fix re-applied at `index_logits.py:110`. Check: box, sm_120 paged logits call with bf16 weights. Not NR.
10. `srt/layers/hyperconnection.py` (content, :252-260). Ours: `_mix_online_fp8`. Upstream: `mix(..., normalized_input)`. C: both; body already used both. Check: `test_hc_mix.py`. Not NR.
11. `srt/layers/moe/fused_moe_triton/layer.py` (content, :1192, :1465). Ours: file-cache wrappers `_weight_loader_impl` / `weight_loader_fused` calling `*_uncached`. Upstream: removed the `tp_rank` local. C: wrappers kept; `tp_rank = self.moe_tp_rank` dropped from both uncached bodies. Check: `test_nvfp4_expert_offload.py`. Not NR.
12. `srt/layers/quantization/modelopt_quant.py` (content, :1094, :2708, :3393). Ours: draft NVFP4 requant on `FP8_BLOCK_SCALES`, local `w13_weight` deinterleave for file cache, expert-streamer gather. Upstream: `_BLOCK_FP8_ALGOS`, `use_flashinfer_trtllm_weight_layout`, `self.moe_runner_config`. C: all three semantics combined. Check: `test_modelopt_nvfp4.py`. Not NR.
13. `srt/managers/schedule_policy.py` (content, :1667). Ours: `req.layer_major = admission.is_layer_major`, `_req_inc_lock_ref`, `set_extend_range`. Upstream: `req.extend_end`, `req.lock = tree_cache.lock(...)`. C: upstream's lock model plus our `layer_major` flag. Check: scheduler layer-major admission test (`unit/layer_major`); box: lock released on finish. Not NR.
14. `srt/managers/scheduler.py` (content, :113, :1482, :6367). Ours: layer-major gate imports, `maybe_lock_process_memory`, SWA allocator imports. Upstream: `get_dp_tp_group`, `page_interleave_shard_size`, `Scheduler(server_args, port_args)`. C: all imports; ctor call upstream's with our memory lock after; `init_layer_major_gate` now reads `get_parallel().attn_dp_enabled`. Dropped import: `compute_dp_attention_world_info`, `msgspec_to_builtins` (unused upstream). Check: unit/layer_major; box: scheduler starts. Not NR.
15. `srt/managers/tokenizer_manager.py` (content, :3059-3101). Ours: skip logprob handling when a batch carries no logprobs (refused request). Upstream: None-safe inner checks. C: outer guard on `output_*_val is not None` kept, upstream's inner checks adopted. Check: a request refused at admission with `top_logprobs_num > 0`. Not NR.
16. `srt/mem_cache/pool_host/base.py` (content, :34, :40). Ours: `write_back_staging_page_capacity`. Upstream: `_host_memory_budget` ContextVar, `available_host_memory_bytes`. C: both kept; dropped the old `KVCache` import (unused upstream). Check: `mem_cache` host pool unit tests; box: HiCache write-back staging env. Not NR.
17. `srt/mem_cache/unified_radix_cache.py` (content, `swa_reprefill_tail_tokens`). Ours: `get_exec().features.enable_decoder_swa_bounded_replay` returns the window. Upstream implements that natively (`self.decoder_swa_bounded_replay`, `session.will_lend_record(req)`). T+ with nothing re-applied (upstream's covers it). Check: cache-hit prefix with replay on. Not NR.
18. `srt/model_executor/model_runner.py` (content, :473, :714). Frozen file. Ours: `create_offloader_from_server_args(...)`, expert hot-cache/arena/pinned/prediction init after `prepare_moe_topk`. Upstream: `create_offloader()`, `prepare_moe_topk(model, model_config)`. C: our collaborator calls kept, no new logic added. Also `self.ps` to `get_parallel()` (:864) and `enable_dp_attention` to `attn_dp_enabled` (:799). Not NR beyond 8.
19. `srt/model_executor/runner/decode_cuda_graph_runner.py` (content, :228). Ours: `_eager_only`. Upstream: `dllm_input_preparation`, `dllm_attention` class attrs. C: all three. Check: DSpark verify overflow re-run test. Not NR.
20. `srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` (content, :147). Ours: `_capture_output_rows`, shared-buffer growth. Upstream: `eager_on_graph(forward_fn)` call form. C: upstream call form, our buffer logic. Check: BCG capture of verify graphs. Not NR.
21. `srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py` (content, :488-502). Ours: `is_breakable_graph_capturing`, `retain_for_current_graph`, `current_capture_stream_handle`. Upstream: `break_graph` body without `pass`. C: our helpers kept, `pass` removed. Not NR.
22. `srt/models/deepseek_v2.py` (content, 4 hunks :709, :1048, :1170, :1359, :1449). Ours: `moe_side_stream`, `v41_router` via `vision_topk`. Upstream: HIP gate partials, `return_moe_output`/`leave_output_sum`, removed `_fuse_finalize_all_reduce`. C: side stream join placed before the `MoEOutput` return; `use_vision_topk = use_vision_topk or self.v41_router`. Dropped: `_fuse_finalize_all_reduce` (upstream removed; no readers). Check: V4.1 MoE forward on box, parity with master. **NR 6**.
23. `srt/models/deepseek_v4.py` (content, 4 hunks incl. a 760-line deleted decoder forward). T+ (upstream file, then ours re-applied): `late_layer_start_of` (:3645), `post_engram_device_lookups` (:3764), `is_in_breakable_prefill_graph` at three sites (:2314, :2654, :3758), EXL3 no-fuse `wqkv_a` (:930), `skip_checkpoint_weight` (:4327), `forward_late_tail`/`make_layer_major_adapter` (:4649, :4684), ram-prefetch `register_moe_gates` (:4773), EXL3 weight adapters (:4944-4962), `exl3` in `_prepare_deepseek_v4_weights` (:5492), layer-major kwargs on `_forward_layers_hc_pre_from_prev`. Moved: EXL3 cast fusion and stats refork to `deepseek_v4_mhc.py`. Dropped: our `_refork_stats_stream` helper (moved), duplicate `is_dsv41` (upstream has it). Check: DSV4 forward on box with EXL3, layer-major long prompt, Engram on. **NR 2, 3**.
24. `srt/models/deepseek_v4_dspark.py` (content, :522). Ours: `supports_sharded_greedy`, `sample_block_greedy_fused(top_prob_out)`. Upstream: `step_logits_dtype`. C: upstream's only; ours dropped (replaced by logits path). Check: DSpark V4 draft graph on box. **NR 1**.
25. `srt/models/qwen4_exp.py` (content, 8 hunks :651, :2168-2224, imports). Ours: PLE cache identity/tag, host token embedding with shared table, `online_fp8_scheme`, `prepare_decode_graph_replay`. Upstream: PLE built on meta, PP `PPMissingLayer`, tc-piecewise import, `linear_attn_parallel_group`. C: upstream structure; our wrapper args moved into the NGram embedding (:651); PP guards wrap our code (:2168, :2221). Check: `test_qwen4_ple_offload.py` (GPU). **NR 7**.
26. `srt/speculative/dspark_components/dspark_draft.py` (content, 4 hunks). T+: upstream's `draft_markov_block`, `DraftNoise`, `greedy_draft`. Re-applied: `DraftBlockResult.top_probs` (:68), `DraftProposal.raw_hidden/draft_hidden`, `top_prob_out` in `draft_markov_block` (:291 uses `step_top_prob`), `sample_draft_block(record_top_prob)` (:299-339), proposer `record_top_prob`, folded `top_probs` (:440). Dropped: `recording_top_prob` wrapper (function kept in gate module, now unused), `confidence_tap`. Check: `test_dspark_top_prob_gate.py::test_the_eager_greedy_proposal_records_each_steps_top_probability`. **NR 1**.
27. `srt/speculative/dspark_components/dspark_draft_sampler.py` (content, 5 hunks). T+. Re-applied: `record_top_prob`, `top_prob_out` buffer, logits kept when the gate is on (:35, :64, :123-126, :155-183). Dropped: `folded_sampling`, `confidence_fn`, `_base_logits_dtype`, sharded/fused greedy paths. Check: DSpark draft graph capture on box; `test_dspark_draft_graph*`. **NR 1**.
28. `srt/speculative/dspark_components/dspark_verify.py` (content, 6 hunks). T+. Re-applied: `forward_verify_with_reverify` (:337), `static_cutoff`/`begin_gated_static_step` (:549, :641, :670), `cutoff_verify_lens` in `accept_draft_tokens` (:961) and `gate_verify_lens` in `accept_and_finalize`. Check: `test_dspark_top_prob_gate.py::test_a_capped_greedy_accept_never_accepts_past_the_cut`; `test_dspark_top_prob_gate_gpu.py::test_the_static_epilogue_stops_the_accept_at_the_gates_cut`. **NR 1**.
29. `srt/speculative/dspark_components/dspark_worker_v2.py` (content, 8 hunks). Ours: top-prob gate init/lens, EXL3 draft graphs, `_draft_sampler_built`, `check_dspark_shared_lm_head_usable`, hot-cache commit, verify dead rows, prefill-peak log. Upstream: `ParallelState` removed (`get_parallel()`), `_hosts_draft`, `get_dp_tp_group`, `coordination_plan`, `random_seed`/`greedy_draft`, simulate-fold. C: all; fixed stale `self.ps` and `enable_dp_attention` in `_init_top_prob_gate` (:548) and `init_cuda_graphs` (:653-696); `_maybe_build_draft_sampler()` takes no memory arg now. Check: `test_dspark_draft_graph_init_idempotent.py`; box: DSpark serve, gate on and off. **NR 1**.
30. `srt/speculative/eagle_worker_common.py` (content, :725). Ours: verify trace. Upstream: fused commit outputs, hisparse accept. C: trace block first, then upstream's. Check: eagle/NEXTN verify unit tests; verify trace env. Not NR.
31. `srt/speculative/eagle_worker_v2.py` (content, :79-83, :211, :371). Ours: `draft_shares_target_embed_and_head`, `require_vocab_weights_materialized`. Upstream: `DPSpecPrefillCoordinationPlan`, `draft_owns_attention`, `_resolve_shared_embed_and_head`. C: all kept. Check: `test_mtp_shared_embed_head.py` (updated for the new ctor). Not NR.
32. `srt/utils/offloader.py` (content, :148, :172). Ours: `create_offloader_from_server_args(server_args, dp_rank, model_config)`. Upstream: `create_offloader()`, V2 reads dp rank itself. C: ours kept, `dp_rank` parameter dropped; callers (`model_runner.py:473`, `test_nvfp4_expert_offload.py`) updated. Check: that test. **NR 8**.
33. `test/registered/kernels/ops/embeddings/test_qwen4_ple_offload.py` (content). Ours: staged-prefetch tests. Upstream: runtime-context fixture + `test_qwen4_ple_offload_avoids_device_table`. C: ours as base, upstream's fixture, test and imports added. Check: GPU run of the file. **NR 7**.
34. `test/registered/kernels/ops/gemm/test_hc_mix.py` (content plus rename: ours `hyperconnection/test_hc_mix_triton.py`, upstream `gemm/test_hc_mix.py`). C: new path and module `sglang.kernels.ops.gemm.hc_mix`, kept our `envs` import. Check: that test. Not NR.
35. `test/registered/unit/layers/quantization/test_modelopt_nvfp4.py` (content, imports). C: union of imports. Not NR.
36. `test/registered/unit/layers/test_dsv4_nonpaged_indexer.py` (content, :1120). Ours: asserted old `make_candidate_indexer` returns None. Upstream: asserts dense backends. C: upstream's, plus a new case: topk_v2 off on Blackwell returns dense blocks. Check: `TestCandidateIndexerGating`. **NR 4**.

### 10.2 Files changed outside the conflict list (upstream moved or removed code)

- `kernels/ops/attention/dsv4/index_logits.py`: receives the sm_120 fp32-weights fix from file 9 (:110); adds a `get_platform` import.
- `srt/layers/attention/dsv4/v41_indexer/__init__.py`: `SGLANG_OPT_USE_TOPK_V2` gates both `_use_deep_gemm_prefill` (:70) and `make_candidate_indexer` (:119, dense blocks when off); top-level `envs` import. Check: `TestCandidateIndexerGating`. NR 4.
- `srt/layers/attention/dsv4/v41_indexer/scoring.py`: `_torch_score_rows_per_chunk` (:273, used :355) honours `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB`. Check: `test_dsv41_torch_score_budget.py`. NR 4.
- `srt/models/deepseek_v4_mhc.py`: EXL3 cast fusion in `_combine` (:457-462); BCG stats-stream refork in `mix_stats` (:258). NR 3.
- `srt/speculative/dspark_components/dspark_draft_top_prob_log.py`: `draft_top_probs` (:28) rebuilt on `_draft_step` with teacher-forced tokens, since `markov_head.sample_block` is gone. Check: `test_dspark_draft_top_prob_log.py`. NR 1.
- `srt/models/deepseek_v4_dspark.py`: see file 24 (conflict file; listed for completeness).
- Tests edited: `test/manual/dsv41/test_dspark_top_prob_gate_gpu.py` (first test on `MarkovSampleStep` + `step_top_prob`); `test/manual/dsv41/test_dsv41_cast_fusion_gpu.py` (`silu_and_mul_clamp` now from `kernels.ops.moe.dsv4`); `test/registered/unit/models/test_mtp_shared_embed_head.py` (no `ps=`, `attn_dp_enabled`, patched `get_pp_stage_load_group`/`draft_tp_context`); `test/registered/unit/speculative/test_dspark_draft_graph_init_idempotent.py` (`_hosts_draft`, no-arg sampler builder); `test/registered/unit/speculative/test_dspark_top_prob_gate.py` (new `sample_draft_block` signature, `DraftNoise`, `greedy_draft`); `test/registered/unit/test_nvfp4_expert_offload.py` (no `dp_rank`). Also `analysis/dsv41-compute-gap/wo_a_bench.py` (import path).
- Tests deleted: `test/registered/unit/layers/attention/test_dsv4_candidate_indexer.py` and `test/registered/unit/kernels/test_dsv41_torch_indexer_chunking.py`. They tested the removed `candidate_indexer` module and the mask-based tail publish. Lost coverage: tail-only publish equivalence and chunking-equals-single-pass for the torch indexer. Replaced by `test_dsv41_torch_score_budget.py` and the new gating case. Worth re-adding against `v41_indexer` after the re-sync.
- Tests added: `test/registered/unit/kernels/test_dsv41_torch_score_budget.py`.
- Merged cleanly by git but worth a glance because they sit next to the above: `kernels/ops/gemm/hc_mix.py` (renamed from `srt/layers/hc_mix_triton.py`), `kernels/ops/gemm/dsv4_wo_a.py`, `srt/mem_cache/memory_pool_host.py`.
