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
