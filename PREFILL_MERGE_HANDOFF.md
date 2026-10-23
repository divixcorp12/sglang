# Prefill-merge / hot-cache calibration handoff (2026-10-16, final state)

**Goal** (team lead, for the owner): recover the VRAM the DSpark arms give to the prefill spike and spend it on the
expert hot cache, with the KV pool pinned at **381,184 tokens** (owner's requirement).

The work went through three stages:
1. chunking the triton FlashMLA merge;
2. a prefill-transient attribution;
3. a startup calibration that sizes the hot cache from measured memory.

**Owner rules that shaped the design:**
- values are computed at startup, never hard-coded;
- the server refuses to start below the KV pin;
- everything new is off by default, and the owner sets the production defaults.

## 1. State

**Branch `prefill-merge-chunk` on `origin`.** It is based on `split-model` 1ddfaae39e. The head before this file is
**a75bb27afb**. No other branch carries this work.

**Worktrees:**
- Laptop: `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-prefill-merge`. Never edit the main checkout.
- divix01: `/data/models/slang/nvfp4-work/wt-prefill-merge-test`, detached at a75bb27afb and clean. All calib runs and
  the served A/B ran here.
- divix01, old and unused: `wt-prefill-merge` (a988627f06) and `wt-prefill-merge-dev` (1d0e669e7f).

**Tests at a75bb27afb:** 200 passed, 56 skipped, EXIT=0.
- Command, on divix01 in wt-prefill-merge-test:

  ```
  PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES=99 taskset -c 56-63 python -m pytest -p no:randomly <files>
  ```

- Files:
  - `test/registered/unit/mem_cache/test_hot_cache_{trial_capture,calibration,retire,vmm_math}.py`
  - `test/registered/unit/mem_cache/test_prefill_cache_release.py`
  - `test/registered/unit/model_executor/test_prefill_peak_log.py`
  - `test/registered/unit/speculative/test_dspark_draft_graph_init_idempotent.py`
  - `test/registered/unit/scripts/test_both_cpu_ab.py`
  - `test/registered/unit/kernels/test_flash_mla_sm120_merge_chunking.py`
  - `test/registered/unit/layers/moe/test_expert_tier_startup.py`

**GPU tests**, last run in calib11b:
- merge-chunk + VMM: 66 passed, 12 skipped;
- `test_expert_residency_gpu.py` in its own process: 72 passed.

## 2. Flags and arms (all off by default)

| Flag | Default | What |
|---|---|---|
| `SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB` | 0 | (A3) triton FlashMLA LSE merge and sink in token chunks. Bitwise. |
| `SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS` | False | (A2) with the budget set, the SWA and extra kernels run per token chunk into one output. Bitwise. +1.8% prefill time at 4096 in isolation (not resolvable in serving). |
| `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB` | arm_env 128 | (A1) existing knob. 32 in the calib arm (bitwise, §27.7). |
| `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP` | False | The startup calibration (§3). |
| `SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL` | False | empty_cache before each new high-water prefill. Superseded by expandable segments; never GPU-tested. |
| `SGLANG_DEBUG_PREFILL_PEAK_LOG` | False | Per-call allocated/reserved inside a DSpark prefill (`prefill_peak_report.py` reads it). |

The arms are defined in `analysis/dsv41-drive/dspark/both_cpu_ab.py`:
- `...-mergechunk`: -hicache1 plus merge budget 128 and its hot cache.
- `...-mergechunk-expseg`: plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 1,035 slots, fraction 0.8623, KV
  pinned.
- `...-mergechunk-expseg-calib`: `-expseg` plus the calibration flag, A1 (indexer 32), A2 and A3 (merge 32).
- Every other arm pins the calibration and the chunk kernels off against an exported shell.

## 3. Calibration design, as built

**Hook.** `managers/scheduler.py` `init_model_worker` calls, in order:
1. `maybe_calibrate_hot_cache(target runner, draft_worker=...)`, between the attention-backend init and
   `init_all_cuda_graphs`;
2. the capture;
3. `require_hot_cache_headroom("after capture")`.

Startup memory probes log around the draft load and the target and draft captures (flag on only). `model_runner.py`
is untouched.

**Sequence** (`layers/moe/hot_cache_calibration.py`):
1. **KV pin check.** The pool must be at least `--max-total-tokens`, else `KvPinMissed`. This always re-raises.
2. **Two trial captures** (`hot_cache_trial_capture.py`):
   - Each captures the target via `capture_cuda_graphs` (not `ModelRunner.init_cuda_graphs`, so the residency updater
     stays disabled), and the draft via its `init_cuda_graphs`.
   - Both go into a private `graph_pool_handle()` swapped in for the global pool.
   - Free and reserved memory are measured while the graphs are alive.
   - Teardown: `backend.cleanup()`, the draft runners' graph attributes cleared, the attention capture metadata
     emptied, gc, the global pool restored, `empty_cache`.
   - The cold trial creates the persistent capture state, which the real capture reuses: per-layer state built on
     the first graph-path forward, the shared input buffers and the DSpark draft sampler. The warm trial then costs
     what the real capture does.
   - After each trial: `reset_trial_routes` (the capture discard with the updater left disabled) and
     `forget_graph_path_state`.
   - Skipped when observability forward hooks are configured.
3. **Warm-up replay.**
   - The dummy EXTEND prefills replay `DSV41_PREFILL_WARMUP_SIZES` (made public in `entrypoints/warmup.py`) up to the
     chunk, then 2 more full chunks.
   - They run under `no_grad`, as serving does; `inference_mode` broke the 1-token cast fusion.
   - Calibration routes go to RAM-tier residents and the forward is stateless.
   - Then one eager TARGET_VERIFY run.
4. **`settle`.** `empty_cache`, then F = free memory and reserved_final.
5. **Sizing:**
   - regrow = highest chunk reserve peak − (reserved_final − tensors the verify left);
   - lowest_free = F − regrow;
   - **margin = the worst allocator gap among the chunk runs + the spread of their reserve peaks** (measured noise);
   - **capture estimate = max(warm trial, cold trial less its residue, eager verify)**;
   - required = margin + capture estimate;
   - room = lowest_free − required;
   - keep_free = regrow + required.
6. **Census.** The updater must be disabled, the copy engine unarmed and every capacity holder in agreement; otherwise
   no resize, a WARNING, and the configured size.
7. **Plan.**
   - With room ≥ 0: `plan_grow`, priced in mapped 2 MiB granules (`VmmHotCacheBacking.bytes_to_back`), emptiest
     layer first.
   - With room < 0: `plan_keep` trims from the fullest layer.
   - The summary is logged before the resize.
8. **Resize.** `set_hot_capacity(strict=False)`, the single capacity authority.
   - A grow goes layer by layer behind a free-memory guard (free − cost ≥ keep_free).
   - A cuMemCreate OOM rolls that layer's mapping back (`HotCacheMapOutOfMemory`) and growing stops short, logged.
   - A holder mismatch afterwards rolls every layer back to its pre-call capacity, logged at WARNING.
   - The census reads a row's SpecScoreRow against its **target** layer (the L+2 scorer).
9. **Post-capture check.** It logs free memory, the chunk reserve peak, the projection, the margin and the measured
   capture against the estimate. It is a WARNING when short and **never refuses**: nothing can shrink after capture,
   because the graphs bake `hot_capacity` into spec_score, spec_head and lease kernel parameters.
10. **One-shot watch.** It compares the first real 4,096-token prefill with the calibration, folding the peak
    logger's nested peak resets.

**The grow bound.** The VMM reservation rows equal the non-static memory (total × (1 − fraction)) in slots, spread over
the layers in proportion to their pinned-tier headroom.
- Without it, the small scale tensors, which stay on torch.empty, were sized for about 6,200 rows and cost the KV pool
  232 MB (calib4).
- `log_grow_bound` logs what each layer would have wanted.
- The bound has never bound.

**Failure handling.** Any exception before the resize is logged at ERROR with its traceback, then cleaned up (trial
routes reset, graph-path state dropped, the census checked), and the launch starts at the configured size. These
re-raise:
- `KvPinMissed`;
- any failure during the resize (holders may be half-moved);
- `CalibrationUnrecoverable`: the cleanup itself failed (for example a sticky CUDA error), or the holders disagree after
  the cleanup.

`strict=True` (tests) re-raises everything. The path was exercised on the GPU in calib11, where a diagnostic bug
started the launch at 1,035.

**Module map** (`python/sglang/srt/layers/moe/`):
- `hot_cache_calibration.py`
- `hot_cache_trial_capture.py`
- `hot_cache_retire.py`: `set_hot_capacity`, `census_error`, `CAPACITY_HOLDERS`.
- `hot_cache_vmm.py`
- `hot_cache_dummy_extend.py`
- `expert_hot_cache.py`: VMM backing and the bounded upper.
- `speculative/dspark_components/dspark_worker_v2.py`: the draft sampler and its hook are built once (idempotent init),
  the one change that applies with the flag off.

## 4. Results

### Calibration launches

All are combined smokes: the -expseg arm, calibration, A1/A2/A3, one 4,096 prompt, the peak logger on.

| run | outcome | slots | peak free (nvidia-smi) | margin | capture: estimate / real |
|---|---|---|---|---|---|
| calib5 | refused: census bug (row score against its own layer) | — | — | — | 232 / — |
| calib6 | served | 1,059 | 472 | 574 | 30 / 214 |
| calib7 | served, **near-OOM** | 1,101 | **38** | 334 | 30 / 214 |
| calib8 | failed: `inference_mode` at the 1-token replay | — | — | — | — |
| calib8b | served | 1,049 | 692 | 508.5 | 232 / 98 |
| calib9 | served | 1,047 | 682 | 508.5 | 232 / 98 |
| calib10 | served, residue diagnostic | 1,053 | 670 | 508.5 | 212 / 98 |
| calib11 | calibration failed (diagnostic walk); launch at 1,035 as designed | 1,035 | 900 | — | — |
| **calib11b** | served | **1,056** | **552** | 508.5 | **100 / 98** |

- **calib11b, the final design.**
  - Cold trial: drop 712 = torch 212 + outside torch 500 (first-use module loads); persistent 648; residue 148.
  - Warm trial: drop 100, residue 36. **Cold − warm = 612 MiB**, all state that persists for the process.
  - Estimate 100 (warm) against the real capture's 98: **within 2%**.
  - Peak free 552 ≥ margin 508.5. +21 slots.
  - Replayed reserve peak 14,722 against the real warm-up's 14,758 (+36). Transient +41.2.
  - Startup 4:59 from START to ready, against calib7's 4:35: **+24 s** (cold trial 8-16 s, warm 4 s, the longer replay
    about 7 s).
- **The margin across launches.** The replay and the worst-chunk rule made it stable at 508.5 (488.5 gap + 20 spread).
  Before that it moved 574 → 334 with the last repeat's growth. The replay put the reserve peak within 36-38 MiB of the
  real one, against −312 in calib7.
- **The residue.**
  - calib10 counted 879 blocks, 134.6 MiB after the cold teardown.
  - About 21 MiB is the process-wide input-buffer pool (mrope_positions 8.7, next_token_logits 3.0, input_ids,
    positions and out_cache_loc 2.9 each).
  - **114 MiB is still unattributed**, likely per-layer state built on the first graph-path forward.
  - It is legitimate persistent state: the warm trial proves the real capture reuses it.

### Served A/B (e)

One sample each at c6d3128a40, peak logger off. Artifacts: `ram-prefetch/prefill-merge-ab-20261016/`.
- OFF = -mergechunk-expseg (1,035 slots). ON = -expseg-calib (1,046 slots; the trial estimate was still the cold 232).
- **serve_compare: clean prompts (5 of 8) +0.62 ms/token, CI −1.88..+2.60. No cost and no gain.**
- STALL flags: OFF p0; ON p2 and p5.
- Free memory (1 s samples):
  - OFF: warm-up minimum 580, serving minimum 580;
  - ON: warm-up minimum 624, serving minimum 624, margin 508.5.
  - **Serving never went below the warm-up minimum.**
- VRAM verify hit rate 43.72% against 43.69%; prefill 13.13% against 13.27%.
- Node 0 minimum during load: 230 / 135 MiB against a 43 MiB watermark. It is the draft weight load's dip in both
  arms, not the trial or the replay.
- 0 OOM lines.
- ON's slow prefills (p3 and p4, 30 s) sat inside an nvme3n1 stall (await 47-52 ms against about 9) that was not
  flagged.

### Earlier results

See DSV_OPTIMIZATION.md §3.7.
- Merge cause: 2,048 MiB at 4,096 tokens against flashinfer's 256.
- expseg2: −1.79 ms/token against mergechunk-r2.
- A0 attribution: attention 903 → 584 with A1+A2+A3. wq_b's 522 MiB transient (266 of it temporary) now binds.
- B1 dropped: hc_post's extra copy falls after the attention peak.

## 5. Open items

1. **The 114 MiB unattributed residue.** The attribution walk was fixed in a75bb27afb (exact dict, list and tuple
   only; never calls a subclass's `items()`) but not yet run on the GPU. It is logging only; nothing should be released
   from it.
2. **Warm-trial residue 36 MiB** (not ≈0). It is counted in F, so it is safe but unexplained.
3. **serve_compare (stream-variance tool) flags stalls in decode only.** Prefill-window stalls go unflagged: ON p3 and
   p4 in (e) had 47-52 ms await during prefill. Owner of the tool: stream-variance.
4. **(c)** a clean long-prompt prefill A/B (logger off) for A2's cost. **(d)** wq_b's 266 MiB temporary (about 20
   slots). Both are parked by team-lead until VRAM binds again.
5. **The trial builds the DSpark draft sampler** at higher free memory than the real capture used to. In a tight launch
   it could build one the real capture would have skipped. It is counted in F.
6. **Order-dependent GPU test.** `test_expert_residency_gpu.py::...graphed_verifies` fails only after the merge-chunk
   and VMM files in one process. Logged by team-lead; run that file in its own process.
7. **Production defaults** are the owner's decision.
   - The calib arm buys about +20 slots per launch: +21 in calib11b, which is below the bound and unmeasurable in
     ms/token.
   - Its value is the safety margin by design (peak free ≥ the measured margin) plus about 24 s of startup.

## 6. Protocol notes and pitfalls

- **CPU tests.** `CUDA_VISIBLE_DEVICES=""` breaks `sglang.test.test_utils` at import (it indexes `[0]` of the
  variable). Use `CUDA_VISIBLE_DEVICES=99`.
- **Waiting on a run.** Wait in the foreground for runs under 10 minutes: a background monitor's events do not wake an
  idle agent. In a remote filter, avoid `| cut` or `| head` stages, which block-buffer; poll the process IDs.
- **Node 1 prelaunch stalls.**
  - Reclaimable slab (dentries and inodes) and other lanes' page cache count against node-1 MemFree.
  - The 02:00 stall needed a root cache drop. The 03:46 one (30 GiB of file pages) cleared by itself within 3 minutes.
  - Never lower the floor.
- **Never in-server allocator snapshots** (`/start_profile MEM`): they OOM-killed the scheduler on node 0.
  `torch.cuda.memory_snapshot()` in-process, as the residue diagnostic uses, is fine.
- **Repo edits.** The edit hook blocks scripted edits of tracked files; use Edit and Write. Throwaway mutants on divix01
  are the exception: resend the command.
- **Commits.** Commit trailer:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01KvT26Vzsfr4r2yo73YRcMR
  ```

  Stage files by name. Never commit `uv.lock`, `.python-version` or `RAM_PREFETCH_*_HANDOFF.md`.
