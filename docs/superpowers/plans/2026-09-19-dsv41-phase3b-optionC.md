# DeepSeek V4.1 Flash — Phase 3b (option C: MoE inside the decode graph, NVMe misses served by an io_uring thread and a device wait) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run DSV4.1 decode at batch size 1 under a `breakable` CUDA graph with the **whole MoE inside the graph**: an in-graph gather of VRAM-missed rows from the pinned RAM tier, exllamav3's fused `exl3_moe` over slot pointer tables, and RAM misses served by **§9.3 option C**, a C++ io_uring thread that reads, splits and publishes rows while a single-block device kernel waits with a bounded poll and fails stop on timeout. Add a minimal next-layer prefetch hook (option F) behind one env var, and measure everything in one GPU window.

**Architecture:**
- **P2 first — the `exl3_moe` probe is a HARD GATE (Task 1).** It runs before anything else and depends on no other task: a manual GPU test runs `ext.exl3_moe` against `exl3_moe_loop` on real layer-3 rows at BS1 top-6, then captures it in a torch CUDA graph over nine slot pointer tables and a device remap and replays it after rewriting slots, remap and routes in place, for `num_active` 6 and −1. **If it fails, Tasks 6–16 are blocked**: the P1 tasks (2–5) still run, Task 17 writes a short record, and execution stops.
- **P1 — graph decode foundation (Tasks 2–5).** The EXL3 gate accepts decode `breakable` at max batch size 1 with prefill disabled; the Engram file-table lookup becomes an eager break; the stream trace, route recording and hot-cache counters ignore warmup and capture forwards; `trace_corpus.py`/`compare_oracle.py` get `--graphs`. Until P3 lands, `FusedMoE.forward`'s existing eager MoE break runs the 3a streamed path unchanged (the fallback, and P6's graph baseline arm).
- **P3 — in-graph MoE (Tasks 6–9).** Framework (R3, on `cc/moe-expert-plugins`, Tasks 6–7, merged by Task 8): a format may declare **pinned-tier graph sources** (`graph_source_kind = "pinned_tier"`); `enable_graph_gather` then builds its segment table over the pinned slabs and installs `PinnedTierRowBackend`, which adds the **host-slot indirection** (`host_rows = pinned.expert_to_slot.index_select(0, plan.expert_ids)`) before `copy_expert_row_segments_gpu` and keeps a device `ram_miss` counter and a `keep` scale. EXL3 side (Task 9): `Exl3FusedMoE` (nine `int64` pointer tables over hot slots + scratch) and `Exl3MoEMethod._apply_graph` = gather → `ext.exl3_moe` + `ext.exl3_moe_gather` fed the gather's `remap`, in-graph. Residency stays host-boundary; GPU residency update, insert-on-miss and the RAM→VRAM doorbell stay OFF. P3's backend alone only counts a RAM miss inside a replay and drops the layer; it is a test configuration, and `_apply_graph` refuses to serve without option C outside tests (Task 14).
- **P4 — option C (Tasks 10–14).** Two new JIT modules on the EXL3 side: `exl3_ram_miss_host` (C++ thread, no CUDA calls: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`) and `exl3_ram_miss` (device kernels: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`), wrapped by `python/sglang/kernels/ops/moe/exl3_ram_miss.py` and served by `python/sglang/srt/layers/moe/exl3_ram_miss.py`. They copy the doorbell's page/watchdog/fail-stop patterns without changing the doorbell:
  - an in-graph **post** kernel writes the layer's deduped RAM-missed expert ids to a host-mapped **request page** and rings a sequence number;
  - a C++ **service thread** (Tasks 10–12: the split, the slot LRU and request service, the thread and watchdog) owns the pinned-slot LRU (always, once option C is on; eager paths reach it through a locked proxy after a pause handshake), picks victims (never a protected id), reads the EXL3 superset rows with io_uring, does the per-name split in C++ (a flat copy plan exported from `Exl3ShardRowSource`/`segment_map`), writes the six pinned slabs, updates the **device-visible pinned map** (a mapped `int32` array the gather reads) and publishes a completion sequence;
  - an in-graph single-block **wait** kernel polls the completion with a bounded budget; on timeout it sets the fatal word and makes the layer's output poisoned-but-harmless; the scheduler's per-batch fail-stop hook (the doorbell's, generalised) raises, and a watchdog `std::abort()`s if the process never gets there. **No copy-from-RAM fallback exists.**
- **P5 — prefetch hook (Task 15).** `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` (off by default): during layer L's graph segment, a device kernel posts the previous token's routes for layer L+1 as *advisory* requests; the thread reads them if idle; the demand post of L+1 finds them already present.
- **P6 — GPU window and record (Tasks 16–17).** Every earlier task already ran its own GPU test (the GPU is available whenever needed); the window reruns them as a regression, then: CUDA tests, capture smoke on the truncated model, eager-vs-graph numerics (R4), forced-miss and timeout fail-stop tests (§9.3 acceptance), the R5 corpus arm with prefetch off and on, a graph-mode Nsight capture analysed under memory caps, and the spec's new §17 (which also corrects the "~40 decode breaks" claim).

**Rulings this plan implements** (`.superpowers/plans-research/dsv41-phase3b/rulings.md`; the owner's Amendment R1' is the scope):
- **R1'** phases P1–P6 as above, in that dependency order. The per-layer eager sync cut of the pre-amendment plan is **dropped** (the untracked `docs/superpowers/plans/2026-09-19-dsv41-phase3b1.md` is superseded; it is not executed and not committed by this plan).
- **R3** framework edits: Task 2's counter re-baseline and Tasks 6–7 are committed on `cc/moe-expert-plugins` in `FWT`, pushed, CPU-tested on `DFWT`, then reach `dsv41` by a plain `git merge` (Task 2 Step 6 and Task 8). Never merge `dsv41` into `cc/moe-expert-plugins`.
- **R4** eager vs graph on the truncated model: identical greedy tokens over 32 and max |Δlogprob| ≤ 1e-3, with an eager-vs-eager control and a debug-eager isolation arm (Task 16, D3).
- **R5** Window C's reduced cold corpus shape (4 sessions, `--prompt-tokens 256 --new-tokens 128`), graphs on, prefetch off and on, plus one `--cuda-graph-trace=graph` Nsight decode capture (Task 16).
- **R6** is P2 (Task 1), with its bar stated there.

**Tech Stack:** Python 3.13, PyTorch 2.13 (cu130), CUDA 13.2 on the RTX 5090 (sm_120); the MoE expert framework (`layers/moe/expert_*.py`, `cc/moe-expert-plugins`); SGLang's breakable CUDA graph backend; the repo's JIT kernel mechanism (`sglang.kernels.jit.utils.load_jit` + `tvm_ffi`, as `expert_doorbell` and `uring_file_reader` use); liburing (`-luring`, as `uring_file_reader` links); exllamav3 at `02aef45` (`ext.exl3_moe`, `ext.exl3_moe_gather`); Nsight Systems 2026.3.2 on divix01.

**Spec:** [`DSV41_REFERENCE.md`](../../../DSV41_REFERENCE.md) §9.3 (options, acceptance criteria), §9.4, §11 (Phase 2b, 3b), §15.1 (m = 1 GEMM error), §16 (Phase 3a numbers, esp. §16.2, §16.4, §16.12). Research: `.superpowers/plans-research/dsv41-phase3b/{rulings,prof-summary,A-framework-graph-path,B-dsv4-graph-capture}.md`. Conventions: `docs/superpowers/plans/2026-09-18-dsv41-phase3a-r2.md`.

**Out of scope:** decode graphs at batch size > 1; prefill graphs; speculative decoding and DSpark; GPU residency update / insert-on-miss; copy-engine (DMA) row copies (3b-3); Engram rows staged before replay; the BLOB host layout; drop-on-miss (option D); restarting production.

## Global Constraints

- **Never commit to `master`.** EXL3 work goes on `dsv41` (worktree `WT`). Never merge `dsv41` into a mainline or production branch.
- **(R3) Framework files are edited only on `cc/moe-expert-plugins` in `FWT`:** `python/sglang/srt/layers/moe/expert_*.py`, `layers/moe/expert_format.py`, `arg_groups/expert_stream_requirements.py`, `arg_groups/memory_hook.py`, `model_executor/model_runner.py`, `python/sglang/test/moe_expert_fakes.py`, `python/sglang/kernels/ops/moe/expert_*.py`, `python/sglang/kernels/jit/csrc/moe/expert_*.cuh`. Each change is pushed to `shared`, CPU-tested on `DFWT`, and brought into `dsv41` by a plain `git merge shared/cc/moe-expert-plugins` (never rebase or cherry-pick). Never merge `dsv41` into `cc/moe-expert-plugins`. Prefer the EXL3 side (`quantization/exl3*.py`, `layers/moe/exl3_*.py`, `arg_groups/expert_stream_requirements_exl3.py`, `engram*`, `scripts/dsv41`) when equally clean.
- **The production doorbell's behaviour for Qwen does not change.** `expert_doorbell.cuh`/`.py` are not edited (Design decision D18); Qwen's doorbell tests (`test/registered/unit/kernels/test_expert_doorbell_copier.py`) run unmodified in Task 16.
- **CPU jobs on divix01:** `taskset -c 0-63` with `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`. Cores 64–71 are reserved (core 71 is production's doorbell spin core). The reader thread's CPU-test pinning uses cores inside 0–63 only.
- **GPU commands** may appear in any task; the owner has granted GPU use whenever needed, and no step waits for an approval or a reply. Every GPU command runs through `$ANA/gpu-run.sh` (created by Task 1: `taskset -c 32-63 flock -n` on `/data/models/slang/nvfp4-work/cc-gpu.lock`) with `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`. **GPU lock rule:** `gpu-run.sh` retries the lock every 60 s for up to 30 min; if it still exits 75 (lock held), append `Task N: BLOCKED — cc-gpu.lock held` to the ledger, continue with the next CPU-only task, and come back to the blocked step afterwards. **Production is down; never start it.**
- **No bulk reads of `/mnt/nvme1` or `/mnt/nvme2`** except through the model's own loader or row source inside a GPU slot. safetensors headers, `config.json`, index files, the tokenizer and the JSONL corpus are fine.
- **Memory caps:** heavy CPU jobs, every Nsight export and every trace query run under `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 <cmd>` (DuckDB: `SET memory_limit='4GB'; SET threads=4`). Analyse traces on divix01; never copy a report to the laptop's `/tmp` or the scratchpad.
- **Nsight decode traces use `--cuda-graph-trace=graph`.** A graph-mode kernel table leaves out the graph body: read wall time, GPU idle and API counts from it, never kernel rankings.
- **Test files:** end with exactly one of the two `__main__` blocks from `test/README.md` (no argparse); registered tests call `register_cpu_ci(...)`/`register_cuda_ci(...)`; GPU or divix01-data tests go in `test/manual/dsv41/`; CPU tests never import `sglang.test.test_utils`.
- **Env vars:** exactly three new ones, all declared in `python/sglang/srt/environ.py` per `.claude/skills/env-var-conventions/SKILL.md`: `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS` (Task 14), the test-only `SGLANG_TEST_DSV41_RAM_MISS_FAULT` (Task 14; Rule 4's `TEST_` verb) and `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` (Task 15). Read the skill before touching `environ.py`.
- **`model_runner.py` / `scheduler.py`:** read `.claude/skills/large-class-style/SKILL.md` first. `scheduler.py` is not edited (the fail-stop reuses its existing doorbell hook, D14). `model_runner.py` gets one condition change inside `maybe_init_expert_hot_cache` (Task 6, framework branch); no `__init__` change.
- **Commits:** stage files by name, never `git add -A`; never amend, rebase, stash or force-push. Messages end with exactly:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
  ```
- **Red/green through commits.** divix01 runs only what is pushed: commit failing tests (`test(<scope>): ... (red)`), push, run red on divix01; commit the implementation, push, run green. Never `scp` into or edit a divix01 worktree. Every divix01 command begins with `git pull --ff-only`.
- **No step stops to ask.** Every failure has a rule in its step: fix it through a red/green commit pair on the task that owns the file, or record `Task N: BLOCKED — <reason>` in the ledger and continue with the next task that does not depend on it.
- The expert directory is always a parameter (`SGLANG_DSV41_EXPERT_DIR`, `DSV41_EXL3_DIR` for manual tests); today `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw`.

## Paths used throughout

| Name | Path |
|---|---|
| Laptop dsv41 worktree (`WT`) | `/home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41` (branch `dsv41`) |
| Laptop framework worktree (`FWT`) | `/home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins` (branch `cc/moe-expert-plugins`) |
| divix01 dsv41 worktree (`DWT`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41` — the brief's `/data/models/slang/nvfp4-work/wt-dsv41` does not exist (checked 2026-09-19 with `ls` over ssh) |
| divix01 framework worktree (`DFWT`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins` |
| divix01 python (`PY`) | `/data/models/slang/.venv/bin/python` (has `cuda-python`) |
| EXL3 checkpoint (`EXL3_DIR`) | `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw` |
| Full-model dir (`FULL`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40` |
| Truncated model (`T3`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-trunc3` (layers 0–2, Engram layer 1, `candidate_source_layer_id=-1`) |
| Corpus (`SESSIONS`) | `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl` |
| Phase 1 analysis (`ANA1`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1` (`pyarrow-shim` for pytest) |
| Phase 3a analysis (`ANA3A`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a` (`env.sh`, `corpus-cold.json`, `prof-summary.md`) |
| This plan's analysis (`ANA`) | `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b` (made by Task 1) |
| Lock | `/data/models/slang/nvfp4-work/cc-gpu.lock` |
| Ledger | `WT/.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md` (lines `Task N: complete`) |

**CPU test commands.** `<FW-CPU> <paths>` means:

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins push shared cc/moe-expert-plugins
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins && \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim \
  taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf <paths>'
```

`<DSV41-CPU> <paths>` is the same with `WT`/`dsv41`/`DWT`:

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41 push shared dsv41
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41 && git pull --ff-only shared dsv41 && \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim \
  taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf <paths>'
```

`<GPU> <cmd>` means, from `DWT` after `git pull --ff-only shared dsv41` (the GPU lock rule applies to exit code 75):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41 && git pull --ff-only shared dsv41 && \
  ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b && \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
  SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build CUDA_HOME=/usr/local/cuda-13.2 \
  PYTHONPATH=$PWD/python:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim \
  $ANA/gpu-run.sh <cmd>'
```

`<FW-GPU> <cmd>` is the same in `DFWT` on `cc/moe-expert-plugins` (`cd .../wt-moe-plugins && git pull --ff-only shared cc/moe-expert-plugins`).

**Suites.**
- `DSV41_SUITE`: `test/registered/unit/layers/moe/test_exl3_expert_layout.py test/registered/unit/layers/moe/test_exl3_row_reader.py test/registered/unit/layers/moe/test_exl3_slot_layout.py test/registered/unit/layers/moe/test_exl3_expert_format.py test/registered/unit/layers/moe/test_exl3_shard_row_source.py test/registered/unit/layers/moe/test_exl3_stream_trace.py test/registered/unit/layers/quantization/test_exl3_method.py test/registered/unit/layers/quantization/test_exl3_moe_method.py test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py test/registered/unit/layers/quantization/test_exl3_ops_cpu.py test/registered/unit/layers/quantization/test_exl3_ext.py test/registered/unit/layers/test_engram_file_table.py test/registered/unit/layers/test_engram_row_cache.py test/registered/unit/layers/attention/test_dsv4_candidate_indexer.py test/registered/unit/models/test_deepseek_v4_exl3_weights.py test/registered/unit/model_loader/test_weight_iterator_skip.py test/registered/unit/test_file_row_reader.py test/registered/unit/test_dsv41_expert_knobs.py test/registered/unit/test_expert_stream_requirements_exl3.py test/manual/dsv41/test_ref_oracle_helpers.py test/manual/dsv41/test_trace_corpus.py test/manual/dsv41/test_tier_sim.py test/manual/dsv41/test_read_benches.py test/manual/dsv41/test_ref_oracle_lazy_experts.py` plus this plan's new dsv41 CPU files as they land.
- `FRAMEWORK_SUITE`: the list in `docs/superpowers/plans/2026-09-18-dsv41-phase3a-r2.md` ("Test suites named in steps") plus this plan's new framework CPU files. Known pre-existing failure: `test_expert_transfer.py::TestExpertRowCopySubmission::test_gpu_submission_uses_fallback_for_nonpinned_sources`.
- Task 2 records both suites' summary lines before any change (`$ANA/baseline-{dsv41,framework}-suite.txt`); later "suite unchanged" steps compare against those lines plus the new tests.

## Facts this plan relies on (read 2026-09-19 at `dsv41` `4bbaa27db6`)

- **DSV4 decode has zero graph breaks today.** Every `bcg_deepseek_v4_*` break is gated on `forward_mode.is_extend()` (`models/deepseek_v4.py:2021-2026`, `:2325-2329`, `:4211-4213`). The "~40 eager breaks/forward" in `DSV41_REFERENCE.md` §8 (the `breakable` row) and §9.3 option A is prefill-only; Task 17 corrects it.
- **The eager MoE break exists.** `FusedMoE.forward` (`layers/moe/fused_moe_triton/layer.py:1592-1605`) runs `forward_streamed_experts_eager = eager_on_graph(True)(...)` when `is_in_breakable_cuda_graph()`, the layer has `_nvfp4_expert_streamer`, and `not streamer.serves_graph_gather(topk_output)`. `serves_graph_gather` is `graph_gather_rows > 0 and 0 < topk_ids.numel() <= graph_gather_rows` (`expert_stream.py:809-816`). So once P3 enables graph gather with `graph_gather_rows = 6`, decode (6 routes) runs in-graph and prefill (> 6 routes) keeps the eager path, with no change to `FusedMoE`.
- **`get_is_capture_mode()` is true at replay too** (`runner_utils/capture_mode.py`: `is_capture_mode or is_in_breakable_cuda_graph()`, and `replay_session()` wraps replays). Only the module flag `capture_mode.is_capture_mode`, set by `model_capture_mode()` around the decode runner's whole `capture()`, separates warmup and capture from replay.
- **Capture discards routes but not counters.** `ModelRunner.init_cuda_graphs` calls `expert_hot_cache_manager.discard_graph_capture_routes()` (`model_runner.py:1329-1330`); it does not re-baseline `_last_gather`/`_last_pinned_cache_stats` (`expert_hot_cache.py:1243-1250`).
- **The graph gather seam.** `ExpertStreamer._gather_graph` (`expert_stream.py:941-1073`) plans on the device, then calls `row_backend.post(tag, plan)`, `delivery = row_backend.resolve(tag, plan)`, `row_backend.copy_residual(tag, delivery)` (`:1056-1059`) and returns `(remap, cache.tensors)`. `ExpertRowPlan` holds `expert_ids` int64 `[C]`, `slots` int32 `[C]`, `count` int32 `[1]` (`expert_row_plan.py:5-20`). `InGraphRowBackend.post` calls `copy_expert_row_segments_gpu(segments, plan.expert_ids, plan.slots, plan.count)`, whose source address is `src_base + expert_id * row_bytes` (`expert_cache_transfer.cuh:165`).
- **`enable_graph_gather` refuses the pinned tier** (`expert_stream.py:844-847`) and reads sources from `getattr(self.layer, name)` (`:854-862`); EXL3's streamed layer has no expert attributes (`exl3.py:308-313`). `require_graph_gather_support` refuses spec-only formats (`expert_format.py:255-274`). The arena requirement is at `model_runner.py:727-730` and `arg_groups/memory_hook.py:153-160`. The pinned tier is built before the hot cache (`model_runner.py:679-681`), so `enable_graph_gather` (called from `ExpertHotCacheManager.from_model`, `expert_hot_cache.py:1326-1330`) can see it.
- **The pinned tier** (`ExpertPinnedHostCache`, `expert_stream.py:128-400`): per-name page-aligned `cudaHostRegister`ed slabs `[capacity, *row_shape]` (`allocate_host_slab`, `expert_host_tier.py`), a device `expert_to_slot` int64 `[E]` refreshed in place by `_refresh_mapping` (`:232-239`), the LRU `PinnedSlotLRU` (`expert_host_tier.py:29-100`: `__contains__`, `touch`, `assign(expert, protected)`, `release(slot)`, `mapping(E)`, `slot_to_expert`, `expert_to_slot` OrderedDict), and `is_pinned` from `format.pinned_tier_options(layer)`. 3a already rewrites pinned slabs on the host and reads them zero-copy from later kernels (`_gather_host_rows_kernel`), with the oracle matching (§16.6): no stale-cache effect across kernel boundaries.
- **The EXL3 split** (`Exl3ShardRowSource.read`, `exl3_shard_row_source.py:130-194`): one page-aligned superset read per expert (`Exl3ExpertRecord.aligned_read(4096)` → `(aligned_offset, aligned_len, row_start)`, `exl3_expert_layout.py:38-42`; EOF-clamped to `min(len, file_size - offset)`), into a page-aligned bounce, then per `RowSegment(name, part, dst_offset, src_offset, nbytes)` of `Exl3ExpertFormat.segment_map()` a copy `dst[name][slot, dst_offset:+nbytes] = bounce[row_start + src_offset:+nbytes]`. The segment map is the same for every layer.
- **The doorbell** (`kernels/jit/csrc/moe/expert_doorbell.cuh`, `kernels/ops/moe/expert_doorbell.py`): request page in `pin_memory=True` host bytes read by device kernels through UVA; poster `st.release.sys` of the head after the record; completion polled with `ld.acquire.gpu`/volatile + `__nanosleep(64)`; sticky `disabled` and `fatal` words; a watchdog thread `std::abort()`s after `fatal_wait_s` (`:720-772`); `ExpertHotCacheManager.doorbell_fail_stop_check(synchronize)` (`expert_hot_cache.py:1711-1727`) is called after every batch result by `Scheduler._expert_doorbell_fail_stop_check` (`managers/scheduler.py:4776`, `:4792-4805`) and returns 0.0 at once when no doorbell exists. Its thread issues CUDA copies, which a running graph replay holds back (E32) and whose launches block (E34) — the reason its waits need drains.
- **JIT build patterns.** `load_jit` (`kernels/jit/utils/compile/loader.py:48`): CUDA modules pass `cuda_files=[...]`, `cuda_wrappers=[(name, name)]` (doorbell); a pure C++ module with its own exports passes `cpp_files=["io/uring_file_reader.cpp"]`, `extra_ldflags=["-luring"]`, `header_only=False` and exports with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` (`uring_file_reader.cpp:702`). `test/registered/unit/kernels/test_uring_file_reader.py` compiles and runs that module in the CPU suite.
- **exllamav3 at `02aef45`** (divix01 `/data/models/slang/nvfp4-work/exllamav3`): `ext.exl3_moe(hidden fp16 [T,H], out fp32 [T,H], expert_count int64 [E+1], token_sorted int64 [A], weight_sorted fp16 [A], temp_state_g, temp_state_u fp16 [C,R,H], temp_intermediate_g, temp_intermediate_u fp16 [C,R,I], act_function, K_gate, K_up, K_down, gate_ptrs_{trellis,suh,svh}, up_ptrs_{...}, down_ptrs_{...} int64 [E], gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1, act_limit, num_active, output_scratch fp32 [slots,H] | None, fused_base int64 | None, count_lo, count_hi, m_tile)` (`exllamav3_ext/quant/exl3_moe.cu:140-200`); `num_active=-1` is the all-fused launch; `ext.exl3_moe_gather(out, scratch, flat_expert, inv_order, expert_start, slot_base, slot_kind, weight_sorted)`; `ext.exl3_moe_max_concurrency(device_index)` gives `C`; exllamav3 builds its deterministic tables on the device as `expert_start = cumsum(count) - count; tables = stack([expert_start, expert_start, (count > 0).long()])` (`exllamav3/modules/block_sparse_mlp.py:1082-1100`). Pointer tables are raw `data_ptr()`s per expert (`modules/multilinear.py:32`, `:106-108`). Every EXL3 linear in V4.1 uses `mcg=False, mul1=True` (`exl3_ops.py` module docstring). `exl3_moe_loop` applies the route weight **before** `w2`; `exl3_moe` applies it after `down`.
- **Model constants:** `hidden_size` 5120, `moe_intermediate_size` 2304, 384 routed experts, top-6, `swiglu_limit` 10.0, `routed_scaling_factor` 1.5; 13,315,584 B per streamed row; Window C budgets `SGLANG_MOE_PINNED_HOST_MB=71680` (5,644 rows, 141–142/layer), `SGLANG_MOE_HOT_GPU_MB=14336` (1,128 slots).
- **Baseline** (`prof-summary.md`, eager, Window C env): ~0.69 s/token untraced; model compute ~19 ms/token; RAM misses 16.9/token; read+split 182 ms/token; `_gather_host_rows_kernel` 172 ms/token; ~780 D2H + ~780 syncs/token; host CPU outside CUDA ~340 ms/token. Window C cold arm sessions 0–3 (`ANA3A/corpus-cold.json`): TTFT 94.8/55.3/53.8/55.9 s, decode tok/s 1.433/1.768/1.510/1.947 (mean 1.665).

## Design decisions

Each item states the choice and why. "Open" items name what would change the choice.

### Phase structure and the gate

- **D1. P2 is a hard gate and runs first.** Task 1 (the probe) uses only code already on `dsv41`, so it runs before any other task. **If it fails its bar, Tasks 6–16 are blocked**: the ledger records the numbers, the P1 tasks (2–5) still run (graph decode with the eager MoE break needs no fused kernel), Task 17 writes the short record (§17.1–17.2), and execution stops. Why first: P3's whole compute path is `ext.exl3_moe` under capture; P4 only makes sense if P3 can exist.
- **D2. Graph shape.** Decode `breakable`, `cuda_graph_bs_decode=[1]`, `cuda_graph_max_bs_decode=1`, prefill `disabled`. Decode `full` stays refused: Engram's file-table lookup (2 layers) remains an eager break. Overlap scheduling stays **on** (§9.3 acceptance asks for capture under breakable + overlap; D4.12 explains why the fail-stop tolerates it).
- **D3. Numerics bars.** R4 (Task 16) on `T3`, four arms per config: eager (`exl3_moe_loop`, graph gather off), graph, an eager control, and — for option C — **debug-eager** (`--debug-cuda-graph`: the same fused in-graph path run eagerly through the capture machinery, graph gather on). Bars:
  - P1 config (eager MoE break): graph vs eager passes R4 (identical greedy tokens over 32, max |Δlogprob| ≤ 1e-3). A failure is a capture bug.
  - Option C (a), the capture-correctness gate: graph vs debug-eager **bitwise identical** tokens and logprobs (FUSED_DET makes both deterministic).
  - Option C (b): debug-eager vs eager (the deliberate kernel change: weight after `down`, fp16 intermediates) is reported against R4's bar. **Decision rule (controller ruling):** if (b) fails while (a) passes, record `R4: FAIL (fused-kernel numerics)` with `first_token_mismatch`, `max_abs_dlogprob` and the probe's per-layer `rel_fused`/`rel_loop`, and continue; this does **not** block declaring P3 done; Task 17 reports it. If (a) is not bitwise, it is still accepted when the tokens are identical and its gap is no larger than P1's eager-vs-graph gap (capture effects outside the MoE); otherwise P3 is not done: fix the capture bug (at most 2 rounds, then record `graph_vs_debug: FAIL` and continue; see Task 16 Step 5).
  - Probe (Task 1): see D7.

### P3 — in-graph MoE

- **D4. A format declares where graph sources live; the framework serves `pinned_tier`.** New attribute `ExpertFormat.graph_source_kind: str` (`"dense"` default for every existing format through `graph_source_kind_of(format)`; EXL3 sets `"pinned_tier"`). `require_graph_gather_support` accepts a spec-only streamer when its kind is `pinned_tier` and it has a pinned tier. `enable_graph_gather` then:
  - takes sources from `streamer.pinned_host_cache.tensors` instead of layer attributes and lifts the pinned-tier refusal for this kind only;
  - builds `ExpertRowSegments` over `(pinned slab[name], cache.tensors[name])`;
  - installs `PinnedTierRowBackend(segments, host_row_map, max_rows)` instead of `InGraphRowBackend`.

  `ModelRunner.maybe_init_expert_hot_cache` and `memory_hook` require the host arena only when some streamed format's kind is `dense`. Why a kind, not a new boolean: the two kinds differ in *where the source row index comes from*, which is what the backend needs to know.
- **D5. Host-slot indirection lives in the row backend, not the planner.** `PinnedTierRowBackend.post` computes `host_rows[:C] = translate(plan)` into a fixed int64 buffer and copies `copy_expert_row_segments_gpu(segments, host_rows, plan.slots, plan.count)`. `ExpertRowPlan.expert_ids` keeps expert ids (counters, residency, prefetch stay correct). In P3 `translate` is `host_row_map.index_select(0, plan.expert_ids)` over the device `pinned.expert_to_slot`, then `ram_miss[0] += (host_rows[:count] < 0).sum()`, `keep = (ram_miss_this_call == 0)` as fp32 `[1]`, and `host_rows.clamp_(min=0)` so a missing row copies slot 0 (a valid address) instead of faulting. All fixed-shape device ops: capture-safe. P4 overrides `post`/`resolve` (D4.x) and keeps `copy_residual` as the copy.
- **D6. EXL3 compute over slots.** `Exl3FusedMoE` (EXL3 side, `layers/quantization/exl3_fused_moe.py`) owns per layer:
  - nine `int64 [S]` pointer tables, `S = cache.capacity + cache.scratch_rows`, built once from `cache.tensors[name][slot, part].data_ptr()` (gate = `w13` part 0, up = `w13` part 1, down = `w2` part 0). `cache.tensors` are never reallocated after startup (checked by the framework's `_check_graph_sources`);
  - static buffers: `expert_count int64 [S+1]`, `ones int64 [6]`, `token_sorted`, `weight_sorted fp16 [6]`, `inv_order`, `scratch fp32 [6, H]`, `out fp32 [1, H]`, `x16 fp16 [1, H]`; the four `temp_*` buffers `[C, R, H|I]` (`C = ext.exl3_moe_max_concurrency(device)`, `R = 16`: one route per slot at BS1, `count_hi = R`) are **one set shared by every layer** (layers run one after another on one stream; per-layer sets would cost ~400 MB);
  - `run(x, topk_weights, remap, keep)`: `expert_count.zero_().index_add_(0, remap, ones)` (not `bincount`, which reads its max to the host), then `expert_count *= (keep > 0)` so a dropped layer runs no expert at all (no NaN from half-written rows times 0); `order = argsort(remap)`, deterministic tables as exllamav3 builds them, `ext.exl3_moe(..., num_active=NUM_ACTIVE, output_scratch=scratch, fused_base=tables[0], count_lo=1, count_hi=16, m_tile=16)`, then `ext.exl3_moe_gather`. `NUM_ACTIVE` is the probe's `"num_active"` output: 6 (BS1 top-6 slots are always distinct, so the static constant gives each expert ~28 SMs instead of 8) when its parity and bitwise replay hold, else −1. Deterministic accumulation (FUSED_DET) is chosen so graph replays are bitwise reproducible, which R4's isolation arm needs. `keep` also multiplies `weight_sorted`.
  - `Exl3MoEMethod.apply` takes `_apply_graph` when `streamer.serves_graph_gather(topk)`; `assert_not_capturing` moves into the eager branches only.
- **D7. Probe bar (Task 1), for each of `num_active` 6 and −1; the gate passes when either passes.** Reference: per-expert fp32 `exl3_linear_reference` (reconstruct + fp32 matmul), SiLU·clamp, route weight, fp32 sum. For 8 route sets on layer 3: `rel(y) = ‖y − ref‖₂ / ‖ref‖₂`. Pass needs all of:
  1. `rel(exl3_moe) ≤ 1.2e-2` (the m = 1 bound ruled in §15.1, where `exl3_gemm` measured 0.66–0.79%) **and** `rel(exl3_moe) ≤ 2·rel(exl3_moe_loop) + 1e-3` (no worse than the loop beyond fp16-intermediate noise: `exl3_moe` weights after `down`, the loop before `w2`);
  2. capture in `torch.cuda.graph` over slot pointer tables, then 4 in-place rewrites of slot rows, remap and routes: each replay **bitwise equal** to an eager deterministic call on the same inputs;
  3. `exl3_moe`, `exl3_moe_gather`, `exl3_moe_max_concurrency` exist in the sglang-built extension.

  Reported, not gated: max abs error, µs per call eager and replayed.

### P4 — option C

- **D8. Where it lives.** EXL3 side, on `dsv41` (R3 prefers it; the reader and split are EXL3-specific): `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (C++ thread, no CUDA), `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` (device kernels), `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (wrappers), `python/sglang/srt/layers/moe/exl3_ram_miss.py` (service, slot table proxy, row backend). Framework edits are limited to the seams in D4, D13 and D14.
- **D9. The thread makes no CUDA calls.** Every write it does is a CPU store into pinned host memory: the six per-name pinned slabs, the host-mapped slot map, the page. The in-graph gather already reads the slabs zero-copy. Why: the doorbell's hard cases (E32: replays hold back copies queued on another stream; E34: a launch blocks while a kernel spins) come from its thread issuing CUDA work while a device kernel waits. With no CUDA calls there is nothing for a spinning wait kernel to block, no drain kernel is needed, and the thread is testable on a CPU-only host (D15).
- **D10. Request page (host bytes, `torch.empty(..., pin_memory=True)`, zeroed; device reads/writes it through UVA). Little-endian u32 words.**

  | Offset | Field | Writer | Meaning |
  |---:|---|---|---|
  | 0 | `demand_head` | device post kernel (`st.release.sys`) | last posted demand sequence (0 = none) |
  | 4 | `demand_done` | thread (`__atomic_store_n` release) | last demand sequence fully served |
  | 8 | `fatal` | wait kernel (`st.release.sys`, max) | first sequence that timed out or failed; sticky, 0 = none |
  | 12 | `stop` | host Python | thread exit request |
  | 16 | `advise_head` | device post kernel | last posted advisory sequence (P5) |
  | 20 | `advise_done` | thread | last advisory sequence consumed |
  | 24 | `thread_busy_seq` | thread | sequence being served (watchdog) |
  | 28 | `thread_heartbeat` | thread | incremented every spin batch (watchdog) |
  | 32..63 | reserved | — | zero (row counters live in the C++ object, per layer, demand and advisory apart) |
  | 64 | demand ring: 16 records × 128 B | device | record `k` holds sequence `s` with `(s - 1) % 16 == k` |
  | 2112 | advisory ring: 64 records × 128 B | device | same layout |

  Record (128 B): `u32 seq; u16 layer; u16 need_count; u16 protect_count; u16 status (thread: 0 pending, 1 served, 2 failed); u32 after (advisory: the demand sequence it was posted after; 0 for demands); i32 need[8]; i32 protect[8]; u8 pad[48]`. The writer stores the payload, fences system-wide, then stores `seq` last; the reader re-reads `seq` after the payload (a seqlock), so a lapped or torn record is an overrun, never a mixed one. `need` = routed ids whose map entry was −1 at post time, deduped (BS1 top-6 ids are distinct; the kernel still dedupes); `protect` = all routed ids of the layer. Sizes: 64 + 16·128 + 64·128 = 10,304 B → one 12 KiB allocation.

  **Slot map:** `int32 [L_streamed, E]` pinned host (`pin_memory=True`), row `r` = streamed layer index `r` (the dense `layer_index_of[layer_id]`), −1 = not in RAM. Written only by the thread (and by the Python proxy under the tier mutex while the thread is paused). The device reads it with plain loads **only after** a wait kernel observed completion (D11).
- **D11. Memory ordering.**
  - Post kernel (1 block; thread 0): writes `row, counts, status=0, after`, the ids, `__threadfence_system()`, the record's `seq` word, `__threadfence_system()`, then `st.release.sys.global.u32 demand_head, seq`. Host: `seq = __atomic_load_n(&demand_head, __ATOMIC_ACQUIRE)`, then the seqlock read of the record.
  - Thread publish: row bytes (CPU `memcpy` out of the bounce, which glibc may do with non-temporal stores), `_mm_sfence()`, then map stores `slot_map[r][e] = slot`, then `_mm_sfence()`, `record.status = 1`, then `__atomic_store_n(&demand_done, seq, __ATOMIC_RELEASE)`. The explicit fences order the non-temporal stores; x86 TSO orders the rest for PCIe readers (device reads of host memory snoop CPU caches).
  - Wait kernel (1 block; thread 0 polls): `ld.acquire.sys.global.u32 demand_done` until `reached(done, seq)` or the timeout; after success `__threadfence_system()`, then it reads `status` and the map. Later kernels (the segment copy, `exl3_moe`) are stream-ordered after it, so they see the slab bytes. Evidence that zero-copy reads of rewritten slabs see fresh bytes across kernel boundaries: 3a (Facts). Task 13's GPU step runs `test_exl3_ram_miss_cuda.py::test_a_rewritten_slot_is_read_fresh`, which checks it on sm_120.
  - Eviction: the thread stores `slot_map[r][victim] = -1` **before** overwriting the victim slot's bytes. No device reader can be reading the victim slot then (D12).
- **D12. Pinned-slot ownership during replay: C++ owns it, always, once option C is on.** `Exl3RamTier` (C++) holds, per streamed layer, `slot_to_expert[capacity]`, `slot_state[capacity]` (FREE/LOADING/READY), an LRU list, and a `hot[E]` byte map; one `std::mutex` guards all of it. Python's `ExpertPinnedHostCache` uses it through `NativePinnedSlotTable` (D13), which takes the same mutex per call. Rules:
  - victims are chosen from READY slots whose expert is neither `hot` (inclusive tier) nor in the request's `protect` set nor in the thread's current in-flight set; LOADING slots are never victims;
  - **protection of rows the current layer needs**: the thread serves a demand for layer `r` by recomputing the missing set itself = `protect` ids that are not READY in `r` (not only the posted `need`), so an advisory eviction that raced the post cannot leave a routed row missing; then it reads that set;
  - ordering argument: the demand for layer `r` is posted after every earlier gather of layer `r` in stream order, and the thread evicts in tier `r` only while serving a request for `r`, so no in-flight kernel reads a victim slot. Advisories for `r+1` are posted during layer `r`, after the previous token's layer `r+1` gather (stream order), and never evict a `protect` id of their own request;
  - eager callers (prefill's `ensure_rows`, boundary promotions reading `_expert_to_slot`) run between `slot_table.before_host_use()` and `slot_table.after_host_use()` (a decorator on `lookup`, `ensure_rows`, `copy_rows`, `gather_rows`; nesting is counted). `before_host_use` synchronizes the current stream, then **pauses** the thread: a handshake in which the thread asks any advisory in flight to give up at its next row, skips advisories posted so far, and acknowledges only between two requests, so no advisory can pick a victim while Python assigns or fills slots. It is bounded by 2 × the per-request timeout + 1 s, else it raises. It then refreshes `pinned.expert_to_slot` from the C++ map. `after_host_use` resumes the thread (also skipping advisories posted during the pause). Eager reads still go through the Python row source; only the LRU bookkeeping is C++.
  - The hot map is pushed by `Exl3RamMissService.on_residency(layer_id, slot_to_expert)`, a residency listener (D14), at every residency boundary and once at startup.
- **D13. Framework seam for the slot table.** `pinned_tier_options(layer)` may return `"slot_table": PinnedSlotTable`; `ExpertPinnedHostCache` uses it instead of `PinnedSlotLRU` and wraps `lookup`, `ensure_rows`, `copy_rows` and `gather_rows` in `self._lru.before_host_use(self)` / `after_host_use(self)` (no-ops on `PinnedSlotLRU`). A table with `bind_capacity(capacity)` learns the tier's capacity before the capacity check. `PinnedSlotTable` is a `typing.Protocol` in `expert_host_tier.py` with exactly the members `ExpertPinnedHostCache` and `_prepare_promotion` use: `capacity`, `slot_to_expert`, `expert_to_slot` (Mapping), `__contains__`, `touch`, `assign`, `release`, `mapping`, `before_host_use`, `after_host_use`.
- **D14. Fail-stop reuses the doorbell's scheduler hook.** `ExpertHotCacheManager` gains `fail_stop_checks: list[Callable[[], None]]` and `register_fail_stop_check(fn)`; `doorbell_fail_stop_check` runs every registered check first (each raises on failure), then the doorbell's own logic unchanged. The scheduler call site stays as it is (no `scheduler.py` edit). It also gains `residency_listeners` notified after `_update_residency`/startup allocation with `(layer_id, hot.slot_to_expert)` (D12).
- **D15. The wait: bounded, single block, fail-stop on timeout.**
  - Budget in nanoseconds from `%globaltimer`, `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS` (default 2000). Why 2 s: a worst-case layer needs ≤ 6 reads; measured 8.1 ms read + 2.2 ms split per row (§16.12) → ~62 ms serial, so 2 s only fires on a hung read or a dead thread, never on a slow drive. Poll loop `ld.acquire.sys` + `__nanosleep(256)`: one resident block of 32 threads, so at most one SM is held.
  - Sticky fast path: the post and wait kernels read `fatal` first; once it is non-zero they post nothing, set `keep = 0`, and return immediately, so a failed forward finishes its remaining ~40 layers in microseconds instead of 40 × 2 s.
  - On timeout, or on `status == 2` (read failed): `fatal = seq` (only if 0), `keep = 0`, device counters `timeouts`/`failures` += 1. The layer's MoE output is zero for that forward (a one-forward drop, the §9.3 "safety valve"), then the process stops.
  - **Stop:** the registered check (D14) reads `fatal` from the host page (no device sync) after every batch result (0xFFFFFFFF marks a planned row found missing after a served request, `unserved_misses`); non-zero raises `RuntimeError("exl3 RAM miss: request <seq> timed out or failed; fail-stop")`, which ends the scheduler. **Watchdog** (C++ thread, 20 ms tick, no CUDA): aborts (`prctl(PR_SET_DUMPABLE, 0); std::abort()`) when `fatal` has been non-zero for longer than `fatal_wait_s` (30 s: the scheduler did not stop) or when one request has been in service longer than `fatal_wait_s` (a hung io_uring read). **There is no copy-from-RAM fallback**: nothing retries a row that is not in RAM.
  - Latency of the stop: the forward that timed out finishes with a dropped layer; without overlap its tokens are the last emitted; with overlap one more forward may run (its layers all take the sticky fast path) before the check raises. Recorded in §17 as a known property.
- **D16. The split in C++.** Python hands the thread flat tables at start (`Exl3RamMissTables`, built by `exl3_ram_miss_tables(layout, format, pinned caches)`):
  - `paths: list[str]` (shard files), `files: int64 [F, 1]` file sizes;
  - `reads: int64 [L, E, 4]` = `(file_index, aligned_offset, aligned_len, row_start)` from `record.aligned_read(4096)`;
  - `segments: int64 [S, 4]` = `(name_index, dst_offset, src_offset, nbytes)` from `segment_map()` in `EXL3_STREAMED_NAMES` order;
  - `slabs: int64 [L, 6]` base addresses of each layer's per-name slabs, `row_bytes: int64 [6]`, `capacity: int64 [L]`.

  The thread reads with its own liburing ring (depth 16, `O_DIRECT` unless `direct=0`), into a `posix_memalign(4096)` bounce of 8 × `slot_bytes`, checks `res == min(aligned_len, file_size − aligned_offset)`, then `memcpy`s each segment into `slab[name] + slot·row_bytes[name] + dst_offset`. Task 10's CPU test compares it byte-for-byte with `Exl3ShardRowSource.read` on the fake checkpoint.
- **D17. Build/JIT.** Two modules, each built by `load_jit` like an existing one:
  - `exl3_ram_miss_host` (built up over Tasks 10–12): `cpp_files=["moe/exl3_ram_miss_host.cpp"]`, `extra_ldflags=["-luring", "-lpthread"]`, `header_only=False`, free functions exported with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` (the `uring_file_reader` pattern). No CUDA headers, so it compiles and runs in the CPU suite.
  - `exl3_ram_miss`: `cuda_files=["moe/exl3_ram_miss.cuh"]`, `cuda_wrappers=[("exl3_ram_miss_post", ...), ("exl3_ram_miss_wait", ...)]` (the doorbell pattern).
- **D18. Doorbell reuse = copied patterns, not shared code.** The page words, release/acquire helpers, `reached()` wraparound compare, the watchdog loop and the scheduler hook path are the doorbell's designs, re-implemented in the new files. `expert_doorbell.cuh`/`.py` are not edited, so Qwen's production behaviour cannot change. The alternative, extracting a shared header, would touch production code for no functional gain.
- **D19. Thread core and spin.** `cpu_core` argument (default −1 = inherit the process affinity; the window's GPU commands run under `taskset -c 32-63`, so the thread lands there). It spins with `_mm_pause()` while a replay is likely (last request < 5 ms ago), else sleeps 50 µs between polls. It never uses cores 64–71.
- **D20. Capture and warmups.** Warmup forwards run the kernels eagerly and post real requests, which the thread serves; the P1 guards keep them out of the trace and residency counters. Capture records the kernels without running them. No counter is reset after capture: the graph trace (D23) records per-step deltas, and the device `timeouts`/`failures` words must stay 0 through warmup anyway (a warmup timeout is a real failure and fails stop at startup).
- **D21. Overlap scheduling is allowed.** Unlike the doorbell, the check needs no stream sync (it reads a host word) and there is no late copy that could land on rows a later forward uses in a way that matters: a fatal process only produces dropped-layer tokens until it stops.

### P5 — prefetch

- **D22. Advisory posts, one env var.** `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` (bool, default False). When on, layer `r`'s post kernel also (a) saves its routed ids into a device `last_routes int32 [L, 6]` row `r` and (b) posts an advisory record for layer `r+1` with `need = last_routes[r+1]` ids whose map entry is −1 (the previous token's routes for `r+1`), `protect = need`. The thread serves demands first: it checks `demand_head` (and a pause request) before each advisory row, and an advisory row already in flight finishes (≤ ~10 ms) before a demand starts; an abandoned advisory releases its rows. Advisory records whose layer already has a newer demand are skipped. The post kernel posts a record for **every** MoE layer in both modes: with `need` empty it is a touch-only record that keeps the C++ LRU's recency right. The wait is armed when something is needed, and always with prefetch on, so the thread's re-check (D12) closes the advisory-eviction race; with prefetch off, a layer with no `need` does not wait. Known limit: the previous token's routes for `r+1` were served one token ago and are usually still in RAM, so a low `advisory_rows` measures this predictor, not the mechanism; another predictor writes its ids into `last_routes[r+1]` before layer `r`'s post.

### Measurement and trace

- **D23. `G` and `f` in graph mode.** The in-graph MoE produces no host `ExpertGatherStats`, and each streamer's `graph_counters` are zeroed every forward by the manager's `_accumulate_registers` (run by the forward observer, before the scheduler's per-batch hook). So the per-batch check reads the **manager's registers**: `G` = the per-batch delta of `sum(r["graph_rows"] for r in manager._registers.values())` (routed rows, routed misses), with the manager handle passed in `attach`; RAM misses = the delta of the thread's per-layer **demand** row counters (advisory rows are counted apart, so `f` excludes prefetch traffic). `Exl3StreamTrace.record_graph_step()` runs from the registered check only when the trace is enabled (`Exl3StreamTrace.enabled`, true when `SGLANG_DSV41_EXPERT_TRACE_PATH` is set) and writes one `"kind": "graph_step"` JSONL line per graph decode step; `tier_sim.load_trace`/`live_summary` count such a line as one decode forward with `vram_misses`/`ram_misses`. One small D2H per batch, measurement runs only.


## File Structure

| File | Responsibility | Task | Branch |
|---|---|---|---|
| `test/manual/dsv41/test_exl3_moe_probe_gpu.py` (new); `$ANA/gpu-run.sh` (divix01, not in git) | P2 gate (`num_active` choice); the GPU lock rule | 1 | dsv41 |
| `python/sglang/srt/layers/moe/expert_hot_cache.py` (modify) | capture re-baseline; `fail_stop_checks`; `residency_listeners`; `_attach_formats` | 2, 7 | framework |
| `python/sglang/srt/layers/moe/expert_format.py` (modify) | `graph_source_kind_of`; `require_graph_gather_support` accepts `pinned_tier` | 6 | framework |
| `python/sglang/srt/layers/moe/expert_row_plan.py` (modify) | `PinnedTierRowBackend` | 6 | framework |
| `python/sglang/srt/layers/moe/expert_stream.py` (modify) | `enable_graph_gather` over the pinned tier; pinned cache `slot_table`, host-use decorator, `bind_capacity` | 6, 7 | framework |
| `python/sglang/srt/layers/moe/expert_host_tier.py` (modify) | `PinnedSlotTable` protocol; `PinnedSlotLRU.before_host_use`/`after_host_use` | 7 | framework |
| `python/sglang/srt/arg_groups/expert_stream_requirements.py`, `memory_hook.py` (modify) | `graph_gather_host_source`; arena rule only for `"arena"` | 6 | framework |
| `python/sglang/srt/model_executor/model_runner.py` (modify) | arena requirement skipped for pinned-tier formats | 6 | framework |
| framework tests (new) | `test/registered/unit/layers/moe/test_expert_capture_baseline.py`, `test_expert_pinned_graph_gather.py`, `test_expert_pinned_graph_gather_cuda.py`, `test_expert_pinned_slot_table.py`, `test_expert_fail_stop_checks.py` | 2, 6, 7 | framework |
| `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (modify) | decode `breakable` bs 1; graph gather over the pinned tier; prefetch rule | 3, 9, 15 | dsv41 |
| `python/sglang/srt/layers/engram.py` (modify) | file-table lookup as an eager break | 3 | dsv41 |
| `python/sglang/srt/layers/moe/exl3_stream_trace.py` (modify) | skip warmup/capture; `enabled`, `record_graph_step` | 4, 14 | dsv41 |
| `python/sglang/srt/layers/quantization/exl3.py` (modify) | capture guard for `record_routes`; `_apply_graph`; option C routes + refusal | 4, 9, 14 | dsv41 |
| `scripts/dsv41/trace_corpus.py`, `compare_oracle.py` (modify); `scripts/dsv41/graph_parity.py` (new) | `--graphs`; R4 arms with per-run env | 5 | dsv41 |
| `scripts/dsv41/tier_sim.py` (modify) | `graph_step` lines in `load_trace`/`live_summary` | 14 | dsv41 |
| `python/sglang/srt/layers/moe/exl3_expert_format.py` (modify) | `graph_source_kind = "pinned_tier"`; `slot_table` option; `attach_hot_cache_manager`; `prefetch_enabled` | 9, 14, 15 | dsv41 |
| `python/sglang/srt/layers/quantization/exl3_fused_moe.py` (new) | `Exl3FusedMoE` (pointer tables, shared temps, `NUM_ACTIVE`, `run`) | 9 | dsv41 |
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (new) | row reader + split; C++ LRU, request service, simulator; thread, pause handshake, watchdog | 10, 11, 12 | dsv41 |
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` (new) | post / wait kernels | 13 | dsv41 |
| `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (new) | `read_rows_once`; `Exl3RamMissHost`; `Exl3RamMissDevice` | 10–13 | dsv41 |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` (new) | `exl3_ram_miss_tables`; `NativePinnedSlotTable`, `Exl3RamMissRowBackend`, `Exl3RamMissService` | 10, 14 | dsv41 |
| `python/sglang/test/dsv41_ram_miss_fixtures.py` (new) | shared CPU fixture of the option C tests | 10 | dsv41 |
| `python/sglang/srt/environ.py` (modify) | `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`, `SGLANG_TEST_DSV41_RAM_MISS_FAULT`, `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` | 14, 15 | dsv41 |
| dsv41 tests (new/modify) | see each task | 1–15 | dsv41 |
| `DSV41_REFERENCE.md` (modify) | §8/§9.3 corrections, §11 status, new §17 | 17 | dsv41 |

Every task is self-contained for its implementer: it names files, commands and expected results. Line anchors cite `dsv41` `4bbaa27db6`; after an earlier task edits a file, find each `old` block by content (each is unique in its file).

**Commit message template** used by every commit step below (`<subject>` is given in the step):
```bash
git commit -m "$(printf '%s\n\n%s\n%s\n' '<subject>' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' \
  'Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF')"
```

---

## Phase P2 — the `exl3_moe` probe (HARD GATE)

### Task 1: `exl3_moe` parity and capture probe on real layer weights — **GATE for Tasks 6–16**

**Runs first and depends on nothing in this plan** (it uses only code already on `dsv41`: `exl3_ext`, `exl3_ops`, the EXL3 layout and `Exl3ShardRowSource`). It also creates the analysis directory, the ledger and the GPU lock wrapper that later tasks use.

**Gate rule:** if Step 5's verdict is FAIL, Tasks 6–16 are blocked. Record `Task 1: FAIL — <which bar>` and the probe JSON path in the ledger, still execute the P1 tasks (2–5; they need no fused kernel), then write the short record of Task 17 (its §17.1–17.2 only) and stop.

**Branch:** `dsv41`. **GPU:** Step 4 (~20 min, ~6 GiB of VRAM).

**Files:**
- Create: `test/manual/dsv41/test_exl3_moe_probe_gpu.py`

**Interfaces:**
- Consumes: `exl3_ext()` (`layers/quantization/exl3_ext.py`), `Exl3Tensors`, `exl3_linear_reference`, `exl3_moe_loop` (`exl3_ops.py`), `build_exl3_expert_layout`, `Exl3ExpertFormat(layout, layer, direct=)`, `Exl3ShardRowSource.for_layer(layout, layer, segments, direct=)`.
- Produces: `fused_moe_slots(ext, bufs, x16, weights, remap, tables, *, act_limit, bits, num_active) -> torch.Tensor` (fp32 `[1, H]`) and `FusedBuffers` inside the test module, the exact call sequence Task 9 productionizes; `$ANA/probe-exl3-moe.json` with `"verdict"` and `"num_active"` (6 or -1; Task 9 reads it); `$ANA/gpu-run.sh`; the ledger file.

- [ ] **Step 1: Write the probe**

Create `test/manual/dsv41/test_exl3_moe_probe_gpu.py`:
```python
"""P2 gate: exllamav3's fused exl3_moe over slot pointer tables, on real layer rows (GPU).

Reads 12 real experts of one layer through the model's own row source into
hot-cache-shaped slot tensors, then for 8 BS1 top-6 route sets compares:
  * exl3_moe (deterministic: output scratch + exl3_moe_gather) over the slots,
  * exl3_moe_loop over the same slots,
against an fp32 reference built from exl3_linear_reference. Then it captures the
fused call in a CUDA graph and replays it after rewriting slot rows, remap,
routes and input in place; each replay must equal an eager call bitwise.

Bars (plan Design decision D7): rel(fused) <= 1.2e-2 and
rel(fused) <= 2 * rel(loop) + 1e-3 for every route set; replay == eager bitwise.
Both num_active = 6 and num_active = -1 are measured; the report's "num_active"
is 6 when that mode passes, else -1 when it passes, and the verdict is PASS when
either passes.
Env: DSV41_EXL3_DIR (checkpoint), DSV41_PROBE_LAYER (default 3),
DSV41_PROBE_OUT (JSON report path).
"""

import json
import os
import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EXL3_DIR = os.environ.get("DSV41_EXL3_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw")
LAYER = int(os.environ.get("DSV41_PROBE_LAYER", "3"))
OUT = os.environ.get("DSV41_PROBE_OUT")
EXPERTS = list(range(0, 384, 32))  # 12 experts -> 12 slots
TOP_K = 6
ROUTE_SETS = 8
REWRITES = 4
R_ROWS = 16  # fused-kernel row tile; one route per slot at BS1
ACT_SILU = 0
ACT_LIMIT = 10.0
REL_BOUND = 1.2e-2


@dataclass
class FusedBuffers:
    expert_count: torch.Tensor
    ones: torch.Tensor
    token_sorted: torch.Tensor
    scratch: torch.Tensor
    out: torch.Tensor
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor


def make_buffers(ext, slots: int, hidden: int, inter: int, device) -> FusedBuffers:
    concurrency = ext.exl3_moe_max_concurrency(device.index)
    half = dict(dtype=torch.float16, device=device)
    return FusedBuffers(
        expert_count=torch.zeros(slots + 1, dtype=torch.long, device=device),
        ones=torch.ones(TOP_K, dtype=torch.long, device=device),
        token_sorted=torch.zeros(TOP_K, dtype=torch.long, device=device),
        scratch=torch.empty((TOP_K, hidden), dtype=torch.float32, device=device),
        out=torch.empty((1, hidden), dtype=torch.float32, device=device),
        temp_state_g=torch.empty((concurrency, R_ROWS, hidden), **half),
        temp_state_u=torch.empty((concurrency, R_ROWS, hidden), **half),
        temp_intermediate_g=torch.empty((concurrency, R_ROWS, inter), **half),
        temp_intermediate_u=torch.empty((concurrency, R_ROWS, inter), **half),
    )


def pointer_tables(slot_tensors: dict, slots: int, device) -> dict:
    """Nine int64 [slots] tables of raw row addresses: gate = w13 part 0, up = part 1, down = w2."""
    def table(name, part):
        rows = slot_tensors[name]
        return torch.tensor([rows[s, part].data_ptr() for s in range(slots)], dtype=torch.long, device=device)

    return {
        f"{proj}_{kind}": table(f"{prefix}_{kind}", part)
        for proj, prefix, part in (("gate", "w13", 0), ("up", "w13", 1), ("down", "w2", 0))
        for kind in ("trellis", "suh", "svh")
    }


def fused_moe_slots(ext, bufs: FusedBuffers, x16, weights, remap, tables, *, act_limit, bits, num_active):
    """exl3_moe over slots, deterministic accumulation; every op is capture-safe.

    ``remap`` int64 [6] names a slot per route (distinct at BS1), ``weights`` fp32 [6].
    """
    bufs.expert_count.zero_().index_add_(0, remap, bufs.ones)
    order = torch.argsort(remap)
    inv_order = torch.empty_like(order).scatter_(0, order, torch.arange(TOP_K, device=remap.device))
    weight_sorted = weights[order].to(torch.float16)
    expert_start = torch.cumsum(bufs.expert_count, 0) - bufs.expert_count
    det = torch.stack([expert_start, expert_start, (bufs.expert_count > 0).long()])
    bufs.out.zero_()
    ext.exl3_moe(
        x16, bufs.out, bufs.expert_count, bufs.token_sorted, weight_sorted,
        bufs.temp_state_g, bufs.temp_state_u, bufs.temp_intermediate_g, bufs.temp_intermediate_u,
        ACT_SILU, bits["gate"], bits["up"], bits["down"],
        tables["gate_trellis"], tables["gate_suh"], tables["gate_svh"],
        tables["up_trellis"], tables["up_suh"], tables["up_svh"],
        tables["down_trellis"], tables["down_suh"], tables["down_svh"],
        False, True, False, True, False, True,
        act_limit, num_active, bufs.scratch, det[0], 1, R_ROWS, 16,
    )
    slots = bufs.expert_count.shape[0] - 1
    ext.exl3_moe_gather(
        bufs.out, bufs.scratch, remap, inv_order,
        det[1, :slots], det[0, :slots], det[2, :slots], weight_sorted,
    )
    return bufs.out


def _load_slots(device):
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout = build_exl3_expert_layout(EXL3_DIR)
    fmt = Exl3ExpertFormat(layout, LAYER, direct=True)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    host = {name: torch.empty((len(EXPERTS),) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    source = Exl3ShardRowSource.for_layer(layout, LAYER, fmt.segment_map(), direct=True)
    source.read(torch.tensor(EXPERTS, dtype=torch.long), host)
    return {name: tensor.to(device) for name, tensor in host.items()}


def _views(slot_tensors, slot):
    from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors

    def t(prefix, part):
        return Exl3Tensors(
            trellis=slot_tensors[f"{prefix}_trellis"][slot, part],
            suh=slot_tensors[f"{prefix}_suh"][slot, part],
            svh=slot_tensors[f"{prefix}_svh"][slot, part],
            mul1=True,
        )

    return (t("w13", 0), t("w13", 1)), t("w2", 0)


def _reference(x16, weights, remap, views):
    from sglang.srt.layers.quantization.exl3_ops import exl3_linear_reference

    out = torch.zeros((1, x16.shape[1]), dtype=torch.float32, device=x16.device)
    for k, slot in enumerate(remap.tolist()):
        (gate_t, up_t), down_t = views[slot]
        gate = exl3_linear_reference(x16, gate_t).clamp(max=ACT_LIMIT)
        up = exl3_linear_reference(x16, up_t).clamp(-ACT_LIMIT, ACT_LIMIT)
        h = F.silu(gate) * up * weights[k].float()
        out += exl3_linear_reference(h.to(torch.float16), down_t)
    return out


def _rel(y, ref):
    return float((y.float() - ref).norm() / ref.norm())


def test_exl3_moe_probe():
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_loop

    ext = exl3_ext()
    missing = [n for n in ("exl3_moe", "exl3_moe_gather", "exl3_moe_max_concurrency") if not hasattr(ext, n)]
    assert not missing, f"extension lacks {missing}"
    device = torch.device("cuda", torch.cuda.current_device())
    slot_tensors = _load_slots(device)
    slots = len(EXPERTS)
    hidden = slot_tensors["w13_suh"].shape[-1]
    inter = slot_tensors["w2_suh"].shape[-1]
    bits = {
        "gate": slot_tensors["w13_trellis"].shape[-1] // 16,
        "up": slot_tensors["w13_trellis"].shape[-1] // 16,
        "down": slot_tensors["w2_trellis"].shape[-1] // 16,
    }
    tables = pointer_tables(slot_tensors, slots, device)
    bufs = make_buffers(ext, slots, hidden, inter, device)
    views = [_views(slot_tensors, s) for s in range(slots)]
    w13 = [v[0] for v in views]
    w2 = [v[1] for v in views]
    gen = torch.Generator(device="cpu").manual_seed(1234)
    report = {"layer": LAYER, "experts": EXPERTS, "bits": bits, "modes": {}}

    def inputs():
        remap = torch.randperm(slots, generator=gen)[:TOP_K].to(device)
        weights = torch.softmax(torch.randn(TOP_K, generator=gen), 0).to(device)
        x16 = (torch.randn((1, hidden), generator=gen) * 0.5).to(device, torch.float16)
        return x16, weights, remap

    route_sets = [inputs() for _ in range(ROUTE_SETS)]
    references = [_reference(x16, w, r, views) for x16, w, r in route_sets]
    loops = [
        exl3_moe_loop(x16, w.view(1, -1), r.view(1, -1), w13, w2, ACT_LIMIT).float()
        for x16, w, r in route_sets
    ]
    # BS1 top-6 slots are always distinct, so num_active = 6 is a static constant a graph
    # can bake in; -1 (all-fused, max concurrency) is the fallback (plan M1).
    for num_active in (TOP_K, -1):
        call = dict(act_limit=ACT_LIMIT, bits=bits, num_active=num_active)
        mode = {"route_sets": [], "replays": []}
        ok_parity = True
        for (x16, weights, remap), ref, loop in zip(route_sets, references, loops):
            fused = fused_moe_slots(ext, bufs, x16, weights, remap, tables, **call).clone()
            rel_fused, rel_loop = _rel(fused, ref), _rel(loop, ref)
            passed = rel_fused <= REL_BOUND and rel_fused <= 2 * rel_loop + 1e-3
            ok_parity &= passed
            mode["route_sets"].append({
                "remap": remap.tolist(),
                "rel_fused": rel_fused,
                "rel_loop": rel_loop,
                "max_abs_fused_vs_loop": float((fused - loop).abs().max()),
                "pass": passed,
            })
        # Capture over static inputs, then rewrite slot rows, remap, routes and input in place.
        x_s, w_s, r_s = inputs()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_s = fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call)
        ok_replay = True
        for step in range(REWRITES):
            perm = torch.randperm(slots, generator=gen).to(device)
            for tensor in slot_tensors.values():
                tensor.copy_(tensor[perm].clone())
            x_new, w_new, r_new = inputs()
            x_s.copy_(x_new)
            w_s.copy_(w_new)
            r_s.copy_(r_new)
            graph.replay()
            replayed = out_s.clone()
            eager = fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call).clone()
            same = torch.equal(replayed, eager)
            ok_replay &= same
            mode["replays"].append({"step": step, "bitwise_equal": same, "max_abs": float((replayed - eager).abs().max())})
        # Rewrites permuted the slots: put the original rows back for the next mode.
        slot_tensors.update(_load_slots(device))
        for name, table in pointer_tables(slot_tensors, slots, device).items():
            tables[name].copy_(table)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            fused_moe_slots(ext, bufs, x_s, w_s, r_s, tables, **call)
        torch.cuda.synchronize()
        mode["eager_us"] = (time.perf_counter() - started) * 1e4
        started = time.perf_counter()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        mode["replay_us"] = (time.perf_counter() - started) * 1e4
        mode["pass"] = ok_parity and ok_replay
        report["modes"][str(num_active)] = mode
        views = [_views(slot_tensors, s) for s in range(slots)]
    passing = [int(m) for m, v in report["modes"].items() if v["pass"]]
    # The fused MoE (Task 9) reads this choice: 6 when it passes, else -1.
    report["num_active"] = TOP_K if TOP_K in passing else (-1 if -1 in passing else None)
    report["verdict"] = "PASS" if report["num_active"] is not None else "FAIL"
    if OUT:
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({"verdict": report["verdict"], "num_active": report["num_active"],
                      **{m: (v["pass"], v["eager_us"], v["replay_us"]) for m, v in report["modes"].items()}}))
    assert report["verdict"] == "PASS", report["modes"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s"]))
```

- [ ] **Step 2: Check it collects on the CPU (skips cleanly)**

`git add test/manual/dsv41/test_exl3_moe_probe_gpu.py`; commit `test(dsv41): exl3_moe parity and capture probe over slot pointer tables (P2 gate)`; push. Run `<DSV41-CPU> test/manual/dsv41/test_exl3_moe_probe_gpu.py`. Expected: `1 skipped` (no GPU).

- [ ] **Step 3: Ledger, analysis directory, GPU lock wrapper**

```bash
mkdir -p /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC
touch /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b; mkdir -p $ANA
cat > $ANA/gpu-run.sh <<"EOS"
#!/usr/bin/env bash
# The plan's GPU lock rule: try cc-gpu.lock every 60 s for up to 30 min, then exit 75.
# Usage: gpu-run.sh <cmd> [args...]  (runs the command on cores 32-63 under the lock)
LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
for attempt in $(seq 1 30); do
  taskset -c 32-63 flock -n -E 75 "$LOCK" "$@"
  rc=$?
  [ "$rc" -ne 75 ] && exit "$rc"
  echo "[$(date --iso-8601=seconds)] cc-gpu.lock held (attempt $attempt/30); retrying in 60 s" >&2
  sleep 60
done
exit 75
EOS
chmod +x $ANA/gpu-run.sh; ls -l $ANA/gpu-run.sh'
```
Expected: the script listed as executable.

- [ ] **Step 4: Run the probe**

`<GPU> env DSV41_EXL3_DIR=/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw DSV41_PROBE_LAYER=3 DSV41_PROBE_OUT=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/probe-exl3-moe.json SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build CUDA_HOME=/usr/local/cuda-13.2 /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -s -rf test/manual/dsv41/test_exl3_moe_probe_gpu.py 2>&1 | tee /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/probe-exl3-moe.log`.

Expected: `1 passed` and a printed line `{"verdict": "PASS", "num_active": 6, ...}` (or `-1`). The probe reads 12 rows (~160 MB) through the model's row source. An `AttributeError` for `exl3_moe*` means the extension was built without the MoE sources: that is a FAIL of bar 3. Exit code 75 from `gpu-run.sh`: apply the GPU lock rule (Global Constraints).

- [ ] **Step 5: Verdict (the gate)**

```bash
ssh divix01 '/data/models/slang/.venv/bin/python -c "
import json
r = json.load(open(\"/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/probe-exl3-moe.json\"))
for m, v in r[\"modes\"].items():
    print(m, v[\"pass\"], max(s[\"rel_fused\"] for s in v[\"route_sets\"]), max(s[\"rel_loop\"] for s in v[\"route_sets\"]), all(x[\"bitwise_equal\"] for x in v[\"replays\"]), v[\"eager_us\"], v[\"replay_us\"])
print(r[\"verdict\"], r[\"num_active\"])"'
```
- `PASS <n>`: append `Task 1: complete — PASS (num_active <n>; max rel fused <x>, loop <y>; replay bitwise; <eager_us>/<replay_us> µs)` to the ledger and continue with Task 2.
- `FAIL` or the file is missing: append `Task 1: FAIL — <bar>` and apply the gate rule above.

---

## Phase P1 — graph decode foundation

### Task 2: Pre-flight baselines, and the framework's post-capture counter re-baseline (merged into `dsv41`)

**Branches:** `cc/moe-expert-plugins` (`FWT`) for Steps 2–5, then `dsv41` (`WT`) for Step 6.

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (end of `discard_graph_capture_routes`, `:1553-1557`)
- Create: `test/registered/unit/layers/moe/test_expert_capture_baseline.py`

**Interfaces:**
- Consumes: `ExpertHotCacheManager.discard_graph_capture_routes`, `_pinned_cache_stats`, `streamers`, `_last_gather`, `_last_pinned_cache_stats`.
- Produces: after `discard_graph_capture_routes()`, `_last_gather[layer] is streamers[layer].last_gather_stats` and `_last_pinned_cache_stats[layer] == _pinned_cache_stats(streamers[layer])`; `$ANA/baseline-{dsv41,framework}-suite.txt`; the ledger file.

- [ ] **Step 1: Baselines (no code change)**

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41 status --short --branch | head -3
git -C /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins status --short --branch | head -3
mkdir -p /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC
touch /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41/.superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md
ssh divix01 'mkdir -p /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b && ls /usr/include/liburing.h && nproc'
```
Expected: both worktrees clean apart from `??` entries under `.superpowers/` and the superseded `docs/superpowers/plans/2026-09-19-dsv41-phase3b1.md`; `/usr/include/liburing.h` listed; `nproc` ≥ 72.

Run `<DSV41-CPU> <DSV41_SUITE>` with `2>&1 | tail -1 | tee /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/baseline-dsv41-suite.txt` appended inside the remote command, and `<FW-CPU> <FRAMEWORK_SUITE>` with `... | tee .../baseline-framework-suite.txt`. Expected: no failure except the known `test_gpu_submission_uses_fallback_for_nonpinned_sources`. Copy both lines into the ledger.

- [ ] **Step 2: Write the failing test**

Create `test/registered/unit/layers/moe/test_expert_capture_baseline.py` in `FWT`:
```python
"""discard_graph_capture_routes re-baselines the eager counters after CUDA-graph capture (CPU)."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_discarding_capture_routes_rebaselines_the_eager_counters():
    capture_stats = object()
    pinned = SimpleNamespace(stats=SimpleNamespace(populated_rows=7, evictions=3))
    manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
    manager.streamers = {
        2: SimpleNamespace(last_gather_stats=capture_stats, pinned_host_cache=pinned),
        5: SimpleNamespace(last_gather_stats=capture_stats, pinned_host_cache=None),
    }
    manager._graph_counters = None
    manager._registers = {}
    manager.residency_policies = {}
    manager._side_pull_snapshots = {}
    manager._last_side_pull_totals = {}
    manager._inflight_promotions = []
    manager._last_gather = {2: None, 5: None}
    manager._last_pinned_cache_stats = {2: (0, 0), 5: (0, 0)}

    manager.discard_graph_capture_routes()

    assert manager._last_gather == {2: capture_stats, 5: capture_stats}
    assert manager._last_pinned_cache_stats == {2: (7, 3), 5: (0, 0)}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```

- [ ] **Step 3: Commit red, push, run red**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/moe-expert-plugins
git add test/registered/unit/layers/moe/test_expert_capture_baseline.py
```
Commit with subject `test(moe): red test that capture discards re-baseline the eager counters`. Run `<FW-CPU> test/registered/unit/layers/moe/test_expert_capture_baseline.py`. Expected: `1 failed` on `assert {2: None, 5: None} == {2: <object ...>, 5: <object ...>}`. If it fails earlier with an `AttributeError` naming a manager attribute, read `discard_graph_capture_routes` for that attribute's type, add it to the fixture empty, commit (`test(moe): complete the capture-baseline fixture (red)`), and rerun until the assertion is the failure.

- [ ] **Step 4: Implement**

In `expert_hot_cache.py` replace:
```python
        self.finish_promotions()
        if getattr(self, "gpu_residency", None) is not None:
            self.gpu_residency.reset_after_capture(self._boundary_clock)

    def _start_doorbell(
```
with:
```python
        self.finish_promotions()
        if getattr(self, "gpu_residency", None) is not None:
            self.gpu_residency.reset_after_capture(self._boundary_clock)
        # Warmup and capture forwards replaced each layer's last gather stats and
        # admitted pinned rows. Start the eager counters from here, so the first
        # real forward counts only its own.
        for layer_id, streamer in self.streamers.items():
            self._last_gather[layer_id] = streamer.last_gather_stats
            self._last_pinned_cache_stats[layer_id] = self._pinned_cache_stats(streamer)

    def _start_doorbell(
```

- [ ] **Step 5: Commit, push, run green**

`git add python/sglang/srt/layers/moe/expert_hot_cache.py`; commit `fix(moe): re-baseline eager gather counters after CUDA-graph capture`. Run `<FW-CPU> <FRAMEWORK_SUITE> test/registered/unit/layers/moe/test_expert_capture_baseline.py`. Expected: the baseline line plus `1 passed`; the known failure only.

- [ ] **Step 6: Merge into `dsv41` (plain merge, R3)**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41
git fetch shared
test "$(git rev-parse shared/cc/moe-expert-plugins)" = "$(git -C ../moe-expert-plugins rev-parse HEAD)" && echo pushed
git merge-tree --write-tree --name-only dsv41 shared/cc/moe-expert-plugins
```
Expected: `pushed`, then one tree hash and no file names (a file name means a conflict. **Conflict rule:** do not merge; `git merge --abort` if a merge was started, record `Task 2: BLOCKED — merge conflict in <files>` in the ledger, and continue with the tasks that do not depend on this merge (Task 1 and the P1 tasks never do)).
```bash
git merge --no-ff shared/cc/moe-expert-plugins -m "$(printf '%s\n\n%s\n%s\n' \
  'merge cc/moe-expert-plugins into dsv41 (post-capture counter re-baseline)' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' \
  'Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF')"
git log -1 --format='%P' | wc -w
git diff --stat shared/cc/moe-expert-plugins HEAD -- python/sglang/srt/layers/moe/expert_hot_cache.py
```
Expected: `2`; no diff output. Run `<DSV41-CPU> <DSV41_SUITE> test/registered/unit/layers/moe/test_expert_capture_baseline.py`. Expected: `baseline-dsv41-suite.txt`'s counts plus `1 passed`. Append `Task 2: complete` to the ledger.

---

### Task 3: EXL3 gate accepts breakable decode at batch size 1; Engram lookup becomes an eager break

**Branch:** `dsv41`.

**Files:**
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (module body and docstring)
- Modify: `python/sglang/srt/layers/engram.py:746-756` (`EngramEmbedding.forward`) and module level (one helper)
- Modify: `test/registered/unit/test_expert_stream_requirements_exl3.py`
- Create: `test/registered/unit/layers/test_engram_lookup_break.py`

**Interfaces:**
- Consumes: `eager_expert_stream_requirements`, `ExpertStreamRequirements`, `Backend`, `eager_on_graph`, `_current_capture_var`.
- Produces: `exl3_expert_stream_requirements` (registered for `exl3`, label `"EXL3"`); `engram._engram_file_table_lookup(file_table, indices)`.

- [ ] **Step 1: Write the failing tests**

First check `PhaseConfig`'s batch-size field names: `grep -n "class PhaseConfig" -A 25 python/sglang/srt/model_executor/cuda_graph_config.py`. The code below uses `bs` and `max_bs`; if the class names them differently, use its names everywhere below.

In `test/registered/unit/test_expert_stream_requirements_exl3.py`, add after the imports:
```python
BREAKABLE_BS1 = CudaGraphConfig(
    decode=PhaseConfig(backend="breakable", bs=[1], max_bs=1),
    prefill=PhaseConfig(backend="disabled"),
)
```
Replace the parametrized case
```python
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="breakable"), prefill=PhaseConfig(backend="disabled"))}, {}, "runs eagerly"),
```
with
```python
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="full", bs=[1], max_bs=1), prefill=PhaseConfig(backend="disabled"))}, {}, "breakable"),
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="breakable", bs=[1, 2], max_bs=2), prefill=PhaseConfig(backend="disabled"))}, {}, "max batch size 1"),
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="breakable", bs=[1], max_bs=1), prefill=PhaseConfig(backend="breakable"))}, {}, "prefill"),
```
and add:
```python
def test_breakable_decode_at_batch_size_one_passes(model_dir):
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1))
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1, disable_overlap_schedule=True))
```

Create `test/registered/unit/layers/test_engram_lookup_break.py`:
```python
"""Engram's file-table lookup is an eager break under a breakable graph capture (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import engram
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Table:
    def __init__(self):
        self.calls = 0

    def lookup(self, indices):
        self.calls += 1
        return indices.float().unsqueeze(-1) * 2


def test_outside_capture_the_lookup_runs_directly():
    table = _Table()
    out = engram._engram_file_table_lookup(table, torch.tensor([[1, 2]]))
    assert table.calls == 1 and out.tolist() == [[[2.0], [4.0]]]


def test_under_capture_the_lookup_ends_the_segment_and_records_a_replay():
    events = []
    capture = SimpleNamespace(
        _end_current_segment=lambda: events.append("end"),
        _begin_new_segment=lambda: events.append("begin"),
        _barrier_fn=None,
        cuda_graph=SimpleNamespace(_break_fns=[]),
    )
    table = _Table()
    token = bcg._current_capture_var.set(capture)
    try:
        out = engram._engram_file_table_lookup(table, torch.tensor([[3]]))
    finally:
        bcg._current_capture_var.reset(token)
    assert events == ["end", "begin"]
    assert len(capture.cuda_graph._break_fns) == 1
    assert table.calls == 1 and out.tolist() == [[[6.0]]]
    capture.cuda_graph._break_fns[0]()  # a replay re-runs the lookup and copies into `out`
    assert table.calls == 2


def test_the_embedding_forward_uses_the_break():
    table = _Table()
    module = SimpleNamespace(file_table=table)
    out = engram.EngramEmbedding.forward(module, torch.tensor([[5]]))
    assert out.tolist() == [[[10.0]]] and table.calls == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```
Before writing the last test, read `EngramEmbedding.forward` (`engram.py:746-760`): if it reads any attribute of `self` before the `file_table` branch, add it to the `SimpleNamespace` with the value that reaches that branch.

- [ ] **Step 2: Commit red, push, run red**

`git add test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/test_engram_lookup_break.py`; commit `test(dsv41): red tests for breakable decode at bs 1 and the Engram lookup break`. Run `<DSV41-CPU> test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/test_engram_lookup_break.py`. Expected: the four new gate cases fail (`runs eagerly` raised, or `DID NOT RAISE` for `full`); the three Engram tests fail with `AttributeError: module 'sglang.srt.layers.engram' has no attribute '_engram_file_table_lookup'`; the old cases pass.

- [ ] **Step 3: Implement the gate**

Replace everything after the docstring of `expert_stream_requirements_exl3.py` with:
```python
from sglang.srt.arg_groups.expert_stream_requirements import (
    ExpertStreamRequirements,
    eager_expert_stream_requirements,
    register_expert_stream_requirements,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend

_EAGER = eager_expert_stream_requirements(
    "EXL3",
    enabled=lambda: envs.SGLANG_DSV41_EXPERT_STREAM.get(),
    enable_hint="SGLANG_DSV41_EXPERT_STREAM=1",
)


class _EagerGraphView:
    """``cfg`` with no graph config, so the shared eager checks skip their graph rule."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def __getattr__(self, name):
        if name == "cuda_graph_config":
            return None
        return getattr(self._cfg, name)


def _check(cfg, budgets) -> None:
    """The eager checks, with decode allowed as a breakable CUDA graph at max batch size 1.

    Everything that still needs the host (the eager MoE fallback, the Engram
    file-table lookup) runs as an eager break; prefill stays eager. Decode
    ``full`` cannot work: the Engram lookup reads its ids on the host.
    """
    graph = cfg.cuda_graph_config
    if graph is None or graph.decode.backend == Backend.DISABLED:
        _EAGER.check(cfg, budgets)
        return
    if graph.decode.backend != Backend.BREAKABLE:
        raise ValueError(
            "EXL3 expert caching needs --cuda-graph-backend-decode breakable "
            "(or disabled); full decode graphs cannot run the Engram file-table lookup"
        )
    if (graph.decode.max_bs or 0) != 1:
        raise ValueError(
            "EXL3 expert caching captures decode graphs at max batch size 1 only; "
            "pass --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1"
        )
    if graph.prefill.backend != Backend.DISABLED:
        raise ValueError(
            "EXL3 expert caching runs prefill eagerly; pass --cuda-graph-backend-prefill disabled"
        )
    _EAGER.check(_EagerGraphView(cfg), budgets)


exl3_expert_stream_requirements = ExpertStreamRequirements("EXL3", _check)
register_expert_stream_requirements(("exl3",), exl3_expert_stream_requirements)
```
In the module docstring replace "EXL3 experts stream eagerly only: no CUDA graphs, no graph gather, no host arena," with "EXL3 experts stream with prefill eager and decode eager or a breakable CUDA graph at max batch size 1; no host arena,".

- [ ] **Step 4: Implement the Engram break**

In `python/sglang/srt/layers/engram.py`, if the module does not import `eager_on_graph` yet, add:
```python
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
```
then add at module level, after the imports:
```python
@eager_on_graph(True)
def _engram_file_table_lookup(file_table, indices):
    """The file-table lookup reads its ids on the host, so under a breakable CUDA
    graph it runs as an eager break; outside a capture the decorator is transparent."""
    return file_table.lookup(indices)
```
In `EngramEmbedding.forward` replace `            return self.file_table.lookup(indices)` with `            return _engram_file_table_lookup(self.file_table, indices)`.

- [ ] **Step 5: Commit, push, run green**

`git add python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py python/sglang/srt/layers/engram.py`; commit `feat(dsv41): breakable decode graphs at bs 1 for EXL3; Engram lookup as an eager break`. Run `<DSV41-CPU> <DSV41_SUITE> test/registered/unit/layers/test_engram_lookup_break.py`. Expected: previous counts + 4 gate cases + 3 Engram tests; no failure. Append `Task 3: complete`.

---

### Task 4: Warmup and capture forwards stay out of the stream trace and route recording

**Branch:** `dsv41`.

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_stream_trace.py` (module function; first line of `Exl3StreamTrace.record`)
- Modify: `python/sglang/srt/layers/quantization/exl3.py` (`_apply_streamed`)
- Modify: `test/registered/unit/layers/moe/test_exl3_stream_trace.py`, `test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py`

**Interfaces:**
- Consumes: module flag `sglang.srt.model_executor.runner_utils.capture_mode.is_capture_mode`.
- Produces: `exl3_stream_trace.capturing_graphs() -> bool`; `record` and the `record_routes` call are no-ops while it is True.

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/unit/layers/moe/test_exl3_stream_trace.py` (add `import torch` if absent; construct `Exl3StreamTrace` the way the file's other tests do if it needs arguments):
```python
def test_capture_forwards_are_not_recorded(monkeypatch):
    from sglang.srt.layers.moe import exl3_stream_trace as module
    from sglang.srt.model_executor.runner_utils import capture_mode

    trace = module.Exl3StreamTrace()
    monkeypatch.setattr(capture_mode, "is_capture_mode", True)
    assert module.capturing_graphs()
    trace.record(0, torch.zeros((1, 6), dtype=torch.long), None, 0)
    assert trace.forwards == 0 and trace.decode_tokens == 0
    monkeypatch.setattr(capture_mode, "is_capture_mode", False)
    assert not module.capturing_graphs()
    trace.record(0, torch.zeros((1, 6), dtype=torch.long), None, 0)
    assert trace.forwards == 1 and trace.decode_tokens == 1
```
Append to `test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py`:
```python
def test_streamed_apply_skips_route_recording_while_capturing(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.srt.model_executor.runner_utils import capture_mode

    recorded = []

    class _Streamer:
        background_read_stats = type("S", (), {"rows": 0})()
        last_gather_stats = None

        def record_routes(self, routed):
            recorded.append(routed.tolist())

        def iter_gather_experts(self, source_ids):
            return iter(())

    layer = type("L", (), {"layer_id": 0})()
    x = torch.zeros((1, 8), dtype=torch.float16)
    weights = torch.ones((1, 2), dtype=torch.float32)
    ids = torch.tensor([[1, 2]])
    monkeypatch.setattr(capture_mode, "is_capture_mode", True)
    Exl3MoEMethod._apply_streamed(layer, _Streamer(), x, weights, ids, None)
    assert recorded == []
    monkeypatch.setattr(capture_mode, "is_capture_mode", False)
    Exl3MoEMethod._apply_streamed(layer, _Streamer(), x, weights, ids, None)
    assert recorded == [[1, 2]]
```

- [ ] **Step 2: Commit red, push, run red**

`git add` both test files; commit `test(dsv41): red tests that capture forwards skip the trace and route recording`. Run `<DSV41-CPU> test/registered/unit/layers/moe/test_exl3_stream_trace.py test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py`. Expected: the two new tests fail (`capturing_graphs` missing; `recorded == [[1, 2]]` during capture).

- [ ] **Step 3: Implement**

In `exl3_stream_trace.py` add at module level:
```python
def capturing_graphs() -> bool:
    """True during the decode runner's warmup and capture forwards, False at replay.

    ``get_is_capture_mode()`` is also true at every breakable replay, so only the
    module flag that ``model_capture_mode()`` sets separates capture from serving.
    """
    from sglang.srt.model_executor.runner_utils import capture_mode

    return bool(capture_mode.is_capture_mode)
```
and make the first statements of `Exl3StreamTrace.record`:
```python
        if capturing_graphs():
            return
```
In `exl3.py`, import `capturing_graphs` next to the existing `get_exl3_stream_trace` import, and in `_apply_streamed` replace `        streamer.record_routes(routed)` with:
```python
        if not capturing_graphs():
            # Warmup and capture forwards route dummy tokens; they must not move residency.
            streamer.record_routes(routed)
```

- [ ] **Step 4: Commit, push, run green**

`git add python/sglang/srt/layers/moe/exl3_stream_trace.py python/sglang/srt/layers/quantization/exl3.py`; commit `fix(dsv41): keep warmup and capture forwards out of the trace and residency routes`. Run `<DSV41-CPU> <DSV41_SUITE> test/registered/unit/layers/test_engram_lookup_break.py`. Expected: previous counts + 2. Append `Task 4: complete`.

---

### Task 5: `--graphs` for the corpus and oracle drivers, and the R4 parity script

**Branch:** `dsv41`.

**Files:**
- Modify: `scripts/dsv41/trace_corpus.py` (`engine_kwargs`, `main`), `scripts/dsv41/compare_oracle.py` (`main`)
- Create: `scripts/dsv41/graph_parity.py`, `test/manual/dsv41/test_graph_parity.py`
- Modify: `test/manual/dsv41/test_trace_corpus.py`
- GPU: Step 5 (~20 min, ~20 GiB), the P1 smoke on the truncated model.

**Interfaces:**
- Produces: `trace_corpus.GRAPH_KWARGS`; `engine_kwargs(args)` honouring `args.graphs`; `graph_parity.compare(eager, graph, *, logprob_tol=1e-3) -> {"first_token_mismatch", "max_abs_dlogprob", "pass"}`, `graph_parity.run_env(arm, graph_gather) -> dict`, `graph_parity.engine_kwargs(model, arm, mem_fraction) -> dict` (arms `eager`, `graph`, `debug`, `control`); CLI `graph_parity.py --model --prompt-file --prompt-tokens --new-tokens --mem-fraction-static [--graph-gather] [--debug-arm] [--control] --out`; `$ANA/env-t3.sh`, `$ANA/prompt-0.txt`, `$ANA/parity-t3-p1.json`.

- [ ] **Step 1: Write the failing tests**

Append to `test/manual/dsv41/test_trace_corpus.py`:
```python
def test_graphs_switch_to_breakable_decode_at_batch_size_one():
    args = SimpleNamespace(model="/m", mem_fraction_static=0.8, chunked_prefill_size=512, new_tokens=128, graphs=True)
    kwargs = trace_corpus.engine_kwargs(args)
    assert "disable_cuda_graph" not in kwargs
    assert kwargs["cuda_graph_backend_decode"] == "breakable"
    assert kwargs["cuda_graph_backend_prefill"] == "disabled"
    assert kwargs["cuda_graph_bs_decode"] == [1] and kwargs["cuda_graph_max_bs_decode"] == 1
    eager = trace_corpus.engine_kwargs(SimpleNamespace(**{**vars(args), "graphs": False}))
    assert eager["disable_cuda_graph"] is True
```
Create `test/manual/dsv41/test_graph_parity.py`:
```python
"""The R4 eager-versus-graph comparison (CPU; the Engine runs are the window's)."""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41")
sys.path.insert(0, SCRIPTS)

import graph_parity  # noqa: E402


def _run(tokens, logprobs):
    return {"tokens": tokens, "logprobs": logprobs}


def test_identical_runs_pass():
    report = graph_parity.compare(_run([1, 2, 3], [-0.1, -0.2, -0.3]), _run([1, 2, 3], [-0.1, -0.2, -0.3]))
    assert report["pass"] and report["first_token_mismatch"] is None and report["max_abs_dlogprob"] == 0.0


def test_a_token_mismatch_fails_and_names_the_step():
    report = graph_parity.compare(_run([1, 2, 3], [-0.1] * 3), _run([1, 9, 3], [-0.1] * 3))
    assert not report["pass"] and report["first_token_mismatch"] == 1


def test_logprob_drift_is_bounded_over_the_common_prefix():
    report = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2 - 2e-3]))
    assert not report["pass"] and report["max_abs_dlogprob"] == pytest.approx(2e-3)
    ok = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2 - 5e-4]))
    assert ok["pass"]


def test_the_bitwise_gate_has_zero_tolerance():
    same = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2]), logprob_tol=0.0)
    off = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2 + 1e-9]), logprob_tol=0.0)
    assert same["pass"] and not off["pass"]


def test_each_arm_sets_its_own_graph_gather_env():
    # The eager and control Engines must launch with graph gather off: the gate refuses
    # graph gather without decode graphs.
    assert graph_parity.run_env("eager", True) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "0"}
    assert graph_parity.run_env("control", True) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "0"}
    assert graph_parity.run_env("graph", True) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1"}
    assert graph_parity.run_env("graph", False) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "0"}
    assert graph_parity.run_env("debug", False) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1"}
    with pytest.raises(ValueError):
        graph_parity.run_env("full", True)


def test_each_arm_builds_its_engine():
    eager = graph_parity.engine_kwargs("/m", "eager", 0.8)
    graph = graph_parity.engine_kwargs("/m", "graph", 0.8)
    debug = graph_parity.engine_kwargs("/m", "debug", 0.8)
    assert eager["disable_cuda_graph"] and "cuda_graph_backend_decode" not in eager
    assert graph["cuda_graph_backend_decode"] == "breakable" and "debug_cuda_graph" not in graph
    assert debug["debug_cuda_graph"] and debug["cuda_graph_max_bs_decode"] == 1


def test_decode_sets_the_env_for_the_engine_and_restores_it(monkeypatch):
    import types

    seen = {}

    class _Engine:
        def __init__(self, **kwargs):
            seen["env"] = os.environ.get("SGLANG_MOE_EXPERT_GRAPH_GATHER")
            seen["kwargs"] = kwargs

        def generate(self, **kwargs):
            return {"meta_info": {"output_token_logprobs": [(-0.5, 7, None)]}}

        def shutdown(self):
            seen["shutdown"] = True

    monkeypatch.setitem(sys.modules, "sglang", types.SimpleNamespace(Engine=_Engine))
    monkeypatch.setenv("SGLANG_MOE_EXPERT_GRAPH_GATHER", "1")
    run = graph_parity._decode("/m", [1, 2], 1, "eager", True, 0.8)
    assert seen["env"] == "0" and seen["shutdown"] and run == {"tokens": [7], "logprobs": [-0.5]}
    assert os.environ["SGLANG_MOE_EXPERT_GRAPH_GATHER"] == "1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
```

- [ ] **Step 2: Commit red, push, run red**

`git add test/manual/dsv41/test_trace_corpus.py test/manual/dsv41/test_graph_parity.py`; commit `test(dsv41): red tests for --graphs and the eager-versus-graph comparison`. Run `<DSV41-CPU> test/manual/dsv41/test_trace_corpus.py test/manual/dsv41/test_graph_parity.py`. Expected: the new trace test fails; `test_graph_parity.py` errors with `No module named 'graph_parity'`.

- [ ] **Step 3: Implement**

In `trace_corpus.py`, above `engine_kwargs`:
```python
# Decode as a breakable CUDA graph at batch size 1, prefill eager (the EXL3 gate's only
# graph shape). The MoE runs in-graph once graph gather serves it, else as an eager break.
GRAPH_KWARGS = dict(
    cuda_graph_backend_decode="breakable",
    cuda_graph_backend_prefill="disabled",
    cuda_graph_bs_decode=[1],
    cuda_graph_max_bs_decode=1,
)
```
Rewrite `engine_kwargs` as:
```python
def engine_kwargs(args) -> dict:
    """The Engine of every corpus run (it satisfies the EXL3 expert-caching gate)."""
    kwargs = dict(
        model_path=args.model,
        tp_size=1,
        disable_shared_experts_fusion=True,
        context_length=4096,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        # BS1 decode is what Phase 3 measures; DSV4 reserves SWA slots per request.
        max_running_requests=4,
        # The hot cache's residency counts routes through the recorder's forward
        # observer; without it dynamic residency never updates (MOE_EXPERT_TRANSFER.md).
        expert_distribution_recorder_mode="per_pass",
        # A cached prefix would turn a later prompt into a one-token extend, which
        # the stream trace would count as a decode token.
        disable_radix_cache=True,
    )
    if getattr(args, "graphs", False):
        kwargs.update(GRAPH_KWARGS)
    else:
        kwargs["disable_cuda_graph"] = True
    return kwargs
```
In `main` add `p.add_argument("--graphs", action="store_true", help="breakable decode graphs at batch size 1")`.

In `compare_oracle.py`, add `import os` and `import sys` if absent, `p.add_argument("--graphs", action="store_true")`, and replace the `engine = sglang.Engine(...)` call with a `kwargs = dict(...)` holding the same arguments minus `disable_cuda_graph=True`, followed by:
```python
    if args.graphs:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from trace_corpus import GRAPH_KWARGS

        kwargs.update(GRAPH_KWARGS)
    else:
        kwargs["disable_cuda_graph"] = True
    engine = sglang.Engine(**kwargs)
```
Prefill stays eager in both modes, so the oracle comparison reads the same prefill logits; `--graphs` exists so one Engine configuration serves every script.

Create `scripts/dsv41/graph_parity.py`:
```python
"""R4 on the truncated model: greedy decode in up to four Engine runs, compared.

Arms, each a fresh Engine (shut down before the next), with its own environment
(the scheduler subprocess inherits os.environ, so each run sets it first):
  eager   - no CUDA graphs, SGLANG_MOE_EXPERT_GRAPH_GATHER=0 (the exl3_moe_loop path);
  graph   - breakable decode graph at bs 1; graph gather as --graph-gather says;
  debug   - (--debug-arm) the graph arm with --debug-cuda-graph: the same in-graph
            path run eagerly through the capture machinery (graph gather on);
  control - (--control) a second eager run, for run-to-run nondeterminism.
Reports eager_vs_graph and debug_vs_eager against R4's bar (1e-3), graph_vs_debug
bitwise (tolerance 0.0: the capture-correctness gate), eager_vs_eager. Exit status 1
when the capture gate fails: graph_vs_debug with --debug-arm, else eager_vs_graph.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trace_corpus import GRAPH_KWARGS  # noqa: E402

LOGPROB_TOL = 1e-3
ARMS = ("eager", "graph", "debug", "control")


def compare(eager: dict, graph: dict, *, logprob_tol: float = LOGPROB_TOL) -> dict:
    """``{"tokens": [...], "logprobs": [...]}`` runs -> mismatch step, max |dlogprob|, pass."""
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(eager["tokens"], graph["tokens"])) if a != b),
        None,
    )
    if mismatch is None and len(eager["tokens"]) != len(graph["tokens"]):
        mismatch = min(len(eager["tokens"]), len(graph["tokens"]))
    common = len(eager["tokens"]) if mismatch is None else mismatch + 1
    deltas = [abs(a - b) for a, b in zip(eager["logprobs"][:common], graph["logprobs"][:common])]
    worst = max(deltas, default=0.0)
    return {
        "first_token_mismatch": mismatch,
        "max_abs_dlogprob": worst,
        "pass": mismatch is None and worst <= logprob_tol,
    }


def run_env(arm: str, graph_gather: bool) -> dict[str, str]:
    """Environment overrides of one arm's Engine."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    on = arm == "debug" or (arm == "graph" and graph_gather)
    return {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1" if on else "0"}


def engine_kwargs(model: str, arm: str, mem_fraction: float) -> dict:
    kwargs = dict(
        model_path=model,
        tp_size=1,
        disable_shared_experts_fusion=True,
        context_length=4096,
        mem_fraction_static=mem_fraction,
        max_running_requests=4,
        expert_distribution_recorder_mode="per_pass",
        disable_radix_cache=True,
    )
    if arm in ("graph", "debug"):
        kwargs.update(GRAPH_KWARGS)
    else:
        kwargs["disable_cuda_graph"] = True
    if arm == "debug":
        kwargs["debug_cuda_graph"] = True
    return kwargs


def _decode(model: str, ids: list[int], new_tokens: int, arm: str, graph_gather: bool, mem_fraction: float) -> dict:
    import sglang

    env = run_env(arm, graph_gather)
    saved = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        engine = sglang.Engine(**engine_kwargs(model, arm, mem_fraction))
        try:
            out = engine.generate(
                input_ids=ids,
                sampling_params={"max_new_tokens": new_tokens, "temperature": 0, "ignore_eos": True},
                return_logprob=True,
            )
        finally:
            engine.shutdown()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    entries = out["meta_info"]["output_token_logprobs"]
    return {"tokens": [e[1] for e in entries], "logprobs": [float(e[0]) for e in entries]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True, help="a text file; its first --prompt-tokens tokens")
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--mem-fraction-static", type=float, default=0.8)
    p.add_argument("--graph-gather", action="store_true", help="the graph arm serves the MoE in-graph")
    p.add_argument("--debug-arm", action="store_true")
    p.add_argument("--control", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.prompt_file) as f:
        ids = tokenizer(f.read()).input_ids[: args.prompt_tokens]
    arms = ["eager", "graph"] + (["debug"] if args.debug_arm else []) + (["control"] if args.control else [])
    runs = {
        arm: _decode(args.model, ids, args.new_tokens, arm, args.graph_gather, args.mem_fraction_static)
        for arm in arms
    }
    report = dict(runs)
    report["eager_vs_graph"] = compare(runs["eager"], runs["graph"])
    if "debug" in runs:
        report["graph_vs_debug"] = compare(runs["debug"], runs["graph"], logprob_tol=0.0)
        report["debug_vs_eager"] = compare(runs["eager"], runs["debug"])
    if "control" in runs:
        report["eager_vs_eager"] = compare(runs["eager"], runs["control"])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if "_vs_" in k}))
    gate = report["graph_vs_debug"] if "debug" in runs else report["eager_vs_graph"]
    sys.exit(0 if gate["pass"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit, push, run green**

`git add scripts/dsv41/trace_corpus.py scripts/dsv41/compare_oracle.py scripts/dsv41/graph_parity.py`; commit `feat(dsv41): --graphs for the corpus and oracle drivers; eager-vs-graph parity script`. Run `<DSV41-CPU> <DSV41_SUITE> test/registered/unit/layers/test_engram_lookup_break.py test/manual/dsv41/test_graph_parity.py`. Expected: previous counts + 8 (1 trace, 7 parity). Record the summary line in the ledger (Task 8 compares against it).

- [ ] **Step 5: P1 smoke on the truncated model (GPU)**

Create the `T3` environment and a prompt file:
```bash
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b; mkdir -p $ANA
cat > $ANA/env-t3.sh <<EOS
. /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a/env.sh
ANA=$ANA
T3=/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-trunc3
export SGLANG_MOE_PINNED_HOST_MB=4096 SGLANG_MOE_HOT_GPU_MB=2048
EOS
/data/models/slang/.venv/bin/python -c "import json; t=json.loads(open(\"/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl\").readline())[\"turns\"][0]; open(\"$ANA/prompt-0.txt\",\"w\").write(t)"'
```
Then run `<GPU> bash -c ". $ANA/env-t3.sh && /data/models/slang/.venv/bin/python scripts/dsv41/graph_parity.py --model \$T3 --prompt-file $ANA/prompt-0.txt --prompt-tokens 256 --new-tokens 32 --mem-fraction-static 0.8 --control --out $ANA/parity-t3-p1.json > $ANA/parity-t3-p1.log 2>&1"` (`$ANA` expands in the wrapper's shell, `\$T3` in the inner shell after `env-t3.sh` defines it; every later `bash -c` GPU command in this plan follows the same quoting). `env-t3.sh` keeps `SGLANG_MOE_EXPERT_GRAPH_GATHER=0` from the Window C env, so the MoE runs as the eager break.

Expected: the graph Engine log contains `Breakable CUDA graph captured: shape=` with `breaks=4` (3 MoE + 1 Engram layer; `T3` has `candidate_source_layer_id=-1`, so one variant); the printed `eager_vs_graph` has `"pass": true` (the same math as eager). **Rule:** a capture error or a failed `eager_vs_graph` is a P1 bug: fix it with a red/green pair in the owning task's files (Task 3 for gate and breaks, Task 4 for the guards) and rerun this step; exit code 75 is the GPU lock rule. Append `Task 5: complete`.

**P1 ends here**: graph decode runs with the eager MoE break, checked on the GPU.

---

## Phase P3 — in-graph MoE

**Tasks 6–16 are blocked if Task 1 fails** (Task 17 then writes its short form). Before starting any of them, check the ledger for `Task 1: complete — PASS`.

### Task 6: Framework — graph gather over a pinned tier (`graph_source_kind`, `PinnedTierRowBackend`)

**Blocked if Task 1 fails.**

**Branch:** `cc/moe-expert-plugins` (`FWT`). Read `.claude/skills/large-class-style/SKILL.md` before Step 3's `model_runner.py` edit.

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_format.py` (`require_graph_gather_support`, `:255-274`; new `graph_source_kind_of`)
- Modify: `python/sglang/srt/layers/moe/expert_row_plan.py` (new `BACKEND_PINNED_TIER`, `PinnedTierRowBackend`)
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (`__init__` near `:679-680`, `enable_graph_gather` `:818-939`, `_check_graph_sources` `:1075-1082`)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (`from_model`, the `require_graph_gather_support` call at `:1094-1095`)
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements.py` (`ExpertStreamRequirements`), `python/sglang/srt/arg_groups/memory_hook.py` (arena rule, `:153-160`)
- Modify: `python/sglang/srt/model_executor/model_runner.py` (`maybe_init_expert_hot_cache`, `:727-730`)
- Create: `test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py` (CPU), `test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py` (CUDA; skips on CPU; runs green in Step 5 and again in Task 16)

**Interfaces:**
- Produces:
  - `graph_source_kind_of(expert_format) -> str` (`"dense"` unless the format sets `graph_source_kind`);
  - `require_graph_gather_support(streamers, *, pinned_tier_ok: bool = False)`;
  - `PinnedTierRowBackend(segments: Mapping[int, ExpertRowSegments], host_row_map: Tensor, capacity: int)` with `host_rows` int64 `[C]`, `ram_miss` int64 `[1]` (cumulative), `keep` fp32 `[1]`, `translate(tag, plan)`, `post`, `resolve`, `copy_residual`;
  - `ExpertStreamer._graph_pinned_tier: bool`;
  - `ExpertStreamRequirements.graph_gather_host_source: str = "arena"`.

- [ ] **Step 1: Write the failing CPU tests**

Create `test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py`:
```python
"""Graph gather over a partial pinned host tier: the row backend and the support check (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.expert_stream_requirements import ExpertStreamRequirements
from sglang.srt.layers.moe import expert_row_plan
from sglang.srt.layers.moe.expert_format import graph_source_kind_of, require_graph_gather_support
from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan, PinnedTierRowBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _plan(ids, count):
    return ExpertRowPlan(
        expert_ids=torch.tensor(ids, dtype=torch.int64),
        slots=torch.arange(len(ids), dtype=torch.int32),
        count=torch.tensor([count], dtype=torch.int32),
    )


@pytest.fixture
def copies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        expert_row_plan,
        "copy_expert_row_segments_gpu",
        lambda segments, rows, slots, count: calls.append((segments, rows.tolist(), slots.tolist(), int(count))),
    )
    return calls


def test_planned_experts_are_copied_from_their_pinned_slots(copies):
    host_map = torch.tensor([-1, 4, -1, 0, -1, 2, -1, -1], dtype=torch.int64)
    backend = PinnedTierRowBackend({0: "segments"}, host_map, 4)
    plan = _plan([3, 5, 1, 7], count=3)  # lane 3 is past the count: its -1 is not a miss
    backend.post(0, plan)
    assert copies == [("segments", [0, 2, 4, 0], [0, 1, 2, 3], 3)]
    assert backend.keep.tolist() == [1.0] and backend.ram_miss.tolist() == [0]
    assert backend.resolve(0, plan).delivered is None


def test_a_row_missing_from_ram_copies_slot_zero_and_drops_the_layer(copies):
    host_map = torch.tensor([-1, 4, -1, 0], dtype=torch.int64)
    backend = PinnedTierRowBackend({0: "segments"}, host_map, 2)
    backend.post(0, _plan([2, 1], count=2))
    assert copies[-1][1] == [0, 4]
    assert backend.keep.tolist() == [0.0] and backend.ram_miss.tolist() == [1]
    backend.post(0, _plan([1, 3], count=2))
    assert backend.keep.tolist() == [1.0] and backend.ram_miss.tolist() == [1]  # cumulative


def test_graph_source_kind_defaults_to_dense():
    assert graph_source_kind_of(SimpleNamespace()) == "dense"
    assert graph_source_kind_of(SimpleNamespace(graph_source_kind="pinned_tier")) == "pinned_tier"


def _streamer(kind, pinned):
    fmt = SimpleNamespace(key="k", supports_graph_gather=False, graph_source_kind=kind)
    return SimpleNamespace(format=fmt, has_spec_only_tensors=True, pinned_host_cache=pinned, layer_id=3)


def test_support_check_accepts_a_pinned_tier_format_only_when_asked():
    require_graph_gather_support([_streamer("pinned_tier", object())], pinned_tier_ok=True)
    with pytest.raises(ValueError, match="does not support graph gather"):
        require_graph_gather_support([_streamer("pinned_tier", object())])
    with pytest.raises(ValueError, match="pinned host tier"):
        require_graph_gather_support([_streamer("pinned_tier", None)], pinned_tier_ok=True)
    with pytest.raises(ValueError, match="does not support graph gather"):
        require_graph_gather_support([_streamer("dense", object())], pinned_tier_ok=True)


def test_requirements_default_to_the_host_arena():
    assert ExpertStreamRequirements("X", lambda cfg, budgets: None).graph_gather_host_source == "arena"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```
Create `test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py`:
```python
"""A captured graph gather reads missed rows from pinned-tier slots, not expert-id rows (CUDA)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


def _tiers(pinned_rows=4, hot_slots=2, top_k=2):
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.moe_expert_fakes import SpecOnlyFormat

    reference = {
        name: (torch.arange(8 * 64, dtype=torch.int32).reshape(8, 64) * (i + 1)).to(torch.int16)
        for i, name in enumerate(NAMES)
    }
    fmt = SpecOnlyFormat(reference)
    fmt.graph_source_kind = "pinned_tier"
    layer = torch.nn.Module()
    layer.layer_id = 0
    layer.top_k = top_k
    streamer = ExpertStreamer(layer, NAMES, format=fmt)
    pinned = ExpertPinnedHostCache(streamer, pinned_rows)
    hot = ExpertHotCache(streamer, hot_slots, scratch_rows=top_k)
    hot.reassign([0, 1])
    streamer.enable_graph_gather(top_k)
    return streamer, pinned, hot, reference


def test_capture_then_replay_reads_the_pinned_slots():
    streamer, pinned, hot, reference = _tiers()
    assert streamer._graph_pinned_tier
    pinned.ensure_rows(torch.tensor([6, 3], device="cuda"))  # slots chosen by the LRU
    ids = torch.tensor([[1, 6]], device="cuda", dtype=torch.int32)
    streamer.gather(ids)  # warm up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        remap, tensors = streamer.gather(ids)
    ids.copy_(torch.tensor([[3, 0]], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    for name in NAMES:
        got = tensors[name][remap.reshape(-1).long()].cpu()
        assert torch.equal(got, reference[name][[3, 0]]), name
    assert streamer.row_backend.keep.item() == 1.0
    assert streamer.row_backend.ram_miss.item() == 0


def test_a_ram_miss_inside_a_replay_is_counted_and_drops_the_layer():
    streamer, pinned, hot, _ = _tiers()
    pinned.ensure_rows(torch.tensor([6], device="cuda"))
    ids = torch.tensor([[1, 6]], device="cuda", dtype=torch.int32)
    streamer.gather(ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        streamer.gather(ids)
    ids.copy_(torch.tensor([[1, 7]], device="cuda", dtype=torch.int32))  # 7 is not in RAM
    graph.replay()
    torch.cuda.synchronize()
    assert streamer.row_backend.keep.item() == 0.0
    assert streamer.row_backend.ram_miss.item() == 1
```
Check `ExpertHotCache.__init__`'s signature (`expert_hot_cache.py:111`: `(streamer, capacity, scratch_rows=0)`) and `register_cuda_ci`'s suite names in an existing CUDA test (`grep -n register_cuda_ci test/registered/unit/layers/moe/test_expert_graph_gather.py`); use the suite that file uses.

- [ ] **Step 2: Commit red, push, run red**

`git add` both test files; commit `test(moe): red tests for graph gather over a pinned host tier`. Run `<FW-CPU> test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py`. Expected: the CPU file errors at import (`cannot import name 'graph_source_kind_of'`); the CUDA file skips (2 skipped).

- [ ] **Step 3: Implement**

`expert_format.py` — add above `require_graph_gather_support`:
```python
def graph_source_kind_of(expert_format: Any) -> str:
    """Where a format's graph gathers read host rows: ``"dense"`` (``[experts, ...]``
    layer tensors or the host arena, indexed by expert id) or ``"pinned_tier"`` (the
    layer's pinned host tier, indexed by pinned slot)."""
    return getattr(expert_format, "graph_source_kind", "dense")
```
and replace `require_graph_gather_support` with:
```python
def require_graph_gather_support(
    streamers: Iterable["ExpertStreamer"], *, pinned_tier_ok: bool = False
) -> None:
    """Raise unless every streamer's format can serve sync-free graph gathers.

    Dense formats need dense, GPU-readable host sources frozen at startup, which
    spec-only tensors lack. A ``pinned_tier`` format serves graph gathers from its
    pinned host tier instead; only the plain graph gather supports that
    (``pinned_tier_ok``), not the GPU residency update or the doorbell, whose
    copies index host rows by expert id.
    """
    for streamer in streamers:
        expert_format = streamer.format
        key = expert_format.key
        if pinned_tier_ok and graph_source_kind_of(expert_format) == "pinned_tier":
            if streamer.pinned_host_cache is None:
                raise ValueError(
                    f"expert format {key!r} of layer {streamer.layer_id} serves graph "
                    "gathers from its pinned host tier; set SGLANG_MOE_PINNED_HOST_MB"
                )
            continue
        unsupported = (
            not expert_format.supports_graph_gather or streamer.has_spec_only_tensors
        )
        if unsupported:
            raise ValueError(
                f"expert format {key!r} of layer {streamer.layer_id} does not support "
                "graph gather; unset SGLANG_MOE_EXPERT_GRAPH_GATHER, "
                "SGLANG_MOE_GPU_RESIDENCY_UPDATE and SGLANG_MOE_EXPERT_DOORBELL"
            )
```
`expert_row_plan.py` — add `BACKEND_PINNED_TIER = "pinned_tier"` beside the other backend names, and after `DoorbellRowBackend`:
```python
class PinnedTierRowBackend:
    """Serves plans of expert ids from a partial pinned host tier, indexed by pinned slot.

    The segment table's sources are the tier's slabs. ``translate`` maps each
    planned expert through ``host_row_map`` (expert id -> pinned slot, -1 when the
    row is not in RAM) into ``host_rows``; ``post`` then copies with the segment
    kernel. A planned row missing from RAM is copied from slot 0 (a valid address),
    added to the cumulative ``ram_miss`` and sets ``keep`` to 0 for this call, so
    the caller can drop the layer's routed output and the host can fail stop.
    Every step is fixed-shape device work, so a CUDA graph can capture it.
    """

    name = BACKEND_PINNED_TIER

    def __init__(
        self,
        segments: Mapping[int, ExpertRowSegments],
        host_row_map: torch.Tensor,
        capacity: int,
    ) -> None:
        self.segments = dict(segments)
        self.host_row_map = host_row_map
        device = host_row_map.device
        self.host_rows = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device=device)
        self.keep = torch.ones(1, dtype=torch.float32, device=device)
        self._lanes = torch.arange(capacity, dtype=torch.int32, device=device)

    def translate(self, tag: int, plan: ExpertRowPlan) -> None:
        rows = self.host_row_map.index_select(0, plan.expert_ids).to(torch.int64)
        missing = (rows < 0) & (self._lanes < plan.count)
        misses = missing.sum().reshape(1)
        self.ram_miss.add_(misses)
        self.keep.copy_((misses == 0).to(torch.float32))
        self.host_rows.copy_(rows.clamp(min=0))

    def post(self, tag: int, plan: ExpertRowPlan) -> None:
        self.translate(tag, plan)
        copy_expert_row_segments_gpu(
            self.segments[tag], self.host_rows, plan.slots, plan.count
        )

    def resolve(self, tag: int, plan: ExpertRowPlan) -> ExpertRowDelivery:
        return ExpertRowDelivery(plan, None)

    def copy_residual(self, tag: int, delivery: ExpertRowDelivery) -> None:
        """``post`` copied every planned row."""
```
`expert_stream.py`:
1. In `ExpertStreamer.__init__`, next to `self.row_backend = None` (`:679`), add `self._graph_pinned_tier = False`.
2. Import `graph_source_kind_of` with the other `expert_format` imports, and `PinnedTierRowBackend` with the `expert_row_plan` imports.
3. In `enable_graph_gather`, replace
```python
        require_graph_gather_support((self,))
```
with
```python
        pinned_tier = graph_source_kind_of(self.format) == "pinned_tier"
        require_graph_gather_support((self,), pinned_tier_ok=True)
```
replace
```python
        if self.pinned_host_cache is not None:
            raise ValueError(
                "graph gather cannot admit rows through the pinned host cache"
            )
```
with
```python
        if self.pinned_host_cache is not None and not pinned_tier:
            raise ValueError(
                "graph gather cannot admit rows through the pinned host cache"
            )
```
replace
```python
        for name in self.tensor_names:
            source = _tensor_data(getattr(self.layer, name))
            if source.device.type == "cpu" and not is_gpu_readable_host_tensor(source):
                raise ValueError(
                    f"graph gather needs registered host rows; {name!r} is pageable"
                )
        device = cache.device
        self._graph_sources = {
            name: _tensor_data(getattr(self.layer, name)) for name in self.tensor_names
        }
```
with
```python
        if pinned_tier:
            # Missed rows are read from the pinned tier's registered slabs by pinned
            # slot (PinnedTierRowBackend), never from layer attributes.
            sources = dict(self.pinned_host_cache.tensors)
        else:
            sources = {
                name: _tensor_data(getattr(self.layer, name)) for name in self.tensor_names
            }
        for name, source in sources.items():
            if source.device.type == "cpu" and not is_gpu_readable_host_tensor(source):
                raise ValueError(
                    f"graph gather needs registered host rows; {name!r} is pageable"
                )
        device = cache.device
        self._graph_sources = sources
        self._graph_pinned_tier = pinned_tier
```
and replace
```python
        self.row_backend = (
            InGraphRowBackend({self.row_tag: self._graph_row_segments})
            if self._graph_row_segments is not None
            else None
        )
```
with
```python
        if pinned_tier:
            self.row_backend = PinnedTierRowBackend(
                {self.row_tag: self._graph_row_segments},
                self.pinned_host_cache.expert_to_slot,
                max_rows,
            )
        else:
            self.row_backend = (
                InGraphRowBackend({self.row_tag: self._graph_row_segments})
                if self._graph_row_segments is not None
                else None
            )
```
4. In `_check_graph_sources`, replace
```python
            current = _tensor_data(getattr(self.layer, name))
```
with
```python
            current = (
                self.pinned_host_cache.tensors[name]
                if self._graph_pinned_tier
                else _tensor_data(getattr(self.layer, name))
            )
```
`expert_hot_cache.py` — replace
```python
        if index(graph_gather_batch_size) or gpu_residency_update or expert_doorbell:
            require_graph_gather_support(streamers.values())
```
with
```python
        if index(graph_gather_batch_size) or gpu_residency_update or expert_doorbell:
            require_graph_gather_support(
                streamers.values(),
                pinned_tier_ok=not (gpu_residency_update or expert_doorbell),
            )
```
`expert_stream_requirements.py` — in `ExpertStreamRequirements` add after `check`:
```python
    # Where graph gathers read host rows: "arena" (SGLANG_MOE_EXPERT_HOST_ARENA,
    # indexed by expert id) or "pinned_tier" (the format's pinned host tier).
    graph_gather_host_source: str = "arena"
```
`memory_hook.py` — replace
```python
        if not envs.SGLANG_MOE_EXPERT_HOST_ARENA.get():
            raise ValueError(
                "SGLANG_MOE_EXPERT_GRAPH_GATHER requires SGLANG_MOE_EXPERT_HOST_ARENA=1"
            )
```
with
```python
        if not envs.SGLANG_MOE_EXPERT_HOST_ARENA.get() and (
            expert_stream_requirements_for(server_args, cfg).graph_gather_host_source
            == "arena"
        ):
            raise ValueError(
                "SGLANG_MOE_EXPERT_GRAPH_GATHER requires SGLANG_MOE_EXPERT_HOST_ARENA=1"
            )
```
`model_runner.py` (`maybe_init_expert_hot_cache`) — replace
```python
            from sglang.srt.layers.moe.expert_format import iter_expert_streamers
```
with
```python
            from sglang.srt.layers.moe.expert_format import (
                graph_source_kind_of,
                iter_expert_streamers,
            )
```
and
```python
            if getattr(self, "expert_host_arena", None) is None:
                raise ValueError(
```
with
```python
            if getattr(self, "expert_host_arena", None) is None and not all(
                graph_source_kind_of(streamer.format) == "pinned_tier"
                for streamer in iter_expert_streamers(self.model)
            ):
                raise ValueError(
```

- [ ] **Step 4: Commit, push, run green**

`git add python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_row_plan.py python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py python/sglang/srt/arg_groups/expert_stream_requirements.py python/sglang/srt/arg_groups/memory_hook.py python/sglang/srt/model_executor/model_runner.py`; commit `feat(moe): graph gather over a pinned host tier (graph_source_kind, PinnedTierRowBackend)`. Run `<FW-CPU> <FRAMEWORK_SUITE> test/registered/unit/layers/moe/test_expert_capture_baseline.py test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py`. Expected: baseline counts + 1 + 5 passed, 2 skipped; the known failure only.

- [ ] **Step 5: Run the CUDA test on the GPU**

`<FW-GPU> /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py test/registered/unit/layers/moe/test_expert_graph_gather.py 2>&1 | tail -5`. Expected: `test_expert_pinned_graph_gather_cuda.py` 2 passed, and the existing graph-gather CUDA tests pass unchanged (the dense path is untouched). A failure is fixed by a red/green pair in this task's files and this step rerun; exit code 75 is the GPU lock rule. Append `Task 6: complete`.

---

### Task 7: Framework — pluggable pinned slot table, fail-stop checks and residency listeners

**Blocked if Task 1 fails.**

**Branch:** `cc/moe-expert-plugins` (`FWT`).

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_host_tier.py` (new `PinnedSlotTable` protocol; `PinnedSlotLRU.before_host_use`)
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (`ExpertPinnedHostCache.__init__`, `lookup`, `ensure_rows`, `copy_rows`, `gather_rows`)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (`from_model` end; `on_expert_distribution` residency branch; `doorbell_fail_stop_check`; new `register_fail_stop_check`, `add_residency_listener`, `_notify_residency_listeners`, `_attach_formats`)
- Create: `test/registered/unit/layers/moe/test_expert_pinned_slot_table.py`, `test/registered/unit/layers/moe/test_expert_fail_stop_checks.py`

**Interfaces:**
- Produces:
  - `PinnedSlotTable` (Protocol): `capacity`, `slot_to_expert`, `expert_to_slot`, `__contains__`, `touch`, `assign(expert_id, protected=frozenset()) -> (slot, evicted | None)`, `release(slot)`, `mapping(num_experts) -> list[int]`, `before_host_use(cache) -> None`, `after_host_use(cache) -> None`;
  - `lookup`, `ensure_rows`, `copy_rows`, `gather_rows` run between `before_host_use` and `after_host_use` (`after` in a `finally`; nested calls call both again, so a table counts depth);
  - `ExpertPinnedHostCache(..., slot_table: PinnedSlotTable | None = None)`; a format passes it through `pinned_tier_options(layer)["slot_table"]`; a table with `bind_capacity(capacity)` gets the tier's capacity before the capacity check;
  - `ExpertHotCacheManager.register_fail_stop_check(fn: Callable[[], None])`, `add_residency_listener(fn: Callable[[int, list[int]], None])`; `doorbell_fail_stop_check` runs the checks first;
  - format hook `attach_hot_cache_manager(manager, streamer)` (optional on a format): `from_model` calls it for every streamer after graph gather is enabled, then notifies the residency listeners once (`_attach_formats`).

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/layers/moe/test_expert_pinned_slot_table.py`:
```python
"""A format can own the pinned tier's slot bookkeeping through a PinnedSlotTable (CPU)."""

import pytest
import torch

from sglang.srt.layers.moe.expert_host_tier import PinnedSlotLRU
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.moe_expert_fakes import SpecOnlyFormat

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


class RecordingTable(PinnedSlotLRU):
    """A PinnedSlotLRU that records every host use, as an external owner would see it."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self.host_uses = 0
        self.depth = 0
        self.max_depth = 0

    def before_host_use(self, cache):
        self.host_uses += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)

    def after_host_use(self, cache):
        self.depth -= 1


def _streamer():
    reference = {
        name: torch.arange(8 * 4, dtype=torch.int16).reshape(8, 4) + i for i, name in enumerate(NAMES)
    }
    layer = torch.nn.Module()
    layer.layer_id = 0
    return ExpertStreamer(layer, NAMES, format=SpecOnlyFormat(reference)), reference


def test_the_cache_uses_a_supplied_slot_table_and_announces_host_use():
    streamer, reference = _streamer()
    table = RecordingTable(3)
    cache = ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=table)
    cache.ensure_rows(torch.tensor([5, 2]))
    assert table.host_uses >= 1
    assert 5 in table and 2 in table
    assert cache.expert_to_slot[5].item() == table.expert_to_slot[5]
    slot = table.expert_to_slot[5]
    assert torch.equal(cache.tensors["w2_suh"][slot], reference["w2_suh"][5])
    before = table.host_uses
    cache.lookup(torch.tensor([5]))
    assert table.host_uses == before + 1
    assert table.depth == 0  # every use was closed
    cache.gather_rows(torch.tensor([5]), {n: torch.empty((1,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in reference.items()})
    assert table.depth == 0 and table.max_depth >= 2  # gather_rows nests lookup/ensure_rows


def test_host_use_is_closed_when_the_call_raises():
    streamer, _ = _streamer()
    table = RecordingTable(1)
    cache = ExpertPinnedHostCache(streamer, 1, device="cpu", slot_table=table)
    with pytest.raises(Exception):
        cache.lookup(torch.tensor([5], device="meta"))  # wrong device: lookup raises
    assert table.depth == 0


def test_a_slot_table_of_another_capacity_is_refused():
    streamer, _ = _streamer()
    with pytest.raises(ValueError, match="capacity"):
        ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=RecordingTable(2))


def test_the_default_lru_needs_no_host_use_hook_behaviour():
    PinnedSlotLRU(2).before_host_use(object())  # no-ops
    PinnedSlotLRU(2).after_host_use(object())


def test_the_cache_binds_its_capacity_into_an_unbound_slot_table():
    streamer, _ = _streamer()

    class Unbound(RecordingTable):
        def __init__(self):
            super().__init__(3)
            self.capacity = None

        def bind_capacity(self, capacity):
            self.capacity = capacity

    table = Unbound()
    ExpertPinnedHostCache(streamer, 3, device="cpu", slot_table=table)
    assert table.capacity == 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```
Create `test/registered/unit/layers/moe/test_expert_fail_stop_checks.py`:
```python
"""The hot-cache manager runs registered fail-stop checks and residency listeners (CPU)."""

from types import SimpleNamespace

import pytest

from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _manager():
    manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
    manager.doorbell = None
    return manager


def test_registered_checks_run_before_the_doorbell_check():
    manager = _manager()
    calls = []
    manager.register_fail_stop_check(lambda: calls.append("a"))
    manager.register_fail_stop_check(lambda: calls.append("b"))
    assert manager.doorbell_fail_stop_check(synchronize=True) == 0.0
    assert calls == ["a", "b"]


def test_a_failing_check_raises_through_the_scheduler_hook():
    manager = _manager()

    def fail():
        raise RuntimeError("fail-stop")

    manager.register_fail_stop_check(fail)
    with pytest.raises(RuntimeError, match="fail-stop"):
        manager.doorbell_fail_stop_check(synchronize=True)


def test_a_manager_without_checks_still_returns():
    assert _manager().doorbell_fail_stop_check() == 0.0


def test_formats_are_offered_the_manager_then_residency_is_pushed():
    manager = _manager()
    manager.caches = {1: SimpleNamespace(slot_to_expert=[4])}
    events = []

    class _Format:
        def attach_hot_cache_manager(self, mgr, streamer):
            events.append(("attach", streamer.layer_id))
            mgr.add_residency_listener(lambda layer_id, experts: events.append(("hot", layer_id, experts)))

    manager.streamers = {1: SimpleNamespace(format=_Format(), layer_id=1), 2: SimpleNamespace(format=SimpleNamespace(), layer_id=2)}
    manager._attach_formats()
    assert events == [("attach", 1), ("hot", 1, [4])]


def test_residency_listeners_get_every_layers_resident_experts():
    manager = _manager()
    manager.caches = {2: SimpleNamespace(slot_to_expert=[5, -1]), 7: SimpleNamespace(slot_to_expert=[1])}
    seen = []
    manager.add_residency_listener(lambda layer_id, experts: seen.append((layer_id, experts)))
    manager._notify_residency_listeners()
    assert sorted(seen) == [(2, [5, -1]), (7, [1])]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```

- [ ] **Step 2: Commit red, push, run red**

`git add` both test files; commit `test(moe): red tests for pluggable pinned slot tables and fail-stop checks`. Run `<FW-CPU> test/registered/unit/layers/moe/test_expert_pinned_slot_table.py test/registered/unit/layers/moe/test_expert_fail_stop_checks.py`. Expected: slot-table tests fail (`unexpected keyword argument 'slot_table'`; `PinnedSlotLRU` has no `before_host_use`); fail-stop tests fail (`no attribute 'register_fail_stop_check'`).

- [ ] **Step 3: Implement**

`expert_host_tier.py` — extend the `typing` import with `Any, Mapping, Protocol`, add before `PinnedSlotLRU`:
```python
class PinnedSlotTable(Protocol):
    """The slot bookkeeping ``ExpertPinnedHostCache`` delegates to.

    ``PinnedSlotLRU`` is the default. A format whose slots are also managed by
    someone else (a native reader thread) supplies its own through
    ``pinned_tier_options(layer)["slot_table"]``; ``before_host_use(cache)`` runs
    before every host-side use of the tier, so that owner can pause and the cache
    can refresh its device slot map.
    """

    capacity: int

    @property
    def slot_to_expert(self) -> Sequence[int]: ...

    @property
    def expert_to_slot(self) -> Mapping[int, int]: ...

    def __contains__(self, expert_id: int) -> bool: ...

    def touch(self, expert_id: int) -> None: ...

    def assign(
        self, expert_id: int, protected: Collection[int] = frozenset()
    ) -> tuple[int, Optional[int]]: ...

    def release(self, slot: int) -> None: ...

    def mapping(self, num_experts: int) -> list[int]: ...

    def before_host_use(self, cache: Any) -> None: ...

    def after_host_use(self, cache: Any) -> None: ...
```
and in `PinnedSlotLRU` add:
```python
    def before_host_use(self, cache) -> None:
        """Nothing else owns these slots."""
        return None

    def after_host_use(self, cache) -> None:
        return None
```
`expert_stream.py` — `ExpertPinnedHostCache.__init__`: add the keyword `slot_table: "PinnedSlotTable | None" = None` after `is_pinned`, import `PinnedSlotTable` from `expert_host_tier`, and replace
```python
        self._lru = PinnedSlotLRU(capacity, is_pinned=is_pinned)
```
with
```python
        if slot_table is not None and hasattr(slot_table, "bind_capacity"):
            # A table built before the tier's size was known (a format's
            # pinned_tier_options) learns it here.
            slot_table.bind_capacity(capacity)
        if slot_table is not None and slot_table.capacity != capacity:
            raise ValueError(
                f"pinned slot table capacity {slot_table.capacity} does not match "
                f"the tier's {capacity} rows"
            )
        self._lru = (
            slot_table
            if slot_table is not None
            else PinnedSlotLRU(capacity, is_pinned=is_pinned)
        )
```
Add at module level (import `functools` if absent):
```python
def _host_use(method):
    """Run a pinned-tier method between its slot table's before/after_host_use."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._lru.before_host_use(self)
        try:
            return method(self, *args, **kwargs)
        finally:
            self._lru.after_host_use(self)

    return wrapper
```
and decorate `ExpertPinnedHostCache.lookup`, `ensure_rows`, `copy_rows` and `gather_rows` with `@_host_use`.
`expert_hot_cache.py`:
1. Add methods to `ExpertHotCacheManager`:
```python
    def register_fail_stop_check(self, check: Callable[[], None]) -> None:
        """Run ``check`` after every batch result, before the doorbell's own check.

        A check raises to stop the process; it must not synchronize the device.
        """
        if not hasattr(self, "fail_stop_checks"):
            self.fail_stop_checks = []
        self.fail_stop_checks.append(check)

    def add_residency_listener(self, listener: Callable[[int, list[int]], None]) -> None:
        """Call ``listener(layer_id, slot_to_expert)`` after startup and every residency update."""
        if not hasattr(self, "residency_listeners"):
            self.residency_listeners = []
        self.residency_listeners.append(listener)

    def _notify_residency_listeners(self) -> None:
        for listener in getattr(self, "residency_listeners", ()):
            for layer_id, cache in self.caches.items():
                listener(layer_id, list(cache.slot_to_expert))

    def _attach_formats(self) -> None:
        """Offer the finished manager to each streamed format that asks for it
        (``attach_hot_cache_manager(manager, streamer)``), then push residency once."""
        for streamer in self.streamers.values():
            hook = getattr(streamer.format, "attach_hot_cache_manager", None)
            if hook is not None:
                hook(self, streamer)
        self._notify_residency_listeners()
```
(import `Callable` from `typing`/`collections.abc` if the module lacks it). At the end of `from_model`, replace
```python
                sort_keys=True,
            ),
        )
        return manager

    def enable_next_layer_prefetch(
```
with
```python
                sort_keys=True,
            ),
        )
        manager._attach_formats()
        return manager

    def enable_next_layer_prefetch(
```
2. In `doorbell_fail_stop_check`, replace
```python
        doorbell = getattr(self, "doorbell", None)
        if doorbell is None:
            return 0.0
```
with
```python
        for check in getattr(self, "fail_stop_checks", ()):
            check()
        doorbell = getattr(self, "doorbell", None)
        if doorbell is None:
            return 0.0
```
3. In `on_expert_distribution`, replace
```python
        elif qualifying:
            self._update_residency(boundary_tokens, mode)
```
with
```python
        elif qualifying:
            self._update_residency(boundary_tokens, mode)
            self._notify_residency_listeners()
```

- [ ] **Step 4: Commit, push, run green**

`git add python/sglang/srt/layers/moe/expert_host_tier.py python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py`; commit `feat(moe): pluggable pinned slot tables, registered fail-stop checks, residency listeners`. Run `<FW-CPU> <FRAMEWORK_SUITE> test/registered/unit/layers/moe/test_expert_capture_baseline.py test/registered/unit/layers/moe/test_expert_pinned_graph_gather.py test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py test/registered/unit/layers/moe/test_expert_pinned_slot_table.py test/registered/unit/layers/moe/test_expert_fail_stop_checks.py`. Expected: Task 6's line + 10 passed; the known failure only. Append `Task 7: complete`.

---

### Task 8: Merge `cc/moe-expert-plugins` into `dsv41` (Tasks 6–7)

**Blocked if Task 1 fails.**

**Files:** none edited by hand.

- [ ] **Step 1: Check the framework part is finished and pushed**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41
git fetch shared
test "$(git rev-parse shared/cc/moe-expert-plugins)" = "$(git -C ../moe-expert-plugins rev-parse HEAD)" && echo pushed
for n in 6 7; do grep -qE "^Task $n: complete" .superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md && echo "Task $n complete"; done
for name in "class PinnedTierRowBackend" "def graph_source_kind_of" "class PinnedSlotTable" "def register_fail_stop_check" "def add_residency_listener" "def _attach_formats"; do
  printf '%s: ' "$name"; git grep -l "$name" shared/cc/moe-expert-plugins -- python/sglang | wc -l
done
git status --short --branch | head -1
```
Expected: `pushed`; `Task 6 complete`, `Task 7 complete`; each name `1`; `## dsv41...shared/dsv41`. If a check fails, finish the missing piece first (push, or complete Task 6/7); if the worktree is not clean, commit or remove only files this plan created, never others'.

- [ ] **Step 2: Dry-run and merge**

```bash
git merge-tree --write-tree --name-only dsv41 shared/cc/moe-expert-plugins
```
Expected: one tree hash only (file names after it mean a conflict: a file name means a conflict. **Conflict rule:** do not merge; `git merge --abort` if a merge was started, record `Task 8: BLOCKED — merge conflict in <files>` in the ledger, and continue with the tasks that do not depend on this merge (Task 1 and the P1 tasks never do); Tasks 9–17 depend on this merge, so they wait, and Tasks 10–12 may proceed since they touch only EXL3 files). Then:
```bash
git merge --no-ff shared/cc/moe-expert-plugins -m "$(printf '%s\n\n%s\n%s\n' \
  'merge cc/moe-expert-plugins into dsv41 (pinned-tier graph gather, slot tables, fail-stop checks)' \
  'Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>' \
  'Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF')"
git log -1 --format='%P' | wc -w
git log --oneline HEAD^1..HEAD^2
git diff --stat shared/cc/moe-expert-plugins HEAD -- python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py python/sglang/srt/layers/moe/expert_format.py python/sglang/srt/layers/moe/expert_row_plan.py python/sglang/srt/layers/moe/expert_host_tier.py python/sglang/srt/arg_groups/expert_stream_requirements.py python/sglang/test/moe_expert_fakes.py
for f in python/sglang/srt/arg_groups/memory_hook.py python/sglang/srt/model_executor/model_runner.py; do
  diff <(git diff HEAD^1 HEAD -- $f | grep '^[-+][^-+]') <(git diff $(git merge-base HEAD^1 HEAD^2) HEAD^2 -- $f | grep '^[-+][^-+]') > /dev/null && echo "$f: framework delta only" || echo "$f: MISMATCH"
done
```
Expected: `2`; `git log` lists only the framework commits of Tasks 6–7 (and nothing from mainline); no `diff --stat` output for the files that were identical on both branches before the merge; `framework delta only` for `memory_hook.py` and `model_runner.py` (they already differ between the branches by mainline changes on `dsv41`, so a no-diff check cannot apply to them). A `MISMATCH` means the merge resolved something by hand: `git reset --hard HEAD^1` (the merge commit is not pushed yet), record `Task 8: BLOCKED — merge changed <file> beyond the framework delta`, and continue with Tasks 10–12.

- [ ] **Step 3: Push and run both suites**

`git push shared dsv41`. Run the Task 7 Step 4 file list with `<DSV41-CPU>` (the merged tree) and `<DSV41-CPU> <DSV41_SUITE> test/registered/unit/layers/test_engram_lookup_break.py test/manual/dsv41/test_graph_parity.py`. Expected: the framework line equals Task 7 Step 4's exactly; the dsv41 line equals the last dsv41 suite line recorded in the ledger by Task 5 (nothing on the EXL3 side uses the new seams yet). Append `Task 8: complete`.

---

### Task 9: EXL3 in-graph MoE — pinned-tier graph source, `Exl3FusedMoE`, `_apply_graph`, gate

**Blocked if Task 1 fails.**

**Branch:** `dsv41`.

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_expert_format.py` (class attribute)
- Create: `python/sglang/srt/layers/quantization/exl3_fused_moe.py`
- Modify: `python/sglang/srt/layers/quantization/exl3.py` (`apply`; new `_apply_graph`)
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (`_check`, registration)
- Create: `test/registered/unit/layers/quantization/test_exl3_fused_moe.py` (CPU), `test/manual/dsv41/test_exl3_graph_apply_gpu.py` (GPU; the task's GPU step, rerun in Task 16)
- Modify: `test/registered/unit/test_expert_stream_requirements_exl3.py`, `test/registered/unit/layers/moe/test_exl3_expert_format.py`

**Interfaces:**
- Consumes: `PinnedTierRowBackend.keep`, `ExpertStreamer.serves_graph_gather`, `ExpertStreamer.gather`, `hot_cache.tensors`, `hot_cache.capacity`, `hot_cache.scratch_rows`, `exl3_ext()`.
- Produces:
  - `Exl3ExpertFormat.graph_source_kind = "pinned_tier"`;
  - `slot_pointer_tables(tensors: Mapping[str, Tensor], slots: int) -> dict[str, Tensor]` (nine `int64 [slots]`, keys `{gate,up,down}_{trellis,suh,svh}`);
  - `route_tables(remap, expert_count, ones, weights, keep) -> (inv_order, weight_sorted, det)` (a dropped layer, `keep == 0`, has an all-zero `expert_count`);
  - `NUM_ACTIVE` (the probe's `"num_active"`, 6 or -1), `shared_temps(device, hidden, inter) -> tuple[Tensor, ...]` (one set of the four `temp_*` buffers per device and shape, shared by every layer);
  - `Exl3FusedMoE(tensors, slots, hidden, inter, top_k, device)`, `.run(x, topk_weights, remap, keep, act_limit) -> fp32 [1, H]`;
  - `exl3_fused_moe_for(layer, streamer) -> Exl3FusedMoE` (cached on the layer as `layer._exl3_fused_moe`);
  - `Exl3MoEMethod._apply_graph(layer, streamer, x, topk_weights, topk_ids, swiglu_limit)`.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/layers/quantization/test_exl3_fused_moe.py`:
```python
"""Pointer tables and route tables of the fused EXL3 MoE over hot-cache slots (CPU)."""

import pytest
import torch

from sglang.srt.layers.quantization.exl3_fused_moe import route_tables, slot_pointer_tables
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


def test_pointer_tables_address_each_slots_part():
    tensors = {name: torch.zeros((5, 2 if name.startswith("w13") else 1, 16), dtype=torch.int16) for name in NAMES}
    tables = slot_pointer_tables(tensors, 5)
    assert sorted(tables) == sorted(f"{p}_{k}" for p in ("gate", "up", "down") for k in ("trellis", "suh", "svh"))
    for slot in range(5):
        assert tables["gate_trellis"][slot].item() == tensors["w13_trellis"][slot, 0].data_ptr()
        assert tables["up_svh"][slot].item() == tensors["w13_svh"][slot, 1].data_ptr()
        assert tables["down_suh"][slot].item() == tensors["w2_suh"][slot, 0].data_ptr()
    assert all(t.dtype == torch.int64 and t.shape == (5,) for t in tables.values())


def test_route_tables_sort_routes_by_slot_and_scale_by_keep():
    remap = torch.tensor([4, 1, 3])
    count = torch.zeros(6, dtype=torch.long)
    weights = torch.tensor([0.5, 0.25, 0.125])
    inv_order, weight_sorted, det = route_tables(remap, count, torch.ones(3, dtype=torch.long), weights, torch.tensor([1.0]))
    assert count.tolist() == [0, 1, 0, 1, 1, 0]
    order = torch.argsort(remap)
    assert torch.equal(inv_order[order], torch.arange(3))
    assert weight_sorted.dtype == torch.float16 and weight_sorted.tolist() == [0.25, 0.125, 0.5]
    assert det[0].tolist() == [0, 0, 1, 1, 2, 3] and det[2].tolist() == [0, 1, 0, 1, 1, 0]
    _, dropped, _ = route_tables(remap, count, torch.ones(3, dtype=torch.long), weights, torch.tensor([0.0]))
    assert dropped.tolist() == [0.0, 0.0, 0.0]
    assert count.tolist() == [0] * 6  # a dropped layer runs no expert at all


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```
Append to `test/registered/unit/layers/moe/test_exl3_expert_format.py`:
```python
def test_the_exl3_format_serves_graph_gathers_from_its_pinned_tier():
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.expert_format import graph_source_kind_of

    assert graph_source_kind_of(Exl3ExpertFormat) == "pinned_tier"
```
In `test/registered/unit/test_expert_stream_requirements_exl3.py` add:
```python
def test_graph_gather_over_the_pinned_tier_needs_breakable_decode(model_dir):
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)
    with pytest.raises(ValueError, match="GRAPH_GATHER"):
        _gate(_launch(model_dir), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)
    with pytest.raises(ValueError, match="PINNED_HOST_MB"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, SGLANG_MOE_PINNED_HOST_MB=0)


def test_the_exl3_requirements_read_graph_gathers_from_the_pinned_tier(model_dir):
    args = _launch(model_dir)
    assert expert_stream_requirements_for(args, args).graph_gather_host_source == "pinned_tier"
```
Also in the same file's parametrized refusals, the existing `SGLANG_MOE_EXPERT_HOST_ARENA` case stays as it is.

Create `test/manual/dsv41/test_exl3_graph_apply_gpu.py`:
```python
"""In-graph EXL3 MoE over pinned-tier graph gather equals the eager streamed apply (GPU, window).

Fake finite EXL3 checkpoint (1 layer, 12 experts); pinned tier of 8 rows, hot cache of
3 slots + 6 scratch; top-6 routes. Captures streamer.gather + Exl3FusedMoE.run in a
CUDA graph, replays for new routes whose rows are all in RAM, and compares with
Exl3MoEMethod._apply_streamed on the same inputs (rel <= 1.2e-2, the probe's bar).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 12, 6


def _layer(tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layer = torch.nn.Module()
    layer.layer_id = 0
    layer.top_k = TOP_K
    fmt = Exl3ExpertFormat(build_exl3_expert_layout(str(tmp_path)), 0, direct=False)
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
        streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
    pinned = ExpertPinnedHostCache(streamer, 8)
    hot = ExpertHotCache(streamer, 3, scratch_rows=TOP_K)
    hot.reassign([0, 1, 2])
    pinned.ensure_rows(torch.arange(8, device="cuda"))
    streamer.enable_graph_gather(TOP_K)
    layer._nvfp4_expert_streamer = streamer
    layer._exl3_allow_p3_only = True  # test configuration: P3's backend without option C (Task 14)
    return layer, streamer


def test_captured_in_graph_moe_matches_the_eager_streamed_apply(tmp_path):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer = _layer(tmp_path)
    gen = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
    weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
    ids = torch.tensor([[0, 3, 5, 1, 7, 6]], device="cuda", dtype=torch.int32)
    Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)  # warm up, allocate
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
    for route in ([2, 4, 6, 0, 1, 3], [7, 5, 3, 1, 0, 2]):
        ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        got = out.float().clone()
        want = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), 10.0).float()
        rel = float((got - want).norm() / want.norm())
        assert rel <= 1.2e-2, (route, rel)
        assert streamer.row_backend.keep.item() == 1.0
```

- [ ] **Step 2: Commit red, push, run red**

`git add test/registered/unit/layers/quantization/test_exl3_fused_moe.py test/registered/unit/layers/moe/test_exl3_expert_format.py test/registered/unit/test_expert_stream_requirements_exl3.py test/manual/dsv41/test_exl3_graph_apply_gpu.py`; commit `test(dsv41): red tests for the in-graph EXL3 MoE over the pinned tier`. Run `<DSV41-CPU> test/registered/unit/layers/quantization/test_exl3_fused_moe.py test/registered/unit/layers/moe/test_exl3_expert_format.py test/registered/unit/test_expert_stream_requirements_exl3.py test/manual/dsv41/test_exl3_graph_apply_gpu.py`. Expected: import error for `exl3_fused_moe`; the format test fails (`"dense" == "pinned_tier"`); the two gate tests fail; the GPU file skips.

- [ ] **Step 3: Implement the fused MoE module**

Create `python/sglang/srt/layers/quantization/exl3_fused_moe.py`:
```python
"""exllamav3's fused exl3_moe over hot-cache slots, for in-graph decode at BS1.

The graph gather returns ``remap``: one hot-cache slot per route (hits in place,
misses in scratch rows), distinct at BS1. The fused kernel treats slots as its
"experts": nine pointer tables hold every slot's row addresses (fixed for the
life of the hot cache), ``expert_count`` marks the routed slots, and the
deterministic path (output scratch + exl3_moe_gather, the FUSED_DET mode) makes
replays bitwise reproducible. Every op here is capture-safe: no host reads, no
allocation that depends on data. This is the call sequence the P2 probe
(test/manual/dsv41/test_exl3_moe_probe_gpu.py) gated.
"""

from __future__ import annotations

from typing import Mapping

import torch

from sglang.srt.layers.quantization.exl3_ext import exl3_ext

ACT_SILU = 0
ROW_TILE = 16  # fused-kernel rows per slot tile; BS1 puts one route on a slot
# The P2 probe's "num_active" (Task 1 Step 5, $ANA/probe-exl3-moe.json): 6 when the
# static six-expert launch passed parity and bitwise replay, else -1 (all-fused).
NUM_ACTIVE = 6

_SHARED_TEMPS: dict = {}


def shared_temps(device, hidden: int, inter: int):
    """The fused kernel's four temp buffers, one set per device and shape for every layer.

    Layers run one after another on one stream, so they never use the set at once;
    per-layer sets would cost ~10 MB x 40 layers.
    """
    key = (str(device), hidden, inter)
    temps = _SHARED_TEMPS.get(key)
    if temps is None:
        concurrency = exl3_ext().exl3_moe_max_concurrency(torch.device(device).index)
        half = dict(dtype=torch.float16, device=device)
        temps = (
            torch.empty((concurrency, ROW_TILE, hidden), **half),
            torch.empty((concurrency, ROW_TILE, hidden), **half),
            torch.empty((concurrency, ROW_TILE, inter), **half),
            torch.empty((concurrency, ROW_TILE, inter), **half),
        )
        _SHARED_TEMPS[key] = temps
    return temps

_PROJECTIONS = (("gate", "w13", 0), ("up", "w13", 1), ("down", "w2", 0))
_KINDS = ("trellis", "suh", "svh")


def slot_pointer_tables(tensors: Mapping[str, torch.Tensor], slots: int) -> dict[str, torch.Tensor]:
    """Nine int64 [slots] tables of row addresses: gate = w13 part 0, up = w13 part 1, down = w2."""
    device = tensors["w13_trellis"].device
    return {
        f"{proj}_{kind}": torch.tensor(
            [tensors[f"{prefix}_{kind}"][slot, part].data_ptr() for slot in range(slots)],
            dtype=torch.int64,
            device=device,
        )
        for proj, prefix, part in _PROJECTIONS
        for kind in _KINDS
    }


def route_tables(remap, expert_count, ones, weights, keep):
    """Fill ``expert_count`` from ``remap``; return (inv_order, weight_sorted fp16, det tables).

    ``keep`` (fp32 [1]) scales every route weight and, when 0, empties ``expert_count``,
    so a dropped layer runs no expert.
    ``det`` is exllamav3's device-built deterministic table stack
    ``[expert_start, expert_start, count > 0]``.
    """
    expert_count.zero_().index_add_(0, remap, ones)
    order = torch.argsort(remap)
    inv_order = torch.empty_like(order).scatter_(
        0, order, torch.arange(order.numel(), device=order.device)
    )
    weight_sorted = (weights[order].float() * keep).to(torch.float16)
    # A dropped layer runs no expert: nothing reads rows that may be half written.
    expert_count.mul_((keep > 0).to(torch.int64))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    det = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
    return inv_order, weight_sorted, det


class Exl3FusedMoE:
    """Static buffers and pointer tables of one streamed layer's in-graph fused MoE."""

    def __init__(self, tensors: Mapping[str, torch.Tensor], slots: int, hidden: int, inter: int, top_k: int, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Exl3FusedMoE must be built before CUDA-graph capture (in a warmup)")
        ext = exl3_ext()
        self.ext = ext
        self.slots = slots
        self.tables = slot_pointer_tables(tensors, slots)
        self.bits = {
            "gate": tensors["w13_trellis"].shape[-1] // 16,
            "up": tensors["w13_trellis"].shape[-1] // 16,
            "down": tensors["w2_trellis"].shape[-1] // 16,
        }
        half = dict(dtype=torch.float16, device=device)
        self.expert_count = torch.zeros(slots + 1, dtype=torch.int64, device=device)
        self.ones = torch.ones(top_k, dtype=torch.int64, device=device)
        self.token_sorted = torch.zeros(top_k, dtype=torch.int64, device=device)
        self.scratch = torch.empty((top_k, hidden), dtype=torch.float32, device=device)
        self.out = torch.empty((1, hidden), dtype=torch.float32, device=device)
        self.x16 = torch.empty((1, hidden), **half)
        (
            self.temp_state_g,
            self.temp_state_u,
            self.temp_intermediate_g,
            self.temp_intermediate_u,
        ) = shared_temps(device, hidden, inter)

    def run(self, x, topk_weights, remap, keep, act_limit: float) -> torch.Tensor:
        """x [1, H] any float dtype; topk_weights [6]; remap int64 [6] slots; keep fp32 [1]."""
        self.x16.copy_(x)
        inv_order, weight_sorted, det = route_tables(remap, self.expert_count, self.ones, topk_weights, keep)
        self.out.zero_()
        t = self.tables
        self.ext.exl3_moe(
            self.x16, self.out, self.expert_count, self.token_sorted, weight_sorted,
            self.temp_state_g, self.temp_state_u, self.temp_intermediate_g, self.temp_intermediate_u,
            ACT_SILU, self.bits["gate"], self.bits["up"], self.bits["down"],
            t["gate_trellis"], t["gate_suh"], t["gate_svh"],
            t["up_trellis"], t["up_suh"], t["up_svh"],
            t["down_trellis"], t["down_suh"], t["down_svh"],
            False, True, False, True, False, True,
            float(act_limit), NUM_ACTIVE, self.scratch, det[0], 1, ROW_TILE, 16,
        )
        self.ext.exl3_moe_gather(
            self.out, self.scratch, remap, inv_order,
            det[1, : self.slots], det[0, : self.slots], det[2, : self.slots], weight_sorted,
        )
        return self.out


def exl3_fused_moe_for(layer, streamer) -> Exl3FusedMoE:
    """The layer's fused MoE, built on first use (a warmup forward, before capture)."""
    fused = getattr(layer, "_exl3_fused_moe", None)
    if fused is None:
        cache = streamer.hot_cache
        slots = cache.capacity + cache.scratch_rows
        fused = Exl3FusedMoE(
            cache.tensors,
            slots,
            hidden=cache.tensors["w13_suh"].shape[-1],
            inter=cache.tensors["w2_suh"].shape[-1],
            top_k=streamer.graph_gather_rows,
            device=cache.device,
        )
        layer._exl3_fused_moe = fused
    return fused
```
Before committing, read the probe's choice on divix01 (`/data/models/slang/.venv/bin/python -c "import json; print(json.load(open('/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/probe-exl3-moe.json'))['num_active'])"` inside `ssh divix01 '...'`) and set `NUM_ACTIVE` to it (6 or -1). `_apply_graph` requires a `swiglu_limit` (DSV4.1 always sets 10.0), so no "no clamp" value of `act_limit` is ever relied on.

- [ ] **Step 4: Implement `_apply_graph`, the format flag and the gate**

`exl3_expert_format.py` — in `class Exl3ExpertFormat`, after `supports_host_arena = False`:
```python
    # Graph gathers read missed rows from the pinned host tier by pinned slot
    # (PinnedTierRowBackend); there is no dense [experts, ...] host source.
    graph_source_kind = "pinned_tier"
```
`exl3.py` — in `Exl3MoEMethod.apply`, delete the first-line `assert_not_capturing("Exl3MoEMethod.apply")`, and replace
```python
        streamer = expert_streamer_of(layer)
        if streamer is not None:
            out = self._apply_streamed(
```
with
```python
        streamer = expert_streamer_of(layer)
        if streamer is not None and streamer.serves_graph_gather(topk):
            out = self._apply_graph(
                layer, streamer, dispatch_output.hidden_states, topk_weights, topk_ids, cfg.swiglu_limit
            )
        elif streamer is not None:
            assert_not_capturing("Exl3MoEMethod.apply")
            out = self._apply_streamed(
```
and put `assert_not_capturing("Exl3MoEMethod.apply")` as the first line of the final `else:` branch (the resident `exl3_moe_loop`). Add the method:
```python
    @staticmethod
    def _apply_graph(layer, streamer, x, topk_weights, topk_ids, swiglu_limit):
        """BS1 decode inside a CUDA graph: device-only gather, then the fused MoE over slots.

        Hits are read in place from the hot cache, misses land in scratch rows from
        the pinned tier (PinnedTierRowBackend), and routes are recorded on the device
        by the planner. A row missing from RAM sets the backend's ``keep`` to 0,
        which drops this layer's routed output; the host fail-stop check stops the
        process after the forward.
        """
        from sglang.srt.layers.quantization.exl3_fused_moe import exl3_fused_moe_for

        if swiglu_limit is None:
            raise NotImplementedError("exl3 in-graph MoE: a swiglu_limit is required (DSV4.1 sets 10.0)")
        remap, _ = streamer.gather(topk_ids)
        fused = exl3_fused_moe_for(layer, streamer)
        out = fused.run(
            x,
            topk_weights.reshape(-1),
            remap.reshape(-1).long(),
            streamer.row_backend.keep,
            swiglu_limit,
        )
        return out.to(x.dtype)
```
`expert_stream_requirements_exl3.py` — add `import dataclasses` at the top of the imports, and in `_check` replace the final `_EAGER.check(_EagerGraphView(cfg), budgets)` with:
```python
    if budgets.graph_gather and not budgets.pinned_budget_mb:
        raise ValueError(
            "EXL3 graph gathers read missed rows from the pinned host tier; "
            "set SGLANG_MOE_PINNED_HOST_MB"
        )
    if budgets.graph_gather and not budgets.hot_budget_mb:
        raise ValueError("EXL3 graph gathers need SGLANG_MOE_HOT_GPU_MB")
    # The shared eager check refuses graph gather; decode graphs may use it.
    _EAGER.check(_EagerGraphView(cfg), dataclasses.replace(budgets, graph_gather=False))
```
and replace the registration with:
```python
exl3_expert_stream_requirements = ExpertStreamRequirements(
    "EXL3", _check, graph_gather_host_source="pinned_tier"
)
register_expert_stream_requirements(("exl3",), exl3_expert_stream_requirements)
```

- [ ] **Step 5: Commit, push, run green**

`git add python/sglang/srt/layers/moe/exl3_expert_format.py python/sglang/srt/layers/quantization/exl3_fused_moe.py python/sglang/srt/layers/quantization/exl3.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`; commit `feat(dsv41): in-graph EXL3 MoE over pinned-tier graph gather and fused exl3_moe`. Run `<DSV41-CPU> <DSV41_SUITE> test/registered/unit/layers/test_engram_lookup_break.py test/manual/dsv41/test_graph_parity.py test/registered/unit/layers/quantization/test_exl3_fused_moe.py test/manual/dsv41/test_exl3_graph_apply_gpu.py`. Expected: Task 8's dsv41 line + 1 format + 2 gate + 2 fused tests passed, 1 skipped (the GPU file).

- [ ] **Step 6: Run the GPU test**

`<GPU> /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/manual/dsv41/test_exl3_graph_apply_gpu.py`. Expected: `1 passed`. A failure is fixed by a red/green pair in this task's files and this step rerun; exit code 75 is the GPU lock rule. Append `Task 9: complete`.

**P3 ends here.** With `SGLANG_MOE_EXPERT_GRAPH_GATHER=1`, decode runs the MoE in-graph. `PinnedTierRowBackend` alone only counts a RAM miss (`ram_miss`) and drops the layer (`keep = 0`); nothing on the host checks that counter. So P3 without option C is a test configuration only: Task 14 makes `_apply_graph` refuse to run on a plain `PinnedTierRowBackend` unless the layer sets `_exl3_allow_p3_only` (as this task's GPU test does), and option C's wait kernel raises the fatal word for any planned row still missing.

---

## Phase P4 — option C (io_uring thread + device wait)

### Task 10: Option C tables and the EXL3 split in C++ (byte-equal to `Exl3ShardRowSource`)

**Blocked if Task 1 fails.**

**Branch:** `dsv41`. CPU only. This task creates the C++ file with its row reader (io_uring superset reads + the per-name split, D16) and the tables that feed it; Tasks 11–12 add the slot bookkeeping and the thread to the same file.

**Files:**
- Create: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`Exl3RamMissTables`, `exl3_ram_miss_tables`)
- Create: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (`Tables`, `tables_from`, `RowReader`, export `exl3_ram_miss_read_rows`)
- Create: `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (`_host_module`, `read_rows_once`)
- Create: `python/sglang/test/dsv41_ram_miss_fixtures.py` (shared CPU fixture for Tasks 10–12 and 15)
- Create: `test/registered/unit/kernels/test_exl3_ram_miss_split.py`

**Interfaces:**
- Produces:
  - `Exl3RamMissTables` (frozen dataclass): `layer_ids: list[int]`, `paths: list[str]`, `file_sizes` int64 `[F]`, `reads` int64 `[L, E, 4]` (file index, aligned offset, aligned length, row start), `segments` int64 `[S, 4]` (name index, dst offset, src offset, bytes), `slabs` int64 `[L, 6]` (slab addresses in `EXL3_STREAMED_NAMES` order), `row_bytes` int64 `[6]`, `capacity` int64 `[L]`, `slot_bytes: int`;
  - `exl3_ram_miss_tables(layout, segments, slabs_by_layer) -> Exl3RamMissTables` (rows in ascending layer id);
  - C++ `Tables`, `tables_from(...)`, `RowReader(tables, direct)` with `open()` and `read(row, experts, slots, step, abandon) -> int` (1 read, 0 failed, -1 abandoned);
  - `read_rows_once(tables, row, experts, slots, *, direct) -> int` (same codes);
  - fixture `ram_miss_setup(tmp_path, *, capacity=3, layers=2, experts=6) -> RamMissSetup` with fields `layout, fmt, specs, slabs, tables` and `reference(layer, experts) -> dict[name, Tensor]`.

- [ ] **Step 1: Write the fixture and the failing tests**

Create `python/sglang/test/dsv41_ram_miss_fixtures.py`:
```python
"""A fake EXL3 checkpoint plus per-layer pinned-slab stand-ins for the option C CPU tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.test.dsv41_fake_exl3 import write_fake_exl3


@dataclass
class RamMissSetup:
    layout: object
    fmt: Exl3ExpertFormat
    specs: dict
    slabs: dict
    tables: object

    def reference(self, layer: int, experts: list[int]) -> dict[str, torch.Tensor]:
        """Exl3ShardRowSource's split of ``experts`` of ``layer`` (the byte oracle)."""
        out = {
            name: torch.empty((len(experts),) + self.specs[name].row_shape, dtype=self.specs[name].dtype)
            for name in EXL3_STREAMED_NAMES
        }
        Exl3ShardRowSource.for_layer(self.layout, layer, self.fmt.segment_map(), direct=False).read(
            torch.tensor(experts), out
        )
        return out


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def ram_miss_setup(tmp_path, *, capacity: int = 3, layers: int = 2, experts: int = 6) -> RamMissSetup:
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables

    write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=experts)
    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    slabs = {
        layer: {
            name: allocate_host_slab(capacity, specs[name].row_shape, specs[name].dtype, register=False)
            for name in EXL3_STREAMED_NAMES
        }
        for layer in range(layers)
    }
    return RamMissSetup(layout, fmt, specs, slabs, exl3_ram_miss_tables(layout, fmt.segment_map(), slabs))
```
Create `test/registered/unit/kernels/test_exl3_ram_miss_split.py`:
```python
"""The C++ row reader reads and splits EXL3 rows byte for byte like Exl3ShardRowSource (CPU)."""

import os

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import read_rows_once
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def test_tables_describe_every_row(tmp_path):
    s = ram_miss_setup(tmp_path)
    assert s.tables.reads.shape == (2, 6, 4)
    assert s.tables.capacity.tolist() == [3, 3]
    assert s.tables.slabs[1, 0].item() == s.slabs[1]["w13_trellis"].data_ptr()
    assert s.tables.slot_bytes % 4096 == 0
    assert s.tables.segments.shape == (len(s.fmt.segment_map()), 4)
    assert s.tables.row_bytes.tolist() == [
        s.slabs[0][n].numel() * s.slabs[0][n].element_size() // 3 for n in EXL3_STREAMED_NAMES
    ]


@pytest.mark.parametrize("layer, experts, slots", [(0, [0], [2]), (1, [5, 2, 3], [0, 1, 2])])
def test_rows_are_split_like_the_python_row_source(tmp_path, layer, experts, slots):
    s = ram_miss_setup(tmp_path)
    assert read_rows_once(s.tables, layer, experts, slots, direct=False) == 1
    reference = s.reference(layer, experts)
    for i, slot in enumerate(slots):
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[layer][name][slot], reference[name][i]), (name, experts[i])


def test_the_last_row_of_a_shard_is_clamped_at_end_of_file(tmp_path):
    s = ram_miss_setup(tmp_path)
    # write_fake_exl3 puts 3 experts per shard: expert 5 of layer 1 is the last row of the last shard.
    assert read_rows_once(s.tables, 1, [5], [0], direct=False) == 1
    assert same_bytes(s.slabs[1]["w2_svh"][0], s.reference(1, [5])["w2_svh"][0])


def test_a_short_file_fails_the_read(tmp_path):
    s = ram_miss_setup(tmp_path)
    path = s.tables.paths[int(s.tables.reads[0, 0, 0])]
    with open(path, "r+b") as f:
        f.truncate(int(s.tables.reads[0, 0, 1]) + 100)  # cut inside expert 0's superset
    assert read_rows_once(s.tables, 0, [0], [0], direct=False) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```

- [ ] **Step 2: Commit red, push, run red**

`git add python/sglang/test/dsv41_ram_miss_fixtures.py test/registered/unit/kernels/test_exl3_ram_miss_split.py`; commit `test(dsv41): red tests for the option C tables and C++ split`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_split.py`. Expected: collection error `No module named 'sglang.kernels.ops.moe.exl3_ram_miss'`.

- [ ] **Step 3: Implement the tables**

Create `python/sglang/srt/layers/moe/exl3_ram_miss.py`:
```python
"""Option C for EXL3 streamed experts (plan D8-D23).

``exl3_ram_miss_tables`` flattens what ``Exl3ShardRowSource`` knows (the per-expert
superset reads and the per-name segment map) plus each layer's pinned slabs into
int64 tensors, so the C++ thread reads and splits rows without Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES


@dataclass(frozen=True)
class Exl3RamMissTables:
    layer_ids: list[int]
    paths: list[str]
    file_sizes: torch.Tensor  # int64 [F]
    reads: torch.Tensor  # int64 [L, E, 4]: file index, aligned offset, aligned length, row start
    segments: torch.Tensor  # int64 [S, 4]: name index, dst offset, src offset, bytes
    slabs: torch.Tensor  # int64 [L, 6]: slab base addresses in EXL3_STREAMED_NAMES order
    row_bytes: torch.Tensor  # int64 [6]
    capacity: torch.Tensor  # int64 [L]
    slot_bytes: int


def exl3_ram_miss_tables(
    layout: Exl3ExpertLayout,
    segments: Sequence[RowSegment],
    slabs_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
) -> Exl3RamMissTables:
    """Tables for the streamed layers in ``slabs_by_layer`` (ascending layer id = row order)."""
    layer_ids = sorted(slabs_by_layer)
    paths: list[str] = []
    file_index: dict[str, int] = {}
    reads = torch.empty((len(layer_ids), layout.num_experts, 4), dtype=torch.int64)
    widest = 0
    for row, layer_id in enumerate(layer_ids):
        for expert in range(layout.num_experts):
            record = layout.records[(layer_id, expert)]
            if record.path not in file_index:
                file_index[record.path] = len(paths)
                paths.append(record.path)
            offset, length, start = record.aligned_read(PAGE_BYTES)
            reads[row, expert] = torch.tensor([file_index[record.path], offset, length, start])
            widest = max(widest, length)
    names = {name: index for index, name in enumerate(EXL3_STREAMED_NAMES)}
    segment_table = torch.tensor(
        [[names[s.name], s.dst_offset, s.src_offset, s.nbytes] for s in segments], dtype=torch.int64
    )
    row_bytes = torch.tensor(
        [sum(s.nbytes for s in segments if s.name == name) for name in EXL3_STREAMED_NAMES], dtype=torch.int64
    )
    slabs = torch.empty((len(layer_ids), len(EXL3_STREAMED_NAMES)), dtype=torch.int64)
    capacity = torch.empty(len(layer_ids), dtype=torch.int64)
    for row, layer_id in enumerate(layer_ids):
        tensors = slabs_by_layer[layer_id]
        rows = {int(tensors[name].shape[0]) for name in EXL3_STREAMED_NAMES}
        if len(rows) != 1:
            raise ValueError(f"layer {layer_id}: pinned slabs disagree on their row count {rows}")
        capacity[row] = rows.pop()
        for name, index in names.items():
            slab = tensors[name]
            if not slab.is_contiguous() or slab.device.type != "cpu":
                raise ValueError(f"layer {layer_id} {name}: slab must be a contiguous CPU tensor")
            per_row = slab.numel() * slab.element_size() // max(int(capacity[row]), 1)
            if per_row != int(row_bytes[index]):
                raise ValueError(f"layer {layer_id} {name}: slab rows hold {per_row} B, expected {int(row_bytes[index])}")
            slabs[row, index] = slab.data_ptr()
    return Exl3RamMissTables(
        layer_ids=layer_ids,
        paths=paths,
        file_sizes=torch.tensor([os.path.getsize(p) for p in paths], dtype=torch.int64),
        reads=reads,
        segments=segment_table,
        slabs=slabs,
        row_bytes=row_bytes,
        capacity=capacity,
        slot_bytes=-(-widest // PAGE_BYTES) * PAGE_BYTES,
    )
```

- [ ] **Step 4: Implement the C++ reader**

Create `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp`:
```cpp
// Option C RAM-miss service for EXL3 streamed experts (DSV41 Phase 3b plan, D8-D19).
//
// This file grows in three plan tasks: the row reader (Task 10: io_uring superset
// reads into a page-aligned bounce, then Exl3ShardRowSource's per-name split into the
// pinned slabs), the C++-owned slot bookkeeping and request service (Task 11), and
// the service thread with its watchdog (Task 12). Nothing here makes a CUDA call:
// every write is a CPU store into (pinned) host memory (plan D9).

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <fcntl.h>
#include <immintrin.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sglang {
namespace exl3_ram_miss {

using tvm::ffi::TensorView;

constexpr int kBounceRows = 8;
constexpr unsigned kQueueDepth = 16;
constexpr int64_t kPage = 4096;

inline int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

struct Segment {
  int64_t name;
  int64_t dst;
  int64_t src;
  int64_t bytes;
};

struct Read {
  int64_t file;
  int64_t offset;
  int64_t length;
  int64_t start;
};

struct Tables {
  int64_t layers = 0;
  int64_t experts = 0;
  int64_t slot_bytes = 0;
  std::vector<std::string> paths;
  std::vector<int64_t> file_sizes;
  std::vector<Read> reads;
  std::vector<Segment> segments;
  std::vector<std::vector<uint8_t*>> slabs;
  std::vector<int64_t> row_bytes;
};

inline std::vector<int32_t> ids_of(TensorView tensor) {
  const auto* data = static_cast<const int64_t*>(tensor.data_ptr());
  return std::vector<int32_t>(data, data + tensor.size(0));
}

inline Tables tables_from(
    TensorView reads,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    const std::string& paths,
    int64_t slot_bytes) {
  Tables t;
  t.layers = reads.size(0);
  t.experts = reads.size(1);
  t.slot_bytes = slot_bytes;
  size_t start = 0;
  while (true) {
    const size_t end = paths.find('\n', start);
    t.paths.push_back(paths.substr(start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  const auto* sizes = static_cast<const int64_t*>(file_sizes.data_ptr());
  t.file_sizes.assign(sizes, sizes + file_sizes.size(0));
  const auto* read_data = static_cast<const int64_t*>(reads.data_ptr());
  t.reads.resize(static_cast<size_t>(t.layers * t.experts));
  for (size_t i = 0; i < t.reads.size(); ++i) {
    t.reads[i] = Read{read_data[4 * i], read_data[4 * i + 1], read_data[4 * i + 2], read_data[4 * i + 3]};
  }
  const auto* segment_data = static_cast<const int64_t*>(segments.data_ptr());
  t.segments.resize(static_cast<size_t>(segments.size(0)));
  for (size_t i = 0; i < t.segments.size(); ++i) {
    t.segments[i] = Segment{segment_data[4 * i], segment_data[4 * i + 1], segment_data[4 * i + 2], segment_data[4 * i + 3]};
  }
  const auto* slab_data = static_cast<const int64_t*>(slabs.data_ptr());
  const int64_t names = slabs.size(1);
  t.slabs.resize(static_cast<size_t>(t.layers));
  for (int64_t row = 0; row < t.layers; ++row) {
    for (int64_t name = 0; name < names; ++name) {
      t.slabs[row].push_back(reinterpret_cast<uint8_t*>(static_cast<intptr_t>(slab_data[row * names + name])));
    }
  }
  const auto* rows = static_cast<const int64_t*>(row_bytes.data_ptr());
  t.row_bytes.assign(rows, rows + row_bytes.size(0));
  return t;
}

// io_uring superset reads of whole expert rows into a page-aligned bounce, then the
// per-name split into the pinned slabs (Exl3ShardRowSource.read's copies).
class RowReader {
 public:
  RowReader(Tables tables, bool direct) : t_(std::move(tables)), direct_(direct) {}

  ~RowReader() {
    if (ring_ready_) io_uring_queue_exit(&ring_);
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
  }

  const Tables& tables() const { return t_; }

  bool open() {
    for (const auto& path : t_.paths) {
      const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (direct_ ? O_DIRECT : 0));
      if (fd < 0) {
        std::fprintf(stderr, "ERROR exl3 RAM miss: open %s: %s\n", path.c_str(), std::strerror(errno));
        return false;
      }
      fds_.push_back(fd);
    }
    if (posix_memalign(reinterpret_cast<void**>(&bounce_), kPage, static_cast<size_t>(kBounceRows * t_.slot_bytes)) != 0) {
      bounce_ = nullptr;
      return false;
    }
    if (io_uring_queue_init(kQueueDepth, &ring_, 0) != 0) return false;
    ring_ready_ = true;
    return true;
  }

  // Read `experts` of streamed row `row` into `slots`, `step` rows per io_uring batch
  // (at most kBounceRows). `abandon()` runs before each batch; true stops the read.
  // Returns 1 when every row landed, 0 on an I/O error or short file, -1 when abandoned.
  int read(
      int64_t row,
      const std::vector<int32_t>& experts,
      const std::vector<int64_t>& slots,
      size_t step,
      const std::function<bool()>& abandon) {
    step = std::max<size_t>(1, std::min<size_t>(step, kBounceRows));
    for (size_t first = 0; first < experts.size(); first += step) {
      if (abandon()) return -1;
      const size_t count = std::min<size_t>(step, experts.size() - first);
      std::vector<int64_t> done(count, 0);
      std::vector<int64_t> expected(count, 0);
      std::vector<const Read*> reads(count);
      for (size_t i = 0; i < count; ++i) {
        reads[i] = &t_.reads[static_cast<size_t>(row * t_.experts + experts[first + i])];
        expected[i] = std::min(reads[i]->length, t_.file_sizes[reads[i]->file] - reads[i]->offset);
      }
      size_t pending = 0;
      auto submit = [&](size_t i) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        const int64_t remaining = reads[i]->length - done[i];
        io_uring_prep_read(
            sqe, fds_[reads[i]->file], bounce_ + i * t_.slot_bytes + done[i], static_cast<unsigned>(remaining),
            static_cast<uint64_t>(reads[i]->offset + done[i]));
        io_uring_sqe_set_data64(sqe, i);
        ++pending;
      };
      for (size_t i = 0; i < count; ++i) submit(i);
      while (pending > 0) {
        if (io_uring_submit_and_wait(&ring_, 1) < 0) return 0;
        io_uring_cqe* cqe;
        unsigned head;
        unsigned seen = 0;
        std::vector<size_t> again;
        bool failed = false;
        io_uring_for_each_cqe(&ring_, head, cqe) {
          ++seen;
          --pending;
          const size_t i = static_cast<size_t>(io_uring_cqe_get_data64(cqe));
          if (cqe->res < 0 || (cqe->res == 0 && done[i] < expected[i])) {
            failed = true;
            continue;
          }
          done[i] += cqe->res;
          if (done[i] < expected[i]) again.push_back(i);
        }
        io_uring_cq_advance(&ring_, seen);
        if (failed) {
          while (pending > 0) {  // drain what is still in flight before the bounce is reused
            io_uring_cqe* rest;
            if (io_uring_wait_cqe(&ring_, &rest) == 0) io_uring_cqe_seen(&ring_, rest);
            --pending;
          }
          return 0;
        }
        for (size_t i : again) submit(i);
      }
      for (size_t i = 0; i < count; ++i) {
        const uint8_t* base = bounce_ + i * t_.slot_bytes + reads[i]->start;
        const int64_t slot = slots[first + i];
        for (const Segment& segment : t_.segments) {
          std::memcpy(
              t_.slabs[row][segment.name] + slot * t_.row_bytes[segment.name] + segment.dst, base + segment.src,
              static_cast<size_t>(segment.bytes));
        }
      }
    }
    return 1;
  }

 private:
  Tables t_;
  bool direct_;
  std::vector<int> fds_;
  uint8_t* bounce_ = nullptr;
  io_uring ring_{};
  bool ring_ready_ = false;
};

}  // namespace exl3_ram_miss

using exl3_ram_miss::TensorView;

// Read `experts` of streamed row `row` into `slots` once, synchronously (tests, tools).
int64_t exl3_ram_miss_read_rows(
    TensorView reads,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    std::string paths,
    int64_t slot_bytes,
    int64_t direct,
    int64_t row,
    TensorView experts,
    TensorView slots) {
  using namespace exl3_ram_miss;
  RowReader reader(tables_from(reads, file_sizes, segments, slabs, row_bytes, paths, slot_bytes), direct != 0);
  if (!reader.open()) return 0;
  const auto* slot_data = static_cast<const int64_t*>(slots.data_ptr());
  return reader.read(
      row, ids_of(experts), std::vector<int64_t>(slot_data, slot_data + slots.size(0)), kBounceRows,
      [] { return false; });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_read_rows, exl3_ram_miss_read_rows);

}  // namespace sglang
```
If the build rejects `std::string` as a typed-function parameter, change it to `tvm::ffi::String paths` and pass `std::string(paths.data(), paths.size())` to `tables_from`; keep that form for every later export in this file. If divix01's liburing lacks `io_uring_sqe_set_data64`/`io_uring_cqe_get_data64` (< 2.2), use `io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(i))` and `reinterpret_cast<uintptr_t>(io_uring_cqe_get_data(cqe))`.

Create `python/sglang/kernels/ops/moe/exl3_ram_miss.py`:
```python
"""Option C RAM-miss service for EXL3 streamed experts: the C++ host module's wrappers.

Page layout and memory ordering are the plan's Design decisions D10-D11. The
device kernels (Task 13) and the host simulator (Task 11) speak the same protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _host_module() -> Module:
    return load_jit(
        "exl3_ram_miss_host",
        cpp_files=["moe/exl3_ram_miss_host.cpp"],
        extra_ldflags=["-luring", "-lpthread"],
        header_only=False,
    )


def _ids(values: Iterable[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64)


def read_rows_once(tables, row: int, experts, slots, *, direct: bool) -> int:
    """Read ``experts`` of streamed row ``row`` into pinned ``slots`` in C++: 1 ok, 0 failed."""
    return int(
        _host_module().exl3_ram_miss_read_rows(
            tables.reads,
            tables.file_sizes,
            tables.segments,
            tables.slabs,
            tables.row_bytes,
            "\n".join(tables.paths),
            tables.slot_bytes,
            int(direct),
            row,
            _ids(experts),
            _ids(slots),
        )
    )
```

- [ ] **Step 5: Commit, push, run green**

`git add python/sglang/srt/layers/moe/exl3_ram_miss.py python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp python/sglang/kernels/ops/moe/exl3_ram_miss.py`; commit `feat(dsv41): option C tables and the EXL3 split in C++`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_split.py` (the first run compiles the module, ~1 min). Expected: `5 passed`. Then `<DSV41-CPU> <DSV41_SUITE>` plus every new dsv41 file so far: the Task 9 line + 5. Append `Task 10: complete`.

---

### Task 11: The C++-owned slot LRU and request service — page protocol, eviction rules, a host-simulated device (pumped, no thread)

**Blocked if Task 1 fails.**

**Branch:** `dsv41`. CPU only. Adds to `exl3_ram_miss_host.cpp` the request page protocol (D10–D11), the per-layer slot bookkeeping with the victim rules (D12), demand/advisory service with separate row counters (I9 of the review), the test-only fault injection, and a host-side simulator of the device kernels. Requests are served by an explicit `pump()` here; Task 12 adds the thread that pumps.

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (insert a second `namespace exl3_ram_miss` block and exports before the final `}  // namespace sglang`)
- Modify: `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (page constants, `new_page`, `page_word`, `sim_post`, `sim_wait`, `Exl3RamMissHost`)
- Create: `test/registered/unit/kernels/test_exl3_ram_miss_tier.py`

**Interfaces:**
- Page: `PAGE_BYTES = 10304`, `RECORD_BYTES = 128`, `DEMAND_RING = 64`, `DEMAND_RECORDS = 16`, `ADVISE_RING = 2112`, `ADVISE_RECORDS = 64`, `MAX_IDS = 8`; `WORDS` offsets `demand_head 0, demand_done 4, fatal 8, stop 12, advise_head 16, advise_done 20, busy_seq 24, heartbeat 28`; record `u32 seq @0 (written last), u16 row @4, u16 need_count @6, u16 protect_count @8, u16 status @10 (0 pending, 1 served, 2 failed), u32 after @12, i32 need[8] @16, i32 protect[8] @48`.
- `COUNTERS` (the C++ `Counter` enum order): `served, touch_only, rows_read, read_errors, evictions, overruns, advisories, advisories_skipped, advisory_rows, late_after_fatal, no_victim, version, running, spin_cpu`.
- `Exl3RamMissHost(tables, *, page, slot_map, direct)` with `pump() -> int` (1 demand, 2 advisory, 0 nothing), `contains`, `touch`, `assign(row, e, protected=(), protected_fallback=True) -> (slot, evicted | None)`, `release`, `mapping(row)`, `slot_to_expert(row)`, `lru_order(row)`, `set_hot(row, experts)`, `version()`, `counters()`, `layer_rows()` (demand rows per layer), `layer_advisory_rows()`, `inject(delay_s=0.0, fail_reads=False, delay_after_demands=0)`, `fatal_seq()`, `stop()`.
- `sim_post(page, row, need, protect, *, advisory=False, after=0) -> int`; `sim_wait(page, seq, timeout_s) -> int` (1 served, 2 failed, 0 timed out, 3 fatal already raised; raises `fatal` like the wait kernel).

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_miss_tier.py`:
```python
"""The C++ slot LRU and request service, pumped by hand against a host-simulated device (CPU)."""

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import (
    DEMAND_RECORDS,
    Exl3RamMissHost,
    new_page,
    page_word,
    sim_post,
    sim_wait,
)
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup, same_bytes

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


@pytest.fixture
def tier(tmp_path, request):
    capacity = getattr(request, "param", 3)
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False)
    yield s, page, slot_map, host
    host.stop()


def _serve(page, host, row, need, protect):
    seq = sim_post(page, row, need=need, protect=protect)
    assert host.pump() == 1
    return sim_wait(page, seq, timeout_s=1.0)


def test_a_demand_is_read_split_and_published(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 1, need=[2, 5], protect=[2, 5]) == 1
    reference = s.reference(1, [2, 5])
    for i, expert in enumerate((2, 5)):
        slot = int(slot_map[1, expert])
        assert slot >= 0 and host.mapping(1)[expert] == slot
        for name in EXL3_STREAMED_NAMES:
            assert same_bytes(s.slabs[1][name][slot], reference[name][i]), (name, expert)
    assert slot_map[0].tolist() == [-1] * 6
    assert host.layer_rows() == [0, 2] and host.counters()["served"] == 1


def test_protected_experts_missing_from_ram_are_read_too(tier):
    s, page, slot_map, host = tier
    # D12: the thread recomputes the missing set from protect, not only from need.
    assert _serve(page, host, 0, need=[1], protect=[1, 4]) == 1
    assert host.contains(0, 1) and host.contains(0, 4) and host.layer_rows() == [2, 0]


def test_eviction_spares_protected_and_hot_experts_and_unmaps_the_victim_first(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[0, 1, 2], protect=[0, 1, 2]) == 1
    host.set_hot(0, [0])
    assert _serve(page, host, 0, need=[], protect=[1]) == 1  # touch-only: 2 becomes the LRU non-hot row
    assert _serve(page, host, 0, need=[4], protect=[4, 1]) == 1
    assert slot_map[0, 2].item() == -1
    assert all(slot_map[0, e].item() >= 0 for e in (0, 1, 4))
    slot = int(slot_map[0, 4])
    assert all(same_bytes(s.slabs[0][n][slot], s.reference(0, [4])[n][0]) for n in EXL3_STREAMED_NAMES)
    assert host.lru_order(0)[-1] == 4 and host.counters()["evictions"] == 1
    assert host.counters()["touch_only"] == 1


@pytest.mark.parametrize("tier", [2], indirect=True)
def test_no_evictable_slot_fails_the_request_and_raises_fatal(tier):
    s, page, slot_map, host = tier
    assert _serve(page, host, 0, need=[0, 1], protect=[0, 1]) == 1
    seq = sim_post(page, 0, need=[3], protect=[3, 0, 1])
    assert host.pump() == 1
    assert sim_wait(page, seq, 1.0) == 2
    assert host.fatal_seq() == seq and host.counters()["no_victim"] == 1
    sim_post(page, 0, need=[], protect=[0])
    assert host.pump() == 1
    assert sim_wait(page, page_word(page, "demand_head"), 1.0) == 3  # sticky


def test_a_failed_read_frees_its_slots_and_reports_failed(tier):
    s, page, slot_map, host = tier
    host.inject(fail_reads=True)
    assert _serve(page, host, 0, need=[1], protect=[1]) == 2
    assert slot_map[0, 1].item() == -1 and not host.contains(0, 1)
    assert host.counters()["read_errors"] == 1


def test_a_record_whose_seq_does_not_match_is_an_overrun(tier):
    s, page, slot_map, host = tier
    seq = sim_post(page, 0, need=[1], protect=[1])
    record = 64 + ((seq - 1) % DEMAND_RECORDS) * 128
    page[record : record + 4].view(torch.int32)[0] = seq + DEMAND_RECORDS  # a lapped ring slot
    assert host.pump() == 1
    assert host.counters()["overruns"] == 1 and not host.contains(0, 1)
    assert sim_wait(page, seq, 1.0) == 2  # never served: the waiting layer fails stop


def test_advisory_rows_are_counted_apart_from_demand_rows(tier):
    s, page, slot_map, host = tier
    sim_post(page, 1, need=[3], protect=[3], advisory=True, after=page_word(page, "demand_head") + 5)
    assert host.pump() == 2
    assert host.contains(1, 3)
    assert host.layer_advisory_rows() == [0, 1] and host.layer_rows() == [0, 0]
    assert host.counters()["advisory_rows"] == 1


def test_python_assign_release(tier):
    s, page, slot_map, host = tier
    slot, evicted = host.assign(1, 3, protected=[3])
    assert evicted is None and host.contains(1, 3) and slot_map[1, 3].item() == slot
    host.assign(1, 4, protected=[4])
    host.assign(1, 0, protected=[0])
    slot5, evicted = host.assign(1, 5, protected=[5])
    assert evicted == 3 and slot_map[1, 3].item() == -1
    host.release(1, slot5)
    assert slot_map[1, 5].item() == -1 and not host.contains(1, 5)
    host.assign(1, 5, protected=[5])
    with pytest.raises(RuntimeError, match="protected"):
        host.assign(1, 2, protected=[0, 4, 5], protected_fallback=False)
    with pytest.raises(ValueError, match="already"):
        host.assign(1, 4)


def test_injected_delay_starts_after_n_demands_that_read(tier):
    import time

    s, page, slot_map, host = tier
    host.inject(delay_s=0.3, delay_after_demands=1)
    started = time.perf_counter()
    assert _serve(page, host, 0, need=[1], protect=[1]) == 1
    assert time.perf_counter() - started < 0.2
    started = time.perf_counter()
    assert _serve(page, host, 0, need=[2], protect=[2]) == 1
    assert time.perf_counter() - started >= 0.3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```

- [ ] **Step 2: Commit red, push, run red**

`git add test/registered/unit/kernels/test_exl3_ram_miss_tier.py`; commit `test(dsv41): red tests for the C++ slot LRU and request service`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_tier.py`. Expected: import error (`cannot import name 'DEMAND_RECORDS'`).

- [ ] **Step 3: Implement the C++ service**

In `exl3_ram_miss_host.cpp`, insert before the final `}  // namespace sglang` line:
```cpp
namespace exl3_ram_miss {

// ---- Request page (plan D10) ----
constexpr int64_t kDemandHead = 0;
constexpr int64_t kDemandDone = 4;
constexpr int64_t kFatal = 8;
constexpr int64_t kAdviseHead = 16;
constexpr int64_t kAdviseDone = 20;
constexpr int64_t kBusySeq = 24;
constexpr int64_t kHeartbeat = 28;
constexpr int64_t kRecordBytes = 128;
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kAdviseRing = kDemandRing + kDemandRecords * kRecordBytes;
constexpr uint32_t kAdviseRecords = 64;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;
constexpr int64_t kRecRow = 4;
constexpr int64_t kRecNeedCount = 6;
constexpr int64_t kRecProtectCount = 8;
constexpr int64_t kRecStatus = 10;
constexpr int64_t kRecAfter = 12;
constexpr int64_t kRecNeed = 16;
constexpr int64_t kRecProtect = 48;
constexpr uint16_t kServed = 1;
constexpr uint16_t kFailed = 2;

enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2 };

enum Counter : int {
  kServedRequests = 0,
  kTouchOnly,
  kRowsRead,
  kReadErrors,
  kEvictions,
  kOverruns,
  kAdvisories,
  kAdvisoriesSkipped,
  kAdvisoryRows,
  kLateAfterFatal,
  kNoVictim,
  kVersion,
  kRunning,
  kSpinCpu,
  kCounterCount,
};

inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release(uint8_t* address, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(address), value, __ATOMIC_RELEASE);
}

inline bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

inline int64_t record_offset(int64_t ring, uint32_t records, uint32_t seq) {
  return ring + static_cast<int64_t>((seq - 1u) % records) * kRecordBytes;
}

struct Request {
  uint32_t seq = 0;
  int64_t row = 0;
  uint32_t after = 0;
  std::vector<int32_t> need;
  std::vector<int32_t> protect;
};

// Seqlock read: the writer stores the payload, fences, then the seq word last, so a
// record whose seq reads `expected` both before and after the payload is whole.
inline bool read_record(const uint8_t* record, uint32_t expected, Request* request) {
  if (load_acquire(record + kRecSeq) != expected) return false;
  uint16_t row, need, protect;
  std::memcpy(&row, record + kRecRow, 2);
  std::memcpy(&need, record + kRecNeedCount, 2);
  std::memcpy(&protect, record + kRecProtectCount, 2);
  std::memcpy(&request->after, record + kRecAfter, 4);
  request->seq = expected;
  request->row = row;
  const auto* need_ids = reinterpret_cast<const int32_t*>(record + kRecNeed);
  const auto* protect_ids = reinterpret_cast<const int32_t*>(record + kRecProtect);
  request->need.assign(need_ids, need_ids + std::min<int>(need, kMaxIds));
  request->protect.assign(protect_ids, protect_ids + std::min<int>(protect, kMaxIds));
  std::atomic_thread_fence(std::memory_order_acquire);
  return load_acquire(record + kRecSeq) == expected;
}

inline void set_status(uint8_t* record, uint16_t status) {
  __atomic_store_n(reinterpret_cast<uint16_t*>(record + kRecStatus), status, __ATOMIC_RELEASE);
}

inline bool listed(const std::vector<int32_t>& ids, int32_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

struct Tier {
  int64_t capacity = 0;
  std::vector<int32_t> slot_to_expert;
  std::vector<uint8_t> state;
  std::vector<uint64_t> stamp;
  std::vector<int32_t> expert_slot;  // assigned slot (LOADING or READY) or -1
  std::vector<uint8_t> hot;
  int64_t rows_demand = 0;
  int64_t rows_advisory = 0;
};

// The pinned-slot bookkeeping of every streamed layer (plan D12) and the service of one
// request at a time. pump_demand/pump_advice are called by one caller at a time: a test's
// pump(), or the Task 12 thread. The Python-facing methods take the same mutex.
class RamTier {
 public:
  RamTier(uint8_t* page, int32_t* slot_map, Tables tables, std::vector<int64_t> capacity, bool direct)
      : page_(page),
        map_(slot_map),
        layers_(tables.layers),
        experts_(tables.experts),
        reader_(std::move(tables), direct),
        tiers_(static_cast<size_t>(layers_)) {
    for (auto& counter : counters_) counter.store(0);
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      tier.capacity = capacity[row];
      tier.slot_to_expert.assign(tier.capacity, -1);
      tier.state.assign(tier.capacity, kFree);
      tier.stamp.assign(tier.capacity, 0);
      tier.expert_slot.assign(experts_, -1);
      tier.hot.assign(experts_, 0);
    }
  }

  bool open() {
    if (!reader_.open()) return false;
    next_demand_ = load_acquire(page_ + kDemandDone) + 1u;
    if (next_demand_ == 0) next_demand_ = 1;
    next_advice_ = load_acquire(page_ + kAdviseDone) + 1u;
    if (next_advice_ == 0) next_advice_ = 1;
    return true;
  }

  uint8_t* page() const { return page_; }
  int64_t busy_since() const { return busy_since_.load(); }
  void set_counter(int index, int64_t value) { counters_[index].store(value); }
  void request_pause(bool paused) { pause_requested_.store(paused); }
  void skip_advice_posted_so_far() { skip_advice_upto_.store(load_acquire(page_ + kAdviseHead)); }
  bool threaded() const { return threaded_.load(); }
  void set_threaded(bool threaded) { threaded_.store(threaded); }

  // Serve the next posted demand record, if any. True when it handled one.
  bool pump_demand() {
    const uint32_t head = load_acquire(page_ + kDemandHead);
    if (head == 0 || !reached(head, next_demand_)) return false;
    if (head - next_demand_ >= kDemandRecords) {
      counters_[kOverruns].fetch_add(head - next_demand_ - (kDemandRecords - 1));
      next_demand_ = head - kDemandRecords + 2u;
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    Request request;
    if (read_record(record, next_demand_, &request)) {
      handle_demand(request, record);
    } else {
      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop
    }
    _mm_sfence();
    store_release(page_ + kDemandDone, next_demand_);
    next_demand_ += 1u;
    return true;
  }

  // Serve (or skip) the next posted advisory record, if any. True when it handled one.
  bool pump_advice() {
    const uint32_t head = load_acquire(page_ + kAdviseHead);
    if (head == 0 || !reached(head, next_advice_)) return false;
    if (head - next_advice_ >= kAdviseRecords) {
      counters_[kAdvisoriesSkipped].fetch_add(head - next_advice_ - (kAdviseRecords - 1));
      next_advice_ = head - kAdviseRecords + 2u;
    }
    uint8_t* record = page_ + record_offset(kAdviseRing, kAdviseRecords, next_advice_);
    Request request;
    const uint32_t skip_upto = skip_advice_upto_.load();
    const bool stale = !read_record(record, next_advice_, &request) ||
                       (skip_upto != 0 && reached(skip_upto, next_advice_)) ||
                       reached(load_acquire(page_ + kDemandHead), request.after + 1u) ||
                       load_acquire(page_ + kFatal) != 0 || pause_requested_.load();
    if (stale) {
      counters_[kAdvisoriesSkipped].fetch_add(1);
    } else {
      in_advice_.store(true);
      counters_[kAdvisories].fetch_add(1);
      serve(request, true);
      in_advice_.store(false);
    }
    store_release(page_ + kAdviseDone, next_advice_);
    next_advice_ += 1u;
    return true;
  }

  // ---- Python-facing bookkeeping; eager callers pause the thread first (Task 12) ----

  bool has(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    return tiers_[row].expert_slot[expert] >= 0;
  }

  void touch(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0) tier.stamp[slot] = ++tick_;
  }

  // A slot for a Python-side read; the map entry is published at once (the device is idle
  // and the thread paused when an eager path calls this). evicted: -1 none, -2 already held.
  int64_t assign(int64_t row, int64_t expert, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    if (tier.expert_slot[expert] >= 0) {
      *evicted = -2;
      return tier.expert_slot[expert];
    }
    const int64_t slot = take_slot_locked(row, protect, fallback, evicted);
    if (slot < 0) return -1;
    tier.slot_to_expert[slot] = static_cast<int32_t>(expert);
    tier.state[slot] = kReady;
    tier.stamp[slot] = ++tick_;
    tier.expert_slot[expert] = static_cast<int32_t>(slot);
    publish_map(row, expert, static_cast<int32_t>(slot));
    counters_[kVersion].fetch_add(1);
    return slot;
  }

  void release(int64_t row, int64_t slot) {
    std::lock_guard<std::mutex> guard(mutex_);
    release_locked(row, slot);
    counters_[kVersion].fetch_add(1);
  }

  void mapping(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const int32_t slot = tier.expert_slot[expert];
      out[expert] = slot >= 0 && tier.state[slot] == kReady ? slot : -1;
    }
  }

  void slot_to_expert(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot) out[slot] = tier.slot_to_expert[slot];
  }

  int64_t lru_order(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    std::vector<int64_t> slots;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kReady) slots.push_back(slot);
    }
    std::sort(slots.begin(), slots.end(), [&](int64_t a, int64_t b) { return tier.stamp[a] < tier.stamp[b]; });
    for (size_t i = 0; i < slots.size(); ++i) out[i] = tier.slot_to_expert[slots[i]];
    return static_cast<int64_t>(slots.size());
  }

  void set_hot(int64_t row, const int64_t* experts, int64_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    std::fill(tier.hot.begin(), tier.hot.end(), 0);
    for (int64_t i = 0; i < count; ++i) {
      if (experts[i] >= 0 && experts[i] < experts_) tier.hot[experts[i]] = 1;
    }
  }

  void layer_rows(int64_t* out, bool advisory) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (int64_t row = 0; row < layers_; ++row) out[row] = advisory ? tiers_[row].rows_advisory : tiers_[row].rows_demand;
  }

  // Test-only faults: sleep `delay_ns` before each advisory read and before each demand
  // read once `after_demands` demands have read rows; report reads as failed.
  void inject(int64_t delay_ns, bool fail_reads, int64_t after_demands) {
    delay_ns_.store(delay_ns);
    fail_reads_.store(fail_reads);
    delay_after_.store(after_demands);
  }

  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i) out[i] = counters_[i].load();
  }

  bool idle() const {
    return reached(load_acquire(page_ + kDemandDone), load_acquire(page_ + kDemandHead)) &&
           load_acquire(page_ + kBusySeq) == 0 && !in_advice_.load();
  }

 private:
  void publish_map(int64_t row, int64_t expert, int32_t slot) {
    __atomic_store_n(map_ + row * experts_ + expert, slot, __ATOMIC_RELEASE);
  }

  int64_t take_slot_locked(int64_t row, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    Tier& tier = tiers_[row];
    *evicted = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) return slot;
    }
    int64_t best = -1;
    int64_t spare = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] != kReady) continue;
      const int32_t expert = tier.slot_to_expert[slot];
      if (tier.hot[expert]) continue;
      if (listed(protect, expert)) {
        if (spare < 0 || tier.stamp[slot] < tier.stamp[spare]) spare = slot;
        continue;
      }
      if (best < 0 || tier.stamp[slot] < tier.stamp[best]) best = slot;
    }
    if (best < 0 && fallback) best = spare;
    if (best < 0) {
      counters_[kNoVictim].fetch_add(1);
      return -1;
    }
    const int32_t victim = tier.slot_to_expert[best];
    publish_map(row, victim, -1);  // unmapped before its bytes are overwritten (D11)
    tier.expert_slot[victim] = -1;
    tier.slot_to_expert[best] = -1;
    tier.state[best] = kFree;
    *evicted = victim;
    counters_[kEvictions].fetch_add(1);
    return best;
  }

  void release_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const int32_t expert = tier.slot_to_expert[slot];
    if (expert >= 0) {
      publish_map(row, expert, -1);
      tier.expert_slot[expert] = -1;
    }
    tier.slot_to_expert[slot] = -1;
    tier.state[slot] = kFree;
  }

  bool demand_pending() const { return !reached(next_demand_ - 1u, load_acquire(page_ + kDemandHead)); }

  // Touch the request's assigned rows; read every protected or needed expert that is not
  // assigned (D12's recompute), evicting only unprotected, non-hot READY rows; publish.
  // An advisory protects only its own ids, reads one row at a time and gives up when a
  // demand is posted or a pause is requested (its rows so far are released).
  bool serve(const Request& request, bool advisory) {
    std::vector<int32_t> wanted = request.protect;
    for (int32_t expert : request.need) {
      if (!listed(wanted, expert)) wanted.push_back(expert);
    }
    std::vector<int32_t> missing;
    std::vector<int64_t> slots;
    bool ok = request.row >= 0 && request.row < layers_;
    if (ok) {
      std::lock_guard<std::mutex> guard(mutex_);
      Tier& tier = tiers_[request.row];
      for (int32_t expert : wanted) {
        if (expert < 0 || expert >= experts_) {
          ok = false;
          break;
        }
        const int32_t slot = tier.expert_slot[expert];
        if (slot >= 0) {
          tier.stamp[slot] = ++tick_;
        } else {
          missing.push_back(expert);
        }
      }
      for (size_t i = 0; ok && i < missing.size(); ++i) {
        int64_t evicted = -1;
        const int64_t slot = take_slot_locked(request.row, wanted, false, &evicted);
        if (slot < 0) {
          ok = false;
          break;
        }
        tier.slot_to_expert[slot] = missing[i];
        tier.state[slot] = kLoading;
        tier.expert_slot[missing[i]] = static_cast<int32_t>(slot);
        slots.push_back(slot);
      }
      if (!ok) {
        for (int64_t slot : slots) release_locked(request.row, slot);
        slots.clear();
      }
    }
    if (ok && !missing.empty()) {
      const int64_t delay = delay_ns_.load();
      if (delay > 0 && (advisory || demands_read_ >= delay_after_.load())) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
      }
      if (fail_reads_.load()) {
        counters_[kReadErrors].fetch_add(1);
        ok = false;
      } else {
        const int result = reader_.read(
            request.row, missing, slots, advisory ? 1 : kBounceRows,
            [&] { return advisory && (demand_pending() || pause_requested_.load()); });
        if (result == 0) counters_[kReadErrors].fetch_add(1);
        ok = result == 1;
      }
      if (!advisory) ++demands_read_;
      _mm_sfence();  // the split's memcpy stores land before the map publishes them (D11)
    }
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!slots.empty()) {
        Tier& tier = tiers_[request.row];
        for (size_t i = 0; i < slots.size(); ++i) {
          if (ok) {
            tier.state[slots[i]] = kReady;
            tier.stamp[slots[i]] = ++tick_;
            publish_map(request.row, missing[i], static_cast<int32_t>(slots[i]));
          } else {
            release_locked(request.row, slots[i]);
          }
        }
        if (ok) (advisory ? tier.rows_advisory : tier.rows_demand) += static_cast<int64_t>(slots.size());
        counters_[kVersion].fetch_add(1);
      }
    }
    if (ok) {
      counters_[kRowsRead].fetch_add(static_cast<int64_t>(slots.size()));
      if (advisory) counters_[kAdvisoryRows].fetch_add(static_cast<int64_t>(slots.size()));
    }
    return ok;
  }

  void handle_demand(const Request& request, uint8_t* record) {
    busy_since_.store(now_ns());
    store_release(page_ + kBusySeq, request.seq);
    if (load_acquire(page_ + kFatal) != 0) counters_[kLateAfterFatal].fetch_add(1);
    const bool ok = serve(request, false);
    if (ok) counters_[request.need.empty() ? kTouchOnly : kServedRequests].fetch_add(1);
    _mm_sfence();
    set_status(record, ok ? kServed : kFailed);
    store_release(page_ + kBusySeq, 0);
    busy_since_.store(0);
  }

  uint8_t* page_;
  int32_t* map_;
  int64_t layers_;
  int64_t experts_;
  RowReader reader_;
  std::vector<Tier> tiers_;
  std::mutex mutex_;
  uint64_t tick_ = 0;
  uint32_t next_demand_ = 1;
  uint32_t next_advice_ = 1;
  int64_t demands_read_ = 0;
  std::atomic<bool> in_advice_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> threaded_{false};
  std::atomic<uint32_t> skip_advice_upto_{0};
  std::atomic<int64_t> busy_since_{0};
  std::atomic<int64_t> delay_ns_{0};
  std::atomic<int64_t> delay_after_{0};
  std::atomic<bool> fail_reads_{false};
  std::atomic<int64_t> counters_[kCounterCount];
};

inline std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_map<int64_t, std::unique_ptr<RamTier>>& registry() {
  static std::unordered_map<int64_t, std::unique_ptr<RamTier>> tiers;
  return tiers;
}

inline RamTier* find(int64_t handle) {
  std::lock_guard<std::mutex> guard(registry_mutex());
  const auto found = registry().find(handle);
  if (found == registry().end()) throw std::runtime_error("exl3 RAM miss: unknown handle");
  return found->second.get();
}

}  // namespace exl3_ram_miss

int64_t exl3_ram_miss_open(
    TensorView page,
    TensorView slot_map,
    TensorView reads,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    TensorView capacity,
    std::string paths,
    int64_t slot_bytes,
    int64_t direct) {
  using namespace exl3_ram_miss;
  const auto* capacity_data = static_cast<const int64_t*>(capacity.data_ptr());
  auto tier = std::make_unique<RamTier>(
      static_cast<uint8_t*>(page.data_ptr()), static_cast<int32_t*>(slot_map.data_ptr()),
      tables_from(reads, file_sizes, segments, slabs, row_bytes, paths, slot_bytes),
      std::vector<int64_t>(capacity_data, capacity_data + capacity.size(0)), direct != 0);
  if (!tier->open()) return -1;
  std::lock_guard<std::mutex> guard(registry_mutex());
  static int64_t next_handle = 1;
  const int64_t handle = next_handle++;
  registry().emplace(handle, std::move(tier));
  return handle;
}

void exl3_ram_miss_close(int64_t handle) {
  using namespace exl3_ram_miss;
  std::unique_ptr<RamTier> tier;
  {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = registry().find(handle);
    if (found == registry().end()) return;
    tier = std::move(found->second);
    registry().erase(found);
  }
}

// 1 served a demand record, 2 an advisory record, 0 nothing posted. Refused while a thread pumps.
int64_t exl3_ram_miss_pump(int64_t handle) {
  auto* tier = exl3_ram_miss::find(handle);
  if (tier->threaded()) throw std::runtime_error("exl3 RAM miss: pump() while the service thread runs");
  if (tier->pump_demand()) return 1;
  return tier->pump_advice() ? 2 : 0;
}

int64_t exl3_ram_miss_contains(int64_t handle, int64_t row, int64_t expert) {
  return exl3_ram_miss::find(handle)->has(row, expert) ? 1 : 0;
}

void exl3_ram_miss_touch(int64_t handle, int64_t row, int64_t expert) {
  exl3_ram_miss::find(handle)->touch(row, expert);
}

void exl3_ram_miss_assign(
    int64_t handle, int64_t row, int64_t expert, TensorView protect, int64_t fallback, TensorView out) {
  auto* result = static_cast<int64_t*>(out.data_ptr());
  int64_t evicted = -1;
  result[0] = exl3_ram_miss::find(handle)->assign(row, expert, exl3_ram_miss::ids_of(protect), fallback != 0, &evicted);
  result[1] = evicted;
}

void exl3_ram_miss_release(int64_t handle, int64_t row, int64_t slot) {
  exl3_ram_miss::find(handle)->release(row, slot);
}

void exl3_ram_miss_mapping(int64_t handle, int64_t row, TensorView out) {
  exl3_ram_miss::find(handle)->mapping(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_slot_to_expert(int64_t handle, int64_t row, TensorView out) {
  exl3_ram_miss::find(handle)->slot_to_expert(row, static_cast<int64_t*>(out.data_ptr()));
}

int64_t exl3_ram_miss_lru_order(int64_t handle, int64_t row, TensorView out) {
  return exl3_ram_miss::find(handle)->lru_order(row, static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_set_hot(int64_t handle, int64_t row, TensorView experts) {
  exl3_ram_miss::find(handle)->set_hot(row, static_cast<const int64_t*>(experts.data_ptr()), experts.size(0));
}

void exl3_ram_miss_inject(int64_t handle, int64_t delay_ns, int64_t fail_reads, int64_t after_demands) {
  exl3_ram_miss::find(handle)->inject(delay_ns, fail_reads != 0, after_demands);
}

void exl3_ram_miss_counters(int64_t handle, TensorView out) {
  exl3_ram_miss::find(handle)->counters(static_cast<int64_t*>(out.data_ptr()));
}

void exl3_ram_miss_layer_rows(int64_t handle, int64_t advisory, TensorView out) {
  exl3_ram_miss::find(handle)->layer_rows(static_cast<int64_t*>(out.data_ptr()), advisory != 0);
}

// ---- Host-side simulated device: the post and wait kernels' protocol, for CPU tests ----

int64_t exl3_ram_miss_sim_post(
    TensorView page, int64_t row, TensorView need, TensorView protect, int64_t advisory, int64_t after) {
  using namespace exl3_ram_miss;
  auto* base = static_cast<uint8_t*>(page.data_ptr());
  const int64_t head_word = advisory ? kAdviseHead : kDemandHead;
  uint32_t seq = load_acquire(base + head_word) + 1u;
  if (seq == 0) seq = 1;
  uint8_t* record =
      base + record_offset(advisory ? kAdviseRing : kDemandRing, advisory ? kAdviseRecords : kDemandRecords, seq);
  const auto need_ids = ids_of(need);
  const auto protect_ids = ids_of(protect);
  const uint16_t row16 = static_cast<uint16_t>(row);
  const uint16_t need_count = static_cast<uint16_t>(std::min<size_t>(need_ids.size(), kMaxIds));
  const uint16_t protect_count = static_cast<uint16_t>(std::min<size_t>(protect_ids.size(), kMaxIds));
  const uint16_t pending = 0;
  const uint32_t after32 = static_cast<uint32_t>(after);
  // Seqlock writer: invalidate seq, fence, payload, fence, seq last (a lapped record
  // still being rewritten can never carry a valid seq).
  store_release(record + kRecSeq, 0u);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  std::memset(record + 4, 0, kRecordBytes - 4);
  std::memcpy(record + kRecRow, &row16, 2);
  std::memcpy(record + kRecNeedCount, &need_count, 2);
  std::memcpy(record + kRecProtectCount, &protect_count, 2);
  std::memcpy(record + kRecStatus, &pending, 2);
  std::memcpy(record + kRecAfter, &after32, 4);
  std::memcpy(record + kRecNeed, need_ids.data(), 4 * need_count);
  std::memcpy(record + kRecProtect, protect_ids.data(), 4 * protect_count);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  store_release(record + kRecSeq, seq);  // payload first, seq last (the seqlock order)
  store_release(base + head_word, seq);
  return seq;
}

// The wait kernel's decision rule: 1 served, 2 failed, 0 timed out (both raise fatal),
// 3 fatal already raised (the sticky fast path).
int64_t exl3_ram_miss_sim_wait(TensorView page, int64_t seq, int64_t timeout_ns) {
  using namespace exl3_ram_miss;
  auto* base = static_cast<uint8_t*>(page.data_ptr());
  const uint32_t want = static_cast<uint32_t>(seq);
  if (load_acquire(base + kFatal) != 0) return 3;
  const int64_t deadline = now_ns() + timeout_ns;
  auto raise_fatal = [&] {
    uint32_t zero = 0;
    __atomic_compare_exchange_n(
        reinterpret_cast<uint32_t*>(base + kFatal), &zero, want, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  };
  while (!reached(load_acquire(base + kDemandDone), want)) {
    if (now_ns() > deadline) {
      raise_fatal();
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
  const uint8_t* record = base + record_offset(kDemandRing, kDemandRecords, want);
  const uint16_t status = __atomic_load_n(reinterpret_cast<const uint16_t*>(record + kRecStatus), __ATOMIC_ACQUIRE);
  if (status == kServed) return 1;
  raise_fatal();
  return 2;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_open, exl3_ram_miss_open);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_close, exl3_ram_miss_close);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pump, exl3_ram_miss_pump);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_contains, exl3_ram_miss_contains);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_touch, exl3_ram_miss_touch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_assign, exl3_ram_miss_assign);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_release, exl3_ram_miss_release);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_mapping, exl3_ram_miss_mapping);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_slot_to_expert, exl3_ram_miss_slot_to_expert);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_lru_order, exl3_ram_miss_lru_order);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_set_hot, exl3_ram_miss_set_hot);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_inject, exl3_ram_miss_inject);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_counters, exl3_ram_miss_counters);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_layer_rows, exl3_ram_miss_layer_rows);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_sim_post, exl3_ram_miss_sim_post);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_sim_wait, exl3_ram_miss_sim_wait);
```
(`exl3_ram_miss::TensorView` is already brought into `namespace sglang` by Task 10's `using`.)

- [ ] **Step 4: Implement the Python wrapper**

Append to `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (add `import atexit, json, sys, weakref` and `from typing import Optional` to its imports):
```python
PAGE_BYTES = 10304
RECORD_BYTES = 128
DEMAND_RING = 64
DEMAND_RECORDS = 16
ADVISE_RING = DEMAND_RING + DEMAND_RECORDS * RECORD_BYTES
ADVISE_RECORDS = 64
MAX_IDS = 8
WORDS = {
    "demand_head": 0,
    "demand_done": 4,
    "fatal": 8,
    "stop": 12,
    "advise_head": 16,
    "advise_done": 20,
    "busy_seq": 24,
    "heartbeat": 28,
}
STATUS = {"pending": 0, "served": 1, "failed": 2}
COUNTERS = (
    "served",
    "touch_only",
    "rows_read",
    "read_errors",
    "evictions",
    "overruns",
    "advisories",
    "advisories_skipped",
    "advisory_rows",
    "late_after_fatal",
    "no_victim",
    "version",
    "running",
    "spin_cpu",
)


def new_page(pin: bool) -> torch.Tensor:
    """A zeroed request page; pinned (device-readable through UVA) for a real device."""
    return torch.zeros(PAGE_BYTES, dtype=torch.uint8, pin_memory=pin)


def page_word(page: torch.Tensor, name: str) -> int:
    offset = WORDS[name]
    return int(page[offset : offset + 4].view(torch.int32)[0]) & 0xFFFFFFFF


def sim_post(page, row: int, need, protect, *, advisory: bool = False, after: int = 0) -> int:
    """Post a record as the device post kernel does; returns its sequence."""
    return int(_host_module().exl3_ram_miss_sim_post(page, row, _ids(need), _ids(protect), int(advisory), after))


def sim_wait(page, seq: int, timeout_s: float) -> int:
    """Wait as the device wait kernel does: 1 served, 2 failed, 0 timed out, 3 fatal already raised."""
    return int(_host_module().exl3_ram_miss_sim_wait(page, seq, int(timeout_s * 1e9)))


_LIVE: "weakref.WeakSet[Exl3RamMissHost]" = weakref.WeakSet()


@atexit.register
def _stop_live() -> None:
    for host in list(_LIVE):
        host.stop()


class Exl3RamMissHost:
    """The C++-owned pinned-slot bookkeeping of every streamed layer and its request service.

    ``tables``: ``Exl3RamMissTables``; ``page``: a ``new_page`` tensor; ``slot_map``:
    int32 ``[layers, experts]`` filled with -1 (pinned for a real device). Row ``r`` is
    streamed layer ``tables.layer_ids[r]``. Without a thread (Task 12) requests are
    served only by ``pump()``.
    """

    def __init__(self, tables, *, page: torch.Tensor, slot_map: torch.Tensor, direct: bool) -> None:
        if page.numel() != PAGE_BYTES or page.dtype != torch.uint8 or page.device.type != "cpu":
            raise ValueError("page must be a CPU uint8 tensor of PAGE_BYTES")
        if slot_map.dtype != torch.int32 or tuple(slot_map.shape) != tuple(tables.reads.shape[:2]):
            raise ValueError("slot_map must be int32 [layers, experts]")
        self._module = _host_module()
        self.tables = tables
        self.page = page
        self.slot_map = slot_map
        self.layers, self.experts = tables.reads.shape[:2]
        self.handle = int(
            self._module.exl3_ram_miss_open(
                page, slot_map, tables.reads, tables.file_sizes, tables.segments, tables.slabs,
                tables.row_bytes, tables.capacity, "\n".join(tables.paths), tables.slot_bytes, int(direct),
            )
        )
        if self.handle < 0:
            raise RuntimeError("exl3 RAM miss service failed to open (files, io_uring or bounce)")
        self._stopped = False
        _LIVE.add(self)

    def pump(self) -> int:
        return int(self._module.exl3_ram_miss_pump(self.handle))

    def contains(self, row: int, expert: int) -> bool:
        return bool(self._module.exl3_ram_miss_contains(self.handle, row, expert))

    def touch(self, row: int, expert: int) -> None:
        self._module.exl3_ram_miss_touch(self.handle, row, expert)

    def assign(self, row: int, expert: int, protected: Iterable[int] = (), protected_fallback: bool = True) -> tuple[int, Optional[int]]:
        out = torch.zeros(2, dtype=torch.int64)
        self._module.exl3_ram_miss_assign(self.handle, row, expert, _ids(protected), int(protected_fallback), out)
        slot, evicted = int(out[0]), int(out[1])
        if evicted == -2:
            raise ValueError(f"expert {expert} already holds a pinned slot")
        if slot < 0:
            raise RuntimeError("every pinned host slot holds a protected expert")
        return slot, (None if evicted < 0 else evicted)

    def release(self, row: int, slot: int) -> None:
        self._module.exl3_ram_miss_release(self.handle, row, slot)

    def mapping(self, row: int) -> list[int]:
        out = torch.empty(self.experts, dtype=torch.int64)
        self._module.exl3_ram_miss_mapping(self.handle, row, out)
        return out.tolist()

    def slot_to_expert(self, row: int) -> list[int]:
        out = torch.empty(int(self.tables.capacity[row]), dtype=torch.int64)
        self._module.exl3_ram_miss_slot_to_expert(self.handle, row, out)
        return out.tolist()

    def lru_order(self, row: int) -> list[int]:
        out = torch.empty(int(self.tables.capacity[row]), dtype=torch.int64)
        count = int(self._module.exl3_ram_miss_lru_order(self.handle, row, out))
        return out[:count].tolist()

    def set_hot(self, row: int, experts: Iterable[int]) -> None:
        self._module.exl3_ram_miss_set_hot(self.handle, row, _ids(e for e in experts if e >= 0))

    def version(self) -> int:
        return self.counters()["version"]

    def inject(self, delay_s: float = 0.0, fail_reads: bool = False, delay_after_demands: int = 0) -> None:
        """Test-only faults (see RamTier::inject)."""
        self._module.exl3_ram_miss_inject(self.handle, int(delay_s * 1e9), int(fail_reads), delay_after_demands)

    def counters(self) -> dict[str, int]:
        out = torch.zeros(len(COUNTERS), dtype=torch.int64)
        self._module.exl3_ram_miss_counters(self.handle, out)
        return dict(zip(COUNTERS, out.tolist()))

    def layer_rows(self) -> list[int]:
        """Rows read for demands, per streamed layer: the RAM misses behind ``f``."""
        out = torch.zeros(self.layers, dtype=torch.int64)
        self._module.exl3_ram_miss_layer_rows(self.handle, 0, out)
        return out.tolist()

    def layer_advisory_rows(self) -> list[int]:
        out = torch.zeros(self.layers, dtype=torch.int64)
        self._module.exl3_ram_miss_layer_rows(self.handle, 1, out)
        return out.tolist()

    def fatal_seq(self) -> int:
        return page_word(self.page, "fatal")

    def stop(self) -> None:
        if not getattr(self, "_stopped", True):
            # One line for the window's records (the corpus arms grep it).
            sys.stderr.write("exl3 RAM miss thread counters " + json.dumps(self.counters()) + "\n")
            self._module.exl3_ram_miss_close(self.handle)
            self._stopped = True
```

- [ ] **Step 5: Commit, push, run green**

`git add python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp python/sglang/kernels/ops/moe/exl3_ram_miss.py`; commit `feat(dsv41): C++-owned pinned-slot LRU and option C request service`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_tier.py`. Expected: `5 + 9 passed`. Append `Task 11: complete`.

---

### Task 12: The service thread — spin loop, pause/resume handshake, watchdog abort

**Blocked if Task 1 fails.**

**Branch:** `dsv41`. CPU only. Adds the thread that pumps Task 11's service (D19), the pause/resume handshake eager callers use (D12; closes the quiesce/advisory race: the thread acknowledges a pause only between requests, and advisories abandon at their next row), and the watchdog (D15).

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (insert a third `namespace exl3_ram_miss` block and exports before the final `}  // namespace sglang`)
- Modify: `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (`Exl3RamMissHost.start_thread`, `pause`, `resume`, `stop`)
- Create: `test/registered/unit/kernels/test_exl3_ram_miss_thread.py`

**Interfaces:**
- Produces: `Exl3RamMissHost.start_thread(*, cpu_core=-1, fatal_wait_s=30.0, spin_us=5000)`, `pause(timeout_s)` (raises `RuntimeError` on timeout), `resume()`, `threaded` (bool); `stop()` stops the thread before closing. C++ `RamThread`; exports `exl3_ram_miss_start_thread`, `exl3_ram_miss_stop_thread`, `exl3_ram_miss_pause`, `exl3_ram_miss_resume`.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_miss_thread.py`:
```python
"""The option C service thread, its pause handshake and its watchdog (CPU, simulated device)."""

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _host(tmp_path, capacity=3, fatal_wait_s=5.0):
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False)
    slot_map = torch.full((2, 6), -1, dtype=torch.int32)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=slot_map, direct=False)
    host.start_thread(fatal_wait_s=fatal_wait_s)
    return s, page, slot_map, host


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_the_thread_serves_demands_without_a_pump(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        assert sim_wait(page, sim_post(page, 0, need=[1, 2], protect=[1, 2]), 10) == 1
        assert host.contains(0, 1) and host.counters()["running"] == 1
        with pytest.raises(RuntimeError, match="pump"):
            host.pump()
    finally:
        host.stop()


def test_a_slow_read_times_out_the_wait_and_raises_fatal(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.inject(delay_s=1.0)
        seq = sim_post(page, 0, need=[1], protect=[1])
        started = time.perf_counter()
        assert sim_wait(page, seq, timeout_s=0.05) == 0
        assert time.perf_counter() - started < 0.5
        assert host.fatal_seq() == seq
        assert sim_wait(page, sim_post(page, 0, need=[], protect=[]), 1.0) == 3  # sticky
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_no_advisory_starts_while_paused(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.pause(timeout_s=1.0)
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=page_word(page, "demand_head") + 5)
        time.sleep(0.2)
        assert host.counters()["advisories"] == 0 and not host.contains(1, 3)
        # Python owns the slots now: an eager assignment cannot race an advisory.
        host.assign(1, 4, protected=[4])
        host.resume()
        # The advisory posted during the pause is skipped (it predates the eager use).
        assert _until(lambda: host.counters()["advisories_skipped"] == 1)
        assert not host.contains(1, 3) and host.contains(1, 4)
    finally:
        host.stop()


def test_a_pause_waits_for_an_advisory_in_flight_and_cuts_it_short(tmp_path):
    s, page, slot_map, host = _host(tmp_path)
    try:
        host.inject(delay_s=0.3)  # every advisory read sleeps first
        sim_post(page, 1, need=[1, 2, 3], protect=[1, 2, 3], advisory=True, after=page_word(page, "demand_head") + 5)
        assert _until(lambda: host.counters()["advisories"] == 1)
        started = time.perf_counter()
        host.pause(timeout_s=2.0)
        waited = time.perf_counter() - started
        assert waited < 1.0  # at most the one row in flight, not three
        assert not any(host.contains(1, e) for e in (1, 2, 3))  # the abandoned advisory released its rows
        host.resume()
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_concurrent_eager_use_and_advisories_never_share_a_slot(tmp_path):
    s, page, slot_map, host = _host(tmp_path, capacity=4)
    stop = threading.Event()
    errors = []

    def eager():
        for expert in range(200):
            try:
                host.pause(timeout_s=2.0)
                try:
                    e = expert % 6
                    if not host.contains(0, e):
                        host.assign(0, e, protected=[e])
                    mapping = host.mapping(0)
                    slots = [m for m in mapping if m >= 0]
                    if len(slots) != len(set(slots)):
                        errors.append(mapping)
                finally:
                    host.resume()
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(repr(error))
        stop.set()

    worker = threading.Thread(target=eager)
    worker.start()
    try:
        expert = 0
        while not stop.is_set():
            sim_post(page, 0, need=[expert % 6], protect=[expert % 6], advisory=True, after=page_word(page, "demand_head") + 5)
            expert += 1
            time.sleep(0.001)
        worker.join(timeout=30)
        assert not errors, errors[:3]
        host.pause(timeout_s=2.0)
        try:
            mapping = host.mapping(0)
            slots = [m for m in mapping if m >= 0]
            assert len(slots) == len(set(slots))
            assert slot_map[0].tolist() == mapping  # the device-visible map equals the READY slots
        finally:
            host.resume()
    finally:
        host.stop()


_ABORT_SCRIPT = textwrap.dedent(
    """
    import pathlib, time
    from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, sim_post, sim_wait
    from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
    import torch
    s = ram_miss_setup(pathlib.Path({tmp!r}))
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.start_thread(fatal_wait_s=0.3)
    host.inject(delay_s=30.0)
    sim_wait(page, sim_post(page, 0, need=[1], protect=[1]), timeout_s=0.05)
    time.sleep(3.0)
    print("still alive")
    """
)


def test_the_watchdog_aborts_a_process_that_does_not_stop_after_fatal(tmp_path):
    script = _ABORT_SCRIPT.format(tmp=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == -6, (result.returncode, result.stderr[-2000:])
    assert "still alive" not in result.stdout
    assert "exl3 RAM miss" in result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
```

- [ ] **Step 2: Commit red, push, run red**

`git add test/registered/unit/kernels/test_exl3_ram_miss_thread.py`; commit `test(dsv41): red tests for the option C service thread and watchdog`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_thread.py`. Expected: every test fails with `AttributeError: 'Exl3RamMissHost' object has no attribute 'start_thread'` (the abort test fails on its return code for the same reason).

- [ ] **Step 3: Implement the thread**

In `exl3_ram_miss_host.cpp`, insert before the final `}  // namespace sglang` line:
```cpp
namespace exl3_ram_miss {

// Pumps one RamTier on its own thread (plan D19): demands first, then advisories; spins
// with _mm_pause() for spin_ns after the last request, else sleeps 50 us between polls.
// pause() is a handshake: it asks every advisory in flight to give up at its next row,
// skips advisories posted so far (resume() skips those posted during the pause), and returns once the loop has acknowledged the pause
// between two requests. While paused the loop takes no request, so an eager caller
// owns the slots until resume(). The watchdog (plan D15), on its own thread so a stuck
// read cannot silence it, aborts the process when the fatal word stays raised for
// fatal_wait without stop() (the process did not fail stop), or when one demand stays in
// service for fatal_wait (a hung read).
class RamThread {
 public:
  RamThread(RamTier* tier, int cpu_core, int64_t fatal_wait_ns, int64_t spin_ns)
      : tier_(tier), page_(tier->page()), cpu_core_(cpu_core), fatal_wait_ns_(fatal_wait_ns), spin_ns_(spin_ns) {}

  ~RamThread() { stop(); }

  void start() {
    tier_->set_threaded(true);
    thread_ = std::thread([this] { run(); });
    watchdog_ = std::thread([this] { watch(); });
  }

  void stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    if (watchdog_.joinable()) watchdog_.join();
    tier_->set_threaded(false);
  }

  bool pause(int64_t timeout_ns) {
    tier_->request_pause(true);
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(true);
    const int64_t deadline = now_ns() + timeout_ns;
    while (!paused_.load()) {
      if (now_ns() > deadline) {
        resume();
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
  }

  // Advisories posted while paused predate the eager use: skip them too.
  void resume() {
    tier_->skip_advice_posted_so_far();
    pause_requested_.store(false);
    tier_->request_pause(false);
  }

 private:
  void run() {
    if (cpu_core_ >= 0) {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(cpu_core_, &cpus);
      pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    }
    tier_->set_counter(kSpinCpu, sched_getcpu());
    tier_->set_counter(kRunning, 1);
    int64_t last_active = now_ns();
    uint32_t heartbeat = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
      store_release(page_ + kHeartbeat, ++heartbeat);
      if (pause_requested_.load()) {
        paused_.store(true);
        while (pause_requested_.load() && !stop_.load()) std::this_thread::sleep_for(std::chrono::microseconds(20));
        paused_.store(false);
        continue;
      }
      if (tier_->pump_demand() || tier_->pump_advice()) {
        last_active = now_ns();
        continue;
      }
      if (now_ns() - last_active < spin_ns_) {
        _mm_pause();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    tier_->set_counter(kRunning, 0);
  }

  void watch() {
    int64_t fatal_since = 0;
    bool reported = false;
    while (!stop_.load()) {
      const uint32_t fatal = load_acquire(page_ + kFatal);
      const int64_t now = now_ns();
      if (fatal != 0) {
        if (!reported) {
          reported = true;
          std::fprintf(stderr, "ERROR exl3 RAM miss: request %u timed out or failed; the process must stop\n", fatal);
          std::fflush(stderr);
        }
        if (fatal_since == 0) fatal_since = now;
      }
      const int64_t busy_since = tier_->busy_since();
      const bool fatal_held = fatal_since != 0 && now - fatal_since > fatal_wait_ns_;
      const bool stuck = busy_since != 0 && now - busy_since > fatal_wait_ns_;
      if (fatal_held || stuck) {
        std::fprintf(
            stderr, "ERROR exl3 RAM miss: %s for %.1f s (fatal %u, busy %u); aborting instead of hanging decode\n",
            stuck ? "a request stayed in service" : "the fatal word stayed raised without the process stopping",
            static_cast<double>(fatal_wait_ns_) / 1e9, fatal, load_acquire(page_ + kBusySeq));
        std::fflush(stderr);
        prctl(PR_SET_DUMPABLE, 0);
        std::abort();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  RamTier* tier_;
  uint8_t* page_;
  int cpu_core_;
  int64_t fatal_wait_ns_;
  int64_t spin_ns_;
  std::thread thread_;
  std::thread watchdog_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> paused_{false};
};

inline std::unordered_map<int64_t, std::unique_ptr<RamThread>>& thread_registry() {
  static std::unordered_map<int64_t, std::unique_ptr<RamThread>> threads;
  return threads;
}

inline RamThread* find_thread(int64_t handle) {
  std::lock_guard<std::mutex> guard(registry_mutex());
  const auto found = thread_registry().find(handle);
  if (found == thread_registry().end()) throw std::runtime_error("exl3 RAM miss: no service thread");
  return found->second.get();
}

}  // namespace exl3_ram_miss

void exl3_ram_miss_start_thread(int64_t handle, int64_t cpu_core, int64_t fatal_wait_ns, int64_t spin_ns) {
  using namespace exl3_ram_miss;
  RamTier* tier = find(handle);
  auto thread = std::make_unique<RamThread>(tier, static_cast<int>(cpu_core), fatal_wait_ns, spin_ns);
  thread->start();
  std::lock_guard<std::mutex> guard(registry_mutex());
  thread_registry()[handle] = std::move(thread);
}

void exl3_ram_miss_stop_thread(int64_t handle) {
  using namespace exl3_ram_miss;
  std::unique_ptr<RamThread> thread;
  {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = thread_registry().find(handle);
    if (found == thread_registry().end()) return;
    thread = std::move(found->second);
    thread_registry().erase(found);
  }
  thread->stop();
}

int64_t exl3_ram_miss_pause(int64_t handle, int64_t timeout_ns) {
  return exl3_ram_miss::find_thread(handle)->pause(timeout_ns) ? 1 : 0;
}

void exl3_ram_miss_resume(int64_t handle) {
  exl3_ram_miss::find_thread(handle)->resume();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_start_thread, exl3_ram_miss_start_thread);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_stop_thread, exl3_ram_miss_stop_thread);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_pause, exl3_ram_miss_pause);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(exl3_ram_miss_resume, exl3_ram_miss_resume);
```

- [ ] **Step 4: Implement the Python side**

In `Exl3RamMissHost.__init__`, after `self._stopped = False`, add `self.threaded = False`. Add the methods:
```python
    def start_thread(self, *, cpu_core: int = -1, fatal_wait_s: float = 30.0, spin_us: int = 5000) -> None:
        """Serve requests on a C++ thread (no more ``pump()``), with the fail-stop watchdog."""
        self._module.exl3_ram_miss_start_thread(self.handle, cpu_core, int(fatal_wait_s * 1e9), int(spin_us * 1e3))
        self.threaded = True

    def pause(self, timeout_s: float) -> None:
        """Hand the slots to the caller: returns once the thread is between requests and idle."""
        if not self.threaded:
            return
        if not self._module.exl3_ram_miss_pause(self.handle, int(timeout_s * 1e9)):
            raise RuntimeError(f"exl3 RAM miss thread did not pause within {timeout_s} s")

    def resume(self) -> None:
        if self.threaded:
            self._module.exl3_ram_miss_resume(self.handle)
```
and replace `stop` with:
```python
    def stop(self) -> None:
        if not getattr(self, "_stopped", True):
            if self.threaded:
                self._module.exl3_ram_miss_stop_thread(self.handle)
                self.threaded = False
            # One line for the window's records (the corpus arms grep it).
            sys.stderr.write("exl3 RAM miss thread counters " + json.dumps(self.counters()) + "\n")
            self._module.exl3_ram_miss_close(self.handle)
            self._stopped = True
```

- [ ] **Step 5: Commit, push, run green**

`git add python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp python/sglang/kernels/ops/moe/exl3_ram_miss.py`; commit `feat(dsv41): option C service thread with pause handshake and fail-stop watchdog`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_tier.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py` five times in a row (a `for i in 1 2 3 4 5; do ...; done` inside the remote command). Expected: `5 + 9 + 6 passed` every time. Then `<DSV41-CPU> <DSV41_SUITE>` plus every new dsv41 file so far: previous counts + 20. Append `Task 12: complete`.

---

### Task 13: Device post and wait kernels (`exl3_ram_miss.cuh`) and their wrapper

**Blocked if Task 1 fails.**

**Branch:** `dsv41`. The CPU part of this task is the wrapper's argument checks; the kernels compile, and their CUDA test runs green, in Step 6 (GPU), and again in Task 16 as a regression.

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`
- Modify: `python/sglang/kernels/ops/moe/exl3_ram_miss.py` (add `STATE_WORDS`, `Exl3RamMissDevice`)
- Create: `test/manual/dsv41/test_exl3_ram_miss_cuda.py` (GPU), `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py` (CPU)

**Interfaces:**
- Consumes: the page protocol (D10–D11), `Exl3RamMissHost`, `new_page`.
- Produces: `STATE_WORDS = {"posted": 0, "pending": 1, "timeouts": 2, "failures": 3, "waits": 4, "polls": 5, "sticky": 6, "advised": 7, "unserved_misses": 8}`; `Exl3RamMissDevice(page, slot_map, *, device, layers, timeout_ms, advise)` with `post(row, planned, count, routes, next_row)`, `wait(row, planned, count, host_rows, keep, ram_miss)`, `stats() -> dict` (synchronizes), `state`, `last_routes`.

- [ ] **Step 1: Write the tests**

Create `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`:
```python
"""The device wrapper refuses pages and slot maps the kernels cannot address (CPU)."""

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import PAGE_BYTES, STATE_WORDS, Exl3RamMissDevice
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_state_words_are_distinct_and_dense():
    assert sorted(STATE_WORDS.values()) == list(range(len(STATE_WORDS)))


def test_a_page_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="page"):
        Exl3RamMissDevice(torch.zeros(10, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=10, advise=False)


def test_a_slot_map_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="slot_map"):
        Exl3RamMissDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((3, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=10, advise=False)


def test_the_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        Exl3RamMissDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=0, advise=False)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```
Create `test/manual/dsv41/test_exl3_ram_miss_cuda.py`:
```python
"""Option C on the GPU: post/wait kernels against the C++ thread (window test).

Fake finite EXL3 checkpoint unless DSV41_EXL3_DIR is set, in which case the
overhead test reads real layer-3 rows (the model's own row source) to measure
the stream-idle time of a forced NVMe miss. DSV41_RAM_MISS_OUT names a JSON
file for the measured numbers.
"""

import json
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EXPERTS = 16
TOP_K = 6


def _service(tmp_path, *, expert_dir=None, layer=0, capacity=8, timeout_ms=2000, advise=False, layers=2):
    from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissDevice, Exl3RamMissHost, new_page
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    if expert_dir is None:
        write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=EXPERTS, hidden=1024, inter=512, finite=True)
        expert_dir = str(tmp_path)
    layout = build_exl3_expert_layout(expert_dir)
    fmt = Exl3ExpertFormat(layout, layer, direct=expert_dir != str(tmp_path))
    specs = {s.name: s for s in fmt.tensor_specs(None)}
    layer_ids = [layer, layer + 1][:layers]
    slabs = {
        lid: {n: allocate_host_slab(capacity, specs[n].row_shape, specs[n].dtype, register=True) for n in EXL3_STREAMED_NAMES}
        for lid in layer_ids
    }
    tables = exl3_ram_miss_tables(layout, fmt.segment_map(), slabs)
    page = new_page(pin=True)
    slot_map = torch.full((len(layer_ids), layout.num_experts), -1, dtype=torch.int32).pin_memory()
    host = Exl3RamMissHost(tables, page=page, slot_map=slot_map, direct=fmt.direct)
    host.start_thread(fatal_wait_s=30.0)
    dev = Exl3RamMissDevice(page, slot_map, device="cuda", layers=len(layer_ids), timeout_ms=timeout_ms, advise=advise)
    return layout, fmt, specs, slabs, host, dev


def _buffers():
    return dict(
        planned=torch.zeros(TOP_K, dtype=torch.int64, device="cuda"),
        count=torch.zeros(1, dtype=torch.int32, device="cuda"),
        routes=torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda"),
        host_rows=torch.zeros(TOP_K, dtype=torch.int64, device="cuda"),
        keep=torch.ones(1, dtype=torch.float32, device="cuda"),
        ram_miss=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )


def _step(dev, b, row=0, next_row=-1):
    dev.post(row, b["planned"], b["count"], b["routes"], next_row)
    dev.wait(row, b["planned"], b["count"], b["host_rows"], b["keep"], b["ram_miss"])


def _set(b, planned, routes):
    b["planned"][: len(planned)].copy_(torch.tensor(planned))
    b["count"].fill_(len(planned))
    b["routes"].fill_(-1)
    b["routes"][: len(routes)].copy_(torch.tensor(routes))


def test_a_miss_is_served_and_translated(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path)
    try:
        b = _buffers()
        _set(b, [3, 5], [3, 5, 0, 1, 2, 4])
        _step(dev, b)
        torch.cuda.synchronize()
        mapping = host.mapping(0)
        assert b["host_rows"][:2].tolist() == [mapping[3], mapping[5]] and min(mapping[3], mapping[5]) >= 0
        assert b["keep"].item() == 1.0 and b["ram_miss"].item() == 0
        assert dev.stats()["timeouts"] == 0 and host.fatal_seq() == 0
    finally:
        host.stop()


def test_a_hung_read_times_out_and_everything_after_is_fast(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path, timeout_ms=50)
    try:
        b = _buffers()
        host.inject(delay_s=5.0)
        _set(b, [7], [7])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _step(dev, b)
        end.record()
        torch.cuda.synchronize()
        assert b["keep"].item() == 0.0 and host.fatal_seq() != 0
        assert start.elapsed_time(end) < 500.0
        start.record()
        for _ in range(40):
            _step(dev, b)
        end.record()
        torch.cuda.synchronize()
        assert start.elapsed_time(end) < 50.0  # the sticky fast path
        assert dev.stats()["sticky"] == 1 and dev.stats()["timeouts"] == 1
    finally:
        host.inject(delay_s=0.0)
        host.stop()


def test_post_and_wait_capture_and_replay(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path)
    try:
        b = _buffers()
        _set(b, [1], [1, 2])
        _step(dev, b)  # warm up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _step(dev, b)
        for planned in ([4], [6, 8], [9, 10, 11]):
            _set(b, planned, planned)
            graph.replay()
            torch.cuda.synchronize()
            mapping = host.mapping(0)
            assert b["host_rows"][: len(planned)].tolist() == [mapping[e] for e in planned]
            assert b["keep"].item() == 1.0
    finally:
        host.stop()


def test_a_rewritten_slot_is_read_fresh(tmp_path):
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu, expert_row_segments
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout, fmt, specs, slabs, host, dev = _service(tmp_path, capacity=1)
    try:
        dest = {n: torch.zeros((1,) + specs[n].row_shape, dtype=specs[n].dtype, device="cuda") for n in EXL3_STREAMED_NAMES}
        segments = expert_row_segments([(slabs[0][n], dest[n]) for n in EXL3_STREAMED_NAMES])
        slots = torch.zeros(TOP_K, dtype=torch.int32, device="cuda")
        b = _buffers()
        for expert in (2, 9, 2):  # capacity 1: each demand evicts and rewrites slot 0
            _set(b, [expert], [expert])
            _step(dev, b)
            copy_expert_row_segments_gpu(segments, b["host_rows"], slots, b["count"])
            torch.cuda.synchronize()
            want = {n: torch.empty((1,) + specs[n].row_shape, dtype=specs[n].dtype) for n in EXL3_STREAMED_NAMES}
            Exl3ShardRowSource.for_layer(layout, 0, fmt.segment_map(), direct=False).read(torch.tensor([expert]), want)
            for n in EXL3_STREAMED_NAMES:
                assert torch.equal(dest[n].cpu().view(torch.uint8), want[n].view(torch.uint8)), (expert, n)
    finally:
        host.stop()


def test_overheads(tmp_path):
    """§9.3 acceptance numbers: hit-path cost per layer, stream idle per forced miss."""
    expert_dir = os.environ.get("DSV41_EXL3_DIR")
    layer = int(os.environ.get("DSV41_PROBE_LAYER", "3")) if expert_dir else 0
    layout, fmt, specs, slabs, host, dev = _service(tmp_path, expert_dir=expert_dir, layer=layer, capacity=16)
    report = {"real_rows": bool(expert_dir)}
    try:
        b = _buffers()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        _set(b, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
        _step(dev, b)  # load six rows
        torch.cuda.synchronize()
        _set(b, [], [0, 1, 2, 3, 4, 5])  # all in RAM: a touch-only post, no wait
        start.record()
        for _ in range(400):
            _step(dev, b)
        end.record()
        torch.cuda.synchronize()
        report["hit_path_us_per_layer"] = start.elapsed_time(end) * 1000 / 400
        for misses, ids in ((1, [6]), (6, [7, 8, 9, 10, 11, 12])):
            _set(b, ids, ids)
            start.record()
            _step(dev, b)
            end.record()
            torch.cuda.synchronize()
            report[f"stream_idle_ms_{misses}_miss"] = start.elapsed_time(end)
            _set(b, [], [])
        report["thread"] = host.counters()
    finally:
        host.stop()
    out = os.environ.get("DSV41_RAM_MISS_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report))
    assert report["hit_path_us_per_layer"] < 100.0
```

- [ ] **Step 2: Commit red, push, run red**

`git add test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/manual/dsv41/test_exl3_ram_miss_cuda.py`; commit `test(dsv41): red tests for the option C device kernels`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/manual/dsv41/test_exl3_ram_miss_cuda.py`. Expected: the CPU file errors (`cannot import name 'STATE_WORDS'`); the GPU file skips.

- [ ] **Step 3: Implement the kernels**

Create `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh`:
```cuda
// Device side of the option C RAM-miss service (DSV41 Phase 3b plan, D10-D15, D22).
//
// post: one block; thread 0 builds the layer's request (need = planned VRAM misses
// whose host-mapped slot map entry is -1; protect = every routed expert), writes it
// into the page's demand ring with volatile stores, fences system-wide and
// release-stores demand_head. The record is posted for every MoE layer (the thread
// uses touch-only records for LRU recency); the wait is armed only when something is
// needed or advisories are on. With `advise`, it also remembers this token's routes
// for `row` and posts the previous token's routes of `next_row` that are not in RAM
// as an advisory record.
// wait: one block; thread 0 polls demand_done with ld.acquire.sys and __nanosleep
// until it reaches the armed sequence or `timeout_ns` of %globaltimer passes, then
// translates the planned experts to pinned slots from the slot map. A timeout, a
// failed request, or a planned row still not in RAM raises the page's fatal word
// (sticky: later posts post nothing and later waits return at once) and sets keep
// to 0, which drops the layer's routed output for this forward.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace exl3_ram_miss_device {

constexpr int kBlock = 32;
constexpr int64_t kDemandHead = 0;
constexpr int64_t kDemandDone = 4;
constexpr int64_t kFatal = 8;
constexpr int64_t kAdviseHead = 16;
constexpr int64_t kRecordBytes = 128;
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kAdviseRing = kDemandRing + 16 * kRecordBytes;
constexpr uint32_t kAdviseRecords = 64;
constexpr int kMaxIds = 8;
constexpr int64_t kRecRow = 4;
constexpr int64_t kRecNeedCount = 6;
constexpr int64_t kRecProtectCount = 8;
constexpr int64_t kRecStatus = 10;
constexpr int64_t kRecAfter = 12;
constexpr int64_t kRecNeed = 16;
constexpr int64_t kRecProtect = 48;
constexpr uint16_t kServed = 1;

constexpr int kPosted = 0;
constexpr int kPending = 1;
constexpr int kTimeouts = 2;
constexpr int kFailures = 3;
constexpr int kWaits = 4;
constexpr int kPolls = 5;
constexpr int kSticky = 6;
constexpr int kAdvised = 7;
constexpr int kUnservedMisses = 8;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint8_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys(uint8_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ int32_t ld_volatile(const int32_t* address) {
  return *reinterpret_cast<const volatile int32_t*>(address);
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

__device__ __forceinline__ bool listed(const int32_t* ids, int count, int32_t id) {
  for (int i = 0; i < count; ++i) {
    if (ids[i] == id) return true;
  }
  return false;
}

__device__ __forceinline__ void write_record(
    uint8_t* record, uint32_t seq, int64_t row, const int32_t* need, int need_count, const int32_t* protect,
    int protect_count, uint32_t after) {
  volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(record);
  volatile uint16_t* halves = reinterpret_cast<volatile uint16_t*>(record);
  // Seqlock writer: invalidate seq before touching the payload, so a lapped record that
  // is half rewritten never passes the thread's read_record seq re-check.
  words[0] = 0u;
  __threadfence_system();
  halves[kRecRow / 2] = static_cast<uint16_t>(row);
  halves[kRecNeedCount / 2] = static_cast<uint16_t>(need_count);
  halves[kRecProtectCount / 2] = static_cast<uint16_t>(protect_count);
  halves[kRecStatus / 2] = 0;
  words[kRecAfter / 4] = after;
  volatile int32_t* need_out = reinterpret_cast<volatile int32_t*>(record + kRecNeed);
  volatile int32_t* protect_out = reinterpret_cast<volatile int32_t*>(record + kRecProtect);
  for (int i = 0; i < kMaxIds; ++i) {
    need_out[i] = i < need_count ? need[i] : -1;
    protect_out[i] = i < protect_count ? protect[i] : -1;
  }
  // The seqlock order the thread's read_record relies on: seq=0, fence, payload, fence, seq last.
  __threadfence_system();
  words[0] = seq;
}

__device__ __forceinline__ void raise_fatal(uint8_t* page, uint32_t seq) {
  if (ld_acquire_sys(page + kFatal) == 0) st_release_sys(page + kFatal, seq);
}

}  // namespace exl3_ram_miss_device

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_post_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ slot_map,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int64_t* __restrict__ routes,
    int64_t route_count,
    int64_t row,
    int64_t experts,
    int64_t advise,
    int32_t* __restrict__ last_routes,
    int64_t next_row) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  if (state[kSticky] != 0 || ld_acquire_sys(page + kFatal) != 0) {
    state[kSticky] = 1;
    state[kPending] = 0;
    return;
  }
  const int32_t* map_row = slot_map + row * experts;
  int32_t need[kMaxIds];
  int32_t protect[kMaxIds];
  int need_count = 0;
  int protect_count = 0;
  const int64_t planned_count = min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), static_cast<int64_t>(kMaxIds));
  for (int64_t i = 0; i < planned_count; ++i) {
    const int32_t expert = static_cast<int32_t>(planned[i]);
    if (expert >= 0 && expert < experts && ld_volatile(map_row + expert) < 0 && !listed(need, need_count, expert)) {
      need[need_count++] = expert;
    }
  }
  for (int64_t i = 0; i < route_count && protect_count < kMaxIds; ++i) {
    const int32_t expert = static_cast<int32_t>(routes[i]);
    if (expert >= 0 && expert < experts && !listed(protect, protect_count, expert)) protect[protect_count++] = expert;
  }
  for (int i = 0; i < need_count && protect_count < kMaxIds; ++i) {
    if (!listed(protect, protect_count, need[i])) protect[protect_count++] = need[i];
  }
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) seq = 1;
  state[kPosted] = static_cast<int32_t>(seq);
  uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
  write_record(record, seq, row, need, need_count, protect, protect_count, 0);
  __threadfence_system();
  st_release_sys(page + kDemandHead, seq);
  state[kPending] = (need_count > 0 || advise != 0) ? static_cast<int32_t>(seq) : 0;
  if (advise == 0) return;
  for (int i = 0; i < kMaxIds; ++i) last_routes[row * kMaxIds + i] = i < protect_count ? protect[i] : -1;
  if (next_row < 0) return;
  const int32_t* next_map = slot_map + next_row * experts;
  int32_t ahead[kMaxIds];
  int ahead_count = 0;
  for (int i = 0; i < kMaxIds; ++i) {
    const int32_t expert = last_routes[next_row * kMaxIds + i];
    if (expert >= 0 && expert < experts && ld_volatile(next_map + expert) < 0) ahead[ahead_count++] = expert;
  }
  if (ahead_count == 0) return;
  uint32_t advice = static_cast<uint32_t>(state[kAdvised]) + 1u;
  if (advice == 0) advice = 1;
  state[kAdvised] = static_cast<int32_t>(advice);
  uint8_t* advice_record = page + kAdviseRing + static_cast<int64_t>((advice - 1u) % kAdviseRecords) * kRecordBytes;
  write_record(advice_record, advice, next_row, ahead, ahead_count, ahead, ahead_count, seq);
  __threadfence_system();
  st_release_sys(page + kAdviseHead, advice);
}

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ slot_map,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    int64_t row,
    int64_t experts,
    int64_t lanes,
    int64_t* __restrict__ host_rows,
    float* __restrict__ keep,
    int64_t* __restrict__ ram_miss,
    int64_t timeout_ns) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  bool ok = state[kSticky] == 0;
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  if (ok && seq != 0) {
    state[kWaits] += 1;
    const uint64_t start = global_ns();
    int64_t polls = 0;
    uint32_t done = ld_acquire_sys(page + kDemandDone);
    while (!reached(done, seq) && static_cast<int64_t>(global_ns() - start) < timeout_ns) {
      __nanosleep(256);
      ++polls;
      done = ld_acquire_sys(page + kDemandDone);
    }
    const int64_t total = static_cast<int64_t>(state[kPolls]) + polls;
    state[kPolls] = static_cast<int32_t>(total < 0x7fffffffLL ? total : 0x7fffffffLL);
    if (!reached(done, seq)) {
      state[kTimeouts] += 1;
      raise_fatal(page, seq);
      ok = false;
    } else {
      __threadfence_system();
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
      if (status != kServed) {
        state[kFailures] += 1;
        raise_fatal(page, seq);
        ok = false;
      }
    }
  }
  state[kPending] = 0;
  const int32_t* map_row = slot_map + row * experts;
  const int64_t planned_count = min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), lanes);
  int64_t misses = 0;
  for (int64_t i = 0; i < lanes; ++i) {
    int64_t slot = 0;
    if (i < planned_count) {
      const int64_t expert = planned[i];
      slot = expert >= 0 && expert < experts ? ld_volatile(map_row + expert) : -1;
      if (slot < 0) {
        ++misses;
        slot = 0;
      }
    }
    host_rows[i] = slot;
  }
  if (misses > 0 && ok) {
    // A served (or unarmed) request left a planned row out of RAM: never expected.
    state[kUnservedMisses] += static_cast<int32_t>(misses);
    raise_fatal(page, 0xFFFFFFFFu);
    ok = false;
  }
  if (!ok) state[kSticky] = 1;
  ram_miss[0] += misses;
  keep[0] = ok ? 1.0f : 0.0f;
}

void exl3_ram_miss_post(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView routes,
    int64_t row,
    int64_t advise,
    tvm::ffi::TensorView last_routes,
    int64_t next_row) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_post_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(slot_map.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int64_t*>(routes.data_ptr()),
      routes.size(0),
      row,
      slot_map.size(1),
      advise,
      static_cast<int32_t*>(last_routes.data_ptr()),
      next_row);
}

void exl3_ram_miss_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    int64_t row,
    tvm::ffi::TensorView host_rows,
    tvm::ffi::TensorView keep,
    tvm::ffi::TensorView ram_miss,
    int64_t timeout_ns) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(slot_map.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      row,
      slot_map.size(1),
      host_rows.size(0),
      static_cast<int64_t*>(host_rows.data_ptr()),
      static_cast<float*>(keep.data_ptr()),
      static_cast<int64_t*>(ram_miss.data_ptr()),
      timeout_ns);
}

}  // namespace sglang
```
Before writing, read how `expert_doorbell.cuh` launches (`host::LaunchKernel::resolve_device`, `host::LaunchKernel(grid, block, stream)(kernel, args...)`, `:430-460`) and copy any include or namespace qualification this file needs that differs from the above.

- [ ] **Step 4: Implement the wrapper**

Append to `python/sglang/kernels/ops/moe/exl3_ram_miss.py`:
```python
STATE_WORDS = {
    "posted": 0,
    "pending": 1,
    "timeouts": 2,
    "failures": 3,
    "waits": 4,
    "polls": 5,
    "sticky": 6,
    "advised": 7,
    "unserved_misses": 8,
}


@cache_once
def _device_module() -> Module:
    names = ("exl3_ram_miss_post", "exl3_ram_miss_wait")
    return load_jit(
        "exl3_ram_miss",
        cuda_files=["moe/exl3_ram_miss.cuh"],
        cuda_wrappers=[(name, name) for name in names],
    )


class Exl3RamMissDevice:
    """The post and wait kernels of option C, capturable in a CUDA graph.

    ``page`` and ``slot_map`` are the host's pinned tensors (device-readable
    through UVA). ``state`` holds the device words ``STATE_WORDS``;
    ``last_routes`` int32 ``[layers, MAX_IDS]`` the previous token's routes per
    layer for advisories (``advise``). ``timeout_ms`` bounds each wait.
    """

    def __init__(self, page, slot_map, *, device, layers: int, timeout_ms: int, advise: bool) -> None:
        if page.numel() != PAGE_BYTES or page.dtype != torch.uint8:
            raise ValueError("page must be a uint8 tensor of PAGE_BYTES")
        if slot_map.dtype != torch.int32 or slot_map.dim() != 2 or slot_map.shape[0] != layers:
            raise ValueError("slot_map must be int32 [layers, experts]")
        if timeout_ms <= 0:
            raise ValueError("the RAM-miss wait timeout must be positive")
        self.page = page
        self.slot_map = slot_map
        self.layers = layers
        self.timeout_ns = int(timeout_ms * 1_000_000)
        self.advise = int(bool(advise))
        self.state = torch.zeros(len(STATE_WORDS), dtype=torch.int32, device=device)
        self.last_routes = torch.full((layers, MAX_IDS), -1, dtype=torch.int32, device=device)
        self._module = None

    def _kernels(self):
        if self._module is None:
            self._module = _device_module()
        return self._module

    def post(self, row: int, planned, count, routes, next_row: int) -> None:
        self._kernels().exl3_ram_miss_post(
            self.page, self.state, self.slot_map, planned, count, routes, row, self.advise, self.last_routes, next_row
        )

    def wait(self, row: int, planned, count, host_rows, keep, ram_miss) -> None:
        self._kernels().exl3_ram_miss_wait(
            self.page, self.state, self.slot_map, planned, count, row, host_rows, keep, ram_miss, self.timeout_ns
        )

    def stats(self) -> dict[str, int]:
        values = self.state.cpu().tolist()
        return {name: values[index] for name, index in STATE_WORDS.items()}
```

- [ ] **Step 5: Commit, push, run green (CPU part)**

`git add python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh python/sglang/kernels/ops/moe/exl3_ram_miss.py`; commit `feat(dsv41): option C post and wait kernels`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/registered/unit/kernels/test_exl3_ram_miss_split.py test/registered/unit/kernels/test_exl3_ram_miss_tier.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py test/manual/dsv41/test_exl3_ram_miss_cuda.py`. Expected: `4 + 5 + 9 + 6 passed`, `5 skipped`.

- [ ] **Step 6: Compile and run the kernels on the GPU**

`<GPU> /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/manual/dsv41/test_exl3_ram_miss_cuda.py -k "not overheads"` (the first run compiles `exl3_ram_miss.cuh`, ~1 min). Expected: `4 passed, 1 deselected`: a miss served and translated, the hung-read timeout plus the sticky fast path, capture and replay of post/wait, and `test_a_rewritten_slot_is_read_fresh` (the host-mapped ordering check of D11 on sm_120). A compile error or a failure is fixed by a red/green pair in this task's files and this step rerun; exit code 75 is the GPU lock rule. Append `Task 13: complete`.

---

### Task 14: Wire option C into EXL3 graph decode — native slot tables, the row backend, the service, fail-stop, the graph trace

**Blocked if Task 1 fails.**

**Branch:** `dsv41`. Read `.claude/skills/env-var-conventions/SKILL.md` before Step 3.

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (add `NativePinnedSlotTable`, `Exl3RamMissRowBackend`, `Exl3RamMissService`)
- Modify: `python/sglang/srt/layers/moe/exl3_expert_format.py` (`pinned_tier_options`, new `attach_hot_cache_manager`)
- Modify: `python/sglang/srt/layers/quantization/exl3.py` (`_apply_graph`: routes into the backend; refusal without option C)
- Modify: `python/sglang/srt/layers/moe/exl3_stream_trace.py` (`enabled`, `record_graph_step`)
- Modify: `scripts/dsv41/tier_sim.py` (`simulate` skips `graph_step` lines), `test/manual/dsv41/test_tier_sim.py`
- Modify: `python/sglang/srt/environ.py` (`SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`, `SGLANG_TEST_DSV41_RAM_MISS_FAULT`)
- Create: `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py` (CPU), `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py` (GPU; Step 5, rerun in Task 16)

**Interfaces:**
- Consumes: `PinnedSlotTable`, `PinnedTierRowBackend`, `register_fail_stop_check`, `add_residency_listener`, `attach_hot_cache_manager` hook (Task 7), `Exl3RamMissHost`, `Exl3RamMissDevice`, `exl3_ram_miss_tables`.
- Produces:
  - `NativePinnedSlotTable(service, layer_id, streamer_of)` implementing `PinnedSlotTable`, capacity bound through `bind_capacity` (Task 7);
  - `Exl3RamMissRowBackend(segments, device_side, row, next_row, capacity, device)` (a `PinnedTierRowBackend` with `routes` int64 `[capacity]`, `translate` = post + wait kernels);
  - `Exl3RamMissService.get()` (process singleton) with `register(layer_id, table)`, `ensure_started()` (starts the thread; applies the test fault), `before_host_use()` / `after_host_use()` (stream sync + pause, resume; nesting counted), `attach(manager, streamer)` (keeps the manager handle), `on_residency(layer_id, slot_to_expert)`, `fail_stop_check()`, `_trace_step()`, `row_of(layer_id)`, `host`, `device_side`, `shutdown()`; `parse_fault(spec) -> (demands, seconds) | None`;
  - `Exl3ExpertFormat.pinned_tier_options(layer)` adds `"slot_table"` when `SGLANG_MOE_EXPERT_GRAPH_GATHER` is on;
  - `Exl3StreamTrace.enabled` (a trace file is open) and `record_graph_step(layer_rows_delta, routed_rows, routed_misses)`, which writes a line `{"forward", "layer": -1, "tokens": 1, "kind": "graph_step", "vram_miss", "ram_miss", "routed_rows", "layer_ram_rows", "experts": [], "counts": [], "t"}` that `tier_sim.load_trace`/`live_summary` read as one decode forward;
  - env `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS = EnvInt(2000)` and the test-only `SGLANG_TEST_DSV41_RAM_MISS_FAULT = EnvStr("")`.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`:
```python
"""Option C wiring on the host: the native slot table under the pinned tier, the fail-stop
check, residency pushes and the per-step graph trace (CPU; the thread runs, no device)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import sim_post, sim_wait
from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 3


@pytest.fixture
def tiers(tmp_path, monkeypatch):
    write_fake_exl3(str(tmp_path), num_layers=LAYERS, num_experts=EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    module.Exl3RamMissService._instance = None
    streamers, caches = {}, {}
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
        for layer_id in range(LAYERS):
            layer = torch.nn.Module()
            layer.layer_id = layer_id
            fmt = Exl3ExpertFormat(layout, layer_id, direct=False)
            streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            options = fmt.pinned_tier_options(layer)
            caches[layer_id] = ExpertPinnedHostCache(streamer, CAPACITY, device="cpu", **options)
            streamers[layer_id] = streamer
    service = module.Exl3RamMissService.get()
    yield service, streamers, caches
    service.shutdown()
    module.Exl3RamMissService._instance = None


def test_the_pinned_tier_runs_on_the_native_slot_table(tiers):
    service, streamers, caches = tiers
    cache = caches[1]
    assert isinstance(cache._lru, module.NativePinnedSlotTable)
    cache.ensure_rows(torch.tensor([4, 2]))
    row = service.row_of(1)
    assert service.host.contains(row, 4) and service.host.contains(row, 2)
    assert cache.expert_to_slot[4].item() == service.host.mapping(row)[4]
    assert service.slot_map[row, 4].item() == cache.expert_to_slot[4].item()


def test_rows_the_thread_loads_reach_the_eager_map_on_next_host_use(tiers):
    service, streamers, caches = tiers
    service.ensure_started()
    row = service.row_of(0)
    assert sim_wait(service.page, sim_post(service.page, row, need=[5], protect=[5]), 10) == 1
    caches[0].lookup(torch.tensor([5]))  # before_host_use refreshes the device-side copy
    assert caches[0].expert_to_slot[5].item() == service.host.mapping(row)[5] >= 0


def test_the_fail_stop_check_raises_once_fatal_is_set(tiers):
    service, streamers, caches = tiers
    service.ensure_started()
    service.fail_stop_check()  # nothing raised yet
    service.host.inject(fail_reads=True)
    row = service.row_of(0)
    assert sim_wait(service.page, sim_post(service.page, row, need=[1], protect=[1]), 10) == 2
    with pytest.raises(RuntimeError, match="exl3 RAM miss"):
        service.fail_stop_check()


def test_attach_registers_once_and_pushes_residency(tiers):
    service, streamers, caches = tiers
    checks, listeners = [], []
    manager = SimpleNamespace(
        register_fail_stop_check=checks.append,
        add_residency_listener=listeners.append,
    )
    for streamer in streamers.values():
        streamer.format.attach_hot_cache_manager(manager, streamer)
    assert len(checks) == 1 and len(listeners) == 1
    listeners[0](1, [3, -1])  # expert 3 is hot in layer 1
    row = service.row_of(1)
    caches[1].ensure_rows(torch.tensor([0, 1, 2]))
    caches[1].ensure_rows(torch.tensor([3]))  # capacity 3: evicts 0, loads 3
    caches[1].ensure_rows(torch.tensor([4]))  # evicts 1, never the hot 3
    assert service.host.contains(row, 3) and not service.host.contains(row, 1)


def test_graph_steps_are_traced_and_read_back_by_tier_sim(tmp_path):
    import os
    import sys

    from sglang.srt.layers.moe.exl3_stream_trace import Exl3StreamTrace

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "dsv41"))
    import tier_sim

    path = tmp_path / "trace.jsonl"
    trace = Exl3StreamTrace(str(path))
    assert trace.enabled and not Exl3StreamTrace().enabled
    trace.record_graph_step(layer_rows_delta=[1, 0, 2], routed_rows=18, routed_misses=5)
    trace.record_graph_step(layer_rows_delta=[0, 1, 0], routed_rows=18, routed_misses=3)
    trace.close()
    assert trace.decode_tokens == 2 and trace.decode_vram_misses == 8 and trace.decode_ram_misses == 4
    live = tier_sim.live_summary(tier_sim.load_trace(str(path)), warmup=0)
    assert live["decode_tokens"] == 2 and live["G"] == 4.0 and live["f"] == 0.5


def test_the_trace_step_reads_the_manager_registers_before_they_are_lost(monkeypatch):
    # The forward observer (_accumulate_registers) zeroes the streamers' graph counters
    # before the scheduler's per-batch check runs; the step must still be recorded.
    from sglang.srt.layers.moe import exl3_stream_trace
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

    manager = ExpertHotCacheManager.__new__(ExpertHotCacheManager)
    manager._layer_ids = [0, 1]
    manager._graph_counters = torch.zeros((2, 2), dtype=torch.int64)
    manager._graph_unique_counters = torch.zeros((2, 2), dtype=torch.int64)
    manager._registers = {}
    manager._layer_index = {}
    manager._gathered_zero_masks = {}
    manager._collect_route_history = False
    manager._collect_affinity = False
    lines = []
    trace = SimpleNamespace(enabled=True, record_graph_step=lambda **kw: lines.append(kw))
    monkeypatch.setattr(exl3_stream_trace, "get_exl3_stream_trace", lambda: trace)
    demand_rows = [[0, 0]]
    service = module.Exl3RamMissService()
    service._manager = manager
    service.host = SimpleNamespace(fatal_seq=lambda: 0, layer_rows=lambda: demand_rows[0])

    manager._accumulate_registers("decode", torch.zeros((2, 4)), [False, False])
    service.fail_stop_check()  # baseline: no line yet
    manager._graph_counters.copy_(torch.tensor([[6, 2], [6, 1]]))  # one replay's gathers
    manager._accumulate_registers("decode", torch.zeros((2, 4)), [False, False])  # the observer
    assert int(manager._graph_counters.sum()) == 0
    demand_rows[0] = [1, 1]
    service.fail_stop_check()
    assert lines == [dict(layer_rows_delta=[1, 1], routed_rows=12, routed_misses=3)]


def test_a_test_fault_spec_parses():
    assert module.parse_fault("") is None
    assert module.parse_fault("40:20") == (40, 20.0)


def test_apply_graph_refuses_the_p3_only_backend(tiers):
    from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    service, streamers, caches = tiers
    streamer = streamers[0]
    streamer.row_backend = PinnedTierRowBackend({0: None}, torch.full((6,), -1, dtype=torch.int64), 6)
    with pytest.raises(RuntimeError, match="option C"):
        Exl3MoEMethod._apply_graph(streamer.layer, streamer, torch.zeros((1, 8)), torch.ones((1, 6)), torch.zeros((1, 6), dtype=torch.long), 10.0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
```
Create `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`:
```python
"""End to end on the GPU (window): a captured EXL3 in-graph MoE whose RAM misses are
served by option C, against the eager streamed apply; and a forced timeout fail-stop."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 16, 6


def _layers(tmp_path, monkeypatch, timeout_ms=2000):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    monkeypatch.setenv("SGLANG_DSV41_RAM_MISS_TIMEOUT_MS", str(timeout_ms))
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True):
        layer = torch.nn.Module()
        layer.layer_id = 0
        layer.top_k = TOP_K
        fmt = Exl3ExpertFormat(layout, 0, direct=False)
        streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
        layer._nvfp4_expert_streamer = streamer
        pinned = ExpertPinnedHostCache(streamer, 8, **fmt.pinned_tier_options(layer))
        hot = ExpertHotCache(streamer, 3, scratch_rows=TOP_K)
        hot.reassign([0, 1, 2])
        streamer.enable_graph_gather(TOP_K)
    checks = []
    manager = type("M", (), {"register_fail_stop_check": lambda self, f: checks.append(f), "add_residency_listener": lambda self, f: f(0, list(hot.slot_to_expert))})()
    fmt.attach_hot_cache_manager(manager, streamer)
    return layer, streamer, service_module.Exl3RamMissService.get(), checks


def test_ram_misses_inside_a_replay_are_served(tmp_path, monkeypatch):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, monkeypatch)
    try:
        gen = torch.Generator(device="cpu").manual_seed(3)
        x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
        weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
        ids = torch.tensor([[0, 3, 5, 1, 7, 6]], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        for route in ([9, 10, 11, 0, 1, 12], [13, 14, 15, 2, 9, 4]):  # misses beyond the 8 pinned rows
            ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            got = out.float().clone()
            want = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), 10.0).float()
            rel = float((got - want).norm() / want.norm())
            assert rel <= 1.2e-2, (route, rel)
            assert streamer.row_backend.keep.item() == 1.0
            for check in checks:
                check()
        assert service.host.counters()["rows_read"] >= 6
    finally:
        service.shutdown()


def test_a_forced_timeout_fails_stop_without_hanging(tmp_path, monkeypatch):
    import time

    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, service, checks = _layers(tmp_path, monkeypatch, timeout_ms=100)
    try:
        x = torch.zeros((1, HIDDEN), device="cuda", dtype=torch.bfloat16)
        weights = torch.full((1, TOP_K), 1.0 / TOP_K, device="cuda")
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda", dtype=torch.int32)
        Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
        service.host.inject(delay_s=10.0)
        ids.copy_(torch.tensor([[13, 14, 15, 0, 1, 2]], device="cuda", dtype=torch.int32))
        started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        assert time.perf_counter() - started < 2.0
        assert streamer.row_backend.keep.item() == 0.0
        with pytest.raises(RuntimeError, match="exl3 RAM miss"):
            for check in checks:
                check()
    finally:
        service.host.inject(delay_s=0.0)
        service.shutdown()
```

- [ ] **Step 2: Commit red, push, run red**

`git add test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`; commit `test(dsv41): red tests for option C wiring into EXL3 graph decode`. Run `<DSV41-CPU> test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py`. Expected: the CPU tests fail (`module ... has no attribute 'Exl3RamMissService'`; the trace test with `no attribute 'enabled'`); the GPU file skips. If the registers test fails earlier with an `AttributeError` on the manager, read `_accumulate_registers` and `_registers_for` (`expert_hot_cache.py:1908-1965`) and set that attribute in the test to its empty value; recommit the test.

- [ ] **Step 3: Implement**

`environ.py` — next to `SGLANG_DSV41_EXPERT_TRACE_PATH`, add:
```python
    # Option C (EXL3 graph decode): how long the in-graph wait for the RAM-miss
    # thread may take per MoE layer, in ms, before the process fails stop.
    SGLANG_DSV41_RAM_MISS_TIMEOUT_MS = EnvInt(2000)
    # Test only: "<demands>:<seconds>" makes the RAM-miss thread sleep before every
    # demand read once that many demands have read rows (forces an Engine-level
    # timeout after capture). Empty: off.
    SGLANG_TEST_DSV41_RAM_MISS_FAULT = EnvStr("")
```
`exl3_ram_miss.py` (srt) — append:
```python
import logging
from collections import OrderedDict
from typing import Callable, Optional

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissDevice, Exl3RamMissHost, new_page
from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend

logger = logging.getLogger(__name__)


class NativePinnedSlotTable:
    """``PinnedSlotTable`` over the C++ service's slot bookkeeping for one streamed layer.

    Created at pinned-tier construction (from ``pinned_tier_options``); the service
    starts on first use, once every layer's slabs exist.
    """

    def __init__(self, service: "Exl3RamMissService", layer_id: int, streamer_of: Callable[[], object]):
        self.service = service
        self.layer_id = layer_id
        # Bound by ExpertPinnedHostCache.__init__ (bind_capacity) to the tier's row count.
        self.capacity: Optional[int] = None
        self.streamer_of = streamer_of
        self._seen_version = -1
        service.register(layer_id, self)

    def bind_capacity(self, capacity: int) -> None:
        self.capacity = int(capacity)

    @property
    def _row(self) -> int:
        self.service.ensure_started()
        return self.service.row_of(self.layer_id)

    @property
    def slot_to_expert(self) -> list[int]:
        return self.service.host.slot_to_expert(self._row)

    @property
    def expert_to_slot(self) -> "OrderedDict[int, int]":
        row = self._row
        mapping = self.service.host.mapping(row)
        return OrderedDict((expert, mapping[expert]) for expert in self.service.host.lru_order(row))

    def __contains__(self, expert_id: int) -> bool:
        return self.service.host.contains(self._row, int(expert_id))

    def touch(self, expert_id: int) -> None:
        self.service.host.touch(self._row, int(expert_id))

    def assign(self, expert_id: int, protected=frozenset()) -> tuple[int, Optional[int]]:
        return self.service.host.assign(self._row, int(expert_id), [int(e) for e in protected])

    def release(self, slot: int) -> None:
        self.service.host.release(self._row, int(slot))

    def mapping(self, num_experts: int) -> list[int]:
        return self.service.host.mapping(self._row)

    def before_host_use(self, cache) -> None:
        self.service.before_host_use()
        version = self.service.host.version()
        if version != self._seen_version:
            cache._refresh_mapping()
            self._seen_version = self.service.host.version()

    def after_host_use(self, cache) -> None:
        self.service.after_host_use()


class Exl3RamMissRowBackend(PinnedTierRowBackend):
    """``PinnedTierRowBackend`` whose translation posts RAM misses to the thread and waits.

    ``routes`` holds the layer's routed experts (the protect set); ``_apply_graph``
    copies them in before each gather. ``post`` = post kernel, wait kernel (which
    writes ``host_rows``, ``keep`` and ``ram_miss``), then the segment copy.
    """

    name = "exl3_ram_miss"

    def __init__(self, segments, device_side: Exl3RamMissDevice, row: int, next_row: int, capacity: int, device) -> None:
        super().__init__(segments, torch.full((1,), -1, dtype=torch.int64, device=device), capacity)
        self.device_side = device_side
        self.row = row
        self.next_row = next_row
        self.routes = torch.full((capacity,), -1, dtype=torch.int64, device=device)

    def translate(self, tag, plan) -> None:
        self.device_side.post(self.row, plan.expert_ids, plan.count, self.routes, self.next_row)
        self.device_side.wait(self.row, plan.expert_ids, plan.count, self.host_rows, self.keep, self.ram_miss)


def parse_fault(spec: str) -> Optional[tuple[int, float]]:
    """``SGLANG_TEST_DSV41_RAM_MISS_FAULT`` = ``"<demands>:<seconds>"``: delay each demand
    read by ``seconds`` once ``demands`` demands have read rows. Empty: no fault."""
    if not spec:
        return None
    demands, seconds = spec.split(":")
    return int(demands), float(seconds)


class Exl3RamMissService:
    """Process-wide option C service: the C++ thread, the device kernels, the hooks."""

    _instance: "Optional[Exl3RamMissService]" = None

    @classmethod
    def get(cls) -> "Exl3RamMissService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.tables: dict[int, NativePinnedSlotTable] = {}
        self.host: Optional[Exl3RamMissHost] = None
        self.device_side: Optional[Exl3RamMissDevice] = None
        self.page = None
        self.slot_map = None
        self._rows: dict[int, int] = {}
        self._manager = None
        self._pause_depth = 0
        self._trace_rows: Optional[list[int]] = None
        self._trace_graph: Optional[list[int]] = None

    def register(self, layer_id: int, table: NativePinnedSlotTable) -> None:
        if self.host is not None:
            raise RuntimeError("exl3 RAM miss: a pinned tier was built after the service started")
        self.tables[layer_id] = table

    def row_of(self, layer_id: int) -> int:
        return self._rows[layer_id]

    def ensure_started(self) -> None:
        if self.host is not None:
            return
        streamers = {layer_id: table.streamer_of() for layer_id, table in sorted(self.tables.items())}
        missing = [layer_id for layer_id, s in streamers.items() if s is None or s.pinned_host_cache is None]
        if missing:
            raise RuntimeError(f"exl3 RAM miss: layers {missing} have no pinned tier yet")
        fmt = next(iter(streamers.values())).format
        tables = exl3_ram_miss_tables(
            fmt.layout, fmt.segment_map(), {layer_id: s.pinned_host_cache.tensors for layer_id, s in streamers.items()}
        )
        pin = torch.cuda.is_available()
        self.page = new_page(pin=pin)
        slot_map = torch.full(tuple(tables.reads.shape[:2]), -1, dtype=torch.int32)
        self.slot_map = slot_map.pin_memory() if pin else slot_map
        self._rows = {layer_id: row for row, layer_id in enumerate(tables.layer_ids)}
        self.host = Exl3RamMissHost(tables, page=self.page, slot_map=self.slot_map, direct=fmt._resolve_direct())
        self.host.start_thread(fatal_wait_s=30.0)
        fault = parse_fault(envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.get())
        if fault is not None:
            demands, seconds = fault
            self.host.inject(delay_s=seconds, delay_after_demands=demands)
            logger.warning("exl3 RAM miss TEST FAULT: demand reads sleep %.1f s after %d demands", seconds, demands)
        logger.info(
            "exl3 RAM miss thread started: %d layers, %d files, slot bytes %d, wait timeout %d ms",
            len(tables.layer_ids), len(tables.paths), tables.slot_bytes, envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
        )

    def before_host_use(self) -> None:
        """Eager pinned-tier use: finish queued device work, then pause the thread (nesting counted)."""
        self.ensure_started()
        if self._pause_depth == 0:
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.current_stream().synchronize()
            self.host.pause(2 * envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get() / 1000 + 1.0)
        self._pause_depth += 1

    def after_host_use(self) -> None:
        self._pause_depth -= 1
        if self._pause_depth == 0:
            self.host.resume()

    def attach(self, manager, streamer) -> None:
        """The format's ``attach_hot_cache_manager``: hooks once, a row backend per layer."""
        self.ensure_started()
        if self._manager is None:
            self._manager = manager
            manager.register_fail_stop_check(self.fail_stop_check)
            manager.add_residency_listener(self.on_residency)
        if not getattr(streamer, "_graph_pinned_tier", False):
            return
        cache = streamer.hot_cache
        if self.device_side is None:
            from sglang.srt.layers.moe.exl3_expert_format import prefetch_enabled

            self.device_side = Exl3RamMissDevice(
                self.page,
                self.slot_map,
                device=cache.device,
                layers=len(self._rows),
                timeout_ms=envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.get(),
                advise=prefetch_enabled(),
            )
        row = self.row_of(streamer.layer_id)
        next_row = row + 1 if row + 1 < len(self._rows) else -1
        previous = streamer.row_backend
        streamer.row_backend = Exl3RamMissRowBackend(
            previous.segments, self.device_side, row, next_row, streamer.graph_gather_rows, cache.device
        )

    def on_residency(self, layer_id: int, slot_to_expert: list[int]) -> None:
        if layer_id in self._rows:
            self.host.set_hot(self.row_of(layer_id), slot_to_expert)

    def fail_stop_check(self) -> None:
        """Per batch (the scheduler's doorbell hook): raise when a wait timed out or failed."""
        if self.host is None:
            return
        fatal = self.host.fatal_seq()
        if fatal:
            raise RuntimeError(
                f"exl3 RAM miss: request {fatal} timed out or failed "
                f"(thread {self.host.counters()}); fail-stop"
            )
        self._trace_step()

    def _graph_rows(self) -> Optional[list[int]]:
        """Routed rows and routed misses of every graph gather so far, from the manager's
        registers (streamers' own graph_counters are zeroed every forward, plan D23)."""
        registers = getattr(self._manager, "_registers", None)
        if not registers:
            return None
        total = sum(phase["graph_rows"].sum(dim=0) for phase in registers.values())
        return [int(value) for value in total.tolist()]

    def _trace_step(self) -> None:
        """One graph decode step's G and RAM misses into the stream trace (trace runs only)."""
        from sglang.srt.layers.moe.exl3_stream_trace import get_exl3_stream_trace

        trace = get_exl3_stream_trace()
        if not trace.enabled:
            return
        graph = self._graph_rows()
        if graph is None:
            return
        rows = self.host.layer_rows()  # demand rows only: advisory reads are not misses
        if self._trace_graph is not None and graph[0] > self._trace_graph[0]:
            trace.record_graph_step(
                layer_rows_delta=[a - b for a, b in zip(rows, self._trace_rows)],
                routed_rows=graph[0] - self._trace_graph[0],
                routed_misses=graph[1] - self._trace_graph[1],
            )
        self._trace_rows, self._trace_graph = rows, graph

    def shutdown(self) -> None:
        if self.host is not None:
            self.host.stop()
```
`exl3_expert_format.py`:
1. Add a module function:
```python
def prefetch_enabled() -> bool:
    """Option F advisories (Task 15 adds the env var; off until then)."""
    return False
```
2. In `pinned_tier_options`, replace `        return {"is_pinned": is_pinned}` with:
```python
        options = {"is_pinned": is_pinned}
        if envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.get():
            # Option C: the C++ RAM-miss thread owns this tier's slots (plan D12).
            # ExpertPinnedHostCache binds the tier's capacity into the table.
            from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService, NativePinnedSlotTable

            options["slot_table"] = NativePinnedSlotTable(
                Exl3RamMissService.get(), self.layer_id, lambda: expert_streamer_of(layer)
            )
        return options
```
3. Add to `Exl3ExpertFormat`:
```python
    def attach_hot_cache_manager(self, manager, streamer) -> None:
        """Option C hooks (fail-stop check, residency pushes, the RAM-miss row backend),
        for a layer whose pinned tier runs on the native slot table."""
        from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService, NativePinnedSlotTable

        tier = getattr(streamer, "pinned_host_cache", None)
        if not isinstance(getattr(tier, "_lru", None), NativePinnedSlotTable):
            return
        Exl3RamMissService.get().attach(manager, streamer)
```
`exl3.py` `_apply_graph` — before `remap, _ = streamer.gather(topk_ids)` add:
```python
        from sglang.srt.layers.moe.expert_row_plan import PinnedTierRowBackend

        if type(streamer.row_backend) is PinnedTierRowBackend and not getattr(layer, "_exl3_allow_p3_only", False):
            # Without option C nothing serves or checks a RAM miss inside a replay.
            raise RuntimeError(
                "exl3 in-graph MoE needs option C (Exl3RamMissRowBackend); the pinned "
                "tier's slot table was not native (SGLANG_MOE_EXPERT_GRAPH_GATHER off at load?)"
            )
        routes = getattr(streamer.row_backend, "routes", None)
        if routes is not None:
            # The option C post protects every routed expert of this layer.
            routes.copy_(topk_ids.reshape(-1))
```
(This Python runs at warmup and capture only; replays do not re-enter it.)
`exl3_stream_trace.py` — add to `Exl3StreamTrace`:
```python
    @property
    def enabled(self) -> bool:
        """A trace file is open (SGLANG_DSV41_EXPERT_TRACE_PATH was set)."""
        return self._file is not None

    def record_graph_step(self, layer_rows_delta, routed_rows: int, routed_misses: int) -> None:
        """One graph decode step: its VRAM misses (G) and the demand rows the RAM-miss
        thread read (f). In-graph MoE layers produce no host gather stats; the option C
        service calls this once per batch with the step's deltas. The line is shaped
        like a one-token forward, so tier_sim.live_summary counts it as a decode token.
        """
        ram = int(sum(layer_rows_delta))
        self.forwards += 1
        self.decode_tokens += 1
        self.decode_vram_misses += int(routed_misses)
        self.decode_ram_misses += ram
        self.vram_misses += int(routed_misses)
        self.ram_misses += ram
        self._last_layer = None  # the next eager call starts a new forward
        if self._file is not None:
            line = {
                "forward": self.forwards,
                "layer": -1,
                "tokens": 1,
                "kind": "graph_step",
                "experts": [],
                "counts": [],
                "vram_miss": int(routed_misses),
                "ram_miss": ram,
                "routed_rows": int(routed_rows),
                "layer_ram_rows": [int(v) for v in layer_rows_delta],
                "t": round(time.monotonic(), 6),
            }
            self._file.write(json.dumps(line) + "\n")
```
`scripts/dsv41/tier_sim.py` — make the first statement of `simulate`:
```python
    # Graph decode steps carry no per-layer routes; only live_summary reads them.
    calls = [call for call in calls if call.get("kind") != "graph_step"]
```
(`load_trace` keeps every line; `live_summary` needs no change: a `graph_step` line is a forward of its own with `tokens == 1`, `vram_miss` and `ram_miss`.) Append to `test/manual/dsv41/test_tier_sim.py`:
```python
def test_graph_steps_mix_with_eager_prefill_lines():
    calls = [
        {"forward": 1, "layer": 0, "tokens": 256, "vram_miss": 9, "ram_miss": 4, "experts": [1], "counts": [9]},
        {"forward": 1, "layer": 1, "tokens": 256, "vram_miss": 9, "ram_miss": 4, "experts": [2], "counts": [9]},
        {"forward": 2, "layer": -1, "tokens": 1, "kind": "graph_step", "vram_miss": 6, "ram_miss": 3, "experts": [], "counts": []},
        {"forward": 3, "layer": -1, "tokens": 1, "kind": "graph_step", "vram_miss": 4, "ram_miss": 1, "experts": [], "counts": []},
    ]
    live = tier_sim.live_summary(calls, warmup=0)
    assert live["decode_tokens"] == 2 and live["G"] == 5.0 and live["f"] == 0.4
```

- [ ] **Step 4: Commit, push, run green**

`git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/exl3_ram_miss.py python/sglang/srt/layers/moe/exl3_expert_format.py python/sglang/srt/layers/quantization/exl3.py python/sglang/srt/layers/moe/exl3_stream_trace.py scripts/dsv41/tier_sim.py test/manual/dsv41/test_tier_sim.py`; commit `feat(dsv41): option C serves EXL3 RAM misses inside the decode graph`. Run `<DSV41-CPU> <DSV41_SUITE>` plus every new dsv41 test file. Expected: previous counts + 8 service tests + 1 tier_sim case; the GPU files skip.

- [ ] **Step 5: Run the GPU tests**

`<GPU> /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_graph_apply_gpu.py test/manual/dsv41/test_exl3_ram_miss_cuda.py -k "not overheads"`. Expected: all pass: RAM misses inside a captured replay are served by option C and match the eager apply; a forced timeout fails stop within 2 s; Task 9's and Task 13's tests still pass on the wired code. A failure is fixed by a red/green pair in this task's files and this step rerun; exit code 75 is the GPU lock rule. Append `Task 14: complete`.

---

## Phase P5 — prefetch hook (option F, minimal)

### Task 15: Next-layer advisory prefetch behind `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`

**Blocked if Task 1 fails.**

**Branch:** `dsv41`. Read `.claude/skills/env-var-conventions/SKILL.md` before Step 3 (the name follows its Rule 4: `ENABLE_` verb after the `DSV41_` family prefix; `EnvBool(False)`, opt-in).

The device side already posts advisories when `advise` is set (Task 13) and the thread serves them demand-first (Tasks 11–12). This task adds the switch, the gate rule, and the tests that the advisory path does what D22 says. The predictor is the one R1' names: the previous token's routes for layer L+1, kept on the device in `Exl3RamMissDevice.last_routes`. Any other predictor writes its ids into that table's row L+1 before layer L's post; nothing else changes.

**Files:**
- Modify: `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/exl3_expert_format.py` (`prefetch_enabled`), `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (`_check`)
- Create: `test/registered/unit/kernels/test_exl3_ram_miss_advisory.py` (CPU)
- Modify: `test/manual/dsv41/test_exl3_ram_miss_cuda.py` (GPU case), `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Produces: `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH = EnvBool(False)`; `prefetch_enabled() -> bool` reads it; the gate refuses it without breakable decode and graph gather.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_miss_advisory.py`:
```python
"""Advisory (next-layer prefetch) records on the RAM-miss thread (CPU, simulated device)."""

import sys
import time

import pytest
import torch

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissHost, new_page, page_word, sim_post, sim_wait
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _host(tmp_path):
    s = ram_miss_setup(tmp_path)
    page = new_page(pin=False)
    host = Exl3RamMissHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), direct=False)
    host.start_thread(fatal_wait_s=5.0)
    return page, host


def _until(predicate, timeout_s=5.0):
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_an_advisory_loads_rows_before_their_demand(tmp_path):
    page, host = _host(tmp_path)
    try:
        sim_post(page, 1, need=[4, 5], protect=[4, 5], advisory=True, after=page_word(page, "demand_head") + 10)
        assert _until(lambda: host.contains(1, 4) and host.contains(1, 5))
        assert host.layer_advisory_rows() == [0, 2] and host.layer_rows() == [0, 0]
        # The demand then finds them in RAM: a touch-only request, no read.
        before = host.counters()["rows_read"]
        assert sim_wait(page, sim_post(page, 1, need=[], protect=[4, 5]), 10) == 1
        assert host.counters()["rows_read"] == before
    finally:
        host.stop()


def test_an_advisory_whose_layer_already_posted_its_demand_is_skipped(tmp_path):
    page, host = _host(tmp_path)
    try:
        seq = sim_post(page, 0, need=[], protect=[0])
        assert sim_wait(page, seq, 10) == 1
        sim_post(page, 1, need=[3], protect=[3], advisory=True, after=seq - 1)  # its demand seq is reached
        assert _until(lambda: host.counters()["advisories_skipped"] >= 1)
        assert not host.contains(1, 3)
    finally:
        host.stop()


def test_a_demand_preempts_an_advisory_in_flight(tmp_path):
    page, host = _host(tmp_path)
    try:
        host.inject(delay_s=0.3)
        sim_post(page, 1, need=[1, 2, 3], protect=[1, 2, 3], advisory=True, after=page_word(page, "demand_head") + 10)
        assert _until(lambda: host.counters()["advisories"] == 1)
        started = time.perf_counter()
        assert sim_wait(page, sim_post(page, 0, need=[], protect=[]), 10) == 1
        assert time.perf_counter() - started < 0.8  # the advisory gave up after at most its first read delay
        assert not any(host.contains(1, e) for e in (1, 2, 3))
    finally:
        host.inject(delay_s=0.0)
        host.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
```
Append to `test/manual/dsv41/test_exl3_ram_miss_cuda.py`:
```python
def test_layer_posts_advise_the_next_layer(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path, advise=True, layers=2)
    try:
        b = _buffers()
        dev.last_routes[1, :3].copy_(torch.tensor([9, 10, 11], dtype=torch.int32))  # the predictor's ids
        _set(b, [2], [2])
        _step(dev, b, row=0, next_row=1)
        torch.cuda.synchronize()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and not all(host.contains(1, e) for e in (9, 10, 11)):
            time.sleep(0.005)
        assert all(host.contains(1, e) for e in (9, 10, 11))
        assert dev.last_routes[0, 0].item() == 2  # layer 0 remembered this token's routes
        assert host.counters()["advisory_rows"] == 3
    finally:
        host.stop()
```
Append to `test/registered/unit/test_expert_stream_requirements_exl3.py`:
```python
def test_prefetch_needs_graph_gather_under_breakable_decode(model_dir):
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=True)
    with pytest.raises(ValueError, match="SGLANG_DSV41_ENABLE_EXPERT_PREFETCH"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=True)
```

- [ ] **Step 2: Commit red, push, run red**

`git add` the three test files; commit `test(dsv41): red tests for next-layer advisory prefetch`. Run `<DSV41-CPU> test/registered/unit/kernels/test_exl3_ram_miss_advisory.py test/registered/unit/test_expert_stream_requirements_exl3.py test/manual/dsv41/test_exl3_ram_miss_cuda.py`. Expected: the three advisory tests pass already (Tasks 11–12 built the thread side; this is their first run, a characterization — if any fails, it is a Task 11 or 12 bug: fix it in `exl3_ram_miss_host.cpp` with its own red/green commit pair before going on); the gate test fails (`envs` has no `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`); the GPU file skips.

- [ ] **Step 3: Implement**

`environ.py`, next to `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`:
```python
    # Option F on top of option C: during MoE layer L of a graph decode step, post
    # the previous token's routes for layer L+1 that are not in RAM as advisory
    # reads for the RAM-miss thread (demands always go first). Off by default.
    SGLANG_DSV41_ENABLE_EXPERT_PREFETCH = EnvBool(False)
```
`exl3_expert_format.py` — replace `prefetch_enabled`'s body with:
```python
def prefetch_enabled() -> bool:
    """Option F advisories (``SGLANG_DSV41_ENABLE_EXPERT_PREFETCH``)."""
    return envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH.get()
```
`expert_stream_requirements_exl3.py` — at the top of `_check`, before `graph = cfg.cuda_graph_config`:
```python
    if envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH.get() and not budgets.graph_gather:
        raise ValueError(
            "SGLANG_DSV41_ENABLE_EXPERT_PREFETCH posts advisories from the in-graph MoE; "
            "it needs SGLANG_MOE_EXPERT_GRAPH_GATHER=1 with breakable decode graphs"
        )
```
(graph gather already requires decode graphs in `memory_hook`, and `_check` requires them to be breakable at bs 1.)

- [ ] **Step 4: Commit, push, run green**

`git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/exl3_expert_format.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`; commit `feat(dsv41): next-layer advisory prefetch for option C (off by default)`. Run `<DSV41-CPU> <DSV41_SUITE>` plus every new dsv41 test file. Expected: previous counts + 3 advisory + 1 gate; GPU files skip.

- [ ] **Step 5: Run the GPU advisory test**

`<GPU> /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/manual/dsv41/test_exl3_ram_miss_cuda.py -k advise`. Expected: `1 passed` (layer 0's post advises layer 1's predicted ids and the thread loads them). A failure is fixed by a red/green pair in the owning file (`exl3_ram_miss.cuh` for the post kernel, Task 13's; the thread, Tasks 11–12's) and this step rerun; exit code 75 is the GPU lock rule. Append `Task 15: complete`.

---

## Phase P6 — GPU window and record

### Task 16: GPU window — regression tests, capture smoke, R4 numerics, §9.3 acceptance, corpus arms, graph-mode Nsight

**Blocked if Task 1 fails.** Needs Tasks 2–15 complete in the ledger.

**Branch:** `dsv41` (no code changes; a bug found here is fixed by a red/green commit pair on the task that owns the file, pushed, pulled, and the failed step rerun).

**Files:** none in the repository. Artifacts go to `ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b` on divix01; `$ANA/window.log` is the chronological record (every step appends one line: `date --iso-8601=seconds`, the step, the exit code).

**How commands run.** Every GPU command is `<GPU> bash -c ". $ANA/env-<x>.sh && <cmd>"` (Global Constraints: `gpu-run.sh` takes the lock; exit code 75 is the GPU lock rule; `\$T3`/`\$FULL` stay escaped so the inner shell expands them after the env file defines them). The `env-*.sh` files source `ANA3A/env.sh`, which sets `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, the EXL3 build dirs, `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`, `FULL`, and the Window C tier budgets; that file `cd`s into the divix01 worktree. Production is down; never start it.

- [ ] **Step 1: Pre-flight**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41
for n in 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do grep -qE "^Task $n: complete" .superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md || echo "Task $n NOT complete"; done
grep -E '^Task 1: complete — PASS' .superpowers/sdd/2026-09-19-dsv41-phase3b-optionC/progress.md
git rev-parse HEAD
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41 && git pull --ff-only shared dsv41 && git rev-parse HEAD && git status --short | head -3; nvidia-smi --query-compute-apps=pid --format=csv,noheader; free -g | head -2'
```
Expected: no `NOT complete`; the PASS line; the two HEADs equal; divix01 clean; no compute process on the GPU; ≥ 90 GiB free RAM. **Rule:** a `NOT complete` task is finished first (its own steps); a compute process on the GPU that is not ours means the lock is held elsewhere: apply the GPU lock rule to this whole task. Then write the environment files:
```bash
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b
cat > $ANA/env-full.sh <<EOS
. /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a/env.sh
ANA=$ANA
export SGLANG_MOE_EXPERT_GRAPH_GATHER=1 SGLANG_DSV41_RAM_MISS_TIMEOUT_MS=2000 SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=0
EOS
cat > $ANA/env-full-p1.sh <<EOS
. /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a/env.sh
ANA=$ANA
export SGLANG_MOE_EXPERT_GRAPH_GATHER=0
EOS
cat > $ANA/env-t3-c.sh <<EOS
. $ANA/env-t3.sh
export SGLANG_MOE_EXPERT_GRAPH_GATHER=1 SGLANG_DSV41_RAM_MISS_TIMEOUT_MS=2000 SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=0
EOS
echo "[$(date --iso-8601=seconds)] window start at $(git -C /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41 rev-parse --short HEAD)" >> $ANA/window.log'
```
(`env-t3.sh` and `prompt-0.txt` exist from Task 5 Step 5.)

- [ ] **Step 2: Regression rerun of every GPU test (~30 min)**

`<GPU> bash -c ". $ANA/env-full.sh && env -u SGLANG_MOE_EXPERT_GRAPH_GATHER /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -rf test/manual/dsv41/test_exl3_moe_probe_gpu.py test/manual/dsv41/test_exl3_graph_apply_gpu.py test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_ops_gpu.py test/manual/dsv41/test_exl3_moe_gpu.py test/manual/dsv41/test_exl3_method_gpu.py test/manual/dsv41/test_exl3_shard_source_gpu.py test/manual/dsv41/test_exl3_stream_apply_gpu.py test/registered/unit/layers/moe/test_expert_pinned_graph_gather_cuda.py test/registered/unit/kernels/test_expert_doorbell_copier.py test/registered/unit/kernels/test_expert_cache_transfer.py > $ANA/cuda-tests.log 2>&1"` (`env -u` unsets graph gather for pytest: the tests set what they need). Expected: all pass, except the doorbell file's known timing-flaky `test_a_second_exhausted_drain_whose_copy_never_lands_aborts_after_the_fatal_wait` (§15.2), which may fail; the doorbell tests otherwise pass unmodified (the doorbell is untouched, D18). **Rule:** any other failure is a regression of the task that owns the test: fix it there (red/green) and rerun this step.

- [ ] **Step 3: §9.3 acceptance at component level, on real rows (~10 min)**

`<GPU> bash -c ". $ANA/env-full.sh && env -u SGLANG_MOE_EXPERT_GRAPH_GATHER DSV41_EXL3_DIR=/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw DSV41_PROBE_LAYER=3 DSV41_RAM_MISS_OUT=$ANA/ram-miss-overheads.json /data/models/slang/.venv/bin/python -m pytest -p no:cacheprovider -q -s test/manual/dsv41/test_exl3_ram_miss_cuda.py -k overheads > $ANA/ram-miss-overheads.log 2>&1"`.

Expected: `1 passed`; the JSON holds `hit_path_us_per_layer` (acceptance 1: per-layer overhead on the RAM-hit path; ×40 is the per-token cost), `stream_idle_ms_1_miss` and `stream_idle_ms_6_miss` (acceptance 2: stream idle on forced NVMe misses; expected ≈ 8–11 ms and ≈ 25–65 ms from §16.4/§16.12's 8.1 ms read + 2.2 ms split per row, less for 6 rows at queue depth 6). Acceptance 3 at component level is Step 2's `test_a_hung_read_times_out_and_everything_after_is_fast` and `test_a_forced_timeout_fails_stop_without_hanging`; Step 7 repeats it at Engine level.

- [ ] **Step 4: Capture smoke and R4 arms on the truncated model (~25 min)**

`<GPU> bash -c ". $ANA/env-t3-c.sh && /data/models/slang/.venv/bin/python scripts/dsv41/graph_parity.py --model \$T3 --prompt-file $ANA/prompt-0.txt --prompt-tokens 256 --new-tokens 32 --graph-gather --debug-arm --control --out $ANA/parity-t3-c.json > $ANA/parity-t3-c.log 2>&1"`, then
```bash
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b; grep -E "Breakable CUDA graph captured|exl3 RAM miss thread started|Debug mode for CUDA graph|exl3 RAM miss: request" $ANA/parity-t3-c.log'
```
`graph_parity.py` gives each Engine its own `SGLANG_MOE_EXPERT_GRAPH_GATHER` (eager and control 0, graph and debug 1), so the eager Engines launch under the option C environment (Task 5). Expected in the log: `exl3 RAM miss thread started: 3 layers` (graph and debug Engines); `Breakable CUDA graph captured: ... breaks=1` for the graph Engine (only the Engram lookup; the three MoE layers are in-graph; `parity-t3-p1.log` from Task 5 shows `breaks=4`); `Debug mode for CUDA graph is enabled` for the debug Engine; no `exl3 RAM miss: request` error. **Rule:** a capture or launch error names the op: fix it in the owning task (Task 9 for the MoE path, Task 3 for breaks, Task 14 for the service) and rerun.

- [ ] **Step 5: R4 verdict**

```bash
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b; for f in parity-t3-p1 parity-t3-c; do /data/models/slang/.venv/bin/python -c "import json; r=json.load(open(\"$ANA/$f.json\")); print(\"$f\", {k: v for k, v in r.items() if \"_vs_\" in k})"; done'
```
Apply D3's decision rules and record the outcome in `$ANA/window.log` and the ledger:
- `parity-t3-p1` `eager_vs_graph.pass` is true (Task 5 established it; a false here is a regression: fix it, rerun Task 5 Step 5).
- `parity-t3-c` `graph_vs_debug.pass` (bitwise, tolerance 0) is the capture-correctness gate. If it is false, compare with the P1 baseline: when the greedy tokens are identical and `graph_vs_debug.max_abs_dlogprob` <= `parity-t3-p1` `eager_vs_graph.max_abs_dlogprob` (the capture-induced gap outside the MoE, e.g. a different cuBLAS algorithm under capture), accept it and record `graph_vs_debug: not bitwise, within P1 capture gap (<x> <= <y>)`. Otherwise P3 is not done: bisect with the graph and debug logs, fix the owning task and rerun Step 4, at most 2 bisect/fix rounds. After that, record `graph_vs_debug: FAIL (<x>)` and continue; Task 17 reports it as an open defect.
- `parity-t3-c` `debug_vs_eager` (and `eager_vs_graph`) against R4's bar: if it fails while `graph_vs_debug` passes, write `R4: FAIL (fused-kernel numerics) — first_token_mismatch <n>, max_abs_dlogprob <x>; probe rel_fused/rel_loop <a>/<b>` and continue; this does not block P3 or the remaining steps, and Task 17 reports it. `eager_vs_eager` separates run-to-run nondeterminism: if the control itself fails R4's bar, record both and attribute the gap to nondeterminism.

- [ ] **Step 6: Full-model graph smoke (~20 min)**

`<GPU> bash -c ". $ANA/env-full.sh && /data/models/slang/.venv/bin/python scripts/dsv41/trace_corpus.py --graphs --model \$FULL --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --skip 0 --n 1 --prompt-tokens 256 --new-tokens 16 --mem-fraction-static 0.80 --out $ANA/smoke-full.json > $ANA/smoke-full.log 2>&1; nvidia-smi --query-gpu=memory.used --format=csv >> $ANA/smoke-full-gpu-mem.txt"`, then
```bash
ssh divix01 'grep -E "Breakable CUDA graph captured|exl3 RAM miss thread started|Expert hot cache startup|max_total_num_tokens|exl3 RAM miss: request" /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/smoke-full.log'
```
Expected: `exl3 RAM miss thread started: 40 layers`; each captured variant logs `breaks=2` (two Engram layers); the run finishes and writes `smoke-full.json`. The `Expert hot cache startup` line gives the effective hot slots; record them. **Ladder** if KV-pool sizing fails at load: `--mem-fraction-static 0.78`; then `SGLANG_MOE_HOT_GPU_MB=12288` in `env-full.sh` at 0.80 (graph gather's scratch rows, 6 × 13.3 MB × 40 layers ≈ 3,048 MiB, come out of the hot budget). Record the setting that ran (`MEM_FRACTION`, `HOT_MB`); Steps 7–9 use it. **Ladder exhausted:** record `Task 16: full model does not fit — <last error>` in the ledger and `window.log`, skip Steps 8–9 on the full model, run Step 7 (it uses `T3`), and continue with Task 17, which reports it.

- [ ] **Step 7: Engine-level fail-stop (acceptance 3, ~10 min)**

The test-only fault makes every demand read sleep 20 s once 50 demands have read rows, so the timeout fires in decode, after capture (warmups on `T3` read far fewer than 50 demands; the log timestamps confirm the phase):

`<GPU> bash -c ". $ANA/env-t3-c.sh && SGLANG_TEST_DSV41_RAM_MISS_FAULT=50:20 SGLANG_MOE_PINNED_HOST_MB=1024 timeout 900 /data/models/slang/.venv/bin/python scripts/dsv41/trace_corpus.py --graphs --model \$T3 --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --skip 0 --n 1 --prompt-tokens 256 --new-tokens 128 --out $ANA/failstop-t3.json > $ANA/failstop-t3.log 2>&1; echo rc=\$? >> $ANA/failstop-t3.log"`, then
```bash
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b; grep -nE "TEST FAULT|Breakable CUDA graph captured|exl3 RAM miss: request|rc=" $ANA/failstop-t3.log; nvidia-smi --query-compute-apps=pid --format=csv,noheader'
```
Expected: `TEST FAULT` logged at startup; the `exl3 RAM miss: request <n> timed out or failed ...; fail-stop` line comes **after** `Breakable CUDA graph captured` (decode, not warmup); `rc=` non-zero and not `124` (the process stopped itself; `timeout` did not kill it); no process left on the GPU. Record the seconds from the error line to process exit (log timestamps; ≤ the 20 s injected sleep plus shutdown). **Rule:** if no timeout fires (the decode had fewer than 50 reading demands), rerun with `SGLANG_TEST_DSV41_RAM_MISS_FAULT=10:20`; if it fires before capture, rerun with a larger count (100).

- [ ] **Step 8: Corpus arms (R5), ~1 h**

Three arms, each Window C's reduced cold shape on the same sessions 0–3 as `ANA3A/corpus-cold.json`, each a fresh Engine. Hot capacity is matched across arms: option C's scratch rows (3,048 MiB) come out of the hot budget, so the `p1` arm (eager MoE break, no scratch) runs with `SGLANG_MOE_HOT_GPU_MB` = Step 6's `HOT_MB` − 3048.

| Arm | Env | Extra env on the command |
|---|---|---|
| `p1` (graphs, eager MoE break) | `env-full-p1.sh` | `SGLANG_MOE_HOT_GPU_MB=<HOT_MB − 3048>` |
| `c` (option C, prefetch off) | `env-full.sh` | — |
| `cpf` (option C, prefetch on) | `env-full.sh` | `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=1` |

For each arm `<a>`: `<GPU> bash -c ". $ANA/env-<env>.sh && <extra env> SGLANG_DSV41_EXPERT_TRACE_PATH=$ANA/trace-<a>.jsonl SGLANG_MOE_HOT_METRICS_FILE=$ANA/hot-metrics-<a>.jsonl /data/models/slang/.venv/bin/python scripts/dsv41/trace_corpus.py --graphs --model \$FULL --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --skip 0 --n 4 --prompt-tokens 256 --new-tokens 128 --mem-fraction-static <MEM_FRACTION> --out $ANA/corpus-<a>.json > $ANA/corpus-<a>.log 2>&1"`.

Then, CPU only and memory-capped, on divix01 in the worktree:
```
ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 taskset -c 0-63 /data/models/slang/.venv/bin/python -c "
import json, statistics, sys
sys.path.insert(0, 'scripts/dsv41')
import tier_sim
base = json.load(open('/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a/corpus-cold.json'))['per_session'][:4]
for arm in ('p1', 'c', 'cpf'):
    r = json.load(open('$ANA/corpus-%s.json' % arm))
    s = r['per_session']
    live = tier_sim.live_summary(tier_sim.load_trace('$ANA/trace-%s.jsonl' % arm))
    print(arm, 'tok/s mean %.3f' % r['mean_decode_tok_s'], 'per session', [round(x['decode_tok_s'], 3) for x in s],
          'TTFT median %.1f s0 %.1f' % (statistics.median(x['ttft_s'] for x in s), s[0]['ttft_s']),
          'G %.2f f %.4f f_after_warmup %.4f decode_tokens %d' % (live['G'], live['f'], live['f_after_warmup'], live['decode_tokens']))
print('windowC', 'tok/s mean %.3f' % (sum(x['decode_tok_s'] for x in base) / 4), [round(x['decode_tok_s'], 3) for x in base])
" | tee $ANA/corpus-summary.txt
for a in p1 c cpf; do echo "$a $(grep -m1 'Expert hot cache startup' $ANA/corpus-$a.log | grep -oE 'slots[^,]*' | head -1)"; grep 'exl3 RAM miss thread counters' $ANA/corpus-$a.log | tail -1; done | tee -a $ANA/corpus-summary.txt
```
Expected: three JSON reports; `corpus-summary.txt` with tok/s, TTFT median and session 0, `G` and `f` (from demand rows only: advisory reads are counted apart, D23) per arm, `decode_tokens` between 4 × 127 and 4 × 128 for every arm (the overlap schedule may add one overshoot step per session; a smaller count for `c`/`cpf` means the graph trace lost steps: a Task 14 bug), the Window C baseline, each arm's effective hot slots (equal across arms within a few slots), and the thread counters of `c` and `cpf` (`advisory_rows`, `advisories`, `advisories_skipped`, `rows_read`). If the hot-slot counts differ by more than 2%, record the difference: §17.6 states it next to `G`.

- [ ] **Step 9: Graph-mode Nsight decode capture (~15 min)**

Write `$ANA/prof-graph.sh` (option C, prefetch off, Window C profile shape: one unseen session `--skip 12`, 256-token prompt, 64 new tokens, a 15 s capture once ≥ 16 graph decode steps are traced):
```bash
ssh divix01 'ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b; cat > $ANA/prof-graph.sh <<"EOS"
#!/usr/bin/env bash
set -uo pipefail
ANA=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b
. $ANA/env-full.sh
cd /data/models/slang/nvfp4-work/cc-expert-prediction/wt-dsv41
T=$ANA/trace-prof-graph.jsonl
rm -f $T $ANA/prof-graph.nsys-rep $ANA/prof-graph.sqlite
SGLANG_DSV41_EXPERT_TRACE_PATH=$T \
$ANA/gpu-run.sh nsys launch --session-new=dsv41graph \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=graph --python-sampling=true --trace-fork-before-exec=true \
  /data/models/slang/.venv/bin/python scripts/dsv41/trace_corpus.py --graphs --model $FULL \
  --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --skip 12 --n 1 --prompt-tokens 256 --new-tokens 64 \
  --mem-fraction-static ${MEM_FRACTION:-0.80} --out $ANA/corpus-prof-graph.json > $ANA/corpus-prof-graph.log 2>&1 &
APP=$!
steps() { [ -f $T ] && grep -c "\"kind\": \"graph_step\"" $T || echo 0; }
until [ "$(steps)" -ge 16 ] || ! kill -0 $APP 2>/dev/null; do sleep 2; done
if kill -0 $APP 2>/dev/null; then
  echo "capture on at step $(steps)"
  nsys start --session=dsv41graph --sample=process-tree --cpuctxsw=process-tree -o $ANA/prof-graph -f true
  sleep 15
  nsys stop --session=dsv41graph
  echo "capture off at step $(steps)"
else
  echo "NO CAPTURE: the app exited before 16 graph steps"
fi
wait $APP; echo "rc=$?"
EOS
chmod +x $ANA/prof-graph.sh'
```
Run it with `ssh divix01 'MEM_FRACTION=<MEM_FRACTION> /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/prof-graph.sh > /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/prof-graph.log 2>&1'` (it takes the lock itself through `gpu-run.sh`; if `HOT_MB` changed in Step 6 it is already in `env-full.sh`). Expected: `capture on`, `capture off`, `rc=0`, `$ANA/prof-graph.nsys-rep`. **Rule:** `NO CAPTURE` means no `graph_step` lines were written (a Task 14 trace bug; Step 8's `decode_tokens` check would show it too): fix it and rerun; `rc=75` is the GPU lock rule.

Analyse on divix01 only, memory-capped (never copy the report to the laptop):
```
cd $ANA && systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 taskset -c 0-63 bash -c '
nsys export --type sqlite -o prof-graph.sqlite -f true prof-graph.nsys-rep &&
nsys stats --report cuda_api_sum --report osrt_sum --format csv -o prof-graph-stats prof-graph.sqlite &&
sqlite3 prof-graph.sqlite ".tables"' 2>&1 | tee prof-graph-analysis.log
```
Then query (graph executions are in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` in graph mode; eager kernels in `CUPTI_ACTIVITY_KIND_KERNEL`; runtime calls in `CUPTI_ACTIVITY_KIND_RUNTIME` with names in `StringIds`; if `.tables` lists no `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, drop that table from both queries and report GPU busy from kernels plus `cuda_api_sum`'s `cudaGraphLaunch` count only):
```
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 sqlite3 $ANA/prof-graph.sqlite "
WITH span AS (SELECT MIN(start) s, MAX(end) e FROM CUPTI_ACTIVITY_KIND_RUNTIME),
gpu AS (SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE UNION ALL SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL)
SELECT (SELECT (e - s) / 1e9 FROM span) AS span_s,
       (SELECT SUM(end - start) / 1e9 FROM gpu) AS gpu_busy_s_upper,
       (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE) AS graph_execs;
SELECT s.value, COUNT(*), SUM(r.end - r.start) / 1e9 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id
 WHERE s.value LIKE 'cudaGraphLaunch%' OR s.value LIKE 'cudaMemcpyAsync%' OR s.value LIKE 'cudaStreamSynchronize%' OR s.value LIKE 'cudaLaunchKernel%'
 GROUP BY s.value;" | tee -a $ANA/prof-graph-analysis.log
```
`gpu_busy_s_upper` sums graph and kernel intervals without removing overlap (an upper bound; with one stream they do not overlap). Tokens in the window = the `graph_step` lines whose `t` falls between the `capture on`/`off` moments (their step numbers are in `prof-graph.log`). Report per token, next to `prof-summary.md`'s eager numbers: ms/token (in capture and just after), GPU busy and idle, `cudaGraphLaunch`, `cudaMemcpyAsync` (D2H), `cudaStreamSynchronize` and `cudaLaunchKernel` counts, scheduler-thread off-CPU time (`osrt_sum`). **Do not rank kernels** from this report: the graph body is not in the kernel table (project `CLAUDE.md`). Then delete `prof-graph.sqlite` and any Nsight skill cache created on divix01 (`rm -rf /tmp/nvidia/nsight_systems/nsys-skill-cache` if it exists); keep the `.nsys-rep`.

- [ ] **Step 10: Close the window**

```bash
ssh divix01 'nvidia-smi --query-compute-apps=pid --format=csv,noheader; echo "[$(date --iso-8601=seconds)] window end" >> /data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3b/window.log'
```
Expected: no compute process of ours. Never start production. Append `Task 16: complete` with the artifact list.

---

### Task 17: Record — `DSV41_REFERENCE.md` §17, corrections, §11 status, owner numbers

**Runs in both outcomes of Task 1.** If Task 1 failed, write the short record instead: Step 1 (the correction), Step 2's §17.1–17.2 only plus a §17.3 "P1 graph decode" paragraph from `parity-t3-p1.json`/`.log` (Task 5), a sentence "3b option C blocked at P2: <failed bar>", and Step 3 with only the status line "Option C blocked at the P2 probe (§17.2)".

**Branch:** `dsv41`. Docs only.

**Files:**
- Modify: `DSV41_REFERENCE.md` (§8 `breakable` row; §9.3 option A row; §11 Phase 3b; new §17 before `## Sources`; TL;DR line on 3b status if the TL;DR lists phase status)

- [ ] **Step 1: Correct the "~40 decode breaks" claim**

In §8's table, replace the `breakable` decode-graph row's text
"**Our `deepseek_v4.py` already wraps every attention layer in `eager_on_graph` (`:620`)**, so DSV4 already has ~40 breaks/forward, and the eager break functions already run under overlap scheduling in the shipping Qwen config"
with
"Our `deepseek_v4.py`'s `eager_on_graph` breaks (attention, low-ratio sources, Engram hashing) fire only for **extend** batches (`forward_mode.is_extend()`, `models/deepseek_v4.py:2021-2026`, `:2325-2329`, `:4211-4213`), so DSV4 **decode** captures with zero breaks of its own. The ~40 breaks per forward are a prefill property. The eager break functions run under overlap scheduling in the shipping Qwen config (corrected in §17)."

In §9.3's option A row, replace "Works: DSV4 already runs ~40 eager breaks/forward under overlap." with "Works under overlap (Qwen ships eager breaks with it), but DSV4 decode has no breaks today: option A would **add** ~40 per decode forward (§17)."

Check with `grep -n "40 breaks\|40 eager breaks" DSV41_REFERENCE.md`: no remaining claim that decode has ~40 breaks.

- [ ] **Step 2: Write §17**

Insert before `## Sources` a section `## 17. Phase 3b findings (option C)` with these subsections, every number taken from the named artifact (write "not run" where a step did not run, never an estimate in its place):
- **17.1 Scope and code state.** R1' phases P1–P6; the commit the window ran (`window.log` first line); budgets (`env-full.sh`, Task 16 Step 6's `MEM_FRACTION` and `HOT_MB`, and the `p1` arm's reduced hot budget).
- **17.2 P2 probe (gate).** Table from `probe-exl3-moe.json` (Task 1; Task 16 Step 2 reran it): for `num_active` 6 and −1, per route set `rel_fused`, `rel_loop`, `max_abs_fused_vs_loop`; replay bitwise; `eager_us`, `replay_us`; the chosen `num_active`; verdict.
- **17.3 Graph decode.** Breaks per capture from `parity-t3-p1.log`, `parity-t3-c.log`, `smoke-full.log` (`breaks=4` → `breaks=1` on `T3`; `breaks=2` on the full model: the two Engram lookups). State: "DSV4 decode had zero breaks before 3b; the ~40-breaks claim was prefill-only (corrected in §8 and §9.3)."
- **17.4 R4 numerics.** From both parity reports: `eager_vs_graph`, `eager_vs_eager`, and for option C `graph_vs_debug` (the bitwise capture gate) and `debug_vs_eager`. State the D3 outcome as recorded in Task 16 Step 5, e.g. `R4: FAIL (fused-kernel numerics)` with its numbers, or `R4: PASS`.
- **17.5 §9.3 acceptance.** A table with the four criteria: (1) RAM-hit path overhead per layer and ×40 per token (`ram-miss-overheads.json`); (2) stream idle on forced 1- and 6-row NVMe misses (same file); (3) timeout fail-stop: component tests (Task 16 Step 2) and the Engine-level run (`failstop-t3.log`: the error line after capture, rc, seconds to exit); (4) capture under breakable with overlap scheduling on (`smoke-full.log`: overlap is on by default in `trace_corpus`; quote the captured line).
- **17.6 Corpus arms (R5).** `corpus-summary.txt` as a table: arm, effective hot slots, tok/s mean and per session, ms/token, TTFT median and session 0, `G`, `f` (demand rows only), `f_after_warmup`; the Window C cold arm's sessions 0–3 as the paired baseline (1.665 tok/s mean, 1,128 hot slots). State the hot-capacity matching: `p1` ran with the hot budget reduced by option C's scratch (3,048 MiB), so `p1`/`c`/`cpf` share one hot capacity, which is below Window C's; any remaining slot difference is named next to `G`. Prefetch: `advisory_rows`, `advisories`, `advisories_skipped`, and the tok/s difference `cpf − c` (state whether it is inside the per-session spread).
- **17.7 Per-token breakdown.** Task 16 Step 9's numbers next to `prof-summary.md`'s eager table (ms/token, GPU busy, GPU idle, D2H copies, stream syncs, launches, scheduler off-CPU). State which of the eager buckets (host ~340 ms, syncs ~220 ms, read+split ~182 ms, zero-copy gather ~172 ms) moved and by how much.
- **17.8 Known properties and open items.** The fail-stop latency (one forward without overlap, up to two with it, D15); the predictor limitation (previous-token routes for L+1 are usually already in RAM, so a low `advisory_rows` measures the predictor, not the mechanism); scratch VRAM (≈3.0 GiB out of the hot budget); anything recorded as BLOCKED, failed or not run in the ledger.

- [ ] **Step 3: §11 Phase 3b status**

Under "**Phase 3b — graph-mode three-tier streaming (open):**", add as its first bullet: "**Option C built and measured (§17):** decode under breakable CUDA graphs at bs 1 with the MoE in-graph (pinned-tier graph gather + fused `exl3_moe`), NVMe misses served by the io_uring thread with a bounded in-graph wait and fail-stop; next-layer advisory prefetch behind `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`." Mark the list items that §17 closes ("The in-graph RAM tier", "The §9.3 miss mechanism and its failure path", "CUDA graphs and graph gather for EXL3", "Profile the ~0.38 s/token ...") with "— done, §17", and leave the rest open.

- [ ] **Step 4: Commit and push**

```bash
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41
git add DSV41_REFERENCE.md
```
Commit `docs(dsv41): Phase 3b option C results (§17), correct the decode-breaks claim`; `git push shared dsv41`. Append `Task 17: complete`. The task's report lists: tok/s per arm vs the Window C baseline, TTFT, G, f, the per-token breakdown, the R4 verdict, and the four §9.3 acceptance results.

---

## Self-review

### Task list

| # | Title | Phase | Gate |
|---:|---|---|---|
| 1 | `exl3_moe` parity and capture probe (**GATE**) | P2 | — |
| 2 | Pre-flight baselines; framework post-capture counter re-baseline (merged) | P1 | — |
| 3 | EXL3 gate: breakable decode at bs 1; Engram lookup as an eager break | P1 | — |
| 4 | Warmup/capture forwards out of the trace and route recording | P1 | — |
| 5 | `--graphs`; R4 parity script (per-arm env, debug arm); P1 GPU smoke | P1 | — |
| 6 | Framework: graph gather over a pinned tier | P3 | Task 1 |
| 7 | Framework: pluggable slot table (host-use decorator), fail-stop checks, residency listeners | P3 | Task 1 |
| 8 | Merge `cc/moe-expert-plugins` (Tasks 6–7) | P3 | Task 1 |
| 9 | EXL3 in-graph MoE (`Exl3FusedMoE`, `_apply_graph`, gate) + GPU test | P3 | Task 1 |
| 10 | Option C tables and the EXL3 split in C++ | P4 | Task 1 |
| 11 | C++ slot LRU and request service (pumped), simulator | P4 | Task 1 |
| 12 | Service thread, pause handshake, watchdog | P4 | Task 1 |
| 13 | Device post/wait kernels + GPU test | P4 | Task 1 |
| 14 | Wire option C into graph decode; fail-stop; graph trace + GPU tests | P4 | Task 1 |
| 15 | Next-layer advisory prefetch + GPU test | P5 | Task 1 |
| 16 | GPU window: regression, R4, §9.3 acceptance, corpus arms, Nsight | P6 | Task 1 |
| 17 | Record (§17, corrections) | P6 | runs either way (short form on a Task 1 FAIL) |

### Coverage: rulings, phases, review findings → tasks

| Item | Where |
|---|---|
| R1' P1: gate, Engram break, recorder guards, driver graph options, eager MoE fallback | Tasks 2–5 (GPU check: Task 5 Step 5) |
| R1' P2 / R6: probe first, parity + capture with pointer tables, bar with reasoning, **hard gate** | Task 1 (D1, D7); Tasks 6–16 say "Blocked if Task 1 fails" |
| R1' P3: pinned-tier graph sources, host-slot indirection, in-graph gather, slot pointer tables, graph-safe `_apply_graph` with `exl3_moe` | Tasks 6, 9 (D4–D6) |
| R1' P3: residency host-boundary; GPU residency update / insert-on-miss / doorbell off | Task 6 (`pinned_tier_ok=False` for them); Tasks 3, 9 gate |
| R1' P4: device post of deduped RAM-missed ids to a host-mapped page | Task 13 (D10–D11) |
| R1' P4: C++ io_uring thread, slot allocation, split in C++, slab writes, device-visible map, completion | Tasks 10 (split), 11 (LRU, service), 12 (thread) |
| R1' P4: bounded single-block wait, fail-stop (fatal word + watchdog + scheduler hook), no copy-from-RAM fallback | Tasks 13 (wait), 12 (watchdog), 7 + 14 (hook), D15 |
| R1' P4: LRU safe for the thread, protected rows of the current layer | Tasks 11–12, 7, 14 (D12–D13) |
| R1' P4: doorbell patterns reused, Qwen doorbell unchanged | D14, D18; Task 16 Step 2 reruns the doorbell tests |
| R1' P5: predicted L+1 ids posted during L, off by default, one env var | Tasks 13, 11–12, 15 (D22) |
| R1' P6: CUDA tests, capture smoke, R4, forced-miss/timeout fail-stop, corpus arm prefetch off/on, graph-mode nsys, §17 + "~40 decode breaks" correction | Tasks 16–17 |
| R3: framework edits on `cc/moe-expert-plugins`, plain merges | Tasks 2, 6, 7, 8 |
| R4 | Task 5 (script), Task 16 Steps 4–5 (D3) |
| R5 | Task 16 Steps 8–9, Task 17 §17.6–17.7 |
| §9.3 acceptance (hit-path overhead; stream idle on a forced miss; timeout fail-stop without hang; capture under breakable + overlap) | Task 16 Steps 3, 3, 2 + 7, 4 + 6 |
| Review C1 (no approval waits; lock rule) | Global Constraints (`gpu-run.sh`, GPU lock rule, "No step stops to ask"); Task 1 Step 3; every GPU step |
| Review C2 (graph trace reads zeroed counters) | D23; Task 14 `_graph_rows`/`_trace_step` over the manager's registers; CPU test `test_the_trace_step_reads_the_manager_registers_before_they_are_lost` |
| Review C3 (eager reference refused under option C env) | Task 5 `run_env`/`engine_kwargs`/`_decode` per arm + CPU tests; Task 16 Step 4 |
| Review I1 (decision rules) | Task 2 Step 6, Task 8 Step 2 (conflict rule), D3 + Task 16 Step 5 (R4 rule), Task 16 Step 6 (ladder exhausted) |
| Review I2 (isolation arm) | D3; Task 5 debug arm; Task 16 Steps 4–5 |
| Review I3 (GPU verification per task) | Task 5 Step 5, Task 6 Step 5, Task 9 Step 6, Task 13 Step 6, Task 14 Step 5, Task 15 Step 5; Task 16 Step 2 reruns |
| Review I4 (merge check) | Task 8 Step 2 (framework-delta comparison for `memory_hook.py`/`model_runner.py`) |
| Review I5 (protect set in the test) | Task 11 `test_a_demand_is_read_split_and_published` uses `protect=[2, 5]`; `test_protected_experts_missing_from_ram_are_read_too` covers D12's recompute |
| Review I6 (broken string) | Task 11 `Exl3RamMissHost.stop` (`"\n"`), Task 12's replacement |
| Review I7 (`live_summary` input; `graph_step` lines) | Task 14 (`record_graph_step` line shape, `simulate` filter, tier_sim test), Task 16 Step 8 (`load_trace`) |
| Review I8 (quiesce race) | D12; Task 12 pause handshake + `test_concurrent_eager_use_and_advisories_never_share_a_slot`; Task 7 host-use decorator; Task 14 pause depth |
| Review I9 (comparable R5 numbers) | Task 11 demand/advisory row counters; D23 (f from demand rows); Task 16 Step 8 (hot slots logged, `p1` hot budget reduced by scratch); §17.6 |
| Review M1 (`num_active`) | Task 1 measures 6 and −1; Task 9 `NUM_ACTIVE` from the probe; D6 |
| Review M2 (per-layer temps) | Task 9 `shared_temps` |
| Review M3 (torn records) | D10–D11; Task 11 `read_record` seqlock; Task 13 `write_record` seq last; Task 11 overrun test |
| Review M4 (store fences) | Task 11 `_mm_sfence()` before map publish and before `demand_done`/status |
| Review M5 (dropped layer still computes) | Task 9 `route_tables` empties `expert_count` when `keep == 0` + test |
| Review M6 (1 ms timeout fires in warmup) | Task 14 `SGLANG_TEST_DSV41_RAM_MISS_FAULT`; Task 11 `delay_after_demands`; Task 16 Step 7 checks the phase |
| Review M7 (P3 miss has no host check) | Task 14 `_apply_graph` refusal + test; Task 9 closing paragraph |
| Review M8 (text vs code) | D22 (touch-only records), D10 (`after`), Task 9 Interfaces (`top_k`), D23 (`enabled`) |
| Review M9 (Task 10 too big) | split into Tasks 10, 11, 12 |
| Review M10 (`$ANA` undefined) | `<GPU>` defines `ANA`; every ssh snippet sets it |

### Type and name consistency

| Name | Defined | Used by |
|---|---|---|
| `fused_moe_slots(..., num_active)`, probe JSON `"num_active"`, `"verdict"` | Task 1 | Task 9 (`NUM_ACTIVE`), Task 16 Step 2 |
| `$ANA/gpu-run.sh` (exit 75 = lock held) | Task 1 Step 3 | every `<GPU>`/`<FW-GPU>` command; `prof-graph.sh` |
| `graph_parity.compare`, `run_env`, `engine_kwargs`, `_decode`; report keys `eager_vs_graph`, `graph_vs_debug`, `debug_vs_eager`, `eager_vs_eager` | Task 5 | Task 16 Steps 4–5, D3 |
| `graph_source_kind_of`, `require_graph_gather_support(..., pinned_tier_ok=)`, `PinnedTierRowBackend` (`host_rows`, `ram_miss`, `keep`, `translate`), `ExpertStreamer._graph_pinned_tier`, `ExpertStreamRequirements.graph_gather_host_source` | Task 6 | Tasks 9, 14 |
| `PinnedSlotTable` (+ `before_host_use`, `after_host_use`, `bind_capacity`), `_host_use` decorator, `register_fail_stop_check`, `add_residency_listener`, `_attach_formats` → `attach_hot_cache_manager(manager, streamer)` | Task 7 | Task 14 |
| `Exl3FusedMoE(tensors, slots, hidden, inter, top_k, device)`, `NUM_ACTIVE`, `shared_temps`, `slot_pointer_tables`, `route_tables`, `exl3_fused_moe_for` | Task 9 | Task 9 `_apply_graph` |
| `Exl3RamMissTables`, `exl3_ram_miss_tables`, C++ `Tables`/`RowReader`, `read_rows_once`, fixture `ram_miss_setup`/`same_bytes` | Task 10 | Tasks 11–15 |
| Page constants (`PAGE_BYTES` 10304, rings at 64/2112, record fields incl. `after` @12, seq written last), `COUNTERS` (14, = C++ `Counter`), `Exl3RamMissHost` (`pump`, `inject(..., delay_after_demands)`, `layer_rows`, `layer_advisory_rows`), `sim_post`, `sim_wait` | Task 11 | Tasks 12–15 (the kernel constants in Task 13 match) |
| `RamThread`; `Exl3RamMissHost.start_thread`, `pause`, `resume`, `threaded` | Task 12 | Tasks 13–15 |
| `Exl3RamMissDevice.post(row, planned, count, routes, next_row)` / `wait(row, planned, count, host_rows, keep, ram_miss)`, `STATE_WORDS` (= C++ `kPosted..kUnservedMisses`), `last_routes` | Task 13 | Tasks 14–15 |
| `NativePinnedSlotTable(service, layer_id, streamer_of)`, `Exl3RamMissRowBackend`, `Exl3RamMissService` (`before_host_use`/`after_host_use` with pause depth, `attach` keeps the manager, `_graph_rows`, `_trace_step`), `parse_fault` | Task 14 | Tasks 15–16 |
| `Exl3StreamTrace.enabled`, `record_graph_step`; line `"kind": "graph_step"` with `forward`, `layer: -1`, `tokens: 1`, `vram_miss`, `ram_miss` | Task 14 | `tier_sim.load_trace`/`live_summary`, `prof-graph.sh` |
| `prefetch_enabled()` | Task 14 (stub `False`), Task 15 (env) | Task 14 service |
| `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS`, `SGLANG_TEST_DSV41_RAM_MISS_FAULT`, `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH` | Tasks 14, 14, 15 | Tasks 14–16 |

Checked by hand: the record layout is identical in `read_record`/`exl3_ram_miss_sim_post` (Task 11) and `write_record` (Task 13), with the seq word written last on both writers; `routes`, `planned` and `host_rows` are int64 on every side; the slot map is int32 on host, kernels and thread; `sim_wait`'s codes (1, 2, 0, 3) match the wait kernel's rules; the C++ `Counter` enum and the Python `COUNTERS` tuple have the same 14 names in the same order.

### Things an implementer must check first (each named in its step)

- Whether `TVM_FFI_DLL_EXPORT_TYPED_FUNC` accepts `std::string`, and whether divix01's liburing has the `*_data64` helpers (Task 10 Step 4 gives the fallbacks).
- Extra manager attributes `_accumulate_registers` may read in Task 14's registers test (the step names the lines to read).

### Top risks

1. **R4 with the fused kernel.** `exl3_moe` applies the route weight after `down` with fp16 intermediates, unlike `exl3_moe_loop`, so debug-eager vs eager may miss 1e-3. The ruling makes this non-blocking when the bitwise capture gate (graph vs debug-eager) passes; §17 reports the numbers.
2. **Host-mapped memory ordering on sm_120.** Checked directly by Task 13's GPU step (`test_a_rewritten_slot_is_read_fresh`, capture/replay) as soon as the kernels exist, not only in the window.
3. **Memory and capture on the full model.** Scratch (~3.0 GiB) comes out of a 14 GiB hot budget on a 32 GB card, plus graph pools for up to four attention variants whose sm_120 capture is untested; Task 16 Step 6 has the ladder and a rule for when it is exhausted.
