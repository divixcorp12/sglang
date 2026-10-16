# Prefill-merge / hot-cache calibration handoff (2026-10-16)

Goal (team lead, for the owner): recover the VRAM the DSpark arms give to the prefill spike and spend it on the
expert hot cache, with the KV pool pinned at **381,184 tokens** (owner's requirement). The work grew from chunking
the triton FlashMLA merge into a prefill-transient attribution and a startup calibration that sizes the hot cache
from a measured prefill peak.

## 1. Branches, worktrees, commits

- Branch **`prefill-merge-chunk`** on `origin` (based on `origin/split-model` 1ddfaae39e, with `ccb87893fa` merged in
  at a83378798b). Head at handoff: **6c7902bb7f** (this file adds one commit on top).
- Laptop worktree: `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-prefill-merge` (never edit the main checkout).
- divix01 worktrees (all detached; update with `git fetch origin && git checkout --detach origin/prefill-merge-chunk`,
  **never while a run uses it**):
  - `/data/models/slang/nvfp4-work/wt-prefill-merge` — first serving runs (mergechunk-r2 etc.), at a988627f06.
  - `/data/models/slang/nvfp4-work/wt-prefill-merge-dev` — served runs/smokes (expseg, expseg2, peak log, A0), at
    1d0e669e7f.
  - `/data/models/slang/nvfp4-work/wt-prefill-merge-test` — CPU tests and the calibration smokes (calib, calib2,
    **calib3 running**), at 6c7902bb7f.
- No other branch carries this work.

## 2. Flags added (all default off unless stated)

| Flag | Default | Where | What |
|---|---|---|---|
| `SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB` | 0 | environ.py; `kernels/ops/attention/flash_mla_sm120_triton.py` | triton FlashMLA LSE merge + sink in token chunks; chunk = budget / (3·H·D·4). Bitwise. |
| `SGLANG_SM120_TRITON_FLASHMLA_CHUNK_KERNELS` | False | same | (A2) with the budget set, also run SWA + extra kernels per token chunk into one output; chunk = budget / (16·H·D). Bitwise. |
| `SGLANG_OPT_EMPTY_CACHE_ON_LARGER_PREFILL` | False | `srt/mem_cache/prefill_cache_release.py` | empty_cache before each DSpark prefill larger than any before (11× in warm-up). Not GPU-tested; superseded by expandable segments. |
| `SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP` | False | `layers/moe/hot_cache_{vmm,retire,calibration,dummy_extend}.py`, expert_hot_cache.py, expert_residency_gpu.py, scheduler.py | design (C): VMM hot cache, calibrate before capture, trim/grow. |
| `SGLANG_DEBUG_PREFILL_PEAK_LOG` | False | `srt/debug_utils/prefill_peak_log.py` (+ `analysis/dsv41-drive/dspark/prefill_peak_report.py`) | per-call allocated/reserved of a DSpark prefill; one `prefill peak log {json}` line per prefill. |
| Arm `...-mergechunk` | — | `analysis/dsv41-drive/dspark/both_cpu_ab.py` | -hicache1 + merge budget 128, hot `hot_mb_for_margin()` = 11721 MiB, `--mem-fraction-static 0.8150 --max-total-tokens 381184`. |
| Arm `...-mergechunk-expseg` | — | same | -mergechunk + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, hot 13144 MiB (1035 slots), fraction 0.8623, KV pinned. Every other arm pins the alloc conf to "". |

Indexer budget used in calib3 is the existing `SGLANG_DSV41_TORCH_PREFILL_INDEXER_SCORE_BUDGET_MB` (arms: 128).

## 3. Results (numbers)

Basis: GPU free = **nvidia-smi `memory.free`** (used + free = 32,203 MiB; the 404 MiB driver reserve is NOT free).
The probes' first "left" figures were total − used and overstated free by 404 MiB.

- **Merge cause (confirmed)**: one-pass merge at 4096 tokens holds out + out_extra + 3 fp32 copies = 2,048 MiB vs
  flashinfer's 256 (+1.75 GiB). GPU: one-pass 2,054 MiB, chunked@64 583, chunked kernels@32 289.
- **Peak log (OFF arm, peaklog run)**: reserved peak 28,242 vs allocated peak 26,363 → **1,879 MiB stranded allocator
  reserve**; live prefill transient 1,459; persistent MoE staging 813.
- **expseg smoke** (expandable segments, 923 slots): reserve gap 438 MiB, memory.free at warm-up peak 2,040 (vs
  482-598), graphs OK, serve_compare clean prompts −0.26 ms/token (CI −0.39..−0.02); node-0 load dip 0.04 GiB with
  410 MiB swap-out (did not recur).
- **mergechunk-r2** (11721 MiB, no expseg): 63.9 ms/token vs A'' 70.9 / A 65.9 (one sample), 598 MiB free.
- **expseg2** (1035 slots, 13144 MiB): serve_compare vs mergechunk-r2, all 8 prompts clean: **−1.79 ms/token**
  (CI −4.51..−0.64); verify hot hit rate 41.71% → 43.61%; misses/fwd 76.8 → 73.2; 706 MiB free at peak; node-0 load
  min 0.14 GiB (above min watermark); VmPTE 236-290 MiB tracks RSS (VA reservations unpopulated, not the cause).
- **A0 attribution** (finer logger, expseg2 config), L2 at 4096 tokens: attention entry 26,767; wq_b +522 transient,
  leaves q 256; peak 27,670 in the core (out+out_extra+merge chunk ≈ 633 over q); MoE 431; wo_b 200; hc_post +160.
  556 MiB held across the layer loop (unattributed: hc_pre wrapper did not fire, fixed for calib3).
- **Step-3 calibration smoke** (calib/): dummy EXTEND works on the speculative target; lowest free 1,018 MiB, margin
  331, 0 retired; calibration vs real 4096: +46 alloc, +152 reserved. Bugs: (A) post-capture "free now" wrong because
  capture released cached reserve → fixed with projection; (B) per-tensor VMM rounding cost ~140 MiB and KV fell to
  372,480 < pin → VMM only for tensors ≥ 32 granules, KV-pin refusal added.
- **calib2** (A2 + merge 32 + calibration with capture estimate, 1035 slots): GPU tests 66 passed EXIT=0. KV held
  381,184; 0 trim; lowest free 1,525.9, margin 298.3, capture estimate 25.3 (TOO SMALL: measured capture ~130-150);
  post-capture projected 1,097.9; memory.free at peak **876** (vs 706 expseg2). Prefill transient 1,259 (A0 1,459);
  attention 704 (A0 903) — now bound by the torch indexer (~q + 384 at budget 128) → A1 back in. Prefill 4096 prompt
  19.70 s (A2+A3+logger) vs 19.35 s (no A2/A3, no logger): not a clean A/B.

## 4. Calibration design (C) as built — and the decisions behind it

Decisions: owner chose (a′) **calibrate before graph capture, every launch**, via **(C) VMM tail release** with
**(b) post-capture check**; routing **(i)** (RAM-resident routes, stateless); **grow approved** (VA for a derived
upper bound); post-warm-up check **log-only**; **A1 dropped then re-added** (indexer now binds); **A4 skipped** (not
bitwise); KV pinned 381,184 is a hard requirement.

Modules (`python/sglang/srt/layers/moe/`):
- `hot_cache_vmm.py` — `VmmHotCacheBacking(device, specs, backed_rows)`: per-tensor VA reservation for the upper bound,
  one handle per 2 MiB granule, `set_backed_rows(n)` maps/unmaps (addresses never move), tensors < 32 granules on
  torch.empty. `tail_release` = pure granule math.
- `hot_cache_retire.py` — **single capacity authority** `set_hot_capacity(manager, keep)` (`retire_hot_slots` =
  shrink-only). Lowers/raises cache, policy, updater (capacity/slot_valid/scratch_columns), row planner, RAM-miss row
  backend hot_capacity, `_spec_hot` + `_bind_spec_scores`, re-pushes hot sets, re-ranks victims, maps/unmaps VMM.
  Refuses after capture (updater enabled), when copy engine armed, without VMM, with scratch rows, below 2×victim
  lanes, above upper/bank, and on any unlisted `*capacity*` attribute (`CAPACITY_HOLDERS` / `EXEMPT_CAPACITY_ATTRS`).
- `hot_cache_dummy_extend.py` — own one-sequence EXTEND ForwardBatch (`_dummy_run` forces TARGET_VERIFY on spec
  runners), `run(n)`; `dummy_verify_forward` = `BaseRunner._dummy_run` on a stand-in (capture estimate);
  `forget_graph_path_state` drops cached `_exl3_fused_moe`.
- `hot_cache_calibration.py` — `maybe_calibrate_hot_cache(model_runner)`; runs shapes (chunk/2, +1, chunk, chunk)
  then the verify estimate; `lowest_free`, `margin`, `required = margin + capture_estimate`, `slot_delta`,
  `plan_capacity` (grow emptiest / trim fullest); `check_kv_pin`; `require_hot_cache_headroom` (post-capture
  projection: free − max(0, peak-run reserve − reserved now), raises `HotCacheHeadroomShort`);
  `watch_first_full_prefill` (one-shot real-vs-calibration transient report); `calibration_context` + guards.
- Guards inside `calibration_context`: `ExpertStreamer.record_routes` skips the policy; `EngramFileTable.lookup`
  returns zeros; `Exl3MoEMethod.apply` → `calibration_routes` (RAM-tier residents for prefill-sized route sets, VRAM hot
  residents for verify-sized ones); model called directly (no recorder); KV writes to slot 0.
- Upper bound: `from_model` passes `upper_capacity = slot_limits[layer] or num_experts` (inclusive pinned-tier
  limit); `GpuResidencyUpdater.max_capacity` = max upper.
- Hooks in `managers/scheduler.py` `init_model_worker`: `self.hot_cache_calibration = maybe_calibrate_hot_cache(...)`
  between `init_all_attention_backends()` and `init_all_cuda_graphs()`; `require_hot_cache_headroom("after capture",
  ...)` after. `model_runner.py` untouched.
- Tests (CPU, `test/registered/unit/mem_cache/`): `test_hot_cache_calibration.py`, `test_hot_cache_retire.py`
  (census over real classes' source + inverse grow/shrink census), `test_hot_cache_vmm_math.py`,
  `test_prefill_cache_release.py`; `test/registered/unit/model_executor/test_prefill_peak_log.py`;
  `test/registered/unit/kernels/test_flash_mla_sm120_merge_chunking.py`. GPU: `test/manual/dsv41/test_hot_cache_vmm_gpu.py`
  + the merge-chunk file. Last CPU run: 150 passed, 56 skipped, EXIT=0 (at 75eac8b26f; 6c7902bb7f adds logger wrappers,
  5 passed).

## 5. In flight: calib3

- Started 2026-10-16 ~00:45 CDT, **pid 1948706** (`/mnt/nvme1/prefill-merge/calib3/run.sh`, worktree
  wt-prefill-merge-test at 6c7902bb7f). Log: `/mnt/nvme1/prefill-merge/calib3/run.log`; server log and peaks:
  `/mnt/nvme1/prefill-merge/calib3/run/{server.log,peaks.json,gpu-mem.csv}`.
- Does: GPU unit tests (merge-chunk + VMM with partial backing; stops if they fail), then -mergechunk-expseg with
  calibration (grow/trim, warm-up-shape runs, capture estimate), A1 indexer budget 32, A2 + merge budget 32, peak
  logger with mHC wrappers, one 4096 prompt.
- Read: `grep -E "hot cache calibration|hot cache capacity|hot cache VMM|KvPinMissed|HotCacheHeadroomShort|Traceback"
  run/server.log` → slots before/after and slot_delta, capture estimate, post-capture projection, real-vs-calibration
  transient delta; `peaks.json` → `kv_tokens` (must be 381184), `run_free_mib`; run.log tail = prefill_peak_report
  (per-op; look for `L*.hc_combine`, `L*.hc_post_with_combine` and `L*` layer entries for the 556 MiB).
- **A refusal (HotCacheHeadroomShort / KvPinMissed) is a valid result** — report it, don't patch in place. The
  capture estimate is known-low (25 vs ~130 MiB), so a grow could run into the post-capture backstop.

## 6. Open issues / next steps (in order)

1. Read calib3 (above).
2. Capture estimate: eager verify peak misses the graph pool's static buffers and the draft graph (0.04-0.25 GB).
   Options: add the capture runners' static buffer sizes (computable from shapes) and a draft estimate via the DSpark
   worker's draft runner.
3. Clean A2 prefill-time A/B, logger off, budgets 32/64/128 → derive the budget from ms per launch vs MiB.
4. wq_b's 266 MiB temporary (522 transient − 256 output) — next binding op after A1/A2.
5. 556 MiB held across layers — attribute from calib3's mHC wrappers; B1 (in-place hc_post) only if two 4-stream
   copies are live together.
6. Post-warm-up check: log-only (lead decision); the one-shot `watch_first_full_prefill` already logs the 4096
   comparison.
7. Then a served A/B of the calibrated arm vs expseg2 (serve_compare against mergechunk-r2/expseg2).

## 7. Protocol notes and pitfalls

- Code reaches divix01 only by push + worktree; `PYTHONPATH=$PWD/python`; interpreter `/data/models/slang/.venv/bin/python`;
  print `sglang.__file__`; read `${PIPESTATUS[0]}`. CPU tests: `CUDA_VISIBLE_DEVICES="" taskset -c 56-63`. GPU tests:
  `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`, `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`.
- Served runs: wait for the quiet box (no compute apps, no server/pytest/spec_margin_capture/prefill_peak_probe) AND both
  locks free (`flock -n` probe); `prelaunch_evict.py --hicache-gib 12.67` before launch; outputs under `/mnt/nvme1` or
  `ram-prefetch/`; TMPDIR=/mnt/nvme1/prefill-merge/tmp; detach with `setsid nohup`; kill by PID only.
- Serving A/B = `spec_margin_capture.py --arm ... --no-verify-split` with `SGLANG_DSV41_CPU_EXPERTS_SPLIT=<split A>`;
  compare with `wt-stream-variance/analysis/dsv41-drive/dspark/serve_compare.py` (clean prompts; nvme3n1 stalls flag).
- **Never in-server allocator snapshots** (`/start_profile MEM`): the dump OOM-killed the scheduler on node 0 twice and
  once a root swapoff. The -mem arms leave node 0 ~0.14 GiB free under mlockall.
- `prefill_peak_probe.py` gives up when its server exits (4bada59bd0); before that fix a refused start held both locks
  for 17 min.
- Don't capture a waiter pid with `pgrep -f <script>` from an ssh command line — it matches the ssh shell itself.
- The repo's edit hook blocks scripted edits of tracked files; use Edit/Write. Throwaway mutants on divix01 are the
  exception (resend the command), always reverted.
- `test/registered/unit/mem_cache/` as a directory has collection errors from other files; name files explicitly.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` +
  `Claude-Session: https://claude.ai/code/session_01KvT26Vzsfr4r2yo73YRcMR`. Stage by name; never commit uv.lock,
  .python-version or RAM_PREFETCH_*_HANDOFF.md.
