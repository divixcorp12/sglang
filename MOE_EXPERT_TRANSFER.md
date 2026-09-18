# MoE expert transfer — paths, results, and how to run it

State as of `f2c6cd2cbc` on `codex/nvfp4-expert-stream-main` — all accepted
optimizations merged, plus the pre-existing test-failure fixes.

Geometry for scale: 48 layers, 512 experts/layer, top_k 10, one expert row =
2,764,800 B. About 480 routed rows/token if nothing is cached. The experiments
below ran a **10,240 MB** hot cache = 3,403 slots + 480 scratch + 48 pull rows.
(Earlier production notes referenced a 12 GiB / 4,180-slot cache; that budget has
**not** been measured with these optimizations — see Backlog.)

---

## TL;DR — current best

| Configuration | tok/s | vs baseline |
|---|---:|---:|
| Baseline (no optimizations) | 13.96 | — |
| + insert-on-miss | 15.98 | +14.2% |
| **+ overlap scheduling + hc_mix CTA cap (SHIP THIS)** | **16.645** | **+19.2%** |
| + prefetch enabled on top | 15.93 | −4.3% ✗ |

**Everything that worked reduces or re-schedules bytes; nothing that worked came
from prediction.** Prefetch is implemented, measured, and deliberately left off.

The governing constraint: at 159.8 miss rows/token x 0.23 ms/row, the PCIe path
is **~61% saturated** inside a 60.1 ms token. Only changes that reduce bytes/token
move the needle.

---

## Prod server run

### Env vars this work added

| Variable | Default | What it does |
|---|---|---|
| `SGLANG_MOE_HOT_INSERT_ON_MISS` | `False` | **The big one (+14.2%).** Each decode boundary promotes the layer's missed experts from their scratch rows into real cache slots, so a miss is paid once instead of every token. Requires `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1` (a residency boundary after every decode forward). Costs ~190 MiB peak. |
| `SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY` | `0.98` | Per-token decay on the insertion score that ranks eviction victims. Lower = forgets faster. |
| `SGLANG_OPT_HC_MIX_MAX_CTAS` | `128` | **On by default (+4.4% under load).** Caps the fused HC-mix grid below the SM count. Uncapped, the kernel launched one CTA per SM behind a device-wide barrier, so any overlapping expert copy froze it from 13 us to 186 us. `0` restores the old all-SM launch. On a GPU with <=128 SMs this is a no-op and the stall returns. |
| `SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE` | `off` | Prefetch pull arm: `off` / `count_zero` / `always`. **Leave off** — measured −4.3%. Requires `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR`. |
| `SGLANG_MOE_EXPERT_PREFETCH_FUSED_TOP1` | `True` | Selects pull candidates with the JIT top-1 kernel in all modes instead of writing a width-16 candidate bank (the bank cost 2.37 ms/token). Only matters when prefetch is on. |

Pre-existing flags the winning config depends on: `SGLANG_MOE_EXPERT_GRAPH_GATHER`,
`SGLANG_MOE_HOT_DYNAMIC`, `SGLANG_MOE_EXPERT_HOST_ARENA`, `SGLANG_MOE_GPU_RESIDENCY_UPDATE`,
`SGLANG_MOE_HOT_GPU_MB`.

Overlap scheduling is **not** an env var — it is the absence of the
`--disable-overlap-schedule` CLI flag. It is worth +4.5% / +2.9% and needs
graph-gather plus GPU residency update.

### Winning config

This is exactly what produced 16.645 tok/s (`servers/combo-b-p1/run-20260917-141132`).
Paths are divix01's; change the four at the top for another host.

```bash
# --- host-specific paths ---
WORK=/data/models/slang/nvfp4-work
MODEL=/mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47
EXPERT_CACHE=/mnt/nvme2/nvfp4-work/qwen38-nvfp4-expert-cache-v1
PLE_CACHE=/mnt/nvme2/ple-cache/qwen38-nvfp4
EXPERT_SEED=/data/models/slang/slang-dev-2bit/qwen3.8-flash-next-24gb-sglang/assets/expert_freq.pt

taskset -c 0-63 env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  PYTHONPATH="$WORK/flashinfer-0.6.18-cu130-overlay:$PWD/python" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  \
  SGLANG_MOE_EXPERT_STREAM=1 \
  SGLANG_MOE_EXPERT_FILE_DIR="$EXPERT_CACHE" \
  SGLANG_MOE_EXPERT_FILE_READER=uring_direct \
  SGLANG_MOE_EXPERT_HOST_ARENA=1 \
  SGLANG_MOE_EXPERT_GRAPH_GATHER=1 \
  SGLANG_MOE_EXPERT_COPY_BACKEND=dma \
  SGLANG_MOE_PINNED_HOST_MB=0 \
  \
  SGLANG_MOE_HOT_GPU_MB=10240 \
  SGLANG_MOE_HOT_DYNAMIC=1 \
  SGLANG_MOE_HOT_SEED="$EXPERT_SEED" \
  SGLANG_MOE_HOT_INSERT_ON_MISS=1 \
  SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY=0.98 \
  SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1 \
  SGLANG_MOE_HOT_DECAY_TOKENS=1 \
  SGLANG_MOE_HOT_PROMOTION_SIGMAS=0 \
  SGLANG_MOE_HOT_BENEFIT_RATIO=2 \
  SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS=0 \
  SGLANG_MOE_GPU_RESIDENCY_UPDATE=1 \
  SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS=64 \
  \
  SGLANG_OPT_HC_MIX_MAX_CTAS=128 \
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 \
  SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0 \
  \
  /data/models/slang/.venv/bin/sglang serve --model-type llm \
    --model-path "$MODEL" \
    --tp 1 \
    --fp4-gemm-backend flashinfer_cutlass \
    --moe-runner-backend flashinfer_cutlass \
    --moe-a2a-backend none \
    --cpu-offload-gb 80 \
    --ple-offload-embedding --ple-offload-backend file --ple-offload-dir "$PLE_CACHE" \
    --page-size 64 \
    --mamba-track-interval 64 \
    --mamba-ssm-dtype bfloat16 \
    --disable-radix-cache \
    --mamba-radix-cache-strategy extra_buffer_lazy \
    --max-mamba-cache-size 1 \
    --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
    --context-length 40000 --max-total-tokens 40000 \
    --max-running-requests 1 \
    --mem-fraction-static 0.95 \
    --language-model-only \
    --cuda-graph-backend-decode breakable \
    --cuda-graph-bs-decode 1 --cuda-graph-max-bs-decode 1 \
    --cuda-graph-backend-prefill disabled \
    --disable-flashinfer-autotune \
    --expert-distribution-recorder-mode per_pass \
    --reasoning-parser auto \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --host 127.0.0.1 --port 31047
```

**Note there is no `--disable-overlap-schedule`.** Its absence *is* the overlap
optimization. Adding that flag costs ~4%.

Deliberately unset, all measured and rejected: `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR`,
`..._PULL_MODE`, `SGLANG_MOE_EXPERT_DOORBELL`.

The tested wrapper is `scripts/expert_prediction/run-shadow-server.sh`, which adds
run provenance, the GPU lock and metrics files:

```bash
OVERLAP_SCHEDULE=1 HOT_GPU_MB=10240 HOT_INSERT_ON_MISS=1 HOT_INSERT_ON_MISS_DECAY=0.98 \
PREFETCH_PREDICTOR= PREFETCH_PULL_MODE=off RUN_KIND=timed PREFETCH_RUN_DIR=/path/to/run \
  scripts/expert_prediction/run-shadow-server.sh myserver 31047 off
```

### What production runs today, and the delta

The live production server is launched by
**`divix01:/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh`** (port 7867,
`0.0.0.0`). That *script* is not in this repo, but the *code* it runs is: the
worktree `main-port-probe-7bc4eb` is checked out on **this same branch**,
`codex/nvfp4-expert-stream-main`, at commit `d43c59b58a`.

**Prod is 158 commits behind, and the gap is the whole problem.** `d43c59b58a`
is an ancestor of `f2c6cd2cbc`, and it contains **zero occurrences of
`INSERT_ON_MISS`** — the flag does not exist in the code prod is running. Setting
it there changes nothing.

Therefore the env changes below are step 2, not step 1:

1. **Get the code to prod first.** Fetch and fast-forward the
   `main-port-probe-7bc4eb` worktree to the tip of
   `shared/codex/nvfp4-expert-stream-main` (do not pin a SHA from this document
   — it is written before the commit that contains it). That advances prod by 158
   commits — far more than our four optimizations — so it needs its own
   validation pass, not a flag flip. (The worktree's own `git status` reports
   "behind 13" against a stale remote-tracking ref; fetch first. The 158 figure
   is computed where both commits are present and is the one to trust.)
2. **Then change exactly three things:**

| Setting | Prod today | Should be | Why |
|---|---|---|---|
| `SGLANG_MOE_HOT_INSERT_ON_MISS` | unset (off) | **`1`** | +14.2%, the largest single win |
| `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` | `4` | **`1`** | Insert-on-miss needs a residency boundary after every decode forward; it refuses to initialise otherwise |
| `--disable-overlap-schedule` | **present** | **remove the flag** | +4.5% / +2.9%. Its absence *is* the optimization |

The hc_mix CTA cap needs no change once the code is there — `SGLANG_OPT_HC_MIX_MAX_CTAS`
defaults to 128. Note that removing `--disable-overlap-schedule` also depends on
code prod does not yet have: the overlap admission path came from
`host-decode-overhead` @ `90914e33ac`. All three changes are gated on step 1.

Keep everything else prod already has, including `--tool-call-parser auto`,
`SGLANG_QWEN4_PLE_FILE_READER=uring`, `SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY=1`,
`SGLANG_FILE_CACHE_MODEL_PATH`, `SGLANG_VLM_CACHE_SIZE_MB=0` and the PLE RSS budget
vars — none are in the benchmark launcher and all are production behaviour.

**Prod runs a 12,288 MB hot cache; every measurement here was at 10,240 MB.** The
direction should hold (more slots, fewer misses) but the magnitude is unverified
at that budget.

### Flag rationale (questions that come up)

- **`--expert-distribution-recorder-mode per_pass` is load-bearing, not
  observability.** The hot-cache manager registers `on_expert_distribution` as a
  forward observer on the global recorder (`model_runner.py:851-853`). With the
  mode unset the recorder no-ops, the manager never sees routing counts, and
  dynamic residency — including insert-on-miss — silently stops working. Do not
  remove it. A cheaper accumulator mode is unlikely to serve, since insert-on-miss
  needs per-forward granularity.
- **`--moe-a2a-backend none` is correct.** All-to-all exists for expert
  parallelism across GPUs; at `--tp 1`, `ep_size=1`, single GPU there is no
  all-to-all to do.
- **`--disable-flashinfer-autotune` is unevaluated.** It buys reproducible kernel
  selection and faster startup, and may cost kernel performance. It was set for
  benchmark determinism and never revisited. Prod carries it too. Good screening-arm
  candidate.
- **Radix cache is a deliberate memory trade, also unevaluated for this workload.**
  Prod uses `--disable-radix-cache` with `--max-mamba-cache-size 1`; the shadow
  launcher supports the alternative (`--mamba-radix-cache-strategy extra_buffer`,
  `--max-mamba-cache-size 8`). On a Mamba hybrid, prefix caching needs per-prefix
  state, and a size-8 cache costs memory that competes directly with the hot cache,
  i.e. more misses. Against that, multi-turn chat re-prefills the whole conversation
  every turn without it. **No TTFT was ever measured here** — all arms report decode
  tok/s — so this is an open question, not a settled one.
- **`--weight-loader-drop-cache-after-load` costs 129 s of startup** re-reading
  weights from NVMe. It exists so A/B arms start from an identical cold page cache.
  Prod carries it too; it is pure startup cost there and can be dropped.

### Caveats before shipping

- Measured only at **BS1** (`--max-running-requests 1`, decode graph captured for
  `bs=[1]`). Concurrency changes the memory split between KV and the hot cache.
- Measured at **10,240 MB** hot cache, not production's 12 GiB.
- `--weight-loader-drop-cache-after-load` is in the benchmark launcher and costs
  129 s of startup re-reading weights. Drop it in production.
- Suite has 8 known pre-existing failures unrelated to this work (fix in flight).

---

## Experiment results — consolidated

Compare arms only within the same matrix; cross-matrix drift is ~0.19 tok/s/day.
Protocol: 8 sessions / 29 turns / 768 tokens, cold server per arm, mode verified.

### Accepted / shipped

| # | Experiment | Branch @ commit | Result | Evidence |
|---|---|---|---|---|
| 1 | **Insert-on-miss residency** | `insert-on-miss` @ `679e1a7fb2` | **13.99 → 15.98 (+14.2%)**; misses 190.5 → 155.1 rows/token; decode H2D promotions 21.3 → 0; +190 MiB | ABBA + 8-arm parity; `servers/iom-*` |
| 2 | **Overlap scheduling** | `host-decode-overhead` @ `90914e33ac` | **+4.5% / +2.9%** over two pairs (~2.2–2.4 ms/token) | `servers/host-overlap-*` |
| 3 | **hc_mix CTA cap** | `hc-mix-stall` @ `732acac42f` | **D 14.089 → 14.713 (+4.4%)**; B flat (13.836 → 13.962) | `matrix/hcmix-20260917-113135`; microbench `analysis/hc-mix-stall/` |
| 4 | Top-1 prefetch selector | `scoring-cost` @ `3baf985d33` | C 13.33 → 13.81; D unchanged. Width-16 bank cost 2.37 ms, scorer 1.71 ms | `servers/scoring-*` |
| — | **Combined (1+2+3+4)** | `combined-iom` @ `f727a001f7` | **combo-B 16.645** | `servers/combo-b-p1/run-20260917-141132` |

### Rejected / closed

| # | Experiment | Result | Why it failed |
|---|---|---|---|
| 5 | **Prefetch (LLaPor pull)** | combo-D 15.928 vs combo-B 16.645, **−4.3%** | Moves +21 rows/token over the link. Precision is fine (77.7%) but coverage is only 19% of misses, a correct pull saves **zero** bytes (same row, earlier), and wrong pulls are pure waste. Plus it starves insert-on-miss: insertions 159.5 → 133.9/token |
| 6 | Insert pull-covered experts | Recovers ~9 rows/token, still +11.5 worse than no-pull | Identity: **C − A = posted x (1 − precision)**. Needs ~100% precision to break even |
| 7 | **Speculative decoding / bigger N** | At α=0.7, N=4 costs **+35%** rows per accepted token | Consecutive tokens share experts (1.68x at N=8) but the shared ones are *already resident*; miss reuse is only 1.28x. Break-even α = 0.84–0.93 |
| 8 | Task 3 compute-window pull placement | Was −4%, re-tested at **+2.9%** after the hc_mix fix | Rejection was confounded by the stall. Now moot: it relocates *pull* copies and prefetch ships off. Conflicts with `scoring-cost` in `serving/runtime.py` |
| 9 | Static per-layer pull gating | Est. 0.1–0.5 ms/token, below noise | Timing cancelled; code kept on `pull-layer-gating` @ `fa5b873a34` |
| 10 | Layer-0 token-id table | Best ~0.85–0.91 ms/token vs 1.41 ceiling | Below the 1 ms gate; no code |
| 11 | Cross-token end-of-step prefetch | 8.7 misses covered at 22% precision with 40 rows | Coverage and precision both too low |
| 12 | Lossless NVFP4 compression | ~0.95 ratio | Not viable |
| 13 | Batch-prep reduction | ≤1.07 ms/step host work | No change needed once overlap is on |

### Backlog (not started / in flight)

| # | Item | Expected | Status |
|---|---|---|---|
| 14 | **Insert-on-miss Stage B** | +480 slots (+14.1% cache) at **no VRAM cost**; curve says −4.02 ms/token, **+7.2% → ~17.8 tok/s** (upper estimate) | **In flight.** Design approved: boundary *proposes* victim shortlist, gather *disqualifies* entries this forward routes to. Risk: per-layer bookkeeping adds ~500–700 graph nodes/step |
| 15 | 12 GiB budget | +741 slots; curve says −6.08 ms/token (independent estimate said −6) | Not started; needs +2 GB VRAM |
| 16 | Reduce the 2.76 MB row size | Directly attacks bytes/token | Not started — the only untried lever in the winning category |
| 17 | Fused scorer kernel | 1.7 ms D scoring cost | Only useful if prefetch is revived |
| 18 | Fix 8 known test failures | Suite to zero | In flight on `fix-known-test-failures` |

### The pattern

Two sophisticated ideas failed for the same structural reason. **On a saturated
link, only reducing bytes helps.** Prefetch *re-times* bytes (−4.3%).
Speculation *re-groups* bytes (−35% at realistic acceptance). Residency policy
*reduces* bytes (+14.2%). Use this as the filter for any new proposal: does it
change bytes/token?

---

## Reference documents

| Document | Contains |
|---|---|
| [`docs/superpowers/plans/2026-09-16-prefetch-continuation-handoff.md`](docs/superpowers/plans/2026-09-16-prefetch-continuation-handoff.md) | **Primary tracking doc.** Full results tables, prefetch cost accounting, timing protocol, lessons, next actions |
| [`docs/superpowers/plans/2026-09-16-hicache-expert-prefetch-optimization-handoff.md`](docs/superpowers/plans/2026-09-16-hicache-expert-prefetch-optimization-handoff.md) | Optimization task list this campaign drew from |
| [`docs/superpowers/plans/2026-09-16-prefetch-throughput-recovery.md`](docs/superpowers/plans/2026-09-16-prefetch-throughput-recovery.md) | Task 3 origin and the throughput-recovery series |
| [`docs/superpowers/plans/2026-09-16-side-stream-expert-pull-handoff.md`](docs/superpowers/plans/2026-09-16-side-stream-expert-pull-handoff.md) | Side-stream pull design, staged |
| [`docs/superpowers/plans/2026-09-16-serving-handoff-stage-c-flags.md`](docs/superpowers/plans/2026-09-16-serving-handoff-stage-c-flags.md) | Serving flag semantics |
| [`docs/superpowers/plans/2026-09-16-prefetch-plan-completion-handoff.md`](docs/superpowers/plans/2026-09-16-prefetch-plan-completion-handoff.md) | Copy-cost measurements (0.2237 ms/row eager, 0.2240 replay) |
| [`docs/superpowers/plans/2026-09-15-moe-expert-prefetch-live.md`](docs/superpowers/plans/2026-09-15-moe-expert-prefetch-live.md) | Live prefetch bring-up |
| [`CLAUDE.md`](CLAUDE.md) | Nsight graph-trace rule; trace-analysis memory bounds |
| `.superpowers/sdd/2026-09-16-hicache-expert-prefetch-optimization-handoff/progress.md` | Detailed per-experiment ledger (git-ignored, local only) |
| `docs/superpowers/plans/2026-09-17-hc-mix-stall-and-task3-reversal.md` | hc_mix mechanism + Task 3 reversal (**on branch `task3-on-hcmix` only**) |

Analysis outputs on divix01, under `/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/`:
`hc-mix-stall/` (CTA sweep), `strategy/` (`static_curves.json` miss-vs-slots, `sim_policy.py`, `sim_pull.py`),
`cross-token/` (`spec_window.py` speculative-decoding study), `copy-compute-contention/`,
`step-tail/`, `l0-token-table/`, `compression/`. Timed runs are under `servers/<arm>/run-*/`
with `hot-cache.metrics.jsonl` (**cumulative — read the last record**).

---

## The shared primitive

All five paths bottom out in one header: `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh`.

| Entry point | Line | Shape |
|---|---|---|
| `copy_expert_rows_gpu` | :181 | one contiguous region; row stride from `source.stride(0)` |
| `copy_expert_row_segments_gpu` | :200 | a segment list (`{src, dst, bytes}` triples), so one launch gathers several disjoint tensors per expert |

`copy_expert_rows_gpu_kernel` (:144) is the segment kernel with a single
synthesized segment (:151-154).

Three properties drive everything downstream:

- **Row count is read on the device**, from `count[0]` (:160, :176), never from a
  host argument. The grid is fixed at `kExpertTransferGridSize`, so how many rows
  actually move is decided at replay time from a GPU tensor. This is what makes
  the kernel legal inside a captured CUDA graph, and why scratch is budgeted as a
  fixed 10 rows/layer rather than sized per token.
- **Work assignment adapts to row count** (:123-141). Fewer rows than warps: each
  row owns a contiguous range of whole warps, so adjacent host units stay in one
  warp. More rows than warps: each warp strides over rows. Either way all 32
  lanes of a warp sit on one row.
- **It is an in-kernel `ld.global.nc` gather**, not a copy-engine DMA
  (`load_expert_host_word_noncoherent`, :17). Measured 12.081 GB/s against the
  copy engine's 13.313 GB/s, both at the host link's PCIe gen3 x16 ceiling.

## 1. In-graph demand gather (miss path)

Pulls rows the router asked for but the hot cache does not hold, into scratch
rows, during decode.

- Python: `ExpertStreamer.enable_graph_gather` (`python/sglang/srt/layers/moe/expert_stream.py:570`), `_gather_graph` (`:650`); planning in `expert_row_plan.py`.
- CUDA: `copy_expert_row_segments_gpu_kernel` (`expert_cache_transfer.cuh:165`).
- Trigger: **inside CUDA graph replay**, current stream, device-only ops.
- Env: `SGLANG_MOE_EXPERT_GRAPH_GATHER`, scratch sized by `..._SCRATCH_ROWS`; needs `SGLANG_MOE_HOT_DYNAMIC` and `SGLANG_MOE_EXPERT_HOST_ARENA`.
- Cost: 0.2239 ms/row + 0.006 ms fixed.
- **Stage B (in flight) targets this path**: land the miss copy directly in its
  victim slot and return the 480 scratch rows to the cache.

## 2. Doorbell side-thread copier

A background CPU thread services posted requests, so a copy can start without
waiting for the graph to reach the miss.

- Python: `expert_doorbell.py`; posted via `InGraphRowBackend.post/resolve` (`expert_stream.py:670-673`).
- CUDA/C++: `expert_doorbell.cuh` — post (:222), wait (:292), drain (:357) kernels; host issuance by `cudaMemcpyAsync`/`cudaMemcpyBatchAsync` (:1040, :1057, :1070-1071); entry points `expert_doorbell_post` (:398), `expert_doorbell_resolve` (:426).
- Trigger: **dedicated CPU spin thread**, pinned by `SGLANG_MOE_EXPERT_DOORBELL_CPU`, decoupled from the replay that posted.
- Env: `SGLANG_MOE_EXPERT_DOORBELL` and `_CPU`/`_TIMEOUT_POLLS`/`_DEGRADED_POLLS`/`_DRAIN_POLLS`/`_MODE`/`_FATAL_WAIT_S`/`_PLAN_CAPACITY`. Requires path 1. **Disabled by default.**
- Cost: 0.209 ms/row (12.4 GiB/s), ~10% better than path 1 — it buys earlier starts, not more bandwidth.
- Semantics to respect: delivery is **all-or-nothing per tag**; `resolve` drains committed requests but a late copy *can* land after resolve. Safety rests on sticky disable plus a watchdog abort ~30 s later, not on the copy being cancelled.
- **This is why Stage B must keep refusing the doorbell** (`check_miss_plans`): a
  thread that writes a slot after a timed-out resolve would corrupt a row that
  Stage B has already committed to the mapping.
- **Gotcha — the CPU pin can fail silently.** `expert_doorbell.cuh:783-788`
  discards the return value of `pthread_setaffinity_np`, with no check, no log
  and no `failed_` store, unlike the fail-stop error paths on either side of it.
  If the requested core is outside the process's allowed set, POSIX returns
  `EINVAL` and the thread silently keeps its inherited mask. **This bites exactly
  under our own `taskset -c 0-63` recipe**, which excludes core 71: a doorbell
  benchmark run that way would measure an *unpinned* spin thread while appearing
  to request core 71. Always read back `spin_cpu` from `stats()`
  (`expert_doorbell.py:74`), which records `sched_getcpu()` right after the pin
  and so reports where the thread actually landed. Verified 2026-09-17:
  production and the historical doorbell runs record `spin_cpu: 71` under no
  restrictive mask, and `run-shadow-server.sh` never enables the doorbell at all,
  so no prefetch measurement is affected.

## 3. Async promotion (residency-driven, off-graph)

Promotes newly-hot experts into free or evicted slots, decided on the host
between forwards.

- Python: `ExpertHotCacheManager._update_residency` (`expert_hot_cache.py:1428`, called :1814); staging `ExpertHotCache.stage_reassign`; submission `submit_hot_cache_promotions` (:688) → `submit_expert_row_copy_batch` (`expert_transfer.py:563`) → `AsyncExpertTransferExecutor.submit` (:255, :272).
- CUDA: `ExpertRowCopyRoutes.copy_rows` (`expert_transfer.py:525`) → `copy_expert_rows_gpu` (`expert_cache_transfer.cuh:181`) or a DMA route; fallback `_copy_rows_fallback`/`_copy_rows_by_bytes` (:661, :698) via `index_copy_`.
- Trigger: **dedicated CUDA stream**, submitted from the host outside capture. Slot metadata is published only once copies complete (`_publish_completed_promotions`, `finish_promotions` :1405).
- Env: no dedicated on/off flag; route chosen by `SGLANG_MOE_EXPERT_COPY_BACKEND`.
- With insert-on-miss on, **decode-time H2D promotions drop to zero** (21.3 → 0):
  residency is maintained from misses instead.

## 4. In-graph residency promotion

A second promotion path, distinct from 3: the decode graph decays scores, decides
promotions on-device, and copies promoted rows inside the replay.

- Python: `GpuResidencyUpdater._copy_promotions` (`expert_residency_gpu.py:302`); decision in `decide_residency_on_device` (`expert_residency.py:350`). Insert-on-miss lives here as `_insert_misses`.
- CUDA: `copy_expert_row_segments_gpu` (`expert_cache_transfer.cuh:165`), or `index_copy_` fallback.
- Trigger: **inside CUDA graph replay**, fixed-shape device ops only, no host thread.
- Env: `SGLANG_MOE_GPU_RESIDENCY_UPDATE`; insert-on-miss via `SGLANG_MOE_HOT_INSERT_ON_MISS`
  (requires `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1`).

Paths 1 and 4 both run in-graph and both write hot-cache rows; 1 is miss-driven
into scratch, 4 is popularity-driven into real slots. Keeping straight which one
owns a given row is a live design constraint — and Stage B merges them.

## 5. Prefetch scoring and the pull path

`expert_prediction/` scores and ranks candidates. The pull arm
(`DedicatedPrefetchSlot` / `PrefetchPuller`, `serving/runtime.py:210-215`) copies a
predicted row into a dedicated per-layer pull row ahead of use.

- Env: `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR`, `..._PULL_MODE`, `..._BUDGET`,
  `..._CANDIDATES`, `..._FUSED_TOP1`, `..._SHADOW_RECALL`, `..._CALIBRATION`.
- **Measured and left off.** Precision 77.7%, but it covers only 19% of misses,
  a correct pull is byte-neutral, and the 48 reserved pull rows cost one
  residency slot per layer. See rows 5–6 of the results table.
- The code stays in tree behind its flags. Revisit only if precision approaches
  100%, the link gets much faster, or batch size rises.

---

## Notes

- `expert_cache_transfer.cuh` was read in full; Python line references come from a
  search pass and are jump targets, not quotations. Line numbers predate the
  merge to `f727a001f7` in places.
- Metrics files are **cumulative** — always read the last JSONL record.
- Timing protocol, drift figures and the two-tier screening rules are in the
  primary tracking doc.
