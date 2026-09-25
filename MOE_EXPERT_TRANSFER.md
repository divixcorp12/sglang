# MoE expert transfer — paths, results, and how to run it

State as of `797be6f678` (`insert-on-miss-stage-b`, fast-forwarded into
`master` 2026-09-18) — all accepted
optimizations merged, plus insert-on-miss **stage 2 (DIRECT)** and the **fused
route planner** (`SGLANG_MOE_EXPERT_FUSED_PLAN=1`), both accepted 2026-09-18 on
acceptance-grade paired serving arms. Suite on that merge:
1 failed / 860 passed, the single failure being the long-characterised doorbell
copier flake (four distinct parametrisations, clears in isolation).

**NEXTN speculative decoding now runs on top of all of it** (2026-09-18, after
lifting the `SGLANG_MOE_GPU_RESIDENCY_UPDATE` speculation guard): **+9.6% over
the shipped config at production's 12,288 MB, 29/29 paired turns** — but only
with 3 draft tokens, and at 31.9 of 32 GB. See
[NEXTN arm](#nextn-speculative-decoding-arm-2026-09-18). Committed, not deployed.
Four draft tokens now start (per-layer slot floor) but **tie** three; keep 3. Prefill
staging was cut from 2.6 GB to 1.3 GB (**−1.5 GB peak VRAM**, see
[Prefill staging and VRAM](#prefill-staging-and-vram-2026-09-18)).

Geometry for scale: 48 layers, 512 experts/layer, top_k 10, one expert row =
2,764,800 B. About 480 routed rows/token if nothing is cached. The experiments
below ran a **10,240 MB** hot cache. Stage 1 splits that into 3,403 slots + 480
scratch + 48 pull rows; **stage 2 needs no scratch, so the same budget gives
3,883 slots + 0 scratch** — the +14.1% capacity that wins the campaign.
(Earlier production notes referenced a 12 GiB / 4,180-slot cache; that budget has
**not** been measured with these optimizations — see Backlog.)

---

## TL;DR — current best

| Configuration | tok/s | vs baseline |
|---|---:|---:|
| Baseline (no optimizations) | 13.96 | — |
| + insert-on-miss (stage 1, SCRATCH) | 15.98 | +14.2% |
| + overlap scheduling + hc_mix CTA cap | 16.645 | +19.2% |
| + insert-on-miss stage 2 (DIRECT) | 19.360 | +38.7% |
| **+ fused route planner — SHIP THIS** | **20.695** | **+48.2%** |
| same config at prod's 12,288 MB (NEXTN arm control) | 22.650 | +62.2% ¹ |
| + NEXTN, 3 draft tokens, 12,288 MB — committed, not deployed | 24.930 | +78.6% ¹ |
| + prefetch enabled on top (of stage 2's predecessor) | 15.93 | −4.3% ✗ |

¹ Different matrix and budget from the baseline row; compare the last two rows
with each other only (paired, same build, same day).

**Stage 2 corrupted a cache slot until `7b498893cd` (2026-09-19).** After a chunked
prompt whose last prefill chunk was under 1,024 tokens, the first verify copied
every miss into slot 0: garbled replies from the second token, and slot 0 kept
wrong rows afterwards. Every arm before that commit ran with the bug; speeds stand,
reply quality of those arms was not re-checked. See
[Stage-2 garbled replies](#stage-2-garbled-replies-2026-09-18--root-cause-and-fix).

**Nothing that worked came from prediction.** Prefetch is implemented, measured,
and deliberately left off. (NEXTN is speculation over *tokens*, not experts: it
moves slightly more bytes per token and wins by running fewer forwards.)

The governing constraint: at ~140 miss rows/token x 0.2239 ms/row, the PCIe path
is ~62% of each token — **stage 2 wins by moving 388.0 MB/token instead of
430.5.** But the link and the compute run **serialized on one stream**, so the
other ~38% is on the critical path too: the fused planner moves no bytes and
wins by cutting ~65 bookkeeping kernels per layer (see
[the decode trace](#nsight-decode-trace-2026-09-18--the-per-row-constant-measured-in-graph)).

Stage 2 accepted 2026-09-18 on an acceptance-grade paired arm: **19.360 vs
18.541 median, faster on 27 of 29 paired turns (p < 1e-5)**, mechanism confirmed
by +480 slots → +3.09 points of hit rate → 9.9% fewer miss rows. See
[Acceptance arm](#acceptance-arm-2026-09-18--the-shipping-decision).

Fused route planner accepted 2026-09-18 on a second acceptance-grade paired arm:
**20.695 vs 19.442 median, faster on 28 of 29 paired turns (p = 5.6e-8)**,
2.77 ms/token saved, hit rate unchanged. The shipped-config control reproduced
the day before's 19.360 within drift. See
[Fused route planner arm](#fused-route-planner-arm-2026-09-18).

---

## Prod server run

### Env vars this work added

| Variable | Default | What it does |
|---|---|---|
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` | `0` (OFF) | **The big one.** `0` OFF, `1` SCRATCH, `2` DIRECT. **Ship `2`.** A miss is paid once instead of every token. SCRATCH (+14.2%) copies the previous forward's misses out of the gather's scratch rows into slots at the boundary. DIRECT lands each miss straight in a victim slot chosen from a shortlist the previous boundary ranked, so the D2D hop disappears **and the 1.33 GB scratch region returns to the cache as +480 slots (+14.1% capacity at no VRAM cost)** — worth a further **+4.4%** over SCRATCH. Requires `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1` and `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`. |
| `SGLANG_MOE_EXPERT_FUSED_PLAN` | `False` | **Ship `1` (+6.4%).** Replaces the graph gather's generic route planning (~65 small torch ops per layer: sorts, scans, masks, counter adds) with one warp-sized JIT kernel. Pure compute: moves no bytes and leaves residency unchanged. BS1 only — `supports_fused_graph_routes` requires one token row with unique IDs and falls back to the generic planner otherwise. Tested against stage 2 by `test_the_fused_route_planner_drives_direct_exactly_like_the_generic_one`. |
| `SGLANG_MOE_HOT_INSERT_ON_MISS` | — | **Deprecated alias** for the above, kept because the enum preserves its 0/1 meaning. `1` selects SCRATCH. Prefer the `_STAGE` form. |
| `SGLANG_MOE_HOT_FUSED_INSERT` | `False` | Runs **stage 1's** boundary insert through a fused masked Triton kernel: one pass instead of two, and idle lanes move no bytes. Cuts that boundary ~45% (5.32 ms/token isolated, 4.28 ms/token in an arm). **Not on the shipping path** — stage 2 has no insert loop to fuse, and this flag is hard-refused on any stage but SCRATCH so an arm cannot report a fused number for an unfused run. Keep off unless you are deliberately running stage 1. |
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

This is the config that produced **20.695 tok/s**
(`servers/fplan-F/run-20260918-032147`, acceptance grade, 29 records, 0 errors),
except that the arm ran a **10,240 MB** hot cache where this block shows
production's 12,288. Paths are divix01's; change the four at the top for another host.

> **Two lines changed from the 16.645 config:**
> `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2` in place of
> `SGLANG_MOE_HOT_INSERT_ON_MISS=1` (19.360), then
> `SGLANG_MOE_EXPERT_FUSED_PLAN=1` (20.695). Everything else is identical.
> Stage 2 requires `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1` and
> `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=1`, both already present.
>
> Do **not** also set `SGLANG_MOE_HOT_FUSED_INSERT`. It applies to stage 1 only
> and is hard-refused on stage 2 — deliberately, so an arm cannot report a fused
> number for an unfused run. See [the kernel's status](#the-fused-insert-kernel-correct-tested-and-off-the-shipping-path).

> ### ⚠ Check `hot_update_decode_forwards` on every stage-2 arm
>
> `scripts/expert_prediction/run-shadow-server.sh` derives it:
>
> ```bash
> hot_update_decode_forwards=$([ "$insert_on_miss" != 0 ] && echo 1 || echo 4)   # correct
> hot_update_decode_forwards=$([ "$insert_on_miss" = 1 ] && echo 1 || echo 4)    # main-tip: WRONG for stage 2
> ```
>
> **Main-tip still tests `= 1`.** Under stage 2, `insert_on_miss=2`, so upstream's
> version yields **4** — a different residency update cadence, applied silently.
> Stage 2 requires 1.
>
> This is the dangerous shape: **the launcher has no test coverage.** Nothing in
> the 861-test suite reads it. Anyone who merges main and resolves this file by
> taking one side wholesale reintroduces the bug, and every arm still runs, still
> reports, and is quietly measuring a different cadence. It arrived once already
> bundled inside a textual conflict on a ~1000-character `printf` and nearly went
> through as "take theirs".
>
> **The check:** the run manifest records `hot_update_decode_forwards`. On any
> stage-2 arm it must read **1**, not 4.

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
  SGLANG_MOE_EXPERT_FUSED_PLAN=1 \
  SGLANG_MOE_EXPERT_COPY_BACKEND=dma \
  SGLANG_MOE_PINNED_HOST_MB=0 \
  \
  SGLANG_MOE_HOT_GPU_MB=12288 \
  SGLANG_MOE_HOT_DYNAMIC=1 \
  SGLANG_MOE_HOT_SEED="$EXPERT_SEED" \
  SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 \
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

**The doorbell and overlap scheduling are mutually exclusive.**
`model_runner.py:797-801` refuses at startup: `SGLANG_MOE_EXPERT_DOORBELL`
requires `--disable-overlap-schedule`. Since the shipping config removes that
flag to gain +4%, the doorbell (path 2 below) **cannot be enabled at all** — the
server will not start. Stage B refuses it independently, for the async-write
reason in path 2. Treat the doorbell as dead in this configuration unless
someone is prepared to give up overlap scheduling and re-measure.

**Sync audit, 2026-09-17: with overlap scheduling on there are ZERO CUDA
synchronisations per decode forward in this code.** The one per-forward
candidate, `doorbell_fail_stop_check(synchronize=True)` at `scheduler.py:4623`,
early-returns on `doorbell is None` and never touches the device. The eager
gather's ~8 syncs/layer are prefill-only here: `..._GRAPH_GATHER_SCRATCH_ROWS`
defaults to 0, so the graph gather is never undersized, and a nonzero value is
refused without speculative decoding. The `log_interval` trace boundary is
non-blocking by construction (`AsyncTelemetry`, background D2H). This is
CI-guarded: 10 registered tests run the decode path under
`torch.cuda.set_sync_debug_mode`, so a newly introduced sync fails a test rather
than silently costing throughput.

The tested wrapper is `scripts/expert_prediction/run-shadow-server.sh`, which adds
run provenance, the GPU lock and metrics files:

```bash
OVERLAP_SCHEDULE=1 HOT_GPU_MB=10240 HOT_INSERT_ON_MISS_STAGE=2 HOT_INSERT_ON_MISS_DECAY=0.98 FUSED_PLAN=1 \
PREFETCH_PREDICTOR= PREFETCH_PULL_MODE=off RUN_KIND=timed PREFETCH_RUN_DIR=/path/to/run \
  scripts/expert_prediction/run-shadow-server.sh myserver 31047 off
```

### What production runs today, and the delta

The production server is launched by
**`divix01:/data/models/slang/nvfp4-work/run-nvfp4-e16c-public.sh`** (port 7867,
`0.0.0.0`). That *script* is not in this repo, but the *code* it runs is. Since
2026-09-19 it runs the worktree **`prod-presets-20260919`**: `797be6f678` plus
`prod-presets-20260919.patch` staged, now at `a3134f39b9`, byte-identical to
this branch's code at that commit (everything below plus the host token
embedding, the stage-2 victim fix, and the presets module). The patch is
`git diff 797be6f678 a3134f39b9` of `python/`, `scripts/` and `test/`; its
effective offload env was verified identical to the previous script's. The
previous worktree `prod-5b91a98` (code at `5b91a9833c`, **has the stage-2
bug**) is untouched.

**As of 2026-09-18 the script carries the full winning config.** Four changes
against the pre-campaign script, each with a backup alongside:

| Setting | Before | Now | Why | Backup |
|---|---|---|---|---|
| `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE` | unset (off) | **`2`** | +14.2% for stage 1, a further +4.4% for stage 2's +480 slots | `.bak-pre-stage2-20260918` |
| `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` | `4` | **`1`** | Insert-on-miss needs a boundary every decode forward. **Stage 2 requires 1 — see the launcher warning above** | same |
| `--disable-overlap-schedule` | present | **removed** | +4.5% / +2.9%. Its absence *is* the optimization | same |
| `SGLANG_MOE_EXPERT_FUSED_PLAN` | unset | **`1`** | +6.4% | `.bak-pre-fusedplan-20260918` |
| NEXTN speculation | none | **`--speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3`** | +9.6%; 4 draft tokens tie with 3 | `.bak-pre-nextn-20260918` |
| `SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT` | unset | **`1`** | Draft 2.46 → 1.45 GB, accept length unchanged | `.bak-pre-nvfp4draft-20260918` |
| `SGLANG_MOE_HOT_GPU_MB` | `12288` | **`15360`** | +6.1% (13,312), +6.65% (14,336), then +6.6% (15,360 with the host embedding); 37k-token peak 31,511 MiB at chunk 4096 | `.bak-pre-stage2fix-20260919` |
| `SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING` | unset | **`1`** | Frees 1.18 GB at no decode cost; pays for the 15,360 MB cache | `.bak-pre-stage2fix-20260919` |
| worktree | `main-port-probe-7bc4eb` | **`prod-stage2fix-20260919`** | the rows above need this code, and stage 2 needs `7b498893cd` | same |

**Since 2026-09-19 the script selects these settings with `--moe-offload-preset
graph-gather`** (`python/sglang/srt/layers/moe/offload_presets.py`) instead of
setting 22 variables by hand; the table above is that preset's content. An
explicitly set variable still wins, and the server logs one `MoE offload preset`
line per value it fills. `--moe-offload-preset doorbell` is the doorbell copier
on the same base (insert-on-miss stage 1, overlap off, no speculative decoding),
smoke-tested for startup only: a coherent reply and its spin thread running on
core 71, not timed; it accepted stage 1 and the fused planner, so nothing was
dropped. Backup of the explicit script: `.bak-pre-presets-20260919`.

Validated after the stage-2 change: healthy in 201 s, `DIRECT`, 4,660 slots,
`scratch_bytes: 0`, coherent generation. **The fused-planner line was added after
that validation and has not been started under the prod script yet** — prod was
left down after the 2026-09-18 traces.

Keep everything else prod already has, including `--tool-call-parser auto`,
`SGLANG_QWEN4_PLE_FILE_READER=uring`, `SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY=1`,
`SGLANG_FILE_CACHE_MODEL_PATH`, `SGLANG_VLM_CACHE_SIZE_MB=0` and the PLE RSS budget
vars — none are in the benchmark launcher and all are production behaviour.

**Prod ran a 12,288 MB hot cache (14,336 MB from the 2026-09-18 update); every measurement before the NEXTN arm was at
10,240 MB.** That arm's control ran the prod config at 12,288 MB: 22.650 median.

**Prod is down; the script was updated 2026-09-19 (15,360 MB, host embedding,
stage-2 fix worktree) and has not been started.**
Every setting in it was measured in the benchmark launcher, which matches prod
on all GPU-relevant flags; prod adds `--tool-call-parser auto`,
`--mamba-radix-cache-strategy extra_buffer_lazy` and `--max-mamba-cache-size 1`
with `--disable-radix-cache`. Validate the first start with a ~37k-token prompt
while watching `nvidia-smi` (the benchmark launcher's probe left ~1,096 MiB).
To go back to 14,336 MB **without** the stage-2 fix:
`cp run-nvfp4-e16c-public.sh.bak-pre-stage2fix-20260919 run-nvfp4-e16c-public.sh`
(prefer stage 0 over that). Older:
`cp run-nvfp4-e16c-public.sh.bak-pre-nvfp4draft-20260918 run-nvfp4-e16c-public.sh`
(that copy still has the broken 4-token NEXTN lines; `.bak-pre-nextn-20260918`
is the last pre-speculation script that starts).

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
- Stage 2 and the fused planner were accepted at **10,240 MB**. At 12,288 MB the
  shipped config ran 22.650 (NEXTN arm control); stage 2's +480 slots over stage 1
  are still unmeasured at that budget.
- `--weight-loader-drop-cache-after-load` is in the benchmark launcher and costs
  129 s of startup re-reading weights. Drop it in production.
- Suite on the merged build is **1 failed / 860 passed**; the single failure is
  the doorbell copier flake (four parametrisations, clears 10/10 in isolation in
  two separate trees). The six earlier base failures are fixed by main-tip.
- **The fused planner is BS1-only by construction.** At more than one token row it
  falls back to the generic planner automatically, so it cannot break a larger
  batch — but its +6.4% does not carry over either.
- **Stage 2 has never run on production hardware under production load** — only
  on divix01's benchmark harness at BS1. It is accepted on a paired serving arm,
  which is the strongest evidence this campaign produced, not on production
  traffic.

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
| 14 | **Insert-on-miss stage 2 (DIRECT)** | `insert-on-miss-stage-b` @ `797be6f678` | **18.541 → 19.360 (+4.4%)** over stage 1 + fused insert kernel | `matrix/accept-20260917-205847`; see backlog row 14 |
| 21 | **Fused route planner** | flag only, on `797be6f678` (planner code predates the campaign) | **19.442 → 20.695 (+6.4%)**, 28/29 paired turns, p = 5.6e-8; 2.77 ms/token; hit rate unchanged | `matrix/fplan-20260918-032147` |
| 24 | Per-layer slot floor for stage 2 | `e1d227a4bd` | Enabler, no speed of its own: lets 4 draft tokens start at 12,288 MB. **N4 vs N3: tie** (11/29 turns, median −1.5%, p = 0.27 two-sided); keep 3 | `matrix/nextn-20260918-123932` |
| 25 | **Prefill staging in place, allocated once** | `844bb9d7a5` | **Peak VRAM 32,143 → 30,587 MiB** (37k-token prompt, NEXTN-3, 12,288 MB); prefill time unchanged | `matrix/memprobe-20260918-144144` |
| 23 | **NEXTN speculative decoding, 3 draft tokens** (guard lifted) | `master` (guard removal + test + launcher switch) | **22.650 → 24.930 (+9.6%)** at 12,288 MB, 29/29 paired turns, p = 1.9e-9; 3.94 ms/token; accept length 2.55. **Not deployed**: 31.9/32 GB, long-context OOM untested | `matrix/nextn-20260918-115808` |
| 27 | **Spend the freed VRAM on the hot cache: 13,312 MB** | flag only (`HOT_GPU_MB`), on `844bb9d7a5` | **25.208 → 27.081 (+6.1% paired median)** vs 12,288 MB, both NEXTN-3; 24/29 turns, p = 2.7e-4; 2.16 ms/token; hit rate 70.17 → 72.11%, 372 → 345 MB/token. 37k-token peak **31,647 MiB at chunk 4096 (safe); chunk 8192 OOMs** | `matrix/cache-20260918-161820`, `matrix/memprobe-20260918-171612` |
| 30 | **Draft (MTP) experts requantized FP8 → NVFP4 at load** | `46735b12b4` (`SGLANG_ENABLE_DRAFT_MOE_NVFP4_REQUANT=1`) | **Draft 2.46 → 1.45 GB; free after startup 3.05 → 4.04 GB.** Accept length 2.546 → 2.564, speed tie (26.096 → 26.198, 19/29 turns, p = 0.07), both at 13,312 MB. Enabler for a bigger cache | `matrix/draft-20260918-173157` |
| 32 | **Spend the draft's freed GB on the hot cache: 14,336 MB** (NVFP4 draft) | flag only, on `46735b12b4` | **26.877 → 28.101 (+6.65% paired median)** vs 13,312 MB, both NVFP4 draft; 27/29 turns, p = 8.1e-7; 2.28 ms/token; 5,048 → 5,437 slots, hit rate 72.67 → 74.20%, 335 → 321 MB/token. 37k-token peak **31,667 MiB at chunk 4096 (safe)** | `matrix/memprobe-d-20260918-180224`, `matrix/draftcache-20260918-180224` |
| 33 | **Token embedding in pinned host memory**, shared with the draft | `34a80a2dae` (`SGLANG_ENABLE_QWEN4_HOST_TOKEN_EMBEDDING=1`) | **Free after startup 3.05 → 4.23 GB** at 14,336 MB; speed tie (28.46 → 28.69, 17/29 turns, p = 0.46). The draft binds the target's host table, no second copy. Enabler | `matrix/hostembed-20260918-203429` |
| 34 | **Spend it on the hot cache: 15,360 MB** (host embedding) | flag only, on `34a80a2dae` | **28.46 → 29.30 (+6.6% paired ratio)** vs 14,336 MB GPU embedding; 24/29 turns, p = 5.5e-4; 2.36 ms/token; 5,825 slots, hit rate 73.98 → 76.13%, 330 → 296 MB/token. 37k-token peak **31,511 MiB at chunk 4096** | `matrix/memprobe-he-20260918-203429`, `matrix/hostembed-20260918-203429` |
| 35 | **Stage-2 victim shortlist kept full** (correctness) | `7b498893cd` | Fixes garbled replies after a short last prefill chunk and a slot-0 corruption that outlived the request. No speed claim | `matrix/garble-*`, [section](#stage-2-garbled-replies-2026-09-18--root-cause-and-fix) |

### Rejected / closed

| # | Experiment | Result | Why it failed |
|---|---|---|---|
| 5 | **Prefetch (LLaPor pull)** | combo-D 15.928 vs combo-B 16.645, **−4.3%** | Moves +21 rows/token over the link. Precision is fine (77.7%) but coverage is only 19% of misses, a correct pull saves **zero** bytes (same row, earlier), and wrong pulls are pure waste. Plus it starves insert-on-miss: insertions 159.5 → 133.9/token |
| 6 | Insert pull-covered experts | Recovers ~9 rows/token, still +11.5 worse than no-pull | Identity: **C − A = posted x (1 − precision)**. Needs ~100% precision to break even |
| 7 | **Speculative decoding / bigger N** | At α=0.7, N=4 costs **+35%** rows per accepted token | Consecutive tokens share experts (1.68x at N=8) but the shared ones are *already resident*; miss reuse is only 1.28x. Break-even α = 0.84–0.93. **Superseded by #23:** the row arithmetic held (NEXTN-3 moved ~4% more bytes/token) but the model left out the ~20 ms of per-forward non-transfer work, which speculation amortizes over 2.55 tokens |
| 8 | Task 3 compute-window pull placement | Was −4%, re-tested at **+2.9%** after the hc_mix fix | Rejection was confounded by the stall. Now moot: it relocates *pull* copies and prefetch ships off. Conflicts with `scoring-cost` in `serving/runtime.py` |
| 9 | Static per-layer pull gating | Est. 0.1–0.5 ms/token, below noise | Timing cancelled; code kept on `pull-layer-gating` @ `fa5b873a34` |
| 10 | Layer-0 token-id table | Best ~0.85–0.91 ms/token vs 1.41 ceiling | Below the 1 ms gate; no code |
| 11 | Cross-token end-of-step prefetch | 8.7 misses covered at 22% precision with 40 rows | Coverage and precision both too low |
| 12 | **Lossless NVFP4 compression** | Whole row ~0.95; **subparts differ enormously** (see below) | The compressible part is too small a share, and GPU decode eats the saving |
| 13 | Batch-prep reduction | ≤1.07 ms/step host work | No change needed once overlap is on |
| 31 | `--enable-torch-compile` (decode) | Smoke only | **Does not start.** Capture fails in `QSAIndexer`, which has no plain-PyTorch path; the breakable graph backend documents "No torch.compile". Needs a compile-safe indexer path and dynamo-disabled break markers, then an A/B; expected gain 0–3% (decode is PCIe-bound). Deferred. `matrix/tcsmoke-20260918-163830` |
| 36 | Hot cache past 15,360 MB (host embedding) | 16,384 MB **OOMs** (peak 32,103 MiB); 15,872 MB runs and is **+1.7%** over 15,360 (30.68 vs 29.88, 21/29 turns, p = 0.024) | Peak 32,029 MiB leaves **66 MiB**; not worth a production OOM. `matrix/memprobe-he-16384-20260918-213009`, `matrix/memprobe-he-15872-20260918-213009`, `matrix/hostembed-big-20260918-213009` |

### Backlog (not started / in flight)

| # | Item | Expected | Status |
|---|---|---|---|
| 14 | **Insert-on-miss stage 2 (DIRECT)** | **+4.4% measured** over stage 1 + fused kernel | **ACCEPTED, ship it.** 19.360 vs 18.541 median, faster on 27/29 paired turns (p < 1e-5). +480 slots at no VRAM cost, +3.09 points hit rate, 9.9% fewer miss rows, 42.5 MB/token less over PCIe |
| 19 | Fused masked insert kernel | ~45% off stage 1's boundary (5.32 ms isolated / 4.28 ms in-arm) | **Kept, default-off, stage-1 only.** Correct and tested, but stage 2 has no insert loop to fuse, so it is on no live path. Retained for a memory-constrained config where the 1.33 GB scratch is affordable but 3883 slots are not |
| 15 | 12 GiB budget | +741 slots; curve says −6.08 ms/token (independent estimate said −6) | **Measured incidentally, not as a paired arm:** the NEXTN arm's control (shipped config at 12,288 MB) ran 22.650, vs 20.695 at 10,240 MB in a different matrix — about −4.2 ms/token, below the curve's −6. Prod already runs 12,288 |
| 16 | Reduce the 2.76 MB row size | Directly attacks bytes/token | **Closed by arithmetic.** Row size is set by the model (NVFP4 4-bit weights + E4M3 blockscales); it is not a tunable. The only mechanism that shrinks it is compression, which is #12 |
| 17 | Fused scorer kernel | 1.7 ms D scoring cost | Only useful if prefetch is revived |
| 18 | Fix 8 known test failures | Suite to zero | In flight on `fix-known-test-failures` |
| 22 | **Fold stage 2's gather bookkeeping into the fused planner** | ≤ ~1.9–2.4 ms/token (trace upper bound, like the planner's 3.8–4.4 that realized 2.77) | Not started. `gather_destinations` (~16–23 kernels/layer) and `_commit_gather` (22) are what remains of the tail after #21. Supersedes the handoff's B2, whose ~1% was scoped against stage 1 |
| 26 | **Chunked prefill 8192** | Long-prompt prefill **−28% / −36%** (31k / 37k tokens) at +420 MiB peak over 4096, measured *before* pre-sizing | **Does not fit at 13,312 MB**: OOM on the first 31k prompt (160 MiB short in attention). Only viable at 12,288 MB or once more VRAM is freed. TTFT only; decode unaffected |
| 28 | Token embedding to pinned host memory | 1.18 GiB (~+450 slots), lossless; decode reads one 5 KB row per token | **Done: #33/#34.** The LM head (also 1.18 GiB) cannot move: every token multiplies against all of it |
| 29 | Prefill MoE in expert groups | Staging ~4x smaller again | Not started; needs the fused-MoE runner to split and re-sum per group. Only after #28 |
| 20 | **Offline blockscale re-coding** (precomputed codebook) | **+1.6%** lossless (6-bit) / **+3.2%** lossy (4-bit) | **DEFERRED — do not pick this up without asking the repo owner first.** Not blocked on evidence; it is an open decision about accuracy budget, and it is the owner's call to make |

### The pattern

Two sophisticated ideas failed for the same structural reason. **On a saturated
link, re-timing or re-grouping bytes does not help.** Prefetch *re-times* bytes
(−4.3%). Speculation *re-groups* bytes (−35% at realistic acceptance). Residency
policy *reduces* bytes (+14.2%).

The filter has a second branch, found by trace rather than theory: the link and
the compute run **serialized on one stream** (~62% / ~38% of a token), so
removing work from the compute side shortens the token as directly as removing
bytes does. The fused planner moves no bytes and won +6.4%. Ask of any new
proposal: **does it change bytes/token, or remove work from that one stream?**
Moving work *around* on it (re-timing) is the pattern that has failed.

### The link is at line rate, and the host caps it at Gen3

`2,764,800 B / 0.23 ms = 12.02 GB/s`, against a practical PCIe Gen3 x16 ceiling
of ~12.3 GB/s. **The transfers run at ~98% of line rate.** There is no packing,
merging, coalescing or batching win available on the wire — a transfer-size
sweep would only re-confirm this ceiling.

`nvidia-smi -q` reports, for the RTX 5090:

```
Device Max : 5      ← the card supports PCIe Gen5
Host Max   : 3      ← the motherboard negotiates Gen3
Current    : 3
```

The card is Gen5-capable and the **host board is Gen3**, confirmed by the owner
as a hardware property, not a BIOS or slot misconfiguration. At Gen4 the 36.8 ms
transfer term would be ~18 ms and at Gen5 ~9 ms — 60.1 → ~41 or ~32 ms/token,
i.e. ~24 or ~31 tok/s against today's 19.360. **The link is worth more than
every software optimization in this document combined, and it is unavailable.**
Record it so nobody re-derives the hope: on this host, `12.3 GB/s` is a wall.

### The idle window is real, fragmented, and still unexploited

Prefetch (#5, #6) and speculation (#7) were closed on *prediction quality*, not
on bandwidth availability. The distinction matters, because the bandwidth is
genuinely there:

- PCIe is busy **31.4 ms of a 51.65 ms token → ~61% duty cycle** (140.3 miss
  rows x 0.2239 ms). Stage 2 cut both the numerator and the token, so the *ratio*
  is unchanged — the link is still the binding term.
- The remaining **~20 ms is idle link time.**

But it is **not one contiguous block.** Per layer: 51.65/48 = 1.076 ms of wall
time, of which 2.92 misses x 0.2239 = 0.654 ms is transfer, leaving **~0.42 ms of
idle link per layer** — under two rows' worth, arriving 48 times per token in
small pieces.

That shape is why filling it is hard rather than merely unattempted. To use
layer L's idle window you must issue transfers for layer L+1, whose routes are
not yet known — so anything that fills it is *speculative* by construction, and
lands back on precision. Prefetch failed at `C − A = posted x (1 − precision)`;
it did not fail for lack of link capacity.

**What is closed vs what is open.** One *approach* to filling the window (one-layer-ahead
top-1 prediction) is closed on measurement. The window itself is not closed, and
larger batch is the other way at it: more tokens per forward amortizes each miss
row over more work without needing any prediction. Revisit from here, not from
prefetch.

### Nsight decode trace 2026-09-18 — the per-row constant, measured in-graph

Production config (stage 2, 12 GiB / 4,660 slots, overlap on), node-mode
capture of 20 s ≈ 456 decode steps. Report:
`cc-e16c-public/run-20260918-004721/profiles/report.nsys-rep` (see inventory).

**The segment kernel is pure link time.** Bucketing
`copy_expert_row_segments_gpu_kernel` by rows moved (21,888 calls = 48 x 456):

| rows | calls | µs/row |
|---:|---:|---:|
| 0 | 7,975 | — (0.5 µs total) |
| 1 | 5,410 | 228.5 |
| 2 | 3,467 | 226.1 |
| 4 | 1,206 | 225.0 |
| 7 | 282 | 224.7 |
| 10 | 40 | 224.5 |

Flat at 225–228 µs/row from 1 to 10 rows = 12.2 GB/s, with no per-call fixed
cost. This independently confirms the 0.2239 ms/row constant above from inside
the graph, and closes kernel-side tuning of this path: duration = misses x row
time. 36% of layer calls miss nothing.

**Transfer and compute are serialized.** The segment kernel and all compute run
on one stream; summed kernel time 14.67 s vs GPU-busy union 14.49 s. The idle
window above is structural, not incidental.

**The small-kernel tail is hot-cache bookkeeping — verified, and partly
removed.** Each layer runs the same 137 kernels (146 on the 12 full-attention
layers); gaps between them average only 0.35 µs. Between the router and the MoE
GEMM sit **107 bookkeeping kernels, ~130 µs/layer, 6.2 ms/token** — more than
twice the MoE runner (54 µs) and as much as the rest of the layer (134 µs).
Split by code order (boundaries +-7 kernels):

| block | kernels/layer | ms/token |
|---|---:|---:|
| generic route planning + counters (`if not fused:` branch) | ~62–69 | ~3.8–4.4 |
| stage 2 `gather_destinations` | ~16–23 | ~1.0–1.5 |
| stage 2 `_commit_gather` | 22 | ~0.9 |

The first block is what `SGLANG_MOE_EXPERT_FUSED_PLAN` replaces — which production
was **not** running (see [the arm](#fused-route-planner-arm-2026-09-18)); it
realized 2.77 ms/token of the 3.8–4.4 upper bound. The other two are backlog #22.
The ordered sequence for one layer is saved as
`run-20260918-004721/profiles/layer-kernel-sequence.txt`.

**Open:** 15.35 GB of plain H2D memcpy in the window (670 ops, ~34 MB/token)
is not the segment kernel and is unattributed — prefill inside the window or
PLE staging are the candidates.

**Gotchas from this capture (each cost a run):**

- **A graph-mode trace's kernel table excludes the graph body.** In the earlier
  graph-mode capture, 0 of 152,184 kernel rows carried a `graphId` and total
  kernel time was 693 ms in a 120 s window. Its top kernel, `index_copy`
  (`expert_stream.py`'s miss stitch, paired 1:1 with `_scatter_hot_rows_kernel`),
  was **prefill only** (1,440 = 48 x 6 tensors x 5 prefills). Use graph mode for
  timing, node mode for attribution.
- **The trace driver flatters hit rate.** It repeats one prompt at temperature 0,
  so the window ran at **84.4%** (1.56 misses per layer call) against 70.6% in
  the acceptance arm, while the cumulative counters said 47% (cold first
  request). Take miss counts from the acceptance arm, not from a trace.
- **SGLang `/health` runs a real 1-token generation** — 3+ s on a cold cache. A
  driver with `curl -m 3 /health` as its liveness test quits immediately.
- **`ssh -n` discards stdin**, so `cat <<EOF | ssh -n host 'cat > f'` writes an
  empty file. Pass content in the command argument (e.g. base64).

### Row size does not change prefetch economics

A natural and incorrect intuition: smaller rows would make wrong predictions
cheaper, so prefetch might become profitable. It does not, because **cost and
benefit both scale with row bytes.** A correct prediction moves `R/B` seconds off
the critical path; a wrong one adds `R/B` seconds of link occupancy. Halving `R`
halves both, and the break-even precision is unchanged. The same cancellation
applies to posting more candidates and to slot capacity (a wasted slot is `R`
bytes, and halving `R` doubles the slot count).

The only genuinely `R`-dependent term is *timeliness* — whether a posted row
completes inside the window before it is needed — and at ~1 posted row per layer
against a ~0.42 ms window that holds ~2 rows, we are not window-limited. So the
one effect that smaller rows would help is not the one that is binding.

Note the reverse for compression specifically: decompression cost scales with
*output* bytes, which do not shrink, so a wrongly prefetched compressed row costs
the same decode work as a correct one. Compression would make prefetch economics
slightly **worse**, not better.

### Compression: the subparts differ enormously, and it still does not pay

The headline "~0.95 ratio" hides the interesting structure. Measured per tensor
(`analysis/compression/tables.txt`, 7 layers x 12 experts):

| Tensor | Share of row | Entropy | zstd-1 ratio | Verdict |
|---|---:|---|---:|---|
| `w13_weight`, `w2_weight` | **~89%** | **3.945 bits of 4** (nibble H0) | **1.0000** (lz4 *expands* to 1.0039) | Incompressible |
| `w13_blockscale`, `w2_blockscale` | **~11%** | **~4.5 bits of 8**, only ~52 distinct byte values | **0.48–0.50** | **Compresses ~2x** |

So the answer to "are subparts compressible" is **yes, and they are at opposite
extremes.** The 4-bit weight codes are at 98.6% of maximum entropy — quantization
to NVFP4 is *designed* to use the full code range, and the sign bit measures a
full 1.000 bits. Nothing will compress them; cross-expert concatenation with a
128 MB window and long-distance matching still returns exactly 1.0000.

The blockscales are the opposite: FP8 E4M3 values that are always positive, drawn
from ~52 of 256 codes, and spatially correlated (order-1 entropy drops 4.5 → 4.1).
They halve.

**But 11% of the row halving is a 5.3% row saving** — the table's own
`scales-only-compressed row ratio` is 0.946–0.947, and an ideal static ANS coder
over both parts reaches only 0.933. Converting the best case: 6.6% of the 36.8 ms
transfer term is **~2.4 ms/token, ~+4%**.

That ceiling then has to pay for decode. CPU decompression is measured at
~0.7 ms/row, which at ~140 rows/token is ~98 ms — two orders of magnitude too
slow, so decode must be on the GPU, where it competes for SMs with MoE compute.
Decoding ~49 MB/token of blockscales at realistic GPU rANS throughput plausibly
costs 0.5–2.4 ms, i.e. **somewhere between "most of the win" and "all of it"** —
and the `_hc_mix` lesson says SM contention during in-flight copies is exactly
where this class of idea dies.

Closed, but closed with a number: the ceiling is ~+4% before decode cost, not
zero. If the link were ever faster, this would need re-deriving rather than
re-assuming.

### Arm gotcha: `iom-matrix.sh` does not set `OVERLAP_SCHEDULE`

`combo-matrix.sh` sets `OVERLAP_SCHEDULE=1`. **`iom-matrix.sh` predates that flag
and never sets it**, so anything built from that template defaults to 0 and the
launcher passes `--disable-overlap-schedule`. An arm built from it is *not* the
winning config, and nothing in its results says so.

This cost a full screening arm on 2026-09-17. The only symptom was a 3.3%
baseline gap against the 16.645 reference, which is easy to explain away as
protocol — and a protocol explanation turned out to be ~60% of it, which is the
most dangerous size for a wrong explanation to be.

**Always check `overlap_schedule` in the run manifest, not the script.** It is
now a field that has demonstrably gone wrong once.

Decomposition of that gap, produced by re-cutting the recorded reference rather
than re-measuring it (a technique worth reusing — it cost zero GPU):

| cut of `combo-b-p1` | median |
|---|---:|
| as recorded, 29 turns | 16.645 |
| same 4 sessions only | 16.643 |
| only turns ≤384 tokens | 16.257 |
| both cuts together | **16.324** |
| screening condition A (overlap **off**) | **16.105** |

Session count explains 0.002 — nothing. The 384-token cap explains 0.32, because
**long generations run faster** (17.99 vs 16.26 tok/s); this is the positive
length/throughput correlation, and it means short-turn protocols understate
absolute tok/s. The residual 0.219 is scheduler plus noise, below the 0.42
screening floor. A merge regression was **not** supported.

### Screening arm 2026-09-17 (overlap **off** — orderings valid, effect sizes not)

`matrix/fused-20260917-195153`, 16 turns, one run each, merged build `797be6f678`.

| | condition | tok/s | ms/token | miss rows/tok | slots | hit rate |
|---|---|---:|---:|---:|---:|---:|
| A | stage 1 unfused | 16.105 | 62.09 | 184.9 | 3403 | 66.21% |
| B | stage 1 + fused kernel | 17.296 | 57.82 | 191.5 | 3403 | 65.69% |
| C | **stage 2 (DIRECT)** | **18.157** | **55.08** | **172.5** | **3883** | **69.45%** |

**B − A = +1.190 (+7.4%). C − B = +0.861 (+5.0%).**

Read this arm carefully: all three conditions shared the wrong scheduler setting,
so the **orderings and the mechanism stand** while the **effect sizes on the
shipping config do not**. The mechanism is the durable part: C moves 7% fewer
miss rows than A and 10% fewer than B, with hit rate 3.8 points higher and
`scratch_bytes: 0`. On a saturated link that is the only thing that ever helps,
and it is visible directly rather than inferred from the clock.

**Stage 2 beat the fused kernel, inverting both predictions** — the team lead's
(−0.40) and the implementer's own paper analysis (net negative). Both were wrong
in the same direction by ~1.26 tok/s. The standing lesson: *when a measurement
misses the model by three times the effect size, the model is what failed.*

Measured noise floor, from two recorded passes of an identical config: **0.050
tok/s** apart (acceptance grade). B−A is ~24x that.

Kernel corroboration, two independent instruments: isolated benchmark **5.32
ms/token**, serving arm **4.28 ms/token**. 20% apart, arm lower — expected, since
the benchmark ran 64 experts/71 slots against the arm's 512/3403.

**C − B is not pure capacity.** C carries no boundary insert loop at all where B
still runs the fused one, so the delta is capacity gain + B's residual loop cost
− C's in-graph overhead: three terms, one measurement. It has not been
decomposed, and a decomposition that happens to reconcile it with the model
should be distrusted.

### Measurement facts worth not rediscovering

Each of these cost real time to establish. Size any future claim against them.

**Greedy decoding is not reproducible on this stack.** Same-config acceptance
repeats produce **0 of 29 identical completions** at temperature 0, with 3.7%
different token totals. Consequence, as a rule: *no serving arm can demonstrate
byte-equivalence of two code paths.* Controlled equivalence evidence lives in
unit tests that fix the routes; never in an arm.

**Noise floors, measured rather than assumed:**

| grade | floor |
|---|---:|
| same-config acceptance repeats | **0.050 tok/s** |
| off-pair acceptance | 0.117 tok/s |
| screening | **~0.42 tok/s** |

**tok/s correlates positively with generation length** (+0.68 to +0.79), so short
turns understate absolute throughput. *Direction matters when judging a result:*
at screening the slowest arm generated the most tokens, so the confound pushed
against the observed ordering and those deltas were conservative. At acceptance
B and C landed 0.4% apart in tokens, which is what makes the paired test clean.

**A fresh-process comparison is not a valid control here.** Two identically-built
unfused managers differ in 8 tensors once the caching allocator is dirty. A
warm-up control that ran clean did so only because it ran in a clean process — it
passed by testing nothing. Anything comparing built caches must dirty the
allocator first.

### Closed: the shared 10-row scratch pool

Recorded because it is attractive from the outside and will be re-proposed.
Killed on paper, two independent reasons:

1. **It is stage 2 plus an extra copy.** Freeing the scratch rows requires the
   insert to happen inside the gather, per layer — exactly where stage 2 already
   puts its work, needing the same victim selection with the same hazard check.
   So it pays stage 2's per-layer node cost *and* its victim machinery, then adds
   a D2D scratch->slot hop that stage 2 does not do at all, while saving 470 rows
   instead of 480. **Dominated on both axes.**
2. **The indexing does not permit it.** Scratch rows are not a separate pool:
   each layer's cache tensor is allocated `capacity + miss_rows`, and
   `scratch_columns = capacity + miss_columns` puts them *inside* that tensor. A
   shared pool needs a second tensor and a new addressing scheme through the
   whole gather and remap path.

### Artifact inventory (divix01)

Under `/data/models/slang/nvfp4-work/cc-expert-prediction/`:

| What | Where |
|---|---|
| Screening arm | `matrix/fused-20260917-195153`, `servers/fused-{A,B,C}` |
| Acceptance arm | `matrix/accept-20260917-205847`, `servers/accept-{B,C}` |
| Analysis scripts | `accept_report.py`, `accept_paired.py`, `baseline_gap.py`, `noisefloor.py`, `determinism.py`, `confound.py`, `fused_replay_overhead.py` |
| Compression study | `analysis/compression/` (`tables.txt`, `results_full.json`) |
| D2D constants bench | `iom-cuda-tests.sh` (carries a caveat block; `.pre-caveat` backup alongside) |
| Worktree | `wt-stageb` @ `797be6f678` |

Under `/data/models/slang/nvfp4-work/cc-e16c-public/` (production-config traces):

| What | Where |
|---|---|
| Decode trace, node mode, 20 s, loaded | `run-20260918-004721/profiles/report.nsys-rep` (laptop copy: `~/data/divix/traces/stage2-decode-nodemode-20260918-004721.nsys-rep`) |
| Graph-mode trace, 120 s (kernel table is prefill-only, see above) | `run-20260918-000659/profiles/report.nsys-rep` |
| Traced launcher (node mode, `--delay=360 --duration=20`) / graph-mode backup | `../run-nvfp4-e16c-public-TRACED.sh` / `.bak-graphmode` |
| Trace load driver (health `-m 60`, quits after 3 consecutive failures) | `drive.sh` |
| One layer's ordered kernel sequence (node-mode trace) | `run-20260918-004721/profiles/layer-kernel-sequence.txt` |

Fused route planner work, under `cc-expert-prediction/`:

| What | Where |
|---|---|
| Arm | `matrix/fplan-20260918-032147`, `servers/fplan-{F,C}/run-2026091803*`; the first C attempt is marked `ABORTED` |
| Arm script / paired analysis | `fplan-arm.sh` / `fplan_paired.py` |
| Worktree | `wt-fusedplan` @ `797be6f678` + the test and launcher change |
| CUDA test runner (`ALLOW_SHARED_GPU=1` waives the census check, keeps the lock) | `run-fusedplan-tests.sh`, logs `logs/cuda-tests-fusedplan-*` |

NEXTN work, under `cc-expert-prediction/`:

| What | Where |
|---|---|
| Arm | `matrix/nextn-20260918-115808`, `servers/nextn-{N,F}/run-20260918-1{15808,21147}`; `matrix/nextn-20260918-{104943,105436,110139}` are `ABORTED` |
| Arm script / paired analysis | `nextn-arm.sh` / `nextn_paired.py` (its "C"/"F" labels mean F/N) |
| Worktree | `wt-nextn` @ `797be6f678` + the guard removal, test and launcher switch (`wt-nextn.patch`) |
| CUDA test runner | `run-nextn-tests.sh`, logs `logs/cuda-tests-nextn-*` |
| N4 vs N3 arm / paired | `matrix/nextn-20260918-123932` / `nextn4_paired.py` (its "C"/"F" mean N3/N4) |
| Memory probe (waits for arms, retries a lost lock) | `mem-probe.sh`; `matrix/memprobe-20260918-{125301,140008,144144}` |
| Staging test runners (lock-queued) | `run-staging-green.sh`, `run-presize-tests.sh`; logs `logs/cuda-tests-{staging,presize}-*` |
| Cache-budget arm | `cache-arm.sh`; `matrix/cache-20260918-161820` (`-161749` is `ABORTED`) |
| 13,312 MB memory probe | `mem-probe-hot.sh` (HOT_MB, CHUNKS); `matrix/memprobe-20260918-171612` |
| torch.compile smoke | `torch-compile-smoke.sh`; `matrix/tcsmoke-20260918-163830` |
| NVFP4 draft arm | `draft-arm.sh`, worktree `wt-draftfp4` (+ `wt-draftfp4.patch`), `draft_paired.py`, `cache_counters.py`, `run_stubbed.py`; `matrix/draft-20260918-173157` |
| NVFP4 draft + 14,336 MB | `draftcache-chain.sh` (runs `mem-probe-draft.sh`, then `draftcache-arm.sh` only if the probe survives); `matrix/memprobe-d-20260918-180224`, `matrix/draftcache-20260918-180224`; `draftcache_paired.py` |

The `_hc_mix` persistent-kernel pattern is recorded as **checked and
inapplicable** in `python/sglang/kernels/ops/moe/expert_insert_rows.py`'s module
docstring: those lanes share nothing, so there is no device-wide barrier to need,
and the one-CTA-per-SM cap would only starve a kernel whose job is to saturate
HBM. Read it there before re-deriving it.

### Retired check: insertions/token is not a byte-identity test at serving level

A and B differ 3.5% in insertions/token, which looks like evidence the fused
kernel is not byte-identical. **It is not evidence.** Greedy runs on this stack
diverge run to run: two passes of an *identical* config produce 0/29 identical
turns with 3.7% different token totals at temperature 0. The check cannot
discriminate a code change from the stack's own nondeterminism. Byte-identity
rests on the unit tests, which control routes and compare every byte of every
cache tensor including untouched scratch rows.

### Fused route planner arm 2026-09-18

`matrix/fplan-20260918-032147`, script `fplan-arm.sh` (a copy of `accept-arm.sh`),
build `797be6f678` in `wt-fusedplan`, acceptance grade (8 sessions / 29 turns /
768 tokens), `OVERLAP_SCHEDULE=1`, 10,240 MB, one cold server per condition,
env verified from `/proc/<pid>/environ`, empty census before, between and after.
F ran first.

| | condition | median tok/s | mean | hit rate |
|---|---|---:|---:|---:|
| C | stage 2, generic planner (shipped) | 19.442 | 19.649 | 71.07% |
| **F** | **stage 2 + `SGLANG_MOE_EXPERT_FUSED_PLAN=1`** | **20.695** | **20.834** | 70.73% |

Paired: **F faster on 28/29 turns, one-sided sign test p = 5.6e-8**, median
+1.254 tok/s (ratio 1.060), **2.77 ms/token saved** (mean 2.94). C reproduced the
previous day's 19.360 within drift.

- **Mechanism is compute, not caching:** F's hit rate is 0.34 points *lower*.
- **Not a length artifact.** The servers generated different lengths on most
  turns (F +1.9% tokens) and long generations run faster, but the 5 identical-length
  turns give the same median, +1.254 (4/5 faster); within 10% length, 12/13.
- **Correctness:** `test_the_fused_route_planner_drives_direct_exactly_like_the_generic_one`
  runs twin stage-2 managers (generic vs fused) through 30 captured decode steps
  and requires identical residency, counters and slot bytes, with the generic
  planner mocked to fail. It kills a mutant that permutes the planner's
  rank-to-expert layout at step 0. Related suites: 298 passed, 0 failed.

**Gotcha: every earlier arm that says `fused_plan=1` ran the generic planner.**
`run-shadow-server.sh` printed `fused_plan=1` as a hard-coded literal in the run
manifest and startup banner (since `ee9c86f189`) and never exported
`SGLANG_MOE_EXPERT_FUSED_PLAN`, which defaults to `False`. The planner, reviewed
and approved in Stage A, was therefore never measured in serving until this arm,
and the claim "fused plan 1" in the prefetch handoff's baseline arms is false.
The launcher now takes `FUSED_PLAN=1`, exports the variable only when set, and
records the real value. Same lesson as the `OVERLAP_SCHEDULE` gotcha: **verify a
flag in the server's environment, not in what the harness says it did.**

### NEXTN speculative decoding arm 2026-09-18

**Why it was refused, and why the refusal was too broad.** `model_runner.py`
refused `SGLANG_MOE_GPU_RESIDENCY_UPDATE` with any speculative algorithm:
"verify commits do not reach the device clock". True, but harmless. The host
clock's `commit` (`expert_residency_clock.py`, fed by `eagle_worker_v2.py`'s
`on_verify_complete_cpu`) only corrects `tokens_since_boundary`, i.e. how far a
boundary decays scores. Boundaries fire on *forward* counts, which the device
counts identically for DECODE and VERIFY. Without the commit the device decays
each verify by its drafted positions (N) instead of the committed tokens (~α·N+1),
and its route counts include those same N positions, so drafted positions are
arguably the consistent unit. Draft forwards never reach it: `observe_forward`
returns on DRAFT, and the MTP layer's experts are FP8, so it has no streamers.
The guard was deleted (the DP-attention and batch-size checks stay).

**The binding constraint is stage 2's slot floor, not the clock.** A verify
gathers `draft_tokens × top_k` routes per layer (40 at 4 tokens), and DIRECT
requires every layer to hold **twice** its gather rows (`_init_insert_direct`'s
capacity guarantee), i.e. 80 slots. The per-layer split follows seed scores, so
the smallest layer sets the floor:

| Draft tokens | Budget | Outcome |
|---|---|---|
| 4 | 12,288 MB | refused at startup: layer 0 has **75** slots for 40 rows |
| 4 | 13,824 MB | starts (5,242 slots), then **OOM on the first prefill** (348 MiB staging alloc, 252 MiB free) |
| **3** | **12,288 MB** | **runs**: 30 rows, needs 60, layer 0 has 75; 31.9 of 32 GB in use |

**Arm:** `matrix/nextn-20260918-115808`, script `nextn-arm.sh` (a copy of
`fplan-arm.sh`), worktree `wt-nextn` = `797be6f678` + this change, acceptance
grade, `OVERLAP_SCHEDULE=1`, both conditions at 12,288 MB with stage 2 + fused
planner, speculation verified from `server_args` in each server log. N ran first.
Two earlier attempts are marked `ABORTED` (the 4-token failures above) and a third
was stopped for other GPU work after 5 turns.

| | condition | median tok/s | mean | decode-phase hit rate | H2D / token ² |
|---|---|---:|---:|---:|---:|
| F | shipped config (no speculation) | 22.650 | 22.640 | 74.44% | 360 MB |
| **N** | **F + NEXTN, 2 steps, top-k 1, 3 draft tokens** | **24.930** | **24.915** | 69.87% | 375 MB |

² Cumulative counters include warm-up (same for both); N's decode rows are under
the `speculative` phase key, not `decode`.

Paired: **N faster on 29/29 turns, one-sided sign test p = 1.9e-9**, median
+2.137 tok/s (ratio 1.096, min +0.22, max +4.11), **3.94 ms/token saved** (mean
4.12). Mean accept length 2.55. Length-matched turns agree: 5/5 identical-length
and 12/12 within 10% are faster.

- **Mechanism: fewer forwards, not fewer bytes.** N moves ~4% *more* bytes per
  token at a 4.6-point lower hit rate, as #7 predicted, but each verify pays the
  ~20 ms of serialized non-transfer work once for ~2.55 tokens.
- **The fused planner does not run on verify forwards** (one-token rows only); N
  wins despite losing it.
- **Correctness:** `test_a_speculative_verify_forward_keeps_direct_exact` replays
  a 2-token TARGET_VERIFY graph with repeated experts under DIRECT with the fused
  flag on, commits random accept counts, and checks per step: gathered bytes,
  every miss lands in a slot no token reads, distinct destinations, no routed
  resident evicted, byte-exact slots, and device `forwards` equal to the host
  clock's. Related suites: 299 passed.
- **Not deployed.** 31.9 of 32 GB with outputs up to 768 tokens; a long prefill
  (context is 40,000) is untested and the 13,824 MB attempt shows how little
  headroom there is. Run one ~30k-token request before shipping.

**Test gotcha: `envs.X.override()` has no `try/finally`** (`environ.py`). A test
that raises inside the `with` leaks the value into every later test in the
process; a failing draft of the test above left `SGLANG_MOE_EXPERT_FUSED_PLAN=true`
set and produced four unrelated-looking `test_expert_graph_gather` failures.

### Prefill staging and VRAM 2026-09-18

**Where the 32 GB goes** (NEXTN-3, 12,288 MB, from the server log):

| Item | Size |
|---|---:|
| Hot cache | 12.0 GiB |
| Target non-expert weights | 10.05 GB (linear attention 3.89 GiB, full attention 1.42, embed_tokens 1.18, lm_head 1.18, shared experts 0.44, other ~1.9; vision skipped by `--language-model-only`) |
| NEXTN draft layer (mostly its FP8 experts) | 2.46 GB |
| KV (target + draft) / Mamba + spec scratch / CUDA graphs | 1.0 / 0.44 / 0.49 GB |
| Free after startup | 4.08 GB, consumed at the prefill peak |

**The prefill peak was staging, and it does not scale with the chunk.** The
eager hot-cache gather stages every routed expert of a layer, and a 2048- or
4096-token chunk routes to 480–495 of 512. It held them **twice** — misses in a
`hot_cache_misses` buffer, then stitched into the kernel's buffer by the
`index_copy` at `expert_stream.py:1033` — and grew both a step at a time
(13–21 regrowths per prefill), leaving each outgrown buffer in the allocator's
cache: **2.6 GB of staging**. So chunk 2048 freed nothing (peak 32,145 vs 32,143
MiB) and prefilled 60% slower.

`844bb9d7a5`: misses take the kernel buffer's first rows so their copies land in
place (hits after them, routes remapped), and the buffer is allocated at a whole
layer's experts on first use. Probe: `mem-probe.sh`, NEXTN-3 at 12,288 MB, a
30,906- then a 37,274-token prompt, `nvidia-smi` sampled every 100 ms:

| Build | chunk | Peak MiB | 31k / 37k prefill | OOM |
|---|---:|---:|---:|---|
| before | 4096 | 32,143 | 35.0 / 38.9 s | no |
| before | 2048 | 32,145 | 56.6 / 61.9 s | no |
| in place | 4096 | 31,387 | 33.9 / 38.7 s | no |
| in place | 8192 | 31,807 | **24.3 / 24.7 s** | no |
| **in place + allocated once** | 4096 | **30,587** | 34.3 / 37.5 s | no |

- **Bigger chunks are faster prefill.** Each expert copied over PCIe serves
  every token of the chunk that routes to it, and prefill is ~90% transfer
  (~50 GB per 4096-token chunk at a 23% prefill hit rate). That is also why
  CUDA graphs for prefill are not worth building: they remove launch overhead
  from a 4.5 s chunk.
- **Tests:** `test_eager_hot_cache_gather_stages_a_layer_once` (old: 2.0 layers
  at peak) and `test_growing_prefill_gathers_do_not_regrow_staging` (old: 1.75
  layers) both fail on the old code. Device-to-device assembly now counts hit
  rows only (three test expectations updated). All MoE suites: 726 passed.
- **Gotcha: a peak-memory test must not index on the device.** Building the
  routed rows to check correctness (`tensors[name][compact]`) allocated 2x the
  staging and made the first version of the regrowth test fail identically on
  old and new code.
- **Gotcha: 10 test files in `test/registered/unit/layers/moe/` do not import**
  on divix01's venv (`pyarrow` has no `PyExtensionType`). A plain pytest run of
  the directory aborts at collection; use `--continue-on-collection-errors`.
- **Gotcha: another session shares `cc-gpu.lock`** with short back-to-back jobs.
  Check-then-launch loses the race (the launcher's `flock --nonblock` exits at
  once); queue with a blocking `flock -w` or retry a launch that leaves no log.

**Cache-budget arm** (`matrix/cache-20260918-161820`, `cache-arm.sh`, acceptance
grade, same build, NEXTN-3 + stage 2 + fused planner, B first):

| | budget | slots | median tok/s | mean | speculative-phase hit rate | H2D / token |
|---|---:|---:|---:|---:|---:|---:|
| N | 12,288 MB | 4,660 | 25.208 | 25.192 | 70.17% | 372 MB |
| **B** | **13,312 MB** | **5,048** | **27.081** | **26.514** | **72.11%** | **345 MB** |

Paired: **B faster on 24/29 turns, one-sided sign test p = 2.7e-4**, median
+1.615 tok/s (ratio 1.061, min -2.44, max +3.69), 2.16 ms/token saved. Accept
length 2.55 (N) vs 2.57 (B): the gain is bytes, not speculation.

- **Safe at chunk 4096, not at 8192.** `mem-probe-hot.sh` at 13,312 MB
  (`matrix/memprobe-20260918-171612`): the 37k-token prompt peaks at 31,647 of
  32,607 MiB (~960 MiB spare, 3.05 GB free after startup). At chunk 8192 the
  first 31k prompt OOMs in attention with our own process holding 31.17 GiB.

**NVFP4 draft experts** (`matrix/draft-20260918-173157`, `draft-arm.sh`, same
worktree `wt-draftfp4` for both, 13,312 MB, chunk 4096, D first):

| | draft experts | draft load | free after startup | accept length | median tok/s | hit rate |
|---|---|---:|---:|---:|---:|---:|
| B | FP8 128x128 block (as shipped) | 2.46 GB | 3.05 GB | 2.546 | 26.096 | 72.37% |
| **D** | **NVFP4, requantized at load** | **1.45 GB** | **4.04 GB** | **2.564** | **26.198** | 72.28% |

- **Why the draft was FP8:** the model card says the MTP module was copied
  byte-for-byte from Qwen's FP8 release; only the main model's routed experts
  were quantized (with calibrated scales). A normal forward never runs the MTP
  layer, so it could not be calibrated for W4A4; FP8 block scales need no
  calibration.
- **How:** `ModelOptMixedPrecisionConfig.get_quant_method` hands an
  `FP8_BLOCK_SCALES` MoE built in draft scope to the existing
  `ModelOptNvFp4OnlineFusedMoEMethod`, which dequantizes each FP8 expert and
  requantizes it to NVFP4 as it loads. Activation scale is per-tensor 1.0
  (per-token scales need the TRT-LLM or CuTe DSL MoE backends; we run
  `flashinfer_cutlass`). Accept length shows 1.0 is good enough.
- **The draft stays resident:** the offloader matches the quant method's exact
  class name `ModelOptNvFp4FusedMoEMethod`, so the online subclass is never
  streamed. `test_requantized_draft_experts_stay_on_the_gpu` guards this.
- **Output cannot change:** every draft token is verified by the target.
- **Spent on the cache** (`draftcache-chain.sh`: probe, then arm only if it
  survives). At 14,336 MB the 37k-token peak is 31,667 MiB at chunk 4096, the
  same ~940 MiB margin as 13,312 MB with the FP8 draft. The arm (E first):

  | | budget | slots | median tok/s | mean | hit rate | H2D / token | accept length |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | D | 13,312 MB | 5,048 | 26.877 | 26.696 | 72.67% | 335 MB | 2.567 |
  | **E** | **14,336 MB** | **5,437** | **28.101** | **28.360** | **74.20%** | **321 MB** | 2.604 |

  Paired: **E faster on 27/29 turns, p = 8.1e-7**, median +1.901 tok/s (ratio
  1.0665), 2.28 ms/token saved. Over the day the stack went 12,288 MB FP8 draft
  (25.21) → 13,312 MB (27.08) → 14,336 MB with the NVFP4 draft (28.10), each
  step a paired same-build arm; cross-arm drift is ~2%, so read the chain by
  its paired steps, not the absolute numbers.
- **Test gotcha:** `test_modelopt_nvfp4.py` imports `sglang.test.test_utils`,
  which pulls in `datasets` and dies on divix01's pyarrow; `run_stubbed.py`
  stubs that one module to run the file there.
- **Script gotcha:** the requested-bytes check needs `grep -qF` on a precomputed
  number; an unquoted pattern lost its quotes through `ssh` and the first launch
  (`-161749`, `ABORTED`) was stopped before it ran.

### Acceptance arm 2026-09-18 — the shipping decision

`matrix/accept-20260917-205847`, acceptance grade, `OVERLAP_SCHEDULE=1`,
8 sessions / 29 turns / 768 tokens, merged build `797be6f678`.

| | B stage 1 + fused | C stage 2 (DIRECT) |
|---|---:|---:|
| median tok/s | 18.541 | **19.360** |
| mean tok/s | 18.536 | 19.381 |
| ms/token | 53.94 | **51.65** |
| insertions / token | 155.5 | 139.9 |
| miss rows / token | 155.7 | **140.3** |
| requested rows / token | 479.8 | 477.9 |
| hit rate | 67.55% | **70.64%** |
| slots | 3403 | **3883** |
| H2D per token | 430.5 MB | **388.0 MB** |
| records / errors | 29 / 0 | 29 / 0 |

**C - B = +0.819 median (+4.4%), 16x the 0.050 tok/s acceptance noise floor.**

**The paired test is the real evidence.** Both arms ran the same 8 sessions and
the same 29 turns, so the comparison is turn-by-turn rather than
distribution-to-distribution: **C is faster on 27 of 29 paired turns**, median
paired delta +0.879, sd 0.518, worst case -0.646. Sign test p < 1e-5. Token
totals are 0.4% apart (9922 vs 9963), which removes the generation-length
confound that complicated the screening arm.

**The mechanism, and it closes exactly.** Stage 2 needs no scratch region, so
`scratch_bytes` goes 1.327 GB -> 0 and that memory becomes cache: 3403 -> 3883
slots, +14.1%. Check the byte accounting against the row accounting:

```
C: 140.3 miss rows x 2,764,800 B = 387.9 MB/token   (reported 388.0)
B: 155.7 miss rows x 2,764,800 B = 430.5 MB/token   (reported 430.5)
```

Rows and bytes agree independently, so the win is physical, not a clock artifact.

Partial-exposure check: 42.5 MB/token saved at the arm's ~8.0 GB/s average H2D
rate would be ~5.3 ms/token if fully exposed; observed gain is 2.29 ms/token, so
roughly 40% of the saved transfer sits on the critical path and the rest was
already overlapped. Treat that as order-of-magnitude agreement, not a tight
prediction — it is a single-rate model over an overlapped pipeline.

**Screening predicted this.** Screening C-B was +0.861; acceptance says +0.819.
The scheduler did not change the conclusion. That is a result, not an assumption
that was made in advance.

### The fused insert kernel: correct, tested, and off the shipping path

`SGLANG_MOE_HOT_FUSED_INSERT` cuts stage 1's boundary insert cost by ~45%
(5.32 ms/token isolated, 4.28 ms/token in a serving arm — two instruments, 20%
apart). It is byte-exact, covered by 31 tests, and **it optimises stage 1 only.**

Stage 2 has no scratch boundary at all, so there is no insert loop to fuse. The
flag hard-refuses any stage but SCRATCH — a deliberate guard, so that an arm can
never report a fused number for an unfused run.

**Decision (2026-09-18): kept, default-off, documented as stage-1 only.** It
costs nothing at runtime and keeps a tested optimisation available if stage 1 is
ever wanted for a memory-constrained config where the 1.33 GB scratch region is
affordable but 3883 slots are not. It is not on the shipping path, and it should
not be cited as part of the shipped result.

### Counter gotcha: hot-cache counters are cumulative and include warm-up

Per-token rates derived from the hot-cache counters must divide by **all** tokens
the server has processed, not just timed ones. The counters run from server start
and include the 5 warm-up requests. Dividing by timed tokens inflates every
per-token rate by roughly 15%.

**The available check that catches it instantly:** `requested_rows/token` has a
hard physical ceiling of `48 layers x top_k 10 = 480`. A screening report gave
547, 558 and 564 — impossible on their face, and nobody checked them against the
bound. With the correct denominator every arm lands at 470-480, a near-constant,
which is what that quantity must be.

Corrected screening figures (the tok/s medians never depended on this, and the
orderings and mechanism are unaffected): A 158.6, B 161.5, C 143.7
insertions/token.

### Backlog #20: offline blockscale re-coding — DEFERRED, ask first

> **Do not start this without consulting the repo owner.** It is deferred by
> decision, not by lack of evidence. The lossy variant changes model numerics,
> and that trade is the owner's to make, not an implementer's.

This is **not** the rejected #12. What made #12 non-viable was that entropy
decoding (zstd/rANS) has to run on the critical path, is sequential per stream,
and competes for SMs with MoE compute — plausibly 0.5–2.4 ms against a ~2.4 ms
saving. A **precomputed codebook** replaces entropy decoding with a LUT gather
from a 16- or 64-entry table resident in L1. Expanding ~49 MB/token of
blockscales is ~74 MB of HBM traffic, **~0.05 ms**. The cost that killed #12
disappears.

The enabling idea: the host arena and the cache slot need not hold the same
form. Store compact on the host, transfer compact, expand into the slot. Host
arena also drops ~3 GB of its 67.9 GB as a side effect.

| Scheme | Blockscale bits | Row saving | Throughput | Accuracy risk |
|---|---:|---:|---:|---|
| 6-bit codebook | 8 → 6 | 2.6% | **~+1.6%** | **None — lossless** |
| 4-bit codebook | 8 → 4 | 5.2% | **~+3.2%** | Needs a real eval |

**6 bits is the lossless floor**, verified not assumed: `results_full.json`
records distinct scale values both per layer (45–61) and per expert
(`scale_ctx/n_distinct`, 39–56). Both exceed 32, so 5 bits would be lossy.
Capturing the true 4.5-bit entropy needs variable-length coding, which is #12
again.

Blockscale share is ~10.5%, derived two independent ways: from NVFP4 format
arithmetic (1 E4M3 byte per 16 elements against 4-bit weights), and from the
archive's `scales-only-compressed row ratio 0.947` against a 0.49 scale ratio.

**Known complications, all real:**

1. **Swizzled layout.** The slot holds `w13_blockscale_swizzled`, read directly
   by the flashinfer_cutlass kernel. Expansion must reproduce the swizzle
   exactly. Static and computable at load, but this is where bugs would hide.
2. **6-bit is not byte-aligned** (4 values per 3 bytes) — awkward precisely in
   the *safe* variant. 4-bit packs cleanly at 2 per byte.
3. **The lossy variant re-quantizes the quantizer.** Blockscales carry each
   16-element block's dynamic range; coarsening to 16 codes adds relative error
   to every weight in the block, against E4M3's ~3% typical precision. This is
   the class of change that looks fine on perplexity and breaks on specific
   tasks. `scripts/fp8_accuracy/` has the harness.

**Ranking.** Below #19 (+7–8%, no accuracy risk) and #14 (+5%, no accuracy risk).
Its significance is categorical rather than numerical: it is the **only live idea
that reduces bytes on the wire**, and the wire is the 36.8 ms term. Everything
else in the queue attacks miss count or D2D traffic.

### Measurement gotcha: the recorded per-row constants are cache-flattered

The comment at `expert_residency_gpu.py:218-219` cites **0.007 ms/row** for
`index_copy_` and **0.054 ms/row** for `copy_expert_row_segments_gpu`. On a
production-shaped working set those are **0.0105 and 0.125 ms/row — 1.5x and
2.3x worse.**

The cause is the benchmark, not the kernels. `iom-cuda-tests.sh` allocates a
single `(80, 2_764_808)` uint8 tensor (221 MB) and copies rows 70–79 into rows
0–9, the same 20 rows, 200 reps. That touches ~55 MB against the RTX 5090's
**128 MB L2**, so it never leaves cache and measures L2 bandwidth rather than
HBM. Re-measured with 48 separate layer tensors (4.9 GiB working set, which is
production's real access pattern), the constants move as above.

Two consequences:

1. **Anywhere those numbers are used to reason about production cost, they
   understate it.** The design decision they justified — `index_copy_` over the
   segment kernel — still stands, and by a *wider* margin (0.0105 vs 0.125 is
   11.9x, against the recorded 7.7x).
2. **The general rule:** a microbenchmark that reuses one tensor across reps on
   this card is measuring L2. Size the working set past 128 MB, and sanity-check
   the implied bandwidth against the card's HBM spec — an impossible number
   (3.1 TB/s was the tell here) is the cheapest available detector.

### Stage-2 garbled replies 2026-09-18 — root cause and fix

**Symptom.** With NEXTN on, a reply garbled from its second token ("We", then
junk) and recovered within a few verifies. The first token came from prefill
and was right; the draft proposed sensibly; the target's first verify had
near-zero logit gaps.

**Trigger, pinned by exact-token prompts** (`garble_prompts.py`, chunk 4096):
a prompt with at least one earlier chunk and a **last chunk under 1,024
tokens** (1,020 garbled, 1,024 clean). Single-chunk prompts were clean. The
1,024 is `SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS`: moved to 2,048, the edge moved
with it (1,124 and 2,043 garbled, 2,048 clean).

**Bisection** (3 garbling prompts + 2 controls per server, all `matrix/garble-*`):

| Change from the prod config | Garbled? |
|---|---|
| NEXTN off | no |
| overlap off | yes |
| `SGLANG_ENABLE_JIT_DEEPGEMM=0` | yes (the target has no FP8-block linears) |
| decode graphs off (forces gather, residency update, stage 2 and fused plan off) | no |
| graphs on, graph gather and residency update off | no |
| graph gather on, residency update off | no |
| **stage 0, everything else as prod** | **no** |

**Root cause.** Stage 2 re-ranks each layer's victim shortlist at every graph
forward and, before the fix, dropped every resident routed since the last
boundary. A short last chunk takes no boundary but routes ~2,000 routes over
512 experts, i.e. nearly every resident, so the first verify saw an empty
shortlist. `gather_destinations` then gave each miss destination 0; the copy
kernel reads the device miss count, not `live`, so it wrote every miss into
**slot 0** and the remap pointed every missed route there. A boundary-sized
last chunk zeroes the route window first, which is why it was clean.

**Evidence.** A throwaway byte-level checker (`debug_residency_check.py`, patched
into the probe worktree only) compared every resident slot against its host row
after each forward: 0 bad through both prefill chunks and the prefill
boundary, then **42-48 of 48 layers with slot 0 holding another expert's rows**
from the first verify on, persisting across requests.

**Fix (`7b498893cd`).** Routed residents are ranked after unrouted ones instead
of dropped, so the shortlist is full whenever a layer holds `miss_rows` slots
(the 2x floor guarantees that). A miss that still finds no victim now counts in
`insertion_truncated`, which stage 2 never incremented; watch it stay 0. The
regression test fails on the old code exactly like production (wrong gathered
rows on the first replay after a short, all-routing prefill).

**Verified.** Stage 2 on: all five prompts clean; the checker stayed at 0 bad
slots over 72 forwards. 137 unit tests pass.

**Consequence for earlier arms.** Every arm before `7b498893cd` ran with the bug.
The speed comparisons stand (both sides of each pair had it), but a stale slot
0 could have read wrong weights for its expert on any later request; reply
quality of those arms was not re-checked. Stage 1 (SCRATCH) is unaffected: it
truncates instead of writing slot 0.

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
