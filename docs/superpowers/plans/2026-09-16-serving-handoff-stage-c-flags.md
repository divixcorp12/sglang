# Serving handoff — running the NVFP4 server with Stage A/C features enabled

Written 2026-09-16 against branch `master`, **unpushed**. Supersedes nothing;
read alongside `2026-09-16-side-stream-expert-pull-handoff.md`, whose Stage C this operationalises.

**Read section 1 before enabling anything.** One of these flags is ready, one changes a recorded
metric's meaning the moment you turn it on, and one is not ready to run at all.

## 1. Readiness, stated honestly

| Flag | State | Safe to serve with? |
|---|---|---|
| `SGLANG_MOE_EXPERT_FUSED_PLAN=1` | Stage A, reviewed, approved, 237/239 (2 pre-existing doorbell failures) | **Yes** |
| `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR=llapor\|apex` | Scoring only, shadow. Works, but see §4 | **Yes, with the §4 caveat** |
| `SGLANG_MOE_EXPERT_PREFETCH_PULL=1` | Stage C complete, reviewed, APPROVED — but **never run on a live server** | **Not yet — see below** |

**UPDATE — the four blockers below are now CLOSED.** Stage C4 (`5203f18828`, `f867e73e8b`) landed the
demand-path row-skip, the `posted` call site, the manager registration and warmup purge, and the
real-path re-derivation; its review returned **APPROVE** with zero CRITICAL/HIGH/MEDIUM and two LOW.
The plan's worked numbers were reproduced through a real `ExpertStreamer` + `ExpertHotCache` +
`PrefetchPuller`: **useful = 2 host rows, wrong = 3, multi-token overlap = 1 physical row** against
route-level `covered = 2`.

**What still stops it being "yes":**

- **It has never run on a live server with a real checkpoint.** Everything above is unit and
  integration testing. That is the remaining gate, and it belongs to Stage D.
- **The fused path is not exercised end-to-end through a live BS1 `_gather_graph` capture** — the
  production shape and the Stage A win. The kernel lane is proven bit-for-bit against the CPU oracle
  across capture + 6 replays with a changing prediction, and the host wiring is identical in the fused
  and non-fused branches, so the reviewer judged the residual risk small but **not zero**.
- **`join_target`'s remap is now redundant but still live** (LOW). Traced by hand as byte-identical
  idempotence today, not a live corruption — but if the two slot derivations ever diverge it becomes
  the "two writers, both plausible" class, and nothing currently pins them to each other.
- **`prefetch_slot=0` is a dummy when prefetch is disabled** (LOW), safe only because
  `prefetch_count` is forced to 0 in lockstep. `-1` is the convention elsewhere in that same file for
  "no destination" and would fail loudly instead.

Enable it for a measured Stage D arm, not for production serving.

**The original blockers, retained for the record** — none of these was speculative:

1. **Bullet 3's demand-path row-skip is not implemented.** The ordinary demand copy in `_gather_graph`
   still writes a row the side pull already delivered. Nothing reads it, so it is not a correctness
   bug — it is wasted work, and it **double-counts** once the delivery counters are consumed. Closing
   it requires excluding a row from the fused planner's output while preserving fixed CUDA-graph
   shapes, in `expert_route_plan.py`. Deferred to its own dispatch deliberately rather than attempted
   blind.
2. **The batched review across C1/C2/C3 has not run.** Every prior stage in this plan changed
   materially at review.
3. **A counter defect was found and fixed only at the end**, and the fix has not been reviewed:
   `covered` sums *routes*, while a capacity-1 pull delivers one physical *row* per posted forward, so
   `covered + wasted` overcounted rows on multi-token overlap. A fourth `posted` counter was added.
   **The plan's 2-vs-3 host-row tests pass under both the correct and the incorrect counter**, because
   that scenario is single-token — so those tests are not evidence the counters are right.
4. The pull additionally requires the hot cache to be allocated with the trailing
   `DedicatedPrefetchSlot` row, which only exists when `SGLANG_MOE_EXPERT_PREFETCH_PULL=1` is also set
   at allocation time. With the flag off the shape is byte-identical to before. `PrefetchPuller`
   raises `ValueError` at setup if the row is missing rather than corrupting anything — that guard is
   working as designed, and if you see it, the flag is on somewhere the allocation did not see it.

**Review outcome.** The batched review across C1/C2/C3 returned **REQUEST CHANGES** with one HIGH and
two MEDIUM findings, all fixed at `eede477278`. C1's ranking and slot arithmetic were reviewed and
came out **clean**. One question the reviewer could not settle without hardware — whether
`argsort(descending=True, stable=True)` breaks ties toward the lower index — was settled directly on
divix01: scores `[5,3,5,1,5]` give order `[0,2,4,1,3]`, so ties break toward the **lower** index and
C1's tie-break rule is correct.

### 🔴 Known uncorrected bias in the side-pull counters

`PullDeliveryStats.counts` is **never reset**, and nothing purges capture-time warmup from it. The
sibling hot-cache counters have `discard_graph_capture_routes` for exactly this — dummy routes execute
once while recording and land in the counts like a real forward — but `PrefetchPuller` is built from a
raw `hot_caches` mapping and has no connection to `ExpertHotCacheManager`, so there is no purge hook
to wire into yet. This was an accepted decision, not an oversight: building that connection now would
guess a shape the wiring dispatch has not chosen.

**Scope of the bias, corrected — it is a ONE-TIME STARTUP COST, not a growing one.** An earlier draft
of this section said "permanent and cumulative across a server lifetime." **That was wrong and is
withdrawn.** `discard_graph_capture_routes()` has exactly one production call site,
`model_runner.py:1319` inside `init_cuda_graphs()`, which runs **once per model runner at process
startup** — not per forward, and no other path recaptures graphs later. So the bias is a fixed
constant added at startup, bounded by (captured decode-graph shapes × target layers with the pull
enabled), and it becomes proportionally *less* significant the longer a server runs, rather than more.

"Permanent" is accurate — nothing ever purges it. "Cumulative" was not.

**RESOLVED at `5203f18828`.** Stage C4 added `register_prefetch_puller` and extended
`discard_graph_capture_routes` to zero a registered puller's `PullDeliveryStats`, so the startup
contamination is now **purged to zero** the same way it already was for `_graph_counters`,
`_registers` and `residency_policies.pending_counts`.

The bias's structure was pinned before deciding to fix it, rather than fixed on a hunch. Every
captured shape runs the target forward **three times** — two warmup replays plus one replay inside the
`torch.cuda.graph()` context — confirmed identically in `full_cuda_graph_backend.py:140`/`:179`,
`breakable_cuda_graph_backend.py:136` and `tc_piecewise_cuda_graph_backend.py:240`. Each replay runs
the full forward including the prefetch taps. So:

> bias per target layer ≈ **3 × N_captured_decode_shapes** samples, once, at startup.

`N_captured_decode_shapes` derives from `get_batch_sizes_to_capture(...)`, filtered by attn-tp
alignment and clamped to `max_running_requests`. **That factor was deliberately not traced to a number
without a real launch** — the formula is measured, the shape count is not, and it is reported that way
rather than invented. Prefill shapes contribute only if a prefill forward is small enough to take the
graph-gather path, which is uncommon for real chunked-prefill batches, so decode dominates.

## 2. Baseline: what is running today

The live script is `divix01:/data/models/slang/nvfp4-work/run-nvfp4-expert-dynamic-hot10g.sh`,
serving on **port 7867**. As of this writing **port 7867 is not listening** — the server is down, so
starting it does not interrupt anything. If it is up when you read this, it is the user's Hermes
backend: you may stop it for experiments without asking, but **notify the user**, and relaunch and
verify health afterwards.

Its current expert-path environment, verbatim from the script:

```
SGLANG_MOE_EXPERT_STREAM=1
SGLANG_MOE_EXPERT_FILE_DIR="$expert_cache"
SGLANG_MOE_EXPERT_FILE_READER=uring_direct
SGLANG_MOE_HOT_GPU_MB=10240
SGLANG_MOE_PINNED_HOST_MB=0
SGLANG_MOE_EXPERT_HOST_ARENA=1
SGLANG_MOE_EXPERT_GRAPH_GATHER=1
SGLANG_MOE_HOT_SEED="$expert_seed"
SGLANG_MOE_HOT_DYNAMIC=1
SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS=8
SGLANG_MOE_HOT_DECAY_TOKENS=16
SGLANG_MOE_HOT_PROMOTION_SIGMAS=0
SGLANG_MOE_HOT_BENEFIT_RATIO=4
SGLANG_MOE_HOT_LOG_INTERVAL=100
SGLANG_MOE_HOT_METRICS_FILE="$metrics"
SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0
SGLANG_MOE_EXPERT_COPY_BACKEND=dma
```

`SGLANG_MOE_EXPERT_GRAPH_GATHER=1` is already on, which is the precondition for everything below.
`SGLANG_MOE_EXPERT_DOORBELL` is **off** and should stay off for these arms — it is Stage E, and two of
its tests fail at HEAD for reasons unrelated to this work.

## 3. Arm B — fused planner only (recommended first)

Copy the script, add **one line**:

```
SGLANG_MOE_EXPERT_FUSED_PLAN=1
```

This is the only change. It replaces ~25-30 tensor ops in route planning with a single 32-thread
warp kernel. It is reviewed, approved, and independently green.

**Its precondition is structural and worth knowing:** the fused planner requires within-row expert
IDs to be unique. That holds because `phy2log` is a function (physical→logical), so distinct logical
experts have disjoint physical replica sets, and `torch.topk` returns distinct indices — duplicates
cannot arise from real routing. `supports_fused_graph_routes` checks this and falls back if it does
not hold. Do not "fix" a fallback by relaxing that check.

**Verify:** the server answers on 7867 and completions are coherent. Then compare decode ms/token
against a baseline arm run in the same session — not against a number from another day.

## 4. Arm C — prediction scoring, shadow only

Add to arm B:

```
SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR=llapor      # or apex
SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR=<checkpoint dir>
SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES=16
SGLANG_MOE_EXPERT_PREFETCH_BUDGET=3
SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL=100
SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE=<path>
```

Leave `SGLANG_MOE_EXPERT_PREFETCH_PULL` unset. Nothing is transferred; the scorer runs and emits
`budget_recall`.

### 🔴 The caveat that matters, and it is not opt-in

**`budget_recall` changed meaning at `c48b3c69e5`, unconditionally — there is no flag that avoids it.**

Before that commit, the candidate bank ranked every expert by score with no residency filter, and
`BudgetRecall.observe` masked residents afterwards, so `budget_recall` counted non-resident coverage
*within the top-W bank*. After it, the bank itself excludes residents before truncating to width, so
the same metric counts coverage *within the non-resident-only bank*. The old offering is a strict
prefix of the new one, so **`budget_recall` can only rise**, for a reason that has nothing to do with
prediction quality — the bank simply reaches deeper into the score ranking.

Consequences for anyone reading numbers off this server:

- **The recorded figures llapor 0.383/0.389 and apex 0.424/0.420 are pre-seam.** A reading of 0.45
  from this server is not an improvement over them. It is a different quantity.
- **The two predictors are not comparable to each other across the seam either.** Residency is
  sampled at two points — the bank filters at the source layer's score trigger, `observe` re-checks at
  the target layer's `TOPK_IDS` write. LLaPor is next-layer, so its write precedes observe by a full
  layer; APEX is same-layer. The write-to-observe distance differs, so the two arms absorb **different
  amounts** of inflation. Since the live LLaPor-vs-APEX question rests on a 0.42-vs-0.38 gap, a seam
  that moves both arms unequally can move that gap either way.
- **Open and unmeasured:** nobody has verified that residency actually changes inside the
  write-to-observe window during a real forward. The window is structurally open (the tensor reference
  is live, not snapshotted). If residency is in practice frozen across one forward, the eviction-driven
  part of this collapses and only the differing-magnitude point survives. Watching one `expert_to_slot`
  tensor across a forward would settle it and needs no benchmark.

Full record in `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md`, entry **E38**.

If you need numbers comparable to the recorded ones, run them at a commit **before** `c48b3c69e5`.

## 5. What NOT to do

- **Do not set `SGLANG_MOE_EXPERT_PREFETCH_PULL=1`** until §1's four items are closed. It is
  default-off because the plan forbids enabling it on synthetic evidence, and the synthetic evidence
  is all we have.
- **Do not run timing arms while the GPU has other tenants.** As of writing the user's own jobs
  (`entry_with_update.py`, a ComfyUI `main.py --use-sage-attention`) have been resident, at times ~4.7
  GiB. The hot cache alone asks for 10240 MB. Correctness work tolerates co-tenancy; **timing does
  not**, and a delivery-cost number taken beside someone else's kernels attributes their work to ours.
  Census `nvidia-smi --query-compute-apps` before and after, and record both.
- **Do not compare a decode ms/token from this arm against a number from another day** without
  re-running its baseline in the same session. Interleave and repeat baseline arms; that is section 11
  of the plan, not a suggestion.

## 6. Expected gain, and where it is bounded

From **E36**, measured in-graph rather than inferred: delivery costs **0.2237 ms/row** eager /
**0.2240** replay (~11.51 GiB/s, R²=0.99996, no knee), agreeing with E28's nsys production trace to
0.06%. The count-zero floor is **6.752 µs**, i.e. 0.324 ms/token over 48 layers — **0.91%** of a
35.5 ms step.

The prize is large: demand delivery is **42.5%** of the 35.5 ms step at 65.07 misses, and **51.1%** of
a 43.8 ms step at 98.58 misses. But `post()` is a node *inside* the decode CUDA graph, so its time is
structurally incapable of falling outside the step — overlap is the only mechanism by which any of it
is recovered, and Stage B measured overlap at **0.0645 ms**, established as launch-order-bound (117 µs
of host dispatch across five sequential API calls). Replay-based overlap remains **unmeasured**.

Do not present the 42-51% as an expected speedup. It is the size of the target, not the size of the
win.

## 7. Rollback

Every new behaviour is default-off except the `expert_to_slot` pass-through at `bank.write`, which is
unconditional by design (§4). To return to today's production behaviour, drop
`SGLANG_MOE_EXPERT_FUSED_PLAN` and the `PREFETCH_*` lines; the remaining environment is byte-identical
to the live script. With `SGLANG_MOE_EXPERT_PREFETCH_PULL` unset the hot-cache allocation shape is
byte-identical to before Stage C.

The branch is **not pushed**. Nothing here has shipped to any RPM or deploy.
