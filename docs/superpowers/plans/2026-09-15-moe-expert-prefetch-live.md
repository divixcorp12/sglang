# MoE Expert Prefetch Candidates (LLaPor/APEX): Phase A In-Graph Shadow Scoring, Phase B Doorbell Wiring — Implementation Plan

> **Dependency, stated first:** Phase B (Tasks B1–B2) is **BLOCKED** until crypto-c9's `cc/doorbell-serving` merges into `master`.
> - That branch owns the whole shared copy layer: the plan interface, the planner, delivered/residual accounting, and both the in-graph and doorbell copy backends.
> - It merges as one piece after review, divix01 tests and a serving A/B. It is in a rework round for a late-copy race, so there is no ETA.
> - Phase A (Tasks 1–7) starts now and touches no copy path.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- **Phase A:** load the trained LLaPor (and APEX) checkpoints in serving from a model-agnostic place.
  - Score each target layer's experts inside the decode CUDA graph with no host syncs.
  - Publish per-target-layer candidate ids (int64 `[C]`) and float32 priorities as stable device tensors.
  - Validate offline (recall of non-resident native experts within a budget) and live in shadow mode: the same metric, no prefetch, logprobs unchanged **relative to the same-arm nondeterminism baseline** (see Task 6; decode is not bit-reproducible run to run on this box), scoring cost measured.
- **Phase B:** once the shared copy layer merges, adapt those candidates to its plan interface, then run the 4-cell prefetch × doorbell matrix with live tok/s.

**Architecture:**
- **Model-independent package.** A new `layers/moe/expert_prediction/serving/` package loads the trained checkpoints, validates them against the live `MoeLayerSpec`s, and builds static-shape bf16 scorers.
- **In-graph scoring through generic hooks.** Scorers attach through one new generic hook, `FeatureStore.after_write`, which fires from the existing TopK taps and pre-mixer adapters. On a decode forward, scoring rewrites each target layer's top-C candidate ids and priorities in a stable device bank, with no host sync.
- **Candidate output contract.** `PrefetchCandidateBank` holds, per target layer, `ids` int64 `[C]` (best first) and `scores` float32 `[C]` (priorities).
  - The bank is residency-agnostic.
  - A `[num_experts]` mask, resident filtering or a budget clamp is one `scatter_`/`index_select` in the Phase B adapter, whichever crypto-c9's interface wants.
- **Shadow metric.** `BudgetRecall` counts, in-graph, the non-resident native routes covered by the first B non-resident candidates. It is read only at log intervals.
- **No copy path in Phase A.** Phase A reads `expert_to_slot` and nothing else from the hot cache.
- **Gates.** The offline gate (Task 3) and the live shadow run (Task 6) decide whether Phase B is worth running.

**Tech Stack:** Python 3.13, torch 2.13 (CUDA graphs, `torch.cuda.set_sync_debug_mode`), msgspec, safetensors, SGLang `envs`. Phase B adds crypto-c9's shared copy layer (`cc/doorbell-serving`).

**Spec and research inputs (read before any task):**
- LLaPor serving: `/home/dimitri/data/divix/crypto/trading-framework/mechanism_hunter/docs/superpowers/plans/2026-09-14-llapor-gpu-only.md` §6–8 (read-only).
- APEX serving: `.../2026-09-14-apex-gpu-only.md` §5, Tasks 5–8 (read-only).
- Shared copy layer: `/home/dimitri/data/divix/crypto/NVFP4_DOORBELL_COPIER.md` §3, §6, and the branch `shared/cc/doorbell-serving` (both read-only; Phase B waits for the merge).
- Trained models: `docs/superpowers/experiments/2026-09-15-expert-predictor-offline-training.md`. Checkpoints are at `/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630/{llapor/pair-NN,apex/layer-NN}`.
- Capture: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438`. Sessions are in `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`.
- Timing evidence: `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md` E26–E32 (E27 overlap, E28 window, E29/E32 doorbell).

## Global Constraints

- **Phase split (hard). Phase A must not modify the copy path.**
  - Off limits: `python/sglang/srt/layers/moe/expert_stream.py`, `expert_route_plan.py`, `expert_hot_cache.py`, `expert_residency_gpu.py`, `expert_transfer.py`, `expert_prefetch.py`, and `python/sglang/kernels/ops/moe/*`.
  - Also off limits: any file `cc/doorbell-serving` adds.
  - Before each Phase A task, run `git fetch -q shared cc/doorbell-serving && git diff --stat master...shared/cc/doorbell-serving`.
  - **Two overlaps are known and allowed** (checked 2026-09-15 at `56a5920489`):
    - `environ.py`: the branch's hunk is at lines 328–333. Phase A adds lines only after `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES` (line 348).
    - `model_runner.py`: the branch's hunks are at 754–798, 1257 and 1631–1700. Phase A changes one condition in `maybe_init_expert_prediction` (line ~803).
  - If any other Phase A file appears in that diff, stop and coordinate with crypto-c9 through the team lead.
- **Phase B interface is TBD.** It is described only as: per target layer, int64 row ids `[C]`, slots `[C]`, and a count.
  - These come from crypto-c9 at merge: exact names, slot and count dtypes, who filters residents and clamps to a budget, delivered/residual accounting, backend selection, and env flags.
  - Do not code against the unmerged branch.
- **Shadow scoring must not change outputs.**
  - Scoring off must be production: logprobs against the pre-change commit `7de955329a`, **judged against the same-arm baseline, not against bit-exactness**. Measured 2026-09-15: REF-vs-REF flips 5 of 768 probe tokens, so exact equality is not a property this server has.
  - Scoring on only appends kernels that read tap buffers and write the bank and counters, so logprobs must stay **within that same baseline** of scoring off. Equality is not available to assert: a build that adds scoring machinery may change kernel selection, fusion or reduction order without changing semantics, and near-ties then resolve differently.
- **Env names do not collide** with the doorbell branch. That branch defines `SGLANG_MOE_EXPERT_DOORBELL*` and keeps `SGLANG_MOE_PREFETCH_MAX_CANDIDATES`. Phase A uses `SGLANG_MOE_EXPERT_PREFETCH_*`.
- **Model-agnostic placement:**
  - Nothing goes in `python/sglang/srt/models/*` and nothing hardcodes Qwen3.8-Next dimensions.
  - Dimensions come from checkpoint manifests, validated against live `MoeLayerSpec` (`layer_id`, `num_experts`, `top_k`, `hidden_size`).
  - Model-specific facts pass only through `python/sglang/srt/layers/moe/expert_prediction/adapters.py`:
    - `_PRE_MIXER_ADAPTERS` / `register_pre_mixer_adapter` (which tensor is pre-mixer);
    - the new `_MIXER_KIND_ADAPTERS` / `register_mixer_kind_adapter` (mixer kind per layer, used only for metric labels).
- **Everything per token lives in the decode CUDA graph.** No `.item()`, `.tolist()`, `.cpu()`, host syncs, Python loops over rows, or dynamic shapes on the decode path. Python runs only at setup, during graph capture, and for eager prefill forwards (which skip scoring). Metric readback happens once per log interval.
- **Frozen `model_runner.py`:** orchestration-only edits (read `.claude/skills/large-class-style/SKILL.md` first). Env vars go through `envs` (read `.claude/skills/env-var-conventions/SKILL.md`). Use `msgspec.Struct`, not dataclass. No defensive `getattr`/`hasattr`.
- **GPU etiquette (hard):**
  - Every GPU process takes `flock` on `/data/models/slang/nvfp4-work/cc-gpu.lock`.
  - Never start while a server on port 7867 is relaunching: if a 7867 process exists, `http://127.0.0.1:7867/health` must be 200 and the run must fit beside production. Only CPU work or GPU unit tests under 1 GiB qualify.
  - Production is taken down only with the user's explicit approval, obtained with AskUserQuestion by the task that needs it. With production down by approval, the GPU must be empty before launch.
  - While the doorbell session owns the GPU, run no GPU work at all. CPU steps set `CUDA_VISIBLE_DEVICES=""`.
- **Code and tests run on divix01.** Laptop commits, then `git push shared HEAD` is followed by a sync of `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree`: `git fetch -q origin master && git checkout -q --detach FETCH_HEAD`. Use `ssh -n divix01 '...'` only; never `ssh -t` and never `tmux capture-pane`.
- **Test command (divix01):** `cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && timeout 900 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs <test files>`
- **Commits:** stage by name, `git commit -m "..." -- <paths>`. Never stage `.omc/`, `.superpowers/`, or `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md`. End messages with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NurjMVe2nS3PGBqBr8M8MZ
  ```
- **Never modify the serving worktree or the production launcher.** The serving worktree is `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb`; the production launcher is `run-nvfp4-e16c-public.sh`. Never relaunch production; the production session owns it.

---

## Design

### Timing facts this plan is built on (measured, E27–E32)

| Fact | Value | Source |
|---|---|---|
| Decode step | One unbroken CUDA graph (`segments=1 breaks=0`) | E28 |
| Order within a layer | norm, mixer, shared expert, router/TopK, route planning, the layer's miss copy, MoE kernels | E28 |
| Window from copy(L) end to copy(L+1) start | 0.267 ms before a linear-attention layer, 0.290 ms before a full-attention layer (p50) | E28 |
| In-graph miss copy | 0.2239 ms/row + 0.006 ms (11.5 GiB/s) | E28 |
| Doorbell copy on a torch-created stream | ≈0.209 ms/row (0.628 ms for 3 rows, 12.4 GiB/s) | E32 |
| Doorbell post → thread sees the request | Below what the bench resolves; price as ≈0, with 0.03 ms as sensitivity. The 29–75 µs "reaction" is an artifact: it compares Python noticing a CUDA event against the thread's timestamp, and E32 read negative. The busy-spin thread (`expert_doorbell.cuh` 792–813) sees a head change within ~1 µs. The trusted figure is end to end: doorbell in graph beats the in-graph kernel by 4–7% at 1–10 rows. | E29/E32, experiment log ~1574 |
| Doorbell waiting on an already-landed request | 15–22 µs | E29 |
| Misses per layer per token | mean 2.72, p90 6 at 4,957 slots; production now has 4,180 slots, so expect more | E28 |
| Idea-2 ceiling (perfect recall, free prediction) | ≈10 ms/token (+15%) | E28 |
| Per-row expert bytes | 2,764,808 | server log |
| Graph scratch per layer | 10 rows (1.33 GiB total) | server log |
| Copy under compute | No measurable contention either way (±0.01 ms compute, ~0.1 GiB/s copy); a copy graph captured on a side stream replays on whichever stream is current | E27 |
| Two copies at once (side copy for L+1 during L's own miss copy) | **Unmeasured** — E27 measured copy vs compute only | Task 7 |

### Where each predictor's output can be used (input to Phase B)

These findings go to crypto-c9 with the Task 3 and Task 7 reports. Phase B picks its budget from them.

- **LLaPor (target L+1, source L's router features).**
  - Candidates for L+1 are ready after TopK(L), inside layer L, before L's gather in launch order.
  - **Gap window** (a copy that starts after L's own miss copy): E28's 0.267/0.290 ms. That fits `floor(0.267 / 0.2299) = 1` in-graph row. The doorbell (reaction ≈0 + 0.209 ms/row) also lands about 1 row without a wait.
  - **Overlap window** (a copy that starts before L's miss copy): gap + L's copy time ≈ 0.267 + 0.006 + 2.72 × 0.2239 ≈ 0.88 ms, about 3 rows. It is valid only if two concurrent copy launches don't slow each other. E27 measured copy vs compute only; Task 7 measures copy vs copy.
- **APEX (target L, source L's pre-mixer).**
  - **Metric:** the success metric is demand-loading wait saved, not whether a whole row lands before the router. A copy that starts at the pre-mixer hook and finishes after routing still saves its head start, provided the demand path joins it instead of restarting it.
  - **Window:** launch-order time from L's pre-mixer hook to L's own miss copy, i.e. norm + mixer + shared expert + router + planning. It is ≈ 0.267 − 0.06 ≈ 0.21 ms (linear) and 0.23 ms (full): E28's gap minus ~0.06 ms of L−1's MoE kernels. The DMA head start is shorter by the scorer's own GPU time, unmeasured until Task 7. The doorbell reaction is negligible, since the old 29–75 µs figure is a bench artifact (see the timing table).
  - **Join semantics (checked in `cc/doorbell-serving` 19c3ac656e, `expert_doorbell.py`):** once the thread commits a request, `resolve` drains it until its copies land, so progress is kept. Delivery is **all-or-nothing** per tag: resolve waits for every posted row, including mispredicted ones.
  - **Consequence:** wait after routing = `max(0, reaction + n_posted·0.209 + 0.007 − window)`. With reaction ≈0 and the linear window (0.207 ms), the wait including the 0.02 ms completed-wait cost is ≈0.03 ms at B=1, ≈0.24 ms at B=2 and ≈0.45 ms at B=3. Break-even is ≈0.13, 1.1 and 2.0 hits. The ideal per-layer bound at B=1 with a perfect hit is ≈0.19 ms, ≈9 ms/token over 48 layers, before scorer cost.
  - **Compared with LLaPor:** APEX's window is only ≈0.06 ms shorter than LLaPor's gap window. LLaPor is clearly ahead only if the overlap window (≈0.88 ms) holds, which needs Task 7.
  - **Decision:** APEX is not ruled out on timing. Task 3 prices it with an oracle bound, scorer cost and both delivery variants (see the Task 3 amendment). It goes live only if it clears the gate and crypto-c9's layer supports a same-layer post at the pre-mixer hook. Its scorer is built (Task 2) and runs in shadow (Task 6).

### Phase A output contract (what Phase B consumes)

- **Handle.** `ExpertPredictionRuntime.prefetch: PrefetchScoring | None`.
- **Structure.**
  - `PrefetchScoring.targets: list[int]`.
  - `PrefetchScoring.next_target: dict[int, int]`: source layer → target layer for LLaPor, `{}` for APEX.
  - `PrefetchScoring.bank.ids_for(T)`: int64 `[C]`, best first.
  - `PrefetchScoring.bank.scores_for(T)`: float32 `[C]`, non-negative priorities. LLaPor gives sigmoid probabilities summed over batch rows; APEX gives softmax zeroed past `top_k + depth(tau)`, summed.
- **When the bank is written.**
  - The bank rows are rewritten in place from `FeatureStore.after_write`: at the source layer's TOPK_WEIGHTS write (LLaPor) or the target's PRE_MIXER write (APEX).
  - For LLaPor, a Phase B reader at layer L's gather or later sees this forward's candidates for L+1.
  - Addresses survive graph capture. Rows never written hold distinct valid ids with score 0.
- **Residents stay in.** The bank does not filter residents. `BudgetRecall` quantifies what filtering plus a B clamp would deliver.

### Phase B interface (TBD, owned by crypto-c9)

- **Assumed, nothing more:** per target layer, int64 row ids `[C]`, slots `[C]`, and a count.
- **Adapter:** Phase B's adapter (Task B1) maps bank → that interface, and nothing else.
- **Decided by crypto-c9 at merge:**
  - where plans are read;
  - resident filtering and the budget clamp;
  - destination rows and slot safety;
  - delivered/residual accounting;
  - backend selection (in-graph vs doorbell);
  - the prefetch-on flag.
- **Candidate format:** if the interface prefers a `[num_experts]` mask, the adapter builds it with one `scatter_` from the same ids and priorities.

### Scratch versus evictable hot slots

Owned by crypto-c9's copy layer. Task 3 still reports budget recall at 1–10 *and* 16/32 rows, so that decision has data.

**Scratch-write ownership (raised by crypto-c9; written down on both sides).** Their doorbell correctness argument needs layer L's scratch rows to be written *only* by layer L's own copy and its residual copy. A second writer fails open: a late copy overwrites a row the layer is already computing with, giving a silently wrong token with no error and no counter.

- **Phase A writes nothing.** It reads `expert_to_slot` and produces ids plus priorities; it allocates no rows and issues no copy (`serving/runtime.py`, `serving/candidates.py`). No collision is possible with anything shipped today.
- **Phase B allocates nothing either.** We do not open a second pool and must not: destinations come from their planner. On their draft, `ExpertRowPlan.for_scratch(capacity, scratch_base, scratch_rows)` addresses `scratch_base + r` — *the same per-layer pool the graph gather uses*, clamped to `scratch_rows` (10 today). So prefetch rows and gather rows are the same pool by their design, and the disjointness argument has to hold inside their layer, not ours.
- **Why it holds today — confirmed-as-of-now by crypto-c9, not frozen** (their rework is in flight, step 3's tests are unwritten, and one outstanding experiment could still change the gather path). The prefetch write and the gather's scratch write are the *same* write, issued once per tag, because we post through their planner rather than beside it.
  1. One outstanding post per tag. **CHANGED — it is a caller obligation, not a layer guarantee** (verified against the landing code). Nothing checks it: `post()` validates tag, capture state and plan shape, then unconditionally allocates the next sequence and overwrites the tag's state. A second post on a live tag **silently orphans the earlier request** — resolve matches only the later sequence, and no counter fires (`skipped_overrun` is a ring overrun, a different condition). **Our obligation:** tag must be the *target* layer, so LLaPor's L and L+1 are distinct tags, and APEX's `current`-mode post replaces L's router-miss post rather than joining it. B1 asserts one in-flight request per tag on our side, because theirs will not.
  2. The reservation rule keeps a posted slot unread and unwritten by anyone else until its resolve. **Confirmed as documented, unenforced in code.** No runtime check backs any clause, including "plan tensors must not change between post and resolve". A contract to honour, silent when violated.
  3. `plan_residual_routes` assigns residual rows that no needed delivered expert occupies. **Confirmed from the function body**, not its docstring: residual rows are drawn from an ordering that puts un-needed rows first, and it refuses a residual plan sharing tensors with the delivered plan.
  4. **A late copy CAN land after its resolve returned** — measured, not theoretical: `late_completions` 1, with copies arriving 28.96 ms and 101.4 s after enqueue, after the drain exhausted and the doorbell was disabled. The bytes do reach the scratch row. The actual guarantee is one step out: an exhausted drain stickily disables the doorbell, records the failing sequence, and **aborts the process ~30 s later**, so no token is computed from that row. Not "no copy lands", but "a copy may land and nothing survives to consume it".
- **Drain recovery is an OPEN question, not a measured fact.** crypto-c9 withdrew "a timeout always exhausts the drain and the drain never recovers a request": a re-aimed gather test saw the drain *succeed* with the copy landing during it, while eight back-to-back diagnostic repeats reproduced the hold 8/8. The difference between those contexts is unexplained. **Nothing in this plan may depend on the drain never recovering.**
- **Do not infer that a scratch row is safe to reuse once resolve returns.** That inference is false on exactly the failing path. The real invariant depends on the abort happening and on nothing reading that row in the ~30 s window before it does — conditions someone can weaken later without noticing they were load-bearing.
- **Honest description of the doorbell at merge:** a modest gain whose failure mode is a server abort, not a degraded request. The user has not ruled on whether that trade is acceptable.
- **The hazard if any of these weakens:** a timed-out-then-late landing, or a second post on a live tag, writes a row residual has since reassigned. B1 Step 1 re-confirms all four against the merged code before any code is written.

### What stays outside the decode graph, and why

1. Checkpoint load, checksum and spec validation, dtype casting, and buffer allocation: one-time setup.
2. Hook installation, and the Python bodies of hooks during graph capture. Replay re-executes the recorded kernels without Python, as for the existing `RouteTaps`.
3. The `rows <= max_rows` shape check in `FeatureStore.write`. It reads a Python int shape, not device data, and never runs during replay.
4. Eager prefill forwards larger than the tap buffers. `FeatureStore.write` skips them (no `spill` when capture is off), so they are never scored. Prefill prefetch is out of scope.
5. Metric readback, once per `SGLANG_MOE_EXPERT_PREDICTOR_LOG_INTERVAL` forwards, only when a metrics file is set.
6. Phase B's copy-path host steps, owned by crypto-c9's layer.

### Existing Python-side logic this supersedes

- **Host-side next-layer policy:** `expert_prefetch.SparseNextLayerPolicy`, `ExpertPrefetchCoordinator.launch`, and `ExpertHotCacheManager.enable_next_layer_prefetch`.
  - They use `.tolist()` on routes and require the pinned host cache.
  - `ModelRunner.maybe_init_expert_hot_cache` refuses them with graph gather.
  - They are not deleted. `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR` and `SGLANG_MOE_PREFETCH_MAX_CANDIDATES > 0` are mutually exclusive (Task 4).
- **Post-forward shadow scoring:** `ExpertPredictionRuntime._score` calls `ExpertPredictor.predict` after the forward. That is too late to feed the same forward's copy, so LLaPor/APEX do not register as `ExpertPredictor`s. Shadow predictors (`affinity`, `popularity`) keep working unchanged.

### Deviations from the research specs (recorded for the write-up)

- **No LLaPor online adaptation, no rollback, no slot leases in Phase A.** Slot safety belongs to crypto-c9's copy layer.
- **Candidates come from summed batch scores**, the specs' batch extension. At batch 1 this equals per-token ranking.
- **Splits are train/dev/shifted_test** (training write-up), not the specs' 70/10/10/10.
- **APEX live prefetch is gated on Task 3's saved-wait pricing**, not ruled out by its window. The specs' "finish useful transfers inside the window" is replaced by "reduce exposed demand-loading latency".

## File Structure

**Phase A (now):**

| File | Responsibility | Task |
|---|---|---|
| `python/sglang/srt/layers/moe/expert_prediction/serving/__init__.py` | package marker | 1 |
| `.../serving/checkpoints.py` | load and verify LLaPor/APEX checkpoints against `MoeLayerSpec` | 1 |
| `.../serving/scorers.py` | static-shape bf16 `LlaporScorer`, `ApexScorer` | 2 |
| `.../serving/candidates.py` | `PrefetchCandidateBank`, `BudgetRecall` device counters | 2 |
| `.../expert_prediction/training/dataset.py` | add `residency_layer` to `load_layer_rows` | 3 |
| `.../expert_prediction/prefetch_pricing.py` | budget recall, doorbell and in-graph copy timing models (pure torch) | 3 |
| `scripts/expert_prediction/prefetch/check_serving_parity.py` | real checkpoints: serving bf16 vs training fp32 | 3 |
| `scripts/expert_prediction/prefetch/price_prefetch.py` | offline gate report from the capture | 3 |
| `.../expert_prediction/feature_store.py` | `after_write` generic hook | 4 |
| `.../expert_prediction/adapters.py` | `register_mixer_kind_adapter`, `mixer_kinds` | 4 |
| `.../serving/runtime.py` | `PrefetchScoring`: scorers, bank, shadow recall, metrics | 4 |
| `.../expert_prediction/runtime.py` | build `PrefetchScoring` from env; log its metrics | 4 |
| `python/sglang/srt/environ.py` | `SGLANG_MOE_EXPERT_PREFETCH_*` | 4 |
| `python/sglang/srt/model_executor/model_runner.py` | one gate condition (orchestration) | 4 |
| `scripts/expert_prediction/run-shadow-server.sh` | predictor knobs, production-current flags, lock | 5 |
| `scripts/expert_prediction/benchmarks/run_capture_sessions.py` | `--max-tokens`, `--session-ids` | 5 |
| `scripts/expert_prediction/prefetch/select_ab_sessions.py` | fixed A/B subset | 5 |
| `scripts/expert_prediction/prefetch/logprob_probe.py` | greedy top-2 logprob capture per arm | 5 |
| `scripts/expert_prediction/prefetch/compare_logprobs.py` | exact and near-tie correctness gates | 5 |
| `scripts/expert_prediction/prefetch/summarize_ab.py` | tok/s, TTFT, shadow budget recall per arm | 5 |
| `scripts/expert_prediction/prefetch/bench_scoring_cost.py` | 47-layer scorer graph cost at bs 1 | 7 |
| `scripts/expert_prediction/prefetch/bench_concurrent_copies.py` | two concurrent existing-kernel copies on real rows | 7 |
| `test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py` | Task 1 tests (CPU) | 1 |
| `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py` | Task 2 tests (CPU + CUDA) | 2 |
| `test/registered/unit/layers/moe/test_expert_prefetch_pricing.py` | Task 3 tests (CPU) | 3 |
| `test/registered/unit/layers/moe/test_expert_prefetch_runtime.py` | Task 4 tests (CUDA, small) | 4 |
| `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md` | Task 3 report | 3 |
| `docs/superpowers/experiments/2026-09-1X-expert-prefetch-shadow-live.md` | Task 6 report (+ Task 7 section) | 6, 7 |

**Phase B (blocked on the doorbell merge):**

| File | Responsibility | Task |
|---|---|---|
| `.../serving/<adapter>.py` (name TBD) | bank → crypto-c9's per-target-layer ids/slots/count | B1 |
| its test file (name TBD) | adapter tests on the merged layer | B1 |
| `docs/superpowers/experiments/2026-09-1X-expert-prefetch-live-ab.md` | 4-cell matrix and live tok/s | B2 |

---

### Task 1: Serving checkpoint loader validated against live layer specs

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/__init__.py` (empty)
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/checkpoints.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py`

**Interfaces:**
- Consumes (existing): `training.llapor.build_predictor(group, *, pca_rank, num_experts)`; `training.apex.Ranker(hidden_size, num_experts)`; `training.apex.OrdinalCDF(hidden_size, num_depths)`; `training.pca.PCAStats(mean, components, explained_variance)`; `contracts.MoeLayerSpec`.
- Checkpoint layout (as written by `scripts/expert_prediction/training/train_{llapor,apex}.py`):
  - `llapor/pair-NN/`: `model.pt` (state dict), `pca.pt` (`{"mean", "components"}`), `manifest.json` (`architecture{group,pca_rank,num_experts}`, `grouping{source_layer,target_layer,group}`, `pca_stats{explained_variance}`, `tensor_checksums{model}`), `DONE`.
  - `apex/layer-NN/`: `ranker.pt`, `cdf.pt`, `manifest.json` (`layer_id`, `architecture{hidden_size,num_experts,top_k}`, `tensor_checksums{ranker}`), `DONE`.
- Produces:
  - `LlaporCheckpoint(source_layer: int, target_layer: int, group: str, pca: PCAStats, model: nn.Module)`
  - `ApexCheckpoint(layer_id: int, top_k: int, ranker: apex.Ranker, cdf: apex.OrdinalCDF)`
  - `state_dict_sha256(state_dict) -> str`
  - `load_prefetch_checkpoints(model_dir: Path, *, predictor: str, specs: Sequence[MoeLayerSpec]) -> dict[int, LlaporCheckpoint | ApexCheckpoint]`, keyed by the layer whose experts are predicted.
  - `PREFETCH_PREDICTORS = ("llapor", "apex")`

- [ ] **Step 1: Write the failing tests**

```python
"""Prefetch checkpoints load only when complete, unmodified and shaped like the live MoE layers."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.training import apex, llapor

HIDDEN, EXPERTS, TOP_K, RANK = 16, 32, 4, 8


def _specs(hidden=HIDDEN):
    return [MoeLayerSpec(layer_id=i, num_experts=EXPERTS, top_k=TOP_K, hidden_size=hidden) for i in (0, 1)]


def _write_llapor(root: Path, *, checksum_override=None, done=True) -> Path:
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import state_dict_sha256

    torch.manual_seed(0)
    directory = root / "llapor" / "pair-00"
    directory.mkdir(parents=True)
    model = llapor.build_predictor("middle", pca_rank=RANK, num_experts=EXPERTS)
    state = model.state_dict()
    torch.save(state, directory / "model.pt")
    torch.save({"mean": torch.randn(HIDDEN), "components": torch.randn(RANK, HIDDEN)}, directory / "pca.pt")
    manifest = {
        "architecture": {"group": "middle", "pca_rank": RANK, "num_experts": EXPERTS},
        "grouping": {"source_layer": 0, "target_layer": 1, "group": "middle"},
        "pca_stats": {"explained_variance": [1.0] * RANK},
        "tensor_checksums": {"model": checksum_override or state_dict_sha256(state)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if done:
        (directory / "DONE").write_text("ok")
    return directory


def _write_apex(root: Path) -> Path:
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import state_dict_sha256

    directory = root / "apex" / "layer-01"
    directory.mkdir(parents=True)
    ranker = apex.Ranker(HIDDEN, EXPERTS)
    cdf = apex.OrdinalCDF(HIDDEN, EXPERTS - TOP_K + 1)
    torch.save(ranker.state_dict(), directory / "ranker.pt")
    torch.save(cdf.state_dict(), directory / "cdf.pt")
    manifest = {
        "layer_id": 1,
        "architecture": {"hidden_size": HIDDEN, "num_experts": EXPERTS, "top_k": TOP_K},
        "tensor_checksums": {"ranker": state_dict_sha256(ranker.state_dict())},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "DONE").write_text("ok")
    return directory


class TestPrefetchCheckpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_llapor_loads_keyed_by_target_layer(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root)
        loaded = load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())
        self.assertEqual(sorted(loaded), [1])
        self.assertEqual((loaded[1].source_layer, loaded[1].target_layer), (0, 1))
        self.assertFalse(loaded[1].model.training)

    def test_apex_loads_with_cdf(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_apex(self.root)
        loaded = load_prefetch_checkpoints(self.root, predictor="apex", specs=_specs())
        self.assertEqual(loaded[1].top_k, TOP_K)
        self.assertEqual(loaded[1].cdf.thresholds().numel(), EXPERTS - TOP_K + 1)

    def test_checksum_mismatch_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root, checksum_override="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())

    def test_incomplete_checkpoint_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root, done=False)
        with self.assertRaisesRegex(ValueError, "DONE"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs())

    def test_hidden_size_mismatch_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_llapor(self.root)
        with self.assertRaisesRegex(ValueError, "hidden"):
            load_prefetch_checkpoints(self.root, predictor="llapor", specs=_specs(hidden=HIDDEN * 2))

    def test_unknown_layer_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        _write_apex(self.root)
        with self.assertRaisesRegex(ValueError, "lacks"):
            load_prefetch_checkpoints(self.root, predictor="apex", specs=_specs()[:1])

    def test_unknown_predictor_raises(self):
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints

        with self.assertRaisesRegex(ValueError, "unknown prefetch predictor"):
            load_prefetch_checkpoints(self.root, predictor="affinity", specs=_specs())


if __name__ == "__main__":
    unittest.main()
```

Copy the `register_*_ci(...)` line that `test/registered/unit/layers/moe/test_expert_prediction_training.py` uses, and put it after the imports, so the file registers like its CPU neighbour.

- [ ] **Step 2: Run the tests to verify they fail**

Run on divix01, CPU only: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py`

Expected: FAIL with `ModuleNotFoundError: ...serving.checkpoints`.

- [ ] **Step 3: Write the implementation**

```python
"""Load trained LLaPor and APEX checkpoints and validate them against the live MoE layer specs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

PREFETCH_PREDICTORS = ("llapor", "apex")


class LlaporCheckpoint(msgspec.Struct, frozen=True):
    source_layer: int
    target_layer: int
    group: str
    pca: PCAStats
    model: torch.nn.Module


class ApexCheckpoint(msgspec.Struct, frozen=True):
    layer_id: int
    top_k: int
    ranker: apex.Ranker
    cdf: apex.OrdinalCDF


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """The digest the training scripts store under ``tensor_checksums``."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def _manifest(directory: Path) -> dict:
    if not (directory / "DONE").exists():
        raise ValueError(f"prefetch checkpoint {directory} is incomplete: no DONE marker")
    return json.loads((directory / "manifest.json").read_text())


def _verified_state(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_sha256(state) != expected_sha256:
        raise ValueError(f"prefetch checkpoint {path} does not match its manifest checksum")
    return state


def _spec(specs: Mapping[int, MoeLayerSpec], layer_id: int, directory: Path) -> MoeLayerSpec:
    if layer_id not in specs:
        raise ValueError(f"prefetch checkpoint {directory} names MoE layer {layer_id}, which the model lacks")
    return specs[layer_id]


def _load_llapor(directory: Path, specs: Mapping[int, MoeLayerSpec]) -> LlaporCheckpoint:
    manifest = _manifest(directory)
    architecture, grouping = manifest["architecture"], manifest["grouping"]
    source = _spec(specs, grouping["source_layer"], directory)
    target = _spec(specs, grouping["target_layer"], directory)
    experts = {architecture["num_experts"], source.num_experts, target.num_experts}
    if len(experts) != 1:
        raise ValueError(f"prefetch checkpoint {directory} expert count {architecture['num_experts']} "
                         f"does not match layers {source.layer_id}/{target.layer_id}")
    pca_state = torch.load(directory / "pca.pt", map_location="cpu", weights_only=True)
    if tuple(pca_state["mean"].shape) != (source.hidden_size,):
        raise ValueError(f"prefetch checkpoint {directory} PCA hidden width {tuple(pca_state['mean'].shape)} "
                         f"does not match layer {source.layer_id} hidden size {source.hidden_size}")
    model = llapor.build_predictor(
        architecture["group"], pca_rank=architecture["pca_rank"], num_experts=target.num_experts
    )
    model.load_state_dict(_verified_state(directory / "model.pt", manifest["tensor_checksums"]["model"]))
    pca = PCAStats(
        mean=pca_state["mean"],
        components=pca_state["components"],
        explained_variance=torch.tensor(manifest["pca_stats"]["explained_variance"]),
    )
    return LlaporCheckpoint(
        source_layer=source.layer_id,
        target_layer=target.layer_id,
        group=architecture["group"],
        pca=pca,
        model=model.eval(),
    )


def _load_apex(directory: Path, specs: Mapping[int, MoeLayerSpec]) -> ApexCheckpoint:
    manifest = _manifest(directory)
    architecture = manifest["architecture"]
    spec = _spec(specs, manifest["layer_id"], directory)
    found = (architecture["hidden_size"], architecture["num_experts"], architecture["top_k"])
    expected = (spec.hidden_size, spec.num_experts, spec.top_k)
    if found != expected:
        raise ValueError(f"prefetch checkpoint {directory} (hidden, experts, top_k)={found} "
                         f"does not match layer {spec.layer_id} {expected}")
    ranker = apex.Ranker(spec.hidden_size, spec.num_experts)
    ranker.load_state_dict(_verified_state(directory / "ranker.pt", manifest["tensor_checksums"]["ranker"]))
    # The training run stores no checksum for cdf.pt.
    cdf = apex.OrdinalCDF(spec.hidden_size, spec.num_experts - spec.top_k + 1)
    cdf.load_state_dict(torch.load(directory / "cdf.pt", map_location="cpu", weights_only=True))
    return ApexCheckpoint(layer_id=spec.layer_id, top_k=spec.top_k, ranker=ranker.eval(), cdf=cdf.eval())


def load_prefetch_checkpoints(
    model_dir: Path, *, predictor: str, specs: Sequence[MoeLayerSpec]
) -> dict[int, LlaporCheckpoint | ApexCheckpoint]:
    """Checkpoints keyed by the MoE layer whose experts they predict."""
    by_layer = {spec.layer_id: spec for spec in specs}
    if predictor == "llapor":
        pairs = [_load_llapor(path, by_layer) for path in sorted((model_dir / "llapor").glob("pair-*"))]
        loaded = {pair.target_layer: pair for pair in pairs}
    elif predictor == "apex":
        layers = [_load_apex(path, by_layer) for path in sorted((model_dir / "apex").glob("layer-*"))]
        loaded = {layer.layer_id: layer for layer in layers}
    else:
        raise ValueError(f"unknown prefetch predictor {predictor!r}; expected one of {PREFETCH_PREDICTORS}")
    if not loaded:
        raise ValueError(f"no {predictor} prefetch checkpoints under {model_dir}")
    return loaded
```

In `test_unknown_predictor_raises`, the loader raises before touching the directory.

- [ ] **Step 4: Run the tests to verify they pass**

Commit, push, sync, then run the Step 2 command. Expected: `7 passed`, `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(moe): load prefetch predictor checkpoints validated against live MoE layer specs" -- \
  python/sglang/srt/layers/moe/expert_prediction/serving/__init__.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/checkpoints.py \
  test/registered/unit/layers/moe/test_expert_prefetch_checkpoints.py
```

---

### Task 2: Static-shape scorers, candidate bank, and budget-recall counters

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`

**Interfaces:**
- Consumes (Task 1): `LlaporCheckpoint`, `ApexCheckpoint`. Consumes (existing): `training.llapor.encode_features`, `training.apex.select_depth`.
- Produces:
  - `LlaporScorer(checkpoint, *, num_experts: int, dtype, device)`, a module. `forward(router_input [rows,H], topk_ids [rows,K] int, topk_weights [rows,K] float) -> float32 [rows,E]` gives independent activation probabilities.
  - `ApexScorer(checkpoint, *, num_experts: int, tau: float, dtype, device)`, a module. `forward(pre_mixer [rows,H]) -> float32 [rows,E]` gives softmax probabilities, zeroed at ranks ≥ `top_k + depth(tau)`.
  - `PrefetchCandidateBank(*, layer_ids: Sequence[int], width: int, device)` has:
    - attributes `ids int64 [layers,width]` and `scores float32 [layers,width]`;
    - `write(target_layer: int, expert_scores: Tensor[rows,E]) -> None`;
    - `ids_for(target_layer) -> Tensor[width]` and `scores_for(target_layer) -> Tensor[width]`.
  - `BudgetRecall(*, layer_ids: Sequence[int], budget: int, device)` has:
    - `observe(*, target_layer, candidate_ids: Tensor[C], topk_ids: Tensor[rows,K], expert_to_slot: Tensor[E]) -> None`;
    - `counts int64 [layers,2]`: column 0 counts non-resident native routes, column 1 counts those covered by the first `budget` non-resident candidates;
    - `snapshot() -> dict[int, tuple[int, int]]`, a host read used only at log intervals.

- [ ] **Step 1: Write the failing tests**

```python
"""Prefetch scoring is device-only: bf16 scorers match training, bank writes and counters never sync."""

import unittest

import torch

from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

HIDDEN, EXPERTS, TOP_K, RANK = 16, 32, 4, 8


def _llapor_checkpoint(group="middle"):
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import LlaporCheckpoint

    torch.manual_seed(1)
    model = llapor.build_predictor(group, pca_rank=RANK, num_experts=EXPERTS).eval()
    pca = PCAStats(mean=torch.randn(HIDDEN), components=torch.randn(RANK, HIDDEN), explained_variance=torch.ones(RANK))
    return LlaporCheckpoint(source_layer=0, target_layer=1, group=group, pca=pca, model=model)


def _apex_checkpoint():
    from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import ApexCheckpoint

    torch.manual_seed(2)
    return ApexCheckpoint(
        layer_id=1, top_k=TOP_K, ranker=apex.Ranker(HIDDEN, EXPERTS).eval(),
        cdf=apex.OrdinalCDF(HIDDEN, EXPERTS - TOP_K + 1).eval(),
    )


def _route_features(rows=3, device="cpu"):
    generator = torch.Generator().manual_seed(3)
    router_input = torch.randn(rows, HIDDEN, generator=generator)
    topk_ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(rows)])
    topk_weights = torch.rand(rows, TOP_K, generator=generator)
    return router_input.to(device), topk_ids.to(device), topk_weights.to(device)


class TestPrefetchScoringCpu(unittest.TestCase):
    def test_llapor_fp32_scorer_equals_training_forward(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        for group in ("outer", "middle"):
            checkpoint = _llapor_checkpoint(group)
            router_input, topk_ids, topk_weights = _route_features()
            u = llapor.encode_features(router_input, topk_ids, topk_weights, pca=checkpoint.pca, num_experts=EXPERTS)
            expected = torch.sigmoid(checkpoint.model(u))
            scorer = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.float32, device=torch.device("cpu"))
            torch.testing.assert_close(scorer(router_input, topk_ids, topk_weights), expected, rtol=1e-5, atol=1e-6)

    def test_apex_scorer_zeroes_ranks_beyond_top_k_plus_depth(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer

        checkpoint = _apex_checkpoint()
        pre_mixer = torch.randn(2, HIDDEN)
        scorer = ApexScorer(checkpoint, num_experts=EXPERTS, tau=0.9, dtype=torch.float32, device=torch.device("cpu"))
        scores = scorer(pre_mixer)
        probabilities = torch.softmax(checkpoint.ranker(pre_mixer), dim=-1)
        depth = apex.select_depth(checkpoint.cdf(pre_mixer), 0.9, EXPERTS - TOP_K)
        for row in range(2):
            kept = (scores[row] > 0).sum().item()
            self.assertEqual(kept, min(EXPERTS, TOP_K + depth[row].item()))
            best = torch.topk(probabilities[row], kept).indices
            torch.testing.assert_close(scores[row, best], probabilities[row, best], rtol=1e-5, atol=1e-6)

    def test_bank_keeps_top_width_of_summed_rows(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank

        bank = PrefetchCandidateBank(layer_ids=[3, 7], width=2, device=torch.device("cpu"))
        scores = torch.zeros(2, EXPERTS)
        scores[0, 5], scores[1, 5], scores[0, 9], scores[1, 11] = 0.4, 0.4, 0.7, 0.6
        bank.write(7, scores)
        self.assertEqual(bank.ids_for(7).tolist(), [5, 9])
        torch.testing.assert_close(bank.scores_for(7), torch.tensor([0.8, 0.7]))
        self.assertEqual(bank.ids_for(3).tolist(), [0, 1])

    def test_budget_recall_counts_nonresident_routes_within_budget(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall

        recall = BudgetRecall(layer_ids=[1], budget=2, device=torch.device("cpu"))
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long)
        expert_to_slot[[2, 4]] = torch.tensor([0, 1])
        candidates = torch.tensor([2, 6, 4, 8, 10], dtype=torch.long)
        topk_ids = torch.tensor([[2, 8, 10, 12]])
        recall.observe(target_layer=1, candidate_ids=candidates, topk_ids=topk_ids, expert_to_slot=expert_to_slot)
        self.assertEqual(recall.snapshot(), {1: (3, 1)})


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchScoringCuda(unittest.TestCase):
    def _parts(self):
        from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        device = torch.device("cuda")
        scorer = LlaporScorer(_llapor_checkpoint(), num_experts=EXPERTS, dtype=torch.bfloat16, device=device)
        bank = PrefetchCandidateBank(layer_ids=[0, 1], width=6, device=device)
        recall = BudgetRecall(layer_ids=[1], budget=3, device=device)
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)
        expert_to_slot[:10] = torch.arange(10, device=device)
        return scorer, bank, recall, expert_to_slot

    def test_decode_step_never_synchronizes(self):
        scorer, bank, recall, expert_to_slot = self._parts()
        router_input, topk_ids, topk_weights = _route_features(rows=1, device="cuda")
        router_input = router_input.to(torch.bfloat16)
        bank.write(1, scorer(router_input, topk_ids, topk_weights))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            bank.write(1, scorer(router_input, topk_ids, topk_weights))
            recall.observe(target_layer=1, candidate_ids=bank.ids_for(1), topk_ids=topk_ids, expert_to_slot=expert_to_slot)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_graph_replay_matches_eager_for_new_routes(self):
        scorer, bank, recall, expert_to_slot = self._parts()
        router_input, topk_ids, topk_weights = _route_features(rows=1, device="cuda")
        router_input = router_input.to(torch.bfloat16)

        def step():
            bank.write(1, scorer(router_input, topk_ids, topk_weights))
            recall.observe(target_layer=1, candidate_ids=bank.ids_for(1), topk_ids=topk_ids, expert_to_slot=expert_to_slot)

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for seed in range(5):
            generator = torch.Generator().manual_seed(100 + seed)
            router_input.copy_(torch.randn(1, HIDDEN, generator=generator).to("cuda", torch.bfloat16))
            topk_ids.copy_(torch.randperm(EXPERTS, generator=generator)[:TOP_K].unsqueeze(0).to("cuda"))
            topk_weights.copy_(torch.rand(1, TOP_K, generator=generator).to("cuda"))
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.topk(scorer(router_input, topk_ids, topk_weights).sum(0), bank.width)
            self.assertEqual(bank.ids_for(1).tolist(), expected.indices.tolist())

    def test_bf16_scorer_agrees_with_fp32_top_candidates(self):
        from sglang.srt.layers.moe.expert_prediction.serving.scorers import LlaporScorer

        checkpoint = _llapor_checkpoint()
        router_input, topk_ids, topk_weights = _route_features(rows=64, device="cuda")
        full = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.float32, device=torch.device("cuda"))
        half = LlaporScorer(checkpoint, num_experts=EXPERTS, dtype=torch.bfloat16, device=torch.device("cuda"))
        top_full = torch.topk(full(router_input, topk_ids, topk_weights), 8).indices
        top_half = torch.topk(half(router_input.to(torch.bfloat16), topk_ids, topk_weights), 8).indices
        overlap = (top_full.unsqueeze(2) == top_half.unsqueeze(1)).any(2).float().mean().item()
        self.assertGreaterEqual(overlap, 0.95)


if __name__ == "__main__":
    unittest.main()
```

Register the file with `register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")`, as `test_expert_graph_gather.py` does.

- [ ] **Step 2: Run the CPU tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`

Expected: FAIL with `ModuleNotFoundError: ...serving.scorers`. CUDA cases skip.

- [ ] **Step 3: Write `scorers.py`**

```python
"""Static-shape prefetch scorers; every op in ``forward`` can be recorded in a decode CUDA graph."""

from __future__ import annotations

import copy

import torch

from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import ApexCheckpoint, LlaporCheckpoint
from sglang.srt.layers.moe.expert_prediction.training import apex


class LlaporScorer(torch.nn.Module):
    """Target-layer expert activation probabilities from one source layer's routing features."""

    def __init__(self, checkpoint: LlaporCheckpoint, *, num_experts: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.num_experts = num_experts
        self.register_buffer("mean", checkpoint.pca.mean.to(device=device, dtype=dtype))
        self.register_buffer("projection", checkpoint.pca.components.T.contiguous().to(device=device, dtype=dtype))
        self.model = copy.deepcopy(checkpoint.model).to(device=device, dtype=dtype).eval()

    def forward(self, router_input: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        h = (router_input.to(self.mean.dtype) - self.mean) @ self.projection
        ids = topk_ids.long()
        mask = torch.zeros((h.shape[0], self.num_experts), dtype=h.dtype, device=h.device)
        route = torch.zeros_like(mask)
        mask.scatter_(1, ids, 1.0)
        route.scatter_(1, ids, topk_weights.to(h.dtype))
        return torch.sigmoid(self.model(torch.cat((h, mask, route), dim=-1)).float())


class ApexScorer(torch.nn.Module):
    """Same-layer softmax probabilities, kept only for ranks below ``top_k + depth(tau)``."""

    def __init__(
        self, checkpoint: ApexCheckpoint, *, num_experts: int, tau: float, dtype: torch.dtype, device: torch.device
    ):
        super().__init__()
        self.top_k = checkpoint.top_k
        self.tau = tau
        self.ranker = copy.deepcopy(checkpoint.ranker.linear).to(device=device, dtype=dtype)
        self.depth_projection = copy.deepcopy(checkpoint.cdf.w).to(device=device, dtype=dtype)
        self.register_buffer("thresholds", checkpoint.cdf.thresholds().detach().float().to(device))
        self.register_buffer("positions", torch.arange(num_experts, device=device))

    def forward(self, pre_mixer: torch.Tensor) -> torch.Tensor:
        x = pre_mixer.to(self.ranker.weight.dtype)
        probabilities = torch.softmax(self.ranker(x).float(), dim=-1)
        cdf_logits = self.thresholds.unsqueeze(0) - self.depth_projection(x).float()
        depth = apex.select_depth(cdf_logits, self.tau, self.thresholds.numel() - 1)
        order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order).scatter_(1, order, self.positions.expand_as(order))
        return probabilities * (rank < (self.top_k + depth).unsqueeze(1))
```

- [ ] **Step 4: Write `candidates.py`**

```python
"""Device buffers the prefetch planner reads, and in-graph counters of prefetch value."""

from __future__ import annotations

from typing import Sequence

import torch


class PrefetchCandidateBank:
    """Per target MoE layer, the ``width`` experts with the most expected routes this forward.

    Rows are rewritten in place so addresses survive CUDA graph capture; ids are
    initialised to distinct experts so an unwritten row is still a valid index.
    """

    def __init__(self, *, layer_ids: Sequence[int], width: int, device: torch.device) -> None:
        if width < 1:
            raise ValueError("prefetch candidate width must be positive")
        self.width = width
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.ids = torch.arange(width, dtype=torch.int64, device=device).repeat(len(self._rows), 1)
        self.scores = torch.zeros((len(self._rows), width), dtype=torch.float32, device=device)

    def write(self, target_layer: int, expert_scores: torch.Tensor) -> None:
        top = torch.topk(expert_scores.sum(dim=0), self.width)
        row = self._rows[target_layer]
        self.ids[row].copy_(top.indices)
        self.scores[row].copy_(top.values)

    def ids_for(self, target_layer: int) -> torch.Tensor:
        return self.ids[self._rows[target_layer]]

    def scores_for(self, target_layer: int) -> torch.Tensor:
        return self.scores[self._rows[target_layer]]


class BudgetRecall:
    """Non-resident native routes, and those the first ``budget`` non-resident candidates cover."""

    def __init__(self, *, layer_ids: Sequence[int], budget: int, device: torch.device) -> None:
        if budget < 1:
            raise ValueError("prefetch budget must be positive")
        self.budget = budget
        self._rows = {layer_id: row for row, layer_id in enumerate(sorted(layer_ids))}
        self.counts = torch.zeros((len(self._rows), 2), dtype=torch.int64, device=device)

    def observe(
        self,
        *,
        target_layer: int,
        candidate_ids: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_to_slot: torch.Tensor,
    ) -> None:
        resident = expert_to_slot >= 0
        offered_mask = ~resident.index_select(0, candidate_ids)
        offered_mask &= torch.cumsum(offered_mask.to(torch.int64), dim=0) <= self.budget
        offered = torch.where(offered_mask, candidate_ids, torch.full_like(candidate_ids, -1))
        native = topk_ids.reshape(-1).long()
        missed = ~resident.index_select(0, native)
        covered = (native.unsqueeze(1) == offered.unsqueeze(0)).any(dim=1)
        row = self.counts[self._rows[target_layer]]
        row[0].add_(missed.sum())
        row[1].add_((missed & covered).sum())

    def snapshot(self) -> dict[int, tuple[int, int]]:
        values = self.counts.cpu().tolist()
        return {layer_id: tuple(values[row]) for layer_id, row in self._rows.items()}
```

- [ ] **Step 5: Run the tests**

Run CPU first with the Step 2 command. Expected: 4 passed, CUDA cases skipped. Then, **only when the GPU is not owned by the doorbell session and 7867 is up and healthy** (these tests use < 100 MiB), take the lock and run: `flock -n /data/models/slang/nvfp4-work/cc-gpu.lock <test command> test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`. Expected: `7 passed`, `EXIT=0`. If the GPU is unavailable, record "CUDA cases pending" in the commit body and run them at the start of Task 4.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(moe): add graph-capturable prefetch scorers, candidate bank and budget recall" -- \
  python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py \
  test/registered/unit/layers/moe/test_expert_prefetch_scoring.py
```

---

### Task 3: Offline gate — budget recall and in-graph/doorbell pricing from the full capture (CPU only)

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_prediction/training/dataset.py` (`LayerRows`, `load_layer_rows`)
- Create: `python/sglang/srt/layers/moe/expert_prediction/prefetch_pricing.py`
- Create: `scripts/expert_prediction/prefetch/check_serving_parity.py`
- Create: `scripts/expert_prediction/prefetch/price_prefetch.py`
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_pricing.py`
- Report: `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md`

**Interfaces:**
- Consumes: Task 1 `load_prefetch_checkpoints`; Task 2 `LlaporScorer`, `ApexScorer`; existing `dataset.load_session_splits`, `dataset.split_mask`, `capture_schema.FORWARD_RESIDENCY` (int16 `[forwards, layers, experts]`, -1 = non-resident, snapshotted at forward end, which is after that forward's in-graph residency update).
- Produces:
  - `load_layer_rows(..., residency_layer: int | None = None)`, whose `LayerRows.resident` is bool `[rows, experts]` or None;
  - `prefetch_pricing.budget_hits(scores, topk_ids, resident, budget) -> (missed [rows], hits [rows])`;
  - `prefetch_pricing.doorbell_saving_ms(*, hits, budget, window_ms, reaction_ms) -> Tensor[rows] float64`;
  - `prefetch_pricing.side_stream_ready_rows(*, budget, window_ms) -> int`: how many of the plan's rows the side-stream copy lands before the target gather reads readiness (an in-graph copy launched one row at a time, best score first; how crypto-c9's in-graph backend schedules rows is TBD);
  - the constants `IN_GRAPH_ROW_MS`, `IN_GRAPH_FIXED_MS`, `DOORBELL_ROW_MS`, `DOORBELL_FIXED_MS`, `DOORBELL_COMPLETED_WAIT_MS`.
- Side-stream saving is `budget_hits(..., side_stream_ready_rows(...)).hits × IN_GRAPH_ROW_MS`. Rows that land late are simply not delivered, so they cost no wait. Whether their copy slows the target's residual copy is unmeasured (Task 7).

- [ ] **Step 1: Write the failing pricing tests**

```python
"""Prefetch value model: misses counted against residency, hits bounded by the budget, late requests charged."""

import unittest

import torch


class TestPrefetchPricing(unittest.TestCase):
    def test_budget_hits_skip_residents_and_respect_budget(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import budget_hits

        resident = torch.zeros(1, 8, dtype=torch.bool)
        resident[0, [1, 2]] = True
        scores = torch.tensor([[0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0, 0.0]])
        topk_ids = torch.tensor([[1, 3, 5, 6]])
        missed, hits = budget_hits(scores, topk_ids, resident, budget=2)
        self.assertEqual((missed.item(), hits.item()), (3, 1))
        missed, hits = budget_hits(scores, topk_ids, resident, budget=3)
        self.assertEqual((missed.item(), hits.item()), (3, 2))

    def test_saving_charges_the_wait_past_the_window(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            DOORBELL_COMPLETED_WAIT_MS, DOORBELL_FIXED_MS, DOORBELL_ROW_MS, IN_GRAPH_ROW_MS, doorbell_saving_ms,
        )

        hits = torch.tensor([0, 1, 2])
        on_time = doorbell_saving_ms(hits=hits, budget=1, window_ms=1.0, reaction_ms=0.03)
        torch.testing.assert_close(on_time, hits.double() * IN_GRAPH_ROW_MS - DOORBELL_COMPLETED_WAIT_MS)
        late = doorbell_saving_ms(hits=hits, budget=2, window_ms=0.2, reaction_ms=0.05)
        wait = 0.05 + 2 * DOORBELL_ROW_MS + DOORBELL_FIXED_MS - 0.2
        torch.testing.assert_close(late, hits.double() * IN_GRAPH_ROW_MS - wait - DOORBELL_COMPLETED_WAIT_MS)

    def test_side_stream_lands_whole_rows_inside_the_window(self):
        from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
            IN_GRAPH_FIXED_MS, IN_GRAPH_ROW_MS, side_stream_ready_rows,
        )

        row = IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS
        self.assertEqual(side_stream_ready_rows(budget=10, window_ms=0.267), 1)
        self.assertEqual(side_stream_ready_rows(budget=10, window_ms=3 * row + 1e-9), 3)
        self.assertEqual(side_stream_ready_rows(budget=2, window_ms=3 * row), 2)
        self.assertEqual(side_stream_ready_rows(budget=4, window_ms=0.1), 0)


if __name__ == "__main__":
    unittest.main()
```

Add the CPU registration line copied from `test_expert_prediction_training.py`.

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES="" <test command> test/registered/unit/layers/moe/test_expert_prefetch_pricing.py`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `prefetch_pricing.py`**

```python
"""Offline value of expert prefetch: budget recall of cache misses and in-graph/doorbell timing models."""

from __future__ import annotations

import math

import torch

# E28 fit of the in-graph miss copy kernel (11.5 GiB/s): 0.2239 ms/row + 0.006 ms per launch.
IN_GRAPH_ROW_MS = 0.2239
IN_GRAPH_FIXED_MS = 0.006
# E32: doorbell thread on a torch-created stream, 3 rows in 0.628 ms (12.4 GiB/s).
DOORBELL_ROW_MS = 0.209
DOORBELL_FIXED_MS = 0.007
# E29: a wait on an already-landed request costs 15-22 us.
DOORBELL_COMPLETED_WAIT_MS = 0.02


def budget_hits(
    scores: torch.Tensor, topk_ids: torch.Tensor, resident: torch.Tensor, budget: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: native experts missing from the hot cache, and those the top-``budget`` non-resident scores cover."""
    offered = scores.masked_fill(resident, float("-inf")).topk(budget, dim=1).indices
    native = topk_ids.long()
    missed = ~resident.gather(1, native)
    covered = (native.unsqueeze(2) == offered.unsqueeze(1)).any(dim=2)
    return missed.sum(dim=1), (missed & covered).sum(dim=1)


def doorbell_saving_ms(*, hits: torch.Tensor, budget: int, window_ms: float, reaction_ms: float) -> torch.Tensor:
    """In-graph copy time the hits avoid, minus the wait for a ``budget``-row request that lands after the window."""
    landing_ms = reaction_ms + budget * DOORBELL_ROW_MS + DOORBELL_FIXED_MS
    return hits.to(torch.float64) * IN_GRAPH_ROW_MS - max(0.0, landing_ms - window_ms) - DOORBELL_COMPLETED_WAIT_MS


def side_stream_ready_rows(*, budget: int, window_ms: float) -> int:
    """Rows a one-launch-per-row side-stream copy lands within ``window_ms``; later rows are not delivered."""
    return max(0, min(budget, math.floor(window_ms / (IN_GRAPH_ROW_MS + IN_GRAPH_FIXED_MS))))
```

- [ ] **Step 4: Add residency to the capture loader**

In `training/dataset.py`, import `FORWARD_RESIDENCY` next to `FORWARD_KIND`. Add `resident: torch.Tensor | None = None` as the last field of `LayerRows`. Add a keyword `residency_layer: int | None = None` to `load_layer_rows`. Inside the per-shard loop, right after `local_forward` is computed and `is_decode_parts` is appended, add:

```python
            if residency_layer is not None:
                table = handle.get_slice(FORWARD_RESIDENCY)[:, residency_layer : residency_layer + 1, :]
                resident_parts.append(table[local_forward.long(), 0, :] >= 0)
```

Declare `resident_parts: list[torch.Tensor] = []` next to `rid_parts`, and pass it into the returned struct:

```python
        resident=torch.cat(resident_parts) if residency_layer is not None and resident_parts else None,
```

Add this test to `test_expert_prefetch_pricing.py`. It writes a two-forward shard with `safetensors.torch.save_file` and a manifest through `capture_reader`'s format, and asserts `rows.resident` matches the per-forward table. Copy the manifest/shard writing helper from `test/registered/unit/layers/moe/test_expert_prediction_training.py`'s dataset join test, and add the `forward.expert_to_slot` tensor:

```python
    def test_load_layer_rows_joins_residency_by_forward(self):
        # Shard: forwards 0 (decode) and 1 (decode), one row each, layer 0 residency differs per forward.
        residency = torch.full((2, 1, 8), -1, dtype=torch.int16)
        residency[0, 0, 3] = 0
        residency[1, 0, 5] = 0
        rows = _load_single_layer_capture(self.tmp, residency=residency)  # helper built from the training test's writer
        self.assertEqual(rows.resident[0].nonzero().flatten().tolist(), [3])
        self.assertEqual(rows.resident[1].nonzero().flatten().tolist(), [5])
```

`_load_single_layer_capture` calls `load_layer_rows(capture_dir, splits, layer_id=0, features=(RouteFeature.TOPK_IDS,), residency_layer=0)` over the shard built by the copied helper. Run the Step 2 command. Expected: `4 passed`.

- [ ] **Step 5: Real-checkpoint parity script** (`scripts/expert_prediction/prefetch/check_serving_parity.py`)

```python
"""Serving bf16 scorers vs training fp32 forwards on real checkpoints and dev decode rows (CPU)."""

import argparse
import json
from pathlib import Path

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer
from sglang.srt.layers.moe.expert_prediction.training import llapor
from sglang.srt.layers.moe.expert_prediction.training.dataset import load_layer_rows, load_session_splits, split_mask
from sglang.srt.layers.moe.expert_prediction.training.metrics import recall_at_budget

ROWS = 4096


def _specs(capture_dir: Path) -> list[MoeLayerSpec]:
    header = json.loads((capture_dir / "capture.json").read_text())
    return [msgspec.convert(layer, MoeLayerSpec) for layer in header["layers"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layers", default="5,20,44")
    args = parser.parse_args()
    specs = _specs(args.capture_dir)
    splits = load_session_splits(args.sessions)
    llapor_models = load_prefetch_checkpoints(args.model_dir, predictor="llapor", specs=specs)
    apex_models = load_prefetch_checkpoints(args.model_dir, predictor="apex", specs=specs)
    experts = specs[0].num_experts
    report = {}
    cpu = torch.device("cpu")
    for target in (int(layer) for layer in args.layers.split(",")):
        pair = llapor_models[target]
        rows = load_layer_rows(
            args.capture_dir, splits, layer_id=pair.source_layer,
            features=(RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS, RouteFeature.PRE_MIXER),
            next_layer_topk=target,
        )
        keep = (split_mask(rows, splits, "dev") & rows.is_decode).nonzero().flatten()[:ROWS]
        router_input = rows.features["router_input"][keep]
        topk_ids, topk_weights = rows.features["topk_ids"][keep], rows.features["topk_weights"][keep]
        labels = rows.features["next_topk_ids"][keep]
        with torch.no_grad():
            u = llapor.encode_features(router_input, topk_ids, topk_weights, pca=pair.pca, num_experts=experts)
            full = torch.sigmoid(pair.model(u))
            half = LlaporScorer(pair, num_experts=experts, dtype=torch.bfloat16, device=cpu)(
                router_input.to(torch.bfloat16), topk_ids, topk_weights
            )
        report[f"llapor_{target}"] = {
            "recall16_fp32": recall_at_budget(torch.topk(full, 16).indices, labels),
            "recall16_bf16": recall_at_budget(torch.topk(half, 16).indices, labels),
        }
        same = load_layer_rows(args.capture_dir, splits, layer_id=target,
                               features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS))
        keep = (split_mask(same, splits, "dev") & same.is_decode).nonzero().flatten()[:ROWS]
        layer = apex_models[target]
        with torch.no_grad():
            full = torch.softmax(layer.ranker(same.features["pre_mixer"][keep]), dim=-1)
            half = ApexScorer(layer, num_experts=experts, tau=1.0, dtype=torch.bfloat16, device=cpu)(
                same.features["pre_mixer"][keep].to(torch.bfloat16)
            )
        labels = same.features["topk_ids"][keep]
        report[f"apex_{target}"] = {
            "recall16_fp32": recall_at_budget(torch.topk(full, 16).indices, labels),
            "recall16_bf16": recall_at_budget(torch.topk(half, 16).indices, labels),
        }
    print(json.dumps(report, indent=2))
    worst = max(abs(v["recall16_fp32"] - v["recall16_bf16"]) for v in report.values())
    raise SystemExit(0 if worst <= 0.005 else 1)


if __name__ == "__main__":
    main()
```

`tau=1.0` makes the APEX depth the full `E-K` endpoint, so the mask keeps every expert and the comparison isolates bf16.

Run on divix01: `CUDA_VISIBLE_DEVICES="" /data/models/slang/.venv/bin/python scripts/expert_prediction/prefetch/check_serving_parity.py --capture-dir /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438 --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --model-dir /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630`. Expected: EXIT=0, bf16 within 0.005 recall@16 of fp32 at every layer.

- [ ] **Step 6: Pricing script** (`scripts/expert_prediction/prefetch/price_prefetch.py`)

**Amendment (APEX feasibility review).** It applies on top of the code below, and it still runs on CPU only.
- **Keep** `doorbell_saving_ms`'s partial-progress accounting, which charges only the wait past the window.
- **Scorer cost:** `--scorer-ms` (default `0,0.02,0.05`), for both predictors. The effective window is `window_ms − scorer_ms`, and net saving also subtracts `scorer_ms` per target layer.
- **Oracle arm:** `--predictor oracle` offers the row's native non-resident experts and posts `min(misses, budget)` rows. Landing time takes a per-row posted count; real predictors post `budget`. Price it under both the APEX (same-layer) and LLaPor (gap) windows; it is the upper bound.
- **Delivery variants:**
  - `all_or_nothing` (today's copier) waits for every posted row.
  - `prefix` (hypothetical copier change) lands rows in priority order and waits only through the deepest offered rank that is an actual miss: `max(0, reaction + (deepest_hit_rank+1)·0.209 + 0.007 − window)`. A row with no hits waits only `DOORBELL_COMPLETED_WAIT_MS`.
  - Price both.
- **Splits:** by mixer kind (linear vs full attention) and by `forward.kind` (prefill vs decode).
- **Reaction:** `--reaction-ms` defaults to `0,0.03`. The old 29–75 µs is a bench artifact, so the gate uses r0.0.
- **Tests:** CPU unit tests for the oracle posted-count path and the prefix wait.

```python
"""Offline prefetch gate: recall of non-resident native experts within a budget, and doorbell ms/token (CPU)."""

import argparse
import json
from pathlib import Path

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.prefetch_pricing import (
    IN_GRAPH_FIXED_MS, IN_GRAPH_ROW_MS, budget_hits, doorbell_saving_ms, side_stream_ready_rows,
)
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer
from sglang.srt.layers.moe.expert_prediction.training.dataset import load_layer_rows, load_session_splits, split_mask

# E28 p50 windows (copy(L) end -> copy(L+1) start) by the target layer's mixer kind.
WINDOW_MS = {"full_attention": 0.290, "linear_attention": 0.267, "unknown": 0.267}
# APEX must land before its own layer's copy: drop the previous layer's MoE kernels (~0.05 ms) and 10 us of slack.
APEX_WINDOW_REDUCTION_MS = 0.06
BATCH = 4096


def _mixer_kinds(model_config: Path, layer_ids) -> dict[int, str]:
    config = json.loads(model_config.read_text())
    config = config.get("text_config", config)
    layer_types = config.get("layer_types") or []
    return {layer: layer_types[layer] if layer < len(layer_types) else "unknown" for layer in layer_ids}


def _scores(predictor, checkpoint, rows, experts):
    cpu = torch.device("cpu")
    if predictor == "llapor":
        scorer = LlaporScorer(checkpoint, num_experts=experts, dtype=torch.float32, device=cpu)
        inputs = (rows.features["router_input"], rows.features["topk_ids"], rows.features["topk_weights"])
    else:
        scorer = ApexScorer(checkpoint, num_experts=experts, tau=1.0, dtype=torch.float32, device=cpu)
        inputs = (rows.features["pre_mixer"],)
    with torch.no_grad():
        return torch.cat([scorer(*(x[i : i + BATCH] for x in inputs)) for i in range(0, inputs[0].shape[0], BATCH)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--predictor", choices=("llapor", "apex"), required=True)
    parser.add_argument("--budgets", default="1,2,3,4,6,8,10,16,32")
    parser.add_argument("--reaction-ms", default="0,0.03")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    header = json.loads((args.capture_dir / "capture.json").read_text())
    specs = [msgspec.convert(layer, MoeLayerSpec) for layer in header["layers"]]
    splits = load_session_splits(args.sessions)
    models = load_prefetch_checkpoints(args.model_dir, predictor=args.predictor, specs=specs)
    kinds = _mixer_kinds(args.model_config, [spec.layer_id for spec in specs])
    budgets = [int(b) for b in args.budgets.split(",")]
    reactions = [float(r) for r in args.reaction_ms.split(",")]
    experts = specs[0].num_experts
    per_layer = {}
    for target, checkpoint in sorted(models.items()):
        if args.predictor == "llapor":
            rows = load_layer_rows(
                args.capture_dir, splits, layer_id=checkpoint.source_layer,
                features=(RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS),
                next_layer_topk=target, residency_layer=target,
            )
            native = rows.features["next_topk_ids"]
        else:
            rows = load_layer_rows(args.capture_dir, splits, layer_id=target,
                                   features=(RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS), residency_layer=target)
            native = rows.features["topk_ids"]
        scores = _scores(args.predictor, checkpoint, rows, experts)
        window = WINDOW_MS[kinds[target]] - (APEX_WINDOW_REDUCTION_MS if args.predictor == "apex" else 0.0)
        per_layer[target] = {"mixer": kinds[target], "window_ms": window}
        for split in ("dev", "shifted_test"):
            keep = (split_mask(rows, splits, split) & rows.is_decode).nonzero().flatten()
            for budget in budgets:
                missed, hits = budget_hits(scores[keep], native[keep], rows.resident[keep], budget)
                entry = {"rows": int(keep.numel()), "missed_per_token": float(missed.double().mean()),
                         "hits_per_token": float(hits.double().mean()),
                         "budget_recall": float(hits.sum() / missed.sum().clamp(min=1))}
                for reaction in reactions:
                    entry[f"saving_ms_r{reaction}"] = float(
                        doorbell_saving_ms(hits=hits, budget=budget, window_ms=window, reaction_ms=reaction).mean()
                    )
                if args.predictor == "llapor":
                    # "gap": the side copy starts after the source layer's own miss copy (E28 window only).
                    # "overlap": it starts before that copy, so the window also spans the source layer's
                    # baseline copy time. Valid only if concurrent copies do not slow each other (Task 7).
                    source_entry = per_layer.get(checkpoint.source_layer, {}).get(f"{split}_b{budgets[0]}", {})
                    source_copy_ms = IN_GRAPH_FIXED_MS + source_entry.get("missed_per_token", 0.0) * IN_GRAPH_ROW_MS
                    for label, side_window in (("gap", window), ("overlap", window + source_copy_ms)):
                        ready = side_stream_ready_rows(budget=budget, window_ms=side_window)
                        side_hits = (budget_hits(scores[keep], native[keep], rows.resident[keep], ready)[1]
                                     if ready else torch.zeros_like(hits))
                        entry[f"side_ready_rows_{label}"] = ready
                        entry[f"side_saving_ms_{label}"] = float(side_hits.double().mean() * IN_GRAPH_ROW_MS)
                per_layer[target][f"{split}_b{budget}"] = entry
        print(json.dumps({"layer": target, **per_layer[target]}), flush=True)
    totals = {}
    for split in ("dev", "shifted_test"):
        for budget in budgets:
            key = f"{split}_b{budget}"
            totals[key] = {
                "budget_recall": sum(v[key]["hits_per_token"] for v in per_layer.values())
                / max(sum(v[key]["missed_per_token"] for v in per_layer.values()), 1e-9),
                **{f"saving_ms_per_token_r{r}": sum(v[key][f"saving_ms_r{r}"] for v in per_layer.values())
                   for r in reactions},
            }
            if args.predictor == "llapor":
                for label in ("gap", "overlap"):
                    totals[key][f"side_saving_ms_per_token_{label}"] = sum(
                        v[key][f"side_saving_ms_{label}"] for v in per_layer.values()
                    )
    args.out.write_text(json.dumps({"predictor": args.predictor, "totals": totals, "layers": per_layer}, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the gate on divix01 (CPU only; ~48 layer loads per predictor)**

```bash
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && mkdir -p /mnt/nvme2/nvfp4-work/prefetch-gate && \
  for p in llapor apex; do CUDA_VISIBLE_DEVICES="" nohup /data/models/slang/.venv/bin/python -u scripts/expert_prediction/prefetch/price_prefetch.py \
    --capture-dir /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438 \
    --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl \
    --model-dir /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630 \
    --model-config /mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json \
    --predictor $p --out /mnt/nvme2/nvfp4-work/prefetch-gate/$p.json > /mnt/nvme2/nvfp4-work/prefetch-gate/$p.log 2>&1; done &'
```

Run the two predictors sequentially: each holds ≤ ~10 GB RAM per layer, and divix01 has ~147 GB free. Poll the logs every ~20 min.

- [ ] **Step 8: Decide and report**

Write `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md` with:
- **Totals:** a table of budget recall and ms/token by budget, split, and reaction.
- **Per-mixer-kind means.**
- **Parity output** from Step 5.
- **A decision under these rules** (all on `shifted_test`, budget ≤ 10). The decision is whether Phase B is worth running, and which budget to give crypto-c9:
  - **GO:** `side_saving_ms_per_token_gap` ≥ 3.0 or `saving_ms_per_token_r0.0` ≥ 3.0 for some budget, with r0.03 reported as sensitivity (the measured reaction is a bench artifact). Record each argmax budget.
  - **Conditional GO:** only `side_saving_ms_per_token_overlap` reaches 3.0. Phase B then depends on Task 7 showing that concurrent copies do not slow each other.
  - **NO-GO:** none of the above. Report to the user before Task 4. Tasks 4–7 still run only if the user says so; the shadow run still measures scoring cost.
  - **Gate metric:** saved demand wait in ms/token *after* scorer cost, not whether rows land inside the window. Don't write "doesn't fit" as a verdict. For each GO, state which arm (predictor or oracle), delivery variant, scorer cost and budget clears 3.0.
  - **Oracle bound:** if even the oracle under `all_or_nothing` stays below 3.0 ms/token for a window, that predictor/window pair is NO-GO on timing, whatever its recall.
  - **APEX live prefetch:** needs the gate *and* same-layer post support at the pre-mixer hook in crypto-c9's layer. Otherwise the doc states the shortfall.
  - **Copier ask:** if `prefix` clears the gate and `all_or_nothing` doesn't, send crypto-c9 the numbers and ask for per-row (prefix) delivery.
  - **Evictable-slot follow-up** (for crypto-c9): flag it if budget recall at 32 exceeds budget 10 by more than 0.15.

Send the decision to the user; do not start Task 4 on NO-GO without their answer.

- [ ] **Step 9: Commit**

```bash
git commit -m "feat(moe): price live expert prefetch offline from the capture's residency" -- \
  python/sglang/srt/layers/moe/expert_prediction/training/dataset.py \
  python/sglang/srt/layers/moe/expert_prediction/prefetch_pricing.py \
  scripts/expert_prediction/prefetch/check_serving_parity.py \
  scripts/expert_prediction/prefetch/price_prefetch.py \
  test/registered/unit/layers/moe/test_expert_prefetch_pricing.py \
  docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md
```

---

### Task 4: Live in-graph shadow scoring runtime — generic tap hook, env vars, runner gate (Phase A)

Touches no copy path. It reads `ExpertHotCacheManager.caches[layer].expert_to_slot` only.

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_prediction/feature_store.py` (`after_write`)
- Modify: `python/sglang/srt/layers/moe/expert_prediction/adapters.py` (mixer-kind registry)
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/runtime.py` (`from_env`, `build`, `on_forward_end`, `close`)
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py` (`maybe_init_expert_prediction` gate only)
- Test: `test/registered/unit/layers/moe/test_expert_prefetch_runtime.py`; additions to `test_expert_prediction_runtime.py` and `test_expert_prediction_adapters.py`

**Interfaces:**
- Consumes: Tasks 1–2. Also the existing `FeatureStore`, `TappedMoeLayer`, `install_pre_mixer_taps`, and `ExpertHotCacheManager.caches[layer].expert_to_slot`.
- Produces:
  - `FeatureStore.after_write: Callable[[int, RouteFeature, int], None] | None`;
  - `register_mixer_kind_adapter(*, architecture, classify)` and `mixer_kinds(*, model, layers)`;
  - `PrefetchScoring.build(...)`, `.from_checkpoints(...)`, `.features_for(predictor)`, `.required_features`, `.targets`, `.next_target`, `.bank`, `.metrics_record()`;
  - `ExpertPredictionRuntime.prefetch` (the Phase A output contract);
  - the env vars:
    - `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR` (`""|llapor|apex`);
    - `SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR`;
    - `SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES` (16);
    - `SGLANG_MOE_EXPERT_PREFETCH_BUDGET` (3; the shadow metric's budget);
    - `SGLANG_MOE_EXPERT_PREFETCH_APEX_TAU` (0.95).

- [ ] **Step 0: Overlap check.** Run the Global Constraints `git diff --stat` against `shared/cc/doorbell-serving`. Proceed only if its overlap with this task's files is still the two known hunks.

- [ ] **Step 1: Write the failing tests** (`test_expert_prefetch_runtime.py`)

```python
"""Live prefetch scoring runs from generic tap writes, rewrites stable bank rows, and never synchronizes."""

import unittest

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore

HIDDEN, EXPERTS, TOP_K = 16, 32, 4


class _Cache:
    def __init__(self, device):
        self.expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.long, device=device)
        self.expert_to_slot[:6] = torch.arange(6, device=device)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchScoringRuntime(unittest.TestCase):
    def _build(self):
        from sglang.srt.layers.moe.expert_prediction.serving import runtime as serving_runtime
        from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import LlaporCheckpoint
        from sglang.srt.layers.moe.expert_prediction.training import llapor
        from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

        device = torch.device("cuda")
        specs = [MoeLayerSpec(layer_id=i, num_experts=EXPERTS, top_k=TOP_K, hidden_size=HIDDEN) for i in range(3)]
        checkpoints = {
            target: LlaporCheckpoint(
                source_layer=target - 1, target_layer=target, group="outer",
                pca=PCAStats(mean=torch.randn(HIDDEN), components=torch.randn(4, HIDDEN), explained_variance=torch.ones(4)),
                model=llapor.build_predictor("outer", pca_rank=4, num_experts=EXPERTS).eval(),
            )
            for target in (1, 2)
        }
        store = FeatureStore(specs=specs, features=serving_runtime.PrefetchScoring.features_for("llapor"),
                             max_rows=1, device=device, hidden_dtype=torch.bfloat16)
        scoring = serving_runtime.PrefetchScoring.from_checkpoints(
            predictor="llapor", checkpoints=checkpoints, specs=specs, store=store,
            hot_caches={i: _Cache(device) for i in range(3)},
            width=8, budget=2, tau=0.95, dtype=torch.bfloat16, device=device,
        )
        return scoring, store

    def _tap(self, store, layer):
        generator = torch.Generator().manual_seed(layer)
        store.write(layer, RouteFeature.ROUTER_INPUT, torch.randn(1, HIDDEN, generator=generator).to("cuda", torch.bfloat16))
        store.write(layer, RouteFeature.TOPK_IDS, torch.randperm(EXPERTS, generator=generator)[:TOP_K].unsqueeze(0).cuda())
        store.write(layer, RouteFeature.TOPK_WEIGHTS, torch.rand(1, TOP_K, generator=generator).cuda())

    def test_source_tap_rewrites_the_next_layers_bank_row_in_place(self):
        scoring, store = self._build()
        self.assertEqual((scoring.targets, scoring.next_target), ([1, 2], {0: 1, 1: 2}))
        ids, scores = scoring.bank.ids_for(2), scoring.bank.scores_for(2)
        pointer = ids.data_ptr()
        self._tap(store, 1)
        torch.cuda.synchronize()
        self.assertEqual(scoring.bank.ids_for(2).data_ptr(), pointer)
        self.assertNotEqual(scores.abs().sum().item(), 0.0)
        self.assertEqual(scoring.bank.scores_for(1).abs().sum().item(), 0.0)

    def test_forward_taps_never_synchronize(self):
        _, store = self._build()
        for layer in range(3):
            self._tap(store, layer)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for layer in range(3):
                self._tap(store, layer)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_replay_updates_candidates_and_metrics_without_python(self):
        scoring, store = self._build()
        inputs = {layer: (torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                          torch.zeros(1, TOP_K, dtype=torch.long, device="cuda"),
                          torch.zeros(1, TOP_K, device="cuda")) for layer in range(3)}
        for layer, (x, ids, w) in inputs.items():
            ids.copy_(torch.arange(TOP_K, device="cuda").unsqueeze(0) + 6 + layer)

        def forward():
            for layer, (x, ids, w) in inputs.items():
                store.write(layer, RouteFeature.ROUTER_INPUT, x)
                store.write(layer, RouteFeature.TOPK_IDS, ids)
                store.write(layer, RouteFeature.TOPK_WEIGHTS, w)

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        before = scoring.metrics_record()["layers"]["2"]["missed_routes"]
        for _ in range(4):
            inputs[1][0].copy_(torch.randn(1, HIDDEN, device="cuda").to(torch.bfloat16))
            graph.replay()
        torch.cuda.synchronize()
        after = scoring.metrics_record()["layers"]["2"]["missed_routes"]
        self.assertEqual(after - before, 4 * TOP_K)


if __name__ == "__main__":
    unittest.main()
```

Register with `register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")`.

- **The first test** guards the direction of the LLaPor wiring: a source-layer write lands in the *next* layer's row.
- **The replay test:** layer 2's native ids are experts 8..11 and the residents are 0..5, so 4 replays × 4 routes are all misses.
- **Runtime tests** (`test_expert_prediction_runtime.py`, using `envs.X.override(...)`): `from_env` raises `ValueError` naming the env var for:
  - `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR="llapor"` with `expert_hot_cache_manager=None`;
  - the predictor set without `..._MODEL_DIR`;
  - the predictor together with `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR`;
  - the predictor together with `SGLANG_MOE_PREFETCH_MAX_CANDIDATES=4`.
- **Adapter tests** (`test_expert_prediction_adapters.py`): registered and unregistered mixer-kind paths, on that file's fake models.

- [ ] **Step 2: Run to verify failure** (GPU < 200 MiB, lock, etiquette): `flock -n /data/models/slang/nvfp4-work/cc-gpu.lock <test command> test/registered/unit/layers/moe/test_expert_prefetch_runtime.py`. Expected: `ModuleNotFoundError: ...serving.runtime`.

- [ ] **Step 3: `FeatureStore.after_write`.** In `__init__`, after `self.spill = None`:

```python
        # Called after an in-buffer write with (layer_id, feature, rows); runs at graph capture, not replay.
        self.after_write: Callable[[int, RouteFeature, int], None] | None = None
```

In `write`, the in-buffer branch becomes:

```python
            else:
                buffer[:rows].copy_(flat[:, :width])
                if self.after_write is not None:
                    self.after_write(layer_id, feature, rows)
```

- [ ] **Step 4: Mixer-kind adapter** (`adapters.py`; metric labels only)

```python
MixerKindClassifier = Callable[[nn.Module], str]


def _qwen4exp_mixer_kind(decoder: nn.Module) -> str:
    name = type(decoder).__name__
    return "full_attention" if "Attention" in name else "linear_attention" if "Linear" in name else "unknown"


_MIXER_KIND_ADAPTERS: dict[str, MixerKindClassifier] = {
    "Qwen4ExpForConditionalGeneration": _qwen4exp_mixer_kind,
}


def register_mixer_kind_adapter(*, architecture: str, classify: MixerKindClassifier) -> None:
    if architecture in _MIXER_KIND_ADAPTERS:
        raise ValueError(f"mixer-kind adapter already registered for {architecture}")
    _MIXER_KIND_ADAPTERS[architecture] = classify


def mixer_kinds(*, model: nn.Module, layers: Sequence[TappedMoeLayer]) -> dict[int, str]:
    """An unregistered architecture reports ``unknown`` for every layer."""
    classify = _MIXER_KIND_ADAPTERS.get(type(model).__name__)
    if classify is None:
        return {layer.spec.layer_id: "unknown" for layer in layers}
    return {layer_id: classify(decoder) for layer_id, decoder in decoder_layers_by_moe_layer(model=model, layers=layers).items()}
```

The key matches `_PRE_MIXER_ADAPTERS`. The decoder classes (`models/qwen4_exp.py:1585,1625`) are identified by name, never imported.

- [ ] **Step 5: Write `serving/runtime.py`**

```python
"""Live expert prefetch scoring: per-target-layer candidates written inside the decode graph from tap writes.

Every per-token op runs from ``FeatureStore.after_write`` during eager forwards and graph
capture, so replay executes recorded kernels only. Phase B hands ``bank`` rows to the
shared copy layer; this module never touches the copy path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.serving.candidates import BudgetRecall, PrefetchCandidateBank
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer

logger = logging.getLogger(__name__)

_FEATURES = {
    "llapor": frozenset({RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS}),
    "apex": frozenset({RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS}),
}
# The last feature each tap writes for a layer; scoring fires on it so the others are fresh.
_SCORE_TRIGGER = {"llapor": RouteFeature.TOPK_WEIGHTS, "apex": RouteFeature.PRE_MIXER}


class PrefetchScoring:
    """Owns scorers, the candidate bank Phase B reads, and shadow budget-recall counters."""

    @staticmethod
    def features_for(predictor: str) -> frozenset[RouteFeature]:
        return _FEATURES[predictor]

    @classmethod
    def build(cls, *, predictor: str, model_dir: Path, specs: Sequence[MoeLayerSpec], **kwargs) -> "PrefetchScoring":
        checkpoints = load_prefetch_checkpoints(model_dir, predictor=predictor, specs=specs)
        return cls.from_checkpoints(predictor=predictor, checkpoints=checkpoints, specs=specs, **kwargs)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        predictor: str,
        checkpoints: Mapping[int, Any],
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        hot_caches: Mapping[int, Any],
        width: int,
        budget: int,
        tau: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PrefetchScoring":
        by_layer = {spec.layer_id: spec for spec in specs}
        if width > min(spec.num_experts for spec in specs):
            raise ValueError("SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES exceeds the expert count")
        missing_caches = sorted(set(checkpoints) - set(hot_caches))
        if missing_caches:
            raise ValueError(f"expert prefetch scoring needs hot caches for layers {missing_caches}")
        if predictor == "llapor":
            scorers = {target: LlaporScorer(c, num_experts=by_layer[target].num_experts, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {c.source_layer: target for target, c in checkpoints.items()}
            next_target = dict(source_of)
        else:
            scorers = {target: ApexScorer(c, num_experts=by_layer[target].num_experts, tau=tau, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {target: target for target in checkpoints}
            next_target = {}
        scoring = cls(
            predictor=predictor, scorers=scorers, source_of=source_of, next_target=next_target, store=store,
            hot_caches=hot_caches,
            bank=PrefetchCandidateBank(layer_ids=list(checkpoints), width=width, device=device),
            recall=BudgetRecall(layer_ids=list(checkpoints), budget=budget, device=device),
        )
        store.after_write = scoring._on_write
        logger.info("MoE expert prefetch scoring: predictor=%s targets=%d width=%d budget=%d state_bytes=%d",
                    predictor, len(checkpoints), width, budget, scoring.state_nbytes)
        return scoring

    def __init__(self, *, predictor, scorers, source_of, next_target, store, hot_caches, bank, recall) -> None:
        self.predictor = predictor
        self.targets = sorted(scorers)
        # Source layer -> target layer for a next-layer predictor; empty for same-layer predictors.
        self.next_target = next_target
        self._scorers = scorers
        self._source_of = source_of
        self._store = store
        self._hot_caches = dict(hot_caches)
        self.bank = bank
        self.recall = recall

    @property
    def required_features(self) -> frozenset[RouteFeature]:
        return _FEATURES[self.predictor]

    @property
    def state_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self._scorers.values() for t in (*s.parameters(), *s.buffers()))

    def _on_write(self, layer_id: int, feature: RouteFeature, rows: int) -> None:
        if feature is RouteFeature.TOPK_IDS and layer_id in self._scorers:
            self.recall.observe(
                target_layer=layer_id, candidate_ids=self.bank.ids_for(layer_id),
                topk_ids=self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                expert_to_slot=self._hot_caches[layer_id].expert_to_slot,
            )
        if feature is not _SCORE_TRIGGER[self.predictor] or layer_id not in self._source_of:
            return
        target = self._source_of[layer_id]
        with torch.no_grad():
            if self.predictor == "llapor":
                scores = self._scorers[target](
                    self._store.view(layer_id, RouteFeature.ROUTER_INPUT, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_WEIGHTS, rows),
                )
            else:
                scores = self._scorers[target](self._store.view(layer_id, RouteFeature.PRE_MIXER, rows))
        self.bank.write(target, scores)

    def metrics_record(self) -> dict:
        """Host read of the device counters; call only at metric log intervals."""
        layers = {str(layer): {"missed_routes": missed, "covered_routes": covered}
                  for layer, (missed, covered) in self.recall.snapshot().items()}
        missed = sum(v["missed_routes"] for v in layers.values())
        covered = sum(v["covered_routes"] for v in layers.values())
        return {"predictor": self.predictor, "budget": self.recall.budget,
                "budget_recall": covered / missed if missed else 0.0, "layers": layers}
```

**Launch order within one forward (LLaPor):**
1. TopK(L) tap writes ROUTER_INPUT, IDS and WEIGHTS.
2. `_on_write(L, TOPK_IDS)` runs `recall.observe(L)` against the bank row written at L−1.
3. `_on_write(L, TOPK_WEIGHTS)` runs `bank.write(L+1)`.
4. L's gather runs, unchanged.

**APEX:** pre-mixer(L) runs `bank.write(L)`, then TopK(L) runs `recall.observe(L)`.

**Why this is the shadow metric:** `observe` reads the live `expert_to_slot` view, and the in-graph residency update rewrites it at the first streamed layer's gather. So the metric counts non-resident native routes at the moment a Phase B planner would filter them, and how many of those the first B non-resident candidates cover. That is the live counterpart of Task 3's `budget_hits`.

- [ ] **Step 6: Env vars, runtime, runner.**
  - **`environ.py`**, directly after `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES` (clear of the doorbell branch's hunk):

```python
    # In-graph expert prefetch candidate scoring (shadow until the shared copy layer consumes it): "", "llapor" or "apex".
    SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR = EnvStr("")
    SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR = EnvStr("")
    # Candidates per target layer kept in the device bank.
    SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES = EnvInt(16)
    # Rows per layer the shadow budget-recall metric credits.
    SGLANG_MOE_EXPERT_PREFETCH_BUDGET = EnvInt(3)
    SGLANG_MOE_EXPERT_PREFETCH_APEX_TAU = EnvFloat(0.95)
```

  - **`expert_prediction/runtime.py`:**
    - **`from_env`.** Read the five vars. When the predictor is set, raise `ValueError` naming the env var unless all of these hold:
      - `tokens_per_request == 1`;
      - `decode_max_bs >= 1`;
      - `expert_hot_cache_manager is not None`;
      - the model dir is set;
      - `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR` is empty, because `spill` and `after_write` would both be live;
      - `SGLANG_MOE_PREFETCH_MAX_CANDIDATES == 0`.

      Pass `prefetch=PrefetchSettings(predictor, model_dir, width, budget, tau)` into `build`, where `PrefetchSettings` is a `msgspec.Struct(frozen=True)` in this file.
    - **`build`.** Add `prefetch: PrefetchSettings | None = None`. Union `PrefetchScoring.features_for(...)` into the features. After the store and taps (and pre-mixer taps) are installed, set `self.prefetch = PrefetchScoring.build(predictor=..., model_dir=..., specs=specs, store=store, hot_caches=manager.caches, width=..., budget=..., tau=..., dtype=hidden_dtype, device=device)`. The attribute defaults to `None`.
    - **`on_forward_end`.** In the existing metrics block, append `{"prefetch": self.prefetch.metrics_record(), "forwards": self.forwards}` as its own JSONL line when `self.prefetch is not None`. For scoring-only runs (`SGLANG_MOE_EXPERT_PREDICTOR` empty), count `forwards` on decode forwards with `rows > 0`.
    - **`close`.** `self.store.after_write = None` when prefetch scoring is on.
  - **`model_runner.py`** (`maybe_init_expert_prediction`), the only runner edit:

```python
        if self.is_draft_worker or not (
            envs.SGLANG_MOE_EXPERT_PREDICTOR.get()
            or envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR.get()
            or envs.SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR.get()
        ):
```

It is a coordinate/select condition, allowed by `large-class-style` §1.3.

- [ ] **Step 7: Run all prediction tests** under the lock (GPU etiquette): Task 1, 2 and 4 files, plus `test_expert_prediction_runtime.py`, `test_expert_prediction_adapters.py`, `test_expert_prediction_graph.py` and `test_expert_prediction_capture_graph.py`. Expected: all pass, and no previously passing test drops.

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(moe): score expert prefetch candidates in the decode graph with shadow budget recall" -- \
  python/sglang/srt/layers/moe/expert_prediction/feature_store.py \
  python/sglang/srt/layers/moe/expert_prediction/adapters.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py \
  python/sglang/srt/layers/moe/expert_prediction/runtime.py \
  python/sglang/srt/environ.py python/sglang/srt/model_executor/model_runner.py \
  test/registered/unit/layers/moe/test_expert_prefetch_runtime.py \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py \
  test/registered/unit/layers/moe/test_expert_prediction_adapters.py
```

---

### Task 5: Launcher knobs, A/B subset, driver flags, logprob gates and summary scripts (Phase A)

**Files:**
- Modify: `scripts/expert_prediction/run-shadow-server.sh`
- Modify: `scripts/expert_prediction/benchmarks/run_capture_sessions.py`
- Create: `scripts/expert_prediction/prefetch/select_ab_sessions.py`, `logprob_probe.py`, `compare_logprobs.py`, `summarize_ab.py`

**Launcher env:**
- `PREFETCH_PREDICTOR=off|llapor|apex` sets `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR`.
- `PREFETCH_BUDGET` (default 3), `PREFETCH_CANDIDATES` (16), `PREFETCH_MODEL_DIR`.
- `HOT_GPU_MB` (default 12288).
- No doorbell knobs: Phase B adds whatever the merged layer needs.

- [ ] **Step 1: Launcher changes**
  - **Env block:** add `SGLANG_MOE_GPU_RESIDENCY_UPDATE=1`, `SGLANG_MOE_GPU_RESIDENCY_MAX_PROMOTIONS=64` (production since E31), `SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR="${predictor}"`, `SGLANG_MOE_EXPERT_PREFETCH_MODEL_DIR="${PREFETCH_MODEL_DIR:-/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630}"`, `SGLANG_MOE_EXPERT_PREFETCH_BUDGET="${PREFETCH_BUDGET:-3}"` and `SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES="${PREFETCH_CANDIDATES:-16}"`.
  - **Predictor variable:** `predictor=${PREFETCH_PREDICTOR:-off}`; `[ "$predictor" = off ] && predictor=""`.
  - **Server flags:** `--context-length 40000 --max-total-tokens 40000`. Keep `--cuda-graph-backend-decode breakable`, `--cuda-graph-max-bs-decode 1` and `--disable-overlap-schedule`.
  - **Hot cache default:** `hot_gpu_mb=${HOT_GPU_MB:-12288}`.
  - **Refusal:** before the empty-GPU check, `ss -ltn 'sport = :7867' | grep -q LISTEN && { echo "REFUSING_TO_START: production on 7867 is up or relaunching" >&2; exit 1; }`.
  - **Lock:** `exec flock --nonblock /data/models/slang/nvfp4-work/cc-gpu.lock env \`.
  - **Header echo:** add `predictor=`, `candidates=`, `budget=`.
  - **Check:** `bash -n scripts/expert_prediction/run-shadow-server.sh` returns 0.

- [ ] **Step 2: Driver flags** (`run_capture_sessions.py`)
  - `--max-tokens` (int, default 4096) replaces the literal `4096` in `_stream_chat`.
  - `--session-ids` (comma list, default empty) runs only the listed sessions, in file order.
  - Test against the fake SSE server: `--max-tokens 7` appears in the request body, and `--session-ids` filters.

- [ ] **Step 3: `select_ab_sessions.py`**

```python
"""Fixed live A/B subset: the first FinanceBench holdout and ConvFinQA val sessions that fit the time budget."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--val", type=int, default=6)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    args = parser.parse_args()
    picked, counts = [], {"holdout": 0, "val": 0}
    limits = {"holdout": args.holdout, "val": args.val}
    with open(args.sessions) as f:
        for line in f:
            session = json.loads(line)
            split = session["split"]
            if split in limits and counts[split] < limits[split] and session["context_chars"] <= args.max_context_chars:
                picked.append(session)
                counts[split] += 1
    with open(args.out, "w") as f:
        for session in picked:
            f.write(json.dumps(session) + "\n")
    print(json.dumps({"sessions": len(picked), "turns": sum(len(s["turns"]) for s in picked), **counts}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `logprob_probe.py` and `compare_logprobs.py`**

```python
"""Greedy first-turn completions with top-2 logprobs, for the prefetch correctness gates."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = []
    with open(args.sessions) as f:
        sessions = [json.loads(line) for line in f][: args.prompts]
    for session in sessions:
        body = json.dumps({
            "model": "default", "messages": [{"role": "user", "content": session["turns"][0][:6000]}],
            "temperature": 0, "max_tokens": args.max_tokens, "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.loads(response.read())["choices"][0]
        tokens = [{"token": t["token"], "top": [[c["token"], c["logprob"]] for c in t["top_logprobs"]]}
                  for t in choice["logprobs"]["content"]]
        results.append({"session_id": session["session_id"], "tokens": tokens})
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
```

```python
"""Compare two arms' greedy tokens: --exact fails on any flip; otherwise flips must start at a near-tie (E31: top-2 margin <= 0.375 nats)."""

import argparse
import json

NEAR_TIE_NATS = 0.375


def _margin(entry):
    top = entry["top"]
    return top[0][1] - top[1][1] if len(top) > 1 else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()
    base, other = json.load(open(args.base)), json.load(open(args.other))
    failures = []
    for a, b in zip(base, other):
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            if x["token"] != y["token"]:
                if args.exact or min(_margin(x), _margin(y)) > NEAR_TIE_NATS:
                    failures.append({"session_id": a["session_id"], "index": index,
                                     "margins": [_margin(x), _margin(y)]})
                break
    print(json.dumps({"compared": len(base), "flips": failures}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: `summarize_ab.py`**

```python
"""Per arm: median decode tok/s and TTFT over turns with >= 64 completion tokens, plus shadow budget recall."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="arm=results.jsonl[:prediction-metrics.jsonl]")
    args = parser.parse_args()
    table = {}
    for spec in args.runs:
        arm, paths = spec.split("=", 1)
        results, _, metrics = paths.partition(":")
        turns = [json.loads(line) for line in open(results) if line.strip()]
        good = [t for t in turns if "error" not in t and (t.get("completion_tokens") or 0) >= 64]
        entry = table.setdefault(arm, {"tok_s": [], "ttft": [], "turns": 0, "errors": 0, "budget_recall": None})
        entry["tok_s"] += [t["decode_tokens_per_sec"] for t in good if t["decode_tokens_per_sec"]]
        entry["ttft"] += [t["ttft"] for t in good if t["ttft"] is not None]
        entry["turns"] += len(good)
        entry["errors"] += sum("error" in t for t in turns)
        if metrics and Path(metrics).exists():
            records = [json.loads(line) for line in open(metrics) if '"prefetch"' in line]
            if records:
                entry["budget_recall"] = records[-1]["prefetch"]["budget_recall"]
    summary = {arm: {"median_decode_tok_s": statistics.median(e["tok_s"]) if e["tok_s"] else None,
                     "median_ttft_s": statistics.median(e["ttft"]) if e["ttft"] else None,
                     "turns": e["turns"], "errors": e["errors"], "budget_recall": e["budget_recall"]}
               for arm, e in table.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Check and commit.** Run `bash -n` on the launcher, each script's `--help` on divix01 (CPU), and the driver against the fake SSE server. Then:

```bash
git commit -m "feat(nvfp4): add prefetch scoring launcher knobs, A/B subset, logprob gates and summary" -- \
  scripts/expert_prediction/run-shadow-server.sh scripts/expert_prediction/benchmarks/run_capture_sessions.py \
  scripts/expert_prediction/prefetch/select_ab_sessions.py scripts/expert_prediction/prefetch/logprob_probe.py \
  scripts/expert_prediction/prefetch/compare_logprobs.py scripts/expert_prediction/prefetch/summarize_ab.py
```

---

### Task 6: Live shadow validation — exact outputs, scoring cost, live budget recall (Phase A; ask the user first)

**Cells** (all on the same launcher flags, 2 FinanceBench + 6 ConvFinQA val, `--max-tokens 768`):

| Cell | Build and env | Checks |
|---|---|---|
| `REF` | detached worktree at `7de955329a`, predictor unset | reference |
| `S0` | this branch, predictor unset | **answer-level agreement vs REF** (primary); flip count reported as a diagnostic against the same-arm baseline; tok/s within REF's pass-to-pass spread: the off path is production |
| `S1` | this branch, `PREFETCH_PREDICTOR=llapor`, same `HOT_GPU_MB` | **answer-level agreement vs S0** (primary); flip count diagnostic vs S1's own p1/p2 baseline. Tok/s S1 − S0 is the in-graph scoring cost. Live `budget_recall` vs Task 3's `shifted_test`/`dev` budget recall at B = `PREFETCH_BUDGET` |
| `S2` (optional) | `PREFETCH_PREDICTOR=apex` | same as S1, for APEX's cost and recall |

**Preconditions:** Tasks 1–5 merged, and their tests pass on the synced divix01 worktree. Task 3's report exists; a NO-GO there needs the user's explicit go-ahead for this task.

- [ ] **Step 1: Ask the user** (AskUserQuestion) for an approved production-down window.
  - **Estimate:** REF, S0, S1 × 2 passes × ~23 min ≈ 2.3 h. S2 adds ≈0.8 h. Task 7's GPU benches add ≈0.75 h if run in the same window.
  - **Confirm** the crypto-c9 production session has released the GPU and will not relaunch 7867 during the window.
  - **Ask** whether to include S2 and Task 7.
  - Do nothing further without a yes.

- [ ] **Step 2: Prepare** (CPU).
  - **REF worktree:** `git -C /data/models/slang/nvfp4-work/cc-expert-prediction/worktree worktree add --detach /data/models/slang/nvfp4-work/cc-expert-prediction/ref-7de955 7de955329a`. Its launcher copy gets only Task 5 Step 1's flag and lock edits applied by `git show <task5-commit>:scripts/expert_prediction/run-shadow-server.sh > <ref>/scripts/expert_prediction/run-shadow-server.sh`, so REF and S0 share flags. Never touch the serving worktree.
  - **Subset:** `select_ab_sessions.py --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl --out /mnt/nvme2/nvfp4-work/benchmarks/prefetch-shadow/sessions.jsonl`.

- [ ] **Step 3: Run the cells.**
  - **Order:** pass 1 runs REF → S0 → S1 (→ S2); pass 2 runs (S2 →) S1 → S0 → REF.
  - **Each launch:**
    1. `ssh -n divix01 'nohup env PREFETCH_PREDICTOR=... <launcher> prefetch-shadow-<cell>-p<pass> 31040 off radix > /dev/null 2>&1 &'`.
    2. Wait for `/health` 200 with a Monitor until-loop.
    3. Warm up with one turn outside the subset.
    4. `logprob_probe.py --port 31040 --sessions <subset> --out .../<cell>-p<pass>-logprobs.json`.
    5. `run_capture_sessions.py --port 31040 --sessions <subset> --results .../<cell>-p<pass>.jsonl --max-tokens 768`.
    6. Copy the run dir's `expert-prediction.metrics.jsonl`.
    7. `pkill -f '[s]glang serve.*--port 31040'` and wait for an empty GPU.
  - **Required log lines:**
    - S1 must log `MoE expert prefetch scoring: predictor=llapor targets=47` and capture the decode graph without errors.
    - S0 must not log it.
    - Otherwise stop and report the log excerpt.

- [ ] **Step 4: Gates.**
  - **Exactness is RETIRED as a pass/fail gate (measured 2026-09-15).** `compare_logprobs.py --exact REF-p1 S0-p1` returned 5 flips; the control `--exact REF-p1 REF-p2` returned **5 flips against the same arm**, same 1/8-multiple margins, largely different indices (only 7 and 43 recur). The gate was measuring run-to-run nondeterminism. Run every `--exact` comparison and report the counts as a diagnostic, but do not stop on them.
    - **Replacement criterion, PRIMARY: answer-level agreement** (ConvFinQA `correct`) between arms, **conditional on both arms producing a parseable answer** (`finish_reason=stop`). Such a disagreement is a stop-and-report regardless of flip counts. It is the only semantic criterion available: a changed answer is the thing we care about and needs no baseline distribution to interpret, whereas a flip count is a proxy whose units are "places nondeterminism happened to land".
    - **A disagreement where either side hit `finish_reason=length` is a TRUNCATION ARTIFACT, not a semantic failure** (measured 2026-09-15). `correct=False` then means "no answer was parsed", not "a wrong answer was computed". Record it; do not stop. Exclude such sessions from answer agreement and state how many were excluded — never let an exclusion read as agreement.
    - **This gate too can fire within a single arm.** On `cfq-val-Single_AON/2011/page_134.pdf-4`, S1-p1 scored True and S1-p2 False — same build, same budget, same probe. All three arms derived the identical value (11287/8512 - 1 = 0.326010338 against expected 0.32601); two ran past the 768-token cap still mid-trace and emitted no `ANSWER:` line, one converged 70 tokens sooner. An early near-tie rank swap plausibly compounds into a few hundred tokens of divergent chain length. **A criterion that fires between two passes of one build is not evidence about a difference between builds** — the third criterion in this plan to fail that way, after exactness and raw flip count.
    - **Report truncation rate per arm** (`finish_reason=length` out of turns) as a first-class number. An arm truncating materially more than others would mean scoring changes generation length, which is a real finding and the thing this session was actually detecting.
    - **Do not raise the token cap to remove the race.** It changes the workload mid-run and invalidates comparability with arms already banked.
    - **Flip count is a DIAGNOSTIC, never a stop on its own.** Report it for every pairing. Materially above that arm's own baseline (≈2x or more), flips at clear margins rather than near-ties, or flips clustering structurally is a stop-and-report.
    - **Build the baseline from the run itself, at zero extra cost.** Two passes per arm were already planned, so every arm yields a same-arm comparison: `--exact S1-p1 S1-p2`, `--exact S2-p1 S2-p2`, plus REF-p1 vs REF-p2. Three or four samples give a stated range instead of a point. Judge each arm against its own same-arm baseline as well as the pooled spread. Do not spend window on a dedicated baseline pass.
    - **A one-sample baseline is a gate that may not be able to fail** — the mirror of the exactness gate, which could not be passed. With n=1 on a count statistic the spread is unknown, so "5 against a baseline of 5" reads identically whether same-arm repeats range 4-6 (where 8 is a real signal) or 2-12 (where 8 is noise). If the measured spread comes out wide, say so and state the consequence: the flip diagnostic cannot then separate a small regression from run-to-run variation, and the answer-level gate is carrying the weight alone. **A gate with its blind spot written down is worth much more than one that looks quantitative.** Do not report only a mean.
    - **Limitation, accepted deliberately:** the tie-break finding is confirmed at the token level, not at the expert-selection level. The expert-level cross-check was declined to preserve GPU window — it could only confirm a mechanism for a conclusion the same-arm control already establishes independently. Not an open action item.
    - **Do not read a flip as "scoring perturbed the forward."** That inference was in this plan and the control falsified it: a build difference can change kernel selection, fusion or reduction order with identical semantics. Distinguish numerical (flips at near-ties, expert sets differing only by last-rank swaps) from semantic (clear margins, or larger expert-set differences) before concluding anything. Credit: crypto-c9 raised this before the control reported.
    - Root-cause it before Phase B; do not relax the gate.
  - **Throughput:** S0's median tok/s must lie within REF's two-pass spread.
  - **Recall:** live `budget_recall` within ±0.05 (absolute) of Task 3's `dev` value at the same budget means the offline gate is trusted for Phase B. Outside that band, the report explains it (residency drift, prompt mix).

- [ ] **Step 5: Write-up.**
  - **File:** `docs/superpowers/experiments/2026-09-1X-expert-prefetch-shadow-live.md`.
  - **Contents:** setup (commits, flags), the cell table (median tok/s, TTFT, errors, budget recall, per pass), the gate outputs, scoring state bytes from the startup log, and scoring cost in ms/token (from S1 − S0 median decode time per token).
  - **Phase B projection:** Task 3's saving minus the measured scoring cost.
  - Commit, push and sync. **Do not relaunch production;** tell the user the GPU is free.

```bash
git commit -m "docs(moe): live shadow validation of in-graph expert prefetch scoring" -- docs/superpowers/experiments/2026-09-1X-expert-prefetch-shadow-live.md
```

---

### Task 7: Phase B timing inputs that need no copy-path code of ours — scorer cost and concurrent copies (Phase A)

**Where it runs:**
- Only scripts that import existing modules read-only. Nothing is added to the copy path.
- GPU work runs under the lock, and only when etiquette allows:
  - production healthy and the job fits beside it (production peaks at 30.2 of 32 GiB, so realistically inside Task 6's window);
  - or with the GPU free.

**Files:**
- Create: `scripts/expert_prediction/prefetch/bench_scoring_cost.py`
- Create: `scripts/expert_prediction/prefetch/bench_concurrent_copies.py`
- Report: a "Phase B timing inputs" section of the Task 6 write-up

- [ ] **Step 1: `bench_scoring_cost.py`** (< 2 GiB)
  - **Build:** load all 47 LLaPor checkpoints with `load_prefetch_checkpoints` (the capture's `capture.json` gives the specs). Create a `FeatureStore` with `max_rows=1` plus `PrefetchScoring.from_checkpoints`.
  - **Capture:** one CUDA graph that runs the 48 layers' tap writes in order with random bf16 inputs.
  - **Measure:** median replay ms over 2,000 replays, compared with a graph of the same tap writes without scoring. The difference is scoring ms/token.
  - **Variants:** `middle`/`outer` groups separately; `width` 16 vs 64; and PCA folded into the first linear (`fc_in.weight @ projection`, bias adjusted), reported as the fused-PCA follow-up's ceiling.
  - **Cross-check:** against Task 6's S1 − S0.

- [ ] **Step 2: `bench_concurrent_copies.py`** (< 4 GiB)
  - **Setup:**
    - Load two real NVFP4 MoE layers' expert rows into the registered host arena, the way `test_expert_graph_gather.py` registers pinned rows (read it and E27's `work/cc-overlap/bench_overlap.py` on divix01 first).
    - Build two `ExpertHotCache`s with 10 scratch rows each.
    - Build `expert_row_segments` per layer.
  - **Copy launches:** `copy_expert_row_segments_gpu` into scratch rows, used exactly as `expert_stream._gather_graph` uses it, from new tensors in the script.
  - **Arms:**
    - layer A's copy alone on the main stream;
    - layer B's copy alone on a side stream;
    - both launched together (side-stream launch first);
    - both together with the side copy launched 0.1/0.2 ms (`torch.cuda._sleep` calibrated) after the main one.
  - **Sweep:** rows ∈ {1, 2, 3, 6, 10} per copy.
  - **Report:** GiB/s of each copy alone and under the other, and the realized overlap saving `(T_seq − T_both) / T_seq`. This is the unmeasured E27 gap that decides between Phase B's gap-window (1 row) and overlap-window (~3 rows) budgets.
  - **Correctness:** assert byte-exact rows in every arm.

- [ ] **Step 3: Multi-stream capture spike (≤ 2 h, report only).**
  - Capture one CUDA graph that switches to a side stream mid-capture for a copy.
  - Check whether replay keeps that copy on the side stream: side stream busy with `torch.cuda._sleep` while main-stream timestamps advance.
  - E27 predicts no. A yes would let crypto-c9's in-graph backend overlap copies without graph breaks.

- [ ] **Step 4: Report and hand off.** Add the "Phase B timing inputs" section (tables, then projected ms/token per budget for gap vs overlap, net of scoring cost). Send its path to the team lead for crypto-c9. Commit the two scripts and the doc update:

```bash
git commit -m "docs(moe): measure prefetch scoring cost and concurrent expert copies for the shared copy layer" -- \
  scripts/expert_prediction/prefetch/bench_scoring_cost.py scripts/expert_prediction/prefetch/bench_concurrent_copies.py \
  docs/superpowers/experiments/2026-09-1X-expert-prefetch-shadow-live.md
```

---

## Phase B — BLOCKED on the `cc/doorbell-serving` merge

Do not start any Phase B step until crypto-c9's shared copy layer is on `master` and synced to divix01. Do not write code against the unmerged branch.

**Scheduling (user decision):** Task 6's live shadow run waits for this merge. It then shares one GPU window with B2, booked through crypto-c9.

### Task B1: Adapt prefetch candidates to crypto-c9's plan interface

**Assumed interface (exact names TBD at merge):** per target layer, int64 row ids `[C]`, slots `[C]`, and a count.

**Pre-merge draft from crypto-c9 (unmerged, may shift with their deadlock fix; confirm in Step 1):**
- **Module:** `python/sglang/srt/layers/moe/expert_row_plan.py`.
  - `ExpertRowPlan(expert_ids int64 [C], slots int32 [C], count int32 [1])`, one per target layer.
  - `ExpertRowPlanner(<hot cache or live-lookup callable>, scratch_base, scratch_rows)`. A bare map tensor raises TypeError.
  - `plan_candidates(candidates int64 [N] or bool [E], plan, priority=None)` filters residents, dedupes, orders by priority, and clamps to `min(C, scratch_rows)`, which is 10 today.
- **Backends:** `InGraphRowBackend` (default) and `DoorbellRowBackend`, with `post(tag, plan)`, `resolve(tag, plan) -> delivery.mask()` and `copy_residual(tag, delivery)`. Residual goes through `plan_residual_routes`. Tag = target layer.
- **Delivery today is all-or-nothing per tag.**
  - Prefix delivery (resolve waits only through the deepest actual miss) is feasible in the design but not built.
  - It needs per-segment copies in priority order and a monotonic delivered count. Per-segment copy cost is unmeasured, and resolve may have to run in bounded chunks.
  - The ask waits on measured scorer cost (Task 3: prefix clears at budget 3, all-or-nothing doesn't).
- **APEX same-layer:** its pre-mixer `post(tag=L)` *replaces* layer L's router-miss post; there is one outstanding post per tag and no second tag space. At routing: `resolve(L)`, then `plan_residual_routes`, then `copy_residual` in-graph.
- **LLaPor:** `plan_candidates` plus `post(L+1)` inside layer L, then `resolve(L+1)` plus residual at layer L+1. `SGLANG_MOE_EXPERT_DOORBELL_MODE=next_layer` is reserved until this task implements it.

- [ ] **Step 1: Read the merged layer.** Record the answers in this plan file, and commit that edit before writing code:
  - the plan type and its field names and dtypes;
  - where a plan for target L+1 must be written relative to layer L's gather;
  - whether the layer filters residents and clamps to a budget itself, or expects the producer to;
  - how destination slots are chosen;
  - which flag turns prefetch on;
  - how the in-graph and doorbell backends are selected;
  - whether a same-layer (APEX) target is supported;
  - **the scratch-write disjointness argument**, re-confirmed against the merged code (see the four properties above). Two of them are caller obligations that nothing enforces and whose violation is silent: one in-flight request per tag, and the reservation rule. **Write an explicit assertion that tag == target layer and that no tag is posted twice while live**, since their layer will not report it. Design for a late copy landing after resolve, and depend on nothing about drain recovery.
- [ ] **Step 2: Write the adapter** (`serving/<adapter>.py`, name TBD). At the point the merged layer designates, map `PrefetchScoring.bank.ids_for(T)` and `.scores_for(T)` into its plan:
  - with resident filtering (`expert_to_slot.index_select(ids) < 0`) and a top-B by priority, only if the interface leaves that to the producer;
  - or into a `[num_experts]` priority mask via one `scatter_`, if that is what it takes.
  - **In-graph rules:** fixed shapes, no host reads. The only runner/runtime change is the wiring call the merged layer's docs name.
- [ ] **Step 3: Tests** (CUDA, lock), mirroring the merged layer's own test style:
  - `set_sync_debug_mode("error")` over a decode step with scoring and the adapter on;
  - byte-exact expert rows under graph replay with random candidates, including all-resident and garbage ids;
  - prefetch-flag off gives a recorded graph identical to Phase A's (no adapter kernels).
- [ ] **Step 4: Commit** the adapter, its test and the wiring change.

### Task B2: 4-cell prefetch × doorbell matrix and live tok/s (ask the user first)

**Cells** (`P` = the merged prefetch flag, `D` = `SGLANG_MOE_EXPERT_DOORBELL`; names confirmed in B1):

| Cell | Env | Checks |
|---|---|---|
| `P0D0` | both off | `--exact` vs Task 6's REF: identical to production |
| `P0D1` | doorbell on, prefetch off | near-tie rule vs P0D0; tok/s |
| `P1D0` | prefetch on through the merged in-graph backend, LLaPor, budget B | near-tie rule vs P0D0; tok/s; delivered/served counters from the merged layer |
| `P1D1` | prefetch on through the doorbell backend | near-tie rule vs P0D1; tok/s |

**Budget and memory:**
- B comes from Task 3's argmax. Use the gap window unless Task 7 showed concurrent copies cost < 10%; then use the overlap window.
- `HOT_GPU_MB` is equal across cells. Prefetch rows and scorer state come out of hot slots; report slot counts per cell.

- [ ] **Step 1: Ask the user** (AskUserQuestion) for a production-down window: ≈4 cells × 2 passes × ~23 min ≈ 3–3.5 h. Confirm B, and whether to add an APEX cell. APEX is only valid if B1 found same-layer support and Task 3 cleared 3 ms/token.
- [ ] **Step 2: Run** with Task 6's procedure. Pass 1 runs P0D0 → P0D1 → P1D0 → P1D1; pass 2 runs the reverse.
- [ ] **Step 3: Gates and summary.**
  - **Correctness:** exact P0D0; near-tie rule for the others, with P0D0-p1 vs P0D0-p2 bounding noise.
  - **Throughput:** median decode tok/s, TTFT, shadow budget recall, the merged layer's delivered/served counters, and hot slots per cell, set against the Task 3/6/7 projections.
  - **Target:** ≥ 5% median decode improvement for a P1 cell over its P0 counterpart with no correctness failure. With 2 runs per cell, label it preliminary.
- [ ] **Step 4: Write-up and commit** `docs/superpowers/experiments/2026-09-1X-expert-prefetch-live-ab.md`. **Do not relaunch production;** tell the user the GPU is free.

---

## Risks

0. **Bus-bound vs latency-bound is unresolved (crypto-c9, measured facts, 2026-09-15).** Prefetch hides latency; it never reduces bytes. If serving is bus-bound, every saving in Task 3 shrinks.
   - divix01's host link is **PCIe gen3 x16**, ~11 GB/s realistic, a platform limit rather than the card's (device reports gen5). E28's 11.5 GiB/s in-graph and E32's 12.4 GiB/s are at that ceiling, so copies and the demand path compete for one capped resource. (Capped is not the same as saturated: M3 measures the link roughly half busy at steady state — see below.)
   - **The baseline rests on M2's direct measurement, not on production counters** (crypto-c9 withdrew the counter argument after reading the definitions). `gather_copy_engine_bytes` is fed only by the eager gather and promotion paths and is structurally incapable of counting doorbell copies, so its 0 only means the eager path did not run; `h2d_bytes` is not a measured transfer at all but `unique_missed x host_bytes_per_expert`, which is why it tracks `backing_source_bytes` — the same multiplication with different constants. Those counters cannot distinguish "the copy engine carried nothing" from "it carried everything". That the in-graph kernel was the production path in those runs is a *configuration* argument (the doorbell was off or in `in_graph` mode), not a counter argument.
   - **Instrumentation gap:** no counter measures the doorbell's transferred bytes; only the copier's own `bytes_copied`/`rows_copied` covers them, which is why a live run shows 750,553 rows copied with both hot-cache copy counters at zero. **Phase B cannot attribute traffic between mechanisms from the metrics file**, and nobody is building that instrumentation now.
   - Geometry: 48 MoE layers x 10 experts = 480 expert rows per token at 2.76 MB, ~1.32 GB uncached per token.
   - **Bus-bound is now measured (crypto-c9, 2026-09-15).** Link gen under load is gen3 x16 across 37 samples of sustained transfer, with 0 errors, so the idle gen1 reading was noise and the cap is the platform. Achievable H2D on the real 7-row x 2.76 MB gather pattern: in-kernel `ld.global.nc` 12.081 GB/s vs copy engine 13.313 GB/s, a ratio of 1.10, both at the gen3 practical ceiling. Neither path is issue-bound. This independently corroborates Task 3's 0.2239 ms/row baseline (11.5 GiB/s vs their 11.25 GiB/s, within ~2%), so treat that baseline as confirmed.
   - **The saving is bounded by idle bus time per token.** A *correct* prefetch moves the same bytes earlier, so it can hide at most what the link is not already carrying: `saving <= step_time - bytes_per_token / 12.08 GB/s`. **Measured at steady state (crypto-c9's M3, two prompts, 2026-09-15):** 28.14 tok/s at 171.56 MiB/token (miss 65.067 rows/token, hit rate 0.8644) and 22.84 tok/s at 259.92 MiB/token (98.577 rows, 0.7945); 47.976 gathers/token in both; hot cache 12 GiB, 4180 slots. On the formula above that is a ~35.5 ms step carrying ~14.9 ms of transfer, and a ~43.8 ms step carrying ~22.6 ms — **roughly 20 ms of idle link per token in both cases, so Task 3's 5-7 ms/token fits with room.** The bus is not saturated and the negative-saving case below is not where this machine sits.
     - **True duty cycle remains NOT MEASURED, and these ratios are not it.** The 0.419/0.492 and 0.515/0.636 figures against M2's 12.081 GB/s are *mean-throughput* ratios; duty cycle is the fraction of wall time with an active transfer, and 1 Hz sampling cannot resolve it. Do not enter them in a duty-cycle cell. Measured PCIe rx (5.938 / 7.684 GB/s) exceeds the derived h2d rate (5.062 / 6.226 GB/s) by 17.3% and 23.4%, consistently in the same direction — expected in sign, since rx carries more than expert rows, but unattributed.
   - **A wrong prefetch adds bytes, so the downside is negative, not zero** (crypto-c9's correction to an earlier version of this risk). Wasted bytes per token are `(B - hits) x 2.76 MB x 48`; at B=2 and Task 3's recall that is ~0.3-0.4 GB/token, comparable to the real miss traffic. Near saturation those bytes come out of the same budget the real misses need, so prefetch makes the step **slower** than no prefetch.
     - **Precision, not just recall, sets the price.** Recall bounds the gain; precision bounds the loss. The re-price must report precision (`hits / B`) per budget and subtract a bus-cost term `wasted_bytes / 12.08 GB/s` whenever the duty cycle is near 1.
     - **Our design already takes the cheaper half of crypto-c9's point:** prefetch writes only dedicated scratch rows, never evicting a resident expert (the residency updater stays the sole writer of hot slots), so there is no double-fetch-on-eviction term. The wasted-byte term remains.
     - This is one more reason the **evictable-hot-slot follow-up stays unbuilt**: near saturation it converts a wasted fetch into two.
   - A gen4/gen5 platform change (BIOS PCIe generation or a chipset-fed slot) is worth 2-4x and is the only lever that raises the ceiling; every predictor, ours included, only redistributes what is under it.
   - **The doorbell does not unlock more bus.** Copy engine 13.313 vs in-kernel 12.081 GB/s is 1.10x, so it buys ~10% plus an unquantified overlap benefit. Phase B's value must not rest on the doorbell moving materially more bytes than the in-graph kernel; it rests on starting the same bytes earlier.
   - crypto-c9's earlier "bandwidth-bound at ~17 GB/s" was withdrawn as unmeasured. A direct gathers / miss-rows / h2d per token measurement is queued; **do not tune against either assumption until it lands.**
   - **Miss-rate mismatch: the earlier reading of this had the sign backwards.** The 64% figure was production warming, a floor, not steady state. M3 measures 1.36 and 2.06 misses/layer on its two prompts. An earlier version of this line called Task 3's ~3.15 conservative and its pricing "pessimistic" — **wrong on the term that sets the verdict**. Saving is `recall x misses x row_cost`, so *fewer* misses means *fewer avoidable copies* and *less to win*. Fewer misses buy idle link (room), not value; those are different quantities and only the first improves with a quieter cache. (crypto-c9 booked this as their own error after I queried it; the room argument they originally made stands, the value inference did not.)
   - **Do not divide by M3's two prompts.** Both were single-context 2,000-token generations — the highest-locality traffic this cache will ever see, with experts recurring turn after turn. A production mix of many short unrelated requests will miss more. Treat 1.36-2.06 as a floor on favourable traffic.
   - **The capacity argument cuts the other way, against the capture being stale.** Task 3's capture ran at 4,957 slots against production's 4,180. *More* capacity should give *fewer* misses, yet the capture showed 3.15 and the smaller production cache showed 1.36-2.06. Capacity does not explain the gap; workload does — so the capture was measuring a harder mix, which is evidence its 3.15 is the more representative figure, not the outdated one.
   - **MEASURED 2026-09-16: every prediction number in this file comes from a ~33% miss regime. Production runs at 13.6-20.5%.** Reading Task 6's own cumulative counters back out of `*-prediction-metrics.jsonl`, all four live arms sat at **3.32, 3.24, 3.32, 3.38 missed experts per row per layer** (S1-p1, S1-p2, S2-p1, S2-p2) — a 32.4-33.8% native miss rate against top_k 10. The offline capture measures **3.2209**. The row denominator is exact, not estimated: it is the `forwards` field co-located with each cumulative snapshot, and recomputing recall from those counters reproduces the reported `budget_recall` exactly in every arm (0.38266, 0.38875, 0.42380, 0.41997). **So the capture is faithful to the live runs — and both are roughly twice as cold as production.** The gate at Task 6 checked live recall against offline recall and passed; it never checked that either resembled production. LLaPor's 47 target layers (1-47) against APEX's 48 (0-47) is the layer-pair architecture, not an indexing fault.
   - **MEASURED 2026-09-16: the oracle ceiling at B=2 is 0.4970** (B=1 0.2743, B=3 0.6661), on the same capture and split, with 87.08/512 experts resident (17.0%). Against it, **APEX's 0.409 is 82% of what a perfect predictor can achieve, and LLaPor's 0.342 is 69%.** A static train-split popularity prior scores **0.0095/0.0181/0.0265** — the learned models beat a no-learning baseline by ~19x, and LLaPor's checkpoints are converged (train loss flat by epoch 30, dev recall@16 flat, dev never worsening while train improves). **The predictors are not the limiting factor; the budget is.** With a mean of 3.22 misses and 2 slots, even perfect prediction is capped near `B/misses`. 39.3% of rows miss 4 or more experts. Raising B lifts the ceiling 34% (0.497 to 0.666); retraining APEX cannot lift it more than 21%.
     - **This is the measurement that should have preceded the break-even table, and did not.** An oracle arm costs one CPU run and bounds every recall claim in this file. `--predictor oracle` was already implemented in `price_prefetch.py` the whole time.
   - **APEX logs no train loss anywhere** — only `dev_kl` per epoch, with patience-based early stopping at 8-9 epochs on most layers. Underfit and converged are indistinguishable for APEX from what is on disk. LLaPor logs both. Fix this before any retraining round; without it, "train longer" can never be answered for APEX.
   - **READ THIS BEFORE THE TABLE BELOW: it prices delivery at zero, because delivery was never enabled.** Task 6 ran the scorers in **shadow with no copy-path changes** — that is the design's whole point and the reason logprob exactness was a sensible gate at all. S1 and S2 issued no prefetch and moved no rows. So the measured **+2.10 and +3.26 ms/token are SCORING COST ONLY**, containing no delivery cost, while the saving they are compared against comes *entirely* from delivery. The table is therefore not wrong so much as **a table about a system nobody has run**. (crypto-c9, after the wasted-byte term was computed.)
     - Two readings, and nothing measured can choose between them: if the wasted bytes genuinely overlap compute, delivery costs ~0 on the critical path and the table stands; if any meaningful fraction lands on the critical path, it comes off net at **four to eight times the scale of the scoring term** and both predictors are negative everywhere by a wide margin.
     - **The overlap argument cannot be rescued from what we have.** The idle link that makes it work was measured without prefetch, and prefetch at B=2 roughly doubles H2D — a no-prefetch measurement cannot show prefetch's own traffic is free.
     - **The one arm that settles it, and which nobody has run: scoring ON and delivery ON, decode tok/s against S0.** That single measured difference contains scoring cost, delivery cost and the saving, all on the critical path, and replaces this entire construction — the Task 3 saving estimate, the recall ratio, the row-cost constant and the waste term. Put it at the top of the next GPU window, ahead of more cells of the shadow design.
     - **It also decides what the B=1 lever buys, in opposite directions.** If delivery is free, B=1 buys margin — it makes the overlap premise robust, does not move tok/s, and costs recall. If delivery is not free, B=1 buys speed, roughly halving the largest cost term while cutting hits far less. Worth pricing either way; not describable until the reading is known.
   - **State break-even as a function of miss rate, not as a verdict.** Scaling Task 3's budget-2 all-or-nothing saving by each predictor's own live recall gives saving per miss/layer of **1.160 ms/token (LLaPor)** and **0.983 (APEX)**, hence (delivery priced at zero, per the warning above):

     | predictor | live recall | break-even misses/layer | net @1.36 | net @2.06 | net @3.15 |
     |---|---|---|---|---|---|
     | llapor | 0.386 | **1.81** (1.15-2.46 over the cost range) | -0.52 | **+0.29** | **+1.55** |
     | apex | 0.422 | **3.32** (3.26-3.38) | -1.92 | -1.24 | -0.16 |

     So **LLaPor is workload-dependent**, not negative: under water on high-locality traffic, positive from roughly 2 misses/layer upward. **APEX needs ~3.3 misses/layer**, above everything measured and marginal even at the capture's 3.15 — its better recall does not cover its higher scoring cost.

     - **CORRECTION 2026-09-16: only the rightmost column is inside the measured regime.** The live recall that sets both rows (0.386, 0.422) was measured at **3.3 misses/layer**. The `net @1.36` and `net @2.06` columns hold that recall fixed while moving the miss count to regimes in which recall has never been observed, and that is not defensible — **the ceiling itself moves with the miss count.** At 3.22 misses the oracle bound at B=2 is 0.497, so 0.386 is 78% of attainable; at 1.36 misses two slots cover most of what is missed, the bound rises steeply, and both the achievable recall and the saving per miss move with it. Ceiling, recall and saving are one coupled quantity and the table prices them as though only the miss count varied. **Treat `break-even misses/layer`, `net @1.36` and `net @2.06` as withdrawn**, not merely uncertain: they are an extrapolation of a measurement out of its regime, on top of a delivery cost already priced at zero. `net @3.15` is the only cell standing on measured ground, and it still omits delivery.
   - **Recall is NOT double-counted in the table above.** `budget_hits` returns `(missed & covered).sum()`, and `doorbell_saving_ms` prices `hits * IN_GRAPH_ROW_MS`, so Task 3's 3.238/3.001 already have each predictor's own recall baked in — "all_or_nothing" names the *delivery* variant, not a recall assumption. The scaling above therefore uses the RATIO `recall_live / recall_t3` (1.129 LLaPor, 1.032 APEX) to move from the offline recall to the live one, never the raw recall. Had it used raw recall both break-evens would be too low by about `1/recall` — roughly 4.7 misses/layer for LLaPor, which would put it above everything measured. (Check raised by crypto-c9; verified against the code, not against the name.)
   - **PROVENANCE, recorded 2026-09-16 because the counters are about to change meaning.** The 1.36-2.06 misses/layer above and the 65-99 demand-miss rows/token below both derive from `unique_missed` feeding `row["miss_rows"]` / `row["unique_miss_rows"]` (`expert_hot_cache.py:1663-1671`), under the semantics in force **at or before `30cc3db418`**, where those counters record *logical misses*. A side-stream Stage C migrates that telemetry to *physical demand-copy rows*. The counter keeps its name across the change, so a future figure compared against these will look comparable and will not be. **Do not average across that seam; treat pre- and post-migration numbers as two series.** The same applies to this file's live `budget_recall` figures (LLaPor 0.383/0.389, APEX 0.424/0.420): they were produced by `BudgetRecall.observe` with residency masked *after* candidate truncation, and Stage C moves the mask before it — identical metric name, identical code path, different population.
   - **The wasted-byte term, which the net figures above EXCLUDE.** At B=2 the predictor offers `48 x 2 = 96` rows/token = **265 MB/token**, against 65-99 measured demand-miss rows/token. Rows that hit would have been fetched anyway, so the incremental traffic is `96 - hits`:

     | miss/layer | predictor | wasted rows/token | wasted MB | bus time @12.08 GB/s | measured idle link |
     |---|---|---|---|---|---|
     | 1.36 (M3 A) | llapor | 70.8 | 195.8 | **16.2 ms** | 20.6 ms |
     | 1.36 | apex | 68.5 | 189.3 | 15.7 ms | 20.6 ms |
     | 2.06 (M3 B) | llapor | 57.8 | 159.9 | **13.2 ms** | 21.2 ms |
     | 2.06 | apex | 54.3 | 150.1 | 12.4 ms | 21.2 ms |
     | 3.15 (capture) | llapor | 37.6 | 104.1 | 8.6 ms | not measured |

   - **It fits, but it eats most of the headroom, and it must not be subtracted as latency.** Bus occupancy is only a cost where it lands on the critical path; with idle link available these bytes overlap compute. So the honest statement is: the term is **8.6-16.2 ms/token of bus time against ~20.6-21.2 ms of idle link — 62% to 79% of the measured headroom consumed**, tightest on the high-locality prompt where there is least to gain. Naively subtracting it from net would overstate the cost as badly as omitting it understates it.
   - **Critically, that idle link was measured WITHOUT prefetch running.** M3 characterises the no-prefetch steady state; turning prefetch on consumes the very headroom that makes it safe. At B=2 prefetch roughly **doubles H2D traffic** (265 MB offered against 171-260 MB currently moved). So "the bus is not saturated" is a much weaker safety argument than the earlier entry in this file implies: it is true of the system as measured, not of the system with prefetch enabled.
   - **Precision is the lever, and B is the knob.** Wasted rows scale with `B x 48` while hits scale with recall x misses, so dropping to B=1 halves the offered traffic and cuts the waste far more than it cuts the hits. Any future arm should price B=1 before B=2.
   - **Three things that could still break this. The first is no longer a hypothesis.**
     - **Recall is not invariant to miss count — established, not suspected.** This was written as a risk; the 2026-09-16 oracle measurement makes it structural. The attainable recall at budget B is bounded near `B/misses`, so lowering the miss count raises the bound and changes what recall means. Every recall figure in this file was taken at 3.3 misses/layer. Scaling them to 1.36 or 2.06 is extrapolation, and the direction is not even reliably favourable: if the misses that survive a warmer cache are the genuinely cold, unpredictable ones, realised recall can fall while the bound rises.
     - **The 3.2 ms saving is a Task 3 estimate, never a measured quantity.** Unchanged, and still the term the whole construction rests on.
     - **The live path and the offline pricing harness run different selection algorithms.** Live (`serving/candidates.py:26`) does `topk(scores.sum(0), width)` with no residency term, truncating to `width` over all 512 experts including residents, and only then masks residents and takes the first `budget` (`:57-58`). Offline (`prefetch_pricing.offered_ids`) does `masked_fill(resident, -inf).topk(budget)` — mask across all 512 first, then truncate. **Truncate-then-mask versus mask-then-truncate.** `SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES` defaults to 16, so truncation bites whenever 15 or 16 of the top 16 are resident — and that is a structurally favoured state, not a corner case: the predictor ranks by expected routes and the residency policy promotes by expected routes, so the two rankings are correlated by construction and residents concentrate exactly where the cut falls. **The published offline figures and the live figures were therefore never two measurements of one thing.** Counter-evidence, stated because it points the other way: live B=2 recall came in *above* the offline forecast (0.383 vs 0.342, 0.424 vs 0.409), which is not what a badly lossy truncation would produce. Measurement queued; `candidates.py` is held unchanged by agreement with crypto-c9 until it lands, because fixing the order first would destroy the attribution permanently.
       - **RESOLVED 2026-09-16, and the hypothesis was wrong: at width 16 the order costs essentially nothing.** Truncate-first vs mask-first at B=2 is **0.3403 vs 0.3419 (LLaPor, delta 0.0016)** and **0.4083 vs 0.4088 (APEX, delta 0.0004)** — under 1.5% relative in the worst case. The top-16 carries a mean of **5.99 non-resident** candidates (median 6) against a budget of 2, so the cut binds on only 7.7% (LLaPor) and 4.2% (APEX) of rows at B=2. The correlation argument — residents concentrate at the top of the ranking and crowd out the budget — is plausible and **does not survive measurement**. The counter-evidence noted above, live recall landing above the offline forecast, was the reliable signal. **This is not the explanation for the gap to the oracle ceiling; nothing is, beyond the budget itself.**
       - **Width sensitivity: 16 sits just below the plateau, and 8 is a cliff.** At B=2, LLaPor reads 0.3077 / 0.3403 / 0.3419 / 0.3419 at widths 8 / 16 / 32 / 64, matching mask-first exactly from 32 up; APEX 0.3736 / 0.4083 / 0.4088 / 0.4088. So `SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES=16` is a sound default and **raising it buys at most 0.0016** — but lowering it to 8 costs 0.033, twenty times more than the ordering does. Do not treat width as free if the B=1 arm tempts a smaller candidate bank.
       - **A misreading of the live path, corrected, because the residue matters.** The entry above described live selection as per-row. It is not: `PrefetchCandidateBank.write` does `topk(expert_scores.sum(dim=0), width)`, summing scores **across every row in the forward** to pick one shared top-`width` candidate list for the whole batch, which `BudgetRecall.observe` then masks per row. The live algorithm is therefore *further* from the offline harness than this file first claimed, not closer. The measured deltas above reproduce only the order-of-operations on per-row scores; **the cross-request aggregation is not reproducible from the capture** — it would require the train/dev-session rows co-scheduled into the same live forward, which the capture does not retain. Grouping by `forward_index` over the visible decode rows moved the number not at all, which bounds nothing about full-batch aggregation. **Treat "the divergence is harmless" as established for the ordering and unmeasured for the batching.**
1. **The window may be too short to pay.**
   - The gap window fits 1 row per layer, in-graph or doorbell.
   - Only an overlapped copy (≈0.88 ms, ~3 rows) approaches E28's ~10 ms/token ceiling, and that needs concurrent copies not to contend (unmeasured; Task 7).
   - A NO-GO at Task 3 or a negative Phase B is a legitimate outcome.
2. **APEX's head start is short, and the copier waits for whole requests.**
   - The head start is ≈0.21–0.23 ms minus reaction and scorer time. A late copy still saves time because the doorbell joins committed copies, but all-or-nothing delivery makes every mispredicted row add ≈0.209 ms of wait, so budgets above 1 need high precision.
   - **2026-09-16: this is the only budget-limiting claim in this file, and it now rests on an unattributed mechanism.** The ≈0.209 ms per mispredicted row is `DOORBELL_ROW_MS` under all-or-nothing delivery — a property of the doorbell. Two doorbell tests fail at base commit `f415048d25` (independently reproduced: 2 failed / 22 passed running `test_expert_graph_gather.py` alone). `..._falls_back_to_correct_rows_when_the_thread_stalls` fails 3/3 in isolation, `AssertionError: 2 != 1` at `:1166` — a genuine defect in merged, default-off code. `..._drain_on_this_path_ends_at_its_budget_and_never_recovers_the_copy` passes 3/3 alone and fails 3/3 when `..._disables_after_a_drain_runs_out...` runs first (`AssertionError: 0 != 1` at `:687`, the drain recovered the copy); pytest reorders `TestCase` methods alphabetically against definition order (636 vs 699), observed with `-v`, which is why a direct harness and pytest were running different experiments and the earlier "13 of 14" reading looked like flakiness. **So "the copy cannot land until the drain retires" is a property of the code plus a particular process history, and nobody knows which of the two the merge brief measured.** Do not cite drain or fail-stop semantics in any argument for putting delivery on the critical path until the leaked state is identified — not because the drain is known broken, but because no test currently measures it reliably. **Corollary for budgets: the one reason written down for keeping budget low is a doorbell wait term, while the measured constraint on value is the budget ceiling itself (B=2 caps even a perfect predictor at 0.497 against 3.22 misses/row). Those pull in opposite directions and only arm D can price them against each other.**
     - **E36, 2026-09-16: serialized delivery cost is now MEASURED on a path that never touches the doorbell.** `InGraphRowBackend.post` (`resolve`/`copy_residual` are structural no-ops), real 6-tensor NVFP4 layout, CUDA-event median of 200 reps, eager and graph-replay arms: **0.2237 ms/row** (~11.51 GiB/s, linear, R²=0.99996, no knee), plus an **unconditional floor of 48 x 6.752 us = 0.324 ms/token** (0.91% of the 35.5 ms step) paid on every token because a count-zero pull still executes its nodes. Corroborated by E28's 0.2239 at 0.06% — an nsys production trace against a direct microbenchmark, which is genuine independent agreement. **The apparent agreement with line 2091's 14.9/22.6 ms is NOT independent** and must not be cited as such: that was bytes divided by the same ~12.08 GB/s, one measurement expressed twice. Row size is **2,764,808 B**, not the 2,764,800 quoted elsewhere in this repo — the difference is the two float32 alpha scalars.
     - **At M3's miss counts, serialized delivery is 42.4% of the 35.5 ms step and 51.5% of the 43.8 ms step — and this is cost ALREADY BEING PAID, not new cost.** Risk-0 above records that M3's misses were served through this same in-graph kernel with the doorbell off or in `in_graph` mode, so the figure is an *attribution* of the existing step, not an addition to it. **Held at configuration-argument strength, deliberately:** no counter distinguishes in-graph bytes from doorbell bytes (this file records that gap), so this is the best evidence in the repo rather than a measured fact in the sense the kernel timings are. Do not firm it up without instrumentation.
     - **Read correctly, that inverts §2148's weighing — but it does not make prefetch free, and the difference is the whole design.** Roughly half the decode step already goes to serialized delivery, bounded independently by the ~20-21 ms idle-link estimate. That is **the size of the prize**. But prefetch's own bytes are *additional* (B=2 offers 96 rows/token = 265 MB against 65-99 demand-miss rows), so the prize is not "delivery is cheap" — it is **conversion**: turning serialized demand-miss delivery into delivery hidden behind compute. Realising it requires overlap, and **E36 structurally cannot measure overlap**, because `post()` never runs concurrently with anything. It confirms; it cannot refute. A budget argument resting on the cost of a *wasted* row (§2148's 0.209 ms) must therefore be re-derived against the value of a *hidden* one, and that re-derivation still needs an overlap fraction nobody has.
     - **Do not cite 0.0645 ms as an overlap ceiling.** Stage B's first physical overlap datum was established as bounded by *eager host dispatch* — five sequential API calls before the matmul launch — not by any GPU dependency. A replay-based measurement of the production path was not built and is flagged unmeasured.
     - **E37, 2026-09-16: the drain mystery is attributed, and the isolated PASS is the artifact.** `_jit_expert_doorbell_module()` (`kernels/ops/moe/expert_doorbell.py:115-116`) is `@functools.cache`'d and called from every `ExpertDoorbellCopier.__init__` (`:337`), so the first copier in a process pays CUDA module load and driver symbol resolution and every later one gets it free. The drain test flips against itself **in one process with no predecessor at all**: cold (`cache miss=1`) `last_polls` 4,020,000 against a 4,000,000 budget, elapsed 1.4108 s, `drain_timeouts` 1; warm (`hit=1`) 782,015 polls (19.5% of budget), 0.2013 s, `drain_timeouts` 0. **The predecessor was never special, only first** — warm `last_polls` agree to 0.008% across three unrelated predecessor configurations (none at all, the faulting `disable` test, and a gather test that never faults or drains). So the alphabetical ordering established earlier was real and *causally irrelevant*; confirming the ordering half of a hypothesis is not confirming the hypothesis. **Production loads the extension once at startup, so every steady-state copier is warm** — meaning the test's isolated pass encodes cold-start timing and asserts it as the drain's general property. Not fixable by teardown: `functools.cache` is process-lifetime and no fixture reaches it. Either warm deliberately or give the test its own subprocess, and in the latter case the assertion must say *cold-start* rather than claim a general property.
     - **What E37 does NOT undermine, contrary to the hypothesis recorded here earlier: `DOORBELL_ROW_MS`.** That constant comes from E32 — *3 rows in 0.628 ms total*. Module load is ~1.2 s. A 0.628 ms measurement cannot contain, let alone be dominated by, a cost roughly 1,900x larger, so 0.209 ms/row is not first-use module loading under any reading. The earlier note in this file suggesting otherwise was wrong and is withdrawn. **What E37 does undermine is narrower and still serious:** "the copy cannot land until the drain retires" is a cold-start property, so every argument resting on *drain semantics* — as opposed to per-row copy time — is unattributed until remeasured warm. §2148's 0.209 ms per mispredicted row is a property of `all_or_nothing` delivery (resolve waits for every posted row), which is a delivery-model choice, not a drain behaviour. That argument survives E37 on its constant and still needs re-deriving against E36's prize, per the entry above.
   - Task 3 prices the oracle bound, scorer cost and a prefix-delivery variant. It runs in shadow (S2) regardless.
   - **Scratch capacity:** posted rows occupy the layer's 10 scratch rows, and uncovered misses need the rest (p90 6 misses per layer). Phase B must cap `budget + residual ≤ scratch rows` with crypto-c9.
3. **In-graph scoring cost is on the main stream.**
   - LLaPor is ≈10 kernels × 47 layers at bs 1, estimated at ~5–9 ms/token, which could eat the whole gain.
   - Task 7 measures it in isolation and Task 6 end to end (S1 − S0). The follow-ups are fused PCA and a fused kernel.
4. **Hot-cache budget.**
   - Phase B's prefetch rows (≈2.64 MiB × B × 48; B = 3 ≈ 380 MiB ≈ 144 slots) and ~100 MB of scorer state come out of `HOT_GPU_MB`, which adds baseline misses.
   - Task 6 keeps the budget equal (scorer state only). Phase B reports slot counts per cell.
5. **Merge overlap with the doorbell branch.**
   - Both edit `environ.py` and `model_runner.py`. The hunks are disjoint as of `56a5920489`: environ 328–333 vs after 348, runner 754–798 vs ~803.
   - The runner edit sits a few lines from their hunk, so a textual conflict is possible after their rework. Task 4 Step 0 re-checks, and a conflict is resolved by re-applying the one-line condition.
6. **Interface drift.** Phase B's interface is TBD. The bank is residency-agnostic ids plus priorities, so B1's adapter absorbs any ids/slots/count or mask shape without touching Phase A.
7. **Shadow must be exact.** An S1 ≠ S0 flip blocks Phase B until it is root-caused.
8. **Timing assumptions.**
   - The pricing uses E27/E28/E32 medians at 4,957 slots; production has 4,180.
   - Recall uses the capture's own residency, so it is faithful. The ms model is approximate (±4 ms/token per E26).
9. **Two runs per cell is preliminary**, against spec §8's five matched runs.
10. **GPU etiquette vs live runs.** Tasks 6 and B2 need production down. Only an approved, coordinated window (no 7867 listener, empty GPU) is the exception.

## Decisions

**Made in this plan:**
- **Phase split:** Phase A touches no copy path. Phase B is an adapter only, with names TBD.
- **Output:** a residency-agnostic bank of int64 ids `[C]` plus float32 priorities. A mask, filtering or a clamp is built in the adapter.
- **Shadow metric:** in-graph `BudgetRecall` at budget B, identical in definition to Task 3's offline `budget_hits`.
- **Env names:** `SGLANG_MOE_EXPERT_PREFETCH_*`, checked for no collision with the doorbell branch.
- **APEX:** runs in shadow. Whether it prefetches live is gated on Task 3's saved-wait pricing (oracle bound, scorer cost, delivery variant), not on whether a row fits the window.
- **Off-path proof:** exact logprobs against `7de955329a`. Scoring-on proof: exact logprobs against scoring off.

**Need the user (asked at the named step):**
1. The production-down window for Task 6 (≈2.3 h; +0.8 h APEX S2; +0.75 h Task 7 benches), and later for B2 (≈3–3.5 h).
2. On a Task 3 NO-GO: whether to still run Task 6 (it measures scoring cost and validates the bank) or stop Phase A after Task 5.
3. Whether to include the APEX shadow arm S2.
4. The live budget B for B2, if Task 3's gap and overlap argmaxes differ and Task 7 is inconclusive.

## Self-review notes

- **Requirement coverage:**
  - model-agnostic checkpoint loading → Task 1;
  - in-graph forward with no host syncs → Tasks 2 and 4, with sync-debug and replay tests;
  - candidate ids/priorities as device tensors → Task 2 bank and the Phase A output contract;
  - shadow and offline recall → Task 3 (offline), Tasks 4 and 6 (live shadow);
  - no copy path in Phase A → Global Constraints plus Task 4 Step 0;
  - Phase B interface TBD → Design and B1;
  - 4-cell matrix and live tok/s → B2;
  - window findings kept → Design, Task 3 pricing, Task 7.
- **Project rules:**
  - `msgspec.Struct` for new containers (`LlaporCheckpoint`, `ApexCheckpoint`, `PrefetchSettings`);
  - no `getattr`/`hasattr` in new code;
  - envs in `environ.py` next to the predictor block;
  - one runner condition;
  - commits stage named paths only.
- **Names are consistent across tasks:**
  - `load_prefetch_checkpoints`;
  - `LlaporScorer`/`ApexScorer`;
  - `PrefetchCandidateBank.ids_for`/`scores_for`;
  - `BudgetRecall.observe`/`snapshot`;
  - `PrefetchScoring.build`/`from_checkpoints`/`features_for`/`targets`/`next_target`/`bank`/`metrics_record`;
  - `budget_hits`, `doorbell_saving_ms`, `side_stream_ready_rows`;
  - `SGLANG_MOE_EXPERT_PREFETCH_{PREDICTOR,MODEL_DIR,CANDIDATES,BUDGET,APEX_TAU}`.
- **Specified by contract, not code:** Task 7's benches (they reuse E27's harness, read at implementation time) and Phase B (its interface is TBD by design).
