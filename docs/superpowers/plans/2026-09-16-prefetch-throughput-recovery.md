# MoE Expert-Prefetch Throughput Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current always-on one-row expert prefetch experiment into a measured, selectively enabled path that improves decode throughput at the production miss regime, or reject it with enough evidence to redirect effort to cache residency.

**Architecture:** First separate scorer cost, empty fork/join cost, and payload cost with graph-equivalent controls. Then add a device-side, width-one delivery decision that can publish the coherent no-post state (`expert_id=-1`, `count=0`), move LLaPor's pull launch behind the source layer's demand transfer, and enable only layers/score regions whose measured useful-pull precision pays for their traffic and join exposure. Keep shadow recall and delivery telemetry independent, and preserve the current demand path as the default and rollback.

**Tech Stack:** Python 3.11, PyTorch CUDA graphs and streams, SGLang MoE expert streaming, TVM-FFI/JIT CUDA transfer kernels, pytest/unittest, JSONL experiment records, NVIDIA Nsight Systems where available.

**Spec:** `docs/superpowers/plans/2026-09-16-side-stream-expert-pull-handoff.md`, amended by `docs/superpowers/plans/2026-09-16-prefetch-plan-completion-handoff.md` and the approved recovery design recorded in this plan's Decision section.

## Global Constraints

- Production port 7867 may be stopped only with explicit user approval; never relaunch it unless the user asks.
- Coordinate every GPU run with the owner of the concurrent Stage C work, take `/data/models/slang/nvfp4-work/cc-gpu.lock`, and record the GPU process census before and after timing.
- Run every divix01 CPU job under `taskset -c 0-63` with `OMP_NUM_THREADS=32 MKL_NUM_THREADS=32`; cores 64-71 are reserved.
- Never kill a process not started by this experiment.
- Keep doorbell delivery off. Its behavior and known test failures are outside this recovery plan.
- `model_runner.py` remains orchestration-only and frozen unless the plan is explicitly amended after reading the repository's large-class guidance.
- Add environment variables only through `python/sglang/srt/environ.py` and follow the repository's environment-variable conventions.
- Use `msgspec.Struct`, not dataclasses, for new runtime configuration structures.
- Preserve CUDA-graph-safe stable addresses and introduce no per-token host read or synchronization.
- Initial performance scope is ordinary BS1 decode, TP=PP=DP=1, Qwen NVFP4, 48 MoE layers, 512 experts/layer, top-k 10.
- Use the production cache size `SGLANG_MOE_HOT_GPU_MB=10240` for the decision run. Label every result with its observed miss regime.
- Qualify test counts by file and retain the two known doorbell failures as pre-existing evidence, not new failures.
- Stage and commit only named files. Never use `git add -A`, `git add .`, or `git stash`; never stage `.omc/`, `.superpowers/`, or the experiment log owned by another session.
- Do not push to `origin`. Ask before the first push to `shared` in a session.

---

## Decision and hypotheses

The current design is not assumed to be a viable production design. It combines a measured scoring tax with one unconditional 2,764,808-byte pull per target layer and a mandatory join before demand delivery.

The experiment must distinguish these terms:

| Arm | Scoring | Pull graph | Payload | What its delta identifies |
|---|---|---|---|---|
| B | Off | Off | Off | Fused demand-only baseline |
| C | On | Off | Off | `C-B`: scorer and selection cost; shadow recall off |
| Cr | On | Off | Off | `Cr-C`: shadow-recall instrumentation cost; diagnostic only |
| N | On | On | Count zero | `N-C`: fork/join, slot, plan, and accounting cost |
| D | On | On | Always one row | `D-N`: unconditional payload, contention, and useful-row overlap |
| E | On | On | Gated zero/one row | Net candidate after scheduling and gating changes |

The following hypotheses are falsifiable and ordered:

1. Unconditional payload is negative because useful pulls only move bytes earlier while wrong pulls add bytes.
2. LLaPor loses additional time because its pull currently competes with the source layer's necessary demand transfer.
3. Per-layer graph/event/accounting overhead is material even with zero payload.
4. A subset of layer/score regions has sufficient useful-pull precision and overlap to produce a net gain.
5. If hypothesis 4 is false at the production miss regime, predictor work stops and the next investment is reducing demand misses through residency/admission policy.

### Predeclared gates

- Two passes per arm are exploratory only. A shipping claim requires three interleaved paired passes with reversed order and the fused baseline rerun in the same session.
- Do not optimize payload scheduling until B/C/N/D has been measured at `HOT_GPU_MB=10240`.
- Continue from D to gating only if telemetry is internally consistent (`posted = useful_posts + wasted`, no negative deltas, physical row accounting matches demand plus side-pull rows) and at least one layer or score band has nonzero observed useful-pull precision.
- Continue to off-main-stream scoring only if gated delivery E improves on N but remains below B by an amount consistent with the measured `C-B` scorer tax.
- Accept prefetch only when the final gated arm produces at least 5% median decode tok/s improvement over B, every one of three paired passes is positive, a session-clustered bootstrap 95% confidence interval for the paired per-turn improvement is entirely above zero, p95 decode latency stays within the matched baseline's repeatability envelope, and the answer-level correctness gate passes.
- A result below those gates is a measured rejection, not a request for another unpriced optimization.

---

### Task 1: Add graph-equivalent no-payload control and truthful delivery telemetry

**Files:**
- Modify: `python/sglang/srt/environ.py:370-383`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/runtime.py:143-183`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py:40-227`
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py:758-803,1679-1734,1868-1916`
- Modify: `test/registered/unit/layers/moe/test_expert_prediction_runtime.py:215`
- Modify: `test/registered/unit/layers/moe/test_expert_prefetch_pull.py`
- Modify: `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py:243-280`
- Modify: `test/registered/unit/layers/moe/test_expert_hot_cache.py:762-910`

**Interfaces:**
- Produces: `SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE` with values `off`, `count_zero`, and `always`; default `off`, resolved once into runtime field `pull_mode: str` instead of `enable_pull: bool`.
- Produces: `SGLANG_MOE_EXPERT_PREFETCH_SHADOW_RECALL`, default true for compatibility, so matched performance arms can disable `BudgetRecall` independently of scoring and pull outcome telemetry.
- Produces: coherent pull-plan publication: no post is always `expert_ids[0] == -1` and `count[0] == 0`; real post is always `expert_ids[0] >= 0` and `count[0] == 1`.
- Produces: metrics fields `side_pull_posted_rows`, `side_pull_useful_posts`, `side_pull_wasted_rows`, `side_pull_covered_routes`, `side_pull_residual_routes`, and `side_pull_useful_precision`.
- Preserves: existing `SGLANG_MOE_EXPERT_PREFETCH_PULL` as a compatibility alias during this plan using this exact resolution table:

| New variable | Legacy variable | Effective mode |
|---|---|---|
| absent | absent or explicit false | `off` |
| absent | explicit true | `always` |
| explicit mode | absent | explicit mode |
| `off` | explicit false | `off` |
| `always` | explicit true | `always` |
| any other pair with both explicit | startup error |

Explicit presence is detected only inside `environ.py`, not by scattered direct environment reads.

- [ ] **Step 1: Write failing configuration tests**

Add cases proving the default is off, all three modes parse, unknown values fail at startup, pull mode requires a predictor, and contradictory compatibility settings fail clearly.

- [ ] **Step 2: Run the configuration test and verify the new mode is absent**

Run on divix01:

```bash
taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py \
  -q
```

Expected: the new cases fail because `SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE` is not defined or validated.

- [ ] **Step 3: Write failing coherent-count-zero CUDA tests**

Add a captured replay test using the real `PrefetchPuller` that changes `should_post` across `[0, 1, 0, 1]` and asserts:

```python
assert (plan.expert_ids.item(), plan.count.item()) == (-1, 0)  # rejected
assert posted == 0 and wasted == 0
assert demand_rows == original_demand_rows
assert dedicated_slot_unchanged
```

For accepted pulls, assert `(expert_id, count) == (candidate, 1)`. The test must exercise the same post/join graph nodes in both states.

- [ ] **Step 4: Run the count-zero tests and verify they fail against `count.fill_(1)`**

Run:

```bash
taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/layers/moe/test_expert_prefetch_pull.py \
  test/registered/unit/layers/moe/test_expert_prefetch_scoring.py \
  -q
```

Expected: the new control reports a posted row or excludes a demand row because the existing plan always publishes a valid candidate and count one.

- [ ] **Step 5: Implement the delivery mode and coherent sentinel**

Parse the mode once at setup. In `PrefetchPuller.post_target`, publish both tensors from a device boolean rather than changing only `count`:

```python
candidate = self.bank.ids_for(target_layer)[:1]
valid = self._should_post[target_layer]
plan.expert_ids.copy_(torch.where(valid, candidate, candidate.new_full((1,), -1)))
plan.count.copy_(valid.to(torch.int32).reshape(1))
self._pipeline.post_target(target)
```

For `count_zero`, keep `valid` false while still executing `post_target` and `join_target`. Ensure `predicted_expert_for`, demand-row skipping, and `pull_outcome_counts` observe the same sentinel.

- [ ] **Step 6: Write failing telemetry-schema tests**

Extend hot-cache tests with a cumulative snapshot containing one useful post, one wasted post, two covered routes, and three residual routes. Assert physical and route quantities remain separate and precision is `(posted - wasted) / posted`.

- [ ] **Step 7: Implement truthful telemetry and remove stale reset comments**

Expose the existing counters without adding new hot-path reductions. Compute `useful_posts = posted - wasted` and `useful_precision` only during periodic host aggregation. Correct comments in `serving/runtime.py` that still say pull counters are never reset; `discard_graph_capture_routes` already resets them.

Implement the independent shadow-recall switch in the same configuration pass. When false, do not allocate `BudgetRecall`, do not invoke `BudgetRecall.observe`, and report `shadow_recall_enabled=false`; pull outcome counters remain active.

- [ ] **Step 8: Run focused tests**

Run the four files from Steps 2, 4, and 6. Expected: all pass, with no host synchronization introduced in graph replay.

- [ ] **Step 9: Commit Task 1**

```bash
git commit -m "feat(moe): add zero-payload prefetch control and truthful pull metrics" -- \
  python/sglang/srt/environ.py \
  python/sglang/srt/layers/moe/expert_prediction/runtime.py \
  python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py \
  python/sglang/srt/layers/moe/expert_hot_cache.py \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py \
  test/registered/unit/layers/moe/test_expert_prefetch_pull.py \
  test/registered/unit/layers/moe/test_expert_prefetch_scoring.py \
  test/registered/unit/layers/moe/test_expert_hot_cache.py
```

---

### Task 2: Build the decomposition benchmark and run B/C/Cr/N/D before optimizing

**Files:**
- Create: `benchmark/expert_delivery/benchmark_prefetch_pipeline.py`
- Create: `benchmark/expert_delivery/benchmark_prefetch_scorers.py`
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/runtime.py`
- Modify: `scripts/expert_prediction/run-shadow-server.sh`
- Modify: `scripts/expert_prediction/prefetch/summarize_ab.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py`
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/serving/calibration.py`
- Modify: `test/registered/unit/layers/moe/test_expert_prediction_runtime.py`
- Create: `test/registered/unit/layers/moe/test_prefetch_pull_calibration.py`
- Create: `docs/superpowers/experiments/2026-09-16-expert-prefetch-throughput-recovery.md`

**Interfaces:**
- Consumes: Task 1 pull modes and telemetry names.
- Produces: real-geometry measurements for count 0/1 fork-to-copy, copy duration, join exposure, and consumer-ready time.
- Produces: a repeatable B/C/Cr/N/D live matrix with exact flags and provenance.
- Produces in separate non-timed profiling runs: a versioned `pull-calibration.json` containing per-layer 256-bin top-score and score-margin histograms with opportunity, source-eligible, target-useful, target-wasted, and physical-demand-row totals plus commit, predictor, checkpoint checksum, cache size, session-set checksum, shape/top-k, and bin-edge provenance. Additive device counters are read only at the profiling run's normal metrics flush/end, never per token.
- Produces: per-layer scorer-plus-selection CUDA time from `benchmark_prefetch_scorers.py` using captured representative features, plus per-target count-zero post/join/accounting time from `benchmark_prefetch_pipeline.py` with every other target disabled.

- [ ] **Step 1: Write parser tests for the extended summarizer**

Use synthetic result and metrics JSONL to assert median tok/s, p50/p95 per-turn decode ms/token, truncation count, observed miss rate, resident slots, posted/useful/wasted rows, and useful precision. Add a paired session-cluster bootstrap with fixed seed `20260916` and 10,000 resamples; assert its 95% interval on a hand-checkable fixture. Define the baseline p95 repeatability envelope as the inclusive minimum/maximum p95 per-turn decode ms/token across same-session B passes. Reject mixed commits or cache sizes in one comparison.

- [ ] **Step 2: Implement summary parsing without changing serving code**

Extend `summarize_ab.py` with the exact grammar `arm=results.jsonl:prediction.metrics.jsonl:hot-cache.metrics.jsonl:run-manifest.json`. Empty middle fields are permitted, for example `B=results.jsonl::hot-cache.metrics.jsonl:run-manifest.json`. The manifest supplies commit, flags, cache size, predictor, pass, session IDs, and provenance; each other file has one fixed role. Do not guess a source by JSON shape.

- [ ] **Step 3: Add the real-geometry pipeline microbenchmark**

Reuse the CUDA-event methodology in `test_expert_gpu_pull.py::test_physical_overlap_of_side_pull_and_origin_compute`. Measure count zero and count one with registered host memory and the production row segments. Report separate intervals for ready-to-copy-start, copy, join exposure, residual demand, and consumer ready. Validate copied bytes before printing timings.

- [ ] **Step 4: Add explicit launch provenance**

Update the server wrapper to pass and print pull mode, shadow-metric state, calibration-histogram state, candidate width, budget, cache size, predictor, checkpoint directory, fused-plan state, commit, and output paths. A run whose startup log lacks the exact requested mode is invalid, not a zero result.

Use this exact arm configuration for each predictor; all unlisted scheduler, generation, residency, and prompt settings remain identical:

| Arm | Fused plan | Predictor | Pull mode | Shadow recall | Calibration | Candidates | Budget | Hot GPU MB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| B | 1 | empty | `off` | 0 | 0 | 16 | 2 | 10240 |
| C | 1 | LLaPor or APEX | `off` | 0 | 0 | 16 | 2 | 10240 |
| Cr | 1 | same as paired C | `off` | 1 | 0 | 16 | 2 | 10240 |
| N | 1 | same predictor | `count_zero` | 0 | 0 | 16 | 2 | 10240 |
| D | 1 | same predictor | `always` | 0 | 0 | 16 | 2 | 10240 |

- [ ] **Step 5: Write and implement the calibration-histogram contract**

Implement `PullCalibrationHistogram` with fixed device counters shaped by target layer and 256 uniform bins over score/margin range `[0,1]`. At score time, stage the current candidate ID, top score, margin, and source eligibility in stable per-target device tensors. At target routing inside `_gather_graph`, update the corresponding opportunity/useful/wasted and physical demand-copy-row bins after the planner has produced its device row count. All writes are additive and graph-captured; the normal metrics snapshot reads them at profiling-run end. Test schema versioning, exact bin boundaries, all-resident/no-candidate behavior, useful versus wasted target outcomes, route-versus-physical-row separation, and reset after graph capture.

Add `SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION=EnvBool(False)`. Extend the launcher with required run kind `timed` or `profiling`; it records the choice in the manifest and refuses calibration in a timed B/C/Cr/N/D launch before starting the server.

- [ ] **Step 6: Verify the benchmark, scorer timing, recorder, and summarizer on divix01**

Print the imported `sglang.__file__` first. Run the pipeline and per-layer scorer benchmarks under the lock with a clean GPU census. Expected: byte validation passes; count zero reports zero payload bytes; count one reports exactly one physical row; every enabled target has a scorer-plus-selection time and an isolated count-zero control time.

- [ ] **Step 7: Run the exploratory B/C/Cr/N/D matrix**

Use the same prompt subset, generation limits, scheduler, residency policy, and exact flag table above. Interleave pass 1 `B→C→Cr→N→D` and pass 2 `D→N→Cr→C→B`, once for LLaPor and once for APEX. Capture pull telemetry, per-layer scorer/control timings, manifests, and startup logs for every arm. Calibration is disabled in every timed arm.

After the timed matrix, run separate non-timed calibration launches for the training and held-out session sets with identical model/cache/residency settings and `SGLANG_MOE_EXPERT_PREFETCH_CALIBRATION=1`. Record separate session-set checksums and never include these launches in throughput comparisons.

Each profiling launch uses fused plan 1, the selected predictor, pull mode `off`, shadow recall 0, candidates 16, budget 2, hot GPU MB 10240, calibration 1, and run kind `profiling`. The calibration observer attaches independently of `PrefetchPuller`: it evaluates candidate usefulness counterfactually from target routing and reads the planner's physical demand-row count without launching speculative payload.

- [ ] **Step 8: Record the decomposition and apply the gate**

The experiment record must state:

- `C-B` scorer/selection cost;
- `Cr-C` shadow-recall instrumentation cost;
- `N-C` empty fork/join/accounting cost;
- `D-N` payload/traffic effect;
- `D-B` net effect;
- observed misses/layer and useful precision;
- whether payload ever completes before join;
- whether each conclusion is measured, inferred, or still unobserved.

If telemetry is inconsistent, stop here and fix Task 1. If no layer or score band records useful posts, reject prediction for this workload and skip Tasks 3-6.

- [ ] **Step 9: Commit tooling and the experiment record separately**

Commit tooling first, then the immutable result record in a second commit. Do not combine benchmark code with conclusions drawn from it.

---

### Task 3: Move LLaPor payload launch behind source demand delivery

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py:529-537,679-757`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py:271-348`
- Modify: `test/registered/unit/layers/moe/test_expert_prefetch_runtime.py`
- Modify: `test/registered/unit/layers/moe/test_expert_graph_gather.py:1173-1260`

**Interfaces:**
- Produces: `PrefetchPuller.prepare_target(target_layer)` to publish the candidate and gate result without launching payload.
- Produces: source-streamer attribute `prefetch_post_after_demand: Optional[Callable[[], None]]`, initialized to `None`, whose callback invokes `post_target(target_layer)` after the source layer's demand backend has completed.
- Preserves: APEX posting at its same-layer PRE_MIXER hook.

- [ ] **Step 1: Write an ordering test for LLaPor**

Record these events through the real graph-gather seam:

```text
score target L+1
prepare target L+1
source L demand post/resolve
post target L+1
```

Assert the order exactly. Add a separate APEX test asserting score/prepare/post still occurs from PRE_MIXER and does not wait for another layer's gather.

- [ ] **Step 2: Run the ordering test and observe immediate post failure**

Expected: current `_on_write` posts before source demand delivery.

- [ ] **Step 3: Split prepare from post**

`prepare_target` writes the stable plan buffers and validity state on the current stream. `post_target` only forks the already-prepared plan. Reject double post or post-before-prepare in eager tests without adding a replay-time host branch.

- [ ] **Step 4: Wire the LLaPor source-to-target callback**

Use the checkpoint-derived `source → target` mapping; never assume `source + 1`. During `PrefetchScoring.from_checkpoints`, attach `prefetch_post_after_demand` to every source-layer streamer, including LLaPor source layer 0, with a callback that closes over the mapped target. Invoke it at the end of `_gather_graph`, after current-layer device and host demand copies and before returning tensors for expert compute. Use this new attribute rather than the retired eager `next_layer_prefetch` coordinator.

- [ ] **Step 5: Verify graph replay and row-skip correctness**

Run prediction runtime, prefetch pull, graph gather, and expert GPU pull tests. Include changing candidates across at least six graph replays and both useful and wrong predictions.

- [ ] **Step 6: Run a D arm against the unchanged N control**

Use the same-session protocol. The decision statistic is the paired change in `D-N`, plus trace evidence that LLaPor payload no longer overlaps source demand transfer.

- [ ] **Step 7: Commit Task 3**

Commit only the two runtime files and their tests. Record the measurement in a later experiment-doc commit.

---

### Task 4: Add static layer gating, then a conditional calibrated width-one selector

**Files:**
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/runtime.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py:10-58`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py`
- Modify: `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`
- Modify: `test/registered/unit/layers/moe/test_expert_prediction_graph.py`
- Create: `scripts/expert_prediction/prefetch/calibrate_pull_gate.py`
- Create: `test/registered/unit/layers/moe/test_prefetch_pull_gate_calibration.py`

**Interfaces:**
- Produces: `SGLANG_MOE_EXPERT_PREFETCH_TARGET_LAYERS=EnvStr("")`, a comma-separated target-layer allowlist; empty means every checkpoint target. It resolves to `target_layers: frozenset[int]`, filters checkpoint/scorer construction, and rejects duplicates, nonintegers, and targets absent from the checkpoint set.
- Extends `SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE` with `gated` in this task and produces `SGLANG_MOE_EXPERT_PREFETCH_GATE_FILE=EnvStr("")`; empty means static-only behavior for `always` mode. `gated` mode requires a file. The artifact's predictor, checkpoint checksum, source commit compatibility, cache size, and layer set are validated at startup. Each enabled layer records `gate_feature` as exactly `top_score` or `score_margin` plus its threshold; runtime resolves these to stable per-target feature-code and threshold tensors. The effective layer set is the intersection of the explicit allowlist and artifact-enabled layers; an empty intersection is a startup error.
- Produces first: a static layer allowlist selected from non-timed production-regime profiling on a training split and checked on held-out sessions.
- Produces only after the confidence-calibration gate passes: a versioned JSON gate artifact containing predictor name, checkpoint checksum, layer allowlist, per-layer score threshold, source commit, cache size, and calibration miss regime.
- Produces: `DeliveryCandidateBank` with stable device tensors `expert_ids int64[target,1]`, `scores float32[target,1]`, and `eligible bool[target,1]`.
- Consumes: Task 1 coherent sentinel and Task 2 production-regime score/outcome samples.

- [ ] **Step 1: Add a static layer allowlist and test that disabled layers are not scored**

The allowlist must control checkpoint loading and scorer construction, not merely suppress payload after paying every layer's scoring cost. Test empty, one-layer, and multi-layer lists; reject duplicates, malformed IDs, and layers absent from the selected checkpoint set. Preserve the checkpoint-declared source-to-target mapping for LLaPor. Add and parse `gated` here. Test static-only `always + TARGET_LAYERS + empty GATE_FILE`, and test that `gated` rejects an empty, mismatched, malformed, or unknown-feature artifact.

- [ ] **Step 2: Select the layer cohort from Task 2 data**

On the training profiling histogram, rank layers using useful-pull precision, wasted physical rows, physical demand-copy rows, measured lead time/join exposure, isolated per-layer scorer-plus-selection cost, and isolated per-target count-zero control cost. Evaluate nested cohorts such as top 4, 8, 16, and all enabled layers in separate timed held-out runs using `TARGET_LAYERS` and no gate file. Retain the smallest cohort within the measurement uncertainty of the best cohort. If no held-out cohort beats B, reject per-token prefetch and skip the remaining steps in this task.

- [ ] **Step 3: Write confidence-calibration tests with a hand-checkable trace**

For each layer and each supported feature (`top_score`, `score_margin`), enumerate the 256 recorded bin boundaries. Calculate physical traffic as `posted physical rows + physical demand-copy rows`, useful precision as `(posted-wasted)/posted`, and predicted net value using measured per-layer scorer-plus-selection, per-target count-zero control, join exposure, and row costs from Task 2. Never substitute residual route count for physical demand-copy rows. The calibration script must reject records outside BS1/top-k-unique scope unless they carry an independently measured physical-row field. Select the feature and threshold pair that maximizes positive expected value on the training histogram; there is no separate minimum-precision parameter. Layers with no positive training cell are disabled, and the resulting artifact is accepted only if the same feature/threshold remains positive on the held-out profiling histogram and timed E run.

- [ ] **Step 4: Implement the offline calibrator**

The calibrator must reject predictor/checkpoint/cache mismatches and emit all provenance needed to reproduce the threshold table. It must not fit and evaluate on the same sessions; use the existing holdout/validation split. Do not create runtime confidence thresholds unless top score or score margin clearly separates useful from wasted pulls on held-out data; static layer gating remains a valid terminal candidate.

- [ ] **Step 5: Write failing device-selector tests**

Cover top-score and margin gates above/below threshold, disabled layer, all-resident experts, NaN/inf scores, deterministic lower-ID ties, unknown feature rejection, and eligibility changes across graph replay without Python. On rejection, assert `expert_id=-1,count=0`; never infer eligibility from a fallback ID.

- [ ] **Step 6: Implement masked width-one selection**

For BS1 delivery, use finite/residency masking and deterministic top-1 selection. Keep the existing width-W bank for shadow analysis. Do not silently change batch-shared semantics for batch sizes greater than one; reject or fall back outside the declared BS1 scope.

- [ ] **Step 7: Add predictor-specific top-one fast paths**

For LLaPor, exploit sigmoid monotonicity only after testing that the selected nonresident ID, selected gate feature value, and zero/one post decision match the current full scorer across threshold boundaries at BS1. For APEX, require the same ID-feature-decision parity before removing softmax/CDF/rank/sort work; a high-ranked resident expert or current depth mask may invalidate ranker-logit argmax equivalence. Each fast path is independently conditional and must be skipped if parity fails. Shadow mode retains the full scorer.

- [ ] **Step 8: Preserve the independently switchable shadow-recall path**

Retain Task 1's independent switch through the specialized selector. Delivery performance arms and their matched C control keep shadow recall off while retaining pull outcome telemetry. The diagnostic Cr arm alone enables shadow recall so `Cr-C` prices it separately. Test that the width-one path does not allocate or invoke `BudgetRecall` when disabled.

- [ ] **Step 9: Run CPU, CUDA, replay, and parity suites**

Require exact selected-ID parity for accepted BS1 inputs and no host synchronization. Run the relevant prediction scoring, prediction graph, prefetch pull, route planner, and graph gather files.

- [ ] **Step 10: Commit layer gating, calibration, and runtime selection separately**

Commit the static layer allowlist/tests first. Commit the offline calibrator/report second, and only if its gate passes commit the device selector/scorer/runtime threshold changes third. Each decision must be independently reviewable and revertible.

---

### Task 5: Measure and remove secondary per-layer overhead only if the gates justify it

**Files:**
- Modify conditionally: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py:40-90,210-227`
- Modify conditionally: `python/sglang/srt/layers/moe/expert_route_plan.py`
- Modify conditionally: `test/registered/unit/layers/moe/test_expert_prefetch_scoring.py`
- Modify conditionally: `test/registered/unit/layers/moe/test_expert_route_plan.py`

**Interfaces:**
- Gate: execute only if E improves on D or N but per-layer bookkeeping remains material in the count-zero trace.
- Produces: fused planner outputs for covered routes, residual routes, wasted flag, and posted row without a second remap pass.

- [ ] **Step 1: Capture a trace proving the secondary kernels are material**

Name the `route_covered_residual`, reduction, and counter kernels and report their total token cost. If below measurement noise, skip this task and record the skip rationale.

- [ ] **Step 2: Write a planner parity test**

Across resident hit, useful pull, wrong pull, count zero, duplicate routes, and changing replay state, compare the fused outputs bit-for-bit with the current reference functions.

- [ ] **Step 3: Fuse outcome accounting and remove the redundant remap**

Preserve route-vs-physical counter meanings. Do not change demand row order or scratch-slot assignment.

- [ ] **Step 4: Re-run N and E**

Accept the change only if the trace confirms the kernels disappeared and repeated end-to-end timing improves beyond run-to-run spread.

- [ ] **Step 5: Commit or discard**

Commit only on measured improvement. Otherwise revert this task's named changes and retain the negative result in the experiment record.

---

### Task 6: Consider off-main-stream scoring only after gated delivery proves value

**Files:**
- Modify conditionally: `python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py`
- Modify conditionally: `python/sglang/srt/layers/moe/expert_gpu_pull.py`
- Modify conditionally: `test/registered/unit/layers/moe/test_expert_prediction_graph.py`
- Modify conditionally: `test/registered/unit/layers/moe/test_expert_gpu_pull.py`

**Interfaces:**
- Gate: execute only when E beats N, E remains below B, and `C-B` accounts for the remaining loss.
- Produces: a captured `selection_ready` event separating feature publication from side-stream scoring, followed by the existing payload-done event.

- [ ] **Step 1: Write dependency tests before moving scoring**

Delay scoring deterministically and prove target planning cannot read a stale candidate, payload cannot start before scoring publishes the candidate, and the consumer cannot read the dedicated row before payload completion.

- [ ] **Step 2: Capture score, selection, and pull on the side stream**

Record a feature-ready event on the main stream, wait on it on the side stream, score/select/publish the plan, record `selection_ready`, then copy and record payload done. Before `_gather_graph` reads `predicted_expert_for` and performs demand-row planning, the target main stream must wait on `selection_ready`; it waits on payload done only at the existing consumption boundary. This permits scoring overlap before planning without allowing a stale prediction to remove a demand row. Preserve stable plan addresses.

- [ ] **Step 3: Verify capture and six changing replays**

Run useful, wrong, rejected, and all-resident cases with changing features. Require exact candidate/remap/bytes parity with the main-stream reference.

- [ ] **Step 4: Re-run C/N/E with traces**

Accept only if scorer work overlaps, no new target stall replaces it, and end-to-end E moves toward or beyond B. Otherwise discard this optional task.

---

### Task 7: Final matched evaluation and decision record

**Files:**
- Modify: `scripts/expert_prediction/prefetch/summarize_ab.py` only if the frozen schema reveals a parser defect
- Modify: `docs/superpowers/experiments/2026-09-16-expert-prefetch-throughput-recovery.md`
- Modify: `docs/superpowers/plans/2026-09-15-moe-expert-prefetch-live.md`
- Modify: `docs/superpowers/plans/2026-09-16-side-stream-expert-pull-handoff.md`
- Modify: `MOE_EXPERT_TRANSFER.md` only after its location and ownership are approved

**Interfaces:**
- Produces: a go/no-go decision for LLaPor and APEX separately.
- Produces: a reproducible final matrix and rollback instructions.

- [ ] **Step 1: Freeze the candidate commit and configuration**

Record commit, predictor checkpoint checksum, gate artifact checksum, flags, cache slots, scratch rows, scorer bytes, CUDA/PyTorch versions, GPU census, prompt IDs, scheduler settings, and residency policy.

- [ ] **Step 2: Run three paired passes**

Interleave and reverse B/C/N/D/E order across passes. Warm every arm identically. Record answer-level correctness only where both finish reasons are `stop`, plus truncation and same-arm nondeterminism controls.

- [ ] **Step 3: Apply the predeclared acceptance gates**

Report median and per-pass tok/s deltas, the fixed-seed 10,000-resample session-clustered bootstrap confidence interval for paired per-turn improvement, p50/p95 per-turn decode ms/token, the inclusive min/max baseline p95 repeatability envelope across same-session B passes, miss regime, useful precision, offered/useful/wasted/residual routes, physical demand and side-pull rows, derived bytes, join exposure, resident slots, VRAM, and errors. Do not substitute budget recall for useful precision or route counts for physical rows.

- [ ] **Step 4: Choose exactly one conclusion**

- **Ship disabled-by-default candidate:** final E meets every acceptance gate. Document its exact supported scope and rollback.
- **Keep experimental:** E is positive but below the 5% or p95 gate. Preserve flags and evidence; do not enable production.
- **Reject per-token prediction:** E fails to beat B repeatably. Disable the path and redirect the next plan to residency/admission changes. Do not add another predictor optimization without new evidence.

- [ ] **Step 5: Update superseded plan statements**

Replace the old four-cell matrix with B/C/N/D/E, mark completed tasks and negative results honestly, correct stale row-size and counter-reset statements, and label every recall figure with its commit and miss regime.

- [ ] **Step 6: Run final verification**

Search changed files for `TODO`, `TBD`, skipped/only tests, placeholder branches, stale `count.fill_(1)` always-on behavior, and unqualified test counts. Run all prediction, pull, route-plan, graph-gather, and hot-cache suites on the validated divix01 tree.

- [ ] **Step 7: Commit the final record and plan updates by named path**

Do not push or relaunch production. Tell the user whether the GPU is free and whether production remains down.

---

## Rollback

The demand-only path remains the reference. A complete runtime rollback is:

```text
SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR=
SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE=off
SGLANG_MOE_EXPERT_PREFETCH_PULL=0
```

Keep `SGLANG_MOE_EXPERT_FUSED_PLAN=1` only if its independent B-vs-current-planner measurement remains positive. Removing prediction and pull settings must restore the same cache allocation, graph nodes, and metrics behavior as the fused demand-only arm.

## Expected outcome

This plan does not assume prefetch will win. Its successful outcome is either a repeatable, gated decode improvement with complete traffic accounting, or a defensible rejection showing which fixed or variable cost prevents a win. In the rejection case, the next optimization target is cache admission/residency because it removes demand rows without predictor scoring, speculative bytes, or per-layer joins.
