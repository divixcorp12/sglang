# DSV4.1 expert prediction and NVMe prefetch: implementation handoff

**Snapshot:** 2026-09-19, `master`, local source `24404a3560`.
**Scope:** a documented experiment and implementation plan. No capture, training, inference, or implementation was performed to create this handoff.
**Primary objective:** predict useful future target experts early enough to bring their weights from NVMe into the bounded RAM cache before demand. This can reduce exposed NVMe wait; it does not make each byte of RAM → VRAM PCIe transfer intrinsically faster.

Use [DSV41_REFERENCE.md](DSV41_REFERENCE.md), especially §§18.4–18.6, as the current experiment record. [DSV4.1 nvme_cache_performance.md](<DSV4.1 nvme_cache_performance.md>) supplies the earlier storage analysis and design background. Its promotion-stall priority and incomplete boundary-arm status have been superseded by the results below. Do not carry its earlier 40–65 ms/token promotion extrapolation into an expected predictor gain.

**Evidence labels:** *measured* denotes an existing experiment; *code observation* denotes behavior established by source; *derived* denotes arithmetic or a trace-based bound; *proposed* denotes work that is not implemented or validated here. Proposed artifact names below describe outputs to produce, not files that already exist.

## Background and the actual DSV4.1 serving problem

The model has **40 target MoE layers**, indexed 0–39, each with **384 routed experts and top-6 selection**, plus one always-active shared expert. The routed expert intermediate width is **2,304**. There are 15,360 routed experts. Their EXL3 weights occupy about **204.5 GB**, and the separate Engram tables occupy about **203 GB**. A raw expert row contains **13,315,596 bytes**; the six streamed tensors contain **13,315,584 bytes**. Use the appropriate representation in byte accounting.

The available host-memory envelope is roughly 90 GB, not enough to hold the model. The measured configuration gives the expert RAM tier **71,680 MiB / 70 GiB**, sufficient for **5,644 rows**, approximately 36.7% of the experts, plus **5 GiB** for Engram. Process state and I/O staging consume additional RAM. The expert VRAM budget is **14,336 MiB**; graph-gather scratch consumes **3,048 MiB**, leaving **888 hot slots**. These are the relevant experiment budgets, not the larger theoretical capacities from early planning.

The cache hierarchy is **inclusive**: a VRAM-hot expert also has a protected RAM copy. VRAM eviction drops the GPU copy and retains RAM. Distinct coverage is therefore not RAM rows plus VRAM rows. Predictions must respect this policy, protect active users, and coexist with demand loading and residency promotions.

**Why RAM prefetch permits more prediction error than VRAM prefetch:** the RAM tier offers more retention capacity and can keep a row for several later layers or tokens without immediately spending PCIe bandwidth or a scarce GPU slot. A prediction can therefore be useful beyond its first intended demand. However, a wrong prediction still consumes roughly 13.3 MB of disk traffic and may evict useful RAM contents. The tolerance is greater in placement and retention, not unlimited in NVMe bandwidth. Predict broadly if scoring is cheap; admit reads selectively. Predictions remain advisory and never alter native expert selection.

Option C serves decode with a native CPU io_uring reader and device-side post/wait kernels inside a breakable CUDA graph. A layer routes, requests missing RAM rows, waits for the RAM service, gathers RAM rows into VRAM, and runs the fused EXL3 MoE. Prefill is eager; the current EXL3 graph route is BS1. The native service's approximately 10 ms missed-row latency includes read and preparation/publication work; it is not a pure device-read measurement.

The measured host has **188 GB RAM** shared with other workloads and an **RTX 5090 with about 32 GB VRAM** on a Gen3 host. The measured nvme2 path is **Gen3 x2**, sustaining about **1.64–1.77 GB/s** in direct expert-read tests (**7.5–8.1 ms/row**). RAM→VRAM gather takes approximately **1.055 ms/row**, about **12 GB/s** for this payload. The planned nvme1 expert path is Gen3 x4 but shares a QLC drive with other workloads; its loaded serving benefit remains to be measured. These are different bottlenecks and measurement conditions, not interchangeable bandwidth figures.

### DSV4.1 representation and architecture facts

| Property | Consequence for prediction |
|---|---|
| Router width **5,120** | Train fresh PCA and predictor weights for this input width. |
| Four mHC residual streams, logically **4 × 5,120** | The raw residual tensor is not the router input. |
| Weighted mHC combination followed by the FFN normalization | Use the actual normalized **5,120-element gate input** as the canonical LLaPor feature. |
| Native router: top-6 of `sqrt(softplus(Wx)) + bias` | Preserve the exact implementation's score, bias, dtype, and selection semantics for self-check and gate baselines. |
| All target layers are routed MoEs; no hash-routed target layers | Layer/horizon heads can cover target IDs 0–383, subject to valid source/target pairs. |
| Engram at layers **1 and 14** | Evaluate predictions crossing or entering these layers separately. |
| Compression layout changes at **19 → 20** | Treat the encoder/decoder transition as its own calibration stratum. |
| Optional bounded prefill replay starts at layer **21** with the current KV-source list | Later layers may process only the request's tail tokens; capture and join only valid per-layer rows. This is distinct from the architecture boundary at layer 20. |
| DSpark taps late target layers **37–39** | Those taps are completed states averaged across streams, suitable for future-token proposals rather than early same-token routing. |

The canonical feature is the tensor that actually reaches the gate after the weighted combination and normalization. It is **not** the mean of the four residual streams, a flattened 20,480-vector, an unnormalized combined state, or a DSpark late-state tap. The source may use fused operations that reuse a precomputed normalized result; select the semantic gate input rather than relying on one eager Python call site.

The routing correction bias affects **expert selection**, while native combination weights use the selected unbiased scores, normalization, and routed scaling (configured as 1.5). Capture the weights the runtime actually uses; do not substitute biased ranking scores. LLaPor's multilabel sigmoid/BCE objective remains appropriate for predicting inclusion in the native top-6 set even though the native router uses sqrt-softplus. Weighted/focal losses and sigmoid outputs do not by themselves produce calibrated admission probabilities.

DSV4.1 uses CSA2 attention with compressed-KV and index reuse. The 20-layer causal encoder/20-layer decoder split is at layer 20; shared KV-source layers are **2, 8, 14, and 20**. Shared attention state does not imply shared expert identities or a proven change in predictability. When `enable_decoder_swa_bounded_replay` is enabled, `late_layer_start = max(kv_source_layer_ids) + 1`, hence **21**: later layers run on a bounded extend-token tail. Record the actual setting and per-layer valid positions; do not assume every prefill token has all 40 layer records. See [`deepseek_v4.py`](python/sglang/srt/models/deepseek_v4.py), the `late_layer_start` initialization and `enter_late_layer_tail` path.

See [`deepseek_v4.py`](python/sglang/srt/models/deepseek_v4.py), the FFN `_hc_combine` path around lines 3647–3690 and `op_mhc_post_attn_pre_mlp`; [`DSV41_REFERENCE.md`](DSV41_REFERENCE.md), §2; and the late DSpark capture around lines 4815–4822.

### What has already been measured

The Phase 3b option C corpus reports **2.781 tokens/s**, compared with **2.780** using the previous-token advisory predictor. The newer boundary study reports:

| Arm | Mean decode tokens/s | VRAM misses/token, G | RAM misses/token | Current interpretation |
|---|---:|---:|---:|---|
| `c32` | **2.823** | **126.9** | **18.52** | Keep the 32-forward decode residency updates. |
| `c0` | **2.729** | **151.8** | **18.98** | Disabling updates worsened the mean; most benefit was in session 0. |
| `casync` | **2.817** | **126.9** | **18.52** | The flag was ineffective for EXL3 on the measured revision. |

Typical measured promotion excess is approximately **40–80 ms per 32-forward boundary**, or **1–2.5 ms/token**, not 40–65 ms/token. The traced **2.1-second burst** was real but not typical; burst frequency across a broader corpus remains unmeasured. Current EXL3 argument validation **rejects** `SGLANG_MOE_HOT_ASYNC_PROMOTIONS=1` instead of silently accepting an ineffective flag. See [reference §18.6](DSV41_REFERENCE.md) and [`expert_stream_requirements_exl3.py`](python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py), lines 52–61.

A separate trace window measured approximately **391 ms/step**: **190 ms** RAM-miss service wait, **128 ms** RAM → VRAM gather, **16.8 ms** compute, roughly **41 ms** amortized burst-related boundary cost, and approximately **11 ms** Engram breaks. This is not the same corpus or a representative promotion-cost estimate. The core conclusion is that transfers dominate and ordinary compute offers little time to hide them.

The existing lookahead probe used graph decode with the EXL3 MoE as an **eager break**, not the fused option C runtime to be deployed. It captured **20,480 records** over 512 tokens and 40 layers. Recomputing each layer's own gate on its recorded input reproduced **20,480/20,480** route sets. This validates that probe's feature semantics, not a future capture implementation.

| Existing probe predictor | All-route recall | NVMe-row recall | Useful NVMe reads/token | Immediately unused NVMe reads/token |
|---|---:|---:|---:|---:|
| L+1 native gate, top-6 | 0.645 | **0.481** | 8.79 | 22.25 |
| L+1 native gate, top-8 | 0.719 | 0.572 | 10.44 | 41.04 |
| L+1 native gate, top-12 | 0.794 | 0.683 | 12.47 | 90.53 |
| L+1 native gate, top-24 | 0.876 | 0.810 | 14.79 | 284.44 |
| L+2 native gate, top-6 | 0.571 | 0.411 | 7.16 | 26.91 |
| L+2 native gate, top-12 | 0.716 | 0.591 | 10.31 | 98.39 |
| Previous-token routes | 0.376 | **0.000** | 0.00 | 0.42 |

Here “NVMe row” means absent from the recorded RAM tier. The probe observed **19.15 RAM misses/token**, or **18.26** over target layers 1–39 that same-token lookahead can reach. The previous token's experts were usually already in RAM, explaining why general route recurrence failed to produce useful cold-row predictions.

Confidence mattered: for L+1, NVMe-tier candidate precision was **0.681** at rank 1, **0.528** at rank 2, **0.375** at rank 3, and **0.193** at ranks 4–6. The probe's heuristic highest-confidence bin held **7.31 candidates/token**, of which **4.52** were useful; the L+2 bin held **8.00**, of which **4.05** were useful. Its score-margin sigmoid is not an already calibrated probability of future expert use.

One layer's lead was estimated at **4.8 ms**, two layers at roughly **10 ms**, against approximately **10.2 ms** per missed row. The reference's **6% L+1 / 10% L+2** improvement figures are estimates, not live prefetch results. Multiple candidates compete for the same lead window; six serial reads took **61.7 ms**. L+4 is an experiment, not an extrapolated guarantee of more savings.

At 13.3 MB/row, L+1 top-6's 22.25 unused reads represent about **296 MB/token**; top-12's 90.53 represent about **1.21 GB/token**. At around 1.7 GB/s, the latter alone costs about **0.71 seconds** of drive service per token. Wider prediction budgets cannot be equated with wider affordable I/O budgets.

The apparent approximately 200 ms outside the trace's wait interval is not established spare drive capacity. The wait includes CPU service; other processes, Engram, and promotions consume resources. Measure actual queue occupancy, device busy intervals, and contention. Also measure reuse of prefetched rows over later tokens: “not used by this token” is not equivalent to “never useful,” but later reuse can be offset by eviction of a more valuable row.

## Relationship to the six storage priorities

1. **Residency promotions:** retain the measured 32-forward policy. Investigate rare bursts as a tail-latency issue; do not prioritize an assumed recurring 10–15% win. A real async redesign remains separate work.
2. **Drive path:** representative expert reads on nvme1 can change the value and deadline of prefetch. A faster link is not established by the idle-drive estimate alone; keep drive/layout provenance in every experiment.
3. **NVMe-to-RAM copying:** a raw BLOB RAM tier might remove the bounce-to-slab split, but current native split time must be isolated. Earlier eager split timings do not establish the saving in option C.
4. **RAM effectiveness:** weighted per-layer allocation may prevent misses without prediction. Preserve inclusive VRAM-hot copies and demand capacity; do not silently increase the memory budget in predictor arms.
5. **VRAM scratch:** shared serial-layer scratch theoretically recovers **234 rows / about 2.90 GiB**. This changes demand rates and must be measured separately from prediction.
6. **Selective overlap and prediction:** gather already-RAM-ready rows while other rows load, then assess selective NVMe prefetch. The approximately **20.8 ms/token** ready-row overlap figure is a trace-based upper bound, not an implemented gain.

These optimizations interact. Recalibrate prediction after drive, cache, layout, scratch, or residency changes. Their estimated gains cannot simply be added.

## Recommended predictor architecture

Start with a **fresh DSV4.1 LLaPor predictor** over the canonical normalized router input, the source layer's top-6 expert mask, and its top-6 weights. Compare it against the existing **native target gate applied early**, then test a learned head augmented with that gate's scores and margins. Do not begin by replacing the router, executing DSpark, or training on a different quantization path.

The initial pipeline has three distinct responsibilities:

1. **Route prediction:** estimate which experts of target layer T will be selected, using features available at source layer S.
2. **RAM/in-flight filtering:** exclude already-ready rows and deduplicate outstanding reads using the actual bounded RAM tier. VRAM-only filtering is insufficient.
3. **Timely cost-based admission:** select only reads likely to save exposed wait after queueing, latency, pollution, bytes, and scorer cost. It is valid to admit no candidate.

Use explicit `(source layer, target layer, horizon)` heads. Start at **L+1**, then **L+2**, then **L+4**. A shared PCA or small shared trunk can reduce memory and compute, but keep target-specific outputs and calibration. A source may emit several target heads, and a target may receive several earlier predictions; runtime identity must distinguish them before deduplication or arbitration.

Optional mHC side features are the **four `attn_pre` coefficients**, norms, and small stream-disagreement summaries. They may recover information discarded by the weighted combination. These are an ablation after a plain canonical-input baseline, not a reason to replace the input with an unvalidated 20,480-vector. Charge extraction and transfer cost as well as model cost.

The current [`contracts.py`](python/sglang/srt/layers/moe/expert_prediction/contracts.py) and capture schema have no mHC-coefficient or stream-summary fields. Version any new optional fields explicitly; do not overload `PRE_MIXER` or `ROUTER_INPUT`, whose storage/shape contracts describe hidden-size features. Check manifests and serving loaders against that version.

Replace Qwen's fixed outer/middle grouping with DSV-specific layer/horizon measurements. Report Engram transitions at 1 and 14, the 19→20 boundary, layer 21, and the late target layers individually before selecting shared groups. A common head shape need not imply a common admission threshold.

## LLaPor reuse and adaptation map

The repository contains useful infrastructure, but its current Qwen workflow is not a complete DSV4.1 NVMe-prefetch implementation.

| Component | Reuse | Required adaptation or audit |
|---|---|---|
| [`training/llapor.py`](python/sglang/srt/layers/moe/expert_prediction/training/llapor.py) | PCA features, expert mask/weighted-route encoding, small heads, BCE/focal machinery | Fresh 5,120-input PCA and 384-output weights; top-6 labels; select DSV groups from data. |
| `OUTER_SOURCE_LAYERS` in that file | None of its architecture-specific indices | Existing sources **0–7 and 39–46** describe Qwen; most of the late group does not exist as DSV L+1 sources. |
| [`train_llapor.py`](scripts/expert_prediction/training/train_llapor.py) | Per-pair training and resumable artifacts | Remove hardcoded `source_layer + 1`; manifest explicit target/horizon; replace fixed budgets **10,12,16,24,32** and mixed recall@16 selection with decode/cold/timing-aware reporting. |
| Training `_evaluate` popularity helper | General candidate/metric routines | It computes popularity using the evaluated target labels. Fit a deployable popularity baseline on training sessions only and freeze it for validation/test. |
| [`training/pca.py`](python/sglang/srt/layers/moe/expert_prediction/training/pca.py) and [`training/dataset.py`](python/sglang/srt/layers/moe/expert_prediction/training/dataset.py) | PCA serialization and layer-pair/shard loading pattern | Fit PCA and frequency weights on training sessions only; enforce row-identity joins and measured resource caps. Avoid all-layer/all-shard concatenation. |
| [`capture_schema.py`](python/sglang/srt/layers/moe/expert_prediction/capture_schema.py), [`capture_frames.py`](python/sglang/srt/layers/moe/expert_prediction/capture_frames.py) | Request, position, token, forward identity and feature naming | Add unambiguous session/branch provenance, tier state at prediction/demand, and timing. Audit phase/tail-row completeness. |
| [`RouteTaps`](python/sglang/srt/layers/moe/expert_prediction/taps.py) | Graph-stable feature-copy mechanism | Current hook accepts only `StandardTopKOutput`; handle DSV's `StandardTopKOutputPacked` and prove replay produces fresh IDs, input, and weights. |
| [`adapters.py`](python/sglang/srt/layers/moe/expert_prediction/adapters.py) `PRE_MIXER` hooks | Do not require these for LLaPor | The Qwen pre-mixer adapter does not describe DSV mHC. Omit `PRE_MIXER` from this capture's required feature set. |
| [`serving/scorers.py`](python/sglang/srt/layers/moe/expert_prediction/serving/scorers.py) | Training/serving score-parity structure | Support DSV feature definitions, horizons, optional side features, and manifest checks. |
| [`serving/checkpoints.py`](python/sglang/srt/layers/moe/expert_prediction/serving/checkpoints.py) | Checksum/dimension/provenance validation | Store feature schema, source, target, horizon, quantization/runtime path, PCA split, and calibration provenance. |
| [`PrefetchScoring`](python/sglang/srt/layers/moe/expert_prediction/serving/runtime.py) | Device scoring trigger and graph-stable buffers | Existing `source_of = {source: target}` permits one target per source; a target-keyed bank also collapses multiple heads. Introduce explicit head/prediction identity. |
| `TOPK_WEIGHTS` trigger in that runtime | Preserve the readiness principle | LLaPor scoring fires after weights are written. Include **TOPK_WEIGHTS**, not only input and IDs, or the current trigger never fires. |
| [`PrefetchCandidateBank`](python/sglang/srt/layers/moe/expert_prediction/serving/candidates.py) | Stable ranking/validity buffers | Existing exclusion uses VRAM `expert_to_slot`. Filter by RAM-ready and in-flight state for an NVMe objective, before truncating the I/O candidate set. |
| Existing `PrefetchPuller` and dedicated side slot | Useful examples of lifetime/telemetry discipline | They implement a RAM→VRAM pull, not a persistent NVMe→RAM prefetch. Build a native RAM-service bridge; do not wire the wrong tier into a successful scorer. |

The often-cited approximately **0.899 Qwen overall recall** is not DSV NVMe-miss recall, timely precision, or end-to-end savings. Model, expert count, representation, budget, cache tier, and denominator all differ. Retain it only as historical evidence that the infrastructure trained a predictor on another task.

## Capture, labels, splits, and resource contract

**Capture the actual EXL3 fused graph decode first.** A successful eager probe or an FP8 run does not establish training/serving feature equivalence for this path. Eager/FP8 data can become explicit transfer-learning ablations later; they must not silently replace the deployment distribution.

Every training pair needs source and target records joined by explicit identity: session, request, token position/token ID, forward index, phase, source/target layer, and branch/accepted-token identity when applicable. Record source availability time and target demand time. Same-token L+2 is not “shift every table by two rows”; next-token prediction is not “shift one global row.” Never join across sessions, unrelated requests, missing prefill-tail records, branch changes, or dropped capture rows.

Capture canonical source input, source top-6 IDs and weights, target native top-6 IDs and weights, and any optional source features actually available by the trigger. Capture tier status and in-flight state at prediction time and at demand time, with enough cache/eviction events to replay policy changes. The existing capture stages its VRAM residency snapshot at forward end; it cannot reconstruct decision-time RAM admission and must not be treated as source-time cache knowledge. See [`capture.py`](python/sglang/srt/layers/moe/expert_prediction/capture.py), `on_forward_end` and `_stage_identity`.

For the gate self-check, recompute a layer's **own** native routing from the captured canonical input and original gate parameters, including correction bias, dtype, selection, and ties. Preserve unambiguous route mismatches as failed integrity checks; do not compensate by relabeling. If packed representation needs decoding, validate it against the native consumer.

Split by **whole sessions/requests**, with train, development, calibration, and locked test partitions. Near-duplicate prefixes must not leak across them. Fit PCA, normalization statistics, frequency weights, and popularity tables only on training data; select architecture on development; fit admission calibration on the calibration split; touch the locked test after choices freeze. Record both decode-only and prefill results, with decode as the initial objective.

Use bounded layer-pair or head-pair shard processing. The current dataset pattern was designed to load one pair rather than an entire capture; preserve that resource discipline while measuring DSV-specific row sizes. Set and record host/GPU memory caps before processing, stream or accumulate PCA statistics in bounded chunks, and materialize only the current training pair/batch. Predictor weights and PCA also consume serving memory and can displace expert slots.

The existing local-analysis policy in [CLAUDE.md](CLAUDE.md) caps analysis memory at **8 GiB** with **`MemorySwapMax=0`**. Respect that cap when choosing shard/pair sizes; a nominal pairwise loader is not proof that a DSV pair fits. Budget capture/artifact disk space and avoid treating temporary storage as unlimited. Full-model GPU work occupies the shared RTX 5090 and displaces production, so future capture/training/serving runs must follow the project's existing scheduling and owner-authorization process. This handoff launches none of that work and creates no additional approval procedure.

## Shared measurement and decision contract

Record these metrics for every relevant arm, using the same denominator definitions:

- **Route quality:** all-route recall; recall conditional on a target demand being absent from RAM; candidate precision among rows actually eligible for NVMe; calibration by layer/horizon.
- **Timing:** prediction availability, queue delay, read completion, RAM publication, first demand, exposed wait saved, late-but-partially-helpful reads, scorer and CPU scheduling cost.
- **Traffic:** physical submitted/completed bytes for demand, prediction, promotions, cancellations, and rereads; RAM→VRAM bytes; bytes per output token.
- **Cache effects:** useful before first demand, reused within a declared later-token window, evicted before use, rows displaced, and counterfactual demand misses caused by pollution.
- **Serving:** tokens/s, total decode wall time, TTFT, p50/p95/p99 inter-token latency, boundary outliers, host/GPU memory, and hot-slot displacement.
- **Integrity:** healthy-run native routes/tokens/logprobs against the **same EXL3 fused path**, metadata joins, slot generations, and demand fallback behavior.

“Saved wait” requires a same-capacity counterfactual or a paired measurement, not merely a prefetched-row hit. A row can be predicted correctly but arrive too late, already be in flight, or evict a more valuable row. “Unused at the predicted token” and “unused before eviction” must be separate counters.

The current native `rows_read` counter increments for successfully published requests. Interrupted advisories may incur physical reads without incrementing it. It cannot alone establish speculative bandwidth. Report cancellation bytes even when no row is retained.

For live comparisons, include a scorer-only arm with I/O disabled. Charge PCA/weights, score compute, feature extraction, CPU work, and any reduced expert-cache capacity. Keep total RAM/VRAM budgets fixed, or explicitly report all displacement and a matched-capacity control. Use session-paired comparisons and confidence intervals over session-level differences, with independent repetitions where practical; individual tokens within a session are not independent samples.

Final throughput measurements run **without a profiler**. Separate diagnostic traces may explain results. Include cold and warm sessions, long-context/prompt cases, the identified architectural boundaries, and representative contended-drive conditions. Predeclare acceptable latency/traffic regression limits on development/calibration before the locked test; there is no universal required candidate precision.

The previously recorded R4 fused-versus-eager numerical divergence remains unresolved, with no end-to-end FP32 reference establishing its cause. Prediction-on/off parity against the same fused EXL3 path isolates this work's effect; it does not resolve or waive that separate issue.

## Experiment sequence and dependencies

The first viable live path is **E0 → E1/E2 → E7/E8 → E9**. E1 supplies baselines; E2 may supply the learned scorer. A good native-gate candidate can reach the live gate without waiting for every learned-model variation. **E3–E6 are controlled ablations**, not mandatory blockers for an initial bounded prototype. **E10 is optional and parked** until the cheaper path has been measured.

Every experiment below produces a self-contained manifest with source revision, model/checkpoint identity, feature schema, data partitions, cache and drive budgets, run provenance, and explicit measured-versus-simulated labels. “Keep” means advance to the next stage, not deploy based on offline recall alone.

## E0. Capture integrity on the actual fused EXL3 path

**Hypothesis and rationale:** a correctly placed graph-safe tap can expose the true source features and native labels without changing the healthy execution. All later results depend on this; the existing packed-TopK exclusion makes a silent empty/stale capture a concrete risk.

**Prerequisites:** fixed EXL3 model, native routing settings, BS1 fused option C baseline, defined session identities, bounded capture/storage budget. Retain 32-forward residency updates and the current safe overlap settings.

**Steps:**
1. Trace the actual gate-input lifetime through fused mHC/normalization and packed TopK. Specify the semantic tensor and its shape rather than a guessed pre-mixer location.
2. Add or adapt capture support for packed IDs/weights and fresh graph-replay writes; require router input, TOPK_IDS, and TOPK_WEIGHTS. Capture optional mHC fields separately.
3. Record session/request/token/forward/phase/layer identity, prediction-ready timestamps, target demand timestamps, and RAM/VRAM/in-flight state.
4. Run a small integrity capture first. Check finite inputs, expert range 0–383, six native selections, valid weights, row counts, replay freshness, and missing-tail/drop accounting.
5. Recompute each layer's own gate and compare with native routes. Validate source→target joins, including prefill/decode transitions and session boundaries.
6. Compare healthy outputs/routes to the same uncaptured fused path; quantify capture overhead before choosing the main corpus.

**Artifacts:** versioned capture manifest; bounded feature shards; row-join audit; packed-output/gate self-check report; healthy-path comparison; capture-cost and resource report.
**Metrics:** complete/omitted rows by layer and phase, exact route-set matches, mismatch causes, memory/storage footprint, added latency.
**Keep/reject:** proceed only with explained omissions, no invalid joins/stale replay data, and a passing native gate/healthy-route integrity check. A capture failure blocks training; do not fill missing labels or use FP8 success as a substitute.

## E1. Native-gate, previous-token, popularity, and oracle baselines

**Hypothesis and rationale:** cheap native-gate lookahead supplies useful but selective cold-row predictions; oracle replay bounds what the storage schedule can hide. Previous-token and popularity baselines reveal whether a learned model adds information beyond recency and static frequency.

**Prerequisites:** E0 valid capture, held-out session split, a replay model with the actual RAM capacity/inclusivity and observed service-time distributions.

**Steps:**
1. Apply each future target's native gate to the available source input for L+1 and L+2. Include L+4 as an explicitly new baseline only where the source/target pair exists.
2. Rank target scores and margins; preserve raw score provenance. Sweep narrow candidate ranks and confidence thresholds rather than blindly issuing every top-k row.
3. Evaluate previous-token routes with exact request/token alignment. Fit per-target popularity on training sessions, then freeze it; do not reuse the current label-derived evaluation helper unchanged.
4. Build an oracle that knows future routes but obeys the same deadlines, bandwidth, queue limits, RAM capacity, and active/hot protections. Label it a nondeployable upper bound.
5. Filter ready/in-flight rows and replay candidate admissions with queueing, pollution, subsequent reuse, and all physical bytes. Separate static prediction metrics from changed-cache replay.

**Artifacts:** baseline rankings and frozen popularity tables; layer/horizon reports; queue/cache replay specification; oracle-bound report.
**Metrics:** all-route versus cold recall, timely eligible precision, exposed wait saved, queue delay, extra bytes, subsequent reuse and pollution.
**Keep/reject:** retain the best feasible baseline, including no-prefetch. If the oracle's net opportunity is negligible under realistic costs, fix storage/cache constraints before scaling model training. Failure to reproduce eager-probe numbers is a distribution difference to explain, not a reason to alter fused-path labels.

## E2. Fresh plain LLaPor for L+1

**Hypothesis and rationale:** a small learned mapping from the current canonical input and route signature can predict the next layer's cold experts better than applying the next gate directly to the current state.

**Prerequisites:** E0 joins and E1 baselines; train/development/calibration/test sessions fixed; pairwise memory budget; fresh target model identity.

**Steps:**
1. Build valid source S→target S+1 pairs for S=0–38. Use normalized 5,120-element input plus a 384-wide source expert mask and weighted route vector.
2. Fit PCA and expert-frequency weights on training sessions only. Start with the existing small-head building blocks; treat PCA rank/head width as development choices rather than inherited Qwen constants.
3. Train fresh 384-output multilabel heads with native top-6 targets. Omit PRE_MIXER; include TOPK_WEIGHTS end to end.
4. Replace fixed +1 metadata assumptions with explicit source/target identity even for this first horizon, preparing later ablations without changing labels.
5. Evaluate narrow eligible candidate budgets and cold/timely metrics; report the legacy broad-budget recall separately if useful. Do not select solely on mixed recall@16.
6. Check training/serving scorer parity, precision/recall calibration, inference cost, PCA/weight bytes, and resulting cache-slot displacement before selecting a checkpoint.

**Artifacts:** per-pair PCA/head checkpoints and manifests; training-only statistics; frozen session splits; scorer-parity report; comparisons with E1 and no prediction.
**Metrics:** decode cold recall and timely precision, predicted net wait saved at equal bytes, calibration, scorer latency, resident bytes and hot-slot cost.
**Keep/reject:** advance only if a development/calibration region improves expected net value over the best feasible E1 baseline after model cost. High overall recall without eligible/timely improvement is insufficient; a native-gate baseline remains a valid first live candidate.

## E3. L+2/L+4 horizons and shared computation

**Hypothesis and rationale:** farther target layers provide more lead time, potentially offsetting reduced accuracy. Sharing source PCA/trunk work may make multi-horizon scoring affordable.

**Prerequisites:** E2 or an equivalently validated head pipeline; measured source/target timings; explicit head identities and valid same-token joins.

**Steps:**
1. Train target-specific S→S+2 heads, then S→S+4 heads, excluding out-of-range targets and never wrapping a tail pair into the next token.
2. Compare independent PCA/heads with shared source PCA, and then a small shared trunk with target-specific output heads. Keep losses and training sessions controlled.
3. Charge all simultaneously active heads. Measure when each score becomes available, not only nominal layer distance.
4. Replay several predictions for the same future expert with explicit source/horizon identity; deduplicate reads and compare earliest-only, best-calibrated, or refreshed-deadline policies on development data.
5. Test shorter lead under a faster drive/decode scenario rather than assuming today's layer timing persists.

**Artifacts:** head manifest indexed by source/target/horizon; sharing ablation table; per-head arrival/deadline distributions; memory/compute and dedup reports.
**Metrics:** timely coverage per physical byte, total scorer cost, shared-model capacity effects, queue depth and deadline misses.
**Keep/reject:** retain only horizons/sharing variants with improved net timely value. More lead that merely admits more inaccurate traffic is a rejection; runtime multihead support remains E8 work, not an existing bank capability.

## E4. mHC side-feature ablation

**Hypothesis and rationale:** four `attn_pre` coefficients, norms, or stream-disagreement summaries may reveal information lost in the canonical weighted combination and improve prediction around difficult transitions.

**Prerequisites:** a passing canonical-input E0/E2 path and source-time access to the side features in the actual fused graph.

**Steps:**
1. Define precisely which four coefficients and which norm/disagreement summaries are available before the scoring trigger. Record feature extraction semantics and dtypes.
2. Compare canonical-only, canonical-plus-coefficients, canonical-plus-norms, and a compact combined feature set on the same split/head sizes.
3. Fit any normalization statistics on training only. Do not silently substitute arithmetic stream means for the canonical input.
   As a separate representation ablation, compare the weighted combined **pre-normalization** 5,120-vector with the canonical normalized input, keeping labels and model capacity matched. Preserve explicit feature names and charge any extra materialization; do not force an otherwise fused path to expose raw streams just to make capture convenient.
4. Measure extra graph nodes, extraction latency, capture size, model bytes, and any synchronization introduced by reading mHC metadata.
5. Analyze changes specifically around Engram layers, 19→20, layer 21, and late layers, as well as aggregate cold demand.

**Artifacts:** side-feature schema; controlled ablation checkpoints; per-stratum results; extraction-cost report.
**Metrics:** incremental timely precision/coverage, calibration, feature cost and serving memory, healthy-route parity.
**Keep/reject:** keep only compact features whose added benefit survives extraction and model cost. Reject features unavailable by the declared source deadline or any result explained by using future state. Raw four-stream models require a separately costed experiment, not a baseline redefinition.

## E5. Native-gate augmentation and hard negatives

**Hypothesis and rationale:** native future-gate scores encode a strong prior, while a learned correction can distinguish confident false positives that waste NVMe reads.

**Prerequisites:** E1 target-gate baseline and E2 scorer, same canonical capture and fixed train/development partitions.

**Steps:**
1. Compare plain LLaPor, native-gate-only, and LLaPor plus future-gate scores/margins. Preserve target-layer-specific bias and score semantics.
2. Test compact rank/margin summaries against full 384-score augmentation, charging the future-gate matrix multiply and feature memory in both cases.
3. Mine hard negatives from training sessions: high-scoring eligible predictions that do not route at the target or are not reused within the declared horizon. Keep immediate-use and later-reuse labels distinct.
4. Compare a restrained ranking/hard-negative objective to the plain loss; preserve positive coverage and report per-expert frequency effects.
5. Recalibrate confidence on held-out calibration sessions; a native score margin is a feature, not a probability label.

**Artifacts:** augmentation/negative-sampling specification; paired checkpoints; calibrated precision curves; scorer-cost breakdown.
**Metrics:** NVMe-eligible false positives and physical bytes at matched timely coverage, rare-expert recall, calibration, extra GPU time.
**Keep/reject:** accept augmentation only if fewer costly false positives or more timely useful reads repay the gate evaluation. Reject apparent gains from negative mining on validation/test labels or from hiding lost useful recall.

## E6. DSV layer/horizon calibration and grouping

**Hypothesis and rationale:** prediction reliability and usable lead differ by target layer and horizon; uniform thresholds and Qwen outer/middle groups waste I/O or suppress useful candidates.

**Prerequisites:** at least one frozen scorer from E1/E2; separate calibration sessions; enough examples to assess each reported stratum.

**Steps:**
1. Produce reliability/precision curves by target and horizon, separating already-cached predictions from eligible cold candidates.
2. Predeclare architecture strata: Engram layers 1/14 and crossing pairs, 19→20, layer 21, late layers, and ordinary interior pairs.
3. Compare per-layer thresholds with pooled strata and a regularized/shrunk calibration for sparse layers. Select complexity by held-out calibration quality, not by fitting every noisy layer independently.
4. Test whether shared head/PCA groups improve cost while preserving the necessary target-specific outputs and thresholds.
5. Freeze a calibration manifest with workload, cache, drive, scorer, and horizon provenance; reevaluate after material serving changes.

**Artifacts:** layer/horizon reliability report; grouping decisions with sample counts; frozen calibration table; sparse-stratum fallback policy.
**Metrics:** calibration error, precision at admitted bytes, eligible candidate counts, per-stratum wait saved and regression tails.
**Keep/reject:** keep granularity that improves held-out scheduling decisions. Reject groups copied from Qwen or a single global threshold whose aggregate score hides a harmful layer. Insufficient support calls for pooled calibration or no prediction in that stratum.

## E7. Cost-aware admission first, then alternative losses

**Hypothesis and rationale:** most practical value may come from choosing which predictions to read, not increasing broad route recall. Learn routes first; make cache state and timing explicit in admission before adding a policy-dependent loss.

**Prerequisites:** valid E1/E2 scores, realistic service/queue timing, inclusive cache replay, frozen calibration partition, physical-byte accounting.

**Steps:**
1. Filter RAM-ready and already-in-flight rows before candidate truncation. Treat an in-flight demand/prefetch as a join/priority event, not another read.
2. Estimate candidate value from calibrated probability of use before eviction, available lead, service/queue delay, exposed wait saved, pollution, and scorer/scheduler cost.
3. Sweep shallow in-flight byte limits, candidate count, deadline/expiry, and probationary occupancy on calibration sessions. Include an explicit no-admission outcome.
4. Compare immediate-token utility with declared short-window future reuse; run the altered cache trajectory, not a fixed snapshot, and count displaced future demands.
5. Freeze a useful admission baseline before testing cost-weighted, cold-row-weighted, or timely-ranking training losses. Recompute their labels under a stated policy; avoid leaking future cache state into runtime features.
6. Evaluate under realistic contention and alternative drive latency, retaining demand service precedence.

**Artifacts:** admission policy specification; calibration sweeps; full queue/cache replay logs; counterfactual pollution report; optional loss ablation with label provenance.
**Metrics:** net exposed wait saved, timely and late utility, total physical bytes, pollution-induced misses, scorer/scheduling cost, sensitivity to contention.
**Keep/reject:** advance a policy only when its value remains positive over the chosen development/calibration conditions and limits are frozen. Reject a policy that improves recall by spending infeasible disk time or hiding cancellation/promotion traffic. Alternative losses must beat the plain-scorer admission baseline.

## E8. Native RAM bridge and demand-priority scheduler

**Hypothesis and rationale:** a useful score can reduce demand wait only if it reaches the correct tier safely and early. The existing VRAM puller and advisory service are not the required finished scheduler.

**Prerequisites:** E0 identities, a frozen initial scorer/admission policy, design of source/target/horizon/forward tags, and a native slot-lifetime/publication contract.

**Current limitation:** [`Service::serve`](python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp), around lines 796–885, reserves its missing advisory set before reading, reads serially, checks for demand between reads, and publishes the request as a batch. If interrupted, it releases the whole reserved set, including already-read rows, without restoring evicted victims. These are current behaviors to redesign, not capabilities to assume.

**Steps:**
1. Extend scoring/runtime identity beyond one `source → target` entry and a target-only bank. Preserve graph-stable storage and freshness across replay, sessions, and optional branches.
2. Define a native RAM advisory bridge carrying target expert, identity, deadline/priority, and calibrated value; bound its queue and bytes. Do not route it into the dedicated VRAM pull slot.
3. Deduplicate ready and in-flight reads; join a demanded speculative row and raise urgency. Admit unsubmitted demand before speculation, acknowledging that an executing read is not instantly preemptible.
4. Reserve only safe slots, protect active consumers and inclusive GPU-hot/loading rows, and use generation/ownership checks so stale completion cannot publish a reassigned slot.
5. Publish READY only after complete validated data is visible. Evaluate retaining individually completed useful rows when later work is cancelled; count all physical I/O either way.
6. Add expiry/cancellation, bounded fairness, fault accounting, and demand fallback. Preserve existing fail-stop behavior; do not invent a partial-expert output fallback.
7. Exercise protocol invariants with controlled completion ordering, stale generations, demand arrival, cancellation, pressure, and shutdown before any throughput claim.

**Artifacts:** native request/slot-state contract; bridge and protocol verification report; physical-byte/cancellation telemetry specification; bounded-queue and memory report.
**Metrics:** duplicate reads, stale publications, blocked demand latency, bytes after cancellation, completed-but-discarded rows, slot occupancy and fairness.
**Keep/reject:** no live A/B until lifecycle and publication checks pass. Throughput cannot compensate for overwritten active data, wrong expert IDs, unbounded demand delay, or suppressed failure signaling. The existing timeout can drop a layer's MoE output before raising; it is not an exact-output fault-recovery guarantee.

## E9. Paired end-to-end EXL3 evaluation

**Hypothesis and rationale:** a small number of timely predictions will improve real decode time after all predictor, memory, scheduling, and pollution costs. Offline recall and replay are selection tools, not this result.

**Prerequisites:** E0 integrity, frozen chosen E1/E2 scorer, E7 admission, E8 safe bridge, calibrated resource limits, and an approved serving measurement window under the project's operational policy.

**Steps:**
1. Define matched arms: unchanged no-prediction baseline; scorer-only with no speculative I/O; native-gate prefetch if viable; selected learned prefetch. Keep 32-forward residency and the same fused numerical path.
2. Hold total RAM/VRAM budgets fixed. Record predictor/PCA bytes, changed hot slots, and a matched-capacity baseline when displacement must be separated from prediction effects.
3. Pair identical sessions and sampling settings; repeat/order-balance where practical. Include cold, warm, long-context/prompt, and representative contended-drive cases.
4. Compare healthy native routes and output tokens/logprobs against the same EXL3 fused path, not FP8 or the historically divergent eager loop.
5. Measure final throughput without a profiler. Use separate bounded traces only to diagnose arrival timing, stalls, and queue behavior.
6. Report session-paired confidence intervals, absolute latency/traffic, TTFT, p95/p99, cancellation bytes, cache pollution, scorer-only overhead, and promotion bursts.
7. Evaluate the frozen candidate once on locked-test sessions; retain failures and negative sessions in the report.

**Artifacts:** arm manifests; session-paired raw outcomes; healthy correctness comparison; latency/traffic/memory tables; confidence intervals; final keep/reject rationale.
**Metrics:** the full shared contract, led by wall-time/tokens-per-second benefit after all costs, plus latency tails and physical bytes per output token.
**Keep/reject:** accept only a reproducible net gain within predeclared correctness, memory, traffic, and latency limits. If the interval includes no gain, expand independent evidence or retain the simpler baseline; do not declare success from a favorable token subset or raw hit rate.

## E10. Optional next-token prediction and DSpark-assisted features

**Hypothesis and rationale:** late target states may predict a future-token working set with more lead than same-token lookahead. DSpark may offer additional future-token features, but its execution can cost more than the saved I/O.

**Prerequisites:** the cheaper same-token path is measured; exact cross-token/branch identities exist; a bounded persistent-RAM reuse objective and an explicit resource budget are defined. This research arm remains parked by default.

**Steps:**
1. First train a small future-token/window head on existing late target states, including the available 37–39 taps if appropriate. Join only subsequent positions of the same accepted request trajectory; count feature availability after the current pass.
2. Compare this cheap head against previous-token routes, training-only popularity, same-token lookahead, and an oracle at equal RAM/byte budgets. It cannot hide earlier misses of the already-completed pass.
3. Only if the remaining opportunity justifies cost, evaluate DSpark-derived hidden features and optional proposed tokens with a newly trained mapping to target `(layer, expert)` scores.
4. Keep the three cases separate: cheap late-target head; DSpark-assisted predictor-only; full DSpark speculative decoding plus prefetch.
5. For predictor-only DSpark, account for draft loading/layout changes, execution and memory even if vocabulary projection or token sampling can be omitted. For full speculation, additionally account for verify shapes, acceptance, branches and bytes per accepted token.
6. Preserve branch-qualified labels and cancellation accounting; draft-token confidence is not target-expert probability, and rejected branches may still have consumed disk bandwidth.

**Artifacts:** cross-token identity/join audit; cheap-head checkpoints; optional draft-to-target mapping specification; full memory/compute/traffic model; separate predictor-only and speculative evaluation reports.
**Metrics:** future-window useful retention, timely target cold coverage, wasted branch bytes, displaced hot capacity, net accepted-token latency; acceptance rate only where full speculation is actually used.
**Keep/reject:** advance DSpark only if it beats the cheaper late-target head and same-token baseline after its complete cost. Qwen acceptance break-even does not establish predictor-only value or DSV speculative value. No advantage from calling draft expert IDs target expert IDs is valid.

### DSpark facts and blockers to preserve in any E10 design

The draft has **three stages**, each with **128 experts/top-3**; the target has **384 experts/top-6 per layer**. Draft stages and target layers have separate routers and weights. There is no identity mapping between their expert IDs. A predictor needs an explicit learned or calibrated mapping to target layer/expert scores.

Target taps at layers **37, 38, and 39** are completed post-layer states averaged across mHC streams. They are not the canonical normalized router inputs described earlier. Draft `forward` returns hidden states; these and optional token proposals may be features, but neither reveals the target's exact intermediate routes in advance. DSpark confidence concerns token proposals.

Draft experts alone are estimated at approximately **6.8 GB** resident, before additional draft state. Using that VRAM displaces a substantial part of the target hot cache; streaming it instead adds transfer traffic. The current full-model serving directory omits MTP draft weights, the DSpark loader has no EXL3 adaptation, and the process-wide EXL3 streamer assumes a 384-expert layout that rejects 128-expert draft layers.

The current target BS1 graph/scratch/request sizing blocks multi-token **full speculative verification**. It does not logically require predictor-only target execution to become multi-token, but predictor-only DSpark still needs draft loading, layout, features, memory, and execution plumbing. None of those costs disappear by omitting verification.

Relevant source: [`deepseek_v4_dspark.py`](python/sglang/srt/models/deepseek_v4_dspark.py) for draft stages, `forward`, `compute_confidence`, and `load_weights`; [`dspark_worker_v2.py`](python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) for orchestration; [`deepseek_v4_exl3_weights.py`](python/sglang/srt/models/deepseek_v4_exl3_weights.py) for target adaptation; [reference §§14.3 and 18.5](DSV41_REFERENCE.md).

## Existing evidence and artifact locations

These are recorded locations from the current reference and prior experiments, not newly checked remote files. Verify availability, revision, checkpoint hashes, and resource limits before a future run. Keep new experiment outputs in a separate run directory and preserve the original evidence.

| Location | Use |
|---|---|
| `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-overlap/` | Existing probe, lookahead analysis, trace analysis, and `boundary/` promotion arms. |
| `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40` | Existing full-target serving directory; it omits the MTP draft weights. |
| `divix01:/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw/` | Original EXL3 checkpoint, including draft tensors; this does not imply the serving loader supports them. |
| `divix01:/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl` | Existing session corpus and split provenance; establish untouched DSV test sessions before tuning. |
| [`2026-09-15-expert-predictor-offline-training.md`](docs/superpowers/experiments/2026-09-15-expert-predictor-offline-training.md) | Historical Qwen training, checkpoint conventions, resource lessons, and metric definitions. |
| [`exl3_ram_miss.py`](python/sglang/srt/layers/moe/exl3_ram_miss.py) and [`exl3_ram_miss_host.cpp`](python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp) | Python/native service boundary, pinned slot ownership, advisory lifecycle, and counters for E8. |

For each new run, record checkpoint and PCA/head hashes, feature/tap version and dtype, source/target/horizon identities, random seeds, training/calibration/test session lists, optimizer/loss and selection rule, cache sizes and policies, drive/file layout, graph mode, scorer memory displacement, and runtime source revision. Store queue/cache replay assumptions alongside simulated results. This makes a later transfer or cache change distinguishable from a predictor improvement.

## Completion record expected from the implementation task

The implementation task should return a manifest of exactly which experiments ran, which stayed proposed, and which were rejected, with reasons and links to their artifacts. Include the frozen scorer/calibration/admission configuration, source/model revisions, session splits, serving resource budget, independent protocol/correctness review, and the matched end-to-end result.

A useful result may be a native-gate policy, a small LLaPor head, or evidence that no prediction beats the current baseline under the measured storage constraints. The success criterion is reduced real expert-service latency with preserved native routing and bounded memory/traffic, not completion of every optional experiment or a high all-route recall figure.
