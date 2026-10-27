---
title: "NCM head: staged acceptance and serving experiments"
description: "Proposed experiments for architecture revision 3: correctness, a small controlled pilot, component attribution, causal cutoff calibration, and serving cost."
---

# NCM head: staged acceptance and serving experiments

Date: 2026-10-18. Revision 2. Status: **Protocol corrected after Astra review; isolated experiment setup authorized. No model capture, training, GPU experiment, or serving run is authorized by the setup request. Local correctness checks and read-only asset inspection are permitted.**

The user requested direct repository preparation without hx. The [Astra experiment review][ER] is the rationale for this revision. The setup lives in [analysis/ncm_head][SETUP]; its readiness report distinguishes CPU-tested reference components from native model/data prerequisites. This protocol does not turn an unverified native path into a runnable experiment by assertion.

This protocol evaluates [architecture revision 3][ARCH], including its fixed future residual targets, token-weighted zero-mean fine residuals, explicit EOS handling, and single-gate prefix head. The user has now requested experiment design; the earlier instruction to defer experiments is superseded for this companion document. The architecture itself is unchanged.

The pre-companion-link architecture SHA-256 is `97918887868cc0aeb4e106ff0f6ec4490033cae97e5df3891da5b42cf80113b8`. Local source was inspected at `d82ae69d0368478b5018d4cbfa92ab4643a43f54`. A future execution must record actual source and checkpoint identities; these are provenance, not a claim about the remote server today.

## 1. What we want to learn

The primary question is whether the full head increases accepted draft tokens enough to pay for its additional work on the intended DSpark serving path. Four narrower questions determine what to retain:

1. Does it improve proposals beyond retraining and adding ordinary adapter capacity?
2. Does persistent history help beyond the latest target feature and context already available to DSpark?
3. Do the future hierarchy and fixed residual supervision help beyond a flat adapter?
4. Does the prefix state earn its serial cost, and does a causal confidence cutoff reduce actual verification cost without changing sampled output semantics?

NCM does not establish any of these results. Its coarse-to-fine generation motivates the hierarchy, while the short continuous head, temporal memory, and acceptance objective are new hypotheses. [Sol S1–S5 and S8][SOL] and [Astra's two reviews][R1] [R2] are the evidence behind the architecture and the controls below.

**Order:** correctness and feasibility → small training/variance pilot → independently budgeted screening → component attribution → confidence calibration → closed-loop serving. A budget stop or preliminary comparison is not evidence against the architecture. Correctness failures stop performance claims; training inadequacy and statistical uncertainty have separate dispositions.

## 2. Primary configuration and scope

| Item | Proposed primary setting | Reason / limit |
| --- | --- | --- |
| Target and draft | The DSV4.1 target and DSpark draft referenced by the existing local serving recipe | Tests the intended integration; checkpoint availability, dimensions, quantization and hashes must be resolved in stage 0. |
| Draft width | `gamma = 5` proposed tokens; verification has up to six input rows including the anchor | This is the current [recipe][ARM], rather than an arbitrary long block chosen to favor recurrence. |
| Serving concurrency | Batch size / active requests = 1 | Primary deployment hypothesis; larger batches are secondary and require a separate capacity check. |
| Sampled target regime | Temperature 0.6, top-p 0.95, no other penalties unless explicitly recorded | Proposed starting regime informed by the historical serving context below. Confirm the actual intended configuration before collecting labels. |
| Draft sampling | Preserve the resolved native draft transform and record every parameter separately from the target transform | Equal target/draft temperature or filtering is not assumed. Actual transformed `q` is required by verification. |
| Secondary decoding | Greedy; sampled temperature 1 with no filtering | Separates deterministic matching from sampled behavior and stresses broad support. Each gets its own confidence calibration if cutoff is evaluated. |
| Reference continuation for future labels | Frozen target, temperature 1, no filtering or penalties | This is the architecture's fixed supervision law, independent of the serving regime. |
| Backbone | Frozen existing parallel DSpark backbone; frozen target, original token embedding, and original LM head | Isolates the proposed head. Preserve trainable copies of the dependency tables as specified in revision 3. |
| Memory / planner / prefix state | Exactly revision 3: width 128; four temporal slots; two planner stages; at most four fine segments; single-gate state | Avoid a simultaneous architecture search. Numeric settings remain hypotheses. |

> 🧠 **From Hindsight memory (F142 — kp-9909d120898342a4b7fee3bd0eee916d)** — The earlier work recorded sampled serving at temperature 0.6/top-p 0.95 and batch size 1, distinguished greedy cutoff gains from sampled serving evidence, and reported that its served comparison stopped on a driver deadlock without a final served result. This motivates the proposed primary regime and a separate serving gate; it supplies no performance result or threshold for this head.

The local [recipe][ARM] names remote paths including `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40` and `dsv41-dspark-draft`. These are configuration strings, not locally verified artifacts. Stage 0 must resolve them on the eventual execution host. No remote resource is presumed idle, approved, or ready.

A small proxy model may debug implementation and distribution tests. It cannot replace the native-model acceptance or latency evidence. Do not silently switch to a different target, quantization, backbone, or horizon if native training is difficult.

Wider blocks are deferred. `gamma = 9` may require compatible backbone weights, positional handling, and a new training protocol. Padding a five-token model is not a valid wider-block experiment.

## 3. Hypotheses and decisive comparisons

| Hypothesis | Comparison | Evidence that would support it | What would not support it |
| --- | --- | --- | --- |
| The complete proposal improves drafting | Full head F versus capacity control C, cutoff off | Higher expected accepted draft tokens at matched anchors and a useful serving result | F beating only an untrained/frozen stock head |
| Persistent memory contributes | PM versus P; F versus PH | Gain with historical memory enabled, particularly on longer contexts, without an unacceptable update cost | Low future-feature MSE alone |
| Prefix state contributes | PH versus P; F versus PM | More accepted tokens and a favorable measured reward/time tradeoff | More accepted tokens with a larger proportional time increase |
| Explicit future supervision contributes | F versus F-noaux | Better actual-prefix acceptance with fixed auxiliary labels | Better auxiliary loss on its own |
| The hierarchy contributes | F versus F-flat | Better acceptance/cost than a matched flat planner with the same memory and prefix interfaces | Calling any improvement over stock DSpark an NCM-specific result |
| Confidence cutoff contributes | Same frozen head, cutoff off versus calibrated cutoff | Less executed verification work and lower end-to-end decode latency with correct sampled output law | Shorter reported prefixes after full verification |

The minimum pilot answers only the first row. It does not establish which component caused a gain. The later controls distinguish useful NCM-inspired structure from generic added capacity; they do not claim to reproduce NCM's discrete latent mechanism.

## 4. Stage 0: feasibility and correctness before training

### 4.1 Freeze an execution manifest

Record source revision plus local diff, architecture snapshot, all target/draft/tokenizer hashes, quantization, feature-layer indices and dimensions, vocabulary size, Markov rank, epsilon, random projection tensor/hash, precision, backend, graph mode, CPU/GPU placement, expert storage/cache settings, and all sampling transforms. Record hardware, driver/runtime, thread counts and seeds.

Resolve the target feature alignment with actual traces: the memory watermark refers to committed **processed** input tokens; the anchor and each future feature refer to the positions specified in the architecture. Do not infer alignment from array lengths. Save a small human-inspectable trace containing input token indices, processed indices, commits, rejected candidates, EOS, and memory updates.

**Setup finding:** the native checkpoint stores a 6-bit EXL3 vocabulary head. The isolated bridge retains its native forward and proposes a reconstructed-frozen-weight straight-through backward surrogate, with the original norm boundary. A weight-only GPU probe is prepared but unrun; real-row parity, feature alignment and gradient checks remain required. This choice is explicit and does not claim exact differentiation through quantized arithmetic.

The frozen, possibly quantized LM head must provide a valid gradient to the trainable adapter input. Confirm a reference backward path and forward parity before committing to the training budget. A custom inference kernel with no input gradient is a feasibility issue; freezing its weights does not remove the need for this gradient. Any changed precision or alternative projection used for training must be documented and checked against actual inference logits. Do not silently train a different head.

Measure capture throughput, one training step, memory, and disk growth on a tiny sample. Project cost from actual measurements before a larger capture. Do not infer GPU-hours from parameter count or the paper.

### 4.2 Architecture invariants

These are tests to implement later, not tests run by this document:

- At zero-initialized adapter/state projections, proposal logits reproduce the existing dependency head within the declared numeric tolerance. Verify both logits and transformed probabilities.
- Memory updates exactly once per committed, processed token; rejected suffixes, graph padding, and unprocessed bonus tokens do not enter it. Test request reuse, reset, rollback, chunked prefill and different batching/chunk boundaries.
- Recomputed memory under current trainable encoder weights matches an uncached reference. Cached learned state from an older checkpoint is not a valid training input.
- `sum_j w_j Delta_j = 0`; reconstructed weighted mean equals `z0`; `J=1` gives zero fine residual. Test unequal segment sizes at `gamma=5`.
- Fixed target tensors do not change with student parameters. Confirm the coarse-input and centering gradient paths numerically. No detached moving residual target may reappear.
- Include the actual EOS feature in its label position, then fixed zero label vectors. Capture truncation without EOS masks both planner losses when the horizon is unknown; it does not create a terminal event. No validity mask or future feature enters model inputs.
- Prefix state resets every round and uses the actual sampled draft prefix. The confidence decision for position `i` is evaluated before drawing `y_i`.
- Changing future labels or future draft candidates while holding the available prefix fixed cannot change earlier proposal/confidence inputs or admission.
- Confidence loss cannot update the token, planner or memory branches through detached features. Invalid proposal distributions are errors, not ordinary low confidence.

### 4.3 Exact sampled-cutoff checks

Use a tiny synthetic vocabulary (four symbols including EOS), horizon up to three, and enumerably small states. Enumerate candidate paths and acceptance/rejection branches using simple distributions with analytically tractable probabilities. Compare the complete output distribution with ancestral target sampling, including variable-length EOS output.

Cover zero/full/partial admission; unequal target and draft probabilities; support zeros; sharp and diffuse distributions; prefix-dependent thresholds; corrected rejection tokens; both valid boundary mechanisms in the architecture; and output-length caps. Admission must depend only on information available before the next draw.

Cover the fallback state machine: pre-request calibration mismatch goes to full verification; a visited invalid confidence enters FULL_REMAINDER; STOPPED never reopens after observing an invalid suffix. Reject intentionally biased variants that inspect `y_i` or later confidence values when deciding to admit `y_i`. These negative controls establish that the test can detect the important class of bug.

The test must exercise the chosen boundary implementation. A retained prefix can be followed by a fresh target sample, or by the architecture's valid accepted/corrected next-proposal mixture when those probabilities have actually been evaluated. Simply cutting the emitted token array and choosing an arbitrary existing bonus token is not justified.

Then run relevant existing and new parity/distribution tests on the real backend. Existing suites include [kernel parity][TESTPARITY], [ragged verification][TESTRAGGED], [backend capability][TESTBACKEND], [STS][TESTSTS], and [speculative sampling][TESTSAMPLE]. Enumeration plus an algorithmic argument supports correctness; a finite Monte Carlo test alone cannot prove exactness. Any correctness failure stops all performance claims.

**Stage-0 exit:** a pinned feasible native path, verified label alignment and invariants, correct cutoff/boundary behavior, and a measured resource estimate. If these cannot be established, report the specific blocker before training rather than replacing the experiment implicitly.

## 5. Data, splits and leakage controls

### 5.1 Proposed registry

Use 832 independent prompt groups, balanced across four declared domains: code, mathematics, ordinary instruction/chat, and long structured reasoning/document use. The exact source names, licenses, versions, selection query and prompt IDs must be frozen in the execution manifest before capture. This document does not claim these datasets have already been assembled.

| Split | Prompt groups | Per domain | Permitted use |
| --- | ---: | ---: | --- |
| Training | 512 | 128 | Token/planner training; rollout distribution matching; confidence-head training |
| Development | 96 | 24 | First 32: feasibility and later checkpoint selection. Remaining 64: one independent fixed-checkpoint screen; later reuse is exploratory. |
| Calibration | 96 | 24 | 48 groups for confidence calibration; disjoint 48 for threshold selection |
| Sealed test | 128 | 32 | One final frozen acceptance/serving evaluation |

Within each split and domain, target approximately half the prompts at 512–2,048 input tokens and half at 4,096–8,192, subject to the model's supported context length. Use real coherent contexts; padding with irrelevant text is not equivalent to long-context use. Record realized lengths and any shortfall. Group related documents, problem variants, code repositories and synthetic templates together **before** splitting. No derivatives of one prompt family may cross splits.

The [existing local sample prompt][SAMPLE] is one structured-reasoning smoke test, not a representative benchmark. Keep it outside aggregate claims unless its role was declared before data selection.

Generate common committed-prefix anchor trajectories from the intended target serving law (temperature 0.6/top-p 0.95), up to 256 tokens or EOS. Candidate anchors occur at offsets `0,32,64,...,224` where a valid nonterminal anchor exists: at most eight blocks per prompt. From each anchor, independently sample its five-position label continuation under the fixed temperature-1 unfiltered reference law. Thus the available history reflects deployment while future supervision preserves the architectural law. Store separate RNG streams and law identities for anchor trajectories and label continuations. Use identical anchor records across arms; no fabricated blocks after early EOS. Temperature-1 anchor trajectories are optional diagnostics, not the primary deployment screen. Common target-law anchors still differ from each deployed head's closed-loop boundaries.

For planner supervision, every future position must have a real reference feature or known post-EOS terminal padding. If the final horizon is unknown, omit both planner losses for that block. Capturing the EOS feature may require processing EOS once for labels; this is label extraction, not generation beyond EOS. Token losses may still use individually valid labels, with their own masks.

The pilot uses 128 training groups and the first 32 development groups, balanced by domain and length. The other 64 development groups are sealed from checkpoint/recipe selection until the independent screen. Training extensions never consume screening or test groups. The pilot uses the scheduled final endpoint, not a development-selected endpoint. Expanded phases may select on the first 32 development groups; reuse of the 64 after the one screen is explicitly exploratory. If measured variance requires more independent screening groups, register additional source-disjoint groups before seeing their outcomes; do not take them from the sealed test.

### 5.2 Three different data contracts

| Data | Conditioning and use | Prohibited shortcut |
| --- | --- | --- |
| Reference futures | One fixed temperature-1 target continuation after each fixed anchor; future features supply fixed coarse/fine labels and reference-token CE | Feeding these future features, labels, or masks into the proposal |
| Distribution-matching rollouts | Target `p_i` evaluated at the **actual recorded draft prefix**; student `q_i` reevaluated at that same prefix | Pairing target probabilities from the reference prefix with a different sampled draft prefix |
| Acceptance/confidence evaluation | Fresh candidates from the frozen evaluated head, with target probabilities at each corresponding prefix and exact sampling transforms | Reusing another head's candidate path as if it were the evaluated head's own output distribution |

Cached target probabilities remain valid for their recorded prefix; their data distribution becomes off-policy as the draft changes. Refresh rollout banks explicitly, label this limitation, and always use fresh own-head paths for final acceptance. A common target-uniform random stream may reduce variance, but different proposals need not produce identical candidates or outputs at the same seed.

### 5.3 Capture and storage

Store the committed target feature sequence needed to recompute the learned memory, frozen backbone features at each anchor, reference futures/labels, position maps, and relevant probability records with hashes. Recompute learned memory from committed features under current weights, using checkpointed/chunked scans if necessary; do not cache trainable memory outputs across optimizer updates. A block must not start with an artificially empty state merely to simplify training.

The existing [capture utility][CAPTURE] can provide useful diagnostics, but it is not a complete training capture for this head. Its top-probability option adds an extra draft LM-head pass, and top-1 probabilities do not provide full distributions, all new feature tensors, or the required current-prefix labels. Instrumented capture timings are not throughput measurements.

Budget using actual dimensions before collection:

```text
committed-feature bytes ≈ N_processed_tokens * d_context * bytes_per_feature
backbone-feature bytes  ≈ N_anchors * gamma * d_draft * bytes_per_feature
target-probability bytes ≈ N_rollout_blocks * gamma * V * bytes_per_probability
```

Full-vocabulary probabilities can dominate storage. Use streamed vocabulary reductions where possible; retained target probability rows for training should use lossless chunked storage with an explicit precision contract. Top-k truncation is not exact TV/overlap supervision. A deliberate compressed approximation requires measured error and a protocol revision; do not introduce it silently.

Proposed initial storage ceilings are 128 GiB for the pilot and 512 GiB for the expanded study, excluding model checkpoints already present. These are spending limits, not statements about available disk. If the measured projection exceeds them, revise the capture layout or budget before proceeding. Record optimizer/checkpoint storage separately, including the two trainable vocabulary tables and their optimizer states.

## 6. Smallest useful pilot

### 6.1 Three arms, with cutoff disabled

| Arm | Definition | Purpose |
| --- | --- | --- |
| B0: stock | Frozen existing DSpark head and backbone; no new memory/planner/prefix branch | Establish current proposal behavior and cost |
| C: capacity control | Same frozen backbone; trainable copies of the original dependency tables; a flat per-position residual adapter using current backbone features, direct anchor, latest processed target feature and pooled block features | Controls for retraining and ordinary extra capacity |
| F: full | Complete revision-3 head, including persistent memory, future planner, fixed auxiliary losses and prefix state | Tests the proposed package |

C has no historical memory, hierarchical future prediction, future auxiliary loss, or recurrent prefix state. Match its active proposal non-vocabulary trainable parameter count to F within 5% (exclude confidence-only and inactive ablation parameters) by selecting a flat hidden width in multiples of 16 before seeing accuracy. Keep its output adapter zero-initialized and dependency-table initialization identical. If the tolerance cannot be met, report the actual difference and revise the control before training. Parameter matching is not latency matching: report MACs and measured time independently.

All three generate all five candidates subject to EOS. Confidence computation is disabled for the primary proposal comparison. This prevents a head from appearing better because it prunes its hard positions. A frozen stock comparison by itself is insufficient: F must also be compared with C.

### 6.2 Fixed pilot training recipe

These are proposed starting hyperparameters, not values validated by NCM or the reviews.

- One training seed initially, with separate recorded RNG streams for reference generation, label projection, initialization, data order, proposal sampling and verifier randomness.
- C and F each receive 512 optimizer updates at effective batch size 32 anchor blocks: 256 reference warm-start updates followed by two refinement rounds of 128 updates each.
- AdamW: learning rate `1e-4` for new small modules and `1e-5` for copied dependency tables; betas `(0.9,0.95)`, weight decay `0.01`, global gradient norm cap `1.0`. Exclude biases and normalization parameters from decay.
- BF16 activations where supported; FP32 optimizer/master arithmetic as required, fixed projection targets and persistent-memory accumulators. Stage 0 establishes the exact frozen projection path.
- For each stage: 5% linear learning-rate warm-up followed by cosine decay to 10% of that stage's peak rate. Preserve optimizer moments across stages. The same schedule applies to C and F.
- Warm start: reference-token CE weight 1; F additionally uses `0.1 L_coarse + 0.1 L_fine`. Average MSE over latent dimensions; fine MSE is segment-token-weighted exactly as in revision 3.
- Refinement: `0.1 L_CE + 0.9 L_TV`, plus the same auxiliary terms for F. Normalize each loss over its own valid blocks/positions before weighting; do not let missing future labels change the scale of the remaining losses.
- Reference CE uses smooth, unfiltered temperature-1 student logits at the reference prefix. TV uses actual transformed target/student distributions at the recorded draft prefix. The two minibatches may be different and must retain their identities.
- No confidence loss in token-head training. No future target or confidence gradient may change these conditioning contracts.

For top-p/top-k student filtering, the proposed optimization surrogate selects support from detached logits, differentiates normalization/probabilities on that selected support, and stops gradients through discrete ranking/support selection. This is a piecewise, biased training surrogate at support changes; the forward distribution used for TV must still equal the actual sampling distribution. Record support-boundary behavior and numerical checks. If that cannot be implemented faithfully, do not claim to have trained on deployment-regime TV. Greedy matching is evaluated separately, not optimized through a nonexistent argmax gradient.

After warm start and before each refinement round, collect 256 training-anchor rollout blocks from each trainable arm using its then-current checkpoint and the primary serving regime. Pool the C/F banks with equal arm weight and expose both arms to the same pooled distribution-matching dataset. Refresh for the second round. This gives 1,024 producer rollout blocks total across two arms and two rounds, at most 5,120 target-prefix probability rows. The bank may be reused across updates; document its age rather than labeling every update on-policy.

Use the final scheduled endpoint for both pilot arms. Earlier endpoints diagnose learning trends only; they cannot replace the final endpoint for the reported pilot comparison. Save losses, per-branch gradient norms, normalization checks, active parameter counts and timings. Before interpreting a weak result, classify training adequacy: absent expected gradients, unstable losses, memorization, or unresolved improvement at the budget boundary are recipe problems. One optional extension is permitted before independent screening: continue BOTH arms for exactly 512 additional updates (two 256-update refinement rounds, each preceded by fresh 256-block C and F producer banks), preserving the original objectives and peak learning rates. Restart the declared per-stage LR schedule, preserve optimizer moments, and select the extended final endpoint for both. Trigger and rationale use training/first-32 diagnostics only; log this as a new recipe version. No unlimited tuning or evaluation-driven extension is permitted.

### 6.3 Pilot evaluation and decision

Evaluate the nested 32 development prompts at all available anchors, with two fresh sampled proposal paths per anchor and one greedy path. Use the acceptance estimator in section 8; keep EOS blocks. Report B0, C and F, per-domain results, paired F−C and F−B0, and measured head cost.

The pilot has three dispositions: **implementation/resource failure**, **training recipe inadequate**, or **valid but statistically preliminary comparison**. It has no architecture-rejection gate and no requirement that the point estimate exceed +0.10. Report paired prompt differences, candidate-draw variability within anchors, anchor variability within prompts, and central 80%/95% descriptive intervals with sidedness explicitly named. With only four prompt groups per domain/length cell, these intervals are preliminary.

### 6.4 Freeze an independent screening design

After the pilot and any declared training extension, freeze checkpoints, measured round-cost assumptions, useful-effect range, interval convention, and independent prompt count. The 64 unused development groups are a minimum available screen, not a powered guarantee. Use normal-approximation sensitivity `n = ceil(((1.96 + 0.842) * sigma_D / delta_useful)^2)` as a planning diagnostic for 80% power of a two-sided 95% interval excluding zero; round to a multiple of eight strata and consider uncertainty in estimated sigma. It is not a substitute for a measured variance model or a promise of coverage.

For example, assumed paired prompt SD 0.5 requires approximately 197 independent groups to detect +0.10, before small-sample/selection corrections. Repeated draws cannot eliminate between-prompt variance. Choose a fixed budget before evaluating the independent screen; if the needed corpus/cost is unavailable, report insufficient precision rather than run an unacknowledged underpowered rejection test.

Use the measured reward/cost relationship to define useful effects: away from EOS/caps, `delta_G >= G0 * ((1+h)/(1-r)-1)`, where h is added round-time fraction and r the desired time/token reduction. The old +0.10 threshold is only an illustrative effect size. A positive lower 95% endpoint supports improvement; an upper endpoint below the predefined useful effect supports futility for this trained recipe; all remaining cases are inconclusive. Training inadequacy overrides architectural futility. No point-estimate-at-the-alternative threshold is added. A funding/budget stop is recorded as such.

## 7. Component attribution after feasibility and a declared spending decision

### 7.1 Factorial core

Use the full training/development splits. Retain B0 and C. Train the following from the same original initialization, with the same dependency-table policy and token losses:

| Arm | Future planner P | Historical memory M | Prefix state H |
| --- | --- | --- | --- |
| P | Yes, with fixed auxiliary labels | No | No |
| PM | Yes | Yes | No |
| PH | Yes | No | Yes |
| F = PMH | Yes | Yes | Yes |

Memory-off arms retain direct anchor and the latest processed target feature; disable historical slot readouts. This isolates historical state from information already available at the current boundary. Prefix-off arms retain the trained predecessor-token dependency correction, fix the additional state contribution to zero, and skip state updates. Keep branch dimensions otherwise aligned; count the resulting parameter and runtime differences.

The four planner arms expose interactions: memory may help only with prefix state, and recurrence may help only when the plan is useful. Do not assume independent additive effects.

For an explicit claim that the NCM-inspired mechanism contributes, also train:

- **F-noaux:** identical inference architecture, but coarse/fine auxiliary loss weights zero. This isolates fixed future supervision.
- **F-flat:** same memory and prefix interfaces, but jointly predict the SAME four segment vectors without conditioning a later prediction on an earlier coarse output; no auxiliary loss. Each segment receives direct anchor, latest feature, global/segment backbone pools, segment embedding, and a segment-specific memory read. Its query uses available global/segment features rather than a predicted coarse vector. Define global output as the token-weighted segment mean and residuals as deviations from that mean; expand segments to token positions exactly as in F. Match active non-vocabulary parameter count within 5% and report actual cost.
- **F-flat-aux:** identical flat computation with the SAME fixed global/residual labels and weights as F. Equal coarse/fine coefficients make this objective algebraically weighted segment-mean regression. F versus F-flat-aux tests hierarchical computation under supervision; the four F/F-noaux/F-flat/F-flat-aux cells identify hierarchy–supervision interaction. No five-position independent prediction is substituted for four segments.

Thus the matrix has eight trained arms (C, P, PM, PH, F, F-noaux, F-flat, F-flat-aux) plus frozen B0. If the supervised flat control is omitted, restrict attribution to the combined package. The flat and hierarchical arms expose equal segment resolution and observed inputs; learned query parametrization and sequential conditioning are treatment differences, not an excuse to remove memory access from the flat arm.

| Interface | C | P/PH | PM/F/F-noaux/F-flat/F-flat-aux |
| --- | --- | --- | --- |
| Current anchor/latest projected feature/global and position backbone features | Yes | Yes | Yes |
| Historical slots and global/segment memory read access | No | No | Yes |
| Four segment pools and piecewise-constant plan at gamma 5 | No explicit plan | Yes | Yes |
| Prefix recurrence | No | PH only | F/F-noaux/F-flat/F-flat-aux only |
| Future auxiliary labels | No | Yes | PM/F/F-flat-aux only |

Record active proposal parameters, confidence parameters separately, initialization and update counts. Inactive tensors never count toward the capacity control.

### 7.2 Expanded training budget

Restart from the original initialization; do not give selected pilot arms extra accumulated training. Each trained arm gets 2,048 updates: 1,024 reference warm-start plus two 512-update refinement rounds, effective batch size 32. Keep the pilot's losses and optimizer. Select expanded checkpoints using only the first 32 development groups; the independent-screen groups cannot select their own checkpoints.

In every phase, ONLY C and F produce rollout banks, with equal producer weight. At expanded scale each refinement round collects 512 blocks per producer; all eight arms train against that same frozen bank for the round. C/F are updated alongside the diagnostic arms and regenerate the next bank, so optional arms cannot change the producer curriculum. Save producer checkpoint hashes and exact shared-bank IDs.

Use one seed for the attribution matrix. Then repeat C and F for two additional seeds under the IDENTICAL two-producer recipe, providing three matched C/F seeds including the matrix pair. Each pair produces 2,048 blocks across its two refinement rounds. Do not switch producer composition or data budget and call the result seed replication. Diagnostic arms receive one seed and support exploratory mechanism conclusions; replicated hierarchy-specific claims require matched flat-aux repeats as a separately budgeted extension.

Before any test evaluation, nominate one C/F checkpoint pair for serving by the predetermined development acceptance criterion, and freeze the decision. Report all three seed results, not only the nominated seed. If F loses to a simpler arm, a simpler future design is a possible conclusion; do not silently replace F as the prespecified primary test arm.

### 7.3 Diagnostics, not substitutes for acceptance

Report coarse/fine MSE, residual norms, per-position TV, proposal entropy, prefix-state norms, memory age/length strata, and accepted-prefix survival. Zeroing or swapping memory at evaluation can diagnose dependence but creates a distribution shift; it does not replace a trained memory-off control. An oracle-future-label diagnostic, if used, is explicitly noncausal and not a valid serving result or a proven upper bound.

Attribute no benefit to planning merely because MSE decreases. The deterministic MSE planner may learn an averaged future that is accurate in feature space but unhelpful for lexical agreement. This is a central risk identified by [Astra][R1], not an unexpected anomaly to exclude.

## 8. Acceptance measurement and statistical unit

For a fixed committed prefix/anchor, let a head generate its own candidate path `Y`. At position `i`, evaluate the target and proposal at exactly that draft prefix, with actual sampling transforms:

```text
rho_i(Y) = min(1, p_i(y_i | prefix, y_<i) / q_i(y_i | prefix, y_<i))
S_j(Y)   = product_{i=1..j} rho_i(Y)
E[A | Y] = sum_{j=1..L(Y)} S_j(Y)
```

Here `L(Y)` is the proposed/admitted length truncated at the first candidate EOS, and `A` counts accepted draft tokens only. Sampled candidates have positive `q_i(y_i)`; impossible candidates or nonfinite probabilities are implementation errors. Integrating acceptance uniforms this way reduces noise without altering the candidate-path expectation.

**This product uses candidate-specific acceptance ratios. It is not the product of predicted confidences or local distribution overlaps.** The latter are not exact prefix-survival probabilities. Local confidence supervision remains `c_i* = sum_v min(p_i(v),q_i(v))` at that particular prefix.

Record actual verifier outputs too. Committed new tokens per round `G` usually equals `A+1` away from terminal/output-cap boundaries, but not universally: accepted EOS ends the sequence and prohibits a bonus. Use the verifier's EOS-aware, cap-aware count for throughput and break-even calculations. Do not inflate gains by adding a nonexistent bonus.

Primary proposal metric: expected accepted draft tokens per block with cutoff off, averaged within prompt first and then equally across the four domains. Report per-position unconditional survival, local overlap, conditional survival among reached positions, and total committed tokens separately. For greedy decoding use deterministic agreement up to the first mismatch; do not reuse sampled overlap as its acceptance label.

For confirmatory acceptance, use all 128 sealed test prompts, up to eight anchors each, eight fresh sampled paths per anchor, and one greedy path. That is at most 8,192 sampled paths per head/checkpoint, with fewer after short completions. C and F get the three declared training seeds; B0 is evaluated on the corresponding independent evaluation streams. Component-screening arms remain on development data unless explicitly registered as secondary test comparisons beforehand.

Use 5,000 paired bootstrap replicates, resampling independent prompt groups within domain/length strata; retain all anchors and paths of each selected prompt together. Do not count tokens, anchors or repeated draws as independent prompts. Report each training seed's paired result; for an aggregate over both factors use crossed resampling: draw seeds and prompt groups independently, but reuse the same sampled prompt IDs for every sampled seed and preserve F/C pairing. Report checkpoint-conditional prompt intervals separately from retraining uncertainty. Three seeds provide a replication check and weak tail information, not assured population power.

F−C in the primary sampled regime is the single primary acceptance contrast. F−B0, greedy and no-filter results are secondary. Apply a declared multiplicity correction such as Holm if making several confirmatory significance claims. Screening confidence intervals are descriptive; do not present selected component comparisons as untouched confirmatory tests.

The nominal 128 test groups are a budget proposal. Before opening test outcomes, use selection/screen variance to freeze either that count with its declared detectable-effect range or a larger source-disjoint count. Preserve the sealed test identity and document the choice. Three seeds and repeated paths do not guarantee power. Leave a wide final interval inconclusive; no significance-driven expansion.

## 9. Train and calibrate confidence only after freezing proposals

The cutoff is mandatory as an architectural capability, but a useful positive threshold remains an empirical question. Test proposal quality first; then freeze the token/planner/memory/prefix parameters for each retained head.

### 9.1 Confidence labels and fitting

Collect fresh own-head sampled paths on the training split under the primary regime, at all available anchors, two paths per anchor. Evaluate full `p` and `q` at each actual path prefix, including positions that might later be cut off. Fit the detached confidence branch to soft local-overlap labels `c_i*`, using binary cross-entropy with soft targets.

Proposed confidence budget: 1,000 AdamW updates, effective batch 256 valid positions, learning rate `1e-3`, betas `(0.9,0.999)`, no weight decay, gradient cap 1, FP32 loss. Choose the endpoint with best development soft-label Brier score among steps 250/500/750/1,000. This budget is separate from token-head training. Every retained head gets its own confidence model and calibration; no old fit is inherited.

The existing [STS recorder][STS] uses prefix-correctness masks and raw confidence. Those records are not automatically the conditional distribution-overlap labels required here. Do not train this branch from `prefix_mask` without changing and justifying the label contract.

Use the first 48 calibration prompts to fit a positive scalar logit temperature per position, constrained to `[0.1,10]`, by soft-label BCE. Use a single shared temperature if a position has fewer than 200 valid positions OR fewer than 32 independent contributing prompt groups; counts from repeated paths do not substitute for prompt support. Temperature scaling preserves ordering and the 0.5 boundary and may leave systematic offset errors; report residual miscalibration rather than silently changing calibrator. Generate two fresh paths per anchor. Freeze the fit before threshold selection on the disjoint remaining 48 prompts.

Report soft-label Brier score, BCE, ten equal-mass-bin calibration error, and predicted-versus-observed acceptance among actually reached prefixes. Separate these populations: unconditional draft-prefix calibration and verifier-reached-prefix calibration can differ. Include prompt-clustered uncertainty and support near the chosen threshold. Reached-prefix estimates use actual verifier reach or candidate-specific survival weights, also applying causal admission for the selected policy. Calibration is not an exactness guarantee; correct verification supplies exactness.

### 9.2 Threshold policy and cost selection

Primary policy is the revision-3 causal conditional cutoff: admit position `i` only if its calibrated pre-draw conditional score is at least `theta`; stop at the first failure. Threshold grid:

```text
theta in {0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95}
```

Do not tune a cumulative-product policy in the primary study. A future product-policy comparison must retain its non-probabilistic interpretation and have its own selection budget.

Build a measured cost table for verification widths 1 through 6, both context strata, and the actual serving backend. Combine this with calibration-set committed-token rewards and measured memory/planner/prefix/confidence costs to select the threshold with lowest predicted time per committed token. Selection happens only on the threshold-selection split. Require both at least 2% predicted improvement and a positive paired prompt-bootstrap lower 95% endpoint versus threshold zero on selection data; report threshold-selection stability under prompt resampling. These are selection diagnostics, not independent confirmatory intervals. Otherwise retain zero and report insufficient evidence for a useful cutoff from this calibration. This 2% is a proposed noise/spending margin, not paper evidence.

The shared backbone and parallel planner still produce their configured five-position features before admission. Do not credit truncation with avoiding that already-completed work. Savings can come from skipped prefix-loop steps and genuinely shortened verification, subject to the backend's graph/padding behavior. If a backend cannot shorten physical verification work, retain it as a correctness control and mark the native-cutoff performance objective unmet.

Evaluate three modes for overhead attribution: confidence entirely off; confidence on with `theta=0`; confidence on with selected threshold. Confidence-on threshold zero still costs time and exercises fallback handling. Store head/model/transform/calibration hashes together. A regime or checkpoint mismatch invalidates the calibration and uses the architecture's pre-request full-verification fallback.

For the final frozen threshold report admitted width, accepted drafts, committed tokens, executed verification rows, padding/graph work, and total latency. A reduction in admitted width with unchanged physical work is a policy result, not a verification speedup.

## 10. Measure serial cost and opportunities for parallelism

Use device events for GPU work and end-to-end CPU wall time for the full path; synchronize only where required by the measurement protocol. Measure warm steady state at the actual shapes, precisions and graph modes. Keep I/O-heavy capture, extra diagnostic LM-head passes and debug synchronization out of headline timing.

Report memory update, coarse planner, parallel fine planner, adapter/vocabulary projection, prefix loop, confidence loop, sampling and target verification separately, as well as their overlapped total. Summing independently timed kernels is not an end-to-end latency measurement. On the CPU-expert path include host scheduling and expert I/O.

The architecture already exposes parallel fine segments and parallel expensive backbone work. Implementing those as independent serial calls would confound the experiment. Batch the fine-segment projection, jointly compute the prefix gate/candidate, and use available fused operations when numerical parity is established. Report launch counts and the critical path. Do not add a second architecture while optimizing these existing operations.

For prefix-state value, compare F with PM and PH with P using:

```text
measured throughput = total committed tokens / total round time
necessary reward/time improvement:
    E[G_with_state] / E[G_without_state]
      > E[T_with_state] / E[T_without_state]
```

The inequality is a steady-state screening criterion on comparable workloads; final closed-loop timing decides. More accepted tokens do not justify recurrence if time rises more. [Astra R2-4][R2] motivates the single-gate design; the [DSpark paper][DSPARK] provides a reason to be skeptical of large recurrent gains at short blocks, not a latency forecast for this host.

Do not preemptively add more recurrent depth, stochastic plans, bigger memory, or longer blocks when a result is weak. Possible later directions should be recorded after the component evidence: remove an unhelpful prefix state, improve shared future planning if later positions fail, or explore a compatible wider backbone if early five-token acceptance is already near saturation. Each is a new experiment, not an undeclared arm.

## 11. Closed-loop serving experiment

Proceed after numerical/correctness gates, training-adequacy review, an explicit resource decision and calibration; an underpowered pilot alone cannot veto this study as an architectural failure. The offline reference-anchor study does not reproduce the distribution of boundaries generated by each deployed head; closed-loop serving is necessary.

### 11.1 Arms and controls

Run four configurations using the nominated frozen checkpoints:

1. B0, cutoff off.
2. C, its independently calibrated selected policy.
3. F, confidence/cutoff off.
4. F, its independently calibrated selected policy.

If F selects zero, say so; its fourth arm measures confidence overhead and cannot be described as a cutoff benefit. Measure confidence-on/zero for C and F in the microbenchmark as well. The F confidence-off serving arm isolates proposal quality plus head cost from truncation.

Use the same target weights/quantization, CPU threads, expert format/residency, cache/prefetch policy, context settings, sampling transforms, request output cap, graph backend and hardware for every arm. Record thermal and storage conditions. Changing expert prefetch policy while changing the head invalidates attribution.

Use 96 of the 128 sealed test prompts, chosen by fixed IDs before results, balanced by domain and length. Partition them into six blocks of 16 prompts; every arm processes every block, with the same per-prompt evaluation seeds and a maximum of 256 new tokens. Randomize arm order in a balanced schedule fixed before running. Preserve natural EOS; do not force all requests to reach the cap or drop short completions.

Predeclare warm-up requests and cache preparation, exclude warm-up consistently, and keep startup/model-load time separately. Hot-swapping heads is allowed only after validation of request-state reset, graph-buffer compatibility and identical base-model state. Otherwise use controlled reloads. Startup overhead must be budgeted even when excluded from steady-state decode metrics.

### 11.2 Outcomes

Primary serving metric: per-prompt decode wall time divided by committed generated tokens, averaged equally across prompts within domain and then equally across domains. Report paired relative changes with the same prompt weights. Also report aggregate tokens/total decode time; the two weight long and short completions differently.

Define decode timing consistently from completion of prefill through final emitted token; separately measure TTFT, total request latency and prompt-memory initialization. Include the same count convention for the initial generated token in every arm. A request with no generated output is an error/special case to explain, not a silently excluded high-latency sample.

Report median and p95 inter-token latency, peak device/host memory, accepted draft tokens, committed tokens per round, round count, effective verification width, actual executed work, graph/padding waste, CPU utilization and expert I/O/cache statistics. Distinguish performance diagnostics from metrics collected with intrusive instrumentation.

Use paired uncertainty over prompts and crossover blocks, preserving shared run conditions. Six blocks provide limited information about run-to-run variability; do not manufacture a narrow interval by treating tokens as independent. Report order effects and per-block results. A stalled or failed run is recorded and diagnosed; repeat only under a predeclared failure/retry rule, not because its result is inconvenient.

### 11.3 Decision rule

The proposed criterion for a worthwhile full-head result is:

- All correctness gates pass, including EOS and causal sampled cutoff.
- Confirmatory F−C acceptance has a positive 95% lower bound in the primary sampled regime; report the magnitude, not only the sign.
- F's selected serving policy reduces primary decode time/token by at least 5% versus B0, with a positive 95% lower bound on the improvement.
- F also has a positive 95% lower bound on improvement versus C's selected policy, supporting the cost of the added mechanisms.
- No unexplained TTFT or p95 inter-token latency regression greater than 2%; report uncertainty rather than treating a noisy point estimate as proof of a regression.

The 5% and 2% margins are proposed engineering decision thresholds for this study, not inherited historical results or paper guarantees. If F improves acceptance but loses on time, retain the diagnosis and reject the claim of a serving win. If a simpler arm wins, report it and propose an explicit architecture revision. If intervals are wide, the result is inconclusive. No outcome automatically authorizes deployment or production changes.

## 12. Budget and stop points

| Stage | Maximum planned work before the next decision | Stop / reconsider condition |
| --- | --- | --- |
| 0 | Tiny alignment/correctness/gradient probes and resource profile | Invalid sampling, feature mismatch, infeasible native backward/capture path |
| Pilot | 128 training prompts; C/F each 512 updates; 1,024 producer blocks; 32 dev groups. Optional bounded paired extension adds 512 updates/arm and 1,024 producer blocks. | Implementation/resource failure, recipe inadequacy, or preliminary result; no architecture-rejection gate |
| Independent screen | Fixed checkpoints; at least 64 untouched groups, final count frozen from useful-effect/variance/cost analysis | Useful-recipe futility only with adequate training and declared precision; otherwise inconclusive |
| Attribution | Eight trained arms, each 2,048 updates; 2,048 shared C/F producer blocks; training plus selection split | Explore components with equal observed inputs and supervision controls |
| Seed confirmation | Matrix C/F pair plus TWO additional matched pairs; each extra pair 2,048 updates/arm and 2,048 producer blocks | Report seed variability; do not confound it with producer-recipe changes |
| Confidence | Frozen retained heads; 1,000 confidence updates each; disjoint 48/48 calibration and threshold prompts | Poor calibration, no beneficial threshold, or incorrect causal boundary |
| Final acceptance | 128 sealed prompts, up to eight anchors and eight sampled paths/anchor/head/seed | Correctness failure; otherwise report once, including inconclusive results |
| Serving | Four arms × 96 prompts, organized into six crossover blocks, plus fixed warm-up | Instability, confounded resource state, or no measurable net gain |

The planned total is two pilot fits, eight matrix fits and four additional confirmatory fits: **14 token-head fits**, conditional on spending decisions. Producer banks total 1,024 pilot + 2,048 matrix + 4,096 replication = 7,168 blocks (35,840 probability rows at gamma 5), plus an optional 1,024-block pilot extension. This excludes confidence and evaluation queries.

The nominal final acceptance study can use 368,640 target conditional rows (six learned checkpoints plus three B0 streams, each up to 8,192 paths × five positions), dwarfing the producer banks. Record logical row counts separately from actual forward calls/batching. Confidence-training maximum per retained checkpoint is 512 prompts × eight anchors × two paths × five = 40,960 rows; calibration adds up to 7,680 rows per checkpoint. Early EOS reduces these ceilings.

The preflight ledger must include processed target tokens for anchor trajectories and five-token label branches, long-prefix learned-memory replay tokens per optimizer update, full-vocabulary probability retention, model/optimizer bytes and disk free space. Checkpointed replay saves activations, not arithmetic. Record measured throughput only after a separately authorized native probe; resource estimates are not experimental results.

These counts bound updates and examples, not execution time. Estimate wall time as measured target-query/capture cost plus measured training cost plus load/warm-up/serving cost. Training from full-vocabulary rows and native target queries may cost much more than the small adapter parameter count suggests.

No threshold sweep, seed sweep, wider-block run, production restart, capture, or GPU launch is initiated by the current setup work. Before execution, resolve source datasets and the native manifest, then present the measured resource estimate. A change to architecture, target model, sampling policy, loss, split, or primary decision rule creates a new protocol version and is logged before viewing new held-out results.

## 13. Required artifacts and evidence ledger

Produce one immutable manifest per protocol version and one result directory per run containing: split IDs/group hashes; source/checkpoint/projection hashes; feature alignment examples; capture schema and probability transforms; training configuration and seed streams; selected/all endpoint checkpoints; confidence fits and threshold selection records; per-prompt acceptance results; EOS-aware verifier records; raw timing traces and cache preparation; and a report of failures, exclusions and limitations.

The final report must distinguish four claims: **correct**, **better proposals**, **useful cutoff**, and **faster serving**. Passing one does not imply the others.

| Design choice | Basis and strength of evidence |
| --- | --- |
| Preserve coarse/fine plan and test it against a flat control | [Sol S1–S4][SOL], [NCM primary paper][NCM], and [Astra review 1][R1]; mechanistic motivation, no demonstrated speculative gain |
| Fixed labels, centered fine residuals, EOS semantics and distinct prefix laws | [Astra findings R2-1, R2-2, R2-3, R2-5][R2] and [revision 3][ARCH]; mathematical/causal constraints |
| Compare single-gate prefix state with Markov-only correction and measure its critical path | [Astra R2-4][R2], [DSpark][DSPARK]; expressivity/cost rationale, native benefit unknown |
| Pilot against both frozen stock and retrained capacity control | Experimental attribution requirement; proposed control, not an NCM result |
| Test cutoff off before threshold tuning | Separates proposal quality from selection and cost; proposed evaluation design |
| Pre-draw conditional cutoff, own-prefix overlap labels, exact verifier boundary | [Architecture cutoff contract][ARCH] and [Astra review 1][R1]; sampled-correctness reasoning |
| Five-token primary horizon | Current local [serving recipe][ARM]; code observation, remote state must be verified |
| Do not reuse current capture/STS records as complete training labels | [Capture implementation][CAPTURE] and [STS implementation][STS]; code-level schema and instrumentation limits |
| Fresh own-head evaluation plus closed-loop serving | Candidate-prefix distribution and serving-boundary shifts; statistical/causal reasoning |
| Prompt-group bootstrap, sealed test, matched seeds and spending gates | Proposed experiment discipline; counts and margins are explicitly unvalidated engineering choices |

[ARCH]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md
[SOL]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-paper-sol-summary-2026-10-18.md
[R1]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-head-astra-architecture-review-2026-10-18.md
[R2]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-head-astra-revision-2-review-2026-10-18.md
[NCM]: https://arxiv.org/abs/2610.14162
[DSPARK]: https://arxiv.org/html/2607.05147v1
[ARM]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/benchmarks/dsv41_baseline/arm_env.py
[CAPTURE]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/analysis/dsv41-drive/dspark/spec_margin_capture.py
[STS]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/srt/speculative/dspark_components/dspark_sts.py
[SAMPLE]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/SAMPLE_PROMPT_DIVIX01.md
[TESTPARITY]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/test/registered/spec/dspark/test_dspark_kernel_parity.py
[TESTRAGGED]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/test/registered/spec/dspark/test_ragged_verify.py
[TESTBACKEND]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/test/registered/spec/dspark/test_ragged_verify_backend_capability.py
[TESTSTS]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/test/registered/spec/dspark/test_dspark_sts.py
[TESTSAMPLE]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/test/registered/kernels/ops/speculative/test_speculative_sampling.py

[ER]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-head-astra-experiment-review-2026-10-18.md
[SETUP]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/analysis/ncm_head/README.md
