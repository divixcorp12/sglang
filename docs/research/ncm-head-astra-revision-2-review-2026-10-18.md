---
title: "Astra second review: future residual supervision and prefix-head cost"
description: "Critical review of architecture revision 2, with fixed residual targets, a constrained hierarchy, and a cheaper prefix-state recommendation."
---

# Astra second review: future residual supervision and prefix-head cost

Date: 2026-10-18. Status: **Review only; proposed corrections have not been applied to the architecture. Experiments remain deferred.**

## Provenance

The user requested another GPT-6 Astra review, emphasizing whether future residual supervision makes sense and whether prefix awareness warrants its serial cost. This document preserves and organizes the second response from `/root/astra_architecture_review`, using `gpt-6-astra`. The parent independently checked the key gradient derivation, operation count, local predecessor-only head, and cited DSpark evidence.

- Reviewed [architecture revision 2][PLAN]: 646 lines; SHA-256 `8558fed82089dee1e54a8f92cea6967b4441349783d1ee7132534e0a60d3f3b3`.
- Previous [Astra review][FIRST] and preserved [GPT-6.1 Sol paper summary][SOL].
- Local code checked at `d82ae69d0368478b5018d4cbfa92ab4643a43f54`.
- Primary external reference: [DSpark][DS], especially §§3.1 and 4.3.2.

Line references below identify the reviewed revision, not future edits. The review is based on the documents, source code, mathematical reasoning, and stated paper results. No experiment, training run, serving benchmark, dataset protocol, or runtime implementation was designed or executed.

## Verdict

**Revise with a small set of architecture corrections. Keep the future residual planner, prefix-aware interface, and native cutoff.**

The revised cutoff resolves the first review's concern; Astra found no new sampled-correctness defect in the stated valid-proposal path. The remaining issues concern the auxiliary objective, the global/local decomposition, label alignment, and avoidable serial work. They are design decisions that can be resolved now without demanding empirical proof before a research proposal can exist.

Future supervision is coherent as prediction of multiscale future-feature statistics. It is not yet a structurally enforced residual decomposition, and its current stop-gradient creates an inadequately justified optimization rule. Prefix state removes a real expressivity limitation, but the particular reset-before GRU has more sequential dependency than the argument for prefix awareness requires. Its net acceptance/latency benefit is unknown.

## Findings

| ID | Severity / type | Finding | Recommended correction |
| --- | --- | --- | --- |
| R2-1 | P2: learning contract | The moving, detached residual target produces a semi-gradient rather than ordinary reconstruction gradients. | Use fixed global/local residual labels and explicit gradient paths. |
| R2-2 | P2: representation contract | Fine predictions can cancel the entire coarse vector. | Enforce token-weighted zero-mean fine residuals. |
| R2-3 | P2: target definition | Variable valid-prefix means do not automatically match a fixed runtime horizon. | Define fixed-horizon EOS and truncation semantics. |
| R2-4 | P2: avoidable serial cost | Reset-before recurrence introduces a second dependent dense phase without being necessary for prefix awareness. | Use a single-gate cell with jointly computed gate/candidate. |
| R2-5 | P2: label alignment | Reference-plan, actual-prefix distribution, and cross-entropy labels require distinct conditioning contracts. | Declare the reference law and each label's valid prefix. |

### R2-1. The residual objective uses a semi-gradient

**Plan:** [lines 440–448][LOSS].

Let `z` be the coarse prediction, `m` a fixed segment target mean, and `f(z)` the fine predictor. The proposed fine loss is:

```text
L_fine = ||f(z) - stop_gradient(m-z)||²
e      = z + f(z) - m
```

Its numerical value equals the squared error of the reconstructed segment. Its gradient with respect to the coarse input, holding other inputs fixed, is nevertheless:

```text
specified stopped-target gradient:  2 J_f(z)^T e
ordinary reconstruction gradient:  2 (I + J_f(z))^T e
```

Detaching the target does not isolate coarse training: the fine predictor still backpropagates through its coarse input. It removes only the direct coarse contribution to the reconstruction gradient.

For example, if `f(z)=c-z`, the reconstruction is `c`, independent of `z`. The ordinary reconstruction gradient is zero, but the specified gradient is `-2(c-m)`. With `c=.7` and `m=1`, that gradient is `.6` even though moving `z` cannot change the reconstructed result in this case.

This does not prove instability or failure. Semi-gradient objectives can be deliberate. It does establish that the stated rule is not ordinary reconstruction optimization and needs a rationale beyond “detach the target.”

**Smallest correction:** define fixed reference decomposition targets:

```text
g*       = global future-feature mean
delta*_j = segment future-feature mean - g*
L_coarse = ||z0 - g*||²
L_fine   = sum_j w_j ||Delta_j - delta*_j||²
```

The fine predictor still consumes the predicted `z0`, as it will at inference. Its gradients now optimize a conventional loss against fixed labels. If auxiliary coarse/fine gradient separation is desired, detach `z0` specifically on the fine-loss input path and document that separate choice. It is not necessary merely because the labels are residuals.

An alternative is to preserve error-correction semantics and optimize the complete reconstruction loss without stopping its direct coarse term. That is mathematically coherent, but does not address the cancellation issue below.

### R2-2. The hierarchy permits coarse cancellation

**Plan:** [lines 224–234][PLANNER] and [lines 444–448][LOSS].

With `z_j=z0+Delta_j`, the fine stage may learn `Delta_j ≈ m_j-z0`. It can replace the coarse prediction with a direct segment prediction. The coarse loss still penalizes a wrong coarse output, so this is not an unrestricted zero-loss degeneracy when all targets are attainable. However, the reconstruction does not require the coarse stage to carry the global component actually delivered downstream.

The current text says the reconstruction gives the stages distinct roles. More precisely, its losses encourage roles while permitting the fine stage to cancel the coarse contribution.

**Recommended correction:** center the predicted residuals using fixed token-count weights:

```text
w_j      = segment_length_j / gamma
raw_j    = FinePredictor(predicted_z0, local_features_j, ...)
Delta_j  = raw_j - sum_k w_k * raw_k
z_i      = z0 + Delta_segment(i)
```

Then:

```text
sum_j w_j * Delta_j = 0
sum_j w_j * z_j     = z0
```

The fine stage cannot replace the global component with a common offset. Combined with fixed targets `delta*_j=m*_j-g*`, this creates an actual global/local decomposition. It does not guarantee interpretable concepts or useful acceptance, and it need not do so to be a valid architecture.

Weights must reflect token counts. For `gamma=5,J=4`, the current segmentation has sizes `2,1,1,1`; uniform `1/J` weighting would not preserve the global token mean. Centering adds a small reduction over fine slots after their parallel computation, not a serial token-generation loop.

### R2-3. Define a consistent horizon, EOS, and truncation contract

**Plan:** [lines 438–450][LOSS].

The coarse label currently averages “valid future positions,” while runtime planning uses fixed `gamma` positions and fixed segments. A shorter valid-prefix mean is a legitimate random-horizon statistic, but it is not the same quantity as a fixed-horizon mean. Segment labels observed only when their tokens exist also have different conditional populations.

That matters particularly when enforcing the fixed weighted decomposition above. Fixed runtime weights and variable label weights cannot be mixed without an explicit target definition.

**Smallest concrete contract:**

1. Define the planner over the fixed `gamma`-position horizon.
2. Specify a fixed training-only terminal representation for label positions after a genuine EOS. It can be an explicitly defined terminal vector in the label space; no real generation after EOS is implied.
3. Treat positions missing because of capture truncation as unknown, not as EOS. Do not silently replace the full-horizon global target with a shorter mean.
4. Mask every loss term whose required target is unavailable. In particular, a residual label requiring the global mean is unavailable when that global mean cannot be formed. These masks are training bookkeeping and cannot become runtime conditioning.

Another consistent convention is possible. The requirement is agreement among target meaning, segment weights, reconstruction, and inference horizon. This is an architecture definition, not a data-collection or training-schedule prescription.

### R2-4. Keep prefix awareness, simplify the recurrence

**Plan:** [lines 250–279][PREFIX] and §15. **Code:** [predecessor-only output bias][MARKOV].

The prefix state provides a genuine capability. At fixed context and position, the existing head's `compute_step_bias` discards `hidden_states` and uses only the preceding token embedding. It cannot distinguish sampled histories `A C` and `B C` when choosing the next token. A recurrent state can retain that distinction.

That argument does not require a reset-before GRU. The current cell uses three `r x 4r` projections, or `12r²` multiply-accumulates per update: `196,608` at `r=128`, excluding embedding projection, confidence, and state-to-correction projection. More consequential than the count is the dense dependency chain: the reset gate must be computed before the candidate projection can consume `reset*s`.

DSpark uses a joint projection for its recurrent gate, candidate, and output components. Its recurrent variant adds only modest gains over Markov, mostly at longer proposals, and Markov remains its deployment default. Its reported `0.2%–1.3%` overhead is full-round latency versus DFlash at batch 128 with contexts 512–4096; it does not establish the cost of this proposed GRU or a small-batch path. [DSpark §§3.1 and 4.3.2][DS]

**Recommended default:** retain the existing-shaped low-rank correction but replace the reset-before cell with a single-gate update:

```text
[g_pre, b_pre] = W_joint * concat(x_i, s_i)
g_i           = sigmoid(g_pre)
b_i           = tanh(b_pre)
s_(i+1)       = g_i*s_i + (1-g_i)*b_i
```

The gate and candidate can be calculated in one joint dense phase. State-dependent gates and nonlinear memory still distinguish earlier histories. For the current `x_i` width `3r`, the joint projection has `8r²` MACs, but this algebraic reduction is not a latency estimate.

Keep `v_i=E_dep(previous_i)+P_state*s_i`, its single low-rank vocabulary correction, and the shared pre-candidate state in confidence. The zero initial state projection remains compatible with baseline-logit initialization.

**Cost-aware implementation requirements, not implementation claims:**

- Precompute contributions from known position features `d_i,z_i` across the block where the linear algebra permits it.
- Avoid an unnecessary dense embedding projection after every sampled token if a checkpoint-specific projected lookup is preferable; account for the extra table storage.
- Skip a state update after the final candidate if nothing consumes that state.
- Do not assume the existing fused Markov kernel supports recurrent state. State production, token synchronization, confidence, vocabulary reduction, and tensor-parallel communication need a compatible schedule.
- A small arithmetic ratio relative to `V*k` is not proof of negligible cost. Additional launches and dependent reductions can matter.

A cheaper alternative is an input-gated affine state:

```text
g_i       = sigmoid(W_g*x_i)
b_i       = W_b*x_i
s_(i+1)   = g_i*s_i + b_i
```

Its coefficients do not depend on `s_i`, enabling associative scans when the tokens are already known. It is less expressive than state-dependent gating. Inference still depends on the newly sampled token in `x_i`; it is not fully parallel token generation. Bounded-history features are another option, but only cover their declared window.

The conclusion is therefore specific: **the prefix-aware interface has a sound representational rationale; the reset-before GRU's extra serial dependency is not justified by that rationale or by the cited evidence.** A single-gate state is a better default research design. Net speed and acceptance benefits remain hypotheses.

### R2-5. Separate three label-conditioning contracts

**Plan:** [lines 438–452][LOSS] and [lines 459–468][OBJECTIVES].

The distinction between reference and actual draft paths is improved, but should cover all objectives explicitly:

| Objective | Required conditioning |
| --- | --- |
| Future-plan labels | A reference continuation under a declared law, sharing the planner's committed prefix and anchor |
| Distribution matching and confidence | Target `p_i` and proposal `q_i` evaluated at the same actual candidate prefix |
| Token cross-entropy | A token label paired with the prefix from which that label is valid |

An unchanged reference next token is not automatically a valid target draw after an arbitrary divergent candidate prefix. The plan's existing warning addresses mismatched `p` and `q`, but does not explicitly address this cross-entropy case.

Specify whether the reference continuation is target-generated or comes from another declared source. That choice determines the auxiliary target distribution; it is more than a data-loading detail. No dataset size, sampling budget, or experiment protocol is needed to state it.

The reference future need not match every later sampled draft. The plan is fixed before those draws and represents a prior forecast. The ephemeral prefix state adapts to the actual branch. Do not repair divergence by feeding post-draw target features into the fixed planner.

## What the future supervision can legitimately claim

The fixed random projection is defensible as a simple label-space choice. Freezing it prevents the label encoder from learning a constant solution. It does not prevent a predictor from learning a nearly constant conditional mean, losing distinctions in the projection's nullspace, emphasizing common hidden-state directions, or being ignored by the token adapter.

Fully specify the projection's distribution, scaling, row normalization, normalization epsilon, feature-layer identity, and serialized weights. There is no architectural necessity to replace it with a learned encoder, but its connection to acceptance is an auxiliary hypothesis.

Under squared error and sufficient capacity, deterministic future-feature predictions approach conditional means. The hierarchy is thus a multiscale future-feature predictor, not a device that independently resolves multiple possible futures or reproduces NCM's stochastic coarse choices. Sol S1, S2, and S7–S8 distinguish those mechanisms [SOL].

The second planning stage adds no new observed information. Its rationale is compact hierarchical computation and communication through the coarse slot. Retaining two small stages is reasonable for the requested coarse-to-fine architecture, provided the plan does not claim that two stages are mathematically necessary or that auxiliary regression guarantees better proposals.

## The revised sampled cutoff passes this review

**Plan:** §§9–11, especially [lines 333–383][CUTOFF]. Astra found the following corrections coherent:

- Conditional acceptance is the default; the product is explicitly a score.
- `STOPPED` is permanent.
- Confidence validity is inspected only at the next visited position.
- Memory/head fallback selection occurs before candidate inspection.
- Invalid proposal recovery is separate from confidence fallback.
- A cutoff boundary can use a fresh target draw or a valid accepted-or-corrected boundary token.
- Physical full-block pre-drafting must preserve the actual proposal and logical admission dependencies.

Recurrent state does not invalidate the proof: `q_i` remains a causal normalized proposal. The accepted mass `min(p_i,q_i)` plus corrected residual mass `(p_i-q_i)_+` still equals `p_i`. This approves the mathematical contract for valid proposals and declared randomness, not an unimplemented backend or its numerical behavior.

## Minimum changes sufficient to settle the research architecture

1. Use fixed global/local targets and explicit ordinary gradient paths.
2. Enforce token-weighted zero-mean fine residuals and a consistent horizon/EOS contract.
3. Use a single-gate recurrent cell while retaining prefix-aware token/confidence interfaces.
4. Declare separate reference-plan, actual-prefix matching, and cross-entropy alignment rules.
5. Preserve the revised cutoff contract and describe planning as an auxiliary inductive bias.

Astra's stated conclusion is that these corrections would make the architecture a coherent research proposal. They do not establish a serving benefit, nor is such proof required to finish specifying a research design. Experiments, budgets, training schedules, and acceptance benchmarks remain for later. The original plan is unchanged by this review.

[PLAN]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md
[FIRST]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-head-astra-architecture-review-2026-10-18.md
[SOL]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-paper-sol-summary-2026-10-18.md
[LOSS]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md:438
[PLANNER]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md:224
[PREFIX]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md:250
[OBJECTIVES]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md:459
[CUTOFF]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md:333
[MARKOV]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/python/sglang/srt/models/deepseek_v4_dspark.py:476
[DS]: https://arxiv.org/html/2607.05147v1
