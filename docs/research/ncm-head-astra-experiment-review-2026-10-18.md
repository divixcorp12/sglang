---
title: "Astra review: NCM experiment power and attribution controls"
description: "Critical review of pilot sensitivity, checkpoint selection, training adequacy, hierarchy controls, replication, and confidence evaluation."
---

# Astra review: NCM experiment power and attribution controls

Date: 2026-10-18. Status: **Review only. Recommendations have not been applied to the experiment plan. No training, capture, simulation, benchmark, or GPU work was performed.**

## Provenance

The user explicitly requested a GPT-6 Astra subagent review, particularly whether the pilot can detect a useful gain and whether the controls fairly isolate NCM ideas. This document preserves and organizes the response from `/root/astra_experiment_review`, using `gpt-6-astra` with high reasoning effort. The parent checked the referenced plan passages and analytical relationships, and saved this edited preservation. The subagent performed read-only inspection and did not modify files.

Reviewed sources:

- [Experiment plan][PLAN], 414 lines, SHA-256 `44018d5809690b28e196686f7263f4882ed45758f39702e52b332780e48c9995`.
- [Architecture revision 3][ARCH], SHA-256 `1496a900000e21bfa3737e2ff0c9274c9ff34362004948261db6abf2d79f2a0b`.
- [GPT-6.1 Sol paper summary][SOL] and [first][R1] / [second][R2] Astra architecture reviews.

Line references identify the reviewed experiment-plan snapshot, not later revisions. Numerical power examples below are analytical approximations under stated assumptions, not measured variability, simulations, or experimental results. No new literature claim is required for these findings.

## Verdict

**The plan is a sound starting protocol, but its 128-training/32-development pilot should remain a feasibility and variance-estimation study. It is not yet a reliable gate for rejecting useful gains or deciding whether NCM-inspired structure deserves further study.**

The most consequential corrections are to separate checkpoint selection from screening, distinguish inadequate training from inadequate statistical power, and complete the supervised flat-planner control before making a hierarchy-specific claim.

The plan already gets several difficult things right: cutoff-disabled proposal comparisons, retrained capacity control, own-prefix target/proposal probabilities, prompt grouping, separate calibration and threshold selection, and a final closed-loop serving comparison. The corrections below strengthen what these experiments can establish; they do not establish that the architecture will succeed or fail.

## Findings at a glance

| ID | Priority | Finding | Minimum correction |
| --- | --- | --- | --- |
| E1 | P1 | The small pilot is not a reliable effect-detection/futility gate | Make it feasibility/variance estimation; specify interval sidedness and size independent screening from observed variability |
| E2 | P1 | Checkpoint selection and screening reuse development outcomes | Fixed pilot endpoint or separate selection and screening groups |
| E3 | P1 | Evaluation power cannot diagnose insufficient training | Add a training-adequacy disposition and a bounded, predefined response |
| E4 | P1 for hierarchy attribution | Missing flat-with-supervision cell; flat resolution differs | Add supervised flat segment prediction with matching interfaces, or limit claims to the combined package |
| E5 | P2 | Parameter matching does not fully specify information access | Declare exact inputs, memory reads, active parameters and initialization per arm |
| E6 | P2 | Producer mixtures change between attribution and replication | Keep the compared training procedure fixed or explicitly test the recipe change |
| E7 | P2 | Three seeds and 128 test prompts do not assure power | Separate checkpoint-conditional from retraining uncertainty; preserve crossed seed/prompt structure |
| E8 | P2 | The useful-effect threshold and anchor distribution are not deployment-derived | Relate gain to measured cost and evaluate native-regime prefixes |
| E9 | P2 | Fit counts omit potentially dominant evaluation/replay costs | Add target-row and memory-replay ledgers |
| E10 | P2 | Calibration counts and threshold selection understate clustering/uncertainty | Report prompt support, reached-prefix calibration and threshold stability |

## E1. The pilot can miss a useful direction through lack of sensitivity

**Plan:** lines 123, 189–197 and 368; [pilot gate][PILOT].

The pilot has **32 independent prompt groups**, with only four groups in each domain-by-length cell when perfectly balanced. Up to eight anchors and two candidate paths improve each prompt's estimate; they do not create 512 independent prompts. Shared contexts, related anchors and repeated draws remain clustered.

The promising rule requires both an estimated gain of at least `0.10` and an 80% lower bound above zero. With unbiased, approximately normal estimates, **a true gain of exactly `0.10` passes the point-estimate requirement only about half the time**, even as estimation becomes more precise. The confidence requirement can reduce that probability further. Expanding to 96 prompts improves precision but does not remove this threshold effect. The actual plan's checkpoint-selection bias is a separate complication addressed in E2.

“80% bootstrap lower bound” needs a sidedness definition. The lower endpoint of a central 80% interval is a one-sided 90% bound, not a one-sided 80% bound.

For illustration, assume fixed checkpoints, no selection bias, independent prompt groups and paired prompt-level F−C standard deviation `sigma_D`. Ignoring stratification and small-sample corrections gives `SE ≈ sigma_D / sqrt(n)`.

| Assumed paired SD, tokens/block | SE at 32 prompts | Approximate probability of passing current gate at true +0.10, interpreting the interval as central 80% | Approximate prompts for 80% power to obtain a positive lower endpoint of a two-sided 95% interval at true +0.10 |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.044 | 50% | 50 |
| 0.50 | 0.088 | 44% | 197 |
| 1.00 | 0.177 | 24% | 785 |

The last column uses:

```text
n ≈ [(1.96 + 0.842) * sigma_D / 0.10]^2
```

These are sensitivity illustrations, not mandatory sample counts or measured standard deviations. Actual paired variability may be smaller or larger.

**Correction:** retain the small pilot, but label a weak result “insufficient evidence from this feasibility recipe.” Do not make `+0.10` plus an 80% bound a prerequisite for attribution work. Use variance estimates to choose a separately frozen screening budget and specify interval sidedness. A deliberate budget stop remains permissible; it should not become evidence that the mechanism lacks useful effects.

## E2. Checkpoint selection and screening reuse the same development outcomes

**Plan:** lines 185–195, 225 and 259–263; [endpoint selection][SELECT].

Each arm selects its best of three endpoints on development acceptance, then the selected F−C comparison is screened on the same prompts. An ordinary paired bootstrap of those selected checkpoints does not account for this selection. Separate selection for C and F does not make the problem cancel: their endpoint variability and learning trajectories can differ.

The optional expansion from 32 to 96 includes the original selection data. It therefore does not create a clean independent evaluation. Calling intervals exploratory limits the claim, but does not make them reliable evidence for stopping a research direction.

**Correction:** use the final scheduled endpoint for the feasibility pilot, or use the first 32 groups solely for selection and reserve the other 64 development groups for one independent screening comparison. Sixty-four is a clean split, not a guarantee of adequate power. If all 96 groups subsequently support component selection, those later comparisons remain exploratory. Preserve the sealed test.

## E3. Training adequacy and statistical power are separate unresolved questions

**Plan:** lines 167–185 and 223–229; [training recipe][TRAIN].

The pilot has at most 1,024 distinct reference anchor blocks. Its 256 warm-start updates at batch 32 provide 8,192 block presentations—approximately eight passes if every prompt supplies eight anchors. More updates repeat these examples; they do not add independent contexts or future realizations.

This could be adequate for adaptation, or inadequate for learning the newly initialized memory, planner and recurrent components. The documents establish neither. One fixed future realization per anchor is a valid regression label, but a small dataset can encourage fitting sampled futures rather than useful conditional statistics.

Equal updates and learning rates establish a fair **fixed-recipe, fixed-update comparison**. They do not establish that both architectures received adequate optimization. Zero output projections initially block token-loss gradients into upstream new branches; auxiliary supervision gives F another early learning path. That is a legitimate effect of the proposed recipe, but learning-speed differences should not be mistaken for an optimized architecture comparison.

**Correction:** add a training-adequacy disposition before statistical futility. Verify that relevant branches receive gradients, record endpoint learning trends, and identify unresolved improvement at the budget boundary. Predeclare at most one bounded training-budget extension or a small shared recipe adjustment using training/selection data. An unconverged or clearly data-limited comparison is “recipe inadequate,” not “mechanism ineffective.” Expanding evaluation prompts cannot fix undertraining.

## E4. The ablations omit a supervised flat planner

**Plan:** lines 53–57 and 216–221; [hierarchy controls][CONTROLS].

The current comparisons identify:

- F versus F-noaux: supervision contribution **within the hierarchical architecture**.
- F-noaux versus F-flat: hierarchical-versus-flat computation **without auxiliary supervision**.
- F versus F-flat: the combined hierarchy-plus-supervision package.

They do not identify whether hierarchical computation helps when both designs receive future supervision, or whether hierarchy and supervision interact. A positive F−F-flat result could arise from generic future regression rather than coarse-conditioned planning.

A natural control jointly predicts the same four segment vectors `mhat_j`, then defines:

```text
ghat       = sum_j w_j * mhat_j
deltahat_j = mhat_j - ghat
```

Apply the same fixed global/residual labels and loss weights. With the proposed equal coarse/fine coefficients:

```text
||ghat - g*||² + sum_j w_j ||deltahat_j - delta*_j||²
    = sum_j w_j ||mhat_j - m*_j||²
```

The identity follows by decomposing each segment error into its weighted mean and a weighted-zero-mean residual; the cross term vanishes. Therefore the existing supervision has a generic segment-regression counterpart that does not require a coarse prediction to condition a second prediction stage.

**Correction:** add `F-flat-aux`, completing the hierarchy-by-supervision comparison, or explicitly limit the conclusion to “the combined package helped.” The additional arm is needed for stronger attribution, not for establishing F's practical value.

Also align the control's resolution. F has four segment vectors with piecewise-constant reconstruction; F-flat currently predicts five per-position vectors. For a clean computation comparison, use the same segment partition, summary interface, available backbone features and memory access. A five-position flat alternative is valid, but it also tests a different output resolution.

The completed core comparison would be:

| Computation | No future auxiliary loss | Same fixed future auxiliary loss |
| --- | --- | --- |
| Coarse-conditioned, centered hierarchy | F-noaux | F |
| Joint flat prediction of the same segments | F-flat | F-flat-aux |

Hold memory and prefix-state interfaces fixed across these four arms. This tests the proposed continuous hierarchy and its supervision, not the entire discrete NCM mechanism.

## E5. Capacity matching does not fully specify information and optimization access

**Plan:** lines 159–164 and 203–221; [capacity control][CAPACITY].

C is a useful package control: it retrains dependency tables and adds ordinary adapter capacity. Its missing history and prefix state are intentional treatment differences, so C does not need to reproduce those mechanisms.

However, “same direct inputs” is too loose for the flat hierarchy control. F has segment-specific pooling and query-dependent memory reads. A flat predictor receiving only one global pool or one generic memory summary has a different information bottleneck, even if parameter counts match.

Define whether matching counts only parameters active in proposal training/inference. The disabled, untrained confidence branch should not determine C's token-model capacity. Inactive parameters in ablated branches are not effective trainable capacity.

**Correction:** add an interface table listing exact input tensors, memory-read access, segment resolution, active parameter count and initialization for each arm. Report equal-update performance and measured cost. Keep the warning that parameter matching is not latency matching. P/PM/PH/F supports memory/prefix effects conditional on a planner; do not generalize those effects to every architecture.

## E6. The matrix and confirmation use different training procedures

**Plan:** lines 227–231 and 375; [producer mixtures][PRODUCERS].

The plan correctly refuses to count the seven-producer matrix as a fourth matched seed. Nevertheless, switching to two producers changes the distillation curriculum and off-policy prefix distribution. Failure to reproduce a matrix gain could reflect that change rather than seed variability. Within the matrix, performance also depends on whether optional diagnostic producers were included.

**Correction:** make a common C/F producer bank the fixed data source for every attribution arm and confirmation pair, or designate the seven-producer matrix as a separate exploratory recipe and perform a two-producer C/F development comparison before calling subsequent seeds confirmation. Record producer composition as part of the treatment definition. The suggested common bank must be explicitly defined and held constant across the arms it is meant to control; it does not imply that differently seeded future banks will contain identical samples.

## E7. Three seeds and 128 test prompts provide conditional evidence, not assured power

**Plan:** lines 229–231 and 257–263; [final statistical protocol][STATS].

Three seeds are a useful minimum replication check, but weak support for the distribution of training outcomes. Eight paths per anchor reduce candidate-sampling variance; they do not multiply the number of independent training outcomes or prompt groups.

Under the fixed-checkpoint normal approximation used in E1, 128 prompts have approximately 80% power for gains of `0.062`, `0.124` or `0.248` when paired prompt SD is respectively `0.25`, `0.50` or `1.00`. Training variance adds uncertainty.

Shared prompts across seeds create a **crossed seed/prompt structure**. A vaguely specified hierarchical bootstrap may resample independent prompts within each seed and discard common prompt effects. For inference over both factors, resample seeds and prompt groups while preserving the same sampled prompt identities across seed comparisons.

As another illustration—not an observed result—if three seed-level mean gains had SD `0.10`, a conventional two-sided t interval based only on those three means would have a half-width around:

```text
4.303 * 0.10 / sqrt(3) ≈ 0.25
```

A bootstrap cannot manufacture reliable tail information from three training outcomes.

**Correction:** distinguish uncertainty conditional on evaluated checkpoints from uncertainty over retraining. Report all seed effects and define crossed resampling. Choose a fixed final sample budget after development variance estimates; do not promise power from the current counts. Retain the prohibition on significance-driven test expansion.

## E8. The usefulness threshold and anchor distribution are not tied to deployment

**Plan:** lines 34–37, 119, 193 and 307–316; [reward/time comparison][COST].

`+0.10` accepted tokens is not inherently useful. Away from terminal boundaries, if baseline committed reward is `G0`, added round-time fraction is `h`, and desired time/token reduction is `r`, the simplified requirement is:

```text
Delta_G >= G0 * [(1+h)/(1-r) - 1]
```

For illustration, `G0=3`, `h=2%`, and desired reduction `r=5%` require approximately `+0.22` committed tokens per round. With no overhead, `+0.10` on `G0=3` yields approximately a 3.2% time/token reduction. Actual verification-width costs and EOS behavior require the measured model already contemplated by the plan.

Separately, the fixed anchor prefixes come from temperature-1 reference continuations, while intended serving uses temperature 0.6/top-p 0.95. The comparison is valid on common reference anchors, but is not automatically representative of native serving boundaries. This matters when a small offline result can prevent reaching the serving study.

**Correction:** keep `+0.10` only as an exploratory convention, and derive the meaningful effect range from measured reward/cost before inferential screening. Preserve temperature-1 future supervision while adding common evaluation anchors drawn from the intended target sampling regime, or prohibit strong deployment-negative conclusions from the reference-anchor pilot. Target-regime anchors improve relevance but still do not replace closed-loop serving boundaries.

## E9. Resource feasibility needs query, replay and evaluation accounting

**Plan:** lines 137–151, 257, 271–277 and 365–377; [resource accounting][RESOURCES].

The stage-0 resource probe is appropriate. Include repeated learned-memory replay over long committed prefixes, not only the five-token adapter forward/backward. Checkpointing reduces peak activation memory; it does not remove this work.

Final acceptance can exceed training-rollout evaluation substantially. The maximum is 8,192 paths per checkpoint/stream, or 40,960 target-prefix probability rows. Six learned checkpoints imply up to 245,760 rows; B0 on three corresponding streams adds up to 122,880, totaling **368,640**. Pilot, matrix and confirmation producer banks together contain at most **71,680** rows. Confidence collection and calibration are additional.

These are maximum logical row counts, not separate-forward counts or runtime predictions; batching, short completions and reuse affect cost.

**Correction:** add a stage-level ledger of target conditional rows, prefix lengths, memory-replay tokens per update, full-vocabulary storage, optimizer storage and retention policy. Let the pilot validate the projection. Fifteen nominal fits and disk ceilings alone cannot establish feasibility.

## E10. Confidence evaluation needs stronger support and uncertainty reporting

**Plan:** lines 271–295; [confidence protocol][CONFIDENCE].

Own-prefix overlap labels, detached confidence fitting, separate calibration/threshold prompts, and confidence-off/on-zero/on-selected modes are strong choices.

However, 200 valid positions do not imply 200 independent calibration examples; many may come from a few prompts. Ten calibration bins and an eleven-value threshold grid on 48 prompt groups can be noisy. A 2% point-estimated cost advantage is a spending margin, not evidence that the selected threshold truly saves 2%.

Positive temperature scaling cannot correct every calibration defect: it preserves ranking and the location of probability 0.5. Systematic offset errors can remain. This does not require a more elaborate calibrator preemptively, but the limitation should be acknowledged.

**Correction:** report calibration support and uncertainty by prompt group and position, including support near the selected threshold. Define the reached-prefix population using actual verifier reach or candidate-specific survival weights; include causal admission when evaluating the selected policy. Report paired uncertainty and selection stability for predicted cost differences, retaining zero when evidence is inadequate. Keep the final independent serving comparison decisive.

## Sampled correctness: no new defect established

The sampled acceptance estimator in plan lines 241–255 is mathematically appropriate under its stated own-path probabilities and independent acceptance randomness. Integrating acceptance coins reduces noise without replacing prompt clustering.

The exhaustive tiny-vocabulary checks, deliberately biased negative controls, live-boundary verification and backend tests in lines 88–100 are also appropriate. **This review establishes no new sampled-correctness flaw.** A finite test remains evidence about implementation, not a proof of exactness over all inputs.

## Concrete minimum revision recommended by Astra

Retain the existing 128/32 pilot, with these changes:

1. Treat it as feasibility, training diagnostics and variance estimation. Use a fixed endpoint or separate checkpoint-selection data.
2. Preserve C/F, cutoff off, matched anchors and two paths initially. Record within-prompt sampling variability so later spending can favor additional independent prompts when appropriate.
3. Replace the architecture go/no-go rule with three dispositions: implementation/resource failure, training recipe inadequate, or valid but statistically preliminary comparison.
4. Before one independently evaluated screen, freeze a practically useful effect range, interval convention, prompt budget and explicit uncertainty outcome. More sampling is justified only by the variance component it reduces.
5. Align flat-control segment/input interfaces and add flat-with-auxiliary-supervision for hierarchy attribution, or restrict the claim to the combined package.
6. Keep producer-bank composition consistent across the comparison being replicated. Preserve the sealed test and closed-loop serving study.

The architecture supplies structural facts: fixed labels avoid moving-target gradients; centering preserves the reconstructed global mean; prefix state can distinguish histories sharing their last token; and causal admission with valid rejection/boundary sampling preserves the target law. It does **not** supply an acceptance effect size, a sufficient training set, negligible recurrent cost, or a useful cutoff threshold. The pilot should measure whether the study can answer those questions rather than prematurely deciding their answers.

[PLAN]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md
[ARCH]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md
[SOL]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-paper-sol-summary-2026-10-18.md
[R1]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-head-astra-architecture-review-2026-10-18.md
[R2]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/research/ncm-head-astra-revision-2-review-2026-10-18.md
[PILOT]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:189
[SELECT]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:185
[TRAIN]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:167
[CONTROLS]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:216
[CAPACITY]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:159
[PRODUCERS]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:227
[STATS]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:257
[COST]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:307
[RESOURCES]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:365
[CONFIDENCE]: /Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md:271
