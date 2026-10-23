# Astra review: learned continuous future labels

Date: 2026-10-18. Reviewer: GPT-6 Astra subagent `astra_tokenizer_review`.
Scope: read-only review of the proposed learned continuous encoder/decoder versus the fixed random projection. No implementation, capture or training was performed by the reviewer. This file preserves the substantive findings in an edited summary; the concrete contract in architecture revision 4 is the parent's response, not a design already approved by Astra.

## Verdict

A learned continuous future encoder is a reasonable alternative representation, but calling it a stronger acceptance candidate overstates the evidence. Reconstruction, prefix predictability and acceptance are different objectives. Preserve fixed random projection as the baseline. Learned representation, coarse-only training and shared coarse/fine conditioning recover part of NCM; they do not reproduce its discrete residual codes, sampled hierarchy or probabilistic objective.

## Findings

- **LC1 — Predictability:** features after consuming a reference token legitimately encode that token for posterior supervision. The prefix planner cannot recover irreducible future randomness. Deterministic squared error can produce a conditional mean outside the oracle decoder's training distribution; better reconstruction can enlarge this gap.
- **LC2 — Bottleneck:** five 128-dimensional continuous vectors can encode five token identities. Width is not an information-rate bound or proof of semantic abstraction. Zero-mean fine residuals enforce a weighted decomposition, not interpretability or absence of global information in residual patterns. Coarse-only reconstruction encourages usefulness but not predictability.
- **LC3 — Decoder bypass:** DSpark features or true earlier future tokens can let a reconstruction decoder ignore its latent; removing all bypasses can instead encourage lexical memorization. Specify decoder inputs, capacity and relationship to the deployed adapter explicitly.
- **LC4 — Frozen artifact and decoder fate:** either discard the reconstruction decoder and acknowledge indirect task alignment, or initialize the deployed adapter from it and train on predicted latents. Freeze normalization, statistics, position convention and EOS handling with the encoder. Arbitrary coordinate rescaling changes MSE while inverse decoder rescaling can preserve reconstruction; a fixed metric is part of the learning contract.
- **LC5 — NCM scope:** learned continuous labels restore task-directed compression, not sampled coarse decisions. NCM reconstruction and diffusion-relative speed results do not establish speculative acceptance gains. The existing recurrent token head already supplies dependence through actual earlier draws.
- **LC6 — EOS:** feeding zeros through a biased or position-aware encoder does not preserve zero terminal labels. Explicitly force post-EOS output labels to the terminal vector. Specify masks for any position mixing. Unknown tails invalidate the whole planner target; they are not shortened horizons.
- **LC7 — Confidence:** a new label representation does not change exact verification, but changed head weights require new proposal/confidence artifact identities and calibration. Admission remains pre-draw; oracle future information cannot enter confidence or runtime conditioning.

For sampled latent `z`, conditional rejection can be exact yet less efficient than using the exact marginal proposal:

```text
E_z sum_x min(p(x), q_z(x)) <= sum_x min(p(x), E_z q_z(x)).
```

For `p=(1/2,1/2)` and equally likely point-mass proposals, conditional-latent acceptance is 1/2 while the marginal matches p and accepts with probability 1. Later-token marginalization requires latent posterior weights conditional on the generated prefix. This comparison does not establish that all practical deterministic heads outperform all latent heads.

## Evidence and disposition

- [Sol paper summary](ncm-paper-sol-summary-2026-10-18.md), S1–S2 (NCM mechanism/training), S4 (few-level limits), S5 (exact reconstruction), S8 (transfer hypotheses).
- [NCM §§3.2–3.4](https://arxiv.org/html/2610.14162v1#S3): learned residual VQ tokenizer and prior over its frozen hierarchy.
- [Second architecture review](ncm-head-astra-revision-2-review-2026-10-18.md): fixed targets, weighted residual centering, EOS and predicted-coarse conditioning.

The review also distinguished Engram's normalized, hashed causal n-gram memory from a future concept codebook. The user subsequently excluded Engram reuse from this design; no Engram input, label source or codebook is selected.

The reviewer recommended specifying a complete reconstruction/coordinate/freeze/deployment contract before any replacement of the baseline. Encoder width, reconstruction decoder, losses and terminal implementation in revision 4 are explicit engineering choices requiring validation, not paper-established optima or reviewed performance results.
