---
title: "NCM paper: GPT-6.1 Sol extraction and architectural implications"
description: "Preserved findings from the paper-reading subagent, with page references and explicit limits on transfer to speculative drafting."
---

# NCM paper: GPT-6.1 Sol extraction and architectural implications

## Provenance and how to use this summary

This document preserves the findings returned by the **GPT-6.1 Sol** paper-reading subagent `/root/paper_extraction` in this conversation on **2026-10-18**. The parent agent organized that response into stable, citable sections and condensed repetition. It is an edited preservation of the agent's findings, not a claim that the subagent originally wrote this file. Parent-agent repository checks are labeled separately.

The subagent extracted all 40 PDF pages and visually checked the core equations and mechanism on printed pp. 5–7, plus the ablation and reconstruction figures on pp. 27 and 31. The parent separately read relevant extracted pages, inspected Figure 4, and checked the released implementation. Numerical figure readings below are explicitly approximate. Table values are reported values, not reproduced measurements.

- Paper: Hermann Kumbong et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*.
- Primary local source: [ncm.pdf](/Users/dnikolaidis/Desktop/ncm.pdf).
- PDF SHA-256: `708a029c7f53d4f9f04ca5c93865afc9b3b5414acab3d7d9b887f313d04a5b4d`.
- Public record: [arXiv:2610.14162](https://arxiv.org/abs/2610.14162).
- Released code examined by the parent: [HazyResearch/ncm, commit c70274273f489abc73832b2b98584a8483b6b203](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203).
- Related design: [Persistent-memory NCM head architecture draft](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-persistent-memory-head.draft.md).

Use section IDs S1–S9 as evidence pointers. The paper remains authoritative if this summary and the source disagree.

## S1. What NCM actually computes

**Source:** §3, pp. 5–7; Figure 4; Equations 3–8; Algorithms 1–2.

NCM is a standalone, two-stage text-generation framework:

1. Train a multiscale residual VQ-VAE to encode a complete fixed-length text block into discrete codes and reconstruct its tokens.
2. Freeze that tokenizer and train a shared transformer to generate the code hierarchy.
3. At inference, predict the levels sequentially; sample all code positions within each level in parallel.
4. Decode the final cumulative latent reconstruction into ordinary token distributions. Algorithm 2 selects output tokens by argmax.

For encoded text `z = E(x)`, a level downsamples the remaining residual, quantizes it, upsamples the selected embeddings, and adds that contribution to the reconstruction:

```text
z_hat[0] = 0
u[k]     = Down[k](z - z_hat[k-1])
r[k,i]   = nearest_code(u[k,i], codebook[k])
z_hat[k] = z_hat[k-1] + R[k](embedding[k](r[k]))
```

Every cumulative reconstruction has full output length, even when the code sequence at the current level is short. The next generator input is the cumulative reconstruction resized to the next level's length. It is not simply a concatenation of previous raw code IDs.

The distribution factorizes as:

```text
p(r[1:K]) = product_k product_i p(r[k,i] | z_hat[k-1])
```

Conditional independence is within a level. Earlier sampled codes provide shared conditioning for many later positions, allowing dependence in the marginal output distribution. The selected block-diagonal attention pattern communicates prior-level information through reconstructed inputs rather than unrestricted attention over all earlier levels.

The word “concept” denotes information carried at coarse scales. It does not establish a vocabulary of named semantic concepts or reasoning operations.

## S2. Training objectives and exposure mismatch

**Source:** §3.3–3.4, p. 7, Equations 9–10; Appendix A, pp. 21–22; Appendix F.3, p. 29.

Tokenizer training combines token-reconstruction cross-entropy, commitment MSE, and quantization MSE with straight-through gradients. Nested dropout retains only a sampled prefix of the hierarchy, encouraging early levels to contain useful information.

Generator training uses clean target codes and optionally corrupted conditioning codes. All levels are packed into one training forward pass with the appropriate masks. This does **not** mean one-pass inference. A level-weight parameter controls the balance between equal weight per code position and equal total weight per level.

The clean-conditioning, uniformly weighted generator objective corresponds to the generator-dependent term of the paper's ELBO. Corruption and nonuniform level weighting modify that objective. The full text likelihood marginalizes over latent hierarchies; it is not identical to the probability of the tokenizer's single assigned hierarchy.

Teacher-provided latent prefixes differ from generated latent prefixes. Uniform code replacement is a robustness mechanism; the paper identifies error-informed corruption as future work. For a speculative adapter, future-derived labels are permissible, but future target states must not become inference-time inputs.

## S3. Actual architecture and serial computation

**Source:** Tables 5–6, pp. 23–24; Algorithm 2, p. 7.

| Dataset | Output tokens | Generator calls / levels | Tokenizer decoder blocks | Generator blocks / parameters |
| --- | ---: | ---: | ---: | --- |
| OpenWebText | 1,024 | 32 | 1 | 12 / 94.65M |
| LM1B | 128 | 12 | 4 | 12 / 97.07M |
| TinyStories | 128 | 12 | 1 | 12 / 88.61M |
| XSum | 48 | 8 | 1 | 12 / 88.02M |
| WMT14 | 64 | 11 | 1 | 12 / 119.96M |
| Sudoku | 89 | 12 | 1 | 8 / 25.46M |

The listed tokenizer encoders have zero transformer blocks. Language-task tokenizer latent width is 384; Sudoku uses 256. The language generators use width 768; Sudoku uses 512. Tokenizers are additional substantial models: approximately 36–144M parameters for the language tasks, separate from generator parameter counts.

NFE counts generator calls, not all computation. The decoder, latent reconstruction operations, and conditioning work also matter. Early generator passes process fewer positions, but cumulative latent updates still represent the complete output block.

**Parent-agent code observation:** the released [OWT config](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/owt.yaml) has a 16-layer, width-1,024 generator and 8,192-entry codebooks, whereas the local paper tables specify 12 layers, width 768, and OWT codebooks of 4,096. These are different source specifications; do not silently combine their parameter counts or claim exact code-to-paper configuration parity. The released OWT hierarchy has lengths `1,4,9,...,1024`, i.e. 32 levels.

## S4. Few-level generation is not validated for drafting

**Source:** Appendix F.1, pp. 26–27, Figure 9; Appendix G, p. 31, Figure 16.

The paper provides no one-, two-, or three-level generation results. Its TinyStories level-count ablation tests 4, 8, 16, and 32 levels. Each tokenizer trains for 40k steps and each generator for 10k steps; conditioning corruption is disabled.

Approximate final Gen. PPL readings from Figure 9 are 300, 145, 100, and 80 respectively. All four full-hierarchy tokenizers reconstruct at least 99.95% of tokens. Therefore, the poor four-level generator cannot be explained merely by its tokenizer being unable to represent the answer: predicting a compact, information-dense hierarchy is difficult.

These limited ablations do not prove a short, target-conditioned hierarchy would fail. They do invalidate an assumption that the paper already demonstrated good one- or two-stage generation.

Truncating a pretrained deep hierarchy is also different from training a shallow one. Figure 16's OWT reconstruction is only a few percent at the earliest levels, roughly 35% around level 16 and 85% around level 24, reaching 99.99% at full depth. Intermediate values are approximate visual readings, not table entries.

## S5. Reconstruction accuracy and exact-token limits

**Source:** Table 1, p. 8; Table 11, p. 31.

| Dataset | Full-hierarchy exact token reconstruction |
| --- | ---: |
| OWT | 99.99% |
| LM1B | 99.86% |
| TinyStories | 99.98% |
| XSum | 99.79% |
| WMT14 | 98.79% |
| Sudoku | 100% |

These measure token recovery from codes extracted from known text. They are neither generated-token acceptance rates nor exact whole-block recovery rates. Errors include names, numerals, usernames, markup, diacritics, and word boundaries. Semantically similar reconstruction can therefore still fail speculative verification immediately.

Oracle-code reconstruction is a useful representation diagnostic, but not a universal mathematical upper bound on acceptance: another hierarchy might decode the text better, and token errors can be correlated. Do not raise mean token accuracy to a block-length power and present it as measured block accuracy.

## S6. Results and comparison boundaries

**Source:** Table 2–3, p. 10; §4.6, pp. 11–12; Table 8, p. 26; Appendix C, p. 24.

| Task | NCM result | Relevant comparator |
| --- | --- | --- |
| TinyStories | 12 NFE, Gen. PPL 15.41, entropy 3.98 | FMLM+: 128 NFE, 15.43, 3.90; AR sampling: 8.89, 4.01 |
| LM1B | 12 NFE, Gen. PPL 93.05, entropy 4.31 | DUO: 128 NFE, 97.30, 4.30 |
| OWT | 32 NFE, Gen. PPL 83.12, entropy 5.42 | DUO: 1,024 NFE, 77.69, 5.55; AR sampling: 35.45, 5.58 |
| XSum | 8 NFE, ROUGE-L 28.0 | Unguided AURORA-LM: 128 NFE, 26.6 |
| WMT14 | 11 NFE, BLEU 22.6 | E2D2: 64 NFE, 24.8; AR: 64 NFE, 25.2 |
| Sudoku hard | 12 NFE, 86.1% solved | FMLM+: 16 NFE, 81.3% solved |

NCM with minimum Bayes risk selection reaches WMT14 BLEU 25.2 using 12 candidates / 132 generator calls. MBR is candidate selection by mutual similarity, not target-model speculative verification.

Reported wall-clock speedups include 21× on LM1B, 10× on OWT, and 31× on XSum against selected diffusion comparators, measured on an H100 SXM 80GB with batch size 16. They cannot be interpreted as speedups over DFlash's one-pass draft or over DSpark serving.

Sudoku is useful evidence for global constraint coordination, but its 12-symbol vocabulary and exact tokenizer reconstruction differ greatly from large-vocabulary token drafting. The unconditional results also do not establish general quality superiority over autoregressive models.

The reported training budgets count text-position exposure, including tokenizer training; they are not FLOP-equivalent comparisons. LM1B is reported at 13.36B text positions and OWT at 127.22B. Hierarchy length, model size, and decoder work change actual compute.

## S7. Evidence for shared structure, and what it does not show

**Source:** Appendix H, pp. 32–34, Figures 17–19.

Fixing the first three TinyStories levels and resampling later levels tends to preserve broad story structure while changing names, objects, and wording. Replacing a few coarse codes can change events or characters across the output.

This supports the interpretation of shared coarse conditioning. It does not show that a coarse plan predicts the exact continuation of an external target model. A deterministic shared vector also does not, by itself, create stochastic dependence among independently sampled token outputs. A sampled latent can create marginal dependence, but can commit too early to one continuation and reduce target-distribution overlap.

## S8. Transfer hypotheses from the Sol analysis

These are the subagent's architectural inferences, not empirical findings of NCM:

- Condition a small shared plan on frozen target prefix features; let a direct token decoder use both the plan and the target context.
- Retain a lightweight sequential token head to preserve dependence on the actually sampled predecessor.
- Compare continuous planning vectors before paying the complexity of a learned discrete tokenizer.
- A shallow refinement stage or latent-assisted Markov head is a plausible alternative to a full residual VQ hierarchy.
- Optimize exact accepted prefixes and measured draft-plus-verify time, rather than semantic coherence or standalone generative perplexity.
- Rich target features may already contain the useful planning information; an added latent module could be redundant.

For greedy verification, proposals can be generated by any causal mechanism and checked against the target. For stochastic exactness, a sampled latent can be held fixed while using explicitly normalized token probabilities conditional on it. Exact rejection sampling can then use that conditional proposal; marginalizing over every latent is not inherently necessary. The probabilities used by verification must match the actual sampling process, including transformations. The original argmax output decoder is not automatically a softmax-sampled proposal.

Persistent temporal memory, four exponential timescales, rank 128, FP32 accumulators, and residual injection before DSpark's vocabulary head were developed in the parent conversation. **They are not architecture choices proposed or validated by the NCM paper or by this paper extraction.**

## S9. Limitations and design implications

**Source:** paper limitations, p. 35, and the evidence above.

The paper notes fixed hierarchy depth and lengths, separately trained tokenizer and generator, limited model scale, and occasional generation artifacts. Its main generators are about 100M parameters, with preliminary LM1B scaling up to 725M. It presents no target-conditioned speculative-head experiment, acceptance measurement, or DSpark/DFlash comparison.

The evidence supports studying shared coarse conditioning as an architectural hypothesis. It does not establish that persistent history compression, a single continuous plan, or a frozen DSpark residual adapter will improve acceptance. Those are precisely the distinctions the linked architecture draft preserves.
