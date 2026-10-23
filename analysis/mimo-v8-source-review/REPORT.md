# Additive public workflow revision v8

Authoritative workspace: divix01:/mnt/nvme4/mimo-reap/calibration-mix-v8.

All 621 v7 histories, original partition assignments and existing annotations are preserved exactly. Added 46 whole public histories, 23 per partition, with 1,385,814 exact MiMo tokens. Combined: 667 histories and 14,616,186 tokens. Primary: 331 histories / 7,260,460 tokens. Repeat: 336 histories / 7,355,726 tokens.

| Added category | Primary | Repeat |
|---|---:|---:|
| Parser/type/metadata behavior | 3 | 3 |
| Numerical arrays and scientific software | 3 | 3 |
| Memory/state handling | 3 | 3 |
| Simulation and measurement methods | 3 | 3 |
| Chemical/biological methods | 3 | 3 |
| Quantitative financial risk methods | 1 | 1 |
| Empirical financial/econometric data methods | 1 | 1 |
| Statistical-methods research | 3 | 3 |
| Typed API/SQL workflows | 3 | 3 |

Twenty-two additions exceed 32,768 tokens, eleven per partition. All episodes remain whole, with no truncation or padding. Pairs use the same tokenizer context bucket and at most a 1.5 length ratio. New repository tasks include DTD mixed content, CBOR null decoding, Altair type-context propagation, quantum-circuit integer overflow, FITS scaling metadata, annotation rounding tolerances, random-state handling, async-iterator semantics, buffer contents and connection-state recovery.

Research additions include robotics/sensor simulation, bioinformatics and electrochemical methods, regression-noise estimation, causal inference, Bayesian methods, backtest overfitting, earnings persistence/volatility, synthetic financial time series and empirical econometric modeling. The latter includes agricultural production-cost modeling; this is quantitative economics rather than a trading-agent capture. Finance admissions are intentionally smaller because many keyword matches were news, social-policy questions or otherwise out of scope. Additional financial-software discovery produced no admissible balanced cohort; its evidence remains reserved.

## Sources and controls

Pinned public sources: [NVIDIA Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3) and [DR-Tulu SFT data](https://huggingface.co/datasets/rl-research/dr-tulu-sft-data). Exact revisions, source URLs, raw hashes, row indices and license annotations are retained in sources.lock.json and each record.

Selection code reads source/content metadata, not expert scores. The investigator knew diagnostic categories, so this is not an independently blinded study. All 46 raw rows were independently matched and lossless histories checked. Every one of the 667 records was retokenized against the local MiMo tokenizer. Selection replay and combined-record reconstruction are byte-identical. Existing examples and annotations are unchanged, repositories remain separate across coding and tools, touched repositories obey two-episode/100000-token combined caps, and paper/lineage and held-out screens pass. These screens do not prove semantic or pretraining decontamination.

Seven adversarial admission checks cover valid additions, wrong partition ownership, repeated groups, preserved-source duplicates, repository token caps, opposite-partition papers and corrupted token streams. Manifests and source/tokenizer bindings are checked again in final-audit.json.

## Limits and next step

Public test outcomes, tool results and scientific claims remain upstream-attributed. No source code or tools were executed and no scientific correctness regrading was performed. Some source SWE captures finish through a native finish-tool payload rather than final text; these are preserved. Source-disclosed limitations and off-topic intermediate retrieval hits remain visible.

The annotation audit found that three inherited quota counts do not reproduce when counted over all existing annotated histories. Their precise policy scope needs separate reconciliation; coverage readiness remains false. Additional content does not establish ranking stability, native precision or task-quality acceptance.

No GPU scoring, pruning, NVFP4 export, model changes or active-pointer updates were performed. New data requires a fresh versioned run from stage zero, after capacity/disk preflight. Never attach v8 data to v11 activation checkpoints. Keep all quality and pruning gates unchanged.

Read RUNBOOK.md, verification.json, semantic-label-review.json, final-audit.json and release-seal.json for commands and evidence. Preparation code and archived attempts are in /mnt/nvme4/mimo-reap/public-workflow-gap-fill-v2.
