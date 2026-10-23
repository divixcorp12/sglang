# Incremental-v1 gate failure and additive v9 follow-up

Authoritative workspace: `/mnt/nvme4/mimo-reap` on divix01.

The 2026-10-21 paired layer-32 gate failed 2/128 checks. Finance layer 22 is 0.9490084985835694; general tool use layer 32 is 0.943502824858757. Threshold 0.95, keep 344 and every existing example remain unchanged. The previous v11 values for those checks were 0.9601139601139601 and 0.9545454545454546. The controller stopped normally; both additions partitions retain committed stage-67 checkpoints. No later scoring or pruning was launched.

## Verified diagnosis

`rank-diagnosis-v8-prefix/diagnose.py` verifies the production input lock and sealed merged snapshot, reconstructs failed expert-set comparisons, and attributes all disputed experts to complete saved histories. It writes `report.json` and `diagnosis.log`. Question previews were corrected to use the task's first user message rather than its system prompt; this did not change scores or ranks.

- Tool expert 373 ranks 64 in primary and 368 in repeat. The new `ethereum-optimism/optimism` indexer-start-height/database workflow supplies 78.5% of its primary score. It is retained.
- Tool expert 270 ranks 352 versus 148; an existing repeat workflow supplies 67.7% of its repeat score. Other disagreements involve statistical evidence/retrieval histories. This is broader than a single new history.
- Finance expert 305 ranks 349 versus 270. The two newly added financial/econometric research histories account for 45.2% and 48.4% of its partition scores. Existing quantitative implementation captures also remain concentrated: expert 63 draws 33.3% from the primary max-stat bootstrap task and 22.8% from primary shrinkage risk, with different dominant repeat workflows.

These are score attributions, not proof that a history is incorrect. No history was removed, reweighted or selected using expert scores. The merge's unchanged input identities, raw statistics and sealed lineage remain valid.

## Public candidate revision: in progress, not released

`public-workflow-gap-fill-v3/` is a separate draft rooted in all 667 immutable v8 histories. It does not edit frozen scoring code or old checkpoint identities. Current source inventory uses pinned NVIDIA Nemotron SFT-SWE train data and DR-Tulu SFT train data. New task families cover persistence/indexing, typed API integration, statistical-method evidence and quantitative finance methods.

Current preparation:

- 284 source-only candidates were normalized using existing whole-history validators and exact model tokenizer.
- 133 survived structural, duplicate, held-out, lineage and context checks.
- `scope-admission.json` records task-mechanism judgments; `scope-review.json` binds them to the screened file.
- The deterministic draft proposal contains 20 histories: nine tool-workflow pairs and one empirical-finance pair. Length buckets, <=1.5 ratio, group/repository caps and ownership checks remain binding.
- **The proposal is incomplete.** `proposal.json` records unfilled finance-risk pairs and a second empirical-finance pair. `release.py build` refuses an incomplete proposal. No `calibration-mix-v9` release or scoring run exists.
- Sixteen additional pinned public SWE training shards (24..39, about 1.8 GiB) were downloaded under `financial-extension/sources`. Goal-pattern discovery found one candidate, a factory/Faker issue with incidental portfolio-example text. The held-out-visible-span check rejected it. It was not admitted.
- Finance benchmark-derived sources and code-only reasoning corpora were reserved; no benchmark traces were added to calibration merely to meet counts.

The extension's corpus-level finance discovery may be broadened by explicit repository identity in a **new** inventory, since some task titles omit the financial repository name. Preserve the existing inventory and rejection receipts. Review actual task mechanisms; a repository name alone is not admission evidence. Also inspect other public **training** trajectory sources with explicit split/lineage provenance. Do not use evaluation-only finance traces or fabricate tool observations.

## Commands already run

```sh
cd /mnt/nvme4/mimo-reap
.venv-compressor/bin/python rank-diagnosis-v8-prefix/diagnose.py
.venv-compressor/bin/python public-workflow-gap-fill-v3/prepare.py inventory
.venv-compressor/bin/python public-workflow-gap-fill-v3/prepare.py normalize
.venv-compressor/bin/python public-workflow-gap-fill-v3/prepare.py screen
.venv-compressor/bin/python public-workflow-gap-fill-v3/finance_extension.py discover
.venv-compressor/bin/python public-workflow-gap-fill-v3/finance_extension.py normalize
.venv-compressor/bin/python public-workflow-gap-fill-v3/finance_extension.py screen
.venv-compressor/bin/python public-workflow-gap-fill-v3/release.py propose
```

All commands write versioned workspace outputs; pytest scratch, when needed, remains `/mnt/nvme1/pytest_scratch`. Scripts are one-shot: do not blindly rerun stages over existing files. Copy them to a new inventory/output version for experiments. An initial wrapper call hit the inherited proposal function's old default output path; its no-overwrite guard rejected the write. The new wrapper now passes the v3 destination explicitly. No frozen v2 file was overwritten.

## Remaining work before readiness

1. Fill the finance candidate supply gap from appropriate public training workflows, preserving existing examples and all cross-domain repository/research ownership. Reserve inadequate or held-out-overlapping material.
2. Inspect full selected histories for action/result/final coherence and record source-bound content decisions. The existing scope review alone is not full content approval.
3. Add meaningful adversarial admission tests for new adapters/policies, then run build, exact-tokenization/source replay, complete parent-preservation and overlap checks, deterministic selection/reconstruction replay, manifest assembly and an independent final-artifact audit. The new `release.py` draft has compiled and its proposal path executed, but build/verify/assemble are not yet validated or run.
4. Seal a genuinely validated revision before preparing a new versioned incremental calibration. Do not resume changed data under `incremental-calibration-v1` checkpoints. The current production-v1 runner intentionally targets frozen v8 and must not be edited in place; a v9 path needs a separately reviewed version.
5. Keep the monitor active for the follow-up, quiet on ordinary progress and explicit about dataset readiness, failures and stalls. Do not resume the failed calibration or bypass native precision, objective baseline, coverage, ranking or pruning/export gates.
