# Current v9 candidate: full history review next

Workspace: `/mnt/nvme4/mimo-reap` on divix01.

**Active candidate:** `public-workflow-gap-fill-v3/finance-balanced-draft-v1/`. It contains 26 histories, 13 per partition, with 410,452 primary tokens and 408,390 repeat tokens. Planned category counts are met: four finance-risk histories, four empirical-finance histories, and six each for persistence/indexing, typed APIs and statistical evidence. This is **not a validated release**. Full content review, new release-path validation, final manifest audit/sealing and a separately reviewed incremental runner remain outstanding. No scoring or pruning was launched.

## What changed, and why

The source-only `finance-union-v1/` combined 133 original screened candidates and seven newly screened finance candidates, deduplicating one identical source history. It preserves all 20 original proposed histories and produces a 24-history partial proposal. Four identity tests and raw-source/exact-token replay for all 24 passed. One finance-risk pair remained unfilled.

The remaining gap was partly an allocation constraint: two suitable, previously unselected finance research histories had both received primary assignments from default hash parity. Neither research-question group is owned by a frozen v8 history. A **new, separately versioned allocation policy** considers both orientations for new, unselected finance research pairs, subject to unchanged context buckets, <=1.5 length ratio, all parent ownership, repository caps and complete paper/dataset lineage separation. It minimizes assignment changes, then breaks ties using a fixed source-hash seed. No expert scores are read. All 24 already proposed histories and their partitions are locked byte-for-object unchanged.

The resulting pair is:

- Primary: underwriting/credit-risk methods research, 20,274 tokens; its discovery partition remains primary.
- Repeat: solver/constraint-programming portfolio optimization research, 16,587 tokens; this previously unowned and unscored group changes from its discovery hash assignment to repeat.

The reverse orientation fails paper-lineage separation because the underwriting history shares a paper with an existing primary draft example. That failure was retained as a constraint, rather than bypassed. The chosen orientation passes the unchanged native repository/lineage checks. `allocation-ledger.json` records both source hashes, original/allocated partitions, parent ownership absence and the locked 24-history hash.

This changes an unscored candidate-allocation rule in a new draft. It does not move any of the 667 frozen v8 histories, alter failed scores/checkpoints, change domain weights or calibration thresholds, or change data under an activation checkpoint. All previous candidate directories and source files remain intact.

## Verification already completed

- Thirteen unit tests passed: duplicate/raw-source/token/partition identity guards; frozen ownership protection; lineage rejection; unchanged context bucket/length bounds; duplicate source rejection; enumeration-order independence; locked history preservation; rejection of moving locked histories to repair imbalance.
- Deterministic allocation replay using reversed candidate enumeration is identical.
- The original independent additions checker accepts the completed 26-history candidate with its explicit ownership ledger.
- All 26 selected histories reconstruct from pinned source rows without native history loss and retokenize exactly with the MiMo tokenizer. All 24 prior draft histories and partitions are preserved; the original 20 are also preserved.
- `partial-validation.json`, `locked-prefix-verification.json`, `test-evidence.json` and `admission-tests.log` record these checks. They explicitly retain `full_content_review=false`, `release_ready=false`, `scoring_launched=false`, `native_precision_verified=false` and `selection_allowed=false`.
- Full native message copies and source provenance for every selected history are in `review-materials/`; `index.json` binds their file hashes to the proposal and marks every content review pending. These are review copies, not new generated observations or admissions.

Commands already run (one-shot output paths; do not overwrite):

```sh
cd /mnt/nvme4/mimo-reap
.venv-compressor/bin/python -m unittest discover -s public-workflow-gap-fill-v3 -p 'test_*finance*_v9.py'
.venv-compressor/bin/python public-workflow-gap-fill-v3/finance_union_v9.py
.venv-compressor/bin/python public-workflow-gap-fill-v3/verify_finance_union_v9.py
.venv-compressor/bin/python public-workflow-gap-fill-v3/balanced_finance_draft_v9.py
.venv-compressor/bin/python public-workflow-gap-fill-v3/prepare_balanced_review_v9.py
```

For balanced-draft source/token verification, the same versioned verification module was imported and its `OUT` set to `finance-balanced-draft-v1` before `main()`. It ran the unchanged source/native-token/additions checks; its log is in that directory. The separate locked-prefix check compares the first 24 records to the union proposal exactly. No frozen helper or production source was edited.

## Additional discovery receipts

- `financial-hedge-extension-v1/`: a real public training FX-futures hedge-sizing history passed native screening at 9,531 tokens, assigned repeat. It is **outside the selected 26**, remains pending full content review and was not used to fill counts. The credit-spread history is 5,432 tokens and fails the unchanged 8,192 minimum.
- The first throwaway native-template length probe counted `BatchEncoding` keys. It is preserved as `financial-mechanics-length-check.invalid-return-shape.json`; the corrected probe explicitly uses `return_dict=False` and validates flat integer IDs. Existing normalization and independent replay use rendered text plus encoding and were unaffected.
- `open-swe-traces/finance-scan-v2/`: two pinned SWE-rebench train shards (5,000 histories, about 1.36 GB) produced two TradingView hits. Actual tasks are callback unsubscription and transparent chart colors, rather than quantitative-finance methods. One is resolved and one unresolved; both are reserved from this finance cohort. Repository names alone do not qualify content.
- `openmle/assessment.json`: all 26,259 released rows have three messages and no observed tool-role calls/results or embedded action/result markers. They are code-proposal SFT examples, not the required action/result workflows. No row was admitted.
- Live FinTrace revision `59bd0b566099a30239ca8494680ecf5bf42a930b` contains only `evaluation/testset.json` and documentation; training trajectories remain unavailable. Do not substitute its evaluation traces.
- Related DR-Tulu discovery metadata is retained. Browsecomp synthetic trajectories, generic personal-finance data and undocumented derivative corpora were not admitted to satisfy the quantitative-method targets.

## Next steps

1. Read `review-materials/index.json` and complete source/action/result/final content decisions for **all 26 selected histories**. Raw-source, structural and token replay do not replace semantic review. Use the complete native `messages.json` files; SWE final summaries may be in native `finish` arguments, and DR-Tulu action/results are embedded XML. Preserve and explain any reservations rather than silently repairing histories.
2. Implement a new versioned release path for this candidate, including independent replay of the ownership allocation and adversarial admission checks. **Do not call the old v3 `release.py build` for this draft:** it still targets the preserved 20-history proposal, and its original exact-parent-owners assertion intentionally does not describe the new allocation ledger. Do not edit frozen production-v1 scoring code.
3. Recheck all 667 frozen histories and annotations, held-out overlap, source/runtime closure pins, source/selection/reconstruction replay, exact tokenization, ownership/caps, then assemble and independently audit/seal the new revision. Until that succeeds, no `calibration-mix-v9` release is ready.
4. Only then prepare a separately reviewed versioned incremental runner and fresh stage-zero scoring for changed histories, using compatible immutable parent score evidence through the verified merge protocol. The failed v1 stage-67 checkpoints and sealed prefix scores remain preserved and must not be resumed as that failed run.

The GPU was not used or claimed. Disk had about 152 GiB free after discovery; no retired activation cleanup was needed. Continue to honor user holds and unrelated jobs. Native-precision, objective-baseline, coverage, ranking and pruning/export gates remain binding.
