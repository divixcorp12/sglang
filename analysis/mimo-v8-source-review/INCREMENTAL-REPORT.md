# Incremental REAP scoring: bounded verification passed

Authoritative experiment: `/mnt/nvme4/mimo-reap/incremental-scoring-v1` on `divix01`.

The evidence supports scoring only the 46 newly added workflows and combining additive raw statistics with the immutable v11 results for the 621 unchanged workflows. This verification does not implement or certify a production evidence merger, complete v8 calibration, or authorize pruning/export.

## Existing evidence reconstruction

All 552 layer/domain aggregates (69 MoE layers × four domains × two partitions) reconstructed from the saved per-history records. Route counts and valid-token counts matched exactly. All 552 complete expert rankings matched; all 276 paired Jaccard gate values reproduced exactly, including the eight existing failures. Another 1,656 split/recombine checks retained identical full expert orders. The maximum relative difference from saved aggregate scores was `1.5605306288162735e-16`.

## Fresh joined versus independent GPU execution

Six real sequential LLM Compressor executions used the promoted v9 runtime with `sglang_sm120` precision: joined, old-only and new-only plans for each partition. Sixteen distinct whole histories covered all four domains in both partitions, including four coding histories of 32,780–36,401 tokens. Selection used domain and length, not expert scores. New histories came before old histories in the joined plans to exercise changed accumulator history.

For every history, raw token routing indices, gate weights and expert-block outputs matched bitwise between joined and separate execution. All final cached tensors matched exactly. All eight domain expert rankings matched after merging raw score/count statistics. Split-versus-joined score differences reached at most `5.5448956612566635e-14` relative, within the producer's existing FP64-delta tolerance (`rel_tol=1e-12`, `abs_tol=1e-8`).

The eight old histories were also compared with saved v11 layer-1 records. Route counts matched exactly; score differences reached at most `1.1013412404281553e-12` absolute and `1.497121828539382e-12` relative. The relative maximum occurred at small score magnitudes and satisfies the unchanged absolute tolerance. V11 did not save token-level routing hashes, so that historical comparison proves route-count equality, rather than individual token-route equality.

## Preservation and limits

Protected v7/v8 dataset files, v11 statistics, trajectory files, checkpoint pointers/manifests and gate artifacts matched their pre-test hashes. All 317 pinned source/input files and source-weight shard size/mtime identities were checked. No source weights, thresholds, scoring implementation or frozen run were edited. All scratch files are in the workspace. GPU jobs were not stopped; the bounded test is finished.

Fresh GPU parity covers embedding, dense layer 0 and MoE layer 1, including full and sliding attention. The all-layer part is a CPU reconstruction audit plus inspection of the independent, observation-only stage path; it is not a fresh 70-layer GPU replay. Weight checks reuse the existing size/mtime shard identity rather than computing a new multi-terabyte weight checksum. Native-precision, coverage, baseline-quality and selection/export gates remain unchanged and unverified where previously unverified.

## Next implementation

Create an additions-only run from stage zero with its own checkpoint identity. Keep all 46 additions in their frozen partitions and score them through every layer. Build a separately versioned combined-evidence artifact that verifies immutable parent scores, identical model/runtime identities, exact old-history/token preservation, complete disjoint additions and unchanged domain weights. Sum raw saliency sums, route counts and valid tokens before applying existing normalization. Recompute the paired layer-32 and final full-model gates. Never place changed data under an old activation checkpoint or average already normalized run scores.

The additions represent about 9.5% of combined tokens, so old-history computation can be avoided. Fixed model loading remains; these correctness checks do not establish a full-run speedup or ETA.

See `RUNBOOK.md`, `cpu-audit.json`, `gpu-parity.json`, `commands.json`, `status.json`, and `protected-before.json` / `protected-after.json` for reproducible evidence.
