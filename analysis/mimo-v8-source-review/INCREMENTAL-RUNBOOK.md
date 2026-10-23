# Incremental calibration verification

Authoritative workspace: `/mnt/nvme4/mimo-reap/incremental-scoring-v1` on `divix01`.

This is verification only. It does not resume v11, run full v8 calibration, prune, export, or promote native-precision/coverage/objective-quality flags.

## Reproduce in a new versioned directory

The script uses a fixed `HERE` output directory and refuses to overwrite prepared inputs or GPU runs. For a new experiment, copy it to a new version and explicitly change `HERE`; preserve this version and receipts. Do not remove checkpoints to make a test rerun appear fresh.

```sh
cd /mnt/nvme4/mimo-reap
.venv-compressor/bin/python -m py_compile incremental-scoring-v1/verify_incremental.py
.venv-compressor/bin/python incremental-scoring-v1/verify_incremental.py prepare
.venv-compressor/bin/python incremental-scoring-v1/verify_incremental.py cpu
.venv-compressor/bin/python -u incremental-scoring-v1/verify_incremental.py gpu
```

Use the literal workspace interpreter. All scratch/output paths are under the workspace. The GPU test requires the existing GPU release guard, no unrelated compute jobs, sufficient memory, and the shared benchmark lock; it never stops services.

## Evidence and invariants

- `prepared.json`: 317 frozen input/source files checked plus source-weight shard size/mtime identity; whole-history selection based only on domain and length, not scores.
- `input-pins.json`, `protected-before.json`, `protected-after.json`: frozen v7/v8 data and v11 score/checkpoint metadata receipts. Original checkpoint activation payloads are not modified or loaded for the early replay.
- `cpu-audit.json`: all 69 MoE layers × four domains × two partitions reaggregated from saved per-history records; routing counts and valid tokens exact; all expert orders, top-344 sets, and all 276 paired gate values checked. Existing eight failures must remain failures in this audit.
- `inputs/{primary,repeat}/{joined,old,new}`: four old and four new whole histories per partition. Joined execution puts the new history before the old one in each domain to change accumulator history. All histories remain in their assigned partition. Coding histories exceed 32K tokens.
- `commands.json`, `status.json`, per-run logs: six actual `llmcompressor.oneshot` sequential executions under exactly the promoted v9 install contexts, using `sglang_sm120` precision, no pruning, no error propagation, no shuffled histories, batch size one. Additional hooks only hash selected token routes, gate weights, and expert-block outputs.
- `runs/*/*/raw-hook-hashes.json`: raw tensor shape/dtype/SHA256 for token-level routes, gates, expert outputs.
- `gpu-parity.json`: joined-versus-independent comparisons, old-versus-v11 layer-1 score/count comparisons, exact final cached activations and full expert rankings. FP64 accumulator deltas are compared at the existing producer's `rel_tol=1e-12, abs_tol=1e-8`; routes/gates/expert outputs/final activations require byte-exact hashes or tensor equality.

## Why additive reuse is plausible

The original observation subclass suppresses pruning and config updates. The sequential pass sets `propagate_error=False`; its loader yields one entire history at a time and does not shuffle. Each stage transforms only that history's hidden states. Domain accumulators collect gate-weighted expert-output norm sums and route counts without feeding statistics back into model execution. This permits additive sufficient-statistic merging when source weights, model/token identity and compute profile remain unchanged.

The per-history files contain FP64 differences of domain accumulators, so reaggregation is not promised bitwise-identical floating-point addition. Exact routes/counts and unchanged rankings are separate requirements; numerical differences must satisfy the producer's unchanged roundoff bounds.

## Limits and intended next implementation

The fresh GPU experiment covers embedding, dense layer 0 and MoE layer 1, including full and sliding attention. It is not a fresh 70-layer equivalence run. The CPU audit covers every previously captured MoE layer. Weight integrity reuses the existing shard size/mtime identity, not a new multi-terabyte weight checksum.

If verification passes, implement a separate additions-only scoring run starting at stage zero, with separate checkpoint identities. A new merged evidence artifact must pin both parent runs, require disjoint sample IDs, exact coverage/token identity and identical compute/weight identities, preserve partition ownership and domain weights, sum raw saliency/count/token statistics before applying the existing normalization, and rerun all unchanged gates. It must reject missing/duplicate histories, incomplete layers, changed identities, negative/nonfinite scores, and forged completion/native/selection flags. Do not average already normalized per-run scores, mix partitions, or transplant additions into old activation checkpoints. A bounded GPU result alone does not certify the unimplemented production merger.
