# Production incremental calibration

Workspace on divix01: `/mnt/nvme4/mimo-reap`.

Implementation: `incremental-production-v1/production_incremental.py`.

Prepared run: `incremental-calibration-v1`.

The frozen parents are `full-scoring-v11` and `calibration-mix-v8`. This run scores only the 46 additions (23 per partition) and combines them with the 621 unchanged histories. It starts additions at stage zero; no changed data is loaded under v11 activation checkpoints. Existing examples, ownership, source provenance, domain weights, model/compute identity and gate thresholds are preserved.

## Start

```sh
cd /mnt/nvme4/mimo-reap
.venv-compressor/bin/python incremental-production-v1/production_incremental.py preflight
.venv-compressor/bin/python incremental-production-v1/production_incremental.py start
```

`start` detaches the controller and prints its PID. `run` runs it in the foreground instead. Neither command stops unrelated GPU processes. The controller and observer use the shared GPU lock, the existing GPU release guard and versioned checkpoint identities. Scratch stays under the run directory. No full calibration was launched while implementing this path.

## Progress, pause and resume

```sh
.venv-compressor/bin/python incremental-production-v1/production_incremental.py status
tail -f incremental-calibration-v1/controller.log
```

Partition logs are `incremental-calibration-v1/primary.log` and `repeat.log`. Status includes each observer's stage progress, committed checkpoint state and its measured stage-average ETA where available; this is not a guaranteed whole-run ETA because attention types and history lengths differ.

```sh
.venv-compressor/bin/python incremental-production-v1/production_incremental.py pause
.venv-compressor/bin/python incremental-production-v1/production_incremental.py status
.venv-compressor/bin/python incremental-production-v1/production_incremental.py resume
```

Pause writes only this run's request file. The active observer finishes its stage, commits a recovery checkpoint and exits, releasing the GPU; the command does not promise immediate release. Confirm `paused` and that the observer has exited before using the GPU for another job. Resume verifies frozen inputs and the existing GPU release guard, clears this run's pause request explicitly, and uses the original observer's verified checkpoint restore. It does not override a global/user GPU hold. For a controller failure, inspect the logs and `state.json` before resuming; committed inputs cannot change.

## Automatic sequence

1. Check frozen release seals, bounded-verification receipts, v11 evidence, exact whole-history token identities, partition assignment, domain weights, available RAM/GPU and disk reserve.
2. Run each additions partition through model layer 32 under the existing v9 observer, stopping at stage 67 with all 70 model layers represented in the execution graph.
3. Validate committed state and trajectory hashes, raw domain reaggregation and exact sample/token coverage. Add raw score sums, route counts and valid tokens to v11 evidence within each partition. Never average normalized run scores or combine partitions.
4. Atomically seal `merges/prefix-32`. Run the unchanged `compare_prefix` implementation: top-344 sets, Jaccard >= 0.95, all 128 checks. On failure, stop and preserve the scores and examples.
5. On pass, automatically resume both additions partitions through all 70 layers, ending at stage 141. Atomically seal `merges/full-69` and run the unchanged `rank_stability.compare`: all 276 checks over 69 MoE layers and four domains.
6. Stop on a failed full-model gate. If the rank gate passes, report calibration complete with native-precision, objective-baseline, coverage and pruning/export gates still required. A calibration pass alone never promotes those flags or changes model weights.

Merged artifacts contain the actual raw statistics, original per-history records and full lineage receipts, separate from activation checkpoint directories. Parent manifest and observer-state snapshots remain sealed inside each merge, so checkpoint rotation cannot erase an earlier gate's proof. Interrupted uncommitted merge directories are archived under `merges/uncommitted` before a fresh merge; existing sealed results are reverified and reused.

## Implementation and validation

```sh
.venv-compressor/bin/python -m py_compile incremental-production-v1/production_incremental.py
.venv-compressor/bin/python -m pytest incremental-production-v1/test_production_incremental.py \
  --basetemp=/mnt/nvme1/pytest_scratch/incremental-production-recheck -q
```

The test suite covers complete synthetic 70-layer protocol fixtures, early failure preventing continuation, early pass continuing both partitions, user pause, checkpoint retirement, interrupted atomic merges, overlapping histories, malformed/negative/nonfinite scores, fractional/missing route counts, wrong token/partition metadata, corrupt states/trajectories, missing layers, changed precision/runtime/weight identity, changed execution graphs, forged gate flags and report mismatches. Synthetic fixtures are protocol tests, not scored model evidence. The reducer also reproduces the real bounded GPU split/join results from `incremental-scoring-v1`.

Preparation separately validates both real v11 partitions against their committed full-model observer state and trajectory records. The earlier fresh GPU parity experiment covered only the first two model layers. This implementation does not turn that bounded experiment into a fresh 70-layer numerical parity claim. Existing source-weight shard identity uses size/mtime plus pinned model/index/runtime files, rather than a new checksum of the entire multi-terabyte checkpoint.

## Files to retain

- `incremental-production-v1/`: source, tests, implementation receipts and this runbook.
- `incremental-calibration-v1/inputs.lock.json`, `preparation.json`: frozen input/runtime/parent pins and admission receipt.
- `incremental-calibration-v1/state.json`, `events.jsonl`, `launch.json`, controller/partition logs.
- Both active additions recovery checkpoints and observer statistics/trajectories.
- `merges/prefix-32` and `merges/full-69`: sealed scores, trajectory records, lineage, parent-statistics snapshots and gate results.

The disk budget includes two recovery generations per partition, one in-flight generation, a serialization margin and overhead. No old checkpoints were deleted during implementation. To change workflows, runtime or policy later, prepare a new versioned run directory; do not alter this run's frozen files.
