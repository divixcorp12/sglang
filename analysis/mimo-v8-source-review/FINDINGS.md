# Full-model v11 ranking diagnosis

The CPU audit reproduced all 276 comparisons from committed per-trajectory scores, checked 419 input files and verified 1,417 leave-one-history counterfactuals using independent scalar summation and stable sorting. Both partitions completed all 70 model layers. The paired layer-32 gate passed, but eight full-model comparisons fell below the unchanged 0.95 top-344 Jaccard threshold.

| Domain | Failed layers | Jaccard |
|---|---|---|
| Tool use | 33 | 0.9490085 |
| Coding | 56, 68 | 0.9490085 |
| Coding | 57, 60 | 0.9435028 |
| Finance | 59 | 0.9490085 |
| Science | 64, 68 | 0.9490085 |

A 0.9490085 result means 335 common retained experts and nine exclusive to each partition. A 0.9435028 result means 334 common retained experts and ten exclusive to each partition. These compare different input partitions; they are not an identical-input numerical-equivalence check.

## Main evidence

The aggregate misses coexist with large individual rank reversals and task concentration. They cannot be dismissed as uniformly tiny differences at the cutoff.

- Coding layer 56, expert 278: primary rank 203 versus repeat 374. A SQLGlot subquery/NULL-parenthesization workflow contributes about 37.8% of its primary score.
- Coding layer 57, expert 112: primary rank 370 versus repeat 226. A PennyLane numerical/quantum workflow contributes about 84.0% of its repeat score.
- Coding layer 57, expert 2: primary rank 375 versus repeat 252. A C++ emulator memory-bookmark/state-validation workflow contributes about 94.5% of its repeat score.
- Coding layer 60, expert 240: primary rank 351 versus repeat 159. An SWC constant/bitwise-semantics workflow contributes about 59.9% of its repeat score.
- Coding layer 68, expert 38: primary rank 368 versus repeat 291. A media metadata/list-representation workflow contributes about 64.4% of its repeat score.
- Finance layer 59, expert 226: primary rank 352 versus repeat 221. A credit-data/SHAP notebook contributes about 45.2% of its repeat score; a max-statistic bootstrap workflow contributes about 29.1% of its primary score.
- Science layer 64, expert 375: primary rank 132 versus repeat 362. A robotics/physics-engine literature workflow contributes about 76.8% of its primary score.
- Science layer 68, expert 230: primary rank 373 versus repeat 283. A VR-training literature workflow contributes about 72.3% of its repeat score.
- Tool layer 33, expert 149: primary rank 249 versus repeat 347. A bounded-statistics/sample-complexity research workflow contributes about 48.5% of its primary score.

The complete report records every disputed expert and its leading histories in both partitions. These are fixed-score attributions, not proof of incorrect examples, task-quality loss or a reason to delete a workflow. Whole-history statistics cannot identify individual responsible tool calls or token regions. Counterfactual omissions were diagnostic only; no example was removed or reweighted.

## Additive response

Broaden independent parser/type/codec behavior, numerical arrays and scientific software, memory/state handling, scientific simulation/measurement and chemical/bioinformatics methods, quantitative finance, statistical inference and typed API/SQL workflows. Select by actual workload content and pinned source metadata, without reading expert scores in selection code. Preserve all 621 v7 histories and all original partition/annotation assignments. Repositories must stay separate across coding and tool domains, with held-out and research-paper lineage checks.

The resulting candidate is calibration-mix-v8. It adds 46 whole public histories; the old v7 data, v11 run, scores, checkpoints and thresholds remain unchanged. Fresh scoring is still required, and native-precision, objective-quality, coverage and pruning/export gates remain binding.

## Repeat

On divix01, from /mnt/nvme4/mimo-reap:

    .venv-compressor/bin/python audit_v11_diagnosis.py

The producer refuses an existing rank-diagnosis-v7 directory. For an independent replay, use a new output directory without replacing this evidence. Read report.json, verification.json and inputs.sha256.json in rank-diagnosis-v7. No GPU or source-task execution is involved.
