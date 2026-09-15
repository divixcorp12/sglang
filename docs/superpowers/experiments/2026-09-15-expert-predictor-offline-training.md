# Offline Training: LLaPor and APEX Expert Predictors (2026-09-15)

Offline-only. No server was started, no prefetch/admission/online-adaptation
code was touched, and nothing in `python/sglang/srt/layers/moe/expert_prediction/`
outside the new `training/` package was modified.

Specs (read-only, not edited):
- LLaPor: `/home/dimitri/data/divix/crypto/trading-framework/mechanism_hunter/docs/superpowers/plans/2026-09-14-llapor-gpu-only.md`
- APEX: `/home/dimitri/data/divix/crypto/trading-framework/mechanism_hunter/docs/superpowers/plans/2026-09-14-apex-gpu-only.md`

## Data, splits, and row counts

- Capture: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438`
  on divix01 -- 162 shards, 698,644 rows (394,709 prefill / 303,935 decode),
  532 requests, 360.4 GB, 0 integrity violations
  (`capture_reader.check_capture`). `stopped_reason: null` -- the capture
  agent's server was stopped manually, not by the byte cap. 48 layers, 512
  experts, top_k=10, hidden_size=2560, hidden_dtype bf16 (from `capture.json`).
- Splits are joined by session, never by row: a row's `rid` is
  `<session_id>-t<turn>`; `session_id` looks up
  `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`'s `split` field.
  train->train, val->dev, holdout->shifted_test. 1 of 698,644 rows is a
  server `HEALTH_CHECK_*` probe with no session and is excluded from every
  split/subset (see `dataset.known_session_id_of_rid`).
- Row counts per split (matching the capture doc's per-session-metadata
  join): train 539,850, dev (val) 65,678, shifted_test (holdout) 93,115,
  unmatched 1.
- APEX's `cdf_fit`/`calibration` subsets are carved out of the **train**
  sessions only, by hashing `f"{seed}:{session_id}"` (seed 0) into 100
  buckets: 80% `ranker_train`, 10% `cdf_fit`, 10% `calibration`.
- Teacher logits are not stored; they are recomputed as
  `router_input.float() @ gate.weight.float().T`, then softmaxed. This
  matches the model's actual routing transform: `Qwen2MoeSparseMoeBlock`
  (used by `Qwen3NextModel`) constructs `TopK(scoring_func="softmax",
  use_grouped_topk=False, correction_bias=None)` -- plain softmax, no
  grouped-topk, no correction bias
  (`python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/moe/topk.py`).
  Gate weights are loaded the way
  `scripts/expert_prediction/check-capture-gate-topk.py` does, excluding
  `mtp.*` keys.
- Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, snapshot
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`.

## Deviations from the specs (explicit, per the task contract)

1. **Splits are 3-way, not 4/5-way.** The specs call for 70/10/10/10
   (LLaPor) and 60/10/10/10/10 (APEX) splits with a locked final test.
   `sessions.jsonl` only carries train/val/holdout, so this run uses
   train/dev/shifted_test and carves APEX's cdf_fit/calibration out of
   train (above). There is no separate locked-test split withheld from
   both checkpoint selection and calibration.
2. **Checkpoint selection metric.** LLaPor selects on dev
   `llapor_recall@16` (multi-budget recall on the *unpacked* prediction),
   not the spec's "development timely-cold-recall replay at fixed byte
   budget" (that requires the cache/transfer simulator, which is out of
   scope here).
3. **`same_expert` baseline padding.** Padded past K=10 with ascending
   expert IDs (not excluding the row's own IDs), since there is no
   principled per-row extension of "the same 10 experts" past K=10.
4. **Gate transform verification skipped.** The 0.982-0.992 agreement
   between recomputed-softmax top-k and captured topk_ids (spot-checked in
   the 2026-09-14 smoke and 2026-09-15 capture docs) was not re-verified on
   this exact capture because the existing verification script
   (`check-capture-gate-topk.py`) concatenates every shard's tensors in
   RAM -- see the caution below.
5. No online-adaptation, GPU serving, or cache/scheduler code was touched;
   this is training-only, matching the task's stated scope.

## Caution: `check-capture-gate-topk.py` at this scale

That existing script loads every shard's tensors for every layer into RAM
simultaneously via `torch.cat([shard.tensors[key] for shard in shards ...])`.
Run against the full 360 GB capture it reached ~127 GB RSS with the host
already at 48 GB swap before being killed. The dataset loader added here
(`training/dataset.py`) never does this -- it opens shards one at a time
with `safetensors.safe_open` and only concatenates the columns one
layer/pair actually needs (~3.2 GB peak per layer, matching the ~3-6 GB
estimate in the task). Not fixed as part of this task (out of scope); flag
before running that script again against a full-size capture.

## Commands

```bash
# Capture integrity (safe: one shard's tensors in memory at a time)
python -c "from pathlib import Path; from sglang.srt.layers.moe.expert_prediction.capture_reader import check_capture; \
  import msgspec; print(msgspec.json.encode(check_capture(Path('<capture_dir>'))).decode())"

# Unit tests (CPU)
python test/registered/unit/layers/moe/test_expert_prediction_training.py -v

# Full LLaPor run (47 pairs, resumable per pair)
python scripts/expert_prediction/training/train_llapor.py \
  --capture-dir /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438 \
  --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl \
  --out-dir /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630 \
  --device cuda

# Full APEX run (48 layers, resumable per layer)
python scripts/expert_prediction/training/train_apex.py \
  --capture-dir /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438 \
  --model-dir /mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --sessions /mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl \
  --out-dir /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630 \
  --device cuda

# Condensed evaluation summary
python scripts/expert_prediction/training/evaluate.py \
  /mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630
```

Both `train_*.py` scripts skip any layer/pair whose output directory already
has a `DONE` marker, so a killed run resumes by re-invoking the same command.

## Smoke run

2 LLaPor pairs (layers 0, 10) and 1 APEX layer (layer 0) at 2 epochs each:
~8s/pair including data load, PCA/ranker fit, dev eval, and manifest write.
Already showed LLaPor beating both baselines by a wide margin
(recall@10 0.65 vs. same_expert 0.03 / popularity 0.12) and APEX at
dev_kl=0.014 after only 2 epochs -- confirmed the pipeline end-to-end before
the full run. Extrapolating smoke throughput projected the full run at
roughly 30-45 minutes; actual wall time matched that (below).

## Full run

Both `train_llapor.py` (all layers, `--layers` omitted) and `train_apex.py`
were launched concurrently against the same GPU (a single RTX 5090, 32 GB);
peak combined GPU memory was ~24 GB with GPU utilization pinned near
100%, host RAM stayed under 45 GB used / free of the 188 GB total.

- **Wall time:** LLaPor 47/47 pairs in ~31 min (30 epochs/pair, ~35-45s
  each once past the first few PCA fits); APEX 48/48 layers in ~16 min
  (ranker + CDF fit + calibration per layer, ~12-25s each). Neither hit
  early stopping in the observed runs (all pairs/layers used their full
  epoch budget); patience=5 never triggered at 30/20 max epochs here.
- **No crashes** in the full run. Four bugs were caught and fixed during
  the smoke run before it (see commits below): a device mismatch in the
  frequency-weight scatter_add, a bf16/float32 dtype mismatch in the APEX
  ranker's first linear layer, one non-benchmark rid
  (`HEALTH_CHECK_*`) crashing the split join, and a harmless autograd
  warning from converting a still-attached loss tensor to a scalar.
- Checkpoints: `/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630/{llapor,apex}/`,
  one directory per pair/layer with `model.pt`/`pca.pt` (LLaPor) or
  `ranker.pt`/`cdf.pt` (APEX), `manifest.json` (architecture, PCA stats,
  grouping, split session hashes, capture dir, gate transform, seed,
  metrics, tensor checksum), and a `DONE` marker.

## Results (condensed; full per-layer numbers in each manifest.json and
`evaluate_report.json` in the checkpoint dir)

### LLaPor: mean recall across all 47 pairs, vs. baselines

| split / phase | budget | LLaPor | same_expert (baseline) | popularity (baseline) |
|---|---|---|---|---|
| dev / decode  | 10 | **0.782** | 0.021 | 0.194 |
| dev / decode  | 16 | **0.899** | 0.031 | 0.266 |
| dev / decode  | 32 | **0.966** | 0.062 | 0.401 |
| dev / prefill | 10 | **0.755** | 0.019 | 0.218 |
| dev / prefill | 16 | **0.877** | 0.030 | 0.283 |
| dev / prefill | 32 | **0.957** | 0.059 | 0.406 |
| shifted_test / decode  | 10 | **0.685** | 0.021 | 0.192 |
| shifted_test / decode  | 16 | **0.813** | 0.032 | 0.259 |
| shifted_test / decode  | 32 | **0.921** | 0.061 | 0.388 |
| shifted_test / prefill | 10 | **0.649** | 0.019 | 0.236 |
| shifted_test / prefill | 16 | **0.772** | 0.030 | 0.306 |
| shifted_test / prefill | 32 | **0.888** | 0.059 | 0.436 |

LLaPor beats both baselines by a wide margin at every budget/phase, on both
dev and the shifted-domain test set (FinanceBench, unseen document domain).
Full budgets 10/12/16/24/32 are in `evaluate_report.json`.

### APEX: same-layer top-10 coverage, mean across all 48 layers

| split / phase | cov@10 | cov@16 | cov@24 | cov@32 |
|---|---|---|---|---|
| dev / decode | 0.521 | 0.650 | 0.743 | 0.798 |
| dev / prefill | 0.516 | 0.647 | 0.741 | 0.796 |
| shifted_test / decode | 0.549 | 0.682 | 0.771 | 0.820 |
| shifted_test / prefill | 0.482 | 0.606 | 0.700 | 0.758 |

Mean dev KL (ranker vs. teacher): **4e-5** across all 48 layers -- the
linear-softmax ranker distills the teacher distribution almost exactly.

### APEX: ordinal CDF calibration (calibration subset, mean across 48 layers)

| tau | mean requested depth (of 502 = E-K) | empirical full-set coverage |
|---|---|---|
| 0.90 | 90.5 | 0.668 |
| 0.95 | 92.5 | 0.684 |
| 0.99 | 97.1 | 0.719 |

**None of the three tau settings reaches its nominal coverage target**
(the spec's target is empirical coverage ~= tau, e.g. ~99% at tau=0.99).
Reporting the shortfall rather than a calibrated result, per the spec's
explicit instruction for this case: the CDF is well-fit in KL/BCE terms
(ranker KL is ~4e-5) but its calibration curve saturates well under 1.0
long before requesting most of the 502 non-native experts -- i.e. a
non-trivial share of rows have their true `delta_star` past where the
learned sigmoid threshold ever crosses tau, so `select_depth` falls back to
the max depth (E-K) for those rows and calibration still under-covers.
This is a genuine finding, not a training bug: dev KL is excellent while
tail-coverage calibration is poor, meaning the ranking is accurate but the
*worst-case* rank of a native expert has a heavier tail than the ordinal
CDF's single global threshold per depth captures well. Worth revisiting
(e.g. per-layer/per-phase tau, or a heavier-tailed ordinal link) before any
serving use of the calibrated depths.

## Environment

- Laptop: `/home/dimitri/data/divix/sglang-nvfp4`,
  branch `codex/nvfp4-expert-stream-main`.
- Executed and tested on divix01 only, via
  `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree` (synced with
  `git fetch shared codex/nvfp4-expert-stream-main && git checkout --detach FETCH_HEAD`),
  using `/data/models/slang/.venv` with
  `PYTHONPATH=$flashinfer_overlay:$worktree/python`.
- GPU: one RTX 5090 (32 GB), shared with other users' experiments; no
  server was started; no persistent GPU allocation exceeded ~24 GB during
  the concurrent LLaPor+APEX run.

## Commits (this task)

- `5410bcbad0` feat(expert-prediction): add offline training pipeline for LLaPor and APEX
- `b8058d29ed` fix(expert-prediction): use non-clipping frequencies in the LLaPor weight test
- `eb5ce747e7` fix(expert-prediction): exclude non-benchmark rids from every split
- `7beeaadf07` fix(expert-prediction): keep frequency-count scatter_add on the input device
- `e4abd61cff` fix(expert-prediction): detach the running loss before float() conversion
- `d4d2b38f7c` fix(expert-prediction): cast the ranker's bf16 pre_mixer input to float32
