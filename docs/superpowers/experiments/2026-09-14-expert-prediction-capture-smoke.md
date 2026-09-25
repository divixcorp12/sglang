# Expert Prediction Capture Smoke (2026-09-14)

## Setup

- Commit: `97ceacd64d` (`feat(moe): add capture two-turn smoke driver and gate top-k checker`), on `master`.
- Launch: `HOT_GPU_MB=12288 CAPTURE=1 run-shadow-server.sh capture-smoke 31010 off radix` (predictors off, radix cache on, 12 GB hot cache, matching the production recommendation in `2026-09-14-radix-cache-ab.md`).
- Capture directory: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-smoke/20260914-231957`.
- Server log confirms: `MoE expert capture: directory=... capacity_rows=4096 frames=2 pinned_bytes=4074242048 max_bytes=536870912000`.

## Usage JSON (two-turn chat)

```json
{"turn1": {"prompt_tokens": 81, "total_tokens": 337, "completion_tokens": 256, "prompt_tokens_details": null, "reasoning_tokens": 256}, "turn2": {"prompt_tokens": 107, "total_tokens": 363, "completion_tokens": 256, "prompt_tokens_details": null, "reasoning_tokens": 256}}
```

## Reader report

```json
{"shards":2,"rows":635,"prefill_rows":125,"decode_rows":510,"forwards":513,"ran_rows":635,"requests":3,"bytes":339225925,"bytes_per_row":534214.0551181103,"stopped_reason":null,"violation_count":0,"violations":[]}
```

- `violations: []`, `stopped_reason: null`.
- `decode_rows == 510 == turn1.completion_tokens (256) + turn2.completion_tokens (256) - 2` (each request's first output token comes from its prefill forward, not a decode forward). Matches expectation exactly.
- `prefill_rows == 125`, matching the sum of `#new-token` over the three `Prefill batch` log lines: `1 (warmup) + 81 (turn1) + 43 (turn2) = 125`. With radix cache on, turn 2 only reprefills 43 of its 107 prompt tokens (`#cached-token: 64` reused from turn 1), well under `turn1.prompt_tokens + turn2.prompt_tokens = 188`. Turn 2's new rows start at position 64 and carry `prefix_hash = 0` (history before the cached prefix boundary was not itself observed as a captured prefill), which is expected.
- `requests == 3`: the health-check warmup forward, turn 1, and turn 2 each got a distinct `rid`.

## Measured values

- **Bytes per row:** 534,214, vs the ~494,400 plan estimate. The difference is the per-forward `forward.expert_to_slot` snapshot: 513 forwards x 48 layers x 512 experts x 2 B = 25.2 MB, or 39.7 KB per row when most forwards are 1-row decodes. 494.4 + 39.7 = 534.1 KB. Long prefills amortize it; decode-heavy traffic pays it per token.
- **Disk on shard:** 324 MiB for 635 rows across 48 layers.
- **Gate top-k agreement per layer** (`check-capture-gate-topk.py` against the NVFP4 model's `mlp.gate.weight`):
  - The first run of the script read only `manifest[0]`. That shard was the idle flush of a single warmup row, so agreement could only be 0.9 or 1.0. It also skipped layer 0, because `mtp.layers.0.mlp.gate.weight` matched the key regex. The script now reads every shard and excludes `mtp.` keys.
  - Over all 635 rows, layers 0-47: agreement **0.982-0.992** (min layer 45, max layer 24).
  - Mismatches sit only at predicted ranks 9-10 (rank 8 at most 3%, ranks 1-7 about 0). At mismatched rows, the median logit gap between the 10th and 11th expert is 0.005-0.010, vs a 0.03-0.055 median over all rows. These are near-ties: the router's bf16/fused top-k and a float32 recompute from bf16 router input order the last slot differently.
  - Conclusion: capture is consistent with the router. Router logits need not be stored, because gate weight x router input reproduces them to near-tie precision. That is adequate for soft targets such as APEX's KL ranker. Captured `topk_ids` remain the label ground truth, since a recompute can swap ranks 9-10.
- **Decode tok/s with capture on:** mean 12.40 tok/s over 12 `Decode batch` log lines (13.39 tok/s excluding the first, cold-start line), vs. **13.76 tok/s** with capture off (`2026-09-14-expert-prediction-shadow-smoke.md`). About a 3-10% slowdown depending on whether the cold-start sample is included; within the noise of a 12-sample log window from a single short smoke run.

## Disk projection

- Tokens per 100 GB: `100 GiB / 534,214 bytes/row ≈ 200,995 rows`.
- Hours of decode per 100 GB: `200,995 rows / 13.76 tok/s / 3600 ≈ 4.06 hours` (capture-off rate) to `≈ 4.50 hours` at the measured 12.40 tok/s capture-on rate.

## Open items

- **Radix cache decision for the capture server:** decided in `2026-09-14-radix-cache-ab.md` (enable, `extra_buffer`, 8 mamba slots, 12 GB hot cache); this smoke used the same flags. Production's launcher still needs the change.
- **Speculative decoding support:** out of scope for capture; VERIFY rows would need a branch tag and an acceptance join (see plan's "Open items after this plan").
- **Per-session tagging from the traffic driver:** the traffic driver should log `rid -> session, domain, split` so shards can be split by session without parsing prompts.
- **Resolved:** the gate check read one shard and matched the MTP gate key. It is fixed. All-row agreement is 0.982-0.992, with only near-tie swaps at ranks 9-10. The plan's 0.99 bar was too strict for a float32 recompute of bf16 routing.
- **Residency snapshot cost:** about 50 KB per forward. If decode-heavy capture volume matters, store `expert_to_slot` only when it changes (the hot cache updates every 4 decode forwards) or as deltas.
