# Expert Prediction Capture Smoke (2026-09-14)

## Setup

- Commit: `97ceacd64d` (`feat(moe): add capture two-turn smoke driver and gate top-k checker`), on `codex/nvfp4-expert-stream-main`.
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

- **Bytes per row:** 534,214 (vs. the ~494,000 estimate in the plan; ~8% higher, likely rounding in the estimate's per-layer overhead term).
- **Disk on shard:** 324 MiB for 635 rows across 48 layers.
- **Gate top-k agreement per layer** (`check-capture-gate-topk.py` against the NVFP4 model's `mlp.gate.weight`):
  - Layer 0: **not computed** — the gate-key regex matches both `model.language_model.layers.0.mlp.gate.weight` and `mtp.layers.0.mlp.gate.weight`, so the script reports "gate key not unique" for layer 0 and skips it. Open item below.
  - Layers 1-47: min **0.9**, most layers at **1.0**. Layers below 0.99: 11, 16, 17, 20, 25, 27, 29, 32, 39, 43, 45, 46 (all at 0.9).
  - Mean over the 47 computed layers: ~0.98.
- **Decode tok/s with capture on:** mean 12.40 tok/s over 12 `Decode batch` log lines (13.39 tok/s excluding the first, cold-start line), vs. **13.76 tok/s** with capture off (`2026-09-14-expert-prediction-shadow-smoke.md`). About a 3-10% slowdown depending on whether the cold-start sample is included; within the noise of a 12-sample log window from a single short smoke run.

## Disk projection

- Tokens per 100 GB: `100 GiB / 534,214 bytes/row ≈ 200,995 rows`.
- Hours of decode per 100 GB: `200,995 rows / 13.76 tok/s / 3600 ≈ 4.06 hours` (capture-off rate) to `≈ 4.50 hours` at the measured 12.40 tok/s capture-on rate.

## Open items

- **Radix cache decision for the capture server:** decided in `2026-09-14-radix-cache-ab.md` (enable, `extra_buffer`, 8 mamba slots, 12 GB hot cache); this smoke used the same flags. Production's launcher still needs the change.
- **Speculative decoding support:** out of scope for capture; VERIFY rows would need a branch tag and an acceptance join (see plan's "Open items after this plan").
- **Per-session tagging from the traffic driver:** the traffic driver should log `rid -> session, domain, split` so shards can be split by session without parsing prompts.
- **New:** `check-capture-gate-topk.py`'s gate-key regex is ambiguous for layer 0 when an `mtp.layers.0...` key exists alongside `model.language_model.layers.0...`; it should anchor on the full prefix (e.g. require `model.language_model.` or exclude `mtp.`) rather than a bare `(^|\.)` boundary. Left unfixed here since the plan's script is the spec and this affects only the offline gate-check tool, not capture correctness.
- **New:** several layers (11, 16, 17, 20, 25, 27, 29, 32, 39, 43, 45, 46) show gate top-k agreement of only 0.9, below the plan's 0.99 threshold. Per the plan, this means router logits should not be dropped from capture for those layers pending further investigation into why the gate-weight-only reconstruction diverges (possibly NVFP4 quantization effects on the gate weights, or shared-expert/bias terms not captured by `router_input @ gate.weight.T`).
