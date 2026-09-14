# Expert Prediction Shadow Mode Smoke Test (2026-09-14)

Commit under test: `a0ec88567d` (`feat(moe): add expert prediction shadow server launcher`),
run in worktree `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree` on divix01.
Both servers ran production's E16c settings via
`scripts/expert_prediction/run-shadow-server.sh`, port 7871, `--max-running-requests 1`.

## Runs

- `shadow-off` (`SGLANG_MOE_EXPERT_PREDICTOR=""`): run dir
  `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/shadow-off/run-20260914-155339`
- `shadow-on` (`SGLANG_MOE_EXPERT_PREDICTOR="affinity,popularity"`): run dir
  `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/shadow-on/run-20260914-155954`

Both requests: `POST /v1/chat/completions`, greedy (`temperature: 0`), `max_tokens: 600`,
`chat_template_kwargs: {"enable_thinking": false}`, prompt "Explain how a B-tree handles
node splits, with a worked example." Sent twice per server; only the second response
(post-warmup) is compared/measured.

## Decode throughput (second request, `gen throughput (token/s)` median across decode
batches, cuda graph: True throughout for both)

| Server | Median tok/s (req 2) |
|---|---|
| shadow-off | 13.76 |
| shadow-on (affinity,popularity) | 6.85 |

Shadow scoring roughly halves decode throughput in this single-request, single-GPU
smoke config (48 tapped MoE layers, `--max-running-requests 1`). No host sync was
observed or expected to cause this; the added kernel-launch and scoring work itself
accounts for the slowdown. Not investigated further per the task scope (only "record",
not "fix" tok/s here).

## Output equality (off vs on, second request)

**DIFFER.** First differing character index: 22.

- `shadow-off-2.json` content starts: `"# B-Tree Node Splits: Explanation and Worked Example"`
- `shadow-on-2.json` content starts: `"# B-Tree Node Splits: A Detailed Explanation\n\n## Cor..."`
- Lengths: off = 1812 chars, on = 1849 chars.

Shadow mode is scoring-only (predict + observe + metrics), it does not gate or alter
expert selection, so this divergence is unexpected from the design intent. Per the
controller's instruction this is recorded as a concern and not investigated/fixed here.

## Checks

- `shadow-on` `server.log` contains:
  `MoE expert prediction shadow mode: predictors=affinity,popularity layers=48 max_rows=1 tap_bytes=3840 predictor_state_bytes=49479680`
  — matches expected format (predictors, layers=48). No `cannot tap layer` warning present.
- All 29 decode batch log lines in `shadow-on/server.log` report `cuda graph: True`.
- `shadow-on/expert-prediction.metrics.jsonl` has 11 records (>= 5 required).
- Last record (`forwards: 1100`) totals:
  - `affinity.total`: `routes=517000`, `recall_at_k=0.4730`, `recall_at_m=0.5943`,
    `cold_recall_at_m=0.1432` (all ratios within [0, 1]).
  - `popularity.total`: `routes=528000`, `recall_at_k=0.2996`, `recall_at_m=0.3844`,
    `cold_recall_at_m=0.0098` (all ratios within [0, 1]).

## Task 6 (CUDA graph replay / no-sync test)

`test/registered/unit/layers/moe/test_expert_prediction_graph.py` +
`test_expert_prediction_runtime.py` on divix01 GPU (commit `c2859408c1`):
`9 passed`, `EXIT=0`. No synchronizing op was triggered under
`torch.cuda.set_sync_debug_mode("error")`; no fix needed.

## Notes

- Production was left down for this smoke test per the controller's direction; it was
  not relaunched afterward (out of scope for this entry).
- GPU was confirmed free (`nvidia-smi --query-compute-apps`) before each server launch
  and confirmed clear again after each `SIGTERM`.
- Nothing crashed, no missing log lines, no framework bug surfaced beyond the two
  findings above (throughput regression under shadow scoring, and the greedy-output
  divergence), both recorded as concerns rather than fixed.
