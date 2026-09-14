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

**Label: shadow scoring overhead, not a prefetch latency result.** Shadow scoring
currently launches thousands of small eager kernels after every decode step (across
all 48 tapped layers x both predictors); the off-vs-on gap above mostly reflects that
per-step host launch overhead, not predictor or framework decode latency. No host sync
was observed or expected to cause it. Not optimized here — a follow-up fix will address
the eager-kernel-launch overhead after this smoke test.

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
  findings above (shadow-scoring kernel-launch overhead lowering decode tok/s, and the
  greedy-output divergence), both recorded as concerns rather than fixed.

## Follow-up: does shadow prediction cause the greedy-output divergence? (2026-09-14)

Commit tested: `d6a67beac3` (fix wave; adds `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL`).
Question: does installing taps/buffers or running scoring change greedy output, or is the
divergence ordinary launch-to-launch nondeterminism (H0 vs H1 vs H2, see
`.superpowers/sdd/2026-09-14-moe-expert-prediction-framework/divergence-brief.md`)?

Same request each run (one request, first after health): `POST /v1/chat/completions`,
prompt "Explain how a B-tree handles node splits, with a worked example.", `max_tokens=600`,
`temperature=0`, `chat_template_kwargs.enable_thinking=false`, model = the served NVFP4
Qwen3.8-Flash-Next model path. Port 7871, 127.0.0.1, production down, GPU confirmed free
before each launch and confirmed clear after each stop.

### Runs

- **A** `div-off-a`: predictor off. Run dir
  `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/div-off-a/run-20260914-161824`.
- **B** `div-off-b`: predictor off (repeat, fresh launch). Run dir
  `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/div-off-b/run-20260914-162252`.
- **C** `div-taps-only`: `affinity,popularity` with
  `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL=1000000000` (taps/buffers installed, but with
  the fix wave's `(eligible-1) % interval == 0` rule the first eligible forward is still
  scored once, then never again). Confirmed via `/proc/<pid>/environ` on divix01 that the env
  var reached the server process (launcher passes it through correctly; no launcher change
  was needed). Run dir
  `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/div-taps-only/run-20260914-162726`.
  `server.log` shows `MoE expert prediction shadow mode: predictors=affinity,popularity
  layers=48 max_rows=1 tap_bytes=3840 predictor_state_bytes=49479680`, no `cannot tap layer`
  warning. No `expert-prediction.metrics.jsonl` was written (log-interval gate never fires
  again after the single scored forward), consistent with scoring running exactly once.
- **D**: not run. Controller condition for D was "A == B through at least 300 chars and C
  differs from A"; A and B already diverged at char 129 (< 300), so D was skipped per the
  brief.

All completions used the full `max_tokens=600` budget (`completion_tokens=600` in every
run's `usage`).

### First-difference table (character index into `choices[0].message.content`)

| Pair | First diff index | Lengths (chars) |
|---|---|---|
| A (off) vs B (off) | 129 | 1845 / 1801 |
| A (off) vs C (taps-only) | 73 | 1845 / 1885 |
| B (off) vs C (taps-only) | 73 | 1801 / 1885 |
| A (off) vs shadow-off-1 (earlier smoke, off) | 73 | 1845 / 1717 |
| B (off) vs shadow-off-1 | 73 | 1801 / 1717 |
| C (taps-only) vs shadow-off-1 | 133 | 1885 / 1717 |
| A (off) vs shadow-on-1 (earlier smoke, `affinity,popularity` scoring every forward) | 22 | 1845 / 1772 |
| B (off) vs shadow-on-1 | 22 | 1801 / 1772 |
| C (taps-only) vs shadow-on-1 | 22 | 1885 / 1772 |
| shadow-off-1 vs shadow-on-1 (Task 7's original comparison) | 22 | 1717 / 1772 |

No pair matched past ~130 characters; every pair, including two fresh off-vs-off launches
(A vs B), diverged well before the 300-char bar the brief set for "taps are inert."

### Verdict: H0

Two off/off fresh launches (A vs B) diverge at char 129 — the same order of magnitude as
off-vs-taps-only (73) and the original off-vs-on Task 7 comparison (22). Divergence does not
grow monotonically with how much predictor machinery is active: A-vs-C (73, taps installed,
scored once) is *smaller* than A-vs-B (129, both predictor off), and C is closer to
shadow-off-1 in divergence point (133) than to shadow-on-1 (22). If taps/buffers (H1) or
per-forward scoring (H2) were perturbing numerics, divergence onset should track predictor
activity; instead it is roughly constant (character range 22-133) regardless of whether the
predictor is off, taps-only, or fully scoring. This matches H0: separate server launches are
not repeatable at greedy decoding (consistent with experiment log E7's prior finding that
`explain`-class replies are non-reproducible across runs), and the original Task 7 off-vs-on
divergence cannot be attributed to expert prediction with this evidence. The exact
divergence character varies run to run (22-133) rather than clustering tightly, so this is
circumstantial, not a proof of exact bitwise cause, but it is sufficient to reject H1/H2 as
the primary explanation for Task 7's observation.

Per the brief, no code fix was attempted for H1/H2 (none is warranted under H0 anyway).

### Commits

- `docs(nvfp4): record expert prediction output divergence check` — this entry
  (`docs/superpowers/experiments/2026-09-14-expert-prediction-shadow-smoke.md`, staged by
  name only).
- No launcher change was needed: `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL` reached the
  shadow server correctly on the first try (verified via `/proc/<pid>/environ`).
