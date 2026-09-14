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
- **D** `div-on`: `affinity,popularity` with `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL=1`
  (scoring every eligible forward, fresh launch). Initially skipped (controller condition "A
  == B through >= 300 chars and C differs from A" wasn't met, A/B diverged at 129), then run
  on controller follow-up because A/B/C/shadow-off-1 all agreed through >= 73 chars while
  shadow-on-1 (the earlier smoke's only scoring-every-step run) uniquely diverged from
  everything at char 22 — an unexplained outlier worth checking against a fresh scoring
  launch. Run dir
  `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/div-on/run-20260914-163435`.
  Median decode `gen throughput (token/s)` over 14 decode-batch log lines: **6.955** (close to
  Task 7's shadow-on median of 6.85, consistent with the known scoring-kernel-launch
  overhead).

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
| A (off) vs D (on, scoring every forward) | 129 | 1845 / 1762 |
| B (off) vs D | 232 | 1801 / 1762 |
| C (taps-only) vs D | 73 | 1885 / 1762 |
| shadow-off-1 vs D | 73 | 1717 / 1762 |
| D vs shadow-on-1 | 22 | 1762 / 1772 |

No pair matched past ~230 characters; every pair, including two fresh off-vs-off launches
(A vs B), diverged well before the 300-char bar the brief set for "taps are inert."

### Verdict: H0 (shadow-on-1 was a launch outlier)

Two off/off fresh launches (A vs B) diverge at char 129 — the same order of magnitude as
off-vs-taps-only (73) and off-vs-D (129/232). A/B/C/shadow-off-1 all agree with each other
through **>= 73 chars** pairwise. D (`affinity,popularity`, scoring every forward, fresh
launch) also agrees with A, B, C, and shadow-off-1 in that same 73-232 char range — it does
**not** reproduce shadow-on-1's early (char-22) divergence from everything else. D vs
shadow-on-1 itself diverges at char 22, i.e. D behaves like an ordinary off/taps-only run
relative to every other run, and only shadow-on-1 stands apart, diverging from all five other
runs (A, B, C, D, shadow-off-1) at exactly char 22.

Per the controller's decision rule: D agreed with A/B/C through >= 73 chars, so
**shadow-on-1 was a launch outlier, and H0 (launch-to-launch nondeterminism) holds** — not
H2 (per-step scoring). If scoring-every-forward reliably perturbed output, D should have
diverged early like shadow-on-1 did; instead D's divergence points (73-232 chars against the
other five runs) fall squarely inside the same noise band as pairs where the predictor was
off or taps-only throughout. Divergence character does not track predictor activity in
either direction: off-vs-off (A-B) can diverge as early as taps-vs-off (C vs A/B, 73) or as
late as 232 (B vs D), and the one run that diverged earliest overall (shadow-on-1, char 22)
is matched no more closely by other scoring runs (D) than by pure off runs. This is
consistent with experiment log E7's prior finding that `explain`-class replies are
non-reproducible across runs, and confirms the original Task 7 off-vs-on divergence (char 22)
cannot be attributed to expert prediction — it was launch noise, and shadow-on-1's early
divergence in that run did not repeat when scoring-every-forward was relaunched fresh (D).

Caveat: n=1 per configuration (except off, n=2, and shadow-on-1-like runs, n=2: shadow-on-1
and D), so this remains circumstantial rather than a bitwise proof; but it directly
contradicts H2 (D was the exact repeat of shadow-on-1's config and did not reproduce its
divergence pattern), and no evidence in either run set supports H1.

Per the brief, no code fix was attempted for H1/H2 (none is warranted under H0 anyway).

### Commits

- `docs(nvfp4): record expert prediction output divergence check` — the original A/B/C
  section above (`docs/superpowers/experiments/2026-09-14-expert-prediction-shadow-smoke.md`,
  staged by name only).
- `docs(nvfp4): add scoring-launch divergence run` — added run D, its table rows, and the
  corrected verdict (same file, staged by name only).
- No launcher change was needed in either pass: `SGLANG_MOE_EXPERT_PREDICTOR_SCORE_INTERVAL`
  reached the shadow server correctly on the first try (verified via `/proc/<pid>/environ`).
