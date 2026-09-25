# Handoff: finishing the MoE expert-prefetch plan (LLaPor vs APEX)

Written 2026-09-16 for a fresh session with no memory of the one that wrote it. Everything below
was checked against the tree or against divix01 at time of writing; anything not checked is marked.
**Re-verify state (section 2) before acting on it** — another session (crypto-c9) commits to this
branch continuously.

---

## 1. The question, and where the answer stands

**Standing goal:** compare decode tokens/sec for two MoE expert predictors used to prefetch expert
weights from host memory before the GPU needs them:

- **LLaPor** — next-layer predictor. 47 models (source layer L predicts layer L+1, targets 1-47).
- **APEX** — same-layer predictor. 48 models (targets 0-47), scores from the layer's pre-mixer state.

Model: Qwen NVFP4, 48 MoE layers, 512 experts/layer, top_k 10. One expert row = **2,764,808 B**.

**What is settled:**

- **The predictors are not the bottleneck.** LLaPor converged (train loss flat by epoch 30, dev
  recall flat, no overfit). Both are ~19x a static popularity prior. At budget B=2, the oracle
  ceiling is **0.497**; APEX 0.409 = 82% of it, LLaPor 0.342 = 69%.
- **Budget is the binding constraint.** The capture data has 3.22 misses/row; B=2 caps even a
  perfect predictor at 0.497, and 39.31% of rows miss 4+ experts.
- **Scoring cost is measured** (Task 6, live): LLaPor ≈ +2.10 ms/token, APEX ≈ +3.26 ms/token over
  scoring-off, at ~62 ms/token baseline.
- **Delivery cost is measured** (E36): 0.2237 ms/row in-graph (~11.51 GiB/s, R²=0.99996); a
  count-zero floor of 0.324 ms/token. At production miss counts, serialized demand delivery is
  42-51% of a decode step — **that is the size of the target, not the size of the win.**

**What is NOT settled — this is the work that remains:**

1. **No tok/s comparison of prefetch-on vs prefetch-off exists.** Task 6 was shadow only (scores,
   transfers nothing). The actual A/B needs arm D (section 7).
2. **Every prediction number we own was measured at ~33% miss rate.** Production steady state is
   **13.6-20.5%** (1.36-2.06 misses/layer). Recall measured at 3.3 misses/layer does not extrapolate
   to 1.4-2.1; the plan's break-even/net figures at 1.36 and 2.06 were **withdrawn** for this reason.
3. **The LLaPor-vs-APEX recall comparison crossed a seam** at `c48b3c69e5` (section 6).

---

## 2. State of the world at handoff (re-check all of it)

| Thing | State | How to re-check |
|---|---|---|
| Local branch | `master`, HEAD `f867e73e8b`, **55 commits ahead of `shared`** | `git status --short --branch` |
| `shared` remote tip | `bd8cc84d1a` | `git log -1 origin/master` |
| divix01 test worktree | `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree` at **`d725c54a31`** — far behind local, clean | section 4 |
| Production (port 7867) | **down** (health 000). Left down deliberately; whether that is intended is an open user question | section 4 |
| GPU | 4.5 / 32.6 GiB used by two processes that are not ours (pids 3349801 3.7 GiB, 3349981 0.76 GiB) | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` |
| `cc-gpu.lock` | zero-byte, not held | `fuser -v /data/models/slang/nvfp4-work/cc-gpu.lock` |
| Uncommitted, mine | `docs/superpowers/plans/2026-09-15-moe-expert-prefetch-live.md` (+21/-1: regime finding, oracle table, withdrawals, E36/E37 blocks, provenance bullet) | `git diff --stat` |
| Untracked, mine | `MOE_EXPERT_TRANSFER.md` at repo root — location and commit undecided | — |
| Untracked, not mine | `docs/superpowers/plans/2026-09-16-serving-handoff-stage-c-flags.md`, `benchmark/expert_delivery/` (crypto-c9) | — |

Stage progress in the side-stream plan: **A, B, C1-C4 committed and reviewed (APPROVE).
Stage D (measurement) and E (doorbell, optional) not started.**

---

## 3. Hard rules (non-negotiable; each has bitten before)

**divix01 access**
- Every ssh: `ssh -n -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes divix01 '<cmd>'`.
  Never `ssh -t`, never `tmux capture-pane`.
- **Every CPU job under `taskset -c 0-63` with capped threads** (`OMP_NUM_THREADS=32 MKL_NUM_THREADS=32`,
  or `torch.set_num_threads`). Cores 64-71 are reserved: core 71 is production's doorbell spin core
  and the doorbell tests use 64-71.
- **The GPU is never free.** Production uses ~25-30 GiB when up. Before any GPU time: message
  crypto-c9 (cross-session, name `crypto-c9`) with duration and memory; always take
  `/data/models/slang/nvfp4-work/cc-gpu.lock`; **the user must approve any production downtime**
  (AskUserQuestion). Note: `2026-09-16-serving-handoff-stage-c-flags.md` §2 says you may stop 7867
  without asking if you notify — that conflicts with this rule; this rule is the user's and is
  stricter, so ask.
- Never kill a process you did not start. Never relaunch production.
- Never modify the serving worktree `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb` or
  `run-nvfp4-e16c-public.sh`. Never edit anything under `/home/dimitri/data/divix/crypto` (read-only).

**Git**
- Stage by name; commit with `git commit -m ... -- <paths>`. No `git add -A`/`.`. **`git stash` is forbidden.**
- Never stage `.omc/`, `.superpowers/` (gitignored scratch; one agent force-added a file there and
  it had to be untracked at `0d0825eca2`), or
  `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md` (crypto-c9 owns and commits it).
- **Never push to `origin`.** Pushing to `shared` (a private bare repo on divix01) was user-approved
  earlier for syncing, but the branch has since accumulated 55 unpushed commits, including
  crypto-c9's Stage C work and an **unpushed doorbell merge (`da3e9297be`, `d4ad7fb8ba`) whose merge
  brief claims two tests pass that demonstrably fail**. Pushing `shared` publishes all of it.
  **Ask the user before the first push this session.** Section 4 gives a no-push alternative.
- Trailers: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01NurjMVe2nS3PGBqBr8M8MZ` (or the new session's own URL).

**Code**
- `model_runner.py` is frozen — orchestration only; read the `large-class-style` skill first.
- `msgspec.Struct`, not `@dataclass`. No defensive `getattr`/`hasattr`.
- Env vars only through `python/sglang/srt/environ.py` (read the `env-var-conventions` skill).
- Delegation: user wants **sonnet agents for everything but the hardest tasks**.

---

## 4. Workflow: push to divix01 and test

Local has no torch/CUDA (`ModuleNotFoundError: torch`). All tests run on divix01 with
`/data/models/slang/.venv/bin/python`.

### 4a. Pre-flight (every time)

```bash
ssh -n -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes divix01 '
  curl -s -o /dev/null -w "7867=%{http_code}\n" --max-time 3 http://127.0.0.1:7867/health
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ps -p $p -o pid,user,etime,args --no-headers | cut -c1-160; done
  fuser -v /data/models/slang/nvfp4-work/cc-gpu.lock 2>&1'
```

Record the GPU census before and after any timing run. **Timing does not tolerate co-tenants**;
correctness tests do.

### 4b. Sync path 1 — via `shared` (after user approves the push)

```bash
git commit -m "..." -- <paths>
git push -q origin master
ssh -n -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes divix01 '
  cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree &&
  git status --short | head -3 &&
  git fetch -q origin master &&
  git checkout -q --detach FETCH_HEAD && git log -1 --oneline'
```

Check `git status` is clean on divix01 first; the worktree is detached, so checkout moves it.

### 4c. Sync path 2 — no push (used for all verification this session)

Ship a commit's tree into a throwaway dir. No remote, no worktree registration:

```bash
git archive <commit> python test scripts | ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=10 divix01 \
  'rm -rf /tmp/dnikolaidis-<name> && mkdir -p /tmp/dnikolaidis-<name> && tar -x -C /tmp/dnikolaidis-<name>'
```

Note `ssh` without `-n` here — stdin carries the tarball. Only committed content ships; that is a
feature (what you test is exactly a commit).

### 4d. Run tests

```bash
ssh -n -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes divix01 '
  cd /tmp/dnikolaidis-<name> &&
  export PYTHONPATH=$PWD/python OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 &&
  taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang,torch; print(sglang.__file__, torch.cuda.is_available())" &&
  taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <test path> -q 2>&1 | tail -20'
```

**Always print `sglang.__file__` first.** If it doesn't resolve under your directory, you are
testing the installed package, and every pass is meaningless.

Relevant suites (`test/registered/unit/layers/moe/`): `test_expert_hot_cache_publication.py`,
`test_expert_graph_gather.py`, `test_expert_graph_gather_scratch.py`, and the prediction tests
`test_expert_prediction_*.py` in the same directory (graph, taps, predictors, adapters, metrics, ...).

**Known failures at HEAD — not yours:** two doorbell tests in `test_expert_graph_gather.py` fail,
independently reproduced at base `f415048d25`:
- `test_doorbell_drain_on_this_path_ends_at_its_budget_and_never_recovers_the_copy` (`:636`) —
  passes in isolation, fails in the suite. **The isolated PASS is the artifact** (E37):
  `_jit_expert_doorbell_module()` is `@functools.cache`'d, so the first test in a process pays
  ~1.2 s of CUDA module load, which lands inside the drain's poll budget.
- `test_doorbell_gather_falls_back_to_correct_rows_when_the_thread_stalls` (`:1142`, assert `:1166`).

**Qualify counts by file.** "2 failed / 22 passed" is `test_expert_graph_gather.py` alone;
"2 / 29" includes `_scratch.py`. Unqualified, a later clean 22 reads as seven tests vanished.

### 4e. Live server runs

Launcher: `scripts/expert_prediction/run-shadow-server.sh <name> <port> <predictors|off> [radix]`.
Env knobs: `PREFETCH_PREDICTOR`, `PREFETCH_MODEL_DIR` (default
`/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630`), `PREFETCH_BUDGET` (default 3),
`PREFETCH_CANDIDATES` (default 16), `HOT_GPU_MB` (default 12288). It serves the divix01
`cc-expert-prediction/worktree`, **refuses to start if 7867 is listening**, and `exec flock --nonblock`s
`cc-gpu.lock` — so it takes the lock for you and fails fast if someone holds it.

Per-launch procedure (from plan Task 6 Step 3):

1. `ssh -n ... divix01 'nohup env PREFETCH_PREDICTOR=... <launcher> <cell>-p<pass> 31040 off radix > /dev/null 2>&1 &'`
2. Wait for `/health` 200 with a Monitor until-loop (not sleep).
3. Warm up with the fixed session `cfq-train-Single_CDW/2015/page_35.pdf-2` — identical every cell.
4. `scripts/expert_prediction/prefetch/logprob_probe.py --port 31040 --sessions <subset> --prompts 8 --max-tokens 96 --out .../<cell>-p<pass>-logprobs.json`
5. `scripts/expert_prediction/benchmarks/run_capture_sessions.py --port 31040 --sessions <subset> --results .../<cell>-p<pass>.jsonl --max-tokens 768`
6. Copy the run dir's `expert-prediction.metrics.jsonl` and `hot-cache.metrics.jsonl`.
7. `pkill -f '[s]glang serve.*--port 31040'` (only your own server), wait for the GPU to empty.

Check the startup log: a scoring arm must log `MoE expert prefetch scoring: predictor=llapor targets=47`
(APEX: `targets=48`); an off arm must not. **Check that the line is present, not merely that the run
didn't error** — see section 8.

---

## 5. How results are measured

### Offline (CPU, capture replay)

- Capture: `/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-full/20260915-054438`
- Checkpoints: `/mnt/nvme2/nvfp4-work/expert-prediction-models/20260915-121630/{llapor,apex}`
- Pricing: `scripts/expert_prediction/prefetch/price_prefetch.py --predictor {llapor,apex,oracle,popularity} --scorer-ms 0,0.02,0.05 --reaction-ms 0,0.03`
  under `taskset -c 0-63` with `OMP_NUM_THREADS=32 MKL_NUM_THREADS=32`. Outputs in
  `/mnt/nvme2/nvfp4-work/prefetch-gate-v2/`.
- Parity: `check_serving_parity.py` (fp32 training vs bf16 serving scorers, recall@16 within 0.005).
- Record: `docs/superpowers/experiments/2026-09-15-expert-prefetch-offline-gate.md`.

**Metric: `budget_recall`** = covered non-resident native routes / total non-resident native routes,
pooled across layers (`serving/runtime.py`, `metrics_record`). Offline counterpart is `budget_hits`.

| B | popularity | LLaPor | APEX | oracle |
|---|---|---|---|---|
| 1 | 0.0095 | 0.196 | 0.231 | 0.2743 |
| 2 | 0.0181 | 0.342 | 0.409 | 0.4970 |
| 3 | 0.0265 | 0.453 | 0.544 | 0.6661 |

(`shifted_test_decode` window; all at the capture's ~33% miss regime.)

### Live (GPU, shadow A/B) — plan Task 6, done

- Record: `docs/superpowers/experiments/2026-09-15-expert-prefetch-live-shadow.md`
- Subset: `/mnt/nvme2/nvfp4-work/benchmarks/prefetch-shadow/sessions.jsonl` (8 sessions, 29 turns,
  `select_ab_sessions.py --holdout 2 --val 6`).
- **tok/s:** median decode tok/s over turns with `completion_tokens >= 64`; ms/token difference
  against the scoring-off arm is the scoring cost. Summarize with `summarize_ab.py arm=results.jsonl[:prediction-metrics.jsonl] ...`.
- **Pass order interleaved and reversed:** pass 1 REF→S0→S1→S2, pass 2 S2→S1→S0→REF. Never compare
  against a baseline from another day — rerun it in the same session.
- **Correctness gate:** answer-level agreement (ConvFinQA `correct`) only where both sides have
  `finish_reason=stop`. Exact-logprob flips are a **diagnostic only** — REF vs itself gave 5 flips,
  so exactness measures run-to-run nondeterminism. Report truncation rate per arm.
  The answer gate can also fire within one build (a same-arm pass pair disagreed) — a criterion
  that fires between two passes of one build is not evidence about a difference between builds.

Results (B=2): LLaPor recall 0.383/0.389, APEX 0.424/0.420; tok/s S0 16.17, S1 15.46/15.83, S2
15.38/15.35. **These recall figures are pre-seam — see section 6.**

### Delivery cost (E36, in-graph)

0.2237 ms/row eager, 0.2240 replay, floor 6.752 µs/layer. Stage B measured side-stream overlap at
**0.0645 ms, launch-order-bound** — do not cite it as the overlap ceiling; replay-based overlap is
unmeasured.

### Ledgers and records

| File | What | Owner |
|---|---|---|
| `docs/superpowers/plans/2026-09-15-moe-expert-prefetch-live.md` | **The plan being finished.** Tasks 1-7 (Phase A), B1-B2 (Phase B), Risks (`:2083`), Decisions (`:2176`) | this session (uncommitted edits) |
| `docs/superpowers/plans/2026-09-16-side-stream-expert-pull-handoff.md` | Side-stream pull plan; Stages A-E; **§11 measurement matrix (arms A-E)** is the live A/B design now | crypto-c9 |
| `docs/superpowers/plans/2026-09-16-serving-handoff-stage-c-flags.md` | How to launch with Stage A/C flags; readiness table; rollback | crypto-c9 (untracked) |
| `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md` | Numbered experiments; **E36** delivery cost, **E37** drain attribution, **E38** recall seam | crypto-c9 only — never stage |
| `docs/superpowers/experiments/2026-09-15-expert-prefetch-{offline-gate,live-shadow}.md` | T3 and T6 records | committed |
| `.superpowers/sdd/2026-09-15-moe-expert-prefetch-live/progress.md` | SDD progress ledger for the plan (rulings, user decisions) | scratch, not committed |
| `.superpowers/sdd/2026-09-16-side-stream-expert-pull-handoff/progress.md` + `stage-*-{brief,report}.md` | crypto-c9's stage ledger | scratch |
| `MOE_EXPERT_TRANSFER.md` | Map of the five expert-transfer paths. **Has stale facts:** row size says 2,764,800 (correct: 2,764,808) and five further verified corrections from this session are not yet applied; state line cites `a5d45d7068` | this session, untracked |

---

## 6. Traps specific to this work

### The `budget_recall` seam at `c48b3c69e5` (E38)

Before it, `PrefetchCandidateBank.write` took the top-W by score with no residency filter and
`BudgetRecall.observe` masked residents afterward. After it, the bank excludes residents before
truncating. The old offering is a strict prefix of the new, so **`budget_recall` can only rise, for
reasons unrelated to prediction quality.** A post-seam 0.45 is not an improvement over 0.42.

**And the arms don't move equally.** Residency is sampled at `bank.write` (source layer's score
trigger) and again at `observe` (target layer's `TOPK_IDS`), `serving/runtime.py:106-125`. LLaPor's
window is a full layer; APEX's is within one layer. So the LLaPor-vs-APEX gap is not comparable
across the seam either.

**Open and cheap:** does residency actually change inside that window during a real forward?
Structurally open (live tensor reference), empirically unverified. Watching one `expert_to_slot`
across a forward settles it. For pre-seam-comparable numbers, run at a commit before `c48b3c69e5`.

The plan's Decisions section still says "a residency-agnostic bank" — superseded by C1; update it.

### The `miss_rows` seam at `0671e34440`

Hot-cache telemetry now separates physical host rows from logical misses. The production 13.6-20.5%
miss rate and the 65-99 demand-miss rows/token in the plan both came from the **old** semantics
(`expert_hot_cache.py`, `unique_missed` → `row["miss_rows"]`). Treat pre- and post-migration numbers
as two series; never average across.

### Selection: live is batch-shared, offline is per-row

`PrefetchCandidateBank.write` ranks on `expert_scores.sum(dim=0)` — one list for the whole batch.
Offline masks then truncates per row. Measured cost at width 16: 0.0016 (LLaPor), 0.0004 (APEX) —
negligible at BS1, but grows with batch size. (This session once described it as per-row with the
file open. Read the line.)

### Regime

Anything measured on the capture or in T6 is at ~3.2-3.4 misses/layer. Production is 1.36-2.06.
State the regime next to every number.

---

## 7. Remaining work, in order

Each item marked **[user]** needs AskUserQuestion before starting.

1. **[user] Commit the plan corrections** (`git diff` on the plan doc first; commit with `-- <path>`).
2. **[user] Decide `MOE_EXPERT_TRANSFER.md`**: location (root vs `docs/`) and commit. Before
   committing, fix row size to 2,764,808, update the state commit, and re-derive the five corrections
   by reading the cited code (they are not written down anywhere else — treat the doc's Python line
   refs as unverified, as its own Notes say).
3. **[user] The doorbell merge** (`da3e9297be`, `d4ad7fb8ba`): its brief claims two tests pass that
   fail. Decide whether to amend the record before anything is pushed.
4. **[user] Push to `shared`** and sync the divix01 worktree (it is at `d725c54a31`). Required before
   any live run, since the launcher serves that worktree.
5. **Settle the residency-window question** (section 6) — a small instrumented forward, GPU but short.
   Coordinate with crypto-c9; take the lock.
6. **[user, production-down window] Stage D matrix** from side-stream plan §11:
   A (current planner) / B (fused) / C (fused + shadow scoring) / **D (fused + scoring + one-row
   side-stream pull)**, with separate LLaPor and APEX C/D arms. Arm D is the first real tok/s A/B of
   prediction. Budget: the pull has one `DedicatedPrefetchSlot` row per layer (`serving/candidates.py:62-76`),
   so delivered rows per forward look capped at 1 regardless of `PREFETCH_BUDGET` — read from the
   code, not measured; confirm from `PullDeliveryStats` before comparing against the B=1 table rows. Protocol as section 4e/5: interleaved, reversed pass 2, baselines rerun
   same-session, GPU census recorded, answer-level gate.
   **By which path is undecided [user]:** measuring delivery through the doorbell inherits E37's
   attribution problem; the in-graph side-stream pull (`SGLANG_MOE_EXPERT_PREFETCH_PULL=1`) is
   reviewed but has **never run on a live server**, and the fused path has no live BS1 capture test.
   Keep doorbell off (Stage E).
7. **Measure in production's regime.** Either run D at production's hot-cache size
   (`HOT_GPU_MB=10240` per the live script) and report the observed miss rate, or state that the
   result is at a different regime. Without this, question 2 of section 1 stays open.
8. **Write up** a new dated experiment record, and update the plan's Task B2 (it still describes the
   old 4-cell prefetch × doorbell matrix; §11 supersedes it). **Do not relaunch production**; tell the
   user the GPU is free.

**Acceptance (from plans):** repeatable net decode improvement beyond run-to-run spread (B2 target:
≥5% median over its P0 counterpart), no answer-level correctness failure, labelled preliminary at
2 passes/cell. Present 42-51% delivery share only as the size of the target.

---

## 8. The failure mode to guard against

Nine times in one day, a check passed because its precondition had silently failed: a test that
passed only when run alone, a gate that couldn't fail at n=1, a metric keeping its name while its
population changed, a lock file with no holder. The shared error: **asserting the conclusion the
evidence makes available rather than the one it makes necessary.**

The defence is to make **"I failed to observe" a third outcome**, distinct from pass and fail:
print `sglang.__file__` before trusting a test, check the scoring log line is *present*, record the
GPU census, qualify every count with its file and every recall with its commit and miss regime, and
check magnitudes before accepting a mechanism (E37's module-load hypothesis for `DOORBELL_ROW_MS` was
off by ~1,900x and had to be withdrawn).

Coordinate with **crypto-c9** (cross-session) on any GPU time and any file in Stage C's set
(`expert_prediction/serving/*`, `expert_hot_cache.py`, `environ.py`, `model_runner.py`). It is
rigorous and will accept a better control over its own — and expects the same in return.
