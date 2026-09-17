# Expert prefetch continuation handoff

## Objective

Use the recorded final-code baseline to start the pipeline changes that attack the measured current-system losses. The original symptom was lower tok/s with prefetch enabled on PCIe Gen3 x16. Correctness validation, a diagnostic Nsight trace, and a reduced timed baseline are now done at `57c842ae6e`. Do not claim a speedup until a change beats matched measurements.

## Authoritative references

- HiCache/production design and the appended `python in hot path` audit: [2026-09-16-hicache-expert-prefetch-optimization-handoff.md](2026-09-16-hicache-expert-prefetch-optimization-handoff.md)
- Measurement protocol and B/C/Cr/N/D arm definitions: [2026-09-16-prefetch-throughput-recovery.md](2026-09-16-prefetch-throughput-recovery.md)
- Earlier operational state/matrix handoff: [2026-09-16-prefetch-plan-completion-handoff.md](2026-09-16-prefetch-plan-completion-handoff.md)
- Persistent decision ledger (rulings, full result table, run directories): `.superpowers/sdd/2026-09-16-hicache-expert-prefetch-optimization-handoff/progress.md`
- Trace launcher and provenance contract: `scripts/expert_prediction/run-shadow-server.sh`; sudo environment relay: `scripts/expert_prediction/trace-env-relay.sh`

## Current code state

Branch: `codex/nvfp4-expert-stream-main`. `shared/codex/nvfp4-expert-stream-main` is at `0283eacb74`. The following are local and on divix01 only (fetched from a git bundle), **not pushed**:

| Commit | Content |
| --- | --- |
| `cdc70383bc` | This handoff (first version). |
| `c27fc6f3f8` | Fix: JIT top-1 kernel used undefined `CUDART_INF_F` and never compiled; plus four CUDA test fixes (capture-writer frame release, sync-test warm-up, budget regex, physical-row assertion). |
| `57c842ae6e` | Launcher relays its environment through a sudo-prefixed `PREFETCH_TRACE_COMMAND` (`sudo` `env_reset` otherwise drops every `SGLANG_*` setting). |

Earlier work in the branch:

| Area | Commits | Result |
| --- | --- | --- |
| No-offer semantics | `ef81a4d426` | Persistent validity; current-forward `id=-1,count=0`; live residency recheck. |
| Planner foundation | `f18b95794e` | Real posted count and optional `[covered,residual,wasted,posted]` outcomes. |
| Production path + Task C | `42a4d268e2`, `7bc343e5a3`, `50f860888f` | Count-safe fused/generic planning, CUDA JIT BS1 fp32 top-1 selector, reference fallback for shadow/calibration/wider diagnostics, physical-row accounting and cache-budget charge. |
| Async metrics/capture | `a9b7ac0b78`, `9128855359`, `3c0f5114bf`, `d879b685f8` | Owned pinned snapshots, event polling, bounded writers, background CPU formatting/I/O, bounded trace admission/order state. |
| Test determinism | `0283eacb74` | Fixes the CPU writer-queue test's scheduling race. |

Important design properties:

- The fast selector is a real JIT CUDA kernel. It is BS1/contiguous/fp32 only and is used only when a pull is actually posted (`pull_mode=always`, logged `serving_top1=True bank_width=1`). Pull modes `off` and `count_zero`, shadow recall, calibration and wider diagnostics use the width-16 reference bank (`serving_top1=False bank_width=16`).
- Planner coverage requires `posted_count == 1`; a stale nonnegative ID with count zero remains a demand route.
- The dedicated speculative row is charged against cache budget and is not a permanent residency slot.
- Optional telemetry does not wait on CUDA, serialize JSON, or write files on the inference path. Required cache-control and doorbell safety synchronization were not removed.

## Validation completed

- CPU: `test_async_telemetry.py` 6 passed on divix01.
- CUDA, 13-file suite from the shadow worktree under the GPU lock: **425 passed, 3 failed** at the fixed code. The three failures also fail at pre-work commit `0fe8d526df` and are not regressions: `test_forward_taps_never_synchronize` (test's own pageable `.to(cuda)` trips sync-debug) and two doorbell tests that are suite-order artifacts. Logs in `cc-expert-prediction/logs/cuda-tests-*.log`.
- Diagnostic trace (LLaPor D, `servers/prefetch-recovery-llapor-trace-d-final/run-20260916-190633`): relay kept configuration; `copy_expert_row_segments_gpu_kernel` is 78.4% of GPU kernel time, `select_prefetch_top1_kernel` 0.1%; GPU metrics captured; stats in `trace/stats/`.

## Baseline results at `57c842ae6e`

Reduced by user direction from the full matrix. Per arm: cold server on 31040, startup mode verified, fixed warm-up, 8 sessions / 29 turns at 768 tokens, fused plan 1, candidates 16, budget 2, hot GPU 10240, calibration 0, not profiled. All arms 29 records, 0 errors. Run directories are listed in the ledger.

| Arm | Median decode tok/s per pass | Mean |
| --- | --- | ---: |
| B (old commit `0fe8d526df`) | 14.176 | 14.18 |
| LLaPor C | 13.384 | 13.38 |
| LLaPor Cr | 13.426, 13.412 | 13.42 |
| LLaPor N | 13.044, 12.897 | 12.97 |
| LLaPor D | 14.263, 14.105, 14.273 | 14.21 |
| APEX D | 13.944 | 13.94 |

What the baseline says about the current system (exploratory):

1. Prefetch is at break-even: LLaPor D ~= old-commit B; APEX D ~1.7% below it (one pass).
2. Scoring is the largest loss: C is ~0.8 tok/s (~5.6%) below B, and D's real pull only recovers it.
3. Empty pull N-C costs ~0.4 tok/s; shadow recall Cr-C is within noise (~0.15 tok/s pass spread).
4. Expert row copies dominate GPU time; selection cost is negligible under the top-1 kernel.
5. Confound: C/Cr/N use the width-16 reference bank, D uses top-1. N-to-D mixes payload and selector effects, and part of C's cost may be the wider bank.

Gaps: no final-code B (D-vs-B rests on the old-commit B), and APEX has only one D pass. Each missing arm is one ~18 min cold run.

## Safety rules

- Do not touch production port 7867. Never stop/relaunch it without explicit user direction. It is currently down.
- `model_runner.py` is frozen. Do not edit it.
- Every GPU run: `/data/models/slang/nvfp4-work/cc-gpu.lock`, before/after GPU census, `taskset -c 0-63`, `OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=32`. Never profile timed arms.
- Do not reset, clean, stash, or alter unrelated `.omc`, untracked docs, `python/uv.lock`, or user files.
- Shadow worktree `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree` is detached at `57c842ae6e`. Deliver new commits to it by push to `shared` (after asking) or by git bundle.

## Next actions

1. Decide with the user whether to push `c27fc6f3f8`/`57c842ae6e` to `shared` and whether to run a final-code B to close the D-vs-B gap.
2. Start the pipeline changes against this baseline, LLaPor first (the ledger ruling treats this baseline as satisfying the Task 3 gate for LLaPor; APEX still needs its own evidence):
   - Reduce scoring cost (the C-B loss), including separating the width-16 bank cost from predictor cost.
   - Task 3: move LLaPor payload launch behind the source layer's demand transfer, measured as paired D-N against the unchanged N control plus trace evidence of no overlap.
3. Benchmark each change with the same per-arm protocol (runner: `cc-expert-prediction/matrix-tail.sh "<predictor> <pass> <arm>" ...` with `MATRIX_DIR` set, or a fresh matrix dir). Use a separate `RUN_KIND=trace` run for traces; a sudo-prefixed trace command is relayed automatically.
4. Optional cleanup on divix01 (ask first): `cc-expert-prediction/baseline-0fe8d526df` worktree, `trace-relay-proto/`, `fix-57c842ae6e*.bundle`, the 3.9 GB trace `report.sqlite`.

## Operational patterns

SSH:

```bash
ssh -n -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes divix01 '<command>'
```

Retry read-only status checks after transient SSH resets. Never use broad `pgrep` patterns that can match the checking shell itself. Stop only PIDs conclusively shown to be self-started shadow processes (the runner SIGTERMs the `sglang serve` child of its own `flock` launcher).

Privileged trace example (tokens only; no quotes or backslashes):

```bash
RUN_KIND=trace PREFETCH_TRACE_COMMAND="sudo -n /usr/local/sbin/nsys-profile profile -f true -o <run_dir>/trace/report --trace=cuda,nvtx,osrt --cuda-graph-trace=node --gpu-metrics-devices=all --delay=240 --duration=240" \
  scripts/expert_prediction/run-shadow-server.sh <name> 31040 off
```

Server startup takes ~200-220 s; choose `--delay` so the capture window opens after health and warm-up. nsys ends the server at the end of `--duration` (launcher rc 137 is expected).
