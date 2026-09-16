# Expert prefetch continuation handoff

## Objective

Finish correctness validation of the production prefetch and nonblocking telemetry work, then resume the shadow throughput matrix. The original symptom was lower tok/s with prefetch enabled on PCIe Gen3 x16. Do not claim a speedup until matched measurements exist.

## Authoritative references

- HiCache/production design and the appended `python in hot path` audit: [2026-09-16-hicache-expert-prefetch-optimization-handoff.md](2026-09-16-hicache-expert-prefetch-optimization-handoff.md)
- Measurement protocol and B/C/Cr/N/D arm definitions: [2026-09-16-prefetch-throughput-recovery.md](2026-09-16-prefetch-throughput-recovery.md)
- Earlier operational state/matrix handoff: [2026-09-16-prefetch-plan-completion-handoff.md](2026-09-16-prefetch-plan-completion-handoff.md)
- Persistent decision ledger: `.superpowers/sdd/2026-09-16-hicache-expert-prefetch-optimization-handoff/progress.md`
- Trace launcher and provenance contract: `scripts/expert_prediction/run-shadow-server.sh`

## Current code state

Branch: `codex/nvfp4-expert-stream-main`, pushed to `shared/codex/nvfp4-expert-stream-main`.

Current required commit: `0283eacb74` (`test: make async telemetry writer queue drop deterministic`). It contains all preceding commits below:

| Area | Commits | Result |
| --- | --- | --- |
| No-offer semantics | `ef81a4d426` | Persistent validity; current-forward `id=-1,count=0`; live residency recheck. |
| Planner foundation | `f18b95794e` | Real posted count and optional `[covered,residual,wasted,posted]` outcomes. |
| Production path + Task C | `42a4d268e2`, `7bc343e5a3`, `50f860888f` | Count-safe fused/generic planning, actual CUDA JIT BS1 fp32 top-1 selector, reference fallback for shadow/calibration/wider diagnostics, physical-row accounting and cache-budget charge. |
| Async metrics/capture | `a9b7ac0b78`, `9128855359`, `3c0f5114bf`, `d879b685f8` | Owned pinned snapshots, event polling, bounded writers, background CPU formatting/I/O, bounded trace admission/order state. |
| Test determinism | `0283eacb74` | Fixes the CPU writer-queue test’s scheduling race. |

Important design properties:

- Fast selector is a real JIT CUDA kernel, not a Python wrapper. It is BS1/contiguous/fp32 only; all other shapes/dtypes and any shadow recall, calibration, or diagnostic consumer use the exact stable top-W reference bank.
- Planner coverage requires `posted_count == 1`; a stale nonnegative ID with count zero remains a demand route.
- Dedicated speculative row is charged against cache budget and is not a permanent residency slot.
- Optional telemetry must not wait on CUDA, serialize JSON, or write files on the inference path. Accepted snapshots are owned; completed snapshots are processed by bounded background writers. Drops are explicit under backpressure.
- Required cache-control and doorbell safety synchronization were not removed.

## Reviews and validation already completed

- Independent reviews found and corrected: stale count not forwarded, tail drain telemetry, cache budget undercharge, phase attribution, writer lock blocking inference, graph-counter unpack mismatch, schedule/close race, unbounded skipped trace state.
- Final review approved bounded trace admission at `d879b685f8`.
- divix01 CPU suite passed:

```bash
cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree
taskset -c 0-63 env OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONPATH=python \
  /data/models/slang/.venv/bin/python -m pytest -q \
  test/registered/unit/layers/moe/test_async_telemetry.py
```

Result: `6 passed` (warnings only).

Local environments lack dependencies for full collection; static `py_compile` and `git diff --check` were run repeatedly. CUDA tests have **not** yet run on the final code.

## Immediate blocker and safety rules

- Do not touch production port 7867. Never stop/relaunch it without explicit user direction.
- Shadow worktree is `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree`, detached at `0283eacb74`.
- Last GPU census: unrelated PID `4024302`, `python entry_with_update.py`, ~3986 MiB. Do not kill it. GPU lock was available but census is not clear, so do not start CUDA tests or a server until it exits.
- Every GPU run must use `/data/models/slang/nvfp4-work/cc-gpu.lock`, a before/after GPU census, `taskset -c 0-63`, `OMP_NUM_THREADS=32`, and `MKL_NUM_THREADS=32`.
- `model_runner.py` is frozen. Do not edit it.
- Do not reset, clean, stash, or alter unrelated `.omc`, untracked docs, `python/uv.lock`, or user files.

## Next actions, in order

1. Wait for a clear GPU census. Confirm both no compute apps and lock availability:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true
```

2. Run CUDA-focused correctness tests from the shadow worktree. Start with telemetry/hot cache, then prefetch lifecycle/planner:

```bash
cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree
taskset -c 0-63 env OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONPATH=python \
  /data/models/slang/.venv/bin/python -m pytest -q \
  test/registered/unit/layers/moe/test_async_telemetry.py \
  test/registered/unit/layers/moe/test_expert_hot_cache.py \
  test/registered/unit/layers/moe/test_expert_prediction_metrics.py \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py \
  test/registered/unit/layers/moe/test_expert_prefetch_runtime.py \
  test/registered/unit/layers/moe/test_expert_prefetch_scoring.py \
  test/registered/unit/layers/moe/test_expert_prefetch_pull.py \
  test/registered/unit/layers/moe/test_expert_gpu_pull.py \
  test/registered/unit/layers/moe/test_expert_route_plan.py \
  test/registered/unit/layers/moe/test_expert_graph_gather.py \
  test/registered/unit/kernels/test_expert_route_plan_fused.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py
```

3. If tests pass, run a short diagnostic Nsight trace before timing. The user configured passwordless metrics access through `sudo -n /usr/local/sbin/nsys-profile`; use it only for a separate `RUN_KIND=trace` diagnostic, never a timed arm. Verify root-owned report-file accessibility after the run. The trace wrapper accepts only simple whitespace-separated tokens; no quotes/backslashes in `PREFETCH_TRACE_COMMAND`.

4. Resume the old matrix on the final commit, from a cold server per arm. The B baseline already completed on old `0fe8d526df` with 29 records and zero error fields. It is a historical baseline, not matched final-code evidence. Run LLaPor pass 1 next: `C -> Cr -> N -> D`; then pass 2 `D -> N -> Cr -> C -> B`; repeat for APEX. Use port 31040 only after each prior shadow process exits and GPU census is clear. Timed arms: fused plan 1, candidates 16, budget 2, hot GPU 10240, shadow recall 0, calibration 0. Do not profile timed arms.

5. Do not implement Task 3 (after-source-demand LLaPor posting), static gate rollout, transfer geometry/cache-hint experiments, promotion reuse, or doorbell changes until the final-code B/C/Cr/N/D evidence is recorded. Task 3 remains hard-gated by that matrix.

## Existing measurement artifacts

- Historical B baseline run: `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/prefetch-recovery-llapor-b-p1/run-20260916-161625`
- It used port 31040, old commit `0fe8d526df`, 29 JSONL result records, zero `error` fields, and was shut down cleanly after a SIGTERM to its verified self-started parent.
- Earlier diagnostic trace: `/data/models/slang/nvfp4-work/cc-expert-prediction/servers/prefetch-recovery-llapor-trace-d/run-20260916-160533`; it was diagnostic-only and predates GPU-metrics authorization. Do not compare it as a performance result.

## Operational SSH pattern

Use exactly:

```bash
ssh -n -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o BatchMode=yes divix01 '<command>'
```

Transient SSH resets occurred previously; retry read-only status checks before interpreting them as job failures. Never use broad `pgrep` patterns that can match the checking shell itself. Stop only PIDs conclusively shown to be self-started shadow processes.

