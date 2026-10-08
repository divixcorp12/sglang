# Persistent CPU expert team

Date: 2026-10-07. Branch `codex/dsv41-persistent-team`, from `codex/dsv41-draft-arrival-magic` at `703953eeb2`.

## Problem

The CPU expert engine (`host/cpu_experts.h`) ran every job on a fresh OpenMP team and held the idle team in a
second OpenMP region (`keep_warm`, `keep_warm_either`), so each job crossed libgomp eight times: the hold's join,
the forward's region start, the four phase barriers inside the forward (`forward_plan.hpp`), the forward's join and
the next hold's start. Since `e7e8479d91` (2026-10-05) the server runs under `taskset -c 0-5,36-41`; torch's bundled
libgomp caches those 12 CPUs at initialization, sees 19 managed threads across the two 10-thread teams, and takes its
throttled path at every one of those crossings: 100 spins, then a futex sleep. Measured on the stall sampler's
per-thread `schedstat` and context-switch counters (`omp-throttle-check-20261007` on divix01): every worker made
700-950 voluntary switches per second while the engine was busy, on-CPU 0.87-0.94, against 0 switches and on-CPU
0.96-0.99 when libgomp was initialized under 0-63. This is what made the expert cores stop showing 100% and what
produced the 1-4 ms barrier tails in the 2026-10-07 stall investigation.

The engine's idle policy also had five states (hold, leader spin, 50 us timed sleep with a draft, untimed sleep
without, released) and two reproduced defects: control state read before the doorbell snapshot could delay stop or
detach until the hold expired, and the timed sleep's predicate omitted the draft head.

## Decision

One `Team` per engine, owned by the engine thread for its lifetime (`host/cpu_experts/team.hpp`):

- `threads - 1` pthreads created at `start()`, each pinned once to `cores[i]` and named like the engine thread; the
  engine thread is worker 0 on `cores[0]`.
- `Team::run(body)` publishes one job through a generation word; `Team::barrier()` is a counting spin barrier the
  quants call where they had `#pragma omp barrier`; the job's end is the same barrier.
- Between jobs the workers run the kernel's register-work loop (`keep_warm.hpp`, unchanged code) for `keep_warm_ns`
  after each job and PAUSE after that, watching the job word. They never sleep and never leave the team.
- The engine thread idles the same way on its submit word, a quantum (`kIdleQuantumNs`, 2 us) at a time, and between
  quanta re-reads the target ring, the draft source word and head (`draft_experts.h`) and `stop_`. Nothing in the
  engine sleeps.

The kernel vtable (`kernel.hpp`) becomes `forward(layer, call, Team&)` plus `warm(word, seen, deadline)`; a
non-virtual `forward(layer, call)` builds a team for the call alone (tests, bench, harnesses). `keep_warm`,
`keep_warm_either`, `run_team`, `CallCores`, the engine's `Doorbell`, `spin_ns`, `release_at`, `idle_budget` and both
`sleep_unless` paths are removed. `SGLANG_DSV41_CPU_EXPERTS_IDLE_SPIN_US` and
`SGLANG_DSV41_DSPARK_CPU_EXPERTS_IDLE_SPIN_US` are removed; `SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US` stays. The RAM
service and copy threads keep their own spin budgets. `hold_trace.hpp` and its test are removed; `worker_trace`
stays. The draft stats lose their hold count (six fields).

Both polling defects disappear structurally: there is no hold snapshot that control state can go stale across, and
there is no sleep predicate.

## Rejected

- **A persistent OpenMP region.** The quants' barriers would still have to become ours, and the engine would stay
  coupled to libgomp's pool and `OMP_*` environment.
- **Initializing libgomp under 0-63 in the launcher.** Proven in `omp-serving-20261007-broad1`; kept as the fallback
  if this branch slips, but it leaves eight libgomp crossings per job.
- **A separate team for the draft.** The draft and the verify are serialized on one GPU stream, so a second team has
  nothing to overlap with, and node 0 has no free physical core for one. Two sources, one executor per node.
- **Parking after a spin budget on our own futex.** Chosen against: dedicated cores, and the park state is where the
  two defects lived.

## Validation

Bit-exactness: phase order and accumulation are unchanged; the frozen references and routed cross-arm references of
the bench, and the engine tests on the instr build's fake kernel, must pass. On divix01, in a private worktree:
`test/registered/unit/kernels/test_cpu_expert*.py test_dspark_*cpu*.py test_cpu_experts_common.py
test_nvfp4_cpu_build.py test_exl3_worker_trace.py test_exl3_ram_miss_stage_trace_causal.py
test_exl3_ram_miss_numa_groups.py test_expert_stream_build_variants.py test_expert_stream_hotpath_golden.py`, then
the CPU-expert files of `test/manual/dsv41` under `cc-gpu.lock`. The production proof is a serving capture with
`stall_sampler.py` under the server mask and no libgomp bootstrap, showing worker voluntary switches near 0 during
decode.
