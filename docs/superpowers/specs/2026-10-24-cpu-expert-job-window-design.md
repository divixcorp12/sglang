# CPU expert job window: a task-graph engine with overlapping jobs

Owner-approved design, 2026-10-24. Branch `cpu-job-window`, cut from `exl3-globals` (77838f55f8).

## Goal

The CPU expert engine stops running one job at a time on the whole team. Each job is split into phase tasks; the
team's workers take the oldest ready task from a window of admitted jobs, with no team barriers. It must be
batch-ready: bigger records (up to the token table's 32 tokens), several records in flight, and hit, miss, staged and
draft jobs interleaved, with no single-job slowdown.

## Evidence this rests on

From the production DSpark trace `divix01:/mnt/nvme1/lm-lend/a32k-prod/prod` (2026-10-18, both engines, about 46,000
target jobs):

- 16-19% of target jobs are submitted while the engine is busy. Their wait has a median of 1.4-1.5 ms; in total it is
  6.5-8.4 s against 47-49 s of compute.
- 98% of that wait is a record's 1-expert miss job (part 1) behind the same record's hit job (part 0).
- The gap between back-to-back jobs has a median under 1 us. Job time grows linearly with expert count from about
  zero, so a job's tail and barriers cost a few us.
- Submit order is sequence order. Draft and target work never wait at the same time: none of 1,073 draft records
  overlaps a target job's submit-to-end span.

So at batch size 1 the window gains little; the owner chose to build it for batching. Multi-token gains come from
tokens sharing an expert inside one job (bench: 258 us per (token, expert) pair at 1x1, 99 us when 6 tokens share 6
experts). The kernel already groups them by expert into chunks; the window never holds a job back to merge it with
another.

## Decisions

- D1. Approach B, a task graph, over a pipelined team (A) or split sub-teams (C).
- D2. EXL3 and NVFP4 are both ported in this feature.
- D3. `Kernel::forward`, `Team::run` and `Team::barrier` are removed. Tests, the bench and `forward_alone` move to the
  job interface; `Team` keeps the pinned workers, their names and the keep-warm idle wait.
- D4. Admission is target ring first, then the draft channel. An always-on counter records admissions that found both
  sources pending; oldest-first across sources is the change to make if it ever counts.
- D5. EXL3's Down is two phases again, with the base's split: DownTiles (tile pairs, any owner; the worker that
  finishes a (chunk, 128-output block) transforms it) and DownOut (the weighted sum, 16-output units over every chunk
  in chunk order). The fused Down was 4-6% slower (40 blocks over 16 workers) and was reverted; a DownOut of 128-output
  blocks that also transformed measured the same way (experts:1 1.045x) and was replaced (2026-10-24).
- D6. The split model's fit records worker time: each claimed run adds its TSC duration to its slot, and the fit takes
  the sum divided by the workers.
- D7. Window size W = 4.
- D8. No batching delay: a job is admitted as soon as it is popped.

## Rules

- EXL3 stays bit-exact on all four tiers (vnni, bw, avx2, scalar) against the current baseline. NVFP4 stays within
  one Q8_0 quantum.
- A task's arithmetic does not depend on which worker runs it. Every reduction that fixes a summation order stays
  inside one task.
- `done_` never passes an unfinished sequence.
- Kernel `run` does not throw. `prepare` may throw `std::invalid_argument`, which the engine turns into `fail_stop`.
- No new environment variables or macro switches.

## Non-goals

- Raising the 8-token chunk limit (CHUNK_M = MAX_M / 2). A 32-token record routing many tokens to one expert decodes
  its tiles once per 8 tokens; that is kernel work for a follow-up.
- Lifting `--max-running-requests 1` or the batch-size-1 decode graph. Serving stays as it is.
- The 256-bit VNNI experiment. Counters (`/mnt/nvme1/exl3-tidy/perf1`) show the decode is port-0 bound (0.86-0.89
  uops/cycle on one core, port 5 at 0.75-0.77, every cycle at the AVX-512 heavy license); at 256 bits port 5 would
  likely bind instead.

## Design

### 1. Architecture

In `moe/expert_stream/host/cpu_experts/`:

- `Team`: the pinned pool. It keeps pinning, thread names and the warm/PAUSE idle wait, and loses `run` and `barrier`.
- `JobWindow`: a ring of W slots. A slot holds its job's sequence and kind (target or draft); the kernel's per-job
  state `Kernel::Job`, with its own scratch arena (today's `thread_local ForwardArena` becomes per slot); the job's
  token table (today's `token_slots_`/`token_weights_`); per phase a next-task counter, a done-task counter and a
  ready flag; its predecessor slot on the same output; its stage-two gate; and its worker-time sum.
- `Scheduler`: the loop every worker runs.
  1. Scan the slots oldest first and take the first with a ready phase.
  2. Claim a run of its tasks with one `fetch_add`.
  3. Run the run; add its TSC duration to the slot's worker time; add the run's length to the phase's done count.
  4. The worker that completes a phase's last task makes the next phase ready or, after the last phase, completes
     the job.
  5. With nothing ready, idle on the window's word (warm, then PAUSE). Admissions, phase transitions and gate openings
     move that word.
- The engine thread is worker 0 and the admitter. Between claims it pops the target ring and the draft channel and
  runs a job's serial setup into a free slot: token expansion, the instr build's `check`, and the kernel's `prepare`.

### 2. Kernel task interface

`binding.hpp`'s `CpuExpertKernel` concept replaces `forward` with:

```cpp
using Job = ...;                     // one slot's state; reused across jobs, so its scratch stops growing
static constexpr int kPhases;
static constexpr int kGatePhase;     // first phase that reads stage-two rows; -1: none
static void prepare(Job&, const Layer&, const ForwardCall&);  // serial, at admit
static Tasks tasks(const Job&, int phase);   // {count, align}, read when the phase becomes ready
static void run(Job&, int phase, int t0, int t1, int worker); // tasks [t0, t1)
static int status(const Job&);       // 0, or 2 when the input refused
```

- `tasks` is read when a phase becomes ready, so a phase may report 0 tasks (NVFP4 after its Q8_0 check fails).
- `worker` indexes per-worker scratch only.
- Each phase function keeps today's loop body, its `worker_share(total, worker, n)` range replaced by `[t0, t1)`.

EXL3 (one expert, DSV4.1 counts):

| # | phase | one task | count |
|---|---|---|---|
| 0 | PrepareGateUp | (chunk, gate or up, 128-input block): transform and quantize | 80 |
| 1 | GateUp | (gate/up GEMV, tile pair), align `PlanTraits::kTileAlign` (DSV4.1 1, generic 4) | 144 |
| 2 | Middle (`kGatePhase`) | (chunk, 128-block): SiLU and quantize | 18 |
| 3 | DownTiles | (chunk, tile pair) of the down GEMV, into `tout_d`; a block's last tile transforms it | 160 |
| 4 | DownOut | 16-output unit: per chunk in order, weighted add into `out` | 320 |

- DownOut zeroes its own columns first for `kOverwrite`, replacing worker 0's memset at job start, with the same
  bits. `kOverwriteByStore` keeps its first-write store and unrouted-row zeroing, per unit.
- Align 8 pairs (16 tiles) on DSV4.1 measured experts:1 at 1.57x the static split: 18 runs of GateUp's 144 pairs over
  16 workers. The DSV4.1 compact kernels take any pair; the generic tile kernels need 8-tile groups.
- `assign_gemvs`, `assign_row_weighted_gemvs`, `claim_groups`, `PlanTraits::kSplitTiles`/`kClaimMixedChunks` and the
  `row_weighted` param are removed. The grouped multi-band traversal stays, inside a claimed run.

NVFP4: PrepareInput (32-value blocks), GateUp (16-row units), Middle (blocks), Down (16-row units). `partial` is per
slot, `[workers][stride]`, indexed by `worker`. After `invalid` is set, Middle and Down report 0 tasks.

Claiming: a run is `ceil(count / workers)` tasks rounded up to `align`, at most what remains: the static split's
share as one contiguous range, which a worker that finishes early follows with the next. Runs of
`max(align, remaining / (2 * workers))` shrank toward single tasks and starved EXL3's band traversal (experts:3 at
1.15x the static split, 54% of its time in single-band kernels against the base's 73% in `traversal_kblock`).

### 3. Scheduling and dependencies

- Window: W slots in sequence order. A full window stops admission; worker 0 then runs tasks.
- Picking: oldest slot first, so the oldest job always gets the workers first.
- Same output: at admit, a job whose `(row, part)` an earlier active slot writes gets that slot as its predecessor;
  its DownOut becomes ready only once the predecessor's DownOut has completed.
- Draft jobs depend only on themselves (their output is the draft's pinned area). Admission follows D4.
- Stage two: a staged job's `kGatePhase` is not made ready until its gate word opens. Workers re-check gate words
  when they find nothing ready. The scheduler records the gate's calls (exactly one per staged job), whether it was
  shut on arrival, and the wait.
- Completion: a finished slot sets `complete[slot]`. The finisher advances `done_` over completed slots in sequence
  order with a CAS loop on the publish word, so each value is published once.
- Stop: no new admissions; admitted slots drain; a shut gate opens on `stop`, as now.

### 4. Engine integration

| today | in the window |
|---|---|
| `busy()`/`idle()`: one job | per slot: kind, id, admit time; the watchdog scans the slots; `stall_state(seq)` reports the slot holding `seq`, its phase and age, and the window |
| `expand_tokens` into the engine's table | into the slot's table, at admit |
| memset then accumulate for a per-token job (store-out off) | removed: the job passes `kOverwrite` and DownOut zeroes its columns |
| `Kernel::check` (instr) | at admit |
| `Kernel::forward`, `fail_stop` on throw | `prepare` throws: `fail_stop`. Status 2 at completion: `fail_stop` naming the row |
| `record_stage_two` | the scheduler's gate stats, same checks |
| `compute_ns_`, `jobs_done_`, `lanes_done_` | at completion |
| `record_fit(part, k, pairs, end - start)` | `record_fit(part, k, pairs, worker_time / workers)` (D6) |
| `cpu_start`/`cpu_end`/`cpu_shape`, fault and switch snapshots | `cpu_start` at admit, `cpu_end` at completion, same fields; snapshots on the engine thread; the instr worker trace records (slot, phase, worker) |
| `done_.store(seq)` | the prefix publication |
| `draft_.serve(record, team)` | `draft_.prepare(record)` builds and admits the call; the draft's completion (channel, stats, trace) runs at slot completion |
| `stop()`: the running job finishes | no admissions, the window drains |

## Testing

- Scheduler, in C++ with the fake kernel (`moe/formats/fake/fake_kernel.cpp`), whose test phases record (job, phase,
  task, worker, time): every task runs exactly once; phases in order within a job; same-`(row, part)` DownOuts in
  sequence order while other outputs interleave; `done_` never passes an unfinished sequence, including when a later
  job finishes first; a staged gate holds only its job; a full window stops admission; `stop()` drains; the watchdog
  fires on a stuck slot; status 2 and a throwing `prepare` fail-stop naming the row. Plus one ThreadSanitizer run.
- EXL3: `test/manual/dsv41/run_exl3_cpu_forward_checks.sh` on all four tiers against the baseline, and both benches'
  validation at the host and bw builds, through a one-job window.
- NVFP4: its forward checks, within one Q8_0 quantum.
- Engine and service: `test_cpu_expert_engines_exl3.py`, `test_cpu_expert_service.py`,
  `test_dspark_shared_cpu_team.py`, `test_dspark_draft_cpu_experts.py`, `test_nvfp4_cpu_experts.py`,
  `test_exl3_ram_miss_cpu_experts.py`, `test_cpu_expert_store_out.py`, `test_exl3_worker_trace.py`, against the base
  run. Nine already fail at the base (six in `test_dspark_shared_cpu_team.py`, three in
  `test_exl3_ram_miss_cpu_experts.py`).
- New engine tests through the FFI: several records in flight with mixed hit, miss and staged jobs, and one 32-token
  record. Their outputs must match the same jobs run one at a time (bit-exact for EXL3).
- Performance (CPU only, no server, 16 workers on 28-43, `bench/ab.sh` against the base):
  - Gate: experts:1/3/5 and the routed rows no slower than the base beyond noise.
  - A new job-stream bench mode: the trace's record pattern (a 3-5 expert hit job, then 1-expert miss jobs as rows
    land, staged misses gated on a timer), several records in flight, and one 32-token record. Reported as record
    completion time (p50, p95) and throughput, against the same driver on the base.
  - Last, a short traced DSpark serve under `cc-gpu.lock` with the smoke prompt: `done` latency, the gate-wait
    counters and the both-pending counter.

## Steps

Each step green before the next; Sonnet implementers, reviewed, with the divix01 runs between steps.

1. `JobWindow` and `Scheduler`, the new concept members, and the fake kernel's job interface, with the scheduler
   tests. `forward` stays for now.
2. EXL3's job interface (the five phases); the bench and the EXL3 tests moved to it; the bit-exact check and the
   single-job A/B.
3. NVFP4's job interface and its tests.
4. The engine and the draft path on the window; `Kernel::forward`, `Team::run` and `Team::barrier` removed; the engine
   and service tests.
5. The job-stream bench mode and the A/Bs; the served check.

## Follow-ups

- Chunks above 8 tokens for batched records.
- The 256-bit VNNI experiment, after the window.
