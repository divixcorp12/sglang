# DeepSeek V4.1 / DSpark CPU experts: complete investigation handoff

Written 2026-10-07 for continuation in a new session. This document contains the
working state, numerical evidence, trace catalog, reproduction recipes, known
limitations, and next leads. Binary traces remain in the directories below;
they are not committed to Git. No new measurement was run while writing this
handoff.

## Start here

The original 1.71 ms calibration number and the roughly 0.6 ms/lane decode
statistic measure different workloads and intervals. That mismatch is real,
but does not by itself prove a kernel regression. We subsequently established:

1. Torch's bundled OpenMP runtime cached a restricted CPU count during startup.
   Two ten-thread expert teams then triggered its oversubscription policy.
   A diagnostic bootstrap that initializes the runtime under CPUs 0–63 removed
   the millisecond post-work barrier tails in measured serving captures.
2. The remaining slow mixed-row forwards really spend CPU time doing unevenly
   assigned matrix work. One worker can receive twice the tile-row work.
   A default-off row-weighted assignment experiment improved the reconstructed
   mixed-row native forward by 22–23%, bit-exactly. Live serving benefit is
   still unmeasured.
3. Astra reproduced two polling defects: a missed control notification can
   delay stop/detach until the hold expires, or indefinitely for infinite hold;
   and the idle sleep predicate omits draft readiness. **Neither is fixed.**
4. A valid counters-off, steady-decode Intel PT trace visibly contains 14.74 ms
   in `keep_warm_either`, followed by a 5.78 ms draft forward. The hold begins
   before the retained trace window. There is no measured GPU-publication
   timestamp in this trace, so we cannot conclude that a ready job waited for
   the whole hold. It does not show recursive calls to `keep_warm_either`.

The original approximately 90 ms group-1 stall remains unexplained and has
not recurred in the later captures. Do not claim that any one finding above
explains it.

Recommended sequence: fix the two reproduced polling defects with bounded
regression tests; measure publication → observation → all-worker release;
then test row weighting in live serving with counters off; finally retune
calibration against realistic routed shapes and complete layer latency.
These are recommendations, not changes made in this handoff.

## Current checkout and scope

| Item | Exact location/state |
|---|---|
| Laptop working checkout | `/Users/dnikolaidis/.codex/worktrees/draft-arrival-magic/sglang-nvfp4` |
| Branch | `codex/dsv41-draft-arrival-magic` |
| Code/results HEAD before this documentation commit | `dceca241707b3a3614dd455d076f0fff1dd89ce6` |
| Remote host | `divix01` |
| Remote private checkout | `/data/models/slang/nvfp4-work/wt-stall-sampler-20261007` |
| Remote shared Git clone | `/data/models/slang/sglang` |
| Remote interpreter | `/data/models/slang/.venv/bin/python` |
| Compiler | `/opt/rh/gcc-toolset-15/root/usr/bin/g++` (15.2.1) |
| Laptop Intel PT artifacts | `/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007` |
| Laptop Nsight/CPU measurements | `/Users/dnikolaidis/Downloads/dsv41-nsight-20261007` |

The laptop worktree was clean before this document; remote tracked files were
also clean at the same HEAD. Remote untracked `generations.json` and temporary
row-image fixture directories are existing harness state: preserve them.
The original checkout `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4` has a
separate uncommitted `CLAUDE.md` change from skill work. Do not overwrite it.
No owned serving/capture process was found in the handoff-time process check.
Do a fresh GPU/lock check before launching anything.

Important constraints carried forward:

- User explicitly declined moving 15 GiB and extra workers from node 1 to node 0.
  The measured later configurations retained **40 GiB / 40 GiB** expert memory
  and ten workers per node; the assumed 60/40 arrangement was not the actual
  measured allocation. No NUMA redistribution was applied.
- No production polling fix, barrier change, fairness policy, or wait-policy
  change was applied. Broad OpenMP initialization is a diagnostic launch path.
- Row weighting defaults off. The native benchmark CMake flag does not enable
  it in full-stack/JIT serving. Live integration remains work to do.
- Keep arithmetic and accumulation order bit-exact. Do not remove barriers
  that protect phase dependencies simply because they appear in a trace.
- Do not change `model_runner.py` as part of this investigation.

> 🧠 **From Hindsight memory (Conventions and patterns)** — the approved shared
> team intentionally prioritizes waiting target jobs and executes target and
> draft work sequentially; validation should preserve bit-exact outputs and
> evaluate complete layer latency and tokens/s. Optional diagnostics should be
> bounded and compile out of normal builds.

The source and current experiment follow that design. The curated Hindsight
page “Row-weighted CPU expert assignment experiment” was stale when checked:
it described a larger joint CPU/GPU split search rather than the implemented
tile-assignment test. A correction titled **“Correction: Row-weighted assignment
experiment scope”** was ingested. Use source and saved artifacts below as the
authority for what actually ran.

## Original problem and calibration semantics

User's original attachment:
`/Users/dnikolaidis/.codex/attachments/d71414ae-f274-415f-9a01-d2ba518d2e84/Pasted text.txt`.

```text
[2026-10-07 00:23:18] CPU experts calibration: row 0, expert 12.7 MiB, 10 reps
  cpu  ms k=1..8: 1.71 3.45 5.15 7.45 11.03 10.71 13.60 14.41
  link ms m=1..8: 0.97 1.94 2.90 3.87 4.83 5.80 6.77 7.73
  layer ms n=1..8 at chosen k: 0.97 1.94 1.94 2.91 3.87 3.88 4.84 5.74
  split n=0..8: 0 0 0 1 1 1 2 2 3
```

The user remembered 0.3–0.45 ms for one expert; decode summaries later showed
roughly 0.658–0.683 ms/lane. Source explains why these are not comparable:

- `host/split_calibration.h::calibration_run` starts before claim/submit and
  waits for CPU completion / DMA query. Calibration warms up and averages ten
  repetitions. With `tokens > 1`, its calibration routes send all six verify
  rows through each selected expert.
- `python/sglang/srt/layers/moe/cpu_experts/service.py::log_stats` subtracts
  calibration counters and prints `forward_ns / lanes / 1e6`. This is a
  cumulative forward-time-per-lane statistic across variable job shapes.
  It excludes request detection and is neither a single-token expert timing
  nor publication-to-completion latency.
- Captured target jobs had about **1.17–1.20 routes per lane**, rather than six.
  Chunking matters as well: the measured later build uses `CHUNK_M=2`, so six
  rows for one expert become three chunks.

A historical forced-higher-CPU B test was reported as median 95.05 → 93.35
ms/token, mean 106.31 → 106.63, acceptance 3.588 → 3.572. That was not a clear
end-to-end win. These are prior reported numbers, not a rerun performed for
this handoff; do not use them as a matched baseline for the new tile experiment.

DSpark's backbone has serial layer dependencies. Parallel draft positions do
not make arbitrary layers independent. CPU expert requests are issued for
layer work; dependent computation requires their outputs. The target gather
path posts CPU work and expert copies, waits for required completions, then
runs routed GPU work and moves through dependent layers. Improving an isolated
expert rate therefore need not improve token rate if it changes overlap or
puts a different operation on the critical path.

Historical production context is in
`docs/superpowers/handoffs/2026-10-07-dsv41-dspark-both-cpu-in-production.md`:
master `473691871d` was launched around 00:13 with target and draft CPU clients,
RTX 5090 EXL3, 10840 MiB GPU hot cache, memory fraction 0.78, DSpark block size 5.
Its eight-session comparison was baseline median 88.30 ms/token, mean 113.5,
versus both-CPU median 95.05, mean 106.3, acceptance 3.588. Promotion was made
knowing that median was slower. **That document's running-server state is
historical, not the current state.**

## Investigation results, in order

All rates below are from their own instrumentation/configuration. They are not
a matched throughput series. Warm-up, TTFT, memory pressure, trace overhead,
and formal gate status vary. Times labeled CPU are thread CPU time, not wall
time. Quantiles in current analyzers use nearest-rank semantics.

### 1. Initial Nsight and reusable completion tracing

Root laptop report: `dsv41-nsight-20261007/findings.md`.
The first higher-CPU capture used one 260-token prompt / 128-token output.
CPU summaries were about 0.730 / 0.640 ms/lane. Graph variants with 8 / 5 tokens
showed mean `lease_stream` 0.397 / 0.810 ms, copy wait 0.424 / 0.835 ms, and
`exl3_moe` 0.256 / 0.258 ms. These kernel elapsed times include dependency
waiting and are not additive exclusive work durations.

The PCIe metric recipe lacked a CUDA device context. It does **not** establish
PCIe bandwidth or saturation. Deep C++ sampling stacks were incomplete;
roughly 2.9% leaf samples were unknown. This motivated compile-time job,
copy-retirement, worker-phase, and hold tracing.

Instrumented capture (`instrumented/`) joined 1080 layers and 2151 NUMA group
events: 987 CPU-later-than-DMA, 79 CPU-only, 13 DMA-later, one ambiguous.
CPU was last, including CPU-only, for **1066/1080 = 98.7%**. This is completion
ordering, not a percentage of end-to-end time attributable to CPU work.
Group 0/1 jobs: 2490/2535; lanes: 6824/6643; routes: 7996/7922.
Wall ms/lane: 0.667/0.975; job p95: 5.425/8.826 ms; queue p95: 2.842/18.121 ms.

The alarming group-1 example was row 20, sequence 12049: **96.683706 ms**,
five lanes, seven routes, six rows, queue wait only 0.003987 ms. Nsight showed
the leader on CPU 18 absent for **90.664700 ms** (13.807400211 → 13.898064911 s),
with all nine workers having 89.340–91.544 ms gaps. No overlapping OS runtime
blocked stack explained it. Thread state was Unknown and external competitors
were not visible. Memory, scheduling, profiler interference, and system pauses
were still candidates. This is the original stall that later runs did not
reproduce.

### 2. Memory/scheduler trace and no-Nsight control

`stall-diagnostic/` recorded heavy memory activity:

| Interval | Swap out | Direct scan | Compaction stalls |
|---|---:|---:|---:|
| Startup | 2,354,727 pages, ~8.98 GiB | 459,599 pages | 8,316 |
| 13.535 s decode bracket | 369,432 pages, ~1.41 GiB | 239,180 pages | 134 |

Decode also had 4,249,944 kswapd scan pages and 1404 global major faults.
Node 0 free memory was 0.84 GiB versus node 1's 12.17 GiB despite system
MemAvailable 45.96 GiB. But **all 68 recorded direct-reclaim beginnings during
decode belonged to NsightAnalysisService TIDs 104928/105170**. The profiler
perturbed the run. Engine leader group 1 had three major faults; group 0 had
none. No OOM counter increase was seen; this is not a complete kernel OOM audit.
Longest reproduced group-1 leader gap was 5.78009 ms, sleeping; at least one
worker continued real matrix work during the gap. Warm-up filled job buffers,
so this capture cannot attribute those stalls to timed jobs.

`no-nsys-stall/` removed Nsight: 128 tokens at 16.361 tokens/s, TTFT 9.963 s,
26 verify steps, acceptance 4.923. Target jobs 2516/2587; wall ms/lane
0.625/0.595; p95 5.112/4.687 ms; maxima 11.858/10.837 ms. All job drops were
zero. No major faults among 23 engine threads during the request bracket;
zero swap out, direct reclaim, or kswapd scan; one global compaction stall.
Existing occupied swap (~15.61 GiB total, ~3.47 GiB server) was not active
paging evidence. Seven startup group-1 jobs had 206 major faults before this
bracket. The bracket includes prefill/TTFT, not solely steady decode.
The 90 ms event did not recur.

### 3. Per-worker work, barriers, and scheduler competitors

`worker-stall/` retained 770 completed forwards >=6 ms, 170 in the measured
request bracket, with no dropped records. Examples:

- Group 1 row 33 seq 23306: forward 11.484659 ms, six rows, ten lanes,
  eleven routes. Worker 2/CPU 20 had gate/up units 576 versus 288, wall
  7.287751 ms / CPU 6.727205 ms; others took 3.066–3.229 ms wall. Down units
  640 versus 320, wall 3.665 / CPU 3.366 ms. Much of the wait was real math
  still running on the heavy worker.
- Row 22 seq 24244: 13.376304 ms forward; gate worker 3 wall 8.176 ms,
  CPU 7.665 ms.
- Separate late-team-entry examples were 2.654491 ms (row 29 seq 21954)
  and 1.919862 ms (row 37 seq 24200).
- Post-all-work barrier tails were 3.172406 ms (row 24 seq 24030) and
  3.842989 ms (draft stage 0 seq 372), with little thread CPU time.

No engine faults/reclaim during the timed bracket explained those tails.
Its 6.993 tokens/s versus the previous 16.361 is not a clean performance
comparison: tracing and memory conditions changed.

The `cpu-route-measure/` native matrix separated routed row count:

| One expert routed rows | Node 0 median ms | Node 1 median ms |
|---|---:|---:|
| 1 | 0.504 | 0.520 |
| 2 | 0.664 | 0.637 |
| 3 | 1.177 | 1.131 |
| 4 | 1.212 | 1.142 |
| 5 | 1.837 | 1.695 |
| 6 | 1.834 | 1.764 |

Rotating weights, ten threads; repeated-weight controls gave one-row
0.464/0.478 ms and six-row 1.757/1.654 ms. Six rows used three `CHUNK_M=2`
chunks. Mixed eleven-route/ten-expert cases moved the heavy chunk front,
middle, and last: last-worker work remained ~1.818 times mean on both nodes.
The overload followed assignment shape, not a permanently bad CPU.
Default, passive/zero-spin, and ACTIVE/GOMP_SPINCOUNT=100000 sweeps were short,
nonrandom policy order, and showed no consistent winner. No production policy
was changed.

Valid scheduler replay `cpu-route-scheduler/replay-pty/` covered 600 forwards,
360879 ftrace records, 53927 target switches, 26888 wakes, zero dropped records.
Concrete tails:

- Forward 488, worker 4/CPU 22: 1.867 ms tail, 1.855 ms runnable, overlapping
  cAdvisor process 6597 / TID 9844.
- Forward 20, worker 7/CPU 25: 1.536 ms tail, 1.520 ms runnable, overlapping
  RayDashboardAgent process 2365045 / TID 2365594.
- Late entry 1.1395 ms included 1.1096 ms runnable.

Pinning a worker does not reserve its CPU; other processes had affinity 0–71.
The first `replay/` attempt was invalid because buffered readiness output
allowed replay after the trace window; `replay-pty/` fixed the collector gate.

`cpu-route-isolation/` temporarily moved 202 cAdvisor/selected DashboardAgent
threads off expert physical CPUs and their SMT siblings. Two 600-forward
replays restored **202/202** original affinity masks. Median benefit was not
consistent; barrier maxima 1.651/2.302 ms remained versus 1.867 ms reference.
Other RayEventHead, Postgres, and transient Python competitors remained.
No permanent service affinity change was made; 90 ms did not recur.

### 4. OpenMP startup budget: confirmed cause of barrier tails

`threading-model/` found Torch's actual bundled `libgomp`, SHA256:
`e28fb2896a3d612b46a27d7f8ef840d34c9efbc29b70a044c42095fb94856890`.

This binary caches available CPUs during initialization. With one CPU at
initialization it cached one; the serving main mask (0–5,36–41) cached twelve;
two ten-thread teams produced a managed-thread count of nineteen and triggered
oversubscription throttling. Initializing under CPUs 0–63 cached sixty-four and
avoided throttling. Widening affinity after loading the runtime changed public
`omp_get_num_procs()` but did not update the private cached budget.
Normal spin was 300000 iterations versus 100 throttled; ACTIVE/INFINITE still
used its throttled path (1000), so changing only those environment values did
not solve the cached-budget problem.

Read-only offsets used for this exact binary were managed 0x237040, cached CPUs
0x237048, normal spin 0x2373c8, throttled spin 0x2373d8. These are diagnostic
ELF offsets, not a portable supported production interface. The helper checks
the SHA before reading them.

`engine-handoff/` used the actual engine, two ten-worker teams, fixed NUMA
first-touch, five fresh arms, 1880 total/1630 nonwarm jobs, bit-exact, no drops.
The twelve-CPU startup arm throttled all 320 short-gap jobs; broad64 throttled
none. Maximum post-work barrier tail was 1833.752 → 54.661 µs with profiling,
589.559 → 19.543 µs without. Group-1 six-row/eight-expert p95 worst barrier was
116.050 → 6.553 µs. A specific 1833 µs tail contained 1829 µs off CPU: 8.931 µs
sleep plus 1820.262 µs runnable. 1206/1280 server-mask barrier release intervals
overlapped sleeping versus zero broad64. Hold final joins were p95 ~1–2 µs,
max ~10–12 µs. Whole forward speedups were inconsistent, up to ~9% on group 1;
no token-rate claim follows.

`omp-serving/` then preloaded libgomp under 0–63 and restored main affinity
0–5,36–41 **before model allocation**, including fresh-child bootstrap.
All 121 runtime samples showed nineteen managed / sixty-four available,
without throttling. In selected >=6 ms forwards, post-work barrier p95 fell
from 986.3/1926.6 µs to 12.6/9.9 µs, maxima from 3843/3172 to 16.9/12.1 µs;
>1 ms tails fell from 5/10 to zero/zero. Long remaining group-1 matrix phases
spent almost all wall time on CPU. Warm-up overflowed job buffers, so this run
cannot attribute all timed jobs. Its outer driver also hit a zombie/proc-maps
shutdown issue, subsequently fixed in `c0c5d703d0`; trace evidence remains usable.

### 5. Scratch/all-job serving capture: remaining work imbalance

`scratch-serving/` is the strongest full-job join after broad initialization.
Runtime budget: all 59 samples nineteen/sixty-four. **5117 timed forwards,
all matched, zero drops**: group 0 target 2482 plus draft 49; group 1 target 2586.

| Interval | Group 0 p95 / max µs | Group 1 p95 / max µs |
|---|---:|---:|
| Scratch preparation, no resize | 5.31 / 25.73 | 8.49 / 22.28 |
| Post-all-work barrier tail | 9.28 / 251.73 | 9.68 / 30.39 |
| Start after engine available | 3.14 / 342.88 | 5.20 / 25.69 |

Raw queue p95 was 2.338/2.082 ms, with 384/287 waits >1 ms, but none remained
>1 ms after the prior job ended. Most large queue waits here were occupied
engine time rather than an idle polling failure.

Group-1 forward 10511, row 0, six rows / ten chunks / eleven routes: 9.874 ms.
Gate phase 6.390 ms: worker 0 had 576 units versus 288, wall 6.388 / CPU
6.339 ms; others 2.752–2.856 ms. Down phase 3.272 ms: worker 0 had 640 units
versus 320, wall 3.270 ms; others 1.380–1.422 ms. Work-unit/CPU correlations
were 0.9996/0.9998, maximum/mean units 1.818. Critical workers in the ten
longest matrix phases had zero measured off-CPU time.

Residual lead: equal units do not guarantee equal time. Forward 10546 workers
0 and 5 each had 516 gate units but CPU times 6.309 versus 4.054 ms.
Cache, frequency, placement, and finer shape cost remain unmeasured explanations.

Of 1080 layer completions, 979 CPU-later, 82 CPU-only, 19 DMA-later.
No engine faults; zero global swap out/direct scan; 274 swap-in pages,
33724 kswapd scans, 33 compaction stalls; node free 0.835/7.495 GiB.
Reported 16.576 tokens/s is one instrumented run, not an established speedup.
**Formal harness verdict FAILED because bootstrap displaced private Python
first in PYTHONPATH.** The trace completed and joins are usable. Launcher order
was fixed afterward; do not rewrite this run as a passed harness or claim a
successful rerun. Scheduler-only report has no CUDA timeline/call stacks or
hardware memory counters.

## Latest row-weighted experiment: implementation and complete result

Implementation `d68fc51fd9`; ownership-test correction/measured HEAD
`33261d8880d87c362473b9a48575b9c845edc74d`; recorded findings `dceca24170`.

`python/sglang/kernels/jit/csrc/exl3/optimized/tile_assignment.hpp` retains the
old mapping and introduces row-weighted aligned partition boundaries.
Each aligned output tile group costs `chunk.m`; adjacent workers use the same
rounded boundary. Gate/up uses `c.chunks[j/2].m`; down uses `c.chunks[j].m`.
The original Unit=2/8 alignment and exactly-once ownership are preserved.
Uniform row counts fall back to the original mapping, including whole-GEMV
striding for large batches. Arithmetic and route accumulation are unchanged.

`EXL3_MOE_CPU_ROW_WEIGHTED_ASSIGNMENT` defaults to **0**. The bare native
benchmark exposes CMake `-DEXL3_ROW_WEIGHTED_ASSIGNMENT=ON`. Full-stack targets
and production JIT do not automatically inherit that CMake option. A live B
test must deliberately wire the macro into its JIT variant/cache digest.

Runner: `benchmarks/dsv41_baseline/run_row_weighted_experiment.py`.
Sixteen serial processes, four alternating AB/BA pairs per node. Ten workers
on CPUs 6–15 / 18–27, respective `numactl --membind`. Eight real 3-bit EXL3
layers, H=5120/I=2304, twelve separately allocated expert slots per layer,
~1.2 GiB working set; five fixture identities repeat at distinct addresses.
Synthetic activations and reconstructed route multiplicities, not captured
activations/routes. Each case has 64 warmups +128 timed forwards. Seven cases,
14336 timed forwards total. **141.1 seconds excluding builds.** All optional
instrumentation counters compiled out. Nodes ran separately, no GPU work.

Fixture `/data/models/exl3_exp/selected_followup/dsv41-eight-layers-unswizzled.bin`:
SHA256 `55de978d5eb3c9d1bc3ac38b5a7937ea724a47fea2c0cf87c04bd1616e188a8c`.
Original frozen refs `/data/models/exl3_exp/threading/reference-e{1,3,5}.bin`.

Times below are median of four per-process mean complete-forward times, ms:

| Routing case | Node 0 A | Node 0 B | Change | Node 1 A | Node 1 B | Change |
|---|---:|---:|---:|---:|---:|---:|
| 6 rows, 10 chunks, heavy first, 11 routes | 5.8926 | 4.5833 | -22.22% | 5.6032 | 4.3371 | -22.60% |
| Same, heavy middle | 5.8687 | 4.5661 | -22.20% | 5.5373 | 4.2992 | -22.36% |
| Same, heavy last | 5.8779 | 4.5322 | -22.89% | 5.5700 | 4.3371 | -22.13% |
| 5 rows, k=3, seeded random | 6.2755 | 5.9988 | -4.41% | 6.0164 | 5.7537 | -4.37% |
| 6 rows, k=2, shared1 | 5.0988 | 5.0949 | -0.08% | 4.8165 | 4.7960 | -0.43% |
| 6 rows, k=2, shared2 | 3.4629 | 3.4546 | -0.24% | 3.4310 | 3.3273 | -3.02% |
| 1 row, k=3 | 1.2996 | 1.3032 | +0.28% | 1.2017 | 1.2049 | +0.27% |

Primary mixed-case p95: node 0 6.1009 → 4.7460 ms, node 1
5.8485 → 4.5093 ms. Uniform controls' p50 changes were all <1%; do not
interpret their mean noise as a weighting gain. One-row p95 rose 1.3–1.5%,
p99 1.5–3.1%; short-run tail equivalence is not proven.

Static assignment calculation for the primary case:

| Phase | Original units | Weighted units |
|---|---|---|
| Gate/up | worker 0 576; others 288 | 316–318 per worker |
| Down | worker 0 640; others 320 | 352 per worker |

This table is static ownership accounting, not a new measured phase trace.
Every process passed 24 frozen output checks and seven new routed cross-arm
reference checks before and after timing, bit-exact, finite and repeatable.
No frozen references were modified. Focused ownership pytest passed with GCC15;
a local native Clang check also passed. Tests cover counts 0/1/2/10/17/65/257,
workers 1/3/10/16/64, rows 1–8, both split units, bounds/alignment/exact ownership,
and uniform mapping preservation. Empty callbacks when workers exceed groups
are existing valid behavior, retained by the test.

Results directory contains `summary.json`, `metadata.json`, sixteen raw
Google Benchmark JSON/log pairs, seven routed refs and build provenance.
`remote-copy-verification.json` reports 49 remote/laptop files with identical
SHA256. A binary SHA256:
`0c0352c5de61dccbc0c4fc073162533791d91c1298f873ec45685e32a6710e84`;
B: `b4617e7ee4bba27a9f38a033c6d7e44a9c91e1173f91b04e931d2b0e70575c06`.

## Astra run-loop audit: two reproduced defects, still unfixed

User requested a `gpt-6-astra` agent, run at xhigh. Audited HEAD
`1756407149393e7493921ebe452771c7b378c0a8`; relevant headers are unchanged in
the later row-weighted/code HEAD. Full report and native probe sources/results:
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/row-weighted-experiment/polling-audit.md`
and sibling `polling-probes/`.

### Control state checked before doorbell snapshot

In `host/cpu_experts.h::run`, around lines 408–418:

```cpp
while (!stop_) {
    // Reads draft attachment / detach control state here.
    // Existing test hook can pause here.
    const auto kick = doorbell_.load();
    // Read head, pop work, or hold using kick and observed head.
}
```

Interleaving: consumer checks old control state; producer publishes stop,
detach, or attach and rings; consumer snapshots the **new kick** with the
**old control state**. Its subsequent hold sees no change. A finite hold waits
until deadline; infinite hold may never return. The idle watchdog does not
rescue the reproduced completed-request state `head == done == 1`.

Bounded CPU-only probes, fake engines, no GPU/model, warmed `instr` host modules,
forced 30 ms pause between control check and snapshot:

| Operation | Ordinary | Forced, 100 ms hold | Forced infinite hold |
|---|---:|---:|---|
| Draft-only stop | 19.506514 ms | 109.351995 ms | blocked at 1.5 s, killed |
| Shared draft detach/stop | 1.144814 ms | 100.287610 ms | blocked at 1.5 s, killed |

Draft-only stop includes its normal up-to-20 ms watchdog join. Attach has the
same source-level ordering defect but was **not** runtime-reproduced.
Fix direction: snapshot the notification before **all** associated state
checks, including the outer stop test, then recheck state as appropriate.
Moving the snapshot only above the draft pointer read is insufficient.

Probe results preserve an initial shared setup failure. The successful shared
probe explicitly sets `ops._DEFAULT_VARIANT = "instr"`; do not silently merge
the failed setup result with the valid reproduction.

### Idle sleep omits draft readiness

Current predicate around line 447:

```cpp
doorbell_.sleep_unless([this] {
    return !jobs_.empty() || stop_.load(std::memory_order_acquire);
}, 50'000);
```

If the CPU reads an old draft head, GPU publishes a new head, and the CPU then
enters this wait, already-ready draft work does not satisfy the predicate.
Using the actual Doorbell header, all twelve ready-before-wait trials slept
**83.445–120.688 µs**. Checking head readiness returned in **67–113 ns**.
This reproduces avoidable sleep, not end-to-end GPU arrival latency.
A post arriving after the final predicate can still wait for timeout because
GPU head publication cannot directly futex-wake the CPU; adding readiness does
not remove every polling delay.

An older capture contained 147958 matched nominal-50 µs waits, median
104.586 µs, maximum 1797.944 µs. It includes startup/idle, not a causal
steady-decode delay distribution.

### Additional source-level windows and intended policy

- Target submit publishes queue entry then rings (around lines 268–269).
  A producer preempted between them can leave ready work while held workers
  watch unchanged kick/head words. This window is not runtime-attributed to
  the 15 ms hold. Existing `cpu_submit` marker precedes actual queue publication;
  measure publication and ring separately.
- Each loop pops target jobs before serving observed draft work. Target backlog
  can delay draft under the intentional priority policy, with no fairness
  budget. Serving starvation has not been demonstrated.
- Hold and forward create separate OpenMP parallel regions (`keep_warm.hpp`,
  `team.hpp`). Worker pinning is cached; the team is not one persistent parallel
  region. Workers poll two words and do a sink update after heating. The leader
  may perform an additional spin interval and then the 50 µs sleep.
- The ordinary fully attached sequence protocol has no obvious lost-update
  defect found in this audit: GPU publication fence/release, acquire head reads,
  one outstanding sequence, zero skipped, completion before `draft_next`.
  That is source inspection, not proof of ideal latency.

`polling-probes/` includes `control_event_probe.py`,
`shared_control_event_probe.py`, result JSONs, `doorbell_ready_probe.cpp/jsonl`,
`SHA256SUMS`, and a packaged `replay_control_events.py`. Six original source/
result files were hash-verified against remote originals. The packaged replay
driver was added locally afterward and **was not run**. Remote originals are
`/data/models/slang/nvfp4-work/astra-poll-audit-20261007/`; copy/commit the driver
deliberately before using it remotely. Prefer turning these cases into proper
bounded regression tests before applying a fix.

## Intel PT / magic-trace: valid steady decode and its limits

Strongest trace:
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/model-steady-decode/draft-delay.fxt.gz`.
Remote directory `/data/models/slang/nvfp4-work/draft-magic-steady-decode-20261007`.
Source HEAD `1756407149393e7493921ebe452771c7b378c0a8`.

Formal warmup passed at round ten; last rates 11.5741225/11.7819782 tokens/s,
zero pending JIT, stable clock gate. Trace arm was **47.938877675 seconds after
first content**, with 345 tokens /100 stream updates (minimums 30 s/256/100).
Snapshot came 1.576548989 s later, 49.530939012 s after first content, around
351–355 streamed tokens. This is not slow model startup.

Metrics were compiled out; independent trigger-only mode retained a forward
threshold of 4000 µs. Sequence 653, reason 1, native elapsed **5.779382 ms**.
91,355 decoded FXT events, 4 MiB Intel PT AUX payload, one PT overflow,
21.372876 ms retained span. Captured only group-0 leader TID 1003930, owning
scheduler PID 1000485: no group-1 team, other workers, or GPU publication marker.

`keep-warm-analysis.json` gives:

| Decoded interval | Duration |
|---|---:|
| Initial visible `keep_warm_either` wrapper, 2350 → 14744186 ns | 14.741836 ms |
| Final worker-clone/GOMP end → wrapper return | 0.659 µs |
| `serve_draft`, 14744298 → 20549569 ns | 5.805271 ms |
| Actual forward, 14745582 → 20524071 ns | 5.778489 ms |
| Next visible hold | 0.354222 ms |

**`inferred_start_time=true` for the first hold.** It began before the ring
window, so 14.74 ms is visible retained occupancy, not a directly observed
function-entry-to-return interval or a fixed 15 ms timeout. Same-symbol active
nesting depth is one: wrapper → namespaced helper → OpenMP clone appearances
are not recursive invocations. There are 5511 clock reads in the initial hold.
Its final join is short, but that does not bound notification arrival to first
notice or all-workers-ready. We lack the timestamp needed to tell how much of
the visible hold preceded the request.

Earlier **instrumented warm-up** trace `model-capture/` had 91,928 FXT events,
five PT overflows, trigger seq64 ~5.045 ms. For 47 sequences 54–100, zero drops:

| CPU interval | p50 µs | p95 µs | max µs |
|---|---:|---:|---:|
| CPU head observation → forward start | 2.419 | 3.808 | 4.677 |
| Forward | 2337.714 | 4709.067 | 6461.751 |
| Final hold join, 46 holds | 0.770 | 2.376 | 3.294 |
| Worker exit spread | 0.225 | 1.860 | 2.140 |
| Hold end → CPU observation | 0.750 | 2.062 | 8.739 |

Seq64: hold 8.808557 ms, join 0.663 µs, exit spread 0.244 µs,
hold end→observe 1.025 µs, observe→start 2.394 µs, forward 5.045252 ms.
GPU/CPU clock anchors shifted **720.9305 µs**, with nonoverlapping offset
intervals. The analyzer correctly refuses their intersection. Its illustrative
anchor-envelope arrival values are **not certified latency/error bounds**.
Do not use this dataset to claim precise or bounded GPU-arrival latency.

`model-trigger-only/` was metrics-off but health/warmup diagnostic, not steady;
native trigger ~5.044085 ms. `failed-worker-capture/`, `failed-trigger-capture/`,
and `failed-steady-state/` are retained failed attempts, not valid timelines.

Trace SHA256:
`3b6bc8680a22246f84777fe01ae474af4f794ec8bc7fbe4ab6e1e3f2c4e24d79`;
raw `magic-work/perf.data`:
`0bff8b387fd51a5b49d10d18ccbb57a8078779dcc3b109df3c7a8ad56f2e36e5`.
Both hashes were recomputed from the laptop files while writing this handoff.
Saved verification manifests also record the original remote/local comparison.

### Viewing and tool gotchas

Open **https://magic-trace.org/** and load `draft-delay.fxt.gz`. It is the
Perfetto-based magic-trace viewer; W/S zoom, A/D pan. Capture runs on the Linux
Intel PT host, not the laptop. `.nsys-rep` files open in NVIDIA Nsight Systems.

Reusable skill: `tools/skills/magic-trace/SKILL.md`, also installed at
`/Users/dnikolaidis/.codex/skills/magic-trace/SKILL.md`. It contains standalone
setup, capture, trigger/address resolution and trace verification scripts;
SGLang-specific notes are in repo `CLAUDE.md`. Read it before another capture.
Pinned tested tool: `/data/models/slang/nvfp4-work/tools/magic-trace-v1.2.4`.
Tested collector hardware: dual Xeon Gold 6154 (Intel Skylake, Intel PT),
72 logical CPUs. Download recipe for the same release:

```bash
mkdir -p "$HOME/.local/bin"
curl -fL --retry 3 \
  https://github.com/janestreet/magic-trace/releases/download/v1.2.4/magic-trace \
  -o "$HOME/.local/bin/magic-trace-v1.2.4"
chmod 755 "$HOME/.local/bin/magic-trace-v1.2.4"
```

- Normal optimized builds work with `sglang_draft_delay_trigger`, a noinline
  function containing volatile asm. **Noinline cannot preserve a call site
  removed by `if constexpr(kMetrics)`.** The separate production trigger DSO
  uses `SGLANG_DRAFT_DELAY_TRIGGER_ONLY=1` and production clocks, no event
  buffers, one gate check per completed forward, fires once.
- Counters-off mode only supports the forward trigger. Set arrival and pending
  thresholds to zero. For arrival timing, explicitly use instrumented mode.
- Multiple loaded JIT DSOs export the same symbol. Resolve the DSO actually
  owning the engine, not the first `nm` hit. `addr:` needs executable-relative
  PIE relocation semantics; raw ASLR addresses can double-apply the bias.
- OMP workers inherit `exl3-cpu-exp0` names. The oldest matching owned TID was
  the leader, verified with ownership manifests/affinity, not name alone.
- Linux perf 6.12 needed `MAGIC_TRACE_PERF_PATH` pointing at
  `benchmarks/dsv41_baseline/magic_trace_perf_compat.py` for trace-start alias
  and compatible invocation. The skill has the portable adapter.
- magic-trace v1.2.4 decodes native C++ trigger arguments using OCaml tagged
  integer conventions. `.passed_val` can be halved and its timestamp nonsense.
  Use the typed native trigger report JSON, not `hits.sexp` for elapsed ns.
- Exit zero / “Snapshot taken” is insufficient. Failed captures were only
  87–88-byte FXTs with zero events and no PT AUX payload. Validate event count,
  AUX bytes, trigger identity, window, overflows and local-copy hashes.
- Four MiB is a history capacity, not a guaranteed lookback duration. Busy
  control flow fills it faster. Initial inferred frames do not prove when a
  function began or whether work was already queued.
- Health diagnostic captures only warmup; `--magic-steady-decode` retains
  formal warmup and the long SSE decode admission gate.

## Code map and reusable instrumentation

Paths below are relative to the exact checkout identified at the top.

| File / symbol | Responsibility / next inspection |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h` | `BasicCpuExpertEngine::run`, submit, attach/detach/stop, `serve_draft`; CPU request head read and compact/route preparation |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/draft_kernels.cuh` | GPU request publication, system fence / head publication; inspect around 94,108–109 |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/keep_warm.hpp` | Warm hold and two-word polling; separate OpenMP region, sink update |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/team.hpp` | Forward team entry and phase execution/barriers, cached worker pinning |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/job_trace.h` | Bounded job/resource trace, independent trigger and typed trigger report |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/trace_gate.h` | Gate file for tracing only the requested bracket |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/hold_trace.hpp` | Hold observations, worker release/final join; compile-time guarded |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/split_calibration.h` | Current CPU/link calibration and split cost assumptions |
| `python/sglang/kernels/jit/csrc/exl3/optimized/forward_plan.hpp` | Chunked gate/up/down work, worker phases and scratch observations |
| `python/sglang/kernels/jit/csrc/exl3/optimized/tile_assignment.hpp` | Original / default-off weighted mapping |
| `python/sglang/srt/layers/moe/cpu_experts/service.py` | Engine setup, stats units, trace export |
| `benchmarks/dsv41_baseline/run_omp_serving_capture.py` | Broad-runtime bootstrap, owned runtime sampling, magic/Nsight capture orchestration |
| `benchmarks/dsv41_baseline/run_stall_capture.py` | Short serving capture and sampler lifecycle |
| `benchmarks/dsv41_baseline/stall_sampler.py` | Process/thread resource and memory sampler |
| `benchmarks/dsv41_baseline/analyze_scratch_serving.py` | Complete job/phase/scratch joins |
| `benchmarks/dsv41_baseline/analyze_draft_arrival.py` | Request arrival joins and clock-calibration validity checks |
| `test/manual/dsv41/omp_bootstrap/sitecustomize.py` | Diagnostic broad-affinity runtime preload and exact-binary private offsets |
| `python/sglang/test/dsv41_ram_miss_fixtures.py` | `warm_host_modules` / `spawn_child` for bounded focused native tests |

`draft_start` originally follows request read and compaction. It excludes
detection/preparation. Later instrumentation added observation/preparation
markers, but publication-to-observation still requires a valid cross-device
clock mapping. Avoid conflating this with hold duration.

Diagnostics use compile-time variants rather than always-on counters. Relevant
switches include `EXL3_MOE_CPU_WORKER_TRACE`, `SGLANG_CPU_EXPERT_HOLD_TRACE`,
`SGLANG_DRAFT_DELAY_TRIGGER_ONLY`; runtime destinations include
`SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX`, `SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY`,
`SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE`, `SGLANG_CPU_EXPERT_HOLD_TRACE_PREFIX`,
`SGLANG_CPU_EXPERT_HOLD_TRACE_CAPACITY`, `SGLANG_EXL3_CPU_SCRATCH_TRACE_PREFIX`,
`SGLANG_CPU_EXPERT_TRACE_GATE`, and `SGLANG_DRAFT_DELAY_TRIGGER_REPORT_PREFIX`.
Use the committed capture drivers to construct the full variant/build flags;
setting a runtime prefix cannot turn on a compiled-out collector. Preserve
drop counts and bracket metadata in analysis; never substitute warm-up records
for a timed interval whose buffer overflowed.

## Artifact catalog

For the first table, prepend laptop root
`/Users/dnikolaidis/Downloads/dsv41-nsight-20261007/` to the relative folder.
Remote paths are under `/data/models/slang/nvfp4-work/` unless stated otherwise.
Directory `findings.md` and capture-command/validation JSONs contain exact
commands and underlying records, not just these summaries.

| Laptop folder/files | Remote directory / retained evidence | Quality / purpose |
|---|---|---|
| Root `higher-cpu-nsys-s2-20261007-023427.nsys-rep`, `...-pcie.nsys-rep`, `.sqlite`, `findings.md`, `decode-kernel-totals.json`, `cpu-thread1.json` | Original higher-CPU capture; exact launch recorded in root findings | Initial CUDA/C++ timeline; PCIe metric collection ineffective |
| `instrumented/` | See folder capture metadata for original run paths | Job/copy completion joins; original ~90 ms group-1 gap; `memory-stall-findings.md` |
| `stall-diagnostic/` | `stall-capture-20261007` | Scheduler/memory trace, profiler reclaim; warm-up job overflow |
| `no-nsys-stall/` | `no-nsys-stall-20261007` | Job/resource sampling without Nsight, no timed reclaim/engine major faults |
| `worker-stall/` | `worker-stall-20261007` | Matrix worker skew plus late entry/barrier tails |
| `cpu-route-measure/` | `cpu-route-measure-20261007-r2` | Native row/chunk cost matrix and wait-policy sweep |
| `cpu-route-scheduler/replay-pty/` | `cpu-route-scheduler-20261007/replay-pty` | Valid scheduler replay; root `replay/` attempt invalid |
| `cpu-route-isolation/replay-isolated-r2/`, `replay-isolated-r3/` | `cpu-route-isolation-20261007` | Temporary affinity experiment, full restoration journals; `comparison.json` |
| `threading-model/` | Provenance in local findings | Bundled-libgomp CPU-budget probes and source/binary audit |
| `engine-handoff/` | Provenance in local findings | Actual-engine broad-budget replay, barrier/hold joins and validation |
| `omp-serving/` | `omp-serving-20261007-broad1` | Broad-budget live capture; job overflow, outer shutdown error |
| `scratch-serving/` | `scratch-serving-20261007-b1` | All-job complete phase/scratch/resource joins; formal Python-path gate failed |

Useful Nsight reports include:

- `stall-diagnostic/stall-cpu-s2-20261007-040932.nsys-rep` and its `-pcie` report.
- `stall-diagnostic/stall-warmup-20261007-0424.nsys-rep`: despite its name,
  overlaps the timed request; supplemental failed ftrace acquisition must not
  be confused with the primary trace.
- `cpu-route-scheduler/replay-pty/scheduler.nsys-rep`.
- `cpu-route-isolation/replay-isolated-r2/scheduler.nsys-rep` and r3 equivalent.
- `scratch-serving/scheduler.nsys-rep` (~63 MiB); exported SQLite ~949 MiB.

For the next table, prepend laptop root
`/Users/dnikolaidis/Downloads/dsv41-magic-trace-20261007/`.

| Laptop folder | Remote path / important contents | Interpretation |
|---|---|---|
| `model-steady-decode/` | `draft-magic-steady-decode-20261007`; `draft-delay.fxt.gz`, `magic-work/perf.data`, `capture-summary.json`, `capture-command.json`, `magic-command.json`, `trace-verification.json`, `local-verification.json`, readiness/arm JSON, progress logs, `keep-warm-analysis.json` | Accepted counters-off steady-decode capture |
| `model-capture/` | Instrumented original capture; exact path in `capture-command.json`; job/hold/worker records, start/end draft clocks, `prior-arrival-hold-analysis.json` | Valid warm-up PT, invalid clock intersection for precise arrival timing |
| `model-trigger-only/` | `draft-magic-trigger-only-20261007` | Valid metrics-off health diagnostic, not steady |
| `failed-worker-capture/`, `failed-trigger-capture/`, `failed-steady-state/` | Failure logs and verification records | Retained failures, do not analyze as accepted timelines |
| `trigger-smoke/`, `skill-validation/`, `skill-real-trace-verification.json` | Native main/worker/fallback trigger tests | Tool/skill validation, not model performance |
| `row-weighted-experiment/results-01/` | `row-weighted-20261007/results-01`, sibling `build-a/`, `build-b/` | Complete native A/B result/provenance/hashes |
| `row-weighted-experiment/polling-audit.md`, `polling-probes/` | `astra-poll-audit-20261007` | Read-only Astra findings, deterministic probe sources/results |
| `row-weighted-experiment/findings.md` | Copy of earlier committed row-weighted handoff | Concise prior experiment report |

Large raw artifacts are already on the laptop; do not download them again
without a reason. To transfer a new trace, copy artifacts only and compare
remote/local SHA256. Keep both the decoded FXT and raw perf data, command logs,
trigger report and verification metadata. Never copy a source tree by scp.

## Reproduction and safe launch recipes

Read `.claude/rules/divix01-run-protocol.md` and repo `CLAUDE.md` first.
Laptop changes → explicit-path commit → push origin → fetch/fast-forward the
private remote checkout. No master push, amend, rebase, force, stash, cleanup of
other work, or bulk staging. These scripts may use existing untracked benchmark
registration: preserve `benchmarks/dsv41_baseline/generations.json`.

Every CPU process must be capped to CPUs **0–63**; 64–71 are reserved for NVMe
IRQs. Limit OpenMP and build parallelism. GPU jobs obey `cc-gpu.lock`; row-image
disk lock is acquired before GPU lock. `run_arm.sh` takes the GPU lock itself;
do not hold it around a parent driver and deadlock its child. Use fresh output
directories. Runtime JIT can take 50–100 seconds per module; warm serving launch
~5.5 minutes, cold ~10–15 minutes. Prewarm exact variants before short child
timeouts. Store large traces on disk/`/mnt/nvme1`, not a RAM-backed `/tmp`.

Always make the private checkout's Python tree first and print its provenance;
the venv can otherwise import unrelated `main-port-probe-7bc4eb` code:

```bash
cd /data/models/slang/nvfp4-work/wt-stall-sampler-20261007
taskset -c 0-63 env OMP_NUM_THREADS=4 PYTHONPATH="$PWD/python" \
  /data/models/slang/.venv/bin/python -c 'import sglang; print(sglang.__file__)'
```

### Focused ownership test and native B rebuild

```bash
taskset -c 0-63 env OMP_NUM_THREADS=4 \
  CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ PYTHONPATH="$PWD/python" \
  /data/models/slang/.venv/bin/python -m pytest -q \
  test/registered/unit/kernels/test_exl3_row_weighted_assignment.py
```

To rebuild, choose new build paths rather than overwriting retained binaries:

```bash
bench_src=python/sglang/kernels/jit/csrc/moe/expert_stream/bench
new_build=/data/models/slang/nvfp4-work/row-weighted-NEW/build-b
taskset -c 0-63 env OMP_NUM_THREADS=4 cmake -S "$bench_src" -B "$new_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
  -DEXL3_TORCH_ROOT=/data/models/slang/.venv/lib/python3.13/site-packages/torch \
  -DEXL3_CXX11_ABI=1 \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK=/data/models/slang/nvfp4-work/cpubench-build/_deps/google_benchmark-src \
  -DEXL3_ROW_WEIGHTED_ASSIGNMENT=ON
taskset -c 0-63 env OMP_NUM_THREADS=4 \
  cmake --build "$new_build" --target exl3_cpu_optimized -j2
```

A uses the same recipe with separate build-a and weighting OFF. Existing
retained binaries can reproduce the original experiment directly:

```bash
taskset -c 0-63 env OMP_NUM_THREADS=10 PYTHONPATH="$PWD/python" \
  /data/models/slang/.venv/bin/python \
  benchmarks/dsv41_baseline/run_row_weighted_experiment.py \
  --a=/data/models/slang/nvfp4-work/row-weighted-20261007/build-a/exl3_cpu_optimized \
  --b=/data/models/slang/nvfp4-work/row-weighted-20261007/build-b/exl3_cpu_optimized \
  --output=/data/models/slang/nvfp4-work/row-weighted-NEW/results-01
```

Defaults reproduce four rounds /128 iterations. Inspect metadata/hash/flags
before treating a rebuilt result as the same code. There is no need for a
suite-wide pytest run for this documentation or this isolated experiment.

### Repeat counters-off steady-decode capture

This is a recipe, not an authorization or claim that it was rerun now.
Reference capture preserves the established higher-CPU B split, model paths,
40/40 GiB allocation, ten workers, hot cache 10840, memory fraction .78,
block size 5, ACT_RESIDUAL=1 / ACT_BLOCK=128, victim lanes 8, static ragged
layer-major path, retune batches zero. Exact reference:
`/data/models/slang/nvfp4-work/scratch-serving-20261007-b1/capture-command.json`.
Draft model `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-dspark-draft`;
resident map under the same project `analysis/dsv41-dspark/cpu-draft-routes/resident-top32.json`.

The fixed higher-CPU split (n=0..40) was:

```text
0,1,1,2,3,3,4,4,5,6,6,7,8,8,9,9,10,11,11,12,12,13,14,14,15,16,16,17,17,18,19,19,20,20,21,22,22,23,24,24,25
```

```bash
taskset -c 0-63 env OMP_NUM_THREADS=10 PYTHONPATH="$PWD/python" \
  MAGIC_TRACE_PERF_PATH="$PWD/benchmarks/dsv41_baseline/magic_trace_perf_compat.py" \
  /data/models/slang/.venv/bin/python \
  benchmarks/dsv41_baseline/run_omp_serving_capture.py \
  --reference=/data/models/slang/nvfp4-work/scratch-serving-20261007-b1/capture-command.json \
  --output=/data/models/slang/nvfp4-work/draft-magic-steady-decode-NEW \
  --seed-build=/data/models/slang/nvfp4-work/scratch-serving-20261007-b1/exl3-build \
  --port=30035 --seconds=30 \
  --magic-trace=/data/models/slang/nvfp4-work/tools/magic-trace-v1.2.4 \
  --no-instrumentation --magic-steady-decode \
  --draft-arrival-trigger-us=0 --draft-pending-trigger-us=0 \
  --draft-forward-trigger-us=4000
```

Inspect driver and saved reference for SHA/registration compatibility when
running newer code. It manages bootstrap and owned process selection; do not
manually reuse a prior PID/TID/address. Arrival measurement requires removing
`--no-instrumentation` and deliberately choosing thresholds/clock validation.

Generic trace validation from the skill:

```bash
/data/models/slang/.venv/bin/python tools/skills/magic-trace/scripts/verify_trace.py \
  /absolute/path/draft-delay.fxt.gz \
  --perf-data /absolute/path/magic-work/perf.data \
  --log /absolute/path/magic.log --output /absolute/path/NEW-verification.json
```

Use actual log filenames from the capture; the placeholders above are not
assertions that every output directory uses `magic.log`.

Nsight notes: node-level CUDA graph tracing added roughly 0.77 µs/node in an
earlier graph (~8153 nodes), about 6 ms host tail /7 ms-token disturbance.
Graph-level mode omits graph-body kernels from some tables, and previous
copy-engine graph capture combinations hung. Follow current harness guards,
not a blanket mode recommendation. Host `cudaMemcpy` duration is not DMA
transfer time: join correlation IDs and bytes. Use a short scheduler-only
capture for CPU scheduling questions. Root metrics wrapper is
`sudo -n /usr/local/sbin/nsys-profile`; actual remote version used was 2026.3.2.
Keep analysis memory bounded (e.g. DuckDB 8 GiB) on remote disk; previous
laptop/RAM-filesystem analysis exhausted memory.

## Next leads and concrete acceptance criteria

1. **Fix only the two reproduced polling defects first.** Add deterministic
   stop/detach tests for finite/infinite hold and bounded cleanup; cover attach
   control order; use the real ready-before-sleep predicate test. Warm modules
   outside child deadlines. Preserve target priority and current CPU share.
   Record exact before/after behavior. Source-level queue→ring delay is a
   separate measurement, not a proven third fix.
2. **Measure arrival and worker transition in steady decode.** Record GPU
   publication, CPU first observation, hold first notice, every worker release,
   forward start/end, queue publication/ring. Cover both groups if possible.
   Preserve valid clock uncertainty intervals; if anchors fail, report failure
   rather than invent a midpoint bound. A short final join is not sufficient.
   Check whether the visible hold was simply idle before publication, delayed
   notification, a late worker, intentional target priority, or extra heating.
3. **Live row-weighted B against a fixed-polling baseline.** Deliberately add a
   JIT variant/cache key for the macro; keep default off. Same CPU/GPU split,
   NUMA budgets, worker/core allocation and OpenMP startup budget. Verify
   bit-exact outputs, then compare counters-off tokens/s, acceptance, complete
   layer latency/p95/tails. Standalone 22% is enough to justify this test, not
   enough to claim a production gain.
4. **Then redesign calibration.** Use realistic per-expert routed row counts
   and CHUNK_M shapes, sustained warm state and rotating weights, both groups
   concurrently with realistic copies/GPU work. Optimize complete layer
   completion/critical path, not isolated CPU ms or assumed PCIe sums. Include
   scheduling/detection overhead, uncertainty and tail behavior. Dense six-row
   calibration is useful as one case, not the sole runtime cost proxy.

Other unresolved leads: original 90 ms all-team gap; equal-work/different-CPU
cost example; nonreserved expert CPUs and other competing services; node-0
memory pressure and profiler-induced reclaim. Intel PT shows control flow,
not memory-stall attribution. No current evidence proves faulty recursion or
a systematic 15 ms “cannot exit” loop while a draft is ready.

## Commit trail and adjacent documents

| Commit | Meaning |
|---|---|
| `1cf2697ff8` | Request preparation timing and Intel PT slow-draft trigger |
| `05eca2ef8f` | Optimized trigger validation / arrival timing documentation |
| `0da713d15f`, `f083b8218b`, `94dc88235d` | Perf alias, PIE relocation, bracketed clocks and publication trigger coverage |
| `28dccb3bd8` | Larger PT history |
| `c17d68c8ff` | Arrival join validation, continue decode after timeout |
| `f0e302eed5` | Health-ready diagnostic, explicitly no benchmark verdict |
| `11785eefcb`, `e0a750be67` | Correct leader, reject empty captures, owning JIT DSO / worker tests |
| `8eaf304f7d` | Portable skill and independent production trigger |
| `d25f412144` | Native C++ argument ABI caveat |
| `1756407149` | Sustained steady-decode capture gate |
| `d68fc51fd9` | Default-off weighted tile assignment |
| `33261d8880` | Preserve original empty callback behavior in ownership checks |
| `dceca24170` | Row-weighted findings and Astra probe reproduction record |

Earlier detailed reports remain useful for exact intermediate commands:

- `docs/superpowers/handoffs/2026-10-07-dsv41-dspark-both-cpu-in-production.md`:
  historical production merge/launch, prefill memory and benchmark caveats.
- `docs/superpowers/handoffs/2026-10-07-dsv41-row-weighted-assignment-and-polling-audit.md`:
  concise latest native experiment/audit.
- Canonical portable tool skill `tools/skills/magic-trace/`; installed skill
  `/Users/dnikolaidis/.codex/skills/magic-trace/`.

Suggested new-session prompt:

> Read `docs/superpowers/handoffs/2026-10-07-dsv41-cpu-stalls-complete-session-handoff.md`
> on branch `codex/dsv41-draft-arrival-magic`. Inspect the saved Astra polling
> probes and current run loop. Continue from the prioritized next steps; keep
> the 40/40 GiB placement and row weighting default off. Do not treat the
> 14.74 ms inferred-start hold as proven queued-request delay. Report a concrete
> plan for deterministic polling fixes and a valid steady-decode arrival capture
> before running another long experiment.
