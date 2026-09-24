# DSV4.1 baseline serving benchmark

A trimmed serving-benchmark harness for DeepSeek-V4.1 Flash (DSV4.1) on divix01,
comparing V2 storage changes one arm at a time, over the real HTTP server
(`sglang serve`), the same way the Qwen3.8 campaign measured its arms.

**Do not run this before reading the SM-clock and noise-floor sections below.**

## Corpus: real text, truncated to a fixed short shape

The current benchmark uses a **32,768-token context and prefix caching**, matching the
saved production launcher. Earlier arms used a 4,096-token context with prefix caching
disabled. The larger context can change KV-pool sizing and the VRAM available for the
hot expert cache (`SGLANG_MOE_HOT_GPU_MB=14336`); prefix caching can also change
prefill work. Treat this as a new benchmark recipe and compare only matched arms.
Historical 2.781 tok/s and V2 storage step breakdowns are useful context, not a
same-configuration baseline. The short corpus below is retained for continuity.

This harness instead reuses the corpus and shape the phase 3a/3b DSV4.1 arms
themselves used: `analysis/dsv41-phase3a/wc-step7.sh` (the command that produced
`corpus-cold.json`) ran `scripts/dsv41/trace_corpus.py --n 8 --skip 0 --prompt-tokens
256 --new-tokens 128` against
`divix01:/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`
(sha256 `249e8a73a32b69aff563471dbae2f4f3a2a9beaa1a3ae5cb03b4c2c549c16c72`), a real
corpus, unchanged. `synthetic_corpus.py` reuses that recipe — the same 8 sessions, the
same 256-token truncation of each one's real first-turn text — but re-decodes the
truncated tokens back to text and wraps each as a one-turn chat session, so it can be
driven over `/v1/chat/completions` (`trace_corpus.py` instead fed raw token ids
straight to an offline `sglang.Engine`, which is not this campaign's serving path).
`run_arm.sh` checks the real corpus's checksum at the start of every run and aborts on
mismatch; it never regenerates or modifies the corpus, and `assert_fits_context`
refuses any prompt+generation that would exceed the configured context before it is
ever sent.

**8 sessions, and the first 4 are exactly `corpus-c.json`'s sessions** (the recorded
2.781 tok/s baseline). The corpus overlap aids workload comparisons, but the new
launch settings prevent a direct throughput comparison with that number. The extra
4 buy statistical power: 8 sessions gives a clean-sweep
sign test p ~= 0.0039 (1/256); 4 alone only reaches p = 0.0625.

A 9th real corpus session (index 8, `fb-financebench_id_04209`, not one of the 8 timed
ones) is the warm-up session — discarded from timing and reused each warm-up round
(see below).

## Server: confirmed live, `sglang.launch_server`

**The DSV4.1 HTTP server is real and has been launched successfully** (2026-09-21,
`divix01:/data/models/slang/nvfp4-work/cc-dsv41-base/analysis/baseline/smoke.sh`):
`/v1/chat/completions` returned 200, and the server log showed `cuda graph: True` at
`#running-req: 1` — the breakable decode CUDA graph, the path every recorded DSV4.1
number describes, not an eager fallback. `arm_env.ServerArgs.argv()` starts from that
smoke launch, with the current production context and cache mode:
`context_length=32768`, prefix caching enabled,
`mem_fraction_static=0.80`, `chunked_prefill_size=512`, `max_prefill_tokens=16384`,
decode CUDA graph `backend=breakable, bs=[1], max_bs=1`, prefill
CUDA graph disabled, `expert_distribution_recorder_mode=per_pass` (the EXL3 gate
refuses dynamic hot caching without it), `disable_shared_experts_fusion`.
**`max_running_requests=1`**, not the 4 in `corpus-c.log`'s Engine dump: decode graphs
exist only at batch size 1, and anything above it runs eagerly and skips the path
under test — the Qwen campaign also used 1, and the driver is sequential, one turn at
a time.

`--reasoning-parser` / `--tool-call-parser` are **not** passed — **verified**, not just
predicted from reading `resolve_chat_encoding_spec`: the smoke launch ran without
either flag and `/v1/chat/completions` returned 200 with correct generation.

The current `arm_env.base_env()` defaults also enable RAM miss leases, GPU residency
updates, DIRECT insert on miss at stage 2, decode cache updates every forward, and
eight RAM miss pack workers. `arm_env(overrides)` can change any of these for an arm.
It also enables Engram graph host-node lookups using io_uring. Async CPU
residency scores remain off because the GPU residency update path rejects them.
The native Engram host-node cache still has a separate 5 GiB budget;
`SGLANG_DSV41_ENGRAM_RAM_GIB=5` controls the Python row cache.
These newer defaults are a new serving recipe; historical throughput cells above
used earlier settings and are not directly comparable.

## Signals that say ready and are not

Three separate systems reported success today while in a state the campaign could not
use. Trust none of them at face value; this is why the gates below exist.

1. **The EXL3 gate validates the wrong format and passes**, unless
   `--expert-distribution-recorder-mode per_pass` is explicit — the Engine happened to
   default it correctly, the server did not, and the failure mode was a launch refusal,
   not a silent bad number, but only because someone was watching the launch.
2. **`/health` returns 200 before compilation has finished.** The smoke log shows a
   34 s Triton kernel compile landing *inside a served request*, after `/health` had
   already returned 200 and after the server printed "ready to roll" — and it happened
   a second time roughly 3 minutes later, for a different shape. A passing health gate
   does not mean the server is ready to be timed.
3. **Clocks-at-rest and clocks-under-load are not the same number, and not in the
   direction a compute-bound intuition predicts.** A single-drive, pre-mirror smoke
   showed decode clocks (~2572 MHz) *below* at-rest clocks (~2947-2970 MHz).
   `nvidia-smi` never says "which state is this"; it just reports whatever
   `clocks.sm` currently is — and see the caveat below before trusting those specific
   numbers going forward.

## JIT compilation: the warm-up must run until the log goes quiet, not for N tokens

Because of finding 2 above, the warm-up is not a fixed-length nicety — it is the part
of the protocol responsible for absorbing serving-time JIT compilation before anything
is timed, and a single occurrence during a timed session invalidates that session.

`run_arm.sh`'s readiness loop (`compile_watch.py`):
1. Runs the warm-up session (256-token prompt, a real 128-token generation — the same
   shape as the timed work, so it exercises the same kernels) in rounds.
2. After each round, scans `server.log` for `took N s to compile after serving
   started` and compares the count to the previous round's.
3. Only proceeds to the timed set once a round adds **zero new** compile events *and*
   the SM clock is stable (see below) — bounded rounds, a clear abort if neither
   happens.
4. Still checks every timed session individually (log byte offset before/after each
   `run_capture_sessions.py` call): if a compile event appears anyway — the second
   smoke occurrence showed this is possible even after a clean warm-up round, for a
   shape not yet seen — the arm **hard-aborts immediately**, naming the session.
5. Every session's compile status is also written to `compile.jsonl`
   (`compiled_during_session`, `compile_events`), and `paired.py` refuses to pair an
   arm carrying any `true` there — a second gate, independent of the hard-abort, in
   case that safety net is ever bypassed.

## SM clock: stability, not a ramp target

An earlier version of this harness gated the warm-up on the clock reaching 0.95 x
`clocks.max.sm`, modeled on a workload that sustains a high clock under load. **Smoke
measurements disproved that premise.** Observed, so nobody re-derives a threshold from
the spec sheet again:

| state | `clocks.sm` |
|---|---:|
| `clocks.max.sm` (spec limit) | 3135 MHz |
| at rest, before/after the probe | ~2947-2970 MHz |
| **during active decode** | **~2572 MHz** |

**These three numbers are from a single-drive, pre-mirror smoke run** (`smoke.sh`
predates `SGLANG_MOE_EXPERT_MIRROR_DIRS` being set), the same regime as the ~2.09
tok/s throughput it also showed. With mirrors on and decode at ~3.92 tok/s
(`DSV41_REFERENCE.md` section 19), the step is materially shorter and nobody has
re-measured whether the clock still dips below rest, by how much, or at all — do not
carry these specific MHz figures forward as the current machine's behavior. What
doesn't depend on re-measuring: a workload whose clock is stimulus-dependent in either
direction should not be gated on a fixed ramp target, and a ramp-fraction gate would
either abort every arm or, loosened enough to pass, certify nothing. What actually
ruins a paired comparison is two arms sitting at *different* points on whatever the
workload's clock behavior turns out to be, not both sitting at the same point.

`run_arm.sh` therefore:
1. Runs the warm-up session (discarded from timing either way) in rounds, sampling
   `clocks.sm` after each round, and refuses to start the timed set until the two most
   recent samples agree within 3% of each other (`clock_ramp.is_stable`) — bounded
   rounds, a clear abort if it never stabilizes.
2. Records `clocks.sm` immediately before and after **every** timed session
   (`clocks.jsonl`, one row per session_id), not just once per run.
3. `paired.py` refuses to pair two arms whose per-session clock samples differ by more
   than 3% in their mean (`clock_ramp.clock_profiles_compatible`) — the same kind of
   hard gate as the tenancy refusal, not an advisory note. **This 3% figure is
   provisional**, chosen from the smoke's spread above; it should be replaced with the
   real measured per-session spread once the noise-floor pair (below) has run.

## Warm-up: not a fixed count

Live evidence (a traced run, mirrors on; not baseline numbers — see the caveat at the
end of this section), the same 252-token prompt repeated:

| request | t since launch s | TTFT s | decode tok/s |
|---:|---:|---:|---:|
| 1 |  | 30.6 | 1.776 |
| 2 |  | 16.7 | 3.154 |
| 3 |  | 16.0 | 3.286 |
| 4 |  | 15.9 | 3.404 |
| 5 |  | 15.8 | 3.477 |
| 6 | 498.8 | 15.65 | 3.5316 |
| 7 | 550.3 | 15.51 | 3.5350 |
| 8 | 601.7 | 15.54 | 3.5400 |
| 9 | 653.2 | 15.55 | 3.5307 |

**The first request is not merely slower, it is a different regime** — roughly half
the plateaued throughput, TTFT roughly double. The mandatory discarded warm-up (rule
6) exists for exactly this: without it, a short arm would be dominated by that first,
unrepresentative request. **Decode is still climbing at request 5** — a fixed round
count would have declared victory after request 2 or 3, well before the number
settled — **and the asymptote itself is now measured**: requests 6-9 sit at
3.53-3.54 tok/s, flat to 0.3% across four consecutive requests, reached at roughly
request 6-7. `run_arm.sh`'s readiness loop therefore requires the two most recent
warm-up rounds' decode tok/s to agree (`clock_ramp.is_stable`, reused rather than
duplicated under a second name — see below), the same successive-agreement shape as
the SM-clock check, not a fixed count.

**This traced asymptote (3.53-3.54) is ~10% below the untraced mirrored baseline
(3.905-3.933, `DSV41_REFERENCE.md` section 19, "Task 1 matched baselines" --
not section 19's opening table, whose "decode is untouched by mirroring" is
superseded there) — nsys node-mode tracing is NOT free
on this machine, provisionally.** This repo's own `CLAUDE.md` records tracing as
negligible for DSV4.1 (390 ms untraced against 391 ms traced), but that was measured
at a ~391 ms step; the step is now roughly a third shorter with the same node count,
so the same absolute per-node tracing cost is a much larger fraction of it. Flagged
here rather than asserted as settled — a single before/after comparison is not
sufficient confirmation, and this campaign has already been burned once today by
treating a plausible-looking number as settled without asking what changed underneath
it. Do not carry the `CLAUDE.md` 390/391 figures forward for this configuration until
that caveat is added there.

**A real bug, caught while adding that check, before any GPU time was spent on it:**
the warm-up loop originally reused one shared `results-warmup.jsonl` across rounds.
`run_capture_sessions.py` resumes by `session_id` — once a single-turn session's one
turn is written, every later call for that same session_id and results file returns
immediately without contacting the server at all. So every round after the first was
silently a no-op: the clock/compile checks on "round 2" onward were reading whatever
state persisted from round 1, not fresh measurements. Fixed by giving each round its
own results file (`results-warmup-<round>.jsonl`); this is also what made the decode
tok/s table above possible to collect in the first place — the bug would have hidden
this instability, not just failed to check for it.

**Caveat on the table above**: recorded under nsys node-mode tracing, against an
**untraced** mirrored baseline of 3.905-3.933 tok/s (`DSV41_REFERENCE.md` section 19,
"Task 1 matched baselines"). Both are mirrors-**on** cells; the matched mirrors-off cell
is 2.903-2.919. Since 2026-09-22 mirroring is **on by default** in `arm_env.base_env()`,
so it is an arm that overrides `SGLANG_MOE_EXPERT_MIRROR_DIRS` to the empty string that
belongs against the mirrors-off cell. Section 20 records what this harness measures against both.
Tracing overhead is unmeasured for this configuration (`PIPELINE_BASELINE.md` section
3.2 notes the same gap for the Engine path). Read this table as warm-up *shape*
evidence — the first-request regime shift, the still-climbing tail — not as baseline
throughput values.

## Cost per arm

**Startup duration is not stable and must never be used as a readiness proxy or a
health signal.** A cold smoke launch took ~430 s to `/health`; a later trace run
reached it in **127.9 s**, because the weights were still in the page cache from the
run before. `corpus-c.log`'s ~200 s figure (`tokenizer_e2e=78.71s`,
`scheduler_e2e=69.87s`, decode graph capture 3.87s) is one more data point in that
same range, not a number to time out against. This is exactly why the readiness gate
is compile-quiet-and-clock-stable rather than "wait N seconds": it does not assume a
fixed startup time, on purpose.

Budget accordingly: startup (128-430 s, observed range) + clock/compile
stabilization (a handful of warm-up rounds) + 8 sessions x (TTFT ~55-99 s + 127 decode
tokens at ~2.2-3.5 tok/s — **prefill is roughly half the wall clock**, not a rounding
error on decode). Call it 20-30 minutes wall time per arm, not a fixed 20-25.

## THE NOISE FLOOR IS UNMEASURED FOR THIS PROTOCOL

Do not read the `c`/`cpf` per-session spread as "measured and small" — it is one pair
of runs, not a calibration, and it predates the SM-clock finding above (an unaccounted
clock swing could easily produce that spread on its own). **The floor to measure is
the per-session repeat spread**, not an aggregate. Greedy decoding on this stack is
also not reproducible (0/29 identical completions at temperature 0 in a separate
campaign), so some floor is guaranteed; its size for this exact protocol, on this
exact hardware state, has not been established.

**Rule: the first arm run under this harness is the same config twice.** Run it as two
arms (e.g. `baseline-1` and `baseline-2`, no overrides) and pair them with `paired.py`.
Record the resulting per-session delta vector and sign-test p-value as this protocol's
noise floor. No later arm comparison may claim a difference smaller than that floor,
per session, without saying so explicitly.

## The comparison is always per-session, never arm-mean-vs-arm-mean

Per-session tok/s varies more than 60% across this corpus's 8 different prompts
(2.18-3.53 tok/s in the recorded 4-session arms) because prompt content — not whatever
changed between arms — dominates it. `paired.py` therefore only ever joins two arms
**by session_id** and reports **per-session** deltas, ratios, win count, and a
one-sided sign test; it has no function that takes two arm-level scalars and returns a
difference. `median_tok_s` exists only to report an arm's own median across its
sessions as context.

## Protocol

1. **tok/s is decode-only**: `(completion_tokens - 1) / (last_token - start - ttft)`,
   `completion_tokens` read from the server's `usage` block, never assumed from the
   text (`run_capture_sessions.py`; `metrics.decode_tokens_per_second` pins the same
   formula independently for the test suite).
2. **The per-session comparison is the only comparison** (above).
3. **Verify env from `/proc/<pid>/environ`**, string-compared against a per-condition
   expected literal; abort on mismatch (`run_arm.sh`, `tenancy.verify_env`) — the live
   server process's real environment, not what the launcher thinks it set.
4. **Result gate**: exactly 8 records and 0 errors, or abort (`results_gate.py`).
5. **Health gate**: up to 900 s of `curl -sf /health`. `/health` runs a real
   generation and is slow cold; never shorten this.
6. **A warm-up before the timed set is mandatory**, not optional — extended into as
   many rounds as it takes for the SM clock, JIT compilation, **and decode tok/s** to
   all settle together (see "Warm-up: not a fixed count" below), discarded from
   timing either way, written to `results-warmup-<round>.jsonl` (one file per round —
   see that section for why one shared file was wrong). **This is a deliberate,
   disclosed deviation from the Engine-based methodology that produced 2.781 tok/s**
   (which ran no warm-up at all): under the HTTP server, a cold first session could be
   timed while still mid-JIT-compile, before the clock has settled, or (now measured)
   at roughly half its plateaued throughput — none of which the Engine path had to
   contend with. This campaign's number may therefore differ from 2.781 by
   construction; that is expected, not a discrepancy to chase down.
7. **Cold server per condition.** `run_arm.sh` launches, verifies, times, and tears
   down one server per invocation.
8. **Card tenancy is recorded in the manifest** (`nvidia-smi` memory used, whether
   production (port 7867) is running, SM clock limit) at run start and run end.
   `paired.py` refuses to pair two arms whose tenancy differs beyond a small
   memory-noise tolerance — a hard gate. Today the card is empty, which is unusual; do
   not assume it stays that way between arms.
9. **The manifest records** commit SHA, corpus checksum, the full merged environment,
   and the 8 pinned session ids (`run_arm.sh`).

## CPU pinning (hard requirement)

Cores 64-71 stay free for every CPU job on divix01 — core 71 is production's doorbell
spin core. `run_arm.sh` pins the server to `taskset -c 32-63` and the driver
(`run_capture_sessions.py`, invoked once per session) to `taskset -c 8-15`, matching
the Qwen campaign's split, and caps `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.

## Files

- `arm_env.py` — the DSV4.1 EXL3 recipe (env vars from phase 3b's `env-full.sh`) and
  the `sglang serve` CLI flags. `arm_env(overrides)` layers a V2 storage-change arm's
  overrides onto the base env; `ServerArgs.argv()` builds the server command line.
- `session_subset.py` — the pinned real corpus path/checksum, the 8-session shape
  (`--n 8 --skip 0 --prompt-tokens 256 --new-tokens 128`), the expected session_id
  order, and the dedicated warm-up session.
- `synthetic_corpus.py` — truncates each real session's first turn to 256 tokens and
  re-decodes it into a one-turn chat session; `assert_fits_context` refuses an
  over-budget request before it is sent.
- `clock_ramp.py` — SM clock sampling, the successive-samples-agree stability check,
  and the cross-arm clock-profile compatibility check.
- `compile_watch.py` — scans `server.log` for serving-time JIT compile events, total or
  within a byte range (one timed session's window).
- `metrics.py` — tok/s formula, median (context only), one-sided sign test. Pure, no I/O.
- `tenancy.py` — tenancy capture/comparison, `/proc/<pid>/environ` parsing and
  verification. Pure except `capture_tenancy`, which shells out to `nvidia-smi`/`ss`.
- `results_gate.py` — the exactly-8-records-0-errors gate.
- `task1_verdict.py` — sha256-pinned loader for the real `task1_arm_verdict.py`
  (divix01, outside any repo — see "One harness, not two" below).
- `generations.py` — this campaign's `{python tree: label}` registry, checked with
  Task 1's own `generation()` lookup.
- `client_latency.py` — percentiles of `run_capture_sessions.py`'s new `chunk_times`
  field, as `client_inter_token_latency_s`: a client-side proxy, explicitly never
  `step_latency` and never compared against its thresholds.
- `report_builder.py` — shapes this campaign's HTTP results into the report schema
  `task1_arm_verdict`'s functions expect; documents the one field (`step_latency`)
  that is honestly unavailable rather than faked, and the server-provenance ceiling.
- `verdict.py` — judges one arm by calling Task 1's `check_arm`, `check_timed_phase`,
  `boot_growth`, `contention`, `generation`, `session_outliers`, and `cross_arm_outliers`
  (imported, never reimplemented)
  plus this campaign's two additions (compile-contamination, the clock-readiness
  note), each labeled as absent from Task 1's own checks and why it was added.
- `run_arm.sh` — preflight (clean tree, optionally pinned to `EXPECT_SHA`), the
  generation gate, builds the synthetic corpus, launches one cold server, runs the
  health gate, verifies env, runs the readiness gate (SM clock stable + JIT compile
  quiet), then the 8-session timed set with per-session clock/cpu_s/compile-status
  samples and boundary samples (hard-aborting on a mid-session compile event), the
  result gate, builds `report.json` and judges it (`verdict.txt`), and writes
  `run-manifest.json`. Usage: `run_arm.sh <arm_name> <port> [KEY=VAL ...]`. Accepts
  `DECODE_LOG_INTERVAL=<n>` (bash env var, not a `KEY=VAL` arg — it's a CLI flag, not
  an `SGLANG_*` env var) to override `--decode-log-interval`; unset by default.
- `decode_log_interval_compare.sh` — runs the same arm at `decode_log_interval=40`
  and `=1`, pairs the two. Written for the team lead to run; not run here (see "One
  harness, not two").
- `paired.py` — the only comparison this harness offers: per-session, refusing to pair
  arms under mismatched tenancy, mismatched clock profile, or any compile-contaminated
  session. Usage: `python paired.py <arm_a_dir> <arm_b_dir> [--a-name NAME] [--b-name
  NAME]`. Does not yet read `verdict.txt`/`report.json`; a natural next step.
- `test_dsv41_baseline.py` — CPU-only unit tests (no network, no GPU, no server, and no
  dependency on the real `task1_arm_verdict.py`, which is not committed anywhere — a
  fake stand-in with its exact call shape exercises this campaign's own glue code):
  the tok/s formula, median/sign-test, the session subset, the synthetic-corpus
  truncation and context-budget assertion, env verification, the result gate, the
  SM-clock stability check, the compile-event log scanning, the clock-profile/
  tenancy/contamination pairing refusals, the generation registry, the report
  shaping, and the verdict orchestration (including that the step-latency gap is
  surfaced, never silently passed).

## One harness, not two: adopting Task 1's provenance and verdict machinery

Task 1 (`docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`,
`analysis/dsv41-drive/{task1-baseline-arms.sh,task1_arm_verdict.py,PIPELINE_BASELINE.md}`
on divix01) already has a matched-baseline harness for the offline `Engine` path, with
a measured noise floor (0.25-0.53% cell sd) and provenance/verdict machinery that has
caught real mistakes over several iterations — documented in `PIPELINE_BASELINE.md`
itself. This campaign reuses that machinery rather than building a second, competing
one: **a second provenance mechanism is worse than none**, because the two would
inevitably drift apart in what "VALID" means.

What's genuinely new here (Task 1 has no equivalent): the HTTP serving path itself,
the JIT-compile contamination gate, and the SM-clock stability gate. Everything else —
provenance shape, page-cache residency, cross-arm leave-one-out outliers, contention
detection, the code-generation registry — is Task 1's own code, imported and called,
never reimplemented (`task1_verdict.py`, `verdict.py`).

**`task1_arm_verdict.py` is not in any repo** (`PIPELINE_BASELINE.md` section 2: a
hand-copied runtime file on divix01). `task1_verdict.load_task1_verdict()` sha256-pins
it before importing, the same discipline `task1-baseline-arms.sh` applies to its own
copy — a silent drift here is exactly the failure that pinning exists to prevent.

**One shared-file change, deliberately not forward-ported**: `scripts/dsv41/
provenance.py`'s `process_tree_cpu_s()` gained an optional `pid` parameter (backward
compatible; default unchanged) so an external driver can sample a server it launched
as a separate process, rather than only ever sampling itself. **This file has now
diverged from Task 1's copy in `wt-dsv41` by exactly this one parameter** — written
down here rather than silently reconciled or silently left unmentioned, per the same
principle that cost this campaign real time today when a divergence went unrecorded
elsewhere. Forward-porting it (or not) is a decision for whoever owns that worktree.

**`step_latency` (the engine-side, Task-1-comparable quantity) stays acknowledged-
absent.** Two things closed part of the gap without substituting into that field:

1. **A client-side proxy now exists, under its own name.** `run_capture_sessions.py`
   gained a purely additive field, `chunk_times` (`time.monotonic()` per streamed
   reasoning/content delta — at `stream_interval: 1`, one per token) — no existing
   behavior or field changed. `client_latency.client_inter_token_latency_s()` turns
   that into percentiles, stored in `report["per_session"][i]["client_inter_token_
   latency_s"]`, **never** in `step_latency`. It is not the same quantity:
   detokenizer, serialization, the socket, the client's event loop and client-process
   scheduling are all inside every gap it measures, and `check_arm`'s thresholds are
   calibrated against the engine-side number. `verdict.py`'s module docstring and
   `client_latency.py` both say this in the same words on purpose, so a future reader
   hits the warning wherever they land.
2. **A genuine engine-side source may exist, but has an unmeasured observer-effect
   risk and is not yet implemented.** `ServerArgs.decode_log_interval` (default 40,
   `--decode-log-interval` confirmed against the actual argparse registration, not
   just the naming convention) governs how often the scheduler prints `Decode batch,
   ..., gen throughput (token/s): X` (`scheduler_components/metrics_reporter.py`),
   computed from `time.perf_counter()` inside the scheduler process itself —
   genuinely engine-side. At `decode_log_interval=1` with this harness's
   `max_running_requests=1`, each log line covers exactly one decode step for one
   request, so `X` inverts cleanly to that step's latency (`1 / X` seconds) with no
   dependency on step-counter alignment across sessions (the interval-1 condition is
   unconditionally true every step).

   **But logging every decode step is an observer effect on the thing being
   measured**, on the path whose latency this would read. At interval 40 the
   scheduler formats and writes one line per 40 steps; at interval 1, every step. At
   ~255 ms/step a formatted log line is probably negligible — the same shape of claim
   ("probably negligible") that was wrong once already today (the clock threshold).
   So this is pursued as a measurement, not an assumption, before any wiring:
   1. Confirm the flag name against `--help` in a real launch, not just the parser
      registration (done here: `--decode-log-interval`, but a launched process is the
      real check).
   2. **Establish the cost empirically**: the same arm at interval 40 and interval 1,
      compared per session. If decode tok/s moves, interval 1 is unusable for timed
      arms (still useful for diagnosis-only runs).
   3. If free, prefer it to the client-side number for anything gate-related and keep
      `client_inter_token_latency_s` as an independent cross-check from the other end
      of the stack — where the two disagree is a measurement of what the HTTP layer
      itself costs, which nobody has measured yet.

   **Not wired into `verdict.judge()` until step 2 has a number.** Step 2 needs GPU
   time this task does not take on its own initiative. `decode_log_interval_compare.sh`
   runs the same arm at `decode_log_interval=40` and `=1` back to back and pairs the
   two arms' decode tok/s per session (`arm_env.ServerArgs.decode_log_interval`, opt-in,
   off by default) — written for the team lead to run, not run here.

**Timed-phase page-cache residency — wired in.** `run_arm.sh` samples
`provenance.resident_bytes()` externally before the server starts, once it is ready,
and after the timed set. `verdict.residency_cache_dict()` maps these to Task 1's
`before`/`ready`/`last` phase shape. Task 1's unmodified `check_timed_phase()` gates
growth from server ready through the timed sessions; `boot_growth()` records startup
growth separately in verdict notes. Sampling is whole timed set rather than per
session, because the HTTP server is a separate process. A missing directory or any
missing boundary is an unacknowledged verdict problem.

**Server-side provenance is capped at `/proc/<pid>/environ`, named explicitly** in
every report (`report["provenance"]["server_provenance_ceiling"]`), per the same
register `PIPELINE_BASELINE.md` uses for its own two open provenance holes (section
7.4). The harness's own provenance (`provenance.capture()`, resolved env, git state,
sglang import path, loaded harness files) is captured from the launcher process with
`sglang` importable from the worktree under test, immediately before the server is
spawned — not from inside the server itself, since the server was told not to be
taught to emit its own provenance yet (a real change to the launch path, not this
task's to make).

## Resolved: the mirror row source is now the default (2026-09-22)

`base_env()` sets `SGLANG_MOE_EXPERT_MIRROR_DIRS=/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash`
(`arm_env.EXPERT_MIRROR_DIRS`; both roots verified present at 205 GB / 50 entries).
Mirroring is a property of the box's storage and production reads mirrored, so an arm
that omitted it was measuring a drive layout nobody runs. To measure the unmirrored
drive, override the var to the **empty string** — dropping the key is no longer possible,
since `base_env()` always supplies one, and `exl3_expert_format.exl3_mirror_config()`
reads an empty value as off.

That last point is why the verdict's `mirror` flag tests the *value* and not the key
(`run_arm.sh`, the `verdict.judge` call): a key-presence test would now judge every
unmirrored arm against the mirrored baseline.

`SGLANG_MOE_EXPERT_MIRROR_WEIGHTS` is still unset, which means equal shares; the recorded
split was 49.6/50.4 (`DSV41_REFERENCE.md` section 20). `run-manifest.json` records the
commit and the full merged environment either way.

**Mirrors confirmed working under load, live** (the same traced run as the warm-up
table above): over a 5 s window, nvme0 read 2,233 MB and nvme3 (`/mnt/nvme4`) 2,247 MB
— a 0.6% split imbalance — while **nvme2, the source, read 0 MB**. This reproduces
Task 0's finding (nvme2 at 0.00% of service-attributed expert bytes) live rather than
from a document, and is independent of the still-open commit-pinning question above.

## Not carried over from earlier drafts of this harness

- An offline-`Engine`-based design (`trace_corpus.py` alone) was tried and discarded:
  the user wants the real serving path, and it has since been confirmed live
  (`smoke.sh`). `trace_corpus.py` is not used anywhere in this harness — a
  `session_id`-on-record patch to it from an earlier iteration was reverted rather than
  kept, since nothing here reads its output any more; `run_capture_sessions.py`'s own
  `results.jsonl` already carries `session_id` per record natively.
- A 12,288-token-context, real-multi-turn-conversation design was also considered and
  discarded, twice — first because prefill is 55-99 s per 256-token prompt and a
  multi-thousand-token session would blow the time budget, then (the sharper reason)
  because raising `context_length` changes the hot-cache/RAM-miss behavior this
  campaign measures. That design remains unused; the short corpus is now run with a
  longer context to match production, so its new results require a new baseline.
