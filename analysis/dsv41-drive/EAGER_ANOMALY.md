# Eager + mirrors "is slower": does not reproduce (2026-09-20)

## Read this first: the 1.54x is one base arm reading 147 GiB LESS, not the mirror arm reading more (2026-09-21)

Nobody should look for extra reads in the mirror arm. There are none.

**What the 1.54x compares.** 397.41 / 257.96 GiB = 1.54: the whole-arm `/proc/diskstats`
deltas (three drives summed, startup included) of two single runs of `run-mirror-arms.sh`
on 2026-09-19, `e-mirror` over `e-base`. No cache counters, no environment dump, no git
revision in either log. The numerator is normal: `e-mirror`'s 397.41 is the ~381 GiB every
eager arm reads in its four sessions, plus 17-24 GiB of startup. The denominator is the
outlier. Against today's eager base arm (380.9 GiB of sessions + ~24.3 GiB of startup =
~405 GiB whole-arm) `e-base` read 405 - 258 = **~147 GiB less**. (Whole-arm to whole-arm; the
"~123 GiB" that once made a cache story look attractive mixed session bytes with whole-arm
bytes.) `e-base` also ran fast: TTFT 85.4 / 55.1 / 24.8 / 5.6 s and decode 3.96 tok/s in
session 3, faster than the graph arms. Its startup (20.2-24.5 GiB in today's base arms)
leaves 233-238 GiB for sessions, and sessions 0+1 alone read 210.8 GiB today, so sessions 2
and 3 together read about 23-27 GiB: almost nothing.

This is the gate's question, reworded: not "why does mirroring read more" but "why did one
base arm read ~147 GiB fewer bytes than seven others, none of which differ from it in code,
corpus, or (as far as anything recorded) configuration". The plan's bar, "resolved only with
evidence explaining the additional reads", cannot be met for the original arm, and there are
no additional reads to explain.

### The determinism argument: why machine state is the least likely class

Per-session bytes are the same to 0.1% in every arm we have, and so are the greedy outputs:

| arm (2026-09-20) | code | order | mirrors | trace | session GiB (s0 / s1 / s2 / s3) | output sha1 (s0-s3, first 8) |
|---|---|---|---|---|---|---|
| r1-mirror (trace lost) | HEAD | 1st | on | on | 127.87 / 82.85 / 92.62 / 77.32 | ad32424c de3bc746 91530418 5142f271 |
| r1-0-base | HEAD | 1 | off | on | 128.03 / 82.89 / 92.67 / 77.33 | same |
| r1-1-mirror | HEAD | 2 | on | on | 128.02 / 82.90 / 92.65 / 77.33 | same |
| r1-2-mirror | HEAD | 3 | on | on | 128.04 / 82.88 / 92.65 / 77.33 | same |
| r1-3-base | HEAD | 4 | off | on | 128.01 / 82.89 / 92.65 / 77.39 | same |
| r2-0-base | HEAD | later | off | **off** | 127.99 / 82.87 / 92.65 / 77.33 | same |
| s19-0-base | `1525e43ab9` | separate | off | **off** | 127.94 / 82.88 / 92.63 / 77.26 | same |

Across the four traced arms the RAM-miss row counts agree to 3 in 30,367, and MiB per read
row is 12.72-12.73 in all 16 traced arm-sessions. What changes between arms is *timing* (TTFT
55 s against 30 s with mirrors) and startup bytes (18.3-24.5 GiB for identical code, so
startup has a cache-dependent buffered component and sessions do not).

**The argument.** Which rows a session reads is a deterministic function of the computation:
weights, config, prompt, numerics and the tier and hot-set policy. Page-cache warmth, drive
state, predecessors, ordering, the mirror choice and tracing change how fast those rows
arrive, not which rows are asked for. They did not change the byte count here across two
commits, both orders, both trace settings and mirrors on or off. So a machine-state
explanation would have to produce a 147 GiB change in *which rows were read* out of
something that has never moved that number by more than 0.1%. It can, but it is the least
likely class. What can change the number is one of two things:

1. **A different computation ran**, so different rows were requested: uncommitted code,
   a different environment (a knob such as `SGLANG_MOE_HOT_SEED`, the hot-tier dynamics or
   the tier sizes), different numerics or prompts. Counters would show fewer rows read.
2. **The rows were requested but did not reach the device**: read through something that
   caches (buffered I/O, or a cache in front of the drive). Counters would show the same rows
   read; diskstats would show fewer bytes.

These are told apart by one observable pair, which is the standing protocol below.

### What the record already excludes, and what it does not

| candidate | status | evidence |
|---|---|---|
| mirroring changes the bytes | excluded | per-row counters, 4 traced arms: identical bytes and decisions |
| tier warming differs between arms | excluded | occupancy 5644/5644 in session 0 in both; admissions equal evictions after |
| ordering (base before / after mirror) | excluded for eager-after-eager | B M M B |
| tracing or counters | excluded | untraced base (r2-0, s19-0) matches |
| committed code | excluded | 1525e43ab9 re-run; and see the reflog below |
| corpus, session count, source drive | excluded | `sessions.jsonl` mtime 09-15; wall time; 0.03 GiB on other drives |
| predecessor was a *graph* arm (`g-mirror`) | **not tested**; mechanism weak | every eager arm today follows an eager arm; under O_DIRECT a predecessor has no channel to change host-issued reads |
| uncommitted edits in `wt-dsv41` at 23:02 | **not excluded** | see below: nothing shows them, nothing rules them out |
| environment exported in a shell | **not excluded** | `env.sh` mtime shows the file was old, not that the process used it; no env dump exists |
| driver | not tested for base | 09-19 used `trace_corpus.py`; today's arms use `eager_arm_driver.py`; `e-mirror` reproduced under the new one |
| non-direct reads somewhere | unsupported either way | config and code say O_DIRECT (`uring_file_reader.cpp:115`), no process-level record from 09-19 |

### Attempt to recover the state of 2026-09-19 (no runs)

- **wt-dsv41 reflog** (`.git/worktrees/wt-dsv41/logs/HEAD`, read with `cat`; no git command
  was run in that worktree): HEAD was `efcef725fc` from 22:35:54 and did not move again until
  23:36:29 (an `am` of the docs commit). The four arms ran 22:45-23:18 (`g-base` ended
  22:54:44, `g-mirror` 23:02:11, `e-base` 23:10:21, `e-mirror` 23:18:23; `e-base` started at
  23:02:12). **No checkout, reset, rebase or commit** falls in that window, and
  `efcef725fc..1525e43ab9` is docs and scripts only. So the *committed* code that ran is the
  code the re-run used. A reflog says nothing about uncommitted edits.
- **Claude session transcripts** (local `~/.claude/projects`, 22:30-23:30 CDT on 09-19): two
  local sessions were active. The one that launched the arms (22:45) made 55 tool calls
  between 22:34 and 23:19: a commit and a fetch at 22:35, a DSpark smoke run on the box that
  ended before the arms began, the arm launch, and reads of results. **None edited python
  under `wt-dsv41`.** The other session edited `exl3.py` at ~23:30 in the laptop worktree,
  after `e-mirror` had finished. Sessions on other machines, and edits made in an editor,
  are not visible here.
- **`~/.bash_history` on divix01**: last written 2026-09-11; non-interactive ssh commands do
  not enter it, so an exported variable would not appear.

Result: nothing shows an uncommitted edit or an unusual environment, and nothing can rule
either out. The residual is exactly "uncommitted tree, environment, unrecorded state".

**Status: retired as non-reproducible; cause unknown; residual = uncommitted tree,
environment, unrecorded state; determinism argues against machine state.** The 257.96 GiB /
1.54x pair must not be cited. Re-running the 09-19 sequence would not change this: it would
most likely read 381 GiB again, which leaves the anomaly unreproducible with one fewer
untested factor, not resolved. It is not worth 35 minutes of GPU.

### Standing protocol: how the next low arm is diagnosed instead of debated

The reason 09-19 is unrecoverable is that nothing recorded what ran. That has changed:
`scripts/dsv41/provenance.py` records, from inside the arm's own process, git HEAD and
dirtiness with a tracked-diff hash and untracked files, the live and exec-time environment,
every `SGLANG_*` knob with its default resolved, `sglang.__file__`, a drive-idle check and
env drift at engine ready. `trace_corpus.py` adds fincore residency and a
meminfo/load/drive sample at every session boundary, and `eager_arm_driver.py` records
per-session diskstats bytes and the output sha1. The task1 arms already do this
(`task1-results/` holds their pre-registration files and the clean-reference manifest; raw
arm outputs stay on divix01).

For any eager arm whose session bytes differ from the table above, or whose TTFT falls
faster than ~55 s, compute per session with the trace and cache counters on:

    implied = (RAM miss rows + bg rows) x 12.72 MiB      # eager_cache_report.py: "MiB/read row"
    actual  = diskstats bytes over the session, all three drives

| what moved | class | check next |
|---|---|---|
| actual and implied both fall; MiB/read row stays 12.72-12.73 | fewer rows were read: a different computation | `provenance` env and git (dirty files, diff hash, `sglang.__file__`); output sha1 against the table; VRAM/RAM miss counts and tier occupancy per session; the arm's `SGLANG_MOE_*` knobs against `env-full.sh` |
| actual falls, implied does not; MiB/read row falls below ~12.7 | the rows were read but not from the device | per-session fincore of the source and mirror shards; the reader mode in the process environment; a buffered code path |
| neither falls | not this anomaly | it is a timing effect |

The threshold is descriptive: 12.72-12.73 is what all 16 traced arm-sessions measured; it is
not a tuned limit.

**What the eager path still needs added, so the protocol is complete** (no code changed here):
`eager_arm_driver.py` does not record what `trace_corpus.py` does, namely fincore residency
of the expert directories at each session boundary and the meminfo/load sample. Without
per-session residency the second row of the table can only be answered after the fact, by a
single end-of-run fincore, which is what this investigation had to do. The counters need the
trace path set: an arm run with `NO_TRACE=1` cannot be diagnosed this way (base with and
without the trace matches, so nothing is lost by leaving it on).

---


DSV41_REFERENCE §19 recorded `e-mirror` 25% slower than `e-base` (2.0126 vs 2.6947 tok/s)
while reading 1.54x more bytes (397.41 vs 257.96 GiB), and named one candidate: the base
arm's pinned host tier warms across sessions, the mirror arm's does not.

**Result: the mirror arm reproduces; the base arm does not.** With cache counters on, the
two arms move identical bytes and make identical cache decisions, and the mirror arm is
*faster* in every session. The tier-warming hypothesis is refuted, because neither arm's
tier warms. The original `e-base` result (257.96 GiB, TTFT 85 -> 55 -> 25 -> 5.6 s) is
what is unexplained, and nothing reproduces it, including the §19 commit's own eager base
(380.7 GiB, ~55 s; see below). **This does not meet the plan's bar for recording the
anomaly resolved** ("only with evidence explaining the additional reads"): there are no
additional reads to explain in the current code or the old, but the original base arm's
*missing* reads are unexplained. The recorded number should be retired as non-reproducible.

Everything below is measured unless marked *inferred*.

## What ran

Same corpus, sessions and settings as §19: graphs off, `SGLANG_MOE_EXPERT_GRAPH_GATHER=0`,
sessions 0-3, 256 prompt / 128 new tokens, `env-full.sh` (70 GiB pinned tier = 5644 rows,
14 GiB hot GPU tier, Engram 5 GiB), HEAD `099eadba33` plus the counter change below.
Drives idle-checked before every arm (0 B/s on nvme0 and nvme2; nvme4 <= 44 KiB/s).
Script `eager-cache-arms.sh`, driver `eager_arm_driver.py`, report `eager_cache_report.py`;
results in `eager-results/`, raw traces and counter files stay on divix01 in
`analysis/dsv41-drive/eager/`.

| arm | order | trace + counters | mean decode tok/s | session bytes GiB |
|---|---|---|---|---|
| r1-0-base | 1 | on | 1.6782 | 380.9 |
| r1-1-mirror | 2 | on | 2.0333 | 380.9 |
| r1-2-mirror | 3 | on | 2.0084 | 380.9 |
| r1-3-base | 4 | on | 1.6855 | 380.9 |
| r2-0-base | later | **off** (as in §19) | 1.7031 | 380.8 |
| first mirror (trace lost, see below) | - | on | 2.0019 | 380.7 |
| s19-0-base (§19 commit `1525e43ab9`) | separate run | **off** | 1.6985 | 380.7 |

Order B M M B (each arm type both before and after the other). The base-with-tracing-off
control exists because §19's arms had no trace; it matches the traced base, so the trace is
not what differs from §19.

Byte volume: 6 complete arms x ~381 GiB of expert reads, plus ~20 GiB of startup reads each,
about 2.3 TiB, plus one more complete mirror arm whose trace I lost (below) and two arms I
aborted early. GPU always via `gpu-run.sh`; nothing on port 7867.

## Per-session table (means of the two traced repeats; the repeats agree to 0.1%)

RAM miss = rows read through the row source in a gather; bg = rows read outside gathers
(promotions, seeding). Tier columns are `PinnedSlotLRU` counters summed over 40 layers.
"Route" units count a repeated expert each time (`PinnedHostCacheStats`).

| s | arm | TTFT s | tok/s | disk GiB (nvme0/2/4) | VRAM miss | RAM miss | bg | tier admit | tier evict | route hit rate | occupancy/cap | Engram hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | 101.2 | 1.429 | 0 / 127.9 / 0.2 | 23931 | 10118 | 180 | 10308 | 5986 | 0.577 | 5644/5644 | 0.046 |
| 0 | mirror | 65.3 | 1.788 | 63.9 / 0.1 / 64.0 | 23931 | 10118 | 180 | 10290 | 5974 | 0.577 | 5644/5644 | 0.046 |
| 1 | base | 55.3 | 1.799 | 0 / 82.9 / 0 | 19802 | 6636 | 36 | 6684 | 6684 | 0.664 | 5644/5644 | 0.294 |
| 1 | mirror | 29.9 | 2.117 | 41.4 / 0.1 / 41.4 | 19801 | 6636 | 36 | 6692 | 6692 | 0.664 | 5644/5644 | 0.294 |
| 2 | base | 54.0 | 1.523 | 0 / 92.6 / 0 | 22386 | 7415 | 45 | 7452 | 7452 | 0.669 | 5644/5644 | 0.242 |
| 2 | mirror | 29.0 | 1.885 | 46.3 / 0.1 / 46.3 | 22386 | 7414 | 45 | 7455 | 7455 | 0.670 | 5644/5644 | 0.242 |
| 3 | base | 55.8 | 1.975 | 0 / 77.3 / 0.1 | 19295 | 6199 | 27 | 6235 | 6235 | 0.677 | 5644/5644 | 0.400 |
| 3 | mirror | 30.1 | 2.294 | 38.6 / 0.1 / 38.6 | 19295 | 6199 | 27 | 6234 | 6234 | 0.677 | 5644/5644 | 0.400 |

Full per-arm tables (with the Engram miss/fill columns): `eager-results/eager-report.md`.

## What this shows

1. **Byte-neutral, and fully accounted for.** Every arm reads 380.9 GiB in sessions.
   (RAM miss + bg rows) x 12.72 MiB/row = 380.9 GiB, so the drive bytes are explained by
   the rows the counters say were read; nothing is read behind the counters' back.
   Mirroring splits the same bytes 50/50 across nvme0/nvme4 (nvme2 carries 0.1 GiB per
   arm) instead of putting them on nvme2.
2. **Identical cache decisions.** RAM miss rows agree to 3 in 30,367; tier admissions and
   evictions agree to ~0.1% (the difference is snapshot blur, below). The row source changes
   who serves the bytes, not which rows get read.
3. **Neither tier warms.** Occupancy reaches capacity (5644/5644) inside session 0 in both
   arms; from session 1 on, admissions equal evictions. The route hit rate moves 0.58 ->
   0.66 -> 0.67 -> 0.68 identically in both arms, and it is the same across all four traced
   arms to 0.1%. The hypothesis predicted more misses and evictions plus non-climbing
   occupancy in the mirror arm, with a warming base arm. Observed: no difference in
   misses, admissions, evictions or occupancy, and no warming in the base arm.
4. **Mirroring helps.** Steady TTFT 29.7 s vs 55.0 s (1.85x, sessions 1-3), decode
   2.01 vs 1.69 tok/s (1.19x, mean of arms) on the same reads. Greedy outputs are
   byte-identical between the arms (sha1 of every session's text matches).
5. **The mirror arm reproduces §19.** TTFT 59.2 / 29.6 / 28.9 / 30.0 s and 2.0126 tok/s
   then, 65.3 / 29.9 / 29.0 / 30.1 s and 2.01-2.03 tok/s now.

## The §19 commit's eager base (2026-09-20, decided)

The open question was whether §19's `e-base` was real code behaviour or system state. I ran
one eager base arm at `1525e43ab9`, the §19 commit, in a separate worktree
(`wt-dsv41-s19` on divix01, detached, with a private copy of the extension build dir
`exl3-build-s19`). Same driver, script, corpus, sessions 0-3, 256/128, `env-full.sh`,
graphs off, trace off (as in §19). Drives idle before the arm (0 B/s on nvme0, nvme2, nvme4).
The process ran from that worktree with `PYTHONPATH` pointing at it (checked in
`/proc/<pid>/environ`).

| | §19 `e-base` (recorded) | `1525e43ab9` base, now | HEAD base, now (r1-0, r1-3, r2-0) |
|---|---|---|---|
| session GiB | ~234 (257.93 total less ~24 startup) | **380.7** | 380.9 / 380.9 / 380.8 |
| TTFT s | 85.4 / 55.1 / 24.8 / 5.6 | **95.4 / 54.9 / 53.6 / 55.4** | ~101 / 55.3 / 54.0 / 55.8 |
| decode tok/s | 1.46 / 2.16 / 3.19 / 3.96 | **1.451 / 1.817 / 1.543 / 1.982** | 1.43 / 1.80 / 1.52 / 1.98 |
| mean tok/s | 2.6947 | **1.6985** | 1.68-1.70 |
| greedy output sha1 (s0-3) | not recorded | `ad32424c de3bc746 91530418 5142f271` | identical |

**Result: the §19 commit reads ~381 GiB and holds ~55 s, like HEAD.** Per-session bytes
match HEAD to 0.1% (127.9 / 82.9 / 92.6 / 77.3 GiB), and the greedy outputs are
byte-identical to HEAD's, so the old code computes the same thing and reads the same rows.
The eager-path changes between `1525e43ab9` and HEAD are therefore not the cause, and there
is no regression to bisect. `e-base` in §19 was not reproducible from its own commit.

What the driver needed to run the old code: nothing that touched the old worktree. The old
`trace_corpus.time_stream` returns no `output_text` (added in `ee4ca6b19b`), so the driver
now wraps the generate stream with `capture_text` to keep the last text itself, and the
script gained `WT=` and `EXL3_BUILD=` to run from another worktree. Both are no-ops on HEAD.

## What is unexplained

Why §19's `e-base` read ~147 GiB less than the same code reads today, with sessions 2 and 3
nearly free (sessions 0+1 today read 210.9 GiB; §19's whole arm, less startup, was ~234 GiB,
leaving ~23 GiB for sessions 2 and 3, against ~170 GiB today: *inferred* from totals, §19 has
no per-session bytes). Code, corpus, environment files, ordering and tracing are all ruled
out (measured: ordering, tracing, the §19 commit itself, corpus mtime, source drive;
*inferred*: environment files by mtime). What remains is state of the machine that day. The
candidate I cannot exclude is page-cache or drive-cache state from the preceding `g-mirror`
arm (it read 127 GiB from nvme2 immediately before), but the config is `uring_direct`, and I
have no per-row counters from §19 to distinguish "not read" from "served from cache".
One §19 detail also does not fit a clean "warm from session 2" story: session 1 decoded at
2.16 tok/s then against 1.80 now, at the same 55 s TTFT.

The number is retired: nothing reproduces it, from its own commit or from HEAD.

Ruled out by measurement:

- Ordering: base, mirror, mirror, base; both base arms alike.
- Tracing or counters: base with no trace path matches the traced base.
- Code: the §19 commit's own base arm (above).
- Fewer sessions than recorded: `e-base.json` has 4 sessions of 128 tokens, and the wall time
  (490.2 s) fits them (388.5 s of sessions, 101.7 s of startup against 74-90 s today).
- Different corpus: `sessions.jsonl` has not changed since 2026-09-15, before §19.
- Different checkpoint copy: §19's diskstats show 257.93 GiB from nvme2 (the source) and
  0.03 GiB from the others.
- Environment (*inferred from mtimes*): `env.sh` and `env-full.sh` predate the §19 run.

## Reader mode and the page cache

The one unexcluded candidate after the §19-commit arm was a buffered read in §19 against
direct reads now. From the recorded artifacts, no GPU:

- §19 ran `uring_direct`. `run-mirror-arms.sh` sources `env-full.sh` -> `env.sh`
  (`SGLANG_MOE_EXPERT_FILE_READER=uring_direct`) and its `run_arm` passes only thread counts,
  `GRAPH_GATHER` and the mirror dirs, so nothing overrides the reader. The recorded logs carry
  no reader line and `e-base.log` no env dump, so this rests on the script and on `env.sh`
  (mtime Sep 19 02:08, before the run). There is no separate direct/buffered flag:
  `_resolve_direct()` reads only this variable.
- Neither `run-mirror-arms.sh` nor `eager-cache-arms.sh` (nor `run-native-mirror-arm.sh`) drops
  caches, calls `fadvise`/`vmtouch`, or syncs. Arms run back to back with whatever the previous
  one left, and in §19 `g-mirror` ran immediately before `e-base`.
- Today's reads are direct (measured). After ~2.3 TiB of arms, `fincore` shows 15.6 GiB of
  the 204.1 GiB source expert files resident, which is startup-scale (each engine start reads
  ~18-24 GiB of non-expert tensors), not the hundreds of GiB buffered reads would leave. That
  is not evidence about §19's process, only that the same config is direct today.
- The "123 GiB missing vs 127 GiB from `g-mirror`" coincidence mixes bases: 381 is session
  bytes today and 258 is §19's whole-arm total with ~24 GiB startup. Whole-arm to whole-arm the
  gap is ~147 GiB. A 127 GiB warm set would also not fit beside the 70 GiB pinned tier in
  188 GiB of RAM, and a cache story predicts `e-mirror` benefits too, which it did not
  (*inferred*).
- The decode pattern says the divergence starts before session 2. Today's base decodes
  1.43 / 1.80 / 1.52 / 1.98 tok/s (graph arms: 2.25 / 3.23 / 2.29 / 3.54), so sessions 1 and 3
  are fast per prompt. §19's `e-base` decoded 1.46 / 2.16 / 3.19 / 3.96: session 1 already 20%
  faster at an identical 55 s TTFT, then session 2 rises instead of dipping. Whatever changed
  cut decode-phase cost from session 1 on; a pure "cache warms in session 2" does not fit.

Conclusion: reader mode was the same, the scripts do not manage the cache, and the config
does not explain §19's arm. Page-cache warming is weakened, not excluded: nothing recorded
in §19 shows the reads were direct except the config. For future e2e arms: back-to-back arms
on one drive are independent only if the reader is genuinely O_DIRECT (checked here with
`fincore` after the arms) or the cache is dropped between them.

## Limits of the counters

- Snapshots are written at most every 0.5 s, so a session's edges blur by up to that much
  activity. Admissions vs (RAM miss + bg) differ by ~11 rows in 10,300 for that reason.
- `PinnedSlotLRU` counters exist only on the eager path. The graph arms use
  `NativePinnedSlotTable` in the C++ service and are not counted.
- VRAM hits are not counted directly; only VRAM miss rows (from the trace) are.
- "Advisory bytes" are zero by configuration: `env-full.sh` sets
  `SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=0` and `SGLANG_MOE_PREFETCH_MAX_CANDIDATES=0`.
  Promotions appear only as `bg` rows (rows read outside gathers); their
  count is not split from seeding.
- The first mirror arm of the first launch has only its driver JSON
  (`eager-results/first-mirror-arm-trace-lost.json`): my script named arms `<label>-<arm>`,
  so a repeated arm deleted its twin's trace. Fixed: arms are now `<label>-<index>-<arm>`.

## What was added

- `PinnedSlotLRU.stats()` and `tier_snapshot()` (`expert_host_tier.py`): hits (`touch`
  calls, per resident route), admissions, evictions, protected evictions, releases,
  occupancy and capacity per layer, plus the tier's route-level lookup hits/misses and
  populated bytes. `EngramRowCache.stats()` gains misses, evictions, filled rows and
  capacity (`engram_row_cache.py`).
- `CacheStatsSink`: with `SGLANG_DSV41_EXPERT_TRACE_PATH` set, both write time-stamped
  cumulative snapshots to `<trace>.cache-stats` (same monotonic clock as the trace).
  With it unset the cost is an `is not None` check per call plus integer increments.
- Counting changes no decision: `test_counting_changes_no_decision` runs the counted table
  and a verbatim copy of the uncounted one through 4000 random operations (assign with and
  without protection, touch, release, a pinned filter) and asserts equal returns, slot
  lists, eviction order and free list.
