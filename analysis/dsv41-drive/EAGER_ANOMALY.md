# Eager + mirrors "is slower": does not reproduce (2026-09-20)

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
