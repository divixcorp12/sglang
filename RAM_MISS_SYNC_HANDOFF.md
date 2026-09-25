# RAM-miss service: synchronization audit and the two follow-ups — handoff

Date: 2026-09-24. Branch `codex/nvfp4-expert-stream-main`, head **`9183637f51`**
(`feat(dsv41-baseline): recipe runs a 100 GiB NUMA-placed tier with two-phase piece
streaming`). Written from a read of the code only; nothing in this file has been run on a
GPU and no code was changed. Every `file:line` below was printed and checked at this head;
regenerate them with the greps in §3.2 and §7 if the branch moves again.

**This file was first written at `7cd00caaee` and rewritten after a rebase that brought
in piece streaming** (38 commits, `2ed5f91226`..`9183637f51`). The production path
changed underneath both items: the recipe now runs two-phase + piece streaming, the wait
side is a new 8-block stream kernel, and the packing pool receives one job per *piece*
rather than per row. Read `DSV41_REFERENCE.md` §24 first; it holds the measured numbers
this file leans on.

**Read `.claude/rules/divix01-run-protocol.md` before running anything on divix01.**
Every command below assumes it (push to `shared`, pull in a worktree, `PYTHONPATH`,
`taskset -c 0-63`, `PIPESTATUS`, `gpu-run.sh` for GPU work, crypto-c9 for GPU time).

Files this is about:

| file | role |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh` (1,854 lines) | device side: post / wait / lease-wait / two-phase W1, W2 / **stream kernel S** / ack / finalize kernels and their TVM-FFI wrappers |
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` (4,577 lines) | host side: `RowReader` (io_uring, bounce banks, sub-reads, pieces, packing, publishing), `RamTier` (tiers, leases, piece masks, demand/advice pumps), `RamThread` (service thread + watchdog) |
| `python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_pack_pool.h` (233 lines) | `PackPool` / `PackJob`: the packing worker threads |
| `python/sglang/kernels/ops/moe/exl3_ram_miss.py` | Python wrapper (`Exl3RamMissHost`, `start_thread` `:686`, trace schema 7) |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | production entry point (`Exl3RamMissRowBackend`, `:518-564`) |
| `benchmarks/dsv41_baseline/arm_env.py:139-143` | **the production recipe**: `PACK_WORKERS=8`, `ENABLE_RAM_MISS_LEASES=1`, `ENABLE_RAM_MISS_TWO_PHASE=1`, `RAM_MISS_HIT_WAIT_US=100`, `ENABLE_RAM_MISS_PIECE_STREAM=1` |

---

## 0. The one-paragraph state

The question that started this was "do we need all this synchronization in the C++
layer; shouldn't the GPU do all the waiting and the C++ just run flat out?" The design
already has that shape: the GPU does every wait (W1, S and the batched wait kernels spin
on pinned-memory words), the host never blocks against the GPU, and every remaining
host-side primitive is cross-processor *ordering* on pinned memory, not waiting. None of
it is removable. Two things came out of the audit, and they are independent:

- **A. The packing pool parks its workers on a condition variable between jobs, and
  under piece streaming a job is one piece: ~208 KB per worker, ~20-40 µs of copying,
  arriving every few hundred µs.** divix01's cores enter C6 (92 µs exit) after 276 µs
  idle, so the wake can now *exceed* the copy it precedes, and the last piece's wake is
  on every demand's critical path. `DSV41_REFERENCE.md` §24.2 already measures the host
  pack tail at **4.3 ms per step**; this is the mechanism most likely inside it. Section 2.
- **B. The device side pays `membar.sys` (the heaviest fence PTX has) in eight places
  where a release store or an acquire load already covers it, and is missing one fence in
  the one place it relies on a re-read for safety.** Cleanup with a formal argument per
  site; a few µs per MoE layer per step. Section 3.
- **C (noticed, not mine): W1 cannot exit early under piece streaming** because
  `lane_result_valid` hides the LOADING tag from it, so it spends its full 100 µs budget
  on every read layer — the 1.9 ms/token that `DSV41_REFERENCE.md` §24.7 lists as item 3.
  It is a small change in the same kernel B touches. Section 4.

Recommended order: measure A (one traced run, no code) → build A → C → B in the next GPU
window. Separate commits.

---

## 1. Who waits on whom (the audit result, so nobody re-derives it)

### 1.1 The production chain per MoE layer with a RAM miss

`post → W1 (lease_stream_hit_wait) → C1 (copy of hit lanes) → A1 (stage ack) → S (stream)
→ A2 (stage ack) → F (finalize)`, all in one stream, captured into the decode graph
(`test_g6_the_flag_on_chain_is_post_w1_c1_a1_s_a2_f_add`). Measured per token over read
layers (§24.3): W1 1.9 ms, C1 23.9 ms, S 60.7 ms, "and S is mostly waiting on the NVMe
read". The batched `wait` kernel and two-phase `rest_wait` (W2) are **not** on the
production path any more; they stay in the file for the flag-off modes.

### 1.2 Waiting

| waiter | waits on | how | where |
|---|---|---|---|
| W1 (`lease_hit_wait_body`, thread 0) | hit lanes' `RowResult.ready` (tag READY) | polls every unclaimed lane's result each pass, `__nanosleep(256)` between passes, bounded by `budget_ns` (100 µs in the recipe) and the D5 deadline | `exl3_ram_miss.cuh:693-716`; `kDemandDone` early-out `:708` |
| S (`lease_stream_kernel`, 8 blocks × 256 threads, each block independent) | each miss lane's `PieceMask` word (area P) and then `demand_done` | leader lanes (`tid < 8`) `ld.acquire.sys64` their lane's mask each pass; thread 0 polls `kDemandDone`; `__nanosleep(256)` only on a pass that copied nothing | `:1107-1360`; mask poll `:1207-1214`; `kDemandDone` `:1258` |
| batched / lease / W2 wait kernels (flag-off modes) | `demand_done` | `ld.acquire.sys` + `__nanosleep(256)` | `:417-422`, `:522-532`, `:860-869` |
| host service thread | `demand_head` / `advise_head` | `_mm_pause` spin for `spin_ns` (5 ms, `kernels/ops/moe/exl3_ram_miss.py:686`) after the last request, then `sleep_for(50us)` polls | `exl3_ram_miss_host.cpp:4392-4426` |
| host service thread (inside `read()`) | NVMe completions | `io_uring_submit_and_wait(1)` only when nothing is packable **and no piece job is in flight**; otherwise `io_uring_submit` + CQ scan (polling) | `:925` (`reap(ready \|\| c.packing > 0)`), `:1848` |
| host service thread | packing workers | `_mm_pause` on `jobs_[…].done()` in `quiesce` (`:1776-1790`); `collect_pieces` each loop turn (`:1728-1750`) | |
| packing workers | a posted job | **`work_cv_.wait` under `mutex_`** — item A | `exl3_ram_miss_pack_pool.h:163` |

There is no `cudaStreamSynchronize`, `cudaEventSynchronize` or `cudaDeviceSynchronize`
in the three files, and §24.3 confirms decode has no per-step blocking sync.

### 1.3 Ordering (load-bearing; do not remove)

| primitive | where | what breaks without it |
|---|---|---|
| `_mm_sfence()` before `store_release(page_ + kDemandDone)` | `host.cpp:2825-2826` | glibc `memcpy` uses non-temporal stores above its threshold; x86 does **not** order NT stores under a release store. The GPU would see `demand_done` and read slab bytes that have not landed. |
| `_mm_sfence()` before `publish_map` after `read()` (D11) | `:3774` | same, for the slot map |
| `_mm_sfence()` in `init_piece_words_locked` | `:3863` | the miss lanes' `PieceMask` words must carry the request's generation before any of its ready words is visible (`:3847-3849` comment) |
| `_mm_sfence()` in `grant_lane_group_locked` | `:3185` | this group's `RowResult` payloads before this group's ready words |
| `_mm_sfence()` in `bump_generation_locked` | `:3486` | slot generation visible before the first byte of a rewrite (LEASE_PROTOCOL.md 6.5) |
| `_mm_sfence()` in `handle_demand` before `set_status` | `:3885` | the record's status is the wait kernels' served/failed verdict |
| `_mm_sfence()` at the end of `PackPool::copy_chunk` | `pack_pool.h:194` | a worker's NT stores; the owner's sfence does not order another core's stores |
| `publish_piece` CAS with `__ATOMIC_RELEASE` | `host.cpp:612-620` | the owner acquired the piece's job (`done()`), so the GPU that acquires the bit sees the bytes; a relaxed CAS would break that chain |
| `atomic_thread_fence(acquire)` between payload and seq re-read in `read_record` | `:2623` | the seqlock re-check would be unordered against the payload loads |
| `__atomic_load_n/store_n` ACQUIRE/RELEASE on every flag word (`demand_*`, `advise_*`, `fatal`, `busy_seq`, ready/ack/terminal/probe/mask words, slot map) | `:2606-2624`, `:2787`, `:3516`, `:3858` | a plain store orders nothing against the payload |
| `PackJob::finished.fetch_add(release)` / `done()` acquire | `pack_pool.h:204`, `:79` | the owner's read of slab bytes and of `first_start`/`last_end` after `done()` |
| `ld.acquire.sys` / `st.release.sys` on the device | `exl3_ram_miss.cuh:150-168` | GPU stores to host-mapped memory are posted PCIe writes; two stores from one thread can land out of order without a fence or a release |
| `__threadfence_system()` as the seqlock **invalidate** fence | `:213`, `:332`, `:357` | store→store ordering; see 3.3 |
| `__threadfence()` (gpu scope) in S's commit | `:1320`, `:1324`, `:1331` | the classic last-block-done pattern: each block's `st.global.cg` copies before its count; the last block's fence before it reads `stream_abort` (`:1333`). Cheap, gpu scope, correct. |
| `ld.global.cv` for the piece bytes in S | `:980-990` | a LOADING lane's host bytes are written while S runs; `.nc` could serve a line cached before the piece was published (comment at `:978-979`, LEASE_PROTOCOL.md E1 amendment `:466`) |
| `__syncthreads()` in S | `:1185`, `:1195`, `:1218`, `:1233`, `:1303` | 256-thread block sharing `sh` (`StreamLanes`); every one separates a leader-only phase from an all-threads phase. Not removable. |

### 1.4 Host-thread coordination (fine as is)

- `RamTier::mutex_` guards the tier tables and `outstanding_` between the service thread
  and Python control calls. `serve()` holds it for short table scans; `read()` runs with it
  **not** held (`host.cpp:3760`). Uncontended in production.
- `PackPool::mutex_`: item A is about the *wait policy* behind it, not the mutex.

### 1.5 Intra-block on the GPU (simplification only)

`__syncthreads()` at `exl3_ram_miss.cuh:1373/:1392` (`lease_stage_ack_kernel`) and
`:1476/:1493` (`lease_ack_kernel`) reduce an 8-thread shared flag; one warp. A warp vote
replaces both barriers and the shared variable (3.2 #14). S's barriers are a different
matter (1.3).

---

## 2. Item A: packing-worker wake latency, now per piece

### 2.1 The mechanism, line by line, under piece streaming

1. `read()` (`host.cpp:850`) admits a batch; with piece streaming each row's two mirror
   parts become 4 sub-reads each (`kSubReads = 4`, `:63`; `split_part`), 8 sub-reads and 8
   pieces per row (`kPieces = 8`, `:64`; `row_geometry`).
2. A sub-read lands → `retire` → `land_sub_read` → `vet_pieces`: every piece whose
   dependency sub-reads have all landed is vetted, in the same loop turn.
3. `pack_one` → `dispatch_ready_pieces` (`:1653-1695`): each vetted piece gets **its own
   `PackJob`**, armed with `chunks = pack_split_` (`:1684`; `= pack_workers_` when unset,
   `:686-690`) and posted (`:1685`). So one piece → 8 chunks → 8 workers, ~1.66 MB / 8 ≈
   **208 KB per worker**, roughly 20-40 µs of copying each.
4. `PackPool::post` (`pack_pool.h:137-145`): take `mutex_`, push, release, `notify_all`.
5. Each worker (`run`, `:148-174`) has been parked in `work_cv_.wait` (`:163`) since the
   queue last emptied. Eight wake, each re-acquires `mutex_` to `claimed.fetch_add` a
   chunk, copies (`copy_chunk`, `:176-205`), `_mm_sfence`s, `finished.fetch_add`s.
6. The owner, polling (`reap(true)` since `c.packing > 0`, then `collect_pieces`,
   `:1728-1750`), sees `done()` and `publish_collected`s (`:1697-1726`): a release CAS
   sets the piece's bit in every `PieceMask` word naming the row (`publish_piece`,
   `:612-620`).
7. S's leader lane for that request lane acquires the mask word on its next pass
   (`cuh:1211-1214`), the block copies its slice (`:1220-1232`).

The wake path (step 5) is: futex wake × 8 → C-state exit × 8 → scheduler → `mutex_`
convoy × 8. It is paid **per posted piece that finds the workers parked**. Between
pieces the workers idle for whatever the drive takes to land the next sub-read.

### 2.2 The numbers and where each comes from

| quantity | value | source |
|---|---|---|
| divix01 CPU | Intel Xeon Gold 6154 @ 3.00 GHz, 72 CPUs, 2 NUMA nodes | `lscpu`, 2026-09-24 |
| cpuidle governor | `menu` | `/sys/devices/system/cpu/cpuidle/current_governor` |
| C-states on cpu0 (exit latency / target residency) | POLL 0/0, C1 2/2 µs, C1E 10/20 µs, **C6 92/276 µs**, none disabled | `/sys/devices/system/cpu/cpu0/cpuidle/state*/{name,latency,residency,disable}`; cpu0 has spent 905 s in C6 over 1.02 G entries |
| kernel cmdline | no `idle=`, no `intel_idle.max_cstate`, no `processor.max_cstate` | `/proc/cmdline` |
| row size | 13,320,192 B (§24.1; task6 measured 13,315,584 B of segment bytes) | `DSV41_REFERENCE.md:3271`; `task6-microbench/results.md` §1 |
| piece / chunk at 8 workers | ≈ 1.66 MB per piece, ≈ 208 KB per worker per piece | arithmetic on `kPieces = 8`, `pack_split_ = 8` |
| chunk copy time | roughly 20-40 µs (cold DRAM bounce → pinned slab, one core) | **estimate**; the 4:4 whole-row measurement gave 0.8-1.1 ms for a 3.3 MB chunk (`PACK_WORKERS.md:280-286`) |
| sub-read cadence | a sub-read is ≈ 1.66 MB; at a few GB/s per drive, pieces land a few hundred µs apart — right at C6's 276 µs residency | inference; the read-wall analysis (§24.4) says reads at credit 32 are four times smaller than before |
| share of demands with `rows_asked == 1` | 67.9% (pre-piece-streaming production) | `PACK_WORKERS.md:196` |
| **host pack tail per step** (last read completion → `done`) | **4.3 ms** with piece streaming (was 24-25 ms) | `DSV41_REFERENCE.md:3317-3318` |
| last piece publish → end of S | 186 µs p50 | `:3321` |
| S start after C1 | 3.1 µs p50 | `:3320` |
| owner extra CPU while workers pack | ~+14 ms per read at W8/c8 (owner polls instead of blocking) | `PACK_WORKERS.md:135-138` |

The 4.3 ms/step tail is the sum over the step's demands of (last sub-read cqe → vet →
post → **wake** → 8 × ~30 µs copies → collect → publish CAS → `demand_done`). With a few
tens of read layers per step that is on the order of 100 µs per demand, which is what a
C6 wake (92 µs) plus a 30 µs chunk copy plus a loop turn adds up to. **That is the
hypothesis, not a measurement**; 2.4 is the measurement. If it holds, option A moves the
per-step tail from 4.3 ms toward ~1-2 ms, i.e. 2-3 ms/token out of ~156 — the same order
as the W1 item (§4), and both are smaller than the read wall §24.4 describes. Worth doing;
not a headline.

### 2.3 What is NOT the problem (do not "fix" these)

- **`notify_all`** is right for N:N: every worker has a chunk of every job.
- **`mutex_`** is only contended at the wake instant; ~20 ns per claim once awake.
- **`_mm_sfence` at `pack_pool.h:194`** must stay (1.3).
- **`publish_piece`'s release CAS** must stay (1.3).
- **The "workers spin" remark** in `PACK_WORKERS.md:288,373` is loose: workers do not
  spin today; the extra CPU is the owner polling and the copies moving cores.
- **The owner's own wake** from `submit_and_wait(1)` between pieces is the same C6
  mechanism, present in inline mode too. Related follow-up, 2.8.

### 2.4 Measure first: what the schema-7 trace already gives

`row_pack_start_k` is the row's `pack_first` = the earliest `first_start` over its piece
jobs (`host.cpp:1739`), stamped at `copy_chunk` entry (`pack_pool.h:177`), i.e. **after**
the wake and the mutex claim. `piece_cqe[k][j]` is the clock at piece j's vetting: the
reap that landed its last dependency (`StageRecord` comment `host.cpp:186-193`; set in `vet_pieces`). So

```
first_piece_wake_delay(row k) = row_pack_start_k − min_j piece_cqe[k][j]
```

is the wake + claim delay of the first piece dispatched for that row: one sample per row.
Later pieces of the row are not individually stamped at job start (only `pack_last`), so a
per-piece number needs a small schema-8 addition: store each job's `first_start` /
`last_end` per piece next to `piece_publish` (the values are already read in
`collect_pieces` at `:1739-1740`). `overlap_timeline.py:203-206` still refuses
`ready_to_pack_ns` in worker mode; add a worker-mode-only `wake_delay_ns` output rather
than un-refusing the inline metrics.

Procedure:

1. One serving arm with the production recipe and the stage trace on
   (`host.enable_trace()` before `start_thread`, `srt/layers/moe/exl3_ram_miss.py:547-549`;
   `get_exl3_stream_trace().enabled`). Two sessions is enough — this is a per-row
   distribution.
2. Export the JSONL (schema 7, `exl3_stream_trace.py:56`); check `request.pack_workers`
   reads 8 and `request.piece_stream` reads 1, or the run is not what you think.
3. Compute `first_piece_wake_delay` per row from `row_pack[].start` and
   `pieces[].cqe` (expanded at `kernels/ops/moe/exl3_ram_miss.py:443-467`). Report p50 /
   p90 / max, and separately for single-row demands.
4. Also report per demand `done − max_k row_pack_end_k` (collect + publish + done latency)
   and `done − last extent cqe` (the whole host tail), so the 4.3 ms/step decomposes.

Prediction, stated so it can be wrong: wake p50 in the tens of µs, p90 ≈ 100 µs (C6 +
scheduler), a bimodal shape (C1E hits ≈ 10-20 µs, C6 hits ≈ 90-120 µs). **Decision rule:**
p90 < 10 µs → the C6 story is wrong, stop, do not build 2.5. p50 > 30 µs → build 2.5.

Free A/B without code: hold `/dev/cpu_dma_latency` at 0 (root; keep the fd open for the
arm's lifetime) or boot with `intel_idle.max_cstate=1` — both box-wide on a shared
72-core machine, so ask first — and repeat. If the wake collapses to single-digit µs and
the per-step tail moves with it, the mechanism is confirmed independently of any pool
change.

### 2.5 Recommended change: spin while a read is in service (option A)

Mirror the policy the service thread applies to itself (`host.cpp:4422-4426`: spin while
recently active, sleep otherwise), keyed on the signal the owner has for free — whether it
is inside `read()`.

**API.** `PackPool::set_active(bool)`. `read()` calls `set_active(true)` right after
`reset_pipeline()` (`host.cpp:889`) and `set_active(false)` from the `Quiesce` guard's
destructor (`:893-896`) after `quiesce()`, so every exit — return, fail, exception —
leaves the pool cold with every copy finished.

**Worker loop (sketch, not code to paste):**

```
while (true) {
  if (claim(&job, &chunk)) { copy_chunk(job, chunk); continue; }   // takes mutex_ only when posted_ moved
  if (active_.load(relaxed)) { _mm_pause(); continue; }             // hot: stay on the core, in C0
  std::unique_lock<std::mutex> lock(mutex_);
  work_cv_.wait(lock, [&] { return stop_ || count_ > 0 || active_; });
  if (stop_ && count_ == 0) return;
}
```

- `posted_`: a new `std::atomic<uint64_t>` the owner increments (release) in `post()`
  after the enqueue; a spinner keeps `seen_` and takes `mutex_` only when
  `posted_.load(acquire) != seen_`. Eight spinners never touch the mutex line otherwise.
- `active_`: `std::atomic<bool>`, written by the owner **under `mutex_`** in
  `set_active`, then `notify_all` when turning on. Read relaxed in the spin, under the
  mutex in the park predicate. That rules out a lost wakeup: cold→hot with workers parked
  → `set_active(true)` notifies; hot→cold while a worker sits between its `active_` check
  and `wait` → the predicate re-checks `count_ > 0` and `active_` under the mutex; a
  `post()` while cold → `notify_all` as today.
- `post()` keeps `notify_all`; glibc's broadcast with no waiters is a user-space check.
- `set_capacity` (`pack_pool.h:129-135`) and `shutdown` unchanged.

**Pinning rule (new invariant).** A spinning worker must not share a core with another
spinning worker. Pin worker *i* to the *i*-th core of `allowed_` ascending (today every
worker gets the whole mask, `:149`). Refuse when `CPU_COUNT(allowed_) < workers` (today
only at zero, `:95-97`). `worker_affinity(i)` then returns one core, so:

- `test_the_pool_pins_its_workers_to_the_allowed_cores_and_refuses_when_none_is_left`
  (`test_exl3_ram_miss_pack_workers.py:197`): expectation becomes "one core of allowed,
  distinct per worker".
- `test_a_worker_may_not_use_cores_64_to_71` (`:188`) holds as written.
- The owner is unpinned in production (`start_thread(cpu_core=-1)`; the production call
  at `srt/layers/moe/exl3_ram_miss.py:549` passes only `fatal_wait_s`). CFS moves a
  spinning unpinned owner off a pinned spinner's core, so this is not blocking; pinning
  the owner via `cpu_core=` is the clean version once someone picks the core.

**Cost.** Eight cores busy for the duration of each request's `read()` (a few ms with
piece streaming), nothing between requests. The owner already burns one core polling
during that window. Cores 64-71 are excluded (`pack_worker_cpus`, `pack_pool.h:84-89`).
The policy question: is 9 cores hot during every RAM-miss read acceptable beside
production's scheduler/tokenizer threads on `0-63`? Someone has to say yes.

**Why not a time window:** the gap before the *first* piece is the NVMe latency itself;
between pieces it is a few hundred µs, i.e. exactly the C6 residency. A window would have
to cover the whole read to help. Keying on `read()` entry/exit is the precise version.

**Tests that constrain the change** (all must stay green, in the three packing modes the
fixture selects): `test_a_failure_returns_only_when_no_copy_is_still_running` (`:259`),
`test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done` (`:299`),
`test_workers_copy_rows_concurrently_and_a_row_is_cut_across_them` (`:220`),
`test_a_row_cut_into_many_more_chunks_than_it_has_lines_is_byte_exact` (`:249`),
`test_no_worker_thread_exists_unless_asked_for_and_close_joins_them` (`:158`); the 30
piece-stream CPU tests in `test_exl3_ram_miss_piece_stream.py` (a job per piece, publish
after done, quiesce on failure); and the `PACK_WORKERS.md` "Safety properties" mutants
that touch the owner↔worker handoff.

New tests worth writing: (a) after `read()` returns the pool reports inactive and no
worker is spinning; (b) a `post()` while cold still completes (no lost wakeup); (c)
`workers > CPU_COUNT(allowed)` throws with every thread joined.

### 2.6 Alternative: per-worker mailbox, N:N only (option B)

Worker *i* always copies chunk *i*: the owner writes the job pointer into worker *i*'s
own cache line (SPSC ring of `set_capacity`'s depth), the worker spins on its line. No
queue, no `mutex_`, no convoy. Second because it drops the `split ≠ workers` modes the
three-mode fixture and `PACK_WORKERS.md` measured. Build A; if the post→first-chunk delay
after A is still dominated by the mutex convoy (it should not be, with `posted_`), B is
next.

### 2.7 Alternative: C-state policy (option C)

`intel_idle.max_cstate=1` or a held `/dev/cpu_dma_latency`: every core wakes in 2-10 µs.
Box-wide, power, shared machine. The A/B in 2.4, not the fix.

### 2.8 Related, same mechanism, not measured

Between pieces, when no job is in flight, the owner blocks in `io_uring_submit_and_wait(1)`
(`host.cpp:925`, `:1848`) and its core can reach C6 too; the completion interrupt then
pays the same 92 µs before the owner sees the CQE. Under piece streaming `c.packing > 0`
is true for much of a read, so the owner mostly polls already; the exposure is the gaps.
Fix shape if it measures: while active, poll `io_uring_peek_cqe` with `_mm_pause` instead
of blocking. Its number is not in the trace (`extent_cqe` is the reap stamp, not the
kernel's completion time); it needs `bpftrace` on io_uring completion → wake.

---

## 3. Item B: device-side fence pass (`exl3_ram_miss.cuh`)

### 3.1 PTX facts this rests on (verified, not remembered)

Compiled on the laptop (`nvcc -arch=sm_90 -ptx`, `/usr/bin/nvcc`) on 2026-09-24:

```
__threadfence_system();                                   →  membar.sys;
asm("st.release.sys.global.u32 [%0], %1;" ...)            →  st.release.sys.global.u32
*reinterpret_cast<volatile unsigned*>(p) = v;             →  st.volatile.global.u32
```

- `membar.sys` is `fence.sc.sys`: sequentially consistent, system scope, the strongest
  fence PTX has. The CUDA guide defines `__threadfence_system()` as
  `cuda::atomic_thread_fence(memory_order_seq_cst, thread_scope_system)`.
- `st.release.sys` orders **every prior memory operation of the thread** before the
  store, at system scope. A `membar.sys` immediately before it adds nothing.
- `ld.acquire.sys` orders **every subsequent memory operation of the thread** after the
  load. A `membar.sys` immediately after it adds nothing for those operations. It does
  **not** order *prior* loads before itself — the bug in #13.
- `bar.sync` (`__syncthreads`) orders at CTA scope and composes with a thread's
  system-scope acquire: host release → thread 0's `ld.acquire.sys` → barrier → other
  threads' later loads is a valid causality chain. That is why S's `membar.sys` at
  `:1259` is not needed for the other 255 threads either.
- `st.volatile` / `ld.volatile` are relaxed at system scope under the model: the compiler
  will not reorder them among themselves, the hardware may.
- `fence.acq_rel.sys` exists (sm_70+) and is sufficient for every fence this file keeps;
  whether SASS makes it cheaper than `membar.sys` is unknown (3.5).

### 3.2 Site table

Line numbers verified against `9183637f51` on 2026-09-24. Regenerate with:

```
grep -n "__threadfence_system()\|__threadfence()\|st_release_sys(\|st_release_sys64(\|ld_acquire_sys(page + kDemandDone)\|__syncthreads()\|words\[kRecSeq / 4\] = \|lane_result_valid(\|^__global__" python/sglang/kernels/jit/csrc/moe/exl3_ram_miss.cuh
```

Kernel entry points: `post` `:276`, `wait` `:395`, `lease_wait` `:486`,
`lease_hit_wait_body` `:636` (shared by `lease_hit_wait_kernel` `:748` and the production
W1 `lease_stream_hit_wait_kernel` `:775`), `lease_rest_wait` `:818`, `lease_stream_kernel`
(S) `:1107`, `lease_stage_ack` `:1362`, `lease_finalize` `:1406`, `lease_ack` `:1466`.
Helpers: `ld_acquire_sys`/`st_release_sys`/64-bit twins `:150-168`, `write_record`
`:205-230`, `raise_fatal` `:232-234`, `lane_result_valid` `:244-272`, `publish_terminal`
`:472-478`, `stream_admit` `:1067-1090`. Launches: ack kernels `:1612`/`:1718`
(`kLeaseLanes` threads), S `:1824` (`kStreamBlocks` × `kStreamThreads` = 8 × 256).
**P** = on the production path (piece streaming); **off** = flag-off modes only.

| # | kernel | site | now | change | argument |
|---|---|---|---|---|---|
| 1 | `write_record` (P) | `:213`, after `words[kRecSeq / 4] = 0u` (`:212`) | `membar.sys` | **keep** | seqlock invalidate: the zero must be visible before any payload store. Store→store needs a fence. |
| 2 | `write_record` (P) | `:228-229`, `membar.sys` then `words[kRecSeq / 4] = seq` (volatile) | fence + volatile store | **`st_release_sys(record + kRecSeq, seq)`**, drop the fence | identical ordering, one instruction, and it is what makes #7/#8 legible. Host reads it with `__atomic_load_n(ACQUIRE)` (`host.cpp:2607`, `:2624`, re-check `:2787`). |
| 3 | `post` (P) | `:332`, hot page, after `*hot = 0` (`:331`) | `membar.sys` | **keep** | as #1 |
| 4 | `post` (P) | `:343`, before `st_release_sys(hot, seq)` at `:344` | `membar.sys` | **remove** | the release store covers every prior store, including the bitmap bytes |
| 5 | `post` (P) | `:357`, lease request, after `gen = 0` (`:356`) | `membar.sys` | **keep** | as #1 (LEASE_PROTOCOL.md 6.3 `:547-560` specifies this shape) |
| 6 | `post` (P) | `:362`, before `st_release_sys64(request + kLeaseLrGen, …)` at `:363` | `membar.sys` | **remove** | the release store covers the payload |
| 7 | `post` (P) | `:366`, before `st_release_sys(page + kDemandHead, seq)` at `:367` | `membar.sys` | **remove** (after #2) | the record's seq store is now a release store; `demand_head`'s release store orders it and everything before it |
| 8 | `post` (off: `advise == 0` returns at `:375`) | `:391`, before `st_release_sys(page + kAdviseHead, advice)` at `:392` | `membar.sys` | **remove** (after #2) | same as #7 for the advice ring |
| 9 | `wait` (off) | `:430`, after the poll loop `:417-422` exits with `reached(done, seq)` | `membar.sys` | **remove** | `done` came from the last `ld_acquire_sys(page + kDemandDone)` (`:421`, or `:417`); the status load `:432` and slot-map loads `:448` are subsequent. See 3.6 for the cross-kernel question. |
| 10 | `lease_wait` (off) | `:545`, after the poll loop `:522-532` | `membar.sys` | **remove** | same; status `:547`, `lane_result_valid` at `:573` |
| 11 | `lease_rest_wait` W2 (off) | `:881`, after the poll loop `:860-869` | `membar.sys` | **remove** | same; status `:883`, `lane_result_valid` at `:916` |
| 12 | **S** `lease_stream_kernel` (P) | `:1259`, thread 0, after `reached(ld_acquire_sys(page + kDemandDone), seq)` at `:1258` | `membar.sys`, once per block per request (8 per layer) | **remove** | thread 0's status load `:1261` and mask re-reads `:1270-1282` are subsequent to its acquire; the other threads' later copies are ordered through the `__syncthreads` at `:1303` (3.1, `bar.sync` bullet) |
| 13 | `lane_result_valid` (P via W1 `:697` and S `:1079`; off via `:573`, `:916`) | `:244-272`: `ld_acquire_sys64 ready` · volatile payload · `ld_acquire_sys64 again` (`:259`) · compare | no fence between payload and re-read | **add a fence between the payload loads and `:259`** — in the batched shape of 3.4 wherever it is called in a loop | an acquire *load* orders later ops after itself, not the earlier payload loads before itself; a torn payload can pass the belt. Host `read_record` gets this right (`host.cpp:2623`). |
| 14 | `lease_stage_ack` (P), `lease_ack` (off) | `__shared__ int` `:1371`/`:1474`; `__syncthreads()` `:1373`/`:1392` and `:1476`/`:1493` | 2 barriers + shared flag per kernel | `bool v = false; if (entry < n && entry < kLeaseLanes) { …; v = !consumed; } if (__any_sync(0xFFu, v) && threadIdx.x == 0) { … }` | one warp of 8 threads (`__launch_bounds__(kLeaseLanes, 1)`, launched with 8 threads). Simplification only. |

Sites that already have the right shape (listed so nobody "fixes" them):
`publish_terminal` (`:472-478`: volatile mask + reason, then `st_release_sys64(gen)` at
`:477`); `raise_fatal` (`:232-234`); S's StreamProbe `st_release_sys64` (`:1242`); the
ack kernels' `ld_acquire_sys(SlotGen)` → `st_release_sys64(LaneAck)` pairs
(`:1384-1389`, `:1484-1489`); every `ld_acquire_sys(page + kFatal)` /
`ld_acquire_sys(lease + kLeaseHeaderShutdown)` early-out; S's three gpu-scope
`__threadfence()` (`:1320`, `:1324`, `:1331`) and five `__syncthreads()` (1.3).

Net on the production path per layer: post 7 `membar.sys` + 3 release stores → 3 + 4;
S 8 `membar.sys` (one per block) → 0; plus the batched fences #13 adds (3.4).

### 3.3 Why the three invalidate fences stay

Writer: `seq = 0; F; payload…; st.release seq`. Reader (host `read_record`):
`s1 = acquire(seq); payload; fence(acquire); s2 = load(seq); accept iff s1 == s2 ==
expected`. Without `F` the writer's payload stores may become visible before its zero:
for a lapped record (host expecting `S`, device rewriting the slot for `S + 16`) the host
reads `s1 == S`, the new payload lands half-way, the host reads `s2 == S` because the
zero has not landed → torn record accepted. A release on the zero store would order it
after *earlier* stores, the wrong direction. `fence.acq_rel.sys` would do (its release
half orders the prior zero before the subsequent payload); leave that downgrade for a
separate, measured commit (3.5). Lapping is structurally rare (stream order plus the
sticky fatal word), which is why the seqlock is a belt — and a belt that is formally
wrong is worse than none, because it looks like protection.

### 3.4 The one design decision: where #13's fence goes

A fence inside `lane_result_valid` costs one `membar.sys` **per call**, and two
production callers call it in a loop:

- **W1** (`lease_hit_wait_body` `:693-716`) calls it once per unclaimed lane per pass and,
  under piece streaming, runs every pass of its 100 µs budget (§4). Up to 8 lanes × ~8
  passes = ~64 fences per layer. Not acceptable inline.
- **S** (`stream_admit` `:1067-1090`, from `:1207`) calls it from each leader lane thread
  (`tid < 8`) until that lane is admitted. Eight threads of one warp executing
  `membar.sys` is one warp instruction, so this is one fence per pass per block until
  admission — LOADING lanes are granted at reservation, so usually the first pass. Cheap.

Preferred shape for W1 (and `lease_wait` `:571-579`, where it is one call per lane after
`done`): hoist the seqlock across the lanes.

```
// per pass, over the unclaimed lanes
for each unclaimed lane i:  ready[i] = ld_acquire_sys64(result_i + kLeaseRrReady); payload_i = volatile loads;
__threadfence_system();                       // ONE fence: every payload load above precedes every re-read below
for each unclaimed lane i:  again = ld_acquire_sys64(result_i + kLeaseRrReady);   // relaxed would do
                            valid_i = tag/gen match on ready[i] && again == ready[i] && expert/slot checks;
```

For S, leave `stream_admit` calling `lane_result_valid` with the fence inside: the per-
thread form is the batched form there. The `ready_seen` / `accept_loading` / `loading`
contract (`:238-243`, `:251-253`) must survive any restructuring — it is what §4 depends
on too.

### 3.5 What this pass does not do

- No `membar.sys` → `fence.acq_rel.sys` downgrade on #1/#3/#5. Formally sufficient;
  unknown whether SASS distinguishes them. Separate commit, measure with the `g` harness
  (`analysis/dsv41-drive/task6-microbench/README.md`, variant `active_p4`).
- No change to the host side. Every host fence is load-bearing (1.3).
- No change to `PackPool` (item A) and no W1 early exit (§4).

### 3.6 Cost expectation and the cross-kernel question

Cost: unmeasured. `LEASE_PROTOCOL.md:559` and `[OPEN 5]` say the same about the two
fences the lease request added. Measured neighbours (task6, `results.md` §2): a serial
`ld.acquire.sys` from the pinned request page costs 711 ns; the `active_p4` stage
(four polls + `membar.sys` + `st.release.sys` + a 4 KiB copy) 7.26 µs. A `membar.sys` is
plausibly 1-3 µs. Removing four from post and eight from S per layer is perhaps 10-30 µs
per layer per step, against a 156 ms step. The value is that afterwards every remaining
fence has one sentence next to it saying what it orders.

The cross-kernel question (#9-#12): thread 0 acquires `demand_done`; the copy kernels and
the fused MoE that read the slab run later in the same stream. Is one thread's acquire
enough for *their* loads? Under the PTX model yes: host stores happen-before the host's
release of `demand_done`, which synchronizes-with thread 0's acquire; kernel completion →
next kernel start is causality order; the later kernels' loads happen-after the host's
stores. The `membar.sys` adds nothing to that chain. Empirically this is what
`test_a_rewritten_slot_is_read_fresh` and
`test_many_rewritten_slots_are_read_fresh_without_a_host_sync[graph]`
(`test/manual/dsv41/test_exl3_ram_miss_cuda.py:153`, `:224`) check, and the plan that
specified the fence (`docs/superpowers/plans/2026-09-19-dsv41-phase3b-optionC.md:187`)
cites "3a (Facts)" as the evidence. S additionally reads a LOADING lane's bytes with
`ld.global.cv` (`:980-990`), which never serves a cached line. If a reviewer is nervous
about #12 alone, keeping it as a labelled belt is defensible; do not keep #4/#6/#7/#8 on
the same grounds — those are on the *writer* side and a release store is by definition
the fence.

### 3.7 Verification plan, and the gap in it

**CPU, on divix01, before and after, same command, counts diffed:**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 \
  /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels \
  -q -p no:randomly -k "ram_miss or lease or piece" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Last recorded: "kernels plus MoE RAM-miss 1,474 passed and 0 failed; piece-stream CPU
tests 82/82" (`DSV41_REFERENCE.md` §24.1, at merge). `test_exl3_ram_miss_device_args.py`
and `test_exl3_lease_block.py` parse the `.cuh` constants with `+ - *` only (see
`d192447606`, "write kAllPieces in decimal so the layout check can parse the device
source") — do not turn a constant into an expression. These prove the host and layout did
not break; **they do not exercise a single device fence.**

**GPU, on divix01, under `gpu-run.sh` (takes `cc-gpu.lock`, pins 32-63), GPU time via
crypto-c9 first** (production holds ~25-30 GiB on the card):

```bash
PYTHONPATH=$PWD/python taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
  test/manual/dsv41/test_exl3_two_phase_parity_cuda.py test/manual/dsv41/test_exl3_two_phase_timing_cuda.py \
  test/manual/dsv41/test_exl3_two_phase_failure_cuda.py test/manual/dsv41/test_exl3_piece_stream_cuda.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```

Last recorded: two-phase GPU 67/67, piece-stream GPU 22/22 (§24.1). All six files are
`skipif` on `torch.cuda.is_available()`, so a run without a GPU reports green by skipping
— read the skip count. Print `sglang.__file__` first (interpreter trap). The cases that
bear on the fences specifically: `test_a_rewritten_slot_is_read_fresh`,
`test_many_rewritten_slots_are_read_fresh_without_a_host_sync[graph]`,
`test_g11_a_late_last_publish_is_judged_on_the_masks_reread_after_kdemanddone` (S's
re-read after the acquire — the site #12 touches), `test_g10_without_an_abort_a_thousand_replays_all_commit`
(S's commit fences), `test_g9_*` (LOADING validation), and the lease-kernel class's
`test_a_lane_that_fails_validation_refuses_the_whole_request_and_publishes_a_terminal`.

**The gap, stated plainly.** No test can catch a *missing* device-side fence except by
luck: `exl3_ram_miss_seqlock_stress` is host-vs-host, and
`test_the_seqlock_reader_never_accepts_a_torn_record` (`test_exl3_ram_miss_tier.py:355`)
drives the host reader against a host writer. Each row of 3.2 stands on its PTX-model
sentence, which is why there is one per row. A reviewer should check the sentences.

---

## 4. Item C (noticed, the reference's item 3): W1 cannot exit early under piece streaming

`DSV41_REFERENCE.md:3334-3335`: "W1 always runs out its budget: on every read layer W1
spends its full 100 µs budget (104 µs p50), about 1.9 ms/token." §24.7 item 3: "Let W1
exit early." The cause is visible in the code:

- `lease_hit_wait_body` (`:693-716`) breaks when `taken == planned_count` (every lane
  claimed), on `kDemandDone`, on the budget, the deadline, or fatal/shutdown.
- Under piece streaming every miss lane is granted at reservation under tag **LOADING**
  (`host.cpp:3139-3195`, `tags[lane]`), and W1 calls `lane_result_valid` with
  `accept_loading = false` (`:697-699`), which reports `ready_seen = false` for a LOADING
  lane (`:262-264`). So W1 cannot tell "not yet published" from "published as LOADING,
  i.e. S's lane, never mine", and it keeps polling a lane that will never turn READY until
  the budget runs out. `kDemandDone` cannot save it either: it is stored only after
  `read()` returns, tens of ms later.

Fix shape: pass the `loading` out-parameter (`:253`) from W1's call and treat a
LOADING lane as settled — break when every unclaimed lane is either claimed or seen
LOADING. A hit lane is READY at reservation too (the loading grant covers "every lane, hit
and miss, in the one call", `host.cpp:3083-3085`), so W1 would typically finish on its
first or second pass instead of its ~8th. That is ~1.9 ms/token by the reference's own
number, from a handful of lines, and it removes most of the fence multiplier 3.4 worries
about. `test_t10_all_miss_w1_stays_bounded` and `test_t11_all_hit_s_copies_and_acknowledges_nothing`
(`test_exl3_piece_stream_cuda.py:689`, `:717`) are the constraints; a new test should
assert W1's pass count on an all-LOADING request is ≤ 2.

Not mine to prioritise; it is listed because B and C edit the same loop and should not
be two conflicting commits.

---

## 5. Order of work and commit shape

1. **A, measurement** (no code): 2.4. One traced arm, one script. Record wake p50/p90/max
   and the per-demand tail decomposition in `PACK_WORKERS.md` under a dated heading, plus
   the decision.
2. **A, change** (one commit: `pack_pool.h` + `read()` entry/exit + the two affinity
   tests): 2.5. Re-run the traced arm; the wake delay should drop to single-digit µs and
   the per-step host pack tail (§24.2's 4.3 ms) should drop with it. If it does not, the
   C6 story was wrong and the spin should be reverted (it costs eight cores).
3. **C** (one commit on `lease_hit_wait_body` + one GPU test): §4.
4. **B** (one commit on `exl3_ram_miss.cuh` only): 3.2 #2, #4, #6-#14 and the 3.4 hoist.
   CPU suite before/after, GPU suite in the next window. Keep #1/#3/#5 as `membar.sys`.
5. **B'** (optional): `fence.acq_rel.sys` on #1/#3/#5 with a `g`-harness measurement.
6. **2.8** if 2.4's wake number is small and the tail is still there.

A, C and B touch different code and can be reviewed independently. Do not fold B into A's
commit: A is a behaviour change with a measurement behind it; B is a formal cleanup whose
only check is the argument.

---

## 6. Open questions

- **Policy:** nine cores hot during every RAM-miss read (2.5).
- **Which core for the owner**, if pinned (`start_thread(cpu_core=)`).
- **Is `fence.acq_rel.sys` cheaper than `membar.sys` on sm_120?** Measurable, unknown.
- **The read wall** (§24.4) is the bigger number: A and C together are ~4 ms/token of a
  156 ms step. Neither should displace §24.7 items 1-2.
- **Does W1's early exit change S's admission timing?** S admits LOADING lanes on its
  first pass regardless of W1; the chain order is unchanged. Worth one look at
  `test_g6_the_flag_on_chain_is_post_w1_c1_a1_s_a2_f_add` after C.

---

## 7. Anchors used above (for the next regeneration)

`grep -n` targets, all verified at `9183637f51`:

- host: `kSubReads`/`kPieces`/`kPieceAlign` `:63-66`; `piece_word` `:604`; `publish_piece`
  `:612`; `set_pack` `:686`; `set_piece_stream` `:698`; `read(` `:850`; `reset_pipeline();`
  `:889`; `struct Quiesce` `:893`; `reap(ready || c.packing > 0)` `:925`; `size_jobs` `:1097`;
  `dispatch_ready_pieces` `:1653`; `job.arm(` `:1684`; `pool_->post(&job)` `:1685`;
  `publish_collected` `:1697`; `collect_pieces` `:1728`; `quiesce` `:1776`;
  `io_uring_submit_and_wait` `:1848`; `PackJob jobs_[` `:1906`; `read_record` `:2606`;
  `pump_demand` `:2768`; `store_release(page_ + kDemandDone` `:2826`;
  `grant_lane_group_locked` `:3139`; `retire_leases` `:3210`; `bump_generation_locked`
  `:3481`; `publish_map` `:3516`; `serve(` `:3609`; `init_piece_words_locked` `:3850`;
  `handle_demand` `:3870`; `RamThread::run` `:4392`; `_mm_sfence` at `:2825`, `:3185`,
  `:3476`, `:3486`, `:3774`, `:3863`, `:3885`.
- pool: `done()` `:79`; `pack_worker_cpus` `:84`; `CPU_COUNT` `:95`; `set_capacity` `:129`;
  `post(` `:137`; `notify_all` `:144`; `run()` `:148`; `setaffinity` `:149`;
  `work_cv_.wait` `:163`; `copy_chunk` `:176`; `_mm_sfence` `:194`; `finished.fetch_add`
  `:204`.
- python: `start_thread` `kernels/ops/moe/exl3_ram_miss.py:686`; recipe
  `benchmarks/dsv41_baseline/arm_env.py:139-143`; env defaults
  `python/sglang/srt/environ.py:1837-1871`; production wiring
  `srt/layers/moe/exl3_ram_miss.py:518, :541-549, :564`.
- docs: `DSV41_REFERENCE.md` §24 `:3260`, §24.2 `:3291`, pack tail `:3317-3318`, §24.3
  `:3343`, §24.7 `:3530`; `LEASE_PROTOCOL.md` E1 amendment `:466`, 6.3 `:547`, OPEN 5
  `:559`, 6.4 `:564`, 11.4 `:1053`; `PACK_WORKERS.md` (unchanged by the rebase)
  `:36, :130-138, :188, :196, :209, :280-288, :373`.

## 8. Reading list, in order

1. `.claude/rules/divix01-run-protocol.md`.
2. `DSV41_REFERENCE.md` §24 — what piece streaming is, its measured verdict, the trace,
   the read wall, and the next-work list this file slots into.
3. `docs/superpowers/plans/2026-09-24-dsv41-piece-streaming.md` — the design S implements.
4. `analysis/dsv41-drive/PACK_WORKERS.md` — the pool's design, the safety-property
   mutation table, and the warning about worker-mode traces (which 2.4 turns into a
   measurement).
5. `analysis/dsv41-drive/LEASE_PROTOCOL.md` §6.3, §6.4 (do not fuse the ack), the E1
   amendment (`.cv`, LOADING), §11.4, `[OPEN 5]`.
6. `analysis/dsv41-drive/task6-microbench/results.md` §2 — the only measured
   system-scope costs.
7. `CLAUDE.md` "Nsight Systems traces" — before reading any trace for kernel durations.
