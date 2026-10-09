# Drive-aware reads for the DSV4.1 EXL3 expert-row reader: design

Worktree `/Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin`, branch `codex/dsv41-ram-prefetch-margin` @ `214344b454`.
All paths below are relative to `python/sglang/kernels/jit/csrc/moe/expert_stream/host/` unless they start with
`python/`, `test/` or `DSV41_`. This is a design only; nothing was built or run.

## 0. Summary

- **The split is not just a host decision.** The device's stream-copy kernel indexes a piece-run table that was computed
  once, at setup, from the same fixed extents: `ffi_exports.h:133-194` (`piece_runs` calls `row_geometry` for every
  `(row, expert)`), consumed at `row_copy_kernels.cuh:186`. If a row's split changes, its piece cuts change too, and
  the device then copies the wrong byte ranges. **So (3) cannot switch between split tables such as 1:1:1 and 1:0:1:**
  their cuts differ (3 reading parts give 2 sub-reads each, 6 pieces; 2 parts give 4 each, 8 pieces;
  `reader_base.h:85-99`).
- **What can move is the drive behind each piece.** All roots hold byte-identical images with identical layouts. The
  builder gives every part the same offset formula (`expert*stride + lo`, `python/.../exl3_ram_miss.py:258-261`), and
  `tables_from` already requires equal file sizes and equal bases across a row's parts (`row_tables.h:194-211`). So any
  sub-read `(offset, length, dest)` can be read from any root by changing only its file index. Pieces, deps, device
  runs, the CPU-expert row-landed path and the lease piece words all stay as they are. This one mechanism, "choose the
  root per sub-read at submit time", serves both changes:
  - (3) is per-sub-read root choice for demand reads.
  - (2) is "every sub-read of a speculative row goes to one gated root, one piece at a time".
- **(2) needs a second reader per NUMA group.** `ReaderCore::read()` is synchronous and single-owner
  (`reader_core.h:72`, `:399-510`). That is why the speculative read holds the group's `turn` mutex for a whole row
  (`ram_tier.h:1448`, `:2363`, `:2658-2670`). A per-piece yield inside one `read()` call is not possible, so
  "without the exclusive turn" means the speculative thread gets its own `RowReader`, with its own io_uring ring, over
  the same Tables.
- **Order:** (0) the 16:10:16 weights, a setting already recommended in §33.16. (1) Shared per-drive in-flight
  accounting, observe-only. (2) Change (3), dynamic demand roots. (3) Change (2), the drive-aware speculative reader.
  (3) is moderate risk and changes no threading. (2) is the high-risk one: threading, pool protocol and tests.

## 1. Where roots and extents are decided today

| Stage | Where | What is fixed |
|---|---|---|
| Weights | `python/sglang/srt/environ.py:501-505`; `python/.../exl3_expert_format.py:413-460` | one weight per root |
| Split | `python/.../exl3_read_split.py:87-103` (`ReadSplit`), `:118-125` (`StaticSplitPolicy`) | the same page split for **every** row: `policy.plan(image.row_stride)` once (`exl3_ram_miss.py:239`) |
| Extents | `exl3_ram_miss.py:252-261` | `extents[row, e, part] = (row*parts+part, e*stride+lo, hi-lo, lo)`; file `row*parts+part` is root `part`'s image |
| Validation | `row_tables.h:107-119` (parts tile the image in part order), `:194-211` (equal size and base across parts) | |
| Sub-reads and pieces | `piece_geometry.h:65-114` (`row_geometry`): parts → `split_part` (`:25-33`) → sub-reads in file order → piece j = sub-read j's dest range mapped into segments, deps bitmask | `kPieces=8`, `kSubReads=4` (`reader_base.h:72-73`) |
| Device copy | `ffi_exports.h:133-194` writes `runs[L,E,8,S,2]` once; `row_copy_kernels.cuh:173-191` copies piece j's runs when bit j is set | setup-time table |
| Admission | `reader_core.h:1049` `plan_pieces` → `:1112-1119` (recomputes the same geometry into `piece_runs_` per slot) → `:1125-1165` `queue_sub_reads` (descriptor index `(slot*parts + part)*kSubReads + k`, `:1133`; `sub_reads_[index] = g.sub[s]`, file included, `:1135`) | |
| Prepare | `reader_core.h:1231-1307` `refill`: first preparation runs `plan_legs` (`:1237`), which cuts the read into legs by **its file's** device limits (`:900`, `limits_[d.read->file]`). The fd is `fds_[d.read->file]` (`:1254`) | |
| Land/vet/publish | `:1168-1223` (vet by `deps` and `sub_dest/sub_done`); `row_reader.h:189-212` publishes vetted pieces | root-independent |
| Drive identity | `reader_core.h:286-296`: drives are distinct `st_dev`, used **only by the trace** (`drive_extents` `:1090-1092`, `:1151-1153`; `drive_bytes` `:1531-1532`) | |

Without piece streaming, one read is issued per part straight from `t_.extents` (`reader_core.h:1072-1088`; `d.read`
points into the const table). Production always runs piece streaming: `ram_tier.h:142` sets
`set_piece_stream(true)` for every group. Both changes below are therefore scoped to piece-stream mode. With the flag
off they keep today's behaviour.

## 2. The shared per-drive accounting (prerequisite for both changes)

**Why the reader must count.** For polled reads, `/sys/block/*/stat` `in_flight` reads 0 (§33.16 item 3). There are
also two NUMA groups, each with **its own ring over the same three drives** (`numa_group.h:33`; `ram_tier.h:1482-1512`
builds one reader per group from the same Tables). The counts must therefore live at **tier scope**, shared by every
reader, not per reader.

**Structure (new header `drive_load.h`).**

```c++
struct alignas(64) DriveSlot {               // one cache line per drive: both groups' threads write it
  std::atomic<int64_t> demand_bytes{0}, spec_bytes{0};
  std::atomic<int32_t> demand_reads{0}, spec_reads{0};   // sub-reads in flight
};
struct DriveLoad { DriveSlot d[kMaxDrives]; int drives; double rate[kMaxDrives]; int cap[kMaxDrives]; };
```

- **Key: the root index, not `st_dev`.** Root `q` is `file % parts` (`exl3_ram_miss.py:258`). The test fixtures put every
  "root" under one `tmp_path` (`python/sglang/test/dsv41_ram_miss_fixtures.py:129-146`, roots `<source_root>_images<i>`),
  so an `st_dev` key would fold them into one drive and leave the policy untestable in CI. `open()` already collects
  `st_dev` (`reader_core.h:286-291`), so it should warn when two roots share a device.
  `kMaxDrives = 4` (`reader_base.h:139`) bounds the root count.
- **Where it counts.** Increment when a descriptor's first leg becomes Inflight in `refill` (`reader_core.h:1277`):
  add the leg's remaining bytes and +1 read on the descriptor's first leg. Decrement when legs are reaped in `process`
  (`:1448`, per leg's bytes). On failure every exit drains (`drain` `:1670-1673`; `Quiesce` `:451-465`), and drained
  CQEs never reach `process`. So each reader keeps a **local mirror** `mine_[root]`, and `drain()`/`reset_pipeline()`
  (`:817-824`) subtract `mine_` from the shared slots and zero it.
  - Invariant: shared = Σ over readers of `mine_`.
  - Invariant: every `read()` return leaves `mine_ == 0`, which matches the existing "every return leaves the ring
    empty" rule (`:372-373`).
- **Cost.** About 4 relaxed RMWs per sub-read: 6-8 sub-reads per row, some cross-socket. That is a few µs per row
  against 1-4 ms reads. The counters are functional state, so they are present in ProdBuild, unlike `ReaderMetrics`
  (`:618-628`). The hot-path golden, shim and stress tests (`test_expert_stream_hotpath_*.py`) must be re-pinned: the
  change adds no syscall and no allocation, only atomics.
- **Observe-only first.**
  - Export `drive_load()` over FFI with the instantaneous counts and time-integrated occupancy. Accumulate
    `∑ in_flight·dt` per root on each change, read with one clock per reap, which already exists (`returned`,
    `reader_core.h:1347`). This gives a reader-side `io_ticks` replacement for `analysis/dsv41-drive/dspark/drive_busy.py`.
  - Add the per-root counts to `StageRecord` beside `drive_bytes/drive_extents` (`reader_base.h:270-272`).

## 3. Change (3): dynamic per-drive root choice for demand reads

### Mechanism

1. **Mirror file map at open.** Build `alt_file_[file][q]` = the file of the same row on root `q`. The candidate is
   `(file - file%parts) + q`. Validate it with `source_paths[f]` equality (`row_tables.h:51`) and equal
   `file_sizes[f]` (already enforced, `row_tables.h:204`, and against disk at `reader_core.h:280-285`).
   - Refuse dynamic mode if any row is not a full mirror set.
   - A root with weight 0 still has its file listed (`exl3_ram_miss.py:262`), so it stays selectable. Only a dynamic
     cap of 0 excludes it.
2. **Choose at first preparation, not at admission.** In `refill`, before `plan_legs` (`reader_core.h:1237`):
   `if (dynamic_ && d.legs == 0) sub_reads_[index].file = alt_file_[...][choose_root(bytes)]`.
   - Late binding matters. The queue is FIFO under credit (`:1231-1250`). A batch of up to 8 rows × 6 sub-reads waits
     for credit, and the queue depth, by default "16 * parts", counts **legs** (`uring_options.h:34`;
     `reader_core.h:725-735`). So choosing when the sub-read is actually issued sees the real load.
   - `plan_legs` then cuts by the chosen file's limits (`:900`), which stays correct.
   - The `d.expected` computed at queue time (`:1141`) and the EOF checks (`:1116`) are unchanged, because sizes are
     equal across roots.
   - `sub_reads_` is already a mutable per-descriptor copy (`:803`, `:1135`). That is why piece mode needs no new
     storage.
3. **Policy.** Pick `argmin_q (inflight_bytes[q] + bytes)/rate[q]`, with `inflight = demand + spec` over all groups.
   - Skip `q` when `demand_reads[q] >= cap[q]`, unless every root is capped; then take the argmin anyway.
   - Choosing a root never blocks. Any root can serve any sub-read, so there is no head-of-line stall in `refill`.
   - Rates and caps come from env, for example SPCC rate 2.2 and cap 2, Samsung rate 3.35 and cap 4, matching the fio
     table in §33.16.
   - Default **off**. With it off the code path is byte-identical, so `test_exl3_ram_miss_piece_stream_parts.py`
     (golden geometry digests, `:47-65`) and `test_expert_stream_reader_golden.py` (SQE logs, `SqeRecord`
     `reader_core.h:36-39`, `:1281-1288`) stay green unchanged.
4. **Trace.** Move the `drive_extents` attribution from queue time (`:1150-1153`) to the moment of choice, because the
   file now changes after queueing. `drive_bytes` at retire (`:1531`) and `account_unfinished` (`:1653`) already read
   `d.read->file` late, so they stay right.

### Answer to "could (3) choose among a few precomputed split tables (1:1:1, 1:0:1, ...)?"

Not as split tables. Each table cuts different pieces (§0), and the device holds one cut per `(row, expert)`
(`row_copy_kernels.cuh:186`). Making that work needs a table id per lane in the lease, a runs tensor of size ×T, and a
wire/kernel change across `lease_layout.h` and `row_copy_kernels.cuh`.

Per-sub-read root choice on today's grid **subsumes** those tables:

- 1:0:1 is "root 1's two sub-reads go to roots 0 and 2".
- 16:10:16 is about 2:1:2 of 5 sixths, or 3:2:3 on an 8-piece grid.

The grid granularity is the price. Equal weights give 6 sub-reads of about 2.2 MB, i.e. 1/6 of a 13,316,096-byte row.
An optional later step is a **root-independent 8-sub-read grid**: a `row_geometry` variant that cuts the image into
`kPieces` equal page-aligned sub-reads whatever the part count.

- It gives 1/8 granularity (about 1.66 MB per piece) and finer piece streaming.
- It changes the device runs, but they are regenerated from the same function at setup (`ffi_exports.h:178`), so host
  and device cannot disagree.
- The golden digests must then be re-recorded under a new name, not overwritten.

### Failure modes and invariants

- **Wrong-row file.** An off-by-`parts` error would read another layer's bytes and pass every length check. Guard it
  with the open-time `source_paths` check, and test it with `poison` (`row_reader.h:234-242`) plus content comparison
  against the eager reader.
- **Faults key on the part.** `fault.part` is matched by the descriptor's **part slot** (`reader_core.h:1456`,
  `:1512`), not by the drive. Under dynamic mode the part slot no longer names a drive. Keep the semantics, which tests
  rely on, and add `fault.root` matched on `file % parts` for drive faults.
- **SPCC tail.** The choice cannot cancel an in-flight 29 ms p99 tail. Hedging (duplicating a late sub-read on another
  root, first CQE wins) would need double-landing protection on `sub_done` (`:1171-1172`). It is out of scope.
- **Cross-group fairness.** Two groups' service threads choose concurrently from relaxed loads. A race costs at most
  one suboptimal pick, not correctness.

### Size and risk

- About 200 LOC of C++ (map, hook, policy, env, trace move), 40 of Python, and 200 of tests.
- Moderate risk: it is on the request hot path, but it adds no thread and no lock.
- Read `.claude/skills/env-var-conventions` before adding `SGLANG_MOE_EXPERT_MIRROR_{DYNAMIC,RATES,CAPS}`
  (`environ.py:501-505`), as the repo rule requires.

## 4. Change (2): drive-aware speculative reads

### Today

- `spec_read` (`ram_tier.h:2356-2418`) spins while `demand_waiting` is set (`:2361-2362`), then locks `spec.turn`
  (`:2363`) and reads the whole row through **the group's demand reader** (`land_pool_row`, `:2468-2487`, calling
  `dist_.group(g).reader.read(...)` at `:2473`).
- The demand side `try_lock`s the same turn and counts `kSpecDelayed` (`:2657-2670`).
- So a demand waits for up to one whole row. Every speculative read also uses all three roots, because it goes through
  the same extents.

### Target

1. **A speculative reader per group.** Add `Source spec_reader` to `SpecGroup` (`ram_tier.h:1444-1463`), constructed
   from a copy of the group's Tables.
   - Create it in `enable_ram_prefetch` (`:1116-1124`) and open it in `RamTier::open()` (`:168-176`) after the demand
     reader, with `set_piece_stream(true)` and no publish target. Pool rows publish nothing to the device;
     `land_pool_row` already passes `nullptr` (`:2473-2475`).
   - `land_pool_row` then reads through `spec.spec_reader`.
   - **Delete** `turn`, `demand_waiting` (`:1448-1449`), the spin at `:2361-2362`, and the demand-side lock
     (`:2657-2670`). `kSpecDelayed` (`tier_protocol.h:62`) becomes `kSpecDeferred`. Rename it in the Python mirror
     too (`python/sglang/kernels/ops/moe/expert_stream_transport.py:1159`, `:1189`).
   - Costs: a second ring per group. IOPOLL without SQPOLL means the speculative thread polls its own ring on its own
     cores (`config.cores`, `:1121`). With SQPOLL a second kernel SQ thread is needed, or `IORING_SETUP_ATTACH_WQ`. In
     fixed read modes each ring registers the slab regions again (`row_reader.h:92-95`), doubling the pinned-memory
     accounting; check `RLIMIT_MEMLOCK`.
2. **One root per speculative row, chosen by load.** Add a per-read `RootPolicy` to `read()`: a new template parameter
   beside `Abandon` and `Progress` (`reader_core.h:398-409`), defaulting to "table root", so demand callers are
   unchanged until (3) lands. The speculative policy:
   - At the row's first sub-read, pick `q*` = argmin `spec+demand bytes / rate` among roots with
     `demand_reads[q] == 0`, counting both groups.
   - If there is none, **defer**.
   - Before **each** later sub-read, recheck `demand_reads[q*] == 0`. If demand arrived, either move the next piece to
     another demand-free root (legal, because any root serves any piece) or defer.
3. **One piece in flight.** Add `c.max_desc_inflight = 1` for speculative calls. In `refill` (`reader_core.h:1244`),
   stop when a descriptor is already in flight. The existing credit cannot express this: it counts legs and reserves a
   descriptor all-or-nothing, and `|| c.pending == 0` admits any width (`:1244`). A demand sub-read that lands on a
   drive mid-piece therefore shares it with at most one speculative piece. That piece is **about 2.2 MB on today's
   grid**, roughly 0.66 ms on a Samsung and 1 ms on the SPCC, **not the 1 MB in the brief**.
   - Reaching 1 MB needs either more than 8 sub-reads (past `kPieces`; `RowGeometry` arrays are `[kPieces]`,
     `piece_geometry.h:49-52`) or partial-leg credit for the speculative reader only. Neither is minimal. Ship 2.2 MB
     (or 1.66 MB with the 8-grid) and measure first.
4. **Deferral needs read-loop changes.** Today `read()` exits when `pending == 0 && !ready && held_empty()`
   (`reader_core.h:476`) and then calls a non-empty queue a bookkeeping failure (`:488-491`). A gated queue head
   would therefore fail the read.
   - Add a "deferred" state: keep looping while the queue is non-empty and the policy defers, `_mm_pause` or a short
     sleep with no ring wait (`reap` with nothing pending must not block, `:481`), and call a **mid-row abandon**
     check.
   - **Abandon** when `spec_stale` turns true (`ram_tier.h:2203-2210`), when a deadline passes (a few ms, far below
     the watchdog's `fatal_wait_ns`, which times `spec.busy`; `ram_thread.h:260-317`), or on stop or hold
     (`spec_wait_unheld`, `:2453-2464`).
   - An abandoned read stops queueing, reaps what is in flight, returns 0, and `land_pool_row` stores `kPoolEmpty`
     (`:2484`). Count it as `kSpecAbandoned`, not `kSpecFailed`, so real I/O errors stay visible.
5. **Promotion boost (required).** A forced miss that finds its expert `kPoolReading` spins until the read ends
   (`ram_tier.h:2122-2134`). With single-root, throttled, deferrable speculative reads that wait can grow from about
   1.5 ms to several ms, or for as long as demand keeps the root busy.
   - Have the promoter set `entry.boost` (an atomic in `PoolEntry`, `ram_prefetch.h`) before spinning.
   - The speculative policy then drops its gate, its single-root rule and its one-piece limit, and spreads the
     remaining sub-reads across roots by the (3) policy, so a promoted read finishes at demand speed.
   - Without this, (2) can make promoted misses slower than today.
6. **Test faults.** `apply_pending_fault(dist_.group(g))` (`:2400`) and `inject_spec` (`:1337-1345`, "holding the
   reader's turn") must target `spec_reader`. The prefill fill (`:1676-1718`) still reads through group 0's demand
   reader with the speculative threads quiesced (`ram_thread.h:135-140`, `ram_tier.h:1395-1404`). That stays, because
   `quiesce_spec` waits for `idle`, which now also covers a deferred read.

### Invariants to keep (and test)

- **No stale publication.** The pool-word protocol is untouched: claim `kPoolReading` → `land_pool_row` → `kPoolLanded`
  or `kPoolEmpty`, with `_mm_sfence` before the landed store (`ram_tier.h:2372-2389`, `:2479-2485`). The `note_serving`
  fence pairing (`:2163-2165`, `:2382`) still decides "promote or drop". Abandonment only adds a path to `kPoolEmpty`.
- **Demand precedence.** This moves from a mutex to drive gating. Without a turn, a demand on group A can run beside a
  speculative read on group A **on another drive**, which is the point. Any concurrency between them on the same drive
  is bounded by one piece.
- **NUMA groups.** Each group keeps its own demand and speculative readers. Only `DriveLoad` is shared, and it is
  lock-free.
- **The registry test needs attention first.** `test_expert_stream_ownership.py:140-157` pins the tier's mutex names to
  `{caller_mutex_, fault_mutex, mutex}` and forbids other lock names, but `ram_tier.h:1448`
  (`std::mutex turn;`), `:2363` (`spec.turn`) and `:2372` (`pool_->mutex(g)`) seem to violate it **today**. Check
  whether it is green on this branch before relying on it. After (2) deletes `turn`, update it so the pool mutex is
  the only addition.

### Size and risk

- About 500-700 LOC of C++ (second reader lifecycle, root policy, deferral, abandon, boost, counters), about 60 of
  Python, and about 400 of tests, several existing ones inverted.
- **High risk**: threading, ownership and the promotion wait.
- **Throughput risk.** About 4,026 issued speculative reads in 131 forwards (§33.16 table 2) is about 30 per forward,
  about 400 MB per 266 ms forward, about 1.5 GB/s that must fit in demand-free windows of single drives. Coverage may
  fall, because more reads defer and go stale. The 19 covered misses per forward are what is at stake. Measure
  `spec_used` per forward, not only tok/s.

## 5. Tests

Existing files to extend, each with a mutant (§`.claude/rules/divix01-run-protocol.md`: apply mutants on divix01 in a
private worktree, revert, re-run green):

- **`test_exl3_ram_miss_piece_stream_parts.py`.** With dynamic off, the golden digests are unchanged. With dynamic on,
  every sub-read's `(offset, length, dest)` is identical to the static geometry and only `file % parts` varies.
  Mutant: choose `alt_file` by `+q` without subtracting the part → red, because the content check reads another
  root's row of another layer.
- **`test_exl3_ram_miss_split.py` and `test_exl3_ram_miss_row_images.py`.** Rows read under a forced skewed load (a
  test hook `set_drive_load(q, bytes)`) match the eager reader byte for byte, and `drive_bytes` per root follows the
  policy (for example root 1 capped at 0 → 0 bytes). Mutant: ignore the cap → red.
- **New `test_expert_stream_drive_load.py`.**
  - After every `read()` return, including injected `cqe_error`, `part_error` and `part_short` faults
    (`read_fault.h`) and a throwing guard, the shared counts are 0.
  - With two readers over the same Tables, counts sum.
  - Mutant: skip the subtract in `drain` → red.
- **`test_exl3_ram_prefetch_thread.py`.**
  - Invert `:86-101` and `:103-117`. A demand is served **while** a slowed speculative read (`inject_spec` delay) is
    in flight on another root, and `spec_deferred` counts.
  - New: a speculative read never submits a piece to a root with `demand_reads > 0`. Use a fake demand count via the
    hook and the per-root SQE log (`SqeRecord.file`).
  - New: a deferral past its deadline or a stale target abandons, leaves `kPoolEmpty`, and counts `spec_abandoned`.
  - New: a promoted read is boosted, i.e. served within one row's time while demand load is held on its root.
    Mutants: no gate re-check between pieces; no boost.
  - Keep `:119-162`: fail, promote, and the GPU miss on a reading row.
  - Keep `:181-252`: pause, stop, watchdog.
- **`test_expert_stream_ownership.py` and `test_expert_stream_sync_primitives.py`.** Update the registries, as noted
  above.
- **`test_expert_stream_hotpath_{golden,shim,stress}.py`.** No new syscalls or allocations on the demand path. Run TSAN
  in `test/manual/dsv41/test_expert_stream_hotpath_tsan.py` over the two-reader setup.
- **Run narrowly**, per the rule: the files above plus `test_*exl3*.py` and `test_*expert*.py` under
  `test/registered/unit/kernels`. Warm the JIT first, since the C++ changes give every module a new build key.

## 6. Counters and trace events for live verification

- **Core counters** (`tier_protocol.h:32-97`, Python names in `expert_stream_transport.py:1159-1190`):
  `kSpecDeferred` (replaces `kSpecDelayed`), `kSpecAbandoned`, `kSpecBoosted`, `kSpecMovedRoot` (a piece moved off its
  row's root), and `kDemandCapped` (a choice that skipped a capped root).
- **Per root, FFI `drive_load()`:**
  - instantaneous `demand/spec bytes and reads`;
  - the time integrals `∑demand_inflight·dt`, `∑spec_inflight·dt` and `∑(demand>0 && spec>0)·dt`. The last one is
    the direct measure of the "hidden cost" overlap and should go to about 0 under (2);
  - bytes per root, which should approach the rates' ratio under (3).
- **InstrBuild JobTrace events** (`job_trace.h:168-180`):
  - `spec_piece` (root, bytes, submit and CQE ns);
  - `spec_defer` and `spec_abandon` (reason: stale, deadline, hold);
  - `spec_boost`;
  - per-demand-sub-read `root` in `StageRecord` (extend `extent_id` or add `extent_root`).
- **Verdict metric.** Rerun `analysis/dsv41-drive/dspark/layer_misses.py --compare` on matched (row, misses) strata. The
  +0.49/+0.58/+1.12/+1.72 ms prefetch penalty at 1-4 misses (§33.16 item 2) should vanish, with `spec_used` per forward
  held near 18. `drive_busy.py` should show the SPCC busy share falling under (3).

## 7. Recommended order

| Step | What | Risk | Size | Gate to proceed |
|---|---|---|---|---|
| 0 | `SGLANG_MOE_EXPERT_MIRROR_WEIGHTS=16:10:16` A/B (setting only) | none | 0 | ms/token and per-root bytes |
| 1 | `DriveLoad` + FFI + counters, observe-only | low | ~250 LOC | the overlap integral reproduces §33.16's hidden cost |
| 2 | Change (3), dynamic demand roots, default off | moderate | ~450 LOC | goldens unchanged when off; A/B on beats step 0 |
| 3 | Change (2), speculative reader per group, gating, abandon, boost | high | ~1,000 LOC | overlap integral ≈ 0, matched-strata penalty gone, `spec_used` held |
| 4 (optional) | 8-sub-read root-independent grid (1.66 MB pieces, 1/8 split) | moderate | ~200 LOC | only if steps 2-3 show the piece size matters |

**Why (3) comes before (2).** (3) touches no threading, and it is useful with the prefetch off. Its per-sub-read root
hook in `refill` is the same hook (2)'s speculative policy plugs into. Once both exist, demand reads also steer away
from roots holding a speculative piece, which closes the remaining same-drive overlap.

**Steelman against this plan.** A simpler (2) could keep one ring and have the service thread interleave speculative
pieces between demand rows. It would avoid a second ring and the deferral loop, but it serializes speculative I/O
behind the service loop and adds the speculative state machine into the hot path that the zero-overhead plan just
cleaned. The second reader isolates that risk, at the cost of a ring and its memory registration.
