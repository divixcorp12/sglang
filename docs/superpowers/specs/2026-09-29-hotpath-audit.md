# Expert-stream RAM-miss hot path: audit and design

Audited at `origin/master` = `ba01695c35`. Read-only: nothing was run. Every claim below comes from reading the code at
that commit. Cost estimates without a measurement are labelled as estimates.

Path prefixes used below:

- `H/` = `python/sglang/kernels/jit/csrc/moe/expert_stream/host/`
- `P/` = `python/sglang/`

## 0. The production configuration this audit is judged against

`benchmarks/dsv41_baseline/arm_env.py:120-216` (`base_env`, which `launch_prod.sh` serves) sets:

| Setting | Value | Consequence for the hot path |
|---|---|---|
| `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES` | 1 | `AnyReader` holds a `RowReader`: O_DIRECT readv straight into the slabs (`H/any_reader.h:318-322`) |
| `SGLANG_MOE_EXPERT_FILE_READER` | `uring_direct` | `direct=true`: O_DIRECT |
| `..._LEASES`, `..._TWO_PHASE`, `..._PIECE_STREAM` | 1 | lease mode, hit grant in the reservation hold, LOADING grant of the miss lanes, and piece publishing |
| `..._COPY_ENGINE`, `..._SM_SMALL_COPIES` | 1 | the copy thread runs and hit lanes are `COPYING` |
| `SGLANG_MOE_HOT_UPDATE_*`, GPU residency, insert-on-miss 2 | on | DIRECT mode, so `gpu_hot_mode_` is set: the hot bitmap is read per demand (`H/ram_tier.h:150-162`) |
| `SGLANG_DSV41_RAM_MISS_PACK_WORKERS` | 8 | ignored by `RowReader` (`H/row_reader.h:48-51`) |
| `SGLANG_DSV41_ENABLE_PREFILL_FILLS` | 1 | the fill thread shares the reader and ring during prefill |
| `SGLANG_EXPERT_STREAM_URING_*` | unset | `MODE=default`, `WAIT_MODE=block`, `READ_MODE=normal`, `READ_CUTS=auto` (so cuts are off), no fixed files (`H/uring_options.h:18-31,105-110`) |
| stage trace | off | `cur_ == nullptr` |
| native prefetch | off (`SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=0`) | `pump_prefetch` returns at `H/ram_tier.h:555` |
| prefill share | off (env default) | `set_prefill_share` is never called |

The production instantiation is `HostExports<Exl3RowLayout, FaultyReader<UringReader>>`
(`P/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp:9-11`). **The test-only fault wrapper is therefore in the production
binary.**

## 1. Where the hot path starts and ends

**Starts:** the device's post kernel stores a demand record's seq word into the host-mapped page. The first
host instruction on the path is `RamThread::run`'s poll (`H/ram_thread.h:370`), which calls `pump_demand`.

**Ends:** the service's `store_release(page_ + kDemandDone, ...)` (`H/ram_tier.h:193`). The path also includes the
per-request work that other threads do concurrently:

- the copy thread, from `CopyEngine::submit` to `copy_completed` / `release_copied`
  (`H/copy_engine.h:297-304,351-445`, `H/ram_tier.h:1115-1166`);
- lease retirement driven by the device's acks (`H/ram_tier.h:898-951`).

**Per-request call chain (production):**

```
RamThread::run                          ram_thread.h:360-380  (loop iteration)
  RamTier::pump_demand                  ram_tier.h:129-197    (per poll; body per demand)
    retire_leases                       ram_tier.h:898        (per poll while leases are outstanding)
    read_record / read_gpu_hot / read_lane_request / terminal_seen
    apply_gpu_hot, defers
    handle_demand -> serve              ram_tier.h:1826, 1562
      reservation hold: take_slot_locked, bump_generation, open_lease_entry,
                        init_piece_words, grant_lane_group(hit + LOADING) -> CopyEngine::submit
      reader_.read(...)                 any_reader.h:409 -> ReaderCore::read reader_core.h:334
        admit / admit_batch / queue_sub_reads / refill / plan_legs / prep_readv   (per row, per SQE)
        reap -> UringReader::submit/reap -> process -> retire -> land_sub_read/vet_pieces  (per CQE)
        RowReader::publish_landed -> publish_collected -> publish_piece (CAS)      (per piece)
      publish hold, S3 hold
    set_status, store kDemandDone
```

**Setup and teardown (off the hot path, free to allocate and lock):** everything in `HostExports::open`, `start_thread`,
`stop_thread` and `close` (`H/ffi_exports.h:662-722,1151-1235`), `RamTier`'s constructor and `init_lease_block`, the
`set_*`/`enable_*` methods that refuse once `threaded_` is set, `ReaderCore::open`, `size_legs` and `size_extents`
(`H/reader_core.h:213-309,611-676`), `UringReader::init`/`configure_resources`, and `CopyEngine::start`/`set_table`.

**Prefill and eager paths (not decode, but still per request):** `fill_begin`, `run_fill`, `assign`, `touch`,
`mapping` and `lru_order`, which Python calls while the service is paused. These are reported separately in the table.

**Python per decode step: nothing in production.** Decode runs as a replayed CUDA graph. `_gather_graph`
(`P/srt/layers/moe/expert_stream.py:1289`, guarded at `:1876`) and the backend's
`post`/`hit_wait`/`stream`/`stage_ack` launches (`P/srt/layers/moe/exl3_ram_miss.py:540-609`) run only at capture.

Per batch, the scheduler runs `fail_stop_check` (`exl3_ram_miss.py:1064-1085`):

- one `page_word` read (`P/kernels/ops/moe/expert_stream_transport.py:697-699`: two tensor views and an `int()`);
- the `_arm_copy_engine` early-return;
- the trace hooks, which return at once when the trace is off (`:1148-1151,1216-1218`).

That is a few Python objects per batch, not per miss. `on_residency`'s `set_hot`, with a `torch.tensor` built by
`_ids` (`expert_stream_transport.py:79-80,1101-1103`), fires only in the non-DIRECT hot mode
(`exl3_ram_miss.py:926-931`). `set_prefill_share` is off (`:843-857,1087-1092`).

## 2. Inventory

The columns are:

- **Req:** C = copy, M = metric/timing/diagnostic/fault, A = allocation, L = lock or kernel sleep.
- **Freq:** how often the item runs (per demand, row, SQE, CQE, piece, loop iteration or job).
- **Prod:** Y = on the production path (row images, default MODE, trace off); otherwise the mode that reaches it.
- **Sev:** H/M/L, an estimate of how much it costs the path.

### 2.1 Copies

| # | Where | Req | What | Freq | Prod | Sev |
|---|---|---|---|---|---|---|
| C1 | `H/row_reader.h:138-140,207-218` | C | Row images: `IORING_OP_READV` whose iovecs are the destination slab rows, so the drive DMAs into the pinned slot | per extent | Y | none (zero-copy) |
| C2 | `H/pack_reader.h:157-176` (inline), `:187-216,222-262` + `H/pack_pool.h` (workers) | C | Packed path: NVMe DMA into the bounce, then a CPU `memcpy` per segment into the slab | per row / per piece | N: `ROW_IMAGES=0`, the env default (`P/srt/environ.py:1870`) | H when used |
| C3 | `H/tier_protocol.h:161-180`, `H/ram_tier.h:430-444,647-669` | - | `memcpy` of record metadata (ids, bitmap), not expert bytes | per demand | Y | n/a |
| C4 | `H/pack_reader.h:181-183`, `H/row_reader.h:222-228` | C | `poison_slot` memset | per row | fault `poison` only | L |

### 2.2 Metrics, timing, diagnostics and fault checks

| # | Where | Req | What | Freq | Prod | Sev |
|---|---|---|---|---|---|---|
| M1 | `H/ram_tier.h:143,155,158,160,172,186,861,860,868-869,826,1675,1756,1795-1796,1840,1489,1499,939,968`; `H/copy_engine.h:479-480` | M | `counters_[k].fetch_add` (seq_cst `lock xadd`) on one 41-slot `std::atomic<int64_t>` array (`:1932`) that the service **and** the copy thread write, so its cache lines ping-pong between them | about 8-14 RMWs per demand, plus 1 per released lane and per eviction | Y | M |
| M2 | `H/ram_tier.h:1721` | M | `counters_[kPiecePublishRefused].store(reader_.publish_refused())` | per read | Y (piece stream) | L |
| M3 | `H/ram_tier.h:1827,1844` (`busy_since_ = now_ns()`), `:224`, `:1369` | M | Clock read for the watchdog's hung-read rule (`H/ram_thread.h:402-405`). It is functional, but it is a clock read on the path | per demand / advisory / fill | Y | L |
| M4 | `H/reader_core.h:405-411` | M | `now_ns()` on **every drain-loop turn**, to gate the `progress` callback. `serve` always passes a progress callback (`H/ram_tier.h:1719`) | per read-loop iteration | Y | M |
| M5 | `H/ram_tier.h:867` | M | `job.submit_ns = now_ns()`, used only for latency counters | per copy job | Y | L |
| M6 | `H/copy_engine.h:462,479-480` | M | issue-time clock pair + 2 RMWs (`kCopyIssueNs`, `kCopyBytes`) | per copy job | Y | L |
| M7 | `H/copy_engine.h:484-489` | M | `record_latency`: clock + RMW + CAS loop on `kCopyLatencyMaxNs` | per copy job | Y | L |
| M8 | `H/copy_engine.h:364,435,437`; `H/ram_thread.h:357,371,375` | M | Spin-pacing clock reads (`now_ns() - last_active < spin_ns_`) | per loop iteration (copy and service threads) | Y | L-M |
| M9 | `H/ram_tier.h:140,194,1243-1259,1273-1279,1565,1679,1787-1793,1832-1838,171,179`; `H/reader_core.h:360,367-372,831,893-897,928-939,942-943,1013,1031-1037,1126-1140,1162-1167,1175,1191,1224-1227,1324-1329,1391,1397-1401,1419-1432,1443`; `H/row_reader.h:158-161,192-193` | M | Stage-trace plumbing: a relaxed load of `trace_on_`, then `cur_`/`c.trace` null checks and `stamp(nullptr)` branches (no clock read when off, as `H/reader_base.h:108-113` promises) | per event: row, SQE, CQE, piece | Y (branches only) | L |
| M10 | `H/reader_core.h:1255` (`++cqes_`), `:1242` (`stale_cqes_`), `:604` (`generation_wraps_`), `:763-765` (`cut_reads_`, `gap_cuts_`), `:790-793` (`fixed_cuts_`, `fanout_sqes_`), `:1381` (`publishes_`), `:1390` (`publish_refused_`); `H/uring_reader.h:162-172` (`note_fanout` + report check) | M | Plain diagnostic counters. Only `cqes_`, `publishes_` and `publish_refused_` run in production; cuts and fixed reads are off | per CQE / piece | Y (3 of them) | L |
| M11 | `H/ram_tier.h:1691-1701` (`apply_pending_fault` acquire load; `delay_ns_`, `fail_reads_`, `abandon_after_` loads), `:189` (`done_stall_ns_`) | M | Tier fault words | per demand | Y | L |
| M12 | `H/reader_core.h:357,363,905,924,1178,1196-1197,1218,1251-1268,1334,1340,1381-1384,1409`; `H/row_reader.h:163,184` | M | Reader fault checks. Per CQE, `process` computes `part = (index / subs_) % parts`, a division done only for the fault, then about 6 compares | per read / row / CQE / piece | Y | L |
| M13 | `H/faulty_reader.h:80-89` | M | `FaultyReader::submit`: `++submits_` + 2 compares. The production binary uses this wrapper (`exl3_ram_miss_host.cpp:9`) | per submit | Y | L |
| M14 | `H/reader_core.h:1118-1125` | M | `sqe_log_` null check (test U10) | per SQE | Y | L |

The following are **not metrics and must stay**, because the protocol uses them:

- `lanes_outstanding_` (`H/ram_tier.h:859,967`), the idle early-out, which pause's refusal reads;
- `lease_changes_` (`:969,1196`), which wakes a deferred demand;
- the page heartbeat (`H/ram_thread.h:362`);
- `kVersion` (`H/ram_tier.h:294,1756`), which Python's LRU view uses for invalidation (`P/srt/layers/moe/exl3_ram_miss.py:386,430`);
- `publish_piece`'s CAS (`H/row_tables.h:112-119`);
- every `store_release`/`load_acquire` of the page and lease block.

### 2.3 Allocations

| # | Where | Req | What | Freq | Prod | Sev |
|---|---|---|---|---|---|---|
| A1 | `H/ram_tier.h:147` + `H/tier_protocol.h:123-130,176-177` | A | `Request request;` is a local with five `std::vector` members; `read_record` `assign`s `need` and `protect` | 2 mallocs + 2 frees per demand | Y | H |
| A2 | `H/ram_tier.h:438` | A | `request->hot_bitmap.assign(...)` (`experts_/8` bytes) | 1 per armed demand | Y (GPU hot) | M |
| A3 | `H/ram_tier.h:665-666` | A | `lane_experts.assign`, `lane_dst.assign` | 2 per armed demand | Y | M |
| A4 | `H/ram_tier.h:696-700` | A | `defers()`: local `wanted` grown by `push_back` (up to 16-24 ids: about 5 reallocs) | per armed demand | Y | M |
| A5 | `H/ram_tier.h:1566-1577,1601,1630` | A | `serve()`: locals `wanted`, `missing`, `slots` grown by `push_back` | about 6-12 per armed demand | Y | H |
| A6 | `H/ram_tier.h:1707-1710` → `H/reader_core.h:339` | A | The `abandon` lambda `[&]` captures `advisory`, `this` and `abandon_after` by reference. Its closure is 24 B, past libstdc++'s 16 B local storage, so building the `std::function` heap-allocates (compiler-dependent; confirm with a malloc counter). `progress` (`[this]`, 8 B) fits locally | 1 per read | Y | M |
| A7 | `H/copy_engine.h:366` | A | `std::deque<CopyJob> fresh;` is **constructed inside the copy thread's loop**. A libstdc++ deque allocates its map and one 512 B node even when empty | 2 mallocs + 2 frees **per loop iteration, while spinning** | Y | H |
| A8 | `H/copy_engine.h:300,378,414,457` | A | `queue_`, `held`, `acking` and `in_flight` deque `push_back`s allocate nodes as they cross node boundaries | per job, amortized | Y | L-M |
| A9 | `H/reader_core.h:662` vs `H/uring_reader.h:232` | A | `completions_` is reserved to `extents+1` but can hold up to `queue_depth()` CQEs (x legs with cuts or fixed reads). It grows once, then keeps its capacity | warm-up only | Y (cuts off: bounded) | L |
| A10 | `H/ram_tier.h:1371` (`packed`), `:338-339,362-363` (`claimed`, `taken`), `:1392-1393` (exception on the fill's error path) | A | Prefill-fill locals | per fill (prefill) | prefill fills | L |
| A11 | `H/ram_tier.h:210` | A | `pump_advice`'s `Request` (like A1) | per advisory | advisories only | L |
| A12 | `H/reader_core.h:706,1114,1315`, `H/ram_tier.h:963`, etc. | A | `throw std::runtime_error(...)` with a `std::string` | error paths only | - | fine |

The service path already reuses `packed_` (`H/ram_tier.h:1684-1688`), `piece_targets_` (`:1807`), the reader's `Call c_`,
`descs_`, `legs_`, `queue_`, `again_`, `iovecs_` and `rows_[]` (`H/reader_core.h:652-687`), and the `PackJob` pool. Those
need no work.

### 2.4 Locks and kernel sleeps

| # | Where | Req | What | Freq | Prod | Sev |
|---|---|---|---|---|---|---|
| L1 | `H/ram_tier.h:900` (from `:134` and the progress callback `:1719`) | L | `retire_leases` takes `mutex_` and scans 16 entries. It runs **on every poll of an idle service loop while any lane is outstanding**, and every 200 us during a read | per loop iteration | Y | H |
| L2 | `H/ram_tier.h:447-451` | L | `apply_gpu_hot` holds `mutex_` over an O(experts) loop | per armed demand | Y | M |
| L3 | `H/ram_tier.h:691` | L | `defers()` | per armed demand | Y | M |
| L4 | `H/ram_tier.h:1589` | L | `serve` reservation hold (S2) | per armed demand | Y | M |
| L5 | `H/ram_tier.h:1734` | L | `serve` publish hold | per armed demand | Y | M |
| L6 | `H/ram_tier.h:1775` | L | `serve` S3/close hold | per armed demand | Y | M |
| L7 | `H/ram_tier.h:1539` | L | `touch_request` | per unarmed record | Y | L |
| L8 | `H/copy_engine.h:297-304`, called at `H/ram_tier.h:870` | L | `CopyEngine::submit` takes the engine mutex **nested inside the tier `mutex_`**, then `notify_one`, a futex wake when the copy thread sleeps | per copy job | Y | M |
| L9 | `H/copy_engine.h:370,498,440` | L | Copy thread: engine mutex every loop iteration; `finish()` per job; `wait_for(1 ms)` when idle | per iteration / job | Y | M |
| L10 | `H/ram_tier.h:1149` (`release_copied`), `:1179` (`prefetch_completed`) | L | **The copy thread takes the tier `mutex_`**. This is the one steady-state contender with the service thread during decode | per copy job | Y | H |
| L11 | `H/uring_reader.h:206-207` | L | `WAIT_MODE=block` (the default): `io_uring_submit_and_wait(1)` puts the service thread to sleep in the kernel until the NVMe completion wakes it | per reap that finds no ready row | Y | H |
| L12 | `H/ram_thread.h:375-378` | L | `sleep_for(50 us)` once the loop has been idle for 5 ms (`spin_us=5000`, `P/kernels/ops/moe/expert_stream_transport.py:835`). It is an idle path, but it delays seeing the first miss after an idle gap | idle loop | Y | L-M |
| L13 | `H/ram_tier.h:1100,1266` (`fault_mutex_`), `:243,251,256` (`trace_mutex_`) | L | fault and trace mutexes | test / trace only | N | - |
| L14 | `H/pack_pool.h:226-234,255` | L | `PackPool::post` mutex + `notify_all`, and the workers' condvar wait | per row / piece | packed path only | M |
| L15 | `H/ffi_exports.h:66-70` | L | `find()` takes `registry_mutex()` and copies a `shared_ptr` (atomic refcount) | per Python FFI call | no per-step call in production | L |

## 3. Copies

**Production (row images): zero-copy is confirmed.** `RowReader` has no bounce and no pool (`H/row_reader.h:48-51,81-86`).
Each read is one `IORING_OP_READV` (`kScatter = true`, `:71`; `H/reader_core.h:1147-1151`). Its iovecs are the
destination slab rows of the caller's unpublished slot (`image_iovecs`, `H/row_reader.h:207-218`), so the NVMe DMA writes
the pinned slab directly. The GPU then moves the bytes itself:

- hit lanes by `cuMemcpyAsync` on the copy thread (`H/copy_engine.h:469-475`), or by the copy wait's SM reads of the
  small tensors;
- miss lanes by the stream kernel, from the published pieces.

No CPU instruction touches expert bytes.

Two conditions carry this guarantee, and only Python enforces one of them:

1. **O_DIRECT is required.** A buffered `RowReader` would copy from the page cache inside the kernel. Python refuses row
   images without O_DIRECT (`open_service_row_images`, `P/srt/layers/moe/exl3_ram_miss.py:320-335`), but
   `RowReader`/`ReaderCore` accept `direct=false` (`H/row_reader.h:42-46`). Recommendation: refuse `images && !direct`
   in `RowReader`'s constructor too.
2. Prefill fills use the same `RowReader` (`H/ram_tier.h:1381-1391`), so they are zero-copy as well.

**Packed path (`PackReader`).** NVMe DMA fills a page-aligned bounce (`posix_memalign`, `H/pack_reader.h:105-112`;
`IORING_OP_READ`, `kScatter = false`). Each finished row, or each vetted piece under piece streaming, is then copied per
segment into the slab:

- inline on the service thread (`H/pack_reader.h:157-176`);
- or by `PackPool` workers (`:187-262`, `H/pack_pool.h`), which are posted through a mutex and condvar and spin on
  `done()` in `quiesce`.

That is one full CPU copy of every expert byte, plus a PackPool lock per row or piece.

It is **reachable in production only if the recipe is changed.** `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES` defaults to
False (`P/srt/environ.py:1870`), so any launch that does not use `base_env` gets the packed path. It is also the only
path for a checkpoint whose row images have not been built. `AnyReader` compiles both paths and chooses at run time
(`H/any_reader.h:318-322`), with a `get_if` branch on every forwarded call (`:414-418`).

**Zero-copy for the packed path is impossible by construction.** The shard layout on disk is not the slab layout, and
the row image exists precisely to make the two match. The options are:

- **(a) Remove it:** require row images and delete `PackReader`, `PackPool`, the bounce and the variant. `Source`
  becomes `RowReader` directly.
- **(b) Keep it as a named non-hot fallback:** its own instantiation, never under leases, piece streaming or the copy
  engine in production. Flip the env default to row images and fail startup on the hot configuration without images.

The recommendation is (b) now and (a) once every checkpoint and test fixture has images. This is decision D4.

## 4. Metrics: the compile-time design

### 4.1 Where the switch lives

Use one build policy, threaded through the types that already carry `Layout` and `Reader`:

```cpp
struct ProdBuild  { static constexpr bool kMetrics = false; static constexpr bool kFaults = false; };
struct InstrBuild { static constexpr bool kMetrics = true;  static constexpr bool kFaults = true;  };

template <class Derived, ExpertRowLayout Layout, AsyncFileReader Reader, class Build> class ReaderCore;
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
class RowReader  : public ReaderCore<RowReader<Layout, Reader, Build>, Layout, Reader, Build>;
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build> class PackReader;  // same shape
template <class Source> class RamTier;   // reads Source::Build
template <class Tier>   class RamThread; // reads Tier::Build
class CopyEngine -> template <class Build> class CopyEngine;
HostExports<Layout, Reader, Build>
```

**How this works with CRTP.** `Build` is an explicit parameter of `ReaderCore`; it is not read from `Derived`.
`Derived` is incomplete while the base is instantiated, so `typename Derived::Build` cannot appear in the base's member
declarations, and `[[no_unique_address]]` members and `if constexpr` in those declarations need `Build`. The derived
class forwards the same `Build` it was given.

**Reader.** The production build uses `UringReader`; only `InstrBuild` wraps it in `FaultyReader<UringReader>`. The
existing `Reader` template parameter already expresses this, and it removes M13.

**What the switch gates:**

- **Metrics.** A `[[no_unique_address]] Stats<Build::kMetrics> stats_` replaces `counters_` for the metric slots.
  `Stats<false>::add()` is an empty inline function, so it compiles to nothing.
- **Trace.** `StageRecord*` parameters become a `Trace<kMetrics>` type whose `false` specialization has no members and
  whose `stamp()` returns `0` from a `constexpr` function. Every `if (c.trace)` / `if (cur_)` site becomes
  `if constexpr (Build::kMetrics) if (c.trace)`, which is greppable and leaves no branch in the production build. The
  StageRing, `trace_mutex_`, `enable_trace` and `drain_trace` exist only in `InstrBuild`.
- **Faults.** `ReadFault fault_`, `stale_*`, `held_`, `publishes_`, `sqe_log_`, the tier's `delay_ns_`, `fail_reads_`,
  `abandon_after_`, `done_stall_ns_`, `fault_mutex_` and `pending_fault_`, and `poison_slot`: all become
  `if constexpr (Build::kFaults)` or members of `FaultState<kFaults>`.
- **`abandon` and `progress`** stop being `std::function` and become template parameters of `read()` (fixing A6).
  `AnyReader::read` already perfect-forwards (`H/any_reader.h:408-411`). The production demand's abandon is then the
  constant `advisory && ...`, which folds away for demands.

**Recommended: two instantiations, not four.** Two independent booleans would ship four combinations, and a metrics-only
or faults-only build has no consumer: the fault tests read counters, and trace runs want counters. This is decision D2.

### 4.2 How Python picks the build

Add a `variant` to the loader key:

- `_host_module_cached(layout, variant)`;
- module name `expert_stream_host_{layout}_{variant}`;
- host TU `moe/exl3_ram_miss_host.cpp` (`ProdBuild`) plus `moe/exl3_ram_miss_host_instr.cpp` (`InstrBuild`), both
  exporting the same symbol names (`P/kernels/ops/moe/expert_stream_transport.py:52-63`).

The service chooses once, at `open`, before `start_thread`. The constraint that the trace must be enabled before the
thread starts already exists (`H/ram_tier.h:240-246`). The instrumented build is chosen when any of these is true:

- the exl3 stream trace is enabled;
- `SGLANG_TEST_DSV41_RAM_MISS_FAULT` is set;
- an explicit override is set. A new env var would follow the `env-var-conventions` skill, which must be read first.

Test-only exports compile only into `InstrBuild`: `inject`, `inject_fault`, `read_rows_faulted`, `read_rows_sqes`,
`trace_*`, `sim_*`, `seqlock_stress` and `inject_lease`. Their Python wrappers raise a clear error on the production
module. Tests default to `InstrBuild`, plus one production-build smoke test.

**Build cost.** Each variant is a full compile of the host TU. Production compiles and caches only `prod`; CI and dev
boxes compile both, which doubles host-module JIT time on a cold cache. That time is not measured here and should be
timed before and after.

### 4.3 The counters production relies on

`stop()` writes the counters to the server log (`P/kernels/ops/moe/expert_stream_transport.py:1197`), and fail-stop
messages include them (`P/srt/layers/moe/exl3_ram_miss.py:1079,1207`). The recommendation is a small `CoreStats` block
kept in the production build:

- **Fields:** `served`, `touch_only`, `rows_read`, `read_errors`, `overruns`, `late_after_fatal`, `evictions`,
  `deferred`, and the functional `version`.
- **One writer per field.** The service thread owns all of these; the copy thread gets its own block holding only
  `copy_errors`.
- **Its own 64 B line per writer thread**, so the lines never ping-pong.
- **Written with a relaxed store of `load + 1`** (`std::atomic_ref<uint64_t>`). On x86 that is a plain `add`: no `lock`
  prefix, no fence, the line always in L1.
- **Read relaxed** by `counters()`.

**Why this is acceptable:** it is about one cycle per field touched, shares nothing with another core, and is the
minimum a post-mortem of a fail-stopped decode needs. **What moves to `InstrBuild`:** every latency, ns and bytes
counter, lease grant/ack/void/double-signal, copy job/lane/fallback, prefetch, piece-publish-refused and stage-trace
item. A strict reading of requirement 2 would drop `CoreStats` too. This is decision D1.

### 4.4 Clock reads that are not metrics

- **Watchdog (M3).** Instead of reading the clock per demand, the service bumps a busy-episode word, a plain
  `store_release`; it already stores `kBusySeq`, `H/ram_tier.h:1828`. The watchdog thread already wakes every 20 ms
  (`H/ram_thread.h:419`): it records when it first saw a given nonzero episode and aborts if that episode persists past
  `fatal_wait`. The clock read moves to the watchdog thread, and detection granularity becomes 20 ms against a 30 s
  deadline. Fills and advisories use the same word.
- **Progress gating (M4).** Call `progress` every N drain-loop turns (a constant, for example 64) instead of on a 200 us
  clock. Once Phase 3 has made retirement lock-free, it is cheap enough to call every turn while
  `lanes_outstanding_ != 0`.
- **Spin pacing (M8).** Replace `now_ns() - last_active < spin_ns_` with an iteration budget calibrated once at
  `start()`. The only remaining clock reads are then on the idle and watchdog threads.

## 5. Allocations: the preallocation plan

The wire format bounds every size: `kMaxIds = 8`, `kLeaseLanes = 8`, `kDemandRecords = kLeaseRing = 16`
(`P/kernels/jit/csrc/moe/expert_stream/lease_layout.h:20,26,48-49`).

| Finding | Replacement | Bound |
|---|---|---|
| A1, A3, A11 | `Request` becomes a POD member `request_` of the service thread: `int32_t need[kMaxIds]`, `protect[kMaxIds]`, `lane_experts[kLeaseLanes]`, `lane_dst[kLeaseLanes]` plus counts. `read_record`/`read_lane_request` copy into those arrays | 8 / 8 / 8 / 8 |
| A2 | `hot_bitmap` becomes a `uint8_t*` into a member scratch buffer sized `(experts_+7)/8` in the constructor, or it is applied straight from the validated record | `experts_/8` bytes |
| A4, A5 | A `FixedVec<T, N>` with an asserting `push_back`: `wanted` N = 2*kMaxIds + kLeaseLanes = 24, `missing` 24, `slots` 24. Members of the service, cleared per request. `reader_.read` takes `std::span<const int32_t>` / `std::span<const int64_t>` instead of vectors (`H/reader_core.h:336-337`, `H/row_reader.h:209,224`) | 24 |
| A6 | template `Abandon`/`Progress` (section 4.1) | - |
| A7, A8 | Copy engine: an SPSC ring (service to copy thread) of `CopyJob`, capacity `kDemandRecords + 1` (at most one job per request slot, plus one prefetch). `in_flight`, `held` and `acking` become fixed rings of the same capacity. No deque | 17 |
| A9 | `completions_.reserve(queue_depth())` at open | queue depth |
| A10 | `run_fill`'s `packed` becomes a member sized at open; `claimed`/`taken` become members, `reserve`d to the tier's row capacity | capacity |
| A12 | Unchanged: error paths only | - |

The eager and Python-facing methods (`assign`, `fill_begin`, `lru_order`, the FFI `ids_of`/`slots_of`) keep their
vectors. They are prefill and eager paths, and they run only when the service is paused.

## 6. Locks: who takes the tier mutex, and the lock-free design

### 6.1 Every holder of `RamTier::mutex_`

| Party | Methods | When, in production |
|---|---|---|
| Service thread | `retire_leases`, `apply_gpu_hot`, `defers`, `serve` (x3), `touch_request`, `judge_prefetch` (returns before locking when prefetch is off), `pump_prefetch` | every demand, and every idle poll with leases out |
| **Copy thread** | `release_copied` (`:1149`), `prefetch_completed` (`:1179`) | every copy job: **the steady-state contender** |
| Fill thread | `run_fill` epilogue (`:1398`) | prefill, while the service is paused |
| Python, eager, under pause | `has`, `touch`, `assign`, `release`, `fill_begin`, `mapping`, `slot_to_expert`, `lru_order`, `layer_rows` | prefill / eager gathers (`exl3_ram_miss.py:386-420`, under `_depth`/`pause`) |
| Python, **not** under pause | `set_hot` via `on_residency` (non-DIRECT hot mode only), `set_prefill_share` (flag off), `pause()`'s own `retire_leases` (after the handshake) | not in the production recipe |
| Tests / introspection | `slot_info`, `lease_entry`, `inject_lease`, `victim_census`, `prefetch_lease` | tests |

Advisory prefetch (`pump_advice`) runs on the service thread and takes no lock of its own beyond `serve`'s.

### 6.2 The io_uring ring

The ring is not protected by a lock; a protocol serializes it. It is created and registered on the opening Python
thread (`ReaderCore::open`), driven by the service thread, and driven by the fill thread only while the service is
paused. `resume()` joins the fill before the service thread may read again (`H/ram_thread.h:323-328`,
`H/ram_tier.h:86,403-405`). That is why the ring cannot use `IORING_SETUP_SINGLE_ISSUER|DEFER_TASKRUN`
(`H/uring_reader.h:332`).

The ring has two owners but no mutex, so it meets requirement 4 as it stands. What does sleep in the kernel is the wait
(L11): production's `WAIT_MODE=block` calls `io_uring_submit_and_wait`. `WAIT_MODE=spin` already exists
(`H/uring_reader.h:208-223`) and never blocks, at the cost of a busy core during reads. That is a configuration
decision (D5), not a code change.

`SINGLE_ISSUER` would need all issuing on one thread, either by moving fills onto the service thread or by giving fills
their own ring. Under `READ_MODE=normal`, a second ring is cheap because nothing is registered. It would also need
`IORING_SETUP_R_DISABLED` plus enabling the ring from the service thread. With `DEFER_TASKRUN`, the spin wait must call
`io_uring_get_events`, because completions post only when the task enters the kernel. The expected benefit is small
(fewer task-work IPIs) and should be measured before anyone builds it (D8).

### 6.3 The lock-free design: single-writer ownership

1. **The service thread owns** `tiers_`, `outstanding_`, `prefetch_lease_`, `judge_`, `tick_`, `CoreStats`, `packed_`
   and `piece_targets_` outright, while it runs. `mutex_` is deleted.
2. **Pause hands ownership over.** `pause()` returns only after the loop has stored `paused_` (seq_cst today, which is
   at least release), and the caller's load acquires it. `resume()`'s `pause_requested_ = false` is a release; the
   loop's load is an acquire. Every "Python, under pause" row in section 6.1 is then a legal single-writer access by
   the new owner. Add a debug assertion, `threaded_ ? paused_ : true`, to each of those methods; in production it can
   be an `InstrBuild` check.
3. **The fill thread touches no tier state.** The `filling[slot] = 0` clear and the release of failed rows move out of
   `run_fill` (`H/ram_tier.h:1397-1409`) into `fill_end()`/`fill_join()` on the caller's thread, after the join, which
   is a synchronization point. The fill thread publishes only `fill_landed_`/`fill_state_`, which are already atomics.
   This also closes a latent race: Python's chunked admission can call `take_admit_slot_locked` on the same row while a
   fill runs (`H/ram_tier.h:350-352`). Today only the mutex prevents a race there.
4. **Copy thread to service: an SPSC completion ring.** Keep the E6 generation check, the `CopyDone` publish and
   `raise_fatal` on the copy thread: they read `slot_gen_` (written by the service with release), and CopyDone is the
   device's signal, so the device waits no longer than today. Replace `release_copied`/`prefetch_completed`'s locked
   section with a push of `{idx, gen, lane mask, prefetch?}` into a ring of capacity `kDemandRecords + 1`. The service
   drains it at the top of `pump_demand` (beside `retire_leases`) and inside the read's progress hook, and applies
   `release_lease_locked` there.
   - **Consequence:** a COPYING slot becomes evictable one service poll later than now (sub-microsecond while spinning,
     up to the drain interval during a read). This can only raise deferrals marginally, never correctness; it is
     decision D7.
   - The copy engine's `outstanding_` becomes an `std::atomic<int64_t>`, decremented by the copy thread and read by
     `wait_idle`, so `finish()` loses its mutex.
5. **Service to copy thread: an SPSC job ring** (section 5, A7). The copy thread polls it. Its idle path sets
   `sleeping_` (seq_cst), re-checks the ring, and then waits on a futex or condvar. `submit` calls `notify` only if
   `sleeping_` is set, after a seq_cst fence, so a spinning copy thread costs the service no syscall. `submit` no longer
   nests a lock inside the tier hold (L8).
6. **Unpaused Python mutators** go through a small SPSC command ring from the scheduler thread, drained by the service
   between requests:
   - `set_hot`: a fixed-size bitmap payload per row, bounded by `experts_`;
   - `set_prefill_share`: becomes an `std::atomic<int64_t>` read relaxed at admission.

   Neither is in the production recipe; they are needed to delete `mutex_` without breaking other modes.
7. **Tests and introspection:** `slot_info`, `lease_entry`, `victim_census` and the rest require pause or `pump()` mode,
   and are asserted.
8. **PackPool (L14):** only on the packed path. If that path is kept (D4), its post/wait becomes an SPSC ring per
   worker with spinning workers. It is out of scope for the production path.

**Complexity (estimate):** about 500-900 changed lines across `ram_tier.h`, `copy_engine.h`, `ram_thread.h` and
`ffi_exports.h`, plus tests.

**Risk: medium-high.** The mutex currently also stands in for invariants the code argues in prose:

- S2, which grants in the same hold as the reservation;
- S6, which never releases a leased slot;
- E1 and E5;
- the quarantine rules (`H/ram_tier.h:1515-1528,927-932`).

Single-writer ownership makes S2 hold trivially. The risk is in the handoff edges: pause, fill end, and copy completion.
Keep `mutex_` in `InstrBuild` for one release as a cross-check, taken but uncontended, and add a TSan build of
`InstrBuild` for the tests.

## 7. Plan outline

Each phase lands on its own and is verified against the phase before it. Every decode A/B uses the `base_env` recipe
and the existing harness (`benchmarks/dsv41_baseline`) on divix01, following `.claude/rules/divix01-run-protocol.md`:
pulled worktree, `PYTHONPATH`, `PIPESTATUS`, GPU lock.

### Phase 0: configuration only (optional, no code)

- **What:** an A/B of `SGLANG_EXPERT_STREAM_URING_WAIT_MODE=spin` against `block`. It removes L11's kernel sleep.
- **Benefit:** it removes the NVMe completion to task-wakeup latency from every read. The size is unmeasured and should
  be measured.
- **Risk:** one core busy for the length of each read.
- **Verify:** ms/token, `perf stat -e context-switches,cycles -t <service tid>`, byte identity.
- **Decision:** D5.

### Phase 1: compile-time metrics and faults gating

- **What:** sections 4.1-4.4. Two host TUs, a `Build` policy, `UringReader` without `FaultyReader` in production,
  `CoreStats`, template `abandon`/`progress`, the watchdog episode word, and iteration-based pacing.
- **Benefit:** it removes about 10-14 `lock xadd` per demand and per job, and the service/copy false sharing on
  `counters_`. It also removes 3-5 clock reads per demand plus one per drain-loop turn, and every fault, trace and SQE-log
  branch.
- **Risk:** low to medium. The main risk is test plumbing (the variant selection) and a wrong `if constexpr` gate
  silently dropping a functional atomic. The list of functional atomics in section 2.2 is the checklist.
- **Verify:**
  - the whole `test/registered/unit/kernels` suite on `InstrBuild`, with counts identical to the merge-base (record the
    command);
  - a production-build smoke test;
  - a static check that `sizeof(Stats<false>) == 1` and `std::is_empty_v`;
  - disassembly of `RamTier<...ProdBuild>::pump_demand`/`serve` and `ReaderCore::process`, checking for no
    `clock_gettime`/`now_ns` calls and counting `lock`-prefixed instructions against the functional list;
  - a decode A/B for byte identity and ms/token;
  - `perf stat -e cycles,instructions,LLC-load-misses -t <service tid>` and the same on the copy thread.
- **Decisions:** D1, D2, D3.

### Phase 2: allocation removal

- **What:** section 5.
- **Benefit:** it removes about 10-25 malloc/free pairs per armed demand, and 2 per copy-thread loop iteration (A7, the
  worst, because it runs while idle-spinning). Allocator locks and cache pollution leave the service core.
- **Risk:** low. Every bound comes from wire-format constants, with `static_assert`s tying them together.
- **Verify:**
  - an `InstrBuild` test hook that counts `operator new` on the service and copy threads, via a `thread_local` counter
    in a replaced global `operator new` in the test module, asserting 0 over N demands after warm-up;
  - on divix01 in production,
    `bpftrace -e 'uprobe:libc:malloc /comm == "exl3-ram-miss" || comm == "exl3-copy-eng"/ { @[comm] = count(); }'`
    over a decode run, expecting 0 per step (check the exact names against `Layout::kName` and `pthread_setname_np`);
  - the kernels suite;
  - decode byte identity.

### Phase 3: lock removal

- **What:** section 6.3.
- **Benefit:** it removes 5-7 mutex round trips per demand, and one per idle poll while leases are out (L1). It ends
  the copy/service contention on the tier mutex and the nested engine lock. Uncontended, a round trip is tens of ns
  (estimate); the cache-line transfer when the copy thread holds the mutex is the larger term. The ms/token effect is
  probably small next to a read of about 100 us. This phase buys determinism and tail latency more than mean latency;
  measure p99 per-request service time in an `InstrBuild` trace.
- **Risk:** medium-high (section 6.3).
- **Verify:**
  - TSan on `InstrBuild` over the kernels suite and `test_exl3_ram_miss_*`;
  - the lease protocol tests (S1-S7, quarantine, copy engine);
  - mutants in a private worktree, for example skipping the completion-ring drain before admission (expect a
    deferral/eviction test to fail), or moving the fill epilogue back onto the fill thread (expect TSan to fire);
  - on divix01, `bpftrace -e 'tracepoint:syscalls:sys_enter_futex /comm == "exl3-ram-miss"/ { @ = count(); }'`,
    expecting 0 during steady decode;
  - a decode A/B for byte identity;
  - a copy-engine soak (`docs/superpowers/plans/2026-09-25-dsv41-copy-engine-soak.md`).
- **Decisions:** D6, D7, D8.

### Phase 4: the packed path

- **What:** D4. At minimum, flip the `ROW_IMAGES` default, or refuse the lease, two-phase, piece-streaming and
  copy-engine configuration without images. Add the C++ `images && !direct` refusal (section 3). Optionally delete
  `PackReader`, `PackPool` and `AnyReader`.
- **Benefit:** one reader in production and no runtime variant. If the path is deleted, about 700 lines less
  (`pack_reader.h` 347, `pack_pool.h` 331, `any_reader.h` 127).
- **Risk:** low if kept as a fallback. If deleted, the risk is to fixtures and tests that exercise the bounce path.
- **Verify:** the kernels suite (with a before/after count delta that is explained), and startup refusal tests.

## 8. Open decisions for the user

- **D1:** keep `CoreStats`, the 8 relaxed, single-writer, line-private counters behind the shutdown and fail-stop logs
  (recommended), or have strictly no counters in the production build.
- **D2:** one `Build` policy with two instantiations (recommended), or independent `EnableMetrics`/`EnableFaults`
  (four builds).
- **D3:** select the instrumented build implicitly (trace or fault env), or only through an explicit new env var.
- **D4:** packed path: delete it, or keep it as a non-hot fallback (recommended for now), and whether to flip the
  `ROW_IMAGES` default.
- **D5:** does requirement 4 cover the I/O wait itself? If so, `WAIT_MODE=spin` (or `sqpoll`/`iopoll`) becomes the
  production default, which costs a busy core during reads.
- **D6:** move the watchdog's clock to the watchdog thread (sampled every 20 ms against a 30 s deadline).
- **D7:** accept that a COPYING lease is released one service poll later (through the completion ring) instead of by
  the copy thread directly.
- **D8:** keep the ring's issuers serialized by protocol (recommended), or restructure fills to allow
  `SINGLE_ISSUER|DEFER_TASKRUN`.
- **Minor:** the 50 us idle sleep after 5 ms (L12). Keep it (power), or raise `spin_us` in production.

## 9. Counterargument and tensions

- **Steelman against Phase 3.** The tier mutex is uncontended except against the copy thread, and costs perhaps
  0.2-0.4 us per demand next to a read of about 100 us. The lease protocol's hardest invariants are written against
  "one `mutex_` hold". Replacing a lock that is cheap and obviously correct with an ownership discipline spread across
  pause, fill end and completion rings trades a measurable-but-small cost for a real risk of silent corruption: a leased
  slot handed out twice. Phases 1 and 2 capture most of the determinism win (no allocator, no seq_cst RMW storm, no
  clock) at low risk. Do Phase 3 only if a Phase 1/2 trace still shows service-side variance attributable to the mutex
  or to copy-thread contention.
- **Tension: requirement 2 against diagnosability.** The production build loses per-request timing. The next latency
  investigation needs the instrumented build, a different binary from the one that misbehaved. `CoreStats` and the
  instrumented build's byte identity (the Phase 1 gate) are the mitigation.
- **Tension: requirement 4 against CPU budget.** A strictly non-sleeping service (spin wait, spinning copy thread, no
  idle sleep) holds one to two cores busy on divix01, where cores 64-71 are reserved and the service runs unpinned
  today.
