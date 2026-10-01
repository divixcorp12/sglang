# The slot-map protocol

How a captured decode step reads EXL3 expert rows that are not in VRAM. The device keeps its own copy of every
layer's RAM-tier map (expert -> pinned slot), types each lane of a plan from it, and posts one record. The host checks
the record, serves the misses into staging slots, and answers with a delta to the map. No lease is granted and no hit
is waited for. This describes the protocol as built (plan `docs/superpowers/plans/2026-09-30-dsv41-device-slot-map.md`).
The lease protocol it replaced (RowResult, W1, Done, deferral) is in `git log` of this file.

Paths are relative to `python/sglang/kernels/jit/csrc/moe/expert_stream/` unless they start with `python/`. The wire
constants are in `lease_layout.h`, mirrored by `python/sglang/kernels/ops/moe/expert_stream_transport.py` and
`expert_lease_block.py`; `test_exl3_ram_miss_device_args` checks the mirrors.

## Parties

- **Device.** One linear chain per layer, in one stream, capturable in a graph
  (`Exl3RamMissRowBackend.post`, `python/sglang/srt/layers/moe/exl3_ram_miss.py`):
  post -> C1 (row copy) -> S (stream) -> CW (copy wait) -> stream wait on the gate -> CC (commit).
  The device owns the **map bank** (`ExpertStreamDevice.map_bank`): `ram_slot[rows, experts]` (int32, -1 unmapped),
  `staging[rows, 8]`, `map_chain[rows]` (starts at 1), `map_applied[rows]` (starts at 0), and per row `ce_ok` and
  `dst_rows` (set at attach by `set_row_copy`, copy engine on) and `cpu_ok` (set when the layer registers with the
  CPU expert service, `set_row_cpu`).
- **Service thread** (`host/ram_tier.h`, `host/ram_thread.h`). The tier's single owner. It handles records in
  sequence, chooses victims, publishes deltas and reads misses.
- **Copy thread** (`host/copy_engine.h`). Copies a record's copy-engine hits with `cuMemcpyAsync`, runs its CPU jobs
  through the CPU expert thread, and publishes CopyDone.
- **CPU expert thread** (`host/cpu_experts.h`). Computes CPU lanes; its output is two parts per row.
- **Watchdog** (`host/ram_thread.h`, `watch`). Samples every 20 ms; aborts on a busy episode held past `fatal_wait` or
  a gate held closed past the copy-wait timeout.

## Wire (v2)

Two pinned host areas, both read by the device through UVA.

**Request page** (2176 B, `kPageBytes`): `demand_head` @0 (the last posted seq, stored with a release) and a 16-record
ring @128, 128 B a record. A record is one 128-byte-aligned block, so the host's L2 fetches its second line with its
first. A record, behind a seqlock on `seq`:

| Offset | Field | |
|---|---|---|
| 0 | `seq` u32 | 0 while the payload is rewritten, the seq stored last |
| 4 | `row` u16 | |
| 6 | `counts` u8 | lanes (at most 8) in bits 0-3, protect ids in bits 4-7 |
| 7 | `flags` u8 | `CAPTURED` = 1 |
| 8 | `chain` u64 | the row's map-chain number; 0 when no lane misses |
| 16 | `epoch` u32 | so the host forms G = epoch << 32 \| seq |
| 20 | `kinds` u32 | lane j's kind (below) in bits 4j..4j+3 |
| 32 | `protect` i16[8] | every routed expert of the request, -1 past the count |
| 48 | `lane_expert` i16[8] | -1 past the lane count |
| 64 | `lane_slot` i16[8] | a hit's RAM slot or a miss's staging slot |
| 80 | `lane_dst` i16[8] | the VRAM destination slot |
| 96 | `lane_weight` f32[8] | the lane expert's routing weight |

Ids are i16, so a launch with more than 32767 experts or slots per row is refused (`kRecIdMax`). The post writes the
payload between the two seq stores as one u32, one 8-byte and seven 16-byte relaxed stores; a page off 128-byte
alignment is refused at the post's launch.

Lane kinds: `HIT_COPY`=1 (the copy thread's DMA; CopyDone), `HIT_SM`=2 (C1; stream order), `HIT_CPU`=3 (the CPU from the
RAM slot; CopyDone), `MISS_GPU`=4 (NVMe into staging, then S; PieceMask), `MISS_CPU`=5 (NVMe into staging, then the
CPU; CopyDone).

**Completion block** (20480 B, `kLeaseBlockBytes`, 4096-aligned), then the delta block:

| Offset | Area | Writer |
|---|---|---|
| 0 | `PieceMask[16][8]`, one 128-byte line per lane: `G << 8 \| piece bits` | host |
| 16384 | `CopyDone[16]`: G once every DMA and CPU job of G completed | copy thread |
| 16512 | the gate | CW closes, copy thread or CW opens |
| 16640 | `kCopyArmed` u32: 1 once the service armed its copy engine | host |
| 16768 | `kSplit` i32[9]: CPU lanes per n eligible lanes | host (`store_split`, at start and on retune) |
| 20480 + 256 row | the row's delta: `tag` u64 @0, `count` u32 @8, `staging` i16[8] @16, 16 `{i16 expert, i16 slot}` entries @32 | host |

`lease_block_bytes(rows) = 20480 + round_up(rows * 256, 4096)`. A delta entry maps `ram_slot[expert] = slot`; slot -1
unmaps. Sixteen entries are an insert and an eviction per lane. The post reads the payload with one u32 load and five
16-byte loads, all issued together after the tag's acquire.

## The chain

1. **post** (one block, thread 0 for the map):
   1. If the plan has lanes, it **applies the row's pending delta**: it waits, with a bounded spin, until the delta's
      `tag` equals `map_chain[row]`, and traps at its deadline if the host never publishes it. If `map_applied[row]`
      already equals that tag it does nothing; otherwise it writes the entries and the staging list and sets
      `map_applied`. So each delta is applied exactly once.
   2. It **types the lanes** (`type_lanes`, transcribed from `python/sglang/srt/layers/moe/ram_slot_map.py`). A mapped
      expert is a hit at its RAM slot; the m-th unmapped expert is a miss into the m-th staging slot. A lane is
      CPU-eligible when the post is captured, the copy engine is armed (`kCopyArmed`), CPU experts are on and the
      row is registered with them (`cpu_ok`), and it is a hit, or a miss with `SGLANG_DSV41_CPU_EXPERTS_MISSES`. The CPU takes the last `split[n]` eligible
      lanes in plan order: the fused plan sorts lanes by residency key, highest first, so those are the coldest. Other
      hits are `HIT_COPY` when copy is allowed (captured, armed, `SGLANG_DSV41_RAM_HIT_COPY=ce`, the row's copy table
      registered, and the lane's destination below the row's `dst_rows`), else `HIT_SM`; other misses are `MISS_GPU`.
      It traps on a plan wider than 8 or than its buffers, an expert out of range, a repeated expert, a hit slot past
      the row's capacity, a miss with no staging slot, or a split entry above n.
   3. With a CPU lane it stages x (fp16) into the row's pinned input row, every thread, before the record.
   4. If any lane misses it bumps `map_chain[row]` and puts the new number in the record: the host answers that
      record with the delta under this number. It writes the hot sidecar, the record (seqlock: seq 0,
      release fence, payload, seq with a release), and `demand_head` with a release.
2. **C1** copies the `HIT_SM` lanes from their RAM slots into their destination slots.
3. **S** copies the `MISS_GPU` lanes from their staging slots, piece by piece as PieceMask bits for G appear.
4. **CW** reads the `HIT_COPY` lanes' small tensors itself when SM small copies are on. If any lane is `HIT_COPY` or a
   CPU kind, it closes the gate for G and checks CopyDone (see "Copy engine"). It leaves `ce_mask` for CC: the
   `HIT_COPY` and CPU lanes in bits 0-7, the CPU lanes in 8-15, the CPU parts used in 16-17.
5. **Stream wait.** `cuStreamWaitValue32` GEQ open on the gate; no SM spins.
6. **CC** traps unless `CopyDone == G` when the gate was armed. It writes `cpu_lanes` for the fused MoE: the CPU lane
   mask in bits 0-7, bit 8 if part 0 (the CPU hits' sum) holds this record's partial, bit 9 if part 1 (the CPU misses')
   does. The route tables seed the output with the flagged parts' sum and leave the CPU lanes out; DIRECT's commit
   reads bits 0-7 and inserts only the copied lanes.

## The host per record

`RamTier::serve_record`, for record G of row r:

1. **Stamp the protect set's recency, then check every lane against the tier.** A hit must name the slot that holds
   its expert; a miss must name a staging slot, for an expert the tier does not hold, and no two misses one slot; no
   expert twice; copy and CPU lanes only on rows registered for them. A record with a miss must carry the row's next
   chain number, one without must carry 0. Any mismatch is a fail-stop: a device map that disagrees with the host's is
   a protocol error, not a case to repair.
2. **Submit the copy job** (the `HIT_COPY` lanes, and the `HIT_CPU` lanes as a part-0 CPU job) at once.
3. **Choose victims and publish the delta before any read.** One victim per miss: a free slot first, else the LRU
   READY slot that is not VRAM-hot, not routed by this record, not filling and not staging. Each miss's staging slot
   becomes its expert's RAM slot and its victim becomes a staging slot; the delta carries the inserts, the evictions
   and the new staging list, entries before the tag (release). A miss with no victim keeps its staging slot, is not
   inserted, and is counted in `ram_insert_skipped`; the delta is still published, with whatever entries the record
   has, possibly none.
4. **Read the misses** into their staging slots, publishing each piece's PieceMask bit as it lands.
5. **CPU lanes go straight from the service to the CPU expert thread.** The service claims one job sequence per job
   the record can need (the hits', one per `MISS_CPU` lane) and submits the hits as part 0 before the copy job. Each
   `MISS_CPU` lane goes into part 1 as soon as its read landed; rows landing together share a job, and every job after
   the record's first adds into the part, so its fp32 sum order follows landing order. The last miss job takes the last
   claimed sequence. The copy job's completion waits for the DMA and for that sequence before it stores CopyDone.
6. The host mirror (`slot_map`, read by the eager Python paths) shows an insert only after its bytes land; a victim's
   unmap is written to it at victim choice, before any read.

A record with no lanes only stamps recency. A record with lanes reads its hot sidecar: if the sidecar
was lapped, an all-`HIT_SM` record is counted as an overrun and skipped, and any other record fails stop.

## Deltas and the bulk delta

The decode chain's map changes travel only as per-row deltas, one per record with a miss, numbered by `map_chain`.
The eager paths (prefill fills, `assign`, `release`) change the tier with the service paused
(`before_host_use`), and each change is appended to a host list. `Exl3RamMissService.after_host_use` takes it (`take_bulk_delta`, int32 `[n, 3]` rows of
`{row, expert, slot}`; it joins a running fill first, so a failed fill's unmaps are in it) and launches
`map_bulk_apply`. That kernel applies every row's pending decode delta first, then the bulk entries, so the device map
ends equal to the host's. The host publishes each row's first delta (tag 1: its staging slots) when the service starts
(`reserve_staging`: the first `min(8, capacity - 1)` slots of every row, taken before any slot is filled), so it evicts
nothing and needs no bulk entry.

## Why it is safe without leases

A slot's bytes are rewritten only by a miss read into it, and a miss reads only into a staging slot. A slot becomes
staging only as a victim. So the question is whether a victim can still be read by anyone:

1. **Not by the record that chose it.** The victim is never one of the record's routed experts, and the record's own
   reads are of its hit slots and its staging slots.
2. **Not by an earlier record.** At most one record is in flight: every layer's chain runs in one stream, and the
   device posts G + 1 only after G's chain ended. That end covers every reader of G: C1 and S by stream order (or PDL,
   below), CW's SM reads before its barrier, and the copy thread and the CPU before CopyDone, which CC waits for.
3. **Not by a later record** until the device has applied the delta that unmapped it, which happens at that record's
   post.
4. **Not by the eager paths.** They run with the service paused and the stream synchronized
   (`before_host_use`/`after_host_use`), and read through the host mirror, which shows a row only once it landed.

A miss's bytes are read only after they land: S waits for PieceMask, the CPU's late job for the read, and the next
post maps the staging slot only after this chain, and so its reads, ended.

## Generations and reuse

`G = epoch << 32 | seq`, 56 bits; the device skips `seq == 0` at the wrap and bumps `epoch`. PieceMask and CopyDone
carry G, so a word left from ring index `idx`'s previous record (G - 16) never matches. The gate is one word, carrying
`seq & 0x1FFFFFFF`; a stale opener fails its CAS on the exact closed(G) value.
Reuse needs no seqlock beyond the record's: only one record is in flight, so the host never writes G + 16's words while
anything reads G's. Delta tags carry the 64-bit chain number, which never repeats.

## Copy engine

Armed by the service once `COPY_ENGINE_ARM_DECODES` (16) decode forwards have run after the first copy-engine capture
(`_copy_engine_barrier`, armed in `fail_stop_check`),
with `CUDA_MODULE_LOADING=EAGER` and a module-load guard; the host then writes `kCopyArmed`. A kernel's first launch
while a copy wait holds the stream can stall the copy thread's driver calls until the watchdog aborts. Only a captured
post can type copy or CPU lanes, so an eager forward is always served by C1 and S.

**The gate.** Its word is `(seq & 0x1FFFFFFF) << 2 | 1`, with bit 31 set while closed. It starts open(0), so a copy
wait that armed nothing passes. The stream waits with `cuStreamWaitValue32` GEQ open, a cyclic compare on the bit-31
encoding.

**The Dekker pair.** Two sides race to open the gate:

- CW: store closed(G) relaxed, `fence.sc.sys` (`__threadfence_system`), then load CopyDone with acquire. If it reads
  G, it opens the gate itself.
- Copy thread (`copy_completed`): store CopyDone = G with release, `seq_cst` fence, load the gate. If it reads
  closed(G), it CASes closed(G) to open(G).

Each side's store precedes its load behind a full fence, so at least one side sees the other's store. Both write the
same word, so a double open is harmless. A stale copy thread for G that meets closed(G + k) fails its CAS.

**CC and the CPU parts.** The gate is only the wake-up; `CopyDone == G` is the commit. The CPU partial sums reach the
fused MoE through the CPU thread's done word, the copy thread, the CopyDone release and CC's acquire.

**Teardown.** With no copy thread left, a closed gate would hold its stream forever. `stop_thread` and close call
`open_closed_gate`, whose CC then traps unless CopyDone is there.

## Fail-stop

Every failure ends the process; the protocol carries no error state.

- **Host:** `fail_stop` prints `FATAL ...` and calls `std::abort()`: a lane the tier does not hold, a miss outside the
  staging list, a map chain out of order, a whole record with a kind or count the device never writes (a torn record
  is an overrun; a malformed one is not), a failed or faulted read, a record with no hot set, a failed
  copy-engine issue or query, a copy-engine ring overflow, a CPU miss whose row never landed, a failed CPU forward,
  among others (`fail_stop` call sites in `host/`). **Not every abort precedes the chain's last piece:** the service
  thread aborts before it publishes the failing piece, but a copy-thread or CPU-thread failure can come after S has
  copied every miss. Either way CopyDone is never stored, so CC never commits the step.
- **Watchdog:** a busy episode held past `fatal_wait`, or a gate closed past the copy-wait timeout, aborts. It runs
  off the copy thread, so a copy thread stuck in a driver call still ends the device's wait.
- **Device:** `__trap()`. The post traps on the typing errors listed under "The chain", a CPU lane with no input row,
  and a delta not published by its deadline or malformed (more than 16 entries, an entry out of range);
  `map_bulk_apply` on an entry out of range. S traps at its deadline on a piece never published, CC on a
  missing CopyDone, the route tables on a part-1 flag with a one-part row. A trap kills the context, and the step never
  emits a token.

## PDL

With `SGLANG_DSV41_ENABLE_LEASE_PDL`, post, S and CW launch with PDL; C1 and CC are plain launches. Each PDL kernel's
first action is `griddepcontrol.wait`, which returns only after its primary completed and flushed its memory, so
completion is transitive: when CW runs, C1 and S have completed. The early `launch_dependents` only lets the
secondary be scheduled.

## Shutdown

`Exl3RamMissService.shutdown` refuses new forwards and then:

1. **`_establish_gpu_completion`:** a device barrier while the service still serves, so every in-flight chain
   finishes normally.
2. **`close_admission`,** a host flag.
3. **`stop`:** opens any closed gate, then joins the service thread and the watchdog. The copy and CPU threads end
   when the tier is destroyed after `close`.
4. **Free the slabs.**

If the barrier fails or times out, or the process is exiting, the tiers are **quarantined**: never unregistered or
freed, with every device buffer (the map bank included) kept alive. A chain waiting on a paused service is not ended by
shutdown; it waits for its deadline. `test/manual/dsv41/test_exl3_task5_item6_shutdown_gpu.py::test_shutdown_ends_a_
gpu_reader_waiting_on_the_service_without_waiting_out_its_timeout` asserts otherwise (from an older design with a
shutdown word) and fails at the merge base too.

## Tests

- **CPU** (the real service against `ChainSim`, `python/sglang/test/dsv41_chain_sim.py`, which types lanes with the
  Python reference): `test/registered/unit/kernels/test_exl3_ram_miss_*.py`, chiefly `test_exl3_ram_miss_slot_map.py`
  and `test_ram_slot_map.py`. Every failure is a subprocess that must die of SIGABRT with its FATAL line
  (`run_host_script`, `assert_aborted` in `python/sglang/test/dsv41_ram_miss_fixtures.py`).
- **GPU** (`test/manual/dsv41/`, the production chain against the real service through `lease_chain_rig.py`):
  - `test_exl3_slot_map_kernels_cuda.py`: the CUDA typing against the reference (200 random maps per switch), a delta
    applied once, the deadline trap, a miss streamed from staging;
  - `test_exl3_lease_kernels_cuda.py`: misses, hits, eviction, captured replays through ring reuse, PDL, the CPU input;
  - `test_exl3_lease_ordering_cuda.py`: a record's victim rewritten under its own delayed readers, copy-engine hits
    typed while the service is paused, armed and unarmed replays, the first armed replay;
  - `test_exl3_copy_engine_cuda.py`, `test_exl3_cpu_lane_order_cuda.py` (CPU hits and CPU misses),
    `test_exl3_moe_split_parity_cuda.py` (the two CPU parts);
  - `test_exl3_ram_miss_graph_gpu.py`: the captured MoE end to end, with the device map checked equal to the host's
    after every multi-layer replay step.
