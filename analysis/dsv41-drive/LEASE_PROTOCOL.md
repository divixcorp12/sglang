# The slot-map protocol

How a captured decode step reads EXL3 expert rows that are not in VRAM. The device keeps its own copy of every
layer's RAM-tier map (expert -> pinned slot), types each lane of a plan from it, and posts one record. The host checks
the record, serves the misses into staging slots, and answers with a delta to the map. No lease is granted and no hit
is waited for. This describes the protocol as built (plan `docs/superpowers/plans/2026-09-30-dsv41-device-slot-map.md`).
The lease protocol it replaced (RowResult, W1, Done, deferral) is in `git log` of this file.

Paths are relative to `python/sglang/kernels/jit/csrc/moe/expert_stream/` unless they start with `python/`. The wire
constants are in `lease_layout.h`, mirrored by `python/sglang/kernels/ops/moe/expert_stream_transport.py` and
`expert_lease_block.py`; `test_exl3_ram_miss_device_args` checks the mirrors.

## The lease channel

The device-to-host handoff every expert-stream client uses, as one template (`LeaseChannel<Spec>`):
`lease_channel_layout.h` (the spec and the gate word), `lease_channel.cuh` (the device half), `host/lease_channel.h`
(the host half) and `stream_wait.h` (the wait node). A client is a `ChannelSpec<Head, Ring, Records, RecordBytes, Done,
Gate>`: where its head word, record ring, done words and gate sit. The record's payload, what the host does with it,
and the client's data areas are its own; only the protocol is shared. The target is the first client
(`TargetChannel` in `lease_layout.h`, over the request page and the completion block below).

- **Page and lease.** Two pinned host areas. The page is device-written and host-read: the head word (u32, the last
  posted seq, stored with a release) and a ring of `Records` records. The lease is host-written and device-read: the
  done words and the gate.
- **The seqlock.** A record's first word is its seq. The device stores 0 there, fences (release), writes the payload,
  then stores the seq last with a release (`begin_record`, `end_record`). The host copies the record once between two
  acquire loads of the seq word and keeps it only if both read the seq it expects (`read_seqlocked`). The head is
  published after the record (`publish_head`).
- **G.** `advance` gives the next seq, never 0; a wrap of the 32-bit seq bumps the epoch. G = epoch << 32 | seq never
  repeats.
- **done[].** One u64 per ring record, at `Done + 8 * ((seq - 1) % Records)`. The host stores G there with a release
  once the record's work is complete (`complete`).
- **The gate.** Its word is `(seq & 0x1FFFFFFF) << 2 | 1`, with bit 31 set while closed (`gate_word`). It starts
  open(0), so a wait that armed nothing passes.
- **The Dekker pair.** Two sides race to open the gate:
  - device (`close_gate`): store closed(G) relaxed, `fence.sc.sys` (`__threadfence_system`), then load done[G] with
    acquire. If it reads G, it opens the gate itself.
  - host (`complete`): store done[G] = G with release, `seq_cst` fence, load the gate. If it reads closed(G), it CASes
    closed(G) to open(G).

  Each side's store precedes its load behind a full fence, so at least one side sees the other's store. Both write the
  same word, so a double open is harmless. A stale completion for G that meets closed(G + k) fails its CAS.
- **The wait node.** `enqueue_gate_wait` queues `cuStreamWaitValue32_v2` GEQ open on the gate, a cyclic compare on the
  bit-31 encoding; under capture it is a memory-op node of the graph.
- **The commit trap.** After the wait, the gate is only the wake-up; done[G] == G is the commit. `commit_or_trap`
  loads it with acquire, which orders the host's results before every later device read of them, and traps otherwise.
  Only a teardown opens a gate without done[G] (`open_closed_gate`), and the process is ending then.

### The second client: the DSpark draft

The draft's CPU experts (`draft_channel.h`, `host/draft_cpu_thread.h`) run on the same channel, so a draft MoE layer's
call sits inside the draft's decode graph as one post, one gate wait and one commit.

- **Spec.** `DraftChannel = ChannelSpec<Head 0, Ring 128, Records 4, RecordBytes 128, Done 640, Gate 768>`, in one
  4096-byte pinned buffer (`kChannelBytes`) that holds the page and the completion block together. The target keeps
  them in two areas.
- **Record.** seq u32 @0 (the seqlock word), stage u16 @4, rows u8 @6, k u8 @7, epoch u32 @8; the other bytes of the
  128-byte slot are unused. It names the CPU share of one call. The inputs and outputs are not in it: they are the
  draft's own pinned areas (`DraftCpuAreas`), `[stages, kMaxRows = 16, ...]` each, indexed by the stage.
- **One record per wait.** The draft posts at most one record before its finish waits on the gate, so the ring never
  laps. The host takes a head that is more than one past the record it expects (a lap), and a record whose seq word
  does not match (torn), as protocol failures and fail-stops.
- **Completer.** The CPU expert engine thread, worker 0 of its team. It reads the record, runs the stage's M-row
  forward over the staged x and routes, and calls `channel::complete` (done[G], then the Dekker open of the gate).
- **Idle.** The thread polls the channel's head between quanta of its own idle wait (`cpu_experts.h`,
  `kIdleQuantumNs`): the GPU's release store of the next head is seen within one, with no syscall. Nothing sleeps,
  because the GPU cannot ring a futex.
- **Failure.** A watchdog checks every 20 ms and fail-stops a record that stays incomplete for `fatal_wait_ns`.
  `stop()` joins both threads and then opens a gate still held closed (`open_closed_gate`), as at the target's teardown.
- **Not shared.** The two clients share the protocol only: separate buffers, areas and threads.

## Parties

- **Device.** One linear chain per layer, in one stream, capturable in a graph
  (`Exl3RamMissRowBackend.post`, `python/sglang/srt/layers/moe/exl3_ram_miss.py`):
  post -> C1 (row copy) -> S (stream) -> CW (copy wait) -> stream wait on the gate -> CC (commit).
  The device owns the **map bank** (`ExpertStreamDevice.map_bank`): `ram_slot[rows, experts]` (int32, -1 unmapped),
  `staging[rows, 8]`, `map_chain[rows]` (starts at 1), `map_applied[rows]` (starts at 0), and per row `ce_ok` and
  `dst_rows` (set at attach by `set_row_copy`, copy engine on) and `cpu_ok` (set when the layer registers with the
  CPU expert service, `set_row_cpu`).
- **Service thread** (`host/ram_tier.h`, `host/ram_thread.h`). The tier's single owner. It handles records in
  sequence, chooses victims, publishes deltas and reads misses. Idle, it spins with PAUSE, or, with
  `SGLANG_DSV41_RAM_MISS_SPIN_CORE` (busy_poll), with no PAUSE on that core, for `SGLANG_DSV41_RAM_MISS_IDLE_SPIN_US`
  (default 5 ms), then sleeps 50 us between polls (20 us between checks while parked); -1 never sleeps.
  `start_thread` refuses busy_poll unless no SMT sibling of the core is in the server's affinity or among the CPU
  experts' cores.
- **Copy thread** (`host/copy_engine.h`). Copies a record's copy-engine hits with `cuMemcpyAsync`, runs its CPU jobs
  through the CPU expert thread, and publishes CopyDone. Idle, it spins with PAUSE on a core of its own
  (`ThreadingConfig.copy_cpus`) for `SGLANG_DSV41_RAM_MISS_COPY_IDLE_SPIN_US` (default 5 ms), then sleeps on a futex
  doorbell that every submit and `stop()` ring (a syscall only while it sleeps); -1 never sleeps.
- **CPU expert thread** (`host/cpu_experts.h`). Computes CPU lanes; its output is two parts per row. Its team
  (`host/cpu_experts/team.hpp`) is its own for its lifetime: `threads - 1` pinned workers that never sleep, the thread
  itself worker 0. Between jobs every worker runs the kernel's register work for
  `SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US`, then PAUSE, until the next job; the thread polls the submit word, the draft
  head, stop and detach the same way. No OpenMP runtime is on the path.
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
CPU; CopyDone), `HIT_SCRATCH`=6 (a GPU drip's scratch slot, then C1s; stream order; see "The GPU drip").

**Completion block** (20480 B, `kLeaseBlockBytes`, 4096-aligned), then the delta block. `CopyDone` and the gate are
the target's lease channel (["The lease channel"](#the-lease-channel)):

| Offset | Area | Writer |
|---|---|---|
| 0 | `PieceMask[16][8]`, one 128-byte line per lane: `G << 8 \| piece bits`; at +8 in the line, `PieceSource`: `G40 << 24 \| slot + 1` | host |
| 16384 | `CopyDone[16]`: G once every DMA and CPU job of G completed | copy thread |
| 16512 | the gate | CW closes, copy thread or CW opens |
| 16640 | `kCopyArmed` u32: 1 once the service armed its copy engine | host |
| 16768 | `kSplit` i32[9]: CPU lanes per n eligible lanes | host (`store_split`, at start and on retune) |
| 16896 | `kScratch` u64[32]: the GPU drip's scratch table | copy thread (`host/gpu_drip.h`) |
| 20480 + 256 row | the row's delta: `tag` u64 @0, `count` u32 @8, `staging` i16[8] @16, 16 `{i16 expert, i16 slot}` entries @32 | host |

`lease_block_bytes(rows, wire) = wire.lease_block_bytes + round_up(rows * wire.delta_stride, 4096)`; at 8 lanes that is
`20480 + round_up(rows * 256, 4096)`. A delta entry maps `ram_slot[expert] = slot`; slot -1
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
3. **S** copies the `MISS_GPU` lanes from their staging slots, piece by piece as PieceMask bits for G appear. When a
   lane's first bits appear and its `PieceSource` word carries G's low 40 bits, S copies from that slot instead
   (bounds-checked like the staging slot); a word left by an earlier record at the ring index names nothing.
4. **CW** reads the `HIT_COPY` lanes' small tensors itself when SM small copies are on. If any lane is `HIT_COPY` or a
   CPU kind, it closes the gate for G and checks CopyDone (see "Copy engine"). It leaves `ce_mask` (int32[3]) for CC:
   `{the HIT_COPY and CPU lanes, the CPU lanes, the CPU output parts used}`, each a bit set indexed by lane (the parts:
   bit 0 for a `HIT_CPU` lane, bit 1 for a `MISS_CPU` lane).
5. **Stream wait.** `cuStreamWaitValue32` GEQ open on the gate; no SM spins.
6. **CC** traps unless `CopyDone == G` when the gate was armed. It writes `cpu_lanes` (int32[2]) for the fused MoE:
   `{the CPU lane mask, the parts}`, where part 0 (the CPU hits' sum) is bit 0 of the parts word if it holds this
   record's partial and part 1 (the CPU misses') is bit 1. The route tables seed the output with the flagged parts' sum
   and leave the CPU lanes out; DIRECT's commit reads the lane mask and inserts only the copied lanes.

## N lanes

The wire is built for `kLanes = round_up(N, 8)` lanes, `N` the widest planned gather (1..32), so `kLanes` is 8, 16, 24
or 32 (`lease_layout.h`'s `LeaseLayout<NumLanes>`; `Wire` is the build's instantiation, `wire_layout(N)` its Python
mirror). At 8 lanes the layout is byte-identical to the one above.

- `kKindWords = kLanes / 8`: the kinds are 4-bit fields, eight to a u32, from `kRecKinds`.
- Packed nibble counts (lanes in bits 0-3, protect ids in bits 4-7 of the `counts` byte) exist only at `kLanes == 8`.
  Above that the `counts` byte is the lane count and the protect count is its own byte at `kRecProtectCount =
  kRecKinds + 4 * kKindWords`.
- The header ends at `kRecHeaderBytes = round_up(kRecProtectCount + (packed ? 0 : 1), 16)`; the four i16 arrays and the
  f32 weights follow it, and `kRecordBytes = round_up(kRecLaneWeight + 4 * kLanes, 128)`.
- The completion block grows with the lanes: `PieceMask[16][kLanes]` is `16 * kLanes * 128` bytes, then `CopyDone`, the
  gate, `kCopyArmed` and `kSplit` (a u32 per eligible-lane count, `kLanes + 1` of them), rounded up to 4096. The delta
  record holds `staging` i16[kLanes] and `2 * kLanes` entries, rounded up to 256.

| kLanes | `kRecordBytes` | `kRecHeaderBytes` | `kPageBytes` | `kLeaseBlockBytes` | `kDeltaStride` | `lease_block_bytes(1)` |
|---|---|---|---|---|---|---|
| 8 | 128 | 32 | 2176 | 20480 | 256 | 24576 |
| 16 | 256 | 32 | 4224 | 36864 | 256 | 40960 |
| 24 | 384 | 48 | 6272 | 53248 | 256 | 57344 |
| 32 | 512 | 48 | 8320 | 69632 | 512 | 73728 |

`kPageBytes = 128 + 16 * kRecordBytes`. The lane masks (`ce_mask`, `cpu_lanes`, the HIT_COPY and CPU sets) are u32, so
lane 31 is bit 31.

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

   **Pool serve (`SGLANG_DSV41_RAM_PREFETCH_POOL_SERVE=1`).** At step 3, a `MISS_GPU` lane whose row already landed in
   its group's pool is not read: the host writes the pool slot into the lane's `PieceSource`, then every PieceMask bit
   in one release. With a victim the pool slot becomes the expert's RAM slot in the delta and the victim takes its place
   in the pool (swapped until the delta is written, as a forced swap's); the lane's staging slot stays staging. With no
   victim the row is not inserted and the entry stays swapped until the group's next posted record, which the device
   posts only after this chain (and so S) ended. A pool row still being read is not waited for: the lane is read.
5. **CPU lanes go straight from the service to the CPU expert thread.** The service claims one job sequence per job
   the record can need (the hits', one per `MISS_CPU` lane) and submits the hits as part 0 before the copy job. Each
   `MISS_CPU` lane goes into part 1 as soon as its read landed; rows landing together share a job, and every job after
   the record's first adds into the part, so its fp32 sum order follows landing order. The last miss job takes the last
   claimed sequence. The copy job's completion waits for the DMA and for that sequence before it stores CopyDone.
6. The host mirror (`slot_map`, read by the eager Python paths) shows an insert only after its bytes land; a victim's
   unmap is written to it at victim choice, before any read.

A record with no lanes only stamps recency. A record with lanes reads its hot sidecar: if the sidecar
was lapped, an all-`HIT_SM` record is counted as an overrun and skipped, and any other record fails stop.

**Deferred publication (`SGLANG_DSV41_RAM_PREFETCH_PROMOTE=1`).** With the flag off, step 3 publishes the delta as
above and nothing here applies. With it on, rows that some source row targets (`RamPrefetchConfig::target`; row 0 and
untargeted rows publish immediately) change step 3:

- **Stash.** The record's combined entries, count and chain are held in the `NumaNodeDistributor` (a pending word per
  row, claimed by CAS) instead of being written. The previous delta of that row must still be chain - 1, else the
  record fails.
- **Row r-1's write.** The serve of row r-1's record in the next forward claims row r's pending delta, writes it
  through the same `write()` (entries before the tag), then releases its swapped victims. This runs in
  `RamTier::pump_demand` for every record that is not torn, before `handle_record` and on the lapped-hot skip path.
- **Backstop.** The same place publishes any pending delta of row r or an earlier row that was stashed in an earlier
  forward (the forward counter advances when the row number does not increase); a delta stashed in the caller's own
  forward, as another group's serve of the same record finds it, is left for row r-1's serve. The claim checks the
  stash's forward against the word it CASes, so a delta re-stashed meanwhile is never taken. It is counted in
  `spec_publish_backstop`. A delta is therefore written before row r's next record is served, and no later than the
  device's deadline allows.
- **No delta.** A group that loses row r-1's claim counts row r's landed entries in `spec_promote_no_delta` only when
  no delta is pending and no group's serve of the same forward claimed one, and only its own pool's entries.
- **Write order.** The claimer copies the entries, frees the pending word, then writes: once the tag is out the device
  may post row r's next record, whose stash must find the word free.
- **Promotion at publish.** The claimer, before writing, maps landed pool rows of row r into the delta: each takes a
  victim (no wanted list), appends the victim's unmap and the expert's `{expert, pool slot}`, and the pool entry becomes
  swapped holding the victim until this delta is written. A delta never exceeds `Wire::kDeltaMaxEntries`; what does
  not fit is skipped and counted. Only a delta already pending can carry promotions, so the device types the promoted
  expert as a hit at row r's next post.
- **Row lock.** `NumaGroup::row_mutex[row]` is per (row, group). The group's service thread takes row_mutex, then the
  pool mutex, while it touches row r's tier. The claimer takes the pool mutex, then try_locks row_mutex; when the row
  is busy it leaves that group's entries landed and counts each in `spec_promote_row_busy`.
- **Trace.** Each deferred write emits `delta_publish` (row, chain, trigger 0 immediate, 1 row r-1's serve, 2 backstop,
  3 flush; promoted count).

## The GPU drip

`SGLANG_DSV41_GPU_DRIP_SLOTS` (default 0, off; `host/gpu_drip.h`). The copy thread copies experts the GPU scorer
predicts for its target layer (`SGLANG_DSV41_RAM_PREFETCH_HORIZON` ahead) from their pinned RAM rows into a few VRAM
scratch slots while no demand copy is pending. A landed slot's word in the scratch table (`Wire::kScratch`, a u64 per
slot: 0, or `tag << 32 | row << 16 | expert`) makes the target's post type that lane `HIT_SCRATCH` (6), and C1s, a
second launch of C1's kernel from the scratch tensors to the row's destinations, copies it in stream order after C1.
The lane takes no staging slot and is not eligible, so its node's split reads one lane fewer.

- **Picks.** After the scorer's kernels, the drip select kernel (`spec_score.cuh`) publishes the target row's
  `SGLANG_DSV41_GPU_DRIP_PICKS` best experts that are mapped in the device map and not VRAM-hot (native scorer: best
  gate score over the record's tokens; learned head: its block score) after the record's candidates, under a seq word
  of its own (`SpecCandidates::kCandDripSeq`). The copy thread reads every posted record's row (a seqlock read of its
  first words) and, for a source row, that slot. A pick is admitted only if its expert is still mapped in the host
  mirror, no slot holds it, a free slot exists (else `drip_no_slot`), and the target row has not posted since the
  source record.
- **Source rows.** A drip copies from a RAM slot the service may evict meanwhile. Every mirror change bumps a per
  `[row, expert]` generation after the store, then fences, before the caller can rewrite the slot it unmapped
  (`RamTier::publish_mirror`). The drip reads the generation, then the mapping, at admission, re-checks both before its
  first chunk, and publishes only if the generation is unchanged after its last chunk completed: a RAM slot is
  rewritten only after its expert is unmapped, so an unchanged generation proves the copy read the expert's bytes.
- **Priority.** One chunk of at most `SGLANG_DSV41_GPU_DRIP_CHUNK_KB` (one mark) is in flight at a time, issued only
  while no demand job is queued and the last demand job's DMA has completed. Drip and demand copies share the copy
  stream, so a demand copy issued meanwhile waits behind at most one chunk. SM-driven demand traffic (C1, S) shares the
  link with at most that one chunk too.
- **Publish.** After the last chunk's mark completed (the completion word orders the DMA's writes) and the source
  check, the copy thread sets the slot's identity for the service, then stores the word with a release. The post loads
  the first `slots` words as atomic u64s, then an acquire fence, before its lanes are typed; C1s reads the scratch in a
  later kernel. A drip never published is never typed.
- **Release.** A record of the target row (scanned by the copy thread) clears a published word, then a `seq_cst`
  fence, then reads the demand head H. A post that read the word before the clear is at most H + 1, and post H + 2
  starts only after H + 1's chain (C1s included) ended, so the slot is reused only once the service handled record
  H + 2 (`handled_through`), which also checked every lane that could name it. A filling drip whose target posts first
  is abandoned (freed when a chunk in flight lands). A lapped or torn record hides rows: every drip is abandoned and
  every published word cleared.
- **Check.** The service checks each `HIT_SCRATCH` lane: a captured record with host lanes, and the slot's identity
  (published or retiring) is the lane's row and expert (`GpuDrip::holds`), else a fail-stop. It counts `drip_used`.
  `HIT_SCRATCH` is device-copied like `HIT_SM`: nothing waits on the host for it, so such a record may lap.
- **Counters.** `drip_issued` (first chunk), `drip_landed` (published), `drip_used`, `drip_abandoned`,
  `drip_no_slot`; InstrBuild events `drip_admit`, `drip_issue`, `drip_land`, `drip_abandon`, `drip_retire` (the copy
  thread's `<layout>-drip` job trace) and `drip_use` (the service's).
- **Outputs.** A scratch lane's destination bytes equal a copied lane's. A lane the drip moves off the CPU is computed
  on the GPU instead, so a step's sums can differ from the drip-off run's in the last bits, as any split change does.

## Deltas and the bulk delta

The decode chain's map changes travel only as per-row deltas, one per record with a miss, numbered by `map_chain`.
The eager paths (prefill fills, `assign`, `release`) change the tier with the service paused
(`before_host_use`), and each change is appended to a host list. `Exl3RamMissService.after_host_use` takes it (`take_bulk_delta`, int32 `[n, 3]` rows of
`{row, expert, slot}`; it joins a running fill first, so a failed fill's unmaps are in it) and launches
`map_bulk_apply`. That kernel applies every row's pending decode delta first, then the bulk entries, so the device map
ends equal to the host's. The host publishes each row's first delta (tag 1: its staging slots) when the service starts
(`reserve_staging`: the first `min(8, capacity - 1)` slots of every row, taken before any slot is filled), so it evicts
nothing and needs no bulk entry.

With deferred publication, every pending delta is written before a caller takes the tier: `RamThread::pause` flushes
after the service threads park, `stop` after they join, and `bulk_delta_count` (so `take_bulk_delta`) before the
bulk is counted (`RamTier::flush_pending`, trigger 3). `map_bulk_apply` skips an unwritten tag, so a missed flush
would leave the device map behind the host's.

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

**Deferred publication keeps this.** A deferred delta is written before row r-1's next gate opens, so the device
has it at row r's next post: the chain check at that post sees every unmap the host made, and a promoted victim is
unmapped there, before any later record can read it.

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

**The gate and the Dekker pair** are the lease channel's (["The lease channel"](#the-lease-channel)): CW is the
device side (`close_gate`), and the copy thread's `copy_completed` is the host side (`complete`).

**CC and the CPU parts.** The gate is only the wake-up; `CopyDone == G` is the commit. The CPU partial sums reach the
fused MoE through the CPU thread's done word, the copy thread, the CopyDone release and CC's acquire.

**Teardown.** With no copy thread left, a closed gate would hold its stream forever. `stop_thread` and close call
`open_closed_gate`, whose CC then traps unless CopyDone is there.

**A verify's CPU lanes (plan 2026-10-06-dsv41-dspark-both-cpu-experts).** A verify gathers with a lane per route (36
at 6 tokens of top-6, on a 40-lane wire whose lane masks are u64). A row's pinned input holds the verify's tokens
(`cpu_tokens_max`, 6 at `speculative_dspark_block_size=5`), then a token table the post writes for every lane: the token
count, a mask of the tokens that route the lane's expert, and each such token's weight (`cpu_token_table.h`). A record's
CPU job runs one forward of the record's tokens from it, and each token's partial is its own `[hidden]` row of the part,
which the route tables add to that token. The record's summed lane weight still serves one-token rows.

**Spill.** With `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` (V) below the lanes, DIRECT gives VRAM victims and
staging slots to the first V lanes only and marks the rest with destination slot -1. The post makes each marked lane a
CPU lane, outside the split: a RAM hit runs from its RAM slot, and an NVMe miss gets slot -1. The host reads each such
miss into a RAM victim of its node's range, taken before the staging misses take theirs. The CPU computes it there,
and the tier keeps it (the record's delta maps it). Every node range is checked at attach to hold staging + lanes +
VRAM-hot slots, so a victim always exists, and a live miss always has its staging slot (live <= V = staging per node).
Both are asserted (fail_record, __trap), never handled. The post overflows only when forced lanes cannot be CPU lanes
at all: before the copy engine arms (or before a row's CPU layer is registered). Then it serves the live prefix,
writes that count, and sets DIRECT's overflow flag and the layer's counter, and the DSpark worker re-runs the verify
eagerly. The draft (the second client) keeps its own channel and areas, on the same stream strictly before the
verify. From Task 11 on it shares the GPU node's CPU expert team: one thread, one job at a time, its hold watching the
doorbell and the channel head.

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
3. **`stop`:** opens any closed gate, then joins the service thread and the watchdog, then flushes every pending
   deferred delta (a no-op with the promote flag off). The copy and CPU threads end
   when the tier is destroyed after `close`.
4. **Free the slabs.**

If the barrier fails or times out, or the process is exiting, the tiers are **quarantined**: never unregistered or
freed, with every device buffer (the map bank included) kept alive. A chain waiting on a paused service is not ended by
shutdown; it waits for its deadline, where it traps (`test_exl3_slot_map_kernels_cuda.py` pins the trap).

## Tests

- **CPU** (the real service against `ChainSim`, `python/sglang/test/dsv41_chain_sim.py`, which types lanes with the
  Python reference): `test/registered/unit/kernels/test_exl3_ram_miss_*.py`, chiefly `test_exl3_ram_miss_slot_map.py`
  and `test_ram_slot_map.py`. Every failure is a subprocess that must die of SIGABRT with its FATAL line
  (`run_host_script`, `assert_aborted` in `python/sglang/test/dsv41_ram_miss_fixtures.py`).
- **Promotion** (CPU, `test_exl3_ram_prefetch_swap.py` for promotion and release, `test_exl3_ram_prefetch_step.py` for
  publication order, backstop, flush and the promote counters): the typed kinds come from `ChainSim.post`. The GPU
  case, a promoted expert typed as a hit by the real post, is a new `test/manual/dsv41/*_cuda.py` on
  `lease_chain_rig.py`.
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
