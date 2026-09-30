# The lease protocol

How a captured decode step reads EXL3 expert rows that are not in VRAM: the device posts a request, the host service
leases the rows' pinned slots and publishes them, and the device copies them and says when it is done reading. This
describes the protocol as it is. Earlier designs (the error words, Terminal, per-lane acknowledgements, SlotGen, the
two-phase and advisory modes, native prefetch) are in `git log` of this file.

Paths are relative to `python/sglang/kernels/jit/csrc/moe/expert_stream/` unless they start with `python/`. The wire
constants are in `lease_layout.h`, mirrored by `python/sglang/kernels/ops/moe/expert_stream_transport.py` and
`expert_lease_block.py`; `test_exl3_ram_miss_device_args` checks the mirrors.

## Parties and wire

- **Device.** One linear chain per layer, in one stream, capturable in a graph
  (`Exl3RamMissRowBackend.post`, `python/sglang/srt/layers/moe/exl3_ram_miss.py`):
  post -> W1 (hit wait) -> C1 (row copy) -> S (stream) -> CW (copy wait) -> stream wait on the gate -> CC (commit).
- **Service thread** (`host/ram_tier.h`, `host/ram_thread.h`). The tier's single owner: it serves demands in sequence,
  reads missing rows into pinned slots, and grants leases.
- **Copy thread** (`host/copy_engine.h`). With the copy engine armed, it copies a captured request's resident lanes
  with `cuMemcpyAsync` and publishes their completion.
- **Watchdog** (`host/ram_thread.h`, `watch`). Samples every 20 ms; aborts on a busy episode held past `fatal_wait` or
  a gate held closed past the copy-wait timeout.

Two pinned host areas, both read by the device through UVA:

- **Request page** (2112 B): `demand_head` (device: the last posted seq), `demand_done` (host: every request up to this
  seq is served), and a 16-record ring. A record is `{seq, row, protect_count, armed, protect[8]}` behind a seqlock on
  `seq`. A record is armed exactly when its request has lanes.
- **Lease block** (28672 B, 4096-aligned), indexed by `idx = (seq - 1) % 16`:
  - `RowResult[idx][lane]` (host): `ready = tag << 56 | G` and `host_slot`. Tags: READY, LOADING, COPYING, CPU.
  - `PieceMask[idx][lane]` (host): `G << 8 | piece bits`, one 128-byte line per lane.
  - `CopyDone[idx]` (host): G once every COPYING and CPU lane of G completed. Then `COPY_GATE`, on its own line.
  - `LaneRequest[idx]` (device): `{gen, count, flags, expert[8], dst_slot[8], weight[8]}`; flag CAPTURED.
  - `Done[idx]` (device): G, written by CW.

## The chain

1. **post** writes the hot sidecar record (GPU-hot mode), the LaneRequest (payload, then `gen` with a release), the
   demand record (seqlock), and `demand_head`, one release ordering them all.
2. **Service.** Under its reservation hold, before reading anything, it grants every lane: a resident expert READY, a
   missing one LOADING, and for a CAPTURED request with the copy engine armed, a resident one COPYING or CPU. CPU
   goes to the last `split[n]` of the n COPYING-eligible lanes: the plan sorts miss lanes by residency key, highest
   first, so those are the lowest-scored RAM hits, and the rest are copied and inserted into VRAM. Each
   grant counts the lease first, writes `host_slot`, and after one `_mm_sfence` stores each ready word with a release.
   It then reads the missing rows as up to four sub-reads per part, publishing each piece's bit in PieceMask as it
   lands. Last comes `_mm_sfence` and a release store of `demand_done` (`RamTier::pump_demand`).
3. **W1** polls each lane's RowResult for at most `hit_wait_ns`. READY lanes go to C1, compacted (source row and
   destination slot in one order). COPYING and CPU lanes are marked for CW. LOADING or unpublished lanes stay S's. It
   never waits on `demand_done`.
4. **C1** copies W1's READY lanes into their destination slots.
5. **S** copies every lane W1 did not claim, piece by piece as bits appear, until the request is served and every
   piece is copied.
6. **CW** reads the small tensors of COPYING lanes itself when SM small copies are on, then publishes `Done[idx] = G`.
   If the request has COPYING or CPU lanes, it closes the gate and checks CopyDone.
7. **Stream wait.** `cuStreamWaitValue32` GEQ open on the gate; no SM spins.
8. **CC** traps unless `CopyDone[idx] == G` (when the gate was armed), and publishes the CPU lanes to the fused MoE.

## Fail-stop

Every failure ends the process. The protocol carries no error state: `demand_done` means served, and nothing else.

- **Host:** `fail_stop` prints `FATAL ...` and calls `std::abort()`. This covers a missing victim slot, a failed or
  faulted read, a GPU-hot request with no hot set (a lapped or malformed sidecar), a failed copy-engine issue or
  query, and a CPU-expert forward that failed. Every abort happens inside `handle_demand`, before the `demand_done`
  store, so S never judges a partial request served.
- **Watchdog:** a busy episode held past `fatal_wait` ("a request stayed in service"), or a gate closed past the
  copy-wait timeout ("a copy wait held the decode stream"), aborts. It runs off the copy thread, so a copy thread
  stuck in a driver call still ends the device's wait.
- **Device:** `__trap()`. W1 traps on a plan wider than 8 lanes. S traps at its deadline while unserved, and on a
  served request with a lane never granted (a lapped record) or a piece never published. CC traps on a missing
  CopyDone. A trap kills the grid and the context, and the step never emits a token.

**S's deadline is the only unbounded device wait, and it is a trap on purpose.** The watchdog's busy episode does not
cover a service thread that exited, an indefinite deferral (no busy episode), an armed record lost to a lap, or
closed admission. The deadline also bounds the at-exit case, where the watchdog is already stopped.

**S has no inter-block commit.** Each block acquires `demand_done` itself and re-reads every lane's mask after that
acquire (a mask read earlier in the pass may predate a later publish). It traps on its own. A failing block kills the
whole grid, so no block can finish on a request another block judged failed.

**No F kernel.** The post rewrites the device's `pending` state word every request, so there is nothing to clear
after the chain. The delivered count is the plan's: every lane is delivered or the process stops.

## Generations

`G = epoch << 32 | seq`, 56 bits. The device skips `seq == 0` at the 32-bit wrap and bumps `epoch`, and the service
advances with the same skip (`skip_zero`), so `demand_done` never takes the value 0 (`test_exl3_ram_miss_wrap`). Every
host-written word a device kernel acts on carries G: the RowResult ready word, the PieceMask word and CopyDone. The
gate carries `seq & 0x1FFFFFFF`. A reader compares the full G, so a word left over from `idx`'s previous request
(G - 16) never matches.

## Reuse

Ring index `idx` is reused by request G + 16. Four facts make every word of it safe to rewrite without a seqlock:

1. **The device posts G + 16 only after G's chain ended.** All layers' chains run in one stream, so at most one armed
   request is in flight. The LaneRequest's `gen`-last write is therefore never met half-written by a service acting on
   it.
2. **The service grants G + 16 only after G's lease row retired.** An `Outstanding` entry stays active until Done(G)
   (and CopyDone for its copied lanes). A demand that would reuse an active entry is deferred and counted in
   `deferred_reuse`.
3. **RowResult has no seqlock.** The grant writes `host_slot` and then, after an sfence, the ready word carrying G + 16.
   No reader of G still runs (fact 2), and a reader of G + 16 reads `host_slot` only after acquiring a ready word that
   names G + 16. `test_exl3_ram_miss_lease_publication` pins the order.
4. **W1, S and CW read RowResults only inside their own chain,** after its post. Nothing reads idx across chains.

A demand whose only victims are leased is deferred rather than refused (counted once in `deferred`, with no busy
episode). The leases it waits for belong to earlier requests whose Done or copies are already on their way.

## Done

`Done[idx] = G`, stored once by CW thread 0 after `__syncthreads(); __threadfence_system()`. It replaces the
per-lane acknowledgements, the SM-read acknowledgement and the terminal record. It covers three kinds of reader:

- **C1 and S** read G's leased slots in earlier kernels. Stream order, or PDL's wait (see below), makes them complete
  before CW starts.
- **CW's own SM reads** of COPYING lanes' small tensors finish before the barrier.
- **Nothing after CW reads a leased slot.** The fused MoE reads the VRAM destinations and the CPU out rows only.

The service acquires Done and releases every non-copy lease of G whose Done equals G (`RamTier::retire_leases`). A
COPYING or CPU lease is released only by the owner draining the copy thread's completion (`release_copied_owned`). If
the row has SM entries, the copy thread hands the job back only once `Done >= G` (`copy_acked`), because CW reads those
slots too. The cost: a hit's lease is held until CW, not until C1. The only contender is a later request on the same
row, which cannot be posted before this chain ends.

## Copy engine

Armed by the service after `COPY_ENGINE_ARM_DECODES` captured decode forwards (`Exl3RamMissService._arm_copy_engine`),
with `CUDA_MODULE_LOADING=EAGER` and a module-load guard. A kernel's first launch while a copy wait holds the stream
can stall the copy thread's driver calls until the watchdog aborts. Only a captured post sets CAPTURED, so an eager
forward is always served by C1.

**The gate.** Its word is `(seq & 0x1FFFFFFF) << 2 | 1`, with bit 31 set while closed. It is initialised to open(0),
so a copy wait that armed nothing passes. The stream waits with `cuStreamWaitValue32` GEQ open, a cyclic compare on
the bit-31 encoding.

**The Dekker pair.** Two sides race to open the gate:

- CW: store closed(G) relaxed, `fence.sc.sys` (`__threadfence_system`), then load CopyDone with acquire. If it
  reads G, it opens the gate itself.
- Copy thread (`copy_completed`): store CopyDone = G with release, `seq_cst` fence, load the gate. If it reads
  closed(G), it CASes closed(G) to open(G).

Each side's store precedes its load behind a full fence, so at least one side sees the other's store and opens. Both
write the same word, so a double open is harmless. A stale copy thread for G that meets closed(G + k) fails its CAS
and leaves the later wait closed.

**CC and the CPU lanes.** The gate is only the wake-up; `CopyDone == G` is the commit. The CPU expert thread's partial
sums reach the fused MoE through its done word, then the copy thread, then the CopyDone release, then CC's acquire.
The stream wait alone is not relied on for the visibility of host-written rows.

**Teardown.** With no copy thread left to follow, a closed gate would hold its stream forever. `stop_thread` and close
call `open_closed_gate`, whose CC then traps unless CopyDone is there. The process is ending either way.

## PDL

With `SGLANG_DSV41_ENABLE_LEASE_PDL`, post, W1, S and CW launch with PDL; C1 and CC are plain launches. Each PDL
kernel's first action is `griddepcontrol.wait`, which returns only after its primary has completed and flushed its
memory. That is the same guarantee a plain launch gets from stream order. Since each primary waited on its own primary
first, completion is transitive: when CW runs, C1 and S have completed, which is what Done relies on. The early
`launch_dependents` only lets the secondary be scheduled; it gives no ordering.

## Shutdown

`Exl3RamMissService.shutdown` refuses new forwards and then:

1. **`_establish_gpu_completion`.** A device barrier while the service still serves, so every in-flight chain
   finishes normally and every lease retires. No shutdown word exists; none is needed, because admission is still open
   during the barrier.
2. **`close_admission`,** a host flag.
3. **`stop`.** It joins the service and copy threads, and opens a closed gate (see "Copy engine").
4. **Free the slabs.**

If the barrier fails or times out, or the process is exiting, the tiers are **quarantined** instead: never
unregistered or freed. A GPU reader of unknown state may still run, and `cudaFreeHost` can synchronize. An interrupt
raised inside the barrier skips `close_admission` and quarantines. A service thread hung in a read ends in the
watchdog's abort.

## Tests

- CPU (the protocol against `LeaseSim`, `python/sglang/test/dsv41_lease_sim.py`):
  `test/registered/unit/kernels/test_exl3_ram_miss_*.py`. Every failure is a subprocess that must die of SIGABRT
  with its FATAL line (`run_host_script`, `assert_aborted` in `python/sglang/test/dsv41_ram_miss_fixtures.py`).
- GPU (the production chain against the real service, `test/manual/dsv41/lease_chain_rig.py`):
  - `test_exl3_lease_kernels_cuda.py`: misses, hits, eviction, captured replays through ring reuse, PDL, mirrors,
    and S's deadline trap;
  - `test_exl3_copy_engine_cuda.py`: armed and unarmed replays, the ballast, SM small copies, and the copy-wait abort;
  - `test_exl3_ram_miss_graph_gpu.py`: the captured MoE end to end.
