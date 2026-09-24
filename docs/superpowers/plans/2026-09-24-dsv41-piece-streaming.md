# DSV4.1 EXL3 RAM-miss: per-piece streaming of NVMe rows to the GPU

Branch `cc/dsv41-direct-two-phase` @ `4215969073`. Design document, not code. `[V]` = verified by reading the
code at that commit, `[I]` = inferred. Line numbers are for that commit. All paths are relative to the worktree.

Revision 2 (2026-09-24) addresses the critic review (verdict REVISE): C1, H1-H3, M1-M6 and the LOW items. Stage 0
(the gain bound) is recorded in §7.1 as a result.

## 0. Decisions taken as given

K=4 sub-reads per mirror half, 8 pieces per row. Readiness is a per-lane generation-tagged **bitmask**. One
streaming kernel **S** replaces W2+C2 (8x256, leaders poll with `ld.acquire.sys`, copies use `ld.global.cv`). The
host publishes a piece only after its bytes are stored and fenced. Miss lanes are granted at reservation in a
"leased, still loading" state. The feature sits behind a new flag, off by default, and all of fail-stop, the single
writer of `keep` (finalize), per-lane leases and acks, the DIRECT commit mask, the W1 `%globaltimer` budget and D5
stay.

The microbenchmark is `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/stream_bench/`
(`stream_bench.cu`, `results.txt`). The disk probe is `…/direct-two-phase-tests/chunk_probe/`.

## 1. Current flow (two-phase)

Host, `exl3_ram_miss_host.cpp`:
1. `serve()` reserves slots under `mutex_` (`:2710-2738`). Each missing expert gets `bump_generation_locked`
   (`:2732`), then `state = kLoading` and `expert_slot[e] = slot` [V].
2. S2 grants the **hit** lanes in that same hold (`:2748-2756`). `grant_lane_group_locked` (`:2258-2305`) requires
   `kReady` (`:2274`) [V].
3. `reader_.read()` (`:2793`, body `:629-704`) blocks until every row is packed. An extent's completion runs
   `retire` (`:1129`). A row becomes `Ready` when its last extent retires (`:1148-1151`). `dispatch_ready_rows`
   (`:1205`) posts one `PackJob` per row. `collect_packed` (`:1236`) calls `finish_row` (`:1261`), which sets
   `packed[ordinal]`.
4. `_mm_sfence` (`:2817`). Rows go `kReady` plus `publish_map` only if `ok` (`:2828`). Otherwise `release_locked`
   runs with the S6 assertion `!leased_locked` (`:2833`) [V].
5. S3 grants the **miss** lanes (`:2857`). `handle_demand` (`:2880`) sets the status. The demand loop then stores
   `kDemandDone` (`:1963`), after `handle_demand` returns (`:1952`) [V].

Credit is `kQueueDepth(16) * parts(2) = 32` SQEs (`:55`, `:799`). Banks are 2 x 8 rows (`:52-54`). The pack pool
queue and `jobs_` are sized `kBounceSlots` (`:602`, `:1370`). A worker's `finished.fetch_add` is its last access to
the job (`pack_pool.h:196`), and `done()` is `finished == chunks` (`:79`) [V].

Device (`srt/layers/moe/exl3_ram_miss.py:366-387`) [V]: `memcpy → post → W1 (:609) → C1 → A1 → W2 (:734) → C2 → A2
(:864) → F (:908) → torch.add(go_total)`. T8 counts this as **10 nodes, 9 edges**, including the add. W1 zeroes
`go_1` (`.cuh:630`), and W2 is the only writer that zeroes `go_2` (`:753`). F keeps when
`kReqFailed == 0 ∧ violated == 0 ∧ go_1+go_2 == planned ∧ kFatal == 0` (`:926-929`). The DIRECT commit masks
`live & (lanes < delivered_count) & keep` (`expert_residency_gpu.py:758-760`) [V].

## 2. Target flow

| t | Who | Writes |
|---|---|---|
| reservation (under `mutex_`) | host | Reserves the slots as today. For each miss lane: store `PieceMask[idx][lane] = gen<<8`, then the RowResult payload. One `_mm_sfence`. Then RowResult.ready is tag 2 (`LOADING`) for miss lanes and tag 1 (`READY`) for hit lanes. One group, one fence (§3.2). |
| read | host owner | Splits each part into 4 sub-reads (§4.1). Each sub-read's retire sets a `landed` bit. Every piece whose dependency mask is covered is dispatched as its own `PackJob`. |
| pack | workers | Copy the chunks with NT stores, `_mm_sfence`, then `finished.fetch_add(release)`, as today. |
| publish | **host owner**, in `collect_packed` | Once a piece job's `done()` holds (acquire), the owner publishes before releasing the job: a generation-checked CAS sets the piece bit on every lane word naming this row (§3.4). |
| after read | host | `ok` gives `kReady` + `publish_map`. `!ok` quarantines leased slots (§3.3). No S3 grant. Status is set, then `kDemandDone` (`:1963`). |
| W1 | device | Zeroes `go_1` **and `go_2`**, the S block counters, and the abort word. Otherwise unchanged. |
| C1, A1 | device | Unchanged. |
| S | device | Covers every lane W1 did not claim: validates its RowResult, streams pieces by mask, watches `kDemandDone`, and commits `go_2` only if every block completed and nothing aborted (§5). |
| A2, F | device | Unchanged. |

## 3. Protocol state

### 3.1 Mask words
The masks get a new host-written area **P** in the lease block, `kLeaseRing(16) x kLeaseLanes(8)` words. Each word
is a `uint64` on its own 128 B line (16 KiB), at a new header offset `kLeaseHeaderPieceOffset`. The word is
`generation56 << 8 | bits8`. Only the service thread writes it (§3.4). The device reads it with `ld.acquire.sys`
and treats `word >> 8 != generation` as 0 bits. There are three layout copies (host `.cpp`, `.cuh`,
`ops/moe/exl3_lease_block.py`) and the layout test must agree.

The device-side generation check is **defence in depth, not the primary guard**. The word is re-initialised at
reservation and fenced before the tag-2 ready word, and S reads the mask only after acquiring that ready word, so
it cannot observe an older generation. The real hazard is on the host: a late publish from an older request
setting bits under the new generation. §3.4 closes it and U8 tests it.

### 3.2 "Leased, still loading"
- **RowResult tag 2 (`kLeaseTagLoading`)**. The payload (host slot, slot generation, expert) is final at
  reservation. W1 cannot claim a tag-2 lane: `ready_seen` requires tag 1 (`.cuh:240`) [V].
- **Grant.** A new mode of `grant_lane_group_locked` accepts miss lanes whose slot is `kLoading` and in this
  request's `slots`. Hit lanes still need `kReady`. The call is all-or-nothing, with one fence, and
  `grants_pending` is false from the start. Both groups' payloads exist at reservation, so the per-group fence rule
  (`:2250-2256`) is met.
- **Tier slot state `kQuarantine = 3`**: a leased slot whose read failed.
  - Entering quarantine **clears the mapping at once**: `expert_slot[e] = -1` and `slot_to_expert[slot] = -1`.
    Nothing is published, because the map was never published for this slot. `release_locked` (`:2638-2646`) then
    finds `expert == -1` on the later release and cannot unmap an expert that was re-read into another slot (M3).
  - `take_slot_locked` skips it: it takes only `kFree`, or evicts `kReady` (`:2607-2614`) [V].
  - `census_locked` counts it as neither free, evictable nor leased (`:2585-2595`) [V]. That is deliberate: a
    quarantined slot can never help a deferred request, so it must not trigger deferral.
- **LEASE_PROTOCOL.md E1 is amended.** E1 says a leased slot's bytes are final and immutable from readiness
  (`LEASE_PROTOCOL.md:436`, `:584`). Tag 2 is a new reader contract. With `gen == G` and tag LOADING:
  - the payload is complete and the lease exists;
  - the bytes of piece `p` are final and immutable once the device has acquired bit `p` of `PieceMask` under `G`;
  - the other bytes are unspecified.

  E1 applies unchanged to tag 1. The `.nc` argument (`:581-586`) holds only for tag 1, which is why S uses `.cv`.
  This amendment is task 1.

### 3.3 Interactions
- **retire_leases / release_lease_locked (`:2321`, `:2371`).** When a release brings `leases[slot]` to 0 on a
  `kQuarantine` slot, it calls `release_locked`. On a `kLoading` slot it only decrements, and the post-read step
  decides.
- **Post-read (`:2822-2839`).** `ok` gives `kReady` + publish. Otherwise a slot with `leases > 0` goes to
  `kQuarantine`, and one with `leases == 0` (already voided, e.g. timeout) goes to `release_locked`.
- **Void / timeout while reading.** The slot stays `kLoading`, so it is never reused while workers write it.
- **Failed read.** `quiesce()` (`:1256`) drains every job, and each drained piece is published or dropped by the
  owner before `read()` returns (§3.4). Then:
  - `read()` returning 0 ends at status `kFailed`, then `kDemandDone`;
  - S sees that and aborts with reason Failed (§5);
  - F writes `keep = 0` and a terminal naming the unacked lanes, then raises fatal;
  - the void frees the quarantined slot.
- **Shutdown / fatal.** Every S leader pass checks `kFatal` and `kLeaseHeaderShutdown`, as W2 does.

### 3.4 Publishing, and replacing S6
**Owner-side publish (H1).** `collect_packed` publishes, not the workers: the workers' `fetch_add` is their last
access to the job, and after it `collect_packed`/`quiesce` may finish the row and `dispatch_ready_rows` may re-arm
the job. For each piece job with `done()` (acquire), before its job is released, the owner does one thing per lane
word the job names:

```
loop: old = load(word); if (old >> 8) != job.gen or (old & bit): fail; CAS(word, old, old | bit, release)
```

- The acquire on `done()` synchronises with every worker's release `fetch_add`, and each worker fenced its NT
  stores first. So the bytes are globally visible before the CAS [I: x86 TSO; M1 tests it].
- The generation check makes a late publish from an older request fail rather than set a bit under the new
  generation. The bit check catches a double publish.
- On failure the owner bumps a counter and sets `c.failed`. The service thread is the only writer of the mask
  words (the grant initialises them, the owner sets bits), so the CAS never contends in practice. It is the guard.

**S6 replacement.** Old S6 guarded the host map and the slots released on failure.
1. The host map is unchanged: `publish_map` and `kReady` happen only on `ok`.
2. The released-slot rule becomes "a leased slot is quarantined, never released". It stays an `assert` at both
   sites.
3. Streaming adds device-visible bytes before the outcome is known. Claim: residency is committed only for a slot
   whose read succeeded. Proof:
   - DIRECT commit needs `keep > 0`, and F keeps only under `:926-929`.
   - S writes a nonzero `go_2` only if all blocks completed, none aborted, and S observed `kDemandDone ≥ seq` with
     status `kServed`.
   - `kServed` means `read() == 1`: every row was packed whole.
   - W1 zeroes `go_2` every replay (C1 fix), so no earlier replay's value can satisfy the equality.
   - On any other path `go_2 = 0` and `kReqFailed = 1`, so F takes the failure branch.

   The partial bytes this leaves in destination slots are the exposure C1 already has (it writes before the outcome
   is known, `exl3_ram_miss.py:376-385`) [V], and fail-stop covers it.

## 4. Host changes (`exl3_ram_miss_host.cpp`, `exl3_ram_miss_pack_pool.h`)

### 4.1 Sub-reads, gated on the flag (M5)
With the flag off, the reader is byte-for-byte today's: `K_eff = 1`. With it on, `admit_batch` (`:890`) splits each
**nonzero** part:
- `len_k = round_up(ceil(len/4), 4096)`. Stop issuing once the bytes run out, so a small part gives fewer than 4
  sub-reads and never a sub-read of 0 or fewer bytes.
- Zero-length parts (mirror weights `0:1`, `:899-906`) issue none, as today.
- The EOF clamp applies per sub-extent.
- The sub-read ordinal is its index `k` in the row's file order.

Descriptor index = `slot*parts*4 + p*4 + k`, so `descs_`/`queue_` grow 4x. Fault hooks keep meaning a *part*
(`(index/4) % parts`), plus a new `fault_.sub`. Credit stays 32.

### 4.2 Piece geometry
The table is static per row start. It is computed in **segment destination coordinates**
(`segment, dst_lo, dst_hi`), and those are the same for the host slab row and the VRAM row. Segments are sorted by
`src_offset` (`exl3_expert_format.py:101`) [V], so file order is monotone over the 9 segments.
- A piece boundary is its sub-read boundary mapped into segment coordinates, rounded down to 128 B.
- The dependency mask is the set of sub-reads its file bytes touch.
- `starts[row]` varies, so the host computes masks at admit. The device table carries only destination sub-runs,
  keyed by piece and row start. This is an open question: one table per distinct `starts mod 4096`, or per-lane
  runs passed through `lane_ctx`.
- Refuse the flag at construction if a slab or VRAM row base is not 128 B aligned.

### 4.3 Reader rework (M2)
Stated because it is hidden in "per piece":
- `jobs_[kBounceSlots]` becomes `jobs_[kBounceSlots*8]`, `runs_` grows 8x, and the pool queue capacity
  (`:602`) becomes `kBounceSlots*8`. Without this, 8x the jobs throws "packing queue overflowed".
- `BounceRow` gains `landed`, `dispatched` and `published` (8 bits each). `c.packing` counts piece jobs.
- These iterate pieces, not rows: `unfinished_jobs` (`:511-516`), `quiesce`, `collect_packed`, the loop exit
  (`c.packing == 0`) and `finish_row`. A row is finished when all its pieces are collected.
- `pack_split` now splits a piece (about 1.66 MB) across workers, not a row. Keep the knob, document the change, and
  re-measure it in task 6.
- Per-piece vetting replaces `take_ready_row`'s `filled >= needed` (`:1164`): each dependency sub-read has
  `done == expected`, and the piece's bytes lie inside its sub-reads' `[dest, dest+done)`.
- **The inline no-pool path (`pack_one`, `:1182`) has no publisher, so the flag is refused when `pack_workers == 0`.**

### 4.4 serve(), trace and env
- `serve()`: the §3.2 grant at the S2 site, the §3.3 post-read, and no S3 miss grant under the flag.
- Trace (`StageRecord`, `stage_records`):
  - `piece_cqe[kTraceRows][8]` and `piece_publish[kTraceRows][8]`;
  - `pieces_published`, `pieces_out_of_order` and `piece_publish_refused`;
  - device state words `kStreamPieces` and `kStreamPolls`, plus S first- and last-copy `%globaltimer` stamps.

  Update `test_exl3_ram_miss_device_args` and the stage-words test.
- Env (per `.claude/skills/env-var-conventions`): `SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM = EnvBool(False)`,
  beside `environ.py:1856`. It is refused unless two-phase, lease mode and `pack_workers > 0` are all on
  (the pattern at `exl3_ram_miss.py:508`).

## 5. Device changes (`exl3_ram_miss.cuh`, ops `.py`, row backend)

`exl3_ram_miss_lease_stream_kernel` runs 8 blocks x 256 threads.

**Per-lane admission.** The leader warp polls each unclaimed lane's RowResult and must pass **the whole
`lane_result_valid` contract** (`:225-248`) on a tag-2 word as on tag 1: the seqlock re-read of `ready`,
`expert == planned[i]`, and `host_slot < capacity`. The implementation parameterises `lane_result_valid` by the
accepted tag rather than duplicating it. A published but invalid result is an identity violation, as in W2.

**Poll loop (every leader pass).**
- Lanes 0..7 of the leader warp each read one lane's mask. A ballot takes the result to shared `(lane, piece)`.
- The block copies its slice with `ld.global.cv.v2.b64` and `st.global.cg`, keeps a per-block `done[lane]`, and
  runs `__syncthreads`.
- Lanes with tag 1 (hits W1 missed) are treated as mask `0xFF`.
- Each pass checks, in order:
  - the D5 deadline → Timeout;
  - `kFatal` or shutdown → Aborted;
  - **`kDemandDone ≥ seq`** (M1): status not `kServed` → Failed. Otherwise a failed read would spin to D5 and be
    reported as Timeout.
  - Status `kServed`: **re-read every lane mask with `ld.acquire.sys` after acquiring `kDemandDone`**, and only
    then judge. Every mask full → keep streaming until copied. Any mask incomplete → Identity.
    - Why the re-read: the masks from earlier in the pass may predate the owner's last CAS. The host stores
      `kDemandDone` last (`host.cpp:1962-1963`), after `read()` returned and so after every CAS (the same service
      thread). Under x86 TSO the CAS is visible before that store, and the device's acquire of `kDemandDone`
      orders the re-read after it, so a mask re-read then is final.
    - Judging on the pre-`kDemandDone` masks would raise a false Identity fatal on a complete request.

**Host-visible progress word (for G2).** The first time any block copies a piece of a request, S stores
`tagged(1, generation)` with `st.release.sys` into a new device-written word `StreamProbe[idx]` in lease area D
(8 bytes per ring index, appended after `Terminal`; layout test updated). The `state` words live in device memory
(`ops/moe/exl3_ram_miss.py:852`), so the host cannot read them.

**Termination and commit (C1).** Every block ends on exactly one of two paths, and neither skips the count:
Blocks count into **one counter word**, `ctr`. The low 16 bits count finished blocks, the high bits count
completed ones.
- **Abort path**: the leader stores `kReqFailed = 1` and `kFailReason` itself (first-writer-wins via `atomicCAS` on
  `kFailReason`), stores `abort = 1`, runs `__threadfence()`, then `old = atomicAdd(ctr, 1)`.
- **Complete path**: after its slices of every lane it runs `__threadfence()`, then
  `old = atomicAdd(ctr, 1 + (1 << 16))`.

The block for which `(old & 0xFFFF) == gridDim.x-1` is last. It decides from `new = old + its own increment`,
never from a separate re-read.
- It runs `__threadfence()`, then reads `abort` with a volatile load (`ld.relaxed.gpu`; an aborting block's store
  is ordered before its `atomicAdd` by its fence).
- It commits only if `(new >> 16) == gridDim.x ∧ abort == 0 ∧` it observed `kDemandDone ≥ seq` with `kServed` and
  the §5 re-read passed. If it has not seen that yet, the leader waits for it, still under D5.
- The commit writes the compacted `host_rows_2`, `dst_slots_2`, `origin_2` and `lane_ctx_2`, owns
  `ram_miss`/`kUnservedMisses`, and does `go_2 = n` as the last store.
- Otherwise `go_2` stays 0 from W1's reset.

A single word removes the case of two relaxed atomics on different words, where the last block sees `finished`
complete but an old `completed`.

No inter-block barrier exists, so S needs **no co-residency** for correctness. There is no hit-wait budget, and S
never writes `keep`.

**Resets (C1).** W1 zeroes `go_2` next to `go_1` (`:630`), plus `ctr` and `abort`. So every word F
and S read is fresh each replay, whatever an aborted S left behind.

**Graph.** Under the flag, `post()` becomes `memcpy → post → W1 → C1 → A1 → S → A2 → F → add`: **9 nodes, 8 edges**,
against today's 10/9. Register the kernel name (D8 pattern).

## 6. Tests

New host tests, CPU, `test/registered/unit/kernels/test_exl3_ram_miss_piece_stream.py`:
- **U1 geometry**, over every row of a synthetic layout with random `starts`, small last parts and `0:1` weights:
  - sub-reads are page aligned, nonzero, and tile each part;
  - pieces partition the needed bytes, with boundaries 128 B aligned;
  - each dependency mask is exact.
- **U2 out of order**: `reverse_cqes` plus a held sub-read. Pieces publish in dependency order, and every bit's
  bytes match the file.
- **U3 bit implies bytes**: poison fill, `pack_delay`. The test thread checks the slab bytes behind every set bit.
- **U4 tags**: tag 2 for misses and tag 1 for hits inside `read()`. The mask is initialised before the ready word.
- **U5 failed read after partial publish**:
  - setup: `part_error` on sub-read 3 of row 1;
  - asserts: `slot_info` shows **state == kQuarantine**, `leases == 1` and `slot_to_expert == -1`; a second request
    never lands on that slot; after the terminal void the slot is `kFree`.
- **U6 double publish**: `piece_publish_refused == 1`, and the read fails.
- **U7 void during read**: the slot stays `kLoading` and is freed post-read.
- **U8 publish primitive** (direct unit test of the CAS helper, exported test-only):
  - a word tagged with another generation, publish refused, word unchanged;
  - a word with the bit already set, publish refused, word unchanged;
  - a fresh word under the right generation, bit set, nothing else changed.

  The single-threaded, quiescing reader cannot reach the stale-publisher state, so the primitive is tested
  directly. A plain `fetch_or` fails the first two cases.
- **U9 quarantine then re-read elsewhere**: expert `e` fails into slot `s` (quarantined). A second request reads `e`
  into slot `t`. Voiding `s` must leave `expert_slot[e] == t`.
- **U10 flag-off reader**: with the flag off, SQE count, descriptor count and packing are identical to the
  baseline.

New GPU tests, `test/manual/dsv41/test_exl3_piece_stream_cuda.py`:
- **G1** parity against the two-phase arm (T9 rules). Run it with a pack delay of 2x `SGLANG_DSV41_RAM_MISS_HIT_WAIT_US`
  per piece, so the read outlasts W1's budget (needed to kill M7).
  - Budget check: 200 us per piece x 8 pieces is 1.6 ms per row, and at most 8 rows is 12.8 ms.
  - The D5 default is `SGLANG_DSV41_RAM_MISS_TIMEOUT_MS = 2000` (`environ.py:1832`) [V], more than 150x larger.
    The test asserts its own margin: total injected delay under 5% of the deadline.
- **G2 streams**, with one clock only: the host holds pieces 1..7 of a one-row request unpublished until it
  acquires `StreamProbe[idx] == tagged(1, gen)` (§5), then releases them. Pass means the request completes.
  Under a wait-for-all mutant S never copies, the probe never fires, and the request times out.
- **G3** failure: keep 0, the terminal names the S lanes, the DIRECT mapping is unchanged, and the reason is Failed
  (not Timeout) at well under the D5 deadline.
- **G4** timeout at about 1x the deadline.
- **G6** graph chain: 9 nodes, 8 edges.
- **G7** replay freshness: replay 1 is served; replay 2 is forced to abort in S (injected fatal on the host). Assert
  in replay 2:
  - `go_2 == 0` and `keep == 0`;
  - no LaneAck with replay 2's generation beyond the lanes C1 copied;
  - replay 1's leases all retired acked.
- **G8** masks full, but a protect row fails: `keep == 0`.
- **G9** S identity: a tag-2 RowResult with a wrong expert, or `host_slot ≥ capacity`, gives a violation.
- **G10** abort commit race: aborting one block in S (fault word) leaves `go_2 == 0` and `kReqFailed == 1`. With
  no abort, 1,000 replays all give `keep == 1`.
- **G11** late last publish:
  - setup: a host fault hook delays the owner's last CAS until just before `kDemandDone`, and a device fault delay
    makes the leader read masks, stall, then read `kDemandDone`;
  - asserts: `keep == 1` and no Identity fatal, over 1,000 replays.

Adapted tests:
- T1, T1b and T2 unchanged.
- T3 unchanged.
- T4 and T4b: the failure path now holds miss leases, and the assertion becomes "leased goes to quarantine".
- T4c: N/A under the flag. Assert `grants_pending == false`.
- T5: the mask includes S lanes.
- T6: the mutant is "S writes keep".
- T7: the shared deadline covers S.
- T8 = G6.
- T9 = G1.
- T10: all-miss, W1 still bounded.
- T11: all-hit, so S copies and acks nothing.

The existing `test_exl3_two_phase_*_cuda.py` run unchanged with the flag off.

| Mutant | Killed by |
|---|---|
| M1 publish before the workers' stores/sfence (publish at dispatch; bench `EARLY_PUBLISH`) | U3, G1 |
| M1b drop only the workers' sfence | Probably undetectable (WC drains fast). Run and record. |
| M2 publish with a plain `fetch_or`, no generation or bit check | U8 (primitive test) |
| M3 S copies a piece before its bit (copies all at start) | G1 (poison + pack delay) |
| M3b S waits for all bits before copying | G2 |
| M4 a failed read releases a still-loading slot | U5 on `state == kQuarantine` (`release_locked` does not touch leases, so no underflow fires) |
| M4b quarantine release unmaps via `slot_to_expert` without clearing it on entry | U9 |
| M5 a piece bit set twice (re-dispatch) | U6 |
| M6 `.nc` loads in S | **Expected undetectable** (bench: not detected). No non-final line is re-read (U1 alignment). M6b (`.nc` + no 128 B rounding) is run to test the argument. |
| M7 W1 accepts tag 2 | G1 with the stated pack delay |
| M8 S commits without `kServed` | G8 |
| M9 S counters/`go_2` not reset in W1 | G7 |
| M10 S writes `keep` | T6 |
| M11 S skips `lane_result_valid` on tag 2 | G9 |
| M12 aborting blocks do not set `kReqFailed` / skip the count | G10, G7 |
| M13 last block commits without fences / without checking `abort` | G10 (race-amplified by a delay in the aborting block before its store) |
| M14 S ignores `kDemandDone` in the poll | G3 (reason becomes Timeout, elapsed ≈ D5) |
| M15 S judges completeness on the masks read before `kDemandDone` (no re-read) | G11 |
| M16 two counter words (`completed`, `finished`) instead of one | G10 (a race-amplifying delay between the two adds; expect a spurious non-commit on a good request, asserted as `keep == 1` over 1,000 replays) |

Record the command next to every suite number. Run mutants in a private worktree and re-run the restored baseline.

## 7. Measurement

### 7.1 Stage 0 (DONE): upper bound on the saving
- Source: the bound-8 node trace `divix01:/mnt/nvme1/dsv41-nsys/tp-on-b8-node-20260924-015735.sqlite`, script
  `/data/models/slang/nvfp4-work/direct-two-phase-tests/gain_bound.py`.
- Per read layer, saving = `min(W2 duration, C2 − 140 us)`, where 140 us is the post-streaming tail of the last
  piece. S starts after C1 and the link is serial, so overlapping S with C1 gives the same bound.
- **Result: about 29.0 ms/token** (C2 34.8, W2 67.3 ms/token). In 226 of 1,276 read layers the window after C1, not
  the copy, is the limit.
- Control: the bound-1024 trace gives 0.1 ms/token (no window), as expected.
- **Approximate, and only for bound-8.** It assumes every piece except the last lands early enough. Three caveats:
  - The formula puts the last piece's landing at W2's end, but the last publish comes before `kDemandDone` by the
    host's post-read work. The true ceiling can be up to 140 us per read layer higher.
  - A layer with `C2 < 140 us` must contribute 0, not a negative saving. Clamp `max(0, C2 − 140 us)` when
    re-running.
  - The trace used W1's poll-count bound of 8, whereas HEAD uses the `%globaltimer` budget (100 us default), and
    the window after C1 depends on W1's exit time.

### 7.1b Stage 0b (to do, CPU analysis + one GPU trace, before task 5)
- One graph-plus-node trace at HEAD with the shipped 100 us budget, flag off.
- Re-run `gain_bound.py` with the clamp above.
- The result replaces 29.0 as the pre-registered ceiling. If it is under 5 ms/token, stop before task 5.

### 7.2 Arms
- **Node trace** (attribution only; never read ms/token or step-tail from it): S's span against C1, and the gap from
  the last piece publish (host trace) to S's end. Use `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp`.
- **Graph-mode paired arms**: `benchmarks/dsv41_baseline/run_arm.sh`, arm A with the flag off and arm B with it on,
  identical otherwise, same tenancy and clock profile, then `paired.py A B`.
- **Pre-registered sample and test**:
  - **8 timed sessions per arm**, sessions 0-7 of `CORPUS_8_SESSION_IDS`, disjoint from warm-up index 8
    (`session_subset.py:57`). Today's 2 sessions cannot reach p < 0.25.
  - The count is set **per arm** through a new arm-level override recorded in the run manifest, not by editing the
    shared `N_SESSIONS` (`session_subset.py:34`), which `EXPECTED_SESSION_IDS` (`:51`) and every other verdict
    depend on.
  - **Interleaved order.** `run_arm.sh` runs a whole arm per invocation (`:416-417`), so drift on this contended
    box would move all 8 differences together and break the sign test's independence. Pre-register:
    - 4 invocations per arm, 2 sessions each, in the fixed order **A B B A A B B A**;
    - session pairs `{0,1} {2,3} {4,5} {6,7}` go to the four A/B pairs;
    - each session is paired with its counterpart in the adjacent opposite-arm invocation;
    - each invocation is a cold server, as `run_arm.sh` already does.
  - One-sided sign test (`metrics.py:44`) at **α = 0.05**: B must win at least 7 of 8 sessions (p = 0.035).
    `paired.py` is fed the concatenated per-arm results.
- **Expected**: a decode improvement at or below the Stage 0b ceiling (about 29.0 ms/token at bound-8). A measured
  gain clearly above the ceiling means the arms differ in something other than this feature, and the run is
  investigated rather than reported.

### 7.3 Kill criterion
Stop and leave the flag off if any of these hold:
- any byte mismatch, fatal, or `piece_publish_refused`;
- a paired median gain under **5 ms/token**;
- fewer than 7/8 wins;
- flag-on read wall time per demand regresses by more than 10% at m=6 against flag-off (the host stage trace).

## 8. Staged tasks (each one a commit, with its tests; estimates re-derived about 2x)

| # | Task | Effort |
|---|---|---|
| 1 | Env flag and its refusals. Lease-block area P (3 layout copies + layout test). Tag 2 constants. Amend `LEASE_PROTOCOL.md` E1 (`:436`, `:584`) and the §2 reader contract for tag 2. No behaviour. | 1 d |
| 2 | Flag-gated sub-read split, piece geometry and per-piece vetting, while still packing whole rows and publishing nothing. U1, U2, U10, plus the existing reader, fault and pack-worker suites green with the flag on and off. De-risks credit and the fault hooks. | 3 d |
| 3 | Per-piece jobs (pool and `jobs_` resize, piece-granular `c.packing`/`quiesce`/exit), owner-side CAS publish, no-pool refusal. U3, U6, U8, M1/M2/M5. | 3 d |
| 4 | Loading grant, quarantine with mapping cleared, retire/post-read changes. U4, U5, U7, U9, adapted T1-T4, M4/M4b. | 2.5 d |
| 5 | Stream kernel S (validation, poll with `kDemandDone` and a mask re-read, one-word completion counter, fenced commit), the `StreamProbe` word in area D, W1 resets, chain rebuild. G1-G11, T5-T11, M3/M3b/M6-M16, run on divix01 through `gpu-run.sh`. | 5.5 d |
| 0b | HEAD node trace at the 100 us budget, clamped `gain_bound.py` re-run (§7.1b). It gates task 5. | 0.5 d |
| 6 | Traces, per-arm session override, the interleaved ABBA arms (8 invocations), paired verdict against §7. | 3 d |

Total about 18 working days.

## 9. Risks
- **Link contention and the C1 window** cap the gain: about 29.0 ms/token at bound-8, re-derived at HEAD in Stage
  0b. The kill criterion covers the rest.
- **4x SQEs** under fixed credit could lengthen reads at m=6. It is compared flag-off vs flag-on in task 6, with
  raising `kQueueDepth` as the fallback.
- **Owner publish latency**: the owner loop polls with `_mm_pause` while jobs are packing, so the latency should be
  about a microsecond [I]. It shows up in the host trace as `piece_publish − last worker end`.
- **Quarantine** holds a slot until its void. Acceptable because a failed demand is fatal. Say so in code.

## 10. Open questions (with recommended answers)
1. **Should S wait for `kServed` even when every mask is full?** Yes (§3.4). It costs microseconds and makes the S6
   replacement a proof.
2. **K as an env knob?** No. Make it a `constexpr` 4.
3. **Mask stride**: 128 B, as benchmarked.
4. **Lease-block version bump?** Yes. Refuse a mismatched layout at attach.
5. **Can `lane_experts` repeat an expert?** The host copies the device's lane list verbatim
   (`exl3_ram_miss_host.cpp:2149`) and does not deduplicate it [V]. The post kernel fills it from `planned`
   (`.cuh:336-337`) [V]. That the planner gives distinct experts per gather is [I]. Recommend: assert distinctness
   when the lane request is read (fail the request otherwise), and have the publish name every lane whose expert
   matches the row, so a repeat is at worst refused, never corrupt.
6. **Device piece table**: one table per distinct `starts mod 4096` (small, static), or per-lane runs in
   `lane_ctx`? Recommend the static table if the distinct count is at most 16 (measure on the real layout in task
   2), else per-lane runs.
7. **T8 node count**: settled. Today's chain is 10 nodes and 9 edges including the `torch.add`, and the streaming
   chain is 9 and 8.
