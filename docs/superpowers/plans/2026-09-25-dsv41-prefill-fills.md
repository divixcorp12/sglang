# DSV41 prefill fills: pinned-tier misses through the native reader

Item 1 of `DSV41_REFERENCE.md` §27.4. A 260-token prefill spends ~12 s with the GPU idle while the scheduler thread
fills the pinned tier. Each fill is an `Exl3ShardRowSource.read` into a bounce buffer followed by a single-threaded
CPU `copy_`, one gather chunk at a time (§27.2).

Flag: **`SGLANG_DSV41_ENABLE_PREFILL_FILLS`** (`EnvBool`, default off; `Dsv41Config.enable_prefill_fills`). With it
off, no new code path runs.

## Design

1. **The native reader does the fills.**
   - Eager pinned-tier misses are read by the RAM-miss service's own `RowReader`, the reader decode demands use.
   - With row images (`SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES`), each row is one O_DIRECT readv per mirror part,
     straight into the pinned slab row. All drives read in parallel, with no bounce buffer and no CPU copy.
   - The flag is refused without row images, and without option C's native slot table
     (`SGLANG_MOE_EXPERT_GRAPH_GATHER`).
2. **A layer's reads are issued once its routing is known.**
   - `Exl3MoEMethod._apply_streamed` has the layer's distinct experts. It reads them back once, together with their
     VRAM hot-cache slots.
   - It opens one host use for the whole layer, which syncs the stream and pauses the service thread once.
   - Inside that host use, it claims pinned slots for every expert that is neither VRAM-hot nor in the tier
     (`RamTier::fill_begin`). Claims go in ascending expert order, which is the gather chunks' order. The whole
     layer's expert set is protected, and a claim never falls back to evicting a protected row.
   - `fill_begin` starts one helper thread, which reads all the claimed rows in a single `RowReader::read`. The
     read's progress callback publishes how many rows have landed, as a prefix count.
3. **Chunk k is gathered while chunk k+1 is read.**
   - The chunk loop is unchanged. Before a chunk's `copy_rows`, `gather_rows` waits only until that chunk's highest
     fill ordinal has landed (`fill_wait`).
   - The helper keeps reading later chunks while the GPU gathers and computes this one.
4. **Overflow.**
   - A long prompt can route more misses than the tier has unprotected rows. Rows that could not be claimed are
     left to the existing chunked admission (`ensure_rows`).
   - Before `ensure_rows` claims anything, the layer's fill is finished (joined), so its slots are ordinary
     residents again.
   - `ensure_rows` then reads through a synchronous native fill.
5. **Copy engine (item 3): not done.**
   - `_gather_host_rows_kernel` already runs at the link rate: 12.33 GB/s, against 12.4-12.5 GB/s for copy-engine
     bursts (§27.2, §27.3). DMA would not move the 78 GB any faster.
   - Its only gain would be overlapping the gather with MoE compute, ~0.5 s per prefill.
   - That needs a second staging set, 64 rows × 13.3 MB ≈ 852 MB of VRAM, which the 0.83 memory fraction leaves no
     room for.
   - It also needs a way around the per-expert `torch.where` syncs in `exl3_moe_accumulate`, which serialize the host
     against the GPU anyway.
   - Left as a follow-up.

## Slot and lease ownership against the decode service

- **The pause is held for the whole layer.** The existing eager contract is unchanged: the outermost host use syncs
  the stream, retires graph-lane leases and pauses the service thread. The fill runs entirely inside that host use,
  so the helper is the reader's only user.
  - `fill_end` joins the helper before the host use ends.
  - As a backstop, `RamThread::resume` joins any fill still running before the service thread may touch the reader.
- **Claimed slots are `kReady`, mapped and published at once.** This is what Python's `assign` already does for
  eager reads. They also carry a new per-slot `filling` flag:
  - `take_slot_locked` and the victim census skip a filling slot, so no admission in the same host use can evict a
    row still being written. Examples are a later chunk's `ensure_rows` or a hot-cache promotion.
  - `release` refuses a filling slot.
- **When the read ends:**
  - The helper clears the flags under `mutex_`.
  - If the read failed, every row that did not land is released, and `fill_wait`/`fill_end` raise.
  - A hung read is caught by the service watchdog: the helper sets `busy_since_`, so the watchdog aborts after
    `max(30 s, 3 x timeout)`, the same contract as a hung demand.
- **Eviction of decode's RAM set is not made worse.**
  - The same misses are admitted, with the same LRU victim choice.
  - Protecting the layer's whole expert set means a later chunk can no longer evict an earlier-needed resident, so
    admissions can only fall.
- **Item 2 (prefill-evict, `SGLANG_DSV41_PREFILL_COLD_ADMIT`) stamps admissions.** `fill_begin` stamps in one
  place, and will take their helper once it lands.

## Sync points

| Sync | Flag off | Flag on |
|---|---|---|
| Stream sync and service pause (`before_host_use`) | once per chunk | once per layer |
| NVMe read then CPU split, on the scheduler thread | per chunk, serial | on the helper thread, overlapped |
| `fill_wait` | none | per chunk, only until that chunk's rows land |

The chunk readbacks stay: the hit counts' `.item()`, `chunk.tolist()`, and the per-expert `torch.where` in the MoE
loop.
- Where hit counts are computed, those readbacks order a slot's reuse after the queued gathers that read it, which
  the overflow path relies on.
- Removing them needs a host-planned gather and MoE loop. That is a separate change.

## Tests

- **C++ / ops, CPU (`test_exl3_ram_miss_prefill_fills.py`):**
  - `fill_begin` claims in order, protects the given set and stops at no victim.
  - Rows land byte-identical to `read_rows_once`.
  - `fill_wait` returns per prefix.
  - A filling slot is never a victim and cannot be released.
  - A failed read releases unlanded rows and raises.
  - `resume` joins a running fill.
  - Row-image tables (direct) and plain tables both work.
- **Tier, CPU:**
  - With a fake `row_fills`, `gather_rows` waits before `copy_rows`, and `ensure_rows` goes native and finishes a
    pending fill first.
  - With the flag off, `ExpertPinnedHostCache` never touches `row_fills`.
  - `test_one_field_per_knob`.
- **GPU (divix01):**
  - The manual RAM-miss suites (`analysis/dsv41-drive/native-prefetch/gpu_suite.sh`).
  - Arms on port 30021, A (flag off) then B (flag on), once each. Report TTFT per session, ms/token and byte
    identity.
