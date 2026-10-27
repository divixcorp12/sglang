# RAM prefetch: the device binds landed pool rows at post time

## Goal
A prefetched row that has landed in the RAM prefetch pool by the time its target layer's post kernel runs is typed as
a RAM hit at its pool slot, whether or not the row's previous record had a miss, so neither "no pending delta" nor
"still reading at row r-1's serve" (F125's two failed assumptions, DSV41_REFERENCE.md §33.37) loses it. Opt-in behind
a flag, default off.

## Acceptance criteria
- With `SGLANG_DSV41_RAM_PREFETCH_BIND=1`, a planned expert that row r's device map does not hold but row r's pool
  table lists (landed, at pool slot P) when the post runs is typed a hit at P (HIT_COPY, HIT_SM or HIT_CPU by the
  existing lane policy), the record carries `kRecFlagPoolBound`, and the post bumps `map_chain` as for a miss.
- The host accepts such a lane only when the record is pool-bound and its pool holds that expert landed at P; it swaps
  P into the tier (victim by LRU outside the record's protect list) and publishes the delta {victim's expert, -1},
  {expert, P}; no NVMe read happens for that expert in that record. Any other hit lane on an unmapped slot still
  fail-stops as today.
- The content invariant (Design §4.4) holds: no slot listed in a table version that a post may still read changes its
  content (reclaimed by a read, or evicted and refilled) before the host has served a record of the row whose post read
  a version without it. CPU unit tests drive each path that could break it (reclaim, demand swap, bound swap, later
  eviction) and each docstring names the mutant that turns it red.
- A post that reads the table while a writer is mid-update binds nothing from it (seqlock check); a GPU test drives a
  torn table and an ordinary one.
- With the flag off the kernel receives no table pointer and loads nothing; posts, records, deltas and publication are
  byte for byte as today; the existing lease, layout and RAM-miss tests stay green.
- The Python wire mirror (`WireLayout.cpp_constants`, `wire_probe`) carries every new constant, and
  `test_expert_stream_lease_layout.py` pins them; `kDeltaStride`, the lease block size and the 128-byte record are
  unchanged.
- Every counter in Design §6 appears in the RAM-miss thread counters line and `counters.json`, is zero with the flag
  off, and has a unit test that drives it nonzero.
- `both_cpu_ab.py` has a bind pair (T3 cap-2, bind on and its off twin, `-fixedsplit` variants) and
  DSV41_REFERENCE.md §33.38 holds the served report skeleton with every number `pending`; the owner's session runs the
  capture.

## Decisions
- D1: Approach B, a device-visible pool table bound at row r's post (owner, 2026-10-14). A (chain on every record)
  leaves "still reading" unfixed and adds a host wait to every post; C (a ring of optional deltas) is the most complex.
- D2: F125 merged to master first (`ee528f2c74`); this feature builds on master.
- D3: No new lane kind: a bound lane is an ordinary hit at the pool slot, marked by record flag bit 1. Kinds 6-15 are
  rejected by `read_record`, and the v2 record has no free per-lane bytes.
- D4: The table lives in each row's 256-byte delta stride (96 bytes used today), so the lease block keeps its size.
- D5: One table per row, written under a per-row host mutex as a seqlock: the generation is odd while a writer
  updates it. The record's flags bits 2-7 carry the table version the post read (generation / 2, mod 64).
- D6: `SGLANG_DSV41_RAM_PREFETCH_BIND` requires `SGLANG_DSV41_RAM_PREFETCH=1`, and the launch refuses it together with
  `SGLANG_DSV41_RAM_PREFETCH_PROMOTE=1` in this version (two mechanisms mapping the same pool entries).
- D7: The served A/B uses the fixed CPU split (§33.36) and the T3 cap-2 arm, as F125's did.
- Evidence (F125 served A/B, 2026-10-14, `ram-prefetch/promote-ab-20261014/`): of 5,940 landed rows, 2,670 found no
  pending delta and 2,541 were still reading at publish; promotion converted pool-served misses into hits
  (741 + 2,164 = 2,905 against off's 2,891) and moved matched time by -0.14 ms per record.

## Rules
- Device and wire files may change in this feature: `lease_layout.h`, `lease_device.cuh`, `lease_kernels.cuh`, with
  their Python mirror (`expert_lease_block.py`) and the layout tests. Nothing else in the record format changes.
- The post loads the table only when the kernel is given a table pointer (flag on), issued beside the delta loads, and
  after the delta tag's acquire when a delta is pending, so a post that applies a swap's delta also sees the table
  version that dropped the entry.
- Pool entry transitions keep `host/ram_prefetch.h`'s threading contract; a published entry leaves the landed state only
  after it is unpublished (a new table version without it).
- Comments follow `.claude/rules/comment-style.md`; tests follow `.claude/rules/unit-test-admission.md`; the env var
  follows `.claude/skills/env-var-conventions` (Dsv41Config field and default test, env docs row, DSV41_REFERENCE §32
  row).
- Update `analysis/dsv41-drive/LEASE_PROTOCOL.md` with the table, the seqlock and the content invariant.
- Tests run on divix01 through the checks runner (`--gpu` for CUDA tests); no task or check launches the server, and
  production on divix01 is untouched.

## Non-goals
- No change to prediction, scoring, or which rows are prefetched.
- F125's promotion is not removed or changed; the two flags are exclusive in this version.
- No new lane kind and no record layout change beyond the flag bits.
- No binding for rows the pool does not serve (the draft model's own CPU-expert path is untouched).

## Open questions

## Design

### 1. Today's mechanism
The post kernel (`lease_kernels.cuh` `exl3_ram_miss_post_kernel`) waits for row r's delta tag to equal
`map_chain[r]`, applies it, then types each lane from the device's mirror of the slot map (`type_lanes`). It bumps
`map_chain[r]` only when a lane is a miss (`lease_kernels.cuh:273`), and the host publishes that chain's delta. So
the host can change the device's map of row r only after a record of row r with a miss, and the change lands at
row r's next post. Pool entries (`host/ram_prefetch.h`, `SpecPool`) are RAM-tier slots of the row, `share` (at most
4) per NUMA group, that the device never maps.

### 2. Approaches considered
- **A. Chain on every record.** One-line device change; every post then waits on the host, and binding still happens at
  row r-1's serve, so "still reading" is unchanged.
- **B. Device-visible pool table (chosen).** Binding at row r's own post, the latest moment, with no wait added.
- **C. A ring of optional deltas.** Ordering optional and owed deltas on one sequence; the most complex.

### 3. Facts the protocol must respect (code survey, 2026-10-14)
- A pool entry's slot is a real tier slot of its group (`reserve_spec_pool`, `ram_tier.h:1022`), below
  `row_capacity`, state `kSpec`, never a victim (`take_victim_locked` takes only `kFree`/`kReady`). The copy paths
  (HIT_COPY, HIT_SM's C1 copy, HIT_CPU's CPU job) can read it as they read any tier slot.
- The host fail-stops a hit lane whose slot it does not map (`classify_lanes_locked`, `ram_tier.h:2120-2122`).
- The speculative thread reclaims the oldest landed entry at any time, lock-free from the device's view
  (`claimable_locked`, `ram_prefetch.h:95-107`): nothing pins an entry the device may be using.
- `take_pooled_locked` (`ram_tier.h:2209`) already swaps a landed entry into the tier for a forced CPU miss: victim by
  LRU, delta `{old, -1}`, `{expert, pool slot}`, the entry kPoolSwapped holding the victim until `release_swapped`.
- The v2 record is exactly 128 bytes; lane kinds 6-15 are rejected by `read_record`; the record flags byte has 7 free
  bits. Each row's delta stride is 256 bytes with 96 used. `map_applied` is device-only.
- Once served, a hit lane's slot is protected by the tier's existing rules (`classify_lanes_locked` stamps LRU and
  disowns it for the job), so a bound slot is safe from the moment the host serves its record.

### 4. Protocol
#### 4.1 The table
Per row, inside the row's delta stride: a u32 generation stored last with a release, then `kPoolTableEntries` =
`kNodes` x `SpecPool::kMaxShare` (8) entries of {expert i16, slot i16}, -1 when empty (40 bytes). Writers take a
per-row host mutex, store an odd generation, rewrite the entries, then release the next even generation. The host
rewrites row r's table when an entry of row r lands and when it unpublishes one.

#### 4.2 Device binding
The post reads the generation (acquire), the entries, and the generation again; on an odd or changed generation it
binds nothing. A planned expert the map misses but the table lists is typed a hit at that slot by the existing lane
policy. A post that bound any lane sets `kRecFlagPoolBound` (flags bit 1), writes the version it read
(generation / 2 mod 64) in flags bits 2-7, and bumps `map_chain`. With no table pointer (flag off) none of this runs.

#### 4.3 Host serve of a bound lane
`classify_lanes_locked` accepts a hit lane on a `kSpec` slot only when the record is pool-bound and the row's pool
holds that expert landed at that slot; it swaps it in through the `take_pooled_locked` path (victim by LRU outside
the record's protect list, delta {old, -1} and {expert, slot}, entry kPoolSwapped until `release_swapped`) and
unpublishes it in the same table rewrite. Anything else still fail-stops.

#### 4.4 The content invariant and pinning
A slot listed in any table version a post may still read must not change content until the host has served a record
of the row whose post read a version without it. Concretely:
- A listed landed entry is pinned: the claimer does not reclaim it. To reclaim or drop it, the host unpublishes it
  (new version) and reclaims it only after serving a record of the row reporting that version or later.
- A listed entry swapped into the tier (by a forced miss's `take_pooled_locked` or by a bound lane) is unpublished in
  the same rewrite, and its slot carries the unpublishing version: `take_victim_locked` does not evict it until a
  served record of the row has reported that version or later.
- The host never moves a row's version more than 31 past the last version a served record reported, so the mod-64
  comparison is exact; past that it delays publishing (counted).

#### 4.5 Flag
`SGLANG_DSV41_RAM_PREFETCH_BIND`, default off, plumbed like `SGLANG_DSV41_RAM_PREFETCH_PROMOTE` (Dsv41Config field,
`RamPrefetchConfig`, FFI arg, `enable_ram_prefetch`, startup log line, launch refusal). Off: no table pointer reaches
the kernel, and the host never writes a table.

### 5. Costs, and what measures each
- **A post that follows a pool-bound record waits for the host's delta.** A pool-bound record bumps the chain as a
  miss does, so a row with a bound hit and no miss now owes a delta it did not before. Measured: the post's delta
  wait after a pool-bound record (InstrBuild event), and the A/B wall time.
- **Pinning can stop a read from being issued.** With all of a row's pool entries pinned, a new read has no claimable
  entry and is dropped. Measured: `spec_bind_pinned_drop`.
- **Each bound row evicts an LRU victim**, as the demand-time swap does today (F125: 5% of victims were routed by
  the layer). Measured: `spec_bind_victim_routed`.
- **The post loads the table every record**: about 40 bytes of pinned host memory, one PCIe round trip (~1-2 us)
  issued beside the delta loads, at most ~40-80 us per verify; the lookup (8 lanes x 8 entries on thread 0) is
  negligible. Measured: post kernel duration in a node-mode trace, flag on vs off.
- **Host bookkeeping**: a table rewrite per landing and per unpublish. Negligible.
- **Flag off costs nothing**: the kernel receives the table pointer only with the flag on.

### 6. Counters
Added to the `Counter` enum and `is_core_counter` (`host/tier_protocol.h`), `COUNTERS`/`CORE_COUNTERS`
(`expert_stream_transport.py`), `spec_margin_capture.counter_summary` and `both_cpu_ab.py` `RAM_KEYS`:
- `spec_bound`: lanes the device bound from the table (counted at serve).
- `spec_bound_used`: bound lanes swapped into the tier (equals `spec_bound` unless a check fails).
- `spec_bind_published`: table rewrites that added an entry.
- `spec_bind_pinned_drop`: reads not issued because every entry of the row was pinned.
- `spec_bind_unpin_wait`: reclaims or evictions deferred by the content invariant.
- `spec_bind_victim_routed`: bound-lane victims the same record's routes name.
- `spec_bind_version_stall`: publications delayed by the 31-version window.

## Testing
- **CPU unit tests** (join `test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py` and
  `test_exl3_ram_prefetch_step.py`, using `prefetch_rig` and `ChainSim`; extend `ChainSim` to model the table read,
  the seqlock and the flag bits): a landed entry is published and a simulated post binds it; the host swaps it in and
  the delta maps it; each content-invariant path (reclaim, demand swap, bound swap, later eviction) waits for the
  reported version; a torn read binds nothing; the version window stalls; every counter driven nonzero; flag off
  writes no table.
- **Layout tests**: the new constants in `test_expert_stream_lease_layout.py` (Python mirror equals the C++ trait,
  v2 literals, stride and record size unchanged).
- **GPU test** (new `test/manual/dsv41/test_exl3_ram_prefetch_bind_cuda.py` on `lease_chain_rig.py`): the real post
  kernel binds a listed expert as a hit at the pool slot with the flag bits and chain bump; a torn table and a null
  table pointer bind nothing; existing lease kernel and ordering GPU tests stay green.
- **Served A/B** (owner-run, not a check): the bind pair with fixed split, 8 sessions, instrumented; report wall,
  matched per record, covered, other reads, the post wait after bound records, and every §6 counter in §33.38.
  Output text is compared against run-to-run variation (§33.37), not for identity.
