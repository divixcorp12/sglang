# DSV4.1 layer-major prefill for long prompts: design

Status: design, revision 2 (2026-09-27). Revision 1 was reviewed against the code (verdict: revise). This
revision addresses every finding; §14 maps each finding to the section that answers it.

Branch: `cc/layer-major-prefill`.

Background: `DSV41_REFERENCE.md` §25.4 (NUMA placement), §27.2, §27.16 (decoder bounded replay), §27.17 (chunk
size, and the traced 4096-token chunk).

## 1. Goal

Serve prompts of at least 250k tokens with the lowest achievable time to first token (TTFT) on divix01's RTX 5090.
Decode must not start cold after a long prefill.

**Today.** With 4096-token chunks, every chunk re-streams its layers' experts. The traced chunk (§27.17) spends:
- 11.8 s waiting on NVMe: 82 GB at 7 GB/s, the line rate of the two Gen3 x4 mirrors;
- then 9.3 s gathering over PCIe: 115 GB at 12.3 GB/s, the link's rate;
- against ~1.5 s of compute.

A 250k prompt is ~61 chunks, ~17-20 min.

**The change.** Layer-major prefill runs every chunk of a long prompt through layer L before layer L+1. Each layer's
experts are then used by many chunks in a row, so they cross NVMe and PCIe once per prompt, not once per chunk.

### Success criteria

1. A 250k-token prompt reaches its first token in under 5 minutes, and never within reach of the scheduler watchdog
   (§5.3).
2. Output equivalence with today's chunked path, within a measured tolerance, defined in §10.2.
   - Bitwise identity is not promised: the chunked path's fused mHC carry-over is reproduced (§6.2) but not proven
     bit-exact under a changed call order.
3. The first decode step after a long prefill costs a normal step plus the hot-cache restore (~1.3 s), not the
   ~200+ ms cold steps seen after prefill today.
4. No allocator OOM retries, and a device peak no higher than a normal 4096-token chunk.

### Assumptions (confirmed with the user)

- Only long prompts take the new path; shorter prompts keep today's chunked path unchanged.
- One request at a time (`max_running_requests=1`). No decode step runs while a long prefill is in flight.
- The GPU hot cache may be borrowed for the whole prefill, provided decode's hot set is recorded first and reloaded
  in bulk before the first decode step.

## 2. Scope and trigger

- **Trigger:** `SGLANG_DSV41_LAYER_MAJOR_PREFILL_MIN_TOKENS` (EnvInt, default 0 = off). A request whose uncached
  prompt suffix is at least this long takes the layer-major path. The intended production value is 32768, set only
  after phase 3.
- **Refused at launch** (a clear `ValueError` naming the conflicting option) when set with any of:
  - `max_running_requests > 1`;
  - a speculative algorithm;
  - `enable_decoder_swa_bounded_replay` off;
  - `enable_encoder_swa_bounded_replay` on (the ring below assumes the paged window allocator);
  - DP attention, context parallelism or two-batch overlap.
- **Refused at admission** by `LayerMajorGate`, falling back to the chunked path with a log line, when the request:
  - asks for prompt logprobs;
  - asks for hidden-state return (today this only raises inside the forward, `deepseek_v4.py:4407-4414`, which
    would stop the scheduler);
  - has a suffix longer than the host state store holds (§6.3).
- **Prefix hits:** a long prompt with a cached prefix runs layer-major over its uncached suffix only. The existing
  boundary cap is unchanged: the match stops `input_len - window` tokens in (`schedule_batch.py:1595-1598`,
  `1613-1616`), with the window from `swa_reprefill_tail_tokens()` (`c515312414`).

## 3. Facts the design rests on

Verified against the code; file:line in the review notes.

- **Model shape.** 40 layers, hidden 5120, `hc_mult` 4, 384 routed experts per layer, one expert row 13,315,584 B.
  - `kv_source_layer_ids` [2, 8, 14, 20]; `index_source_layer_ids` [2, 8, 14, 20, 24, 28, 32, 36];
    `candidate_source_layer_id` 20.
  - `engram_layer_ids` [1, 14]; `sliding_window` 128; `index_topk` 512.
- **The layer loop** (`_forward_layers_hc_pre_from_prev`, `deepseek_v4.py:4427-4607`) carries `hidden` [T, 4, 5120]
  bf16 and `prev_pre` [T, 4] fp32 between layers.
  - At prefill sizes it also carries the fused post-combine-norm output, `normalized` [T, 5120] bf16, from
    `mhc_post_combine_norm_prefill` (`:3519-3536`, on by default via `SGLANG_OPT_USE_TILELANG_MHC_POST`).
  - The next layer consumes it directly (`:3239-3243`). Dropping it changes both the next layer's input and the
    current layer's post kernel (`:3537-3541`).
- **Per-forward attention state** lives in the per-forward `DSV4Metadata` object; the backend holds one
  `forward_metadata` slot (`deepseek_v4_backend.py:2429-2448`). Inside it:
  - a per-token page table [T, ceil(seq/256)] int32;
  - window page indices;
  - freshly allocated top-k buffers [T, 512] int32, written in place by index-source layers and read by the later
    layers of the same ratio;
  - layer 20's candidate masks [rows, lc] bool, built today for every row of the chunk.

  At 250k context one chunk's metadata is ~50 MB.
- **Compressed KV and index-K** go to per-layer pools at `raw_out_loc // ratio`, not to metadata.
  - The ratio-2 pairing state is per layer and per request slot, so running chunks in order within a layer is safe.
  - 4096-token chunk boundaries are even, as the pairing requires.
- **Engram:** history is per request slot. Each extend's hashes use `forward_batch.engram_history` (the preceding
  n-1 tokens), so each sub-batch needs its own history window (precedent: `encoder_swa_replay.py:59-66`).
- **Decoder bounded replay:** layers 21-39 process each extend's last `min(128, extend_len)` rows, with the window
  floored at the tail start (`late_layer_tail_layout`).
  - For the prompt, only the final chunk's tail matters; every earlier chunk's tail is dead work.
  - The chunked-path tail is `min(128, final_chunk_len)` rows, which can be fewer than 128.
- **The window pool is paged and fixed-cap.**
  - The recipe uses the paged `SWATokenToKVPoolAllocator`, one window slot per extend token, with
    `full_to_swa_index_mapping` baked into each metadata at build.
  - The cap is 10,752 slots at 4096-token chunks (logged: `DSV4 SWA sizing: mode=cap, swa_tokens=10752`).
  - `RequestWindow` is `None` in production: it exists only under encoder replay.
  - `alloc_extend_swa_tail` (`allocator/swa.py:345-409`, bs=1, NPU-only caller today) already allocates full KV for
    every token and window KV for the tail only.
- **Admission:** `SWAPrefillBudget.check_prefill` (`prefill_budget.py:208-245`) refuses an unchunked 250k extend
  (`swa_never_fits` leads to `NO_TOKEN`). `alloc_extend` would raise "Prefill out of memory". Both need a
  layer-major branch.
- **Scheduler watchdog:** 300 s on `forward_ct`, which increments only at `run_batch` entry (`scheduler.py:4346`;
  `invariant_checker.py:493-515`). A blocking pass does not bump it.
- **Overlap scheduling is on** for this recipe. Anything the pass issues on side streams must be joined into the
  current stream before `forward_batch_generation` returns (`scheduler.py:4383-4455`).
- **Hot cache:**
  - Per-layer slot tensors hold capacity + scratch (+ an optional pull row), at fixed addresses captured by decode
    graphs and the copy-engine table. Capacity is uneven per layer; raw ~1,267 rows at 16100 MB before clamps.
  - Nothing ties slot bytes to the mapping, and generations increment only on insert. A hit on a borrowed slot
    silently reads wrong weights.
  - The pinned tier is inclusive: the C++ `hot[]` bitmap protects hot rows, refreshed per armed demand and per eager
    host use.
  - Pinned rows are immutable file images.
- **Hot-slot writers under EXL3 DIRECT:**
  - graph-gather miss copies, in-graph and by the copy engine;
  - native prefetch, planned at layer L and committed at L+1 of the same forward;
  - **eager forwards small enough for `_apply_graph`** (`exl3.py:432-439`), which read hot slots and insert misses;
  - seeding;
  - warmup and capture.
  - `before_host_use` parks the RAM-miss service thread, waits for the copy engine to go idle and retires leases. It
    refuses while a graph lease is outstanding. It does not cover other streams.
- **EXL3 prefill MoE:**
  - It loops per expert in ascending id (`exl3_ops.py:231-235`).
  - `EXL3_ROW_VIEWS.select` takes one buffer set (six tensors) per call; rows spread across several buffers need one
    `select` per buffer. Its 48-entry cache is a view-cache bound, not an error.
  - The prefill staging buffer `_STAGING` (~852 MB) is module-global, lazy and grow-only. The normal path, including
    the tail pass, uses it.
- **Host NUMA:**
  - Node 0 is kept at ~4 GiB spare by the recipe's 60 GiB tier share (`DSV41_REFERENCE.md:4716-4721`).
  - Node 1 has ~81 GB free with the server down (measured 2026-09-27), ~40 GB after its 40 GiB tier share.
  - H2D bandwidth is the same from either node, and load on node 0 cuts it 27-43% (§25.4).

## 4. Architecture

```
Scheduler
  LayerMajorGate.admit(req)  -- one extend for the whole suffix; layer-major budget and allocation (§5.1)
TpModelWorker.forward_batch_generation  -- seam before model_runner.forward (tp_worker.py:659)
  LayerMajorPrefill.run(batch)
    ChunkPlan.build            sub-batches, window-ring mapping, per-chunk metadata kept on host (§6.1, §6.4)
    ExpertBorrow.begin()       verify restore sources, snapshot mapping, pause writers (§7.1)
    model.forward_layer_major(plan, store, expert_source, heartbeat)
        embed chunks -> StateStore
        for L in 0..20:
            LayerExpertSource.make_resident(L) ; prefetch L+1, L+2
            for chunk k in order:
                upload meta_k (L-invariant part cached per chunk) ; state_k H2D ; run layer L ; state_k D2H
                heartbeat.tick()
        tail inputs -> dedicated tensors
    ExpertBorrow.restore() ; ExpertBorrow.end()
    tail pass: layers 21-39 on the final chunk's tail, tail-only logits (§5.5)
    join all side streams into the current stream
  -> sampling as today; whole prompt inserted into the radix cache once (§6.5)
```

### 4.1 Units

| Unit | File | Does |
|---|---|---|
| `LayerMajorGate` | `layer_major_prefill.py` | Launch refusals; per-request admit and fallback; capacity check against the StateStore |
| `LayerMajorBudget` | `mem_cache/prefill_budget.py` (branch) | Admission cost: full and compressed KV for every token, window KV for the ring only |
| `alloc_extend_layer_major` | `mem_cache/allocator/swa.py` | Full KV for every token; window slots for the ring only; mapping lifecycle (§6.4) |
| `ChunkPlan` | `layer_major_prefill.py` | Sub-batch bounds, per-chunk `engram_history`, window-ring mapping, per-chunk metadata build and host storage |
| `StateStore` | `layer_major_state.py` | Pinned host store of per-chunk carried state on NUMA node 1; double-buffered device staging |
| `ExpertBorrow` | `layer_major_experts.py` | Restore-source verification, mapping snapshot, writer pauses, restore, fail-stop |
| `LayerExpertSource` | `layer_major_experts.py` | One layer's 384 experts resident in borrowed slots; per-buffer `select` views; prefetch |
| `PassHeartbeat` | `layer_major_prefill.py` | Advances a progress counter the watchdog reads, per chunk-layer |
| `DeepseekV4Model.forward_layer_major` | `deepseek_v4.py` | Layer-outer, chunk-inner loop; tail inputs; tail pass |
| backend `install_forward_metadata(meta)` | `deepseek_v4_backend.py` | Installs a chunk's metadata; sets `encoder_replay` and clears per-forward flags as `init_forward_metadata` does |

`LayerMajorGate`, `LayerMajorBudget`, the allocator branch, `ChunkPlan`'s mapping and history, `StateStore` sizing
and `ExpertBorrow`'s bookkeeping are all testable on CPU with fakes.

## 5. Execution flow

### 5.1 Admission and allocation

- When `LayerMajorGate` admits a request, `PrefillAdder._select_prefill_admission` (`schedule_policy.py:1347-1426`)
  takes the whole uncached suffix as one extend.
- `check_prefill` takes a layer-major branch (`LayerMajorBudget`). It costs:
  - full and compressed KV for every token, against `rem_total_tokens`;
  - window KV for one ring (`chunk + page_size` = 4352 slots), against the window pool's 10,752.

  A request that does not fit is refused to the chunked path, never left waiting.
- `alloc_for_extend` calls `alloc_extend_layer_major`, modelled on `alloc_extend_swa_tail`:
  - full and compressed slots for every token;
  - a ring of window slots owned by the request;
  - no window mapping yet, since ChunkPlan maps it chunk by chunk (§6.4).
- The layer-major mark reaches the forward as a kw-only argument to `forward_batch_generation`, never by mutating
  the ScheduleBatch or the ForwardBatch after `init_new`.

### 5.2 Worker

- `tp_worker.forward_batch_generation` branches before `model_runner.forward` (`tp_worker.py:659`) into
  `LayerMajorPrefill.run`. `model_runner.py` is not touched.
- Sampling then runs as today (`tp_worker.py:704-719`).
- Before returning, every side stream the pass used (copy streams, the expert loader) is joined into the current
  stream, so the overlap scheduler's relay and D2H read finished results.

### 5.3 Watchdog and liveness

- `PassHeartbeat.tick()` runs after every (layer, chunk) step and advances a progress counter. The watchdog treats
  that counter as progress alongside `forward_ct`.
- This is a small change in `invariant_checker.py`: a heartbeat source the pass registers for its duration.
- The pass is not interruptible. An abort received during it takes effect when it returns, as for any forward today.
- `/health_generate` may report 503 during a long pass, as it can during long chunked prefills today. It is left
  unchanged.

### 5.4 Layers 0-20

For each layer L:
- `LayerExpertSource.make_resident(L)` (§7.2).
- For each chunk k in order:
  - upload chunk k's metadata to the device (§6.1);
  - copy chunk k's carried state host-to-device on a copy stream, overlapped with chunk k-1's compute;
  - `install_forward_metadata(meta_k)`;
  - run layer L;
  - copy the updated carried state device-to-host;
  - `heartbeat.tick()`.

Layer 20 builds candidate masks only on the final chunk, and only for its tail rows. That is new code in the
candidate publish (`deepseek_v4_backend.py:3342-3382`) behind a metadata flag that ChunkPlan sets.

### 5.5 Tail pass (layers 21-39)

- **Rows:** exactly the chunked path's tail, `min(128, final_chunk_len)` rows of the final chunk, with the window
  floored at the tail start. A final chunk shorter than 128 tokens gives a shorter tail, as today.
- **Inputs first:** before layer 21 runs, the final chunk's post-layer-20 hidden rows, `prev_pre`, `normalized`,
  positions, input ids, Engram hash ids, layer 20's c1 top-k rows and the candidate masks are copied into dedicated
  tensors, because the tail uses `_STAGING` and would clobber anything parked there.
- **Expert path:** the tail runs on the normal expert path, after `ExpertBorrow.restore()` (§7.4). It reads hot slots
  in place, so it must never run on borrowed bytes.
- **Logits:** from the tail rows only, by rewriting `LogitsMetadata.extend_seq_lens` to the tail's (precedent:
  dspark, `deepseek_v4.py:5172-5183`). The bounded-replay epilogue's scatter back to [T, 5120] and [T, 20480]
  (`:4874-4881`) is skipped: at 250k it would allocate ~12.8 GB.

## 6. Per-chunk state

### 6.1 Attention metadata

- **Built once, stored on host.** ChunkPlan builds each chunk's metadata once, right after mapping that chunk's
  window slots (§6.4), and moves its device tensors to pinned host memory in the StateStore. At 250k, 61 x ~50 MB is
  ~3 GB, which cannot stay on the device.
- **Uploaded per use.** Before (L, k), the chunk's metadata is uploaded into a reusable device slot. The page table
  and window indices do not change with the layer, so they are uploaded once per chunk per layer.
- **Top-k buffers travel with the chunk.** The index-source layers (2, 8, 14, 20) write them, and they are read back
  until the next index-source layer.
- `install_forward_metadata` also sets `self.encoder_replay` (False) and resets the per-forward flags that
  `init_forward_metadata` resets (`backend:2429-2443`).

### 6.2 Carried state

| Buffer | Bytes per token | Why |
|---|---:|---|
| `hidden` [4, 5120] bf16 | 40,960 | layer input |
| `prev_pre` [4] fp32 | 16 | layer input |
| `normalized` [5120] bf16 | 10,240 | the fused post-combine-norm output the next layer consumes. Carrying it reproduces the chunked path's call graph, and with it `next_combined` is non-None, so the current layer keeps its fused post kernel |
| top-k bundle | ~8,192 | from each index-source layer to its last consumer |
| **Total** | **~59.4 KB** | |

### 6.3 The StateStore

- **Size:** ~59.4 KB per token + ~50 MB of metadata per chunk, ~15.6 GB + ~3 GB at 262,144 tokens.
- **Placement:** pinned on **NUMA node 1** (explicit `mbind`, then CUDA host registration). Node 1 has ~40 GB free
  after its tier share; node 0 has ~4 GiB (§3).
- **Allocation:** once at startup, sized from `CONTEXT_LENGTH`.
  - It checks node 1's free memory first and sizes down with a logged reason. `LayerMajorGate` then caps the
    accepted suffix to what fits.
  - It refuses to start if node 1 is under a margin. The arm_env note about a reth/nimbus stack keeping node 1 at
    ~9 GB free is from 2026-09-22; phase 1 re-measures node 1 with production-like load.
- **Device staging:** two ~240 MB carried-state buffers, a metadata slot of ~50 MB and the tail-input tensors. These
  are dedicated allocations, not `_STAGING`, which the tail pass uses.
  - Their ~0.55 GB comes out of the activation headroom. §8 budgets it.

### 6.4 Window KV ring

- **Why a ring:** layer L on chunk k needs L's window KV only for the 127 positions before the chunk, which layer L
  wrote while processing chunk k-1. So a request-owned ring of `chunk + page_size` = 4352 window slots suffices,
  17 pages.
- **Mapping:** for each chunk in order, ChunkPlan sets `full_to_swa_index_mapping` for that chunk's positions to ring
  slots. It maps page-aligned so the predecessor page and the chunk's 16 pages are disjoint within the 17. It then
  builds the chunk's metadata, which bakes the mapping in.
  - Mappings of earlier chunks are overwritten as the ring wraps. That is safe because each chunk's metadata already
    holds its indices.
- **Reuse across layers:** every layer pass reuses the same slot indices, because each layer has its own window
  buffer indexed by slot.
- **After layer 20:**
  - the mapping for positions outside the final window is cleared, so `free_swa` cannot release reused ring slots
    more than once;
  - the ring slots holding the final window stay as the request's window KV;
  - the rest of the ring is freed.
- **Tombstones:** the request's `swa_evicted_seqlen` is set, page-aligned, to the window start so the radix insert
  tombstones everything below.
- **Late layers** write their window for the tail rows only, as today.

### 6.5 After the pass

- The radix tree receives full KV for every prompt token and window KV for the final window only. That is the same
  shape today's tree has after its window slots are evicted.
- A later mid-prompt hit re-prefills the window under the bounded-replay rule.
- HiCache write-through of a 250k insert is measured in phase 3. It happens after the first token is sampled, so it
  does not add to TTFT, but it can delay the first decode step.

## 7. Expert residency

### 7.1 Begin: verify, snapshot, pause

`ExpertBorrow.begin()` runs, in order:

1. **Verify restore sources.** For every hot slot, confirm its expert's pinned row is resident and protected. If any
   is missing, refuse the pass before touching anything, and fall back to the chunked path.
2. **Snapshot** every layer's slot-to-expert mapping; this is decode's hot set. Scratch rows and the pull row are not
   part of it and are not restored.
3. **Pause the hot-slot writers:**
   - the RAM-miss service thread and copy engine, via `before_host_use` (the pause is held for the pass);
   - native prefetch, not armed on the layer-major path;
   - DIRECT miss inserts: the layer-major MoE never routes through `_apply_graph`, whatever the row count. The
     gather-rows threshold check is bypassed for marked forwards.
   - seeding, warmup and capture: none run while serving.
   - The GPU residency updater only writes scores and victims during prefill. `record_routes` is skipped on the
     layer-major path, so prefill routes do not feed decode's residency statistics, as for today's EXL3 prefill
     boundaries (`expert_residency_gpu.py:568-572`).
4. **Check the pause's own limits.** The C++ `fatal_wait_s` and pause-duration semantics are confirmed in phase 2
   before relying on a multi-minute hold. If the pause cannot be held that long, the pass re-acquires it per layer.

### 7.2 The borrowed area and loading a layer

- **The area:** the hot cache's capacity rows, excluding scratch and pull rows, form a pool of 13.3 MB rows. That is
  three layer sets of 384: one computing, one loading, one prefetching.
- **Views:** a layer's 384 rows spread across several donor buffers. `LayerExpertSource` groups them by buffer and
  issues one `select` per buffer.
- **Compute order:** it accumulates in ascending expert id across all buffers, preserving the MoE's accumulation
  order.
- **Pinned rows:** rows already in the pinned tier are copied host-to-device into the area.
- **NVMe rows:** the rest are read from NVMe into a dedicated landing buffer on node 1, then copied to the area.
  - The buffer is six per-name arrays, 512-byte aligned, CUDA-registered, ~64 rows (~850 MB).
  - It needs a new reader entry point in `exl3_ram_miss_host.cpp`. It overrides the per-name slab bases for one call
    and reuses row-image mode's O_DIRECT readv.
  - It requires row images (the recipe sets `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1`) and runs while the service
    is paused.
  - Each layer's load is its own busy window, well under the reader watchdog's `max(30 s, 3 x timeout)`.
- **The pinned tier is not written by layers 0-20**, so decode's RAM set for those layers survives. The tail pass
  (layers 21-39) runs on the normal path and still admits and evicts pinned rows for those layers, as today.
- **Estimate:** ~0.5 s per layer (~2.4 GB from NVMe at 7 GB/s, overlapped with 5.1 GB over the link at 12.3 GB/s),
  hidden behind the previous layer's compute.

### 7.3 Compute on borrowed rows

- The layer-major MoE path reads only from `LayerExpertSource` views. It never uses the normal hot-hit or `_STAGING`
  path, never arms native prefetch and never takes `_apply_graph`.
- A debug assertion (`SGLANG_DSV41_LAYER_MAJOR_VERIFY`) checks at every layer that no hot-hit or graph path was
  entered.

### 7.4 Restore and end

- **Restore:** after layer 20 and before the tail, `restore()` copies every snapshotted slot's expert back from its
  pinned row, looking up the pinned slot at restore time. That is ~16 GB in ~1.3 s at the link's ceiling, if nothing
  else is on the link.
- **Check:** in phase 2 validation, and whenever `SGLANG_DSV41_LAYER_MAJOR_VERIFY` is on, every restored slot is
  checksummed against its pinned row. Otherwise a sample is checked.
- **End:** `end()` releases the pauses after the tail pass.

### 7.5 Failures

- An exception anywhere runs `restore()` then `end()` in a `finally`, clears the ring mapping, releases the request's
  window ring and full KV slots through the normal abort path, and fails the request.
- A failed `restore()` fail-stops the server, because decode must never run on wrong weights. Verifying sources in
  `begin()` makes this unlikely.

## 8. Memory budget

**Device.** Today a 4096-token chunk peaks with ~470 MiB free (§27.17).

| Item | Size | Where from |
|---|---:|---|
| One layer on one 4096 chunk: activations | at or below today's full-chunk peak | as today |
| Indexer logits, dense fp32 [T, lc] | up to ~4 GB per chunk at 250k (layer 20, ratio 1) | **must be chunked** (phase 0 measures the path taken) |
| Carried-state staging, 2 x ~240 MB | ~0.48 GB | new, from headroom |
| Metadata slot | ~50 MB | new, from headroom |
| Tail inputs | ~6 MB | new |
| Borrowed hot area | 0 extra | the hot cache itself |

- The indexer logits are the gating item. Unchunked, they break at 250k on today's path too.
  - Phase 0 measures which indexer path sm_120 takes (`SGLANG_OPT_USE_TOPK_V2`, deep_gemm availability) and its
    peak at 128k.
  - The design requires the row-chunked form: the torch fallback already chunks within 1 GiB, and the dense fp4
    path must adopt the same bound.
- The ~0.53 GB of new staging exceeds today's ~470 MiB headroom. The recipe gives it room by lowering
  `--mem-fraction-static` by 0.02 (0.90 to 0.88, ~650 MiB). The KV pool shrinks from ~374k to ~330k tokens, still
  above 262k.

**Host (node 1):** StateStore ~18.6 GB at 262k (§6.3) + landing buffer ~0.85 GB.

## 9. Engram

ChunkPlan computes each chunk's `engram_history` window from the prompt tokens before the chunk. Hash ids are
computed per chunk, in chunk order, and kept with the chunk; the commit order is unchanged. Nothing else in Engram
depends on layer 1 and layer 14 running in the same forward.

## 10. Testing

### 10.1 Unit, CPU

Under `test/registered/unit/...`, as `unittest.TestCase` per the divix01 collection note.

- **LayerMajorGate:** each launch refusal; each per-request fallback, including hidden-state return and store
  capacity.
- **LayerMajorBudget:** a 250k extend is admitted when full KV and the ring fit, and refused to the chunked path
  otherwise. The chunked path's budget is unchanged.
- **Allocator branch:** full slots for every token, ring slots only for window, and no leaked or double-freed window
  slot after release (property test over prompt lengths, including non-multiples of 4096 and a final chunk of
  1-127 tokens).
- **ChunkPlan:**
  - window mapping: every chunk sees its 127 predecessors in every layer, and the final window lands in the kept
    slots;
  - tombstone alignment;
  - per-chunk `engram_history`;
  - tail = `min(128, final_chunk_len)`.
- **StateStore:** sizing, node placement, the round trip of carried state, metadata and top-k, and capacity refusal.
- **ExpertBorrow with fakes:**
  - `begin()` refuses when a restore source is missing;
  - the mapping is unchanged after the pass;
  - bytes are restored;
  - `restore()` and `end()` still run after an exception injected at any step.
- **PassHeartbeat:** the watchdog sees progress through a simulated multi-minute pass.

### 10.2 GPU equivalence

Under `test/manual/dsv41/`, plus a debug-dump probe modelled on `swa_window_probe.py`.

- **What is compared:** layer 20's output per chunk and the final logits, layer-major against chunked.
- **Lengths:** 8k, 16k, 32k and 33,000 (final chunk 232 tokens), and one length with a final chunk under 128
  tokens (32,868).
- **Tolerance, per compared tensor:**
  - cosine similarity at least 0.9999;
  - max abs difference within 2x the chunked path's own run-to-run difference, measured on the same prompt;
  - greedy 64-token output identical at 8k and 16k;
  - at 32k and above, the first divergence no earlier than the chunked path's own run-to-run divergence.
- **A second, bitwise reference:** the fused mHC post kernel is disabled on both sides
  (`SGLANG_OPT_USE_TILELANG_MHC_POST=0`). With it disabled, bitwise identity of layer 20's output is expected; any
  mismatch there is a bug, not rounding.

### 10.3 GPU restore and safety

- Checksums of every hot slot before the pass and after `restore()`.
- The verify assertion (§7.3) is on for every equivalence run.
- A decode step forced between chunks under a debug flag must be refused.

### 10.4 Performance (phase 3)

- TTFT at 32k, 128k and 250k.
- The first decode step after the prefill; steady decode ms/token.
- Peak VRAM and OOM retries; node 1 memory.
- HiCache insert time.

## 11. Phases

Each phase ships behind the flag and ends with its tests green.

0. **Measure, before any code.** One untraced ~128k prompt on today's recipe:
   - per-chunk times from `server.log`;
   - peak VRAM;
   - which indexer path runs;
   - node 1 free memory under load.

   If the indexer logits do not fit at 128k (§8), row-chunking the dense indexer becomes phase 0b, before
   everything else. The chunked path needs it anyway for long contexts.
1. **Correctness at 8-32k, with no hot-cache borrow:**
   - gate, budget and allocator branch, ChunkPlan, StateStore;
   - `install_forward_metadata`, `forward_layer_major`, the tail pass with tail-only logits, the heartbeat;
   - unit tests, and the equivalence tests of §10.2.

   Experts come through the existing prefill path. The pass is slower than the chunked path here and is not run
   above 32k, since the link traffic is unchanged.
2. **The borrow:**
   - ExpertBorrow, LayerExpertSource, the landing buffer and reader entry point;
   - restore, fail-stop, verify;
   - the restore and safety tests of §10.3.

   This phase is what makes 250k runnable and fast.
3. **Measure and decide:** the performance arms of §10.4 (the first run above 32k), a `DSV41_REFERENCE.md` section,
   and a recipe decision on the threshold and the `--mem-fraction-static` change.

## 12. Estimate for a 250k prompt (after phase 2; not measured)

| Part | Estimate |
|---|---|
| Layers 0-20 compute: 21 layers x 61 chunks x ~50-70 ms (§27.17's traced kernels, ~67 ms per layer-chunk) | ~65-90 s |
| Expert loading | ~0 visible (hidden behind compute) |
| State and metadata copies (~0.6 GB per chunk-layer pair, both ways, overlapped) | ~0-20 s visible, depending on overlap |
| Restore | ~1.3 s |
| Tail pass | ~2-3 s |
| **Total** | **~1.5-2 min** |

That leaves about 2.5x headroom to the 5-minute target, and 3x to the watchdog even without the heartbeat.

**Risks to the estimate:**
- attention and indexer cost beyond 30k context (phase 0);
- host launch overhead of ~170k eager kernels per chunk, spread across 21 layers.

## 13. Risks

- **Attention and indexer cost and memory at 100k+ context** (phase 0; the indexer logits are a hard blocker if not
  chunked).
- **Node 1 memory under production load** (phase 0 and phase 1).
- **The C++ pause** held for minutes, and its `fatal_wait_s` (phase 2, §7.1 step 4).
- **Unlisted per-forward state** in the runner, which the layer-major path bypasses: the expert-distribution recorder
  per pass, the forward context, the WAR snapshot. Phase 1 lists what `model_runner.forward` does around the model
  call and reproduces what matters. The equivalence probe is the backstop.
- **Concurrent lanes.** `prefill-fills` and `prefill-evict` touch the same prefill expert path. Implementation
  rebases on master at each phase.
- **A skipped native-prefetch commit** leaving bytes in a mapped slot (unconfirmed). Not relevant during the pass,
  since prefetch is not armed. The restore overwrites all snapshotted slots regardless.

## 14. Review findings addressed (revision 1 to revision 2)

| Finding | Where |
|---|---|
| Node 0 has ~4 GiB spare | §6.3: StateStore and landing buffer on node 1; §3 |
| SWA budget and allocator refuse an unchunked extend | §5.1: `LayerMajorBudget` and `alloc_extend_layer_major` |
| No RequestWindow; ring on the paged allocator | §6.4: ring on `full_to_swa_index_mapping`, mapping lifecycle, tombstones |
| Watchdog at 300 s | §5.3: `PassHeartbeat`; §12 margin |
| Fused mHC carry-over | §6.2: `normalized` carried; §10.2: tolerance plus a fusion-off bitwise reference |
| Phase 1 not useful alone at 250k | §11: phase 1 limited to 8-32k; borrow before any long run |
| Kept metadata and indexer logits exceed headroom | §6.1: metadata on host; §8: indexer chunking required, fraction -0.02 |
| Tail scatter of ~12.8 GB | §5.5: tail-only logits (dspark precedent) |
| Tail pass uses `_STAGING` | §5.5: tail inputs in dedicated tensors; §6.3: staging not `_STAGING` |
| Tail definition when the final chunk is under 128 tokens | §5.5 and §10.2: `min(128, final_chunk_len)`, tested |
| Unlisted hot-slot writers; no byte-to-mapping tie | §7.1: writer list and pauses, `_apply_graph` bypass; §7.3: verify assertion |
| `select` is per buffer | §7.2: grouped by buffer, ascending-id order |
| Restore source verified only at the end | §7.1 step 1: verified in `begin()`, refusal before touching anything |
| NVMe landing needs aligned, registered per-name arrays | §7.2 |
| Hidden-state refusal only in-forward | §2: refused in `LayerMajorGate` |
| Overlap scheduling stream joins | §5.2 |
| `install_forward_metadata` must mirror `init_forward_metadata` | §6.1 |
| `record_routes` and residency stats | §7.1 step 3 |
| HiCache insert cost; abort mid-pass | §6.5, §7.5, §5.3 |
