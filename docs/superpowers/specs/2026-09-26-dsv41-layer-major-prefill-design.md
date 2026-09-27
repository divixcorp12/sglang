# DSV4.1 layer-major prefill for long prompts: design

Status: design, awaiting review (2026-09-26). Branch `cc/layer-major-prefill`.
Background: `DSV41_REFERENCE.md` §27.2, §27.16 (decoder bounded replay), §27.17 (chunk size and the traced 4096 chunk).

## 1. Goal

Serve prompts of at least 250k tokens with the lowest possible time to first token (TTFT) on divix01's RTX 5090.
Decode must not pay for it with a cold GPU hot cache.

Today, with 4096-token chunks, every chunk re-streams its layers' experts. The traced chunk (§27.17) spends 11.8 s
waiting on NVMe (82 GB at 7 GB/s, the two Gen3 x4 mirrors' line rate) and 9.3 s gathering over PCIe (115 GB at
12.3 GB/s, the link), one after the other, against ~1.5 s of compute. A 250k prompt is ~61 chunks, ~17-20 min.

Layer-major prefill runs every chunk of a long prompt through layer L before layer L+1. Each layer's experts are
then needed for many chunks in a row and cross NVMe and PCIe once per prompt, not once per chunk.

### Success criteria

1. A 250k-token prompt reaches its first token in under 5 minutes.
2. Greedy output matches today's chunked path:
   - Byte-identical 64-token outputs at 8k, 16k and 32k.
   - At longer lengths, divergence no earlier than the chunked path's own run-to-run noise (§27.7 recorded divergence
     at characters 75-109 at 30k).
3. The first decode step after a long prefill costs a normal step plus the hot-cache restore (~1.3 s once), not the
   ~200+ ms-per-step cold steps seen after prefill today.
4. No allocator OOM retries.

### Assumptions (confirmed with the user)

- Only long prompts take the new path. Shorter prompts keep today's chunked path unchanged.
- One request at a time (`max_running_requests=1`): no decode step runs while a long prefill is in flight.
- The GPU hot cache may be borrowed for the whole prefill, provided decode's hot set is recorded first and reloaded
  in bulk before the first decode step.

## 2. Scope and trigger

- `SGLANG_DSV41_LAYER_MAJOR_PREFILL_MIN_TOKENS` (EnvInt, default 0 = off).
  - A request whose uncached prompt suffix is at least this long takes the layer-major path.
  - The intended production value is 32768, set in the recipe after phase 3.
- Refused at launch when set with any of:
  - `max_running_requests > 1`;
  - a speculative algorithm;
  - `enable_decoder_swa_bounded_replay` off (the tail pass depends on it).
- Refused per request with prompt logprobs or hidden-state return. The bounded-replay code already rejects both.
- **Prefix hits:** a long prompt with a cached prefix runs layer-major over its uncached suffix only. The match
  boundary rule of `c515312414` (cap at `input_len - 128`) is unchanged.
- **Length limit:** the longest suffix the path accepts is bounded by the host state store's size (§6.2). A longer
  prompt falls back to the chunked path and logs why.

## 3. Facts the design rests on

From the code (file:line in the investigation notes; config from `dsv41-full40/config.json`):

- **Layers:** 40, hidden 5120, `hc_mult` 4, 384 routed experts per layer, one expert row is 13,315,584 B.
  - `kv_source_layer_ids` [2, 8, 14, 20]; `index_source_layer_ids` [2, 8, 14, 20, 24, 28, 32, 36];
    `candidate_source_layer_id` 20.
  - `engram_layer_ids` [1, 14]; `sliding_window` 128; `index_topk` 512.
- **Loop:** V4.1 runs `_forward_layers_hc_pre_from_prev` (`deepseek_v4.py:4427-4607`).
  - State between layers: `hidden_states` [T, 4, 5120] bf16 and `prev_pre` [T, 4] fp32, i.e. 40,976 B per token.
  - Fused carry-overs (`precomputed_attn`, `combined_attn`, `normalized_attn`) are optional and may be None.
- **Late layers:** with decoder bounded replay, layers 21-39 process only each extend's last `min(128, len)` tokens.
  - They read layer 20's compressed KV and index-K for all positions, layer 20's top-k (layers 21-23) and layer 20's
    candidate masks, for the tail rows only.
  - A whole prompt therefore needs layers 21-39 once, over its last 128 tokens.
- **Attention dependencies:** layer L on chunk k needs:
  - L's window KV for the 127 positions before the chunk;
  - the compressed KV and index-K of L's source layer, up to the chunk end;
  - the top-k and candidate buffers produced earlier in the same chunk's forward.

  All are satisfiable layer-major if chunks run in order within each layer and each chunk keeps its own metadata and
  carried buffers. The ratio-2 compressor's pairing state carries from one chunk to the next per layer, which
  in-order execution preserves.
- **Engram:** hash ids are computed once per forward and the hasher commits n-gram history per extend
  (`engram.py:536-545`). Layer-major computes them per chunk, in chunk order, and keeps them.
- **Hot cache:** per-layer slot tensors at fixed addresses that decode's CUDA graphs capture. Contents may be
  overwritten while no decode runs.
  - Every hot expert's pinned row is protected (inclusive tier), so hot bytes can be restored from pinned memory.
- **Pinned tier:** fixed per-layer split (~201 rows per layer) with fixed, registered slabs. It cannot be rebalanced
  at runtime.
- **EXL3 prefill MoE:** per-expert `exl3_linear` over `Exl3Tensors` views. `EXL3_ROW_VIEWS.select` accepts rows from
  any buffer (at most 48 buffers).
- **`model_runner.py` is frozen.** `encoder_swa_replay.py` is the precedent for building extra ForwardBatches inside
  `tp_worker`.

## 4. Architecture

```
Scheduler (PrefillAdder) --one unchunked extend, marked layer-major--> TpModelWorker
  -> LayerMajorPrefill (new collaborator, model_executor/layer_major_prefill.py)
       builds ChunkPlan: N sub-ForwardBatches of <= chunk tokens, each with its own attention metadata
       ExpertBorrow.begin()      snapshot hot mapping, pause residency writers
       model.forward_layer_major(plan, state_store, expert_source)
            embed chunks -> state_store
            for L in 0..20:
                expert_source.make_resident(L) (L+1, L+2 prefetching)
                for k in chunks (in order):
                    state H2D (double-buffered) ; install meta_k ; run layer L ; state D2H
            ExpertBorrow.restore()   hot bytes back from pinned, mapping untouched
            tail pass: layers 21-39 on the last 128 tokens, hc_combine + norm -> logits
       ExpertBorrow.end()        resume residency writers
  -> sampling as today; whole prompt inserted into the radix cache once
```

### 4.1 Units

| Unit | File | Does | Depends on |
|---|---|---|---|
| `LayerMajorGate` | `layer_major_prefill.py` | Decides per request whether the path applies (threshold, refusals, store capacity) | server args, env |
| `ChunkPlan` | `layer_major_prefill.py` | Splits one extend into ordered sub-batches; builds and keeps one attention metadata per chunk; assigns window-ring slots | ForwardBatch, dsv4 backend |
| `StateStore` | `layer_major_state.py` | Pinned host storage of per-chunk carried state (hidden, `prev_pre`, top-k bundles); double-buffered device staging | pinned allocation at startup |
| `ExpertBorrow` | `layer_major_experts.py` | Snapshot of the hot mapping; pause and resume of residency writers; restore of hot bytes; fail-stop on a failed restore | hot cache manager, RAM-miss service |
| `LayerExpertSource` | `layer_major_experts.py` | Makes one layer's 384 experts resident in the borrowed area; prefetches the next layers; serves `EXL3_ROW_VIEWS.select` views | pinned tier, NVMe reader, `ExpertBorrow` |
| `DeepseekV4Model.forward_layer_major` | `deepseek_v4.py` | The layer-outer, chunk-inner loop and the tail pass | all of the above |
| dsv4 backend `install_forward_metadata(meta)` | `deepseek_v4_backend.py` | Installs a kept metadata object and re-activates the request window | existing metadata builders |

Every unit is testable alone. `ChunkPlan`, `StateStore`, `LayerMajorGate` and `ExpertBorrow`'s bookkeeping run on CPU
with fakes.

## 5. Execution flow

1. **Scheduler.**
   - `PrefillAdder` admits a qualifying request's whole uncached suffix as one extend.
     - The chunk-budget checks (`rem_chunk_tokens`, `fit_chunk`) are bypassed for it.
     - The KV-pool capacity check stays.
   - `alloc_for_extend` reserves full and compressed KV slots for every prompt token.
   - The layer-major mark reaches the forward as a kw-only argument, never by mutating the ScheduleBatch
     (`forward-batch-init-new-purity`).
   - The whole prompt is inserted into the radix cache once, after the forward. There are no per-chunk stashes.
2. **Worker.** `tp_worker.forward_batch_generation` hands a marked batch to `LayerMajorPrefill.run` instead of the
   normal runner.
3. **ChunkPlan.**
   - Splits the extend into sub-batches of `CHUNKED_PREFILL_SIZE` tokens.
   - Builds each chunk's attention metadata with the dsv4 backend's existing builders and keeps it.
   - Builds the late-layer tail metadata for the prompt's final 128 tokens only.
4. **Embed.** Each chunk's embeddings, repeated into the 4 hc streams, are written to the `StateStore`. The chunk's
   Engram hash ids are computed in chunk order and kept.
5. **Layers 0-20.** For each layer L:
   - `LayerExpertSource.make_resident(L)`.
   - For each chunk k in order:
     - copy the chunk's state (and its top-k bundle, if L consumes one) host-to-device on a copy stream, overlapped
       with the previous chunk's compute;
     - `install_forward_metadata(meta_k)`;
     - run layer L;
     - copy the updated state (and any top-k bundle L produced) device-to-host.
6. **Restore.** `ExpertBorrow.restore()` (§7.4).
7. **Tail.**
   - Layers 21-39 run once on the prompt's last 128 tokens, on the normal expert path, which reads the now-restored
     hot slots.
   - Then `hc_combine` + norm, and logits for the last token in the layout the logits processor expects
     (`last_index = extend_len - 1`).
8. **End.** `ExpertBorrow.end()` resumes residency writers. Sampling and output proceed as today.

## 6. Per-chunk state

### 6.1 Attention metadata

- One metadata object per chunk, kept for the whole pass.
- `install_forward_metadata` sets `self.forward_metadata` and re-runs `request_window.activate`. Tail metadata is
  installed only for the tail pass.

### 6.2 Carried buffers and the StateStore

| Buffer | Bytes per token | Producer, consumers | Lifetime |
|---|---:|---|---|
| hidden [4, 5120] bf16 | 40,960 | every layer | the whole pass |
| `prev_pre` [4] fp32 | 16 | every layer | the whole pass |
| top-k bundle (sparse page indices, raw indices, lengths) | ~8 KB | layers 2, 8, 14, 20 produce; later layers consume | producer to last consumer |

- **Sizing:** ~49 KB per token, ~12.8 GB pinned at 262,144 tokens, allocated once at startup on NUMA node 0 and
  sized from `CONTEXT_LENGTH`.
  - Phase 1 first checks node 0's free memory with the expert tier's 100 GiB pinned (60/40 across nodes). If it does
    not fit, the store is smaller and `LayerMajorGate` caps the accepted length to it.
- **Candidate masks:** layer 20 builds them only for the final chunk's tail rows. Non-final chunks skip them; at 250k
  context each would be ~1 GB.
- **Device staging:** two state buffers of ~168 MB each and the top-k bundles (~32 MB per chunk) come out of the
  64-row prefill staging buffer (~852 MB), which the pass does not use.

### 6.3 Window KV ring

- Layer L on chunk k needs L's window KV only for the 127 positions before the chunk, which layer L itself wrote
  while processing chunk k-1.
- Positions map to a request-owned ring of `chunk + page_size` window-pool slots (4096 + 256). Every layer pass reuses
  the ring, since each layer has its own window buffer indexed by slot.
- After layer 20, the ring slots that hold the final 128-token window stay as the request's real window KV; the rest
  are freed. Layers 21-39 write their window for the tail only, as today.
- The window pool must hold the ring for the request (~4.4k slots, about today's cap). The plan confirms exact
  sizing against the pool's cap mode.

### 6.4 After the pass

- The radix tree receives full KV for every prompt token and window KV for the tail only. That is the same shape as
  today's tree after its window slots are evicted.
- A later hit that ends mid-prompt re-prefills the 128-token window under the existing bounded-replay rule.

## 7. Expert residency

### 7.1 The record and the pause

- `ExpertBorrow.begin()` snapshots every layer's hot slot-to-expert mapping. That snapshot is decode's hot set.
- The mapping stays frozen for the pass: no generation bumps, no metadata changes. Only slot bytes are borrowed.
- Paused for the pass, and resumed in `end()`: the GPU residency updater and hot-cache inserts, async promotions,
  the decode prefetchers, and the RAM-miss service thread (its existing `before_host_use` pause).

### 7.2 The borrowed area

- The hot cache's ~1,210 slots (~16 GB across the per-layer slot tensors) are treated as one pool of 13.3 MB rows.
- That is three layer sets of 384: the layer being computed, the next layer loading, and one more prefetching.
- Compute reads weights from these rows through `EXL3_ROW_VIEWS.select`. The borrowed rows span at most 40 buffers,
  under its 48-buffer limit.
- During the pass, layers 0-20's MoE never uses the normal hot-hit or staging path.

### 7.3 Loading a layer

- All 384 experts are loaded. At these lengths nearly all are routed, and routes are known only once the layer runs.
- Rows resident in the pinned tier are copied host-to-device into the area.
- Rows not resident are read from NVMe into a dedicated pinned staging ring of ~64 rows (~850 MB), then copied to
  the area.
  - The pinned tier itself is never written, so decode's RAM set survives the pass (today's prefill evicts ~94% of
    layers 0-20's).
  - This needs one new reader entry point in `exl3_ram_miss_host.cpp` that lands row images in a caller-supplied
    pinned buffer instead of claimed tier slots, reusing `run_fill`'s O_DIRECT readv path.
- **Estimate:** ~0.5 s per layer (~2.4 GB from NVMe at 7 GB/s, overlapped with 5.1 GB over the link at 12.3 GB/s),
  hidden behind the previous layer's compute (~61 chunks x ~50 ms).

### 7.4 Restore

- After layer 20, `ExpertBorrow.restore()` copies every hot slot's expert back from its protected pinned row, per
  the snapshot: ~16 GB in ~1.3 s.
- A debug check (`SGLANG_DSV41_LAYER_MAJOR_VERIFY_RESTORE`) compares sampled slots against their pinned copies.

### 7.5 Failure handling

- An exception anywhere in the pass runs `restore()` and `end()` in a `finally`, then fails the request.
- A failed `restore()` fail-stops the server: decode must never run on wrong weights.

## 8. Memory budget

- **GPU during the pass:** one layer on one 4096-token chunk has about the activation peak of today's full chunk
  (~1.4 GB, leaving ~470 MiB free, §27.17). The pass's own device buffers come out of the unused prefill staging
  buffer (§6.2). Target: the peak is no higher than a normal 4096 chunk.
- **Host:** ~12.8 GB (state store) + ~0.85 GB (staging ring), pinned on node 0.

## 9. Testing

- **Unit, CPU** (`test/registered/unit/...`, unittest.TestCase per the divix01 collection note):
  - **ChunkPlan:** chunk bounds; metadata order; window-ring assignment, so every chunk sees its 127 predecessors in
    every layer and the final window lands in the kept slots.
  - **StateStore:** sizing from context length; the round trip of state and top-k bundles; capacity refusal.
  - **LayerMajorGate:** the threshold and each refusal.
  - **ExpertBorrow with fake caches:** the mapping is unchanged after the pass, bytes are restored, and restore and
    resume still run after an exception injected mid-pass.
- **GPU equivalence** (`test/manual/dsv41/`, plus a debug-dump probe like `swa_window_probe.py`):
  - Layer 20's output per chunk and the final logits, layer-major against chunked, on 8k and 16k prompts. Expected:
    bitwise, or cosine 1.0000, since each layer runs the same kernels with the same accumulation order.
  - Greedy 64-token output equality at 8k, 16k and 32k (`prefix_equiv.py`-style client).
- **GPU restore check:** checksums of every hot slot before the pass and after `restore()`.
- **Performance arms:**
  - TTFT at 32k, 128k and 250k;
  - the first decode step after the prefill;
  - steady decode ms/token;
  - peak VRAM and OOM retries (driver: `analysis/dsv41-drive/prefill-chunk/`).

## 10. Phases

Each phase ships behind the flag and is useful on its own.

0. **Measure attention and indexer cost at long context.** One untraced ~128k prompt on today's recipe, reading
   per-chunk times from `server.log`.
   - If chunk time stays flat, the 250k estimate stands.
   - If it grows, the design adds an indexer step before phase 1.
1. **Layer-major pass on the existing expert path:**
   - `LayerMajorGate`, `ChunkPlan`, `StateStore`, `install_forward_metadata`, `forward_layer_major`;
   - the window ring, the top-k bundles, candidate masks for the tail only, the tail pass;
   - scheduler and worker wiring;
   - unit and equivalence tests.

   Running one layer's chunks back to back should keep its pinned rows warm, so most NVMe reads happen once per layer
   even before phase 2.
2. **Borrowed hot area:** `ExpertBorrow`, `LayerExpertSource`, the staging-ring reader entry point, restore and
   fail-stop, and the restore check.
3. **Measurement arms**, a `DSV41_REFERENCE.md` section, and a recipe decision on the threshold.

## 11. Estimate for a 250k prompt (not measured)

| Part | Estimate |
|---|---|
| Layers 0-20 compute: 21 layers x 61 chunks x ~50 ms | ~64 s |
| Expert loading | ~0 s visible (hidden behind compute) |
| State copies | ~0 s visible (overlapped) |
| Restore | ~1.3 s |
| Tail pass | ~2-3 s |
| **Total** | **~1.5-2 min** |

The largest unknown is attention and indexer cost beyond 30k context, which phase 0 measures.

## 12. Risks

- **Attention and indexer cost at 100k+ context**, which has never been measured (phase 0).
- **Host memory on node 0** for a 12.8 GB pinned store alongside the expert tier.
- **Hidden per-forward assumptions** in the dsv4 backend or the model loop beyond those listed in §3. The equivalence
  probe is the guard.
- **The prefill staging buffer's lifetime:** if it is allocated lazily or shared in ways §6.2 does not allow for, the
  pass needs its own ~370 MB and the recipe's headroom (~470 MiB) gets thin.
- **Pausing the RAM-miss service and residency writers:** anything that runs outside those pauses and touches hot
  slots would read borrowed bytes. Restore checksums and the no-decode rule guard against it.
