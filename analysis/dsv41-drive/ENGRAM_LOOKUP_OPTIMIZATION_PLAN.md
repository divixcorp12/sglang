# Engram lookup synchronization: optimization plan

## Goal and baseline

Reduce served decode latency caused by reading Engram hash IDs back to the CPU, while preserving exact row selection and output parity. The measured call is `EngramFileTable.lookup` at `python/sglang/srt/layers/engram_file_table.py:105`: its pageable `.cpu()` read blocks the scheduler for 76.6 s in one 419 s Nsight window. That is host blocking time, **not** a throughput improvement estimate. See `ENGRAM_LOOKUP_SYNC.md` for the trace and its limitations.

Use the HTTP serving harness in `benchmarks/dsv41_baseline/`, with breakable decode graphs at batch size 1. The recorded comparison point is `DSV41_REFERENCE.md` §20's leases-off, mirrors-off arm (2.102 token-weighted tok/s); compare new arms against a fresh matched control as well. Keep leases, mirrors, expert cache, corpus, clocks, and server settings fixed within each pair. The requested target configuration is a **20 GiB pinned Engram RAM cache**, up from the current 5 GiB pageable cache. Treat pinning and capacity as separate changes in the measurements.

## Why the obvious cache is insufficient

Both Engram layers' hash IDs are produced together before the model layer loop (`deepseek_v4.py:4423-4444`), then consumed at layers 1 and 14. Each layer needs 24 rows per token; the two file tables total about 203 GB. A partial GPU row cache can avoid file reads for row hits, but a CPU decision on **any** miss still requires GPU-to-host synchronization. Row hit rate alone therefore cannot predict whether it removes a graph break. The useful quantity is the fraction of layer calls and complete decode steps with **zero** misses, plus how misses cluster.

## Work packages and decision gates

### 1. Establish a trustworthy control and demand trace

- Run two unmodified A/A HTTP arms through `run_arm.sh` and `paired.py` to establish the session-paired noise floor. Use the harness's existing SHA, clean-tree, generation, JIT, clock, tenancy, and result-count gates. Record decode throughput by session, TTFT, and output parity information.
- Collect a short targeted Nsight trace of the same served configuration. Count pageable D2H copies, host blocking time, graph segments/breaks, and per-step timing. Keep tracing separate from the full throughput arms.
- Derive per-token hash IDs for the fixed corpus offline using the existing bit-exact hasher and cache simulator, or instrument a bounded trace outside timed arms. Record per-layer unique rows, repeated IDs, all-hit layer/step rates for candidate GPU capacities, and the actual RAM-cache miss rate. Avoid adding a per-step CPU read to the timed path just to measure this.
- Measure free GPU memory at steady decode before assigning a device-cache budget; report the expert-hot-cache capacity lost for each proposed Engram budget.

**Gate:** choose a device-cache experiment only if measured whole-layer or whole-step hit rates and VRAM tradeoff suggest it can remove meaningful synchronization. A high *row* hit rate is not sufficient.

### 2. Pin the Engram RAM cache and raise its budget to 20 GiB

- Change `EngramRowCache.data` in `python/sglang/srt/layers/engram_row_cache.py` from a NumPy-owned array to a page-aligned, CUDA-registered host slab with a NumPy view. Reuse the chunked registration pattern in `moe/expert_host_tier.py`; keep the backing allocation alive as long as the view is used and unregister only after outstanding GPU readers/copies finish. Fail clearly if registration or allocation fails, rather than silently running a partially pinned cache.
- Keep tag/age bookkeeping on the CPU. Check the **total** host footprint of the 20 GiB data slab, tag/age arrays, file-read buffers, and transient lookup output; confirm it coexists with the roughly 70 GiB pinned expert tier on the serving host. Pinning 20 GiB may displace useful page cache or hit CUDA registration limits even when allocation succeeds.
- Make the actual host-to-device upload source pinned. Today `EngramRowCache.lookup` creates `out[inverse]` and `EngramFileTable.lookup` creates NumPy `weight_rows`/`scale_rows`; these temporary arrays can still be pageable after the cache slab is pinned. Use a bounded reusable pinned batch-staging buffer or a direct gather from pinned cache slots, and cover cache misses and duplicates. Do not overwrite or evict a source slot before its asynchronous GPU read finishes.
- Set `SGLANG_DSV41_ENGRAM_RAM_GIB=20` in the candidate serving arm via the harness override, and only change `arm_env.py`'s default after the candidate passes. Measure a 2x2 set when feasible: 5 GiB/pageable (control), 5 GiB/pinned, 20 GiB/pageable, 20 GiB/pinned. This separates pinning effects from capacity effects. The existing 1.5M-token corpus simulation found nearly identical row hit rates at 5 and 20 GiB; a larger cache needs a measured benefit on the served or broader target workload, not an assumed one.

**Gate:** retain 20 GiB as the target only if startup succeeds with stable host/GPU memory, the larger cache does not harm expert traffic or throughput, and the paired serving result justifies its extra resident memory. Report pin registration time, RAM/cache occupancy, page-cache effects, Engram hit/miss rate, H2D copy timing, and the unchanged D2H ID stall separately.

### 3. Reduce the two Engram graph breaks without adding one

The current decode graph has three segments and two breaks, created by `_engram_file_table_lookup` at layers 1 and 14. A separate early staging break would make **three** breaks and is not the default design.

- **First prototype: two breaks to one.** Move both file-table embedding lookups to one eager break immediately after `EngramHasher` produces both layers' hash IDs. Read both layers' IDs together into pinned host staging, fetch/dequantize both sets of rows, and pass their ready embedding outputs into the subsequent captured graph where layers 1 and 14 consume them. This requires splitting Engram embedding lookup from the residual/gate computation; preserve the latter at its original layer. The graph-capture stub must not read uninitialized IDs. Compare full served throughput because fetching layer-14 rows before layer 0 may sacrifice useful overlap.
- **Second prototype, if the early layer-14 fetch hurts: keep two breaks but remove the late ID readback.** Repurpose the first existing Engram break as the producer for both layers' pinned ID transfers. Complete layer 1's lookup there; retain the layer-14 break only to consume its already staged host IDs and fetch missing file rows. This keeps the break count at two and can overlap layer-14 ID transfer with layers 1–13. Use a copy stream ordered after the hash producer, a separate completion event for layer 14, and a host-ID entry point in `EngramFileTable` that bypasses `.cpu()`. Measure event wait, GPU idle time, and break count; pinning without delayed consumption only moves the synchronization.
- A CPU-side shadow hash is a lower-priority fallback. Decode tokens are relayed through `FutureMap`'s GPU token buffer and the current hasher commits history on GPU inside the decode graph. First prove token and predecessor availability on the host without a new readback; account for graph padding and speculative verify semantics. Only then prototype it, with exact hash parity and CPU cost measured.

**Gate:** keep the least complex variant that reduces paired served latency beyond the A/A noise floor and lowers the targeted host stall. Revert variants that only move the wait to a different API call.

### 4. Test a zero-break CUDA graph with host lookup nodes

An explicit graph can contain CPU host-function nodes as well as GPU kernels and memcpy nodes. Test this after the simpler break changes: a host node could perform the file lookup **inside one graph replay**, even though the GPU still waits for rows at each dependency. The proposed C++ slot/atomic sample graphs only an H2D copy after a CPU worker already filled the slot; Engram also needs the graph-produced hash IDs moved to the host and the CPU file read triggered by those IDs.

- Build a small native proof with fixed batch-1 buffers and this dependency chain: GPU hash → pinned D2H of both layers' IDs → layer-specific CPU host lookup nodes → pinned H2D of row bytes → GPU dequant/gather → each Engram layer. Fork the layer-14 lookup branch early so file I/O may run alongside earlier model layers, then join only at layer 14. Keep layer 1's join near its consumer. Use explicit graph dependencies or captured cross-stream events; do not assume independent nodes will overlap without measuring.
- Keep host callbacks CUDA-free. NVIDIA forbids CUDA API calls from host functions, so the callback cannot call the present PyTorch GPU operations or launch more CUDA work. It may read already-completed pinned ID data and fill pinned row buffers using a CPU-only cache/file reader. CUDA memcpy and dequantization must be separate graph nodes. Use stable pointers, bounded slot sizes, completion ownership, and a fail-stop error channel so failed reads cannot publish stale rows.
- Compare graph node/callback duration, scheduler host block, GPU idle time, file-read overlap, and served throughput with the two-break and one-break variants. Zero graph breaks is a structural goal; the performance gate remains end-to-end latency and correctness.

**Gate:** proceed only if the native proof can run the actual file miss path without CUDA calls in callbacks, preserve exact outputs, and improve the paired served result. If it merely moves a long wait into a host node, retain the simpler implementation.

### 5. Pursue GPU residency only if the demand trace supports it

- Prototype an exact-ID, layer-tagged GPU slot map and compressed row store, with the current file table as the miss source. Account for both 256-byte FP8 weights and 8-byte scales; publish a slot only after both have arrived. Preserve eviction safety while a graph can still read a slot.
- Keep the miss path exact. A GPU hit kernel plus a CPU `any_miss` check on every step still synchronizes every step; measure this explicitly before investing in cache policy. A true fast path needs a graph-compatible way to continue on all-hit calls and a correct, bounded fallback on misses, or a GPU-accessible backing tier.
- Compare end-to-end throughput with the expert cache reduced by the *same* VRAM budget. Reject a nominal Engram gain if the displaced expert rows make the whole server slower.

**Gate:** implement the cache only after the prototype demonstrates fewer actual break/sync events and a positive paired served result. If misses remain present nearly every layer call, stop this path and focus on early staging or host-side hash production.

## Correctness and performance acceptance

- Unit tests: `test/registered/unit/layers/test_engram_file_table.py`, `test_engram_row_cache.py`, `test_engram_lookup_break.py`, and any new focused staging/cache tests. Cover duplicate IDs, layer separation, cold misses, eviction, empty batches, capture stubs, registration failure/cleanup, and buffer reuse after GPU completion.
- Model checks: `test/manual/dsv41/test_engram_parity.py` and a matched small served corpus; compare row IDs and Engram BF16 outputs to the old lookup before accepting output-level drift. Check decode, prefill/extend, and target-verify paths if touched.
- Performance: at least one A/A pair and paired A/B comparisons under the same serving setup. Report median per-session decode delta and wins, token-weighted rate, TTFT, pageable D2H count/block time, pinned/pageable H2D timing, event wait time, graph breaks, host and GPU memory, Engram hit rate, and expert miss rate. A microbenchmark alone is not the success criterion.
- Stop when the paired improvement is within the measured noise floor or when synchronization is reduced but throughput is unchanged; document the result instead of attributing the 76.6 s block directly to recoverable wall time.

## Recommended order

Run package 1, then package 2 to test the requested 20 GiB pinned host cache, then package 3 to attack the measured D2H ID stall. Package 4 is a native zero-break proof if the measured value justifies its complexity. Treat package 5 as conditional research: the existing short cold served run reported only about 24% Engram row-cache hits (`DSV41_REFERENCE.md` §17), and even a much higher row-hit rate would not by itself eliminate a per-layer miss check.
