# DSV41 decode: Engram lookups without host nodes

Goal: remove the two CUDA host nodes from the DSV41 decode graph, the precondition for copy-engine C1 (1b) and
copy-engine prefetch (`2026-09-25-dsv41-copy-compute-overlap.md`, sections 3 and 5).

Branch `cc/engram-no-hostnode` (from `cc/copy-overlap` at `f54ce6f553`, with `cc/dsv41-pinned-numa` merged).
Flag `SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT` (`environ.py`), default off; it requires
`SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1`, which the recipe sets. Evidence lives under
`divix01:/data/models/slang/nvfp4-work/engram-no-hostnode/`.

## 1. What the host nodes do today

Captured by `_capture_engram_file_lookup` (`python/sglang/srt/layers/engram.py`) for the bs-1 decode graph, at
layers 1 and 14, once per layer:

| Node | What |
|---|---|
| memcpy D2H | the layer's 24 hash ids (int64 `[1, 24]`, a column of the `[T, 2, 24]` `hash_ids` that `engram_hash_ids_and_commit` produced at the start of the step) into pinned `ids` |
| **host** | `EngramHostLookup::callback` (`engram_host_node.cpp`): `Store::enqueue` hands the ids to the Store's io_uring worker and **blocks the stream** until it has looked every id up in the 5 GiB native row cache and read the misses with O_DIRECT into pinned `rows` (`[24, 264]` uint8: 256 fp8 weight bytes and 8 e8m0 scale bytes per row); status into pinned `status` |
| memcpy H2D x2 | `rows` into `packed_gpu`, `status` into `status_gpu` |
| kernels | `torch._assert_async(status_gpu == 0)`, then the dequant (fp8 x e8m0 to bf16 `[1, 24, 256]`) |

Cost (node trace, section 2 of the copy-overlap plan): p50 1.17 ms, p90 2.21 ms, max 33.8 ms, 2.5 ms of stream stall
per step. And the launch: a graph with host nodes makes `cudaGraphLaunch` block (~104 ms/step in the graph-mode
trace) while holding the context lock, which deadlocks a second thread's `cudaMemcpyAsync`.

## 2. The replacement

The same Store, cache and io_uring worker; only who calls them and how completion reaches the stream change.

```
step start   hash kernel -> post(layer 1) -> post(layer 14)          [device, in the graph]
                               |                 |
service      poll post words, Store::enqueue(ids -> pinned rows), status, release done   [one C++ thread]
                               |                 |
layer 1      wait(layer 1): spin on done, copy rows to device, assert, dequant          [device, in the graph]
layer 14     wait(layer 14): same
```

- **Control block** per captured lookup: 48 int32 words of pinned host memory (device-visible through UVA).
  `kPostSeq` (word 0), `kDoneSeq` (16) and `kStatus` (17), `kFatalSeq` (32) and `kFatalStatus` (33), each group on
  its own 64-byte line. Offsets are defined three times (`engram_ring.cuh`, `engram_host_node.cpp` `ring::`,
  `sglang/kernels/ops/embeddings/engram_ring.py`); a CPU test checks the C++ and Python copies agree and the GPU
  tests exercise the kernel's.
- **Post kernel** (`engram_ring_post`, one thread): `seq = ++counter` (a device word; 0 is skipped), copy the 24
  ids into pinned `ids`, `__threadfence_system()`, `st.release.sys` `kPostSeq = seq`.
- **Service** (`RingService`, `engram_host_node.cpp`): one thread for all lookups, polling in registration order
  (layer 1 before layer 14). For each lookup whose `kPostSeq` (acquire load) differs from the last seen: if
  `kFatalSeq` is clear, `Store::enqueue` the ids into pinned `rows` (the same call the host node made, so the same
  cache, the same io_uring worker, the same chunking and failure statuses); else refuse with status 4. Then store
  `kStatus` and release-store `kDoneSeq = seq`. Idle for 200 us, it sleeps 20 us per pass (about 70 us of latency
  with timer slack, hidden behind layer 0). **It never calls the CUDA API**; that is what removes the deadlock
  mechanism, not only the host nodes.
- **Wait kernel** (`engram_ring_wait`, 256 threads): thread 0 returns status 6 at once if `kFatalSeq` is latched;
  else acquire-spins on `kDoneSeq == seq` with `__nanosleep(256)` until `timeout_ns`, then reads `kStatus`. Any
  non-zero status is written to `status_gpu` and latched into `kFatalStatus`/`kFatalSeq`. On success the block copies
  the 6,336 row bytes from pinned `rows` to `packed_gpu` (16-byte volatile loads). Then the unchanged
  `_assert_async` and dequant, so the flag-on output is computed by the same ops as flag off.
- **Early post.** The hash ids exist before layer 0 runs. `DeepseekV4Model._forward_layers_hc_pre_from_prev` calls
  `post_engram_device_lookups` right after the hasher, so both posts run at the start of the step: layer 1's
  lookup overlaps layer 0 (~2.9 ms of device time, above the host node's p90) and layer 14's overlaps layers 0-13.
  The wait sits where the host node sat. A forward that reaches the lookup without an early post (the unit tests)
  posts there.

Everything stays in the graph: no eager break is added, and the graph contains only kernel nodes for the lookup.

## 3. Memory-ordering rules (LEASE_PROTOCOL discipline)

| Edge | Writer | Reader |
|---|---|---|
| ids -> post seq | device: ids stores, `__threadfence_system()`, `st.release.sys` seq | host: `__atomic_load_n(ACQUIRE)` seq, then ids |
| rows, status -> done seq | host: rows (Store memcpy), status, `__atomic_store_n(RELEASE)` seq | device thread 0: `ld.acquire.sys` seq, then status |
| thread 0 -> the block's row loads | `ld.acquire.sys` by thread 0, `__syncthreads()`, `__threadfence_system()` | all threads: `ld.volatile` of rows |
| fatal latch | device: fatal status, `st.release.sys` fatal seq | host: acquire load before serving |

Reuse rules. The pinned `ids`/`rows` are single-buffered. That is safe because every post and wait is a kernel in
the same stream: step i+1's post cannot run until step i's wait (and copy of the rows) completed, and the service
writes the rows only after seeing that post. A graph replay that is launched early (a non-blocking launch, the
overlap scheduler one step ahead) queues behind the previous replay on the stream and changes nothing. Replays on
a second stream could break this, so the graph keeps the existing first-replay-stream binding
(`BreakableCUDAGraph.replay`), because the lookup is retained through `retain_for_current_graph` like the host
node's context.

## 4. Failure behavior

- A device wait that times out (`ENGRAM_DEVICE_WAIT_TIMEOUT_MS`, 10 s, arbitrary: far above the 33.8 ms worst
  host node) or sees a non-zero status (2 bad id, 3 I/O failure, 4 refused) sets `status_gpu`, and
  `torch._assert_async` turns it into a device-side assert: the step fails, the process stops. That is the flag-off
  behavior for a failed lookup too.
- The fatal word is sticky: later waits fail at once (status 6) and the service refuses later requests (status 4)
  without touching the cache, so a lapped or late publish cannot be read as a fresh one.
- Capture refuses a host node: with the flag on, `BreakableCUDAGraphCapture(forbid_host_nodes=True)` (passed by
  the breakable-graph backend) counts host nodes, child graphs included, in each segment's capturing graph
  (`cuStreamGetCaptureInfo`, before `capture_end`) and raises.
- The flag without `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1` raises at `EngramFileTable.open`.
