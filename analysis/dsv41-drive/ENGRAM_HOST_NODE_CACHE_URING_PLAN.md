# Engram host node: cache and io_uring integration

## Starting point and scope

The opt-in layer-1 decode graph already captures pinned hash-ID D2H, a native
CUDA host callback, pinned packed-row H2D, and dequantization. Layer 14 keeps
its eager break. The callback in `engram_host_node.cpp` currently does one
`pread` for each weight and scale row, bypassing the 5 GiB
`EngramRowCache` and the existing `PagedRowSource`/io_uring reader. Replace
that callback's data path while preserving the tested graph ordering and
default-off flag. Do not change the cache budget or pin the whole cache in
this task; the graph's small ID and row staging buffers are already pinned.

## Architectural constraint

`UringFileReader` and its native ring enforce creator-thread ownership.
`shared_uring_file_reader()` is normally created on the Python scheduler
thread, while CUDA chooses a host-callback thread. Therefore the callback
must never call the existing shared reader directly. It also must not call
Python, acquire the GIL, or use any CUDA API. Give this graph route a native
Engram I/O worker that owns its ring for its entire lifetime. The callback
submits a bounded request containing stable pinned pointers, waits for worker
completion, and returns only after all destination bytes are valid. The graph
then uploads the rows. Keep worker, cache, file descriptors, and callback
contexts alive until all graph replays are complete; shut them down in that
order. Avoid global mutable staging pointers shared between graph shapes.

## Implementation sequence

1. **Extract a CPU-only lookup interface.** In `engram_row_cache.py` and
   `engram_file_table.py`, define a packed-row lookup that takes CPU int64 IDs
   and fills a caller-owned `(N, dim + dim/32)` uint8 destination in order.
   Preserve the current layer tag (`layer_id << 40`), 8-way set mapping,
   duplicate-ID behavior, LRU updates, and hit/miss counters. No allocation
   or temporary `out[inverse]` should be required on the callback path.
   Choose one coherent cache owner for both graph layer 1 and eager layer 14:
   preferably move the cache state/policy behind a native object and have
   the eager Python wrapper call it, so two independent 5 GiB caches are not
   created. Preserve the existing eager behavior when the opt-in flag is off.

2. **Create the native I/O worker.** In `engram_host_node.cpp` (or a small
   dedicated native source) add a worker with a bounded queue, a single
   io_uring issuer thread, and a clean stop/drain path. Open each table's
   shard with `O_DIRECT` on the worker, as the current cache miss path does.
   Use `liburing` `IORING_OP_READ` to read *only unique cache misses*.
   Weight and scale rows are below 4 KiB and safetensors offsets may be
   unaligned: collect the distinct 4 KiB pages containing the missing rows,
   coalesce adjacent pages, read them into page-aligned bounce storage, then
   scatter both weight and scale bytes into packed cache entries. Handle
   short reads, EINTR/EAGAIN, file-end pages, and failed CQEs; drain every
   in-flight read before releasing/reusing bounce storage. The existing
   `file_row_reader.py` and `csrc/io/uring_file_reader.cpp` page planner are
   references for semantics, but their thread-owned shared instance is not a
   callback API. Link the extension with `-luring` and fail startup clearly
   if io_uring cannot initialize; do not silently fall back to `pread`.

3. **Wire graph callback and eager lookup to the same store.** At table
   construction, register both layer-1 and layer-14 safetensors offsets,
   row widths, row counts, and cache tags with the native store. Keep the
   table/worker object in `_EngramHostLookupContext`. In `run()`, set nonzero
   status and zero the entire pinned row destination first, enqueue its
   request, wait, and set status 0 only after every requested row is complete.
   On invalid IDs, I/O errors, or shutdown, leave status nonzero and rows
   cleared so the captured GPU assert stops use of stale data. Make the
   eager layer-14 lookup use this same store when the opt-in graph route is
   enabled, while its graph break remains. The ordinary eager route stays as
   before when the flag is disabled. Serialize cache mutation and protect
   captured buffer reuse until the previous replay finishes. No callback
   operation may require a Python thread to service its request.

4. **Make io_uring use observable.** Add native counters for cache accesses,
   hits, unique misses, submitted io_uring SQEs, completed CQEs, direct-read
   bytes, failures, and worker queue/wait time. Expose a low-frequency
   snapshot to existing Engram trace logging, avoiding per-token Python work
   on the graph path. A warm-hit replay must show zero new SQEs; a cold
   replay must show SQEs/CQEs and exact requested rows. Distinguish cache
   bytes from bounce and pinned staging bytes in memory accounting.

5. **Focused validation before serving.** Add unit tests for cold/warm rows,
   duplicate IDs, layer separation, set collision/eviction, cross-page rows,
   unaligned table offsets, short reads/truncated files, invalid IDs,
   capture/replay with changing IDs, deduplicated graphs, and worker cleanup.
   Compare byte-for-byte packed rows and BF16 output to `EngramFileTable.lookup`.
   Assert an instrumented cold path actually issues io_uring SQEs and the
   callback contains no `pread`/Python path. Preserve layer 1 = no break,
   layer 14 = one break in the combined graph test. Run the focused suite on
   `divix01` using the isolated checkout/GPU lock, then compare several real
   hash-ID sets against the actual layer-1 shard.

6. **Paired performance check.** On `divix01`, use the existing serving
   harness and a fresh matched control to compare the current eager cache
   path, the direct-`pread` graph proof, and the io_uring cache graph path.
   Keep model, corpus, GPU settings, RAM budget, and all other cache settings
   fixed. Record graph breaks, D2H host block, hit rate, SQE/CQE count,
   storage bytes, callback wait, decode throughput, and memory. A correctness
   proof alone does not establish a speedup. Leave the opt-in flag off by
   default and report whether to proceed to the separately planned 20 GiB
   pinned cache.

## Acceptance gates

- Same BF16 results for changing and repeated IDs, including actual shard.
- Cold miss goes through io_uring; warm hit produces no file SQE. No `pread`
  fallback in the graph callback.
- No thread-ownership error, deadlock, stale row, or callback lifetime race
  across repeated capture/replay and teardown.
- One layer-14 break remains; layer 1 has none. Baseline mode is unchanged.
- A paired serving result and resource report are attached before claiming
  any performance improvement.
