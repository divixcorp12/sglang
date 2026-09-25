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
- **Sequence word** (`counter`): one int32 of pinned memory that only the two kernels touch. It is not device
  memory because the lookup is built inside the capture, where `torch.zeros(device=...)` records a memset that
  would reset it on every replay (the first GPU run failed exactly so: every replay posted seq 1, and the wait
  passed at once on the stale done word, returning the previous step's rows).
- **Post kernel** (`engram_ring_post`, one thread): `seq = ++counter` (0 is skipped), copy the 24
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

## 5. Design guidance, and where the implementation stands against it

- **Zero-copy rows, and no CUDA call on the service thread.** The service writes the rows into the pinned `rows`
  buffer (torch pinned memory, device-readable through UVA like a `cudaHostRegister`ed slab) and makes no CUDA
  API call at all. There is no graph memcpy node: the wait kernel itself reads the 6,336 row bytes over PCIe with
  SM loads, as C1 reads pinned slabs. One difference from the guidance: those loads land in a small device buffer
  and the existing torch dequant ops read it, instead of a dequant kernel reading pinned memory directly. That keeps
  the flag-on arithmetic the same op chain as flag off, so byte-identity holds by construction; the extra device
  hop is 6 KB per lookup. (Rows here are 264 B, `head_dim` 256 + 8 scale bytes, 24 per layer.)
- **Post early, wait late.** Both posts run right after the hash kernel at the start of the step; the waits sit at
  layers 1 and 14 (section 2). Spin counts are measured (section 8).
- **Kept:** the NVMe miss path (io_uring, O_DIRECT, chunking) and the 5 GiB 8-way set-associative cache are the
  same `Store` calls the host node made.
- **Follow-up, not built:** a GPU-side tag probe of a pinned cache (hits read without a host round trip) needs
  lease/seqlock eviction safety like the expert tier. Section 8 says whether it would pay: the spins that remain
  are at layer 1, on steps whose lookups include NVMe misses, which a tag probe cannot serve either.

## 6. Tests

Commands and logs under `divix01:.../engram-no-hostnode/suite/`; every count is read from pytest's own status.

- **CPU** (`analysis/dsv41-drive/engram-no-hostnode/cpu_suite.sh`; GPU hidden, its cases run under the lock
  instead): `CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 python -m pytest
  test/registered/unit/kernels test/registered/unit/layers/test_engram_*.py -q -p no:randomly -k "ram_miss or
  lease or piece or expert or engram"`, then `EXIT=${PIPESTATUS[0]}`. Head `480d2ee456`: **1255 passed, 414
  skipped, 37 deselected, EXIT 0** (`cpu_final.log`); base `369acc8e2a`: 1249 passed, 414 skipped, EXIT 0
  (`cpu_base.log`). The difference is the six cases of `test_engram_device_wait.py` (the service against a
  simulated device: per-sequence serving, publish order, failed-lookup status, fatal latch after a device timeout,
  close).
- **GPU** (`gpu_tests.sh`, under `cc-gpu.lock`, cores 32-63), head `480d2ee456`:
  `test_engram_device_wait_gpu.py` + `test_engram_file_table.py`: **23 passed, EXIT 0**; with
  `SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT=1 SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=1` exported,
  `test_engram_parity.py`, `test_engram_row_cache_direct.py`, `test_engram_cache_sim.py`,
  `test_exl3_vs_fp8_engram_wkv.py`, `test_engram_device_wait_gpu.py`, `test_engram_lookup_break.py`,
  `test_engram_row_cache.py`: **46 passed, EXIT 0**. The GPU file covers zero host nodes at capture and exact rows
  over five replays (early and late post, with and without graph dedup), the capture refusing a host node, a slow
  service (launch returns in < 20 ms, the wait holds the stream, rows exact, spins counted), a timed-out wait
  failing with a device assert and latching the fatal word, and the flag requiring the native store.
- **Mutants**, each applied in the private worktree, run, and reverted; the runs above are the restored baseline.

| Mutant | Caught by | Result |
|---|---|---|
| service publishes `kDoneSeq` before serving (host-side publish order) | CPU `test_done_is_published_only_after_the_rows_and_status` | 2 failed |
| wait kernel does not spin (missing wait) | GPU exact-rows tests | 6 failed |
| `engram_ring_wait` not launched | GPU tests (uninitialized status asserts) | 8 failed |
| post kernel publishes the sequence before the ids, unfenced | GPU `test_the_post_publishes_its_sequence_after_every_id`, with a test-only stall after the first id | 1 failed; **without the stall hook it passed**, so the hook was added |

A removed `__threadfence_system()` alone, with the stores left in order, is not caught: on this x86 host the posted
PCIe writes arrive in order anyway. The release store is still what the PTX model requires.

## 7. Smokes, 100 GiB `arm_env` recipe

`analysis/dsv41-drive/engram-no-hostnode/smoke.sh` (cold server per arm, six greedy requests), order
off/on/on/off in one session at `31ea1fd082`, plus `s-on` at `6c5b97f093` (the same code plus the spin counters).
`compare_arms.py --include-warmup` (`smoke/compare_abba.json`). `compare_arms.py` needed a fix first: since the
route log, a `graph_routes` record follows every graph step and split each request's decode into one-step blocks,
so trace ms/token read 0 and the wall validation subtracted no prefill.

| Arm | ms/token, trace (wall-validated) | step p50 / p90 ms | stalls (multi-row > 10 ms) |
|---|---:|---:|---:|
| off 1 | 132.0 (134.6) | 127.7 / 161.4 | 2 of 2029 |
| on 1 | 129.4 (132.2) | 125.7 / 158.8 | 0 of 2001 |
| on 2 | 129.6 (132.2) | 125.8 / 158.8 | 3 of 2060 |
| off 2 | 132.2 (134.9) | 127.5 / 161.7 | 0 of 2056 |
| on, spin counters | 129.1 (131.7) | 125.6 / 158.5 | 3 of 2012 |

**-2.6 ms/token** (off mean 132.1, on mean 129.5); the arms of each flag sit within 0.2 ms of each other.
Responses are **byte-identical**, 6 of 6 for every arm against off 1, and each arm's two reps agree.

## 8. How often the wait spins

From `s-on`'s last `engram` cache-stats line (489 decode steps, cumulative to the last prefill):

| Layer | waits | spun | spin total | mean spin | max spin |
|---|---:|---:|---:|---:|---:|
| 1 | 489 | 89 (18%) | 190.4 ms | 2.14 ms | 26.6 ms |
| 14 | 489 | 0 | 0 | - | - |

Layer 14's lookup is always hidden behind layers 0-13. Layer 1's costs 0.39 ms/step on average, against the host
nodes' 2.5 ms: it spins only when the lookup outlasts layer 0 (~2.9 ms), which is when it reads NVMe misses. A
GPU tag probe would not remove those spins. Posting earlier would: the ids depend only on the token, so the host
could post them as soon as the sampled token is read back, or the next token's likely rows could be prefetched.

## 9. `cudaGraphLaunch`: the 104 ms block is an nsys artifact

- **Graph-mode nsys smokes, off and on** (`smoke/g-{off,on}`, `launch_durations.py`,
  `smoke/launch_durations.json`): `cudaGraphLaunch` p50 **113.7 ms off, 116.2 ms on** (489 launches each). The
  flag changes nothing here.
- **Without nsys**, py-spy on the scheduler (`smoke/pyspy-{on,off}`, 90 s at 250 Hz): `graph.replay()` is
  **0.07% of samples on, 0.10% off** (0.1-0.2 ms per step), while `torch.cuda.synchronize` on the previous step's
  result is 64-65% in both. Production's launch does not block, with or without host nodes.
- **The probe under nsys** (`probe_nsys.sh`, `probe_nsys_short/`): the same host-node-free device-wait probe
  (`--engram-device-wait --rows 2 --ahead 1 --timeout-ms 50 --replays 5`) launches in 0.064 ms p50 with 0 timeouts
  plain; under `nsys profile --cuda-graph-trace=graph` every `cudaGraphLaunch` blocks ~2 s (the whole graph,
  40 waits x 50 ms), the copy service's `cudaMemcpyAsync` blocks alongside it (max 2,007 ms), and 158 of 200 waits
  time out. With a 5 s timeout and 60 replays the traced run did not finish in 15 minutes.

Graph-mode CUPTI tracing makes the launch synchronous and holds off another thread's CUDA calls. It produces the
~104 ms figure (`DSV41_REFERENCE.md` 24.9, the copy-overlap plan) by itself, and it deadlocks any service-issued
copy that a graph waits on: **1b cannot be profiled with `--cuda-graph-trace=graph`.** The copy-overlap plan's
untraced probe deadlock with host nodes is real (section 10).

## 10. The deadlock probe, re-run

`ce_probe.py --engram-device-wait` replaces the probe's two stand-in host nodes with the real device-wait lookups of
layers 1 and 14 (posted at the step start with fresh pseudo-random ids into the real 101 GB tables, O_DIRECT
misses through the native store) and checks the captured graph for host nodes (`probe/`, `probe.sh`, `31ea1fd082`):

| Run | Requests | Timeouts | Graph host nodes | post -> done p50 (copy engine alone) | `graph.replay()` p50 / max |
|---|---:|---:|---:|---|---:|
| device wait, 30 replays back to back | 1,200 | 0 | 0 | 1956.7 us (1964.3) | 0.009 / 3.2 ms |
| device wait, `--ahead 1`, 60 replays, 200 ms timeout | 2,400 | 0 | 0 | 1957.3 us (1965.1) | 0.055 / 3.0 ms |
| device wait, 300 replays back to back | 12,000 | 0 | 0 | 1956.7 us (1963.2) | 0.009 / 3.3 ms |
| device wait, `--ahead 1`, 300 replays | 12,000 | 0 | 0 | 1957.7 us (1963.6) | 0.029 / 3.1 ms |
| **host nodes** (control, same session), `--ahead 1`, 60 replays | 2,400 | **14** | 2 | - | 0.061 / **2,802.6 ms** |

**The deadlock is gone** with the device wait: 27,600 service-issued copy requests, 0 timeouts, alongside 1,208
Engram lookups per layer, while the host-node control still deadlocks (14 timeouts, one 2.8 s replay).

The first run of the device-wait arms counted 1 timeout each, in the eager warm-up step: the lookups' first
launches loaded their kernel modules while a probe wait was spinning, and the service's copy waited for the load.
Warming the lookups before the copy service starts removed it (`probe_run1/` against `probe/`). 1b's copy thread
has the same exposure: it must not issue copies that a spinning wait depends on before every module the graph
uses is loaded (warm-up does this, or `CUDA_MODULE_LOADING=EAGER`).

## 11. Recommendation

- **Turn the flag on in the recipe** (a separate change; `arm_env` defaults were left alone here): -2.6 ms/token,
  byte-identical, zero host nodes asserted at capture.
- **1b is unblocked** on the deadlock the copy-overlap probe found: with the host nodes gone, a service thread's
  `cudaMemcpyAsync` completed 27,600 graph-awaited requests with no timeout, back to back and one step ahead.
  Two conditions carry into 1b: no module loading while a wait spins (section 10), and no graph-mode nsys capture
  of a run whose graph waits on service-issued copies (section 9).
