# DSV41 decode: overlapping expert-row copies with compute

Goal: decode runs at 116 ms/token, and the RAM-to-GPU copy of VRAM-miss rows (C1) is 72 ms of it. Priorities: overlap
copy with compute first, then fuse elementwise chains, then trim the Python between steps.

Branch `cc/copy-overlap` (from `cc/dsv41-pinned-numa` at `37c8927027`). Every number below is measured unless it says
*estimate*. Evidence lives under `divix01:/data/models/slang/nvfp4-work/copy-overlap/`.

## 1. What the decode step is made of (investigation A)

Source: node trace `/mnt/nvme1/dsv41-nsys/default-node-20260924-193728.sqlite` (96 steps x 40 layers, one graph per
step, 6,112 kernels per step, **all on one stream**). Script: `analysis/dsv41-drive/copy-overlap/layer_chain.py`;
output `layer_chain_default_node.json`.

### 1.1 Per-layer chain, in stream order

| Group | What | Kernels / step | ms / step |
|---|---|---:|---:|
| attn | mHC mix, norms, q/kv `exl3_gemv` x4, rope, sparse MLA, o-proj (`Kernel2`, grid 8) | 1,485 | 7.50 |
| shared | shared expert: `exl3_gemv` gate, up, `silu_mul_clamp`, down, plus casts. **Runs before routing** | 440 | 1.67 |
| route | router `tiny_n_gemm`, `_router_triton_kernel` top-k, fused plan, DIRECT `gather_destinations` | 1,254 | 1.39 |
| chain | post, W1 (`stream_hit_wait`), **C1**, A1, **S**, A2, F | 280 | 95.15 |
| book | `torch.add` go_total, DIRECT `commit_gather`, the MoE apply's sort / scan / index prep | 2,560 | 2.00 |
| moe | `exl3_moe_kernel` (all 6 routed experts in one launch, grid 28) and `exl3_moe_gather` | 80 | 4.02 |

Chain detail per step: C1 72.26 ms, S 21.30, post 0.82, W1 0.57, stage acks 0.14, F 0.06. Kernels sum to 111.7 ms of a
116.5 ms span; the rest is about 6,100 inter-kernel gaps of ~0.2 us plus the Engram host nodes (1.6).

The dependency chain within a layer is: attention, then routing, then post, W1, C1, S, then F, then the MoE kernel,
then the next layer's attention. Only three pieces of work do not need this layer's missed rows:

- **the shared expert** (1.67 ms/step): it needs neither the routing nor the rows;
- **`commit_gather`** (part of book): it needs F's `go_total`/`keep`, but nothing reads its writes until the next
  forward, so it can run beside the MoE kernel;
- **the resident experts' share of `exl3_moe_kernel`**: 160 of the 240 routed lanes per token are VRAM hits
  (79.9 misses per token, section 24.9 of `DSV41_REFERENCE.md`), but the kernel runs all six in one launch.

Everything else is on the critical path: routing needs attention, the next layer's attention needs this layer's
MoE output.

### 1.2 C1's launch shape, and the SM ceiling

C1 (`copy_expert_row_segments_gpu_kernel`) is **8 CTAs x 256 threads**; S is also 8 CTAs; every other chain kernel is
1 CTA. So the chain occupies at most 8 of the 170 SMs: SM occupancy never blocked overlap, the single stream did.

Bench `analysis/dsv41-drive/copy-overlap/c1_bench.py` (`bench/c1_node0.jsonl`, six real segments, pinned node-0 rows,
GB/s at p50, 30 reps):

| Variant | 1 row | 2 rows | 4 rows | with a concurrent HBM-bound GEMV (4 rows) | GEMV slowdown (4 rows) |
|---|---:|---:|---:|---:|---:|
| copy engine (`cudaMemcpyAsync` per segment) | 13.46 | 13.54 | 13.61 | 13.52 | 1.01x |
| C1 as shipped (grid 8) | 12.13 | 12.25 | 12.30 | 12.04 | 1.04x |
| C1 at grid 16 / 32 / 64 | 12.14 | 12.24 | 12.29 | 12.04 | 1.08 / 1.25 / 1.33x |
| grid 8, 2 / 4 / 8 loads in flight per thread | 12.13-12.15 | 12.24-12.25 | 12.29 | 12.03 | 1.09-1.15x |

- **An SM copy gets 12.1-12.3 GB/s however it is launched.** Neither more CTAs nor more loads in flight lifts it, so
  this is not a latency limit. The copy engine gets 13.5-13.6 GB/s, about 11% more.
- **Under concurrent compute** the SM copy loses 2-7% and slows the GEMV by 1-33% as the grid grows. The copy engine
  loses 1% and slows nothing.
- The NUMA bench's "zc" 12.2 GB/s path was `torch.bitwise_or`, not C1, but lands at the same ceiling.

### 1.3 Lease constraints on moving C1

The seqlock re-read of LEASE_PROTOCOL 11.4 is in **W1** (`lease_stream_hit_wait_kernel`), not in C1. C1 is a plain
copy of W1's compacted plan (`host_rows_1`, `dst_slots_1`, `go_1`). What makes it safe is the lease: the service
cannot reuse a leased host slot until `stage_ack(1)` runs after C1 in stream order. A copy-engine C1 therefore needs
no seqlock of its own, but it needs the lease held until its copies complete, and a completion the graph can wait on.

### 1.4 Can the graph drive the copy engine itself?

- **Captured memcpy nodes** have fixed addresses. C1's source row and destination slot are chosen on the device each
  step (W1 and DIRECT's victim list), so a captured node cannot follow them.
- **Device-side graph updates** (`cudaGraphKernelNodeUpdatesApply`) cover kernel nodes only, not memcpy nodes.
  **Device graph launch** can run memcpy nodes, but with the same fixed addresses; one graph per (host row, slot) pair
  is 8,063 x 1,128.
- **A graph break around C1**: the host does not know the rows without a device-to-host sync.
- **The C++ RAM-miss service thread** knows each lane's host slot when it grants it. With the destination slots in
  the LaneRequest, it could issue the copies itself. This is the only route (section 3, 1b).

## 2. Engram host nodes, and the launch block

The decode graph contains exactly **two host nodes per step**: `EngramHostLookup::callback` at layers 1 and 14
(`python/sglang/srt/layers/engram_host_node.cpp`). Each blocks the stream while the host looks the step's hashed
n-gram ids up in the native row cache and reads the misses with io_uring into pinned staging; a graph copy node then
moves the rows up. From the node trace: 192 host nodes in 96 steps, **p50 1.17 ms, p90 2.21 ms, max 33.8 ms**,
2.5 ms of stream stall per step. There are no other host nodes and no other host-blocking points in the decode graph;
the RAM-miss waits are device spins.

## 3. Design

### 1a. Shared expert and `commit_gather` on a side stream (implemented, section 4)

- **What:** one process-wide side stream (`python/sglang/srt/layers/moe/moe_side_stream.py`). The shared expert forks
  before the router and runs beside route + chain. `commit_gather` forks after F and runs beside the MoE kernel. The
  MoE layer joins before the shared-expert add, so every fork is joined within its layer.
- **Expected saving** *(estimate)*: up to 1.67 (shared) + ~0.8 (commit, about 22 of book's 64 kernels per layer)
  = **~2.5 ms/token**, less whatever the side work slows the main stream's small kernels by.
- **Risk:** the breakable decode graph has given wrong output with side streams before
  (`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`, see `expert_stream_requirements_exl3.py`). This fork is narrower: no graph
  break sits between fork and join. Allocator reuse across streams is covered by `record_stream` on every
  cross-stream tensor. The correctness bar is the byte-identical smoke.

### 1b. C1 (and S's copy) on the copy engine, issued by the service thread

**Expected saving** *(estimate from the bench)*: C1 72.3 ms at 12.25 GB/s becomes about 65.3 ms at 13.55 GB/s:
**~7 ms/token**. S's copy share (about 9.8 rows per token) adds ~1 ms. The copies also leave the SMs, which is what
1a and 1c need so that compute does not slow the copy.

**(i) LaneRequest carries destination slots.** Today a LaneRequest is 64 B: gen (8), count (4), row (4),
`kMaxIds` lane experts (4 each). The post kernel knows `plan.slots` (DIRECT's destinations) but does not publish them.
Grow the record to 128 B with `int32 dst_slot[kMaxIds]`, move `kLeaseLaneAck` and everything after it by the region
map, and update the `static_assert`s on both sides (`exl3_ram_miss.cuh`, `exl3_ram_miss_host.cpp`) and the Python
layout mirror. The seqlock shape of the write is unchanged: clear gen, fence, payload, release gen.

**(ii) The service issues the copies.** At grant time, for each granted RAM-hit lane, the service issues six
`cudaMemcpyAsync` (one per segment, pinned slab row to hot slot) on its own non-blocking stream, then a 4-byte
`cudaMemcpyAsync` of the request's generation from a pinned ring into a device completion word. Under piece
streaming, each published piece gets the same treatment: the copy engine replaces S's SM copy. The graph's W1/C1
become one wait kernel that spins on the completion word (with the deadline and fatal paths W1 has today).

**(iii) Service-observed completion replaces the stage acks.** The service holds each lane's lease until an event
recorded after its copies has completed (`cudaEventQuery` in the polling loop), then retires it. The device no longer
acknowledges stage 1 (or stage 2). LEASE_PROTOCOL 6.3, 7.2-7.5 and area P change. The invariant restated: no host slot
is rewritten while a copy that reads it may be in flight (the service owns both), and no hot slot is read before its
copy completes (the graph's wait on the completion word, ordered before the MoE kernel).

**Probe (ii), the precondition, and its result: DEADLOCK.** `analysis/dsv41-drive/copy-overlap/ce_probe.py` builds a
graph of 40 x (140 filler kernels, post, wait) and replays it back to back. A C++ service thread polls the post word,
issues the copies and posts completion.

| Graph | Requests | Timeouts | post -> done p50, vs copy engine alone | host issue cost | `cudaGraphLaunch` p50 |
|---|---:|---:|---|---:|---:|
| no host nodes, 1 / 2 / 4 rows, 30 replays | 1,200 each | 0 | 982.6 vs 988.7 / 1956.7 vs 1969.4 / 3906.8 vs 3916.2 us | 17 / 29 / 58 us | 9 us |
| no host nodes, 2 rows, 300 replays | 12,000 | 0 | 1957.6 vs 1964.0 us | 30 us | 9 us |
| **2 host nodes per step** (as Engram), 2 rows | - | **stalled** | - | - | blocks |

- Without host nodes the path is sound: completion lands at copy-engine speed and issuing hides behind the first
  copy.
- With host nodes, `cudaGraphLaunch` blocks, as production's does (104 ms per step). While blocked, the main thread
  holds the driver's context lock: its stack is `cuGraphLaunch` on a condvar, and the service thread's
  `cudaMemcpyAsync` waits in `cuMemcpyHtoDAsync_v2` on `pthread_rwlock_rdlock`. The graph spins on a word only those
  copies can write, so only the device timeout breaks it. Stacks: `probe_rows2_hostnodes_DEADLOCK_stacks.txt`.
- **1b is therefore blocked on section 5 below.** It is not safe to ship while the decode graph has host nodes.

**Risks beyond the deadlock:** the service thread becomes a CUDA API caller (context binding, error handling at
shutdown and quarantine), and 58 us of API time per 4-row request lands on the thread that also reaps io_uring.
A dedicated copy-issuing thread keeps the reaper's latency.

### 1c. Per-expert pipelining of the MoE GEMM

Start each missed expert's GEMM as its row lands, and the resident experts' GEMM before any copy finishes.
- **Expected saving** *(estimate)*: at most the resident share of `exl3_moe_kernel`, 160/240 of 3.85 ms = **~2.5
  ms/token**, if a two-launch split costs no more than one launch. Per-miss pipelining adds almost nothing beyond
  that: the last row still gates the layer, and one expert's GEMM is ~16 us.
- **Risk:** the MoE kernel at M=1 is latency-bound (grid 28); two launches may cost more than the overlap buys. It
  also needs a resident/missed mask in the kernel. Only worth it once C1 is off the SMs (1b), or it slows the copy.

### 1d. How this sets up next-layer prefetch (`DSV41_REFERENCE.md` 24.7 item 2)

Prefetch is where the time is: the link idles about 34 ms per step (116 ms step, ~82 ms of copies), and within-layer
overlap can hide at most ~5 ms of compute (1a + 1c). Prefetch has to copy layer L+1's likely rows during layer L's
attention and MoE.
- 1a builds the fork/join discipline a prefetch stream needs inside the graph.
- 1b is what makes a prefetch copy cheap: issued by the service, on the copy engine, with no SM cost. It needs the
  same LaneRequest, lease and completion machinery, driven by an advisory instead of a demand.
- With SM copies, a prefetch C1 on a side stream would compete with attention's GEMVs (the bench's 1.25-1.33x
  slowdown at wide grids) and is capped at the same 12.3 GB/s.

## 4. What was implemented (C)

Flag `SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM` (`environ.py`), default off; flag off is the previous single-stream order.
Files: `moe_side_stream.py` (new), `deepseek_v2.py` (`DeepseekV2MoE.__init__`, `forward_normal`), `expert_stream.py`
(`_gather_graph`), `expert_residency_gpu.py` (`pending_commit_tensors`). Tests:
`test/manual/dsv41/test_moe_side_stream_gpu.py`.

Results: section 7.

## 5. Unblocking 1b: remove the decode graph's host nodes (analysis only)

- **What they do:** see section 2. The ids come off the device through a graph D2H node; the callback does the cache
  lookup and io_uring reads synchronously on the stream (`Store::submit` waits for the worker), then a copy node
  moves the rows up. Cost 2.5 ms of stream stall per step, max 33.8 ms.
- **The RAM-miss pattern fits.** Replace the host node with a device post kernel (write the ids and a sequence
  number to mapped memory) and a device wait kernel on a completion word. The Engram io_uring worker thread
  already exists; it would poll the post word, the way the RAM-miss service does, instead of being called from the
  callback. The hash ids depend only on the input token ids, so the post can sit at the very start of the step, or
  the host can compute them before launch. Either way the lookup overlaps layers 0-13 instead of stalling the stream,
  **saving up to 2.5 ms/token by itself**.
- **Does it remove the launch block?** In the probe, yes: the same graph without host nodes launched in 9 us (p50,
  max 3.6 ms) over 300 back-to-back replays; with host nodes it blocked, back to back and one step ahead (section 8).
  One mechanism, one probe; to be confirmed on the real graph by a graph-mode trace (`cudaGraphLaunch` duration)
  after the change.
- **What a non-blocking launch changes:** the host returns from the launch immediately and would reach the next
  step's first device-to-host dependency (the sampled token's readback) sooner. It then blocks there instead, so the
  GPU step does not shorten. What it buys is that no host API call ever waits behind a blocked launch, which 1b
  needs. Nothing needs a new sync: the scheduler already reads the sampled token through its existing readback. What
  needs checking is anything that assumed "launch returned" meant "previous step done", such as the RAM-miss
  service's `_trace_step` lagged readback and the Engram store's staging reuse. With a blocking launch the host could
  never be two steps ahead; now it can.
- **Effort** *(estimate)*: 2-3 days. A device post/wait pair on the RAM-miss model, the Engram worker switched from
  callback to polling, staging double-buffered per step, and the graph parity and Engram golden tests re-run.
- **Risks:** the Engram lookup becomes asynchronous to the stream, so a slow read (the 33.8 ms max) now stalls the
  wait kernel with a device timeout instead of a host node. Staging reuse across two in-flight steps needs a
  generation check. `test_engram_parity.py` and `dsv41_engram_golden.npz` must stay exact.

## 6. Elementwise catalog and fusion plan (item 2)

From `layer_chain.py`: 6.3 ms/step of kernels of 5 us or less, plus ~1.2 ms of gaps between the ~6,100 kernels.
Attribution is by stream position within the layer, checked against the code.

| Chain (per layer) | Source op | Kernels / layer | ms / step | Fusion |
|---|---|---:|---:|---|
| gemv casts | `unrolled_elementwise` before and after each `exl3_gemv` (attn x4, shared x3) | ~14 | ~0.93 | take bf16 in and write bf16 out inside `exl3_gemv_int8_sq` |
| route plan | router output handling, `plan_unique_routes`, DIRECT `gather_destinations` (hazard compare, argsort, index_select, where, copy) | ~29 | ~1.25 | one JIT kernel: routes plus victim assignment |
| DIRECT commit | `commit_gather`: `where`, 5 scatters, `scatter_add`, 3 fills, 2 sums, `insertion_truncated` | ~22 | ~0.8 | one JIT kernel; also makes the 1a fork one launch |
| MoE prep | sort (`bitonicSortKVInPlace`), scan (`DeviceScan*`), `indexFuncSmallIndex`, casts, `cat` before `exl3_moe_kernel` | ~20 | ~0.6 | fold into the fused planner's output (it already knows the unique routes) |
| mHC | `_hc_mix_stats_partial`, `_hc_mix_reduce_sinkhorn`, `_mhc_post_split_h`, `_hc_combine_norm` (x2 per layer) | 8 | ~0.89 | already Triton; merge stats + reduce |
| shared-expert glue | `CatArrayBatchedCopy` + `silu_mul_clamp` + cast | 3 | ~0.07 | fold into the down-proj input |

*Estimate:* the four JIT fusions (casts, plan, commit, MoE prep) would remove ~85 of ~153 kernels per layer, about
3.5 ms/step of kernel time and gaps. Do this after the overlap work.

## 7. Measurements of 1a

**Tests** (commands recorded in `divix01:.../copy-overlap/suite/*.log`):
- CPU: `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 python -m pytest test/registered/unit/kernels -q -p no:randomly -k "ram_miss or lease or piece or expert"`
  at `f26c910abe`: **1629 passed, 37 deselected, PIPESTATUS 0** (`cpu_head.log`).
- GPU: `analysis/dsv41-drive/copy-overlap/gpu_suite.sh` (the seven manual RAM-miss files of 24.8/24.9, the RAM-miss
  graph file, the side-stream file): head `b4e77e5ebb` **120 passed, exit 0**; base `37c8927027` **116 passed, exit 0**.
  The difference is the four new side-stream tests. (A first head run without the suite's `SGLANG_EXL3_SRC` env
  failed 18 on the environment; `gpu_head.log`.)
- Mutants on `moe_side_stream.py`, each reverted and the file re-run green: dropping the join's wait fails the
  ordering test; dropping the fork's wait fails the captured test.

**Smokes**, 100 GiB `arm_env` recipe, cold server per arm, order off/on/on/off, same session
(`analysis/dsv41-drive/copy-overlap/smoke.sh`, `compare_arms.py --include-warmup`; `smoke/compare_abba.json`):

| Arm | ms/token, trace (wall-validated) | step p50 / p90 | 1-row submit->done p50 | stalls (multi-row > 10 ms) |
|---|---:|---:|---:|---:|
| off 1 | 131.8 (134.4) | 126.9 / 161.4 | 2604 us | 1 of 2004 |
| on 1 | 131.9 (134.5) | 127.4 / 161.9 | 2599 us | 2 of 2023 |
| on 2 | 132.1 (134.7) | 128.2 / 161.6 | 2601 us | 0 of 2026 |
| off 2 | 131.9 (134.6) | 128.0 / 161.1 | 2604 us | 0 of 2026 |

Responses are **byte-identical**, 6 of 6 for every arm against off 1, and each arm's reps agree. **No gain:** the
arms sit within 0.3 ms/token of each other, with the order effect as large as the flag's.

**Node-mode traces, flag on and off**, same prompts, 393 paired steps (`smoke/node-{on,off}/trace.sqlite`).
Attribution only; node mode inflates small-kernel cost, so these are not ms/token:
- The fork works. Shared-expert GEMVs run beside the router and plan kernels, and `commit_gather`'s scatters run
  beside the MoE prep and `exl3_moe_kernel`. Per layer, norm-to-post fell 87 -> 62 us and F-to-MoE 65 -> 42 us.
- The main-stream kernels running beside the GEMV slow down (`plan_unique_routes` 3.1 us against ~1.3).
- Paired per step: time outside the chain **-1.92 ms**, but C1 **+1.05 ms** and S -0.14. Node-mode span -1.69 ms.
  The graph-mode smokes, which have far smaller per-kernel overhead, show none of it.

**Verdict:** 1a is correct and measurably overlapped, but saves nothing at the wall. Leave the flag off. The
within-layer compute that can move off the chain is too small (about 2 ms/step in node mode, less in graph mode) and
partly comes back as a longer C1. Per the brief, no graph-mode trace was taken because the smoke showed no gain.
The node traces above are the confirmation that the overlap exists.

## 8. Python between steps (item 3)

`py-spy record --rate 250 --duration 90` on the scheduler during the off arm's decode (`smoke/pyspy-off/`, 21,119
samples):

| Where the scheduler thread was | Share |
|---|---:|
| waiting in `process_batch_result_decode` -> `torch.cuda` `synchronize` (the previous step's result) | 65.0% |
| eager prefill forward (the six requests' prefills) | 28.5% |
| decode Python outside that wait | 6.2% |

Of the decode Python outside the wait, **82%** is `_expert_doorbell_fail_stop_check` -> `stage_records`, the RAM-miss
stage-trace drain, which runs only because the smoke sets `SGLANG_DSV41_EXPERT_TRACE_PATH`. The rest (prepare
for decode, graph `replay()`, allocation) is about 0.9 s over ~470 steps, **~2 ms per step, all while the GPU runs**.
**Item 3 has no decode value today:** the host waits for the GPU two-thirds of the time and never gates it.
If the stage trace is on during a measurement, it costs host time but is hidden the same way.

The same profile shows `graph.replay()` taking about 1% of the decode Python, while section 24.9's graph-mode nsys
trace put the host in `cudaGraphLaunch` for 104 ms per step. The two disagree; whether nsys's graph tracing moves the
wait into the launch is not established. The probe below settles what matters for 1b either way.

**Probe (ii) again, as the overlap scheduler launches** (`--ahead 1`: launch step i, then wait for step i-1; 2 host
nodes per step; 200 ms device timeout; `probe_rows2_hostnodes_ahead1.json`): 2,400 requests, **14 timeouts**. Every
timeout fell in one `replay()` call that blocked for 2.8 s. So the deadlock also occurs one step ahead, not only
under back-to-back launches.

## 9. Recommended next step

1. **Remove the Engram host nodes (section 5).** It saves up to 2.5 ms/token directly, and it is the precondition
   for 1b's ~7-8 ms/token and for any copy-engine prefetch. Re-run `ce_probe.py --host-nodes --ahead 1` against the
   new scheme's launch pattern, and take a graph-mode trace to confirm `cudaGraphLaunch` no longer blocks.
2. Then **1b** as specified in section 3, then **prefetch (1d)** on the same machinery.
3. Drop 1a (flag stays off) and 1c until C1 is off the SMs. Elementwise fusion (section 6) is worth ~3.5 ms/step in
   kernel time. Given 1a's result, measure one fusion before committing to all four.
