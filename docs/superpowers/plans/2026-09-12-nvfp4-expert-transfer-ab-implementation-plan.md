# NVFP4 Expert Transfer A/B Implementation Plan

**Goal:** Add two interchangeable expert-row transfer engines for the existing NVIDIA ModelOpt NVFP4 hot cache: a graph-capable GPU pull kernel and a CUDA copy-engine DMA backend.

**Scope:** This iteration builds and validates the transfer engines behind one fixed-plan contract. It does not launch the server, change cache sizing, or replace the existing residency policy. End-to-end policy integration follows once both engines can move the same six NVFP4 tensors correctly.

## Common contract

Both engines consume the existing `FixedRowTransferPlan` fields:

- `source_rows`: global expert rows (`int64`, fixed CUDA allocation)
- `destination_slots`: hot-cache slots (`int32`, fixed CUDA allocation)
- `count`: active prefix length (`int32`, fixed CUDA scalar)

Each submission copies all six ModelOpt NVFP4 expert tensors on the executor's transfer stream and publishes one completion event. Metrics must identify the requested backend, actual backend, row count, and bytes submitted.

The comparison is deliberately honest:

- `gpu`: a fixed-grid CUDA kernel reads the device plan and pulls registered/pinned host rows into HBM. It must not call `.item()` or allocate tensors while captured.
- `dma`: SGLang's existing `transfer_embedding_ranges_direct` batches CPU-to-CUDA row copies through `cudaMemcpyBatchAsync` (or SGLang's runtime fallback). It consumes a CPU-visible plan and is intended for uncaptured control-plane transfers.

## Task 1: GPU planned-row copy kernel

**Owner files**

- `python/sglang/kernels/jit/csrc/moe/expert_cache_transfer.cuh`
- `python/sglang/kernels/ops/moe/expert_cache_transfer.py`
- `test/registered/unit/kernels/test_expert_cache_transfer.py`

**Steps**

1. Add a focused CUDA test that copies selected rows from pinned/registered CPU storage into selected CUDA slots using fixed device plan tensors.
2. Adapt the established HiSparse JIT pattern (`hisparse.cuh` and its Python wrapper), including non-caching host loads and cache-global device stores where appropriate.
3. Expose `copy_expert_rows_gpu(source, destination, source_rows, destination_slots, count)`.
4. Validate contiguous layout, matching dtype and row width, CUDA destination/plan placement, and pinned or CUDA-registered CPU source.
5. Use a fixed launch grid and read `count` only on device. Do not use Python scalar extraction in the launch path.
6. Add a CUDA graph replay test with stable tensor addresses and changed plan contents between replays.

## Task 2: CUDA copy-engine DMA backend

**Owner files**

- `python/sglang/srt/layers/moe/expert_dma.py`
- `test/registered/unit/layers/moe/test_expert_dma.py`

**Steps**

1. Add a focused test for selected pinned-CPU rows copied into selected CUDA slots.
2. Flatten each expert tensor to a contiguous two-dimensional row matrix.
3. Reuse `sgl_kernel.kvcacheio.transfer_embedding_ranges_direct` with one-row ranges; do not introduce a second memcpy-batching extension.
4. Expose a small backend object that accepts CPU source/destination row sequences and runs on the current CUDA stream.
5. Detect unavailability of the AOT primitive and fall back to per-row non-blocking `copy_`, recording the actual path used.
6. Test validation and fallback behavior without expanding into full server tests.

## Task 3: Common backend selection and six-tensor integration

**Owner files**

- `python/sglang/srt/layers/moe/expert_transfer.py`
- `python/sglang/srt/layers/moe/expert_stream.py`
- focused existing tests in `test/registered/unit/layers/moe/`

**Steps**

1. Add `SGLANG_MOE_EXPERT_COPY_BACKEND=gpu|dma` with `gpu` as the prototype default.
2. Convert one expert movement into six backend calls under the shared transfer stream and one completion event.
3. Preserve generation-qualified slot publication and the file-backed fallback for pinned-host misses.
4. Add backend, rows, bytes, submissions, and fallback counters to the existing metrics sink.
5. Keep exact-demand transfers higher priority than speculative promotions.

## Task 4: A/B microbenchmark

**Owner files**

- `benchmark/kernels/moe/benchmark_expert_cache_transfer.py`

**Steps**

1. Generate identical fixed expert-row plans for both engines.
2. Report warmup, median latency, effective GiB/s, rows per submission, and transfer size as parseable text/JSON.
3. Measure GPU kernel and DMA paths on the same stream and pinned source allocation.
4. Add an optional CUDA-graph replay measurement for the GPU backend only.

## Focused verification

- Compile the touched Python modules.
- Run only the new and directly affected unit tests.
- Run `git diff --check`.
- Do not launch or stop the live SGLang server without separate approval.
