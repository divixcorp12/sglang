# ModelOpt NVFP4 Selected-Expert Streaming Design

Date: 2026-09-11

## Objective

Run NVIDIA's `Qwen3.8-Flash-Next-NVFP4` checkpoint on one 32 GiB Blackwell
GPU by retaining routed-expert weights in host memory and staging only the
experts selected for the current MoE layer. Preserve the checkpoint's ModelOpt
NVFP4 representation, the existing PLE host-offload path, and the FP8 MTP/NEXTN
weights.

The first milestone prioritizes correct startup and generation. It does not
attempt hot-expert placement, transfer/compute overlap, direct host-pointer
GEMV, CUDA graph support, or concurrent requests.

## Current Failure

PLE host offload prevents the 47.69 GiB n-gram table from being allocated on
the GPU. Model construction then fails while allocating routed-expert tensors.
Each decoder layer contains 512 routed experts, while only 10 experts are used
per token. Stock grouped CPU offload stages complete modules and is also
rejected when PLE offload is active because it would stage the PLE table back
to the GPU.

The NVIDIA checkpoint uses a mixed layout:

- decoder routed experts: ModelOpt NVFP4, group size 16;
- PLE n-gram table: FP8;
- MTP routed experts: FP8 block quantization;
- other parameters: BF16 or their checkpoint-declared type.

Only the decoder routed-expert ModelOpt NVFP4 tensors are in scope for the
initial streamer.

## Supported First-Milestone Configuration

The selected-expert path is feature-gated by
`SGLANG_MOE_EXPERT_STREAM=1`. It initially requires:

- one CUDA device;
- tensor parallel size 1 and expert parallel size 1;
- `moe_runner_backend=flashinfer_cutlass`;
- standard dispatch (`moe_a2a_backend=none`);
- ModelOpt NVFP4 decoder experts;
- one running request;
- overlap scheduling disabled;
- CUDA graphs disabled;
- PLE embedding host offload enabled.

Unsupported combinations fail during argument or model validation with a
specific error. With the environment variable unset, existing SGLang behavior
is unchanged.

## Architecture

### Expert-only host placement

Extend the V1 CPU offloader with a gated expert-only mode. In that mode it
moves only ModelOpt NVFP4 routed-expert parameters into pinned CPU storage and
does not install the stock whole-module functional-call wrapper.

The initial offload set includes the load-time expert tensors:

- `w13_weight`;
- `w2_weight`;
- `w13_weight_scale`;
- `w2_weight_scale`;
- `w13_weight_scale_2`;
- `w2_weight_scale_2`;
- any full-expert derived scale placeholders created during initialization.

The input activation scales are small. The two scalar activation quantization
values produced by ModelOpt post-processing remain on the GPU.

The compatibility check permits `--cpu-offload-gb` together with PLE offload
only when selected-expert streaming is enabled. Generic offload remains
rejected.

### CPU post-processing

The FlashInfer CUTLASS backend consumes swizzled block scales. The stock
`swizzle_blockscale` helper always materializes its result on CUDA, which would
reintroduce full-expert GPU allocation. Add a device-preserving CPU-capable
variant or an explicit CPU implementation for the streaming path.

After checkpoint loading, construct and pin these final per-expert sources:

- packed `w13_weight` and `w2_weight`;
- CUTLASS-swizzled `w13_blockscale_swizzled` and
  `w2_blockscale_swizzled`;
- `g1_alphas` and `g2_alphas`.

All row tensors must be contiguous, pinned, share identical logical expert
ordering, and have expert count as dimension zero. Intermediate raw scales may
remain in host memory during initial development; releasing redundant host
copies is a later memory optimization.

### Compact staging

Adapt Haberstroh's `ExpertStreamer` pattern for the six final NVFP4 tensors.
A Triton row-copy kernel reads pinned host rows through UVA and writes them to
reusable CUDA buffers. Copies are issued on the same stream as the subsequent
FlashInfer invocation, so stream ordering provides correctness without a host
synchronization.

For small decode batches, do not deduplicate selected IDs. Ten routed entries
produce ten staged rows, and the IDs are replaced with `0..9`. Duplicate
experts therefore occupy duplicate staging rows but preserve exact token-slot
mapping without discovering a data-dependent unique count on the host.

For larger prefill batches, deduplicate selected IDs and use the inverse map as
the compact expert IDs. Staging buffers can grow to 512 rows, but buffers are
shared sequentially across decoder layers, so at most one complete decoder
layer's routed-expert payload is resident as staging storage.

### FlashInfer execution

Build a temporary `FlashInferCutlassMoeQuantInfo` from the staged tensors:

1. scalar `w13_input_scale_quant`;
2. compact `w13_blockscale_swizzled`;
3. compact `g1_alphas`;
4. scalar `w2_input_scale_quant`;
5. compact `w2_blockscale_swizzled`;
6. compact `g2_alphas`.

The packed compact weights replace the resident full-expert weights. Routed
IDs are already remapped into the compact range. TP and EP remain one. The
existing FlashInfer CUTLASS NVFP4 fused kernel is otherwise unchanged and
infers its available expert count from the staged weight tensors.

The model's router, shared expert, dense components, Mamba/GDN state, PLE
implementation, and FP8 MTP layer are not modified by the initial streamer.

## Runtime Data Flow

1. The router produces original expert IDs in `[0, 511]` and routing weights.
2. The streamer derives staged source IDs and compact routed IDs.
3. One gather operation per tensor copies aligned expert rows into reusable
   CUDA staging buffers.
4. The dispatch output is copied structurally with compact IDs; the original
   routing weights are retained.
5. FlashInfer CUTLASS evaluates the compact NVFP4 experts.
6. The normal combine path returns the routed result and the shared expert is
   combined by existing model code.
7. The next decoder layer reuses the staging buffers after the current stream
   has consumed them.

## Error Handling and Diagnostics

Startup validation reports which unsupported property was detected instead of
silently falling back. Post-load validation checks source device, pinned state,
contiguity, expert dimension, corresponding row counts, and required dtypes.
Gather validates routed-ID bounds. The first active layer logs source and
staging shapes and byte counts; routine inference does not log per layer.

Any staging or FlashInfer failure terminates the request rather than executing
with incomplete weights. Removing `SGLANG_MOE_EXPERT_STREAM=1` restores the
stock path.

## Verification

1. Unit-test CPU scale swizzling against the existing CUDA result.
2. Unit-test compact ID remapping, including duplicate IDs and all-expert
   prefill input.
3. Unit-test aligned gathering of packed weights, FP8 block scales, and FP32
   alpha tensors.
4. On the RTX 5090, compare a small synthetic resident NVFP4 MoE invocation
   with the compact staged invocation using identical inputs and routing.
5. Start the checkpoint with a small context/token pool and NEXTN disabled.
6. Run deterministic short generation and inspect memory use.
7. Enable NEXTN and verify draft execution separately.
8. Restore the 262144-token configuration and measure peak GPU memory, host
   memory, prefill rate, decode rate, and stability.

## Initial Bring-up Configuration

The first server run adds `SGLANG_MOE_EXPERT_STREAM=1` and uses approximately
55 GiB of expert-only CPU offload. It retains FP8 PLE host offload and the
FlashInfer CUTLASS backends. It temporarily uses chunked prefill size 1024, a
small context/token pool, one request, no overlap schedule, no radix cache, no
CUDA graphs, and no speculative decoding.

Settings are restored one at a time after base generation succeeds: NEXTN,
larger token pools, then the 262144 context target.

## Non-goals

The first milestone does not support TP or EP greater than one, A2A dispatch,
multiple concurrent streams, CUDA graph capture, dynamic hot-expert placement,
expert residency caching, VMM elasticity, transfer prefetch, or non-ModelOpt
quantization formats. These are separate follow-up designs after correctness
and baseline performance are measured.
