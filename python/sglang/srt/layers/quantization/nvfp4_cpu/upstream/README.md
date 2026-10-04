# Pinned GGML CPU kernels

Source: ggml-org/llama.cpp commit
`889edf43ddae0cfe9a4564a882764dc879759870`.
License: MIT; see [LICENSE.llama.cpp](LICENSE.llama.cpp).

This is a small CPU-only extraction, not a complete GGML backend or a CUDA
import. No llama.cpp checkout, Python package or downloaded source is needed
at build time.

`nvfp4.c` contains unchanged bodies of:

- `ggml_vec_dot_nvfp4_q8_0`, `ggml/src/ggml-cpu/arch/x86/quants.c`.
- `quantize_row_q8_0_ref`, `ggml/src/ggml-quants.c`.

`kernels.h` supplies the selected `block_nvfp4`, `block_q8_0`, FP4 LUT and
constants from `ggml-common.h`, unchanged half/UE4M3 conversion bodies from
`ggml-impl.h`, and unchanged x86 reduction/int8 dot helpers from
`ggml-cpu/arch/x86/quants.c`. Surrounding GGML headers, dispatch tables,
visibility macros and unrelated formats have been replaced by a small
standalone declaration shim. CPU FP16 macros bind to the pinned compute
conversions rather than GGML global lookup tables; baseline and adapted
kernels use the same binding. The FP16 conversion originates in FP16, as
acknowledged in the upstream source. The helpers have their original ISA
preprocessor guards. Native x86 AVX2 builds also require FMA, as upstream does.

## Adaptation boundary

`../optimized/dot_nvfp4.h` is derived from the same dot-product function.
The changes are its signature, weight/scale access, and nibble interleave:
GPU bytes hold adjacent columns; GGML bytes hold columns j and j+8 per group.
Integer multiply/reduction, doubled FP4 LUT, activation deltas and floating
accumulation remain GGML arithmetic. Signed GPU E4M3 scales are supported;
GGML's baseline instead flips the weight sign nibble and uses unsigned scales.
Shapes divisible by 16 get zero padding in activation scratch and at most
32 bytes of stack weight scratch for a partial 64-value block. GPU slabs are
never overwritten, cached in a second format, or persistently repacked.

`../optimized/moe_mul1.cpp` and its headers (`quant.hpp`, `shapes.hpp`, `forward_plan.hpp`) are SGLang glue:
registration, the OpenMP forward plan, GPU scale addressing, global alphas, FP16 input, SiLU, routing and callback ABI.
Its Q8 wrapper zeros blocks with a zero FP16 delta before calling the original
quantizer, avoiding reciprocal overflow on tiny FP32 inputs. Nonfinite inputs
or deltas above finite FP16 range return status 2 without publishing output.
The kernel reads the GPU layout directly; the unchanged upstream dot product
remains only as the reference in the GGML differential check
(`test/registered/unit/kernels/nvfp4_cpu_ggml_check.cpp`).

## Reproduce the import

Fetch the listed files from the pinned commit and extract the named function
bodies without edits. The only non-comment content of `nvfp4.c` outside those
bodies is its `kernels.h` include. Source file hashes used for this extraction:

- `ggml/src/ggml-cpu/arch/x86/quants.c`: `3c489fdc77e3ab484a5f188bff9c60c0e9483c951ed65c1a562a04c249d5628e`
- `ggml/src/ggml-quants.c`: `5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab`
- `ggml/src/ggml-impl.h`: `43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3`
- `ggml/src/ggml-common.h`: `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`
