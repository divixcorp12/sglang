# GPTQ INT3 CPU experts

This is an explicit CPU backend for the experimental MiMo hybrid checkpoint. It preserves the original GPTQ integer codes and signed FP16 scale bits. It does not requantize weights or activations, alter the checkpoint, load checkpoint Python, or enable GPU/production serving.

`formats/gptq_int3/format.h` binds `GptqInt3CpuKernel` to the existing expert-stream host. The transport build is `("gptq_int3", "sigmoid")`; Python discovers the format explicitly with `get_format("gptq_int3")`. Generic `quant_method="gptq"` configurations are deliberately not auto-selected. The format advertises a real CPU kernel but `engine_served=False`, with GPU apply, speculative draft and graph support disabled. Whole-model serving/loader integration remains separate work.

## Storage and arithmetic

Accepted canonical projections are INT3, symmetric, group128, contiguous group indices, no act-order permutation:

| Tensor | Shape | Dtype |
|---|---|---|
| qweight | `[K/32*3, N]` | int32 |
| qzeros | `[K/128, N/32*3]` | int32; every unpacked zero is 3 (GPTQ offset +1 gives 4) |
| scales | `[K/128, N]` | FP16, finite and nonzero, either sign |
| g_idx | `[K]` | int32; `arange(K)//128` |

The lossless runtime adapter in `srt/layers/moe/cpu_experts/gptq_int3.py` handles both cross-word joins (codes10 and21). Each `[N,K]` matrix becomes a low2 plane `[N,K/4]`, a high1 plane `[N,K/8]` and transposed scales `[N,K/128]`. Gate rows precede up rows. Six slabs contain gate/up low2, high1, scales, then down low2, high1, scales. No full decoded FP16 layer is cached.

Every weight used by a dot is exactly `FP16((code-4)*FP32(scale))`, promoted to FP32 for accumulation. Gate/up outputs, optional clamp outputs, SiLU output, its product with up, and down outputs each have explicit round-to-nearest-even FP16 boundaries. Route-weight multiplication and ordered reduction use FP32. SIMD reductions may differ from a scalar or GPU reduction; bit-exact GEMM outputs are not claimed. Activations remain floating point; no INT8/VNNI activation quantization is used.

Scalar, AVX2/FMA/F16C, and AVX512F/BW/VL/FMA/F16C builds are selected by the existing compile-time `-march` policy. This translation unit uses `-O3 -ffp-contract=off`; explicit SIMD FMAs are intentional. Other formats' compiler flags are unchanged. Layer dimensions must be multiples of128 in `[128,65536]`, capacity positive, activation0 (SwiGLU), with at most8 routes and4096 rows per job. Invalid policy, nonfinite inputs/results, zero/nonfinite scales and QDQ overflow fail explicitly.

## Scheduling and ownership

The existing `Team` and `JobWindow` own execution. There is no BesTLA scheduler or extra worker pool. Four job phases perform gate/up+SiLU, down, ordered route reduction, and output publication. The first down reader is `kGatePhase=1`. A single transactional `kOutPhase=3` reads an accumulation destination only after the window's same-output dependency, validates the entire result, then writes it. Overwrite-by-store has overwrite semantics. Invalid jobs leave output untouched.

Scratch is owned by each reusable Job, with a 512 MiB float-buffer limit per job, including promoted inputs, routed intermediate/down values and final results, plus bounded routing metadata. One reusable allocation avoids accumulating several buffers' high-water marks. There is no shared or thread-local scratch. Slab owners stay in `CpuExpertLayerSpec.keep`, retained by the host. Callers must not mutate them while a job reads them. Existing leases, stage gates, admission limits and completion rules apply unchanged.

## Read-only checkpoint adapter

```python
import torch
from sglang.srt.layers.moe.cpu_experts.gptq_int3 import (
    GptqInt3CpuTrait, load_exported_expert,
)

rows = load_exported_expert(
    "/mnt/nvme2/mimo-autoround-candidates/mimo-v26-pro-reap288-w3-rtn-experimental-v1",
    69, 0,
    expected_manifest_sha256="7b1445ed9cd90d94cfc98e3cde385c618d4a133dc8adcd0cf031f890aa6b5c65",
)
# Stack selected experts in the exact slot order used by the caller; one expert here.
slabs = {name: tensor.unsqueeze(0) for name, tensor in rows.items()}
spec = GptqInt3CpuTrait(act_limit=0).layer_spec(slabs, capacity=1)
# For an already configured host with the GPTQ INT3 transport/kernel:
# host.set_cpu_layer(row, spec)
```

The reader requires an externally trusted manifest SHA, completed format/policy, matching payload hash/size, exact tensor ownership and no path escape. It hashes the same bytes passed to safetensors. It is a bounded expert reader, not an automatic full-model loader. `prepare_projection` also accepts independently verified canonical tensors directly; their provenance remains the caller's responsibility.

## Verification

CPU-only native tests live at `test/registered/kernels/ops/moe/test_gptq_int3_cpu.py`. On x86-64 Linux:

```sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CXX=g++ \
  python -m pytest -qs test/registered/kernels/ops/moe/test_gptq_int3_cpu.py
```

Set `GPTQ_INT3_REAL_ROOT` to the immutable export above to additionally validate experts `(1,0)`, `(32,17)` and `(69,0)` against an independent canonical decoder. The manifest pin is explicit in that optional test. `GPTQ_INT3_SANITIZER_CXX=clang++` selects a separate compiler when the main compiler's sanitizer runtime is unavailable; numerical builds still use `CXX`.

Tests cover exact layout/scale bits, random packed bytes and cross-word joins, all65536 half bit patterns,100000 random float32-to-half conversions, unsupported/corrupt policy/metadata/data, manifest/payload/owner/path binding, independent full SwiGLU reference, duplicate/skipped/zero-weight routes, output modes, transactional failures, and four simultaneous jobs on a real Team. Down slabs are protected with `PROT_NONE` until their gate opens. An ASan/UBSan standalone stress executable runs the same real kernel and scheduler.

Bounded timings print warm minimums for two synthetic token rows, eight unpinned workers and one real expert, including construction/destruction of the Team. These are component measurements under a shared host load, not serving throughput, model quality or whole-model readiness.

Verified on divix01 (two Xeon Platinum 8280 CPUs), 2026-10-25: 74 feature cases passed with GCC 15.2 numerical builds and Clang 21 ASan/UBSan; the existing optimized JobWindow regression also passed. The real experts' maximum relative L2 error against the independently decoded FP16-boundary reference was `9.20486e-5` (test bound `.002`). AVX512 warm minima were 3.34/3.54/3.61 ms for experts `(1,0)`/`(32,17)`/`(69,0)` respectively, for two rows and eight workers. A separate compile with AVX512DQ explicitly disabled passed. This is bounded CPU component evidence, not whole-model quality or FFI/serving qualification. The isolated remote test checkout is `/mnt/nvme4/mimo-reap/gptq-int3-cpu-dev-v1`; neither environment nor immutable checkpoint was modified.

## Upstream provenance

`bestla_int3.h` adapts Intel BesTLA `decompress_s3_s8` / `unpack_{1,2}bits` from [intel/neural-speed](https://github.com/intel/neural-speed/tree/bfd5d3c17dee18a20e2768f855f2d8fe132fc579/bestla/bestla), commit `bfd5d3c17dee18a20e2768f855f2d8fe132fc579`, specifically `kernel_ref.h`, `kernel_avx2.h` and `kernel_avx512f.h`. Copyright(c)2023 Intel Corporation; Apache2.0 terms are copied in `LICENSE.bestla`.

Adaptations use byte pointers and unaligned-safe scalar loads, fixed128-element groups with32/64-element SIMD blocks, existing engine ISA guards, and an AVX512F broadcast equivalent that avoids requiring AVX512DQ. Upstream dispatch, tails, threading, activation quantization and GEMV scaling are not imported. In particular, applying the group scale after accumulation would change FP16 QDQ rounding; this backend rounds each weight before the dot instead.
