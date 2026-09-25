# DSV41 decode: fusion round 2, the EXL3 cast glue

Goal: remove small kernels from the BS1 EXL3 decode graph (`DSV41_REFERENCE.md` §27.3, §27.4 item 6) with fused
kernels whose outputs are bit-identical to today's. Branch `dsv41-decode-fusion-2` from `origin/master` at
`b7478bd35e`. Flag: `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION` (default off). Evidence lives under
`divix01:/data/models/slang/nvfp4-work/decode-fusion-2/`.

## 1. The catalog

Source: the node-mode trace `pcie-node-20260925-170510` (master `69df716412`, copy engine and layer fusion on),
100 decode steps with ≥ 2,000 kernels each. Every step has **2,756 kernels**. One median step was listed kernel by
kernel and every kernel assigned to its source by name and neighbour (scratchpad `kmix.py`, `groups.py`; the
steady-state layer is 66 kernels). Node mode inflates tiny kernels and their gaps (CLAUDE.md), so the µs column only
ranks the groups.

| Group (per step) | Kernels | < 3 µs | GPU µs (node) | Source |
|---|---:|---:|---:|---|
| attention: q_norm, rope, k-norm-rope, page mark/split, sparse MLA, wo_a, compress/index | 537 | 376 | 2,943 | `deepseek_v4.py` `_forward_prepare`, flashinfer, `dsv4` JIT |
| RAM-miss chain: post, hit-wait, S, CW, stage acks, finalize, go_total adds | 400 | 247 | 92,468 | `exl3_ram_miss.py` (waits dominate) |
| mHC: stats partial, sinkhorn, post split, combine+norm, x2 | 320 | 160 | 895 | `mhc.py`, `hc_combine_norm.py` |
| **cast bf16 -> fp16 before an EXL3 gemv** | **283** | **283** | 457 | `exl3_ops.exl3_linear` `x.to(fp16)` |
| EXL3 gemv (7 dense linears per layer) | 283 | 0 | 4,193 | exllamav3 `exl3_gemm` |
| **cast fp16 -> bf16 after an EXL3 gemv** | **283** | **283** | 472 | `exl3_ops.exl3_linear` `y.to(out_dtype)` |
| residency bookkeeping (fused in round 1) | 160 | 120 | 430 | `dsv41_layer_fusion.cuh` |
| step head, layer-0 routing glue, indexer glue | 123 | 120 | 119 | various |
| **MoE combine: `out.to(bf16)`, `* 1.5`, `+= shared`** | **120** | **120** | 75 | `exl3.py` `_apply_graph`/`apply`, `deepseek_v2.py` |
| router: tiny gemm, Triton top-k | 80 | 40 | 230 | |
| routed MoE: `exl3_moe_kernel`, gather | 80 | 0 | 4,117 | |
| **shared expert: `cat`, `silu_mul_clamp`** | **80** | **80** | 57 | `exl3.py` `apply`, `deepseek_v2.py` `DeepseekV2MLP` |
| Engram | 7 | 2 | 30 | |

Per steady-state layer, the seven EXL3 linears are wq_a, wq_b, wkv, wo_b and the shared expert's gate, up and down.
Each runs as `cast -> gemv -> cast` (`exl3_linear`), and the merged gate_up adds a `cat`:

```
attention: cast wq_a cast | q_norm | cast wq_b cast | rope | cast wkv cast | k-norm-rope ... wo_a | cast wo_b cast
shared:    cast gate cast | cast up cast | cat | silu_mul_clamp | cast down cast
MoE tail:  exl3_moe_gather | to(bf16) | mul 1.5 | add shared
```

Three of those casts convert the same bf16 tensor to fp16 again (wq_a and wkv read one input, gate and up read
one input), and the shared expert's intermediate makes a round trip fp16 -> bf16 -> fp16 around `silu_mul_clamp`.

## 2. Candidates, ranked

A removed tiny kernel is worth ~0.93 µs in graph mode: round 1 removed 3,440 kernels per step for 3.2 ms/token.

| # | Fusion | Kernels / layer | Risk | Bit-exact? | Decision |
|---|---|---:|---|---|---|
| F1 | gate_up: one input cast, both gemvs write one fp16 buffer, one output cast, no `cat` | -3 | low: inside `Exl3LinearMethod.apply` | yes: casts are elementwise, `cat` is a copy | **build** |
| F2 | fp16 copy of the sublayer input written by `hc_combine_norm`, used by wq_a, wkv and gate_up | -3 | medium: producer and consumers are in different modules | yes: the same RN conversion of the same bf16 values | **build** |
| F3 | shared expert: `silu_mul_clamp` reads the fp16 gate/up and writes the fp16 down input | -2 (-6 with F1, F2) | low: reuses the kernel's own `silu_and_mul` | yes: bf16 rounding kept between each step | **build** |
| F4 | MoE combine: `to(bf16)` and `* routed_scaling_factor` in one kernel | -1 | low | yes | **build** |
| F5 | wq_b's output cast folded into the q rope | -1 | medium: a second rope kernel | yes if the rope math is copied | not built (below) |
| F6 | wkv's output cast folded into `fused_k_norm_rope_flashmla` | -1 | medium: kv-pool write path | yes | not built (below) |
| F7 | wo_b's output cast folded into the mHC post | -1 | medium: two Triton readers | yes | not built (below) |
| -- | wq_a out cast + q_norm + wq_b in cast | -2 | high | no: flashinfer's RMSNorm reduction order cannot be reproduced | rejected |
| -- | mHC stats partial + sinkhorn | -2 | high | no: reduction order changes (round 1, §5) | rejected |
| -- | RAM-miss acks and go_total adds | -4 | high: lease protocol, other lanes | yes | out of scope |

F1-F4 remove **9 kernels per layer, 360 per step** (2,756 -> 2,396). At ~0.93 µs each that is **~0.3-0.4 ms/token,
*estimate*** -- well under the "few ms" guessed in §27.4 item 6, because the remaining small kernels are mostly
attention and RAM-miss work that is not glue. F5-F7 would add 3 per layer for similar risk each and are left for a
later round.

## 3. Design

Everything is behind `SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION`, read once where each object is built, and only for BS1
rows (`x.shape[0] == 1`) on the gemv path. With the flag off no new code runs.

- **The fp16 sublayer input (F2).** `_hc_combine` runs `hc_combine_norm` with a second fp16 output and publishes
  `(y, y16)` in a one-slot registry (`exl3_ops.publish_half_input`). A consumer takes `y16` only when its input is
  the same tensor object, at the same `_version`, on the same stream; otherwise it casts. The slot holds a reference,
  so the object cannot be recycled, and the next publish replaces it.
- **EXL3 linears (F1, F2).** `Exl3LinearMethod.apply` takes the published fp16 input or casts once for all parts,
  writes every part into one `[1, parts * out]` fp16 buffer (a slice of one row is contiguous), and casts once.
- **Shared expert (F3).** `DeepseekV2MLP.forward` hands BS1 EXL3 MLPs with a swiglu limit to
  `exl3_swiglu_mlp`: gate_up into the fp16 buffer, `exl3_silu_mul_clamp_half` into the fp16 down input, down, one
  cast.
- **MoE combine (F4).** `Exl3MoEMethod` graph path: `exl3_scale_to_bf16(out_fp32, rsf)` replaces
  `out.to(bf16) * rsf`. The `+= shared` stays in `deepseek_v2.py`.
- **Kernels.** `python/sglang/kernels/jit/csrc/moe/dsv41_cast_fusion.cuh` (silu variant, which includes the existing
  `silu_and_mul` and compiles with its `-use_fast_math`; scale-to-bf16), wrappers in
  `python/sglang/kernels/ops/moe/dsv41_cast_fusion.py`; the fp16 output of `hc_combine_norm` is a constexpr branch of
  the existing Triton kernel.

## 4. Tests

- CPU: the flag and `Dsv41Config` field (`test_one_field_per_knob`); the registry's identity/version rules.
- GPU parity (`test/manual/dsv41/test_dsv41_cast_fusion_gpu.py`), all compared as bits against the unfused chain:
  each kernel against its torch chain; `Exl3LinearMethod.apply` flag on vs off for 1, 2, 3 parts; `exl3_swiglu_mlp`
  vs the unfused `DeepseekV2MLP` path; `hc_combine_norm` with and without the fp16 output; the fused chain captured
  in a CUDA graph with zero host nodes and replayed on new inputs.
- Integration flag on and off: the round-1 set (`test_exl3_ram_miss_graph_gpu.py`, `test_exl3_graph_apply_gpu.py`,
  `test_moe_side_stream_gpu.py`, `test_exl3_task5_item4_gpu.py`), the round-1 parity file, and
  `analysis/dsv41-drive/native-prefetch/gpu_suite.sh`.
- Mutants: one per kernel, run in the private worktree and reverted.

## 5. Measurement

- Node-mode traces, copy engine off, flag off and on, for kernels per step (counts only).
- Full arms on port 30021, A (flag off) then B (flag on), once each, under `rowimg-disk.lock`, as in
  `2026-09-25-dsv41-final-arms.md`. Report ms/token pooled and per session and byte-identity of the outputs.

## 6. Results

(filled in below as the work lands)
