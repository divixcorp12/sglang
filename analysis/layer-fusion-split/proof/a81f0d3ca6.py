"""Proof: the cast-python-move commit is a pure relocation (Repro, mechanical-refactor-verify)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanical_refactor_reproduction_utils import Repro

BASE = "501e536a6bfb6e188c47d6ec7172de708de52e60"
TARGET = "a81f0d3ca651ac16995934d6cd8991a6d72a1cca"
OLD = "python/sglang/kernels/ops/moe/dsv41_cast_fusion.py"
NEW = "python/sglang/kernels/ops/moe/exl3_cast_fusion.py"
OLD_MOD = "sglang.kernels.ops.moe.dsv41_cast_fusion"
NEW_MOD = "sglang.kernels.ops.moe.exl3_cast_fusion"
EXL3 = "python/sglang/srt/layers/quantization/exl3.py"
TEST = "test/manual/dsv41/test_dsv41_cast_fusion_gpu.py"

HEADER = '''"""JIT wrappers for the EXL3 decode cast fusion (SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION).

* ``exl3_silu_mul_clamp_half`` -- the shared expert's ``gate_up.to(bf16)``, ``silu_and_mul_clamp`` and the down
  projection's ``.to(fp16)`` as one kernel on exl3_gemm's fp16 output;
* ``exl3_scale_to_bf16`` -- the routed MoE output's ``out.to(bf16) * routed_scaling_factor``.

Both are bit-identical to the torch chains; the flag-off path runs those chains, and the parity tests compare against
them. The two kernels need opposite fast-math flags, and each loader carries its own: ``_silu_module`` builds with
``-use_fast_math``, ``_scale_module`` without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module
'''

SYMBOLS = [
    "_silu_module",
    "_scale_module",
    "exl3_silu_mul_clamp_half",
    "exl3_scale_to_bf16",
]

repro = (
    Repro(BASE, TARGET)
    .extract_symbols_to_new_module(
        OLD, NEW, symbols=SYMBOLS, header=HEADER, order=SYMBOLS
    )
    .delete_file(OLD)
    .repath_import(
        EXL3, old_module=OLD_MOD, new_module=NEW_MOD, name="exl3_silu_mul_clamp_half"
    )
    .repath_import(
        EXL3, old_module=OLD_MOD, new_module=NEW_MOD, name="exl3_scale_to_bf16"
    )
    .remove_import(TEST, "from sglang.kernels.ops.moe.dsv41_cast_fusion import")
    .add_import(
        TEST,
        "from sglang.kernels.ops.moe.exl3_cast_fusion import exl3_scale_to_bf16, exl3_silu_mul_clamp_half",
    )
)
sys.exit(1 if repro.run() else 0)
