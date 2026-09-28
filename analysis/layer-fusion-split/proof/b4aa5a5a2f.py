"""Proof: the python-move commit is a pure relocation (Repro, mechanical-refactor-verify)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanical_refactor_reproduction_utils import Repro

BASE = "b54253cd01f7e2d209c7066c5b9f37405bacfa89"
TARGET = "b4aa5a5a2f25e8d5b68d968ccbe18fe2798271f8"
OLD = "python/sglang/kernels/ops/moe/dsv41_layer_fusion.py"
RES = "python/sglang/kernels/ops/moe/expert_residency_direct_gather.py"
EXL3 = "python/sglang/kernels/ops/moe/exl3_route_tables.py"
OLD_MOD = "sglang.kernels.ops.moe.dsv41_layer_fusion"
RES_MOD = "sglang.kernels.ops.moe.expert_residency_direct_gather"
EXL3_MOD = "sglang.kernels.ops.moe.exl3_route_tables"
CHECKS = "test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py"

EXL3_HEADER = '''"""JIT wrapper for the EXL3 fused MoE's route tables (SGLANG_DSV41_ENABLE_LAYER_FUSION).

``exl3_moe_route_tables`` launches one kernel in place of ``exl3_fused_moe.route_tables`` and the copies around it in
``Exl3FusedMoE.run`` (22 kernels per layer), with bit-identical results; the flag-off path runs the torch chain, and
the parity test compares against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
'''

RES_HEADER = '''"""JIT wrappers for the DIRECT residency kernels of ``GpuResidencyUpdater`` (SGLANG_DSV41_ENABLE_LAYER_FUSION).

Each wrapper launches one kernel in place of a per-layer chain of small torch ops, with bit-identical results:

* ``direct_gather_destinations`` -- ``GpuResidencyUpdater.gather_destinations`` (26 kernels per layer), with the
  route-slot lookup ``expert_to_slot.index_select(0, flat.long())`` folded in;
* ``direct_commit_gather`` -- ``GpuResidencyUpdater.commit_gather`` (41 kernels per layer).

The torch chains stay the reference: the flag-off path runs them, and the parity tests compare against them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ID_DTYPES = (torch.int32, torch.int64)
'''

EXL3_SYMBOLS = ["_route_tables_module", "exl3_moe_route_tables"]
RES_SYMBOLS = [
    "_gather_module",
    "_commit_module",
    "direct_gather_destinations",
    "direct_commit_gather",
]

repro = (
    Repro(BASE, TARGET)
    # _ID_DTYPES survives in the source here (re-derived copy); _FLOAT_DTYPES leaves with the route tables.
    .extract_symbols_to_new_module(
        OLD,
        EXL3,
        symbols=EXL3_SYMBOLS,
        header=EXL3_HEADER,
        order=EXL3_SYMBOLS,
        drop_assigns=["_FLOAT_DTYPES"],
    )
    .extract_symbols_to_new_module(
        OLD,
        RES,
        symbols=RES_SYMBOLS,
        header=RES_HEADER,
        order=RES_SYMBOLS,
        drop_assigns=["_ID_DTYPES"],
    )
    .delete_file(OLD)
    .repath_import(
        "python/sglang/srt/layers/moe/expert_residency_gpu.py",
        old_module=OLD_MOD,
        new_module=RES_MOD,
        name="direct_gather_destinations",
    )
    .repath_import(
        "python/sglang/srt/layers/moe/expert_residency_gpu.py",
        old_module=OLD_MOD,
        new_module=RES_MOD,
        name="direct_commit_gather",
    )
    .repath_import(
        "python/sglang/srt/layers/quantization/exl3_fused_moe.py",
        old_module=OLD_MOD,
        new_module=EXL3_MOD,
        name="exl3_moe_route_tables",
    )
    .repath_import(
        "test/manual/dsv41/test_dsv41_layer_fusion_gpu.py",
        old_module=OLD_MOD,
        new_module=EXL3_MOD,
        name="exl3_moe_route_tables",
    )
    .repath_import(
        CHECKS, old_module=OLD_MOD, new_module=RES_MOD, name="_gather_module"
    )
    .repath_import(
        CHECKS, old_module=OLD_MOD, new_module=RES_MOD, name="_commit_module"
    )
    .repath_import(
        CHECKS, old_module=OLD_MOD, new_module=EXL3_MOD, name="_route_tables_module"
    )
)
sys.exit(1 if repro.run() else 0)
