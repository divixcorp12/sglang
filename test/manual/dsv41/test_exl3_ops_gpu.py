"""EXL3 kernels against exllamav3's reconstruct + had_r_128 composition (synthetic weights)."""

import os

import pytest
import torch

from sglang.srt.layers.quantization.exl3_ops import (
    exl3_dense_weight,
    exl3_linear,
    exl3_linear_reference,
    random_exl3_tensors,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)

# (in, out, bits): wq_a, routed w1, routed w2, wo_a slice, indexer wq_b, Engram wkv
SHAPES = [(5120, 1280, 5), (5120, 2304, 3), (2304, 5120, 3), (4096, 1024, 5), (1280, 4096, 5), (6144, 25600, 5)]


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("in_f,out_f,bits", SHAPES)
@pytest.mark.parametrize("rows", [1, 8, 144])
def test_gemm_matches_reference(in_f, out_f, bits, rows):
    t = random_exl3_tensors(in_f, out_f, bits, device="cuda", seed=in_f + out_f + bits)
    x = torch.randn(rows, in_f, device="cuda", dtype=torch.bfloat16)
    assert _rel(exl3_linear(x, t, torch.float32), exl3_linear_reference(x, t)) < 5e-3


@pytest.mark.parametrize("in_f,out_f,bits", SHAPES)
def test_dense_path_matches_reference(in_f, out_f, bits):
    t = random_exl3_tensors(in_f, out_f, bits, device="cuda", seed=7 * in_f + bits)
    x = torch.randn(512, in_f, device="cuda", dtype=torch.bfloat16)
    ref = exl3_linear_reference(x, t)
    assert _rel(exl3_linear(x, t, torch.float32), ref) < 5e-3
    assert _rel(x.half().float() @ exl3_dense_weight(t).float(), ref) < 5e-3


def test_head_shape_slices():
    t = random_exl3_tensors(5120, 129280, 6, device="cuda", seed=6)
    x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
    ref = exl3_linear_reference(x, t)
    assert _rel(x.half().float() @ exl3_dense_weight(t).float(), ref) < 5e-3
    assert _rel(exl3_linear(x, t, torch.float32), ref) < 5e-3


def test_bf16_output_and_leading_dims():
    t = random_exl3_tensors(5120, 1280, 5, device="cuda", seed=1)
    x = torch.randn(2, 3, 5120, device="cuda", dtype=torch.bfloat16)
    y = exl3_linear(x, t)
    assert y.dtype == torch.bfloat16 and y.shape == (2, 3, 1280)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
