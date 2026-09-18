"""Exl3LinearMethod.apply on real module types, against the kernel oracle."""

import os

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod
from sglang.srt.layers.quantization.exl3_ops import exl3_linear_reference, random_exl3_tensors

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)
CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}


def _load(layer, t, shard=None):
    for name in ("trellis", "suh", "svh"):
        args = (getattr(layer, name), getattr(t, name)) + (() if shard is None else (shard,))
        getattr(layer, name).weight_loader(*args)
    mul1 = torch.tensor(1, dtype=torch.int32, device="cuda")
    getattr(layer, "mul1").weight_loader(*((layer.mul1, mul1) + (() if shard is None else (shard,))))


@pytest.mark.parametrize("rows", [1, 300])
def test_merged_gate_up(rows):
    torch.set_default_device("cuda")
    layer, method = nn.Module(), Exl3LinearMethod(Exl3Config.from_config(CFG))
    method.create_weights(layer, 5120, [2304, 2304], 5120, 4608, torch.bfloat16)
    g = random_exl3_tensors(5120, 2304, 5, device="cuda", seed=1)
    u = random_exl3_tensors(5120, 2304, 5, device="cuda", seed=2)
    _load(layer, g, 0)
    _load(layer, u, 1)
    method.process_weights_after_loading(layer)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    y = method.apply(layer, x).float()
    ref = torch.cat([exl3_linear_reference(x, g), exl3_linear_reference(x, u)], dim=-1)
    assert float((y - ref).norm() / ref.norm()) < 1e-2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
