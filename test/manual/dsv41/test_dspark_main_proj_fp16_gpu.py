"""The DSpark V4 draft's EXL3 main_proj on a BOS-like target hidden row (|x| ~3e5), with and without fit_fp16_rows.

Exl3LinearMethod casts its input to fp16, so a row past 65504 comes out NaN in every output column (the 2026-10-17
NaN burst, dspark-nan-20261017); fit_fp16_rows scales that row into range, and main_norm's scale invariance makes
the normalized result match a smaller power-of-two scaling of the same row.
"""

import os

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod
from sglang.srt.layers.quantization.exl3.ops import random_exl3_tensors
from sglang.srt.models.deepseek_v4_dspark import fit_fp16_rows

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)
CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
IN, OUT = 15360, 5120  # the V4.1 draft's main_proj: three target layers' hidden -> hidden


def _rms_norm(y):
    y = y.float()
    return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)


def _main_proj():
    layer, method = nn.Module(), Exl3LinearMethod(Exl3Config.from_config(CFG))
    method.create_weights(layer, IN, [OUT], IN, OUT, torch.bfloat16)
    t = random_exl3_tensors(IN, OUT, 4, device="cuda", seed=3)
    for name in ("trellis", "suh", "svh"):
        getattr(layer, name).weight_loader(getattr(layer, name), getattr(t, name))
    layer.mul1.weight_loader(layer.mul1, torch.tensor(1, dtype=torch.int32, device="cuda"))
    method.process_weights_after_loading(layer)
    return lambda x: method.apply(layer, x)


@pytest.mark.parametrize("rows", [1, 6])
def test_bos_row(rows):
    with torch.device("cuda"):
        proj = _main_proj()
        torch.manual_seed(0)
        x = (torch.randn(rows, IN) * 3).to(torch.bfloat16)
        x[0, 123] = 315392.0
        assert torch.isnan(proj(x)[0]).all()
        out = _rms_norm(proj(fit_fp16_rows(x)))
        assert torch.isfinite(out).all()
        smaller = x.clone()
        smaller[0] *= 2.0**-12
        assert torch.allclose(out[0], _rms_norm(proj(smaller))[0], rtol=0, atol=5e-2)
        if rows > 1:
            assert torch.equal(proj(fit_fp16_rows(x))[1:], proj(x)[1:])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
