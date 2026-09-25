"""Which EXL3 calls take the SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION path.

The fused path computes one row; handing it anything else would compute garbage silently, so the predicate's
negative branches are the contract here. Bit parity of the fused path itself is the GPU suite's job
(test/manual/dsv41/test_dsv41_cast_fusion_gpu.py).
"""

import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import exl3
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod, exl3_cast_fusion_mlp
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
IN, OUT = 32, 16


def _method(fusion: bool) -> Exl3LinearMethod:
    with envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.override(fusion):
        return Exl3LinearMethod(Exl3Config.from_config(CFG))


def _layer(parts: int = 2) -> nn.Module:
    layer = nn.Module()
    layer.exl3_in = IN
    layer.exl3_tensors = [object()] * parts
    return layer


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fused(x16, parts):
        seen.append("fused")
        return torch.zeros(1, OUT * len(parts), dtype=torch.float16)

    def unfused(x, t):
        seen.append("unfused")
        return torch.zeros(*x.shape[:-1], OUT, dtype=x.dtype)

    monkeypatch.setattr(exl3, "exl3_gemm_bs1", fused)
    monkeypatch.setattr(exl3, "exl3_half_input", lambda x: x.reshape(1, -1).to(torch.float16))
    monkeypatch.setattr(exl3, "exl3_linear", unfused)
    return seen


def test_a_bf16_row_takes_the_fused_path_with_the_merged_width(calls):
    y = _method(True).apply(_layer(parts=2), torch.zeros(1, IN, dtype=torch.bfloat16))
    assert calls == ["fused"]
    assert y.shape == (1, 2 * OUT) and y.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "fusion, shape, dtype, bias",
    [
        (False, (1, IN), torch.bfloat16, False),
        (True, (2, IN), torch.bfloat16, False),
        (True, (1, IN), torch.float32, False),
        (True, (1, IN), torch.float16, False),
        (True, (1, IN), torch.bfloat16, True),
    ],
    ids=["flag-off", "two-rows", "fp32", "fp16", "bias"],
)
def test_everything_else_stays_on_the_unfused_path(calls, fusion, shape, dtype, bias):
    x = torch.zeros(*shape, dtype=dtype)
    _method(fusion).apply(_layer(parts=1), x, torch.zeros(OUT, dtype=dtype) if bias else None)
    assert calls == ["unfused"]


def test_the_mlp_is_fused_only_when_both_linears_are_exl3_with_the_flag():
    def linear(method):
        module = nn.Module()
        module.quant_method = method
        return module

    on, off, other = linear(_method(True)), linear(_method(False)), linear(UnquantizedLinearMethod())
    assert exl3_cast_fusion_mlp(on, on)
    assert not exl3_cast_fusion_mlp(on, off)
    assert not exl3_cast_fusion_mlp(off, on)
    assert not exl3_cast_fusion_mlp(on, other)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
