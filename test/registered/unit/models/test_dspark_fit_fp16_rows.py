"""fit_fp16_rows: the DSpark V4 draft's target-hidden projection survives an fp16-activation main_proj (no kernels).

A BOS token's target hidden reaches ~3e5; an EXL3 main_proj casts it to fp16 (inf), and main_norm turned the row NaN,
which poisoned that position's draft KV. The fake main_proj below casts its input to fp16 the same way.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark, fit_fp16_rows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BOS_PEAK = 315392.0  # the BOS row's largest entry in the 2026-10-17 collection (dspark-nan-20261017)


def _rms_norm(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)


def _draft(weight: torch.Tensor):
    """A fake unbound `self` for project_target_hidden: an fp16-activation main_proj (as EXL3's) and an RMSNorm."""

    def main_proj(x):
        return (x.to(torch.float16).float() @ weight).to(x.dtype), None

    return SimpleNamespace(stages=[SimpleNamespace(main_proj=main_proj, main_norm=_rms_norm)])


def _rows(dtype):
    torch.manual_seed(0)
    x = (torch.randn(4, 256) * 20).to(dtype)
    x[1, 7] = 65504.0  # the largest fp16 value: in range, untouched
    x[2, 3] = _BOS_PEAK
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_in_range_rows_unchanged(dtype):
    x = _rows(dtype)
    out = fit_fp16_rows(x)
    assert out.dtype == dtype
    for row in (0, 1, 3):
        assert torch.equal(out[row], x[row])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_overflowing_row_scaled_by_power_of_two(dtype):
    x = _rows(dtype)
    out = fit_fp16_rows(x)
    assert torch.isfinite(out[2].to(torch.float16)).all()
    ratio = (x[2].float() / out[2].float()).unique()
    assert ratio.numel() == 1 and float(torch.log2(ratio)).is_integer()
    assert 128.0 < float(out[2].abs().max()) <= 256.0


def test_non_finite_rows_stay_non_finite():
    x = torch.ones(2, 8)
    x[0, 0] = float("nan")
    out = fit_fp16_rows(x)
    assert torch.isnan(out[0, 0]) and torch.equal(out[1], x[1])


def test_projection_finite_and_scale_invariant():
    torch.manual_seed(1)
    weight = torch.randn(256, 64) / 16
    x = _rows(torch.bfloat16)
    draft = _draft(weight)
    unfitted = _rms_norm(draft.stages[0].main_proj(x)[0])
    assert not torch.isfinite(unfitted[2]).all()  # the bug: the BOS row turns NaN
    out = DeepseekV4ForCausalLMDSpark.project_target_hidden(draft, x)
    assert torch.isfinite(out).all()
    reference = _rms_norm(x.float() @ weight)  # the same projection in fp32, no fp16 cast
    assert torch.allclose(out[2], reference[2], rtol=0, atol=2e-2)
    for row in (0, 1, 3):
        assert torch.equal(out[row], unfitted[row])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
