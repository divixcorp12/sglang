"""_extra_kv_to_64 must only route splittable page sizes to _split_kv_pages_to_64.

Regression for the DSV4 c128 extra pool (pbs=2 on sm_120), which
_split_kv_pages_to_64 asserts on (it only accepts src_pbs % 64 == 0 and
src_pbs >= 64). See flash_mla_sm120.py:_extra_kv_to_64.
"""

import torch

from sglang.kernels.ops.attention import flash_mla_sm120 as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _extra(pbs: int) -> torch.Tensor:
    # Shape/content do not matter for this routing test; only shape[1] (pbs) is read.
    return torch.zeros(4, pbs, 8, dtype=torch.uint8)


def test_pbs2_extra_pool_returned_unsplit():
    extra = _extra(2)
    out = mod._extra_kv_to_64(extra, extra_idx=None)
    assert out is extra


def test_pbs64_extra_pool_returned_unchanged():
    extra = _extra(64)
    out = mod._extra_kv_to_64(extra, extra_idx=None)
    assert out is extra


def test_pbs128_extra_pool_is_split(monkeypatch):
    calls = []

    def fake_split(kv_u8, src_pbs, touched_indices=None, tag=""):
        calls.append((kv_u8, src_pbs, touched_indices, tag))
        return "split-result"

    monkeypatch.setattr(mod, "_split_kv_pages_to_64", fake_split)

    extra = _extra(128)
    out = mod._extra_kv_to_64(extra, extra_idx=None)

    assert out == "split-result"
    assert len(calls) == 1
    kv_u8, src_pbs, touched_indices, tag = calls[0]
    assert kv_u8 is extra
    assert src_pbs == 128
    assert tag == ":extra"
