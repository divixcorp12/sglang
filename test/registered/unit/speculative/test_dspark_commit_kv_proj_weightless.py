"""Regression coverage for Task 10: an EXL3-quantized DSpark draft linear (or
shared lm_head) has no dense `.weight` (`Exl3LinearMethod.create_weights`
registers `trellis`/`suh`/`svh`/`mul1` instead). The fused commit-kv-proj
capability checks and `_base_logits_dtype` used to read `linear.weight`
unguarded and raised `AttributeError` for such a linear; they must instead
recognize "no dense weight" as "the fused/dense path does not apply" so
`CommitKvProj.execute` takes the per-linear torch path
(`commit_kv_proj`, `linear(main_x)[0]`), which EXL3 already serves correctly
through its quant method's module `__call__`.

Fail-first: reverting the guards in dspark_draft_model.py /
dspark_draft_sampler.py makes `test_weightless_*` below raise
`AttributeError: ... object has no attribute 'weight'` instead of passing.
"""

import types
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark import dspark_draft_model
from sglang.srt.speculative.dspark_components import dspark_draft_sampler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DenseLinear(torch.nn.Module):
    """Stand-in for an unquantized (bf16/fp8) wkv linear: has `.weight`."""

    quant_method = None

    def __init__(self, weight, weight_scale_inv=None):
        super().__init__()
        self.weight = weight
        if weight_scale_inv is not None:
            self.weight_scale_inv = weight_scale_inv

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype)), None


class _WeightlessLinear(torch.nn.Module):
    """Stand-in for an EXL3-quantized ReplicatedLinear: exposes a
    `quant_method` (`applies_without_weight = True`, matching
    `Exl3LinearMethod`) and registers no dense `.weight` at all, per
    `Exl3LinearMethod.create_weights` (trellis/suh/svh/mul1 only)."""

    def __init__(self, out_features: int):
        super().__init__()
        self.quant_method = SimpleNamespace(applies_without_weight=True)
        self._out_features = out_features

    def forward(self, x):
        # Deterministic, weight-independent output so the test can check
        # CommitKvProj.execute actually returned *this* module's output
        # (proving it took the per-linear path) rather than some fused
        # result that happened to also avoid raising.
        bs = x.shape[0]
        return (
            torch.full((bs, self._out_features), 7.0, dtype=x.dtype),
            None,
        )


def _block_quant_method():
    quant_config = SimpleNamespace(weight_block_size=[128, 128])
    return SimpleNamespace(
        block_quant=True,
        w8a8_block_fp8_linear=lambda **kw: None,
        quant_config=quant_config,
    )


class TestDequantSupported(CustomTestCase):
    """_dequant_supported: unchanged dense/fp8 behaviour, new weightless case."""

    def test_dense_bf16_is_supported(self):
        linear = _DenseLinear(torch.randn(4, 8, dtype=torch.bfloat16))
        self.assertTrue(dspark_draft_model._dequant_supported(linear))

    def test_fp8_with_matching_block_scale_is_supported(self):
        out_dim, in_dim, block = 256, 384, 128
        weight = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
        scale = torch.rand(2, 3)
        linear = _DenseLinear(weight, weight_scale_inv=scale)
        self.assertTrue(dspark_draft_model._dequant_supported(linear))

    def test_fp8_with_mismatched_block_scale_is_not_supported(self):
        out_dim, in_dim = 256, 384
        weight = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
        scale = torch.rand(1, 1)  # wrong grid for a 128x128 block size
        linear = _DenseLinear(weight, weight_scale_inv=scale)
        self.assertFalse(dspark_draft_model._dequant_supported(linear))

    def test_weightless_linear_is_not_supported(self):
        """The bug: this used to raise AttributeError on `linear.weight`."""
        linear = _WeightlessLinear(out_features=8)
        self.assertFalse(dspark_draft_model._dequant_supported(linear))


class TestBlockQuantStackApplies(CustomTestCase):
    def test_dense_block_fp8_stack_applies(self):
        out_dim, in_dim = 256, 384
        weight = torch.randn(out_dim, in_dim).to(torch.float8_e4m3fn)
        linear = _DenseLinear(weight)
        linear.quant_method = _block_quant_method()
        self.assertTrue(
            dspark_draft_model._block_quant_stack_applies(wkv_linears=[linear])
        )

    def test_weightless_linear_does_not_apply(self):
        """Even a quant_method claiming block_quant=True must defer to the
        weightless case rather than raising on `linear.weight.dtype`."""
        linear = _WeightlessLinear(out_features=8)
        linear.quant_method = _block_quant_method()
        self.assertFalse(
            dspark_draft_model._block_quant_stack_applies(wkv_linears=[linear])
        )


class TestFusedCommitKvProjSupported(CustomTestCase):
    def test_dense_bf16_linears_are_supported(self):
        linears = [
            _DenseLinear(torch.randn(4, 8, dtype=torch.bfloat16)) for _ in range(3)
        ]
        self.assertTrue(
            dspark_draft_model._fused_commit_kv_proj_supported(wkv_linears=linears)
        )

    def test_weightless_linears_are_not_supported(self):
        """Root-cause regression test: with EXL3 wkv linears,
        _fused_commit_kv_proj_supported must return False (routing
        CommitKvProj.execute to the torch path) instead of raising
        AttributeError('ReplicatedLinear' object has no attribute 'weight')."""
        linears = [_WeightlessLinear(out_features=8) for _ in range(3)]
        self.assertFalse(
            dspark_draft_model._fused_commit_kv_proj_supported(wkv_linears=linears)
        )

    def test_mixed_dense_and_weightless_are_not_supported(self):
        linears = [
            _DenseLinear(torch.randn(4, 8, dtype=torch.bfloat16)),
            _WeightlessLinear(out_features=8),
        ]
        self.assertFalse(
            dspark_draft_model._fused_commit_kv_proj_supported(wkv_linears=linears)
        )


class TestCommitKvProjExecuteRoutesToTorchPath(CustomTestCase):
    def test_weightless_linears_route_to_module_call_and_return_their_output(self):
        """execute() takes `main_x.is_cuda and _fused_commit_kv_proj_supported(...)`
        to pick the triton/fused path; on CPU (this test's only option without
        a GPU) `main_x.is_cuda` is already False, so this also exercises the
        torch path end-to-end. `test_weightless_linears_are_not_supported`
        above independently proves the capability check itself returns False
        rather than raising, which is what keeps this same routing correct
        once `main_x.is_cuda` is True on real hardware."""
        linears = [_WeightlessLinear(out_features=8) for _ in range(3)]
        main_x = torch.randn(5, 16)
        out = dspark_draft_model.CommitKvProj.execute(
            main_x=main_x, wkv_linears=linears
        )
        self.assertEqual(len(out), 3)
        for kv in out:
            self.assertEqual(tuple(kv.shape), (5, 8))
            self.assertTrue(torch.equal(kv, torch.full((5, 8), 7.0)))

    def test_commit_kv_proj_torch_path_direct(self):
        """Same as above but calling the exact function
        `_fused_commit_kv_proj_supported` gates around (`commit_kv_proj`),
        matching the docstring: "unsupported quant schemes fall back to the
        per-linear torch path in execute()"."""
        linears = [_WeightlessLinear(out_features=8) for _ in range(2)]
        main_x = torch.randn(3, 16)
        out = dspark_draft_model.commit_kv_proj(main_x=main_x, wkv_linears=linears)
        self.assertEqual(len(out), 2)
        for kv in out:
            self.assertTrue(torch.equal(kv, torch.full((3, 8), 7.0)))


class TestBaseLogitsDtype(CustomTestCase):
    def _model(self, lm_head, markov_dtype=torch.float32):
        markov_head = torch.nn.Linear(4, 4, dtype=markov_dtype)
        return SimpleNamespace(lm_head=lm_head, markov_head=markov_head)

    def test_dense_floating_weight_uses_its_own_dtype(self):
        lm_head = SimpleNamespace(weight=torch.randn(4, 4, dtype=torch.bfloat16))
        model = self._model(lm_head, markov_dtype=torch.float32)
        self.assertEqual(
            dspark_draft_sampler._base_logits_dtype(model), torch.bfloat16
        )

    def test_packed_non_floating_weight_falls_back_to_markov_head(self):
        """Pre-existing behaviour: a packed (e.g. int8) `weight` carries no
        logits dtype, so the kernel's activation dtype (markov_head) wins."""
        lm_head = SimpleNamespace(weight=torch.zeros(4, 4, dtype=torch.int8))
        model = self._model(lm_head, markov_dtype=torch.float16)
        self.assertEqual(dspark_draft_sampler._base_logits_dtype(model), torch.float16)

    def test_weightless_head_falls_back_to_markov_head(self):
        """The bug: EXL3's shared lm_head has no `.weight` at all
        (applies_without_weight=True), so `model.lm_head.weight` used to
        raise AttributeError instead of falling through to markov_head."""
        lm_head = SimpleNamespace(quant_method=SimpleNamespace(applies_without_weight=True))
        model = self._model(lm_head, markov_dtype=torch.bfloat16)
        self.assertEqual(
            dspark_draft_sampler._base_logits_dtype(model), torch.bfloat16
        )


if __name__ == "__main__":
    unittest.main()
