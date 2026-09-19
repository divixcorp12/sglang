"""Task 9: DSpark must accept a quantized (EXL3) target `lm_head`.

`DSparkWorkerV2.__init__` used to reject any target `lm_head` without a dense
`.weight`, which is wrong for a quantized head whose quant method can apply
without one (e.g. EXL3, `Exl3LinearMethod.applies_without_weight = True`).
This tests the extracted guard function
`check_dspark_shared_lm_head_usable` directly, per the same pattern as
test_dspark_residency_commit.py (`DSparkWorkerV2` cannot be constructed on
CPU).

Also covers the fp32 LM-head path
(`DeepseekV4ForCausalLMDSpark._logits_from_x_post_hc`, gated by
`SGLANG_DSPARK_FP32_LM_HEAD`): it must raise a clear `ValueError` naming the
env var and the quantized head instead of an `AttributeError` when the
shared head has no dense `.weight`.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    check_dspark_shared_lm_head_usable,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DenseLmHead:
    """A plain unquantized lm_head: has `.weight`, no quant_method."""

    def __init__(self):
        self.weight = torch.zeros(4, 4)


class _AppliesWithoutWeightQuantMethod:
    """Stands in for Exl3LinearMethod: applies without a dense weight."""

    applies_without_weight = True

    def apply(self, layer, x, bias):
        return x


class _NeedsWeightQuantMethod:
    """A quant method that does need a dense weight (not applies_without_weight)."""

    applies_without_weight = False

    def apply(self, layer, x, bias):
        return x


class _QuantizedLmHeadNoWeight:
    """Mirrors an EXL3 lm_head: no `.weight`, only quantized params."""

    def __init__(self, quant_method):
        self.quant_method = quant_method
        self.trellis = torch.zeros(1)
        self.suh = torch.zeros(1)
        self.svh = torch.zeros(1)
        self.mul1 = torch.zeros(1)


class CheckDSparkSharedLmHeadUsableTest(CustomTestCase):
    def test_accepts_head_with_dense_weight(self):
        # Must not raise.
        check_dspark_shared_lm_head_usable(_DenseLmHead())

    def test_accepts_weightless_head_whose_quant_method_applies_without_weight(self):
        lm_head = _QuantizedLmHeadNoWeight(_AppliesWithoutWeightQuantMethod())
        self.assertFalse(hasattr(lm_head, "weight"))
        # Must not raise: EXL3-shaped head is usable via applies_without_weight.
        check_dspark_shared_lm_head_usable(lm_head)

    def test_rejects_none(self):
        with self.assertRaisesRegex(RuntimeError, "no `lm_head`"):
            check_dspark_shared_lm_head_usable(None)

    def test_rejects_weightless_head_whose_quant_method_needs_weight(self):
        lm_head = _QuantizedLmHeadNoWeight(_NeedsWeightQuantMethod())
        with self.assertRaisesRegex(RuntimeError, "neither a `weight`"):
            check_dspark_shared_lm_head_usable(lm_head)

    def test_rejects_weightless_head_with_no_quant_method_at_all(self):
        lm_head = _QuantizedLmHeadNoWeight(None)
        with self.assertRaisesRegex(RuntimeError, "neither a `weight`"):
            check_dspark_shared_lm_head_usable(lm_head)


def _make_dspark_model_for_logits(lm_head):
    """A minimal DeepseekV4ForCausalLMDSpark-shaped `self` for
    `_logits_from_x_post_hc`, avoiding a real model construction (needs a GPU
    / full config)."""
    model = object.__new__(DeepseekV4ForCausalLMDSpark)
    model.lm_head = lm_head
    model.stages = [SimpleNamespace(norm=lambda x: x)]
    model._opt_markov_w2_tp_shard = True
    return model


class DSparkFp32LmHeadRefusalTest(CustomTestCase):
    def test_fp32_lm_head_raises_value_error_naming_env_var_and_head(self):
        lm_head = _QuantizedLmHeadNoWeight(_AppliesWithoutWeightQuantMethod())
        model = _make_dspark_model_for_logits(lm_head)
        model._use_fp32_lm_head = True

        with envs.SGLANG_DSPARK_FP32_LM_HEAD.override(True):
            with self.assertRaisesRegex(
                ValueError,
                "SGLANG_DSPARK_FP32_LM_HEAD.*quantized",
            ):
                model._logits_from_x_post_hc(torch.zeros(2, 4))

    def test_fp32_lm_head_off_does_not_touch_missing_weight(self):
        # With the flag off, the quantized-head apply path must be used and
        # must not raise AttributeError from an unconditional `.weight` read.
        lm_head = _QuantizedLmHeadNoWeight(_AppliesWithoutWeightQuantMethod())
        model = _make_dspark_model_for_logits(lm_head)
        model._use_fp32_lm_head = False

        x = torch.zeros(2, 4)
        # project_through_lm_head will call quant_method.apply(lm_head, x, None),
        # which the stand-in just returns x for; must not raise.
        out = model._logits_from_x_post_hc(x)
        self.assertTrue(torch.equal(out, x))


if __name__ == "__main__":
    unittest.main()
