"""The DSpark draft confidence head under SGLANG_RAGGED_VERIFY_MODE=static (SGLANG_DSPARK_CONFIDENCE_CUT)."""

from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4_dspark import (
    DeepseekV4ForCausalLMDSpark,
    build_dspark_v4_confidence_head,
)
from sglang.srt.models.dspark import build_confidence_head
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The DSV4.1 draft: text_config hidden_size 5120, dspark_markov_rank 256, and no confidence_head_with_markov key;
# its checkpoint stores mtp.2.confidence_head.proj.weight as F16 [1, 5376] with no bias.
_DSV41_HIDDEN, _DSV41_MARKOV_RANK = 5120, 256


def _build_v4_head(*, mode: str, cut: bool):
    with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode), envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(cut):
        return build_dspark_v4_confidence_head(
            config=SimpleNamespace(hidden_size=_DSV41_HIDDEN), markov_rank=_DSV41_MARKOV_RANK
        )


def test_static_builds_no_head_unless_the_cut_is_on():
    """Off, static stays head-less (production unchanged); on, both builders build the head under static."""
    assert _build_v4_head(mode="static", cut=False) is None
    assert _build_v4_head(mode="compact", cut=False) is not None
    dense_config = SimpleNamespace(hidden_size=64, markov_rank=16)
    with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(False):
            assert build_confidence_head(dense_config) is None
        with envs.SGLANG_DSPARK_CONFIDENCE_CUT.override(True):
            assert build_confidence_head(dense_config).with_markov


def test_static_cut_head_matches_the_dsv41_checkpoint():
    """The DSV4.1 config resolves with_markov on (eq. 7's Markov features), so the head's shape is the checkpoint's."""
    head = _build_v4_head(mode="static", cut=True)
    assert head.with_markov
    assert tuple(head.proj.weight.shape) == (1, _DSV41_HIDDEN + _DSV41_MARKOV_RANK)
    assert head.proj.bias is None


class _FakeDraftModel:
    """The attributes DeepseekV4ForCausalLMDSpark.load_weights reads, holding a real confidence head."""

    num_fused_shared_experts = 0

    def __init__(self, confidence_head):
        self.config = SimpleNamespace(o_groups=8, n_routed_experts=128)
        self.quant_config = SimpleNamespace(get_name=lambda: "exl3")
        self.confidence_head = confidence_head

    def named_parameters(self):
        return [(f"confidence_head.{name}", param) for name, param in self.confidence_head.named_parameters()]

    def _remap_dspark_weight_name(self, name):
        return DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(self, name)

    def _assert_confidence_head_loaded(self, **kwargs):
        return DeepseekV4ForCausalLMDSpark._assert_confidence_head_loaded(self, **kwargs)


def test_exl3_load_fills_the_static_cut_head():
    """Under EXL3 the plain F16 confidence weight passes adapt_exl3_weights untouched and lands in the head.

    The head's absence under static drops the name in the remap; with the cut on it must load, and the loader's
    completeness check must see it."""
    head = _build_v4_head(mode="static", cut=True)
    weight = torch.randn(1, _DSV41_HIDDEN + _DSV41_MARKOV_RANK).to(torch.float16)
    DeepseekV4ForCausalLMDSpark.load_weights(
        _FakeDraftModel(head), iter([("mtp.2.confidence_head.proj.weight", weight)])
    )
    torch.testing.assert_close(head.proj.weight, weight.float(), rtol=0, atol=0)
