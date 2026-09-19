"""The safetensors iterators never read the tensors a model asks to skip."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.srt.models.deepseek_v4_exl3_weights import (
    is_streamed_expert_weight,
    streamed_expert_skip_hook,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _shard(tmp_path):
    path = str(tmp_path / "model-00001.safetensors")
    save_file(
        {
            "layers.0.ffn.experts.3.w1.trellis": torch.zeros(4, dtype=torch.int16),
            "layers.0.ffn.shared_experts.w1.trellis": torch.ones(4, dtype=torch.int16),
            "layers.0.attn.norm.weight": torch.ones(2),
        },
        path,
    )
    return [path]


def _skip(name):
    return ".ffn.experts." in name


def test_single_thread_iterator_skips(tmp_path):
    names = [n for n, _ in safetensors_weights_iterator(_shard(tmp_path), skip_name=_skip)]
    assert sorted(names) == ["layers.0.attn.norm.weight", "layers.0.ffn.shared_experts.w1.trellis"]


def test_multi_thread_iterator_skips(tmp_path):
    names = [
        n
        for n, _ in buffered_multi_thread_safetensors_weights_iterator(
            _shard(tmp_path), max_workers=2, skip_name=_skip
        )
    ]
    assert sorted(names) == ["layers.0.attn.norm.weight", "layers.0.ffn.shared_experts.w1.trellis"]


def test_default_skips_nothing(tmp_path):
    assert len(list(safetensors_weights_iterator(_shard(tmp_path)))) == 3


def test_source_takes_the_models_skip_hook():
    config = SimpleNamespace(model_path="/unused", revision=None)
    model = SimpleNamespace(skip_checkpoint_weight=_skip)
    assert DefaultModelLoader.Source.init_new(config, model).skip_weight is _skip
    assert DefaultModelLoader.Source.init_new(config, SimpleNamespace()).skip_weight is None


def test_only_streamed_exl3_routed_experts_are_skipped():
    routed = "layers.12.ffn.experts.301.w2.trellis"
    assert is_streamed_expert_weight(routed, "exl3", True)
    assert not is_streamed_expert_weight(routed, "exl3", False)
    assert not is_streamed_expert_weight(routed, "fp8", True)
    assert not is_streamed_expert_weight(routed, None, True)
    assert not is_streamed_expert_weight("layers.12.ffn.shared_experts.w2.trellis", "exl3", True)
    assert not is_streamed_expert_weight("layers.12.ffn.gate.weight", "exl3", True)


def test_hook_exists_only_when_it_can_skip():
    assert streamed_expert_skip_hook("exl3", False) is None
    assert streamed_expert_skip_hook("fp8", True) is None
    assert streamed_expert_skip_hook(None, True) is None
    assert streamed_expert_skip_hook("exl3", True) is not None


def test_non_streaming_model_gives_the_loader_no_skip_hook():
    # A DeepSeek V4 model sets skip_checkpoint_weight from the hook factory, so a
    # non-streaming one exposes None and the loader keeps its no-skip path
    # (fastsafetensors then does not raise).
    config = SimpleNamespace(model_path="/unused", revision=None)
    model = SimpleNamespace(skip_checkpoint_weight=streamed_expert_skip_hook("exl3", False))
    assert DefaultModelLoader.Source.init_new(config, model).skip_weight is None


def test_streaming_model_hook_skips_routed_experts():
    hook = streamed_expert_skip_hook("exl3", True)
    assert hook("layers.12.ffn.experts.301.w2.trellis")
    assert not hook("layers.12.ffn.shared_experts.w2.trellis")
    assert not hook("layers.12.attn.norm.weight")
    config = SimpleNamespace(model_path="/unused", revision=None)
    model = SimpleNamespace(skip_checkpoint_weight=hook)
    assert DefaultModelLoader.Source.init_new(config, model).skip_weight is hook


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
