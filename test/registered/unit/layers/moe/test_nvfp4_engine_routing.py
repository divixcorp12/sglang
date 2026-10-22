"""Which NVFP4 MoE layers the native expert engine serves, and that every other model keeps the legacy path (CPU).

Registry resolution (NVFP4 from a ModelOpt config, MXFP4 from ``quantization_config.store_dtype``, the MiMo profile from
its architectures), the engine/legacy decision of ``ModelOptNvFp4FusedMoEMethod.bind_expert_engine`` for a DSV4.1, a
MiMo and a Qwen-like model, what an engine-served layer allocates, and that the offloader leaves such a layer alone
while still demanding the legacy layer's parameters.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.expert_engine.formats import (
    format_for_quant_method,
    format_for_quantization_config,
)
from sglang.srt.layers.moe.expert_engine.formats.mxfp4 import Mxfp4ExpertFormat
from sglang.srt.layers.moe.expert_engine.formats.nvfp4 import (
    Nvfp4ExpertFormat,
    engine_profile_name_for,
    engine_serves_nvfp4,
)
from sglang.srt.layers.moe.expert_engine.profiles import get_profile
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.utils.offloader import _iter_streamed_nvfp4_parameters
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DSV4_LAYER = "model.layers.3.mlp.experts"
QWEN_LAYER = "model.layers.3.mlp.experts"  # the same path shape: only the architecture tells them apart


# ---- registry resolution ----


def test_mxfp4_is_detected_by_store_dtype_inside_an_fp8_checkpoint():
    assert format_for_quantization_config({"quant_method": "fp8", "store_dtype": "mxfp4"}) is Mxfp4ExpertFormat
    assert format_for_quantization_config({"quant_method": "fp8", "store_dtype": "MXFP4"}) is Mxfp4ExpertFormat
    assert format_for_quantization_config({"quant_method": "fp8"}) is None
    assert format_for_quantization_config({"quant_method": "fp8", "store_dtype": "fp8"}) is None
    config = SimpleNamespace(get_name=lambda: "fp8", store_dtype="mxfp4")
    assert format_for_quantization_config(config) is Mxfp4ExpertFormat
    assert format_for_quantization_config(SimpleNamespace(get_name=lambda: "fp8")) is None


def test_nvfp4_is_detected_from_a_modelopt_config_or_checkpoint_mapping():
    assert format_for_quantization_config(SimpleNamespace(get_name=lambda: "modelopt_fp4")) is Nvfp4ExpertFormat
    assert format_for_quantization_config({"quant_method": "modelopt", "quant_algo": "NVFP4"}) is Nvfp4ExpertFormat
    assert format_for_quantization_config({"quant_method": "modelopt_fp4"}) is Nvfp4ExpertFormat
    assert format_for_quantization_config({"quant_method": "modelopt", "quant_algo": "FP8"}) is None
    # store_dtype wins over the quant method name.
    assert (
        format_for_quantization_config({"quant_method": "modelopt_fp4", "store_dtype": "mxfp4"}) is Mxfp4ExpertFormat
    )


def test_the_mimo_profile_resolves_from_its_architectures_with_the_sigmoid_router():
    for architecture in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"):
        profile = get_profile(SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])))
        assert profile.name == "mimo_v26" and profile.router == "sigmoid"


# ---- the decision ----


def _method():
    """A ModelOptNvFp4FusedMoEMethod without its constructor (which needs a Blackwell platform)."""
    method = object.__new__(ModelOptNvFp4FusedMoEMethod)
    method.quant_config = None
    return method


@pytest.fixture
def streaming(monkeypatch):
    """Engine streaming on, with the process's model architectures chosen per test."""
    from sglang.srt.layers.moe.expert_engine.formats import mxfp4

    state = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
    monkeypatch.setattr(mxfp4, "expert_stream_enabled", lambda: True)
    monkeypatch.setattr(
        "sglang.srt.runtime_context.process_model_config",
        lambda: SimpleNamespace(hf_config=SimpleNamespace(architectures=state.architectures)),
    )
    return state


def test_a_dsv41_nvfp4_layer_is_served_by_the_engine(streaming):
    method = _method()
    assert not method.engine_streamed and method.expert_format_name is None
    method.bind_expert_engine(DSV4_LAYER)
    assert method.engine_streamed
    assert method.expert_format_name == "nvfp4" and method.expert_profile_name == "dsv41"
    assert format_for_quant_method(method) is Nvfp4ExpertFormat


def test_a_converted_mimo_layer_is_served_by_the_engine_from_layer_one(streaming):
    streaming.architectures = ["MiMoV2ForCausalLM"]
    dense, moe = _method(), _method()
    dense.bind_expert_engine("model.layers.0.mlp.experts")
    moe.bind_expert_engine("model.layers.1.mlp.experts")
    assert not dense.engine_streamed
    assert moe.engine_streamed and moe.expert_profile_name == "mimo_v26"


def test_a_qwen_like_model_keeps_the_legacy_streamer(streaming):
    streaming.architectures = ["Qwen3MoeForCausalLM"]
    method = _method()
    method.bind_expert_engine(QWEN_LAYER)
    assert not method.engine_streamed
    assert method.expert_format_name is None and method.expert_profile_name is None
    assert format_for_quant_method(method) is None


def test_the_dspark_draft_stages_stay_resident(streaming):
    method = _method()
    method.bind_expert_engine("stages.0.mlp.experts")
    assert not method.engine_streamed


def test_nothing_is_served_when_the_engine_switch_is_off(monkeypatch):
    from sglang.srt.layers.moe.expert_engine.formats import mxfp4

    monkeypatch.setattr(mxfp4, "expert_stream_enabled", lambda: False)

    def no_model_config():
        raise AssertionError("a process that never streams must not read the model config")

    monkeypatch.setattr("sglang.srt.runtime_context.process_model_config", no_model_config)
    assert engine_profile_name_for(DSV4_LAYER) is None
    method = _method()
    method.bind_expert_engine(DSV4_LAYER)
    assert not method.engine_streamed


def test_engine_serves_nvfp4_by_architecture_and_module_path():
    assert engine_serves_nvfp4(["DeepseekV4ForCausalLM"], DSV4_LAYER) == "dsv41"
    assert engine_serves_nvfp4(["DeepseekV4ForCausalLMDSpark"], "stages.1.mlp.experts") is None
    assert engine_serves_nvfp4(["MiMoV2ForCausalLM"], "model.layers.1.mlp.experts") == "mimo_v26"
    assert engine_serves_nvfp4(["MiMoV2ForCausalLM"], "model.layers.0.mlp.experts") is None
    assert engine_serves_nvfp4(["Qwen3MoeForCausalLM"], QWEN_LAYER) is None
    assert engine_serves_nvfp4(None, QWEN_LAYER) is None


# ---- what a served layer allocates ----


def test_an_engine_served_layer_allocates_no_expert_tensor_and_records_the_dims(streaming):
    method = _method()
    method.bind_expert_engine(DSV4_LAYER)
    layer = SimpleNamespace(moe_runner_config=SimpleNamespace(is_gated=True))
    method.create_weights(layer, 256, 4096, 2048, torch.bfloat16)
    assert (layer.nvfp4_num_experts, layer.nvfp4_hidden, layer.nvfp4_inter) == (256, 4096, 2048)
    assert layer.nvfp4_gated and layer.nvfp4_streamed
    assert layer.w13_input_scale_quant is None and layer.w2_input_scale_quant is None
    assert Nvfp4ExpertFormat.layer_dims(layer) == (256, 4096)
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale", "w13_weight_scale_2"):
        assert not hasattr(layer, name), name
    fmt = Nvfp4ExpertFormat.for_layer(SimpleNamespace(layer_id=3, **vars(layer)))
    assert (fmt.layer_id, fmt.experts, fmt.hidden, fmt.intermediate) == (3, 256, 4096, 2048)


# ---- the offloader ----


class _Experts(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        self.quant_method = method


def test_the_offloader_skips_an_engine_served_layer_but_still_demands_a_legacy_layers_parameters(streaming):
    served = _method()
    served.bind_expert_engine(DSV4_LAYER)
    assert list(_iter_streamed_nvfp4_parameters(torch.nn.Sequential(_Experts(served)))) == []

    streaming.architectures = ["Qwen3MoeForCausalLM"]
    legacy = _method()
    legacy.bind_expert_engine(QWEN_LAYER)
    with pytest.raises(RuntimeError, match="missing streamable parameters"):
        list(_iter_streamed_nvfp4_parameters(torch.nn.Sequential(_Experts(legacy))))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
