"""The expert engine's format and profile registries, and the startup checks built on them (CPU).

Covers format resolution (by name, quantization config and quant method), profile resolution (DeepSeek V4 and
MiMoV2 model configs), an unknown transport key, cap violations and the refusal of CPU experts or a draft on
a format or profile that only has a stub.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import EXL3_BUILD
from sglang.srt.layers.moe.expert_engine.formats import (
    format_for_quant_method,
    format_for_quantization_config,
    get_format,
    registered_formats,
)
from sglang.srt.layers.moe.expert_engine.formats.exl3 import Exl3ExpertFormat
from sglang.srt.layers.moe.expert_engine.profiles import (
    ProfileCaps,
    get_profile,
    get_profile_by_name,
    profile_for_architectures,
    profile_for_hf_config,
    profile_for_model,
    registered_profiles,
)
from sglang.srt.layers.moe.expert_engine.service import ExpertEngineService
from sglang.srt.layers.moe.expert_engine.startup import (
    ENGINE_MAX_EXPERTS,
    check_profile_caps,
    check_requested_capabilities,
    resolve_transport_build,
)
from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXL3_CONFIG = Exl3Config(3.0, 6, "mul1", "1.4")


class _Adapter:
    def __init__(self, *, cpu: bool):
        self.name = "fake_draft"
        self.supports_cpu_experts = cpu


class _Format:
    """A format object with only the facts the startup checks read."""

    key = "fakefmt"

    def __init__(self, *, transport_build="exl3", cpu=True, draft=True):
        self.transport_build = transport_build
        self.supports_cpu_experts = cpu
        self.supports_draft = draft


class _Profile:
    """A profile object with only the facts the startup checks read."""

    name = "fakeprof"

    def __init__(self, *, router="sqrt_softplus", adapter=None, caps=None):
        self.router = router
        self._adapter = adapter
        self.caps = caps or ProfileCaps(max_experts=384, max_top_k=8, hidden_alignment=8)

    def draft_adapter(self):
        return self._adapter


@pytest.fixture
def no_bound_service():
    ExpertEngineService._instance = None
    yield
    ExpertEngineService._instance = None


# ---- format registry ----


def test_get_format_returns_the_registered_class():
    assert get_format("exl3") is Exl3ExpertFormat
    assert registered_formats() == ("exl3", "mxfp4", "nvfp4")


def test_an_unknown_format_name_raises_naming_the_registered_formats():
    with pytest.raises(ValueError, match=r"no expert format 'nvfp9'.*\['exl3', 'mxfp4', 'nvfp4'\]"):
        get_format("nvfp9")


def test_a_format_resolves_from_its_quantization_config():
    assert format_for_quantization_config({"quant_method": "exl3"}) is Exl3ExpertFormat
    assert format_for_quantization_config(EXL3_CONFIG) is Exl3ExpertFormat
    assert format_for_quantization_config({"quant_method": "fp8"}) is None
    assert format_for_quantization_config(None) is None


def test_a_format_resolves_from_a_quant_method_by_its_declared_name():
    assert format_for_quant_method(Exl3MoEMethod(EXL3_CONFIG, streamed=True)) is Exl3ExpertFormat

    class Plain(FusedMoEMethodBase):
        def apply(self, layer, dispatch_output):
            raise NotImplementedError

    assert format_for_quant_method(Plain()) is None
    assert format_for_quant_method(object()) is None


def test_capabilities_of_the_exl3_and_dense_formats():
    assert Exl3ExpertFormat.engine_served and Exl3ExpertFormat.supports_cpu_experts
    assert Exl3ExpertFormat.supports_draft and Exl3ExpertFormat.supports_dead_rows
    assert Exl3ExpertFormat.supports_transfer_executor
    assert Exl3ExpertFormat.transport_build == "exl3"
    six = DenseLayerFormat([f"t{i}" for i in range(6)])
    two = DenseLayerFormat(["a", "b"])
    assert not six.engine_served and six.transport_build is None
    assert six.supports_transfer_executor and not two.supports_transfer_executor
    assert six.cpu_trait(source="target") is None


# ---- profile registry ----


def _model_config(*architectures):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=list(architectures)))


def test_a_deepseek_v4_model_config_resolves_to_the_dsv41_profile():
    profile = get_profile(_model_config("DeepseekV4ForCausalLM"))
    assert profile.name == "dsv41" and profile.router == "sqrt_softplus"
    assert profile_for_hf_config(SimpleNamespace(architectures=["DeepseekV4ForCausalLMDSpark"])).name == "dsv41"
    assert profile_for_model(SimpleNamespace(config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]))).name == "dsv41"
    assert registered_profiles() == ("dsv41", "mimo_v26")


def test_a_mimo_model_config_resolves_to_the_mimo_v26_profile():
    for architecture in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"):
        profile = get_profile(_model_config(architecture))
        assert profile.name == "mimo_v26" and profile.router == "sigmoid"
    assert profile_for_hf_config(SimpleNamespace(architectures=["MiMoV2ForCausalLM"])).name == "mimo_v26"


def test_a_model_without_a_profile_is_refused_naming_its_architectures_and_the_registered_profiles():
    with pytest.raises(ValueError, match=r"Qwen3MoeForCausalLM.*registered: \['dsv41', 'mimo_v26'\]"):
        get_profile(_model_config("Qwen3MoeForCausalLM"))
    with pytest.raises(ValueError, match="no expert-engine model profile"):
        profile_for_architectures(None)


def test_an_unknown_profile_name_raises():
    with pytest.raises(ValueError, match="no model profile 'qwen3'"):
        get_profile_by_name("qwen3")


def test_the_dsv41_profile_decides_which_modules_stream_and_which_are_the_draft():
    profile = get_profile_by_name("dsv41")
    assert profile.is_streamed_module("model.layers.12.mlp.experts")
    assert not profile.is_streamed_module("stages.0.mlp.experts")
    adapter = profile.draft_adapter()
    assert adapter.is_draft_module("stages.0.mlp.experts") and adapter.is_draft_module("model.stages.2.mlp.experts")
    assert not adapter.is_draft_module("model.layers.12.mlp.experts")
    model = torch.nn.Module()
    model.add_module("model", torch.nn.Module())
    model.model.add_module("layers", torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()]))
    for layer in model.model.layers:
        layer.add_module("mlp", torch.nn.Module())
        layer.mlp.add_module("experts", torch.nn.Module())
    assert [path for path, _ in profile.streamed_modules(model)] == [
        "model.layers.0.mlp.experts",
        "model.layers.1.mlp.experts",
    ]


def test_the_skip_hook_needs_streaming_and_an_engine_served_format():
    profile = get_profile_by_name("dsv41")
    routed = "layers.12.ffn.experts.5.w1.trellis"
    hook = profile.streamed_weight_skip_hook(streaming=True, expert_format=get_format("exl3"))
    assert hook(routed)
    assert not hook("layers.12.ffn.shared_experts.w2.trellis")
    assert not hook("layers.12.ffn.gate.weight")
    assert profile.streamed_weight_skip_hook(streaming=False, expert_format=get_format("exl3")) is None
    assert profile.streamed_weight_skip_hook(streaming=True, expert_format=None) is None
    assert profile.streamed_weight_skip_hook(streaming=True, expert_format=DenseLayerFormat(["a"])) is None


def test_the_dsv41_strata_are_the_handoffs():
    strata = get_profile_by_name("dsv41").strata
    assert [strata(layer) for layer in (1, 14, 2, 15, 20, 21, 37, 39, 8)] == [
        "enters engram", "enters engram", "leaves engram", "leaves engram", "19->20", "replay start",
        "late", "late", "interior",
    ]


# ---- startup checks ----


def test_the_transport_key_is_the_formats_half_and_the_profiles_router():
    assert resolve_transport_build(expert_format=Exl3ExpertFormat, profile=get_profile_by_name("dsv41")) == EXL3_BUILD


def test_an_unknown_transport_key_raises_naming_the_jit_key():
    with pytest.raises(ValueError, match=r"\('exl3', 'sigmoid'\).*expert_stream_host_exl3_sigmoid"):
        resolve_transport_build(expert_format=Exl3ExpertFormat, profile=_Profile(router="sigmoid"))


def test_a_format_without_a_transport_build_is_refused_naming_both():
    with pytest.raises(ValueError, match="fakefmt.*no expert-engine transport build.*fakeprof"):
        resolve_transport_build(expert_format=_Format(transport_build=None), profile=_Profile())


def test_the_service_refuses_an_unbuilt_pairing_at_construction(no_bound_service):
    with pytest.raises(ValueError, match="expert_stream_host_exl3_sigmoid"):
        ExpertEngineService(expert_format=Exl3ExpertFormat, profile=_Profile(router="sigmoid"))
    assert ExpertEngineService._instance is None


def test_the_service_binds_one_pairing_per_process(no_bound_service):
    pairing = dict(expert_format=Exl3ExpertFormat, profile=get_profile_by_name("dsv41"))
    with pytest.raises(RuntimeError, match="no expert engine service is bound"):
        ExpertEngineService.get()
    service = ExpertEngineService.bind(**pairing)
    assert service.build_key == EXL3_BUILD
    assert ExpertEngineService.bind(**pairing) is service and ExpertEngineService.get() is service
    other = _Profile()
    with pytest.raises(RuntimeError, match="bound to format 'exl3' with profile 'dsv41'.*'fakefmt' with profile 'fakeprof'"):
        ExpertEngineService.bind(expert_format=_Format(), profile=other)


@pytest.mark.parametrize(
    "experts,top_k,hidden,message",
    [
        (385, 6, 4096, "385 experts per layer, over the profile's cap of 384"),
        (256, 9, 4096, "routes top-9, over the profile's cap of 8"),
        (256, 6, 4100, "hidden size 4100 is not a multiple of 8"),
    ],
)
def test_a_model_beyond_the_profile_caps_is_refused(experts, top_k, hidden, message):
    with pytest.raises(ValueError, match=message):
        check_profile_caps(get_profile_by_name("dsv41"), experts=experts, top_k=top_k, hidden=hidden)


def test_a_model_inside_the_caps_passes_and_top_k_may_be_unknown():
    profile = get_profile_by_name("dsv41")
    check_profile_caps(profile, experts=256, top_k=6, hidden=4096)
    check_profile_caps(profile, experts=ENGINE_MAX_EXPERTS, top_k=None, hidden=8)


def test_a_profile_claiming_more_than_the_engine_supports_is_refused():
    greedy = _Profile(caps=ProfileCaps(max_experts=512, max_top_k=8, hidden_alignment=8))
    with pytest.raises(ValueError, match="claims up to 512 experts and top-8, beyond the engine's 384"):
        check_profile_caps(greedy, experts=256, top_k=6, hidden=4096)


def test_cpu_experts_on_a_stub_format_are_refused_at_startup():
    with pytest.raises(ValueError, match="CPU experts were requested, but expert format 'fakefmt' has no CPU kernel"):
        check_requested_capabilities(
            expert_format=_Format(cpu=False), profile=_Profile(), cpu_experts=True, draft=False
        )
    check_requested_capabilities(expert_format=_Format(cpu=False), profile=_Profile(), cpu_experts=False, draft=False)
    check_requested_capabilities(expert_format=_Format(), profile=_Profile(), cpu_experts=True, draft=False)


def test_a_draft_on_a_profile_without_an_adapter_is_refused():
    with pytest.raises(ValueError, match="profile 'fakeprof' has no draft adapter"):
        check_requested_capabilities(
            expert_format=_Format(), profile=_Profile(adapter=None), cpu_experts=False, draft=True
        )


def test_a_draft_on_a_format_that_does_not_support_one_is_refused():
    with pytest.raises(ValueError, match="'fakefmt' does not support one"):
        check_requested_capabilities(
            expert_format=_Format(draft=False),
            profile=_Profile(adapter=_Adapter(cpu=True)),
            cpu_experts=False,
            draft=True,
        )


def test_cpu_draft_experts_on_a_stub_adapter_or_format_are_refused():
    for fmt, adapter in ((_Format(), _Adapter(cpu=False)), (_Format(cpu=False), _Adapter(cpu=True))):
        with pytest.raises(ValueError, match="CPU draft experts were requested"):
            check_requested_capabilities(
                expert_format=fmt, profile=_Profile(adapter=adapter), cpu_experts=False, draft=True, cpu_draft=True
            )
    check_requested_capabilities(
        expert_format=_Format(),
        profile=_Profile(adapter=_Adapter(cpu=True)),
        cpu_experts=True,
        draft=True,
        cpu_draft=True,
    )


def test_the_exl3_quant_config_asks_for_cpu_draft_experts_through_the_profile(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

    class _Stub(FusedMoE):
        def __init__(self):
            torch.nn.Module.__init__(self)

    with envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS.override(True):
        draft = EXL3_CONFIG.get_quant_method(_Stub(), "stages.0.mlp.experts")
        target = EXL3_CONFIG.get_quant_method(_Stub(), "model.layers.3.mlp.experts")
    assert draft.draft and draft.cpu_draft and not draft.streamed
    assert target.streamed and not target.draft and not target.cpu_draft
