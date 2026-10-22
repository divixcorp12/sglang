"""The MXFP4 expert format and the MiMo V2.6 Pro profile of the expert engine (CPU).

Row geometry at MiMo's real shapes (288 and 384 experts), the 512 B row alignment, the host scale interleave against
an independent reference (and FlashInfer's when it and a GPU are present), and the profile's caps, streamed modules,
gate registration and refusals. The slot-apply GPU test is test_mxfp4_slot_moe_gpu.py.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.expert_engine import startup
from sglang.srt.layers.moe.expert_engine.formats import mxfp4
from sglang.srt.layers.moe.expert_engine.formats.mxfp4 import (
    MXFP4_STREAMED_NAMES,
    SIDE_ROW_ELEMENTS,
    Mxfp4ExpertFormat,
    interleave_scale_rows,
)
from sglang.srt.layers.moe.expert_engine.profiles.mimo_v26 import MimoV26Profile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, INTER = 6144, 2048  # MiMo V2.6 Pro


def _format(experts=288, hidden=HIDDEN, inter=INTER, **kwargs):
    return Mxfp4ExpertFormat(
        layer_id=1, num_experts=experts, hidden=hidden, intermediate=inter, **kwargs
    )


def _reference_interleave(scales: torch.Tensor) -> torch.Tensor:
    """block_scale_interleave of one [rows, cols] matrix, written out index by index (no reshape/permute)."""
    rows, cols = scales.shape
    out = torch.empty_like(scales)
    flat = out.reshape(-1)
    k_tiles = cols // 4
    for row in range(rows):
        m_tile, within = divmod(row, 128)
        r_hi, r_lo = divmod(within, 32)  # row within the tile = r_hi * 32 + r_lo
        for col in range(cols):
            k_tile, k_in = divmod(col, 4)
            # [m_tile][k_tile][r_lo][r_hi][k_in]
            dest = (((m_tile * k_tiles + k_tile) * 32 + r_lo) * 4 + r_hi) * 4 + k_in
            flat[dest] = scales[row, col]
    return out


@pytest.mark.parametrize("experts", [288, 384])
def test_row_geometry_at_mimo_shapes(experts):
    fmt = _format(experts)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    assert tuple(specs) == MXFP4_STREAMED_NAMES
    assert fmt.num_experts(None) == experts
    assert specs["w13_weight"].row_shape == (4096, 3072)
    assert specs["w13_weight"].dtype == torch.int8
    assert specs["w13_weight_scale"].row_shape == (4096, 192)
    assert specs["w2_weight"].row_shape == (6144, 1024)
    assert specs["w2_weight_scale"].row_shape == (6144, 64)
    assert {n: s.row_bytes for n, s in specs.items()} == {
        "w13_weight": 12_582_912,
        "w13_weight_scale": 786_432,
        "w2_weight": 6_291_456,
        "w2_weight_scale": 393_216,
        "mxfp4_weight_global_scale": 512,
        "swiglu_limit": 512,
    }
    assert all(spec.residence == "host" for spec in specs.values())


def test_every_row_tensor_is_512_byte_aligned():
    for experts, hidden, inter in ((288, HIDDEN, INTER), (384, HIDDEN, INTER), (8, 128, 128), (4, 256, 384)):
        fmt = _format(experts, hidden, inter)
        assert fmt.row_alignment_violations() == []


def test_unpadded_side_tensors_would_violate_alignment():
    # The C++ layout comment describes the side tensors as 4 B each; the reader needs 512 B multiples, which is why
    # the format pads them to SIDE_ROW_ELEMENTS fp32.
    assert SIDE_ROW_ELEMENTS * 4 == 512
    assert 4 % 512 != 0


def test_misaligned_dims_are_refused():
    with pytest.raises(ValueError, match="multiples of 128"):
        _format(8, hidden=100)
    with pytest.raises(ValueError, match="multiples of 128"):
        _format(8, inter=192)
    with pytest.raises(ValueError, match="0 experts"):
        _format(0)


def test_format_capabilities_are_stubbed():
    fmt = _format()
    assert fmt.key == "mxfp4" and fmt.transport_build == "mxfp4"
    assert fmt.engine_served
    assert not fmt.supports_cpu_experts and not fmt.supports_draft
    assert fmt.cpu_trait(source="target") is None
    assert fmt.source(None, "w13_weight") is None


def test_layer_dims_and_for_layer():
    layer = SimpleNamespace(
        layer_id=3,
        mxfp4_num_experts=288,
        mxfp4_hidden=HIDDEN,
        mxfp4_inter=INTER,
        mxfp4_use_swiglu_step=False,
        mxfp4_swiglu_limit=None,
    )
    assert Mxfp4ExpertFormat.layer_dims(layer) == (288, HIDDEN)
    fmt = Mxfp4ExpertFormat.for_layer(layer)
    assert fmt.layer_id == 3 and fmt.num_experts(layer) == 288 and fmt.swiglu_limit is None


@pytest.mark.parametrize("rows,cols", [(128, 4), (256, 8), (512, 12)])
def test_interleave_matches_independent_reference(rows, cols):
    generator = torch.Generator().manual_seed(rows * 1000 + cols)
    scales = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, generator=generator)
    assert torch.equal(interleave_scale_rows(scales), _reference_interleave(scales))


def test_prepare_host_rows_interleaves_per_expert_without_mutating():
    fmt = _format(4, 256, 128)
    generator = torch.Generator().manual_seed(0)
    # w13 scales: [experts, 2 * inter, hidden / 32] = [4, 256, 8]
    scales = torch.randint(0, 256, (4, 256, 8), dtype=torch.uint8, generator=generator)
    before = scales.clone()
    out = fmt.prepare_host_rows(None, "w13_weight_scale", scales)
    assert torch.equal(scales, before)
    assert out.shape == scales.shape and out.dtype == torch.uint8
    assert out.data_ptr() != scales.data_ptr()
    for expert in range(4):
        assert torch.equal(out[expert], _reference_interleave(scales[expert]))
        assert torch.equal(out[expert], fmt.prepare_host_rows(None, "w13_weight_scale", scales[expert]))


def test_prepare_host_rows_accepts_e8m0_view():
    fmt = _format(2, 128, 128)
    raw = torch.randint(0, 255, (256, 4), dtype=torch.uint8)
    out = fmt.prepare_host_rows(None, "w13_weight_scale", raw.view(torch.float8_e8m0fnu))
    assert out.dtype == torch.uint8 and torch.equal(out, _reference_interleave(raw))


def test_prepare_host_rows_other_tensors():
    fmt = _format(4, 256, 128, swiglu_limit=7.0)
    weights = torch.randint(0, 256, (4, 256, 128), dtype=torch.uint8)
    as_int8 = fmt.prepare_host_rows(None, "w13_weight", weights)
    assert as_int8.dtype == torch.int8 and torch.equal(as_int8.view(torch.uint8), weights)
    side = fmt.prepare_host_rows(None, "mxfp4_weight_global_scale", torch.ones(4))
    assert side.shape == (4, SIDE_ROW_ELEMENTS) and bool((side == 1.0).all())
    assert fmt.side_row_value("swiglu_limit") == 7.0
    assert fmt.side_row_value("mxfp4_weight_global_scale") == 1.0
    assert _format().side_row_value("swiglu_limit") == 0.0
    with pytest.raises(ValueError, match="no streamed tensor"):
        fmt.prepare_host_rows(None, "w1", weights)
    with pytest.raises(ValueError, match="rows % 128"):
        interleave_scale_rows(torch.zeros(100, 4, dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        interleave_scale_rows(torch.zeros(128, 4, dtype=torch.float32))


def test_interleave_matches_flashinfer_block_scale_interleave():
    flashinfer = pytest.importorskip("flashinfer")
    if not torch.cuda.is_available():
        pytest.skip("flashinfer's block_scale_interleave needs a CUDA device (GPU test, runs on divix01)")
    for rows, cols in ((4096, 192), (6144, 64)):
        scales = torch.randint(0, 256, (3, rows, cols), dtype=torch.uint8)
        mine = interleave_scale_rows(scales)
        for expert in range(3):
            theirs = flashinfer.block_scale_interleave(scales[expert].cuda()).cpu().reshape(rows, cols)
            assert torch.equal(mine[expert], theirs)


# --- the profile ---


class _Gate(torch.nn.Module):
    def __init__(self, experts=4, hidden=8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(experts, hidden))
        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(experts))


class _Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()
        self.experts = torch.nn.Module()


class _Dense(torch.nn.Module):
    pass


def _fake_mimo(layers=4):
    decoder_layers = torch.nn.ModuleList()
    for index in range(layers):
        layer = torch.nn.Module()
        layer.mlp = _Dense() if index == 0 else _Moe()
        decoder_layers.append(layer)
    inner = torch.nn.Module()
    inner.layers = decoder_layers
    inner.start_layer, inner.end_layer = 0, layers
    model = torch.nn.Module()
    model.model = inner
    model.config = SimpleNamespace(num_experts_per_tok=8)
    mtp = torch.nn.Module()
    mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
    mtp.layers[0].mlp = _Moe()
    model.mtp = mtp
    return model


def test_profile_facts():
    profile = MimoV26Profile()
    assert profile.name == "mimo_v26" and profile.router == "sigmoid"
    assert profile.caps.max_experts == 384 and profile.caps.max_top_k == 8
    assert profile.caps.hidden_alignment == 8
    assert profile.config_type is None
    assert profile.strata(5) is None
    assert "MiMoV2ForCausalLM" in profile.architectures


def test_profile_caps_accept_mimo_and_refuse_beyond():
    profile = MimoV26Profile()
    for experts in (288, 384):
        startup.check_profile_caps(profile, experts=experts, top_k=8, hidden=HIDDEN)
    with pytest.raises(ValueError, match="385 experts"):
        startup.check_profile_caps(profile, experts=385, top_k=8, hidden=HIDDEN)
    with pytest.raises(ValueError, match="top-9"):
        startup.check_profile_caps(profile, experts=384, top_k=9, hidden=HIDDEN)
    with pytest.raises(ValueError, match="multiple"):
        startup.check_profile_caps(profile, experts=384, top_k=8, hidden=6148)


def test_streamed_modules_are_moe_layers_one_and_up():
    model = _fake_mimo(4)
    # A dense layer 0 has no experts; give it one anyway to prove layer 0 is excluded by the profile.
    model.model.layers[0].mlp.experts = torch.nn.Module()
    paths = [path for path, _ in MimoV26Profile().streamed_modules(model)]
    assert paths == [f"model.layers.{i}.mlp.experts" for i in (1, 2, 3)]
    assert not MimoV26Profile().is_streamed_module("mtp.layers.0.mlp.experts")


def test_register_gates_registers_moe_layers_only():
    from sglang.srt.layers.moe import ram_prefetch

    ram_prefetch.clear_router_gates()
    try:
        model = _fake_mimo(4)
        assert MimoV26Profile().register_gates(model) == 3
        gates = ram_prefetch.registered_gates()
        assert sorted(gates) == [1, 2, 3]
        assert gates[2].weight is model.model.layers[2].mlp.gate.weight
        assert gates[2].bias is model.model.layers[2].mlp.gate.e_score_correction_bias
        assert gates[2].top_k == 8
    finally:
        ram_prefetch.clear_router_gates()


def test_skip_hook_skips_only_streamed_routed_expert_tensors():
    profile = MimoV26Profile()
    fmt = Mxfp4ExpertFormat
    hook = profile.streamed_weight_skip_hook(streaming=True, expert_format=fmt)
    assert hook("model.layers.7.mlp.experts.12.gate_proj.weight")
    assert hook("model.layers.7.mlp.experts.12.down_proj.weight_scale")
    assert not hook("model.layers.7.mlp.gate.weight")
    assert not hook("model.layers.7.self_attn.qkv_proj.weight")
    assert not hook("mtp.layers.0.mlp.experts.0.gate_proj.weight")
    assert profile.streamed_weight_skip_hook(streaming=False, expert_format=fmt) is None
    assert profile.streamed_weight_skip_hook(streaming=True, expert_format=None) is None


def test_draft_stub_refuses_at_startup():
    profile = MimoV26Profile()
    with pytest.raises(NotImplementedError, match="no draft adapter yet"):
        profile.draft_adapter()
    with pytest.raises(NotImplementedError, match="no draft adapter yet"):
        startup.check_requested_capabilities(
            expert_format=Mxfp4ExpertFormat, profile=profile, cpu_experts=False, draft=True
        )


def test_cpu_experts_refuse_on_the_mxfp4_stub():
    with pytest.raises(ValueError, match="no CPU kernel"):
        startup.check_requested_capabilities(
            expert_format=Mxfp4ExpertFormat, profile=MimoV26Profile(), cpu_experts=True, draft=False
        )
    startup.check_requested_capabilities(
        expert_format=Mxfp4ExpertFormat, profile=MimoV26Profile(), cpu_experts=False, draft=False
    )


def test_streams_experts_selects_mimo_fp4_layers_only(monkeypatch):
    mimo = SimpleNamespace(is_fp4_experts=True, is_dsv4_fp4_experts=False)
    dsv4 = SimpleNamespace(is_fp4_experts=True, is_dsv4_fp4_experts=True)
    fp8 = SimpleNamespace(is_fp4_experts=False, is_dsv4_fp4_experts=False)
    prefix = "model.layers.5.mlp.experts"
    monkeypatch.setattr(mxfp4, "expert_stream_enabled", lambda: False)
    assert not mxfp4.streams_experts(mimo, prefix)
    monkeypatch.setattr(mxfp4, "expert_stream_enabled", lambda: True)
    assert mxfp4.streams_experts(mimo, prefix)
    assert not mxfp4.streams_experts(None, prefix)
    assert not mxfp4.streams_experts(dsv4, prefix)
    assert not mxfp4.streams_experts(fp8, prefix)
    assert not mxfp4.streams_experts(mimo, "model.layers.0.mlp.experts")
    assert not mxfp4.streams_experts(mimo, "mtp.layers.0.mlp.experts")
