"""The NVFP4 expert format of the expert engine (CPU).

Row geometry at the real DSV4.1 (hidden 4096, intermediate 2048) and MiMo V2.6 Pro (6144 / 2048, the shape of
``csrc/nvfp4/optimized/shapes.hpp``) shapes, the 512 B row alignment (alphas are padded rows), the host scale swizzle
against an independent reference, the CPU trait over the engine's tensor names, and the refusal to build a row source
from the expert-file cache. The slot-apply GPU test is test_nvfp4_slot_moe_gpu.py; registry resolution and the
legacy/engine selection are in test/registered/unit/layers/moe/test_nvfp4_engine_routing.py.
"""

import re
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.expert_engine import startup
from sglang.srt.layers.moe.expert_engine.formats import get_format
from sglang.srt.layers.moe.expert_engine.formats.nvfp4 import (
    NVFP4_STREAMED_NAMES,
    SIDE_ROW_ELEMENTS,
    Nvfp4EngineCpuTrait,
    Nvfp4ExpertFormat,
)
from sglang.test.expert_stream_sources import MOE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DSV41 = (4096, 2048)
MIMO = (6144, 2048)


def _format(experts=288, shape=MIMO, **kwargs):
    hidden, inter = shape
    return Nvfp4ExpertFormat(layer_id=1, num_experts=experts, hidden=hidden, intermediate=inter, **kwargs)


def _reference_swizzle(scales: torch.Tensor) -> torch.Tensor:
    """The 128x4 block-scale swizzle of one [rows, cols] matrix (rows % 128 == 0, cols % 4 == 0), index by index."""
    rows, cols = scales.shape
    out = torch.empty_like(scales)
    flat = out.reshape(-1)
    for row in range(rows):
        m_tile, within = divmod(row, 128)
        r_hi, r_lo = divmod(within, 32)
        for col in range(cols):
            k_tile, k_in = divmod(col, 4)
            flat[(((m_tile * (cols // 4) + k_tile) * 32 + r_lo) * 4 + r_hi) * 4 + k_in] = scales[row, col]
    return out


@pytest.mark.parametrize("experts", [288, 384])
def test_row_geometry_at_mimo_shapes(experts):
    fmt = _format(experts, MIMO)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    assert tuple(specs) == NVFP4_STREAMED_NAMES
    assert fmt.num_experts(None) == experts
    assert specs["w13_weight"].row_shape == (4096, 3072) and specs["w13_weight"].dtype == torch.uint8
    assert specs["w2_weight"].row_shape == (6144, 1024) and specs["w2_weight"].dtype == torch.uint8
    assert specs["w13_blockscale_swizzled"].row_shape == (4096, 384)
    assert specs["w13_blockscale_swizzled"].dtype == torch.float8_e4m3fn
    assert specs["w2_blockscale_swizzled"].row_shape == (6144, 128)
    assert specs["g1_alphas"].row_shape == specs["g2_alphas"].row_shape == (SIDE_ROW_ELEMENTS,)
    assert specs["g1_alphas"].dtype == torch.float32
    assert {n: s.row_bytes for n, s in specs.items()} == {
        "w13_weight": 12_582_912,
        "w2_weight": 6_291_456,
        "w13_blockscale_swizzled": 4096 * 384,  # shapes.hpp: row_bytes(MimoV26ProShape)[kSf13]
        "w2_blockscale_swizzled": 6144 * 128,
        "g1_alphas": 512,
        "g2_alphas": 512,
    }
    assert all(spec.residence == "host" for spec in specs.values())


def test_row_geometry_at_dsv41_shapes():
    specs = {spec.name: spec for spec in _format(256, DSV41).tensor_specs(None)}
    assert {n: s.row_shape for n, s in specs.items()} == {
        "w13_weight": (4096, 2048),
        "w2_weight": (4096, 1024),
        "w13_blockscale_swizzled": (4096, 256),
        "w2_blockscale_swizzled": (4096, 128),
        "g1_alphas": (128,),
        "g2_alphas": (128,),
    }


def test_the_row_bytes_are_the_cpu_kernels_slab_rows_plus_the_alpha_padding():
    """quant.hpp's row_bytes(hidden, intermediate): n*h, h*n/2, the swizzled scales, then 4 B alphas (here 512 B)."""
    def up(value, multiple):
        return -(-value // multiple) * multiple

    for hidden, inter in (DSV41, MIMO, (256, 384)):
        fmt = _format(8, (hidden, inter))
        sizes = [spec.row_bytes for spec in fmt.tensor_specs(None)]
        assert sizes[:4] == [
            inter * hidden,
            hidden * inter // 2,
            up(2 * inter, 128) * up(hidden // 16, 4),
            up(hidden, 128) * up(inter // 16, 4),
        ]
        assert sizes[4:] == [512, 512]


def test_the_names_are_the_cpp_layouts_in_order_and_the_alphas_are_the_small_rows():
    text = (MOE / "formats" / "nvfp4" / "row_layout.h").read_text()
    names = re.findall(r'"([^"]+)"', re.search(r"kNames = \{(.*?)\};", text, re.S).group(1))
    assert tuple(names) == NVFP4_STREAMED_NAMES
    assert "padded row of 128 fp32" in text  # the comment says what the rows are


def test_every_row_tensor_is_512_byte_aligned_and_the_image_geometry_is_page_strided():
    for experts, shape in ((288, MIMO), (256, DSV41), (8, (64, 64)), (4, (128, 256))):
        fmt = _format(experts, shape)
        assert fmt.row_alignment_violations() == []
        offsets, image, stride = fmt.image_geometry()
        assert all(offset % 512 == 0 for offset in offsets) and image % 512 == 0
        assert stride % 4096 == 0 and image <= stride < image + 4096
        assert offsets[0] == 0 and image == sum(s.row_bytes for s in fmt.tensor_specs(None))


def test_a_shape_with_an_unaligned_row_is_refused_naming_the_tensor():
    with pytest.raises(ValueError, match=r"not 512 B aligned.*w13_weight"):
        _format(4, (48, 16))
    with pytest.raises(ValueError, match="multiples of 16"):
        _format(4, (100, 64))
    with pytest.raises(ValueError, match="0 experts"):
        _format(0, (64, 64))


def test_non_gated_layers_have_a_half_height_w13():
    gated = {s.name: s for s in _format(8, (256, 256)).tensor_specs(None)}
    plain = {s.name: s for s in _format(8, (256, 256), gated=False).tensor_specs(None)}
    assert plain["w13_weight"].row_shape == (256, 128) and gated["w13_weight"].row_shape == (512, 128)


def test_capabilities_and_the_transport_pairing():
    fmt = get_format("nvfp4")
    assert fmt is Nvfp4ExpertFormat
    assert fmt.key == "nvfp4" and fmt.transport_build == "nvfp4" and fmt.engine_served
    assert fmt.supports_cpu_experts and not fmt.supports_draft and not fmt.supports_dead_rows
    assert fmt.supports_transfer_executor and not fmt.requires_swiglu_limit
    from sglang.srt.layers.moe.expert_engine.profiles import get_profile_by_name

    # DSV4.1 and a converted MiMo find their builds through the profile's router.
    assert startup.resolve_transport_build(expert_format=fmt, profile=get_profile_by_name("dsv41")) == (
        "nvfp4",
        "sqrt_softplus",
    )
    assert startup.resolve_transport_build(expert_format=fmt, profile=get_profile_by_name("mimo_v26")) == (
        "nvfp4",
        "sigmoid",
    )


def test_a_draft_on_nvfp4_refuses_at_startup_and_cpu_experts_do_not():
    from sglang.srt.layers.moe.expert_engine.profiles import get_profile_by_name

    startup.check_requested_capabilities(
        expert_format=Nvfp4ExpertFormat, profile=get_profile_by_name("dsv41"), cpu_experts=True, draft=False
    )
    with pytest.raises(ValueError, match="does not support one"):
        startup.check_requested_capabilities(
            expert_format=Nvfp4ExpertFormat, profile=get_profile_by_name("dsv41"), cpu_experts=False, draft=True
        )


def test_the_mimo_nvfp4_shape_passes_the_mimo_caps():
    from sglang.srt.layers.moe.expert_engine.profiles import get_profile_by_name

    startup.check_profile_caps(get_profile_by_name("mimo_v26"), experts=384, top_k=8, hidden=6144)
    startup.check_profile_caps(get_profile_by_name("dsv41"), experts=256, top_k=6, hidden=4096)


# ---- host preparation ----


def test_scales_are_swizzled_like_the_reference():
    fmt = _format(2, (64, 64))  # w13 scale rows 128 x 4 cols, w2 scale rows 64 -> 128 x 4 cols
    generator = torch.Generator().manual_seed(3)
    raw13 = torch.randint(0x30, 0x40, (2, 128, 4), dtype=torch.uint8, generator=generator).view(torch.float8_e4m3fn)
    out = fmt.prepare_host_rows(None, "w13_blockscale_swizzled", raw13)
    assert out.dtype == torch.float8_e4m3fn and tuple(out.shape) == (2, 128, 4)
    for expert in range(2):
        assert torch.equal(out[expert].view(torch.uint8), _reference_swizzle(raw13[expert].view(torch.uint8)))
    # Checkpoint-order and swizzled scales have one shape here, so the input is always taken as checkpoint order.
    assert not torch.equal(out.view(torch.uint8), raw13.view(torch.uint8))
    # w2: 64 logical rows pad to 128 (the pad is zero), 64/16 = 4 columns.
    raw2 = torch.randint(0x30, 0x40, (2, 64, 4), dtype=torch.uint8, generator=generator).view(torch.float8_e4m3fn)
    out2 = fmt.prepare_host_rows(None, "w2_blockscale_swizzled", raw2)
    assert tuple(out2.shape[1:]) == fmt.tensor_specs(None)[3].row_shape == (128, 4)
    padded = torch.zeros(128, 4, dtype=torch.uint8)
    padded[:64] = raw2[1].view(torch.uint8)
    assert torch.equal(out2[1].view(torch.uint8), _reference_swizzle(padded))


def test_scales_of_the_wrong_dtype_or_shape_are_refused():
    fmt = _format(2, (64, 64))
    with pytest.raises(TypeError, match="E4M3"):
        fmt.prepare_host_rows(None, "w13_blockscale_swizzled", torch.zeros(2, 128, 4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="not the checkpoint's"):
        fmt.prepare_host_rows(None, "w13_blockscale_swizzled", torch.zeros(2, 100, 4, dtype=torch.float8_e4m3fn))


def test_alphas_are_widened_to_a_padded_row_with_the_value_repeated():
    fmt = _format(3, (64, 64))
    alphas = torch.tensor([0.5, 2.0, 3.0])
    for shape in ((3,), (3, 1)):
        rows = fmt.prepare_host_rows(None, "g1_alphas", alphas.reshape(shape))
        assert rows.shape == (3, SIDE_ROW_ELEMENTS) and rows.dtype == torch.float32 and rows.is_contiguous()
        assert torch.equal(rows[:, 0], alphas) and bool((rows == alphas[:, None]).all())
    with pytest.raises(TypeError, match="fp32"):
        fmt.prepare_host_rows(None, "g2_alphas", alphas.to(torch.bfloat16))


def test_weights_pass_through_as_uint8_and_unknown_names_are_refused():
    fmt = _format(2, (64, 64))
    signed = torch.zeros(2, 128, 32, dtype=torch.int8)
    assert fmt.prepare_host_rows(None, "w13_weight", signed).dtype == torch.uint8
    with pytest.raises(ValueError, match="no streamed tensor 'w13_weight_scale'"):
        fmt.prepare_host_rows(None, "w13_weight_scale", torch.zeros(1))


# ---- row source ----


def test_the_expert_file_cache_is_not_a_row_source_and_the_error_says_why():
    fmt = _format(8, (64, 64))
    for call in (
        lambda: fmt.default_row_source(None, fmt.tensor_specs(None), "auto"),
        lambda: fmt.engine_tables({}),
    ):
        with pytest.raises(NotImplementedError) as error:
            call()
        text = str(error.value)
        assert "four per-tensor files" in text and "g1_alphas" in text and "row-image" in text


# ---- CPU experts ----


def _slabs(capacity=4, hidden=64, inter=64):
    fmt = _format(capacity, (hidden, inter))
    return {s.name: torch.zeros((capacity, *s.row_shape), dtype=s.dtype) for s in fmt.tensor_specs(None)}


def test_the_cpu_trait_is_registered_and_names_the_engines_slabs():
    trait = get_format("nvfp4").cpu_trait(source="target")
    assert isinstance(trait, Nvfp4EngineCpuTrait)
    assert trait.name == "nvfp4" and trait.slab_names == NVFP4_STREAMED_NAMES
    assert trait.x_dtype == torch.float16 and trait.out_dtype == torch.float32
    from sglang.srt.layers.moe.cpu_experts.service import cpu_trait_for

    assert isinstance(cpu_trait_for("nvfp4", source="draft"), Nvfp4EngineCpuTrait)
    with pytest.raises(ValueError, match="no expert format"):
        cpu_trait_for("nvfp9", source="target")


def test_the_trait_reads_the_shape_off_the_slabs():
    slabs = _slabs(4, 128, 256)
    assert Nvfp4EngineCpuTrait.dims_of(slabs) == (128, 256)
    assert Nvfp4EngineCpuTrait().hidden_size(slabs) == 128


def test_a_layer_without_activation_scales_is_refused_and_one_with_them_gets_the_kernels_slab_order():
    module = SimpleNamespace(
        nvfp4_cpu_params=lambda layout, s13, s2: f"{layout}:{s13}:{s2}".encode(),
        nvfp4_cpu_kernel_address=lambda: 4242,
    )
    trait = Nvfp4EngineCpuTrait(module=module)
    trait.act_limit = 0.0
    slabs = _slabs(4)
    with pytest.raises(ValueError, match="activation-scale reciprocals"):
        trait.layer_spec(slabs, 4)
    trait.set_layer_scales(slabs["w13_weight"], 0.25, 4.0)
    spec = trait.layer_spec(slabs, 4)
    assert spec.capacity == 4 and (spec.hidden, spec.intermediate) == (64, 64) and spec.act_limit == 0.0
    # Kernel slab order w13, w2, sf13, sf2, gate_alpha, down_alpha, then the absent up_alpha.
    assert spec.slabs[:6] == tuple(
        (slabs[name].data_ptr(), slabs[name][0].numel() * slabs[name].element_size()) for name in NVFP4_STREAMED_NAMES
    )
    assert spec.slabs[4][1] == spec.slabs[5][1] == 512 and spec.slabs[6] == (0, 0)
    assert spec.params == b"0:0.25:4.0" and trait.kernel_address() == 4242


def test_the_trait_needs_the_layers_activation_limit_first():
    trait = Nvfp4EngineCpuTrait(module=SimpleNamespace(nvfp4_cpu_params=lambda *a: b""))
    slabs = _slabs(2)
    trait.set_layer_scales(slabs["w13_weight"], 1.0, 1.0)
    with pytest.raises(ValueError, match="activation limit"):
        trait.layer_spec(slabs, 2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
