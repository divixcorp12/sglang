"""NVFP4 ``moe_apply`` over hot-cache slot buffers against the dense FlashInfer CUTLASS NVFP4 path, at real shapes (GPU).

The one-layer test of the spec. The slot ABI is the dense call with remapped ids: the slot buffers stand in for the
``[E, ...]`` weight tensors, ``E`` is the slot count and the expert ids are slot ids, so on the same experts the two
outputs must be identical (the same kernel, the same bytes). Shapes are the real ones: DSV4.1 (hidden 4096,
intermediate 2048) and MiMo V2.6 Pro (6144 / 2048), with synthetic weights.

The dense side swizzles scales on the GPU with ``swizzle_blockscale`` and keeps per-expert alphas of size E, as
``ModelOptNvFp4FusedMoEMethod`` does; the slot side takes the same checkpoint-order bytes through the format's host
``prepare_host_rows`` (scales swizzled on the CPU, alphas widened to 512 B rows), so this also checks that the host
preparation feeds the kernel the same bytes. The streaming through the engine (copy engine, RAM tier, NVMe reads) needs
NVFP4 row images and is not exercised here.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="NVFP4 slot-apply tests require a CUDA GPU.",
)

EXPERTS = 8
SHAPES = {"dsv41": (4096, 2048), "mimo": (6144, 2048)}
# The activation scales (reciprocals, as the quant method keeps them): ~ 448 * 6 / amax for standard-normal inputs and
# for the SwiGLU outputs.
W13_INPUT_SCALE_QUANT, W2_INPUT_SCALE_QUANT = 672.0, 96.0


def _require_blackwell_flashinfer():
    pytest.importorskip("flashinfer")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 FlashInfer CUTLASS MoE needs Blackwell")


def _random_experts(hidden, inter, generator):
    """Random checkpoint-layout experts: packed E2M1 bytes, E4M3 scales near 1.0 (bytes 0x30..0x3f: 0.5..1.9), and
    alphas of a magnitude that keeps the outputs finite."""

    def weights(rows, cols):
        return torch.randint(0, 256, (EXPERTS, rows, cols), dtype=torch.uint8, generator=generator)

    def scales(rows, cols):
        return torch.randint(0x30, 0x40, (EXPERTS, rows, cols), dtype=torch.uint8, generator=generator).view(
            torch.float8_e4m3fn
        )

    def alphas():
        return (torch.rand(EXPERTS, generator=generator) * 0.004 + 0.001).to(torch.float32)

    return {
        "w13_weight": weights(2 * inter, hidden // 2),
        "w2_weight": weights(hidden, inter // 2),
        "w13_scale": scales(2 * inter, hidden // 16),
        "w2_scale": scales(hidden, inter // 16),
        "g1_alphas": alphas(),
        "g2_alphas": alphas(),
    }


def _runner_config(swiglu_limit):
    return SimpleNamespace(
        activation="silu",
        is_gated=True,
        swiglu_limit=swiglu_limit,
        gemm1_clamp_limit=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        apply_router_weight_on_input=False,
        routed_scaling_factor=None,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )


@pytest.fixture(autouse=True)
def no_distributed_state(monkeypatch):
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass

    # No distributed state in a unit test: a world-size-1 group makes use_symmetric_memory a no-op.
    monkeypatch.setattr(
        flashinfer_cutlass, "get_parallel", lambda: SimpleNamespace(tp_group=SimpleNamespace(world_size=1))
    )
    monkeypatch.setattr(flashinfer_cutlass, "is_allocation_symmetric", lambda: True)


def _scale_scalars():
    return (
        torch.tensor(W13_INPUT_SCALE_QUANT, dtype=torch.float32, device="cuda"),
        torch.tensor(W2_INPUT_SCALE_QUANT, dtype=torch.float32, device="cuda"),
    )


def _dense_reference(runner_config, experts, x, topk_ids, topk_weights):
    """The dense method's call: scales swizzled on the GPU, alphas of size E, swiglu vectors of size E."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMoeQuantInfo,
        fused_experts_none_to_flashinfer_cutlass,
        materialize_swiglu_params_for_cutlass,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    w13_in, w2_in = _scale_scalars()
    alpha, beta, limit = materialize_swiglu_params_for_cutlass(runner_config, EXPERTS, torch.device("cuda"))
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="fp4",
        w13_weight=experts["w13_weight"].cuda(),
        w2_weight=experts["w2_weight"].cuda(),
        output_dtype=torch.bfloat16,
        quant_scales=[
            w13_in,
            swizzle_blockscale(experts["w13_scale"].cuda()),
            experts["g1_alphas"].cuda(),
            w2_in,
            swizzle_blockscale(experts["w2_scale"].cuda()),
            experts["g2_alphas"].cuda(),
        ],
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        apply_routed_scaling_factor=False,
    )
    dispatch = StandardDispatchOutput(
        hidden_states=x, hidden_states_scale=None, topk_output=StandardTopKOutput(topk_weights, topk_ids, None)
    )
    return fused_experts_none_to_flashinfer_cutlass(dispatch, quant_info, runner_config).hidden_states


def _slot_tensors(fmt, experts, resident):
    """The hot cache's buffers for the experts in ``resident``: slot i holds expert resident[i], rows built by the
    format's own host preparation (scales swizzled on the CPU, alphas widened to their 512 B rows)."""
    index = torch.tensor(resident)

    def prepared(name, tensor):
        return fmt.prepare_host_rows(None, name, tensor[index]).cuda()

    return {
        "w13_weight": prepared("w13_weight", experts["w13_weight"]),
        "w2_weight": prepared("w2_weight", experts["w2_weight"]),
        "w13_blockscale_swizzled": prepared("w13_blockscale_swizzled", experts["w13_scale"]),
        "w2_blockscale_swizzled": prepared("w2_blockscale_swizzled", experts["w2_scale"]),
        "g1_alphas": prepared("g1_alphas", experts["g1_alphas"]),
        "g2_alphas": prepared("g2_alphas", experts["g2_alphas"]),
    }


def _layer():
    w13_in, w2_in = _scale_scalars()
    return SimpleNamespace(w13_input_scale_quant=w13_in, w2_input_scale_quant=w2_in)


def _routes(resident, tokens, top_k):
    """Each token routes ``top_k`` distinct resident experts: the expert ids, and the same routes as slot ids."""
    expert_ids = torch.stack(
        [torch.tensor(resident)[torch.randperm(len(resident))[:top_k]] for _ in range(tokens)]
    ).to(torch.int32)
    slot_of = {expert: slot for slot, expert in enumerate(resident)}
    slot_ids = torch.tensor([[slot_of[int(e)] for e in row] for row in expert_ids], dtype=torch.int32)
    return expert_ids, slot_ids


@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("model", sorted(SHAPES))
def test_moe_apply_over_slots_equals_dense(model, tokens, swiglu_limit):
    _require_blackwell_flashinfer()
    from sglang.srt.layers.moe.expert_engine.formats.nvfp4 import Nvfp4ExpertFormat

    hidden, inter = SHAPES[model]
    generator = torch.Generator().manual_seed(1234 + tokens)
    experts = _random_experts(hidden, inter, generator)
    resident = [3, 0, 6, 1, 5]  # five hot-cache slots out of eight experts, in a non-identity order
    runner_config = _runner_config(swiglu_limit)
    fmt = Nvfp4ExpertFormat(
        layer_id=1, num_experts=EXPERTS, hidden=hidden, intermediate=inter, runner_config=runner_config
    )
    slots = _slot_tensors(fmt, experts, resident)
    assert slots["w13_weight"].shape[0] == len(resident)
    assert slots["g1_alphas"].shape == (len(resident), 128)

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    expert_ids, slot_ids = _routes(resident, tokens, top_k=4)
    weights = torch.softmax(torch.randn(tokens, 4), dim=-1).to(torch.float32)

    dense = _dense_reference(runner_config, experts, x, expert_ids.cuda(), weights.cuda())
    sloted = fmt.moe_apply(_layer(), x, slot_ids.cuda(), weights.cuda(), slots)

    assert dense.shape == sloted.shape == (tokens, hidden) and sloted.dtype == torch.bfloat16
    assert torch.isfinite(dense.float()).all() and bool((dense != 0).any())
    # The same kernel with remapped ids: bit-identical, not merely close.
    torch.testing.assert_close(sloted, dense, rtol=0, atol=0)


def test_keep_zero_drops_the_routed_output():
    _require_blackwell_flashinfer()
    from sglang.srt.layers.moe.expert_engine.formats.nvfp4 import Nvfp4ExpertFormat

    hidden, inter = SHAPES["mimo"]
    experts = _random_experts(hidden, inter, torch.Generator().manual_seed(7))
    runner_config = _runner_config(None)
    fmt = Nvfp4ExpertFormat(
        layer_id=1, num_experts=EXPERTS, hidden=hidden, intermediate=inter, runner_config=runner_config
    )
    slots = _slot_tensors(fmt, experts, [0, 1, 2])
    x = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
    slot_ids = torch.tensor([[0, 1], [2, 1]], dtype=torch.int32, device="cuda")
    weights = torch.full((2, 2), 0.5, dtype=torch.float32, device="cuda")
    args = SimpleNamespace(
        streamer=SimpleNamespace(hot_cache=SimpleNamespace(tensors=slots)),
        keep=torch.zeros(1, dtype=torch.float32, device="cuda"),
    )
    dropped = fmt.moe_apply(_layer(), x, slot_ids, weights, args)
    assert bool((dropped == 0).all())
    args.keep = torch.ones(1, dtype=torch.float32, device="cuda")
    kept = fmt.moe_apply(_layer(), x, slot_ids, weights, args)
    assert bool((kept != 0).any())


def test_a_layer_without_activation_scales_is_refused():
    _require_blackwell_flashinfer()
    from sglang.srt.layers.moe.expert_engine.formats.nvfp4 import Nvfp4ExpertFormat

    fmt = Nvfp4ExpertFormat(layer_id=1, num_experts=EXPERTS, hidden=256, intermediate=256, runner_config=_runner_config(None))
    layer = SimpleNamespace(w13_input_scale_quant=None, w2_input_scale_quant=None)
    slots = {name: torch.zeros(1) for name in fmt.names}
    with pytest.raises(RuntimeError, match="activation scales"):
        fmt.moe_apply(layer, torch.zeros(1, 256), torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1), slots)
