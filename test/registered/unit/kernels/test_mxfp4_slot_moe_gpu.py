"""MXFP4 ``moe_apply`` over hot-cache slot buffers against the dense FlashInfer CUTLASS MXFP4 path (GPU, SM120).

The slot ABI is the dense call with remapped ids: the slot buffers stand in for the ``[E, ...]`` weight tensors,
``E`` is the slot count and the expert ids are slot ids. On the same experts the two outputs must therefore be
identical. The dense side interleaves scales with ``flashinfer.block_scale_interleave`` (what
``Mxfp4FlashinferCutlassMoEMethod.process_weights_after_loading`` does); the slot side uses the format's host
``prepare_host_rows``, so this also checks that the host interleave feeds the kernel the same bytes.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MXFP4 slot-apply tests require a CUDA GPU.",
)

HIDDEN, INTER, EXPERTS = 256, 256, 8


def _require_sm120_flashinfer():
    pytest.importorskip("flashinfer")
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("MXFP4 streaming runs on SM120 (MXFP8 activations x MXFP4 weights)")


def _random_experts(generator):
    """Random checkpoint-layout experts: packed E2M1 weights and E8M0 scales near 1.0 (bytes 120..130)."""

    def weights(rows, cols):
        return torch.randint(0, 256, (EXPERTS, rows, cols), dtype=torch.uint8, generator=generator).view(
            torch.int8
        )

    def scales(rows, cols):
        return torch.randint(120, 131, (EXPERTS, rows, cols), dtype=torch.uint8, generator=generator)

    return {
        "w13_weight": weights(2 * INTER, HIDDEN // 2),
        "w13_weight_scale": scales(2 * INTER, HIDDEN // 32),
        "w2_weight": weights(HIDDEN, INTER // 2),
        "w2_weight_scale": scales(HIDDEN, INTER // 32),
    }


@pytest.fixture
def runner_config(monkeypatch):
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass

    # No distributed state in a unit test: a world-size-1 group makes use_symmetric_memory a no-op.
    monkeypatch.setattr(
        flashinfer_cutlass, "get_parallel", lambda: SimpleNamespace(tp_group=SimpleNamespace(world_size=1))
    )
    monkeypatch.setattr(flashinfer_cutlass, "is_allocation_symmetric", lambda: True)
    return SimpleNamespace(moe_tp_size=1, moe_tp_rank=0, moe_ep_size=1, moe_ep_rank=0)


def _dense_reference(runner_config, experts, x, topk_ids, topk_weights, swiglu_limit):
    """The dense method's call: scales interleaved by FlashInfer, per-expert side tensors sized to E."""
    from flashinfer import block_scale_interleave

    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_cutlass,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    def interleaved(scale):
        scale = scale.cuda()
        return block_scale_interleave(scale).reshape_as(scale)

    quant_info = FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=experts["w13_weight"].cuda(),
        w2_weight=experts["w2_weight"].cuda(),
        w13_weight_scale=interleaved(experts["w13_weight_scale"]),
        w2_weight_scale=interleaved(experts["w2_weight_scale"]),
        mxfp4_weight_global_scale=torch.ones(EXPERTS, dtype=torch.float32, device="cuda"),
        swiglu_limit=(
            None
            if swiglu_limit is None
            else torch.full((EXPERTS,), swiglu_limit, dtype=torch.float32, device="cuda")
        ),
    )
    dispatch = StandardDispatchOutput(
        hidden_states=x, hidden_states_scale=None, topk_output=StandardTopKOutput(topk_weights, topk_ids, None)
    )
    return _fused_experts_flashinfer_mxfp4_cutlass(dispatch, quant_info, runner_config).hidden_states


def _slot_tensors(fmt, experts, resident):
    """The hot cache's buffers for the experts in ``resident``: slot i holds expert resident[i], rows built by the
    format's own host preparation (scales interleaved, side tensors widened to their 512 B rows)."""
    index = torch.tensor(resident)
    side = lambda name: torch.full((len(resident),), fmt.side_row_value(name))
    return {
        "w13_weight": fmt.prepare_host_rows(None, "w13_weight", experts["w13_weight"][index]).cuda(),
        "w13_weight_scale": fmt.prepare_host_rows(None, "w13_weight_scale", experts["w13_weight_scale"][index]).cuda(),
        "w2_weight": fmt.prepare_host_rows(None, "w2_weight", experts["w2_weight"][index]).cuda(),
        "w2_weight_scale": fmt.prepare_host_rows(None, "w2_weight_scale", experts["w2_weight_scale"][index]).cuda(),
        "mxfp4_weight_global_scale": fmt.prepare_host_rows(None, "mxfp4_weight_global_scale", side("mxfp4_weight_global_scale")).cuda(),
        "swiglu_limit": fmt.prepare_host_rows(None, "swiglu_limit", side("swiglu_limit")).cuda(),
    }


@pytest.mark.parametrize("swiglu_limit", [None, 7.0])
@pytest.mark.parametrize("tokens", [1, 5])
def test_moe_apply_over_slots_equals_dense(runner_config, tokens, swiglu_limit):
    _require_sm120_flashinfer()
    from sglang.srt.layers.moe.expert_engine.formats.mxfp4 import Mxfp4ExpertFormat

    generator = torch.Generator().manual_seed(1234 + tokens)
    experts = _random_experts(generator)
    resident = [3, 0, 6, 1, 5]  # five hot-cache slots out of eight experts, in a non-identity order
    fmt = Mxfp4ExpertFormat(
        layer_id=1,
        num_experts=EXPERTS,
        hidden=HIDDEN,
        intermediate=INTER,
        swiglu_limit=swiglu_limit,
        runner_config=runner_config,
    )
    slots = _slot_tensors(fmt, experts, resident)
    assert slots["w13_weight"].shape[0] == len(resident)

    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=None)
    top_k = 4
    # Each token routes top_k distinct resident experts.
    expert_ids = torch.stack(
        [torch.tensor(resident)[torch.randperm(len(resident))[:top_k]] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k), dim=-1).to(torch.float32)
    slot_of = {expert: slot for slot, expert in enumerate(resident)}
    slot_ids = torch.tensor([[slot_of[int(e)] for e in row] for row in expert_ids], dtype=torch.int32)

    dense = _dense_reference(
        runner_config, experts, x, expert_ids.cuda(), weights.cuda(), swiglu_limit
    )
    sloted = fmt.moe_apply(None, x, slot_ids.cuda(), weights.cuda(), slots)

    assert dense.shape == sloted.shape == (tokens, HIDDEN)
    assert torch.isfinite(dense.float()).all()
    # The same kernel with remapped ids: bit-identical, not merely close.
    torch.testing.assert_close(sloted, dense, rtol=0, atol=0)


def test_keep_zero_drops_the_routed_output(runner_config):
    _require_sm120_flashinfer()
    from sglang.srt.layers.moe.expert_engine.formats.mxfp4 import Mxfp4ExpertFormat

    experts = _random_experts(torch.Generator().manual_seed(7))
    fmt = Mxfp4ExpertFormat(
        layer_id=1, num_experts=EXPERTS, hidden=HIDDEN, intermediate=INTER, runner_config=runner_config
    )
    slots = _slot_tensors(fmt, experts, [0, 1, 2])
    x = torch.randn(2, HIDDEN, dtype=torch.bfloat16, device="cuda")
    slot_ids = torch.tensor([[0, 1], [2, 1]], dtype=torch.int32, device="cuda")
    weights = torch.full((2, 2), 0.5, dtype=torch.float32, device="cuda")
    args = SimpleNamespace(
        streamer=SimpleNamespace(hot_cache=SimpleNamespace(tensors=slots)),
        keep=torch.zeros(1, dtype=torch.float32, device="cuda"),
    )
    dropped = fmt.moe_apply(None, x, slot_ids, weights, args)
    assert bool((dropped == 0).all())
    args.keep = torch.ones(1, dtype=torch.float32, device="cuda")
    kept = fmt.moe_apply(None, x, slot_ids, weights, args)
    assert bool((kept != 0).any())
