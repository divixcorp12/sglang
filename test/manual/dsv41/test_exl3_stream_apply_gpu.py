"""The streamed Exl3MoEMethod.apply equals the resident eager loop with the real kernels (GPU, Window C)."""

import json
import os
import types

import pytest
import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.test.dsv41_fake_exl3 import HIDDEN, INTER, write_fake_exl3

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC")),
    reason="needs a GPU and SGLANG_EXL3_SRC",
)

CFG = {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
NUM_EXPERTS = 6
LAYER = 1


@pytest.mark.parametrize(
    "tokens, max_gather_rows, hot_experts",
    [
        (1, None, [3, 4]),  # one chunk, mixed hits (hot cache + pinned tier)
        (5, None, [3, 4]),
        (5, 2, [3, 4]),  # >= 3 chunks reuse the staging
        (1, None, [2, 0, 1]),  # all hot, slots [2,0,1]: row_of_source is [1,2,0]
    ],
)
@pytest.mark.parametrize("route_plan", [False, True])
def test_streamed_apply_equals_resident(tmp_path, tokens, max_gather_rows, hot_experts, route_plan):
    from sglang.srt.layers.moe.exl3_expert_format import exl3_expert_layout_for
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
    from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
    from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_moe_loop

    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=NUM_EXPERTS, finite=True)
    exl3_expert_layout_for.cache_clear()
    with (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_EXPERT_DIR.override(str(tmp_path)),
        envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
        envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"),
        envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan),
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = torch.nn.Module()
        layer.layer_id = LAYER
        method.create_weights(layer, NUM_EXPERTS, HIDDEN, INTER, torch.bfloat16)
        method.process_weights_after_loading(layer)
    streamer = layer._nvfp4_expert_streamer
    if max_gather_rows is not None:
        streamer.format.max_gather_rows = max_gather_rows
    ExpertHotCache(streamer, len(hot_experts)).reassign(hot_experts)
    ExpertPinnedHostCache(streamer, max(2, len(hot_experts)))
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    generator = torch.Generator().manual_seed(tokens)
    x = (torch.randn(tokens, HIDDEN, generator=generator) * 0.05).to(torch.bfloat16).cuda()
    if sorted(hot_experts) == [0, 1, 2]:
        topk_ids = torch.tensor([sorted(hot_experts)] * tokens, dtype=torch.int32).cuda()
    elif max_gather_rows is not None:  # all six experts route: three chunks of two
        topk_ids = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [0, 3, 5], [1, 4, 2], [5, 0, 4]][:tokens], dtype=torch.int32
        ).cuda()
    else:
        topk_ids = torch.stack(
            [torch.randperm(NUM_EXPERTS, generator=generator)[:3] for _ in range(tokens)]
        ).to(torch.int32).cuda()
    topk_weights = torch.rand(tokens, 3, generator=generator).cuda()
    dispatch = types.SimpleNamespace(
        hidden_states=x,
        topk_output=types.SimpleNamespace(topk_weights=topk_weights, topk_ids=topk_ids),
    )
    chunks = []
    iterate = streamer.iter_gather_experts
    iterate_host = streamer.iter_gather_experts_host

    def recording(ids, **kwargs):
        for chunk, row_of_source, rows in iterate(ids, **kwargs):
            chunks.append(("device", chunk.tolist()))
            yield chunk, row_of_source, rows

    def recording_host(ids, experts, **kwargs):
        for chunk, row_of_source, rows in iterate_host(ids, experts, **kwargs):
            chunks.append(("host", chunk))
            yield chunk, row_of_source, rows

    streamer.iter_gather_experts = recording
    streamer.iter_gather_experts_host = recording_host
    got = method.apply(layer, dispatch).hidden_states
    if max_gather_rows == 2:
        assert len(chunks) == 3
    assert {path for path, _ in chunks} == {"host" if route_plan else "device"}

    with open(tmp_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    def tensors(expert, w):
        parts = {}
        for kind in ("trellis", "suh", "svh"):
            name = f"layers.{LAYER}.ffn.experts.{expert}.{w}.{kind}"
            with safe_open(str(tmp_path / weight_map[name]), "pt") as f:
                parts[kind] = f.get_tensor(name).cuda()
        return Exl3Tensors(mul1=True, **parts)

    w13 = [(tensors(e, "w1"), tensors(e, "w3")) for e in range(NUM_EXPERTS)]
    w2 = [tensors(e, "w2") for e in range(NUM_EXPERTS)]
    want = exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, 10.0)
    assert torch.isfinite(want.float()).all()
    assert torch.equal(got, want)



@pytest.mark.parametrize("route_plan", [False, True])
def test_many_experts_route_plan_equals_resident(tmp_path, route_plan):
    """80 experts, 40 tokens x 6 routes with a repeated expert and dropped routes: two 64-expert chunks, through
    an 8-expert hot cache and a pinned tier that holds them all."""
    from sglang.srt.layers.moe.exl3_expert_format import exl3_expert_layout_for
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
    from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
    from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_moe_loop

    num_experts, tokens, topk = 80, 40, 6
    write_fake_exl3(str(tmp_path), num_layers=2, num_experts=num_experts, finite=True)
    exl3_expert_layout_for.cache_clear()
    with (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_EXPERT_DIR.override(str(tmp_path)),
        envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
        envs.SGLANG_MOE_EXPERT_FILE_READER.override("uring"),
        envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan),
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = torch.nn.Module()
        layer.layer_id = LAYER
        method.create_weights(layer, num_experts, HIDDEN, INTER, torch.bfloat16)
        method.process_weights_after_loading(layer)
    streamer = layer._nvfp4_expert_streamer
    streamer.format.max_gather_rows = 64
    ExpertHotCache(streamer, 8).reassign([3, 17, 29, 41, 50, 62, 71, 79])
    ExpertPinnedHostCache(streamer, num_experts)
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    generator = torch.Generator().manual_seed(7)
    topk_ids = torch.stack([torch.randperm(num_experts, generator=generator)[:topk] for _ in range(tokens)])
    topk_ids = topk_ids.to(torch.int32)
    topk_ids[5, 3] = -1
    topk_ids[9, 2] = topk_ids[9, 1]
    assert len(set(topk_ids[topk_ids >= 0].tolist())) > 64
    x = (torch.randn(tokens, HIDDEN, generator=generator) * 0.05).to(torch.bfloat16).cuda()
    topk_ids = topk_ids.cuda()
    topk_weights = torch.rand(tokens, topk, generator=generator).cuda()
    dispatch = types.SimpleNamespace(
        hidden_states=x,
        topk_output=types.SimpleNamespace(topk_weights=topk_weights, topk_ids=topk_ids),
    )
    got = method.apply(layer, dispatch).hidden_states

    with open(tmp_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    def tensors(expert, w):
        parts = {}
        for kind in ("trellis", "suh", "svh"):
            name = f"layers.{LAYER}.ffn.experts.{expert}.{w}.{kind}"
            with safe_open(str(tmp_path / weight_map[name]), "pt") as f:
                parts[kind] = f.get_tensor(name).cuda()
        return Exl3Tensors(mul1=True, **parts)

    w13 = [(tensors(e, "w1"), tensors(e, "w3")) for e in range(num_experts)]
    w2 = [tensors(e, "w2") for e in range(num_experts)]
    want = exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, 10.0)
    assert torch.isfinite(want.float()).all()
    assert torch.equal(got, want)


def _planned_inputs(experts=8, hidden=5120, inter=2304, tokens=40):
    from sglang.srt.layers.quantization.exl3_ops import Exl3RoutePlan, random_exl3_tensors

    w13 = [
        (
            random_exl3_tensors(hidden, inter, 3, device="cuda", seed=3 * e),
            random_exl3_tensors(hidden, inter, 3, device="cuda", seed=3 * e + 1),
        )
        for e in range(experts)
    ]
    w2 = [random_exl3_tensors(inter, hidden, 3, device="cuda", seed=3 * e + 2) for e in range(experts)]
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    topk_ids = torch.stack([torch.randperm(experts, device="cuda")[:6] for _ in range(tokens)]).to(torch.int32)
    topk_weights = torch.rand(tokens, 6, device="cuda")
    return x, topk_ids, topk_weights, w13, w2, Exl3RoutePlan.from_topk(topk_ids)


def test_planned_chunk_body_never_syncs():
    """With the plan built, a chunk's expert compute issues no host sync, so the host can run ahead of its gather."""
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_accumulate_planned

    x, _, topk_weights, w13, w2, plan = _planned_inputs()
    out = torch.zeros(x.shape[0], x.shape[1], device="cuda")
    exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, 10.0, plan.experts)  # warm the kernels
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, 10.0, plan.experts)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_the_where_loop_does_sync():
    """The control for the test above: sync debug mode does catch the per-expert torch.where readback."""
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_accumulate

    x, topk_ids, topk_weights, w13, w2, plan = _planned_inputs()
    out = torch.zeros(x.shape[0], x.shape[1], device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match="synchroniz"):
            exl3_moe_accumulate(out, x, topk_weights, topk_ids, w13, w2, 10.0, plan.experts)
    finally:
        torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
