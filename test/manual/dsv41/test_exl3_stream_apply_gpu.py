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
        (1, None, [0, 1, 2]),  # every routed expert is hot: rows are hot-cache slots
    ],
)
def test_streamed_apply_equals_resident(tmp_path, tokens, max_gather_rows, hot_experts):
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
    ):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG))
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
    if hot_experts == [0, 1, 2]:
        topk_ids = torch.tensor([hot_experts] * tokens, dtype=torch.int32).cuda()
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

    w13 = [(tensors(e, "w1"), tensors(e, "w3")) for e in range(NUM_EXPERTS)]
    w2 = [tensors(e, "w2") for e in range(NUM_EXPERTS)]
    want = exl3_moe_loop(x, topk_weights, topk_ids, w13, w2, 10.0)
    assert torch.isfinite(want.float()).all()
    assert torch.equal(got, want)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
