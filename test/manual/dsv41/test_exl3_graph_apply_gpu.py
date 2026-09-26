"""In-graph EXL3 MoE over pinned-tier graph gather (GPU, window).

Fake finite EXL3 checkpoint (1 layer, 12 experts); pinned tier of 8 rows (experts 0-7),
hot cache of 3 slots (experts 0-2) + 6 scratch; top-6 routes. Captures streamer.gather +
Exl3FusedMoE.run in a CUDA graph and replays it for new routes.

Routes whose rows are all in RAM: every slot the gather names holds exactly the bytes a
fresh read of its expert from the checkpoint gives; the output meets the P2 probe's bars
(plan Design decision D7) against an fp32 reference over those rows (rel <= 1.2e-2 and
<= 2 * rel(loop) + 1e-3, the loop being Exl3MoEMethod._apply_streamed / exl3_moe_loop)
and a tighter bar for these fixed rows (5e-3; measured 2.2e-3); and a replay equals an
eager _apply_graph call bit for bit. Graph vs loop is bounded loosely (the loop is the
less accurate arm: ~1.5e-2 here, probe 1.15e-2 vs fused 1.16e-3). The loop reads the
same pinned slabs but not the device translate, segment copy or slot map, so that bound
is a second check on slot and segment bugs; the exact row check is the first.

A route with a row outside the pinned tier (P3 alone, no option C): keep drops to 0,
ram_miss counts the miss, no expert runs and the output is exactly zero; the next
all-in-RAM replay has keep 1 again and repeats its earlier output bit for bit.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 12, 6
PINNED_EXPERTS = 8  # experts 0-7 are in RAM
ACT_LIMIT = 10.0
REL_BOUND = 1.2e-2  # the probe's bar (D7)
REL_TIGHT = 5e-3  # these fixed rows: 2.1e-3..2.2e-3 measured
# Graph vs loop: 1.51e-2..1.56e-2 measured on these fixed rows. One wrong expert moves the
# loop's output by 3.5e-2 (the lowest-weight route) to 1.36; permuted weights by 1.10.
LOOSE_BOUND = 2.5e-2


def _layer(tmp_path):
    """(layer, streamer, source rows {name: [EXPERTS, ...]} read afresh from the checkpoint)."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layer = torch.nn.Module()
    layer.layer_id = 0
    layer.top_k = TOP_K
    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
        streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
    pinned = ExpertPinnedHostCache(streamer, PINNED_EXPERTS)
    hot = ExpertHotCache(streamer, 3, scratch_rows=TOP_K)
    hot.reassign([0, 1, 2])
    pinned.ensure_rows(torch.arange(PINNED_EXPERTS, device="cuda"))
    streamer.enable_graph_gather(TOP_K)
    layer._nvfp4_expert_streamer = streamer
    layer._exl3_allow_p3_only = True  # test configuration: P3's backend without option C (Task 14)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    source = {name: torch.empty((EXPERTS,) + spec.row_shape, dtype=spec.dtype) for name, spec in specs.items()}
    Exl3ShardRowSource.for_layer(layout, 0, fmt.segment_map(), direct=False).read(
        torch.arange(EXPERTS, dtype=torch.long), source
    )
    return layer, streamer, source


def _reference(x, weights, slots, tensors):
    """fp32 routed output over the hot-cache rows at ``slots`` (the probe's reference)."""
    import torch.nn.functional as F

    from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_linear_reference

    def view(prefix, slot, part):
        return Exl3Tensors(
            trellis=tensors[f"{prefix}_trellis"][slot, part],
            suh=tensors[f"{prefix}_suh"][slot, part],
            svh=tensors[f"{prefix}_svh"][slot, part],
            mul1=True,
        )

    x16 = x.to(torch.float16)
    out = torch.zeros((1, x.shape[1]), dtype=torch.float32, device=x.device)
    for k, slot in enumerate(slots.tolist()):
        gate = exl3_linear_reference(x16, view("w13", slot, 0)).clamp(max=ACT_LIMIT)
        up = exl3_linear_reference(x16, view("w13", slot, 1)).clamp(-ACT_LIMIT, ACT_LIMIT)
        h = F.silu(gate) * up * weights[k].float()
        out += exl3_linear_reference(h.to(torch.float16), view("w2", slot, 0))
    return out


def _rel(y, ref):
    return float((y.float() - ref).norm() / ref.norm())


@pytest.mark.parametrize("fused_in_topk", [False, True])
def test_apply_casts_and_scales_bit_identically_with_the_cast_fusion_flag(tmp_path, fused_in_topk):
    """Exl3MoEMethod.apply, SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION on against off: the routed output after the cast
    to bf16 and the routed_scaling_factor, which the flag moves into one kernel."""
    from types import SimpleNamespace

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod

    layer, _, _ = _layer(tmp_path)
    layer.should_fuse_routed_scaling_factor_in_topk = fused_in_topk
    config = Exl3Config.from_config(
        {"quant_method": "exl3", "version": "1.4.2", "bits": 3.02, "head_bits": 6, "codebook": "mul1"}
    )
    methods = {}
    for flag in (False, True):
        with envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.override(flag):
            methods[flag] = Exl3MoEMethod(config, streamed=True)
        methods[flag].moe_runner_config = SimpleNamespace(
            apply_router_weight_on_input=False, swiglu_limit=ACT_LIMIT, routed_scaling_factor=1.5
        )
    gen = torch.Generator(device="cpu").manual_seed(3)
    for route in ([0, 3, 5, 1, 7, 6], [2, 4, 6, 0, 1, 3]):
        x = (torch.randn((1, HIDDEN), generator=gen) * 4).to("cuda", torch.bfloat16)
        weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
        ids = torch.tensor([route], device="cuda", dtype=torch.int32)
        dispatch = SimpleNamespace(hidden_states=x, topk_output=StandardTopKOutput(weights, ids, None))
        off = methods[False].apply(layer, dispatch).hidden_states
        on = methods[True].apply(layer, dispatch).hidden_states
        assert on.dtype == off.dtype == torch.bfloat16
        assert torch.equal(on.view(torch.int16), off.view(torch.int16)), route
        unscaled = Exl3MoEMethod._apply_graph(layer, layer._nvfp4_expert_streamer, x, weights, ids, ACT_LIMIT)
        assert torch.equal(off, unscaled if fused_in_topk else unscaled * 1.5)


def test_captured_in_graph_moe_matches_the_eager_streamed_apply(tmp_path):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer, source = _layer(tmp_path)
    backend = streamer.row_backend
    gen = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
    weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
    ids = torch.tensor([[0, 3, 5, 1, 7, 6]], device="cuda", dtype=torch.int32)
    Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)  # warm up, allocate
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)

    def replay(route):
        ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        return out.clone()

    outputs = {}
    for route in ([2, 4, 6, 0, 1, 3], [7, 5, 3, 1, 0, 2]):
        got = replay(route)
        outputs[tuple(route)] = got
        assert backend.keep.item() == 1.0
        # An eager gather of the same routes remaps them to the same slots (the replay's).
        assert torch.equal(Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT), got)
        remap, tensors = streamer.gather(ids)
        slots = remap.reshape(-1).tolist()
        for k, expert in enumerate(route):
            for name, rows in tensors.items():
                assert torch.equal(rows[slots[k]].cpu(), source[name][expert]), (route, k, expert, name)
        ref = _reference(x, weights.reshape(-1), remap.reshape(-1), tensors)
        loop = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), ACT_LIMIT)
        rel, rel_loop = _rel(got, ref), _rel(loop, ref)
        assert rel <= REL_BOUND and rel <= 2 * rel_loop + 1e-3, (route, rel, rel_loop)
        assert rel <= REL_TIGHT, (route, rel)
        assert _rel(got, loop.float()) <= LOOSE_BOUND, (route, _rel(got, loop.float()))

    # Expert 8 is neither hot nor in the pinned tier: a RAM miss drops the layer.
    misses = backend.ram_miss.item()
    dropped = replay([8, 0, 3, 1, 7, 6])
    assert backend.keep.item() == 0.0
    assert backend.ram_miss.item() == misses + 1
    assert int(layer._exl3_fused_moe.expert_count.count_nonzero()) == 0  # no expert ran
    assert torch.isfinite(dropped).all() and int(dropped.count_nonzero()) == 0

    # keep is per call, not sticky.
    route = [2, 4, 6, 0, 1, 3]
    assert torch.equal(replay(route), outputs[tuple(route)])
    assert backend.keep.item() == 1.0
    assert backend.ram_miss.item() == misses + 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
