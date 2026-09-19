"""In-graph EXL3 MoE over pinned-tier graph gather equals the eager streamed apply (GPU, window).

Fake finite EXL3 checkpoint (1 layer, 12 experts); pinned tier of 8 rows, hot cache of
3 slots + 6 scratch; top-6 routes. Captures streamer.gather + Exl3FusedMoE.run in a
CUDA graph, replays for new routes whose rows are all in RAM, and checks each replay
with the P2 probe's bars (plan Design decision D7) against an fp32 reference over the
gathered slot rows: rel <= 1.2e-2 and rel <= 2 * rel(loop) + 1e-3, where the loop is
Exl3MoEMethod._apply_streamed (exl3_moe_loop). The loop is the less accurate arm
(~1.5e-2 on these rows, probe: 1.15e-2 vs fused 1.16e-3), so graph vs loop is only
bounded loosely; that bound still catches a wrong expert or weight, since the loop
reads its rows from the row source, not from the graph gather. A replay also equals
an eager _apply_graph call bit for bit.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 12, 6
ACT_LIMIT = 10.0
REL_BOUND = 1.2e-2
LOOSE_BOUND = 4e-2  # graph vs loop: ~1.5e-2 measured, a wrong expert or weight gives ~1


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


def _layer(tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    layer = torch.nn.Module()
    layer.layer_id = 0
    layer.top_k = TOP_K
    fmt = Exl3ExpertFormat(build_exl3_expert_layout(str(tmp_path)), 0, direct=False)
    with envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"):
        streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
    pinned = ExpertPinnedHostCache(streamer, 8)
    hot = ExpertHotCache(streamer, 3, scratch_rows=TOP_K)
    hot.reassign([0, 1, 2])
    pinned.ensure_rows(torch.arange(8, device="cuda"))
    streamer.enable_graph_gather(TOP_K)
    layer._nvfp4_expert_streamer = streamer
    layer._exl3_allow_p3_only = True  # test configuration: P3's backend without option C (Task 14)
    return layer, streamer


def test_captured_in_graph_moe_matches_the_eager_streamed_apply(tmp_path):
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod

    layer, streamer = _layer(tmp_path)
    gen = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.randn((1, HIDDEN), generator=gen) * 0.5).to("cuda", torch.bfloat16)
    weights = torch.softmax(torch.randn((1, TOP_K), generator=gen), -1).cuda()
    ids = torch.tensor([[0, 3, 5, 1, 7, 6]], device="cuda", dtype=torch.int32)
    Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)  # warm up, allocate
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
    for route in ([2, 4, 6, 0, 1, 3], [7, 5, 3, 1, 0, 2]):
        ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        got = out.clone()
        assert streamer.row_backend.keep.item() == 1.0
        # An eager gather of the same routes remaps them to the same slots (the replay's).
        assert torch.equal(Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT), got)
        remap, tensors = streamer.gather(ids)
        ref = _reference(x, weights.reshape(-1), remap.reshape(-1), tensors)
        loop = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), ACT_LIMIT)
        rel, rel_loop = _rel(got, ref), _rel(loop, ref)
        assert rel <= REL_BOUND and rel <= 2 * rel_loop + 1e-3, (route, rel, rel_loop)
        assert _rel(got, loop.float()) <= LOOSE_BOUND, (route, _rel(got, loop.float()))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
