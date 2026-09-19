"""In-graph EXL3 MoE over pinned-tier graph gather equals the eager streamed apply (GPU, window).

Fake finite EXL3 checkpoint (1 layer, 12 experts); pinned tier of 8 rows, hot cache of
3 slots + 6 scratch; top-6 routes. Captures streamer.gather + Exl3FusedMoE.run in a
CUDA graph, replays for new routes whose rows are all in RAM, and compares with
Exl3MoEMethod._apply_streamed on the same inputs (rel <= 1.2e-2, the probe's bar).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN, INTER, EXPERTS, TOP_K = 1024, 512, 12, 6


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
    Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)  # warm up, allocate
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, 10.0)
    for route in ([2, 4, 6, 0, 1, 3], [7, 5, 3, 1, 0, 2]):
        ids.copy_(torch.tensor([route], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        got = out.float().clone()
        want = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids.long(), 10.0).float()
        rel = float((got - want).norm() / want.norm())
        assert rel <= 1.2e-2, (route, rel)
        assert streamer.row_backend.keep.item() == 1.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
