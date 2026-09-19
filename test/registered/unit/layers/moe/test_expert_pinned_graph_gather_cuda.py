"""A captured graph gather reads missed rows from pinned-tier slots, not expert-id rows (CUDA)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NAMES = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")


@pytest.fixture(params=[False, True], ids=["unfused", "fused"], autouse=True)
def fused_plan(request, monkeypatch):
    """Run every test under both planners: the fused kernel writes the plan's expert ids itself.

    The spec-only format serves only the default row source, so an ambient
    SGLANG_MOE_EXPERT_ROW_SOURCE (e.g. ``shards`` from a model env file) is cleared.
    """
    monkeypatch.delenv("SGLANG_MOE_EXPERT_ROW_SOURCE", raising=False)
    monkeypatch.setenv("SGLANG_MOE_EXPERT_FUSED_PLAN", "1" if request.param else "0")
    return request.param


def _tiers(pinned_rows=4, hot_slots=2, top_k=2):
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.test.moe_expert_fakes import SpecOnlyFormat

    reference = {
        name: (torch.arange(8 * 64, dtype=torch.int32).reshape(8, 64) * (i + 1)).to(torch.int16)
        for i, name in enumerate(NAMES)
    }
    fmt = SpecOnlyFormat(reference)
    fmt.graph_source_kind = "pinned_tier"
    layer = torch.nn.Module()
    layer.layer_id = 0
    layer.top_k = top_k
    streamer = ExpertStreamer(layer, NAMES, format=fmt)
    pinned = ExpertPinnedHostCache(streamer, pinned_rows)
    hot = ExpertHotCache(streamer, hot_slots, scratch_rows=top_k)
    hot.reassign([0, 1])
    streamer.enable_graph_gather(top_k)
    return streamer, pinned, hot, reference


def test_capture_then_replay_reads_the_pinned_slots(fused_plan):
    streamer, pinned, hot, reference = _tiers()
    assert streamer._graph_pinned_tier
    assert streamer._fused_plan_enabled == fused_plan
    pinned.ensure_rows(torch.tensor([6, 3], device="cuda"))  # slots chosen by the LRU
    ids = torch.tensor([[1, 6]], device="cuda", dtype=torch.int32)
    streamer.gather(ids)  # warm up outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        remap, tensors = streamer.gather(ids)
    ids.copy_(torch.tensor([[3, 0]], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    for name in NAMES:
        got = tensors[name][remap.reshape(-1).long()].cpu()
        assert torch.equal(got, reference[name][[3, 0]]), name
    assert streamer.row_backend.keep.item() == 1.0
    assert streamer.row_backend.ram_miss.item() == 0


def test_a_ram_miss_inside_a_replay_is_counted_and_drops_the_layer():
    streamer, pinned, hot, reference = _tiers()
    pinned.ensure_rows(torch.tensor([6], device="cuda"))
    ids = torch.tensor([[1, 6]], device="cuda", dtype=torch.int32)
    streamer.gather(ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        remap, tensors = streamer.gather(ids)
    ids.copy_(torch.tensor([[1, 7]], device="cuda", dtype=torch.int32))  # 7 is not in RAM
    graph.replay()
    torch.cuda.synchronize()
    assert streamer.row_backend.keep.item() == 0.0
    assert streamer.row_backend.ram_miss.item() == 1
    missed = remap.reshape(-1)[1].item()
    assert missed >= hot.capacity  # the miss landed in a scratch row
    # The clamped lane copied pinned slot 0. The tier is inclusive, so the hot
    # cache's reassign put experts 0 and 1 there before 6: look the slot up.
    slot_zero_expert = pinned.expert_to_slot.tolist().index(0)
    assert slot_zero_expert != 7
    for name in NAMES:
        assert torch.equal(tensors[name][missed].cpu(), reference[name][slot_zero_expert]), name


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
