"""CUDA tests for the expert format and row-source plugin seams."""

import unittest

import torch

from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-a", runner_config="1-gpu-small")

EXPERTS = 8
_HOST_SHAPES = {
    "w13_weight": (3, 8),
    "w2_weight": (5, 4),
    "w13_blockscale_swizzled": (3, 2),
    "w2_blockscale_swizzled": (5, 1),
}


def _nvfp4_layer(seed=21, pinned=True, experts=EXPERTS):
    """NVFP4-named host rows (optionally pinned) and CUDA alphas, as in test_expert_graph_gather."""
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    for name in NVFP4_STREAM_TENSORS[:4]:
        rows = torch.randint(
            0, 256, (experts,) + _HOST_SHAPES[name], dtype=torch.uint8, generator=generator
        )
        if "blockscale" in name:
            rows = rows.view(torch.float8_e4m3fn)
        if pinned:
            rows = rows.pin_memory()
        setattr(layer, name, torch.nn.Parameter(rows, requires_grad=False))
    for name in NVFP4_STREAM_TENSORS[4:]:
        values = torch.rand(experts, generator=generator).cuda()
        setattr(layer, name, torch.nn.Parameter(values, requires_grad=False))
    layer.top_k = 4
    return layer


def _source_bytes(layer, name, ids):
    source = getattr(layer, name).data
    return source[ids.long().to(source.device)].view(torch.uint8).cpu()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFormatSeamCuda(unittest.TestCase):
    def test_hot_and_pinned_slots_keep_their_source_shapes(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        layer = _nvfp4_layer(pinned=False)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        hot = ExpertHotCache(streamer, 2)
        pinned = ExpertPinnedHostCache(streamer, 2)
        self.assertEqual(hot.device, layer.g1_alphas.device)
        for name in NVFP4_STREAM_TENSORS:
            source = getattr(layer, name).data
            self.assertEqual(tuple(hot.tensors[name].shape[1:]), tuple(source.shape[1:]))
            self.assertEqual(hot.tensors[name].dtype, source.dtype)
        self.assertEqual(pinned.cached_names, NVFP4_STREAM_TENSORS[:4])
        for name in pinned.cached_names:
            source = getattr(layer, name).data
            self.assertEqual(
                tuple(pinned.tensors[name].shape), (2,) + tuple(source.shape[1:])
            )
            self.assertEqual(pinned.tensors[name].dtype, source.dtype)

    def test_graph_gather_still_detects_a_rebound_source(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _nvfp4_layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        cache = ExpertHotCache(streamer, 3, scratch_rows=4)
        cache.reassign([1, 4, 6])
        streamer.enable_graph_gather(4)
        ids = torch.tensor([[1, 2, 4, 7]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )
        layer.w13_weight.data = layer.w13_weight.data.clone().pin_memory()
        with self.assertRaisesRegex(RuntimeError, "moved after graph gather"):
            streamer.gather(ids)


if __name__ == "__main__":
    unittest.main()
