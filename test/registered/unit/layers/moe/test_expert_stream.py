import unittest
import warnings
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    enable_breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


class _Layer(torch.nn.Module):
    pass


class _StreamedFusedMoEHarness(FusedMoE):
    def __init__(self, streamer):
        torch.nn.Module.__init__(self)
        self._nvfp4_expert_streamer = streamer
        self._use_ascend_fuseep = False

    def forward_impl(self, hidden_states, topk_output, pre_quant_input=None):
        compact_ids, streamed_tensors = self._nvfp4_expert_streamer.gather(
            topk_output.topk_ids
        )
        selected_rows = streamed_tensors["host_rows"][compact_ids.long()]
        return hidden_states + selected_rows.reshape_as(hidden_states)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertStreamer(unittest.TestCase):
    def _make_layer(self, experts=16, pin_host_rows=True):
        layer = _Layer()
        host_rows = torch.arange(experts * 12, dtype=torch.uint8).reshape(experts, 3, 4)
        if pin_host_rows:
            host_rows = host_rows.pin_memory()
        layer.host_rows = torch.nn.Parameter(host_rows, requires_grad=False)
        layer.gpu_rows = torch.nn.Parameter(
            torch.arange(experts * 5, dtype=torch.float32, device="cuda").reshape(
                experts, 5
            ),
            requires_grad=False,
        )
        return layer

    def test_decode_preserves_duplicate_rows_without_deduplication(self):
        layer = self._make_layer()
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        topk_ids = torch.tensor([[9, 3, 9, 7]], device="cuda", dtype=torch.int32)

        compact_ids, tensors = streamer.gather(topk_ids)

        self.assertEqual(compact_ids.tolist(), [[0, 1, 2, 3]])
        expected_ids = topk_ids.reshape(-1).to(torch.long).cpu()
        self.assertTrue(
            torch.equal(tensors["host_rows"].cpu(), layer.host_rows[expected_ids])
        )
        self.assertTrue(
            torch.equal(
                tensors["gpu_rows"].cpu(),
                layer.gpu_rows[expected_ids.to(layer.gpu_rows.device)].cpu(),
            )
        )

    def test_prefill_deduplicates_and_returns_inverse_mapping(self):
        layer = self._make_layer()
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        topk_ids = torch.arange(68, device="cuda", dtype=torch.int32).remainder(11)
        topk_ids = topk_ids.reshape(17, 4)

        compact_ids, tensors = streamer.gather(topk_ids)

        reconstructed_host = tensors["host_rows"][compact_ids.to(torch.long)]
        reconstructed_gpu = tensors["gpu_rows"][compact_ids.to(torch.long)]
        self.assertTrue(
            torch.equal(
                reconstructed_host.cpu(),
                layer.host_rows[topk_ids.to(torch.long).cpu()],
            )
        )
        self.assertTrue(
            torch.equal(
                reconstructed_gpu.cpu(),
                layer.gpu_rows[topk_ids.to(torch.long)].cpu(),
            )
        )
        self.assertEqual(tensors["host_rows"].shape[0], 11)

    def test_decode_buffer_covers_more_routes_than_experts(self):
        layer = self._make_layer(experts=4)
        streamer = ExpertStreamer(layer, ("gpu_rows",))
        topk_ids = torch.tensor(
            [[3, 1, 2], [1, 3, 0]], device="cuda", dtype=torch.int32
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compact_ids, tensors = streamer.gather(topk_ids)

        self.assertEqual(compact_ids.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(tensors["gpu_rows"].shape[0], 6)
        self.assertTrue(
            torch.equal(
                tensors["gpu_rows"].cpu(),
                layer.gpu_rows[topk_ids.reshape(-1).to(torch.long)].cpu(),
            )
        )

    def test_pageable_cpu_sources_use_bounded_staging(self):
        layer = self._make_layer(experts=4, pin_host_rows=False)
        streamer = ExpertStreamer(layer, ("host_rows",))
        topk_ids = torch.tensor(
            [[3, 1, 2], [1, 3, 0]], device="cuda", dtype=torch.int32
        )

        compact_ids, tensors = streamer.gather(topk_ids)

        expected_ids = topk_ids.reshape(-1).to(torch.long).cpu()
        self.assertEqual(compact_ids.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertTrue(
            torch.equal(tensors["host_rows"].cpu(), layer.host_rows[expected_ids])
        )

    def test_breakable_graph_replays_streamed_host_rows_for_new_routes(self):
        layer = _Layer()
        layer.host_rows = torch.nn.Parameter(
            torch.tensor([[1.0], [3.0], [5.0], [7.0]]), requires_grad=False
        )
        model = _StreamedFusedMoEHarness(ExpertStreamer(layer, ("host_rows",)))
        hidden_states = torch.zeros((1, 2), device="cuda")
        topk_ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        topk_output = StandardTopKOutput(
            torch.ones((1, 2), device="cuda"),
            topk_ids,
            torch.empty((1, 4), device="cuda"),
        )
        output = torch.empty_like(hidden_states)
        graph = BreakableCUDAGraph()
        stream = torch.cuda.Stream()

        with (
            enable_breakable_cuda_graph(),
            BreakableCUDAGraphCapture(graph, stream=stream),
        ):
            output.copy_(model(hidden_states, topk_output))

        topk_ids.copy_(torch.tensor([[2, 3]], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(output, torch.tensor([[5.0, 7.0]], device="cuda"))

    def test_hot_cache_handles_pinned_and_cuda_sources(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = self._make_layer(experts=4)
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        cache = ExpertHotCache(streamer, capacity=1)
        cache.reassign([3])
        ids = torch.tensor([[3, 1, 3]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        for name in ("host_rows", "gpu_rows"):
            source = getattr(layer, name)
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].cpu(),
                    source[ids.long().to(source.device)].cpu(),
                )
            )
        stats = streamer.last_gather_stats
        self.assertEqual((stats.hot_hit_rows, stats.miss_rows), (2, 1))
        self.assertEqual(stats.source_bytes, 32)
        self.assertEqual(stats.h2d_bytes, 12)
        self.assertEqual(stats.d2d_bytes, 116)

    def test_pinned_host_cache_populates_on_demand_and_evicts_bounded_rows(self):
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        layer = self._make_layer(experts=4, pin_host_rows=False)
        layer._nvfp4_file_source_bytes_per_expert = 12
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, capacity=2)

        first_ids = torch.tensor([[1, 3, 1]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(first_ids)
        self.assertTrue(
            torch.equal(
                tensors["host_rows"][compact.long()].cpu(),
                layer.host_rows[first_ids.cpu()],
            )
        )
        self.assertEqual(cache.slot_to_expert.count(-1), 0)
        self.assertEqual(cache.stats.populated_rows, 2)
        self.assertEqual(cache.stats.evictions, 0)
        self.assertEqual(streamer.last_gather_stats.pinned_host_hit_rows, 0)
        self.assertEqual(streamer.last_gather_stats.pinned_host_miss_rows, 3)
        self.assertEqual(streamer.last_gather_stats.pinned_host_populated_bytes, 24)

        second_ids = torch.tensor([[1, 2, 1]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(second_ids)
        self.assertTrue(
            torch.equal(
                tensors["host_rows"][compact.long()].cpu(),
                layer.host_rows[second_ids.cpu()],
            )
        )
        self.assertEqual(streamer.last_gather_stats.pinned_host_hit_rows, 2)
        self.assertEqual(streamer.last_gather_stats.pinned_host_miss_rows, 1)
        self.assertEqual(cache.stats.populated_rows, 3)
        self.assertEqual(cache.stats.evictions, 1)
        self.assertEqual(streamer.last_gather_stats.pinned_host_populated_bytes, 12)
        self.assertLessEqual(cache.residency_bytes, 2 * streamer.host_bytes_per_expert)

    def test_hot_misses_resolve_through_pinned_then_file_rows(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        layer = self._make_layer(experts=4, pin_host_rows=False)
        layer._nvfp4_file_source_bytes_per_expert = 12
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        hot_cache = ExpertHotCache(streamer, capacity=1)
        hot_cache.reassign([3])
        pinned_cache = ExpertPinnedHostCache(streamer, capacity=2)
        pinned_cache.ensure_rows(torch.tensor([1], device="cuda"))
        ids = torch.tensor([[3, 1, 2]], device="cuda", dtype=torch.int32)
        compact_ids, tensors = streamer.gather(ids)
        self.assertTrue(
            torch.equal(
                tensors["host_rows"][compact_ids.long()].cpu(),
                layer.host_rows[ids.cpu()],
            )
        )
        stats = streamer.last_gather_stats
        self.assertEqual((stats.hot_hit_rows, stats.pinned_host_hit_rows), (1, 1))

        self.assertEqual(stats.pinned_host_miss_rows, 1)
        self.assertEqual(stats.source_bytes, 52)
        self.assertEqual(stats.h2d_bytes, 24)

    def test_pinned_cache_hits_do_not_select_host_rows_on_cpu(self):
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        layer = self._make_layer(experts=4, pin_host_rows=False)
        layer._nvfp4_file_source_bytes_per_expert = 12
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, capacity=2)
        cache.ensure_rows(torch.tensor([1, 3], device="cuda"))
        ids = torch.tensor([[3, 1]], device="cuda", dtype=torch.int32)
        with patch("torch.index_select", side_effect=AssertionError("CPU row select")):
            compact_ids, tensors = streamer.gather(ids)
        self.assertTrue(
            torch.equal(
                tensors["host_rows"][compact_ids.long()].cpu(),
                layer.host_rows[ids.cpu()],
            )
        )
        self.assertEqual(streamer.last_gather_stats.pinned_host_hit_rows, 2)

    def test_rejects_mismatched_expert_dimensions(self):
        layer = _Layer()
        layer.a = torch.nn.Parameter(
            torch.zeros((4, 8), dtype=torch.uint8).pin_memory(),
            requires_grad=False,
        )
        layer.b = torch.nn.Parameter(
            torch.zeros((5, 8), dtype=torch.uint8).pin_memory(),
            requires_grad=False,
        )

        with self.assertRaisesRegex(ValueError, "expert count"):
            ExpertStreamer(layer, ("a", "b"))

    def test_rejects_out_of_range_expert_ids(self):
        layer = self._make_layer(experts=4)
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        topk_ids = torch.tensor([[0, 4]], device="cuda", dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "outside"):
            streamer.gather(topk_ids)


if __name__ == "__main__":
    unittest.main()
