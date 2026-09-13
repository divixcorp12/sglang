"""Sync-free expert gather: CUDA-graph replays match source rows for any routes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.expert_stream import NVFP4_STREAM_TENSORS, ExpertStreamer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

EXPERTS = 8
TOP_K = 4
_HOST_SHAPES = {
    "w13_weight": (3, 8),
    "w2_weight": (5, 4),
    "w13_blockscale_swizzled": (3, 2),
    "w2_blockscale_swizzled": (5, 1),
}


def _layer(seed=21, pinned=True):
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    for name in NVFP4_STREAM_TENSORS[:4]:
        rows = torch.randint(
            0,
            256,
            (EXPERTS,) + _HOST_SHAPES[name],
            dtype=torch.uint8,
            generator=generator,
        )
        if "blockscale" in name:
            rows = rows.view(torch.float8_e4m3fn)
        if pinned:
            rows = rows.pin_memory()
        setattr(layer, name, torch.nn.Parameter(rows, requires_grad=False))
    for name in NVFP4_STREAM_TENSORS[4:]:
        values = torch.rand(EXPERTS, generator=generator).cuda()
        setattr(layer, name, torch.nn.Parameter(values, requires_grad=False))
    layer.top_k = TOP_K
    return layer


def _source_bytes(layer, name, ids):
    source = getattr(layer, name).data
    return source[ids.long().to(source.device)].view(torch.uint8).cpu()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestExpertGraphGather(unittest.TestCase):
    def _graph_streamer(self, layer, resident=(1, 4, 6)):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        cache = ExpertHotCache(streamer, len(resident), scratch_rows=TOP_K)
        cache.reassign(list(resident))
        streamer.enable_graph_gather(TOP_K)
        return streamer, cache

    def _assert_rows(self, layer, ids, compact, tensors):
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )

    def test_eager_graph_gather_returns_source_rows_and_counts_misses(self):
        layer = _layer()
        streamer, cache = self._graph_streamer(layer)
        ids = torch.tensor([[1, 2, 2, 6]], dtype=torch.int32, device="cuda")

        compact, tensors = streamer.gather(ids)

        self.assertIs(tensors, cache.tensors)
        self.assertEqual(compact.dtype, ids.dtype)
        self._assert_rows(layer, ids, compact, tensors)
        self.assertEqual(streamer.graph_counters.tolist(), [4, 2])
        self.assertEqual(cache.resident_experts(), frozenset({1, 4, 6}))

    def test_graph_gather_performs_no_host_synchronization(self):
        streamer, _ = self._graph_streamer(_layer())
        ids = torch.tensor([[0, 4, 7, 0]], dtype=torch.int32, device="cuda")
        streamer.gather(ids)
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            streamer.gather(ids)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_replay_captured_on_all_hot_routes_serves_misses_and_reassignment(self):
        layer = _layer()
        streamer, cache = self._graph_streamer(layer)
        ids = torch.tensor([[1, 4, 6, 1]], dtype=torch.int32, device="cuda")
        outputs = {
            name: torch.empty(
                (TOP_K,) + tuple(tensor.shape[1:]),
                dtype=tensor.dtype,
                device="cuda",
            )
            for name, tensor in cache.tensors.items()
        }

        def gather_into_outputs():
            compact, tensors = streamer.gather(ids)
            for name, output in outputs.items():
                output.copy_(tensors[name][compact.reshape(-1).long()])

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            gather_into_outputs()
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gather_into_outputs()

        def replay_and_check(routes):
            ids.copy_(torch.tensor([routes], dtype=torch.int32, device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            for name, output in outputs.items():
                self.assertTrue(
                    torch.equal(
                        output.view(torch.uint8).cpu(),
                        _source_bytes(layer, name, ids.reshape(-1)),
                    ),
                    f"{name} routes={routes}",
                )

        for routes in ([0, 2, 3, 5], [1, 2, 6, 7], [3, 3, 4, 3], [1, 4, 6, 1]):
            replay_and_check(routes)
        cache.reassign([0, 2, 3])
        replay_and_check([0, 2, 4, 7])
        replay_and_check([5, 5, 5, 5])

    def test_counters_and_residency_match_the_eager_gather(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy

        graph_layer, eager_layer = _layer(), _layer()
        graph_streamer, graph_cache = self._graph_streamer(graph_layer)
        eager_streamer = ExpertStreamer(eager_layer, NVFP4_STREAM_TENSORS)
        eager_cache = ExpertHotCache(eager_streamer, 3)
        eager_cache.reassign([1, 4, 6])
        for streamer, cache in (
            (graph_streamer, graph_cache),
            (eager_streamer, eager_cache),
        ):
            streamer.residency_policy = ExpertResidencyPolicy(
                EXPERTS, cache.capacity, device=cache.device
            )
        requested = misses = 0
        for routes in ([1, 2, 2, 6], [0, 0, 0, 0], [4, 6, 1, 4], [7, 3, 1, 5]):
            ids = torch.tensor([routes], dtype=torch.int32, device="cuda")
            graph_streamer.gather(ids)
            eager_streamer.gather(ids)
            requested += eager_streamer.last_gather_stats.requested_rows
            misses += eager_streamer.last_gather_stats.miss_rows

        self.assertEqual(graph_streamer.graph_counters.tolist(), [requested, misses])
        torch.testing.assert_close(
            graph_streamer.residency_policy.pending_counts,
            eager_streamer.residency_policy.pending_counts,
            rtol=0,
            atol=0,
        )

    def test_graph_gather_reads_rows_from_the_registered_host_arena(self):
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _layer(pinned=False)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        arena = ExpertHostArena()
        arena.bind(streamer)
        try:
            cache = ExpertHotCache(streamer, 2, scratch_rows=TOP_K)
            cache.reassign([1, 4])
            streamer.enable_graph_gather(TOP_K)
            ids = torch.tensor([[1, 3, 3, 7]], dtype=torch.int32, device="cuda")

            compact, tensors = streamer.gather(ids)

            self._assert_rows(layer, ids, compact, tensors)
            self.assertEqual(streamer.graph_counters.tolist(), [4, 3])
        finally:
            arena.close()

    def test_enable_rejects_pageable_sources_and_missing_scratch_rows(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        pageable = ExpertStreamer(_layer(pinned=False), NVFP4_STREAM_TENSORS)
        ExpertHotCache(pageable, 1, scratch_rows=TOP_K)
        with self.assertRaisesRegex(ValueError, "pageable"):
            pageable.enable_graph_gather(TOP_K)

        short = ExpertStreamer(_layer(), NVFP4_STREAM_TENSORS)
        ExpertHotCache(short, 1, scratch_rows=TOP_K - 1)
        with self.assertRaisesRegex(ValueError, "scratch row"):
            short.enable_graph_gather(TOP_K)

    def test_routes_wider_than_the_scratch_rows_take_the_eager_gather(self):
        layer = _layer()
        streamer, _ = self._graph_streamer(layer)
        ids = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 0]], dtype=torch.int32, device="cuda"
        )

        with patch.object(streamer, "_gather_graph") as graph_gather:
            compact, tensors = streamer.gather(ids)

        graph_gather.assert_not_called()
        self._assert_rows(layer, ids, compact, tensors)
        self.assertFalse(streamer.serves_graph_gather(SimpleNamespace(topk_ids=ids)))
        self.assertTrue(streamer.serves_graph_gather(SimpleNamespace(topk_ids=ids[:1])))

    def test_manager_reserves_scratch_rows_and_counts_replayed_gathers(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        layer = _layer()
        layer.layer_id = 0
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        model = torch.nn.Module()
        model.add_module("0", layer)
        streamer = layer._nvfp4_expert_streamer
        options = dict(
            seed_path=None,
            dynamic=True,
            update_prefill_tokens=16,
            min_residence_forwards=0,
            benefit_ratio=1.0,
            graph_gather_batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "scratch rows"):
            ExpertHotCacheManager.from_model(
                model, budget_bytes=streamer.bytes_per_expert * (TOP_K - 1), **options
            )

        manager = ExpertHotCacheManager.from_model(
            model, budget_bytes=streamer.bytes_per_expert * (TOP_K + 2), **options
        )

        cache = manager.caches[0]
        self.assertEqual((cache.capacity, cache.scratch_rows), (2, TOP_K))
        self.assertEqual(streamer.graph_gather_rows, TOP_K)
        resident = sorted(cache.resident_experts())
        missing = [expert for expert in range(EXPERTS) if expert not in resident]
        capture_routes = [missing[1]] * TOP_K
        streamer.gather(
            torch.tensor([capture_routes], dtype=torch.int32, device="cuda")
        )
        manager.discard_graph_capture_routes()
        self.assertEqual(streamer.graph_counters.tolist(), [0, 0])
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 0.0)

        routes = [resident[0], missing[0], missing[0], resident[1]]
        streamer.gather(torch.tensor([routes], dtype=torch.int32, device="cuda"))
        counts = torch.zeros((1, EXPERTS), dtype=torch.int64)
        for expert in routes:
            counts[0, expert] += 1
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, extend_num_tokens=0)
        manager.on_expert_distribution(batch, {"global_physical_count": counts})
        manager.on_expert_distribution(batch, {"global_physical_count": counts})

        decode = manager.snapshot_counters()["decode"]["0"]
        self.assertEqual(decode["requested_rows"], 4)
        self.assertEqual(decode["miss_rows"], 2)
        self.assertEqual(decode["hot_hits"], 2)
        self.assertEqual(decode["requested_unique_experts"], 3)
        self.assertEqual(decode["h2d_bytes"], 2 * streamer.host_bytes_per_expert)
