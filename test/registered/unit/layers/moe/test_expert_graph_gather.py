"""Sync-free expert gather: CUDA-graph replays match source rows for any routes."""

import os
import time
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


DOORBELL_SPIN_CORE = int(os.environ.get("DOORBELL_SPIN_CORE", "71"))


def _streamed_model(seeds):
    model = torch.nn.Module()
    for layer_id, seed in enumerate(seeds):
        layer = _layer(seed=seed)
        layer.layer_id = layer_id
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        model.add_module(str(layer_id), layer)
    return model


def _manager(model, doorbell, **doorbell_budgets):
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager

    layers = list(model.children())
    return ExpertHotCacheManager.from_model(
        model,
        budget_bytes=len(layers)
        * layers[0]._nvfp4_expert_streamer.bytes_per_expert
        * (TOP_K + 2),
        seed_path=None,
        dynamic=False,
        update_prefill_tokens=16,
        min_residence_forwards=0,
        benefit_ratio=1.0,
        graph_gather_batch_size=1,
        expert_doorbell=doorbell,
        doorbell_cpu_core=DOORBELL_SPIN_CORE,
        **doorbell_budgets,
    )


def _cache_rows(manager):
    return {
        (layer_id, name): tensor.view(torch.uint8).cpu()
        for layer_id, cache in manager.caches.items()
        for name, tensor in cache.tensors.items()
    }


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

    def test_verify_shaped_routes_share_scratch_rows_between_duplicate_misses(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        routes = 4 * TOP_K
        cache = ExpertHotCache(streamer, 3, scratch_rows=routes)
        cache.reassign([1, 4, 6])
        streamer.enable_graph_gather(routes)
        ids = torch.tensor(
            [[0, 1, 2, 3], [2, 3, 4, 5], [5, 0, 6, 7], [7, 2, 1, 3]],
            dtype=torch.int32,
            device="cuda",
        )

        compact, tensors = streamer.gather(ids)

        self._assert_rows(layer, ids, compact, tensors)
        self.assertEqual(streamer.graph_counters.tolist(), [routes, 12])
        self.assertEqual(streamer.graph_unique_counters.tolist(), [3, 5])
        scratch = compact[compact >= cache.capacity]
        self.assertEqual(scratch.unique().numel(), 5)

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

    def test_eager_misses_use_the_copy_engine_for_registered_rows(self):
        from sglang.srt.layers.moe.expert_dma import (
            ExpertDMABackend,
            _aot_transfer_available,
        )
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        if not _aot_transfer_available():
            self.skipTest("the copy-engine range transfer is not built")
        layer = _layer(pinned=False)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        arena = ExpertHostArena()
        arena.bind(streamer)
        try:
            cache = ExpertHotCache(streamer, 2, scratch_rows=TOP_K)
            cache.reassign([1, 4])
            streamer.enable_graph_gather(TOP_K)
            streamer.expert_copy_backend = "dma"
            ids = torch.tensor(
                [[2, 3, 5, 1], [6, 0, 7, 4]], dtype=torch.int32, device="cuda"
            )
            original = ExpertDMABackend.copy_rows

            with patch.object(
                ExpertDMABackend, "copy_rows", autospec=True, side_effect=original
            ) as copy_rows:
                compact, tensors = streamer.gather(ids)

            self._assert_rows(layer, ids, compact, tensors)
            self.assertEqual(copy_rows.call_count, len(_HOST_SHAPES))
            self.assertEqual(streamer._dma_backend.actual_backend, "dma")
            self.assertEqual(
                streamer.last_gather_stats.copy_engine_bytes,
                6 * streamer.host_bytes_per_expert,
            )
        finally:
            arena.close()

    def test_eager_misses_keep_the_pull_kernel_without_the_copy_engine(self):
        from sglang.srt.layers.moe.expert_dma import ExpertDMABackend
        from sglang.srt.layers.moe.expert_host_arena import ExpertHostArena
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        conditions = {
            "not built": patch(
                "sglang.srt.layers.moe.expert_stream._aot_transfer_available",
                return_value=False,
            ),
            "capturing": patch(
                "torch.cuda.is_current_stream_capturing", return_value=True
            ),
        }
        for condition, unavailable in conditions.items():
            with self.subTest(condition):
                layer = _layer(pinned=False)
                streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
                arena = ExpertHostArena()
                arena.bind(streamer)
                try:
                    cache = ExpertHotCache(streamer, 2, scratch_rows=TOP_K)
                    cache.reassign([1, 4])
                    streamer.expert_copy_backend = "dma"
                    ids = torch.tensor(
                        [[2, 3, 5, 1], [6, 0, 7, 4]], dtype=torch.int32, device="cuda"
                    )

                    with unavailable, patch.object(
                        ExpertDMABackend, "copy_rows"
                    ) as copy_rows:
                        compact, tensors = streamer.gather(ids)
                    torch.cuda.synchronize()

                    self._assert_rows(layer, ids, compact, tensors)
                    copy_rows.assert_not_called()
                    self.assertEqual(streamer.last_gather_stats.copy_engine_bytes, 0)
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

    def test_gather_refuses_layer_tensors_rebound_after_enable(self):
        layer = _layer()
        streamer, _ = self._graph_streamer(layer)
        layer.w13_weight.data = torch.zeros_like(layer.w13_weight.data).pin_memory()
        ids = torch.tensor([[0, 4, 7, 0]], dtype=torch.int32, device="cuda")

        with self.assertRaisesRegex(RuntimeError, "moved"):
            streamer.gather(ids)

    def test_decode_updates_between_replays_serve_promoted_experts(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        layer = _layer()
        layer.layer_id = 0
        layer._nvfp4_expert_streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        model = torch.nn.Module()
        model.add_module("0", layer)
        streamer = layer._nvfp4_expert_streamer
        manager = ExpertHotCacheManager.from_model(
            model,
            budget_bytes=streamer.bytes_per_expert * (TOP_K + 2),
            seed_path=None,
            dynamic=True,
            update_prefill_tokens=16,
            min_residence_forwards=0,
            benefit_ratio=0.0,
            graph_gather_batch_size=1,
            update_decode_forwards=2,
        )
        cache = manager.caches[0]
        self.assertEqual(cache.resident_experts(), frozenset({0, 1}))
        ids = torch.tensor([[5, 5, 5, 5]], dtype=torch.int32, device="cuda")
        outputs = {
            name: torch.empty(
                (TOP_K,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device="cuda"
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
        manager.discard_graph_capture_routes()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, extend_num_tokens=0, batch_size=1
        )

        def decode_step(routes):
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
            counts = torch.zeros((1, EXPERTS), dtype=torch.int64)
            for expert in routes:
                counts[0, expert] += 1
            manager.on_expert_distribution(batch, {"global_physical_count": counts})

        decode_step([5, 5, 5, 5])
        self.assertEqual(cache.resident_experts(), frozenset({0, 1}))
        decode_step([5, 5, 5, 5])
        self.assertEqual(cache.resident_experts(), frozenset({0, 5}))
        decode_step([5, 1, 0, 5])
        decode_step([2, 2, 2, 2])
        self.assertEqual(cache.resident_experts(), frozenset({2, 5}))
        decode_step([2, 0, 5, 7])

        decode = manager.snapshot_counters()["decode"]["0"]
        self.assertEqual(decode["requested_rows"], 10)
        self.assertEqual(decode["routed_rows"], 20)
        self.assertEqual(decode["miss_rows"], 6)
        self.assertEqual(decode["routed_miss_rows"], 15)
        self.assertEqual(decode["promotions"], 2)
        self.assertEqual(decode["evictions"], 2)

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
        self.assertEqual(decode["requested_rows"], 3)
        self.assertEqual(decode["routed_rows"], 4)
        self.assertEqual(decode["miss_rows"], 1)
        self.assertEqual(decode["routed_miss_rows"], 2)
        self.assertEqual(decode["hot_hits"], 2)
        self.assertEqual(decode["requested_unique_experts"], 3)
        self.assertEqual(decode["h2d_bytes"], streamer.host_bytes_per_expert)

    def test_doorbell_gather_writes_the_same_cache_rows_as_the_in_graph_copy(self):
        """Flag on, the thread (not the fallback) writes every layer's hot and scratch rows
        byte-for-byte as the in-graph kernel does, for 0 misses, all-miss plans and random plans.

        Two layers with different rows post under different tags, so a request copied through
        the wrong layer's segments differs from the in-graph copy.
        """
        plain_model, doorbell_model = _streamed_model((21, 33)), _streamed_model((21, 33))
        plain, doorbell = _manager(plain_model, False), _manager(doorbell_model, True)
        for manager in (plain, doorbell):
            for cache in manager.caches.values():
                for tensor in cache.tensors.values():
                    tensor[cache.capacity :].view(torch.uint8).zero_()
        try:
            self.assertIsNone(plain.doorbell)
            for layer in plain_model.children():
                self.assertIsNone(layer._nvfp4_expert_streamer.doorbell)
            generator = torch.Generator().manual_seed(5)
            no_miss_steps = 0
            steps = 0
            for step in range(12):
                for layer_id in sorted(plain.caches):
                    resident = sorted(plain.caches[layer_id].resident_experts())
                    missing = [e for e in range(EXPERTS) if e not in resident]
                    if step == 0 and resident:
                        routes = [resident[0]] * TOP_K
                        no_miss_steps += 1
                    elif step == 1:
                        routes = missing[:TOP_K]
                    else:
                        routes = torch.randint(0, EXPERTS, (TOP_K,), generator=generator).tolist()
                    ids = torch.tensor([routes], dtype=torch.int32, device="cuda")
                    for model in (plain_model, doorbell_model):
                        layer = model.get_submodule(str(layer_id))
                        compact, tensors = layer._nvfp4_expert_streamer.gather(ids)
                        self._assert_rows(layer, ids, compact, tensors)
                    steps += 1
                    torch.cuda.synchronize()
                    plain_rows, doorbell_rows = _cache_rows(plain), _cache_rows(doorbell)
                    for key, rows in plain_rows.items():
                        self.assertTrue(torch.equal(rows, doorbell_rows[key]), f"{key} step={step}")
            stats = doorbell.doorbell.stats()
            self.assertGreater(no_miss_steps, 0)
            self.assertEqual(stats["timeouts"], 0)
            self.assertEqual(stats["serviced"], steps)
            self.assertEqual(stats["copy_errors"], 0)
            self.assertEqual(stats["invalid_records"], 0)
            self.assertEqual(stats["late_completions"], 0)
        finally:
            doorbell.doorbell.stop()

    def test_doorbell_gather_replays_after_a_quiesced_capture(self):
        """A capture taken with the thread quiesced replays thread-served copies for changing
        routes. Capture only records the doorbell launches, so a gather run while the thread is
        still quiesced stands in for any wait that timed out around a capture: it leaves the
        copier degraded, and resuming must clear that before serving."""
        model = _streamed_model((21,))
        manager = _manager(model, True, doorbell_drain_polls=40_000)
        layer = model.get_submodule("0")
        streamer = layer._nvfp4_expert_streamer
        cache = manager.caches[0]
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
        outputs = {
            name: torch.empty(
                (TOP_K,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device="cuda"
            )
            for name, tensor in cache.tensors.items()
        }

        def gather_into_outputs():
            compact, tensors = streamer.gather(ids)
            for name, output in outputs.items():
                output.copy_(tensors[name][compact.reshape(-1).long()])

        try:
            side_stream = torch.cuda.Stream()
            with torch.cuda.stream(side_stream):
                gather_into_outputs()
            torch.cuda.current_stream().wait_stream(side_stream)
            graph = torch.cuda.CUDAGraph()
            manager.quiesce_doorbell()
            try:
                with torch.cuda.graph(graph):
                    gather_into_outputs()
                torch.cuda.synchronize()
                streamer.gather(torch.tensor([[4, 5, 6, 7]], dtype=torch.int32, device="cuda"))
                torch.cuda.synchronize()
                paused = manager.doorbell.stats()
                self.assertEqual(paused["timeouts"], 1)
                self.assertEqual(paused["degraded"], 1)
            finally:
                manager.resume_doorbell()
            manager.discard_graph_capture_routes()
            self.assertEqual(manager.doorbell.stats()["timeouts"], 0)
            self.assertEqual(manager.doorbell.stats()["degraded"], 0)
            before = manager.doorbell.stats()["serviced"]
            replays = ([5, 6, 7, 5], [0, 7, 3, 2], [4, 4, 4, 4], [1, 2, 3, 4], [7, 6, 5, 4])
            for routes in replays:
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
            stats = manager.doorbell.stats()
            self.assertEqual(stats["timeouts"], 0)
            self.assertEqual(stats["serviced"] - before, len(replays))
        finally:
            manager.doorbell.stop()

    def test_doorbell_timeout_waits_for_the_copy_the_thread_already_queued(self):
        """A copy the thread queued before its wait timed out lands before the gather returns.

        The copier's stream is kept busy so the thread's copy is queued behind that work and the
        wait times out. Returning at the timeout would let the queued copy land after a later
        forward's fallback wrote the same scratch rows; the drain makes the completion word reach
        the request before the gather returns.
        """
        model = _streamed_model((21,))
        manager = _manager(
            model, True, doorbell_timeout_polls=20_000, doorbell_drain_polls=80_000_000
        )
        layer = model.get_submodule("0")
        streamer = layer._nvfp4_expert_streamer
        copier = manager.doorbell
        resident = sorted(manager.caches[0].resident_experts())
        missing = [e for e in range(EXPERTS) if e not in resident]
        stall_in = torch.randn((2048, 2048), device="cuda")
        stall_out = torch.empty_like(stall_in)
        try:
            torch.cuda.synchronize()
            with torch.cuda.stream(copier.stream):
                for _ in range(40):
                    torch.matmul(stall_in, stall_in, out=stall_out)
            ids = torch.tensor([missing[:TOP_K]], dtype=torch.int32, device="cuda")
            compact, tensors = streamer.gather(ids)
            torch.cuda.current_stream().synchronize()
            returned = copier.stats()
            self.assertEqual(returned["timeouts"], 1)
            self.assertEqual(returned["drain_timeouts"], 0)
            self.assertGreaterEqual(returned["done"], returned["posted"])
            torch.cuda.synchronize()
            self._assert_rows(layer, ids, compact, tensors)
            deadline = time.perf_counter() + 5.0
            while copier.stats()["late_completions"] == 0 and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertEqual(copier.stats()["late_completions"], 1)
        finally:
            copier.stop()

    def test_doorbell_gather_falls_back_to_correct_rows_when_the_thread_stalls(self):
        model = _streamed_model((21,))
        manager = _manager(model, True, doorbell_drain_polls=40_000)
        layer = model.get_submodule("0")
        streamer = layer._nvfp4_expert_streamer
        copier = manager.doorbell
        resident = sorted(manager.caches[0].resident_experts())
        missing = [e for e in range(EXPERTS) if e not in resident]
        try:
            copier.pause()
            ids = torch.tensor([missing[:TOP_K]], dtype=torch.int32, device="cuda")
            compact, tensors = streamer.gather(ids)
            torch.cuda.synchronize()
            self._assert_rows(layer, ids, compact, tensors)
            stalled = copier.stats()
            self.assertEqual(stalled["timeouts"], 1)
            self.assertEqual(stalled["serviced"], 0)

            copier.resume()
            ids = torch.tensor([missing[::-1][:TOP_K]], dtype=torch.int32, device="cuda")
            compact, tensors = streamer.gather(ids)
            torch.cuda.synchronize()
            self._assert_rows(layer, ids, compact, tensors)
            recovered = copier.stats()
            self.assertEqual(recovered["skipped_abandoned"], 1)
            self.assertEqual(recovered["late_completions"], 0)
        finally:
            copier.stop()
