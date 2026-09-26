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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRowSourceRoutingCuda(unittest.TestCase):
    def test_uncached_pageable_rows_read_every_name_in_one_call(self):
        from sglang.test.moe_expert_fakes import CountingRowSource

        layer = _nvfp4_layer(pinned=False)
        source = CountingRowSource(
            {name: getattr(layer, name).data for name in NVFP4_STREAM_TENSORS[:4]}
        )
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS, row_source=source)
        ids = torch.tensor([[6, 1, 3]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(source.calls[0].names, NVFP4_STREAM_TENSORS[:4])
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )

    def test_cached_misses_read_every_name_in_one_call(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.test.moe_expert_fakes import CountingRowSource

        layer = _nvfp4_layer(pinned=False)
        source = CountingRowSource(
            {name: getattr(layer, name).data for name in NVFP4_STREAM_TENSORS[:4]}
        )
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS, row_source=source)
        ExpertHotCache(streamer, 1).reassign([2])
        before = len(source.calls)
        ids = torch.tensor([[2, 5, 7]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self.assertEqual(len(source.calls), before + 1)
        self.assertEqual(sorted(source.calls[-1].rows), [5, 7])
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].view(torch.uint8).cpu(),
                    _source_bytes(layer, name, ids),
                ),
                name,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPinnedTierCuda(unittest.TestCase):
    def test_pinned_slabs_are_registered_page_aligned_and_unrounded(self):
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache
        from sglang.srt.utils.cuda_host_registry import is_gpu_readable_host_tensor

        layer = torch.nn.Module()
        layer.host_rows = torch.nn.Parameter(
            torch.zeros(8, 3000, dtype=torch.uint8), requires_grad=False
        )
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 3)
        slab = cache.tensors["host_rows"]
        self.assertEqual(slab.data_ptr() % 4096, 0)
        self.assertTrue(is_gpu_readable_host_tensor(slab))
        self.assertEqual(slab.untyped_storage().nbytes(), 3 * 3000 + 4096)
        cache.close()
        self.assertFalse(is_gpu_readable_host_tensor(slab))

    def test_cached_gather_copies_misses_beyond_the_pinned_capacity(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        experts = 16
        layer = torch.nn.Module()
        layer.host_rows = torch.nn.Parameter(
            torch.randint(0, 256, (experts, 3, 4), dtype=torch.uint8),
            requires_grad=False,
        )
        layer.gpu_rows = torch.nn.Parameter(
            torch.rand(experts, 5, device="cuda"), requires_grad=False
        )
        layer._nvfp4_file_source_bytes_per_expert = 12
        streamer = ExpertStreamer(layer, ("host_rows", "gpu_rows"))
        ExpertHotCache(streamer, 1).reassign([3])
        pinned = ExpertPinnedHostCache(streamer, 2)
        ids = torch.tensor([[3, 0, 5, 7], [9, 11, 3, 13]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        cpu_ids = ids.long().cpu()
        self.assertTrue(
            torch.equal(tensors["host_rows"][compact.long()].cpu(), layer.host_rows.data[cpu_ids])
        )
        self.assertTrue(
            torch.equal(
                tensors["gpu_rows"][compact.long()].cpu(),
                layer.gpu_rows.data[cpu_ids.cuda()].cpu(),
            )
        )
        stats = streamer.last_gather_stats
        self.assertEqual((stats.hot_hit_rows, stats.pinned_host_miss_rows), (1, 6))
        self.assertEqual(pinned.stats.evictions, 4)


def _spec_only_reference(names=None, experts=EXPERTS, seed=5):
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "w13_trellis": ((2, 8), torch.int16),
        "w13_suh": ((2, 4), torch.float16),
        "w13_svh": ((2, 2), torch.float16),
        "w2_trellis": ((1, 8), torch.int16),
        "w2_suh": ((1, 2), torch.float16),
        "w2_svh": ((1, 4), torch.float16),
    }
    names = tuple(shapes) if names is None else names
    return {
        name: torch.randint(
            0, 256, (experts,) + shapes[name][0] + (shapes[name][1].itemsize,),
            dtype=torch.uint8, generator=generator,
        ).view(shapes[name][1]).reshape((experts,) + shapes[name][0])
        for name in names
    }


def _spec_only_streamer(reference):
    from sglang.test.moe_expert_fakes import SpecOnlyFormat

    layer = torch.nn.Module()
    layer.layer_id = 0
    return ExpertStreamer(layer, tuple(reference), format=SpecOnlyFormat(reference))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSpecOnlyCuda(unittest.TestCase):
    def _assert_rows(self, reference, ids, compact, tensors):
        for name, tensor in reference.items():
            self.assertTrue(
                torch.equal(
                    tensors[name][compact.long()].cpu().view(torch.uint8),
                    tensor[ids.long().cpu()].view(torch.uint8),
                ),
                name,
            )

    def test_eager_gather_reads_hot_pinned_and_cold_rows_through_the_row_source(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        reference = _spec_only_reference()
        streamer = _spec_only_streamer(reference)
        hot = ExpertHotCache(streamer, 2)
        hot.reassign([0, 1])
        pinned = ExpertPinnedHostCache(streamer, 2)
        pinned.ensure_rows(torch.tensor([2], device="cuda"))
        ids = torch.tensor([[0, 2, 5], [1, 6, 2]], device="cuda", dtype=torch.int32)
        compact, tensors = streamer.gather(ids)
        self._assert_rows(reference, ids, compact, tensors)
        stats = streamer.last_gather_stats
        self.assertEqual(stats.hot_hit_rows, 2)
        self.assertEqual((stats.pinned_host_hit_rows, stats.pinned_host_miss_rows), (1, 2))
        self.assertEqual(stats.host_read_rows, 2)

    def test_promotions_go_through_the_pinned_tier_in_chunks(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        reference = _spec_only_reference()
        streamer = _spec_only_streamer(reference)
        # Expert 2 is protected, so every promotion chunk holds at most 2 rows.
        pinned = ExpertPinnedHostCache(streamer, 3, is_pinned=lambda expert: expert == 2)
        pinned.ensure_rows(torch.tensor([2], device="cuda"))
        self.assertEqual(pinned.evictable_rows(), 2)
        hot = ExpertHotCache(streamer, 5)
        hot.reassign([4, 0, 7, 3, 1])
        self.assertEqual(sorted(hot.resident_experts()), [0, 1, 3, 4, 7])
        self.assertIn(2, pinned._expert_to_slot)
        for slot, expert in enumerate(hot.slot_to_expert):
            for name, tensor in reference.items():
                self.assertTrue(
                    torch.equal(
                        hot.tensors[name][slot].cpu().view(torch.uint8),
                        tensor[expert].view(torch.uint8),
                    ),
                    (name, expert),
                )
        self.assertEqual(pinned.stats.populated_rows, 6)

    def test_non_six_spec_only_promotion_reads_through_the_row_source(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        reference = _spec_only_reference(names=("w13_trellis", "w2_trellis"))
        streamer = _spec_only_streamer(reference)
        hot = ExpertHotCache(streamer, 3)
        hot.reassign([1, 2, 5])
        for slot, expert in enumerate(hot.slot_to_expert):
            for name, tensor in reference.items():
                self.assertTrue(
                    torch.equal(hot.tensors[name][slot].cpu(), tensor[expert]),
                    (name, expert),
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGatherExpertsCuda(unittest.TestCase):
    def test_chunked_gather_matches_one_gather_through_hot_and_cold_rows(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _nvfp4_layer(pinned=False)
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        ExpertHotCache(streamer, 2).reassign([1, 4])
        ids = torch.tensor([6, 1, 3, 4, 0], device="cuda")
        row_of_source, rows = streamer.gather_experts(ids)
        whole = {
            name: rows[name][row_of_source.long()].view(torch.uint8).cpu()
            for name in NVFP4_STREAM_TENSORS
        }
        pieces = {name: [] for name in NVFP4_STREAM_TENSORS}
        for _, chunk_rows_of_source, chunk_rows in streamer.iter_gather_experts(
            ids, chunk_rows=2
        ):
            for name in NVFP4_STREAM_TENSORS:
                pieces[name].append(
                    chunk_rows[name][chunk_rows_of_source.long()].view(torch.uint8).cpu()
                )
        for name in NVFP4_STREAM_TENSORS:
            self.assertTrue(torch.equal(torch.cat(pieces[name]), whole[name]), name)
            self.assertTrue(torch.equal(whole[name], _source_bytes(layer, name, ids)), name)
        stats = streamer.last_gather_stats
        self.assertEqual((stats.requested_rows, stats.hot_hit_rows), (5, 2))

    def test_host_rows_match_the_device_gather_through_hot_and_cold_rows(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        ids = [6, 1, 3, 4, 0]
        for hot in ([1, 4], [6, 1, 3, 4, 0], [7]):  # mixed, all hit, all miss (7 never routes)
            layer = _nvfp4_layer(pinned=False)
            streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
            ExpertHotCache(streamer, len(hot)).reassign(hot)
            source_ids = torch.tensor(ids, device="cuda")

            def picked(rows, row_of_source):
                index = torch.as_tensor(row_of_source, device="cuda").long()
                return {n: rows[n][index].view(torch.uint8).cpu() for n in NVFP4_STREAM_TENSORS}

            got = [
                (chunk, row_of_source, picked(rows, row_of_source))
                for chunk, row_of_source, rows in streamer.iter_gather_experts_host(
                    source_ids, ids, chunk_rows=2
                )
            ]
            want = [
                (chunk.tolist(), row_of_source.tolist(), picked(rows, row_of_source))
                for chunk, row_of_source, rows in streamer.iter_gather_experts(
                    source_ids, chunk_rows=2
                )
            ]
            self.assertEqual(len(got), len(want))
            for (g_chunk, g_rows_of, g_rows), (w_chunk, w_rows_of, w_rows) in zip(got, want):
                self.assertEqual((g_chunk, g_rows_of), (w_chunk, w_rows_of), hot)
                for n in NVFP4_STREAM_TENSORS:
                    self.assertTrue(torch.equal(g_rows[n], w_rows[n]), (hot, n))

    def test_staging_stays_within_the_cap_and_eager_gathers_above_it_are_refused(self):
        from sglang.srt.layers.moe import expert_stream
        from sglang.srt.layers.moe.expert_format import DenseLayerFormat
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        class CappedDenseFormat(DenseLayerFormat):
            max_gather_rows = 16

        layer = _nvfp4_layer(pinned=False, experts=80)
        streamer = ExpertStreamer(
            layer, NVFP4_STREAM_TENSORS, format=CappedDenseFormat(NVFP4_STREAM_TENSORS)
        )
        ExpertHotCache(streamer, 1).reassign([0])
        expert_stream._STAGING.clear()
        ids = torch.arange(40, device="cuda")
        for _, row_of_source, rows in streamer.iter_gather_experts(ids):
            self.assertLessEqual(row_of_source.numel(), 16)
        self.assertTrue(expert_stream._STAGING)
        self.assertLessEqual(
            max(buffer.shape[0] for buffer in expert_stream._STAGING.values()), 64
        )
        from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy

        routes = torch.arange(40, device="cuda", dtype=torch.int32).reshape(10, 4)
        streamer.residency_policy = ExpertResidencyPolicy(80, 1, device=routes.device)
        with self.assertRaisesRegex(ValueError, "max_gather_rows"):
            streamer.gather(routes)
        # A refused forward records no routes.
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
