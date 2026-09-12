import unittest
import warnings

import torch

from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


class _Layer(torch.nn.Module):
    pass


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
