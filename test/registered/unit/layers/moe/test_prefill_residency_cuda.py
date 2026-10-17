"""Layer-major expert residency through a real hot cache on the GPU: hot and resident hits mixed in one gather (CUDA)."""

import unittest

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_stream import ExpertStreamer
from sglang.srt.layers.moe.prefill_residency import PrefillResidency
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")


class _Layer(torch.nn.Module):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestResidencyThroughTheHotCache(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()

    def _layer(self, experts=64, hot=8, row_bytes=4096):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _Layer()
        layer.host_rows = torch.nn.Parameter(
            torch.randint(0, 256, (experts, row_bytes), dtype=torch.uint8).pin_memory(), requires_grad=False
        )
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertHotCache(streamer, capacity=hot)
        cache.reassign(list(range(0, experts, experts // hot)))  # experts 0, 8, 16, ...
        return layer, streamer

    def _check(self, layer, streamer, ids, host):
        source = torch.tensor(ids, device="cuda")
        if host:
            row_of_source, rows = streamer._gather_experts_host(source)
            row_of_source = torch.tensor(row_of_source, device="cuda")
        else:
            row_of_source, rows = streamer._gather_experts(source)
        torch.cuda.synchronize()
        staged = rows["host_rows"][row_of_source.long()].cpu()
        self.assertTrue(torch.equal(staged, layer.host_rows[torch.tensor(ids)]), ids)
        return streamer.last_gather_stats

    def test_chunks_of_a_layer_read_each_miss_once_and_stage_the_same_bytes(self):
        for host in (False, True):
            expert_stream._STAGING.clear()
            layer, streamer = self._layer()
            PrefillResidency.for_streamer(streamer, 32).activate(streamer)
            first = self._check(layer, streamer, [0, 1, 2, 3, 8, 9], host)  # hot 0 and 8
            self.assertEqual((first.hot_hit_rows, first.miss_rows, first.residency_hit_rows), (2, 4, 0))
            # Hot, resident and new experts in one gather, in an order that interleaves them.
            second = self._check(layer, streamer, [1, 8, 10, 2, 16, 3, 11], host)
            self.assertEqual((second.hot_hit_rows, second.miss_rows, second.residency_hit_rows), (2, 2, 3))
            # Every one hot or resident now: staged, not read in place, and nothing read.
            third = self._check(layer, streamer, [9, 0, 10, 11, 1], host)
            self.assertEqual((third.hot_hit_rows, third.miss_rows, third.residency_hit_rows), (1, 0, 4))

    def test_an_all_hot_gather_still_reads_the_hot_slots_in_place(self):
        layer, streamer = self._layer()
        PrefillResidency.for_streamer(streamer, 8).activate(streamer)
        source = torch.tensor([0, 8, 16], device="cuda")
        _, rows = streamer._gather_experts(source)
        self.assertIs(rows, streamer.hot_cache.tensors)


if __name__ == "__main__":
    unittest.main()
