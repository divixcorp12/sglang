"""CPU tests for chunked expert gathers and the format's staging cap."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_format import DenseLayerFormat
from sglang.srt.layers.moe.expert_residency import ExpertResidencyPolicy
from sglang.srt.layers.moe.expert_stream import ExpertGatherStats, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class CappedDenseFormat(DenseLayerFormat):
    def __init__(self, tensor_names, max_gather_rows):
        super().__init__(tensor_names)
        self.max_gather_rows = max_gather_rows


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


def _streamer(experts=16, max_gather_rows=None):
    layer = torch.nn.Module()
    layer.rows = torch.arange(experts * 4, dtype=torch.int32).reshape(experts, 4)
    expert_format = (
        None
        if max_gather_rows is None
        else CappedDenseFormat(("rows",), max_gather_rows)
    )
    streamer = ExpertStreamer(layer, ("rows",), format=expert_format)
    streamer.hot_cache = SimpleNamespace(lookup=_all_miss, capacity=1)

    def copy_rows(source_ids, outputs):
        for name, output in outputs.items():
            torch.index_select(getattr(layer, name), 0, source_ids.long(), out=output)
        return 0

    streamer._copy_source_rows = copy_rows
    return layer, streamer


class _ClearStaging(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()


class TestStagingFloor(_ClearStaging):
    def _staged_rows(self):
        return expert_stream._STAGING[("rows", torch.int32, "cpu", (4,))].shape[0]

    def test_default_floor_stages_a_whole_layer(self):
        _, streamer = _streamer(experts=80)
        ids = torch.tensor([[1, 2], [3, 1]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        streamer._gather_cached(source_ids, compact_ids, ids)
        self.assertEqual(self._staged_rows(), 80)

    def test_max_gather_rows_caps_the_floor(self):
        _, streamer = _streamer(experts=80, max_gather_rows=8)
        ids = torch.tensor([[1, 2], [3, 1]])
        source_ids, compact_ids = streamer._plan_eager_routes(ids)
        streamer._gather_cached(source_ids, compact_ids, ids)
        # NO_DEDUP_LIMIT still pads deduplicated staging to 64 rows.
        self.assertEqual(self._staged_rows(), 64)


class TestGatherExperts(_ClearStaging):
    def test_rows_are_indexed_by_source_position(self):
        layer, streamer = _streamer()
        ids = torch.tensor([9, 2, 5])
        row_of_source, rows = streamer.gather_experts(ids)
        self.assertEqual(tuple(row_of_source.shape), (3,))
        self.assertTrue(torch.equal(rows["rows"][row_of_source.long()], layer.rows[ids]))
        self.assertEqual(streamer.last_gather_stats.requested_rows, 3)

    def test_invalid_ids_are_refused(self):
        _, streamer = _streamer(max_gather_rows=4)
        cases = (
            (torch.tensor([[1, 2]]), "1-D"),
            (torch.tensor([], dtype=torch.long), "nonempty"),
            (torch.tensor([0, 1, 2, 3, 4]), "max_gather_rows"),
            (torch.tensor([16]), "outside"),
            (torch.tensor([3, 3]), "distinct"),
        )
        for ids, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    streamer.gather_experts(ids)

    def test_routes_are_recorded_only_by_record_routes(self):
        _, streamer = _streamer()
        streamer.residency_policy = ExpertResidencyPolicy(16, 4, device="cpu")
        streamer.gather_experts(torch.tensor([1, 2]))
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 0.0)
        streamer.record_routes(torch.tensor([[1, 2], [2, 5]]))
        self.assertEqual(float(streamer.residency_policy.pending_counts.sum()), 4.0)

    def test_prefetch_hooks_are_refused(self):
        _, streamer = _streamer()
        streamer.next_layer_prefetch = lambda ids: None
        with self.assertRaisesRegex(ValueError, "prefetch"):
            streamer.gather_experts(torch.tensor([1]))


class TestIterGatherExperts(_ClearStaging):
    def test_chunks_equal_one_gather_and_sum_their_stats(self):
        layer, streamer = _streamer()
        ids = torch.tensor([3, 9, 1, 14, 6])
        chunks = []
        for chunk, row_of_source, rows in streamer.iter_gather_experts(ids, chunk_rows=2):
            chunks.append((chunk.tolist(), rows["rows"][row_of_source.long()].clone()))
        self.assertEqual([chunk for chunk, _ in chunks], [[3, 9], [1, 14], [6]])
        self.assertTrue(torch.equal(torch.cat([rows for _, rows in chunks]), layer.rows[ids]))
        stats = streamer.last_gather_stats
        self.assertEqual(
            (stats.requested_rows, stats.miss_rows, stats.unique_miss_rows), (5, 5, 5)
        )

    def test_the_format_cap_is_the_default_chunk(self):
        _, streamer = _streamer(max_gather_rows=2)
        sizes = [
            chunk.numel()
            for chunk, _, _ in streamer.iter_gather_experts(torch.tensor([3, 9, 1, 14, 6]))
        ]
        self.assertEqual(sizes, [2, 2, 1])

    def test_chunks_above_the_cap_are_refused(self):
        _, streamer = _streamer(max_gather_rows=2)
        with self.assertRaisesRegex(ValueError, "max_gather_rows"):
            list(streamer.iter_gather_experts(torch.tensor([1, 2, 3]), chunk_rows=3))

    def test_ids_must_be_distinct_across_chunks(self):
        _, streamer = _streamer()
        with self.assertRaisesRegex(ValueError, "distinct"):
            list(streamer.iter_gather_experts(torch.tensor([1, 2, 1]), chunk_rows=2))

    def test_no_ids_yield_nothing(self):
        _, streamer = _streamer()
        self.assertEqual(
            list(streamer.iter_gather_experts(torch.tensor([], dtype=torch.long))), []
        )


class TestSumGatherStats(unittest.TestCase):
    def test_counts_add_and_a_fallback_marks_the_sum(self):
        total = expert_stream._sum_gather_stats(
            [
                ExpertGatherStats(2, 1, 1, gather_fallback_used=False, host_read_rows=1),
                ExpertGatherStats(3, 0, 3, gather_fallback_used=True, host_read_rows=2),
            ]
        )
        self.assertEqual((total.requested_rows, total.hot_hit_rows, total.miss_rows), (5, 1, 4))
        self.assertTrue(total.gather_fallback_used)
        self.assertEqual(total.host_read_rows, 3)


if __name__ == "__main__":
    unittest.main()
