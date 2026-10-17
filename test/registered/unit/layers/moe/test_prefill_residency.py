"""Layer-major expert residency (prefill_residency.py): a chunk's misses are kept for the layer's later chunks (CPU)."""

import contextlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_stream import ExpertStreamer, host_row_of_source
from sglang.srt.layers.moe.prefill_residency import Exl3LayerMajorResidency, PrefillResidency
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPERTS = 16


def _all_miss(ids):
    return torch.full_like(ids, -1), torch.zeros_like(ids, dtype=torch.bool)


def _streamer(layer_id=0):
    """A streamer whose hot cache misses everything and whose source copies are recorded."""
    layer = torch.nn.Module()
    layer.layer_id = layer_id
    layer.rows = torch.arange(EXPERTS * 4, dtype=torch.int32).reshape(EXPERTS, 4) + 1000 * layer_id
    streamer = ExpertStreamer(layer, ("rows",))
    streamer.hot_cache = SimpleNamespace(
        lookup=_all_miss, capacity=1, expert_to_slot=torch.full((EXPERTS,), -1), device=torch.device("cpu")
    )
    reads = []

    def copy_rows(source_ids, outputs):
        reads.append(source_ids.tolist())
        for name, output in outputs.items():
            torch.index_select(getattr(layer, name), 0, source_ids.long(), out=output)
        return 0

    streamer._copy_source_rows = copy_rows
    return layer, streamer, reads


def _residency(streamer, rows=8):
    return PrefillResidency.for_streamer(streamer, rows)


class _ClearStaging(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()

    def tearDown(self):
        expert_stream._STAGING.clear()


class TestGather(_ClearStaging):
    def _gather(self, streamer, ids):
        """The device-order gather: every source's staged row, as the MoE reads it."""
        source = torch.tensor(ids)
        row_of_source, rows = streamer._gather_experts(source)
        return rows["rows"][row_of_source.long()]

    def _gather_host(self, streamer, ids):
        row_of_source, rows = streamer._gather_experts_host(torch.tensor(ids))
        return rows["rows"][torch.tensor(row_of_source)]

    def test_a_later_chunk_reads_only_the_experts_no_earlier_chunk_read(self):
        layer, streamer, reads = _streamer()
        for gather in (self._gather, self._gather_host):
            expert_stream._STAGING.clear()
            reads.clear()
            _residency(streamer).activate(streamer)
            self.assertTrue(torch.equal(gather(streamer, [1, 3, 5]), layer.rows[[1, 3, 5]]))
            self.assertTrue(torch.equal(gather(streamer, [3, 5, 7]), layer.rows[[3, 5, 7]]))
            self.assertTrue(torch.equal(gather(streamer, [1, 7]), layer.rows[[1, 7]]))
            self.assertEqual(reads, [[1, 3, 5], [7]], gather.__name__)
            stats = streamer.last_gather_stats
            self.assertEqual((stats.miss_rows, stats.residency_hit_rows, stats.hot_hit_rows), (0, 2, 0))

    def test_a_full_buffer_stops_adopting_and_the_rest_are_read_each_chunk(self):
        layer, streamer, reads = _streamer()
        _residency(streamer, rows=2).activate(streamer)
        self._gather(streamer, [1, 3, 5])
        self.assertTrue(torch.equal(self._gather(streamer, [1, 3, 5]), layer.rows[[1, 3, 5]]))
        self.assertEqual(reads, [[1, 3, 5], [5]])

    def test_without_residency_every_chunk_reads_again(self):
        layer, streamer, reads = _streamer()
        for gather in (self._gather, self._gather_host):
            reads.clear()
            self.assertTrue(torch.equal(gather(streamer, [1, 3]), layer.rows[[1, 3]]), gather.__name__)
            self.assertTrue(torch.equal(gather(streamer, [1, 3]), layer.rows[[1, 3]]), gather.__name__)
            self.assertEqual(reads, [[1, 3], [1, 3]])
            self.assertEqual(streamer.last_gather_stats.residency_hit_rows, 0)

    def test_activating_another_layer_forgets_the_previous_layers_rows(self):
        layer0, streamer0, _ = _streamer(0)
        layer1, streamer1, reads1 = _streamer(1)
        residency = _residency(streamer0)
        residency.activate(streamer0)
        self._gather(streamer0, [2, 4])
        residency.activate(streamer1)
        self.assertIsNone(streamer0.prefill_residency)
        # Layer 1's experts 2 and 4 are its own rows, read from its own source.
        self.assertTrue(torch.equal(self._gather(streamer1, [2, 4]), layer1.rows[[2, 4]]))
        self.assertEqual(reads1, [[2, 4]])

    def test_a_buffer_refuses_a_layer_whose_rows_do_not_fit(self):
        _, streamer, _ = _streamer()
        residency = PrefillResidency(rows=2, num_experts=EXPERTS, specs={"rows": ((8,), torch.int32)},
                                     device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "do not fit"):
            residency.activate(streamer)


class TestPrefillFills(unittest.TestCase):
    def test_fills_skip_resident_experts(self):
        _, streamer, _ = _streamer()
        claimed = []

        @contextlib.contextmanager
        def host_use():
            yield

        streamer.pinned_host_cache = SimpleNamespace(
            row_fills=object(), host_use=host_use, finish_fills=lambda: None,
            prefetch_rows=lambda experts, protected: claimed.append(experts) or 0,
        )
        residency = _residency(streamer)
        residency.activate(streamer)
        residency.adopt(torch.tensor([3, 5]), {"rows": torch.zeros(2, 4, dtype=torch.int32)}, 2)
        with streamer.prefill_fills(torch.tensor([1, 3, 5, 7])):
            pass
        self.assertEqual(claimed, [[1, 7]])


class TestHostRowOfSource(unittest.TestCase):
    def test_resident_hits_are_staged_not_read_in_place(self):
        slots, hits = [4, -1, 6], [True, True, True]
        self.assertEqual(host_row_of_source(slots, hits), [4, -1, 6])
        self.assertEqual(host_row_of_source(slots, hits, in_place=False), [0, 1, 2])
        self.assertEqual(host_row_of_source([4, -1], [True, False], in_place=False), [1, 0])


class TestLayerMajorResidency(unittest.TestCase):
    def test_each_layer_of_a_pass_is_served_in_turn_and_none_after(self):
        _, streamer0, _ = _streamer(0)
        _, streamer1, _ = _streamer(1)
        caches = {0: SimpleNamespace(capacity=1, streamer=streamer0), 1: SimpleNamespace(capacity=1, streamer=streamer1),
                  2: SimpleNamespace(capacity=0, streamer=None)}
        runner = SimpleNamespace(expert_hot_cache_manager=SimpleNamespace(caches=caches))
        residency = Exl3LayerMajorResidency(runner, rows=4)
        residency.begin(range(0, 3))
        residency.make_resident(0)
        self.assertIs(streamer0.prefill_residency, residency.buffer)
        residency.make_resident(1)
        self.assertIsNone(streamer0.prefill_residency)
        self.assertIs(streamer1.prefill_residency, residency.buffer)
        residency.make_resident(2)  # no hot cache: streams as before
        self.assertIsNone(streamer1.prefill_residency)
        residency.make_resident(1)
        residency.restore()
        self.assertIsNone(streamer1.prefill_residency)
        residency.end()
        self.assertEqual(residency.buffer.rows, 4)

    def test_a_model_without_hot_caches_allocates_nothing(self):
        runner = SimpleNamespace(expert_hot_cache_manager=None)
        residency = Exl3LayerMajorResidency(runner, rows=4)
        residency.begin(range(0, 2))
        residency.make_resident(0)
        residency.restore()
        residency.end()
        self.assertIsNone(residency.buffer)


if __name__ == "__main__":
    unittest.main()
