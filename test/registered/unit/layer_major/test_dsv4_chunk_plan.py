import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_v4_layer_major import (
    ChunkSpan,
    DeepseekV4LayerMajorAdapter,
    chunk_spans,
    engram_history,
    min_ring_len,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _adapter(*, chunk: int, page: int) -> DeepseekV4LayerMajorAdapter:
    a = DeepseekV4LayerMajorAdapter.__new__(DeepseekV4LayerMajorAdapter)
    a.chunk, a.page = chunk, page
    return a


class TestDsv4ChunkPlan(unittest.TestCase):
    def test_spans_cover_the_suffix_with_a_short_final_chunk(self):
        spans = chunk_spans(prefix_len=0, seq_len=33000, chunk=4096)
        self.assertEqual(len(spans), 9)
        self.assertEqual(spans[0], ChunkSpan(index=0, start=0, end=4096))
        self.assertEqual(spans[-1], ChunkSpan(index=8, start=32768, end=33000))

    def test_first_chunk_with_prefix_is_not_remapped(self):
        # The ring maps suffix positions only; chunk 0's predecessor window lives in the prefix's own slots.
        spans = chunk_spans(prefix_len=512, seq_len=512 + 8192, chunk=4096)
        self.assertEqual([(s.start, s.end) for s in spans], [(512, 4608), (4608, 8704)])

    def test_engram_history_pads_left(self):
        self.assertEqual(engram_history([5, 6, 7, 8], start=2, n=3), [0, 5, 6])
        self.assertEqual(engram_history([5, 6, 7, 8], start=4, n=3), [6, 7, 8])

    def test_min_ring_len_matches_production_geometry(self):
        # page (256) >= window (128), so margin's page-ceil is exactly one page.
        self.assertEqual(min_ring_len(chunk=4096, page=256, window=128), 4096 + 256)

    def test_ring_below_minimum_raises(self):
        # window (128) > page (64): margin's page-ceil is 2 pages, but the ring only holds 1.
        a = _adapter(chunk=4096, page=64)
        with self.assertRaises(ValueError):
            a._check_ring_len(a.chunk + a.page)

    def test_backend_and_allocator_read_lazily_at_construction(self):
        # TpModelWorker builds the adapter before ModelRunner sets attn_backend (missing entirely) and
        # before init_memory_pools() replaces the token_to_kv_pool_allocator placeholder (None). Construction
        # itself must not touch either field.
        runner = SimpleNamespace(model=SimpleNamespace(model=object()), page_size=256,
                                 server_args=SimpleNamespace(chunked_prefill_size=4096),
                                 token_to_kv_pool_allocator=None)
        a = DeepseekV4LayerMajorAdapter(runner)
        self.assertIsNone(a.allocator)
        with self.assertRaises(AttributeError):
            _ = a.backend
        runner.attn_backend = "the-backend"
        runner.token_to_kv_pool_allocator = "the-allocator"
        self.assertEqual(a.backend, "the-backend")
        self.assertEqual(a.allocator, "the-allocator")


if __name__ == "__main__":
    unittest.main()
