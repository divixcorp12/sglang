import unittest

from sglang.srt.models.deepseek_v4_layer_major import ChunkSpan, chunk_spans, engram_history, keep_window_start


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

    def test_keep_window_floors_to_a_page(self):
        self.assertEqual(keep_window_start(seq_len=33000, window=128, page=256), 32768)
        self.assertEqual(keep_window_start(seq_len=32868, window=128, page=256), 32512)

    def test_engram_history_pads_left(self):
        self.assertEqual(engram_history([5, 6, 7, 8], start=2, n=3), [0, 5, 6])
        self.assertEqual(engram_history([5, 6, 7, 8], start=4, n=3), [6, 7, 8])


if __name__ == "__main__":
    unittest.main()
