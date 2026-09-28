"""Pure unit tests for tail_run_spans, the C1 fix's span-selection helper (deepseek_v4_layer_major.py).
The commit-portable write-coverage RED/GREEN test lives in test_c1_late_layer_write_coverage.py, which
uses no symbol added by the fix so it can run unchanged against 43d7813a41 too."""

import unittest

from sglang.srt.models.deepseek_v4_layer_major import DSV4_WINDOW, chunk_spans, tail_run_spans
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 4096


class TestTailRunSpans(unittest.TestCase):
    """The spans selected must cover decode's reach [s - window, s)."""

    def _spans(self, seq_len):
        return chunk_spans(prefix_len=0, seq_len=seq_len, chunk=CHUNK)

    def test_exact_multiple_of_chunk_needs_only_the_final_span(self):
        spans = self._spans(CHUNK * 2)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [1])

    def test_final_span_of_8_rows_also_needs_the_penultimate_tail(self):
        spans = self._spans(CHUNK * 2 + 8)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [1, 2])

    def test_final_span_of_127_rows_also_needs_the_penultimate_tail(self):
        spans = self._spans(CHUNK * 2 + 127)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [1, 2])

    def test_final_span_of_exactly_the_window_needs_no_penultimate_tail(self):
        spans = self._spans(CHUNK * 2 + DSV4_WINDOW)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [2])

    def test_single_span_suffix_needs_only_itself(self):
        # No penultimate span exists; this is the prefix case (review focus item 1), not C1.
        spans = self._spans(8)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [0])


if __name__ == "__main__":
    unittest.main()
