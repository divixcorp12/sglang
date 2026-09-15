import unittest

from sglang.srt.layers.moe.expert_prediction.capture_schema import rows_per_request
from sglang.srt.layers.moe.expert_prediction.prefix_hash import PrefixHasher, SeenPrefixes


class TestPrefixHasher(unittest.TestCase):
    def test_same_prefix_same_hash_across_requests(self):
        hasher = PrefixHasher(max_requests=8)
        first = hasher.hash_rows(rid="a", positions=[0, 1, 2], token_ids=[5, 6, 7])
        second = hasher.hash_rows(rid="b", positions=[0, 1, 2], token_ids=[5, 6, 9])
        self.assertEqual(first[:2], second[:2])
        self.assertNotEqual(first[2], second[2])
        self.assertNotIn(0, first + second)

    def test_chunks_continue_the_chain(self):
        whole = PrefixHasher(max_requests=8).hash_rows(
            rid="a", positions=[0, 1, 2, 3], token_ids=[1, 2, 3, 4]
        )
        chunked = PrefixHasher(max_requests=8)
        parts = chunked.hash_rows(rid="a", positions=[0, 1], token_ids=[1, 2])
        parts += chunked.hash_rows(rid="a", positions=[2, 3], token_ids=[3, 4])
        self.assertEqual(whole, parts)

    def test_gap_marks_the_rest_of_the_request_unknown(self):
        hasher = PrefixHasher(max_requests=8)
        hasher.hash_rows(rid="a", positions=[0, 1], token_ids=[1, 2])
        self.assertEqual(hasher.hash_rows(rid="a", positions=[5, 6], token_ids=[3, 4]), [0, 0])
        self.assertEqual(hasher.hash_rows(rid="a", positions=[7], token_ids=[5]), [0])

    def test_request_starting_after_zero_is_unknown(self):
        hasher = PrefixHasher(max_requests=8)
        self.assertEqual(hasher.hash_rows(rid="a", positions=[64, 65], token_ids=[1, 2]), [0, 0])

    def test_evicts_oldest_request(self):
        hasher = PrefixHasher(max_requests=1)
        hasher.hash_rows(rid="a", positions=[0], token_ids=[1])
        hasher.hash_rows(rid="b", positions=[0], token_ids=[1])
        self.assertEqual(hasher.hash_rows(rid="a", positions=[1], token_ids=[2]), [0])


class TestSeenPrefixes(unittest.TestCase):
    def test_prefill_duplicates_dropped_decode_and_unknown_kept(self):
        seen = SeenPrefixes()
        self.assertEqual(seen.keep_mask(hashes=[11, 12], is_prefill=True), [True, True])
        self.assertEqual(
            seen.keep_mask(hashes=[11, 12, 13], is_prefill=True), [False, False, True]
        )
        self.assertEqual(seen.keep_mask(hashes=[13], is_prefill=False), [True])
        self.assertEqual(seen.keep_mask(hashes=[0, 0], is_prefill=True), [True, True])


class TestRowsPerRequest(unittest.TestCase):
    def test_extend_uses_extend_lengths(self):
        self.assertEqual(
            rows_per_request(is_extend=True, batch_size=2, extend_seq_lens=[3, 5]), (3, 5)
        )

    def test_decode_is_one_row_per_request(self):
        self.assertEqual(
            rows_per_request(is_extend=False, batch_size=3, extend_seq_lens=None), (1, 1, 1)
        )


if __name__ == "__main__":
    unittest.main()
