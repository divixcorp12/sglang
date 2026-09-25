"""convert_logprob_style tolerates a batch output that carries no logprobs.

A request that asked for top logprobs but was refused at admission (it never ran a forward) reaches the
TokenizerManager in a batch whose logprob lists are all None. ``len(None)`` there stopped the TokenizerManager and
the server (DSV4.1 copy-engine soak s2, docs/superpowers/plans/2026-09-25-dsv41-copy-engine-soak.md).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

LISTS = (
    "input_token_logprobs_val input_token_logprobs_idx output_token_logprobs_val output_token_logprobs_idx "
    "input_top_logprobs_val input_top_logprobs_idx output_top_logprobs_val output_top_logprobs_idx "
    "input_top_logprobs_val_flat input_top_logprobs_idx_flat input_top_logprobs_flat_null_prefix "
    "input_token_ids_logprobs_val input_token_ids_logprobs_idx output_token_ids_logprobs_val "
    "output_token_ids_logprobs_idx"
).split()


def state():
    s = SimpleNamespace(**{name: [] for name in LISTS if "flat" not in name})
    s.obj = SimpleNamespace(top_logprobs_num=5, token_ids_logprob=[1, 2])
    return s


# unittest.TestCase, not CustomTestCase: sglang.test.test_utils imports pyarrow, which fails collection on divix01.
class TestLogprobStyleWithoutLogprobs(unittest.TestCase):
    def test_batch_without_logprob_lists(self):
        manager = SimpleNamespace(add_logprob_to_meta_info=Mock())
        recv = SimpleNamespace(**{name: None for name in LISTS})
        st = state()
        TokenizerManager.convert_logprob_style(manager, {}, st, 5, [1, 2], False, recv, 0)
        self.assertEqual(st.output_top_logprobs_val, [])
        self.assertEqual(st.output_token_ids_logprobs_val, [])
        manager.add_logprob_to_meta_info.assert_called_once()

    def test_batch_with_logprob_lists_still_extends(self):
        manager = SimpleNamespace(add_logprob_to_meta_info=Mock())
        recv = SimpleNamespace(**{name: [[0.5]] for name in LISTS})
        recv.input_top_logprobs_val_flat = None
        st = state()
        TokenizerManager.convert_logprob_style(manager, {}, st, 5, [1, 2], False, recv, 0)
        self.assertEqual(st.input_top_logprobs_val, [0.5])
        self.assertEqual(st.output_top_logprobs_val, [0.5])
        self.assertEqual(st.input_token_ids_logprobs_val, [0.5])
        self.assertEqual(st.output_token_ids_logprobs_val, [0.5])


if __name__ == "__main__":
    unittest.main()
