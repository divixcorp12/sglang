"""Grammar token-id sync runs only when there is another TP rank to agree with.

On one rank the all-reduce is an identity, and issuing it creates the NCCL
communicator on the first grammar request: ~512 MiB of device memory allocated
mid-serving, which crashed a single-GPU server with little headroom left.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import sampler as sampler_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sampler():
    # Skip __init__: it resolves process groups from the parallel state.
    s = object.__new__(sampler_mod.Sampler)
    s.tp_sync_group = object()
    return s


def _sync(world_size, grammars):
    ids = torch.tensor([3, 1, 4])
    with mock.patch.object(
        sampler_mod.dist, "get_world_size", return_value=world_size
    ), mock.patch.object(sampler_mod.dist, "all_reduce") as all_reduce:
        _sampler()._sync_token_ids_across_tp(ids, SimpleNamespace(grammars=grammars))
    return all_reduce


class TestSamplerTokenSync(CustomTestCase):
    def test_single_rank_grammar_skips_the_collective(self):
        self.assertFalse(_sync(1, [object()]).called)

    def test_multi_rank_grammar_syncs(self):
        all_reduce = _sync(2, [object()])
        all_reduce.assert_called_once()
        self.assertEqual(all_reduce.call_args.kwargs["op"], sampler_mod.dist.ReduceOp.MIN)

    def test_multi_rank_without_grammar_does_not_sync(self):
        self.assertFalse(_sync(2, None).called)


if __name__ == "__main__":
    unittest.main()
