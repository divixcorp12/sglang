"""The block-verify dead-row guard (replace_dead_target_rows): which target-probability rows of a cut verify are
replaced, and that _accept_sampling_core applies it under block verification only (CPU; the kernel's output law is
checked by the GPU distribution test)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark import dspark_accept
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    replace_dead_target_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, NUM_ROWS, VOCAB = 4, 6, 8


def _rows(seed=0):
    generator = torch.Generator().manual_seed(seed)
    target = torch.softmax(torch.randn(BS, NUM_ROWS, VOCAB, generator=generator), dim=-1)
    draft = torch.softmax(torch.randn(BS, NUM_ROWS - 1, VOCAB, generator=generator), dim=-1)
    return target, draft


class TestReplaceDeadTargetRows(unittest.TestCase):
    def _check(self, verify_lens):
        target, draft = _rows()
        # Poison what the guard must replace: dead rows may hold NaN or Inf.
        for b, length in enumerate(verify_lens):
            target[b, length:] = float("nan")
        out = replace_dead_target_rows(
            target_probs=target,
            draft_probs=draft,
            verify_lens=torch.tensor(verify_lens),
        )
        self.assertTrue(torch.isfinite(out).all())
        uniform = torch.full((VOCAB,), 1.0 / VOCAB)
        for b, length in enumerate(verify_lens):
            for r in range(NUM_ROWS):
                if r < length:
                    self.assertTrue(torch.equal(out[b, r], target[b, r]), (b, r))
                elif r < NUM_ROWS - 1:
                    self.assertTrue(torch.equal(out[b, r], draft[b, r]), (b, r))
                else:
                    self.assertTrue(torch.allclose(out[b, r], uniform), (b, r))
        self.assertTrue(torch.allclose(out.sum(-1), torch.ones(BS, NUM_ROWS)))

    def test_ragged_cuts(self):
        self._check([1, 2, 4, 5])

    def test_cut_one_keeps_only_the_root_row(self):
        self._check([1, 1, 1, 1])

    def test_full_verify_replaces_nothing(self):
        target, draft = _rows()
        out = replace_dead_target_rows(
            target_probs=target,
            draft_probs=draft,
            verify_lens=torch.full((BS,), NUM_ROWS),
        )
        self.assertTrue(torch.equal(out, target))

    def test_live_bonus_row_of_a_full_request_is_untouched(self):
        # verify_len == S: row S - 1 is a live bonus row and must keep its target distribution.
        target, draft = _rows()
        out = replace_dead_target_rows(
            target_probs=target,
            draft_probs=draft,
            verify_lens=torch.tensor([NUM_ROWS, 2, NUM_ROWS, 3]),
        )
        self.assertTrue(torch.equal(out[0], target[0]))
        self.assertTrue(torch.equal(out[2], target[2]))
        self.assertFalse(torch.equal(out[1, NUM_ROWS - 1], target[1, NUM_ROWS - 1]))

    def test_does_not_mutate_inputs(self):
        target, draft = _rows()
        before = target.clone()
        replace_dead_target_rows(
            target_probs=target, draft_probs=draft, verify_lens=torch.tensor([1, 2, 3, 4])
        )
        self.assertTrue(torch.equal(target, before))


class TestGuardWiring(unittest.TestCase):
    """_accept_sampling_core hands the kernel guarded rows under block verification and the raw rows otherwise."""

    def _run(self, *, block_verification, cutoff_verify_lens):
        target, draft = _rows()
        seen = {}
        buffers = (
            torch.arange(NUM_ROWS * BS).view(BS, NUM_ROWS),
            torch.zeros(BS, NUM_ROWS, dtype=torch.long),
            torch.zeros(BS, NUM_ROWS, dtype=torch.long),
            torch.zeros(BS * NUM_ROWS, dtype=torch.int32),
            torch.zeros(BS, NUM_ROWS, dtype=torch.int32),
            torch.zeros(BS, dtype=torch.int32),
        )

        def fake_kernel(**kwargs):
            seen["target_probs"] = kwargs["target_probs"]

        sampling_info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones(BS, 1),
        )
        with (
            mock.patch.object(dspark_accept.SoftmaxTemp, "execute", return_value=target.view(-1, VOCAB)),
            mock.patch.object(dspark_accept, "_get_or_create_chain_verify_buffers", return_value=buffers),
            mock.patch.object(dspark_accept, "chain_speculative_sampling_triton", fake_kernel),
            mock.patch.object(dspark_accept, "_is_npu", False),
        ):
            dspark_accept._accept_sampling_core(
                candidates=torch.zeros(BS, NUM_ROWS, dtype=torch.long),
                target_logits=torch.zeros(BS * NUM_ROWS, VOCAB),
                draft_probs=draft,
                sampling_info=sampling_info,
                draft_input=None,
                gamma=NUM_ROWS - 1,
                verify_num_draft_tokens=NUM_ROWS,
                cutoff_verify_lens=cutoff_verify_lens,
                block_verification=block_verification,
            )
        return target, draft, seen["target_probs"]

    def test_block_with_cut_is_guarded(self):
        lens = torch.tensor([1, 2, 3, 6])
        target, draft, fed = self._run(block_verification=True, cutoff_verify_lens=lens)
        self.assertTrue(torch.equal(fed[0, 1:5], draft[0, 1:5]))
        self.assertTrue(torch.equal(fed[3], target[3]))

    def test_token_by_token_is_byte_identical(self):
        lens = torch.tensor([1, 2, 3, 6])
        target, _, fed = self._run(block_verification=False, cutoff_verify_lens=lens)
        self.assertTrue(torch.equal(fed, target))

    def test_block_without_cut_is_unchanged(self):
        target, _, fed = self._run(block_verification=True, cutoff_verify_lens=None)
        self.assertTrue(torch.equal(fed, target))


if __name__ == "__main__":
    unittest.main()
