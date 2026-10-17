"""Losslessness of DSpark's sampled accept paths, on the real kernels.

A synthetic position-dependent Markov target and draft over a small vocabulary drive ``accept_draft_tokens``. Each
verify's output (accepted drafts, then the bonus) is completed to the full chain length by sampling the target, so a
lossless accept makes the completed sequence an exact draw from the target's joint distribution, which the test
enumerates. A chi-square test over the joint and one per position compares the counts to it; a self-check shows the
same counts reject a reference shifted by 1%.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.dspark_sampled_accept import (
    sampled_chain_accept,
    sampled_chain_bonus_weights,
    sampled_chain_accept_triton,
    sampled_verify_target_probs,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_verify import accept_draft_tokens
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
VOCAB = 4
NUM_SLOTS = 6  # speculative_num_draft_tokens: the anchor and five drafts.
NUM_SAMPLES = 1 << 21
_CHUNK = 1 << 15
TEMPERATURE = 0.8
# Per-test false-alarm rate; a 1% shift at these sample counts sits far beyond it (test_detects_a_one_percent_shift).
P_VALUE_FLOOR = 1e-6


def _chi2_pvalue(observed: torch.Tensor, expected: torch.Tensor) -> float:
    """Pearson chi-square upper tail, cells with expectation under 5 pooled; Wilson-Hilferty for the chi-square
    quantile (accurate to well below the floor used here at these degrees of freedom)."""
    observed = observed.double().flatten()
    expected = expected.double().flatten()
    # A draw in a cell the target cannot produce (filtered out by top-k/top-p) fails outright.
    impossible = expected == 0
    if float(observed[impossible].sum()) > 0:
        return 0.0
    observed, expected = observed[~impossible], expected[~impossible]
    small = expected < 5
    pooled_observed, pooled_expected = observed[small].sum(), expected[small].sum()
    observed, expected = observed[~small], expected[~small]
    if float(pooled_expected) >= 5:
        observed = torch.cat([observed, pooled_observed.view(1)])
        expected = torch.cat([expected, pooled_expected.view(1)])
    stat = float(((observed - expected) ** 2 / expected).sum())
    dof = observed.numel() - 1
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _filter(probs: torch.Tensor, top_k, top_p) -> torch.Tensor:
    """Top-k, then top-p (keep a token while the mass before it is under top_p), renormalized after each."""
    if top_k is not None:
        kth = probs.topk(top_k, dim=-1).values[..., -1:]
        probs = torch.where(probs >= kth, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(-1, keepdim=True)
    if top_p is not None:
        sorted_p, order = probs.sort(dim=-1, descending=True)
        keep_sorted = (sorted_p.cumsum(-1) - sorted_p) < top_p
        keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


class _Chain:
    """Target and draft: a [slot, previous token] table of next-token logits each, the draft a noisy copy of the
    target so acceptance is partial. The anchor is token 0."""

    def __init__(self, *, seed: int, top_k=None, top_p=None, greedy_draft: bool):
        g = torch.Generator().manual_seed(seed)
        target = torch.randn((NUM_SLOTS, VOCAB, VOCAB), generator=g) * 1.5
        draft = target + torch.randn((NUM_SLOTS, VOCAB, VOCAB), generator=g)
        self.target_logits = target.to(DEVICE)
        self.draft_logits = draft.to(DEVICE)
        self.draft_probs = torch.softmax(self.draft_logits, dim=-1)
        self.top_k, self.top_p = top_k, top_p
        self.greedy_draft = greedy_draft
        self.target_probs = _filter(
            torch.softmax(self.target_logits / TEMPERATURE, dim=-1), top_k, top_p
        )

    def sampling_info(self, n: int):
        return SimpleNamespace(
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_k_sampling=self.top_k is not None,
            need_top_p_sampling=self.top_p is not None,
            temperatures=torch.full((n, 1), TEMPERATURE, device=DEVICE),
            top_ks=torch.full(
                (n,), self.top_k or VOCAB, dtype=torch.int32, device=DEVICE
            ),
            top_ps=torch.full((n,), self.top_p or 1.0, device=DEVICE),
        )

    def propose(self, n: int):
        """[n, S] candidates and the [n, S - 1, V] draft distribution each draft was taken from."""
        candidates = torch.zeros((n, NUM_SLOTS), dtype=torch.int64, device=DEVICE)
        draft_probs = torch.empty((n, NUM_SLOTS - 1, VOCAB), device=DEVICE)
        prev = candidates[:, 0]
        for t in range(1, NUM_SLOTS):
            q = self.draft_probs[t - 1][prev]
            if self.greedy_draft:
                x = self.draft_logits[t - 1][prev].argmax(-1)
            else:
                x = torch.multinomial(q, 1).squeeze(-1)
            candidates[:, t] = x
            draft_probs[:, t - 1] = q
            prev = x
        return candidates, draft_probs

    def target_rows(self, candidates: torch.Tensor) -> torch.Tensor:
        """[n * S, V] target logits, row i conditioned on candidate i."""
        rows = torch.stack(
            [self.target_logits[i][candidates[:, i]] for i in range(NUM_SLOTS)], dim=1
        )
        return rows.reshape(-1, VOCAB)

    def accept(
        self, candidates, draft_probs, target_logits, verify_lens, *, block_verify=False
    ):
        n = candidates.shape[0]
        draft_block = DraftBlockResult(
            draft_tokens=candidates[:, 1:],
            corrected_logits=(
                None
                if self.greedy_draft
                # SoftmaxTemp divides by the request temperature; this undoes it so the accept's q is draft_probs.
                else TEMPERATURE * draft_probs.clamp_min(1e-30).log()
            ),
            greedy_mask=torch.zeros((n,), dtype=torch.bool, device=DEVICE),
            temperatures=torch.full((n,), TEMPERATURE, device=DEVICE),
            greedy_draft=self.greedy_draft,
        )
        kwargs = {"block_verify": True} if block_verify else {}
        correct_len, bonus, _ = accept_draft_tokens(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=self.sampling_info(n),
            draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
            gamma=NUM_SLOTS - 1,
            verify_num_draft_tokens=NUM_SLOTS,
            cutoff_verify_lens=verify_lens,
            **kwargs,
        )
        return correct_len.to(torch.int64), bonus

    def verify_and_complete(self, n: int, verify_lens, **accept_kwargs):
        """[n, S] completed sequences and the accept's correct_len. Dead rows (past verify_lens) carry what a cut
        verify leaves there: zero candidates, garbage logits and draft probabilities."""
        candidates, draft_probs = self.propose(n)
        target_logits = self.target_rows(candidates).view(n, NUM_SLOTS, VOCAB)
        if verify_lens is not None:
            slots = torch.arange(NUM_SLOTS, device=DEVICE)
            dead = slots[None, :] >= verify_lens[:, None]
            candidates = candidates.masked_fill(dead, 0)
            target_logits = torch.where(
                dead[..., None], torch.randn_like(target_logits) * 4, target_logits
            )
            draft_probs = torch.where(
                dead[:, 1:, None],
                torch.softmax(torch.randn_like(draft_probs) * 4, -1),
                draft_probs,
            )
        # flashinfer's top-p renorm scratch grows with the row count; chunk below it.
        chunks = [
            self.accept(
                candidates[s : s + _CHUNK],
                draft_probs[s : s + _CHUNK],
                target_logits[s : s + _CHUNK].reshape(-1, VOCAB),
                None if verify_lens is None else verify_lens[s : s + _CHUNK],
                **accept_kwargs,
            )
            for s in range(0, n, _CHUNK)
        ]
        correct_len = torch.cat([c[0] for c in chunks])
        bonus = torch.cat([c[1] for c in chunks])
        if verify_lens is not None:
            assert bool((correct_len <= verify_lens - 1).all())
        slots = torch.arange(NUM_SLOTS, device=DEVICE)
        seq = torch.where(
            slots[None, :] < correct_len[:, None], candidates.roll(-1, dims=1), 0
        )
        seq.scatter_(1, correct_len[:, None], bonus[:, None])
        prev = torch.zeros((n,), dtype=torch.int64, device=DEVICE)
        for t in range(NUM_SLOTS):
            fill = t > correct_len
            sample = torch.multinomial(self.target_probs[t][prev], 1).squeeze(-1)
            seq[:, t] = torch.where(fill, sample, seq[:, t])
            prev = seq[:, t]
        return seq, correct_len

    def joint(self) -> torch.Tensor:
        """[V ** S] exact target probability of each sequence, token 0 most significant."""
        joint = self.target_probs[0][0]
        for t in range(1, NUM_SLOTS):
            last = torch.arange(joint.numel(), device=DEVICE) % VOCAB
            joint = (joint[:, None] * self.target_probs[t][last]).reshape(-1)
        return joint


def _pvalues(seq: torch.Tensor, joint: torch.Tensor) -> dict:
    n = seq.shape[0]
    weights = VOCAB ** torch.arange(NUM_SLOTS - 1, -1, -1, device=DEVICE)
    codes = (seq * weights).sum(-1)
    counts = torch.bincount(codes, minlength=joint.numel())
    out = {"joint": _chi2_pvalue(counts, joint * n)}
    by_pos = joint.view((VOCAB,) * NUM_SLOTS)
    for t in range(NUM_SLOTS):
        dims = tuple(d for d in range(NUM_SLOTS) if d != t)
        marginal = by_pos.sum(dims) if dims else by_pos
        out[f"pos{t}"] = _chi2_pvalue(torch.bincount(seq[:, t], minlength=VOCAB), marginal * n)
    return out


_CUTS = [None, 1, 2, 3, 4, 5, 6, "mixed"]
_FILTERS = [(None, None), (2, None), (None, 0.75), (3, 0.8)]


def _verify_lens(cut, n: int, seed: int):
    if cut is None:
        return None
    if cut == "mixed":
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        return torch.randint(1, NUM_SLOTS + 1, (n,), device=DEVICE, generator=g)
    return torch.full((n,), cut, dtype=torch.int64, device=DEVICE)


class TestSampledAcceptDistribution(CustomTestCase):
    def _check(self, *, greedy_draft: bool, block_verify: bool = False, seed0: int):
        for fi, (top_k, top_p) in enumerate(_FILTERS):
            chain = _Chain(seed=seed0 + fi, top_k=top_k, top_p=top_p, greedy_draft=greedy_draft)
            joint = chain.joint()
            self.assertAlmostEqual(float(joint.sum()), 1.0, places=5)
            for ci, cut in enumerate(_CUTS):
                torch.manual_seed(seed0 * 1000 + fi * 100 + ci)
                seq, _ = chain.verify_and_complete(
                    NUM_SAMPLES,
                    _verify_lens(cut, NUM_SAMPLES, seed0 + ci),
                    block_verify=block_verify,
                )
                for name, pvalue in _pvalues(seq, joint).items():
                    with self.subTest(top_k=top_k, top_p=top_p, cut=cut, test=name):
                        self.assertGreater(pvalue, P_VALUE_FLOOR)

    def test_greedy_draft_token_by_token(self):
        self._check(greedy_draft=True, seed0=11)

    def test_sampled_draft_classic_control(self):
        """The classic kernel with a sampled draft and no cut, as a control for the harness itself."""
        for fi, (top_k, top_p) in enumerate(_FILTERS):
            chain = _Chain(seed=40 + fi, top_k=top_k, top_p=top_p, greedy_draft=False)
            torch.manual_seed(40 + fi)
            seq, _ = chain.verify_and_complete(NUM_SAMPLES, None)
            for name, pvalue in _pvalues(seq, chain.joint()).items():
                with self.subTest(top_k=top_k, top_p=top_p, test=name):
                    self.assertGreater(pvalue, P_VALUE_FLOOR)

    def test_detects_a_one_percent_shift(self):
        """Power: the counts a correct accept produces reject a reference with 1% of position 1's mass moved."""
        chain = _Chain(seed=7, greedy_draft=True)
        torch.manual_seed(7)
        seq, _ = chain.verify_and_complete(NUM_SAMPLES, None)
        joint = chain.joint().view((VOCAB,) * NUM_SLOTS)
        marginal = joint.sum((0, 2, 3, 4, 5))
        hi, lo = int(marginal.argmax()), int(marginal.argmin())
        shifted = marginal.clone()
        shifted[hi] -= 0.01
        shifted[lo] += 0.01
        counts = torch.bincount(seq[:, 1], minlength=VOCAB)
        self.assertGreater(_chi2_pvalue(counts, marginal * NUM_SAMPLES), P_VALUE_FLOOR)
        self.assertLess(_chi2_pvalue(counts, shifted * NUM_SAMPLES), P_VALUE_FLOOR)

    def test_reference_filter_matches_the_verify_target(self):
        """The test's own top-k/top-p reference is the target the accept builds."""
        for top_k, top_p in _FILTERS[1:]:
            chain = _Chain(seed=3, top_k=top_k, top_p=top_p, greedy_draft=True)
            candidates, _ = chain.propose(64)
            got = sampled_verify_target_probs(
                target_logits=chain.target_rows(candidates),
                sampling_info=chain.sampling_info(64),
                draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
                verify_num_draft_tokens=NUM_SLOTS,
                bs=64,
            )
            want = torch.stack(
                [chain.target_probs[i][candidates[:, i]] for i in range(NUM_SLOTS)], 1
            )
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


class TestSampledChainParity(CustomTestCase):
    """The triton chain accept against its torch reference on a production-width vocabulary, so the blocked bonus
    search crosses vocab blocks."""

    def _inputs(self, *, bs, vocab, one_hot, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        candidates = torch.randint(0, vocab, (bs, NUM_SLOTS), device=DEVICE, generator=g)
        # Mass on each next candidate, so chains run past the first slot and stop on a mix of rows.
        next_hot = torch.zeros((bs, NUM_SLOTS, vocab), device=DEVICE)
        next_hot[:, :-1].scatter_(-1, candidates[:, 1:, None], 1.0)
        target = 0.4 * torch.softmax(
            torch.randn((bs, NUM_SLOTS, vocab), device=DEVICE, generator=g) * 3, -1
        ) + 0.6 * next_hot
        target[:, -1] /= target[:, -1].sum(-1, keepdim=True)
        draft = 0.3 * torch.softmax(
            torch.randn((bs, NUM_SLOTS - 1, vocab), device=DEVICE, generator=g) * 3, -1
        ) + 0.7 * next_hot[:, :-1]
        uniforms = torch.rand((bs, NUM_SLOTS), device=DEVICE, generator=g)
        verify_lens = torch.randint(1, NUM_SLOTS + 1, (bs,), device=DEVICE, generator=g)
        return dict(
            candidates=candidates,
            target_probs=target,
            draft_probs=None if one_hot else draft,
            verify_lens=verify_lens,
            uniforms=uniforms,
        )

    def test_triton_matches_torch(self):
        for one_hot in (True, False):
            with self.subTest(one_hot=one_hot):
                inputs = self._inputs(bs=256, vocab=129280, one_hot=one_hot, seed=5)
                want_len, want_bonus = sampled_chain_accept(**inputs)
                got_len, got_bonus = sampled_chain_accept_triton(**inputs)
                torch.testing.assert_close(got_len, want_len.to(torch.int32))
                self.assertGreater(int(want_len.max()), 1)
                self.assertGreater(int((want_len == 0).sum()), 0)
                # The triton bonus is the inverse-CDF draw of the reference's distribution, up to float32
                # summation order: its token holds the draw within a 1e-5 band of the float64 CDF.
                _, weights = sampled_chain_bonus_weights(**inputs)
                cdf = weights.double().cumsum(-1)
                total = cdf[:, -1]
                draw = inputs["uniforms"][:, -1].double() * total
                rows = torch.arange(weights.shape[0], device=DEVICE)
                hi = cdf[rows, got_bonus]
                lo = hi - weights[rows, got_bonus].double()
                eps = 1e-5 * total
                self.assertTrue(bool((weights[rows, got_bonus] > 0).all()))
                self.assertTrue(bool(((lo - eps <= draw) & (draw <= hi + eps)).all()))
                agree = float((got_bonus == want_bonus).float().mean())
                self.assertGreaterEqual(agree, 0.95)


class _BlockHead:
    """sample_block over fixed per-step logits: each step's sampler sees base_logits[:, step]."""

    def sample_block(self, base_logits, *, first_prev_tokens, hidden_states, sampler):
        del first_prev_tokens, hidden_states
        tokens = [sampler(base_logits[:, s], s) for s in range(base_logits.shape[1])]
        return torch.stack(tokens, 1), base_logits


class TestSampledGreedyDraft(CustomTestCase):
    def _draft(self, *, greedy_draft_sampled: bool):
        bs, steps = 256, NUM_SLOTS - 1
        base_logits = torch.randn((bs, steps, 64), device=DEVICE)
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.ones((bs, 1), device=DEVICE),
            top_ks=torch.full((bs,), 64, dtype=torch.int32, device=DEVICE),
        )
        block = sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=torch.zeros((bs,), dtype=torch.int64, device=DEVICE),
            draft_hidden=None,
            sampling_info=sampling_info,
            markov_head=_BlockHead(),
            device=DEVICE,
            tp_sync=SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0)),
            greedy_draft_sampled=greedy_draft_sampled,
        )
        return block, base_logits.argmax(-1)

    def test_flag_on_drafts_the_argmax(self):
        block, argmax = self._draft(greedy_draft_sampled=True)
        self.assertTrue(block.greedy_draft)
        torch.testing.assert_close(block.draft_tokens, argmax)

    def test_flag_off_samples(self):
        block, argmax = self._draft(greedy_draft_sampled=False)
        self.assertFalse(block.greedy_draft)
        self.assertFalse(bool((block.draft_tokens == argmax).all()))


if __name__ == "__main__":
    unittest.main()
