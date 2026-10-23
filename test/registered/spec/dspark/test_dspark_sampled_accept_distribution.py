"""Losslessness of DSpark's sampled accept on upstream's kernel (``chain_speculative_sampling_triton``).

A synthetic position-dependent Markov target and draft over a small vocabulary drive ``accept_draft_tokens``. Each
verify's output (accepted drafts, then the bonus) is completed to the full chain length by sampling the target, so a
lossless accept makes the completed sequence an exact draw from the target's joint distribution, which the test
enumerates. A chi-square test over the joint and one per position compares the counts to it; a self-check shows the
same counts reject a reference shifted by 1%.

Dead rows (past a cut's verify_len) carry what the eager sampled path leaves there: the real candidates and finite
garbage target logits. Two layouts are pinned beside it:

* NaN or Inf dead-row logits under block verification, which the dead-row guard in ``_accept_sampling_core`` keeps
  from biasing the live tokens (this case fails on a tree without the guard);
* zero-filled dead candidates (``DsparkVerifyEpilogue``, ``fill_value=0``), which bias a sampled accept. That test
  asserts the harness detects the bias, so a future fold of the sampled accept into the epilogue fails loudly.

Mutants, run once and not committed: acceptance + 0.05 and block h + 0.05 in ``reject_sampling.py`` must each fail.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import accept_draft_tokens
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
VOCAB = 4
NUM_SLOTS = 6  # speculative_num_draft_tokens: the anchor and five drafts.
NUM_SAMPLES = 1 << 20
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
    dof = observed.numel() - 1
    if dof < 1:
        # One possible outcome, and no draw outside it.
        return 1.0
    stat = float(((observed - expected) ** 2 / expected).sum())
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

    def __init__(
        self, *, seed: int, top_k=None, top_p=None, greedy_draft: bool, device=DEVICE
    ):
        self.device = device
        g = torch.Generator().manual_seed(seed)
        target = torch.randn((NUM_SLOTS, VOCAB, VOCAB), generator=g) * 1.5
        draft = target + torch.randn((NUM_SLOTS, VOCAB, VOCAB), generator=g)
        self.target_logits = target.to(device)
        self.draft_logits = draft.to(device)
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
            temperatures=torch.full((n, 1), TEMPERATURE, device=self.device),
            top_ks=torch.full(
                (n,), self.top_k or VOCAB, dtype=torch.int32, device=self.device
            ),
            top_ps=torch.full((n,), self.top_p or 1.0, device=self.device),
        )

    def propose(self, n: int):
        """[n, S] candidates and the [n, S - 1, V] draft distribution each draft was taken from."""
        candidates = torch.zeros((n, NUM_SLOTS), dtype=torch.int64, device=self.device)
        draft_probs = torch.empty((n, NUM_SLOTS - 1, VOCAB), device=self.device)
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
        self,
        candidates,
        draft_tokens,
        draft_probs,
        target_logits,
        verify_lens,
        *,
        block_verification=False,
        ragged=False,
    ):
        n = candidates.shape[0]
        draft_block = DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=(
                None
                if self.greedy_draft
                # SoftmaxTemp divides by the request temperature; this undoes it so the accept's q is draft_probs.
                else TEMPERATURE * draft_probs.clamp_min(1e-30).log()
            ),
            greedy_mask=torch.zeros((n,), dtype=torch.bool, device=self.device),
            temperatures=torch.full((n,), TEMPERATURE, device=self.device),
            greedy_draft=self.greedy_draft,
            top_probs=draft_probs.max(-1).values,
        )
        cut = {}
        if verify_lens is not None and ragged:
            # accept_draft_tokens reads only the layout's verify_lens; the rest is placeholder.
            dummy = torch.zeros((n,), dtype=torch.int32, device=self.device)
            cut["cutoff_layout"] = RaggedVerifyLayout(
                verify_lens=verify_lens,
                graph_num_tokens=n * NUM_SLOTS,
                extend_start_loc=dummy,
                qo_indptr_device=dummy,
            )
        else:
            cut["cutoff_verify_lens"] = verify_lens
        correct_len, bonus, _ = accept_draft_tokens(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=draft_block,
            sampling_info=self.sampling_info(n),
            draft_input=SimpleNamespace(max_top_k=None, uniform_top_k_value=None),
            gamma=NUM_SLOTS - 1,
            verify_num_draft_tokens=NUM_SLOTS,
            block_verification=block_verification,
            **cut,
        )
        return correct_len.to(torch.int64), bonus

    def verify_and_complete(
        self,
        n: int,
        verify_lens,
        *,
        block_verification=False,
        ragged=False,
        dead_logits="garbage",
        zero_fill=False,
    ):
        """[n, S] completed sequences and the accept's correct_len.

        Dead rows (past verify_lens) hold the real candidates, and target logits that are finite garbage
        ("garbage"), NaN ("nan") or +Inf ("inf"). ``zero_fill`` zeroes the dead candidates instead, the folded
        epilogue's layout; the draft block still carries the real drafts, as the epilogue's does."""
        real, draft_probs = self.propose(n)
        candidates = real
        target_logits = self.target_rows(real).view(n, NUM_SLOTS, VOCAB)
        if verify_lens is not None:
            slots = torch.arange(NUM_SLOTS, device=self.device)
            dead = slots[None, :] >= verify_lens[:, None]
            if dead_logits == "garbage":
                junk = torch.randn_like(target_logits) * 4
            else:
                junk = torch.full_like(
                    target_logits, math.nan if dead_logits == "nan" else math.inf
                )
            target_logits = torch.where(dead[..., None], junk, target_logits)
            if zero_fill:
                candidates = real.masked_fill(dead, 0)
        # flashinfer's top-p renorm scratch grows with the row count; chunk below it.
        chunks = [
            self.accept(
                candidates[s : s + _CHUNK],
                real[s : s + _CHUNK, 1:],
                draft_probs[s : s + _CHUNK],
                target_logits[s : s + _CHUNK].reshape(-1, VOCAB),
                None if verify_lens is None else verify_lens[s : s + _CHUNK],
                block_verification=block_verification,
                ragged=ragged,
            )
            for s in range(0, n, _CHUNK)
        ]
        correct_len = torch.cat([c[0] for c in chunks])
        bonus = torch.cat([c[1] for c in chunks])
        if verify_lens is not None:
            assert bool((correct_len <= verify_lens - 1).all())
        return self._complete(real, correct_len, bonus), correct_len

    def _complete(self, real, correct_len, bonus) -> torch.Tensor:
        """Accepted drafts, then the bonus, then target draws to the full chain length."""
        n = real.shape[0]
        slots = torch.arange(NUM_SLOTS, device=self.device)
        seq = torch.where(
            slots[None, :] < correct_len[:, None], real.roll(-1, dims=1), 0
        )
        seq.scatter_(1, correct_len[:, None], bonus[:, None])
        prev = torch.zeros((n,), dtype=torch.int64, device=self.device)
        for t in range(NUM_SLOTS):
            fill = t > correct_len
            sample = torch.multinomial(self.target_probs[t][prev], 1).squeeze(-1)
            seq[:, t] = torch.where(fill, sample, seq[:, t])
            prev = seq[:, t]
        return seq

    def direct_sample(self, n: int) -> torch.Tensor:
        """[n, S] exact draws from the target chain, no accept in the way (the harness's own reference)."""
        seq = torch.zeros((n, NUM_SLOTS), dtype=torch.int64, device=self.device)
        prev = torch.zeros((n,), dtype=torch.int64, device=self.device)
        for t in range(NUM_SLOTS):
            seq[:, t] = torch.multinomial(self.target_probs[t][prev], 1).squeeze(-1)
            prev = seq[:, t]
        return seq

    def joint(self) -> torch.Tensor:
        """[V ** S] exact target probability of each sequence, token 0 most significant."""
        joint = self.target_probs[0][0]
        for t in range(1, NUM_SLOTS):
            last = torch.arange(joint.numel(), device=self.device) % VOCAB
            joint = (joint[:, None] * self.target_probs[t][last]).reshape(-1)
        return joint


def _pvalues(seq: torch.Tensor, joint: torch.Tensor) -> dict:
    n = seq.shape[0]
    weights = VOCAB ** torch.arange(NUM_SLOTS - 1, -1, -1, device=seq.device)
    codes = (seq * weights).sum(-1)
    counts = torch.bincount(codes, minlength=joint.numel())
    out = {"joint": _chi2_pvalue(counts, joint * n)}
    by_pos = joint.view((VOCAB,) * NUM_SLOTS)
    for t in range(NUM_SLOTS):
        dims = tuple(d for d in range(NUM_SLOTS) if d != t)
        marginal = by_pos.sum(dims) if dims else by_pos
        out[f"pos{t}"] = _chi2_pvalue(
            torch.bincount(seq[:, t], minlength=VOCAB), marginal * n
        )
    return out


_CUTS = [None, 1, 2, 3, 4, 5, 6, "mixed", "ragged"]
_FILTERS = [(None, None), (2, None), (None, 0.75), (3, 0.8)]
_RAGGED_WEIGHTS = [0.35, 0.1, 0.1, 0.05, 0.05, 0.35]  # verify_len 1..6, mostly the extremes


def _verify_lens(cut, n: int, seed: int):
    if cut is None:
        return None
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if cut == "mixed":
        return torch.randint(1, NUM_SLOTS + 1, (n,), device=DEVICE, generator=g)
    if cut == "ragged":
        weights = torch.tensor(_RAGGED_WEIGHTS, device=DEVICE)
        return 1 + torch.multinomial(weights, n, replacement=True, generator=g)
    return torch.full((n,), cut, dtype=torch.int64, device=DEVICE)


class TestHarnessCpu(CustomTestCase):
    """The harness's own statistics and reference, on the CPU: no kernel involved."""

    def test_chi2_accepts_exact_counts_and_rejects_a_shift(self):
        expected = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
        n = 100_000
        self.assertGreater(_chi2_pvalue(expected * n, expected * n), 0.99)
        shifted = expected.clone()
        shifted[0] -= 0.01
        shifted[3] += 0.01
        self.assertLess(_chi2_pvalue(shifted * n, expected * n), P_VALUE_FLOOR)

    def test_chi2_impossible_cell_and_single_outcome(self):
        expected = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64) * 1000
        self.assertEqual(_chi2_pvalue(torch.tensor([499, 500, 1.0]), expected), 0.0)
        self.assertEqual(
            _chi2_pvalue(torch.tensor([1000.0, 0.0]), torch.tensor([1000.0, 0.0])), 1.0
        )

    def test_filter_matches_a_per_row_reference(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(32, VOCAB) * 2, -1)
        for top_k, top_p in _FILTERS[1:]:
            got = _filter(probs, top_k, top_p)
            for row, want_row in zip(probs, got):
                p = row.clone()
                if top_k is not None:
                    keep = set(p.argsort(descending=True)[:top_k].tolist())
                    p = torch.tensor(
                        [v if i in keep else 0.0 for i, v in enumerate(p.tolist())]
                    )
                    p = p / p.sum()
                if top_p is not None:
                    order = p.argsort(descending=True).tolist()
                    mass, keep = 0.0, set()
                    for i in order:
                        if mass < top_p:
                            keep.add(i)
                        mass += float(p[i])
                    p = torch.tensor(
                        [v if i in keep else 0.0 for i, v in enumerate(p.tolist())]
                    )
                    p = p / p.sum()
                torch.testing.assert_close(want_row, p, atol=1e-6, rtol=1e-5)

    def test_exact_draws_pass_and_a_shifted_reference_fails(self):
        """Power of the joint and per-position tests, with the target itself as the sampler."""
        for fi, (top_k, top_p) in enumerate(_FILTERS):
            chain = _Chain(
                seed=60 + fi,
                top_k=top_k,
                top_p=top_p,
                greedy_draft=False,
                device=torch.device("cpu"),
            )
            torch.manual_seed(60 + fi)
            seq = chain.direct_sample(1 << 18)
            joint = chain.joint()
            self.assertAlmostEqual(float(joint.sum()), 1.0, places=5)
            with self.subTest(top_k=top_k, top_p=top_p, check="exact"):
                for name, pvalue in _pvalues(seq, joint).items():
                    self.assertGreater(pvalue, P_VALUE_FLOOR, name)
            # Move 1% of position 1's mass from its likeliest to its least likely token.
            by_pos = joint.view((VOCAB,) * NUM_SLOTS)
            marginal = by_pos.sum((0, 2, 3, 4, 5))
            hi, lo = int(marginal.argmax()), int(marginal.argmin())
            shifted = marginal.clone()
            shifted[hi] -= 0.01
            shifted[lo] += 0.01
            counts = torch.bincount(seq[:, 1], minlength=VOCAB)
            with self.subTest(top_k=top_k, top_p=top_p, check="shifted"):
                self.assertLess(
                    _chi2_pvalue(counts, shifted * seq.shape[0]), P_VALUE_FLOOR
                )


@unittest.skipUnless(HAS_CUDA, "the accept kernel and its softmax need CUDA")
class TestSampledAcceptDistribution(CustomTestCase):
    def _check(self, *, greedy_draft: bool, block_verification: bool, seed0: int):
        for fi, (top_k, top_p) in enumerate(_FILTERS):
            chain = _Chain(
                seed=seed0 + fi, top_k=top_k, top_p=top_p, greedy_draft=greedy_draft
            )
            joint = chain.joint()
            self.assertAlmostEqual(float(joint.sum()), 1.0, places=5)
            for ci, cut in enumerate(_CUTS):
                torch.manual_seed(seed0 * 1000 + fi * 100 + ci)
                seq, _ = chain.verify_and_complete(
                    NUM_SAMPLES,
                    _verify_lens(cut, NUM_SAMPLES, seed0 + ci),
                    block_verification=block_verification,
                    ragged=cut == "ragged",
                )
                for name, pvalue in _pvalues(seq, joint).items():
                    with self.subTest(top_k=top_k, top_p=top_p, cut=cut, test=name):
                        self.assertGreater(pvalue, P_VALUE_FLOOR)

    def test_token_by_token_sampled_draft(self):
        self._check(greedy_draft=False, block_verification=False, seed0=11)

    def test_token_by_token_one_hot_draft(self):
        self._check(greedy_draft=True, block_verification=False, seed0=21)

    def test_block_sampled_draft(self):
        self._check(greedy_draft=False, block_verification=True, seed0=31)

    def test_block_one_hot_draft(self):
        self._check(greedy_draft=True, block_verification=True, seed0=41)

    def test_block_nan_or_inf_dead_rows(self):
        """Block verification reads every row. A dead row whose target probabilities are NaN or Inf must not change
        the live tokens: the dead-row guard in ``_accept_sampling_core`` swaps those rows for valid distributions.

        EXPECTED TO FAIL on a tree without that guard: NaN in row L makes the block rule's h_{L-1} NaN, so
        tau = L - 1 is never chosen and the capped output is biased."""
        for greedy_draft, seed0 in ((False, 71), (True, 81)):
            chain = _Chain(seed=seed0, greedy_draft=greedy_draft)
            joint = chain.joint()
            for di, dead_logits in enumerate(("nan", "inf")):
                for ci, cut in enumerate((2, 4, "mixed", "ragged")):
                    torch.manual_seed(seed0 * 1000 + di * 100 + ci)
                    seq, _ = chain.verify_and_complete(
                        NUM_SAMPLES,
                        _verify_lens(cut, NUM_SAMPLES, seed0 + ci),
                        block_verification=True,
                        ragged=cut == "ragged",
                        dead_logits=dead_logits,
                    )
                    for name, pvalue in _pvalues(seq, joint).items():
                        with self.subTest(
                            greedy_draft=greedy_draft,
                            dead_logits=dead_logits,
                            cut=cut,
                            test=name,
                        ):
                            self.assertGreater(pvalue, P_VALUE_FLOOR)

    def test_zero_filled_dead_candidates_are_detected_as_biased(self):
        """Sampled accepts must stay on real candidates. The folded epilogue (``DsparkVerifyEpilogue``,
        ``fill_value=0``) zero-fills the candidates past verify_len; a sampled accept fed that layout takes the
        zero token as the draft at the first dead slot, which biases the bonus. The harness must see it: some
        p-value falls under the floor for every kernel mode. If this ever passes, either the accept stopped
        reading dead candidates (then lift the constraint) or the harness lost its power."""
        for block_verification in (False, True):
            for greedy_draft in (False, True):
                chain = _Chain(
                    seed=91, top_k=None, top_p=None, greedy_draft=greedy_draft
                )
                joint = chain.joint()
                torch.manual_seed(91)
                seq, _ = chain.verify_and_complete(
                    NUM_SAMPLES,
                    _verify_lens("ragged", NUM_SAMPLES, 91),
                    block_verification=block_verification,
                    ragged=True,
                    zero_fill=True,
                )
                pvalues = _pvalues(seq, joint)
                with self.subTest(
                    block_verification=block_verification, greedy_draft=greedy_draft
                ):
                    self.assertLess(min(pvalues.values()), P_VALUE_FLOOR, pvalues)

    def test_block_accepts_at_least_token_by_token(self):
        """Same drafts, targets and cuts: block verification's mean accepted length is not below token-by-token's."""
        n = NUM_SAMPLES // 2
        for greedy_draft in (False, True):
            for fi, (top_k, top_p) in enumerate(_FILTERS):
                chain = _Chain(
                    seed=50 + fi, top_k=top_k, top_p=top_p, greedy_draft=greedy_draft
                )
                torch.manual_seed(50 + fi)
                real, draft_probs = chain.propose(n)
                target_logits = chain.target_rows(real)
                verify_lens = _verify_lens("mixed", n, 50 + fi)
                means = {}
                for block_verification in (False, True):
                    lens = torch.cat(
                        [
                            chain.accept(
                                real[s : s + _CHUNK],
                                real[s : s + _CHUNK, 1:],
                                draft_probs[s : s + _CHUNK],
                                target_logits[s * NUM_SLOTS : (s + _CHUNK) * NUM_SLOTS],
                                verify_lens[s : s + _CHUNK],
                                block_verification=block_verification,
                            )[0]
                            for s in range(0, n, _CHUNK)
                        ]
                    ).double()
                    means[block_verification] = (
                        float(lens.mean()),
                        float(lens.std()) / math.sqrt(n),
                    )
                (tbt, tbt_se), (block, block_se) = means[False], means[True]
                print(
                    f"greedy_draft={greedy_draft} top_k={top_k} top_p={top_p}: mean correct_len "
                    f"token-by-token {tbt:.4f}, block {block:.4f}"
                )
                with self.subTest(
                    greedy_draft=greedy_draft, top_k=top_k, top_p=top_p
                ):
                    self.assertGreaterEqual(
                        block, tbt - 4 * math.hypot(tbt_se, block_se)
                    )

    def test_sampled_draft_control_without_a_cut(self):
        """The harness against the kernel with a sampled draft and no cut (nothing the cut paths can break)."""
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
        from sglang.srt.speculative.dflash_utils import (
            build_speculative_verify_target_probs,
        )

        for top_k, top_p in _FILTERS[1:]:
            chain = _Chain(seed=3, top_k=top_k, top_p=top_p, greedy_draft=True)
            candidates, _ = chain.propose(64)
            got = build_speculative_verify_target_probs(
                next_token_logits=chain.target_rows(candidates),
                sampling_info=chain.sampling_info(64),
                draft_token_num=NUM_SLOTS,
                bs=64,
            ).view(64, NUM_SLOTS, VOCAB)
            want = torch.stack(
                [chain.target_probs[i][candidates[:, i]] for i in range(NUM_SLOTS)], 1
            )
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
