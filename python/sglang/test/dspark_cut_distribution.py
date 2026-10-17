"""A toy target and DSpark-shaped draft for checking that a verify cut keeps sampled speculative decoding exact.

The target is an order-1 Markov chain over a small vocabulary, so the exact law of the first ``length`` generated tokens
is a product of its rows. The draft proposes gamma tokens per verify from its own Markov chain (DSpark's Markov head:
draft k is drawn given draft k - 1), and the confidence of draft k is a table indexed by the token before it, as the
real head's Markov embedding is (dspark_planner.compute_confidence's prev_seq). ``generate`` decodes many independent
sequences verify by verify: draft, cut, verify, accept (``accept_fn``, the real AcceptSampling on the GPU or
``reference_accept`` on the CPU), append the accepted tokens and the bonus. The target rows past a verify's width are
replaced by random distributions, as a dead row's logits are not the model's. ``chi_square_pvalue`` then compares the
sequences' counts with the exact law.

Used by test_dspark_confidence_cut_sampled.py (CPU, reference accept) and test_dspark_confidence_cut_gpu.py (the real
accept kernels).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

# verify_lens_fn(confidence [n, gamma], drafts [n, gamma]) -> [n] int64 widths in [1, gamma + 1]. A real cut reads the
# confidence only; drafts are there so a mutant can read the drawn tokens.
VerifyLensFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# accept_fn(candidates [n, gamma + 1], target_probs [n, gamma + 1, V], draft_probs [n, gamma, V], verify_lens [n])
#   -> (correct_len [n], bonus [n])
AcceptFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class MarkovToy:
    def __init__(self, *, vocab: int, gamma: int, temperature: float, seed: int = 0, start_token: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.vocab = vocab
        self.gamma = gamma
        self.temperature = temperature
        self.start_token = start_token
        self.target_logits = 1.5 * torch.randn((vocab, vocab), generator=generator)
        # Close to the target, so most drafts are accepted and verifies reach several widths, but not equal to it.
        self.draft_logits = self.target_logits + 0.8 * torch.randn((vocab, vocab), generator=generator)
        # confidence[k - 1, t]: draft k's confidence when the token before it is t.
        self.confidence = 0.15 + 0.8 * torch.rand((gamma, vocab), generator=generator)

    def target_probs(self) -> torch.Tensor:
        return torch.softmax(self.target_logits / self.temperature, dim=-1)

    def draft_probs(self) -> torch.Tensor:
        return torch.softmax(self.draft_logits / self.temperature, dim=-1)

    def exact_sequence_probs(self, length: int) -> torch.Tensor:
        """[vocab ** length] float64: the target's law of the first ``length`` tokens after the start token, indexed
        by sequence_index."""
        p = self.target_probs().to(torch.float64)
        probs = p[self.start_token]
        last = torch.arange(self.vocab)
        for _ in range(length - 1):
            probs = (probs[:, None] * p[last]).reshape(-1)
            last = torch.arange(self.vocab).repeat(probs.numel() // self.vocab)
        return probs


def sequence_index(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    index = torch.zeros(tokens.shape[0], dtype=torch.int64, device=tokens.device)
    for j in range(tokens.shape[1]):
        index = index * vocab + tokens[:, j]
    return index


def generate(
    toy: MarkovToy,
    *,
    n: int,
    length: int,
    verify_lens_fn: VerifyLensFn,
    accept_fn: AcceptFn,
    generator: torch.Generator,
    device="cpu",
    widths_out: Optional[list] = None,
) -> torch.Tensor:
    """[n, length] int64: n independent sequences decoded verify by verify. ``widths_out`` collects each verify's
    widths, so a test can check that the cut actually cut."""
    gamma, vocab = toy.gamma, toy.vocab
    p_rows = toy.target_probs().to(device)
    q_rows = toy.draft_probs().to(device)
    confidence_table = toy.confidence.to(device)
    out = torch.full((n, length), -1, dtype=torch.int64, device=device)
    filled = torch.zeros(n, dtype=torch.int64, device=device)
    prev = torch.full((n,), toy.start_token, dtype=torch.int64, device=device)
    positions = torch.arange(gamma, device=device)
    while True:
        rows = torch.nonzero(filled < length).view(-1)
        m = rows.numel()
        if m == 0:
            return out
        anchor = prev[rows]
        drafts = torch.empty((m, gamma), dtype=torch.int64, device=device)
        draft_probs = torch.empty((m, gamma, vocab), dtype=torch.float32, device=device)
        token = anchor
        for k in range(gamma):
            q = q_rows[token]
            token = torch.multinomial(q, 1, generator=generator).view(-1)
            draft_probs[:, k] = q
            drafts[:, k] = token
        prev_seq = torch.cat([anchor[:, None], drafts[:, :-1]], dim=1)
        confidence = confidence_table[positions[None, :], prev_seq]
        verify_lens = verify_lens_fn(confidence, drafts).to(device=device, dtype=torch.int64)
        if widths_out is not None:
            widths_out.append(verify_lens.cpu())
        candidates = torch.cat([anchor[:, None], drafts], dim=1)
        target_probs = p_rows[candidates]
        dead = torch.arange(gamma + 1, device=device)[None, :] >= verify_lens[:, None]
        garbage = torch.softmax(
            3.0 * torch.randn((m, gamma + 1, vocab), generator=generator, device=device), dim=-1
        )
        target_probs = torch.where(dead[..., None], garbage, target_probs)
        correct_len, bonus = accept_fn(
            candidates=candidates, target_probs=target_probs, draft_probs=draft_probs, verify_lens=verify_lens
        )
        correct_len = correct_len.to(torch.int64)
        bonus = bonus.to(torch.int64)
        emitted = torch.where(
            torch.arange(gamma + 1, device=device)[None, :] < correct_len[:, None],
            torch.cat([drafts, drafts[:, :1]], dim=1),
            bonus[:, None],
        )
        base = filled[rows]
        for j in range(gamma + 1):
            pos = base + j
            keep = (j <= correct_len) & (pos < length)
            out[rows[keep], pos[keep]] = emitted[keep, j]
        filled[rows] = base + correct_len + 1
        prev[rows] = bonus


def reference_accept(
    *,
    candidates: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    verify_lens: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A torch model of AcceptSampling's chain rejection sampling over every row (the speculative sampling kernel),
    then CapCorrectLen and the bonus gather: the bonus is predicts[accept_index[correct_len]], the accepted draft
    there if the chain went past the cap, else the final residual (or all-accepted target) draw."""
    m, rows, vocab = target_probs.shape
    gamma = rows - 1
    device = target_probs.device
    ar = torch.arange(m, device=device)
    coins = torch.rand((m, gamma), generator=generator, device=device)
    num_accept = torch.zeros(m, dtype=torch.int64, device=device)
    going = torch.ones(m, dtype=torch.bool, device=device)
    for k in range(1, gamma + 1):
        draft = candidates[:, k]
        p = target_probs[ar, k - 1, draft]
        q = draft_probs[ar, k - 1, draft]
        going = going & (q > 0) & (coins[:, k - 1] * q < p)
        num_accept += going.to(torch.int64)
    all_accept = num_accept == gamma
    p_row = target_probs[ar, num_accept]
    q_row = draft_probs[ar, num_accept.clamp(max=gamma - 1)]
    final_dist = torch.where(all_accept[:, None], p_row, (p_row - q_row).clamp_min(0.0))
    final = torch.multinomial(final_dist + 1e-30, 1, generator=generator).view(-1)
    correct_len = torch.minimum(num_accept, verify_lens - 1)
    past_cap = num_accept > correct_len
    bonus = torch.where(past_cap, candidates[ar, (correct_len + 1).clamp(max=gamma)], final)
    return correct_len, bonus


def chi_square_pvalue(*, counts: torch.Tensor, probs: torch.Tensor) -> float:
    """Pearson's goodness-of-fit p-value of ``counts`` against ``probs``; cells with an expected count under 5 are
    pooled into one."""
    counts = counts.to(torch.float64).cpu()
    probs = probs.to(torch.float64).cpu()
    n = counts.sum()
    expected = n * probs
    small = expected < 5.0
    obs = torch.cat([counts[~small], counts[small].sum().view(1)])
    exp = torch.cat([expected[~small], expected[small].sum().view(1)])
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    stat = ((obs - exp) ** 2 / exp).sum()
    df = obs.numel() - 1
    return float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64), stat / 2.0))


def sequence_pvalue(toy: MarkovToy, sequences: torch.Tensor) -> float:
    length = sequences.shape[1]
    probs = toy.exact_sequence_probs(length)
    counts = torch.bincount(sequence_index(sequences.cpu(), toy.vocab), minlength=probs.numel())
    return chi_square_pvalue(counts=counts, probs=probs)


class _WidthTimedEvent:
    """A CUDA-event stand-in whose end event times the verify by its forced width, from a fixed affine T(w)."""

    def __init__(self, ms_of_current_width: Callable[[], float]):
        self._ms_of_current_width = ms_of_current_width
        self.ms = 0.0

    def record(self):
        self.ms = self._ms_of_current_width()

    def query(self):
        return True

    def elapsed_time(self, end):
        return end.ms


def calibrated_cut(*, gamma: int, base_ms: float, per_row_ms: float, device="cpu", **cut_kwargs):
    """A ConfidenceCut whose T(w) calibration ran over forced widths timed at exactly base_ms + per_row_ms * (w - 1)."""
    from sglang.srt.speculative.dspark_components.dspark_confidence_cut import (
        CALIBRATION_VERIFIES,
        COPY_ENGINE_OFF,
        CalibrationGate,
        ConfidenceCut,
    )

    holder = {}

    def ms_of_current_width() -> float:
        width = holder["cut"]._timed_width
        return 0.0 if width is None else base_ms + per_row_ms * (width - 1)

    cut = ConfidenceCut(
        gamma=gamma, device=device, new_event=lambda: _WidthTimedEvent(ms_of_current_width), **cut_kwargs
    )
    holder["cut"] = cut
    gate = CalibrationGate(engine_ready=True, copy_engine=COPY_ENGINE_OFF, serving_request=True)
    confidence = torch.full((1, gamma), 0.5, device=device)
    for _ in range(CALIBRATION_VERIFIES):
        assert cut.admit_calibration(gate)
        cut.verify_lens(confidence)
        cut.begin_verify()
        cut.end_verify()
    cut.verify_lens(confidence)
    assert cut.calibrated, "the toy calibration did not freeze T(w)"
    return cut


def token_at_slot_mutant(verify_lens_fn: VerifyLensFn, *, token: int) -> VerifyLensFn:
    """A cut whose admission of slot k reads draft k itself: it never verifies a draft position (k >= 2) whose drawn
    token is ``token``, deferring it to the next verify. Rejection sampling on the drafts it keeps is no longer exact."""

    def mutant(confidence: torch.Tensor, drafts: torch.Tensor) -> torch.Tensor:
        lens = verify_lens_fn(confidence, drafts)
        gamma = drafts.shape[1]
        slots = torch.arange(1, gamma + 1, device=drafts.device)[None, :]
        hit = (drafts == token) & (slots >= 2)
        first_hit = torch.where(hit, slots, torch.full_like(slots, gamma + 1)).min(dim=1).values
        return torch.minimum(lens, first_hit - 1)

    return mutant
