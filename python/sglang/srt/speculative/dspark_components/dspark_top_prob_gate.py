"""DSpark's top-probability verify gate (``SGLANG_DSPARK_TOP_PROB_GATE``).

With p_i the draft's softmax top-1 probability at draft position i (over its Markov-corrected step logits), a verify
keeps draft i while p_1 * ... * p_i is at least the threshold and stops at the first draft where it falls below; the
anchor is always verified. The verify still runs its gamma + 1 rows in the one captured graph, but the rows past the
cut route no experts (layers/moe/verify_dead_rows.py) and the accept is capped at the cut (the verify epilogue's
static cutoff, or the eager accept's), so a cut draft is never accepted.

The probabilities come from the draft proposal itself: the folded greedy proposal's sharded argmax also reduces each
step's softmax denominator (sharded_greedy_step's top_prob_out), and the block sampler paths record them per step.

Greedy batches of one request only: the live rows are a prefix of the verify's flat rows only at bs 1, and the gate's
accept cap is the greedy accept's. Any other batch verifies every draft, as with the gate off.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.environ import envs


class TopProbGate(msgspec.Struct, frozen=True):
    threshold: float

    @classmethod
    def from_envs(cls) -> Optional[TopProbGate]:
        """The configured gate, or None when it is off (threshold 0)."""
        threshold = float(envs.SGLANG_DSPARK_TOP_PROB_GATE.get())
        if threshold == 0.0:
            return None
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"SGLANG_DSPARK_TOP_PROB_GATE must be in (0, 1] (0 turns it off), got {threshold}."
            )
        return cls(threshold=threshold)

    def applies(self, *, bs: int, sampling_info) -> bool:
        return bs == 1 and (sampling_info is None or sampling_info.is_all_greedy)

    def verify_lens(self, top_probs: torch.Tensor) -> torch.Tensor:
        """[bs] int64 rows to verify in [1, gamma + 1], from the [bs, gamma] top-1 probabilities."""
        return gate_verify_lens(top_probs=top_probs, threshold=self.threshold)


def gate_verify_lens(*, top_probs: torch.Tensor, threshold: float) -> torch.Tensor:
    # The prefix of kept drafts, not their count: a NaN probability, or a product that rounds back up to the
    # threshold after falling below it, must still end the prefix.
    above = torch.cumprod(top_probs.float(), dim=1) >= threshold
    num_kept = torch.cumprod(above.to(torch.int64), dim=1).sum(dim=1)
    return num_kept + 1


def step_top_prob(step_logits: torch.Tensor) -> torch.Tensor:
    """[rows] softmax top-1 probability exp(max - logsumexp) of [rows, vocab] logits."""
    logits = step_logits.float()
    return torch.exp(logits.amax(dim=-1) - torch.logsumexp(logits, dim=-1))


def recording_top_prob(sampler, *, top_prob_out: Optional[torch.Tensor]):
    """``sampler``, also writing each step's top-1 probability into column step_idx of ``top_prob_out``."""
    if top_prob_out is None:
        return sampler

    def recording(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
        top_prob_out[:, step_idx].copy_(step_top_prob(step_logits))
        return sampler(step_logits, step_idx)

    return recording
