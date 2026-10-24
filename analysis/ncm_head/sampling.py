"""Reference sampling and causal cutoff, independent of SGLang serving kernels."""

import math
from dataclasses import dataclass
from enum import Enum

import torch


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    greedy: bool = False

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive; use greedy flag")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid sampling settings")


def probabilities(logits, settings):
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite proposal logits")
    if settings.greedy:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(
            logits.dtype
        )
    scaled = logits.float() / settings.temperature
    support = torch.ones_like(scaled, dtype=torch.bool)
    if settings.top_k:
        # Exactly k, stable token-index tie breaking. Native tie semantics need parity verification.
        order = torch.argsort(scaled.detach(), dim=-1, descending=True, stable=True)
        support = torch.zeros_like(support).scatter_(
            -1, order[..., : min(settings.top_k, logits.shape[-1])], True
        )
    masked = scaled.masked_fill(~support, -torch.inf)
    if settings.top_p < 1:
        order = torch.argsort(masked.detach(), dim=-1, descending=True, stable=True)
        p = masked.detach().softmax(-1).gather(-1, order)
        keep = (p.cumsum(-1) - p) < settings.top_p
        sorted_support = torch.zeros_like(support).scatter_(-1, order, keep)
        support = support & sorted_support
    return scaled.masked_fill(~support, -torch.inf).softmax(-1)


def validate_distribution(p):
    if (
        p.ndim != 1
        or not torch.isfinite(p).all()
        or (p < 0).any()
        or not torch.isclose(p.sum(), p.new_tensor(1.0), atol=2e-6, rtol=2e-6)
    ):
        raise ValueError("invalid normalized distribution")


def overlap(p, q):
    return torch.minimum(p, q).sum(-1)


class Mode(Enum):
    ACTIVE = "active"
    FULL_REMAINDER = "full_remainder"
    STOPPED = "stopped"


class Admission:
    def __init__(self, threshold, compatible=True):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid threshold")
        self.threshold = threshold
        self.mode = Mode.ACTIVE if compatible else Mode.FULL_REMAINDER

    def visit(self, predict):
        # Callable is never evaluated after stop or fallback: no suffix invalidity can reopen.
        if self.mode == Mode.STOPPED:
            return False
        if self.mode == Mode.FULL_REMAINDER:
            return True
        score = float(predict())
        if not math.isfinite(score) or not 0 <= score <= 1:
            self.mode = Mode.FULL_REMAINDER
            return True
        if score < self.threshold:
            self.mode = Mode.STOPPED
            return False
        return True


def propose(
    head,
    plan,
    anchor,
    settings,
    generator,
    eos_id,
    *,
    threshold=0.0,
    confidence=None,
    compatible=True,
):
    policy = Admission(threshold, compatible and confidence is not None)
    state, previous = head.initial_state(plan), anchor
    tokens, qs = [], []
    for i in range(head.config.gamma):
        logits, v = head.step_logits(plan, state, previous, i)
        q = probabilities(logits, settings)
        validate_distribution(q)
        if not policy.visit(
            lambda state=state, v=v, i=i: confidence(plan, state, v, i)
        ):
            break
        token = int(torch.multinomial(q, 1, generator=generator))
        tokens.append(token)
        qs.append(q)
        if token == eos_id:
            break
        if i + 1 < head.config.gamma:
            state = head.observe(plan, state, token, i)
        previous = token
    return tokens, qs, policy.mode


def expected_accepted(tokens, target_rows, proposal_rows, eos_id):
    if not (len(tokens) == len(target_rows) == len(proposal_rows)):
        raise ValueError("candidate-prefix row alignment")
    survival, total = 1.0, 0.0
    for token, p, q in zip(tokens, target_rows, proposal_rows):
        validate_distribution(p)
        validate_distribution(q)
        if float(q[token]) <= 0:
            raise ValueError("sampled candidate has zero proposal mass")
        survival *= min(1.0, float(p[token] / q[token]))
        total += survival
        if token == eos_id:
            break
    return total


def verify(tokens, qs, target, generator, eos_id, max_output):
    """Target callback receives the actually accepted local prefix; fresh target boundary.

    Returns committed output including EOS/correction/bonus. Reference implementation only.
    """
    if len(tokens) != len(qs) or max_output < 1:
        raise ValueError("invalid proposal/output limit")
    output = []
    for token, q in zip(tokens, qs):
        p = target(output)
        validate_distribution(p)
        validate_distribution(q)
        if float(q[token]) <= 0:
            raise ValueError("impossible sampled token")
        if float(torch.rand((), generator=generator, device=q.device)) < min(
            1.0, float(p[token] / q[token])
        ):
            output.append(token)
            if token == eos_id or len(output) == max_output:
                return output
        else:
            residual = (p - q).clamp_min(0)
            if not float(residual.sum()) > 0:
                raise ValueError("rejection with zero residual mass")
            output.append(
                int(
                    torch.multinomial(residual / residual.sum(), 1, generator=generator)
                )
            )
            return output
    if len(output) < max_output:
        p = target(output)
        validate_distribution(p)
        output.append(int(torch.multinomial(p, 1, generator=generator)))
    return output
