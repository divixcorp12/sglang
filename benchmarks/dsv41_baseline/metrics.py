"""Pure computations for the DSV4.1 baseline arms: no network, no GPU, no imports of sglang.

`decode_tokens_per_second` mirrors the formula `run_capture_sessions.py` already applies
per record (rule 1 of MOE_EXPERT_TRANSFER.md:674-702): decode-only, from the server's
`usage.completion_tokens`, never assumed from the text. It is re-implemented here, not
imported, so a unit test can pin the formula independently of the driver.
"""

from __future__ import annotations

import math
import statistics


def decode_tokens_per_second(
    *, completion_tokens: int, ttft: float, total: float
) -> float | None:
    """(completion_tokens - 1) / decode_seconds, decode_seconds = total - ttft.

    None when there are fewer than 2 completion tokens (no decode interval) or the
    server never reported usage, matching `run_capture_sessions.py`'s own guard.
    """
    if not completion_tokens or completion_tokens <= 1:
        return None
    decode_seconds = total - ttft
    if decode_seconds <= 0:
        return None
    return (completion_tokens - 1) / decode_seconds


def median_tok_s(per_session_tok_s: list[float]) -> float:
    """The median across sessions. Descriptive only: never the arm-comparison statistic.

    Per-session tok/s varies 62%+ across this corpus's different prompts (2.18-3.53 in
    the recorded arms) because prompt content, not the change under test, dominates it.
    An arm's median (or mean) is useful context but must never be diffed against
    another arm's median/mean as a finding — see `paired.py`.
    """
    if not per_session_tok_s:
        raise ValueError("no per-session tok/s values to aggregate")
    return statistics.median(per_session_tok_s)


def one_sided_sign_test(*, wins: int, n: int) -> float:
    """P(X >= wins) under X ~ Binomial(n, 0.5): the one-sided sign test p-value.

    Tests whether `wins` (arm B beating arm A on a paired session) is more than chance.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= wins <= n):
        raise ValueError(f"wins={wins} out of range for n={n}")
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / (2**n)
