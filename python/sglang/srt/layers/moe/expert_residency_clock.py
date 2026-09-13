"""Residency boundary accounting for dynamic expert hot caches."""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from operator import index
from typing import Any

logger = logging.getLogger(__name__)

_MAX_OUTSTANDING_VERIFIES = 16


class ForwardKind(Enum):
    """How a forward observed by the hot cache advances the residency clock."""

    PREFILL = "prefill"
    DECODE = "decode"
    IDLE = "idle"
    VERIFY = "verify"
    DRAFT = "draft"


def classify_forward(forward_batch: Any) -> tuple[ForwardKind, int]:
    """Return a forward's kind and the tokens it adds to the residency clock.

    Draft-worker forwards add nothing: any forward carrying a draft spec input
    (draft decode, draft extend after prefill, draft idle) and DRAFT_EXTEND_V2.
    Target verify and idle forwards carry a verify input. A TARGET_VERIFY adds
    its drafted tokens per request; the speculative worker later commits the
    accepted count.
    """
    mode = forward_batch.forward_mode
    spec_info = getattr(forward_batch, "spec_info", None)
    if (spec_info is not None and spec_info.is_draft_input()) or (
        mode.is_draft_extend_v2()
    ):
        return ForwardKind.DRAFT, 0
    batch_size = getattr(forward_batch, "batch_size", 1)
    if mode.is_target_verify():
        return ForwardKind.VERIFY, batch_size * getattr(spec_info, "draft_token_num", 1)
    if mode.is_extend_without_speculative():
        return ForwardKind.PREFILL, forward_batch.extend_num_tokens or 0
    if mode.is_idle():
        return ForwardKind.IDLE, batch_size
    return ForwardKind.DECODE, batch_size


class ResidencyBoundaryClock:
    """Decide residency boundaries from target forwards and committed tokens.

    Draft forwards never touch the clock. A prefill of at least
    ``update_prefill_tokens`` tokens is a boundary, and so is every
    ``update_decode_forwards``-th decode or verify forward since the previous
    boundary. A verify counts its drafted tokens provisionally, and ``commit``
    adds the difference between the oldest outstanding verify's accepted and
    drafted tokens to the current window. A verify that reaches a boundary
    carries its own provisional tokens into the next window, where its commit
    usually lands. Under the overlap scheduler a later verify can close that
    window first, so the correction then lands in a newer window; the window
    count is clamped at zero, so decay is approximate for such windows but a
    boundary never advances by negative tokens. ``forwards`` counts target
    forwards and drives minimum residence.
    """

    def __init__(
        self,
        update_prefill_tokens: int,
        update_decode_forwards: int,
        *,
        enabled: bool = True,
    ) -> None:
        self.update_prefill_tokens = index(update_prefill_tokens)
        self.update_decode_forwards = index(update_decode_forwards)
        self.enabled = enabled
        self.forwards = 0
        self.tokens_since_boundary = 0
        self.decode_forwards_since_boundary = 0
        self._provisional: deque[int] = deque(maxlen=_MAX_OUTSTANDING_VERIFIES)
        self._warned_dropped_provisional = False

    def observe(self, kind: ForwardKind, tokens: int) -> int | None:
        """Count one forward; return the tokens a boundary advances by, or None."""
        if kind is ForwardKind.DRAFT:
            return None
        self.forwards += 1
        carried = 0
        if kind is ForwardKind.PREFILL:
            self.tokens_since_boundary += tokens
            qualifying = tokens >= self.update_prefill_tokens
        else:
            if kind is ForwardKind.VERIFY:
                self._track_provisional(tokens)
                carried = tokens
            else:
                self.tokens_since_boundary += tokens
            if kind is not ForwardKind.IDLE:
                self.decode_forwards_since_boundary += 1
            qualifying = (
                self.update_decode_forwards > 0
                and self.decode_forwards_since_boundary >= self.update_decode_forwards
            )
        if not (self.enabled and qualifying):
            self.tokens_since_boundary += carried
            return None
        boundary_tokens = self.tokens_since_boundary
        self.tokens_since_boundary = carried
        self.decode_forwards_since_boundary = 0
        return boundary_tokens

    def commit(self, accepted_tokens: int) -> None:
        """Correct the current window by the oldest outstanding verify's accepted minus drafted tokens."""
        if self._provisional:
            self.tokens_since_boundary = max(
                0,
                self.tokens_since_boundary
                + index(accepted_tokens)
                - self._provisional.popleft(),
            )

    def _track_provisional(self, tokens: int) -> None:
        if (
            len(self._provisional) == self._provisional.maxlen
            and not self._warned_dropped_provisional
        ):
            logger.warning(
                "Expert residency clock dropped an uncommitted verify token count; "
                "speculative commits are not reaching the hot cache manager"
            )
            self._warned_dropped_provisional = True
        self._provisional.append(tokens)
