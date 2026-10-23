"""Online refit of the confidence cut's calibration from the server's own accepts (SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL).

The offline STS files are fitted once, on one prompt and one sampling setting. This refits, per draft position k, a
slope f_k on the cut's own confidence logits, c'_k = sigmoid(f_k * logit(c_k)) with c_k the confidence the cut would
read without it (the head's, re-tempered for a sampled verify), from the verifies the server has run: an online
logistic fit with one slope per position, the STS family relative to the offline file (f = 1 is the offline fit).
Greedy and sampled verifies keep separate buffers and slopes, since their acceptance differs (argmax match against
min(1, p/q)); any temperature and any drift across prompts is what the buffer sees.

Labels. A verify of w rows ran the accept's step for positions 1..w on live rows, so whether drafts 1..k were all
accepted is known for every k <= w: the uncapped accept count m = correct_len + cap_trim_lens says so (positions past
w read dead rows and are not used). Position k's sample is a verify with w >= k whose drafts 1..k-1 were accepted
(m >= k - 1), labelled m >= k: the conditional acceptance the cut's c_k stands for. Selecting on w >= k is selecting on
confidences, not on outcomes, so it does not bias the fit for a given confidence.

Off the hot path. Each cut verify copies [logit(c_1..c_gamma), w, m] (gamma + 2 floats) device-to-host without
blocking, into a pinned ring slot with an event; later verifies take the slots whose event has completed, in order,
and drop a sample when its slot is still busy. Every ``interval`` new samples a refit runs on a worker thread over a
snapshot of the bounded buffer; a later verify picks up the slopes when the thread is done and copies them to the
device without blocking. No device sync and no host wait is added.

Guard. The cut reads max(c_k, c'_k) per position. The rule's width is non-decreasing in every confidence (raising c_j
raises E(w) only for w > j), so the refit can widen a verify but never cut harder than the offline STS. The max is a
per-position function of the same inputs, and the slopes come from earlier verifies only, so the stopping-form
argument (dspark_confidence_cut.py) still holds.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from collections import deque
from typing import Callable, Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

# The newest verifies each mode keeps for a refit (~5 minutes of decode at ~13 verifies/s).
REFIT_BUFFER_VERIFIES = 4096
# Fewer samples than this at a position keep its slope at 1 (the offline fit).
REFIT_MIN_SAMPLES = 64
# A slope is kept in [1 / SLOPE_LIMIT, SLOPE_LIMIT].
SLOPE_LIMIT = 10.0
NEWTON_STEPS = 25
# Device-to-host ring slots in flight.
RING_SLOTS = 64

MODES = ("greedy", "sampled")


class SlopeFit(msgspec.Struct, frozen=True):
    slopes: list[float]
    samples: list[int]
    # Mean log loss per sample at slope 1 and at the fitted slope (NaN where a position kept slope 1).
    nll_before: list[float]
    nll_after: list[float]


def _mean_nll(x: torch.Tensor, y: torch.Tensor, slope: float) -> float:
    logits = x * slope
    return float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y))


def fit_slopes(
    *, logits: torch.Tensor, widths: torch.Tensor, taken: torch.Tensor, min_samples: int = REFIT_MIN_SAMPLES
) -> SlopeFit:
    """Per position k = 1..gamma, the slope f maximising the likelihood of "draft k accepted" given drafts 1..k-1 were,
    under c = sigmoid(f * logit), over the verifies that ran position k (w >= k). ``logits`` [n, gamma] are the cut's
    confidence logits, ``widths`` [n] the verify widths, ``taken`` [n] the uncapped accept counts."""
    logits = logits.to(torch.float64)
    gamma = logits.shape[1]
    slopes, samples, nll_before, nll_after = [], [], [], []
    for k in range(1, gamma + 1):
        x = logits[:, k - 1]
        sel = (widths >= k) & (taken >= k - 1) & torch.isfinite(x)
        x = x[sel]
        y = (taken[sel] >= k).to(torch.float64)
        n = int(x.numel())
        samples.append(n)
        if n < min_samples or float(x.abs().max()) == 0.0:
            slopes.append(1.0)
            nll_before.append(float("nan"))
            nll_after.append(float("nan"))
            continue
        # Newton on the one-parameter log loss: convex in f.
        f = 1.0
        for _ in range(NEWTON_STEPS):
            p = torch.sigmoid(x * f)
            grad = float(((p - y) * x).sum())
            hess = float((p * (1 - p) * x * x).sum())
            if hess <= 1e-12:
                break
            step = grad / hess
            f = min(max(f - step, 1.0 / SLOPE_LIMIT), SLOPE_LIMIT)
            if abs(step) < 1e-6:
                break
        slopes.append(f)
        nll_before.append(_mean_nll(x, y, 1.0))
        nll_after.append(_mean_nll(x, y, f))
    return SlopeFit(slopes=slopes, samples=samples, nll_before=nll_before, nll_after=nll_after)


class _ModeState:
    def __init__(self):
        self.buffer: deque = deque(maxlen=REFIT_BUFFER_VERIFIES)
        self.since_refit = 0
        self.slopes: Optional[torch.Tensor] = None
        self.future: Optional[concurrent.futures.Future] = None
        self.submitted_at = 0.0
        self.num_snapshot = 0
        self.refit_ct = 0
        # Keeps each pinned source alive until its device copy has run.
        self.staging: Optional[torch.Tensor] = None


class OnlineRefit:
    def __init__(
        self,
        *,
        gamma: int,
        device,
        interval: int,
        new_event: Optional[Callable[[], object]] = None,
        executor=None,
    ):
        self.gamma = gamma
        self.device = torch.device(device)
        self.interval = int(interval)
        self._new_event = new_event or (lambda: torch.cuda.Event())
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dspark-cut-refit"
        )
        pin = self.device.type == "cuda"
        self._ring = torch.empty((RING_SLOTS, gamma + 2), dtype=torch.float32, pin_memory=pin)
        self._slot_events: list[Optional[object]] = [None] * RING_SLOTS
        # (slot, mode) in issue order
        self._pending: deque = deque()
        self._next_slot = 0
        self.dropped_ct = 0
        self._modes = {mode: _ModeState() for mode in MODES}

    def apply(self, confidence: torch.Tensor, *, sampled: bool) -> torch.Tensor:
        """The cut's [bs, gamma] confidences with the refit's guard: max(offline, refit) per position."""
        slopes = self._modes[MODES[sampled]].slopes
        if slopes is None:
            return confidence
        offline = confidence.float()
        refit = torch.sigmoid(torch.logit(offline) * slopes)
        # A NaN offline confidence stays NaN (torch.maximum propagates it), which the cut reads as 0.
        return torch.maximum(offline, refit)

    def observe(
        self,
        *,
        confidence: torch.Tensor,
        verify_lens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        sampled: bool,
    ) -> None:
        """One bs-1 cut verify's outcome. ``confidence`` is what the cut read before the refit's guard."""
        self._drain()
        self._poll()
        slot = self._next_slot
        event = self._slot_events[slot]
        if event is not None:
            # Still in flight (or not drained yet): never wait, drop this sample.
            self.dropped_ct += 1
            return
        record = torch.cat(
            [
                torch.logit(confidence.float().reshape(-1)[: self.gamma]),
                verify_lens.reshape(-1)[:1].to(torch.float32),
                (correct_len.reshape(-1)[:1] + cap_trim_lens.reshape(-1)[:1]).to(torch.float32),
            ]
        )
        self._ring[slot].copy_(record, non_blocking=True)
        event = self._new_event()
        event.record()
        self._slot_events[slot] = event
        self._pending.append((slot, MODES[sampled]))
        self._next_slot = (slot + 1) % RING_SLOTS

    def _drain(self) -> None:
        while self._pending:
            slot, mode = self._pending[0]
            if not self._slot_events[slot].query():
                return
            self._pending.popleft()
            row = self._ring[slot].tolist()
            self._slot_events[slot] = None
            state = self._modes[mode]
            state.buffer.append((row[: self.gamma], int(row[self.gamma]), int(row[self.gamma + 1])))
            state.since_refit += 1
            if state.since_refit >= self.interval and state.future is None:
                self._submit(mode, state)

    def _submit(self, mode: str, state: _ModeState) -> None:
        state.since_refit = 0
        snapshot = list(state.buffer)
        logits = torch.tensor([r[0] for r in snapshot], dtype=torch.float64)
        widths = torch.tensor([r[1] for r in snapshot], dtype=torch.int64)
        taken = torch.tensor([r[2] for r in snapshot], dtype=torch.int64)
        state.submitted_at = time.monotonic()
        state.future = self._executor.submit(fit_slopes, logits=logits, widths=widths, taken=taken)
        state.num_snapshot = len(snapshot)

    def _poll(self) -> None:
        for mode, state in self._modes.items():
            if state.future is None or not state.future.done():
                continue
            future, state.future = state.future, None
            try:
                fit = future.result()
            except Exception:
                logger.exception(f"dspark confidence cut: online refit ({mode}) failed; the slopes stay as they were")
                continue
            elapsed_ms = (time.monotonic() - state.submitted_at) * 1e3
            staging = torch.tensor(fit.slopes, dtype=torch.float32, pin_memory=self.device.type == "cuda")
            state.staging = staging
            state.slopes = staging.to(self.device, non_blocking=True)
            state.refit_ct += 1
            fmt = lambda values: [None if math.isnan(v) else round(v, 4) for v in values]  # noqa: E731
            logger.info(
                f"dspark confidence cut: online refit {state.refit_ct} ({mode}) over the last {state.num_snapshot} "
                f"verifies: slopes {fmt(fit.slopes)}, samples per position {fit.samples}, log loss "
                f"{fmt(fit.nll_before)} -> {fmt(fit.nll_after)}, {elapsed_ms:.1f} ms off the step; the cut reads "
                f"max(offline, refit) per position (dropped samples so far {self.dropped_ct})"
            )
