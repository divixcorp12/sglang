"""DSpark's confidence-head verify cut (``SGLANG_DSPARK_CONFIDENCE_CUT``).

With c_k the draft's STS-calibrated confidence that draft k is accepted given drafts 1..k-1 were, S_k = c_1 * ... * c_k
is the probability draft k is accepted, and a verify of w rows (the anchor and drafts 1..w-1) yields E(w) =
1 + S_1 + ... + S_{w-1} expected tokens. The cut verifies w* = argmax_w E(w) / T(w) rows, with T(w) the verify
milliseconds at w live rows. It feeds the same dead-row mask and accept cap as the top-prob gate
(dspark_top_prob_gate.py), so it shares that gate's limit to greedy batches of one request.

T(w) is measured on the running server once it is truly serving (CalibrationGate): the first CALIBRATION_VERIFIES
eligible verifies after that run forced widths 1..gamma+1 round-robin, each timed with CUDA events read lazily at later
verifies, so the hot path never waits on the host, and T(w) is the affine least-squares fit over all of them. Until
then the cut does not apply and every row is verified. A forced width is safe for the same reason any cut is: the
accept is capped at it.
"""

from __future__ import annotations

import logging
import math
import statistics
import sys
from typing import Callable, Optional

import msgspec
import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX, WARMUP_RID_PREFIX
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Lead's choice, 16 samples per width at gamma 5; not tuned.
CALIBRATION_VERIFIES = 96

COPY_ENGINE_OFF = "off"
COPY_ENGINE_PENDING = "pending"
COPY_ENGINE_ARMED = "armed"


class CalibrationGate(msgspec.Struct, frozen=True):
    """Whether a verify may be timed for T(w). Every condition must hold, so a warm-up verify (tiny context, cold
    caches, the copy engine and its split model not yet armed) never enters the fit."""

    # Engine init (CUDA graph capture included) and the startup warm-ups are done (triton_load_watch.serving_started).
    engine_ready: bool
    # The EXL3 RAM-miss copy engine: COPY_ENGINE_OFF when not in use, else pending or armed. It arms after its CPU
    # split and split model calibrate.
    copy_engine: str
    # Every request in the batch is a serving request, not a startup warm-up or a health check.
    serving_request: bool

    @property
    def open(self) -> bool:
        return self.engine_ready and self.copy_engine != COPY_ENGINE_PENDING and self.serving_request

    def describe(self) -> str:
        return (
            f"engine init and startup warm-up done {self.engine_ready}, "
            f"RAM-miss copy engine {self.copy_engine}, serving request {self.serving_request}"
        )


def is_serving_rid(rid) -> bool:
    return isinstance(rid, str) and not rid.startswith((WARMUP_RID_PREFIX, HEALTH_CHECK_RID_PREFIX))


def ram_miss_copy_engine_state() -> str:
    """The EXL3 RAM-miss copy engine's state, read off the service singleton without importing its module."""
    module = sys.modules.get("sglang.srt.layers.moe.exl3_ram_miss")
    service = None if module is None else module.Exl3RamMissService._instance
    if service is None or not service.copy_engine:
        return COPY_ENGINE_OFF
    return COPY_ENGINE_ARMED if service.copy_engine_armed else COPY_ENGINE_PENDING


class ConfidenceCut:
    def __init__(
        self,
        *,
        gamma: int,
        device,
        new_event: Optional[Callable[[], object]] = None,
    ):
        self.gamma = gamma
        self.device = device
        self.num_widths = gamma + 1
        # Any object with CUDA Event's record() / query() / elapsed_time(end); tests inject a CPU fake.
        self._new_event = new_event or (lambda: torch.cuda.Event(enable_timing=True))
        self._num_forced = 0
        self._timed_width: Optional[int] = None
        self._start_event = None
        self._pending: list[tuple[int, object, object]] = []
        self._samples: list[list[float]] = [[] for _ in range(self.num_widths)]
        self._cost_ms: Optional[list[float]] = None
        self._cost_ms_tensor: Optional[torch.Tensor] = None
        self._disabled = False
        self._calibration_started = False
        self._num_held_verifies = 0

    @classmethod
    def from_envs(cls, *, gamma: int, device) -> Optional[ConfidenceCut]:
        """The configured cut, or None when it is off or the launch collects STS data."""
        if not envs.SGLANG_DSPARK_CONFIDENCE_CUT.get():
            return None
        if envs.SGLANG_DSPARK_STS_COLLECT_PATH.get():
            # STS collection labels every draft position against the target's argmax, and a cut row's logits are
            # not valid, so a collect launch builds the head but verifies every row.
            logger.info("dspark confidence cut: STS collect mode, the cut is off and every row is verified")
            return None
        return cls(gamma=gamma, device=device)

    @property
    def calibrated(self) -> bool:
        return self._cost_ms is not None

    @property
    def disabled(self) -> bool:
        return self._disabled

    def cost_ms(self) -> Optional[list[float]]:
        return self._cost_ms

    def applies(self, *, bs: int, sampling_info) -> bool:
        return bs == 1 and (sampling_info is None or sampling_info.is_all_greedy)

    def admit_calibration(self, gate: CalibrationGate) -> bool:
        """Whether an eligible verify before T(w) froze may run the cut (a forced, timed width). A held verify covers
        every row and is not timed."""
        if not gate.open:
            self._num_held_verifies += 1
            return False
        if not self._calibration_started:
            self._calibration_started = True
            logger.info(
                f"dspark confidence cut: T(w) calibration starts at eligible verify {self._num_held_verifies + 1}, "
                f"after {self._num_held_verifies} held back ({gate.describe()}): {CALIBRATION_VERIFIES} verifies, "
                f"widths 1..{self.num_widths} round robin"
            )
        return True

    def verify_lens(self, confidence: torch.Tensor) -> torch.Tensor:
        """[bs] int64 rows to verify in [1, gamma + 1], from the [bs, gamma] per-position confidences."""
        bs = confidence.shape[0]
        if self._cost_ms_tensor is not None:
            return cut_verify_lens(confidence=confidence, cost_ms=self._cost_ms_tensor)
        if self._disabled:
            return self._full_width(bs=bs, device=confidence.device)
        self._drain_ready_samples()
        if self._num_forced < CALIBRATION_VERIFIES:
            width = self._num_forced % self.num_widths + 1
            self._timed_width = width
            return torch.full((bs,), width, dtype=torch.int64, device=confidence.device)
        # Every forced verify is issued but some timings are still in flight: verify all rows until they land.
        self._timed_width = None
        return self._full_width(bs=bs, device=confidence.device)

    def begin_verify(self) -> None:
        if self._timed_width is None:
            return
        self._start_event = self._new_event()
        self._start_event.record()

    def end_verify(self) -> None:
        if self._timed_width is None or self._start_event is None:
            return
        end_event = self._new_event()
        end_event.record()
        self._pending.append((self._timed_width, self._start_event, end_event))
        self._num_forced += 1
        self._timed_width = None
        self._start_event = None

    def _full_width(self, *, bs: int, device) -> torch.Tensor:
        return torch.full((bs,), self.num_widths, dtype=torch.int64, device=device)

    def _drain_ready_samples(self) -> None:
        while self._pending and self._pending[0][2].query():
            width, start_event, end_event = self._pending.pop(0)
            ms = float(start_event.elapsed_time(end_event))
            if math.isfinite(ms):
                self._samples[width - 1].append(ms)
        if self._num_forced == CALIBRATION_VERIFIES and not self._pending:
            self._freeze()

    def _freeze(self) -> None:
        if any(not samples for samples in self._samples):
            self._disable(reason=f"a width has no valid sample (counts {[len(s) for s in self._samples]})")
            return
        fit = AffineCostFit.from_samples(self._samples)
        if fit.base_ms <= 0.0:
            self._disable(reason=f"non-positive fitted T(1) {fit.base_ms} ms")
            return
        cost_ms = fit.cost_ms(num_widths=self.num_widths)
        self._cost_ms = cost_ms
        self._cost_ms_tensor = torch.tensor(cost_ms, dtype=torch.float32, device=self.device)
        medians = [round(statistics.median(samples), 4) for samples in self._samples]
        logger.info(
            f"dspark confidence cut: T(w) ms = {[round(ms, 4) for ms in cost_ms]} "
            f"(a={fit.base_ms:.4f}, b={fit.per_row_ms:.4f}/row, r2={fit.r2:.4f}; "
            f"per-width median ms {medians}, counts {[len(samples) for samples in self._samples]})"
        )

    def _disable(self, *, reason: str) -> None:
        self._disabled = True
        logger.error(f"dspark confidence cut disabled, every verify covers all rows: {reason}")


class AffineCostFit(msgspec.Struct, frozen=True):
    """T(w) = base_ms + per_row_ms * (w - 1), least squares over every sample, with per_row_ms clamped at 0 so a wider
    verify never costs less than a narrower one."""

    base_ms: float
    per_row_ms: float
    r2: float

    @classmethod
    def from_samples(cls, samples_by_width: list[list[float]]) -> AffineCostFit:
        """``samples_by_width[w - 1]`` holds width w's verify ms; at least two widths must have samples."""
        points = [(width_idx, ms) for width_idx, samples in enumerate(samples_by_width) for ms in samples]
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        sxx = sum((x - mean_x) ** 2 for x, _ in points)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
        # A negative slope is noise (a dead row costs nothing extra, never less); the best fit with slope 0 is the mean.
        per_row_ms = max(sxy / sxx, 0.0)
        base_ms = mean_y - per_row_ms * mean_x
        ss_res = sum((y - base_ms - per_row_ms * x) ** 2 for x, y in points)
        ss_tot = sum((y - mean_y) ** 2 for _, y in points)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
        return cls(base_ms=base_ms, per_row_ms=per_row_ms, r2=r2)

    def cost_ms(self, *, num_widths: int) -> list[float]:
        return [self.base_ms + self.per_row_ms * width_idx for width_idx in range(num_widths)]


def cut_verify_lens(*, confidence: torch.Tensor, cost_ms: torch.Tensor) -> torch.Tensor:
    """[bs] int64 w* in [1, gamma + 1] maximising expected tokens per verify ms, from [bs, gamma] confidences and the
    [gamma + 1] cost table. Ties go to the narrower verify."""
    c = confidence.float()
    # A NaN confidence ends the accepted prefix there instead of poisoning every later width's score.
    accept_prob = torch.cumprod(torch.where(torch.isnan(c), torch.zeros_like(c), c), dim=1)
    expected_tokens = 1.0 + torch.nn.functional.pad(torch.cumsum(accept_prob, dim=1), (1, 0))
    return torch.argmax(expected_tokens / cost_ms, dim=1) + 1
