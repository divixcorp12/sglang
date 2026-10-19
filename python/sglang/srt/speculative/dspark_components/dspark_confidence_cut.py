"""DSpark's confidence-head verify cut (``SGLANG_DSPARK_CONFIDENCE_CUT``).

With c_k the draft's STS-calibrated confidence that draft k is accepted given drafts 1..k-1 were, S_k = c_1 * ... * c_k
is the probability draft k is accepted, and a verify of w rows (the anchor and drafts 1..w-1) yields E(w) =
1 + S_1 + ... + S_{w-1} expected tokens. The cut verifies w* = argmax_w E(w) / T(w) rows, with T(w) the verify
milliseconds at w live rows. It feeds the same dead-row mask and accept cap as the top-prob gate
(dspark_top_prob_gate.py), so it shares that gate's limit to batches of one request, greedy ones unless
SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED is on.

Why the cut keeps sampled decoding exact (SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED). Rows 0..w-1 are live: the anchor and
drafts d_1..d_{w-1}. Row j's target distribution p_{j+1} is the one for position j+1, so a w-row verify runs the
rejection-sampling step for every position k <= w: accept d_k with probability min(1, p_k(d_k) / q_k(d_k)), else draw
from the residual (p_k - q_k)+ and stop. The accept cap (CapCorrectLen) keeps the first w of those positions: when
d_1..d_w are all accepted, the bonus is predicts[accept_index[w - 1]], which is d_w, accepted by position w's step, and
when position w rejects, it is that step's residual draw. Either way the token at position w is position w's ordinary
rejection-sampling output, whose law given the prefix is p_w, so a cut verify is speculative sampling stopped after
position w. Stopped sampling is exact if the decision to run position k's step (w >= k) is independent of d_k given
the prefix d_1..d_{k-1}: d_k is then still distributed as q_k on that event, so the accept/residual step yields p_k,
and when the step is not run the next verify drafts position k afresh. (A rule that read d_k could run the step only
for drafts it likes, which is not q_k; that is the bias the distribution tests' mutant shows.)

The cut's rule meets that condition with a draft to spare. Write E(w) = 1 + S_1 + ... + S_{w-1} and
T(w) = a + b * (w - 1), with a > 0 and b >= 0 (AffineCostFit), and r(w) = E(w) / T(w). Then
r(w + 1) = (T(w) r(w) + b * (S_w / b)) / (T(w) + b) is a weighted mean of r(w) and S_w / b, so r(w + 1) > r(w) exactly
when S_w > b r(w) (with b = 0 directly: when S_w > 0). Once that fails at some w, r(w + 1) >= S_w / b >= S_{w+1} / b
(S_k = c_1 ... c_k is non-increasing because each c_k is in [0, 1]; with b = 0, S_w = 0 makes every later S zero), so
it fails at every later width too: r rises and then stays or falls, and the argmax (ties to the narrower width) is the
first w where r stops rising. So w >= k holds exactly when r rose at steps 1..k-1, and those read S_1..S_{k-1}, that
is c_1..c_{k-1}. c_j is the head's output on the draft's hidden state at position j
(computed from the anchor and mask tokens only: the draft forward never sees a drafted token) and the Markov
embedding of the token before draft j (compute_confidence's prev_seq: the anchor, then d_1..d_{j-1}). So w >= k is a
function of the context, the anchor and d_1..d_{k-2}, never of d_{k-1} or d_k. Sampled verifies take the rule in
that stopping form (stop_rule_verify_lens), so float rounding in a whole-curve argmax cannot make w read a later
confidence; greedy verifies keep cut_verify_lens, and the two agree whenever the scores are exact.

Nothing else on the cut verify reads a drawn token past the cut. The dead-row mask and the Engram -1 posts are set from
w alone; attention is causal, so a dead row cannot change a live row's logits, and the dead rows' logits feed only
positions past w, which the cap drops; the static graph's folded (greedy-only) epilogue never runs a sampled batch.
The T(w) calibration's forced widths are a round robin, independent of every token.

A cost model must keep the condition: row r's price may depend on d_1..d_r (row r carries d_r, and keeping it runs
position r + 1's step), but the decision must be taken row by row. A per-row cost conditioned on a row's own token
(the "miss-conditioned cost" TODO) makes r(w) non-unimodal, and its whole-curve argmax then reads later rows' tokens
to decide earlier ones, which breaks the condition.

T(w) is measured on the running server once it is truly serving (CalibrationGate): the first CALIBRATION_VERIFIES
eligible verifies after that run forced widths 1..gamma+1 round-robin, each timed with CUDA events read lazily at later
verifies, so the hot path never waits on the host, and T(w) is the affine least-squares fit over all of them. Until
then the cut does not apply and every row is verified. A forced width is safe for the same reason any cut is: the
accept is capped at it.

That table describes the server as it was in the calibration's seconds. The first ~35 s of a decode run slow (a drive
whose read latency is 7-9 ms instead of 1 ms: 2026-10-18's sampled-finish captures read 270-380 ms per step against 123-134
ms after, at the same routed experts and lanes), so a frozen fit overstates every width's cost for the rest of the run
(a=131.9, b=51.8 against the realized 92 + 25 (w - 1)). With SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL > 0 the cut
keeps timing its own verifies (the width read back asynchronously, no host sync) and refits T(w) by least squares over
the last LIVE_FIT_WINDOW of them every that many verifies, so the table follows the server out of a slow phase.
"""

from __future__ import annotations

import logging
import math
import statistics
import sys
from collections import deque
from typing import Callable, Optional

import msgspec
import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX, WARMUP_RID_PREFIX
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Lead's choice, 16 samples per width at gamma 5; not tuned.
CALIBRATION_VERIFIES = 96
# The live refit's window (about 35 s of decode at 140 ms a verify) and the least it fits: two widths and this many samples.
LIVE_FIT_WINDOW = 192
LIVE_FIT_MIN_SAMPLES = 48
# Timed verifies in flight at once (the pinned width ring's slots).
LIVE_RING_SLOTS = 64

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
        cut_sampled: bool = False,
        sampled_rescale: Optional[torch.Tensor] = None,
        draft_cost: bool = False,
        refit=None,
        live_fit_interval: int = 0,
    ):
        self.gamma = gamma
        self.device = device
        self.num_widths = gamma + 1
        # SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED: sampled bs-1 batches are cut (and calibrate T(w)) too.
        self.cut_sampled = cut_sampled
        # [gamma] T_head / T_sampled per position (SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED_STS_PATH), or None.
        self._sampled_rescale = sampled_rescale
        # SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST: the rule divides by draft ms + T(w), not T(w) alone.
        self.draft_cost = draft_cost
        # SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL: a dspark_confidence_refit.OnlineRefit, or None.
        self.refit = refit
        # Any object with CUDA Event's record() / query() / elapsed_time(end); tests inject a CPU fake.
        self._new_event = new_event or (lambda: torch.cuda.Event(enable_timing=True))
        self._num_forced = 0
        self._timed_width: Optional[int] = None
        self._draft_start_event = None
        self._start_event = None
        # (width, draft start event or None, verify start event, verify end event)
        self._pending: list[tuple[int, object, object, object]] = []
        self._samples: list[list[float]] = [[] for _ in range(self.num_widths)]
        self._draft_samples: list[float] = []
        self._verify_cost_ms: Optional[list[float]] = None
        self._draft_ms: Optional[float] = None
        self._cost_ms: Optional[list[float]] = None
        self._cost_ms_tensor: Optional[torch.Tensor] = None
        self._disabled = False
        self._calibration_started = False
        self._num_held_verifies = 0
        # SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL: refit T(w) every this many timed cut verifies; 0 is off.
        self._live_fit_interval = live_fit_interval
        self._timing_live = False
        # The widths come back through a pinned ring (allocated at first use): the armed verify's slot, then
        # (slot, start event, end event) of each timed verify, oldest first, and the recent (width, ms) window.
        self._live_ring: Optional[torch.Tensor] = None
        self._live_next_slot = 0
        self._live_armed_slot: Optional[int] = None
        self._live_pending: list[tuple[int, object, object]] = []
        self._live_window: deque[tuple[int, float]] = deque(maxlen=LIVE_FIT_WINDOW)
        self._live_since_fit = 0
        self._live_fit_ct = 0

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
        if torch.device(device).type == "cuda":
            # A cut verify's accept kernels first run at T(w) calibration, after the warm-ups (which never cut), so
            # they would device-load mid-serving; load them now, at worker init.
            from sglang.kernels.ops.speculative.dspark.dspark_accept import preload_cut_accept_kernels

            preload_cut_accept_kernels(device=device, verify_num_draft_tokens=gamma + 1)
        cut_sampled = envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED.get()
        sampled_sts_path = envs.SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED_STS_PATH.get()
        if sampled_sts_path and not cut_sampled:
            raise ValueError(
                "SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED_STS_PATH needs SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED=1; set it or "
                "unset the path."
            )
        sampled_rescale = None
        if sampled_sts_path:
            from sglang.srt.runtime_context import get_spec

            sampled_rescale = sampled_sts_rescale(
                gamma=gamma,
                head_sts_path=get_spec().speculative_dspark_confidence_sts_path,
                sampled_sts_path=sampled_sts_path,
                device=device,
            )
        if cut_sampled:
            logger.info(
                "dspark confidence cut: sampled bs-1 verifies are cut too (SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED), "
                + (
                    f"re-tempered by T_head / T_sampled = {[round(float(f), 4) for f in sampled_rescale]} "
                    f"({sampled_sts_path})"
                    if sampled_rescale is not None
                    else "with the head's STS temperatures"
                )
            )
        draft_cost = envs.SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST.get()
        refit_interval = envs.SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL.get()
        refit = None
        if refit_interval > 0:
            from sglang.srt.speculative.dspark_components.dspark_confidence_refit import (
                REFIT_BUFFER_VERIFIES,
                OnlineRefit,
            )

            refit = OnlineRefit(gamma=gamma, device=device, interval=refit_interval)
            logger.info(
                f"dspark confidence cut: online refit every {refit_interval} verifies over the last "
                f"{REFIT_BUFFER_VERIFIES}, greedy and sampled apart (SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL)"
            )
        live_fit_interval = envs.SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL.get()
        if live_fit_interval > 0:
            logger.info(
                f"dspark confidence cut: T(w) is refit live every {live_fit_interval} cut verifies over the last "
                f"{LIVE_FIT_WINDOW} (SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL)"
            )
        return cls(
            gamma=gamma,
            device=device,
            cut_sampled=cut_sampled,
            sampled_rescale=sampled_rescale,
            draft_cost=draft_cost,
            refit=refit,
            live_fit_interval=live_fit_interval,
        )

    @property
    def calibrated(self) -> bool:
        return self._cost_ms is not None

    @property
    def disabled(self) -> bool:
        return self._disabled

    def cost_ms(self) -> Optional[list[float]]:
        """The per-width cost the rule divides by: T(w), plus the draft ms with SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST."""
        return self._cost_ms

    def verify_cost_ms(self) -> Optional[list[float]]:
        """The fitted T(w) alone."""
        return self._verify_cost_ms

    def draft_ms(self) -> Optional[float]:
        """The median draft ms over the calibration verifies (None before T(w) freezes, or with no draft timed)."""
        return self._draft_ms

    def applies(self, *, bs: int, sampling_info) -> bool:
        return bs == 1 and (self.cut_sampled or sampling_info is None or sampling_info.is_all_greedy)

    def offline_confidence(self, confidence: torch.Tensor, *, sampled: bool) -> torch.Tensor:
        """The head's [bs, gamma] confidences, re-tempered to the sampled STS file's temperatures for a sampled verify
        when one is loaded: what the offline calibration says."""
        if sampled and self._sampled_rescale is not None:
            return rescale_confidence(confidence, self._sampled_rescale)
        return confidence

    def cut_confidence(self, confidence: torch.Tensor, *, sampled: bool) -> torch.Tensor:
        """The [bs, gamma] confidences the cut reads: the offline ones, raised by the online refit where it says more
        (never lowered). Per-position maps of this step's confidences only, so the stopping form's argument holds."""
        offline = self.offline_confidence(confidence, sampled=sampled)
        if self.refit is None:
            return offline
        return self.refit.apply(offline, sampled=sampled)

    def observe_outcome(
        self,
        *,
        confidence: torch.Tensor,
        verify_lens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        sampled: bool,
    ) -> None:
        """A bs-1 cut verify's accept, for the online refit (no-op without one). ``confidence`` is the head's."""
        if self.refit is None:
            return
        self.refit.observe(
            confidence=self.offline_confidence(confidence, sampled=sampled),
            verify_lens=verify_lens,
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            sampled=sampled,
        )

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

    def verify_lens(self, confidence: torch.Tensor, *, sampled: bool = False) -> torch.Tensor:
        """[bs] int64 rows to verify in [1, gamma + 1], from the [bs, gamma] per-position confidences. ``sampled``
        verifies take the rule in its stopping form (module docstring)."""
        bs = confidence.shape[0]
        if self._cost_ms_tensor is not None:
            if self._live_fit_interval > 0:
                self._drain_live_samples()
            if sampled:
                lens = stop_rule_verify_lens(confidence=confidence, cost_ms=self._cost_ms_tensor)
            else:
                lens = cut_verify_lens(confidence=confidence, cost_ms=self._cost_ms_tensor)
            if self._live_fit_interval > 0:
                self._arm_live_timing(lens)
            return lens
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

    def begin_draft(self) -> None:
        """Before a step's draft, until T(w) freezes: an event a timed verify's begin_verify closes, so the draft ms
        (the proposal, the confidence and every launch up to the verify's) is timed on the same verifies as T(w),
        read as lazily. An untimed step's event is replaced by the next step's."""
        if self._cost_ms is not None or self._disabled:
            return
        self._draft_start_event = self._new_event()
        self._draft_start_event.record()

    def begin_verify(self) -> None:
        if self._timed_width is None and not self._timing_live:
            return
        self._start_event = self._new_event()
        self._start_event.record()

    def end_verify(self) -> None:
        if self._timing_live:
            if self._start_event is not None and self._live_armed_slot is not None:
                end_event = self._new_event()
                end_event.record()
                self._live_pending.append((self._live_armed_slot, self._start_event, end_event))
            self._timing_live = False
            self._live_armed_slot = None
            self._start_event = None
            return
        if self._timed_width is None or self._start_event is None:
            return
        end_event = self._new_event()
        end_event.record()
        self._pending.append((self._timed_width, self._draft_start_event, self._start_event, end_event))
        self._num_forced += 1
        self._timed_width = None
        self._draft_start_event = None
        self._start_event = None

    def _arm_live_timing(self, lens: torch.Tensor) -> None:
        """Times the coming verify for the live refit: its width is copied to a pinned host slot without a sync (the
        copy is ahead of the verify on the stream, so the end event's completion covers it). A full ring skips the
        verify, so a slot is never rewritten before its sample is read."""
        if len(self._live_pending) >= LIVE_RING_SLOTS:
            return
        if self._live_ring is None:
            self._live_ring = torch.zeros(LIVE_RING_SLOTS, dtype=torch.int64, pin_memory=lens.is_cuda)
        slot = self._live_next_slot
        self._live_next_slot = (slot + 1) % LIVE_RING_SLOTS
        self._live_ring[slot : slot + 1].copy_(lens.reshape(-1)[:1], non_blocking=True)
        self._live_armed_slot = slot
        self._timing_live = True

    def _drain_live_samples(self) -> None:
        """Moves the completed timed verifies into the window, in order, and refits T(w) every interval of them."""
        while self._live_pending and self._live_pending[0][2].query():
            slot, start_event, end_event = self._live_pending.pop(0)
            ms = float(start_event.elapsed_time(end_event))
            width = int(self._live_ring[slot])
            if math.isfinite(ms) and 1 <= width <= self.num_widths:
                self._live_window.append((width, ms))
                self._live_since_fit += 1
        if self._live_since_fit >= self._live_fit_interval:
            self._live_since_fit = 0
            self._refit_live()

    def _refit_live(self) -> None:
        by_width: list[list[float]] = [[] for _ in range(self.num_widths)]
        for width, ms in self._live_window:
            by_width[width - 1].append(ms)
        if len(self._live_window) < LIVE_FIT_MIN_SAMPLES or sum(1 for s in by_width if s) < 2:
            return
        fit = AffineCostFit.from_samples(by_width)
        if fit.base_ms <= 0.0:
            return
        self._install_cost(fit)
        self._live_fit_ct += 1
        if self._live_fit_ct == 1 or self._live_fit_ct % 8 == 0:
            logger.info(
                f"dspark confidence cut: live T(w) refit #{self._live_fit_ct} over {len(self._live_window)} verifies "
                f"(a={fit.base_ms:.4f}, b={fit.per_row_ms:.4f}/row, r2={fit.r2:.4f}; widths seen "
                f"{[len(s) for s in by_width]})"
            )

    def _install_cost(self, fit: AffineCostFit) -> None:
        """Makes ``fit`` the table the rule divides by: T(w), plus the calibration's draft ms with the draft cost on."""
        verify_cost_ms = fit.cost_ms(num_widths=self.num_widths)
        self._verify_cost_ms = verify_cost_ms
        # The draft costs the same at every width (gamma drafts, one draft forward), so it adds to a alone: the cost
        # stays affine with b unchanged, and the stopping form's argument holds (a + draft > 0).
        self._cost_ms = (
            [ms + self._draft_ms for ms in verify_cost_ms]
            if self.draft_cost and self._draft_ms is not None
            else verify_cost_ms
        )
        table = torch.tensor(self._cost_ms, dtype=torch.float32)
        if self._cost_ms_tensor is None:
            self._cost_ms_tensor = table.to(self.device)
        else:
            # A refit on the serving path: written in place from pinned memory, so it syncs nothing.
            self._cost_ms_tensor.copy_(table.pin_memory() if self._cost_ms_tensor.is_cuda else table, non_blocking=True)

    def _full_width(self, *, bs: int, device) -> torch.Tensor:
        return torch.full((bs,), self.num_widths, dtype=torch.int64, device=device)

    def _drain_ready_samples(self) -> None:
        while self._pending and self._pending[0][3].query():
            width, draft_start_event, start_event, end_event = self._pending.pop(0)
            ms = float(start_event.elapsed_time(end_event))
            if math.isfinite(ms):
                self._samples[width - 1].append(ms)
            if draft_start_event is not None:
                draft_ms = float(draft_start_event.elapsed_time(start_event))
                if math.isfinite(draft_ms):
                    self._draft_samples.append(draft_ms)
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
        self._draft_ms = statistics.median(self._draft_samples) if self._draft_samples else None
        if self.draft_cost and self._draft_ms is None:
            logger.warning("dspark confidence cut: no draft was timed, so the cost leaves the draft out")
        self._install_cost(fit)
        verify_cost_ms, cost_ms = self._verify_cost_ms, self._cost_ms
        medians = [round(statistics.median(samples), 4) for samples in self._samples]
        if self._draft_ms is None:
            draft = "draft ms not timed"
        else:
            draft = (
                f"draft ms {self._draft_ms:.4f} (median; mean {statistics.fmean(self._draft_samples):.4f}, "
                f"n {len(self._draft_samples)}), "
                + (
                    f"in the cost: {[round(ms, 4) for ms in cost_ms]}"
                    if cost_ms is not verify_cost_ms
                    else "not in the cost (SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST off)"
                )
            )
        logger.info(
            f"dspark confidence cut: T(w) ms = {[round(ms, 4) for ms in verify_cost_ms]} "
            f"(a={fit.base_ms:.4f}, b={fit.per_row_ms:.4f}/row, r2={fit.r2:.4f}; "
            f"per-width median ms {medians}, counts {[len(samples) for samples in self._samples]}); {draft}"
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


def rescale_confidence(confidence: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """sigmoid(logit(c) * factor) per position: a head output c = sigmoid(raw / T_head) re-tempered to
    sigmoid(raw / T) when factor = T_head / T. 0 and 1 stay put, and a NaN stays NaN (the cut reads it as 0)."""
    return torch.sigmoid(torch.logit(confidence.float()) * factor)


def sampled_sts_rescale(*, gamma: int, head_sts_path: Optional[str], sampled_sts_path: str, device) -> torch.Tensor:
    """[gamma] T_head / T_sampled: the head applies ``head_sts_path``'s temperatures (none: 1), and a sampled verify
    wants ``sampled_sts_path``'s."""
    from sglang.srt.speculative.dspark_components.dspark_sts import load_sts_calibration_from_path

    head = load_sts_calibration_from_path(head_sts_path).temperatures if head_sts_path else [1.0] * gamma
    sampled = load_sts_calibration_from_path(sampled_sts_path).temperatures
    if len(head) != gamma or len(sampled) != gamma:
        raise ValueError(
            f"dspark confidence cut: STS files fit for gamma {len(head)} ({head_sts_path}) and {len(sampled)} "
            f"({sampled_sts_path}), the runtime gamma is {gamma}"
        )
    return torch.tensor([h / s for h, s in zip(head, sampled)], dtype=torch.float32, device=device)


def _tokens_per_ms(*, confidence: torch.Tensor, cost_ms: torch.Tensor) -> torch.Tensor:
    """[bs, gamma + 1] E(w) / T(w) for w = 1..gamma + 1; column w - 1 reads confidences 1..w-1 only."""
    c = confidence.float()
    # A NaN confidence ends the accepted prefix there instead of poisoning every later width's score.
    accept_prob = torch.cumprod(torch.where(torch.isnan(c), torch.zeros_like(c), c), dim=1)
    expected_tokens = 1.0 + torch.nn.functional.pad(torch.cumsum(accept_prob, dim=1), (1, 0))
    return expected_tokens / cost_ms


def cut_verify_lens(*, confidence: torch.Tensor, cost_ms: torch.Tensor) -> torch.Tensor:
    """[bs] int64 w* in [1, gamma + 1] maximising expected tokens per verify ms, from [bs, gamma] confidences and the
    [gamma + 1] cost table. Ties go to the narrower verify."""
    return torch.argmax(_tokens_per_ms(confidence=confidence, cost_ms=cost_ms), dim=1) + 1


def stop_rule_verify_lens(*, confidence: torch.Tensor, cost_ms: torch.Tensor) -> torch.Tensor:
    """cut_verify_lens's w* as a stopping rule: widen while E(w + 1) / T(w + 1) beats E(w) / T(w), and stop at the first
    width where it does not. Equal to the argmax for an affine T(w) (module docstring), and by construction width
    w + 1 is chosen from confidences 1..w only, which is what keeps a sampled verify exact."""
    rate = _tokens_per_ms(confidence=confidence, cost_ms=cost_ms)
    rises = (rate[:, 1:] > rate[:, :-1]).to(torch.int64)
    return 1 + torch.cumprod(rises, dim=1).sum(dim=1)
