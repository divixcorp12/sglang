"""Size the expert hot cache at startup from a measured prefill peak (SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP, step 3).

After the KV pools and attention backends exist and before any CUDA graph is captured, the scheduler's hook calls
``maybe_calibrate_hot_cache``. It runs one dummy EXTEND prefill of the configured chunk size through the target model
twice (allocator cache emptied first), measures the lowest free device memory the prefill reaches, and, when that is
below the margin, retires just enough trailing hot-cache slots (hot_cache_retire.retire_hot_slots) to restore it.
Trim-only: it never grows the cache.

Measured, all in this launch:
  F0, R0: cudaMemGetInfo free and torch reserved before the first run (after empty_cache);
  P1, A1: torch's reserved and allocated peaks in run 1; R1, F1: reserved and free after it;
  P2, R2: the same over run 2.
  non-torch growth = max(0, (F0 - F1) - (R1 - R0)): device memory the run took outside torch (cuBLAS workspaces,
    module loads), which it keeps;
  lowest free = F0 - (P1 - R0) - non-torch growth;
  margin = max(P1 - A1, P2 - R1): the allocator's own fragmentation term, the reserve beyond the live peak in run 1
    or what a repeat of the same prefill still grew.
The post-capture check (``check_headroom``) measures the same free memory again after capture and after the HTTP
warm-up's peak, logs it against the margin, and refuses to report healthy below it.

The forward leaves no state behind: route recording and Engram row-cache inserts are skipped under
``calibrating()``, the model is called directly (not through ModelRunner.forward, so no expert-distribution recorder
or forward-pass counter runs), KV writes go to the reserved slot 0 (out_cache_loc and req_pool_indices are zero, as
for graph capture), and the GPU residency updater is still disabled before capture.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

MIB = 1 << 20
_CALIBRATING = contextvars.ContextVar("hot_cache_calibrating", default=False)


def calibrating() -> bool:
    """True inside the calibration forward: state-changing paths skip themselves."""
    return _CALIBRATING.get()


@contextlib.contextmanager
def calibration_context() -> Iterator[None]:
    token = _CALIBRATING.set(True)
    try:
        yield
    finally:
        _CALIBRATING.reset(token)


@dataclass(frozen=True)
class RunPeak:
    reserved_peak: int
    allocated_peak: int
    reserved_after: int
    free_after: int


@dataclass(frozen=True)
class Calibration:
    tokens: int
    free_before: int
    reserved_before: int
    runs: tuple
    slot_bytes: int

    @property
    def nontorch_growth(self) -> int:
        r1 = self.runs[0]
        return max(0, (self.free_before - r1.free_after) - (r1.reserved_after - self.reserved_before))

    @property
    def lowest_free(self) -> int:
        return self.free_before - (self.runs[0].reserved_peak - self.reserved_before) - self.nontorch_growth

    @property
    def margin(self) -> int:
        r1, r2 = self.runs[0], self.runs[-1]
        return max(0, r1.reserved_peak - r1.allocated_peak, r2.reserved_peak - r1.reserved_after)

    @property
    def slots_to_retire(self) -> int:
        short = self.margin - self.lowest_free
        return max(0, math.ceil(short / self.slot_bytes)) if short > 0 else 0

    def summary(self) -> dict:
        return {
            "tokens": self.tokens,
            "free_before_mib": round(self.free_before / MIB, 1),
            "reserved_before_mib": round(self.reserved_before / MIB, 1),
            "runs": [{k: round(v / MIB, 1) for k, v in vars(r).items()} for r in self.runs],
            "nontorch_growth_mib": round(self.nontorch_growth / MIB, 1),
            "lowest_free_mib": round(self.lowest_free / MIB, 1),
            "margin_mib": round(self.margin / MIB, 1),
            "slots_to_retire": self.slots_to_retire,
        }


def plan_keep(capacity: Mapping[int, int], retire: int, floor: int) -> Dict[int, int]:
    """Each layer's kept slots after retiring ``retire`` slots one at a time from the fullest layer (lowest layer id
    on ties), never below ``floor``. Refuses when the layers cannot give up that many."""
    keep = dict(capacity)
    spare = sum(max(0, c - floor) for c in keep.values())
    if retire > spare:
        raise ValueError(f"cannot retire {retire} hot slots: only {spare} sit above the {floor}-slot floor")
    for _ in range(retire):
        layer = max(keep, key=lambda l: (keep[l], -l))
        keep[layer] -= 1
    return keep


@dataclass(frozen=True)
class CalibrationResult:
    calibration: Calibration
    keep: Dict[int, int]
    retire: Optional[object] = None  # hot_cache_retire.RetireResult
    extra: dict = field(default_factory=dict)


def _measure(fn, device) -> RunPeak:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return RunPeak(
        reserved_peak=torch.cuda.max_memory_reserved(device),
        allocated_peak=torch.cuda.max_memory_allocated(device),
        reserved_after=torch.cuda.memory_reserved(device),
        free_after=torch.cuda.mem_get_info(device)[0],
    )


def calibrate(run_prefill, *, tokens: int, slot_bytes: int, device) -> Calibration:
    """Two measured runs of ``run_prefill`` (the dummy EXTEND forward), allocator cache emptied first."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info(device)[0]
    reserved_before = torch.cuda.memory_reserved(device)
    with calibration_context():
        runs = (_measure(run_prefill, device), _measure(run_prefill, device))
    return Calibration(tokens, free_before, reserved_before, runs, slot_bytes)


def maybe_calibrate_hot_cache(model_runner) -> Optional[CalibrationResult]:
    """The scheduler's pre-capture hook. Off (the default) it returns None and touches nothing.

    Takes the live ModelRunner: the dummy forward needs its model, attention backend and pools, which are not
    narrowable to plain values (large-class-style 1.6). It reads them and returns a result; it writes nothing back."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP.get():
        return None
    manager = getattr(model_runner, "expert_hot_cache_manager", None)
    if manager is None or not manager.caches:
        logger.info("hot cache calibration: no hot cache, nothing to size")
        return None
    from sglang.srt.layers.moe.hot_cache_dummy_extend import dummy_extend_forward
    from sglang.srt.layers.moe.hot_cache_retire import retire_hot_slots
    from sglang.srt.runtime_context import get_schedule

    tokens = int(get_schedule().chunked_prefill_size or get_schedule().max_prefill_tokens)
    cache = next(iter(manager.caches.values()))
    device = cache.device
    run = dummy_extend_forward(model_runner, tokens)
    calibration = calibrate(run, tokens=tokens, slot_bytes=cache.bytes_per_expert, device=device)
    floor = 2 * getattr(manager.gpu_residency, "victim_lanes", 0)
    capacity = {layer: c.capacity for layer, c in manager.caches.items()}
    keep = plan_keep(capacity, calibration.slots_to_retire, floor)
    retired = retire_hot_slots(manager, keep) if keep != capacity else None
    logger.info(
        "hot cache calibration: %s; slots %d -> %d",
        calibration.summary(),
        sum(capacity.values()),
        sum(keep.values()),
    )
    return CalibrationResult(calibration, keep, retired)


class HotCacheHeadroomShort(RuntimeError):
    pass


def require_hot_cache_headroom(stage: str, result: Optional[CalibrationResult], device=None) -> None:
    """The scheduler's post-capture check: refuse to start (raise) when free memory is below the calibrated margin.
    After capture the calibration's prefill reserve is still cached by the allocator, so free memory now is what
    the next prefill of that size can count on. Off (no calibration) it does nothing."""
    if result is not None and not check_headroom(stage, result, device):
        raise HotCacheHeadroomShort(
            f"hot cache calibration: free memory {stage} is below the margin; see the log line above"
        )


def check_headroom(stage: str, result: Optional[CalibrationResult], device=None) -> bool:
    """Post-capture / post-warm-up check: free device memory now, against the calibrated margin. Logs every value;
    returns False (the caller refuses to report healthy) when below."""
    if result is None:
        return True
    free = torch.cuda.mem_get_info(device)[0]
    margin = result.calibration.margin
    ok = free >= margin
    (logger.info if ok else logger.error)(
        "hot cache calibration check %s: free %.1f MiB, margin %.1f MiB%s",
        stage,
        free / MIB,
        margin / MIB,
        "" if ok else f" -- SHORT by {(margin - free) / MIB:.1f} MiB",
    )
    return ok
