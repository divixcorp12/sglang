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


def resident_routes(topk_ids: torch.Tensor, residents) -> torch.Tensor:
    """The calibration's routes for one MoE layer: the real router's tokens x top-k shape and distinct-expert count
    (measured from ``topk_ids``, capped by the residents), drawn round-robin from ``residents`` (experts already in
    the layer's RAM tier), so the prefill reads nothing into the tier. Each token's top-k stay distinct whenever the
    pool holds at least top-k experts."""
    flat = topk_ids.reshape(-1)
    distinct = int(torch.unique(flat[flat >= 0]).numel())
    pool = [int(e) for e in residents if int(e) >= 0][: max(distinct, topk_ids.shape[-1])]
    if len(pool) < topk_ids.shape[-1]:
        raise RuntimeError(
            f"hot cache calibration: {len(pool)} RAM-resident experts cannot fill a top-{topk_ids.shape[-1]} route"
        )
    picks = torch.arange(flat.numel(), device=topk_ids.device) % len(pool)
    return torch.tensor(pool, dtype=topk_ids.dtype, device=topk_ids.device)[picks].reshape(topk_ids.shape)


def calibration_routes(topk_ids: torch.Tensor, streamer) -> torch.Tensor:
    """``topk_ids`` unchanged outside the calibration; inside it, ``resident_routes`` over the streamer's RAM tier."""
    if not calibrating():
        return topk_ids
    hot = getattr(streamer, "hot_cache", None)
    if hot is not None and topk_ids.numel() <= getattr(streamer, "graph_gather_rows", 0):
        # The capture estimate's verify-sized forward takes the graph-gather path, whose misses insert into the hot
        # cache: route it onto the VRAM residents instead, so every row is a hit.
        return resident_routes(topk_ids, hot.slot_to_expert)
    tier = getattr(streamer, "pinned_host_cache", None)
    if tier is None:
        raise RuntimeError("hot cache calibration needs the pinned RAM tier to route onto")
    return resident_routes(topk_ids, tier.slot_to_expert)


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
    # Graph capture's dummy verify forward, run eagerly: its allocated peak above what it started with, which the
    # captured graph's pool will hold once capture runs after the retire.
    capture_estimate: int = 0
    # torch allocated before the first run, the base of the calibration's transient.
    allocated_before: Optional[int] = None

    @property
    def peak_run(self) -> RunPeak:
        return max(self.runs, key=lambda r: r.reserved_peak)

    @property
    def nontorch_growth(self) -> int:
        last = self.runs[-1]
        return max(0, (self.free_before - last.free_after) - (last.reserved_after - self.reserved_before))

    @property
    def lowest_free(self) -> int:
        """Over the whole run sequence: the highest reserve it reached, and what it took outside torch."""
        return self.free_before - (self.peak_run.reserved_peak - self.reserved_before) - self.nontorch_growth

    @property
    def margin(self) -> int:
        """The allocator's fragmentation at the highest run, or what the final repeat of the full chunk still grew."""
        peak, last, before_last = self.peak_run, self.runs[-1], self.runs[-2]
        return max(0, peak.reserved_peak - peak.allocated_peak, last.reserved_peak - before_last.reserved_after)

    @property
    def required(self) -> int:
        """Free memory the prefill peak must leave before capture: the margin plus what capture will take."""
        return self.margin + self.capture_estimate

    @property
    def slots_to_retire(self) -> int:
        short = self.required - self.lowest_free
        return max(0, math.ceil(short / self.slot_bytes)) if short > 0 else 0

    @property
    def slot_delta(self) -> int:
        """Slots to add (positive: whole slots the room above ``required`` buys) or retire (negative)."""
        room = self.lowest_free - self.required
        return room // self.slot_bytes if room >= 0 else -self.slots_to_retire

    def summary(self) -> dict:
        return {
            "tokens": self.tokens,
            "free_before_mib": round(self.free_before / MIB, 1),
            "reserved_before_mib": round(self.reserved_before / MIB, 1),
            "runs": [{k: round(v / MIB, 1) for k, v in vars(r).items()} for r in self.runs],
            "nontorch_growth_mib": round(self.nontorch_growth / MIB, 1),
            "lowest_free_mib": round(self.lowest_free / MIB, 1),
            "margin_mib": round(self.margin / MIB, 1),
            "capture_estimate_mib": round(self.capture_estimate / MIB, 1),
            "required_mib": round(self.required / MIB, 1),
            "slots_to_retire": self.slots_to_retire,
            "slot_delta": self.slot_delta,
        }


def plan_capacity(
    capacity: Mapping[int, int], delta: int, floor: int, upper: Mapping[int, int]
) -> Dict[int, int]:
    """Each layer's slots after adding ``delta`` slots (one at a time to the emptiest layer that is below its upper
    bound, lowest layer id on ties; whatever no layer can take is left over) or retiring ``-delta``
    (``plan_keep``)."""
    if delta <= 0:
        return plan_keep(capacity, -delta, floor)
    keep = dict(capacity)
    for _ in range(delta):
        open_layers = [l for l in keep if keep[l] < upper.get(l, keep[l])]
        if not open_layers:
            break
        layer = min(open_layers, key=lambda l: (keep[l], l))
        keep[layer] += 1
    return keep


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


def calibration_shapes(chunk: int) -> tuple:
    """The prefill sizes the calibration runs, in order: the startup warm-up's largest (half a chunk, one token
    more, the full chunk) so the reserve their mix strands is measured, then the full chunk again, whose further
    growth is the repeat term of the margin."""
    if chunk <= 2:
        return (chunk, chunk)
    half = chunk // 2
    return (half, half + 1, chunk, chunk)


def calibrate(run_prefill, *, tokens: int, slot_bytes: int, device, run_verify=None) -> Calibration:
    """Measured runs of ``run_prefill(n)`` (the dummy EXTEND forward) over ``calibration_shapes(tokens)``, allocator
    cache emptied first, then one of ``run_verify`` (capture's dummy forward) for the capture estimate."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info(device)[0]
    reserved_before = torch.cuda.memory_reserved(device)
    allocated_before = torch.cuda.memory_allocated(device)
    capture_estimate = 0
    shapes = calibration_shapes(tokens)
    with calibration_context():
        runs = tuple(_measure(lambda n=n: run_prefill(n), device) for n in shapes[:-1])
        runs += (_measure(lambda: run_prefill(shapes[-1]), device),)
        if run_verify is not None:
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            peak = _measure(run_verify, device)
            capture_estimate = max(0, peak.allocated_peak - allocated)
    return Calibration(tokens, free_before, reserved_before, runs, slot_bytes, capture_estimate, allocated_before)


class KvPinMissed(RuntimeError):
    pass


def check_kv_pin(*, pinned: Optional[int], sized: Optional[int]) -> None:
    """Calibration sizes the hot cache around a pinned KV pool (--max-total-tokens). Refuse when the pool came out
    smaller than the pin: the memory under the hot cache's upper bound left too little for it (the step-3 smoke's
    372,480 against 381,184). Without a pin there is nothing to check."""
    if pinned and sized is not None and sized < pinned:
        raise KvPinMissed(
            f"hot cache calibration: the KV pool has {sized} tokens, below --max-total-tokens {pinned}; "
            "raise --mem-fraction-static or lower SGLANG_MOE_HOT_GPU_MB"
        )


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
    from sglang.srt.layers.moe.hot_cache_dummy_extend import (
        dummy_extend_forward,
        dummy_verify_forward,
        forget_graph_path_state,
    )
    from sglang.srt.layers.moe.hot_cache_retire import set_hot_capacity
    from sglang.srt.runtime_context import get_schedule

    check_kv_pin(
        pinned=get_schedule().max_total_tokens, sized=getattr(model_runner, "max_total_num_tokens", None)
    )
    tokens = int(get_schedule().chunked_prefill_size or get_schedule().max_prefill_tokens)
    cache = next(iter(manager.caches.values()))
    device = cache.device
    run = dummy_extend_forward(model_runner, tokens)
    verify = dummy_verify_forward(model_runner) if model_runner.spec_algorithm.is_speculative() else None
    calibration = calibrate(
        run, tokens=tokens, slot_bytes=cache.bytes_per_expert, device=device, run_verify=verify
    )
    forget_graph_path_state(model_runner.model)
    floor = 2 * getattr(manager.gpu_residency, "victim_lanes", 0)
    capacity = {layer: c.capacity for layer, c in manager.caches.items()}
    upper = {
        layer: min(getattr(c, "upper_capacity", c.capacity), manager.gpu_residency.max_capacity)
        for layer, c in manager.caches.items()
    }
    keep = plan_capacity(capacity, calibration.slot_delta, floor, upper)
    retired = set_hot_capacity(manager, keep) if keep != capacity else None
    logger.info(
        "hot cache calibration: %s; slots %d -> %d (upper bound %d)",
        calibration.summary(),
        sum(capacity.values()),
        sum(keep.values()),
        sum(upper.values()),
    )
    result = CalibrationResult(calibration, keep, retired)
    watch_first_full_prefill(model_runner, result)
    return result


def compare_with_real(
    calibration: Calibration, real_allocated_peak: int, real_reserved_peak: int, real_allocated_in: int = None
) -> dict:
    """The real full-size prefill against the calibration's highest run. Live memory is compared as transients
    (peak less allocated at entry) when the entry is known, since capture moves the baseline between the two;
    ``exceeds`` when the real reserve peak is above the calibrated one by more than the margin (the forced routing
    or the run sequence did not hold)."""
    peak = calibration.peak_run
    delta_reserved = real_reserved_peak - peak.reserved_peak
    report = {
        "calibration_allocated_peak_mib": round(peak.allocated_peak / MIB, 1),
        "calibration_reserved_peak_mib": round(peak.reserved_peak / MIB, 1),
        "real_allocated_peak_mib": round(real_allocated_peak / MIB, 1),
        "real_reserved_peak_mib": round(real_reserved_peak / MIB, 1),
        "delta_reserved_mib": round(delta_reserved / MIB, 1),
        "margin_mib": round(calibration.margin / MIB, 1),
        "exceeds": delta_reserved > calibration.margin,
    }
    if real_allocated_in is not None and calibration.allocated_before is not None:
        real_t = real_allocated_peak - real_allocated_in
        cal_t = peak.allocated_peak - calibration.allocated_before
        report.update(
            calibration_transient_mib=round(cal_t / MIB, 1),
            real_transient_mib=round(real_t / MIB, 1),
            delta_transient_mib=round((real_t - cal_t) / MIB, 1),
        )
    return report


def watch_first_full_prefill(model_runner, result: CalibrationResult) -> None:
    """One-shot: measure the first real EXTEND forward of the calibrated size (the HTTP warm-up's) and log its peaks
    beside the calibration's, loudly when it exceeds them by more than the margin; then remove itself. The wrapper is
    an instance attribute on the runner's ``forward``, gone after that forward."""
    original = model_runner.forward
    tokens = result.calibration.tokens

    def measured_forward(forward_batch, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or getattr(forward_batch, "extend_num_tokens", None) != tokens:
            return original(forward_batch, *args, **kwargs)
        del model_runner.forward  # restore the class method before anything else can re-enter
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_in = torch.cuda.memory_allocated()
        out = original(forward_batch, *args, **kwargs)
        torch.cuda.synchronize()
        report = compare_with_real(
            result.calibration,
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
            real_allocated_in=allocated_in,
        )
        (logger.error if report["exceeds"] else logger.info)(
            "hot cache calibration vs the first real %d-token prefill: %s%s",
            tokens,
            report,
            " -- the real prefill exceeds the calibration by more than the margin" if report["exceeds"] else "",
        )
        result.extra["first_real_prefill"] = report
        return out

    model_runner.forward = measured_forward


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
    reserved = torch.cuda.memory_reserved(device)
    projected = projected_free(result.calibration, free=free, reserved=reserved)
    margin = result.calibration.margin
    ok = projected >= margin
    (logger.info if ok else logger.error)(
        "hot cache calibration check %s: free %.1f MiB, torch reserved %.1f MiB, projected free at a %d-token "
        "prefill %.1f MiB, margin %.1f MiB%s",
        stage,
        free / MIB,
        reserved / MIB,
        result.calibration.tokens,
        projected / MIB,
        margin / MIB,
        "" if ok else f" -- SHORT by {(margin - projected) / MIB:.1f} MiB",
    )
    return ok


def projected_free(calibration: Calibration, *, free: int, reserved: int) -> int:
    """Free memory a prefill of the calibrated size would leave from now: whatever of its reserve peak torch no longer
    holds (something released cached blocks since calibration) has to come out of free memory again."""
    return free - max(0, calibration.peak_run.reserved_peak - reserved)
