"""Live expert prefetch scoring: per-target-layer candidates written inside the decode graph from tap writes.

Every per-token op runs from ``FeatureStore.after_write`` during eager forwards and graph
capture, so replay executes recorded kernels only. ``bank`` rows feed the shared copy
layer; a non-off ``SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE`` also posts and joins a
captured side-stream pull of each target's top candidate through ``expert_gpu_pull.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from sglang.kernels.ops.moe.expert_cache_transfer import expert_row_segments
from sglang.kernels.ops.moe.expert_prefetch_top1 import (
    select_prefetch_top1_cuda,
    supports_prefetch_top1_cuda,
)
from sglang.srt.layers.moe.expert_gpu_pull import ExpertGpuPullPipeline, ExpertGpuPullTarget
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_prediction.serving.candidates import (
    BudgetRecall,
    DedicatedPrefetchSlot,
    PrefetchCandidateBank,
)
from sglang.srt.layers.moe.expert_prediction.serving.calibration import PullCalibrationHistogram
from sglang.srt.layers.moe.expert_prediction.serving.checkpoints import load_prefetch_checkpoints
from sglang.srt.layers.moe.expert_prediction.serving.scorers import ApexScorer, LlaporScorer
from sglang.srt.layers.moe.expert_row_plan import ExpertRowPlan

logger = logging.getLogger(__name__)


def _layer_tensor(layer: torch.nn.Module, name: str) -> torch.Tensor:
    value = getattr(layer, name)
    return value.data if isinstance(value, torch.nn.Parameter) else value


def pull_outcome_counts(
    flat_ids: torch.Tensor,
    missed_mask: torch.Tensor,
    predicted_expert: torch.Tensor,
    posted_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-forward pull outcome: covered/residual actual-miss ROUTE counts, a waste flag,
    and the PHYSICAL row count this forward actually delivered.

    ``flat_ids`` are this forward's routed expert ids and ``missed_mask`` marks
    the ones the ordinary residency check already found non-resident (the
    demand-path miss set); only those can be "covered" by a speculative pull.
    ``covered`` counts actual-miss ROUTES equal to ``predicted_expert`` -- in a
    multi-token forward several tokens can route to the same predicted expert,
    so this can exceed 1. ``residual`` counts actual-miss routes that are not,
    and so still cross the ordinary demand path. ``wasted`` is true when a real
    prediction (``predicted_expert != -1``) covered no route this forward --
    the posted pull's row went unused.

    ``posted`` is the PHYSICAL row count: 1 when a real prediction was posted
    this forward (regardless of how many routes it covered, or none), 0 when
    nothing was posted. A capacity-1 side pull delivers exactly one row per
    posted forward; ``covered`` (route-level) must never stand in for it --
    summing ``covered`` across a multi-token forward with overlap on the
    predicted expert overcounts the physical rows actually copied. All four
    are device tensors; nothing here reads the device.
    """
    valid = predicted_expert.reshape(()) >= 0
    if posted_count is not None:
        valid = valid & (posted_count.reshape(()) == 1)
    matches = missed_mask & valid & (flat_ids == predicted_expert)
    covered = matches.sum()
    residual = missed_mask.sum() - covered
    wasted = valid & (covered == 0)
    posted = valid.to(torch.int64)
    return covered, residual, wasted, posted


def route_covered_residual(
    flat_ids: torch.Tensor,
    missed_mask: torch.Tensor,
    predicted_expert: torch.Tensor,
    slot_index: int,
    demand_remap: torch.Tensor,
    posted_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Redirect actual-miss routes for ``predicted_expert`` to the dedicated ``slot_index``.

    Only rows ``missed_mask`` marks non-resident can be redirected -- a
    resident hit already has its own slot and must keep it. Every other row
    keeps its existing ``demand_remap`` destination, the ordinary demand-scratch
    row the miss path already assigned it. ``predicted_expert`` at -1 (nothing
    posted this forward) never equals a real expert id, so nothing is
    redirected.
    """
    valid = predicted_expert.reshape(()) >= 0
    if posted_count is not None:
        valid = valid & (posted_count.reshape(()) == 1)
    covered = missed_mask & valid & (flat_ids == predicted_expert)
    return torch.where(covered, torch.full_like(demand_remap, slot_index), demand_remap)


class PullDeliveryStats:
    """Device counters for one target layer's pull outcomes; host-read only via ``snapshot``.

    ``counts`` is strictly additive between resets. ``ExpertHotCacheManager.
    discard_graph_capture_routes`` zeros it with the sibling graph/residency counters,
    removing the captured warmup replay before normal serving resumes. A periodic
    overlay-fresh flush stores the latest cumulative snapshot rather than a diff, so it
    remains correct across that reset.
    ``counts[3]`` (``posted``) is the PHYSICAL row count and is what feeds a "rows
    delivered" telemetry sink (e.g. ``ExpertHotCacheManager.record_side_pull_delivery``);
    ``counts[0]`` (``covered``) is ROUTE-level and can overcount physical rows on a
    multi-token forward with overlap on the predicted expert -- never substitute one for
    the other.
    """

    def __init__(self, device: torch.device | None = None, *, counts: torch.Tensor | None = None) -> None:
        if counts is None:
            if device is None:
                raise ValueError("PullDeliveryStats needs a device or a persistent counter view")
            counts = torch.zeros(4, dtype=torch.int64, device=device)
        if counts.dtype != torch.int64 or counts.numel() != 4 or not counts.is_contiguous():
            raise ValueError("PullDeliveryStats counters must be a contiguous int64[4] tensor")
        self.counts = counts

    def add(
        self, covered: torch.Tensor, residual: torch.Tensor, wasted: torch.Tensor, posted: torch.Tensor
    ) -> None:
        self.counts[0].add_(covered)
        self.counts[1].add_(residual)
        self.counts[2].add_(wasted.to(torch.int64))
        self.counts[3].add_(posted)

    def snapshot(self) -> tuple[int, int, int, int]:
        """Host read of cumulative (covered, residual, wasted, posted); metric log intervals only."""
        values = self.counts.cpu().tolist()
        return values[0], values[1], values[2], values[3]


class PrefetchPuller:
    """Speculative one-row side-stream pull of a target layer's best non-resident candidate.

    Reserves ``DedicatedPrefetchSlot``'s trailing row of each target's hot-cache
    tensor allocation (plan section 7.1) and drives a captured
    ``ExpertGpuPullPipeline`` pull for it. ``post_target`` forks the pull from
    the current stream -- call it after the caller's own demand copies for this
    step, since the whole point is that the pull overlaps real compute rather
    than sitting immediately before its own join. ``join_target`` folds the
    pull back in after the target layer's actual routing and returns the
    covered/residual remap.

    Construction asserts every reservation against the hot cache's *real*
    tensor allocation and *real* ``expert_to_slot``, not the formula alone: a
    hot cache built without the trailing row raises here, at setup, rather
    than corrupting a demand-scratch row silently at the first pull.
    """

    def __init__(
        self,
        *,
        bank: PrefetchCandidateBank,
        layer_ids: Sequence[int],
        hot_caches: Mapping[int, Any],
        device: torch.device,
        pull_mode: str = "always",
        fused_top1: bool = False,
    ) -> None:
        if pull_mode not in {"count_zero", "always"}:
            raise ValueError(f"unsupported prefetch pull mode: {pull_mode}")
        self.bank = bank
        self._pipeline = ExpertGpuPullPipeline(device)
        self._slots: dict[int, DedicatedPrefetchSlot] = {}
        self._targets: dict[int, ExpertGpuPullTarget] = {}
        self._plans: dict[int, ExpertRowPlan] = {}
        self._selected_valid: dict[int, torch.Tensor] = {}
        self._should_post: dict[int, torch.Tensor] = {}
        # The GPU residency owner can replace expert_to_slot between graph
        # replays. Keep the cache object and read its live map when posting.
        self._hot_caches = dict(hot_caches)
        self._fused_top1 = fused_top1
        self._counter_layers = tuple(sorted(layer_ids))
        self._counter_rows = {layer_id: row for row, layer_id in enumerate(self._counter_layers)}
        # A single contiguous bank makes every periodic delivery snapshot one
        # D2H transfer, instead of one transfer per target layer.
        self._outcome_counter_bank = torch.zeros(
            (len(self._counter_layers), 4), dtype=torch.int64, device=device
        )
        self.stats: dict[int, PullDeliveryStats] = {}
        for layer_id in self._counter_layers:
            cache = hot_caches.get(layer_id)
            if cache is None:
                raise ValueError(f"expert prefetch pull needs a hot cache for target layer {layer_id}")
            slot = DedicatedPrefetchSlot(capacity=cache.capacity, demand_rows=cache.scratch_rows)
            allocation_rows = next(iter(cache.tensors.values())).shape[0]
            slot.assert_within_allocation(allocation_rows)
            slot.assert_excluded_from_mapping(cache.expert_to_slot)
            streamer = cache.streamer
            pairs = [
                (_layer_tensor(streamer.layer, name), cache.tensors[name]) for name in streamer.tensor_names
            ]
            segments = expert_row_segments(pairs)
            expert_ids = torch.full((1,), -1, dtype=torch.int64, device=device)
            slots_t = torch.full((1,), slot.index, dtype=torch.int32, device=device)
            count = torch.zeros(1, dtype=torch.int32, device=device)
            plan = ExpertRowPlan(expert_ids=expert_ids, slots=slots_t, count=count)
            self._slots[layer_id] = slot
            self._targets[layer_id] = self._pipeline.create_target(
                f"prefetch_pull_{layer_id}", segments, plan, slot.index
            )
            self._plans[layer_id] = plan
            self._selected_valid[layer_id] = torch.zeros(1, dtype=torch.bool, device=device)
            self._should_post[layer_id] = torch.tensor(
                pull_mode == "always", dtype=torch.bool, device=device
            )
            self.stats[layer_id] = PullDeliveryStats(
                counts=self._outcome_counter_bank[self._counter_rows[layer_id]]
            )

    def post_target(self, target_layer: int) -> None:
        target = self._targets.get(target_layer)
        if target is None:
            return
        plan = self._plans[target_layer]
        candidate = self.bank.ids_for(target_layer)[:1]
        valid = self._should_post[target_layer] & self.bank.valid_for(target_layer)[:1]
        valid = valid & (
            self._hot_caches[target_layer].expert_to_slot.index_select(0, candidate) < 0
        )
        plan.expert_ids.copy_(torch.where(valid, candidate, candidate.new_full((1,), -1)))
        plan.count.copy_(valid.to(torch.int32).reshape(1))
        self._selected_valid[target_layer].copy_(valid)
        self._pipeline.post_target(target)

    def select_and_post_target(self, target_layer: int, scores: torch.Tensor) -> bool:
        """Select and post a serving candidate, using JIT only when it is exact.

        Returns whether the JIT kernel was used.  Its fallback is the complete
        reference candidate-bank path, not an approximation.  The JIT owns the
        persistent ``id``, ``valid`` and ``count`` buffers consumed by both the
        side pull and the target gather planner.
        """
        target = self._targets.get(target_layer)
        if target is None:
            return False
        plan = self._plans[target_layer]
        expert_to_slot = self._hot_caches[target_layer].expert_to_slot
        if self._fused_top1 and supports_prefetch_top1_cuda(scores, expert_to_slot):
            select_prefetch_top1_cuda(
                scores,
                expert_to_slot,
                plan.expert_ids,
                self._selected_valid[target_layer],
                plan.count,
            )
            self._pipeline.post_target(target)
            return True
        self.bank.write(target_layer, scores, expert_to_slot=expert_to_slot)
        self.post_target(target_layer)
        return False

    def predicted_expert_for(self, target_layer: int) -> Optional[torch.Tensor]:
        """This forward's posted prediction for ``target_layer``, or ``None`` if it is not a pull target.

        The same int64 ``[1]`` device tensor ``join_target`` reads back for its
        stats. Safe to read before ``join_target``'s stream join: ``post_target``
        wrote it on the caller's own (main) stream, so ordinary same-stream
        ordering already covers the value, unlike the pulled row's data, which
        lives on the side stream ``join_target`` joins.
        """
        plan = self._plans.get(target_layer)
        return None if plan is None else plan.expert_ids

    def posted_count_for(self, target_layer: int) -> Optional[torch.Tensor]:
        """The current forward's persistent posted-row count for graph planning."""
        plan = self._plans.get(target_layer)
        return None if plan is None else plan.count

    def outcome_counters_for(self, target_layer: int) -> Optional[torch.Tensor]:
        """Persistent planner-owned outcome counters for a fused target gather."""
        stats = self.stats.get(target_layer)
        return None if stats is None else stats.counts

    def snapshot_delivery_stats(self) -> dict[int, tuple[int, int, int, int]]:
        """Read the whole target bank with one host transfer for a real consumer."""
        values = self._outcome_counter_bank.cpu().tolist()
        return {
            layer_id: tuple(values[row])
            for layer_id, row in self._counter_rows.items()
        }

    def discard_delivery_stats(self) -> None:
        """Reset all target counters in a single graph-stable device operation."""
        self._outcome_counter_bank.zero_()

    def slot_for(self, target_layer: int) -> int:
        """The dedicated slot index reserved for ``target_layer``'s pull."""
        return self._slots[target_layer].index

    def join_target(
        self,
        target_layer: int,
        *,
        flat_ids: torch.Tensor,
        missed_mask: torch.Tensor,
        demand_remap: torch.Tensor,
        planner_owned: bool = False,
        clear_after_join: bool = False,
    ) -> torch.Tensor:
        target = self._targets.get(target_layer)
        if target is None:
            return demand_remap
        self._pipeline.join_target(target)
        if planner_owned:
            # The fused planner has already written its final remap and all
            # four outcomes into this target's persistent counter tensor.
            if clear_after_join:
                self._clear_plan(target_layer)
            return demand_remap
        predicted = self._plans[target_layer].expert_ids
        posted_count = self._plans[target_layer].count
        covered, residual, wasted, posted = pull_outcome_counts(
            flat_ids, missed_mask, predicted, posted_count
        )
        self.stats[target_layer].add(covered, residual, wasted, posted)
        remap = route_covered_residual(
            flat_ids,
            missed_mask,
            predicted,
            self._slots[target_layer].index,
            demand_remap,
            posted_count,
        )
        if clear_after_join:
            self._clear_plan(target_layer)
        return remap

    def _clear_plan(self, target_layer: int) -> None:
        plan = self._plans[target_layer]
        plan.expert_ids.fill_(-1)
        plan.count.zero_()
        self._selected_valid[target_layer].zero_()

    def join_unsupported_target(self, target_layer: int) -> None:
        """Drain and clear a target whose gather is outside graph support.

        The source scoring hook can still have forked a count-zero-or-one pull.
        Joining before eager/tail gather keeps the no-op fork/join capture
        structure valid; clearing after the join prevents stale metadata from
        leaking into the next replay.
        """
        target = self._targets.get(target_layer)
        if target is None:
            return
        self._pipeline.join_target(target)
        # The payload still crossed the side stream even though this target
        # cannot consume its dedicated row. Attribute it as a physical wasted
        # post rather than silently dropping tail/eager traffic from telemetry.
        posted = (self._plans[target_layer].count.reshape(()) == 1).to(torch.int64)
        zero = self.stats[target_layer].counts.new_zeros(())
        self.stats[target_layer].add(zero, zero, posted.bool(), posted)
        self._clear_plan(target_layer)

_FEATURES = {
    "llapor": frozenset({RouteFeature.ROUTER_INPUT, RouteFeature.TOPK_IDS, RouteFeature.TOPK_WEIGHTS}),
    "apex": frozenset({RouteFeature.PRE_MIXER, RouteFeature.TOPK_IDS}),
}
# The last feature each tap writes for a layer; scoring fires on it so the others are fresh.
_SCORE_TRIGGER = {"llapor": RouteFeature.TOPK_WEIGHTS, "apex": RouteFeature.PRE_MIXER}


class PrefetchScoring:
    """Owns scorers, the candidate bank Phase B reads, and shadow budget-recall counters."""

    @staticmethod
    def features_for(predictor: str) -> frozenset[RouteFeature]:
        return _FEATURES[predictor]

    @classmethod
    def build(cls, *, predictor: str, model_dir: Path, specs: Sequence[MoeLayerSpec], **kwargs) -> "PrefetchScoring":
        checkpoints = load_prefetch_checkpoints(model_dir, predictor=predictor, specs=specs)
        return cls.from_checkpoints(predictor=predictor, checkpoints=checkpoints, specs=specs, **kwargs)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        predictor: str,
        checkpoints: Mapping[int, Any],
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        hot_caches: Mapping[int, Any],
        width: int,
        budget: int,
        tau: float,
        dtype: torch.dtype,
        device: torch.device,
        pull_mode: str = "off",
        shadow_recall: bool = True,
        calibration: bool = False,
        calibration_file: Path | None = None,
        calibration_provenance: str = "",
        fused_top1: bool = True,
    ) -> "PrefetchScoring":
        by_layer = {spec.layer_id: spec for spec in specs}
        if width > min(spec.num_experts for spec in specs):
            raise ValueError("SGLANG_MOE_EXPERT_PREFETCH_CANDIDATES exceeds the expert count")
        missing_caches = sorted(set(checkpoints) - set(hot_caches))
        if missing_caches:
            raise ValueError(f"expert prefetch scoring needs hot caches for layers {missing_caches}")
        if predictor == "llapor":
            scorers = {target: LlaporScorer(c, num_experts=by_layer[target].num_experts, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {c.source_layer: target for target, c in checkpoints.items()}
            next_target = dict(source_of)
        else:
            scorers = {target: ApexScorer(c, num_experts=by_layer[target].num_experts, tau=tau, dtype=dtype, device=device)
                       for target, c in checkpoints.items()}
            source_of = {target: target for target in checkpoints}
            next_target = {}
        # A real serving pull consumes exactly one candidate. Observability
        # consumers need the reference top-W ordering, so only collapse the
        # bank when shadow recall, calibration, and count-zero diagnostics are
        # all absent.
        serving_top1 = (
            fused_top1 and pull_mode == "always" and not shadow_recall and not calibration
        )
        bank = PrefetchCandidateBank(
            layer_ids=list(checkpoints),
            width=1 if serving_top1 else width,
            device=device,
        )
        puller = (
            PrefetchPuller(
                bank=bank,
                layer_ids=list(checkpoints),
                hot_caches=hot_caches,
                device=device,
                pull_mode=pull_mode,
                fused_top1=serving_top1,
            )
            if pull_mode != "off"
            else None
        )
        histogram = (
            PullCalibrationHistogram(layer_ids=list(checkpoints), device=device)
            if calibration
            else None
        )
        if puller is not None:
            for layer_id in checkpoints:
                hot_caches[layer_id].streamer.prefetch_puller = puller
        if histogram is not None:
            managers = set()
            for layer_id in checkpoints:
                cache = hot_caches[layer_id]
                cache.streamer.prefetch_calibration = histogram
                manager = getattr(cache, "_owner_manager", None)
                if manager is not None:
                    managers.add(manager)
            for manager in managers:
                manager.register_prefetch_calibration(histogram)
        scoring = cls(
            predictor=predictor, scorers=scorers, source_of=source_of, next_target=next_target, store=store,
            hot_caches=hot_caches,
            bank=bank,
            recall=(
                BudgetRecall(layer_ids=list(checkpoints), budget=budget, device=device)
                if shadow_recall
                else None
            ),
            puller=puller,
            calibration=histogram,
            calibration_file=calibration_file,
            calibration_provenance=calibration_provenance,
        )
        store.after_write = scoring._on_write
        logger.info("MoE expert prefetch scoring: predictor=%s targets=%d width=%d bank_width=%d serving_top1=%s budget=%d state_bytes=%d pull_mode=%s shadow_recall=%s calibration=%s",
                    predictor, len(checkpoints), width, bank.width, serving_top1, budget,
                    scoring.state_nbytes, pull_mode, shadow_recall, calibration)
        return scoring

    def __init__(self, *, predictor, scorers, source_of, next_target, store, hot_caches, bank, recall, puller=None, calibration=None, calibration_file=None, calibration_provenance="") -> None:
        self.predictor = predictor
        self.targets = sorted(scorers)
        # Source layer -> target layer for a next-layer predictor; empty for same-layer predictors.
        self.next_target = next_target
        self._scorers = scorers
        self._source_of = source_of
        self._store = store
        self._hot_caches = dict(hot_caches)
        self.bank = bank
        self.recall = recall
        self.puller = puller
        self._serving_top1 = bool(puller is not None and puller._fused_top1)
        self.calibration = calibration
        self._calibration_file = calibration_file
        self._calibration_provenance = calibration_provenance
        if self.calibration is not None and self._calibration_file is not None:
            self.calibration.write(self._calibration_file, json.loads(calibration_provenance or "{}"), complete=False)

    @property
    def required_features(self) -> frozenset[RouteFeature]:
        return _FEATURES[self.predictor]

    @property
    def state_nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self._scorers.values() for t in (*s.parameters(), *s.buffers()))

    def _on_write(self, layer_id: int, feature: RouteFeature, rows: int) -> None:
        if self.recall is not None and feature is RouteFeature.TOPK_IDS and layer_id in self._scorers:
            self.recall.observe(
                target_layer=layer_id, candidate_ids=self.bank.ids_for(layer_id),
                candidate_valid=self.bank.valid_for(layer_id),
                topk_ids=self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                expert_to_slot=self._hot_caches[layer_id].expert_to_slot,
            )
        if feature is not _SCORE_TRIGGER[self.predictor] or layer_id not in self._source_of:
            return
        target = self._source_of[layer_id]
        with torch.no_grad():
            if self.predictor == "llapor":
                scores = self._scorers[target](
                    self._store.view(layer_id, RouteFeature.ROUTER_INPUT, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_IDS, rows),
                    self._store.view(layer_id, RouteFeature.TOPK_WEIGHTS, rows),
                )
            else:
                scores = self._scorers[target](self._store.view(layer_id, RouteFeature.PRE_MIXER, rows))
        if self._serving_top1 and self.puller is not None:
            # The real CUDA kernel writes the pull plan directly.  This branch
            # is selected only when no reference-bank consumer exists; on an
            # unsupported shape/dtype the puller executes the exact fallback.
            self.puller.select_and_post_target(target, scores)
            return
        # Passing expert_to_slot here (rather than the prior zero-arg call) is the seam
        # documented on BudgetRecall: bank.write now excludes residency before truncating
        # to width, so BudgetRecall.observe's own residency re-check at :57 sees a bank
        # that is already non-resident-only, not top-W-then-filtered.
        self.bank.write(target, scores, expert_to_slot=self._hot_caches[target].expert_to_slot)
        if self.calibration is not None:
            candidates = self.bank.ids_for(target)
            ordered_scores = self.bank.scores_for(target) / rows
            self.calibration.stage(
                target,
                candidates[:1],
                ordered_scores[:1],
                ordered_scores[:1] - ordered_scores[1:2] if self.bank.width > 1 else ordered_scores[:1],
                self.bank.valid_for(target)[:1]
                & (self._hot_caches[target].expert_to_slot.index_select(0, candidates[:1]) < 0),
            )
        if self.puller is not None:
            self.puller.post_target(target)

    def metrics_record(self) -> dict:
        """Host read of the device counters; call only at metric log intervals."""
        if self.recall is None:
            record = {"predictor": self.predictor, "shadow_recall_enabled": False}
            if self.calibration is not None:
                record["pull_calibration"] = self.calibration.snapshot()
            return record
        layers = {str(layer): {"missed_routes": missed, "covered_routes": covered}
                  for layer, (missed, covered) in self.recall.snapshot().items()}
        missed = sum(v["missed_routes"] for v in layers.values())
        covered = sum(v["covered_routes"] for v in layers.values())
        record = {"predictor": self.predictor, "budget": self.recall.budget,
                "budget_recall": covered / missed if missed else 0.0, "layers": layers,
                "shadow_recall_enabled": True}
        if self.calibration is not None:
            record["pull_calibration"] = self.calibration.snapshot()
        return record

    def write_calibration(self, *, complete: bool = True) -> None:
        if self.calibration is None or self._calibration_file is None:
            return
        provenance = json.loads(self._calibration_provenance or "{}")
        self.calibration.write(self._calibration_file, provenance, complete=complete)
