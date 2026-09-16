"""Score predictor candidates against native routes on device and report totals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import torch

COUNTER_NAMES = (
    "rows",
    "routes",
    "hits_at_k",
    "hits_at_m",
    "cold_routes",
    "cold_hits_at_m",
    "cold_candidates",
)


def _membership(candidates: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Bool ``[rows, num_experts]``; out-of-range ids land in a dropped sentinel column."""
    in_range = (candidates >= 0) & (candidates < num_experts)
    columns = torch.where(in_range, candidates, torch.full_like(candidates, num_experts))
    mask = torch.zeros(
        (candidates.shape[0], num_experts + 1), dtype=torch.bool, device=candidates.device
    )
    mask.scatter_(1, columns, True)
    return mask[:, :num_experts]


def score_candidates(
    *,
    candidates: torch.Tensor,
    actual: torch.Tensor,
    top_k: int,
    num_experts: int,
    resident: torch.Tensor | None,
) -> torch.Tensor:
    """Counts ordered as ``COUNTER_NAMES`` for one layer of one forward, without host syncs.

    A route is cold when ``resident`` marks its expert absent; with no
    residency mask every route is cold.
    """
    valid = (actual >= 0) & (actual < num_experts)
    safe_actual = actual.clamp(0, num_experts - 1)
    hits_m = _membership(candidates, num_experts).gather(1, safe_actual) & valid
    hits_k = _membership(candidates[:, :top_k], num_experts).gather(1, safe_actual) & valid
    candidate_valid = (candidates >= 0) & (candidates < num_experts)
    if resident is None:
        cold = valid
        cold_candidates = candidate_valid
    else:
        cold = valid & ~resident[safe_actual]
        cold_candidates = candidate_valid & ~resident[candidates.clamp(0, num_experts - 1)]
    return torch.stack(
        (
            torch.full((), actual.shape[0], dtype=torch.int64, device=actual.device),
            valid.sum(),
            hits_k.sum(),
            hits_m.sum(),
            cold.sum(),
            (hits_m & cold).sum(),
            cold_candidates.sum(),
        )
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


class ShadowMetrics:
    """Cumulative ``[predictors, layers, counters]`` totals kept on device."""

    def __init__(
        self, *, predictor_names: Sequence[str], layer_ids: Sequence[int], device: torch.device
    ) -> None:
        self._predictor_names = tuple(predictor_names)
        self._layer_ids = tuple(layer_ids)
        self._layer_index = {layer_id: i for i, layer_id in enumerate(self._layer_ids)}
        self._totals = torch.zeros(
            (len(self._predictor_names), len(self._layer_ids), len(COUNTER_NAMES)),
            dtype=torch.int64,
            device=device,
        )

    def add(self, *, predictor_index: int, target_layer: int, counts: torch.Tensor) -> None:
        self._totals[predictor_index, self._layer_index[target_layer]].add_(counts)

    def snapshot(self) -> dict:
        """Synchronizes with the device; returns JSON-ready per-layer counters and totals."""
        return self.snapshot_from_host(self._totals.cpu())

    def snapshot_from_host(self, totals: torch.Tensor) -> dict:
        """Format an already-owned CPU copy without a device read or file write.

        ``AsyncTelemetry`` calls this only on its background writer after the
        stream-recorded D2H copy completed.  The explicit ``snapshot`` method
        remains available for diagnostics that intentionally request a sync.
        """
        if totals.device.type != "cpu":
            raise ValueError("ShadowMetrics snapshot_from_host needs CPU-owned totals")
        values = totals.tolist()
        result = {}
        for predictor_index, name in enumerate(self._predictor_names):
            per_layer = values[predictor_index]
            layers = {
                str(layer_id): dict(zip(COUNTER_NAMES, per_layer[i]))
                for i, layer_id in enumerate(self._layer_ids)
                if per_layer[i][0] > 0
            }
            total = dict(zip(COUNTER_NAMES, (sum(column) for column in zip(*per_layer))))
            total["recall_at_k"] = _ratio(total["hits_at_k"], total["routes"])
            total["recall_at_m"] = _ratio(total["hits_at_m"], total["routes"])
            total["cold_recall_at_m"] = _ratio(total["cold_hits_at_m"], total["cold_routes"])
            total["cold_precision_at_m"] = _ratio(
                total["cold_hits_at_m"], total["cold_candidates"]
            )
            result[name] = {"layers": layers, "total": total}
        return result

    def append_jsonl(self, path: Path, *, forwards: int, eligible_forwards: int) -> None:
        self.append_jsonl_from_host(
            path,
            self._totals.cpu(),
            forwards=forwards,
            eligible_forwards=eligible_forwards,
        )

    def append_jsonl_from_host(
        self,
        path: Path,
        totals: torch.Tensor,
        *,
        forwards: int,
        eligible_forwards: int,
        timestamp_ns: int | None = None,
        telemetry: dict[str, int] | None = None,
    ) -> None:
        """Append a record from an owned CPU snapshot on a writer thread."""
        record = {
            "timestamp_ns": time.time_ns() if timestamp_ns is None else timestamp_ns,
            "forwards": forwards,
            "eligible_forwards": eligible_forwards,
            "predictors": self.snapshot_from_host(totals),
        }
        if telemetry is not None:
            record["telemetry"] = telemetry
        with path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, sort_keys=True) + "\n")
