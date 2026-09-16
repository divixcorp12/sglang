"""Graph-captured score-band accounting for non-timed prefetch profiling."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import torch


class PullCalibrationHistogram:
    """Per-target score and margin histograms with device-only additive counters.

    Candidate state is rewritten in place by the scorer and consumed later by
    the target's graph gather.  The only device-to-host transfer is
    :meth:`snapshot`, called from a normal metrics flush or profiling teardown.
    """

    SCHEMA_VERSION = 2
    BINS = 256
    _FIELDS = (
        "opportunity",
        "source_eligible",
        "target_useful",
        "target_wasted",
        "physical_demand_rows",
    )

    def __init__(self, *, layer_ids: Sequence[int], device: torch.device) -> None:
        ordered = tuple(sorted(layer_ids))
        if not ordered:
            raise ValueError("pull calibration needs at least one target layer")
        self._rows = {layer_id: row for row, layer_id in enumerate(ordered)}
        count = len(ordered)
        self.candidate_ids = torch.full((count,), -1, dtype=torch.int64, device=device)
        self.top_scores = torch.zeros(count, dtype=torch.float32, device=device)
        self.margins = torch.zeros(count, dtype=torch.float32, device=device)
        self.source_eligible = torch.zeros(count, dtype=torch.bool, device=device)
        shape = (count, self.BINS, len(self._FIELDS))
        self._score_counts = torch.zeros(shape, dtype=torch.int64, device=device)
        self._margin_counts = torch.zeros_like(self._score_counts)
        # This is intentionally independent of candidate validity and target
        # residency: calibration must price every scorer/control invocation,
        # not just the subset that became a pull opportunity.
        self._target_observations = torch.zeros(count, dtype=torch.int64, device=device)

    @classmethod
    def bin_index(cls, value: torch.Tensor) -> int:
        """Return the uniform ``[0, 1]`` bin; 1.0 belongs to the final bin."""
        normalized = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        return min(int((normalized * cls.BINS).item()), cls.BINS - 1)

    @classmethod
    def _bin_tensor(cls, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return (normalized * cls.BINS).to(torch.int64).clamp_max_(cls.BINS - 1)

    def stage(
        self,
        target_layer: int,
        candidate_id: torch.Tensor,
        top_score: torch.Tensor,
        margin: torch.Tensor,
        source_eligible: torch.Tensor,
    ) -> None:
        """Stage stable per-target tensors without reading a device value on host."""
        row = self._rows[target_layer]
        self.candidate_ids[row].copy_(candidate_id.reshape(()))
        self.top_scores[row].copy_(top_score.reshape(()))
        self.margins[row].copy_(margin.reshape(()))
        self.source_eligible[row].copy_(source_eligible.reshape(()))
        self._target_observations[row].add_(1)

    def record_target(
        self,
        target_layer: int,
        flat_ids: torch.Tensor,
        missed_mask: torch.Tensor,
        physical_demand_rows: torch.Tensor,
        expert_to_slot: torch.Tensor | None = None,
    ) -> None:
        """Add one target-routing result to both score and margin histograms."""
        row = self._rows.get(target_layer)
        if row is None:
            return
        candidate = self.candidate_ids[row]
        valid = candidate >= 0
        target_nonresident = valid
        if expert_to_slot is not None:
            target_nonresident = valid & (expert_to_slot[candidate.clamp_min(0)] < 0)
        eligible = valid & self.source_eligible[row] & target_nonresident
        useful = (missed_mask & (flat_ids == candidate)).any()
        values = torch.stack(
            (
                eligible.to(torch.int64),
                self.source_eligible[row].to(torch.int64),
                (eligible & useful).to(torch.int64),
                (eligible & ~useful).to(torch.int64),
                eligible.to(torch.int64) * physical_demand_rows.reshape(()).to(torch.int64),
            )
        ).reshape(1, -1)
        self._score_counts[row].index_add_(0, self._bin_tensor(self.top_scores[row]).reshape(1), values)
        self._margin_counts[row].index_add_(0, self._bin_tensor(self.margins[row]).reshape(1), values)

    def reset(self) -> None:
        """Drop graph-capture warmup counts without changing captured addresses."""
        self._score_counts.zero_()
        self._margin_counts.zero_()
        self._target_observations.zero_()

    def snapshot(self) -> dict:
        """Materialize the versioned host contract at a normal metrics boundary."""
        def serialize(counts: torch.Tensor) -> dict[str, list[int]]:
            values = counts.cpu().tolist()
            return {
                field: [int(bin_values[index]) for bin_values in values]
                for index, field in enumerate(self._FIELDS)
            }

        return {
            "schema_version": self.SCHEMA_VERSION,
            "bin_count": self.BINS,
            "range": [0.0, 1.0],
            "bin_edges": "uniform [0,1], lower-inclusive; 1.0 is in bin 255",
            "layers": {
                str(layer_id): {
                    "target_observations": int(self._target_observations[row].item()),
                    "score": serialize(self._score_counts[row]),
                    "margin": serialize(self._margin_counts[row]),
                }
                for layer_id, row in self._rows.items()
            },
        }

    def write(self, path: Path, provenance: dict, *, complete: bool = True) -> None:
        """Atomically publish the profiling artifact at a normal metrics flush."""
        payload = self.snapshot()
        payload["provenance"] = provenance
        payload["complete"] = complete
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)

    @staticmethod
    def require_complete(path: Path) -> dict:
        payload = json.loads(path.read_text())
        if not payload.get("complete", False):
            raise ValueError("incomplete pull calibration artifact is invalid for decisions")
        return payload
