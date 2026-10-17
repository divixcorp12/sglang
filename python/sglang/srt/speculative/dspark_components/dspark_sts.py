from __future__ import annotations

from pathlib import Path
from typing import Optional

import msgspec
import torch


class DSparkStsCalibration(msgspec.Struct, frozen=True, omit_defaults=True):
    temperatures: list[float]
    dataset: str = ""
    num_samples: int = 0
    ece_before: list[float] = []
    ece_after: list[float] = []

    def __post_init__(self) -> None:
        if not self.temperatures:
            raise ValueError("DSparkStsCalibration requires at least one temperature.")
        for temperature in self.temperatures:
            if temperature <= 0:
                raise ValueError(
                    "DSparkStsCalibration temperatures must all be > 0, got "
                    f"{self.temperatures}."
                )

    def to_json(self) -> str:
        return msgspec.json.encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, data: str) -> DSparkStsCalibration:
        return msgspec.json.decode(data.encode("utf-8"), type=cls)


def load_sts_calibration_from_path(path: str) -> DSparkStsCalibration:
    with open(path, "r", encoding="utf-8") as f:
        return DSparkStsCalibration.from_json(f.read())


class StsDataRecorder:
    """Shards of a collection launch's verifies, ``<path_stem>.<n>.pt``, each a dict of [n, gamma] tensors (and [n]
    temperatures):

    - ``logits``: the confidence head's pre-STS logits.
    - ``prefix_mask``: draft k matched the target's argmax and so did every draft before it (the greedy label).
    - ``sampled_prefix_mask``: the verify's own accept took draft k (the label under the request's sampling: for a
      sampled request, the rejection-sampling outcome; for a greedy one, ``prefix_mask``).
    - ``accept_rates``: draft k's acceptance probability given the drafts before it were accepted, min(1, p/q) for a
      sampled request (1 or 0 for a greedy one), so cumprod over k is the expected ``sampled_prefix_mask``. NaN when
      the caller passed none.
    - ``temperatures``: the request's sampling temperature (0 for greedy; NaN when the caller passed none).
    - ``draft_q``, ``draft_top1``, ``draft_entropy``: the draft's step distribution q_k at draft k (the one d_k was
      drawn from): q_k(d_k), max q_k and its entropy in nats. NaN when the caller passed none.
    """

    DRAFT_FEATURES = ("draft_q", "draft_top1", "draft_entropy")

    def __init__(self, *, path_stem: str, gamma: int, flush_every: int) -> None:
        self.path_stem = path_stem
        self.gamma = int(gamma)
        self.flush_every = int(flush_every)
        self._buffers: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "logits",
                "prefix_mask",
                "sampled_prefix_mask",
                "accept_rates",
                "temperatures",
                *self.DRAFT_FEATURES,
            )
        }
        self._shard_ct = 0

    def _prefix_mask(self, counts: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.gamma).view(1, -1)
        counts = counts.detach().to(device="cpu", dtype=torch.int64).view(-1, 1)
        return (positions < counts).to(torch.float32)

    def record(
        self,
        *,
        confidence_raw: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        sampled_num_correct_drafts: Optional[torch.Tensor] = None,
        accept_rates: Optional[torch.Tensor] = None,
        temperatures: Optional[torch.Tensor] = None,
        draft_features: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        logits = confidence_raw.detach().to(device="cpu", dtype=torch.float32)
        bs = logits.shape[0]
        prefix_mask = self._prefix_mask(num_correct_drafts)
        self._buffers["logits"].append(logits)
        self._buffers["prefix_mask"].append(prefix_mask)
        self._buffers["sampled_prefix_mask"].append(
            prefix_mask
            if sampled_num_correct_drafts is None
            else self._prefix_mask(sampled_num_correct_drafts)
        )
        self._buffers["accept_rates"].append(
            torch.full((bs, self.gamma), float("nan"))
            if accept_rates is None
            else accept_rates.detach().to(device="cpu", dtype=torch.float32)
        )
        self._buffers["temperatures"].append(
            torch.full((bs,), float("nan"))
            if temperatures is None
            else temperatures.detach().to(device="cpu", dtype=torch.float32).view(-1)
        )
        for key in self.DRAFT_FEATURES:
            self._buffers[key].append(
                torch.full((bs, self.gamma), float("nan"))
                if draft_features is None
                else draft_features[key].detach().to(device="cpu", dtype=torch.float32)
            )
        if len(self._buffers["logits"]) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffers["logits"]:
            return
        shard_path = Path(f"{self.path_stem}.{self._shard_ct}.pt")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: torch.cat(values, dim=0) for key, values in self._buffers.items()},
            shard_path,
        )
        for values in self._buffers.values():
            values.clear()
        self._shard_ct += 1
