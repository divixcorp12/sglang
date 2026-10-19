from __future__ import annotations

import time
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
    - ``target_entropy``, ``target_top1`` [n, gamma + 1]: the target's sampling distribution per verify row 0..gamma
      (after temperature and top-k/top-p, as AcceptSampling builds it): its entropy in nats and top-1 probability; the
      bonus of a verify that accepted m drafts was drawn from row m. ``target_p`` [n, gamma]: p_k(d_k). NaN for a
      greedy verify or when the caller passed none.
    - ``markov_prev_tokens`` [n, gamma] int64: the tokens the head's Markov embedding reads (the anchor, then drafts
      1..gamma-1); -1 when the caller passed none.
    - ``record_ns`` [n] int64: CLOCK_MONOTONIC when the verify was recorded, to map rows back to the request that was
      being served (a client logging its requests' CLOCK_MONOTONIC spans).
    - ``head_hidden`` [n, gamma, hidden] float16: the confidence head's hidden input, for retraining the head offline.
      Written for at most HEAD_HIDDEN_MAX_VERIFIES verifies per launch, and only in a shard where every verify has it.
    """

    DRAFT_FEATURES = ("draft_q", "draft_top1", "draft_entropy")
    TARGET_FEATURES = ("target_entropy", "target_top1", "target_p")
    # 16,384 verifies x gamma 5 x hidden 5,120 x 2 bytes = 0.78 GiB per launch at most.
    HEAD_HIDDEN_MAX_VERIFIES = 16384

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
                *self.TARGET_FEATURES,
                "markov_prev_tokens",
                "record_ns",
            )
        }
        self._head_hidden: list[Optional[torch.Tensor]] = []
        self.head_hidden_ct = 0
        self._shard_ct = 0

    def records_head_hidden(self) -> bool:
        return self.head_hidden_ct < self.HEAD_HIDDEN_MAX_VERIFIES

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
        target_features: Optional[dict[str, torch.Tensor]] = None,
        head_hidden: Optional[torch.Tensor] = None,
        markov_prev_tokens: Optional[torch.Tensor] = None,
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
        for key in self.TARGET_FEATURES:
            width = self.gamma if key == "target_p" else self.gamma + 1
            self._buffers[key].append(
                torch.full((bs, width), float("nan"))
                if target_features is None
                else target_features[key].detach().to(device="cpu", dtype=torch.float32)
            )
        self._buffers["markov_prev_tokens"].append(
            torch.full((bs, self.gamma), -1, dtype=torch.int64)
            if markov_prev_tokens is None
            else markov_prev_tokens.detach().to(device="cpu", dtype=torch.int64)
        )
        self._buffers["record_ns"].append(torch.full((bs,), time.monotonic_ns(), dtype=torch.int64))
        if head_hidden is not None and self.records_head_hidden():
            self._head_hidden.append(head_hidden.detach().to(device="cpu", dtype=torch.float16))
            self.head_hidden_ct += bs
        else:
            self._head_hidden.append(None)
        if len(self._buffers["logits"]) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffers["logits"]:
            return
        shard_path = Path(f"{self.path_stem}.{self._shard_ct}.pt")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard = {key: torch.cat(values, dim=0) for key, values in self._buffers.items()}
        if self._head_hidden and all(h is not None for h in self._head_hidden):
            shard["head_hidden"] = torch.cat(self._head_hidden, dim=0)
        torch.save(shard, shard_path)
        for values in self._buffers.values():
            values.clear()
        self._head_hidden.clear()
        self._shard_ct += 1

    def close(self) -> None:
        """Writes the verifies still buffered as a last, shorter shard (same name and format). Writes nothing when none
        are pending, so a second call, or one after an exact multiple of ``flush_every``, adds no file."""
        self.flush()
