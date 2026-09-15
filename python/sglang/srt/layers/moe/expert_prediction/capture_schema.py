"""Row identity and tensor names for captured expert-prediction training data."""

from __future__ import annotations

from enum import IntEnum
from typing import Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature

SCHEMA_VERSION = 1

CAPTURE_FEATURES = (
    RouteFeature.PRE_MIXER,
    RouteFeature.ROUTER_INPUT,
    RouteFeature.TOPK_IDS,
    RouteFeature.TOPK_WEIGHTS,
)

ROW_FORWARD = "row.forward_index"
ROW_REQUEST = "row.request_index"
ROW_POSITION = "row.position"
ROW_TOKEN = "row.token_id"
# 0 means the request's earlier tokens were not observed.
ROW_PREFIX_HASH = "row.prefix_hash"
FORWARD_INDEX = "forward.index"
FORWARD_KIND = "forward.kind"
FORWARD_ROWS = "forward.rows"
# int16 [forwards, layers, experts]; -1 marks a non-resident expert.
FORWARD_RESIDENCY = "forward.expert_to_slot"


class CaptureKind(IntEnum):
    PREFILL = 0
    DECODE = 1


class ForwardRecord(msgspec.Struct, frozen=True):
    forward_index: int
    kind: CaptureKind
    rids: tuple[str, ...]
    rows_per_request: tuple[int, ...]


def feature_key(layer_id: int, feature: RouteFeature) -> str:
    return f"layer.{layer_id}.{feature.value}"


def disk_dtype(feature: RouteFeature, hidden_dtype: torch.dtype) -> torch.dtype:
    if feature is RouteFeature.TOPK_IDS:
        return torch.int16
    if feature is RouteFeature.TOPK_WEIGHTS:
        return torch.float32
    return hidden_dtype


def rows_per_request(
    *, is_extend: bool, batch_size: int, extend_seq_lens: Sequence[int] | None
) -> tuple[int, ...]:
    if is_extend:
        return tuple(int(rows) for rows in extend_seq_lens)
    return (1,) * batch_size
