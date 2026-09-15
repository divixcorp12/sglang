"""Streaming dataset loader: joins capture shards to session splits.

Streams only the requested tensor keys per shard via safetensors safe_open so
one layer's (or one adjacent pair's) rows fit in RAM (~3-6 GB) without ever
materializing the full capture. Splits are joined by session, never by row:
a rid is `<session_id>-t<turn>`; the session_id looks up sessions.jsonl's
`split` field.

Deviation from the specs (documented per team-lead instructions): the specs
call for 70/10/10/10 (LLaPor) and 60/10/10/10/10 (APEX) splits. This loader
only has three named splits from the benchmark sessions file -- train, dev,
shifted_test -- and carves APEX's cdf_fit/calibration subsets out of train by
hashing session_id with a fixed seed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import msgspec
import torch
from safetensors import safe_open

from sglang.srt.layers.moe.expert_prediction.capture_reader import read_manifest
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    FORWARD_KIND,
    ROW_FORWARD,
    ROW_REQUEST,
    CaptureKind,
    feature_key,
)
from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature

_RID_PATTERN = re.compile(r"^(.*)-t(\d+)$")
_SPLIT_OF_SESSION_SPLIT = {"train": "train", "val": "dev", "holdout": "shifted_test"}


class SessionSplits(msgspec.Struct, frozen=True):
    split_of_session: dict[str, str]

    def session_id_of_rid(self, rid: str) -> str:
        match = _RID_PATTERN.match(rid)
        if match is None:
            raise ValueError(f"rid {rid!r} does not match '<session_id>-t<turn>'")
        return match.group(1)

    def split_of_rid(self, rid: str) -> str:
        return self.split_of_session[self.session_id_of_rid(rid)]

    def known_session_id_of_rid(self, rid: str) -> str | None:
        """Like session_id_of_rid, but None for a non-benchmark rid (e.g. a
        server HEALTH_CHECK_* probe) instead of raising."""
        match = _RID_PATTERN.match(rid)
        return match.group(1) if match is not None else None


def load_session_splits(sessions_path: Path) -> SessionSplits:
    split_of_session: dict[str, str] = {}
    with open(sessions_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split_of_session[record["session_id"]] = _SPLIT_OF_SESSION_SPLIT[record["split"]]
    return SessionSplits(split_of_session=split_of_session)


def carve_apex_train_subsets(splits: SessionSplits, *, seed: int = 0) -> dict[str, str]:
    """Partition train sessions into ranker_train/cdf_fit/calibration (80/10/10)."""
    subset_of_session: dict[str, str] = {}
    for session_id, split in splits.split_of_session.items():
        if split != "train":
            continue
        digest = hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest()
        bucket = int(digest, 16) % 100
        if bucket < 80:
            subset_of_session[session_id] = "ranker_train"
        elif bucket < 90:
            subset_of_session[session_id] = "cdf_fit"
        else:
            subset_of_session[session_id] = "calibration"
    return subset_of_session


class _ShardEntry(msgspec.Struct, frozen=True):
    name: str
    first_forward: int


def _manifest_entries(capture_dir: Path) -> list[_ShardEntry]:
    return [
        _ShardEntry(name=entry["shard"], first_forward=entry["first_forward"])
        for entry in read_manifest(capture_dir)
    ]


class LayerRows(msgspec.Struct, frozen=True):
    """Rows for one source layer, optionally joined to a next-layer label."""

    features: dict[str, torch.Tensor]
    is_decode: torch.Tensor  # [N] bool
    rids: list[str]  # [N]; resolve to a split/subset via SessionSplits


def load_layer_rows(
    capture_dir: Path,
    splits: SessionSplits,
    *,
    layer_id: int,
    features: Sequence[RouteFeature],
    next_layer_topk: int | None = None,
) -> LayerRows:
    """Stream `features` for `layer_id`, plus `next_layer_topk`'s topk_ids if set."""
    entries = _manifest_entries(capture_dir)
    wanted_keys = {feature_key(layer_id, feature): feature.value for feature in features}
    next_key = None
    if next_layer_topk is not None:
        next_key = feature_key(next_layer_topk, RouteFeature.TOPK_IDS)
        wanted_keys = {**wanted_keys, next_key: "next_topk_ids"}

    parts: dict[str, list[torch.Tensor]] = {name: [] for name in wanted_keys.values()}
    is_decode_parts: list[torch.Tensor] = []
    rid_parts: list[str] = []

    for entry in entries:
        path = capture_dir / entry.name
        with safe_open(str(path), framework="pt") as handle:
            keys = set(handle.keys())
            if not set(wanted_keys) <= keys:
                continue
            row_forward = handle.get_tensor(ROW_FORWARD)
            if row_forward.numel() == 0:
                continue
            row_request = handle.get_tensor(ROW_REQUEST)
            forward_kind = handle.get_tensor(FORWARD_KIND)
            request_ids = json.loads(handle.metadata()["request_ids"])

            local_forward = row_forward - entry.first_forward
            is_decode_parts.append(forward_kind[local_forward] == int(CaptureKind.DECODE))
            rid_parts += [request_ids[i] for i in row_request.tolist()]
            for disk_key, name in wanted_keys.items():
                parts[name].append(handle.get_tensor(disk_key))

    features_out = {name: torch.cat(tensors) for name, tensors in parts.items()}
    for name in ("topk_ids", "next_topk_ids"):
        if name in features_out:
            features_out[name] = features_out[name].long()
    return LayerRows(
        features=features_out,
        is_decode=torch.cat(is_decode_parts) if is_decode_parts else torch.empty(0, dtype=torch.bool),
        rids=rid_parts,
    )


def _known_split_of_rid(splits: SessionSplits, rid: str) -> str | None:
    """None for rows outside the benchmark (e.g. server HEALTH_CHECK_* probes),
    which never belong to any split."""
    session_id = splits.known_session_id_of_rid(rid)
    return None if session_id is None else splits.split_of_session.get(session_id)


def split_mask(rows: LayerRows, splits: SessionSplits, split: str) -> torch.Tensor:
    return torch.tensor(
        [_known_split_of_rid(splits, rid) == split for rid in rows.rids], dtype=torch.bool
    )


def apex_subset_mask(
    rows: LayerRows, splits: SessionSplits, subset_of_session: dict[str, str], subset: str
) -> torch.Tensor:
    """Mask over train-split rows whose session fell into `subset` (see carve_apex_train_subsets)."""
    return torch.tensor(
        [
            subset_of_session.get(splits.known_session_id_of_rid(rid)) == subset
            for rid in rows.rids
        ],
        dtype=torch.bool,
    )
