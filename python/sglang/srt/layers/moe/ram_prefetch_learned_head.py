"""The learned L+2 head file (safetensors) and its serving loader.

One file holds a folded head: ``spec_head.SpecHeadWeights`` tensors (bf16 matrices, fp32 biases and
tables), and one JSON metadata entry (``HeadMeta``). The export script writes it with
``write_head_file``; serving reads it with ``read_head_file`` only (no pickle, no torch.load).
"""

from __future__ import annotations

from typing import Any, Sequence

import msgspec
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sglang.kernels.ops.moe import spec_head

HEAD_FORMAT_VERSION = 1
HEAD_HORIZON = 2
_META_KEY = "head"

# l2_fast_kernels.MATRIX_FIELDS: bf16 in the file; every other tensor is fp32.
MATRIX_FIELDS = frozenset(
    {"w_pca", "w_in_pca", "w_in_sparse", "w_r1", "w_r2", "w_gate", "w_out", "agg_w1", "agg_w2"}
)
TENSOR_NAMES = tuple(name for name in spec_head.SpecHeadWeights.__struct_fields__ if name != "rows")
_AGG_NAMES = ("agg_w1", "agg_row_bias", "agg_w2", "agg_b2")


class HeadRow(msgspec.Struct, frozen=True):
    """One bias-table row: the (source, target) model layer pair it scores and its stratum."""

    source_layer: int
    target_layer: int
    stratum: int


class HeadMeta(msgspec.Struct, frozen=True):
    format_version: int
    checkpoint_sha256: str
    threshold: float
    threshold_source: dict[str, Any]
    horizon: int
    rows: tuple[HeadRow, ...]


class LearnedHeadFile(msgspec.Struct, frozen=True):
    """A loaded file; ``row_of_target`` maps a prefetch source map row (one with a target) to its head row, and
    ``head_row_by_target`` maps that source's target map row (the device's ``entry.target``) to the same head row."""

    path: str
    tensors: dict[str, torch.Tensor]
    meta: HeadMeta
    row_of_target: dict[int, int]
    head_row_by_target: dict[int, int]


class ServingHead(msgspec.Struct, frozen=True):
    """Device state of one launch's head.

    ``src_w`` is fp32 [HEAD_MAX_TOKENS, top_k] staging for the routing weights and ``positions`` is
    int64 [HEAD_MAX_TOKENS] (position 0 is the anchor); ``threshold`` is fp32 [1]. ``row_of_target`` maps a
    target map row (the device's ``entry.target``) to its head row.
    """

    weights: spec_head.SpecHeadWeights
    workspace: spec_head.SpecHeadWorkspace
    threshold: torch.Tensor
    positions: torch.Tensor
    src_w: torch.Tensor
    row_of_target: dict[int, int]
    meta: HeadMeta


def write_head_file(path: str, *, tensors: dict[str, torch.Tensor], meta: HeadMeta) -> None:
    """Write ``tensors`` (named by ``SpecHeadWeights`` fields) with ``meta`` as one JSON metadata entry."""
    contiguous = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(contiguous, path, metadata={_META_KEY: msgspec.json.encode(meta).decode()})


def _read_meta(path: str) -> HeadMeta:
    try:
        with safe_open(path, framework="pt") as f:
            raw = (f.metadata() or {}).get(_META_KEY)
    except Exception as e:
        raise ValueError(f"learned head file {path!r} is missing or unreadable: {e}") from e
    if raw is None:
        raise ValueError(f"learned head file {path!r} has no {_META_KEY!r} metadata entry")
    try:
        return msgspec.json.decode(raw, type=HeadMeta)
    except msgspec.DecodeError as e:
        raise ValueError(f"learned head file {path!r} has malformed metadata: {e}") from e


def _check_tensors(path: str, tensors: dict[str, torch.Tensor], meta: HeadMeta) -> None:
    for name in TENSOR_NAMES:
        if name not in tensors:
            raise ValueError(f"learned head file {path!r} lacks tensor {name!r}")
        want = torch.bfloat16 if name in MATRIX_FIELDS else torch.float32
        if tensors[name].dtype != want:
            raise ValueError(f"learned head file {path!r}: {name} is {tensors[name].dtype}, expected {want}")
    if any(tensors[name].numel() == 0 for name in _AGG_NAMES):
        raise ValueError(f"learned head file {path!r} has no aggregator (agg_* tensors); the T3 head needs it")
    if tensors["row_bias"].shape[0] != len(meta.rows):
        raise ValueError(
            f"learned head file {path!r}: row_bias has {tensors['row_bias'].shape[0]} rows, "
            f"metadata lists {len(meta.rows)}"
        )


def read_head_file(path: str) -> tuple[dict[str, torch.Tensor], HeadMeta]:
    """Load and validate a head file; ValueError names the file and the first defect."""
    meta = _read_meta(path)
    if meta.format_version != HEAD_FORMAT_VERSION:
        raise ValueError(
            f"learned head file {path!r} has format_version {meta.format_version}, "
            f"this build reads {HEAD_FORMAT_VERSION}"
        )
    if meta.horizon != HEAD_HORIZON:
        raise ValueError(f"learned head file {path!r} has horizon {meta.horizon}, expected {HEAD_HORIZON}")
    try:
        tensors = load_file(path)
    except Exception as e:
        raise ValueError(f"learned head file {path!r} is unreadable: {e}") from e
    _check_tensors(path, tensors, meta)
    return tensors, meta


def load_learned_head(
    path: str, *, layer_ids: Sequence[int], targets: torch.Tensor
) -> LearnedHeadFile:
    """Read ``path`` and map every source map row with a target to its head row by (source, target) layer pair.

    ``targets`` is ram_prefetch's int64 [rows, 2] table (target row, gate index) or (-1, -1);
    ``layer_ids[row]`` is the model layer of a map row. Raises ValueError naming uncovered pairs, or when two source
    rows share one target map row (the device keys the head row by target).
    """
    tensors, meta = read_head_file(path)
    head_row = {(r.source_layer, r.target_layer): i for i, r in enumerate(meta.rows)}
    row_of_target: dict[int, int] = {}
    head_row_by_target: dict[int, int] = {}
    uncovered: list[tuple[int, int]] = []
    for row, target_row in enumerate(targets[:, 0].tolist()):
        if target_row < 0:
            continue
        pair = (layer_ids[row], layer_ids[target_row])
        if pair not in head_row:
            uncovered.append(pair)
            continue
        if target_row in head_row_by_target:
            raise ValueError(f"learned head file {path!r}: two source rows share prefetch target map row {target_row}")
        row_of_target[row] = head_row_by_target[target_row] = head_row[pair]
    if uncovered:
        raise ValueError(
            f"learned head file {path!r} has no row for (source layer, target layer) pairs {uncovered}"
        )
    return LearnedHeadFile(
        path=path, tensors=tensors, meta=meta, row_of_target=row_of_target, head_row_by_target=head_row_by_target
    )


def serving_head(head: LearnedHeadFile, *, device: torch.device | str, top_k: int) -> ServingHead:
    """Upload ``head`` and allocate the launch's workspace and static input buffers on ``device``."""
    t = head.tensors
    weights = spec_head.SpecHeadWeights(
        rows=len(head.meta.rows),
        **{name: t[name].to(device=device).contiguous() for name in TENSOR_NAMES},
    )
    workspace = spec_head.new_workspace(
        device=device,
        pca_rank=t["w_pca"].shape[0],
        hidden=t["b_in"].shape[0],
        experts=t["w_out"].shape[0],
    )
    return ServingHead(
        weights=weights,
        workspace=workspace,
        threshold=torch.tensor([head.meta.threshold], dtype=torch.float32, device=device),
        positions=torch.arange(spec_head.HEAD_MAX_TOKENS, dtype=torch.int64, device=device),
        src_w=torch.zeros(spec_head.HEAD_MAX_TOKENS, top_k, dtype=torch.float32, device=device),
        row_of_target=dict(head.head_row_by_target),
        meta=head.meta,
    )


def describe(head: LearnedHeadFile) -> str:
    """The startup log text: file, checkpoint sha, threshold and that the learned scorer is active."""
    meta = head.meta
    return (
        f"RAM prefetch learned L+2 scorer active: file={head.path} "
        f"checkpoint_sha256={meta.checkpoint_sha256} threshold={meta.threshold!r} rows={len(meta.rows)}"
    )
