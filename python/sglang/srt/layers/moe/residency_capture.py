"""Side files of the residency capture: each graph forward's snapshot and label records.

The record layout is ``residency_capture_ref``'s. One record per forward, in ``seq`` order, appended by
``GraphRouteLog`` beside the RouterCapture files (raw little-endian, numpy structured records):

* ``<prefix>.snap.bin``: per record ``stamp`` uint64 ``[layers]``, ``masks`` uint64 ``[layers, HORIZONS,
  MASK_WORDS]``, ``valid`` uint8 ``[layers, HORIZONS]``.
* ``<prefix>.label.bin``: per record ``stamp`` uint64 ``[layers]``, ``kinds`` uint8 ``[layers, tokens, topk]``.
* ``<prefix>.seq.bin``: int64 ``[records, 3]``, the forward's ``seq``, pass id and live token count, the keys
  of its ``graph_routes`` line and its RouterCapture record.
* ``<prefix>.json``: the schema, shapes, ``VALIDITY_RULE`` and one ``rows`` entry per map row
  (``layer_id``, ``stratum``, ``gate_has_bias``). The record count is the files' size.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

import msgspec
import numpy as np
import torch

from sglang.srt.layers.moe.dsv41_strata import stratum
from sglang.srt.layers.moe.residency_capture_ref import HORIZONS, MASK_WORDS, VALIDITY_RULE


class ResidencyRow(msgspec.Struct, frozen=True):
    """One map row's header entry."""

    layer_id: int
    stratum: str
    gate_has_bias: bool


class ResidencyRecords(msgspec.Struct, frozen=True):
    """A loaded capture; arrays are numpy, first axis the record."""

    header: dict
    rows: list[ResidencyRow]
    validity_rule: str
    masks: Any  # uint64 [records, layers, HORIZONS, MASK_WORDS]
    valid: Any  # bool [records, layers, HORIZONS]
    snap_stamp: Any  # uint64 [records, layers]
    kinds: Any  # uint8 [records, layers, tokens, topk]
    label_stamp: Any  # uint64 [records, layers]
    keys: Any  # int64 [records, 3]: seq, pass id, live tokens


def residency_rows(layer_ids: Sequence[int], *, gates: Mapping[int, Any]) -> list[ResidencyRow]:
    """The header entry of each map row; ``gates`` is ``ram_prefetch.registered_gates()``, and a row without a
    registered gate (or with a hash-routed one, bias None) has no correction bias."""
    rows = []
    for layer in layer_ids:
        gate = gates.get(int(layer))
        rows.append(
            ResidencyRow(
                layer_id=int(layer),
                stratum=stratum(int(layer)),
                gate_has_bias=gate is not None and gate.bias is not None,
            )
        )
    return rows


def _snap_dtype(layers: int) -> np.dtype:
    return np.dtype(
        [
            ("stamp", "<u8", (layers,)),
            ("masks", "<u8", (layers, HORIZONS, MASK_WORDS)),
            ("valid", "u1", (layers, HORIZONS)),
        ]
    )


def _label_dtype(layers: int, tokens: int, topk: int) -> np.dtype:
    return np.dtype([("stamp", "<u8", (layers,)), ("kinds", "u1", (layers, tokens, topk))])


class ResidencyCapture:
    """Appends the residency records of a ``GraphRouteLog``; ``rows`` is one entry per map row."""

    def __init__(self, prefix: str, *, rows: list[ResidencyRow], schema: int) -> None:
        self.prefix = prefix
        self.rows = list(rows)
        self.schema = schema
        self.records = 0
        self._files: Optional[list] = None

    def _open(self, *, run: str, depth: int, tokens: int, topk: int) -> None:
        header = {
            "schema": self.schema,
            "run": run,
            "layer_ids": [row.layer_id for row in self.rows],
            "rows": [msgspec.structs.asdict(row) for row in self.rows],
            "validity_rule": VALIDITY_RULE,
            "horizons": HORIZONS,
            "mask_words": MASK_WORDS,
            "tokens": tokens,
            "topk": topk,
            "depth": depth,
            "snap_fields": ["stamp", "masks", "valid"],
            "label_fields": ["stamp", "kinds"],
        }
        with open(self.prefix + ".json", "w") as f:
            json.dump(header, f)
        self._files = [open(self.prefix + suffix, "ab") for suffix in (".snap.bin", ".label.bin", ".seq.bin")]

    def write(
        self,
        *,
        run: str,
        depth: int,
        keys: tuple[int, int, int],
        masks: torch.Tensor,
        valid: torch.Tensor,
        snap_stamp: torch.Tensor,
        kinds: torch.Tensor,
        label_stamp: torch.Tensor,
    ) -> int:
        """Append one forward's record (one ring slot's host tensors); returns its record number. ``keys`` is
        (seq, pass id, live tokens). The first call writes the ``.json`` header."""
        layers, tokens, topk = (int(n) for n in kinds.shape)
        if self._files is None:
            self._open(run=run, depth=depth, tokens=tokens, topk=topk)
        snap = np.zeros(1, dtype=_snap_dtype(layers))
        snap["stamp"][0] = snap_stamp.numpy().view(np.uint64)
        snap["masks"][0] = masks.numpy().view(np.uint64)
        snap["valid"][0] = valid.numpy()
        label = np.zeros(1, dtype=_label_dtype(layers, tokens, topk))
        label["stamp"][0] = label_stamp.numpy().view(np.uint64)
        label["kinds"][0] = kinds.numpy()
        snap_file, label_file, seq_file = self._files
        snap_file.write(snap.tobytes())
        label_file.write(label.tobytes())
        seq_file.write(np.asarray(keys, dtype="<i8").tobytes())
        self.records += 1
        return self.records - 1

    def flush(self) -> None:
        for f in self._files or []:
            f.flush()


class ResidencyRings:
    """The device rings of the residency records, one slot per ``GraphRouteLog`` slot.

    Sized once, by a warmup forward; ``record`` is captured in the graph and never casts.
    """

    def __init__(self, *, depth: int, layers: int, tokens: int, topk: int, device) -> None:
        self.masks = torch.zeros((depth, layers, HORIZONS, MASK_WORDS), dtype=torch.int64, device=device)
        self.valid = torch.zeros((depth, layers, HORIZONS), dtype=torch.uint8, device=device)
        self.snap_stamp = torch.zeros((depth, layers), dtype=torch.int64, device=device)
        self.kinds = torch.zeros((depth, layers, tokens, topk), dtype=torch.uint8, device=device)
        self.label_stamp = torch.zeros((depth, layers), dtype=torch.int64, device=device)

    def tensors(self) -> list[torch.Tensor]:
        """Ring order, which ``GraphRouteLog._emit`` unpacks."""
        return [self.masks, self.valid, self.snap_stamp, self.kinds, self.label_stamp]

    def record(
        self,
        row: int,
        slot: torch.Tensor,
        *,
        masks: torch.Tensor,
        valid: torch.Tensor,
        snap_stamp: torch.Tensor,
        kinds: torch.Tensor,
        label_stamp: torch.Tensor,
    ) -> None:
        """Copy one layer's records into ``slot``; ``kinds`` is ``[tokens, topk]``, a narrower forward filling its
        first token rows."""
        for name, tensor, dtype in (
            ("masks", masks, torch.int64),
            ("valid", valid, torch.uint8),
            ("snap_stamp", snap_stamp, torch.int64),
            ("kinds", kinds, torch.uint8),
            ("label_stamp", label_stamp, torch.int64),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"residency {name} must be {dtype}, got {tensor.dtype}")
        tokens, topk = int(kinds.shape[0]), int(kinds.shape[-1])
        if kinds.dim() != 2 or tokens > self.kinds.shape[2] or topk != self.kinds.shape[3]:
            raise ValueError(
                f"residency labels hold [{self.kinds.shape[2]}, {self.kinds.shape[3]}], got {tuple(kinds.shape)}"
            )
        self.masks[:, row].index_copy_(0, slot, masks.reshape(1, HORIZONS, MASK_WORDS))
        self.valid[:, row].index_copy_(0, slot, valid.reshape(1, HORIZONS))
        self.snap_stamp[:, row].index_copy_(0, slot, snap_stamp.reshape(1))
        self.kinds[:, row, :tokens].index_copy_(0, slot, kinds.reshape(1, tokens, topk))
        self.label_stamp[:, row].index_copy_(0, slot, label_stamp.reshape(1))


def _parse_rows(header: dict) -> list[ResidencyRow]:
    entries = header.get("rows")
    if not entries:
        raise ValueError("residency header has no per-row entries (layer_id, stratum, gate_has_bias)")
    return [msgspec.convert(entry, type=ResidencyRow) for entry in entries]


def load_residency(prefix: str) -> ResidencyRecords:
    """Read ``prefix``'s side files; raises ValueError on a header without per-row entries."""
    with open(prefix + ".json") as f:
        header = json.load(f)
    rows = _parse_rows(header)
    layers, tokens, topk = len(rows), int(header["tokens"]), int(header["topk"])
    snap = np.fromfile(prefix + ".snap.bin", dtype=_snap_dtype(layers))
    label = np.fromfile(prefix + ".label.bin", dtype=_label_dtype(layers, tokens, topk))
    keys = np.fromfile(prefix + ".seq.bin", dtype="<i8").reshape(-1, 3)
    if not len(snap) == len(label) == len(keys):
        raise ValueError(f"residency files disagree: {len(snap)} snapshots, {len(label)} labels, {len(keys)} keys")
    return ResidencyRecords(
        header=header,
        rows=rows,
        validity_rule=header["validity_rule"],
        masks=snap["masks"],
        valid=snap["valid"].astype(bool),
        snap_stamp=snap["stamp"],
        kinds=label["kinds"],
        label_stamp=label["stamp"],
        keys=keys,
    )
