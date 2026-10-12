"""R1 stamp join: each B0 residency record's prediction-to-demand window per (seq, source layer, horizon).

The window runs from the snapshot stamp of the source row to the label stamp of the target row of the horizon, both
%globaltimer ns. Records with no valid target, an unstamped side or a demand before its snapshot are counted in the
audit and get no window. Capture tools are only read.

Inputs: a block dir (``router.res.*`` files) or a corpus dir holding ``blocks/<block-id>/``.
Usage: from r1_stamps import block_windows, corpus_windows
"""

from __future__ import annotations

import os
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)

from sglang.srt.layers.moe.residency_capture import load_residency  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import HORIZONS, horizon_targets  # noqa: E402

RESIDENCY_PREFIX = "router.res"
BLOCKS_DIR = "blocks"


class Window(msgspec.Struct, frozen=True):
    snap_ns: int
    demand_ns: int

    @property
    def window_ms(self) -> float:
        return (self.demand_ns - self.snap_ns) / 1e6


class StampAudit(msgspec.Struct):
    """Requested (record, source row, horizon) cells: joined + missing + inversions + nonpositive."""

    joined: int = 0
    missing: int = 0  # no target row, or the record's valid flag is clear
    inversions: int = 0  # demand stamp earlier than the snapshot stamp
    nonpositive: int = 0  # a stamp is zero (unstamped)

    def add(self, other: "StampAudit") -> None:
        self.joined += other.joined
        self.missing += other.missing
        self.inversions += other.inversions
        self.nonpositive += other.nonpositive


def block_windows(block_dir: str, *, horizons=(1, 2, 4)) -> tuple[dict[tuple[int, int, int], Window], StampAudit]:
    res = load_residency(os.path.join(block_dir, RESIDENCY_PREFIX))
    layer_ids = [row.layer_id for row in res.rows]
    targets, _ = horizon_targets(layer_ids)
    seqs = res.keys[:, 0]
    snap = res.snap_stamp.astype("uint64")
    label = res.label_stamp.astype("uint64")
    windows: dict[tuple[int, int, int], Window] = {}
    audit = StampAudit()
    for source, layer in enumerate(layer_ids):
        for h in horizons:
            assert 1 <= h <= HORIZONS, h
            target = targets[source][h - 1]
            if target < 0:
                audit.missing += len(seqs)
                continue
            valid = res.valid[:, source, h - 1].astype(bool)
            audit.missing += int((~valid).sum())
            for q in valid.nonzero()[0]:
                s, d = int(snap[q, source]), int(label[q, target])
                if s == 0 or d == 0:
                    audit.nonpositive += 1
                elif d < s:
                    audit.inversions += 1
                else:
                    audit.joined += 1
                    windows[(int(seqs[q]), layer, h)] = Window(snap_ns=s, demand_ns=d)
    return windows, audit


def corpus_windows(corpus_dir: str, *, blocks: list[str], horizons=(1, 2, 4)
                   ) -> tuple[dict[tuple[str, int, int, int], Window], StampAudit]:
    out: dict[tuple[str, int, int, int], Window] = {}
    audit = StampAudit()
    for block in blocks:
        windows, one = block_windows(os.path.join(corpus_dir, BLOCKS_DIR, block), horizons=horizons)
        out.update({(block, *key): w for key, w in windows.items()})
        audit.add(one)
    return out, audit
