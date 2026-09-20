"""Expert rows for the streaming framework, read from K mirrored EXL3 checkpoints.

Every root holds a byte-identical copy of the checkpoint, each on its own
drive. One expert row is still one page-aligned superset read into the shared
bounce ring, but ``Exl3RowReader.read_split`` serves it from all the roots at
once: one page-aligned sub-range per root, in a single submit, as the split
policy plans. Everything after the read -- the bounce ring, the per-name split
into destination rows, the byte accounting -- is ``Exl3ShardRowSource``'s, so
``ExpertStreamer`` cannot tell the two sources apart.
"""

from __future__ import annotations

import os
from typing import Sequence

from sglang.srt.layers.moe.exl3_expert_format import RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout
from sglang.srt.layers.moe.exl3_read_split import SplitPolicy
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.exl3_shard_row_source import (
    BOUNCE_ROWS,
    Exl3ShardRowSource,
    shared_row_reader,
)
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES


class Exl3MirrorRowSource(Exl3ShardRowSource):
    """An ``ExpertRowSource`` (``PER_NAME`` host layout) over one layer's shard
    rows, each read from ``roots`` at once.

    ``reader.source_root`` must be the directory the layout was built from: a
    record's path relative to it picks the file inside each root, so the roots
    may have different absolute prefixes.
    """

    def __init__(
        self,
        reader: Exl3RowReader,
        layer_id: int,
        segments: Sequence[RowSegment],
        *,
        roots: Sequence[str],
        policy: SplitPolicy,
        bounce_rows: int = BOUNCE_ROWS,
    ) -> None:
        roots = tuple(roots)
        if not roots:
            raise ValueError("a mirror row source needs at least one root")
        if reader.source_root is None:
            raise ValueError(
                "a mirror row source needs a reader built with source_root, the "
                "directory the layout's paths live under"
            )
        planned = len(policy.plan(PAGE_BYTES).part_bytes)
        if planned != len(roots):
            raise ValueError(
                f"split policy plans {planned} parts for {len(roots)} roots"
            )
        super().__init__(reader, layer_id, segments, bounce_rows=bounce_rows)
        self.roots = roots
        self.policy = policy
        # Fail here, naming the root and the file, rather than on some later
        # read. Whether a copy is complete is the reader's size check, on the
        # first read of each file.
        layer_paths = {
            record.path
            for (layer, _expert), record in reader.layout.records.items()
            if layer == layer_id
        }
        for path in sorted(layer_paths):
            for root in roots:
                mirror = reader._mirror_path(root, path)
                if not os.path.isfile(mirror):
                    raise FileNotFoundError(
                        f"mirror root {root} has no copy of {path}: {mirror} "
                        "does not exist"
                    )

    @classmethod
    def for_layer(
        cls,
        layout: Exl3ExpertLayout,
        layer_id: int,
        segments: Sequence[RowSegment],
        *,
        direct: bool,
        roots: Sequence[str],
        policy: SplitPolicy,
        source_root: str,
    ) -> "Exl3MirrorRowSource":
        return cls(
            shared_row_reader(layout, direct, source_root),
            layer_id,
            segments,
            roots=roots,
            policy=policy,
        )

    def _read_rows(self, experts: Sequence[int], addresses: Sequence[int]) -> list[int]:
        return self.reader.read_split(
            [(self.layer_id, expert) for expert in experts],
            addresses,
            roots=self.roots,
            policy=self.policy,
        )
