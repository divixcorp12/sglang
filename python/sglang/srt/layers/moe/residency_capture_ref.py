"""Host reference of the residency capture records; the CUDA snapshot and label kernels transcribe it.

Snapshot record, per (forward, layer row T): HORIZONS x MASK_WORDS uint64 words (bit e of horizon h's 384-bit mask is
set iff ram_slot[target row of horizon h][e] >= 0; all zero when the target is -1), a per-horizon valid flag
(target != -1), and one uint64 %globaltimer stamp.

Label record, per (forward, layer): each route's kind as uint8 in route order, plus one uint64 %globaltimer stamp.
Routes in the plan's lanes carry ram_slot_map.LaneKind (1..5); routes outside them carry KIND_VRAM_HOT.
"""

from __future__ import annotations

from typing import Sequence

HORIZONS = 4
MASK_WORDS = 6
KIND_VRAM_HOT = 0
VALIDITY_RULE = "invalid: no target row"


def horizon_targets(layer_ids: Sequence[int]) -> tuple[list[list[int]], list[list[bool]]]:
    """Per map row T and horizon h in 1..4: the row whose model layer id is layer_ids[T] + h, or -1 when no streamed
    row has it; valid is target != -1."""
    row_of = {layer: row for row, layer in enumerate(layer_ids)}
    targets = [[row_of.get(layer + h, -1) for h in range(1, HORIZONS + 1)] for layer in layer_ids]
    valid = [[t != -1 for t in row] for row in targets]
    return targets, valid


def snapshot_masks(ram_slot: Sequence[Sequence[int]], *, targets: Sequence[int]) -> list[list[int]]:
    """[HORIZONS][MASK_WORDS] uint64 words: the residency mask of each target row, all zero for target -1."""
    masks = []
    for target in targets:
        words = [0] * MASK_WORDS
        if target >= 0:
            for e, slot in enumerate(ram_slot[target]):
                if slot >= 0:
                    words[e // 64] |= 1 << (e % 64)
        masks.append(words)
    return masks


def route_labels(*, routes: Sequence[int], planned: Sequence[int], count: int, lane_kind: Sequence[int]) -> list[int]:
    """Each route's kind in route order: lane_kind of its lane when it is among the first ``count`` planned experts,
    else KIND_VRAM_HOT."""
    lane_of = {e: j for j, e in enumerate(planned[:count])}
    return [int(lane_kind[lane_of[e]]) if e in lane_of else KIND_VRAM_HOT for e in routes]
