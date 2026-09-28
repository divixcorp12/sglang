"""Golden characterization of the expert-stream reader (plan 2026-09-28-reader-crtp-uring-registration Task 1).

Generated at 4fe0c37a41 (origin/master, mirror3 merge) BEFORE the CRTP split. Every later task must keep it green
unedited: it is the proof that the split and the io_uring option layer leave the default reader byte-for-byte what it
was. Per shape and mode it pins the read's result, a digest of every slab byte, the SQE log as a sorted set (the
refill order depends on completion timing, so only the set is deterministic), the descriptor count, the ring credit
and the stage record's deterministic fields. Regenerate only with the user's approval:
    PYTHONPATH=python python test/registered/unit/kernels/test_expert_stream_reader_golden.py [tmp_dir]
"""

import hashlib
import json
import pathlib
import sys
import tempfile

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import read_rows_sqes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# (id, mirror_weights, row_images)
SHAPES = [
    ("one_root", None, False),
    ("halves", (1.0, 1.0), False),
    ("three_roots", (1.0, 1.0, 1.0), False),
    ("three_roots_zero_mid", (1.0, 0.0, 1.0), False),
    ("images_one_root", None, True),
    ("images_halves", (1.0, 1.0), True),
    ("images_three_roots", (1.0, 1.0, 1.0), True),
]
# (id, faults): the bounce path's three packers and the direct path with and without piece streaming.
BOUNCE_MODES = [
    ("inline", {}),
    ("workers", {"pack_workers": 2, "pack_split": 3}),
    ("pieces", {"pack_workers": 2, "piece_stream": True}),
]
IMAGE_MODES = [
    ("direct", {}),
    ("direct_split_traced", {"pack_split": 3}),  # images ignore workers but trace pack_split (old set_pack)
    ("direct_pieces", {"piece_stream": True}),
]
# Stage-record fields that do not depend on time or on completion order.
STAGE_KEYS = [
    "rows", "batches", "bytes", "extents", "rows_asked", "useful_bytes", "submitted_bytes", "retried_bytes",
    "cancelled_bytes", "pack_workers", "pack_split", "piece_stream", "pieces_vetted", "pieces_published",
    "piece_publish_refused",
]
EXPERTS = [10, 3, 7, 0, 11, 5, 1, 8, 2, 9, 4]
SLOTS = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]

GOLDEN = {}


def _slab_digest(obj, h):
    if isinstance(obj, torch.Tensor):
        h.update(obj.contiguous().view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            h.update(str(key).encode())
            _slab_digest(obj[key], h)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _slab_digest(item, h)


def _measure(root, weights, images, faults):
    dims = {} if images else dict(hidden=256, inter=512)
    s = ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)
    for slabs in (s.slabs.values() if isinstance(s.slabs, dict) else s.slabs):
        for t in (slabs.values() if isinstance(slabs, dict) else [slabs]):
            t.view(torch.uint8).fill_(0x5A)  # a fixed prior content, so unread bytes are pinned too
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, direct=False, **faults)
    h = hashlib.sha256()
    _slab_digest(s.slabs, h)
    return {
        "result": result,
        "slabs": h.hexdigest(),
        "sqes": hashlib.sha256(json.dumps(sorted(log)).encode()).hexdigest(),
        "sqe_count": info["sqes"],
        "descriptors": info["descriptors"],
        "credit": info["credit"],
        "stage": {key: int(record[key]) for key in STAGE_KEYS},
    }


def _cases():
    for shape, weights, images in SHAPES:
        for mode, faults in IMAGE_MODES if images else BOUNCE_MODES:
            yield f"{shape}/{mode}", weights, images, faults


@pytest.mark.parametrize("key, weights, images, faults", list(_cases()), ids=[c[0] for c in _cases()])
def test_reader_matches_the_base_commit(tmp_path, key, weights, images, faults):
    root = tmp_path / "ckpt"
    root.mkdir()
    assert _measure(root, weights, images, faults) == GOLDEN[key]


if __name__ == "__main__":
    golden = {}
    for key, weights, images, faults in _cases():
        with tempfile.TemporaryDirectory(dir=sys.argv[1] if len(sys.argv) > 1 else None) as d:
            root = pathlib.Path(d) / "ckpt"
            root.mkdir()
            golden[key] = _measure(root, weights, images, faults)
    print("GOLDEN = " + json.dumps(golden, indent=4, sort_keys=True))
