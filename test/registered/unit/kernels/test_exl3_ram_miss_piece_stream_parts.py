"""Piece streaming over N mirror parts (docs/superpowers/plans/2026-09-28-mirror3-piece-stream.md).

GOLDEN pins the geometry and the piece-streaming SQE stream of 1- and 2-part rows to what the C++ produced at
1732aff4ba, before sub-reads per part became a function of the row's reading parts. The independent Python model in
test_exl3_ram_miss_piece_stream is edited by that same change, so it cannot be what shows the change left these rows
alone. The digests hash offsets and lengths only: file contents never enter them.

Regenerate (only at a commit whose 1- and 2-part cut is known good): python test_exl3_ram_miss_piece_stream_parts.py
"""

import hashlib
import json
import pathlib
import sys
import tempfile

import pytest

from sglang.kernels.ops.moe.expert_stream_transport import piece_geometry, read_rows_sqes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# (id, mirror_weights, row_images): every 1- and 2-root shape production or the suites run.
SHAPES = [
    ("one_part", None, False),
    ("halves", (1.0, 1.0), False),
    ("zero_first_part", (0.0, 1.0), False),
    ("three_to_one", (3.0, 1.0), False),
    ("images_one_root", None, True),
    ("images_halves", (1.0, 1.0), True),
    ("images_zero_second", (1.0, 0.0), True),
]

GOLDEN = {}


def _geometry_digest(tables):
    h = hashlib.sha256()
    layers, experts = tables.extents.shape[:2]
    for row in range(layers):
        for expert in range(experts):
            h.update(json.dumps(piece_geometry(tables, row, expert), sort_keys=True).encode())
    return h.hexdigest()


def _sqe_digest(tables):
    """A two-batch piece-streaming read of 11 rows: its SQEs (as a set: the order is the refill's, pinned elsewhere),
    its descriptor count and its credit."""
    experts = list(range(11))[::-1]
    slots = [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4]
    result, log, info, _ = read_rows_sqes(tables, 1, experts, slots, direct=False, piece_stream=True, pack_workers=2)
    assert result == 1
    return [hashlib.sha256(json.dumps(sorted(log)).encode()).hexdigest(), info["descriptors"], info["credit"]]


def _measure(tmp_path, weights, images):
    """Geometry digest, then (bounce path only) the SQE digest, descriptors and credit. The direct mode's SQEs depend
    on whether the filesystem takes O_DIRECT, so only its geometry is pinned."""
    dims = {} if images else dict(hidden=256, inter=512)
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)
    out = [_geometry_digest(s.tables)]
    if not images:
        out += _sqe_digest(s.tables)
    return out


@pytest.mark.parametrize("name, weights, images", SHAPES, ids=[shape[0] for shape in SHAPES])
def test_one_and_two_part_rows_are_cut_and_read_as_at_the_base_commit(tmp_path, name, weights, images):
    assert _measure(tmp_path, weights, images) == GOLDEN[name]


if __name__ == "__main__":
    golden = {}
    for name, weights, images in SHAPES:
        with tempfile.TemporaryDirectory(dir=sys.argv[1] if len(sys.argv) > 1 else None) as d:
            path = pathlib.Path(d) / "ckpt"
            path.mkdir()
            golden[name] = _measure(path, weights, images)
    print("GOLDEN = " + json.dumps(golden, indent=4))
