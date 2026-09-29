"""Piece streaming over N mirror parts (docs/superpowers/plans/2026-09-28-mirror3-piece-stream.md).

GOLDEN pins the piece geometry of 1- and 2-part row-image rows to what the C++ produced at
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

from sglang.kernels.ops.moe.expert_stream_transport import piece_geometry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# (id, mirror_weights): every 1- and 2-root shape production or the suites run. The packed path's shard shapes
# (one_part, halves, zero_first_part, three_to_one), whose SQE digest was the bounce read's, were deleted with the
# packed path (plan 2026-09-29-hotpath-zero-overhead D4, Task 6); the remaining entries are unedited.
SHAPES = [
    ("images_one_root", None),
    ("images_halves", (1.0, 1.0)),
    ("images_zero_second", (1.0, 0.0)),
]

GOLDEN = {
    "images_one_root": [
        "e7b4c9bd3f235e5dbc4447994bac0dfcfc2a2e2fd0458004c0698f8a49519b7a"
    ],
    "images_halves": [
        "5b16a9887ade9567a76d00ab2eb6f88dcbfed1f229fa51169424571768c787e8"
    ],
    "images_zero_second": [
        "0e7b3ac6f4e3c83ba9c91b89944b59e727053afe452d7e45e4e3f0b33e6520d3"
    ]
}


def _geometry_digest(tables):
    h = hashlib.sha256()
    layers, experts = tables.extents.shape[:2]
    for row in range(layers):
        for expert in range(experts):
            h.update(json.dumps(piece_geometry(tables, row, expert), sort_keys=True).encode())
    return h.hexdigest()


def _measure(tmp_path, weights):
    """The geometry digest. The direct mode's SQEs depend on whether the filesystem takes O_DIRECT, so only its
    geometry is pinned."""
    s = ram_miss_setup(tmp_path, capacity=12, experts=12, mirror_weights=weights)
    return [_geometry_digest(s.tables)]


@pytest.mark.parametrize("name, weights", SHAPES, ids=[shape[0] for shape in SHAPES])
def test_one_and_two_part_rows_are_cut_and_read_as_at_the_base_commit(tmp_path, name, weights):
    assert _measure(tmp_path, weights) == GOLDEN[name]


if __name__ == "__main__":
    golden = {}
    for name, weights in SHAPES:
        with tempfile.TemporaryDirectory(dir=sys.argv[1] if len(sys.argv) > 1 else None) as d:
            path = pathlib.Path(d) / "ckpt"
            path.mkdir()
            golden[name] = _measure(path, weights)
    print("GOLDEN = " + json.dumps(golden, indent=4))
