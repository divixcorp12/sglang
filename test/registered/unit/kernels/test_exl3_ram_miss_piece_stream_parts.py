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

GOLDEN = {
    "one_part": [
        "e222060697b71979b177d07d4f192f0f61f921818e699bbad6b16ad739b57a1a",
        "bb2511479b9a392125c6eadd9b571e46d4d929087f4e18398ed2a5817773af77",
        64,
        16
    ],
    "halves": [
        "cbe140acf895ec5b27dbb49da5415b39314ff7686a85ee9db3165aad0f0f9855",
        "ba78686a5c34f75237609f3f8bdee250d3fb57dcc86eb58a831fdc8671c95db6",
        128,
        32
    ],
    "zero_first_part": [
        "8c69cd450bc2ccb6c1f9d1f3871334dfbb5a52c494aea2a90ce22588959b5f91",
        "a7d1a94384f1df05e47a4066b75b33c1c5fcdad3662e6a5f1d87bd8dd8c23fcb",
        128,
        32
    ],
    "three_to_one": [
        "176c3c4a23c98c1c779d7d411223950cea45290463a045a2c2191be5391bc059",
        "742d8088cb2dc9768118a1e2b6b8c2fe38f78cdb957a49b4f9bd1891251530da",
        128,
        32
    ],
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
