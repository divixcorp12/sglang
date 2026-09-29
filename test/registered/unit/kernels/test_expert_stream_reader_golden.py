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

# (id, mirror_weights). The packed path's shard shapes (one_root, halves, three_roots, three_roots_zero_mid) and its
# modes (inline, workers, pieces), and the direct_split_traced mode (a traced pack_split the reader no longer has),
# were deleted with the packed path (plan 2026-09-29-hotpath-zero-overhead D4, Task 6): the remaining entries are
# unedited.
SHAPES = [
    ("images_one_root", None),
    ("images_halves", (1.0, 1.0)),
    ("images_three_roots", (1.0, 1.0, 1.0)),
]
# (id, faults): the direct path with and without piece streaming.
IMAGE_MODES = [
    ("direct", {}),
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

GOLDEN = {
    "images_halves/direct": {
        "credit": 32,
        "descriptors": 32,
        "result": 1,
        "slabs": "61a06a584c5f4f41343e63d88346279c32b263480126661f4aadb9c39f87cba2",
        "sqe_count": 22,
        "sqes": "cc2e527b063f76e3a9697ff5d4b49a10bf8405349b194f605a19a4446501efc4",
        "stage": {
            "batches": 2,
            "bytes": 844800,
            "cancelled_bytes": 0,
            "extents": 22,
            "pack_split": 0,
            "pack_workers": 0,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 844800,
            "useful_bytes": 844800
        }
    },
    "images_halves/direct_pieces": {
        "credit": 32,
        "descriptors": 128,
        "result": 1,
        "slabs": "61a06a584c5f4f41343e63d88346279c32b263480126661f4aadb9c39f87cba2",
        "sqe_count": 77,
        "sqes": "6efe11f38526423f64266f0faadf5eb925f435dcd5f1650b715d5e30fb8310c6",
        "stage": {
            "batches": 2,
            "bytes": 844800,
            "cancelled_bytes": 0,
            "extents": 77,
            "pack_split": 0,
            "pack_workers": 0,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 844800,
            "useful_bytes": 844800
        }
    },
    "images_one_root/direct": {
        "credit": 16,
        "descriptors": 16,
        "result": 1,
        "slabs": "61a06a584c5f4f41343e63d88346279c32b263480126661f4aadb9c39f87cba2",
        "sqe_count": 11,
        "sqes": "376c7eb1872a2b3d481c71d051dbec8c6035cb73b42477867793586ca01365a1",
        "stage": {
            "batches": 2,
            "bytes": 844800,
            "cancelled_bytes": 0,
            "extents": 11,
            "pack_split": 0,
            "pack_workers": 0,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 844800,
            "useful_bytes": 844800
        }
    },
    "images_one_root/direct_pieces": {
        "credit": 16,
        "descriptors": 64,
        "result": 1,
        "slabs": "61a06a584c5f4f41343e63d88346279c32b263480126661f4aadb9c39f87cba2",
        "sqe_count": 44,
        "sqes": "0559971bb24de5a4f139b5548b0f9b07182b7668bab0ec6867b9c852ce17e536",
        "stage": {
            "batches": 2,
            "bytes": 844800,
            "cancelled_bytes": 0,
            "extents": 44,
            "pack_split": 0,
            "pack_workers": 0,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 844800,
            "useful_bytes": 844800
        }
    },
    "images_three_roots/direct": {
        "credit": 48,
        "descriptors": 48,
        "result": 1,
        "slabs": "61a06a584c5f4f41343e63d88346279c32b263480126661f4aadb9c39f87cba2",
        "sqe_count": 33,
        "sqes": "13ea26b82c8af0fbb53fae3ed2f24325d188892fbc1b45145ea0aeea274d4348",
        "stage": {
            "batches": 2,
            "bytes": 844800,
            "cancelled_bytes": 0,
            "extents": 33,
            "pack_split": 0,
            "pack_workers": 0,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 844800,
            "useful_bytes": 844800
        }
    },
    "images_three_roots/direct_pieces": {
        "credit": 48,
        "descriptors": 192,
        "result": 1,
        "slabs": "61a06a584c5f4f41343e63d88346279c32b263480126661f4aadb9c39f87cba2",
        "sqe_count": 66,
        "sqes": "a8499e0ee9e9418e3e086c42d4517ecf99a4d2bbf78005529ced11c9430eb709",
        "stage": {
            "batches": 2,
            "bytes": 844800,
            "cancelled_bytes": 0,
            "extents": 66,
            "pack_split": 0,
            "pack_workers": 0,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 844800,
            "useful_bytes": 844800
        }
    }
}


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


def _measure(root, weights, faults):
    s = ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights)
    for slabs in (s.slabs.values() if isinstance(s.slabs, dict) else s.slabs):
        for t in (slabs.values() if isinstance(slabs, dict) else [slabs]):
            t.view(torch.uint8).fill_(0x5A)  # a fixed prior content, so unread bytes are pinned too
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, **faults)
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
    for shape, weights in SHAPES:
        for mode, faults in IMAGE_MODES:
            yield f"{shape}/{mode}", weights, faults


@pytest.mark.parametrize("key, weights, faults", list(_cases()), ids=[c[0] for c in _cases()])
def test_reader_matches_the_base_commit(tmp_path, key, weights, faults):
    root = tmp_path / "ckpt"
    root.mkdir()
    assert _measure(root, weights, faults) == GOLDEN[key]


if __name__ == "__main__":
    golden = {}
    for key, weights, faults in _cases():
        with tempfile.TemporaryDirectory(dir=sys.argv[1] if len(sys.argv) > 1 else None) as d:
            root = pathlib.Path(d) / "ckpt"
            root.mkdir()
            golden[key] = _measure(root, weights, faults)
    print("GOLDEN = " + json.dumps(golden, indent=4, sort_keys=True))
