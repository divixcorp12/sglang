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

GOLDEN = {
    "halves/inline": {
        "credit": 32,
        "descriptors": 32,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 22,
        "sqes": "6aec279c012f2ac72b6fe169eb032354f8e7e66e0d2044ef12dfbc76283d8faa",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
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
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "halves/pieces": {
        "credit": 32,
        "descriptors": 128,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 88,
        "sqes": "cccea8f733c92bfc00848f45191e39c1a89258efc0ea6550a2e49ea939171013",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 88,
            "pack_split": 2,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "halves/workers": {
        "credit": 32,
        "descriptors": 32,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 22,
        "sqes": "6aec279c012f2ac72b6fe169eb032354f8e7e66e0d2044ef12dfbc76283d8faa",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 22,
            "pack_split": 3,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
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
    "images_halves/direct_split_traced": {
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
            "pack_split": 3,
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
    "images_one_root/direct_split_traced": {
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
            "pack_split": 3,
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
    },
    "images_three_roots/direct_split_traced": {
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
            "pack_split": 3,
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
    "one_root/inline": {
        "credit": 16,
        "descriptors": 16,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 11,
        "sqes": "21447b39e6c37ea5d4235fdb9503ca1f1e3ea956c7081a650194c2d11758951c",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
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
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "one_root/pieces": {
        "credit": 16,
        "descriptors": 64,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 44,
        "sqes": "c94f70fd591d0edbf99f68cc6fffa6eac830913b677d75a710a55bb11cec0420",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 44,
            "pack_split": 2,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "one_root/workers": {
        "credit": 16,
        "descriptors": 16,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 11,
        "sqes": "21447b39e6c37ea5d4235fdb9503ca1f1e3ea956c7081a650194c2d11758951c",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 11,
            "pack_split": 3,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "three_roots/inline": {
        "credit": 48,
        "descriptors": 48,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 33,
        "sqes": "43f420cc00faa412cbbf1e860b45974cf5f16ffad7d0bb62b67a711daf43ae65",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
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
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "three_roots/pieces": {
        "credit": 48,
        "descriptors": 192,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 66,
        "sqes": "da4164d083d396f1d3f450d5a10de2affec583cc7ab5ddb9bdb08645113fc777",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 66,
            "pack_split": 2,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "three_roots/workers": {
        "credit": 48,
        "descriptors": 48,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 33,
        "sqes": "43f420cc00faa412cbbf1e860b45974cf5f16ffad7d0bb62b67a711daf43ae65",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 33,
            "pack_split": 3,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "three_roots_zero_mid/inline": {
        "credit": 48,
        "descriptors": 48,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 22,
        "sqes": "f0ce673b9917265cf59c27df204f43efd55212306b2daba7fefc4fc5c53dd7f8",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
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
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "three_roots_zero_mid/pieces": {
        "credit": 48,
        "descriptors": 192,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 88,
        "sqes": "fef4b978dff0d4bcfbf4315b7996ae2c2f55ddb126e1d4bba99798815969c00f",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 88,
            "pack_split": 2,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 1,
            "pieces_published": 88,
            "pieces_vetted": 88,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
        }
    },
    "three_roots_zero_mid/workers": {
        "credit": 48,
        "descriptors": 48,
        "result": 1,
        "slabs": "c3c287db18c1ab1a5fbd713b46278036f370e1d01e50286fb3599de15ab1f33a",
        "sqe_count": 22,
        "sqes": "f0ce673b9917265cf59c27df204f43efd55212306b2daba7fefc4fc5c53dd7f8",
        "stage": {
            "batches": 2,
            "bytes": 1713152,
            "cancelled_bytes": 0,
            "extents": 22,
            "pack_split": 3,
            "pack_workers": 2,
            "piece_publish_refused": 0,
            "piece_stream": 0,
            "pieces_published": 0,
            "pieces_vetted": 0,
            "retried_bytes": 0,
            "rows": 0,
            "rows_asked": 11,
            "submitted_bytes": 1724416,
            "useful_bytes": 1672704
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


def _measure(root, weights, images, faults):
    dims = {} if images else dict(hidden=256, inter=512)
    s = ram_miss_setup(root, capacity=12, experts=12, mirror_weights=weights, row_images=images, **dims)
    for slabs in (s.slabs.values() if isinstance(s.slabs, dict) else s.slabs):
        for t in (slabs.values() if isinstance(slabs, dict) else [slabs]):
            t.view(torch.uint8).fill_(0x5A)  # a fixed prior content, so unread bytes are pinned too
    result, log, info, record = read_rows_sqes(s.tables, 1, EXPERTS, SLOTS, direct=images, **faults)
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
