"""EngramFileTable and PagedRowSource with real O_DIRECT reads and a non-page-aligned base offset.

The registered tests read with ``direct=False`` because ``tmp_path`` may be tmpfs.
This one needs a disk-backed scratch directory, so it lives under test/manual.
Files are ~350 KB and deleted afterwards.
"""

import json
import os
import shutil
import struct
import tempfile

import numpy as np
import pytest
import torch

from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers.engram_row_cache import EngramRowCache
from sglang.srt.model_loader.file_row_reader import PagedRowSource, shared_uring_file_reader

SCRATCH = os.environ.get(
    "DSV41_DIRECT_SCRATCH",
    "/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-phase3a/scratch-direct",
)
N, DIM, BLOCK = 5000, 64, 32
WEIGHT_KEY, SCALE_KEY = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"


def _fs_type(path):
    mounts = {}
    with open("/proc/mounts") as f:
        for line in f:
            _, mount, fstype, *_ = line.split()
            mounts[mount] = fstype
    path = os.path.realpath(path)
    while path not in mounts:
        path = os.path.dirname(path)
    return mounts[path]


@pytest.fixture
def scratch():
    try:
        os.makedirs(SCRATCH, exist_ok=True)
    except OSError as e:
        pytest.skip(f"scratch dir unavailable: {e}")
    fs = _fs_type(SCRATCH)
    if fs in ("tmpfs", "overlay"):
        pytest.skip(f"{SCRATCH} is on {fs}, which may reject O_DIRECT")
    # A fresh path per test: the shared reader caches file ids by path.
    directory = tempfile.mkdtemp(prefix="run-", dir=SCRATCH)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _write_shard(path):
    rng = np.random.default_rng(1)
    weight = rng.integers(0, 0x70, (N, DIM), dtype=np.uint8)  # finite fp8 values
    scale = rng.integers(120, 130, (N, DIM // BLOCK), dtype=np.uint8)
    header = {
        WEIGHT_KEY: {"dtype": "F8_E4M3", "shape": [N, DIM], "data_offsets": [0, N * DIM]},
        SCALE_KEY: {
            "dtype": "F8_E8M0",
            "shape": [N, DIM // BLOCK],
            "data_offsets": [N * DIM, N * DIM + N * DIM // BLOCK],
        },
    }
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(weight.tobytes())
        f.write(scale.tobytes())
    return 8 + len(blob), weight, scale


def _has_direct_fd(path):
    """True if this process holds an open fd on ``path`` with O_DIRECT set."""
    real = os.path.realpath(path)
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(f"/proc/self/fd/{fd}") != real:
                continue
            with open(f"/proc/self/fdinfo/{fd}") as f:
                flags = int(next(l for l in f if l.startswith("flags:")).split()[1], 8)
        except (OSError, StopIteration):
            continue
        if flags & os.O_DIRECT:
            return True
    return False


def _ids():
    rng = np.random.default_rng(7)
    random_ids = rng.integers(0, N, 400)
    return torch.tensor(np.concatenate([[0, N - 1, 0, N - 1, 17, 17], random_ids]))


def test_paged_row_source_direct_with_unaligned_base_offset(scratch):
    path = os.path.join(scratch, "model-00047-of-00048.safetensors")
    data_start, weight, scale = _write_shard(path)
    assert data_start % 4096 != 0 and os.path.getsize(path) % 4096 != 0
    ids = _ids()
    for rows, base, table in (
        (DIM, data_start, weight),
        (DIM // BLOCK, data_start + N * DIM, scale),
    ):
        source = PagedRowSource(
            shared_uring_file_reader(), path, rows, N, direct=True, base_offset=base
        )
        assert source.direct and source._file != source._buffered_file
        out = torch.empty((ids.numel(), rows), dtype=torch.uint8)
        source.read_rows(ids, out)
        assert np.array_equal(out.numpy(), table[ids.numpy()])
    assert _has_direct_fd(path), "direct=True fell back to a buffered open"


def test_file_table_direct_with_cache_matches_memmap(scratch):
    path = os.path.join(scratch, "model-00047-of-00048.safetensors")
    _write_shard(path)
    plain = EngramFileTable(path, WEIGHT_KEY, SCALE_KEY, N, DIM)
    cache = EngramRowCache(capacity_rows=256, row_bytes=DIM + DIM // BLOCK)
    cached = EngramFileTable(
        path, WEIGHT_KEY, SCALE_KEY, N, DIM, cache=cache, cache_tag=1, direct=True
    )
    assert cached._weight_rows.direct and cached._scale_rows.direct
    assert _has_direct_fd(path), "direct=True fell back to a buffered open"
    for ids in (
        _ids().reshape(-1, 2),
        torch.tensor([[0, N - 1], [N - 1, 0]]),
        torch.tensor([[5, 5, 5], [9, 5, 9]]),
    ):
        assert torch.equal(cached.lookup(ids), plain.lookup(ids))
    hits_before = cache.hits
    repeat = torch.tensor([[5, 9], [0, N - 1]])  # all resident and just touched
    assert torch.equal(cached.lookup(repeat), plain.lookup(repeat))
    assert cache.hits - hits_before == repeat.numel()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
