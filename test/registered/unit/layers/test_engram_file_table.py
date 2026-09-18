"""The memmap Engram table dequantizes exactly like EngramEmbedding's torch path."""

import json
import struct

import pytest
import torch

from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS, DIM, BLOCK = 50, 64, 32


def _write(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, tensor) in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for raw in blobs:
            f.write(raw)


@pytest.fixture
def table(tmp_path):
    torch.manual_seed(0)
    weight = (torch.randn(ROWS, DIM) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 130, (ROWS, DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    _write(
        tmp_path / "model-00047-of-00048.safetensors",
        {
            "layers.1.engram.q_weight": ("BF16", torch.ones(4, 8, dtype=torch.bfloat16)),
            "layers.1.engram.embed.weight": ("F8_E4M3", weight),
            "layers.1.engram.embed.scale": ("F8_E8M0", scale),
        },
    )
    return tmp_path, weight, scale


def test_lookup_matches_reference_dequant(table):
    directory, weight, scale = table
    t = EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS, dim=DIM)
    idx = torch.tensor([[3, 0, 49], [7, 7, 1]])
    got = t.lookup(idx)
    rows = weight[idx].float().unflatten(-1, (-1, BLOCK))
    want = (rows * scale[idx].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16 and got.shape == (2, 3, DIM)
    assert torch.equal(got, want)


def test_missing_layer_raises(table):
    directory, *_ = table
    with pytest.raises(FileNotFoundError, match="layers.14.engram.embed.weight"):
        EngramFileTable.open(str(directory), layer_id=14, num_embeddings=ROWS, dim=DIM)


def test_row_count_mismatch_raises(table):
    directory, *_ = table
    with pytest.raises(ValueError, match="rows"):
        EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS + 1, dim=DIM)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
