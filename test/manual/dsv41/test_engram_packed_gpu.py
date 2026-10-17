"""Captured Engram lookups over a page-packed table replay the split table's rows bit for bit (GPU).

Both decode routes: the device-wait post/wait (no host node) and the host-node lookup. The packed files are made
from the split shard by scripts/dsv41/engram_relayout.py, one per layer, with the real 264 B row geometry.
"""

import importlib.util
import json
import os
import struct
import types

import pytest
import torch

from sglang.srt.layers import engram
from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers.engram_host_node import native_engram_host_node
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA CUDA device")

ROWS, DIM, BLOCK = 100, 256, 32
ROW_BYTES = DIM + DIM // BLOCK
ID_SETS = (([3, 0, 49], [3, 1, 3]), ([2, 10, 48], [4, 4, 2]), ([99, 98, 0], [14, 15, 16]), ([95, 90, 89], [99, 0, 7]))

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41", "engram_relayout.py")
_spec = importlib.util.spec_from_file_location("engram_relayout", _PATH)
relayout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relayout)


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
def dirs(tmp_path, monkeypatch):
    torch.manual_seed(0)
    split, packed = tmp_path / "split", tmp_path / "packed"
    split.mkdir()
    packed.mkdir()
    tensors = {"layers.1.engram.test_padding": ("U8", torch.zeros(7, dtype=torch.uint8))}
    for layer, (mean, lo, hi) in ((1, (0, 118, 130)), (14, (5, 110, 117))):
        tensors[f"layers.{layer}.engram.embed.weight"] = ("F8_E4M3", (torch.randn(ROWS, DIM) * 4 + mean).to(torch.float8_e4m3fn))
        tensors[f"layers.{layer}.engram.embed.scale"] = (
            "F8_E8M0", torch.randint(lo, hi, (ROWS, DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        )
    shard = split / "model-00047-of-00048.safetensors"
    _write(shard, tensors)
    monkeypatch.setattr(relayout, "capture_running", lambda: [])
    for layer in (1, 14):
        out = packed / f"engram-layer{layer}-packed.safetensors"
        assert relayout.main([
            "--source", str(shard), "--layer", str(layer), "--out", str(out), "--same-drive", "--lock", "",
            "--hash-threads", "2", "--chunk-pages", "2",
        ]) == 0
    return split, packed


def _layers(directory, store):
    out = []
    for layer_id in (1, 14):
        table = EngramFileTable.open(str(directory), layer_id=layer_id, num_embeddings=ROWS, dim=DIM)
        # A private store: the shared one keeps rows cached by (layer, id) across tables that differ.
        table._native_store = store
        out.append(types.SimpleNamespace(
            file_table=table, dim=DIM, layer_id=layer_id, tp_size=1, _posted_lookup=None, verify_dead_rows=None
        ))
    return out


def _replay(directory, monkeypatch, device_wait: bool) -> list:
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING", "1")
    monkeypatch.setenv("SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT", "1" if device_wait else "0")
    native = native_engram_host_node()
    native.get_shared_store(1 << 20, ROW_BYTES)
    layer1, layer14 = _layers(directory, native.create_store(64 * ROW_BYTES, ROW_BYTES))
    batch = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))
    ids = torch.tensor([[[3, 0, 49], [7, 1, 7]]], device="cuda", dtype=torch.int64)
    graph = BreakableCUDAGraph()
    with torch.cuda.stream(torch.cuda.Stream()):
        with BreakableCUDAGraphCapture(graph, forbid_host_nodes=device_wait):
            out1 = engram.EngramEmbedding.forward(layer1, ids[:, 0], batch)
            out14 = engram.EngramEmbedding.forward(layer14, ids[:, 1], batch)
    results = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for ids1, ids14 in ID_SETS:
            ids.copy_(torch.tensor([[ids1, ids14]], device="cuda"))
            graph.replay()
            stream.synchronize()
            results.append((out1.cpu().clone(), out14.cpu().clone()))
    for table in (layer1.file_table, layer14.file_table):
        assert table.rows_per_page == (15 if directory.name == "packed" else 0)
    return results


@pytest.mark.parametrize("device_wait", [True, False])
def test_captured_packed_lookups_replay_the_split_tables_rows(dirs, monkeypatch, device_wait):
    split, packed = dirs
    want, got = _replay(split, monkeypatch, device_wait), _replay(packed, monkeypatch, device_wait)
    for step, ((w1, w14), (g1, g14)) in enumerate(zip(want, got)):
        assert torch.equal(w1, g1) and torch.equal(w14, g14), f"step {step}"
        assert w1.abs().sum() > 0 and w14.abs().sum() > 0
