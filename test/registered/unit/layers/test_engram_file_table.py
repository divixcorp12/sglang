"""The memmap Engram table dequantizes exactly like EngramEmbedding's torch path."""

import json
import struct

import numpy as np
import pytest
import torch

from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers import engram
from sglang.srt.layers.engram_host_node import native_engram_host_node
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS, DIM, BLOCK = 100, 64, 32


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
            "layers.1.engram.test_padding": ("U8", torch.zeros(7, dtype=torch.uint8)),
            "layers.1.engram.embed.weight": ("F8_E4M3", weight),
            "layers.1.engram.embed.scale": ("F8_E8M0", scale),
            "layers.14.engram.embed.weight": ("F8_E4M3", weight),
            "layers.14.engram.embed.scale": ("F8_E8M0", scale),
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


def test_native_host_node_is_opt_in_by_default(table, monkeypatch):
    monkeypatch.delenv("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING", raising=False)
    directory, *_ = table
    t = EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS, dim=DIM)
    assert t._host_node_extension is None


def _prime_native_store():
    # Tiny test cache; production open() uses a fixed 5 GiB data budget.
    extension = native_engram_host_node()
    extension.get_shared_store(1 << 20, DIM + DIM // BLOCK)


def _create_native_store(table, capacity_rows=8):
    extension = native_engram_host_node()
    row_bytes = table.dim + table.dim // table.block
    try:
        store = extension.create_store(capacity_rows * row_bytes, row_bytes)
    except RuntimeError as exc:
        if "io_uring_queue_init failed" in str(exc):
            pytest.skip(f"io_uring is unavailable: {exc}")
        raise
    store.register_table(
        table.path, table.weight_offset, table.scale_offset, table.num_embeddings,
        table.dim, table.dim // table.block, table._tag,
    )
    return store


def test_native_store_packed_rows_cold_warm_and_eviction(table):
    directory, *_ = table
    path = str(directory / "model-00047-of-00048.safetensors")
    file_table = EngramFileTable(
        path,
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        ROWS,
        DIM,
    )
    store = _create_native_store(file_table)
    crossing = [
        row
        for row in range(ROWS)
        if (file_table.weight_offset + row * DIM) % 4096 + DIM > 4096
    ]
    assert crossing
    ids = np.array([crossing[0], 3, crossing[0]], dtype=np.int64)
    packed = store.lookup(ids, file_table._tag)
    expected = np.concatenate(
        (file_table.weight[ids], file_table.scale[ids]), axis=1
    )
    assert np.array_equal(packed, expected)
    cold = store.stats()
    assert cold["unique_misses"] == 2
    assert cold["submitted_sqes"] > 0 and cold["completed_cqes"] > 0

    assert np.array_equal(store.lookup(ids, file_table._tag), expected)
    warm = store.stats()
    assert warm["submitted_sqes"] == cold["submitted_sqes"]
    assert warm["hits"] - cold["hits"] == ids.size

    colliding = np.arange(9, dtype=np.int64)
    got = store.lookup(colliding, file_table._tag)
    want = np.concatenate(
        (file_table.weight[colliding], file_table.scale[colliding]), axis=1
    )
    assert np.array_equal(got, want)
    assert store.stats()["evictions"] > 0


# A row per 4 KiB page, so the page count equals the row count and the arithmetic below
# is readable. 4200 rows is 16.4 MiB of weight pages, just past the 16 MiB bounce buffer.
WIDE_DIM, WIDE_ROWS = 4096, 4200


@pytest.fixture
def wide_table(tmp_path):
    torch.manual_seed(0)
    weight = (torch.randn(WIDE_ROWS, WIDE_DIM) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(
        118, 130, (WIDE_ROWS, WIDE_DIM // BLOCK), dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    _write(
        tmp_path / "model-00047-of-00048.safetensors",
        {
            "layers.1.engram.embed.weight": ("F8_E4M3", weight),
            "layers.1.engram.embed.scale": ("F8_E8M0", scale),
        },
    )
    return tmp_path, weight, scale


def test_a_lookup_larger_than_the_bounce_buffer_is_chunked_not_refused(wide_table):
    # Before this was chunked, one request had to fit every miss in the fixed 16 MiB
    # bounce buffer or return -E2BIG. That is a batch-1 decode assumption: eager layer 14
    # drives this same store with prefill-sized batches, and on 2026-09-22 the overflow
    # surfaced as "Engram io_uring lookup failed: -7" and killed the scheduler mid-warmup.
    directory, *_ = wide_table
    file_table = EngramFileTable(
        str(directory / "model-00047-of-00048.safetensors"),
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        WIDE_ROWS,
        WIDE_DIM,
    )
    store = _create_native_store(file_table, capacity_rows=WIDE_ROWS)
    ids = np.arange(WIDE_ROWS, dtype=np.int64)
    assert WIDE_ROWS * WIDE_DIM > (16 << 20), "fixture no longer overflows the buffer"

    packed = store.lookup(ids, file_table._tag)

    expected = np.concatenate((file_table.weight[ids], file_table.scale[ids]), axis=1)
    assert np.array_equal(packed, expected)
    stats = store.stats()
    # The point of the test: it took more than one pass, and every pass stayed in budget.
    assert stats["read_chunks"] > 1, stats["read_chunks"]
    assert stats["bounce_bytes_peak"] <= stats["bounce_bytes_allocated"]
    assert stats["unique_misses"] == WIDE_ROWS


def test_chunking_still_serves_a_request_that_fits_in_one_pass(table):
    # The small path must not regress into needless splitting.
    directory, *_ = table
    file_table = EngramFileTable(
        str(directory / "model-00047-of-00048.safetensors"),
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        ROWS,
        DIM,
    )
    store = _create_native_store(file_table, capacity_rows=ROWS)
    ids = np.arange(8, dtype=np.int64)
    got = store.lookup(ids, file_table._tag)
    assert np.array_equal(
        got, np.concatenate((file_table.weight[ids], file_table.scale[ids]), axis=1)
    )
    assert store.stats()["read_chunks"] == 1


def test_native_host_node_truncated_file_fails_closed(table, tmp_path):
    directory, *_ = table
    path = str(directory / "model-00047-of-00048.safetensors")
    file_table = EngramFileTable(
        path,
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        ROWS,
        DIM,
    )
    truncated_path = tmp_path / "truncated.safetensors"
    cutoff = file_table.scale_offset + DIM // BLOCK - 1
    with open(path, "rb") as source, open(truncated_path, "wb") as destination:
        destination.write(source.read(cutoff))
    extension = native_engram_host_node()
    row_bytes = DIM + DIM // BLOCK
    try:
        store = extension.create_store(8 * row_bytes, row_bytes)
    except RuntimeError as exc:
        if "io_uring_queue_init failed" in str(exc):
            pytest.skip(f"io_uring is unavailable: {exc}")
        raise
    ids = np.array([0], dtype=np.int64)
    rows = np.full((1, row_bytes), 0xA5, dtype=np.uint8)
    status = np.zeros((1,), dtype=np.int32)
    context = extension.EngramHostLookup(
        store,
        str(truncated_path),
        file_table.weight_offset,
        file_table.scale_offset,
        ROWS,
        DIM,
        DIM // BLOCK,
        file_table._tag,
        1,
        ids.ctypes.data,
        rows.ctypes.data,
        status.ctypes.data,
        ids.nbytes + rows.nbytes + status.nbytes,
        rows.nbytes + status.nbytes,
    )
    context.run()
    assert status[0] != 0
    assert not rows.any()
    assert store.stats()["failures"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA CUDA device")
def test_native_callback_failure_clears_staging_rows_and_sets_status(table, monkeypatch):
    _prime_native_store()
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING", "1")
    directory, *_ = table
    t = EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS, dim=DIM)
    ids = torch.tensor([-1], dtype=torch.int64)
    rows = torch.full(
        (1, DIM + DIM // BLOCK), 0xA5, dtype=torch.uint8, pin_memory=True
    )
    status = torch.zeros((1,), dtype=torch.int32, pin_memory=True)
    context = t._host_node_extension.EngramHostLookup(
        t._native_store,
        t.path,
        t.weight_offset,
        t.scale_offset,
        t.num_embeddings,
        t.dim,
        t.dim // t.block,
        t._tag,
        ids.numel(),
        ids.data_ptr(),
        rows.data_ptr(),
        status.data_ptr(),
        8 * ids.numel() + rows.numel() * rows.element_size() + status.element_size(),
        rows.numel() * rows.element_size() + status.element_size(),
    )
    context.run()
    assert status.item() == 2
    assert torch.count_nonzero(rows).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA CUDA device")
def test_ordinary_breakable_graph_can_replay_on_multiple_streams():
    value = torch.tensor([1.0], device="cuda")
    graph = BreakableCUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        with BreakableCUDAGraphCapture(graph):
            output = value + 1

    assert not graph._retained_host_callbacks
    for replay_stream in (torch.cuda.Stream(), torch.cuda.Stream()):
        with torch.cuda.stream(replay_stream):
            graph.replay()
        replay_stream.synchronize()
        assert output.item() == 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA CUDA device")
@pytest.mark.parametrize("deduplicate", [False, True])
def test_layer1_host_node_replays_file_rows_and_layer14_keeps_one_break(
    table, monkeypatch, deduplicate
):
    _prime_native_store()
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING", "1")
    directory, weight, scale = table
    table1 = EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS, dim=DIM)
    table14 = EngramFileTable.open(str(directory), layer_id=14, num_embeddings=ROWS, dim=DIM)
    layer1 = type(
        "Layer", (), {"file_table": table1, "dim": DIM, "layer_id": 1, "tp_size": 1}
    )()
    layer14 = type(
        "Layer", (), {"file_table": table14, "dim": DIM, "layer_id": 14, "tp_size": 1}
    )()
    decode_mode = type("DecodeMode", (), {"is_decode": lambda self: True})()
    batch = type("Batch", (), {"forward_mode": decode_mode})()
    ids1 = torch.tensor([[3, 0, 49]], device="cuda", dtype=torch.int64)
    ids14 = torch.tensor([[7, 1, 7]], device="cuda", dtype=torch.int64)
    registry = DedupedCudaGraphRegistry() if deduplicate else None
    graph = BreakableCUDAGraph(deduped_cuda_graph=registry)
    capture_stream = torch.cuda.Stream()
    replay_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        with BreakableCUDAGraphCapture(graph):
            out1 = engram.EngramEmbedding.forward(layer1, ids1, batch)
            out14 = engram.EngramEmbedding.forward(layer14, ids14, batch)
    assert len(graph._segments) == 2
    assert len(graph._break_fns) == 1

    with torch.cuda.stream(replay_stream):
        before = table1._native_store.stats()
        id_sets = (
            (
                ([3, 0, 49], [7, 1, 7]),
                ([3, 0, 49], [7, 1, 7]),
                ([2, 10, 48], [4, 4, 2]),
                ([49, 6, 0], [1, 3, 8]),
            )
            if not deduplicate
            else (
                ([20, 21, 22], [23, 24, 23]),
                ([20, 21, 22], [23, 24, 23]),
                ([30, 31, 32], [33, 34, 33]),
                ([40, 41, 42], [43, 44, 43]),
            )
        )
        for replay_index, (new_ids1, new_ids14) in enumerate(id_sets):
            ids1.copy_(torch.tensor([new_ids1], device="cuda"))
            ids14.copy_(torch.tensor([new_ids14], device="cuda"))
            graph.replay()
            replay_stream.synchronize()
            for got, row_ids in ((out1, new_ids1), (out14, new_ids14)):
                idx = torch.tensor(row_ids)
                want = (
                    weight[idx].float().unflatten(-1, (-1, BLOCK))
                    * scale[idx].float().unsqueeze(-1)
                ).flatten(-2).to(torch.bfloat16)
                assert torch.equal(got.cpu()[0], want)
            if replay_index == 0:
                cold = table1._native_store.stats()
                assert cold["submitted_sqes"] > before["submitted_sqes"]
                assert cold["completed_cqes"] > before["completed_cqes"]
                assert cold["direct_read_bytes"] > before["direct_read_bytes"]
                assert cold["unique_misses"] - before["unique_misses"] == 5
            elif replay_index == 1:
                warm = table1._native_store.stats()
                assert warm["submitted_sqes"] == cold["submitted_sqes"]
                assert warm["hits"] - cold["hits"] == 6

    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        with pytest.raises(RuntimeError, match="first replay stream"):
            graph.replay()
    if registry is not None:
        registry.close()


def test_missing_layer_raises(table):
    directory, *_ = table
    with pytest.raises(FileNotFoundError, match="layers.13.engram.embed.weight"):
        EngramFileTable.open(str(directory), layer_id=13, num_embeddings=ROWS, dim=DIM)


def test_row_count_mismatch_raises(table):
    directory, *_ = table
    with pytest.raises(ValueError, match="rows"):
        EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS + 1, dim=DIM)


def test_lookup_raises_under_cuda_graph_capture(table, monkeypatch):
    # lookup() makes a host sync (.cpu()) and only works eagerly; under capture
    # it must fail loudly instead of hitting an opaque CUDA capture error.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    directory, *_ = table
    t = EngramFileTable.open(str(directory), layer_id=1, num_embeddings=ROWS, dim=DIM)
    with pytest.raises(RuntimeError, match="disable-cuda-graph"):
        t.lookup(torch.tensor([0]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
