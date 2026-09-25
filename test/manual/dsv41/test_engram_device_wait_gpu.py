"""The Engram device-wait lookup captured in a breakable CUDA graph (GPU).

With SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT the decode graph's layer-1 and layer-14 lookups
are a post kernel and a wait kernel served by the native service thread, with no host node.
"""

import json
import multiprocessing
import os
import struct
import time
import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers import engram
from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers.engram_host_node import native_engram_host_node
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import DedupedCudaGraphRegistry
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA CUDA device")

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


def _make_table(directory):
    torch.manual_seed(0)
    weights = {}
    tensors = {"layers.1.engram.test_padding": ("U8", torch.zeros(7, dtype=torch.uint8))}
    for layer, (mean, lo, hi) in ((1, (0, 118, 130)), (14, (5, 110, 117))):
        weight = (torch.randn(ROWS, DIM) * 4 + mean).to(torch.float8_e4m3fn)
        scale = torch.randint(lo, hi, (ROWS, DIM // BLOCK), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        tensors[f"layers.{layer}.engram.embed.weight"] = ("F8_E4M3", weight)
        tensors[f"layers.{layer}.engram.embed.scale"] = ("F8_E8M0", scale)
        weights[layer] = (weight, scale)
    _write(directory / "model-00047-of-00048.safetensors", tensors)
    return weights


def _want(weight, scale, row_ids):
    idx = torch.tensor(row_ids)
    return (
        (weight[idx].float().unflatten(-1, (-1, BLOCK)) * scale[idx].float().unsqueeze(-1))
        .flatten(-2)
        .to(torch.bfloat16)
    )


def _layer(directory, layer_id, store):
    table = EngramFileTable.open(str(directory), layer_id=layer_id, num_embeddings=ROWS, dim=DIM)
    # A private store: the shared one keeps rows cached by (layer, id) across tests whose tables differ.
    table._native_store = store
    return types.SimpleNamespace(file_table=table, dim=DIM, layer_id=layer_id, tp_size=1, _posted_lookup=None)


def _private_store():
    native = native_engram_host_node()
    native.get_shared_store(1 << 20, DIM + DIM // BLOCK)
    return native.create_store(64 * (DIM + DIM // BLOCK), DIM + DIM // BLOCK)


def _batch():
    return types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True))


def _open(directory, monkeypatch, *, device_wait: bool):
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING", "1")
    monkeypatch.setenv("SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT", "1" if device_wait else "0")
    store = _private_store()
    return _layer(directory, 1, store), _layer(directory, 14, store)


def _capture_two_layers(layer1, layer14, ids, *, early_post: bool, forbid_host_nodes: bool, registry=None):
    """Capture the decode step's shape: hash ids exist first, then layer 1 and layer 14 look up."""
    graph = BreakableCUDAGraph(deduped_cuda_graph=registry)
    batch = _batch()
    with torch.cuda.stream(torch.cuda.Stream()):
        with BreakableCUDAGraphCapture(graph, forbid_host_nodes=forbid_host_nodes):
            if early_post:
                engram.post_engram_device_lookups(
                    [types.SimpleNamespace(embed=layer1, layer_hash_index=0),
                     types.SimpleNamespace(embed=layer14, layer_hash_index=1)],
                    ids, batch,
                )
            out1 = engram.EngramEmbedding.forward(layer1, ids[:, 0], batch)
            out14 = engram.EngramEmbedding.forward(layer14, ids[:, 1], batch)
    return graph, out1, out14


ID_SETS = (
    ([3, 0, 49], [3, 1, 3]),
    ([3, 0, 49], [3, 1, 3]),
    ([2, 10, 48], [4, 4, 2]),
    ([49, 6, 0], [1, 3, 8]),
    ([99, 98, 0], [10, 11, 12]),
)


@pytest.mark.parametrize("early_post", [True, False])
@pytest.mark.parametrize("deduplicate", [False, True])
def test_device_wait_graph_has_no_host_nodes_and_replays_exact_rows(tmp_path, monkeypatch, early_post, deduplicate):
    weights = _make_table(tmp_path)
    layer1, layer14 = _open(tmp_path, monkeypatch, device_wait=True)
    ids = torch.tensor([[[3, 0, 49], [7, 1, 7]]], device="cuda", dtype=torch.int64)
    registry = DedupedCudaGraphRegistry() if deduplicate else None
    graph, out1, out14 = _capture_two_layers(
        layer1, layer14, ids, early_post=early_post, forbid_host_nodes=True, registry=registry
    )
    assert len(graph._segments) == 1 and not graph._break_fns
    assert len(graph._retained_host_callbacks) == 2
    assert layer1._posted_lookup is None and layer14._posted_lookup is None
    replay_stream = torch.cuda.Stream()
    contexts = dict(zip((1, 14), graph._retained_host_callbacks))
    with torch.cuda.stream(replay_stream):
        for step, (ids1, ids14) in enumerate(ID_SETS, start=1):
            ids.copy_(torch.tensor([[ids1, ids14]], device="cuda"))
            graph.replay()
            replay_stream.synchronize()
            for layer, row_ids, out in ((1, ids1, out1), (14, ids14, out14)):
                # Stage by stage: the posted ids, the served rows, the rows the wait copied, the dequant.
                context = contexts[layer]
                weight, scale = weights[layer]
                packed = torch.cat([weight.view(torch.uint8), scale.view(torch.uint8)], dim=1)[row_ids]
                where = f"layer {layer} step {step}: control {context.control.tolist()}"
                assert context.ids.tolist() == row_ids, where
                assert torch.equal(context.rows, packed), where
                assert torch.equal(context.packed_gpu.cpu(), packed), where
                assert torch.equal(out.cpu()[0], _want(weight, scale, row_ids)), where
    for context in graph._retained_host_callbacks:
        assert context.native_context.served() == len(ID_SETS)
    if registry is not None:
        registry.close()


def test_forbidding_host_nodes_rejects_the_host_node_capture(tmp_path, monkeypatch):
    _make_table(tmp_path)
    layer1, layer14 = _open(tmp_path, monkeypatch, device_wait=False)
    ids = torch.tensor([[[3, 0, 49], [7, 1, 7]]], device="cuda", dtype=torch.int64)
    with pytest.raises(RuntimeError, match=r"captured 2 CUDA host node\(s\)"):
        _capture_two_layers(layer1, layer14, ids, early_post=False, forbid_host_nodes=True)


def test_the_wait_holds_the_stream_until_a_slow_service_serves(tmp_path, monkeypatch):
    """The service is delayed past the whole graph's device time: the rows are right only if the wait kernel
    really waits (a wait that returns early copies the previous step's rows), and the launch returns at once
    because nothing in the graph runs on the host."""
    weights = _make_table(tmp_path)
    layer1, layer14 = _open(tmp_path, monkeypatch, device_wait=True)
    ids = torch.tensor([[[3, 0, 49], [7, 1, 7]]], device="cuda", dtype=torch.int64)
    graph, out1, out14 = _capture_two_layers(layer1, layer14, ids, early_post=True, forbid_host_nodes=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph.replay()
        stream.synchronize()
        for context in graph._retained_host_callbacks:
            context.native_context.set_test_delay_us(100_000)
        ids.copy_(torch.tensor([[[5, 6, 7], [8, 9, 10]]], device="cuda"))
        start = time.perf_counter()
        graph.replay()
        launch_s = time.perf_counter() - start
        stream.synchronize()
        total_s = time.perf_counter() - start
    assert launch_s < 0.02 < 0.1 <= total_s
    for context in graph._retained_host_callbacks:
        stats = context.native_context.wait_stats()
        assert stats["waits"] == 2 and stats["spins"] >= 1 and stats["spin_us_max"] >= 100_000, stats
    assert torch.equal(out1.cpu()[0], _want(*weights[1], [5, 6, 7]))
    assert torch.equal(out14.cpu()[0], _want(*weights[14], [8, 9, 10]))


def test_the_post_publishes_its_sequence_after_every_id(tmp_path, monkeypatch):
    """The post kernel stalls after its first id: a sequence published before the ids (a reordered or unfenced
    post) lets the service read the other ids from the previous step while the kernel stalls."""
    weights = _make_table(tmp_path)
    layer1, layer14 = _open(tmp_path, monkeypatch, device_wait=True)
    monkeypatch.setattr(engram, "ENGRAM_POST_TEST_STALL_NS", 20_000_000)
    ids = torch.tensor([[[3, 0, 49], [7, 1, 7]]], device="cuda", dtype=torch.int64)
    graph, out1, out14 = _capture_two_layers(layer1, layer14, ids, early_post=True, forbid_host_nodes=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for ids1, ids14 in (([3, 0, 49], [7, 1, 7]), ([3, 20, 21], [7, 22, 23])):
            ids.copy_(torch.tensor([[ids1, ids14]], device="cuda"))
            graph.replay()
            stream.synchronize()
            assert torch.equal(out1.cpu()[0], _want(*weights[1], ids1))
            assert torch.equal(out14.cpu()[0], _want(*weights[14], ids14))


def _timeout_worker(directory, connection):
    try:
        os.environ["SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING"] = "1"
        os.environ["SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT"] = "1"
        engram.ENGRAM_DEVICE_WAIT_TIMEOUT_MS = 50
        layer14 = _layer(directory, 14, _private_store())
        ids = torch.tensor([[1, 2]], device="cuda", dtype=torch.int64)
        graph = BreakableCUDAGraph()
        with torch.cuda.stream(torch.cuda.Stream()):
            with BreakableCUDAGraphCapture(graph, forbid_host_nodes=True):
                engram.EngramEmbedding.forward(layer14, ids, _batch())
        (context,) = graph._retained_host_callbacks
        context.native_context.set_test_delay_us(1_000_000)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            try:
                graph.replay()
                stream.synchronize()
                connection.send(("no error", ""))
                return
            except RuntimeError as exc:
                message = str(exc)
        control = context.control.tolist()
        connection.send(("raised", json.dumps({"message": message, "control": control})))
    except BaseException as exc:
        connection.send(("error", repr(exc)))


def test_a_timed_out_wait_fails_loudly_and_latches_the_fatal_word(tmp_path):
    from sglang.kernels.ops.embeddings import engram_ring

    _make_table(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_timeout_worker, args=(tmp_path, child))
    process.start()
    assert parent.poll(300), "the timeout worker did not report"
    kind, payload = parent.recv()
    process.join(60)
    assert kind == "raised", payload
    result = json.loads(payload)
    assert "device-side assert" in result["message"] or "timed out or failed" in result["message"]
    assert result["control"][engram_ring.FATAL_SEQ] == 1
    assert result["control"][engram_ring.FATAL_STATUS] == engram_ring.DEVICE_TIMEOUT


def test_the_flag_needs_the_native_store(tmp_path, monkeypatch):
    _make_table(tmp_path)
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING", "0")
    with envs.SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT.override(True):
        with pytest.raises(ValueError, match="set both"):
            EngramFileTable.open(str(tmp_path), layer_id=1, num_embeddings=ROWS, dim=DIM)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
