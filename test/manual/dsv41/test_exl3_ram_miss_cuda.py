"""Option C on the GPU: post/wait kernels against the C++ thread (window test).

Fake finite EXL3 checkpoint unless DSV41_EXL3_DIR is set, in which case the
overhead test reads real layer-3 rows (the model's own row source) to measure
the stream-idle time of a forced NVMe miss. DSV41_RAM_MISS_OUT names a JSON
file for the measured numbers.
"""

import json
import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EXPERTS = 16
TOP_K = 6


def _service(tmp_path, *, expert_dir=None, layer=0, capacity=8, timeout_ms=2000, advise=False, layers=2):
    from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissDevice, Exl3RamMissHost, new_page
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3

    if expert_dir is None:
        write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=EXPERTS, hidden=1024, inter=512, finite=True)
        expert_dir = str(tmp_path)
    layout = build_exl3_expert_layout(expert_dir)
    fmt = Exl3ExpertFormat(layout, layer, direct=expert_dir != str(tmp_path))
    specs = {s.name: s for s in fmt.tensor_specs(None)}
    layer_ids = [layer, layer + 1][:layers]
    slabs = {
        lid: {n: allocate_host_slab(capacity, specs[n].row_shape, specs[n].dtype, register=True) for n in EXL3_STREAMED_NAMES}
        for lid in layer_ids
    }
    tables = exl3_ram_miss_tables(layout, fmt.segment_map(), slabs)
    page = new_page(pin=True)
    slot_map = torch.full((len(layer_ids), layout.num_experts), -1, dtype=torch.int32).pin_memory()
    host = Exl3RamMissHost(tables, page=page, slot_map=slot_map, direct=fmt.direct)
    host.start_thread(fatal_wait_s=30.0)
    dev = Exl3RamMissDevice(page, slot_map, device="cuda", layers=len(layer_ids), timeout_ms=timeout_ms, advise=advise)
    return layout, fmt, specs, slabs, host, dev


def _close(host, slabs):
    """Stop the thread, then unregister the slabs: a freed but still registered range makes
    the next test's cudaHostRegister of the reused addresses fail (cudaErrorHostMemoryAlreadyRegistered)."""
    from sglang.srt.layers.moe.expert_host_tier import release_host_slabs

    host.stop()
    release_host_slabs([slab for names in slabs.values() for slab in names.values()])


def _buffers():
    return dict(
        planned=torch.zeros(TOP_K, dtype=torch.int64, device="cuda"),
        count=torch.zeros(1, dtype=torch.int32, device="cuda"),
        routes=torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda"),
        host_rows=torch.zeros(TOP_K, dtype=torch.int64, device="cuda"),
        keep=torch.ones(1, dtype=torch.float32, device="cuda"),
        ram_miss=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )


def _step(dev, b, row=0, next_row=-1):
    dev.post(row, b["planned"], b["count"], b["routes"], next_row)
    dev.wait(row, b["planned"], b["count"], b["host_rows"], b["keep"], b["ram_miss"])


def _set(b, planned, routes):
    b["planned"][: len(planned)].copy_(torch.tensor(planned))
    b["count"].fill_(len(planned))
    b["routes"].fill_(-1)
    b["routes"][: len(routes)].copy_(torch.tensor(routes))


def test_a_miss_is_served_and_translated(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path)
    try:
        b = _buffers()
        _set(b, [3, 5], [3, 5, 0, 1, 2, 4])
        _step(dev, b)
        torch.cuda.synchronize()
        mapping = host.mapping(0)
        assert b["host_rows"][:2].tolist() == [mapping[3], mapping[5]] and min(mapping[3], mapping[5]) >= 0
        assert b["keep"].item() == 1.0 and b["ram_miss"].item() == 0
        assert dev.stats()["timeouts"] == 0 and host.fatal_seq() == 0
    finally:
        _close(host, slabs)


def test_a_hung_read_times_out_and_everything_after_is_fast(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path, timeout_ms=50)
    try:
        b = _buffers()
        host.inject(delay_s=5.0)
        _set(b, [7], [7])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _step(dev, b)
        end.record()
        torch.cuda.synchronize()
        assert b["keep"].item() == 0.0 and host.fatal_seq() != 0
        assert start.elapsed_time(end) < 500.0
        start.record()
        for _ in range(40):
            _step(dev, b)
        end.record()
        torch.cuda.synchronize()
        assert start.elapsed_time(end) < 50.0  # the sticky fast path
        assert dev.stats()["sticky"] == 1 and dev.stats()["timeouts"] == 1
    finally:
        host.inject(delay_s=0.0)
        _close(host, slabs)


def test_post_and_wait_capture_and_replay(tmp_path):
    layout, fmt, specs, slabs, host, dev = _service(tmp_path)
    try:
        b = _buffers()
        _set(b, [1], [1, 2])
        _step(dev, b)  # warm up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _step(dev, b)
        for planned in ([4], [6, 8], [9, 10, 11]):
            _set(b, planned, planned)
            graph.replay()
            torch.cuda.synchronize()
            mapping = host.mapping(0)
            assert b["host_rows"][: len(planned)].tolist() == [mapping[e] for e in planned]
            assert b["keep"].item() == 1.0
    finally:
        _close(host, slabs)


def test_a_rewritten_slot_is_read_fresh(tmp_path):
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu, expert_row_segments
    from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    layout, fmt, specs, slabs, host, dev = _service(tmp_path, capacity=1)
    try:
        dest = {n: torch.zeros((1,) + specs[n].row_shape, dtype=specs[n].dtype, device="cuda") for n in EXL3_STREAMED_NAMES}
        segments = expert_row_segments([(slabs[0][n], dest[n]) for n in EXL3_STREAMED_NAMES])
        slots = torch.zeros(TOP_K, dtype=torch.int32, device="cuda")
        b = _buffers()
        for expert in (2, 9, 2):  # capacity 1: each demand evicts and rewrites slot 0
            _set(b, [expert], [expert])
            _step(dev, b)
            copy_expert_row_segments_gpu(segments, b["host_rows"], slots, b["count"])
            torch.cuda.synchronize()
            want = {n: torch.empty((1,) + specs[n].row_shape, dtype=specs[n].dtype) for n in EXL3_STREAMED_NAMES}
            Exl3ShardRowSource.for_layer(layout, 0, fmt.segment_map(), direct=False).read(torch.tensor([expert]), want)
            for n in EXL3_STREAMED_NAMES:
                assert torch.equal(dest[n].cpu().view(torch.uint8), want[n].view(torch.uint8)), (expert, n)
    finally:
        _close(host, slabs)


def test_overheads(tmp_path):
    """§9.3 acceptance numbers: hit-path cost per layer, stream idle per forced miss."""
    expert_dir = os.environ.get("DSV41_EXL3_DIR")
    layer = int(os.environ.get("DSV41_PROBE_LAYER", "3")) if expert_dir else 0
    layout, fmt, specs, slabs, host, dev = _service(tmp_path, expert_dir=expert_dir, layer=layer, capacity=16)
    report = {"real_rows": bool(expert_dir)}
    try:
        b = _buffers()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        _set(b, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
        _step(dev, b)  # load six rows
        torch.cuda.synchronize()
        _set(b, [], [0, 1, 2, 3, 4, 5])  # all in RAM: a touch-only post, no wait
        start.record()
        for _ in range(400):
            _step(dev, b)
        end.record()
        torch.cuda.synchronize()
        report["hit_path_us_per_layer"] = start.elapsed_time(end) * 1000 / 400
        for misses, ids in ((1, [6]), (6, [7, 8, 9, 10, 11, 12])):
            _set(b, ids, ids)
            start.record()
            _step(dev, b)
            end.record()
            torch.cuda.synchronize()
            report[f"stream_idle_ms_{misses}_miss"] = start.elapsed_time(end)
            _set(b, [], [])
        report["thread"] = host.counters()
    finally:
        _close(host, slabs)
    out = os.environ.get("DSV41_RAM_MISS_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report))
    assert report["hit_path_us_per_layer"] < 100.0
