"""SGLANG_DSV41_ENABLE_PREFILL_FILLS wired through option C (CPU): the native slot table is the pinned tier's
row_fills, eager reads land from the row images through the service's reader, and the flag is refused where the
service cannot serve it."""

import contextlib
import faulthandler
import shutil

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_fake_exl3 import write_fake_exl3
from sglang.test.dsv41_ram_miss_fixtures import ROW_IMAGE_DIM, write_row_images

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

LAYERS, EXPERTS, CAPACITY = 2, 6, 4


@pytest.fixture(autouse=True)
def hang_guard():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _build(tmp_path, *, row_images=True, fills=True):
    source = tmp_path / "ckpt"
    source.mkdir()
    write_fake_exl3(str(source), num_layers=LAYERS, num_experts=EXPERTS, hidden=ROW_IMAGE_DIM, inter=ROW_IMAGE_DIM)
    layout = build_exl3_expert_layout(str(source))
    root = tmp_path / "mirror"
    shutil.copytree(source, root)
    fmt0 = Exl3ExpertFormat(layout, 0, direct=True, source_root=str(source))
    write_row_images(layout, fmt0.segment_map(), str(source), [str(root)])
    stack = contextlib.ExitStack()
    for env, value in (
        (envs.SGLANG_MOE_EXPERT_ROW_SOURCE, "shards"),
        (envs.SGLANG_MOE_EXPERT_GRAPH_GATHER, True),
        (envs.SGLANG_MOE_EXPERT_MIRROR_DIRS, str(root)),
        (envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES, True),
        (envs.SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES, row_images),
        (envs.SGLANG_DSV41_ENABLE_PREFILL_FILLS, fills),
    ):
        stack.enter_context(env.override(value))
    module.Exl3RamMissService._instance = None
    streamers, caches = {}, {}
    for layer_id in range(LAYERS):
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        fmt = Exl3ExpertFormat(layout, layer_id, direct=True, source_root=str(source))
        streamer = ExpertStreamer(layer, fmt.names, layer_id=layer_id, format=fmt)
        layer._nvfp4_expert_streamer = streamer
        caches[layer_id] = ExpertPinnedHostCache(streamer, CAPACITY, device="cpu", **fmt.pinned_tier_options(layer))
        streamers[layer_id] = streamer
    return stack, layout, source, streamers, caches


@pytest.fixture
def tiers(tmp_path):
    stack, layout, source, streamers, caches = _build(tmp_path)
    service = module.Exl3RamMissService.get()
    yield service, layout, source, streamers, caches
    service.shutdown()
    module.Exl3RamMissService._instance = None
    stack.close()


def _reference(layout, source, layer, experts):
    from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

    fmt = Exl3ExpertFormat(layout, layer, direct=False, source_root=str(source))
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    out = {n: torch.empty((len(experts),) + specs[n].row_shape, dtype=specs[n].dtype) for n in EXL3_STREAMED_NAMES}
    Exl3ShardRowSource.for_layer(layout, layer, fmt.segment_map(), direct=False).read(torch.tensor(experts), out)
    return out


def _same(outputs, reference):
    return all(
        torch.equal(outputs[n].contiguous().view(torch.uint8), reference[n].contiguous().view(torch.uint8))
        for n in EXL3_STREAMED_NAMES
    )


def test_the_native_slot_table_is_the_tiers_row_fills(tiers):
    service, layout, source, streamers, caches = tiers
    assert caches[1].row_fills is caches[1]._lru
    assert isinstance(caches[1]._lru, module.NativePinnedSlotTable)


def test_ensure_rows_lands_the_row_images_through_the_service_reader(tiers, monkeypatch):
    service, layout, source, streamers, caches = tiers
    monkeypatch.setattr(streamers[1], "read_host_rows", lambda *a, **k: pytest.fail("the eager row source read"))
    caches[1].ensure_rows(torch.tensor([4, 2]))
    assert service.host.tables.row_images
    outputs = {n: torch.empty_like(caches[1].tensors[n][:2]) for n in EXL3_STREAMED_NAMES}
    caches[1].copy_rows(torch.tensor([4, 2]), outputs)
    assert _same(outputs, _reference(layout, source, 1, [4, 2]))


def test_a_layers_prefetch_serves_its_chunks_and_ends_with_the_host_use(tiers, monkeypatch):
    service, layout, source, streamers, caches = tiers
    streamer, cache = streamers[0], caches[0]
    monkeypatch.setattr(streamer, "read_host_rows", lambda *a, **k: pytest.fail("the eager row source read"))
    cache.ensure_rows(torch.tensor([5]))
    experts = [0, 3, 5, 1]
    outputs = []
    with streamer.prefill_fills(torch.tensor(experts)):
        assert service._pause_depth == 1  # one host use for the whole layer
        for chunk in ([0, 1], [3, 5]):
            out = {n: torch.empty_like(cache.tensors[n][:2]) for n in EXL3_STREAMED_NAMES}
            cache.gather_rows(torch.tensor(chunk), out)
            outputs.append((chunk, out))
    assert service._pause_depth == 0
    for chunk, out in outputs:
        assert _same(out, _reference(layout, source, 0, chunk))
    assert cache.stats.populated_rows == 4  # 5 by ensure_rows, then 0, 1, 3 by the prefetch


def test_the_flag_is_refused_without_row_images(tmp_path):
    stack, layout, source, streamers, caches = _build(tmp_path, row_images=False)
    try:
        with pytest.raises(RuntimeError, match="ROW_IMAGES"):
            caches[0].ensure_rows(torch.tensor([1]))
    finally:
        module.Exl3RamMissService._instance = None
        stack.close()


def test_the_flag_is_refused_without_the_native_slot_table(tmp_path):
    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    layer = torch.nn.Module()
    layer.layer_id = 0
    fmt = Exl3ExpertFormat(layout, 0, direct=False, source_root=str(tmp_path))
    with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(False), envs.SGLANG_DSV41_ENABLE_PREFILL_FILLS.override(True):
        with pytest.raises(RuntimeError, match="GRAPH_GATHER"):
            fmt.pinned_tier_options(layer)


def test_without_the_flag_the_tier_has_no_row_fills(tmp_path):
    stack, layout, source, streamers, caches = _build(tmp_path, fills=False)
    try:
        assert all(cache.row_fills is None for cache in caches.values())
    finally:
        module.Exl3RamMissService._instance = None
        stack.close()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
