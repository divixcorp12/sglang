"""Shared host slab allocations preserve layout, placement, and ownership."""

import gc
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe import expert_host_tier as tier
from sglang.srt.layers.moe import host_numa
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_arena_views_preserve_layout_and_retain_one_owner() -> None:
    slabs = tier.allocate_host_slab_arena(
        3, {"weights": ((1000,), torch.float32), "scales": ((7,), torch.int16)}, register=False
    )
    weights, scales = slabs.values()
    owner = weights._expert_stream_slab_arena
    assert scales._expert_stream_slab_arena is owner
    assert weights.shape == (3, 1000)
    assert scales.shape == (3, 7)
    assert weights.dtype == torch.float32
    assert scales.dtype == torch.int16
    assert weights.nbytes == 12000
    assert scales.nbytes == 42
    assert weights.is_contiguous() and scales.is_contiguous()
    assert weights.data_ptr() == owner.data_ptr()
    assert scales.data_ptr() == owner.data_ptr() + 3 * tier.PAGE_BYTES
    assert owner.nbytes == 3 * tier.PAGE_BYTES + 42
    assert owner.data_ptr() % tier.PAGE_BYTES == 0
    assert scales.data_ptr() % tier.PAGE_BYTES == 0
    assert weights.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()
    reference = weakref.ref(owner)
    del owner, weights, slabs
    gc.collect()
    assert reference() is not None
    scales.fill_(9)
    assert torch.all(scales == 9)
    del scales
    gc.collect()
    assert reference() is None


def test_arena_registers_and_releases_one_shared_span(monkeypatch: pytest.MonkeyPatch) -> None:
    register, unregister = Mock(), Mock()
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.pool_host.common",
        SimpleNamespace(_cuda_host_register=register, _cuda_host_unregister=unregister),
    )
    slabs = tier.allocate_host_slab_arena(
        2, {"weights": ((4096,), torch.uint8), "scales": ((512,), torch.float32)}, register=True
    )
    owner = slabs["weights"]._expert_stream_slab_arena
    register.assert_called_once()
    assert register.call_args.args[0] is owner
    assert register.call_args.kwargs == {}
    tier.release_host_slabs(list(slabs.values()))
    unregister.assert_called_once()
    assert unregister.call_args.args[0] is owner


def test_arena_preserves_each_slab_numa_row_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    allocate = Mock(side_effect=lambda nbytes, runs, row_bytes: tier.allocate_host_slab(
        1, (nbytes,), torch.uint8, register=False
    ).view(-1))
    monkeypatch.setattr(host_numa, "allocate_bound", allocate)
    slabs = tier.allocate_host_slab_arena(
        4, {"weights": ((4096,), torch.uint8), "scales": ((2048,), torch.uint8)},
        register=False, placement=((0, 1), (1, 1)),
    )
    allocate.assert_called_once_with(
        24576, [(0, 0, 8192), (1, 8192, 8192), (0, 16384, 4096), (1, 20480, 4096)], 1
    )
    assert slabs["weights"]._expert_stream_slab_arena is slabs["scales"]._expert_stream_slab_arena


@pytest.mark.parametrize("rows", [0, 2])
def test_empty_arena_and_scalar_rows(rows: int) -> None:
    assert tier.allocate_host_slab_arena(rows, {}, register=False) == {}
    slabs = tier.allocate_host_slab_arena(rows, {"scale": ((), torch.float32)}, register=False)
    assert slabs["scale"].shape == (rows,)
    assert slabs["scale"].nbytes == rows * 4


@pytest.mark.parametrize("enabled", [False, True])
def test_cache_uses_shared_arena_only_with_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_SLAB_ARENA", "1" if enabled else "0")
    layer = torch.nn.Module()
    layer.weights = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(4, 8))
    layer.scales = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    layer._nvfp4_file_source_bytes_per_expert = 40
    streamer = ExpertStreamer(layer, ("weights", "scales"))
    cache = ExpertPinnedHostCache(streamer, 2, device="cpu")
    try:
        weights, scales = cache.tensors.values()
        assert hasattr(weights, "_expert_stream_slab_arena") is enabled
        assert (weights.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()) is enabled
        cache.ensure_rows(torch.tensor([1, 3]))
        assert torch.equal(weights, layer.weights[[1, 3]])
        assert torch.equal(scales, layer.scales[[1, 3]])
    finally:
        cache.close()


@pytest.mark.parametrize("value", ["", "true", "2"])
def test_cache_rejects_invalid_arena_option(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING_SLAB_ARENA", value)
    layer = torch.nn.Module()
    layer.weights = torch.nn.Parameter(torch.ones(4, 8))
    layer._nvfp4_file_source_bytes_per_expert = 32
    streamer = ExpertStreamer(layer, ("weights",))
    with pytest.raises(ValueError, match="SLAB_ARENA must be 0 or 1"):
        ExpertPinnedHostCache(streamer, 2, device="cpu")
