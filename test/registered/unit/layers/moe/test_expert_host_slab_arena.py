"""Shared host slab allocations preserve layout, placement, and ownership."""

import gc
import os
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


def test_arena_binds_its_whole_span_in_2mib_stripes(monkeypatch: pytest.MonkeyPatch) -> None:
    allocate = Mock(side_effect=lambda nbytes, runs, row_bytes: tier.allocate_host_slab(
        1, (nbytes,), torch.uint8, register=False
    ).view(-1))
    monkeypatch.setattr(host_numa, "allocate_bound", allocate)
    huge = host_numa.HUGE_BYTES
    slabs = tier.allocate_host_slab_arena(
        4, {"weights": ((1 << 20,), torch.uint8), "scales": ((2048,), torch.uint8)},
        register=False, placement=((0, 1), (1, 1)),
    )
    # The stripes ignore the slab join at 4 MiB: each slab's rows are spread over both nodes 2 MiB at a time.
    allocate.assert_called_once_with((4 << 20) + 8192, [(0, 0, huge), (1, huge, huge), (0, 2 * huge, 8192)], 1)
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


def _arena_bytes(rows, row_bytes_by_slab):
    """The arena's byte count: each slab's rows at the next page-aligned offset."""
    offset = 0
    for row_bytes in row_bytes_by_slab:
        offset = -(-offset // tier.PAGE_BYTES) * tier.PAGE_BYTES + rows * row_bytes
    return offset


def _totals(ranges):
    totals = {}
    for node, lo, hi in ranges:
        totals[node] = totals.get(node, 0) + hi - lo
    return totals


def test_many_slab_arena_binds_in_2mib_stripes_within_2mib_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # 24 named slabs of 12 rows x 700,000 B, 60/40.
    rows, row_bytes, count, placement = 12, 700_000, 24, ((0, 60), (1, 40))
    nbytes = _arena_bytes(rows, [row_bytes] * count)
    huge = host_numa.HUGE_BYTES
    calls = []
    monkeypatch.setattr(host_numa, "_mbind", lambda address, length, node: calls.append((address, length, node)))
    specs = {f"slab{i}": ((row_bytes,), torch.uint8) for i in range(count)}
    slabs = tier.allocate_host_slab_arena(rows, specs, register=False, placement=placement)
    owner = slabs["slab0"]._expert_stream_slab_arena
    base = owner.data_ptr()
    assert base % huge == 0
    assert owner.nbytes == nbytes
    # The layout is unchanged: page-aligned offsets from the owner, the same shapes.
    offset = 0
    for name, slab in slabs.items():
        offset = -(-offset // tier.PAGE_BYTES) * tier.PAGE_BYTES
        assert slab.data_ptr() - base == offset and slab.shape == (rows, row_bytes)
        offset += rows * row_bytes
    # The mbind ranges tile the arena out to its 2 MiB end and follow the stripes one 2 MiB piece at a time.
    end = -(-nbytes // huge) * huge
    assert calls[0][0] == base and calls[-1][0] + calls[-1][1] == base + end
    for (address, length, node), (following_address, _, following) in zip(calls, calls[1:]):
        assert address + length == following_address and node != following
        assert (following_address - base) % huge == 0
    pieces = [node for _, length, node in calls for _ in range(length // huge)]
    assert pieces == [node for node, _, _ in host_numa.stripe_runs(nbytes, placement)]
    bound = _totals([(node, address, address + length) for address, length, node in calls])
    assert owner._numa_bound_bytes == bound
    for node, share in placement:
        assert abs(bound[node] - end * share / 100) <= huge, (node, bound)


def test_arena_with_a_single_node_binds_its_whole_span_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(host_numa, "_mbind", lambda address, length, node: calls.append((address, length, node)))
    slabs = tier.allocate_host_slab_arena(
        3, {"weights": ((1000,), torch.float32), "scales": ((7,), torch.int16)}, register=False, placement=((1, 1),)
    )
    owner = slabs["weights"]._expert_stream_slab_arena
    assert calls == [(owner.data_ptr(), host_numa.HUGE_BYTES, 1)]  # 3 pages + 42 B, bound out to the 2 MiB end
    assert slabs["scales"].data_ptr() == owner.data_ptr() + 3 * tier.PAGE_BYTES


def test_zero_row_arena_with_a_placement_binds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    bind = Mock()
    monkeypatch.setattr(host_numa, "_mbind", bind)
    slabs = tier.allocate_host_slab_arena(0, {"weights": ((1000,), torch.float32)}, register=False, placement=((0, 1),))
    assert slabs["weights"].shape == (0, 1000)
    bind.assert_not_called()


@pytest.mark.skipif(not os.path.exists("/sys/devices/system/node/node1"), reason="needs a second NUMA node")
def test_arena_real_policy_alternates_node_every_2mib() -> None:
    # The dsv41 EXL3 slabs' rows, 37 rows, 60/40: the real mbind policy follows the stripes across every slab join.
    rows_bytes = (8_847_360, 20_480, 9_216, 4_423_680, 4_608, 10_240)
    names = ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh")
    placement = ((0, 60), (1, 40))
    try:
        slabs = tier.allocate_host_slab_arena(
            37, {name: ((row_bytes,), torch.uint8) for name, row_bytes in zip(names, rows_bytes)},
            register=False, placement=placement,
        )
    except OSError as error:
        pytest.skip(f"mbind not permitted: {error}")
    owner = slabs["w13_trellis"]._expert_stream_slab_arena
    base, huge = owner.data_ptr(), host_numa.HUGE_BYTES
    stripes = host_numa.stripe_runs(_arena_bytes(37, rows_bytes), placement)
    assert len(stripes) > 100 and base % huge == 0
    for node, start, _ in stripes:
        for address in (base + start, base + start + huge - tier.PAGE_BYTES):
            assert host_numa.address_policy(address) == (2, frozenset({node})), hex(start)
    end = stripes[-1][1] + huge
    assert owner._numa_bound_bytes == _totals([(node, start, start + huge) for node, start, _ in stripes])
    assert host_numa.address_policy(base + end)[0] == 0  # the slack past the 2 MiB end is not bound
