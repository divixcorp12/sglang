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


def _arena_runs(rows, row_bytes_by_slab, placement):
    """The byte runs allocate_host_slab_arena asks for: each page-aligned slab's rows split in placement order."""
    runs, offset = [], 0
    for row_bytes in row_bytes_by_slab:
        offset = -(-offset // tier.PAGE_BYTES) * tier.PAGE_BYTES
        for node, first, count in host_numa.split_rows(rows, placement):
            runs.append((node, offset + first * row_bytes, offset + (first + count) * row_bytes))
        offset += rows * row_bytes
    return offset, runs


def _totals(ranges):
    totals = {}
    for node, lo, hi in ranges:
        totals[node] = totals.get(node, 0) + hi - lo
    return totals


def test_many_slab_arena_binds_on_2mib_boundaries_within_2mib_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    # 24 named slabs of 12 rows x 700,000 B, 60/40: every slab join and every in-slab split is a node change.
    # Rounding each change to its nearest 2 MiB independently drifts node 0 by 16 MiB here; the carried error must not.
    rows, row_bytes, count, placement = 12, 700_000, 24, ((0, 60), (1, 40))
    nbytes, runs = _arena_runs(rows, [row_bytes] * count, placement)
    asked = _totals(runs)
    span = -(-nbytes // tier.PAGE_BYTES) * tier.PAGE_BYTES
    merged = [list(runs[0])]
    for node, lo, hi in runs[1:]:
        if merged[-1][0] == node:
            merged[-1][2] = hi
        else:
            merged.append([node, lo, hi])
    cuts = [0] + [(lo + host_numa.HUGE_BYTES // 2) // host_numa.HUGE_BYTES * host_numa.HUGE_BYTES for _, lo, _ in merged[1:]]
    naive = _totals([(node, cuts[i], (cuts + [span])[i + 1]) for i, (node, _, _) in enumerate(merged)])
    assert abs(naive[0] - asked[0]) > 2 * host_numa.HUGE_BYTES

    calls = []
    monkeypatch.setattr(host_numa, "_mbind", lambda address, length, node: calls.append((address, length, node)))
    specs = {f"slab{i}": ((row_bytes,), torch.uint8) for i in range(count)}
    slabs = tier.allocate_host_slab_arena(rows, specs, register=False, placement=placement)
    owner = slabs["slab0"]._expert_stream_slab_arena
    base = owner.data_ptr()
    assert base % host_numa.HUGE_BYTES == 0
    assert owner.nbytes == nbytes
    # The layout is unchanged: page-aligned offsets from the owner, the same shapes.
    offset = 0
    for name, slab in slabs.items():
        offset = -(-offset // tier.PAGE_BYTES) * tier.PAGE_BYTES
        assert slab.data_ptr() - base == offset and slab.shape == (rows, row_bytes)
        offset += rows * row_bytes
    # The mbind ranges tile the page-rounded arena, change node only on 2 MiB boundaries, and keep each total.
    assert calls[0][0] == base and calls[-1][0] + calls[-1][1] == base + span
    for (address, length, node), (following_address, _, following) in zip(calls, calls[1:]):
        assert address + length == following_address and node != following
        assert (following_address - base) % host_numa.HUGE_BYTES == 0
    bound = _totals([(node, address, address + length) for address, length, node in calls])
    assert owner._numa_bound_bytes == bound
    for node in asked:
        assert abs(bound[node] - asked[node]) <= host_numa.HUGE_BYTES, (node, bound, asked)


def test_arena_with_a_single_node_binds_its_whole_span_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(host_numa, "_mbind", lambda address, length, node: calls.append((address, length, node)))
    slabs = tier.allocate_host_slab_arena(
        3, {"weights": ((1000,), torch.float32), "scales": ((7,), torch.int16)}, register=False, placement=((1, 1),)
    )
    owner = slabs["weights"]._expert_stream_slab_arena
    assert calls == [(owner.data_ptr(), 4 * tier.PAGE_BYTES, 1)]
    assert slabs["scales"].data_ptr() == owner.data_ptr() + 3 * tier.PAGE_BYTES


def test_zero_row_arena_with_a_placement_binds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    bind = Mock()
    monkeypatch.setattr(host_numa, "_mbind", bind)
    slabs = tier.allocate_host_slab_arena(0, {"weights": ((1000,), torch.float32)}, register=False, placement=((0, 1),))
    assert slabs["weights"].shape == (0, 1000)
    bind.assert_not_called()


@pytest.mark.skipif(not os.path.exists("/sys/devices/system/node/node1"), reason="needs a second NUMA node")
def test_arena_slab_joins_change_node_on_2mib_boundaries_of_the_real_policy() -> None:
    # The dsv41 EXL3 slabs' rows, 37 rows, 60/40: the real mbind policy flips exactly at each planned boundary.
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
    base = owner.data_ptr()
    nbytes, runs = _arena_runs(37, rows_bytes, placement)
    bindings = host_numa.plan_bindings(nbytes, runs)
    assert len(bindings) > 1 and base % host_numa.HUGE_BYTES == 0
    for (node, _, end), (following, _, _) in zip(bindings, bindings[1:]):
        assert end % host_numa.HUGE_BYTES == 0
        assert host_numa.address_policy(base + end - tier.PAGE_BYTES) == (2, frozenset({node}))
        assert host_numa.address_policy(base + end) == (2, frozenset({following}))
    assert owner._numa_bound_bytes == _totals(bindings)
