"""Only explicitly owned host allocations may become registered I/O regions: one per slab tensor, with its row size."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_separate_slabs_remain_separate_registration_regions():
    first = torch.empty((4, 16), dtype=torch.uint8)
    second = torch.empty((2, 64), dtype=torch.uint8)
    regions = transport._table_buffer_regions(SimpleNamespace(keepalive=(first, second)))
    assert regions.tolist() == [[first.data_ptr(), 64, 16], [second.data_ptr(), 128, 64]]


def test_arena_slabs_are_registered_per_slab_not_as_the_whole_arena():
    # A 2.69 GB arena is past io_uring's 1 GiB per-buffer limit: the owner is ignored, each slab is its own region
    # (registered as row-aligned chunks), and the arena's padding is never registered.
    arena = torch.empty(4096, dtype=torch.uint8)
    first, second = arena[:64].view(4, 16), arena[2048:2176].view(2, 64)
    first._expert_stream_slab_arena = arena
    second._expert_stream_slab_arena = arena
    regions = transport._table_buffer_regions(SimpleNamespace(keepalive=(first, second)))
    assert regions.tolist() == [[first.data_ptr(), 64, 16], [second.data_ptr(), 128, 64]]


def test_owners_must_be_contiguous_cpu_tensors():
    with pytest.raises(ValueError, match="tensors"):
        transport._table_buffer_regions(SimpleNamespace(keepalive=(object(),)))
    strided = torch.empty((4, 16), dtype=torch.uint8)[:, ::2]
    with pytest.raises(ValueError, match="contiguous CPU"):
        transport._table_buffer_regions(SimpleNamespace(keepalive=(strided,)))


def test_tables_without_owners_supply_no_inferred_address_span():
    regions = transport._table_buffer_regions(SimpleNamespace())
    assert regions.dtype == torch.int64 and regions.shape == (0, 3)


@pytest.mark.parametrize("has_owner", [False, True])
def test_region_metadata_stays_on_cpu_under_a_non_cpu_default(has_owner):
    slab = torch.empty((4, 16), dtype=torch.uint8, device="cpu")
    tables = SimpleNamespace(keepalive=(slab,) if has_owner else ())
    with torch.device("meta"):
        regions = transport._table_buffer_regions(tables)
    assert regions.device.type == "cpu"
    assert regions.tolist() == ([[slab.data_ptr(), 64, 16]] if has_owner else [])
