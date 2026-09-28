"""Only explicitly owned host allocations may become registered I/O regions."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_separate_slabs_remain_separate_registration_regions():
    first = torch.empty(64, dtype=torch.uint8)
    second = torch.empty(128, dtype=torch.uint8)
    regions = transport._table_buffer_regions(SimpleNamespace(keepalive=(first, second)))
    assert regions.tolist() == [[first.data_ptr(), 64], [second.data_ptr(), 128]]


def test_shared_arena_is_registered_once_including_padding():
    arena = torch.empty(4096, dtype=torch.uint8)
    first, second = arena[:64], arena[2048:2176]
    first._expert_stream_slab_arena = arena
    second._expert_stream_slab_arena = arena
    regions = transport._table_buffer_regions(SimpleNamespace(keepalive=(first, second)))
    assert regions.tolist() == [[arena.data_ptr(), arena.numel()]]


def test_region_metadata_must_actually_contain_the_slab():
    arena = torch.empty(4096, dtype=torch.uint8)
    unrelated = torch.empty(64, dtype=torch.uint8)
    unrelated._expert_stream_slab_arena = arena
    with pytest.raises(ValueError, match="contain"):
        transport._table_buffer_regions(SimpleNamespace(keepalive=(unrelated,)))


def test_tables_without_owners_supply_no_inferred_address_span():
    regions = transport._table_buffer_regions(SimpleNamespace())
    assert regions.dtype == torch.int64 and regions.shape == (0, 2)
