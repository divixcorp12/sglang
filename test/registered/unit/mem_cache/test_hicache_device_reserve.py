"""HiCache's post-sizing device memory: write-back staging is capped, and the KV budget reserves room for it."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    dsv4_hicache_staging_bytes,
)
from sglang.srt.mem_cache.kv_cache_configurator import (
    check_hicache_staging_within_reserve,
    hicache_runtime_reservation_gb,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20


def _host_alloc_patches():
    alloc = mock.Mock(return_value=torch.empty(1, dtype=torch.uint8))
    return (
        mock.patch.object(memory_pool_host, "host_memory_budget_bytes", return_value=1 << 40),
        mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}),
    )


def _paged_pool(*, layers: int, item_bytes: int, num_host_pages: int) -> DeepSeekV4PagedHostPool:
    budget, alloc = _host_alloc_patches()
    with budget, alloc:
        return DeepSeekV4PagedHostPool(
            pool_name="swa",
            device_buffers=[torch.empty(1, dtype=torch.uint8) for _ in range(layers)],
            item_bytes=item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=1,
            layout="page_first",
        )


def _state_pool(*, layers: int, row_elems: int, num_host_pages: int) -> DeepSeekV4StateHostPool:
    state_pools = [
        SimpleNamespace(
            ring_size=2,
            kv_score_buffer=SimpleNamespace(kv_score=torch.empty((4, row_elems), dtype=torch.uint8)),
        )
        for _ in range(layers)
    ]
    budget, alloc = _host_alloc_patches()
    with budget, alloc:
        return DeepSeekV4StateHostPool(
            pool_name="c4_state",
            state_pools=state_pools,
            num_host_pages=num_host_pages,
            swa_page_size=2,
            layout="page_first",
        )


class TestWriteBackStagingCap(CustomTestCase):
    def test_paged_pool_staging_is_capped_in_bytes(self):
        # The production SWA mirror staged all 28 pages x 40 layers x 149760 B = 160 MiB of device memory.
        with envs.SGLANG_HICACHE_WRITE_BACK_STAGING_MAX_MB.override(1):
            pool = _paged_pool(layers=4, item_bytes=65536, num_host_pages=28)
        self.assertEqual(pool.staging_buffer.shape[0], 4)
        self.assertLessEqual(pool.staging_buffer.nbytes, 1 * MIB)

    def test_paged_pool_under_the_cap_keeps_the_page_chunk(self):
        with envs.SGLANG_HICACHE_WRITE_BACK_STAGING_MAX_MB.override(32):
            pool = _paged_pool(layers=1, item_bytes=1024, num_host_pages=1208)
        self.assertEqual(pool.staging_buffer.shape[0], 64)

    def test_a_page_row_larger_than_the_cap_still_stages_one_page(self):
        with envs.SGLANG_HICACHE_WRITE_BACK_STAGING_MAX_MB.override(1):
            pool = _paged_pool(layers=4, item_bytes=MIB, num_host_pages=28)
        self.assertEqual(pool.staging_buffer.shape[0], 1)

    def test_state_pool_staging_is_capped_in_bytes(self):
        with envs.SGLANG_HICACHE_WRITE_BACK_STAGING_MAX_MB.override(1):
            pool = _state_pool(layers=2, row_elems=65536, num_host_pages=28)
        self.assertLessEqual(pool.staging_buffer.nbytes, 1 * MIB)
        self.assertGreaterEqual(pool.staging_buffer.shape[0], 1)


class TestHiCacheDeviceReserve(CustomTestCase):
    def test_no_reservation_without_hierarchical_cache(self):
        self.assertEqual(hicache_runtime_reservation_gb(enable_hierarchical_cache=False), 0.0)

    def test_reservation_is_the_configured_budget(self):
        with envs.SGLANG_HICACHE_DEVICE_RESERVE_MB.override(64):
            self.assertEqual(hicache_runtime_reservation_gb(enable_hierarchical_cache=True), 64 / 1024)

    def test_staging_over_the_reserve_refuses_to_start(self):
        with envs.SGLANG_HICACHE_DEVICE_RESERVE_MB.override(64):
            check_hicache_staging_within_reserve(staging_bytes=64 * MIB)
            with self.assertRaisesRegex(ValueError, "SGLANG_HICACHE_DEVICE_RESERVE_MB"):
                check_hicache_staging_within_reserve(staging_bytes=64 * MIB + 1)

    def test_dsv4_staging_bytes_sums_the_mirrored_pools(self):
        with envs.SGLANG_HICACHE_WRITE_BACK_STAGING_MAX_MB.override(1):
            paged = _paged_pool(layers=4, item_bytes=65536, num_host_pages=28)
            state = _state_pool(layers=2, row_elems=65536, num_host_pages=28)
        entries = [
            PoolEntry(name=name, host_pool=pool, device_pool=None, layer_mapper=lambda i: i)
            for name, pool in (("swa", paged), ("c4_state", state))
        ]
        entries.append(
            PoolEntry(name="anchor", host_pool=SimpleNamespace(), device_pool=None, layer_mapper=lambda i: i)
        )
        self.assertEqual(
            dsv4_hicache_staging_bytes(entries),
            paged.staging_buffer.nbytes + state.staging_buffer.nbytes,
        )


if __name__ == "__main__":
    unittest.main()
