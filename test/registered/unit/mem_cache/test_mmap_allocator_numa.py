"""SGLANG_HICACHE_HOST_NUMA_NODE: alloc_mmap binds the host pool to the node before touching it (CPU).

The placement is read back page by page with move_pages(2) (query mode: no nodes given), so the test needs a machine
with at least two NUMA nodes; elsewhere only the unset path runs.
"""

import ctypes
import os
import platform

import pytest
import torch

import sglang.srt.mem_cache.storage.mmap.mmap_allocator as alloc
from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TWO_NODES = os.path.isdir("/sys/devices/system/node/node1") and platform.machine() == "x86_64"
PAGE = 4096
MOVE_PAGES = 279  # x86_64


def _page_nodes(tensor: torch.Tensor) -> set:
    base = tensor.data_ptr()
    pages = tensor.numel() * tensor.element_size() // PAGE
    addrs = (ctypes.c_void_p * pages)(*[base + i * PAGE for i in range(pages)])
    status = (ctypes.c_int * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.syscall(ctypes.c_long(MOVE_PAGES), 0, ctypes.c_ulong(pages), addrs, None, status, 0)
    assert rc == 0, os.strerror(ctypes.get_errno())
    return set(status)


@pytest.mark.skipif(not TWO_NODES, reason="needs two NUMA nodes")
@pytest.mark.parametrize("node", [0, 1])
def test_the_pool_lands_on_the_node_it_is_bound_to(node, caplog):
    with envs.SGLANG_HICACHE_HOST_NUMA_NODE.override(node), caplog.at_level("INFO"):
        tensor = alloc_mmap((8, 1 << 20), torch.uint8)  # 8 MiB, every page faulted in by the allocator
    assert _page_nodes(tensor) == {node}
    assert f"bound to NUMA node {node}" in caplog.text
    tensor.fill_(7)  # writable and whole
    assert int(tensor.sum()) == 7 * tensor.numel()


def test_unset_keeps_the_first_touch_path(monkeypatch):
    def refuse(*args):
        raise AssertionError("an unset node must not bind")

    monkeypatch.setattr(alloc, "_alloc_bound", refuse)
    assert envs.SGLANG_HICACHE_HOST_NUMA_NODE.get() is None
    tensor = alloc_mmap((2, 4096), torch.uint8)
    assert tensor.shape == (2, 4096)
