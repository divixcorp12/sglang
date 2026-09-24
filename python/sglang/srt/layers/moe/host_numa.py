"""NUMA placement for the pinned host expert tier.

A placement is an ordered list of (node, bytes). Each slab is one anonymous mapping whose row ranges are bound
(``mbind(MPOL_BIND)``) to the nodes in that order, in proportion to their bytes, before any page is touched. The
slab stays one contiguous range, so every reader that addresses a row as ``base + slot * row_bytes`` (the C++
RAM-miss service, io_uring fixed buffers, the copy kernels) is unaffected by where its pages live.

Binding replaces first touch, which put the tier wherever the faulting thread ran and let an over-sized tier spill
onto a nearly full node, where the allocation stalled in reclaim instead of failing (arm_env's SERVER_CORES note).
``check_capacity`` refuses a placement a node cannot hold before anything is allocated.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import platform
from collections import Counter
from typing import Sequence

import torch

PAGE_BYTES = mmap.PAGESIZE
MIB = 1 << 20
# Left free on every node the tier uses: the server allocates after the tier (bounce buffers, the Engram slab,
# CUDA's host-side state) with the default local policy, so a node filled to the byte would push those elsewhere.
NODE_HEADROOM_BYTES = 4 << 30
_MPOL_BIND = 2
_SYSCALLS = {"x86_64": {"mbind": 237, "move_pages": 279}, "aarch64": {"mbind": 235, "move_pages": 239}}
_NODE_ROOT = "/sys/devices/system/node"

Placement = tuple[tuple[int, int], ...]  # ((node, bytes), ...), in binding order


def parse_placement(value: str) -> Placement:
    """``"0:65536,1:30720"`` (node:MiB, comma separated) as ((0, bytes), (1, bytes)); empty gives ()."""
    value = value.strip()
    if not value:
        return ()
    placement = []
    for entry in value.split(","):
        node, sep, mib = entry.strip().partition(":")
        if not sep or not node.strip().isdigit() or not mib.strip().isdigit():
            raise ValueError(f"NUMA placement {value!r}: {entry.strip()!r} is not node:MiB")
        if int(mib) == 0:
            raise ValueError(f"NUMA placement {value!r}: node {node.strip()} has 0 MiB; leave it out")
        placement.append((int(node), int(mib) * MIB))
    nodes = [node for node, _ in placement]
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"NUMA placement {value!r} names a node twice")
    return tuple(placement)


def node_memory(node: int, root: str = _NODE_ROOT) -> dict[str, int]:
    """Bytes free and reclaimable (file-backed page cache) on ``node``, from sysfs."""
    path = os.path.join(root, f"node{node}", "meminfo")
    if not os.path.exists(path):
        raise ValueError(f"NUMA node {node} does not exist ({path} is missing)")
    fields = {}
    with open(path) as f:
        for line in f:
            # "Node 0 MemFree:   123456 kB"
            parts = line.split()
            if len(parts) >= 4 and parts[-1] == "kB":
                fields[parts[2].rstrip(":")] = int(parts[3]) * 1024
    return {
        "free": fields.get("MemFree", 0),
        "reclaimable": fields.get("Active(file)", 0) + fields.get("Inactive(file)", 0),
    }


def check_capacity(placement: Placement, *, headroom: int = NODE_HEADROOM_BYTES, root: str = _NODE_ROOT) -> None:
    """Refuse a placement some node cannot hold with ``headroom`` to spare, counting its page cache as reclaimable."""
    short = []
    for node, nbytes in placement:
        memory = node_memory(node, root)
        available = memory["free"] + memory["reclaimable"]
        if nbytes + headroom > available:
            short.append(
                f"node {node}: asked {nbytes / MIB:.0f} MiB + {headroom / MIB:.0f} MiB headroom, "
                f"{memory['free'] / MIB:.0f} MiB free + {memory['reclaimable'] / MIB:.0f} MiB page cache"
            )
    if short:
        raise ValueError("pinned host tier does not fit its NUMA placement: " + "; ".join(short))


def split_rows(rows: int, placement: Placement) -> list[tuple[int, int, int]]:
    """``rows`` as (node, first row, row count) runs in placement order, in proportion to each node's bytes.

    Largest remainder, ties to the earlier node, so the counts sum to ``rows`` exactly; a node whose share rounds to
    no rows gets no run.
    """
    total = sum(nbytes for _, nbytes in placement)
    exact = [rows * nbytes / total for _, nbytes in placement]
    counts = [math.floor(x) for x in exact]
    order = sorted(range(len(placement)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: rows - sum(counts)]:
        counts[i] += 1
    runs, start = [], 0
    for (node, _), count in zip(placement, counts):
        if count:
            runs.append((node, start, count))
            start += count
    return runs


def _syscall(name: str) -> int:
    numbers = _SYSCALLS.get(platform.machine())
    if numbers is None:
        raise RuntimeError(f"NUMA placement is not supported on {platform.machine()}")
    return numbers[name]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _mbind(address: int, length: int, node: int) -> None:
    mask_words = node // 64 + 1
    mask = (ctypes.c_ulong * mask_words)()
    mask[node // 64] = 1 << (node % 64)
    rc = _libc.syscall(
        ctypes.c_long(_syscall("mbind")),
        ctypes.c_void_p(address),
        ctypes.c_ulong(length),
        ctypes.c_int(_MPOL_BIND),
        mask,
        ctypes.c_ulong(mask_words * 64 + 1),
        ctypes.c_uint(0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mbind to NUMA node {node} failed: {os.strerror(err)}")


def allocate_bound(nbytes: int, runs: Sequence[tuple[int, int, int]], row_bytes: int) -> torch.Tensor:
    """A fresh page-aligned uint8 host tensor of ``nbytes`` whose row runs are bound to their nodes.

    A run boundary that falls inside a page is rounded down to the page, so a page shared by two rows goes to the
    later run's node. Nothing is touched here: pages land on their node when first written or registered.
    """
    if nbytes == 0:
        return torch.empty(0, dtype=torch.uint8)
    mapping = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    tensor = torch.frombuffer(mapping, dtype=torch.uint8)  # holds the mapping alive
    base = tensor.data_ptr()
    for node, first, count in runs:
        lo = (first * row_bytes) // PAGE_BYTES * PAGE_BYTES
        hi = min(nbytes, (first + count) * row_bytes)
        if hi > lo:
            _mbind(base + lo, -(-(hi - lo) // PAGE_BYTES) * PAGE_BYTES, node)
    return tensor


def page_nodes(tensor: torch.Tensor, samples: int = 64) -> Counter:
    """Which NUMA node holds each of ``samples`` evenly spaced pages of ``tensor`` (negative: not resident)."""
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes == 0:
        return Counter()
    first = tensor.data_ptr() // PAGE_BYTES * PAGE_BYTES
    last = (tensor.data_ptr() + nbytes - 1) // PAGE_BYTES * PAGE_BYTES
    n = max(1, min(samples, (last - first) // PAGE_BYTES + 1))
    pages = (ctypes.c_void_p * n)(*[first + (last - first) // PAGE_BYTES * i // max(n - 1, 1) * PAGE_BYTES for i in range(n)])
    status = (ctypes.c_int * n)()
    rc = _libc.syscall(
        ctypes.c_long(_syscall("move_pages")), ctypes.c_int(0), ctypes.c_ulong(n), pages, None, status, ctypes.c_int(0)
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"move_pages query failed: {os.strerror(err)}")
    return Counter(int(s) for s in status)
