"""NUMA placement for the pinned host expert tier.

A placement is an ordered list of (node, bytes). Each slab is one anonymous mapping, 2 MiB aligned, whose row
ranges are bound (``mbind(MPOL_BIND)``) to the nodes in that order before any page is touched. The slab stays one
contiguous range, so every reader that addresses a row as ``base + slot * row_bytes`` (the C++ RAM-miss service,
io_uring fixed buffers, the copy kernels) is unaffected by where its pages live.

The rows are divided in proportion to the nodes' bytes, but the binding is not exact to the row: every place where
the node changes is rounded to a 2 MiB boundary of the mapping (``plan_bindings``), so the rows around it lie partly
or wholly on the neighbouring node. With two nodes each node's bound bytes stay within 2 MiB of what its rows asked
for (with more nodes, within (nodes - 1) x 2 MiB), because each boundary is rounded in whichever direction keeps the
running per-node error smallest, not independently. The last binding runs past the slab to the next 2 MiB boundary
(bound to the last node, counted in its bound bytes, never touched), so the slab's last huge page is not split either.

Why 2 MiB: an ``mbind`` range splits the mapping's VMA at its ends, and a split that is not 2 MiB aligned leaves the
huge page around it backed by 4 KiB pages. An io_uring registered buffer that mixes 4 KiB and 2 MiB folios does not
coalesce, and the 6.12 kernel then charges its pin accounting against every page of every buffer registered before
it, so registering the tier grew quadratically (plan 2026-09-28-reader-crtp-uring-registration, Tasks 9 and 10).

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
# Transparent huge page size: every mapping base and every node change in it sits on a multiple of this.
HUGE_BYTES = 2 << 20
# Left free on every node the tier uses: the server allocates after the tier (bounce buffers, the Engram slab,
# CUDA's host-side state) with the default local policy, so a node filled to the byte would push those elsewhere.
NODE_HEADROOM_BYTES = 4 << 30
_MPOL_BIND = 2
_MPOL_F_ADDR = 1 << 1
_SYSCALLS = {
    "x86_64": {"mbind": 237, "move_pages": 279, "get_mempolicy": 239},
    "aarch64": {"mbind": 235, "move_pages": 239, "get_mempolicy": 236},
}
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
        ctypes.c_ulong(_MPOL_BIND),
        mask,
        ctypes.c_ulong(mask_words * 64 + 1),
        ctypes.c_uint(0),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mbind to NUMA node {node} failed: {os.strerror(err)}")


def plan_bindings(nbytes: int, runs: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """The ``mbind`` ranges, as (node, start, end) byte offsets from a 2 MiB-aligned base, for byte ``runs``.

    ``runs`` are (node, start, end) byte ranges in address order, as the rows ask for them. The ranges returned tile
    ``[0, nbytes rounded up to HUGE_BYTES)`` with no gap or overlap, one range per change of node (adjacent runs on
    the same node merge, and so do the gaps between runs), and every range start but the first, and the last range's
    end, is a multiple of ``HUGE_BYTES``.

    The tail past ``nbytes``, up to the next 2 MiB boundary, is bound to the last run's node. The caller maps it as
    slack and never touches it; it exists so that the mapping's last huge page lies wholly inside one binding. Ending
    at the page-rounded ``nbytes`` instead split the VMA there and left that huge page on 4 KiB folios, so the last
    registered io_uring chunk of every mapping could not coalesce (plan 2026-09-28-reader-crtp-uring-registration,
    final review Important 1). The tail counts as bound to that node: in the totals the ranges give (so in
    ``_numa_bound_bytes``) and in the running error below, so the last node change shifts to compensate for it.

    Each node change goes to the multiple of ``HUGE_BYTES`` just below or just above its ideal place (the next run's
    start), whichever leaves the two nodes' running errors (bytes bound minus bytes asked, so far) smaller; the error
    is carried forward instead of each boundary rounding to its nearest, which would drift about 1 MiB per boundary
    over an arena's many slab joins. With two nodes each node's total, tail included, stays within ``HUGE_BYTES`` of
    its ask (plus the gaps between runs, which no run asks for). A range can round away to nothing: a run shorter
    than 2 MiB between two others is then bound to its neighbours' node.
    """
    end_all = -(-nbytes // HUGE_BYTES) * HUGE_BYTES
    segments: list[list[int]] = []  # [node, start, end, bytes asked], same-node runs merged
    for node, lo, hi in runs:
        hi = min(hi, nbytes)
        if hi <= lo:
            continue
        if segments and segments[-1][0] == node:
            segments[-1][2] = hi
            segments[-1][3] += hi - lo
        else:
            segments.append([node, lo, hi, hi - lo])
    if not segments:
        return []
    error: Counter = Counter()
    for node, lo, hi, asked in segments:
        error[node] += (hi - lo) - asked  # gaps inside a merged segment
    error[segments[0][0]] += segments[0][1]  # a gap before the first run
    error[segments[-1][0]] += end_all - segments[-1][2]  # the tail past the last run, up to the 2 MiB end
    bindings, start = [], 0
    for (node, _, hi, _), (following, ideal, _, _) in zip(segments, segments[1:]):
        error[node] += ideal - hi  # the gap up to the next run goes to this node
        down = max(start, min(end_all, ideal // HUGE_BYTES * HUGE_BYTES))
        up = max(start, min(end_all, -(-ideal // HUGE_BYTES) * HUGE_BYTES))
        end = min(
            (down, up),
            key=lambda r: (max(abs(error[node] + r - ideal), abs(error[following] - (r - ideal))), abs(r - ideal), r),
        )
        error[node] += end - ideal
        error[following] -= end - ideal
        if end > start:
            if bindings and bindings[-1][0] == node:
                bindings[-1] = (node, bindings[-1][1], end)
            else:
                bindings.append((node, start, end))
        start = end
    node = segments[-1][0]
    if end_all > start:
        if bindings and bindings[-1][0] == node:
            bindings[-1] = (node, bindings[-1][1], end_all)
        else:
            bindings.append((node, start, end_all))
    return bindings


def allocate_bound(nbytes: int, runs: Sequence[tuple[int, int, int]], row_bytes: int) -> torch.Tensor:
    """A fresh 2 MiB-aligned uint8 host tensor of ``nbytes`` whose (node, first row, row count) runs are bound.

    The binding follows ``plan_bindings``: node changes sit on 2 MiB boundaries of the tensor, so a row across one,
    and whole rows within 2 MiB of one, are bound to the neighbouring node, and the last binding runs past the tensor
    to the next 2 MiB boundary. The actual bytes bound per node, that tail included, are in the tensor's
    ``_numa_bound_bytes`` ({node: bytes}). The mapping is over-allocated for the alignment and the tail: nbytes
    rounded up to 2 MiB, plus 2 MiB. The tail is bound but never touched; the slack before the base and past the
    tail is neither bound nor touched. Nothing is touched here: pages land on their node when first written or
    registered.
    """
    if nbytes == 0:
        tensor = torch.empty(0, dtype=torch.uint8)
        tensor._numa_bound_bytes = {}
        return tensor
    mapping = mmap.mmap(
        -1, -(-nbytes // HUGE_BYTES) * HUGE_BYTES + HUGE_BYTES, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    )
    whole = torch.frombuffer(mapping, dtype=torch.uint8)
    offset = -whole.data_ptr() % HUGE_BYTES
    tensor = whole[offset : offset + nbytes]  # holds the mapping alive
    base = tensor.data_ptr()
    bound: Counter = Counter()
    byte_runs = [(node, first * row_bytes, (first + count) * row_bytes) for node, first, count in runs]
    for node, lo, hi in plan_bindings(nbytes, byte_runs):
        _mbind(base + lo, hi - lo, node)
        bound[node] += hi - lo
    tensor._numa_bound_bytes = dict(bound)
    return tensor


def address_policy(address: int) -> tuple[int, frozenset[int]]:
    """The memory policy governing ``address`` (mode, nodes), without faulting its page in."""
    mode = ctypes.c_int()
    mask = (ctypes.c_ulong * 16)()
    rc = _libc.syscall(
        ctypes.c_long(_syscall("get_mempolicy")),
        ctypes.byref(mode),
        mask,
        ctypes.c_ulong(16 * 64),
        ctypes.c_void_p(address),
        ctypes.c_ulong(_MPOL_F_ADDR),
    )
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"get_mempolicy failed: {os.strerror(err)}")
    return mode.value, frozenset(i for i in range(16 * 64) if mask[i // 64] >> (i % 64) & 1)


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
