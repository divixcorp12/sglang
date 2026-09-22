"""Opt-in CUDA host callback backed by a native worker and shared io_uring row cache."""

from __future__ import annotations

import os
from functools import lru_cache

from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def native_engram_host_node():
    source = os.path.join(os.path.dirname(__file__), "engram_host_node.cpp")
    return load(
        name="engram_host_node_cpp",
        sources=[source],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-luring"],
        verbose=False,
    )
