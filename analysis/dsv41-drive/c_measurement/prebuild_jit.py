#!/usr/bin/env python3
"""Compile the production gather kernel's JIT module with NO GPU and NO CUDA context, so the measurement window contains no compile.

The JIT targets the local GPU's architecture, detected through torch.cuda; with the GPU hidden that fails ("sm00"). This
script answers the two queries that detection makes (device index, capability) with the RTX 5090's (12, 0) BEFORE the build,
so nvcc builds the same sm_120 target the window will look up in the cache. It never initialises CUDA.
Run:  CUDA_VISIBLE_DEVICES= CUDA_HOME=/usr/local/cuda-13.2 PYTHONPATH=<repo>/python taskset -c <cpu> python prebuild_jit.py
"""
import sys, time
import torch
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_capability = lambda device=None: (12, 0)
t = time.time()
from sglang.kernels.ops.moe import expert_cache_transfer as e
m = e._jit_expert_cache_transfer_module()
print("built/loaded", m, "in %.1f s" % (time.time() - t))
import sglang.kernels.jit.utils.arch as arch
print("arch used:", arch._CUDA_ARCH)
