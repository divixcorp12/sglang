"""SglangCpuExpertsLayer and SglangCpuExpertsForward match their ctypes mirrors in pool.py, field by field."""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
ABI = REPO / "python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"
CXX = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")

PROBE = r"""
#include <cstddef>
#include <cstdio>
#include "cpu_experts_abi.h"
#define F(T, f) std::printf(#T " " #f " %zu\n", offsetof(T, f));
int main() {
    F(SglangCpuExpertsLayer, abi_version) F(SglangCpuExpertsLayer, capacity) F(SglangCpuExpertsLayer, hidden)
    F(SglangCpuExpertsLayer, intermediate) F(SglangCpuExpertsLayer, activation) F(SglangCpuExpertsLayer, act_limit)
    F(SglangCpuExpertsLayer, slab_count) F(SglangCpuExpertsLayer, slabs) F(SglangCpuExpertsLayer, slot_bytes)
    F(SglangCpuExpertsLayer, params)
    F(SglangCpuExpertsForward, abi_version) F(SglangCpuExpertsForward, rows) F(SglangCpuExpertsForward, layer)
    F(SglangCpuExpertsForward, x) F(SglangCpuExpertsForward, slots) F(SglangCpuExpertsForward, weights)
    F(SglangCpuExpertsForward, out) F(SglangCpuExpertsForward, k) F(SglangCpuExpertsForward, threads)
    F(SglangCpuExpertsForward, accumulate)
    std::printf("SglangCpuExpertsLayer sizeof %zu\nSglangCpuExpertsForward sizeof %zu\n",
                sizeof(SglangCpuExpertsLayer), sizeof(SglangCpuExpertsForward));
}
"""


@pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")
def test_the_ctypes_mirrors_match_the_c_layout(tmp_path):
    from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertsForwardCall, CpuExpertsLayer

    (tmp_path / "probe.cpp").write_text(PROBE)
    exe = tmp_path / "probe"
    subprocess.run([CXX, "-I", str(ABI.parent), str(tmp_path / "probe.cpp"), "-o", str(exe)], check=True)
    c = {}
    for line in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.splitlines():
        struct, field, value = line.split()
        c[(struct, field)] = int(value)
    for struct, mirror in (("SglangCpuExpertsLayer", CpuExpertsLayer), ("SglangCpuExpertsForward", CpuExpertsForwardCall)):
        for name, _ in mirror._fields_:
            assert c[(struct, name)] == getattr(mirror, name).offset, (struct, name)
        assert c[(struct, "sizeof")] == ctypes.sizeof(mirror), struct
