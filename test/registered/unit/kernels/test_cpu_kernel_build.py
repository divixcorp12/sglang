"""How each format's CPU kernel is compiled into its expert-stream host module (CPU, no build)."""

import pytest

from sglang.kernels.ops.moe import cpu_kernel_build as ckb
from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_engine.config import CpuExpertsConfig, ExpertStreamConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _flags(build, source):
    return dict(build.source_cflags)[source]


def test_the_exl3_kernel_takes_fast_math_openmp_and_the_residual_128_options():
    build = ckb.cpu_kernel_build("exl3", ExpertStreamConfig())
    assert build.cpp_files == ("exl3/optimized/kernel.cpp",)
    flags = _flags(build, "exl3/optimized/kernel.cpp")
    assert {"-Ofast", "-fopenmp", "-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128"} <= set(flags)
    assert "-fno-finite-math-only" not in flags  # it changes EXL3's bits
    assert build.ldflags == ("-fopenmp",)


def test_the_nvfp4_c_file_is_compiled_as_c11():
    build = ckb.cpu_kernel_build("nvfp4", ExpertStreamConfig())
    assert build.cpp_files == ("nvfp4/optimized/kernel.cpp", "nvfp4/upstream/nvfp4.c")
    flags = _flags(build, "nvfp4/upstream/nvfp4.c")
    assert flags[:3] == ("-x", "c", "-std=c11")
    assert "-std=c++20" in _flags(build, "nvfp4/optimized/kernel.cpp")


def test_a_format_without_a_cpu_kernel_compiles_none_and_the_fake_has_its_own():
    assert ckb.cpu_kernel_build("mxfp4", ExpertStreamConfig()) is None
    assert ckb.cpu_kernel_build("fake", ExpertStreamConfig()).cpp_files == ("moe/formats/fake/fake_kernel.cpp",)


@pytest.mark.parametrize("max_m", [1, 3, 17, 18, -2])
def test_cpu_max_m_must_be_even_in_2_to_16(max_m):
    with envs.SGLANG_EXL3_CPU_MAX_M.override(max_m):
        with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_MAX_M"):
            ckb.max_m_defines()


def test_cpu_max_m_maps_to_a_define_and_0_keeps_the_kernels_own():
    """Default 16 (8 tokens per chunk, register_band_halves); 0 builds the kernel source's default (MAX_M 4)."""
    assert envs.SGLANG_EXL3_CPU_MAX_M.default == 16
    with envs.SGLANG_EXL3_CPU_MAX_M.override(8):
        assert ckb.max_m_defines() == ["-DEXL3_MOE_CPU_MAX_M=8"]
    with envs.SGLANG_EXL3_CPU_MAX_M.override(0):
        assert ckb.max_m_defines() == []


def _fake_compiler(monkeypatch, version, clang=False):
    monkeypatch.setattr(ckb.toolchain, "host_compiler_path", lambda: "/fake/g++")
    monkeypatch.setattr(ckb, "_compiler_identity", lambda compiler: (version, clang))


def test_the_kernel_compiler_must_be_gcc_15(monkeypatch):
    with envs.SGLANG_EXL3_CPU_CXX.override(""):
        _fake_compiler(monkeypatch, "15.2.1")
        ckb.check_kernel_compiler()
        _fake_compiler(monkeypatch, "14.2.0")
        with pytest.raises(RuntimeError, match="require GCC 15"):
            ckb.check_kernel_compiler()
        _fake_compiler(monkeypatch, "15.0.0", clang=True)
        with pytest.raises(RuntimeError, match="rather than Clang"):
            ckb.check_kernel_compiler()


def test_a_named_sglang_exl3_cpu_cxx_must_be_the_host_compiler(monkeypatch):
    _fake_compiler(monkeypatch, "15.2.1")
    with envs.SGLANG_EXL3_CPU_CXX.override("/other/g++"):
        with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_CXX"):
            ckb.check_kernel_compiler()
    with envs.SGLANG_EXL3_CPU_CXX.override("/fake/g++"):
        ckb.check_kernel_compiler()


def test_only_a_cpu_experts_launch_checks_the_exl3_compiler(monkeypatch):
    """A launch without CPU experts still builds the EXL3 host module (the kernel is linked in), so it must not need
    GCC 15."""
    _fake_compiler(monkeypatch, "14.2.0")
    ckb.cpu_kernel_build("exl3", ExpertStreamConfig())
    with pytest.raises(RuntimeError, match="require GCC 15"):
        ckb.cpu_kernel_build("exl3", ExpertStreamConfig(cpu_experts=CpuExpertsConfig(enabled=True)))
