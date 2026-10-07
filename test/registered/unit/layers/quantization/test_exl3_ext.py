"""Pure helpers of the exllamav3 extension loader (no build, no GPU)."""

import contextlib
import os

import pytest

from sglang.srt.layers.quantization.exl3 import ext as exl3_ext
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_sources_are_sorted_c_cpp_cu_only(tmp_path):
    for rel in ("b.cu", "a.cpp", "sub/c.c", "sub/d.cuh", "e.h", "sub/f.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    got = [os.path.relpath(p, tmp_path) for p in exl3_ext.extension_sources(str(tmp_path))]
    assert got == ["a.cpp", "b.cu", os.path.join("sub", "c.c")]


def test_wrong_commit_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(exl3_ext, "_checkout_commit", lambda src: "deadbeef")
    with pytest.raises(RuntimeError, match="expected exllamav3"):
        exl3_ext._checked_ext_dir(str(tmp_path))


def test_missing_src_raises(monkeypatch):
    monkeypatch.setattr(exl3_ext.envs.SGLANG_EXL3_SRC, "get", lambda: "")
    exl3_ext.exl3_ext.cache_clear()
    with pytest.raises(RuntimeError, match="SGLANG_EXL3_SRC"):
        exl3_ext.exl3_ext()


def _defines(residual, block):
    with exl3_ext.envs.SGLANG_EXL3_CPU_ACT_RESIDUAL.override(residual), exl3_ext.envs.SGLANG_EXL3_CPU_ACT_BLOCK.override(
        block
    ):
        return exl3_ext.cpu_act_defines()


def test_cpu_act_options_default_to_upstream():
    assert _defines(False, 0) == []
    assert exl3_ext.build_flavor([]) == ""


def test_cpu_act_options_map_to_defines_and_flavor():
    defines = _defines(True, 128)
    assert defines == ["-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128"]
    assert exl3_ext.build_flavor(defines) == "_resid_b128_cpu_v1"
    assert exl3_ext.build_flavor(_defines(False, 64)) == "_b64"
    assert exl3_ext.build_flavor(_defines(True, 0)) == "_resid"


def _cpu_experts_defines(**flags):
    """cpu_act_defines with SGLANG_DSV41_CPU_EXPERTS on and only the given SGLANG_EXL3_CPU_ACT_* flags set."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(exl3_ext.envs.SGLANG_DSV41_CPU_EXPERTS.override(True))
        for name in ("SGLANG_EXL3_CPU_ACT_RESIDUAL", "SGLANG_EXL3_CPU_ACT_BLOCK", "SGLANG_EXL3_CPU_MAX_M"):
            field = getattr(exl3_ext.envs, name)
            if name in flags:
                stack.enter_context(field.override(flags[name]))
            elif field.is_set():
                stack.callback(os.environ.__setitem__, name, os.environ[name])
                field.clear()
        return exl3_ext.cpu_act_defines()


@pytest.mark.parametrize(
    "flags",
    [{}, {"SGLANG_EXL3_CPU_ACT_RESIDUAL": True}, {"SGLANG_EXL3_CPU_ACT_RESIDUAL": True, "SGLANG_EXL3_CPU_ACT_BLOCK": 128}],
)
def test_cpu_experts_always_build_the_optimized_kernel(flags):
    defines = _cpu_experts_defines(**flags)
    assert defines == ["-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128"]
    assert exl3_ext.optimized_cpu(defines) and exl3_ext.build_flavor(defines) == "_resid_b128_cpu_v1"


@pytest.mark.parametrize(
    "flags", [{"SGLANG_EXL3_CPU_ACT_RESIDUAL": False}, {"SGLANG_EXL3_CPU_ACT_BLOCK": 64}, {"SGLANG_EXL3_CPU_ACT_BLOCK": 0}]
)
def test_cpu_experts_refuse_another_accuracy_flavor(flags):
    with pytest.raises(ValueError, match="SGLANG_DSV41_CPU_EXPERTS"):
        _cpu_experts_defines(**flags)


def test_cpu_max_m_maps_to_a_define_and_its_own_flavor():
    defines = _cpu_experts_defines(SGLANG_EXL3_CPU_MAX_M=8)
    assert defines == ["-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128", "-DEXL3_MOE_CPU_MAX_M=8"]
    assert exl3_ext.optimized_cpu(defines) and exl3_ext.build_flavor(defines) == "_resid_b128_m8_cpu_v1"


@pytest.mark.parametrize("max_m", [1, 3, 10, -2])
def test_cpu_max_m_must_be_even_in_2_to_8(max_m):
    with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_MAX_M"):
        _cpu_experts_defines(SGLANG_EXL3_CPU_MAX_M=max_m)


def test_cpu_max_m_needs_the_optimized_kernel():
    """The vendored kernel (moe_mul1.cpp) keeps its own MAX_M: setting the variable without CPU experts is refused."""
    with exl3_ext.envs.SGLANG_DSV41_CPU_EXPERTS.override(False), exl3_ext.envs.SGLANG_EXL3_CPU_MAX_M.override(8):
        with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_MAX_M"):
            exl3_ext.cpu_act_defines()


@pytest.mark.parametrize("block", [8, 100, -16])
def test_cpu_act_block_must_be_a_multiple_of_16(block):
    with pytest.raises(ValueError, match="multiple of 16"):
        _defines(False, block)


def test_vendored_kernel_replaces_upstreams_only(tmp_path):
    for rel in ("cpu/moe_mul1.cpp", "cpu/moe_handoff.cu", "quant/moe_mul1.cpp"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    got = exl3_ext.extension_sources(str(tmp_path), "/vendored/moe_mul1.cpp")
    assert "/vendored/moe_mul1.cpp" in got
    assert str(tmp_path / "cpu" / "moe_mul1.cpp") not in got
    assert str(tmp_path / "quant" / "moe_mul1.cpp") in got
    assert str(tmp_path / "cpu" / "moe_handoff.cu") in got


def test_vendored_kernel_is_in_the_tree():
    assert os.path.isfile(exl3_ext.VENDORED_CPU_KERNEL)
    assert os.path.isfile(exl3_ext.OPTIMIZED_CPU_KERNEL)


def _load_args(tmp_path, monkeypatch, residual, block):
    import torch.utils.cpp_extension as cpp_extension

    ext_dir = tmp_path / "exllamav3" / "exllamav3_ext"
    (ext_dir / "cpu").mkdir(parents=True)
    (ext_dir / "cpu" / "moe_mul1.cpp").write_text("")
    calls = []
    monkeypatch.setattr(exl3_ext, "_checked_ext_dir", lambda src: str(ext_dir))
    monkeypatch.setattr(exl3_ext, "check_cpu_compiler", lambda: None)
    # _cxx: the CXX the build saw, which load() reads from the environment.
    monkeypatch.setattr(cpp_extension, "load", lambda **kw: calls.append(dict(kw, _cxx=os.environ.get("CXX"))) or kw)
    exl3_ext.exl3_ext.cache_clear()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(exl3_ext.envs.SGLANG_EXL3_SRC.override(str(tmp_path)))
            stack.enter_context(exl3_ext.envs.SGLANG_EXL3_BUILD_DIR.override(str(tmp_path / "build")))
            # None leaves the flag unset (the caller cleared it).
            if residual is not None:
                stack.enter_context(exl3_ext.envs.SGLANG_EXL3_CPU_ACT_RESIDUAL.override(residual))
            if block is not None:
                stack.enter_context(exl3_ext.envs.SGLANG_EXL3_CPU_ACT_BLOCK.override(block))
            exl3_ext.exl3_ext()
    finally:
        exl3_ext.exl3_ext.cache_clear()
    (kw,) = calls
    return kw, ext_dir


def test_default_build_is_upstreams(tmp_path, monkeypatch):
    kw, ext_dir = _load_args(tmp_path, monkeypatch, False, 0)
    assert kw["name"] == "sglang_exl3_ext"
    assert kw["build_directory"] == str(tmp_path / "build")
    assert kw["sources"] == [str(ext_dir / "cpu" / "moe_mul1.cpp")]
    assert kw["extra_cflags"] == ["-Ofast"]
    assert kw["extra_include_paths"] == [str(ext_dir)]


def test_flavored_build_has_its_own_name_directory_and_kernel(tmp_path, monkeypatch):
    kw, ext_dir = _load_args(tmp_path, monkeypatch, True, 128)
    assert kw["name"] == "sglang_exl3_ext_resid_b128_cpu_v1"
    assert kw["build_directory"] == str(tmp_path / "build" / "resid_b128_cpu_v1")
    assert kw["sources"] == [exl3_ext.OPTIMIZED_CPU_KERNEL, exl3_ext.OPTIMIZED_TORCH_OPS]
    assert kw["extra_cflags"] == ["-Ofast", "-DEXL3_MOE_CPU_ACT_RESIDUAL=1", "-DEXL3_MOE_CPU_ACT_BLOCK=128", "-march=native", "-std=c++20", "-fopenmp", "-pthread"]
    assert kw["extra_ldflags"] == ["-fopenmp"]
    assert kw["extra_include_paths"] == [str(ext_dir), str(ext_dir / "cpu")]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


def test_cpu_experts_build_the_optimized_extension(tmp_path, monkeypatch):
    """With CPU experts on, the build is the optimized flavor even with both accuracy flags left unset."""
    monkeypatch.delenv("SGLANG_EXL3_CPU_ACT_RESIDUAL", raising=False)
    monkeypatch.delenv("SGLANG_EXL3_CPU_ACT_BLOCK", raising=False)
    with exl3_ext.envs.SGLANG_DSV41_CPU_EXPERTS.override(True):
        kw, _ = _load_args(tmp_path, monkeypatch, None, None)
    assert kw["name"] == "sglang_exl3_ext_resid_b128_cpu_v1"
    assert kw["sources"] == [exl3_ext.OPTIMIZED_CPU_KERNEL, exl3_ext.OPTIMIZED_TORCH_OPS]


def test_other_accuracy_flavors_keep_the_generic_kernel(tmp_path, monkeypatch):
    kw, _ = _load_args(tmp_path, monkeypatch, False, 64)
    assert kw["sources"] == [exl3_ext.VENDORED_CPU_KERNEL]
    assert "-fopenmp" not in kw["extra_cflags"]
    assert kw["extra_ldflags"] == []


def test_the_cpu_compiler_is_sglang_exl3_cpu_cxx_over_cxx(monkeypatch):
    monkeypatch.setenv("CXX", "/system/g++")
    asked = []
    monkeypatch.setattr(exl3_ext.subprocess, "check_output", lambda cmd, **kw: asked.append(cmd[0]) or "15.2.1\n")
    with exl3_ext.envs.SGLANG_EXL3_CPU_CXX.override("/gcc15/g++"):
        exl3_ext.check_cpu_compiler()
    assert asked and set(asked) == {"/gcc15/g++"}


def test_only_the_optimized_build_sees_sglang_exl3_cpu_cxx_as_cxx(tmp_path, monkeypatch):
    """The CPU compiler is scoped to the extension's own build: CXX is that compiler during load() and restored after,
    so the server's other JIT builds keep theirs."""
    monkeypatch.setenv("CXX", "/system/g++")
    with exl3_ext.envs.SGLANG_EXL3_CPU_CXX.override("/gcc15/g++"):
        optimized, _ = _load_args(tmp_path / "optimized", monkeypatch, True, 128)
        upstream, _ = _load_args(tmp_path / "upstream", monkeypatch, False, 0)
    assert optimized["_cxx"] == "/gcc15/g++" and upstream["_cxx"] == "/system/g++"
    assert os.environ["CXX"] == "/system/g++"


def test_optimized_compiler_rejects_known_rounding_change(monkeypatch):
    monkeypatch.setenv("CXX", "/new/g++")
    monkeypatch.setattr(exl3_ext.subprocess, "check_output", lambda *args, **kw: "17.0.0\n")
    with pytest.raises(RuntimeError, match="require GCC 15"):
        exl3_ext.check_cpu_compiler()
