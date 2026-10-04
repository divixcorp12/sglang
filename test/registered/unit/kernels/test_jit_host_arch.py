"""CPU-only tests for the JIT host build flags: the resolved -march and the host compiler.

`-march=native` must never reach build.ninja, because the build key hashes that text
and a cached .so built for one CPU would then match the key on another and SIGILL.
The host compiler is asked for the concrete name `native` stands for.
"""

from __future__ import annotations

import logging
import stat
import textwrap

import pytest

from sglang.kernels.jit.utils.compile import cache, ninja, toolchain
from sglang.kernels.jit.utils.compile.spec import BuildSpec
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The real `-Q --help=target` line, tabs included.
_FAKE_COMPILER = textwrap.dedent(
    """\
    #!/bin/sh
    case "$*" in
      *"-Q --help=target"*)
        {help_branch}
        ;;
      *"-E -dM"*)
        case "$*" in
          *-march=native*) echo "{native_macros}" ;;
          *) echo "{named_macros}" ;;
        esac
        ;;
    esac
    """
)


def _fake_compiler(
    tmp_path,
    *,
    help_branch,
    named_macros="#define __AVX512F__ 1",
    native_macros="#define __AVX512F__ 1",
):
    path = tmp_path / "fake-cxx"
    path.write_text(
        _FAKE_COMPILER.format(
            help_branch=help_branch,
            named_macros=named_macros,
            native_macros=native_macros,
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


_ANSWERS_SKYLAKE = "printf '  -march=\\t\\t\\t\\tskylake-avx512\\n'"


def _resolve(monkeypatch, compiler):
    # `cache_once` keeps its results for the process; the undecorated function is the case under test.
    monkeypatch.setattr(toolchain, "host_compiler_path", lambda: compiler)
    return toolchain.host_arch_flags.__wrapped__()


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("SGLANG_JIT_HOST_MARCH", raising=False)
    monkeypatch.setattr(toolchain, "is_hip_runtime", lambda: False)


def test_native_resolves_to_the_concrete_name(tmp_path, monkeypatch):
    compiler = _fake_compiler(tmp_path, help_branch=_ANSWERS_SKYLAKE)
    assert _resolve(monkeypatch, compiler) == ["-march=skylake-avx512"]


def test_a_name_with_different_macros_is_refused_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    compiler = _fake_compiler(
        tmp_path,
        help_branch=_ANSWERS_SKYLAKE,
        named_macros="#define __AVX512F__ 1\n#define __AVX512VNNI__ 1",
    )
    with caplog.at_level(logging.WARNING, logger=toolchain.logger.name):
        assert _resolve(monkeypatch, compiler) == []
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 1


def test_a_name_that_is_a_subset_of_native_is_kept(tmp_path, monkeypatch):
    # divix01: native adds __ABM__ and __RTM__ over skylake-avx512, and the name adds __SGX__
    # (enclave instructions no codegen emits), so exact equality refuses a name that is safe.
    compiler = _fake_compiler(
        tmp_path,
        help_branch=_ANSWERS_SKYLAKE,
        native_macros="#define __AVX512F__ 1\n#define __ABM__ 1\n#define __RTM__ 1",
        named_macros="#define __AVX512F__ 1\n#define __SGX__ 1",
    )
    assert _resolve(monkeypatch, compiler) == ["-march=skylake-avx512"]


def test_a_compiler_that_cannot_answer_gives_the_default_arch(
    tmp_path, monkeypatch, caplog
):
    compiler = _fake_compiler(tmp_path, help_branch="exit 1")
    with caplog.at_level(logging.WARNING, logger=toolchain.logger.name):
        assert _resolve(monkeypatch, compiler) == []
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 1


def test_a_missing_compiler_gives_the_default_arch(tmp_path, monkeypatch):
    assert _resolve(monkeypatch, str(tmp_path / "absent")) == []


def test_hip_takes_no_host_arch(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "is_hip_runtime", lambda: True)
    compiler = _fake_compiler(tmp_path, help_branch=_ANSWERS_SKYLAKE)
    assert _resolve(monkeypatch, compiler) == []


def test_override_default_keeps_the_compilers_arch(tmp_path, monkeypatch):
    compiler = _fake_compiler(tmp_path, help_branch=_ANSWERS_SKYLAKE)
    with envs.SGLANG_JIT_HOST_MARCH.override("default"):
        assert _resolve(monkeypatch, compiler) == []


def test_override_names_the_arch_verbatim(tmp_path, monkeypatch):
    # The compiler is not consulted: the point is building on one machine for another.
    with envs.SGLANG_JIT_HOST_MARCH.override("x86-64-v4"):
        assert _resolve(monkeypatch, str(tmp_path / "absent")) == ["-march=x86-64-v4"]


# ---------------------------------------------------------------------------
# What reaches build.ninja and the key
# ---------------------------------------------------------------------------


def _spec() -> BuildSpec:
    return BuildSpec(
        module_args=("host_arch",),
        cpp_files=(),
        cuda_files=(),
        cpp_wrappers=(("run", "Kernel::run"),),
        cuda_wrappers=(("run_cu", "Kernel::run_cu"),),
        cflags=("-O3",),
        cuda_cflags=("-O3",),
        ldflags=(),
        include_paths=(),
        header_only=True,
    )


def _lines(text: str) -> dict:
    return {
        line.split(" = ", 1)[0]: line.split(" = ", 1)[1]
        for line in text.splitlines()
        if " = " in line
    }


def _generate(monkeypatch, *, march, cxx):
    monkeypatch.setattr(toolchain, "host_arch_flags", lambda: [f"-march={march}"])
    monkeypatch.setattr(toolchain, "host_compiler_path", lambda: cxx)
    return ninja.generate(_spec())


def test_both_halves_get_the_resolved_arch_and_nvcc_the_host_compiler(monkeypatch):
    text = _generate(monkeypatch, march="skylake-avx512", cxx="/opt/gcc15/bin/g++")
    fields = _lines(text)
    assert "-march=skylake-avx512" in fields["cxxflags"].split()
    assert "-Xcompiler -march=skylake-avx512" in fields["cudaflags"]
    assert "-ccbin /opt/gcc15/bin/g++" in fields["cudaflags"]
    assert "-march=native" not in text


def test_the_resolved_arch_and_the_compiler_reach_the_build_key(monkeypatch):
    def key(**kwargs):
        text = _generate(monkeypatch, **kwargs)
        return cache.compute_build_key(_spec(), build_file=text)

    base = key(march="skylake-avx512", cxx="/opt/gcc15/bin/g++")
    assert base == key(march="skylake-avx512", cxx="/opt/gcc15/bin/g++")
    assert base != key(march="x86-64-v4", cxx="/opt/gcc15/bin/g++")
    assert base != key(march="skylake-avx512", cxx="/usr/bin/g++")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
