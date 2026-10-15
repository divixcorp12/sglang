"""CPU-only tests for `warm.load_in_parallel`: cold JIT builds in child processes.

The loads are fakes that consult a flag file the way `load_jit` consults the
cache, so the probe/build/load sequence is checked without a compiler.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from sglang.kernels.jit.utils.compile import warm
from sglang.kernels.jit.utils.compile.loader import JitCompileForbidden
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_FAKE = '''
import os, pathlib
from sglang.kernels.jit.utils.compile.loader import JitCompileForbidden
from sglang.srt.environ import envs

def load(root, name):
    built = pathlib.Path(root) / name
    if not built.exists():
        if envs.SGLANG_CRASH_ON_JIT_COMPILE.get():
            raise JitCompileForbidden(name)
        built.write_text(str(os.getpid()))
    with open(pathlib.Path(root) / "loads", "a") as f:
        f.write(f"{name} {os.getpid()}\\n")
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """An importable module `fake_jit_loads` whose `load(root, name)` builds
    `root/name` unless the compile-forbidding flag is set, and logs each load."""
    (tmp_path / "fake_jit_loads.py").write_text(_FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    # Children find the module through PYTHONPATH, as they find sglang.
    monkeypatch.setenv(
        "PYTHONPATH", f"{tmp_path}:{':'.join(p for p in sys.path if p)}"
    )
    root = tmp_path / "cache"
    root.mkdir()
    return root


def _loads(root: pathlib.Path):
    lines = (root / "loads").read_text().split() if (root / "loads").exists() else []
    return list(zip(lines[::2], map(int, lines[1::2])))


def test_all_cached_starts_no_process(fake, monkeypatch):
    (fake / "a").write_text("")
    (fake / "b").write_text("")
    monkeypatch.setattr(warm, "_build_in_child", lambda load: pytest.fail("spawned"))
    warm.load_in_parallel(
        [("fake_jit_loads", "load", (str(fake), n)) for n in ("a", "b")]
    )
    assert [name for name, _ in _loads(fake)] == ["a", "b"]


def test_misses_are_built_in_children_then_loaded_here(fake):
    import os

    (fake / "a").write_text("")
    warm.load_in_parallel(
        [("fake_jit_loads", "load", (str(fake), n)) for n in ("a", "b", "c", "b")]
    )
    # Built by children, not by this process.
    for name in ("b", "c"):
        assert int((fake / name).read_text()) != os.getpid()
    here = [name for name, pid in _loads(fake) if pid == os.getpid()]
    # The hit loads during the probe; each miss loads once it is built.
    assert here == ["a", "b", "c"]


def test_a_failed_child_leaves_the_error_to_the_load_here(fake, monkeypatch):
    """A child's failure is logged only: the load in this process repeats the
    build and raises with the compiler's full output."""
    calls = []
    monkeypatch.setattr(warm, "_build_in_child", calls.append)
    warm.load_in_parallel([("fake_jit_loads", "load", (str(fake), "a"))])
    assert calls == [("fake_jit_loads", "load", (str(fake), "a"))]
    assert (fake / "a").exists()  # built here, after the child did nothing


def test_child_failure_does_not_raise(fake):
    warm._build_in_child(("fake_jit_loads", "no_such_function", ()))


@pytest.mark.parametrize(
    "flag", ["SGLANG_CRASH_ON_JIT_COMPILE", "SGLANG_JIT_FORCE_RECOMPILE"]
)
def test_flags_that_rule_out_children_load_in_order(fake, monkeypatch, flag):
    monkeypatch.setattr(warm, "_build_in_child", lambda load: pytest.fail("spawned"))
    (fake / "a").write_text("")
    with getattr(envs, flag).override(True):
        if flag == "SGLANG_CRASH_ON_JIT_COMPILE":
            with pytest.raises(JitCompileForbidden):
                warm.load_in_parallel(
                    [("fake_jit_loads", "load", (str(fake), n)) for n in ("a", "b")]
                )
        else:
            warm.load_in_parallel(
                [("fake_jit_loads", "load", (str(fake), n)) for n in ("a", "b")]
            )
    assert [name for name, _ in _loads(fake)] == ["a"] + (
        ["b"] if flag == "SGLANG_JIT_FORCE_RECOMPILE" else []
    )
