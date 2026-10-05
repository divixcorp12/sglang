"""A test that spawns a child Python under a timeout must warm, in the parent, the JIT host modules the child loads: on
a cold cache the child would compile them (50-100 s each, serialized by the JIT build lock), and the timeout would
measure the compiler. Observed 2026-10-04: nine ``run_host_script`` callers hit ``TimeoutExpired`` at 60 s under
``pytest -n 8`` after a flag change gave every module a new key."""

import subprocess

from sglang.kernels.ops.moe import expert_stream_transport as ops
from sglang.test import dsv41_ram_miss_fixtures as fixtures
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _record(monkeypatch):
    events = []
    monkeypatch.setattr(ops, "_host_module", lambda *args: events.append(("load", args)))
    monkeypatch.setattr(
        fixtures.subprocess, "run", lambda *args, **kwargs: events.append(("spawn", args)) or subprocess.CompletedProcess(args, 0)
    )
    return events


def _loads(events):
    return [args for kind, args in events if kind == "load"]


def test_run_host_script_loads_the_childs_host_module_before_spawning_it(monkeypatch, tmp_path):
    events = _record(monkeypatch)
    fixtures.run_host_script(tmp_path, "print('reached')")
    assert [kind for kind, _ in events].count("spawn") == 1
    assert events[-1][0] == "spawn", events  # every load precedes the timed child
    assert ("exl3", fixtures.HOST_SCRIPT_VARIANT, fixtures.HOST_SCRIPT_LANES, 1) in _loads(events)


def test_the_child_script_builds_the_host_the_parent_warmed():
    head = fixtures._HOST_SCRIPT_HEAD
    assert f'variant="{fixtures.HOST_SCRIPT_VARIANT}"' in head
    assert f"wire_layout({fixtures.HOST_SCRIPT_LANES})" in head


def test_a_child_also_loads_the_default_build_through_host_layout(monkeypatch):
    """The child has no conftest, so its ``host_layout()`` loads the default build (prod), whatever variant the test
    names: that module is as cold as the named one."""
    events = _record(monkeypatch)
    fixtures.warm_host_modules("instr", lanes=8, nodes=2)
    assert set(_loads(events)) == {("exl3", "instr", 8, 2), ("exl3", "prod", 8, 1)}


def test_warming_ignores_the_parents_conftest_default_variant(monkeypatch):
    events = _record(monkeypatch)
    monkeypatch.setattr(ops, "_DEFAULT_VARIANT", "instr")  # the conftest's autouse fixture
    fixtures.warm_host_modules()
    assert set(_loads(events)) == {("exl3", "prod", 8, 1)}
    assert ops._DEFAULT_VARIANT == "instr"  # restored


def test_an_aborting_child_is_not_asked_to_write_a_core_file():
    """Every service failure is a deliberate ``std::abort``. On divix01 ``core_pattern`` pipes to systemd-coredump with
    ``ulimit -c unlimited``, and each of these children maps ~440 MB: under ``-n 8`` the dump outlasted the 60 s timeout
    (observed 2026-10-04, with every JIT module already warm)."""
    probe = subprocess.run(
        [sys.executable, "-c", fixtures.NO_CORE_DUMP + "import resource; print(resource.getrlimit(resource.RLIMIT_CORE))"],
        capture_output=True, text=True,
    )
    assert probe.stdout.strip() == "(0, 0)", probe


def test_run_host_script_children_disable_core_dumps(monkeypatch, tmp_path):
    events = _record(monkeypatch)
    fixtures.run_host_script(tmp_path, "print('reached')")
    script = next(args for kind, args in events if kind == "spawn")[0][2]
    assert script.startswith(fixtures.NO_CORE_DUMP)
