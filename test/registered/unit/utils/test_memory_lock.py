"""srt/utils/memory_lock.py (SGLANG_ENABLE_SCHEDULER_MLOCK): one mlockall with MCL_CURRENT|MCL_FUTURE|MCL_ONFAULT,
a failure warned and never raised, nothing when off (CPU, no server)."""

import ast
import logging
from pathlib import Path

import pytest

import sglang.srt.utils.memory_lock as ml
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(ml, "_locked", False)


def test_the_lock_is_called_once_with_current_future_and_onfault(caplog):
    calls = []
    log = logging.getLogger("t")
    with caplog.at_level(logging.INFO):
        assert ml.lock_process_memory(log, mlockall=lambda flags: calls.append(flags) or 0)
        assert ml.lock_process_memory(log, mlockall=lambda flags: calls.append(flags) or 0)
    assert calls == [ml.MCL_CURRENT | ml.MCL_FUTURE | ml.MCL_ONFAULT] == [7]
    assert "VmLck" in caplog.text


def test_a_failure_is_warned_and_not_raised(caplog):
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING):
        assert not ml.lock_process_memory(log, mlockall=lambda flags: -1)
    assert "failed" in caplog.text
    assert not ml._locked  # a later call may retry


def test_off_by_default_nothing_is_locked(monkeypatch):
    called = []
    monkeypatch.setattr(ml, "lock_process_memory", lambda log, mlockall=None: called.append(1) or True)
    assert not envs.SGLANG_ENABLE_SCHEDULER_MLOCK.get()
    assert not ml.maybe_lock_process_memory(logging.getLogger("t")) and not called
    with envs.SGLANG_ENABLE_SCHEDULER_MLOCK.override(True):
        assert ml.maybe_lock_process_memory(logging.getLogger("t")) and called == [1]


def test_locked_bytes_reads_vmlck(tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmLck:\t  2048 kB\n")
    assert ml.locked_bytes(str(status)) == 2048 * 1024
    assert ml.locked_bytes(str(tmp_path / "missing")) is None


def test_the_scheduler_process_locks_right_after_building_the_scheduler():
    import sglang.srt.managers.scheduler as scheduler

    tree = ast.parse(Path(scheduler.__file__).read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_scheduler_process")
    text = ast.unparse(run)
    assert text.index("scheduler = Scheduler(") < text.index("maybe_lock_process_memory(logger)") < text.index(
        "pipe_writer.send(scheduler.get_init_info())")
