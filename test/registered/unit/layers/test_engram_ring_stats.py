"""The Engram device-wait counters (ring_stats) are logged per layer: every 60 s as a window, and as a run total at
exit (CPU)."""

import ast
import logging
from pathlib import Path

import pytest

import sglang.srt.layers.engram_file_table as eft
from sglang.srt.layers.engram_file_table import log_ring_stats, ring_layer_totals, ring_stats_line
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _row(layer, waits, spins, spin_ns, spin_us_max):
    return {"layer": layer, "served": waits, "waits": waits, "spins": spins, "spin_ns": spin_ns,
            "spin_us_max": spin_us_max}


class _Extension:
    def __init__(self):
        self.rows = []

    def ring_stats(self):
        return list(self.rows)


class _Stop(Exception):
    pass


def _run_loop(monkeypatch, loop, ext, snapshots):
    """Run the logger's loop for one window per snapshot: each sleep installs the next ring_stats() rows."""
    pending = list(snapshots)

    def sleep(_):
        if not pending:
            raise _Stop
        ext.rows = pending.pop(0)

    monkeypatch.setattr(eft.time, "sleep", sleep)
    with pytest.raises(_Stop):
        loop()


@pytest.fixture
def hooks(monkeypatch):
    """log_ring_stats with its thread and atexit hook captured instead of started."""
    captured = {}

    class _Thread:
        def __init__(self, target, name, daemon):
            captured["loop"], captured["name"] = target, name

        def start(self):
            pass

    monkeypatch.setattr(eft.threading, "Thread", _Thread)
    monkeypatch.setattr(eft.atexit, "register", lambda fn: captured.setdefault("exit", fn))
    return captured


def test_totals_sum_every_lookup_of_a_layer_and_keep_the_largest_spin():
    totals = ring_layer_totals([_row(1, 10, 2, 3000, 2), _row(14, 5, 0, 0, 0), _row(1, 4, 1, 1000, 7)])
    assert totals == {
        1: {"waits": 14, "spins": 3, "spin_ns": 4000, "spin_us_max": 7},
        14: {"waits": 5, "spins": 0, "spin_ns": 0, "spin_us_max": 0},
    }


def test_the_window_line_is_the_difference_and_the_total_line_is_the_run():
    before = {"waits": 100, "spins": 10, "spin_ns": 50_000, "spin_us_max": 9}
    now = {"waits": 300, "spins": 30, "spin_ns": 250_000, "spin_us_max": 12}
    assert ring_stats_line(14, now, before, 60.0) == (
        "engram ring stats layer 14: waits 200, spinning waits 20, spin_ns 200000, mean 1000 ns per wait "
        "(10000 per spinning wait), max 12 us since start, over 60.0 s"
    )
    assert ring_stats_line(1, now, None, 240.5).startswith(
        "engram ring stats layer 1 (run total): waits 300, spinning waits 30, spin_ns 250000, mean 833 ns per wait"
    )


def test_a_layer_that_never_spun_reads_zero_not_a_division_error():
    zero = {"waits": 0, "spins": 0, "spin_ns": 0, "spin_us_max": 0}
    assert "mean 0 ns per wait (0 per spinning wait)" in ring_stats_line(1, zero, None, 1.0)


def test_the_loop_logs_each_layer_that_waited_and_exit_logs_the_run_total(hooks, monkeypatch, caplog):
    ext = _Extension()
    log_ring_stats(ext)
    assert hooks["name"] == "engram-ring-stats"
    caplog.set_level(logging.INFO, logger=eft.logger.name)
    _run_loop(monkeypatch, hooks["loop"], ext, [
        [_row(1, 50, 5, 5000, 3), _row(14, 50, 0, 0, 0)],
        [_row(1, 80, 5, 5000, 3), _row(14, 50, 0, 0, 0)],  # layer 14 did not wait: no line
    ])
    lines = [r.getMessage() for r in caplog.records]
    assert [l.split(":")[0] for l in lines] == [
        "engram ring stats layer 1", "engram ring stats layer 14", "engram ring stats layer 1"]
    assert "waits 30, spinning waits 0, spin_ns 0" in lines[-1]

    caplog.clear()
    hooks["exit"]()
    totals = [r.getMessage() for r in caplog.records]
    assert totals[0].startswith("engram ring stats layer 1 (run total): waits 80, spinning waits 5, spin_ns 5000")
    assert totals[1].startswith("engram ring stats layer 14 (run total): waits 50, spinning waits 0")


def test_exit_falls_back_to_the_last_window_once_the_lookups_are_closed(hooks, monkeypatch, caplog):
    ext = _Extension()
    log_ring_stats(ext)
    _run_loop(monkeypatch, hooks["loop"], ext, [[_row(14, 20, 2, 900, 1)]])
    ext.rows = []
    caplog.set_level(logging.INFO, logger=eft.logger.name)
    caplog.clear()
    hooks["exit"]()
    assert [r.getMessage().split(", spin_ns")[0] for r in caplog.records] == [
        "engram ring stats layer 14 (run total): waits 20, spinning waits 2"]


def test_a_second_registration_of_the_same_extension_is_ignored(hooks):
    ext = _Extension()
    log_ring_stats(ext)
    hooks.clear()
    log_ring_stats(ext)
    assert hooks == {}


def test_open_starts_the_ring_logger_next_to_the_store_logger():
    tree = ast.parse(Path(eft.__file__).read_text())
    open_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "open")
    text = ast.unparse(open_fn)
    assert "log_native_stats(table._native_store)" in text
    assert "log_ring_stats(table._host_node_extension)" in text
