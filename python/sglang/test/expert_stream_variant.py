"""Expert-stream tests load the instrumented host build unless a test names a variant: most use its test-only entry
points (faults, the stage trace). Plan 2026-09-29-hotpath-zero-overhead Task 8.

A conftest takes the default in one line: ``from sglang.test.expert_stream_variant import *``.

Hooks, not an autouse fixture: pytest 9.1 binds a conftest's fixtures to the first Directory node collected for its
path, so a command line that revisits the directory (kernels/a.py, ../b.py, kernels/c.py) collects a second node that
sees none of them, and c.py would load the production build. A conftest's hooks are looked up by the item's path, which
that does not affect (test_expert_stream_variant_conftest.py)."""

import pytest

__all__ = ["pytest_runtest_setup", "pytest_runtest_teardown"]

_saved = []


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # tryfirst: before any fixture of the item runs, so a fixture that builds a host builds the instrumented one.
    from sglang.kernels.ops.moe import expert_stream_transport as ops

    _saved.append(ops._DEFAULT_VARIANT)
    ops._DEFAULT_VARIANT = "instr"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    try:
        return (yield)
    finally:
        # After every fixture's teardown, a test's own monkeypatch of the variant included.
        from sglang.kernels.ops.moe import expert_stream_transport as ops

        if _saved:
            ops._DEFAULT_VARIANT = _saved.pop()
