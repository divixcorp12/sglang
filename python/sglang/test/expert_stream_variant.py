"""Expert-stream tests load the instrumented host build unless a test names a variant: most use its test-only entry
points (faults, the stage trace). Plan 2026-09-29-hotpath-zero-overhead Task 8.

A conftest takes the fixture in one line: ``from sglang.test.expert_stream_variant import instrumented_expert_stream_host``
(pytest collects an imported fixture as if the conftest defined it)."""

import pytest


@pytest.fixture(autouse=True)
def instrumented_expert_stream_host(monkeypatch):
    from sglang.kernels.ops.moe import expert_stream_transport as ops

    monkeypatch.setattr(ops, "_DEFAULT_VARIANT", "instr")
