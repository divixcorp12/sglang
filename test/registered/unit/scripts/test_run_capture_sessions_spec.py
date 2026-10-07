"""The session driver records a speculative server's per-request details (accept length's inputs) (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _driver():
    path = os.path.join(ROOT, "scripts", "expert_prediction", "benchmarks", "run_capture_sessions.py")
    spec = importlib.util.spec_from_file_location("run_capture_sessions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sglext_chunk_carries_the_spec_details():
    driver = _driver()
    chunk = {"choices": [], "sglext": {"spec_tokens_details": {"spec_verify_ct": 40, "spec_accept_length": 3.2}}}
    assert driver.chunk_spec_details(chunk) == {"spec_verify_ct": 40, "spec_accept_length": 3.2}
    assert driver.chunk_spec_details({"choices": [{"delta": {}}]}) is None
    assert driver.chunk_spec_details({"choices": [], "sglext": {"cached_tokens_details": {}}}) is None
