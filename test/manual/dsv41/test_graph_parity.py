"""The R4 eager-versus-graph comparison (CPU; the Engine runs are the window's)."""

import os
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41")
sys.path.insert(0, SCRIPTS)

import graph_parity  # noqa: E402


def _run(tokens, logprobs):
    return {"tokens": tokens, "logprobs": logprobs}


def test_identical_runs_pass():
    report = graph_parity.compare(_run([1, 2, 3], [-0.1, -0.2, -0.3]), _run([1, 2, 3], [-0.1, -0.2, -0.3]))
    assert report["pass"] and report["first_token_mismatch"] is None and report["max_abs_dlogprob"] == 0.0


def test_a_token_mismatch_fails_and_names_the_step():
    report = graph_parity.compare(_run([1, 2, 3], [-0.1] * 3), _run([1, 9, 3], [-0.1] * 3))
    assert not report["pass"] and report["first_token_mismatch"] == 1


def test_logprob_drift_is_bounded_over_the_common_prefix():
    report = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2 - 2e-3]))
    assert not report["pass"] and report["max_abs_dlogprob"] == pytest.approx(2e-3)
    ok = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2 - 5e-4]))
    assert ok["pass"]


def test_the_bitwise_gate_has_zero_tolerance():
    same = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2]), logprob_tol=0.0)
    off = graph_parity.compare(_run([1, 2], [-0.1, -0.2]), _run([1, 2], [-0.1, -0.2 + 1e-9]), logprob_tol=0.0)
    assert same["pass"] and not off["pass"]


def test_each_arm_sets_its_own_graph_gather_env():
    # The eager and control Engines must launch with graph gather off: the gate refuses
    # graph gather without decode graphs.
    assert graph_parity.run_env("eager", True) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "0"}
    assert graph_parity.run_env("control", True) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "0"}
    assert graph_parity.run_env("graph", True) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1"}
    assert graph_parity.run_env("graph", False) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "0"}
    assert graph_parity.run_env("debug", False) == {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1"}
    with pytest.raises(ValueError):
        graph_parity.run_env("full", True)


def test_each_arm_builds_its_engine():
    eager = graph_parity.engine_kwargs("/m", "eager", 0.8)
    graph = graph_parity.engine_kwargs("/m", "graph", 0.8)
    debug = graph_parity.engine_kwargs("/m", "debug", 0.8)
    assert eager["disable_cuda_graph"] and "cuda_graph_backend_decode" not in eager
    assert graph["cuda_graph_backend_decode"] == "breakable" and "debug_cuda_graph" not in graph
    assert debug["debug_cuda_graph"] and debug["cuda_graph_max_bs_decode"] == 1
    # The capture line ("Breakable CUDA graph captured: ... breaks=N") is an info log.
    assert graph["log_level"] == "info" and debug["log_level"] == "info"


def test_decode_sets_the_env_for_the_engine_and_restores_it(monkeypatch):
    import types

    seen = {}

    class _Engine:
        def __init__(self, **kwargs):
            seen["env"] = os.environ.get("SGLANG_MOE_EXPERT_GRAPH_GATHER")
            seen["kwargs"] = kwargs

        def generate(self, **kwargs):
            return {"meta_info": {"output_token_logprobs": [(-0.5, 7, None)]}}

        def shutdown(self):
            seen["shutdown"] = True

    monkeypatch.setitem(sys.modules, "sglang", types.SimpleNamespace(Engine=_Engine))
    monkeypatch.setenv("SGLANG_MOE_EXPERT_GRAPH_GATHER", "1")
    run = graph_parity._decode("/m", [1, 2], 1, "eager", True, 0.8)
    assert seen["env"] == "0" and seen["shutdown"] and run == {"tokens": [7], "logprobs": [-0.5]}
    assert os.environ["SGLANG_MOE_EXPERT_GRAPH_GATHER"] == "1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
