"""The corpus driver's session selection and CLI (CPU; the Engine run is Window C's)."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41")
sys.path.insert(0, SCRIPTS)

import trace_corpus  # noqa: E402


def test_first_turns_skip_then_take(tmp_path):
    sessions = tmp_path / "sessions.jsonl"
    sessions.write_text("\n".join(json.dumps({"turns": [f"s{i}", "later"]}) for i in range(6)) + "\n")
    assert list(trace_corpus._first_turns(str(sessions), 3)) == ["s0", "s1", "s2"]
    assert list(trace_corpus._first_turns(str(sessions), 2, skip=3)) == ["s3", "s4"]
    assert list(trace_corpus._first_turns(str(sessions), 5, skip=4)) == ["s4", "s5"]


def test_engine_and_sampling_settings():
    args = SimpleNamespace(model="/m", mem_fraction_static=0.85, chunked_prefill_size=512, new_tokens=128)
    kwargs = trace_corpus.engine_kwargs(args)
    assert kwargs["disable_cuda_graph"] and kwargs["disable_shared_experts_fusion"]
    assert kwargs["disable_radix_cache"]
    assert kwargs["expert_distribution_recorder_mode"] == "per_pass"
    assert kwargs["max_running_requests"] == 4
    assert trace_corpus.sampling_params(args) == {"max_new_tokens": 128, "temperature": 0, "ignore_eos": True}


def test_help_lists_the_options():
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "trace_corpus.py"), "--help"],
        capture_output=True, text=True, check=True,
    )
    for flag in ("--model", "--sessions", "--skip", "--new-tokens", "--out"):
        assert flag in result.stdout


def test_time_stream_reports_ttft_and_decode_rate():
    ticks = iter([10.0, 12.0, 14.0])  # start, first chunk, done
    timing = trace_corpus.time_stream(iter(["a", "b", "c"]), new_tokens=5, clock=lambda: next(ticks))
    assert timing == {"ttft_s": 2.0, "decode_tok_s": 4 / 2.0}


def test_time_stream_zero_decode_time_gives_zero_rate():
    ticks = iter([0.0, 1.0, 1.0])
    timing = trace_corpus.time_stream(iter(["a"]), new_tokens=5, clock=lambda: next(ticks))
    assert timing == {"ttft_s": 1.0, "decode_tok_s": 0.0}


def test_time_stream_that_yields_nothing_is_an_error():
    with pytest.raises(RuntimeError, match="no chunks"):
        trace_corpus.time_stream(iter([]), new_tokens=5)


def test_mean_decode_tok_s():
    sessions = [{"decode_tok_s": 2.0}, {"decode_tok_s": 4.0}]
    assert trace_corpus.mean_decode_tok_s(sessions) == 3.0


def test_mean_decode_tok_s_with_no_sessions_is_an_error():
    with pytest.raises(ValueError, match="no sessions"):
        trace_corpus.mean_decode_tok_s([])


def test_graphs_switch_to_breakable_decode_at_batch_size_one():
    args = SimpleNamespace(model="/m", mem_fraction_static=0.8, chunked_prefill_size=512, new_tokens=128, graphs=True)
    kwargs = trace_corpus.engine_kwargs(args)
    assert "disable_cuda_graph" not in kwargs
    assert kwargs["cuda_graph_backend_decode"] == "breakable"
    assert kwargs["cuda_graph_backend_prefill"] == "disabled"
    assert kwargs["cuda_graph_bs_decode"] == [1] and kwargs["cuda_graph_max_bs_decode"] == 1
    # A graph run's checks read info logs: the capture line, the RAM-miss thread start,
    # the hot cache's startup slots (the Engine's default level is error).
    assert kwargs["log_level"] == "info"
    eager = trace_corpus.engine_kwargs(SimpleNamespace(**{**vars(args), "graphs": False}))
    assert eager["disable_cuda_graph"] is True


def test_dspark_sets_speculative_kwargs_and_forces_eager():
    args = SimpleNamespace(
        model="/m", mem_fraction_static=0.85, chunked_prefill_size=512,
        new_tokens=128, dspark="/draft/dir",
    )
    kwargs = trace_corpus.engine_kwargs(args)
    assert kwargs["speculative_algorithm"] == "DSPARK"
    assert kwargs["speculative_draft_model_path"] == "/draft/dir"
    assert kwargs["speculative_dspark_block_size"] == 5
    assert kwargs["disable_cuda_graph"] is True


def test_dspark_overrides_graphs_since_the_gate_refuses_speculation_under_a_graph():
    args = SimpleNamespace(
        model="/m", mem_fraction_static=0.85, chunked_prefill_size=512,
        new_tokens=128, graphs=True, dspark="/draft/dir",
    )
    kwargs = trace_corpus.engine_kwargs(args)
    assert kwargs["disable_cuda_graph"] is True
    assert "cuda_graph_backend_decode" not in kwargs
    assert kwargs["speculative_algorithm"] == "DSPARK"


def test_time_stream_captures_completion_tokens_and_spec_verify_ct_from_meta_info():
    chunks = [
        {"meta_info": {"completion_tokens": 1}},
        {"meta_info": {"completion_tokens": 5, "spec_verify_ct": 2}},
    ]
    ticks = iter([10.0, 12.0, 14.0])
    timing = trace_corpus.time_stream(iter(chunks), new_tokens=5, clock=lambda: next(ticks))
    assert timing["completion_tokens"] == 5
    assert timing["spec_verify_ct"] == 2


def test_time_stream_without_meta_info_omits_the_spec_fields():
    ticks = iter([10.0, 12.0, 14.0])
    timing = trace_corpus.time_stream(iter(["a", "b", "c"]), new_tokens=5, clock=lambda: next(ticks))
    assert "completion_tokens" not in timing
    assert "spec_verify_ct" not in timing


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
