"""logprob_probe's request body: greedy and non-thinking by default, sampled and thinking for a prompts file (CPU)."""

import argparse
import importlib.util
import json
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _probe():
    spec = importlib.util.spec_from_file_location(
        "logprob_probe", os.path.join(ROOT, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**kw):
    base = dict(temperature=0, top_p=None, max_tokens=96, top_logprobs=2, prompts_file=[], prompts=6, sessions=None)
    return argparse.Namespace(**(base | kw))


def test_the_defaults_are_the_greedy_non_thinking_logprob_request():
    body = _probe().request_body(_args(), "hi")
    assert body == {
        "model": "default", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 96,
        "logprobs": True, "top_logprobs": 2, "chat_template_kwargs": {"enable_thinking": False},
    }


def test_the_sampling_flags_reach_the_payload_and_a_prompts_file_turns_thinking_on():
    body = _probe().request_body(
        _args(temperature=0.6, top_p=0.95, max_tokens=768, top_logprobs=0, prompts_file=["x.json"]), "hi")
    assert body["temperature"] == 0.6 and body["top_p"] == 0.95 and body["max_tokens"] == 768
    assert "logprobs" not in body and "top_logprobs" not in body
    assert body["chat_template_kwargs"] == {"thinking": True}


def test_prompts_files_are_read_in_order_and_ids_name_file_and_index(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(["p0", "p1"]))
    b.write_text(json.dumps(["q0"]))
    got = _probe().sessions_from(_args(prompts_file=[str(a), str(b)], prompts=10**6))
    assert got == [("a.json:0", "p0"), ("a.json:1", "p1"), ("b.json:0", "q0")]
