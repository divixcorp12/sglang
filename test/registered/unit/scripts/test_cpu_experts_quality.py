"""CPU experts' quality gate comparison (scripts/dsv41/cpu_experts_quality.py)."""

import importlib.util
import math
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
_spec = importlib.util.spec_from_file_location(
    "cpu_experts_quality", os.path.join(_ROOT, "scripts", "dsv41", "cpu_experts_quality.py")
)
quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quality)


def _tok(token, *top):
    return {"token": token, "top": [list(t) for t in top]}


def _probe(*tokens, session="s"):
    return [{"session_id": session, "tokens": list(tokens)}]


def test_identical_runs_match_with_zero_kl():
    run = _probe(_tok("a", ("a", -0.1), ("b", -2.5)), _tok("c", ("c", -0.2), ("d", -1.9)))
    result = quality.compare(run, run)
    assert result["greedy_match_rate"] == 1 and result["flip_positions"] == [] and result["e31_pass"]
    assert result["kl_positions"] == 2 and result["kl_max"] == pytest.approx(0)
    assert result["chosen_logprob_abs_delta_max"] == pytest.approx(0)


def test_kl_is_over_the_shared_top_k_renormalised_and_stops_after_the_first_flip():
    off = _probe(_tok("a", ("a", math.log(0.6)), ("b", math.log(0.3)), ("z", math.log(0.05))), _tok("x", ("x", -0.01)))
    on = _probe(_tok("b", ("b", math.log(0.5)), ("a", math.log(0.4)), ("y", math.log(0.05))), _tok("x", ("x", -9)))
    result = quality.compare(off, on)
    p, q = [2 / 3, 1 / 3], [4 / 9, 5 / 9]  # a, b renormalised over the two tokens both list
    assert result["kl_positions"] == 1
    assert result["kl_max"] == pytest.approx(sum(pi * math.log(pi / qi) for pi, qi in zip(p, q)))
    assert result["flip_positions"] == [0] and result["greedy_match_rate"] == 0
    assert result["chosen_logprob_abs_delta_max"] == pytest.approx(abs(math.log(0.6) - math.log(0.4)))
    # margins 0.69 (off) and 0.22 (on): the on run's near-tie starts the flip, so E31 holds.
    assert result["e31_pass"]


def test_a_flip_past_a_clear_margin_in_both_runs_fails_e31():
    off = _probe(_tok("a", ("a", -0.05), ("b", -3.0)))
    on = _probe(_tok("b", ("b", -0.05), ("a", -3.0)))
    assert not quality.compare(off, on)["e31_pass"]


def test_different_prompts_are_refused():
    with pytest.raises(ValueError, match="different prompts"):
        quality.compare(_probe(session="s1"), _probe(session="s2"))
