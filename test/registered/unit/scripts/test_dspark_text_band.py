"""The DSpark text bar: a run's text may leave the base run's only at a near-tie of the base (§33.2, §33.9) (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _band():
    spec = importlib.util.spec_from_file_location("dspark_text_band", os.path.join(ROOT, "scripts", "dsv41", "dspark_text_band.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokens(*rows):
    return [{"token": chosen, "top": top} for chosen, top in rows]


def test_identical_text_passes_with_no_flips():
    band = _band()
    run = [{"session_id": "s", "tokens": _tokens(("a", [["a", -0.1], ["b", -2.0]]))}]
    assert band.compare(run, run)["pass"] and band.compare(run, run)["flips"] == []


def test_a_flip_within_the_band_passes_and_one_past_it_or_off_the_top_k_fails():
    band = _band()
    base = [{"session_id": "s", "tokens": _tokens(("a", [["a", -0.1], ["b", -1.2], ["c", -3.0]]))}]
    near = [{"session_id": "s", "tokens": _tokens(("b", [["b", -0.2], ["a", -0.3]]))}]
    far = [{"session_id": "s", "tokens": _tokens(("c", [["c", -0.2], ["a", -0.3]]))}]
    off = [{"session_id": "s", "tokens": _tokens(("z", [["z", -0.2], ["a", -0.3]]))}]
    assert band.compare(base, near)["pass"]  # 1.1 behind the base argmax
    assert not band.compare(base, far)["pass"]  # 2.9 behind
    assert not band.compare(base, off)["pass"]  # not in the base's top-k: unbounded
    assert band.compare(base, near)["flips"][0]["gap"] == 1.1


def test_different_prompts_are_refused():
    band = _band()
    try:
        band.compare([{"session_id": "a", "tokens": []}], [{"session_id": "b", "tokens": []}])
    except ValueError:
        return
    raise AssertionError("ran different prompts")
