"""The Task 1 arm verdict rejects a run that cannot be a baseline, for the reason it names."""

import copy
import importlib.util
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "task1_arm_verdict.py")
_spec = importlib.util.spec_from_file_location("task1_arm_verdict", _PATH)
verdict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verdict)

ROOT, HEAD = "/wt/new", "b" * 40
MIRRORS = "/mnt/nvme0/x:/mnt/nvme4/x"


def _report(mirror=True, traced=True):
    env = {"SGLANG_MOE_EXPERT_GRAPH_GATHER": "1"}
    if mirror:
        env[verdict.MIRRORS_ENV] = MIRRORS
    if traced:
        env[verdict.TRACE_ENV] = "/out/x.trace"
    session = {"cpu_s": 3.0, "step_latency": {"steps": 127, "multi_token_chunks": 0, "step_s_p50": 0.3}}
    return {
        "provenance": {
            "sglang_file": f"{ROOT}/python/sglang/__init__.py",
            "git": {"head": HEAD, "dirty": False, "untracked_in_package_count": 0},
            "unavailable": {},
            "sglang_env": env,
            "sglang_env_resolved": {verdict.READER: "uring_direct", "SGLANG_MOE_EXPERT_GRAPH_GATHER": True},
            "drive_idle_check": {"idle": True},
            "sglang_env_drift_at_engine_ready": {},
        },
        "per_session": [copy.deepcopy(session) for _ in range(4)],
    }


def _check(report, **kw):
    kw = {"mirror": True, "traced": True, **kw}
    return verdict.check_arm(report, root=ROOT, head=HEAD, **kw)


def test_a_clean_arm_is_valid():
    assert _check(_report()) == ([], [])
    assert _check(_report(mirror=False, traced=False), mirror=False, traced=False)[0] == []


def _mutate(path, value):
    def apply(r):
        node = r
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return r
    return apply


@pytest.mark.parametrize(
    "mutation, needle",
    [
        (_mutate(["provenance", "sglang_file"], "/wt/dirty-wip/python/sglang/__init__.py"), "imported sglang from"),
        (_mutate(["provenance", "git", "head"], "c" * 40), "git head"),
        (_mutate(["provenance", "git", "dirty"], True), "not clean"),
        (_mutate(["provenance", "sglang_env_resolved", verdict.READER], "mmap"), "page cache"),
        (_mutate(["provenance", "drive_idle_check", "idle"], False), "not idle"),
        (_mutate(["provenance", "sglang_env", verdict.MIRRORS_ENV], ""), "mirrors on"),
        (_mutate(["provenance", "sglang_env", verdict.TRACE_ENV], ""), "traced=True"),
        (_mutate(["provenance", "unavailable"], {"git": "x"}), "unavailable"),
        (_mutate(["per_session", 0, "step_latency"], {"unavailable": "no counts"}), "no step latency"),
        (_mutate(["per_session"], []), "0 sessions"),
    ],
)
def test_each_way_an_arm_can_be_unattributable_is_named(mutation, needle):
    problems, _ = _check(mutation(_report()))
    assert any(needle in p for p in problems), problems


def test_a_filter_that_over_redacts_fails_the_arm():
    """Bug: the first Task 1 arm lost 35 knob values to a substring TOKEN filter and still passed."""
    r = _report()
    r["provenance"]["sglang_env_resolved"]["SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS"] = verdict.REDACTED
    r["provenance"]["sglang_env_resolved"]["EXA_API_KEY"] = verdict.REDACTED  # a real secret is fine
    problems, _ = _check(r)
    assert any("PREFILL_TOKENS" in p and "not secrets" in p for p in problems)
    assert not any("EXA_API_KEY" in p for p in problems)


def test_the_verdict_and_provenance_agree_on_what_a_secret_is():
    prov_spec = importlib.util.spec_from_file_location(
        "provenance", os.path.join(os.path.dirname(_PATH), "..", "..", "scripts", "dsv41", "provenance.py")
    )
    prov = importlib.util.module_from_spec(prov_spec)
    prov_spec.loader.exec_module(prov)
    assert verdict.SECRET_NAME.pattern == prov._SECRET_NAME.pattern
    assert verdict.SECRET_NAME.flags == prov._SECRET_NAME.flags
    assert verdict.REDACTED == prov.REDACTED


def test_a_result_without_provenance_is_refused():
    problems, _ = _check({"per_session": []})
    assert problems and "no provenance" in problems[0]


def test_mirrors_off_arm_must_not_have_the_mirror_variable():
    problems, _ = _check(_report(mirror=True), mirror=False)
    assert any("mirrors off" in p for p in problems)


def test_multi_token_chunks_are_a_note_not_a_failure():
    r = _report()
    r["per_session"][1]["step_latency"]["multi_token_chunks"] = 3
    problems, notes = _check(r)
    assert problems == [] and "3 chunks" in notes[0]


def test_expert_page_cache_growth_rejects_the_arm():
    assert verdict.check_cache({"expert_resident_bytes_before": 0, "expert_resident_bytes_after": 1 << 20}) == []
    assert verdict.check_cache({"expert_resident_bytes_before": 0, "expert_resident_bytes_after": 5 << 30})
    assert verdict.check_cache({"expert_resident_bytes_before": None, "expert_resident_bytes_after": 0})
