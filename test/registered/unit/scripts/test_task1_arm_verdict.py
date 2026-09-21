"""The Task 1 arm verdict rejects a run that cannot be a baseline, for the reason it names."""

import copy
import importlib.util
import json
import os
import sys

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


def _cache(before, after):
    return {"expert_resident_by_dir_before": before, "expert_resident_by_dir_after": after}


def test_page_cache_growth_rejects_the_arm_and_names_the_directory():
    """Bug: the summed gate said an arm's residency grew 1.40 GiB without saying where."""
    quiet = {"/src": 16 << 30, "/m0": 0, "/m4": 30 << 30}
    assert verdict.check_cache(_cache(quiet, dict(quiet, **{"/m4": (30 << 30) + (1 << 20)}))) == []
    problems = verdict.check_cache(_cache(quiet, dict(quiet, **{"/m4": (31 << 30) + (1 << 29)})))
    assert len(problems) == 1 and "/m4" in problems[0] and "1.50 GiB" in problems[0]


def test_growth_is_judged_per_directory_not_summed_across_a_shrinking_one():
    before = {"/src": 20 << 30, "/m4": 0}
    after = {"/src": 10 << 30, "/m4": 2 << 30}  # the sum fell, but /m4 grew 2 GiB
    assert any("/m4" in p for p in verdict.check_cache(_cache(before, after)))


def test_unmeasured_or_mismatched_residency_is_a_problem():
    assert verdict.check_cache({})
    assert verdict.check_cache(_cache({"/a": 0}, {"/b": 0}))


def test_residency_notes_name_the_session_where_a_directory_changed():
    report = {
        "expert_residency": {"before_engine": {"/m4": 100 << 20}, "after_engine_ready": {"/m4": 100 << 20}},
        "per_session": [
            {"expert_resident_bytes": {"/m4": 100 << 20}},
            {"expert_resident_bytes": {"/m4": (100 << 20) + (1400 << 20)}},
        ],
    }
    notes = verdict.residency_notes(report)
    assert notes == ["residency of /m4 changed +1400.0 MiB at session_1"]
    assert verdict.residency_notes({}) == []


# --- the timed-phase gate: what the independence claim is about ---------------------------------

GiB = 1 << 30
SRC, M0, M4 = "/src", "/m0", "/m4"


def _phased(before, ready, per_session):
    report = _report()
    report["expert_residency"] = {"before_engine": before, "after_engine_ready": ready}
    report["per_session"] = [
        dict(row, expert_resident_bytes=per_session[min(i, len(per_session) - 1)])
        for i, row in enumerate(report["per_session"])
    ]
    return report


def _run_main(monkeypatch, capsys, tmp_path, report, cache=None):
    arm = tmp_path / "arm.json"
    arm.write_text(json.dumps(report))
    argv = ["v", str(arm), "--root", ROOT, "--head", HEAD, "--mirror", "on", "--trace", "T",
            "--summary-json", str(tmp_path / "regime.json")]
    if cache is not None:
        (tmp_path / "cache.json").write_text(json.dumps(cache))
        argv += ["--cache", str(tmp_path / "cache.json")]
    monkeypatch.setattr(sys, "argv", argv)
    code = verdict.main()
    return code, capsys.readouterr().out, json.loads((tmp_path / "regime.json").read_text())


def test_boot_growth_is_a_note_and_the_regime_not_a_failure(monkeypatch, capsys, tmp_path):
    """Boot-time weight loading grew the source dir 5 GiB; the sessions only shrank it. The independence
    claim concerns the timed sessions, so this arm is valid and says which regime it was in."""
    before = {SRC: 16 * GiB, M4: 30 * GiB}
    ready = {SRC: 21 * GiB, M4: 30 * GiB}
    report = _phased(before, ready, [{SRC: 21 * GiB - (600 << 20), M4: 30 * GiB}])
    code, out, summary = _run_main(monkeypatch, capsys, tmp_path, report)
    assert code == 0 and out.rstrip().endswith("VALID")
    assert "REGIME boot-populated" in out and "residency of /src changed +5120.0 MiB at engine_ready" in out
    assert summary["regime"] == "boot-populated" and summary["boot_growth_bytes"][SRC] == 5 * GiB
    assert summary["timed_growth_bytes"][SRC] < 0 and summary["valid"] is True


def test_a_quiet_boot_still_reports_its_regime_and_boot_growth(monkeypatch, capsys, tmp_path):
    """A field that appears only on exception cannot be used to compute a baseline."""
    flat = {SRC: 16 * GiB, M4: 30 * GiB}
    code, out, summary = _run_main(monkeypatch, capsys, tmp_path, _phased(flat, flat, [flat]))
    assert code == 0 and "REGIME boot-warm" in out
    assert summary["regime"] == "boot-warm" and summary["boot_growth_bytes"] == {SRC: 0, M4: 0}


def test_timed_phase_growth_fails_the_arm_and_names_the_directory(monkeypatch, capsys, tmp_path):
    flat = {SRC: 16 * GiB, M4: 30 * GiB}
    leaked = {SRC: 16 * GiB, M4: 31 * GiB + (1 << 29)}
    code, out, summary = _run_main(monkeypatch, capsys, tmp_path, _phased(flat, flat, [flat, leaked]))
    assert code == 1 and "1.50 GiB during the timed sessions in /m4" in out
    assert summary["valid"] is False


def test_an_arm_without_per_session_residency_is_gated_on_the_whole_arm_and_regime_unknown(monkeypatch, capsys, tmp_path):
    """Arm 6 has no per-session samples: it must not be judged under the phase gate."""
    cache = _cache({SRC: 16 * GiB, M4: 30 * GiB}, {SRC: 17 * GiB + (1 << 29), M4: 30 * GiB})
    code, out, summary = _run_main(monkeypatch, capsys, tmp_path, _report(), cache)
    assert code == 1 and "grew 1.50 GiB across the arm in /src" in out
    assert summary["regime"].startswith("unknown") and summary["timed_growth_bytes"] is None


def test_unmeasured_timed_residency_is_a_problem(monkeypatch, capsys, tmp_path):
    ok = {SRC: 16 * GiB}
    code, out, _ = _run_main(monkeypatch, capsys, tmp_path, _phased(ok, ok, [{SRC: None}]))
    assert code == 1 and "could not be measured" in out
