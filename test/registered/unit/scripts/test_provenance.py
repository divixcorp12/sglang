"""Arm provenance: what the running process saw, with no key silently missing."""

import importlib.util
import json
import os
import subprocess

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")


def _load(name, *parts):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, *parts))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prov = _load("provenance", "scripts", "dsv41", "provenance.py")

REQUIRED_KEYS = {
    "schema", "host", "utc", "pid", "argv", "cwd", "python", "harness_files", "sglang_env",
    "sglang_env_at_exec", "sglang_file", "sglang_env_resolved", "git", "unavailable",
}


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "python" / "pkg").mkdir(parents=True)
    (tmp_path / "python" / "pkg" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", "python/pkg/a.py")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return tmp_path


def test_process_env_reports_only_steering_variables_and_redacts_secrets(monkeypatch):
    for k in [k for k in os.environ if k.startswith(("SGLANG_", "SGL_"))]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("SGLANG_MOE_EXPERT_FILE_READER", "uring_direct")
    monkeypatch.setenv("SGL_LEGACY_ALIAS", "1")
    monkeypatch.setenv("SGLANG_API_TOKEN", "hunter2")
    monkeypatch.setenv("UNRELATED_VAR", "no")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    env = prov.process_env()
    assert env["SGLANG_MOE_EXPERT_FILE_READER"] == "uring_direct"
    assert env["SGL_LEGACY_ALIAS"] == "1"
    assert env["OMP_NUM_THREADS"] == "16"
    assert env["SGLANG_API_TOKEN"] == prov.REDACTED
    assert "UNRELATED_VAR" not in env
    assert "hunter2" not in json.dumps(env)


def test_redaction_hits_secrets_and_spares_knobs_whose_names_only_contain_token():
    """Bug: a substring match on TOKEN redacted SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS from the
    first Task 1 arm's json, so a steering value in force was unrecoverable from the record."""
    for name in ("SGLANG_API_TOKEN", "HF_TOKEN", "EXA_API_KEY", "SGLANG_SECRET_KEY", "DB_PASSWORD", "GITHUB_ACCESS_TOKEN"):
        assert prov._redact(name, "v") == prov.REDACTED, name
    for name in (
        "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS", "SGLANG_USE_AITER_FP8_PER_TOKEN",
        "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE", "SGLANG_MAX_NEW_TOKENS_LIMIT", "SGLANG_MM_AVOID_RETOKENIZE",
    ):
        assert prov._redact(name, "256") == "256", name


def test_resolved_env_redacts_only_secret_named_knobs():
    resolved = prov.resolved_env()
    redacted = {k for k, v in resolved.items() if v == prov.REDACTED}
    assert redacted == {k for k in resolved if prov._SECRET_NAME.search(k)}
    assert resolved["SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS"] != prov.REDACTED


def test_exec_env_parses_nul_separated_environ(tmp_path):
    f = tmp_path / "environ"
    f.write_bytes(b"SGLANG_A=1\0PATH=/bin\0SGLANG_B=x=y\0SGLANG_SECRET_KEY=s\0")
    env = prov.exec_env(str(f))
    assert env == {"SGLANG_A": "1", "SGLANG_B": "x=y", "SGLANG_SECRET_KEY": prov.REDACTED}


def test_env_drift_names_added_removed_and_changed():
    before = {"SGLANG_A": "1", "SGLANG_B": "2", "SGLANG_C": "3"}
    after = {"SGLANG_A": "1", "SGLANG_B": "9", "SGLANG_D": "4"}
    assert prov.env_drift(before, after) == {
        "SGLANG_B": ["2", "9"],
        "SGLANG_C": ["3", None],
        "SGLANG_D": [None, "4"],
    }
    assert prov.env_drift(before, before) == {}


def test_resolved_env_shows_a_default_that_the_environment_does_not(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_EXPERT_FILE_READER", raising=False)
    resolved = prov.resolved_env()
    assert resolved["SGLANG_MOE_EXPERT_FILE_READER"] == "mmap"
    monkeypatch.setenv("SGLANG_MOE_EXPERT_FILE_READER", "uring_direct")
    assert prov.resolved_env()["SGLANG_MOE_EXPERT_FILE_READER"] == "uring_direct"
    json.dumps(resolved)  # every value must survive being written to the arm json


def test_git_state_clean_dirty_and_untracked(repo):
    pkg = str(repo / "python" / "pkg")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    clean = prov.git_state(pkg)
    assert clean["head"] == head
    assert clean["toplevel"] == os.path.realpath(repo) or clean["toplevel"] == str(repo)
    assert clean["dirty"] is False and clean["dirty_files"] == []
    assert clean["untracked_in_package_count"] == 0

    (repo / "python" / "pkg" / "a.py").write_text("x = 2\n")
    (repo / "python" / "pkg" / "new.py").write_text("y = 1\n")
    dirty = prov.git_state(pkg)
    assert dirty["head"] == head
    assert dirty["dirty"] is True and dirty["dirty_file_count"] == 1
    assert dirty["tracked_diff_sha1"] != clean["tracked_diff_sha1"]
    assert dirty["untracked_in_package"] == ["python/pkg/new.py"]


def test_git_state_refuses_a_directory_outside_any_worktree(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        prov.git_state(str(tmp_path))


def _write_diskstats(path, sectors):
    rows = [f"   259       0 {dev} 1 0 {sectors[name]} 0 0 0 0 0 0 0 0" for name, dev in prov.DRIVES.items()]
    path.write_text("\n".join(rows) + "\n")


def test_drive_idle_check_flags_a_busy_drive(tmp_path):
    stats = tmp_path / "diskstats"
    state = {"nvme0": 0, "nvme2": 0, "nvme4": 0}
    _write_diskstats(stats, state)

    def sleep(seconds):
        state["nvme2"] += 8 * 1024 * 1024 // 512 * 2  # 8 MiB/s over 2 s
        _write_diskstats(stats, state)

    result = prov.drive_idle_check(seconds=2.0, diskstats=str(stats), sleep=sleep)
    assert result["idle"] is False
    assert result["bytes_per_s"]["nvme2"] == pytest.approx(8 * 1024 * 1024)
    assert result["bytes_per_s"]["nvme0"] == 0


def test_drive_idle_check_reports_an_idle_drive(tmp_path):
    stats = tmp_path / "diskstats"
    _write_diskstats(stats, {"nvme0": 5, "nvme2": 5, "nvme4": 5})
    result = prov.drive_idle_check(seconds=1.0, diskstats=str(stats), sleep=lambda s: None)
    assert result["idle"] is True
    assert set(result["bytes_per_s"]) == set(prov.DRIVES)


def test_drive_idle_check_unreadable_diskstats_is_null_with_reason(tmp_path):
    result = prov.drive_idle_check(seconds=0.0, diskstats=str(tmp_path / "missing"), sleep=lambda s: None)
    assert result["idle"] is None
    assert "FileNotFoundError" in result["unavailable"]


def test_capture_records_the_imported_sglang_and_its_worktree():
    import sglang

    out = prov.capture({"driver": "/x/driver.py"})
    assert REQUIRED_KEYS <= set(out)
    assert out["sglang_file"] == sglang.__file__
    assert out["harness_files"] == {"driver": "/x/driver.py"}
    assert out["host"] and out["utc"].endswith("+00:00")
    json.dumps(out)  # the arm json must be writable
    if out["git"] is not None:
        assert len(out["git"]["head"]) == 40
        assert isinstance(out["git"]["dirty"], bool)
    else:
        assert "git" in out["unavailable"]


def test_capture_never_omits_a_key_when_a_source_is_unavailable(monkeypatch):
    def refuse(*a, **k):
        raise OSError("no such thing")

    monkeypatch.setattr(prov, "exec_env", refuse)
    monkeypatch.setattr(prov, "git_state", refuse)
    monkeypatch.setattr(prov, "resolved_env", refuse)
    out = prov.capture()
    assert REQUIRED_KEYS <= set(out)
    for field in ("sglang_env_at_exec", "git", "sglang_env_resolved"):
        assert out[field] is None
        assert "OSError: no such thing" in out["unavailable"][field]


def _log(times, tokens):
    return list(zip(times, tokens))


def test_step_latency_is_exact_when_every_chunk_is_one_token():
    # The first chunk (end of prefill, at t=10) starts the clock: 127 decode steps of 0.5 s, one a 2.0 s stall.
    times = [10.0 + 0.5 * i for i in range(128)]
    times[-1] += 1.5
    out = prov.step_latency(_log(times, range(1, 129)))
    assert out["steps"] == 127 and out["multi_token_chunks"] == 0
    assert out["step_s_p50"] == pytest.approx(0.5)
    assert out["step_s_p99"] == pytest.approx(0.5) and out["step_s_max"] == pytest.approx(2.0)
    assert len(out["step_s"]) == 127


def test_step_latency_flags_chunks_that_carry_several_tokens():
    """A chunk of 3 tokens must not be read as one slow step: it is divided and counted."""
    out = prov.step_latency(_log([0.0, 1.0, 4.0], [1, 2, 5]))
    assert out["multi_token_chunks"] == 1
    assert out["step_s"] == pytest.approx([1.0, 1.0])


def test_step_latency_refuses_to_guess_without_token_counts():
    out = prov.step_latency(_log([0.0, 1.0, 2.0], [1, None, 3]))
    assert "unavailable" in out and "step_s_p50" not in out
    assert "unavailable" in prov.step_latency(_log([0.0, 1.0, 2.0], [1, 3, 3]))  # a chunk with no new token
    assert "unavailable" in prov.step_latency(_log([0.0], [1]))


def test_timed_chunks_passes_the_stream_through_and_logs_cumulative_tokens():
    ticks = iter([1.0, 2.0, 3.0])
    log = []
    chunks = [{"meta_info": {"completion_tokens": 1}}, {"meta_info": {"completion_tokens": 2}}, "bare"]
    assert list(prov.timed_chunks(iter(chunks), log, clock=lambda: next(ticks))) == chunks
    assert log == [(1.0, 1), (2.0, 2), (3.0, None)]


def test_process_tree_cpu_s_counts_this_process():
    before = prov.process_tree_cpu_s()
    sum(i * i for i in range(2_000_000))
    assert prov.process_tree_cpu_s() > before


def test_arm_harnesses_embed_provenance_in_the_result_json():
    for parts in (("analysis", "dsv41-drive", "eager_arm_driver.py"), ("scripts", "dsv41", "trace_corpus.py")):
        with open(os.path.join(_ROOT, *parts)) as f:
            source = f.read()
        assert '"provenance": prov' in source, parts
        assert "provenance.capture(" in source and "drive_idle_check()" in source, parts
        assert "provenance.timed_chunks(" in source and "provenance.step_latency(" in source, parts
        assert "provenance.process_tree_cpu_s()" in source, parts


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
