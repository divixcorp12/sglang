"""The both-CPU-experts server A/B driver: the probe server must not write into the timed server's metrics file (CPU)."""

import importlib.util
import json
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
METRICS = "SGLANG_MOE_HOT_METRICS_FILE"


def _ab():
    spec = importlib.util.spec_from_file_location(
        "both_cpu_ab", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "both_cpu_ab.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_probe_server_writes_its_own_metrics_file_and_keeps_the_arms_env():
    ab = _ab()
    for arm in ab.ARMS:
        timed = ab._overrides(arm, "/out")
        probe = ab._probe_overrides(arm, "/out")
        # summarize() reads the timed server's last record; an appending probe server would overwrite it.
        assert probe[METRICS] != timed[METRICS]
        assert {k: v for k, v in probe.items() if k != METRICS} == {k: v for k, v in timed.items() if k != METRICS}


def test_a_dspark_arm_gets_a_health_budget_past_its_measured_startup_and_run_arm_honours_it(monkeypatch):
    ab = _ab()
    # 2026-10-06 smoke: dspark-both loaded weights at 435 s and the draft at ~870 s, so run_arm.sh's fixed 900 s gate
    # killed a server that was still starting (no defect in it).
    assert ab._health_timeout_s(False) == 900
    assert ab._health_timeout_s(True) >= 2400
    seen = {}
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, env, cwd: seen.update(env=env) or type("R", (), {"returncode": 0})())
    ab.run_timed("dspark-both", "/out")
    assert seen["env"]["DSV41_HEALTH_TIMEOUT_S"] == str(ab._health_timeout_s(True))
    script = open(os.path.join(ROOT, "benchmarks", "dsv41_baseline", "run_arm.sh")).read()
    assert "${DSV41_HEALTH_TIMEOUT_S:-900}" in script
    assert "seq 1 $((health_timeout_s / 5))" in script


def test_the_timed_dspark_arms_take_the_recipes_mem_fraction_and_prod_does_not(monkeypatch):
    ab = _ab()
    monkeypatch.delenv("DSV41_MEM_FRACTION_STATIC", raising=False)
    envs = {}
    monkeypatch.setattr(
        ab.subprocess, "run", lambda cmd, env, cwd: envs.update({cmd[1]: env}) or type("R", (), {"returncode": 0})()
    )
    for arm in ab.ARMS:
        ab.run_timed(arm, "/out")
    assert envs["dspark-both"]["DSV41_MEM_FRACTION_STATIC"] == "0.78"
    assert envs["dspark-draft-only"]["DSV41_MEM_FRACTION_STATIC"] == "0.78"
    assert "DSV41_MEM_FRACTION_STATIC" not in envs["prod"]


def test_the_prefetch_arm_is_dspark_both_with_the_prefetch_on_and_its_a_states_it_off(monkeypatch):
    ab = _ab()
    options = {
        "SGLANG_DSV41_RAM_PREFETCH": "1",
        "SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN": "1",
        "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER": "1",
        "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE": "2",
    }
    a, b = ab.ARMS["dspark-both"][0], ab.ARMS["dspark-both-prefetch"][0]
    assert a["SGLANG_DSV41_RAM_PREFETCH"] == "0"
    assert {k: b[k] for k in options} == options
    assert {k: v for k, v in a.items() if k != "SGLANG_DSV41_RAM_PREFETCH"} == {
        k: v for k, v in b.items() if k not in options
    }
    assert ab.ARMS["dspark-both-prefetch"][1] is True
    for k in options:
        monkeypatch.setenv(k, "9")
    assert {k: ab._overrides("dspark-both-prefetch", "/out")[k] for k in options} == options
    assert ab._overrides("dspark-both", "/out")["SGLANG_DSV41_RAM_PREFETCH"] == "0"


def test_the_private_build_caches_reach_every_arms_server(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", "/private/jit")
    monkeypatch.setenv("SGLANG_EXL3_BUILD_DIR", "/private/exl3")
    for arm in ab.ARMS:
        overrides = ab._overrides(arm, "/out")
        assert overrides["SGLANG_JIT_CACHE_DIR"] == "/private/jit"
        assert overrides["SGLANG_EXL3_BUILD_DIR"] == "/private/exl3"


PROBE = [{"session_id": "s0", "tokens": [{"token": 1, "top": [[1, -0.1], [2, -2.0]]}]}]


def _arm_run(root, arm, rows_read=2000, used=0, warmups=(100,), probe=PROBE, timed_tokens=100):
    run = root / "servers" / arm / "run-1"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        json.dumps({"decode_tokens_per_sec": 2.0, "completion_tokens": timed_tokens, "spec_tokens_details": {}}) + "\n"
    )
    for n, tokens in enumerate(warmups, 1):
        (run / f"results-warmup-{n}.jsonl").write_text(json.dumps({"completion_tokens": tokens}) + "\n")
    counters = {"rows_read": rows_read, "spec_issued": 900, "spec_used": used}
    (run / "server.log").write_text("noise\nexl3 RAM miss thread counters " + json.dumps(counters) + "\n")
    if probe is not None:
        (root / f"{arm}.probe.json").write_text(json.dumps(probe))


def test_summarize_reports_the_ram_counters_and_the_prefetch_arms_text_against_its_a(tmp_path):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both", rows_read=3000)
    _arm_run(tmp_path, "dspark-both-prefetch", rows_read=2000, used=700)
    summary = ab.summarize(str(tmp_path))
    b = summary["dspark-both-prefetch"]
    assert b["ram"]["spec_used"] == 700 and b["ram"]["rows_read"] == 2000
    assert b["warmup_rounds"] == 1 and b["lifetime_tokens"] == 200
    assert b["ram_rows_per_token_lifetime"] == 10.0 and summary["dspark-both"]["ram_rows_per_token_lifetime"] == 15.0
    assert "ram_rows_per_timed_token" not in b
    assert b["text_vs_reference"]["pass"] is True


def test_the_lifetime_ratio_counts_every_warmup_round_the_server_ran(tmp_path):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both", rows_read=5000, warmups=(100, 100, 100, 100))
    summary = ab.summarize(str(tmp_path))["dspark-both"]
    assert summary["warmup_rounds"] == 4 and summary["lifetime_tokens"] == 500
    assert summary["ram_rows_per_token_lifetime"] == 10.0 and summary["ram"]["rows_read"] == 5000


def test_a_probe_pair_that_cannot_be_compared_is_recorded_not_raised(tmp_path):
    ab = _ab()
    other = [{**PROBE[0], "session_id": "s1"}]
    _arm_run(tmp_path, "prod", probe=PROBE)
    _arm_run(tmp_path, "dspark-both", probe=PROBE)
    _arm_run(tmp_path, "dspark-both-prefetch", probe=other)
    summary = ab.summarize(str(tmp_path))
    assert "different prompts" in summary["dspark-both-prefetch"]["text_vs_reference"]["error"]
    assert "different prompts" in summary["dspark-both-prefetch"]["text_vs_prod"]["error"]
    assert (tmp_path / "summary.json").exists()


def test_a_missing_reference_probe_is_recorded(tmp_path):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both", probe=None)
    _arm_run(tmp_path, "dspark-both-prefetch")
    summary = ab.summarize(str(tmp_path))
    assert summary["dspark-both-prefetch"]["text_vs_reference"] == "missing reference probe"
    assert "text_vs_reference" not in summary["dspark-both"]
