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


def test_the_prefetch_arm_is_dspark_both_with_the_prefetch_on_and_its_a_states_it_off():
    ab = _ab()
    a, b = ab.ARMS["dspark-both"][0], ab.ARMS["dspark-both-prefetch"][0]
    assert a["SGLANG_DSV41_RAM_PREFETCH"] == "0" and b["SGLANG_DSV41_RAM_PREFETCH"] == "1"
    assert {k: v for k, v in a.items() if k != "SGLANG_DSV41_RAM_PREFETCH"} == {
        k: v for k, v in b.items() if k != "SGLANG_DSV41_RAM_PREFETCH"
    }
    assert ab.ARMS["dspark-both-prefetch"][1] is True


def test_the_private_build_caches_reach_every_arms_server(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", "/private/jit")
    monkeypatch.setenv("SGLANG_EXL3_BUILD_DIR", "/private/exl3")
    for arm in ab.ARMS:
        overrides = ab._overrides(arm, "/out")
        assert overrides["SGLANG_JIT_CACHE_DIR"] == "/private/jit"
        assert overrides["SGLANG_EXL3_BUILD_DIR"] == "/private/exl3"


def test_summarize_reports_the_ram_counters_and_the_prefetch_arms_text_against_its_a(tmp_path):
    ab = _ab()
    for arm, rows_read, used in (("dspark-both", 3000, 0), ("dspark-both-prefetch", 2000, 700)):
        run = tmp_path / "servers" / arm / "run-1"
        run.mkdir(parents=True)
        (run / "results.jsonl").write_text(
            json.dumps({"decode_tokens_per_sec": 2.0, "completion_tokens": 100, "spec_tokens_details": {}}) + "\n"
        )
        counters = {"rows_read": rows_read, "spec_issued": 900, "spec_used": used}
        (run / "server.log").write_text("noise\nexl3 RAM miss thread counters " + json.dumps(counters) + "\n")
        probe = [{"session_id": "s0", "tokens": [{"token": 1, "top": [[1, -0.1], [2, -2.0]]}]}]
        (tmp_path / f"{arm}.probe.json").write_text(json.dumps(probe))
    summary = ab.summarize(str(tmp_path))
    b = summary["dspark-both-prefetch"]
    assert b["ram"]["spec_used"] == 700 and b["ram"]["rows_read"] == 2000
    assert b["ram_rows_per_timed_token"] == 20.0 and summary["dspark-both"]["ram_rows_per_timed_token"] == 30.0
    assert b["text_vs_reference"]["pass"] is True
