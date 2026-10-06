"""The both-CPU-experts server A/B driver: the probe server must not write into the timed server's metrics file (CPU)."""

import importlib.util
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
