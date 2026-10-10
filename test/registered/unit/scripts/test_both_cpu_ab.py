"""The both-CPU-experts server A/B driver: the probe server must not write into the timed server's metrics file (CPU)."""

import importlib.util
import json
import os

import pytest

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
        "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "0",
        "SGLANG_DSV41_RAM_PREFETCH_SCORER": "cpu",
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


def test_the_top_k_only_arm_differs_from_the_prefetch_arm_in_that_option_alone(monkeypatch):
    ab = _ab()
    b, c = ab.ARMS["dspark-both-prefetch"][0], ab.ARMS["dspark-both-prefetch-topk"][0]
    assert c["SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"] == "1"
    assert {k: v for k, v in c.items() if k != "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"} == {
        k: v for k, v in b.items() if k != "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"
    }
    assert ab.ARMS["dspark-both-prefetch-topk"][1] is True
    assert ab.REFERENCE["dspark-both-prefetch-topk"] == "dspark-both"
    monkeypatch.setenv("SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY", "0")
    assert ab._overrides("dspark-both-prefetch-topk", "/out")["SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"] == "1"


def test_every_arm_but_the_prefetch_pins_it_off_against_an_exported_shell(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("SGLANG_DSV41_RAM_PREFETCH", "1")
    for arm in ab.ARMS:
        expected = "1" if arm.startswith("dspark-both-prefetch") else "0"
        assert ab._overrides(arm, "/out")["SGLANG_DSV41_RAM_PREFETCH"] == expected, arm
        assert ab._probe_overrides(arm, "/out")["SGLANG_DSV41_RAM_PREFETCH"] == expected, arm


def test_the_private_build_caches_reach_every_arms_server(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", "/private/jit")
    monkeypatch.setenv("SGLANG_EXL3_BUILD_DIR", "/private/exl3")
    for arm in ab.ARMS:
        overrides = ab._overrides(arm, "/out")
        assert overrides["SGLANG_JIT_CACHE_DIR"] == "/private/jit"
        assert overrides["SGLANG_EXL3_BUILD_DIR"] == "/private/exl3"


PROBE = [{"session_id": "s0", "tokens": [{"token": 1, "top": [[1, -0.1], [2, -2.0]]}]}]


def _arm_run(root, arm, rows_read=2000, used=0, warmups=(100,), probe=PROBE, timed_tokens=100, counters=None):
    run = root / "servers" / arm / "run-1"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        json.dumps({"decode_tokens_per_sec": 2.0, "completion_tokens": timed_tokens, "spec_tokens_details": {}}) + "\n"
    )
    for n, tokens in enumerate(warmups, 1):
        (run / f"results-warmup-{n}.jsonl").write_text(json.dumps({"completion_tokens": tokens}) + "\n")
    counters = counters or {"rows_read": rows_read, "spec_issued": 900, "spec_used": used}
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


def test_the_summary_reports_drive_load_and_wasted_reads_per_lifetime_token(tmp_path):
    ab = _ab()
    counters = {"rows_read": 2000, "spec_issued": 900, "spec_landed": 800, "spec_used": 700}
    _arm_run(tmp_path, "dspark-both-prefetch", counters=counters)
    b = ab.summarize(str(tmp_path))["dspark-both-prefetch"]
    assert b["lifetime_tokens"] == 200 and b["ram_rows_per_token_lifetime"] == 10.0
    assert b["nvme_rows_per_token_lifetime"] == 14.5
    assert b["spec_wasted_per_token_lifetime"] == 0.5


def test_the_drive_load_fields_need_their_counters(tmp_path):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both", counters={"rows_read": 2000})
    a = ab.summarize(str(tmp_path))["dspark-both"]
    assert a["ram_rows_per_token_lifetime"] == 10.0
    assert "nvme_rows_per_token_lifetime" not in a and "spec_wasted_per_token_lifetime" not in a


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


def test_the_gpu_arm_differs_from_the_prefetch_arm_in_the_scorer_alone(monkeypatch):
    ab = _ab()
    b, g = ab.ARMS["dspark-both-prefetch"][0], ab.ARMS["dspark-both-prefetch-gpu"][0]
    assert (b["SGLANG_DSV41_RAM_PREFETCH_SCORER"], g["SGLANG_DSV41_RAM_PREFETCH_SCORER"]) == ("cpu", "gpu")
    assert {k: v for k, v in g.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"} == {
        k: v for k, v in b.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"
    }
    assert ab.ARMS["dspark-both-prefetch-gpu"][1] is True and ab.REFERENCE["dspark-both-prefetch-gpu"] == "dspark-both"
    t = ab.ARMS["dspark-both-prefetch-gpu-topk"][0]
    assert {k: v for k, v in t.items() if k != "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"} == {
        k: v for k, v in g.items() if k != "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"
    }
    assert t["SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"] == "1"
    monkeypatch.setenv("SGLANG_DSV41_RAM_PREFETCH_SCORER", "cpu")
    assert ab._overrides("dspark-both-prefetch-gpu", "/out")["SGLANG_DSV41_RAM_PREFETCH_SCORER"] == "gpu"


def test_the_floors_arm_differs_from_the_gpu_arm_in_the_committed_margin_floors_alone(monkeypatch):
    from sglang.srt.layers.moe.ram_prefetch import load_margin_floors

    ab = _ab()
    g, f = ab.ARMS["dspark-both-prefetch-gpu"][0], ab.ARMS["dspark-both-prefetch-gpu-floors"][0]
    key = "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"
    assert {k: v for k, v in f.items() if k != key} == {k: v for k, v in g.items() if k != key}
    assert g[key] == "" and ab.PREFETCH_OFF[key] == ""
    floors = load_margin_floors(f[key], rows=40)
    assert floors[0] == float("-inf") and floors[14] == 0.25
    assert ab.ARMS["dspark-both-prefetch-gpu-floors"][1] is True
    assert ab.REFERENCE["dspark-both-prefetch-gpu-floors"] == "dspark-both"
    monkeypatch.setenv(key, "/elsewhere.json")
    assert ab._overrides("dspark-both-prefetch-gpu-floors", "/out")[key] == f[key]
    assert ab._overrides("dspark-both", "/out")[key] == ""


def test_the_driver_runs_the_arms_in_the_order_given(monkeypatch, tmp_path):
    """The reversed A/B (B first) relies on it."""
    ab = _ab()
    ran = []
    monkeypatch.setattr(ab, "run_timed", lambda arm, out: ran.append(("timed", arm)) or 0)
    monkeypatch.setattr(ab, "run_probe", lambda arm, out: ran.append(("probe", arm)) or 0)
    monkeypatch.setattr(ab, "summarize", lambda out: {})
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path), "dspark-both-prefetch-gpu", "dspark-both"])
    ab.main()
    assert ran == [
        ("timed", "dspark-both-prefetch-gpu"),
        ("probe", "dspark-both-prefetch-gpu"),
        ("timed", "dspark-both"),
        ("probe", "dspark-both"),
    ]


def _session_rows(root, arm, ms):
    run = root / "servers" / arm / "run-1"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        "".join(
            json.dumps({"session_id": f"s{i}", "decode_tokens_per_sec": 1000.0 / v, "completion_tokens": 10,
                        "spec_tokens_details": {}}) + "\n"
            for i, v in enumerate(ms)
        )
    )
    counters = {"rows_read": 100, "spec_used": 50, "spec_promoted": 5, "spec_late": 3}
    (run / "server.log").write_text("exl3 RAM miss thread counters " + json.dumps(counters) + "\n")


def test_the_summary_pairs_sessions_and_reports_reads_still_in_flight(tmp_path):
    ab = _ab()
    _session_rows(tmp_path, "dspark-both", [100.0, 200.0, 50.0])
    _session_rows(tmp_path, "dspark-both-prefetch-gpu", [90.0, 210.0, 40.0])
    g = ab.summarize(str(tmp_path))["dspark-both-prefetch-gpu"]
    assert g["paired_gain_pct_vs_reference"] == pytest.approx({"s0": 10.0, "s1": -5.0, "s2": 20.0})
    assert g["paired_gain_pct_median"] == pytest.approx(10.0)
    assert g["spec_in_flight_at_use"] == pytest.approx(0.1) and g["ram"]["spec_late"] == 3


def _baseline(root, ms=(100.0, 100.0, 100.0)):
    _session_rows(root, "dspark-both", list(ms))
    (root / "dspark-both.probe.json").write_text("{}")
    (root / "dspark-both.metrics.jsonl").write_text("{}\n")
    (root / "commit.txt").write_text("abc123\n")


def test_a_saved_baseline_is_imported_instead_of_rerunning_the_reference_arm(monkeypatch, tmp_path):
    ab = _ab()
    base, out = tmp_path / "base", tmp_path / "out"
    _baseline(base)
    ran = []
    monkeypatch.setattr(ab, "run_timed", lambda arm, o: ran.append(("timed", arm)) or _session_rows(out, arm, [90.0, 95.0, 110.0]) or 0)
    monkeypatch.setattr(ab, "run_probe", lambda arm, o: ran.append(("probe", arm)) or 0)
    monkeypatch.setattr(ab, "_compare", lambda a, b: {"ok": True})
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(out), "dspark-both-prefetch-gpu-floors", "--baseline", str(base)])
    ab.main()
    assert ran == [("timed", "dspark-both-prefetch-gpu-floors"), ("probe", "dspark-both-prefetch-gpu-floors")]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["dspark-both"]["baseline_from"] == {"dir": str(base), "commit": "abc123"}
    assert summary["dspark-both-prefetch-gpu-floors"]["paired_gain_pct_median"] == 5.0
    assert (out / "dspark-both.probe.json").exists()


@pytest.mark.parametrize("missing", ["dspark-both.probe.json", "dspark-both.metrics.jsonl", "servers"])
def test_a_baseline_missing_any_of_its_files_is_refused_before_any_arm_runs(monkeypatch, tmp_path, missing):
    import shutil

    ab = _ab()
    base = tmp_path / "base"
    _baseline(base)
    target = base / missing
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    monkeypatch.setattr(ab, "run_timed", lambda arm, o: pytest.fail("ran an arm"))
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path / "out"), "dspark-both-prefetch-gpu-floors", "--baseline", str(base)])
    with pytest.raises(SystemExit, match="baseline"):
        ab.main()


def test_a_baseline_cannot_be_both_imported_and_rerun(monkeypatch, tmp_path):
    ab = _ab()
    base = tmp_path / "base"
    _baseline(base)
    monkeypatch.setattr(ab, "run_timed", lambda arm, o: pytest.fail("ran an arm"))
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path / "out"), "dspark-both", "--baseline", str(base)])
    with pytest.raises(SystemExit, match="baseline"):
        ab.main()


MISS_CUT = {"SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "1", "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3"}


def test_the_miss_cut_arm_is_dspark_both_with_the_cut_on_against_dspark_both():
    ab = _ab()
    a, c = ab.ARMS["dspark-both"][0], ab.ARMS["dspark-both-misscut"][0]
    assert {k: c[k] for k in MISS_CUT} == MISS_CUT
    assert {k: v for k, v in c.items() if k not in MISS_CUT} == {k: v for k, v in a.items() if k not in MISS_CUT}
    assert ab.ARMS["dspark-both-misscut"][1] is True
    assert ab.REFERENCE["dspark-both-misscut"] == "dspark-both"


def test_the_prefetch_miss_cut_arm_is_the_gpu_prefetch_arm_with_the_cut_on():
    ab = _ab()
    a, c = ab.ARMS["dspark-both-prefetch-gpu"][0], ab.ARMS["dspark-both-prefetch-gpu-misscut"][0]
    assert {k: c[k] for k in MISS_CUT} == MISS_CUT
    assert {k: v for k, v in c.items() if k not in MISS_CUT} == {k: v for k, v in a.items() if k not in MISS_CUT}
    assert ab.REFERENCE["dspark-both-prefetch-gpu-misscut"] == "dspark-both"


IDLE = "SGLANG_DSV41_RAM_PREFETCH_IDLE_DRIVE"
IDLE_DEADLINE = "SGLANG_DSV41_RAM_PREFETCH_IDLE_DEADLINE_US"


def test_the_idle_drive_arm_is_the_prefetch_miss_cut_arm_with_idle_drive_reads_against_the_miss_cut_arm():
    """Design 2026-10-09-dsv41-drive-aware-reads change (2): compared with dspark-both-misscut, the same arm without
    the prefetch, so the prefetch's net effect with idle-drive reads is what the A/B reads."""
    ab = _ab()
    a, c = ab.ARMS["dspark-both-prefetch-gpu-misscut"][0], ab.ARMS["dspark-both-prefetch-gpu-misscut-idle"][0]
    assert (c[IDLE], c[IDLE_DEADLINE]) == ("1", "4000")
    assert {k: v for k, v in c.items() if k != IDLE} == {k: v for k, v in a.items() if k != IDLE}
    assert ab.ARMS["dspark-both-prefetch-gpu-misscut-idle"][1] is True
    assert ab.REFERENCE["dspark-both-prefetch-gpu-misscut-idle"] == "dspark-both-misscut"


def test_every_other_arm_pins_idle_drive_reads_off_against_an_exported_shell(monkeypatch):
    ab = _ab()
    monkeypatch.setenv(IDLE, "1")
    monkeypatch.setenv(IDLE_DEADLINE, "9")
    for arm in ab.ARMS:
        want = ("1" if arm == "dspark-both-prefetch-gpu-misscut-idle" else "0", "4000")
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert (overrides[IDLE], overrides[IDLE_DEADLINE]) == want, arm


def test_the_summary_keeps_the_idle_drive_counters():
    ab = _ab()
    assert {"spec_deferred", "spec_abandoned", "spec_boosted", "spec_moved_root"} <= set(ab.RAM_KEYS)


def test_every_arm_but_the_miss_cut_pins_it_off_against_an_exported_shell(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("SGLANG_DSV41_CPU_SPLIT_MISS_CUT", "2")
    monkeypatch.setenv("SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX", "5")
    for arm in ab.ARMS:
        want = MISS_CUT if "-misscut" in arm else {
            "SGLANG_DSV41_CPU_SPLIT_MISS_CUT": "0", "SGLANG_DSV41_CPU_SPLIT_MISS_CUT_MAX": "3"}
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert {k: overrides[k] for k in want} == want, arm


DYNAMIC = "SGLANG_MOE_EXPERT_MIRROR_DYNAMIC"
CAPS = "SGLANG_MOE_EXPERT_MIRROR_CAPS"


def test_the_cap_arm_is_the_gpu_arm_with_the_per_drive_in_flight_cap(monkeypatch):
    """Design 2026-10-09-dsv41-drive-aware-reads change (3): roots nvme0, nvme4 (the SPCC), nvme2 capped at 4, 2, 4
    sub-reads in flight; compared with dspark-both like the other prefetch arms."""
    ab = _ab()
    g, c = ab.ARMS["dspark-both-prefetch-gpu"][0], ab.ARMS["dspark-both-prefetch-gpu-cap"][0]
    assert (c[DYNAMIC], c[CAPS]) == ("1", "4,2,4")
    assert {k: v for k, v in c.items() if k not in (DYNAMIC, CAPS)} == {
        k: v for k, v in g.items() if k not in (DYNAMIC, CAPS)
    }
    # The caps follow the roots' order in the recipe the arms run on.
    assert ab.arm_env.EXPERT_MIRROR_DIRS.split(":") == [
        "/mnt/nvme0/dsv41_flash", "/mnt/nvme4/dsv41_flash", "/mnt/nvme2/dsv41_flash"]
    assert ab.ARMS["dspark-both-prefetch-gpu-cap"][1] is True
    assert ab.REFERENCE["dspark-both-prefetch-gpu-cap"] == "dspark-both"
    monkeypatch.setenv(DYNAMIC, "0")
    monkeypatch.setenv(CAPS, "9,9,9")
    assert (ab._overrides("dspark-both-prefetch-gpu-cap", "/out")[DYNAMIC],
            ab._overrides("dspark-both-prefetch-gpu-cap", "/out")[CAPS]) == ("1", "4,2,4")


def test_every_other_arm_pins_the_dynamic_root_choice_off_against_an_exported_shell(monkeypatch):
    ab = _ab()
    monkeypatch.setenv(DYNAMIC, "1")
    monkeypatch.setenv(CAPS, "4,2,4")
    for arm in ab.ARMS:
        if arm == "dspark-both-prefetch-gpu-cap":
            continue
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert (overrides[DYNAMIC], overrides[CAPS]) == ("0", ""), arm
