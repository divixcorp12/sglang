"""The both-CPU-experts server A/B driver: the probe server must not write into the timed server's metrics file (CPU)."""

import importlib.util
import json
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
METRICS = "SGLANG_MOE_HOT_METRICS_FILE"
STS_COLLECT = "SGLANG_DSPARK_STS_COLLECT_PATH"
CONFIDENCE_CUT = "SGLANG_DSPARK_CONFIDENCE_CUT"
STS_PATH_FLAG = "--speculative-dspark-confidence-sts-path"
BLOCK_FLAG = "--speculative-use-block-verification"
SAMPLED_KNOBS = (
    "SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED",
    "SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST",
    "SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL",
    "SGLANG_DSPARK_USE_GREEDY_DRAFT",
)


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
        # A collecting probe server's recorder would overwrite the timed server's shards from shard 0.
        assert (probe[STS_COLLECT] != timed[STS_COLLECT]) == (arm == STS_COLLECT_ARM), arm
        own = (METRICS, STS_COLLECT)
        assert {k: v for k, v in probe.items() if k not in own} == {k: v for k, v in timed.items() if k not in own}


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


def test_every_prefetch_arm_pins_its_scorer_and_floors_against_the_defaults_and_the_shell(monkeypatch):
    """The scorer defaults to gpu with the packaged floors; each arm keeps the scorer and floors it was measured with."""
    from sglang.srt.layers.moe.ram_prefetch import MARGIN_FLOORS

    ab = _ab()
    assert ab.MARGIN_FLOORS == MARGIN_FLOORS
    scorer, floors = "SGLANG_DSV41_RAM_PREFETCH_SCORER", "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"
    monkeypatch.setenv(scorer, "tpu")
    monkeypatch.setenv(floors, "/elsewhere.json")
    for arm in ab.ARMS:
        env = ab._overrides(arm, "/out")
        if arm.startswith("dspark-both-prefetch"):
            assert env[scorer] == ("gpu" if "-gpu" in arm else "cpu"), arm
            assert env[floors] == (MARGIN_FLOORS if arm.endswith("-floors") else ""), arm
        else:
            # Prefetch off: the scorer is unused, and the gate refuses a leaked explicit gpu rather than ignore it.
            assert env["SGLANG_DSV41_RAM_PREFETCH"] == "0" and env[floors] == "", arm


def test_the_driver_runs_the_arms_in_the_order_given(monkeypatch, tmp_path):
    """The reversed A/B (B first) relies on it."""
    ab = _ab()
    ran = []
    monkeypatch.setattr(ab, "run_timed", lambda arm, out, **_: ran.append(("timed", arm)) or 0)
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
    monkeypatch.setattr(
        ab, "run_timed", lambda arm, o, **_: ran.append(("timed", arm)) or _session_rows(out, arm, [90.0, 95.0, 110.0]) or 0
    )
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
    monkeypatch.setattr(ab, "run_timed", lambda arm, o, **_: pytest.fail("ran an arm"))
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path / "out"), "dspark-both-prefetch-gpu-floors", "--baseline", str(base)])
    with pytest.raises(SystemExit, match="baseline"):
        ab.main()


def test_a_baseline_cannot_be_both_imported_and_rerun(monkeypatch, tmp_path):
    ab = _ab()
    base = tmp_path / "base"
    _baseline(base)
    monkeypatch.setattr(ab, "run_timed", lambda arm, o, **_: pytest.fail("ran an arm"))
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


def test_the_weights_arm_is_the_miss_cut_arm_with_the_mirror_weighted_by_drive_speed():
    ab = _ab()
    a, w = ab.ARMS["dspark-both-misscut"][0], ab.ARMS["dspark-both-weights-misscut"][0]
    assert w["SGLANG_MOE_EXPERT_MIRROR_WEIGHTS"] == "16:10:16"  # nvme0 : nvme4 (SPCC) : nvme2, by fio rate
    assert {k: v for k, v in w.items() if k != "SGLANG_MOE_EXPERT_MIRROR_WEIGHTS"} == {
        k: v for k, v in a.items() if k != "SGLANG_MOE_EXPERT_MIRROR_WEIGHTS"}
    assert ab.ARMS["dspark-both-weights-misscut"][1] is True
    assert ab.REFERENCE["dspark-both-weights-misscut"] == "dspark-both-misscut"


def test_the_max_m8_arm_is_the_miss_cut_arm_with_four_tokens_per_cpu_chunk():
    ab = _ab()
    a, m = ab.ARMS["dspark-both-misscut"][0], ab.ARMS["dspark-both-misscut-maxm8"][0]
    assert m["SGLANG_EXL3_CPU_MAX_M"] == "8"  # CHUNK_M 4: an expert shared by up to 4 tokens decodes its weights once
    assert {k: v for k, v in m.items() if k != "SGLANG_EXL3_CPU_MAX_M"} == {
        k: v for k, v in a.items() if k != "SGLANG_EXL3_CPU_MAX_M"}
    assert ab.ARMS["dspark-both-misscut-maxm8"][1] is True
    assert ab.REFERENCE["dspark-both-misscut-maxm8"] == "dspark-both-misscut"


PROMOTE = "SGLANG_DSV41_RAM_PREFETCH_PROMOTE"
PROMOTE_ON = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote"
PROMOTE_OFF = f"{PROMOTE_ON}-off"
POOL_SERVE = "SGLANG_DSV41_RAM_PREFETCH_POOL_SERVE"
POOL_SERVE_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-poolserve"
PROMOTE_POOL_SERVE_ARM = f"{PROMOTE_ON}-poolserve"
GPU_DRIP = "SGLANG_DSV41_GPU_DRIP_SLOTS"
GPU_DRIP_ARM = f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit-gpudrip"
GPU_DRIP_LEAN_ARM = f"{GPU_DRIP_ARM}-c64p2"
PIN01_ARM = f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit-pin01"
PIN01_GPU01_ARM = f"{PIN01_ARM}-gpu01"
PIN01_SPLIT_MODEL_ARM = f"{PIN01_ARM}-splitmodel"
PIN01_MEM_ARM = f"{PIN01_ARM}-mem"
PIN01_SPLIT_MODEL_MEM_ARM = f"{PIN01_SPLIT_MODEL_ARM}-mem"
# Production's arm (-expseg with production's additions) and the confidence cut's two arms built on it.
PROD_ARM = f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-mergechunk-expseg-prod"
STS_COLLECT_ARM, CONFIDENCE_CUT_ARM = f"{PROD_ARM}-stscollect", f"{PROD_ARM}-confcut"
SAMPLED_CUT_ARM = f"{CONFIDENCE_CUT_ARM}-sampled"
SAMPLED_CUT_BLOCK_ARM = f"{SAMPLED_CUT_ARM}-block"
LAYER_MAJOR_ARM = f"{PROD_ARM}-layermajor"
LAYER_MAJOR_SMALL_HOT_ARM, LAYER_MAJOR_RESIDENCY_ARM = f"{LAYER_MAJOR_ARM}-smallhot", f"{LAYER_MAJOR_ARM}-res"
LAYER_MAJOR_ARMS = (LAYER_MAJOR_ARM, LAYER_MAJOR_SMALL_HOT_ARM, LAYER_MAJOR_RESIDENCY_ARM)
LAYER_MAJOR_LEND_ARM = f"{LAYER_MAJOR_ARM}-lend"
CHUNKED_ARM = f"{PROD_ARM}-chunked"
PROD_ARMS = (PROD_ARM, STS_COLLECT_ARM, CONFIDENCE_CUT_ARM, SAMPLED_CUT_ARM, SAMPLED_CUT_BLOCK_ARM, *LAYER_MAJOR_ARMS,
             LAYER_MAJOR_LEND_ARM, CHUNKED_ARM)
PIN01_HICACHE_ARMS = (f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1", f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-engram0",
                      f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-decay995", f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-tiebreak",
                      f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-mergechunk",
                      f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-mergechunk-expseg",
                      f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-mergechunk-expseg-calib",
                      f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-fixedcosts", f"{PIN01_SPLIT_MODEL_MEM_ARM}-hicache1-refine",
                      *PROD_ARMS)
PIN01_SPLIT_MODEL_FIXED_COSTS_ARM, PIN01_SPLIT_MODEL_REFINE_ARM = PIN01_HICACHE_ARMS[7:9]
MLOCK = "SGLANG_ENABLE_SCHEDULER_MLOCK"
SPLIT_MODEL = "SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL"
SPLIT_MODEL_COSTS = "SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS"
SPLIT_MODEL_REFINE = "SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES"
FULL_LAYERS = "SGLANG_MOE_PINNED_HOST_FULL_LAYERS"
SPLIT_LAYERS = "SGLANG_DSV41_CPU_EXPERTS_SPLIT_LAYERS"
PROMOTE_PAIR_ARMS = (
    PROMOTE_ON, PROMOTE_OFF, f"{PROMOTE_ON}-fixedsplit", f"{PROMOTE_OFF}-fixedsplit", POOL_SERVE_ARM,
    f"{POOL_SERVE_ARM}-fixedsplit", PROMOTE_POOL_SERVE_ARM, f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit",
    f"{PROMOTE_POOL_SERVE_ARM}-cores13-fixedsplit",
    f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit-confgate",
    GPU_DRIP_ARM,
    GPU_DRIP_LEAN_ARM,
    PIN01_ARM,
    PIN01_GPU01_ARM,
    PIN01_SPLIT_MODEL_ARM,
    PIN01_MEM_ARM,
    PIN01_SPLIT_MODEL_MEM_ARM,
    *PIN01_HICACHE_ARMS,
)
SPLIT = "SGLANG_DSV41_CPU_EXPERTS_SPLIT"
CALIBRATION = "SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION"
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


IDLE_SPREAD = "SGLANG_DSV41_RAM_PREFETCH_IDLE_SPREAD"
IDLE_PIECES = "SGLANG_DSV41_RAM_PREFETCH_IDLE_PIECES"
IDLE2_ARMS = (
    "dspark-both-prefetch-gpu-misscut-idle2",
    "dspark-both-prefetch-gpu-misscut-idle2-floors",
    "dspark-both-prefetch-gpu-misscut-idle2-l2",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-t3",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-cap2",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2",
    "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap3",
    *PROMOTE_PAIR_ARMS,
)
IDLE_ARMS = ("dspark-both-prefetch-gpu-misscut-idle", *IDLE2_ARMS)


def test_the_idle2_floors_arm_is_the_idle2_arm_with_the_packaged_margin_floors_alone():
    ab = _ab()
    floors = "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"
    i, f = ab.ARMS["dspark-both-prefetch-gpu-misscut-idle2"][0], ab.ARMS["dspark-both-prefetch-gpu-misscut-idle2-floors"][0]
    assert (i[floors], f[floors]) == ("", ab.MARGIN_FLOORS)
    assert {k: v for k, v in f.items() if k != floors} == {k: v for k, v in i.items() if k != floors}
    assert ab.ARMS["dspark-both-prefetch-gpu-misscut-idle2-floors"][1] is True
    assert ab.REFERENCE["dspark-both-prefetch-gpu-misscut-idle2-floors"] == "dspark-both-misscut"


def test_the_idle2_arm_is_the_idle_arm_with_spread_ties_and_two_pieces_against_the_miss_cut_arm():
    ab = _ab()
    a, c = ab.ARMS["dspark-both-prefetch-gpu-misscut-idle"][0], ab.ARMS["dspark-both-prefetch-gpu-misscut-idle2"][0]
    assert (a[IDLE_SPREAD], a[IDLE_PIECES]) == ("0", "1")
    assert (c[IDLE], c[IDLE_SPREAD], c[IDLE_PIECES]) == ("1", "1", "2")
    knobs = (IDLE_SPREAD, IDLE_PIECES)
    assert {k: v for k, v in c.items() if k not in knobs} == {k: v for k, v in a.items() if k not in knobs}
    assert ab.ARMS["dspark-both-prefetch-gpu-misscut-idle2"][1] is True
    assert ab.REFERENCE["dspark-both-prefetch-gpu-misscut-idle2"] == "dspark-both-misscut"


def test_every_other_arm_pins_idle_drive_reads_off_against_an_exported_shell(monkeypatch):
    ab = _ab()
    monkeypatch.setenv(IDLE, "1")
    monkeypatch.setenv(IDLE_DEADLINE, "9")
    monkeypatch.setenv(IDLE_SPREAD, "1")
    monkeypatch.setenv(IDLE_PIECES, "3")
    for arm in ab.ARMS:
        want = ("1" if arm in IDLE_ARMS else "0", "4000")
        knobs = ("1", "2") if arm in IDLE2_ARMS else ("0", "1")
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert (overrides[IDLE], overrides[IDLE_DEADLINE]) == want, arm
            assert (overrides[IDLE_SPREAD], overrides[IDLE_PIECES]) == knobs, arm


HORIZON = "SGLANG_DSV41_RAM_PREFETCH_HORIZON"
L2_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2"
T3_ARM = "dspark-both-prefetch-gpu-misscut-idle2-l2-t3"
L2_CAP2_ARM = f"{L2_ARM}-cap2"
T3_CAP2_ARM = f"{T3_ARM}-cap2"
T3_CAP3_ARM = f"{T3_ARM}-cap3"
LEARNED_HEAD = "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD"
PER_LAYER = "SGLANG_DSV41_RAM_PREFETCH_PER_LAYER"
SPEC_SHARE = "SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE"


def test_the_t3_cap3_arm_is_the_t3_arm_with_three_rows_per_layer_and_three_pool_slots_alone():
    """Mutant: the cap-3 arm keeps two pool slots (a third same-group read finds no slot and is dropped) or changes
    another knob -- red."""
    ab = _ab()
    a, b = ab.ARMS[T3_ARM][0], ab.ARMS[T3_CAP3_ARM][0]
    assert (a[PER_LAYER], a[SPEC_SHARE], b[PER_LAYER], b[SPEC_SHARE]) == ("1", "2", "3", "3")
    knobs = (PER_LAYER, SPEC_SHARE)
    assert {k: v for k, v in b.items() if k not in knobs} == {k: v for k, v in a.items() if k not in knobs}
    assert ab.ARMS[T3_CAP3_ARM][1] is True
    assert ab.REFERENCE[T3_CAP3_ARM] == "dspark-both-misscut"


def test_the_engram_verify_graph_arm_is_the_misscut_arm_with_the_flag_alone_on_the_device_wait():
    """Mutant: the arm also changes another knob, drops the device wait (the flag then does nothing), or is not
    compared with its A -- red."""
    ab = _ab()
    flag = "SGLANG_DSV41_ENGRAM_VERIFY_IN_GRAPH"
    a, b = ab.ARMS["dspark-both-misscut"][0], ab.ARMS["dspark-both-misscut-engram-verify-graph"][0]
    assert flag not in a and {k: v for k, v in b.items() if k != flag} == a and b[flag] == "1"
    env = {**ab.arm_env.base_env(), **ab._overrides("dspark-both-misscut-engram-verify-graph", "/out")}
    assert env["SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT"] == "1"
    assert ab.REFERENCE["dspark-both-misscut-engram-verify-graph"] == "dspark-both-misscut"


def test_each_cap2_arm_is_its_cap1_arm_with_two_rows_per_layer_alone():
    """Mutant: a cap-2 arm also changes another knob (or is built from the wrong arm), so the A/B does not read the
    budget alone -- red."""
    ab = _ab()
    for cap1, cap2 in ((L2_ARM, L2_CAP2_ARM), (T3_ARM, T3_CAP2_ARM)):
        a, b = ab.ARMS[cap1][0], ab.ARMS[cap2][0]
        assert (a[PER_LAYER], b[PER_LAYER]) == ("1", "2")
        assert {k: v for k, v in b.items() if k != PER_LAYER} == {k: v for k, v in a.items() if k != PER_LAYER}
        assert ab.ARMS[cap2][1] is True
        assert ab.REFERENCE[cap2] == "dspark-both-misscut"


def test_the_l2_arm_is_the_idle2_arm_with_horizon_2_alone_and_no_floors_against_the_miss_cut_arm():
    """Mutant: the arm also changes the floors or another idle2 knob, so the horizon is not what the A/B reads -- red."""
    ab = _ab()
    i, l2 = ab.ARMS["dspark-both-prefetch-gpu-misscut-idle2"][0], ab.ARMS[L2_ARM][0]
    assert (i[HORIZON], l2[HORIZON]) == ("1", "2")
    assert (i["SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"], l2["SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"]) == ("", "")
    assert {k: v for k, v in l2.items() if k != HORIZON} == {k: v for k, v in i.items() if k != HORIZON}
    assert ab.ARMS[L2_ARM][1] is True
    assert ab.REFERENCE[L2_ARM] == "dspark-both-misscut"


def test_every_other_arm_pins_horizon_1_against_an_exported_shell(monkeypatch):
    """Mutant: an arm leaves the horizon unset and inherits a shell's 2 -- red."""
    ab = _ab()
    monkeypatch.setenv(HORIZON, "3")
    for arm in ab.ARMS:
        want = "2" if arm in (L2_ARM, T3_ARM, L2_CAP2_ARM, T3_CAP2_ARM, T3_CAP3_ARM, *PROMOTE_PAIR_ARMS) else "1"
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[HORIZON] == want, arm


def test_the_t3_arm_is_the_l2_arm_with_the_head_file_alone_and_every_other_arm_pins_it_empty(monkeypatch):
    """Mutant: the T3 arm also changes another knob, or an arm inherits a shell's head path -- red."""
    ab = _ab()
    monkeypatch.setenv(LEARNED_HEAD, "/shell/head.safetensors")
    l2, t3 = ab.ARMS[L2_ARM][0], ab.ARMS[T3_ARM][0]
    assert (l2[LEARNED_HEAD], t3[LEARNED_HEAD]) == ("", ab.T3_HEAD)
    assert {k: v for k, v in t3.items() if k != LEARNED_HEAD} == {k: v for k, v in l2.items() if k != LEARNED_HEAD}
    assert ab.ARMS[T3_ARM][1] is True
    assert ab.REFERENCE[T3_ARM] == "dspark-both-misscut"
    for arm in ab.ARMS:
        want = ab.T3_HEAD if arm in (T3_ARM, T3_CAP2_ARM, T3_CAP3_ARM, *PROMOTE_PAIR_ARMS) else ""
        if arm in PROD_ARMS:
            want = ab.arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD  # production's checksummed copy of the same head
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[LEARNED_HEAD] == want, arm


def test_every_arm_pins_promote_off_but_the_promote_arm_against_an_exported_shell(monkeypatch):
    """Mutant: an arm leaves the flag unset and inherits a shell's 1, or the on arm is not on -- red."""
    ab = _ab()
    monkeypatch.setenv(PROMOTE, "1")
    for arm in ab.ARMS:
        want = "1" if arm.startswith(PROMOTE_ON) and not arm.removeprefix(PROMOTE_ON).startswith("-off") else "0"
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[PROMOTE] == want, arm


def test_every_arm_pins_pool_serve_off_but_the_pool_serve_arm_which_differs_from_promote_off_in_it_alone(monkeypatch):
    """Mutant: an arm inherits a shell's 1, the pool-serve arm changes another knob or is compared with the wrong
    arm -- red."""
    ab = _ab()
    monkeypatch.setenv(POOL_SERVE, "1")
    for arm in ab.ARMS:
        want = "1" if arm.startswith((POOL_SERVE_ARM, PROMOTE_POOL_SERVE_ARM)) else "0"
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[POOL_SERVE] == want, arm
    for serve, reference in ((POOL_SERVE_ARM, PROMOTE_OFF), (PROMOTE_POOL_SERVE_ARM, PROMOTE_ON)):
        on, off = ab.ARMS[serve][0], ab.ARMS[reference][0]
        assert {k: v for k, v in on.items() if k != POOL_SERVE} == {k: v for k, v in off.items() if k != POOL_SERVE}
        assert ab.REFERENCE[serve] == reference
        assert ab.REFERENCE[f"{serve}-fixedsplit"] == f"{reference}-fixedsplit"


def test_the_promote_pair_differs_in_the_flag_alone_and_the_on_arm_is_compared_with_its_off_twin():
    """Mutant: the twin also changes another knob, or the on arm is built from the wrong arm -- red."""
    ab = _ab()
    base = ab.ARMS[T3_CAP2_ARM][0]
    on, off = ab.ARMS[PROMOTE_ON][0], ab.ARMS[PROMOTE_OFF][0]
    assert (on[PROMOTE], off[PROMOTE]) == ("1", "0")
    assert {k: v for k, v in on.items() if k != PROMOTE} == {k: v for k, v in off.items() if k != PROMOTE}
    assert off == base
    assert ab.ARMS[PROMOTE_ON][1] is True and ab.REFERENCE[PROMOTE_ON] == PROMOTE_OFF
    assert ab.REFERENCE[PROMOTE_OFF] == "dspark-both-misscut"


def test_the_fixed_split_option_adds_the_table_and_calibration_off_to_both_arms_of_the_pair(monkeypatch, tmp_path):
    """Mutant: the option reaches one arm only, leaves calibration on, or drops the table -- red."""
    ab = _ab()
    monkeypatch.setenv(SPLIT, "0:3,1:5")
    for arm in (PROMOTE_ON, PROMOTE_OFF):
        plain, fixed = ab._overrides(arm, "/out"), ab._overrides(f"{arm}-fixedsplit", "/out")
        assert SPLIT not in plain and CALIBRATION not in plain
        assert (fixed[SPLIT], fixed[CALIBRATION]) == ("0:3,1:5", "0")
        assert {k: v for k, v in fixed.items() if k not in (SPLIT, CALIBRATION, METRICS)} == {
            k: v for k, v in plain.items() if k != METRICS
        }
        assert ab._probe_overrides(f"{arm}-fixedsplit", "/out")[SPLIT] == "0:3,1:5"
    monkeypatch.delenv(SPLIT)
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path), f"{PROMOTE_ON}-fixedsplit"])
    with pytest.raises(SystemExit, match=SPLIT):
        ab.main()


def test_the_cores13_arm_differs_from_its_base_in_node_1s_cpu_expert_cores_alone():
    """Mutant: the arm also changes node 0's team or another knob, or lists a core another thread holds -- red."""
    ab = _ab()
    base = ab.ARMS[f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit"][0]
    cores = ab.ARMS[f"{PROMOTE_POOL_SERVE_ARM}-cores13-fixedsplit"][0]
    knobs = ("SGLANG_DSV41_CPU_EXPERTS_THREADS", "SGLANG_EXPERT_NUMA_CORES")
    assert {k: v for k, v in cores.items() if k not in knobs} == {k: v for k, v in base.items() if k not in knobs}
    from sglang.srt.layers.moe.cpu_experts.threading_config import parse_numa_cores

    plan = parse_numa_cores(cores["SGLANG_EXPERT_NUMA_CORES"])
    assert plan[0]["cpu"] == list(range(6, 16))
    assert len(plan[1]["cpu"]) == int(cores["SGLANG_DSV41_CPU_EXPERTS_THREADS"]) == 13
    assert not set(plan[1]["cpu"]) & set(range(28, 32)) and 35 not in plan[1]["cpu"]
    # The launch refuses node 0's RAM core named both here and by the spin core the recipe sets.
    assert "ram" not in plan[0] and ab.arm_env.arm_env(cores)["SGLANG_DSV41_RAM_MISS_SPIN_CORE"]


def test_the_gate_arm_is_its_fixed_split_reference_with_the_gate_alone_and_every_other_arm_pins_it_off(monkeypatch):
    """Mutant: an arm inherits a shell's gate, the gate arm changes another knob, is compared with the wrong arm, or
    loses the driver's fixed split -- red."""
    ab = _ab()
    gate, arm = "SGLANG_DSPARK_TOP_PROB_GATE", f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit-confgate"
    reference = f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit"
    assert arm == "dspark-both-prefetch-gpu-misscut-idle2-l2-t3-cap2-promote-poolserve-fixedsplit-confgate"
    monkeypatch.setenv(gate, "0.9")
    monkeypatch.setenv(SPLIT, "0:3,1:5")
    for name in ab.ARMS:
        assert ab._overrides(name, "/out")[gate] == ("0.4" if name == arm else "0"), name
    on, off = ab.ARMS[arm][0], ab.ARMS[reference][0]
    assert {k: v for k, v in on.items() if k != gate} == {k: v for k, v in off.items() if k != gate}
    assert ab.ARMS[arm][1] is True and ab.REFERENCE[arm] == reference
    assert ab._overrides(arm, "/out")[SPLIT] == "0:3,1:5"


def test_the_gpu_drip_arm_is_the_promote_pool_serve_fixed_split_arm_with_the_drip_alone(monkeypatch):
    """Every other arm pins the drip off against a shell's slots; the drip arm adds the replay's settings to the arm it
    is compared with and takes the fixed split table as that arm does. Mutant: an arm inherits the shell's slots, the
    drip arm changes another knob or is compared with the wrong arm, or it loses the split table -- red."""
    ab = _ab()
    monkeypatch.setenv(GPU_DRIP, "7")
    monkeypatch.setenv(SPLIT, "0:3,1:5")
    for arm in ab.ARMS:
        want = "24" if arm in (GPU_DRIP_ARM, GPU_DRIP_LEAN_ARM) else "0"
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[GPU_DRIP] == want, arm
    base = f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit"
    drip, plain = ab.ARMS[GPU_DRIP_ARM][0], ab.ARMS[base][0]
    knobs = {GPU_DRIP: "24", "SGLANG_DSV41_GPU_DRIP_PICKS": "4", "SGLANG_DSV41_GPU_DRIP_CHUNK_KB": "256"}
    assert {k: drip[k] for k in knobs} == knobs and plain[GPU_DRIP] == "0"
    assert {k: v for k, v in drip.items() if k not in knobs} == {k: v for k, v in plain.items() if k not in knobs}
    assert ab.ARMS[GPU_DRIP_ARM][1] is True and ab.REFERENCE[GPU_DRIP_ARM] == base
    assert ab._overrides(GPU_DRIP_ARM, "/out")[SPLIT] == "0:3,1:5"
    assert {"drip_issued", "drip_landed", "drip_used", "drip_abandoned"} <= set(ab.RAM_KEYS)
    lean = ab.ARMS[GPU_DRIP_LEAN_ARM][0]
    lean_knobs = {"SGLANG_DSV41_GPU_DRIP_PICKS": "2", "SGLANG_DSV41_GPU_DRIP_CHUNK_KB": "64"}
    assert {k: lean[k] for k in lean_knobs} == lean_knobs
    assert {k: v for k, v in lean.items() if k not in lean_knobs} == {
        k: v for k, v in drip.items() if k not in lean_knobs
    }
    assert ab.REFERENCE[GPU_DRIP_LEAN_ARM] == base and ab._overrides(GPU_DRIP_LEAN_ARM, "/out")[SPLIT] == "0:3,1:5"


def test_the_pin_arms_add_their_knobs_alone_and_every_other_arm_pins_them_empty(monkeypatch):
    """pin01 is its fixed-split reference with layers 0 and 1 held whole on a budget raised for them, and gpu01 is pin01
    with the two layers' split tables; every other arm states both knobs empty against a shell's. The budget buys each
    full layer's 204 rows a node and leaves the other layers' 7:6 remainder at least today's rows. Mutant: an arm
    inherits a shell's list or tables, a pin arm changes another knob, is compared with the wrong arm, or loses the
    fixed split -- red."""
    ab = _ab()
    monkeypatch.setenv(FULL_LAYERS, "5")
    monkeypatch.setenv(SPLIT_LAYERS, "5:0")
    monkeypatch.setenv(SPLIT, "0:3,1:5")
    for arm in ab.ARMS:
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[FULL_LAYERS] == ("0,1" if arm in (PIN01_ARM, PIN01_GPU01_ARM, PIN01_SPLIT_MODEL_ARM,
                                                               PIN01_MEM_ARM, PIN01_SPLIT_MODEL_MEM_ARM,
                                                               *PIN01_HICACHE_ARMS)
                                              else ""), arm
            assert overrides[SPLIT_LAYERS] == (ab.ARMS[arm][0][SPLIT_LAYERS] if arm == PIN01_GPU01_ARM else ""), arm
    reference = f"{PROMOTE_POOL_SERVE_ARM}-fixedsplit"
    pin, plain, gpu = ab.ARMS[PIN01_ARM][0], ab.ARMS[reference][0], ab.ARMS[PIN01_GPU01_ARM][0]
    budget = {"SGLANG_MOE_PINNED_HOST_MB", "SGLANG_MOE_PINNED_HOST_NUMA_MB"}
    knobs = budget | {FULL_LAYERS}
    assert {k: v for k, v in pin.items() if k not in knobs} == {k: v for k, v in plain.items() if k not in knobs}
    assert {k: v for k, v in gpu.items() if k != SPLIT_LAYERS} == {k: v for k, v in pin.items() if k != SPLIT_LAYERS}
    assert ab.REFERENCE[PIN01_ARM] == reference and ab.REFERENCE[PIN01_GPU01_ARM] == PIN01_ARM
    for arm in (PIN01_ARM, PIN01_GPU01_ARM):
        assert ab.ARMS[arm][1] is True and ab._overrides(arm, "/out")[SPLIT] == "0:3,1:5"
    nodes = [(int(n), int(mib) << 20) for n, mib in (e.split(":") for e in pin["SGLANG_MOE_PINNED_HOST_NUMA_MB"].split(","))]
    assert sum(b for _, b in nodes) == int(pin["SGLANG_MOE_PINNED_HOST_MB"]) << 20
    row = 13_315_584  # one DSV4.1 EXL3 expert row (2026-10-14 startup log: 111664487424 bytes / 8386 rows)
    rest = [b - 2 * 204 * row for _, b in nodes]
    assert sum(rest) // row >= 24 * 210 + 14 * 209 and abs(rest[0] / rest[1] - 57344 / 49152) < 1e-3
    for spec in gpu[SPLIT_LAYERS].split(";"):
        _, table = spec.split(":")
        split = [int(v) for v in table.split(",")]
        fixed = [0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19,
                 20, 21, 21, 22, 23, 23, 24, 24, 25, 26, 26]
        assert len(split) == len(fixed) and all(a <= b for a, b in zip(split, fixed))  # only lanes to the link


def test_the_split_model_arm_is_pin01_with_the_model_on_and_every_other_arm_pins_it_off(monkeypatch):
    """-splitmodel is pin01 with SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL=1 at the default costs, compared with pin01; every
    other arm states the model off and its costs empty against a shell's. The fixed split stays: the model replaces it
    on the device only where the copy engine can take hits. Mutant: the arm changes another knob, an arm inherits the
    shell's flag or costs, or the arm is compared with the wrong one -- red."""
    ab = _ab()
    monkeypatch.setenv(SPLIT_MODEL, "1")
    monkeypatch.setenv(SPLIT_MODEL_COSTS, "copy_lane=9")
    monkeypatch.setenv(SPLIT_MODEL_REFINE, "7")
    for arm in ab.ARMS:
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[SPLIT_MODEL] == ("1" if arm in (PIN01_SPLIT_MODEL_ARM, PIN01_SPLIT_MODEL_MEM_ARM,
                                                             *PIN01_HICACHE_ARMS)
                                              else "0"), arm
            assert overrides[SPLIT_MODEL_COSTS] == (ab.SPLIT_MODEL_FIXED_COSTS if arm == PIN01_SPLIT_MODEL_FIXED_COSTS_ARM
                                                    else ""), arm
            assert overrides[SPLIT_MODEL_REFINE] == ("50" if arm == PIN01_SPLIT_MODEL_REFINE_ARM else "0"), arm
    model, pin = ab.ARMS[PIN01_SPLIT_MODEL_ARM][0], ab.ARMS[PIN01_ARM][0]
    assert {k: v for k, v in model.items() if k != SPLIT_MODEL} == {k: v for k, v in pin.items() if k != SPLIT_MODEL}
    assert ab.REFERENCE[PIN01_SPLIT_MODEL_ARM] == PIN01_ARM and ab.ARMS[PIN01_SPLIT_MODEL_ARM][1] is True


def test_the_fixed_cost_and_refine_arms_are_hicache1_with_one_knob_each():
    """-fixedcosts names every cost (the offline fits), so it skips the calibration; -refine is the calibrated model with
    the live refinement, compared with -fixedcosts. Both keep -hicache1's -mem server flag. Mutant: the fixed costs
    miss a key or differ from the fixtures, or -refine is compared with the calibrated arm -- red."""
    from sglang.srt.layers.moe.ram_slot_map import (
        SPLIT_MODEL_FIXTURE_MS,
        split_model_overrides_complete,
        split_model_words,
    )

    ab = _ab()
    base = ab.ARMS[PIN01_HICACHE_ARMS[0]][0]
    for arm, key in ((PIN01_SPLIT_MODEL_FIXED_COSTS_ARM, SPLIT_MODEL_COSTS), (PIN01_SPLIT_MODEL_REFINE_ARM,
                                                                            SPLIT_MODEL_REFINE)):
        other = ab.ARMS[arm][0]
        assert {k: v for k, v in other.items() if k != key} == {k: v for k, v in base.items() if k != key}, arm
        assert ab.server_flags(arm) == ("--weight-loader-drop-cache-after-load",) and ab.ARMS[arm][1] is True
    assert split_model_overrides_complete(ab.SPLIT_MODEL_FIXED_COSTS, 2)
    assert split_model_words(2, ab.SPLIT_MODEL_FIXED_COSTS) == split_model_words(2, base=SPLIT_MODEL_FIXTURE_MS)
    assert ab.REFERENCE[PIN01_SPLIT_MODEL_REFINE_ARM] == PIN01_SPLIT_MODEL_FIXED_COSTS_ARM
    assert ab.REFERENCE[PIN01_SPLIT_MODEL_FIXED_COSTS_ARM] == PIN01_HICACHE_ARMS[0]


def test_the_mem_arms_add_the_mlock_and_the_cache_drop_alone(monkeypatch):
    """-mem is its arm with the scheduler's mlockall and --weight-loader-drop-cache-after-load, nothing else; every
    other arm pins the mlock off against a shell's and adds no server flag; the capture and the timed run both append
    the arm's flags. The ON -mem arm is compared with the OFF -mem arm. Mutant: a -mem arm changes another knob, the
    flag is not appended, or an arm inherits the shell's mlock -- red."""
    ab = _ab()
    monkeypatch.setenv(MLOCK, "1")
    for arm in ab.ARMS:
        mem = "-mem-" in arm + "-"
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[MLOCK] == ("1" if mem else "0"), arm
        drop = ("--weight-loader-drop-cache-after-load",) if mem else ()
        assert ab.server_flags(arm) == drop + ab.ARM_MEMORY_FLAGS.get(arm, ()) + ab.ARM_SPEC_FLAGS.get(arm, ()), arm
    for base in (PIN01_ARM, PIN01_SPLIT_MODEL_ARM):
        mem, plain = ab.ARMS[base + "-mem"][0], ab.ARMS[base][0]
        assert {k: v for k, v in mem.items() if k != MLOCK} == {k: v for k, v in plain.items() if k != MLOCK}
        assert ab.ARMS[base + "-mem"][1] is True
    assert ab.REFERENCE[PIN01_MEM_ARM] == PIN01_ARM and ab.REFERENCE[PIN01_SPLIT_MODEL_MEM_ARM] == PIN01_MEM_ARM
    capture = open(os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "spec_margin_capture.py")).read()
    # long-prompt-mode moved the argv into spec_margin_capture.server_argv(arm).
    assert "*both_cpu_ab.server_flags(arm)]" in capture
    seen = {}
    monkeypatch.setattr(ab.subprocess, "run", lambda cmd, env, cwd: seen.update(env) or type("R", (), {"returncode": 0}))
    ab.run_timed(PIN01_MEM_ARM, "/out")
    assert seen["DSV41_EXTRA_SERVER_ARGS"].endswith("--weight-loader-drop-cache-after-load")


def test_the_hicache_and_engram_arms_add_their_knob_alone(monkeypatch):
    """-hicache1 is the ON -mem arm with the HiCache host pools on node 1, -engram0 that with the native Engram cache
    at 0; every other arm states the Engram cache at its default 5 against a shell's, and no other arm names the node.
    Both keep the -mem server flag. Mutant: an arm changes another knob or loses the cache drop -- red."""
    ab = _ab()
    monkeypatch.setenv("SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB", "0")
    hicache, engram0 = PIN01_SPLIT_MODEL_MEM_ARM + "-hicache1", PIN01_SPLIT_MODEL_MEM_ARM + "-hicache1-engram0"
    for arm in ab.ARMS:
        overrides = ab._overrides(arm, "/out")
        assert overrides["SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB"] == ("0" if arm == engram0 else "5"), arm
        assert ("SGLANG_HICACHE_HOST_NUMA_NODE" in ab.ARMS[arm][0]) == (arm in PIN01_HICACHE_ARMS), arm
    base, h, e = (ab.ARMS[a][0] for a in (PIN01_SPLIT_MODEL_MEM_ARM, hicache, engram0))
    assert {k: v for k, v in h.items() if k != "SGLANG_HICACHE_HOST_NUMA_NODE"} == base
    assert h["SGLANG_HICACHE_HOST_NUMA_NODE"] == "1"
    assert {k: v for k, v in e.items() if k != "SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB"} == \
        {k: v for k, v in h.items() if k != "SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB"}
    assert ab.server_flags(hicache) == ab.server_flags(engram0) == ("--weight-loader-drop-cache-after-load",)
    assert ab.REFERENCE[hicache] == PIN01_SPLIT_MODEL_MEM_ARM and ab.REFERENCE[engram0] == hicache


def test_the_decay_arm_is_arm_a_with_the_hot_decay_alone(monkeypatch):
    """-decay995 is -hicache1 with SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY=0.995; every other arm states 0.98 against a
    shell's. Mutant: another arm inherits the shell's decay, or the arm changes another knob -- red."""
    ab = _ab()
    monkeypatch.setenv("SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY", "0.9")
    decay, arm_a = PIN01_HICACHE_ARMS[2], PIN01_HICACHE_ARMS[0]
    for arm in ab.ARMS:
        assert ab._overrides(arm, "/out")["SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY"] == ("0.995" if arm == decay else "0.98")
    d, a = ab.ARMS[decay][0], ab.ARMS[arm_a][0]
    assert {k: v for k, v in d.items() if k != "SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY"} == \
        {k: v for k, v in a.items() if k != "SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY"}
    assert ab.REFERENCE[decay] == arm_a and ab.server_flags(decay) == ("--weight-loader-drop-cache-after-load",)


def test_the_tiebreak_arm_is_arm_a_with_the_tiebreak_alone(monkeypatch):
    """-tiebreak is -hicache1 with SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK=1; every other arm states 0 against
    a shell's. Mutant: another arm inherits the shell's flag, or the arm changes another knob -- red."""
    ab = _ab()
    monkeypatch.setenv("SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK", "1")
    tiebreak, arm_a = PIN01_HICACHE_ARMS[3], PIN01_HICACHE_ARMS[0]
    for arm in ab.ARMS:
        assert ab._overrides(arm, "/out")["SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK"] == (
            "1" if arm == tiebreak else "0"), arm
    t, a = ab.ARMS[tiebreak][0], ab.ARMS[arm_a][0]
    key = "SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK"
    assert {k: v for k, v in t.items() if k != key} == {k: v for k, v in a.items() if k != key}
    assert ab.REFERENCE[tiebreak] == arm_a and ab.server_flags(tiebreak) == ("--weight-loader-drop-cache-after-load",)


def test_the_mergechunk_arm_is_arm_a_with_the_merge_budget_and_its_hot_cache_alone(monkeypatch):
    """-mergechunk is -hicache1 with the triton merge budget and a hot cache sized by hot_mb_for_margin, launched with
    its own fraction and the KV pool pinned; every other arm states the budget 0 against a shell's and adds no memory
    flag. Mutant: another arm inherits the shell's budget, the arm changes another knob, or the pin is dropped -- red."""
    ab = _ab()
    budget = "SGLANG_SM120_TRITON_FLASHMLA_MERGE_BUDGET_MB"
    monkeypatch.setenv(budget, "64")
    merge, arm_a = PIN01_HICACHE_ARMS[4], PIN01_HICACHE_ARMS[0]
    merges = (merge, PIN01_HICACHE_ARMS[5], *PROD_ARMS)  # -mergechunk, -mergechunk-expseg and those built on it
    calib = PIN01_HICACHE_ARMS[6]  # -mergechunk-expseg-calib: its own budget (A3)
    for arm in ab.ARMS:
        want = str(ab.MERGE_CHUNK_MB) if arm in merges else "32" if arm == calib else "0"
        assert ab._overrides(arm, "/out")[budget] == want, arm
        assert (arm in ab.ARM_MEMORY_FLAGS) == (arm in merges + (calib,)), arm
    m, a = ab.ARMS[merge][0], ab.ARMS[arm_a][0]
    knobs = (budget, "SGLANG_MOE_HOT_GPU_MB")
    assert {k: v for k, v in m.items() if k not in knobs} == {k: v for k, v in a.items() if k not in knobs}
    assert m["SGLANG_MOE_HOT_GPU_MB"] == str(ab.hot_mb_for_margin())
    flags = ab.server_flags(merge)
    assert flags[0] == "--weight-loader-drop-cache-after-load"
    assert flags[flags.index("--max-total-tokens") + 1] == str(ab.MERGE_KV_TOKENS)
    fraction = flags[flags.index("--mem-fraction-static") + 1]
    assert fraction == ab.fraction_keeping_kv(ab.hot_mb_for_margin(), ab.MERGE_KV_TOKENS)
    assert ab.REFERENCE[merge] == arm_a


def test_the_expseg_arm_is_mergechunk_on_expandable_segments_with_its_hot_cache_alone(monkeypatch):
    """-expseg is -mergechunk with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and a hot cache sized from its smoke
    (923 slots, 2,040 MiB free, 29.39 GiB) less 3 held-back slots: 1035 slots, KV still pinned; every other arm states the
    allocator config empty against a shell's. Mutant: another arm inherits the shell's config, or the size moves -- red."""
    ab = _ab()
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.5")
    expseg, merge = PIN01_HICACHE_ARMS[5], PIN01_HICACHE_ARMS[4]
    on_expseg = (expseg, PIN01_HICACHE_ARMS[6], *PROD_ARMS)  # -expseg and the arms built on it
    for arm in ab.ARMS:
        assert ab._overrides(arm, "/out")["PYTORCH_CUDA_ALLOC_CONF"] == (
            "expandable_segments:True" if arm in on_expseg else ""), arm
    e, m = ab.ARMS[expseg][0], ab.ARMS[merge][0]
    knobs = ("PYTORCH_CUDA_ALLOC_CONF", "SGLANG_MOE_HOT_GPU_MB")
    assert {k: v for k, v in e.items() if k not in knobs} == {k: v for k, v in m.items() if k not in knobs}
    assert int(e["SGLANG_MOE_HOT_GPU_MB"]) * 2**20 // ab.HOT_SLOT_BYTES == 1035
    flags = ab.server_flags(expseg)
    assert flags[flags.index("--max-total-tokens") + 1] == "381184"
    assert flags[flags.index("--mem-fraction-static") + 1] == ab.fraction_keeping_kv(ab.EXPSEG_HOT_GPU_MB, 381184)
    assert ab.REFERENCE[expseg] == merge


def test_the_calib_arm_is_expseg_with_the_calibration_and_the_prefill_cuts_alone(monkeypatch):
    """-expseg-calib is -expseg with the startup hot-cache calibration and A1/A2/A3 (indexer budget 32, chunk kernels,
    merge budget 32), at -expseg's fraction and KV pin; every other arm states the calibration and the chunk kernels off
    against a shell's. Mutant: another arm inherits the shell's flag, or the arm changes another knob -- red."""
    ab = _ab()
    monkeypatch.setenv(ab.CALIBRATE, "1")
    monkeypatch.setenv(ab.CHUNK_KERNELS, "1")
    calib, expseg = PIN01_HICACHE_ARMS[6], PIN01_HICACHE_ARMS[5]
    for arm in ab.ARMS:
        overrides = ab._overrides(arm, "/out")
        # Production's arms calibrate the hot cache without the per-chunk kernels, as production does; the residency
        # pair's fixed smaller hot cache does not (test_the_residency_pair_...).
        residency_pair = (LAYER_MAJOR_SMALL_HOT_ARM, LAYER_MAJOR_RESIDENCY_ARM)
        want = (
            ("1", "1") if arm == calib
            else ("0", "0") if arm in residency_pair
            else ("1", "0") if arm in PROD_ARMS
            else ("0", "0")
        )
        assert (overrides[ab.CALIBRATE], overrides[ab.CHUNK_KERNELS]) == want, arm
    c, e = ab.ARMS[calib][0], ab.ARMS[expseg][0]
    knobs = {ab.CALIBRATE: "1", ab.CHUNK_KERNELS: "1", ab.MERGE_BUDGET: "32", ab.INDEXER_BUDGET: "32"}
    assert {k: v for k, v in c.items() if k not in knobs} == {k: v for k, v in e.items() if k not in knobs}
    assert {k: c[k] for k in knobs} == knobs
    assert ab.server_flags(calib) == ab.server_flags(expseg) and ab.REFERENCE[calib] == expseg


def test_hot_mb_for_margin_buys_whole_slots_of_the_measured_spare():
    """At the probe's own free memory and margin the arm keeps the probe's slots; each slot of extra margin costs one
    slot, and the MiB it returns floors back to that slot count."""
    ab = _ab()
    slot = ab.HOT_SLOT_BYTES
    for margin in (0, 93, 94, 200, 508):
        mb = ab.hot_mb_for_margin(margin)
        slots = mb * 2**20 // slot
        assert -(-slots * slot // 2**20) == mb
        worst = ab.PROBE_FREE_AT_PEAK_MIB * 2**20 - (ab.PROBE_PRE_LOAD_GIB - ab.MIN_PRE_LOAD_GIB) * 2**30
        assert slots == ab.PROBE_HOT_SLOTS + int((worst - margin * 2**20) // slot)
        assert worst - (slots - ab.PROBE_HOT_SLOTS) * slot >= margin * 2**20


def test_the_sessions_flag_reaches_the_timed_run_and_defaults_to_the_eight_corpus_sessions(monkeypatch, tmp_path):
    """Mutant: the smoke's subset is ignored and all eight sessions run -- red."""
    ab = _ab()
    seen = []
    monkeypatch.setattr(
        ab.subprocess, "run", lambda cmd, env, cwd: seen.append(env["DSV41_SESSION_INDICES"]) or type("R", (), {"returncode": 0})()
    )
    monkeypatch.setattr(ab, "run_probe", lambda arm, out: 0)
    monkeypatch.setattr(ab, "summarize", lambda out: {})
    for argv in ([], ["--sessions", "0,1"]):
        monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path), "dspark-both", *argv])
        ab.main()
    assert seen == ["0,1,2,3,4,5,6,7", "0,1"]


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


def _drive_rows(busy_ms_by_session):
    def counters(sectors, busy):
        return {"nvme4": {"device": "nvme2n1p1", "reads": sectors // 1000, "sectors_read": sectors, "writes": 0,
                          "sectors_written": 0, "io_ticks_ms": busy, "time_in_queue_ms": 0}}  # fmt: skip

    rows = [{"label": "before_server", "diskstats_sectors": {}}]
    for t, (sid, busy) in enumerate(busy_ms_by_session):
        rows.append({"label": "session_start", "session_id": sid, "monotonic": t, "drive_counters": counters(0, 0)})
        rows.append({"label": "session_end", "session_id": sid, "monotonic": t + 1,
                     "drive_counters": counters(10**9 // 512 * 2, busy)})  # fmt: skip
        rows.append({"label": f"session_{sid}"})
    return "".join(json.dumps(r) + "\n" for r in rows)


def test_summarize_adds_per_session_drive_load_and_flags_across_both_arms_and_keeps_the_rest(tmp_path):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both-misscut", rows_read=3000)
    _arm_run(tmp_path, "dspark-both-misscut-maxm8", rows_read=2000)
    # 2 GB per session; busy s per GB 0.31, 0.32 | 0.31, 0.45 -> median 0.315, and 0.45 is +43%.
    for arm, sessions in (("dspark-both-misscut", [("s0", 620), ("s1", 640)]),
                          ("dspark-both-misscut-maxm8", [("s0", 620), ("s1", 900)])):  # fmt: skip
        (tmp_path / "servers" / arm / "run-1" / "boundary-samples.jsonl").write_text(_drive_rows(sessions))
    summary = ab.summarize(str(tmp_path))
    a, b = summary["dspark-both-misscut"], summary["dspark-both-misscut-maxm8"]
    assert a["ram"]["rows_read"] == 3000 and b["text_vs_reference"]["pass"] is True  # the existing fields stay
    assert [s["session_id"] for s in b["drives_per_session"]["sessions"]] == ["s0", "s1"]
    assert b["drives_per_session"]["sessions"][1]["drives"]["nvme4"]["busy_s_per_gb"] == pytest.approx(0.45)
    assert a["drive_flags"] == []
    assert [(f["session_id"], f["drive"]) for f in b["drive_flags"]] == [("s1", "nvme4")]
    assert json.loads((tmp_path / "summary.json").read_text())["dspark-both-misscut-maxm8"]["drive_flags"]


def test_an_arm_without_per_session_drive_counters_gets_a_note_not_an_error(tmp_path):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both")
    summary = ab.summarize(str(tmp_path))["dspark-both"]
    assert summary["drives_per_session"]["sessions"] == [] and summary["drives_per_session"]["notes"]
    assert summary["drive_flags"] == [] and summary["ram"]["rows_read"] == 2000


def test_main_prints_the_per_session_drive_table(tmp_path, monkeypatch, capsys):
    ab = _ab()
    _arm_run(tmp_path, "dspark-both")
    (tmp_path / "servers" / "dspark-both" / "run-1" / "boundary-samples.jsonl").write_text(_drive_rows([("s0", 620)]))
    monkeypatch.setattr(ab, "run_timed", lambda arm, out, **_: 0)
    monkeypatch.setattr(ab, "run_probe", lambda arm, out: 0)
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path), "dspark-both"])
    ab.main()
    out = capsys.readouterr().out
    assert "busy s per GB" in out and "0.31" in out


def test_run_arm_brackets_the_warm_up_and_every_timed_session_with_drive_counters():
    script = open(os.path.join(ROOT, "benchmarks", "dsv41_baseline", "run_arm.sh")).read()
    loop = script[script.index('for session_id in "${timed_session_ids[@]}"'):]
    loop = loop[: loop.index("\ndone\n")]
    start, run, end = (loop.index(s) for s in (
        "drive_sample 'session_start' \"$session_id\"", "run_capture_sessions.py",
        "drive_sample 'session_end' \"$session_id\"",
    ))  # fmt: skip
    assert start < run < end  # the samples bracket the session's own requests
    body = script[script.index("drive_sample() {"):]
    assert "provenance.drive_sample()" in body[: body.index("\n}\n")]
    warm = script.index("for round in $(seq 1")
    assert script.index("drive_sample 'warmup_start'") < warm < script.index("drive_sample 'warmup_end'")
    assert script.index("drive_sample 'warmup_end'") < script.index("label': 'server_ready'")


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


TWO_STAGE = "SGLANG_DSV41_CPU_TWO_STAGE"


def test_the_two_stage_arm_is_the_miss_cut_arm_with_the_two_stage_cpu_miss_on():
    ab = _ab()
    a, t = ab.ARMS["dspark-both-misscut"][0], ab.ARMS["dspark-both-misscut-twostage"][0]
    assert t[TWO_STAGE] == "1"
    assert {k: v for k, v in t.items() if k != TWO_STAGE} == {k: v for k, v in a.items() if k != TWO_STAGE}
    assert ab.ARMS["dspark-both-misscut-twostage"][1] is True
    assert ab.REFERENCE["dspark-both-misscut-twostage"] == "dspark-both-misscut"


def test_every_arm_but_the_two_stage_pins_it_off_against_an_exported_shell(monkeypatch):
    ab = _ab()
    monkeypatch.setenv(TWO_STAGE, "1")
    for arm in ab.ARMS:
        want = "1" if arm.startswith("dspark-both-misscut-twostage") else "0"
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[TWO_STAGE] == want, arm


SPAN = ("SGLANG_DSV41_CPU_TWO_STAGE_W13_WEIGHTS", "SGLANG_DSV41_CPU_TWO_STAGE_W2_WEIGHTS")


def test_the_spcc13_arm_is_the_two_stage_arm_with_the_spcc_on_w13_only():
    ab = _ab()
    t, c = ab.ARMS["dspark-both-misscut-twostage"][0], ab.ARMS["dspark-both-misscut-twostage-spcc13"][0]
    assert (c[SPAN[0]], c[SPAN[1]]) == ("6:10:6", "1:0:1")  # nvme0 : nvme4 (SPCC) : nvme2
    assert {k: v for k, v in c.items() if k not in SPAN} == {k: v for k, v in t.items() if k not in SPAN}
    assert ab.ARMS["dspark-both-misscut-twostage-spcc13"][1] is True
    assert ab.REFERENCE["dspark-both-misscut-twostage-spcc13"] == "dspark-both-misscut"


def test_every_arm_but_spcc13_pins_the_span_weights_unset_against_an_exported_shell(monkeypatch):
    ab = _ab()
    for name in SPAN:
        monkeypatch.setenv(name, "1:1:1")
    for arm in ab.ARMS:
        want = ("6:10:6", "1:0:1") if arm == "dspark-both-misscut-twostage-spcc13" else ("", "")
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert (overrides[SPAN[0]], overrides[SPAN[1]]) == want, arm


def test_prod_is_the_best_served_arm_with_both_startup_calibrations_and_the_checksummed_head():
    # MERGE_PLAN_1016.md: production = -mergechunk-expseg (the best served arm, prefetch on: owner, 2026-10-16) with the
    # table and hot-cache calibrations on (owner rule) and the prefetch head's production copy; every other key the arm
    # states is at its library default.
    ab = _ab()
    arm_env = ab.arm_env
    arm = ab.MERGE_CHUNK_EXPSEG_ARM
    best, prod = arm_env.arm_env(ab.ARMS[arm][0]), arm_env.prod_env()
    differs = {k: (best.get(k), prod.get(k)) for k in set(best) | set(prod) if best.get(k) != prod.get(k)}
    head = "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD"
    assert differs.pop(head) == (ab.T3_HEAD, arm_env.PROD_RAM_PREFETCH_LEARNED_HEAD)
    prefetch = set()
    stated_defaults = {
        "SGLANG_DSPARK_TOP_PROB_GATE": "0",
        STS_COLLECT: "",
        "SGLANG_DSPARK_USE_GREEDY_DRAFT": "0",
        "SGLANG_DSV41_CPU_EXPERTS_SPLIT_LAYERS": "",
        "SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK": "0",
        "SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB": "5",
        "SGLANG_DSV41_GPU_DRIP_SLOTS": "0",
        "SGLANG_MOE_EXPERT_MIRROR_CAPS": "",
        "SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": "0",
        "SGLANG_MOE_HOT_INSERT_ON_MISS_DECAY": "0.98",
    }
    calibrations = {"SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION": ("0", "1"),
                    "SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP": ("0", "1")}
    # Layer-major prefill under DSpark: production since 2026-10-17 (owner), never served on the best arm.
    layer_major = {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": ("0", "8192"),
                   "SGLANG_ENABLE_LAYER_MAJOR_PREFILL_DSPARK": (None, "1"),
                   # Lent residency rows: production since 2026-10-18 (owner), after the served lend validation.
                   "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS": (None, "360")}
    # The sampled confidence cut: production since 2026-10-18 (owner), after the served sampled A/B (DO 3.12). The refit
    # interval 0 is the library default, stated by the arms.
    cut = {k: ("0", "1") for k in ("SGLANG_DSPARK_CONFIDENCE_CUT", "SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED",
                                   "SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST")}
    # The live T(w) refit every 32 timed cut verifies: production since 2026-10-20 (owner); the non-prod arms leave it unset.
    cut["SGLANG_DSPARK_CONFIDENCE_CUT_LIVE_FIT_INTERVAL"] = (None, "32")
    rest ={k: v for k, v in differs.items() if k not in prefetch}
    assert rest == {**{k: (v, None) for k, v in stated_defaults.items()}, **calibrations, **layer_major, **cut}
    # Its server flags are production's: the page-cache drop, its fraction and the KV pin.
    argv = arm_env.ServerArgs.prod().argv()
    flags = ab.server_flags(arm)
    assert flags == ("--weight-loader-drop-cache-after-load", "--mem-fraction-static",
                     arm_env.PROD_DSPARK_MEM_FRACTION_STATIC, "--max-total-tokens",
                     str(arm_env.PROD_DSPARK_MAX_TOTAL_TOKENS))
    assert ab.EXPSEG_HOT_GPU_MB == int(arm_env.PROD_DSPARK_HOT_GPU_MB)
    assert all(flag in argv for flag in flags)


def test_the_prod_arm_is_production_with_the_arms_stated_defaults_and_takes_no_fixed_split(monkeypatch, tmp_path):
    """-prod is production's env and flags, plus the off-states every arm pins, and runs without a driver split table
    (production calibrates it). Mutant: the arm drops a production setting, keeps -expseg's head or calibration off, or
    takes the shell's split -- red."""
    ab = _ab()
    arm_env = ab.arm_env
    assert ab.PROD_ARM == PROD_ARM and ab.PROD_ARMS == PROD_ARMS
    env, prod = arm_env.arm_env(ab.ARMS[PROD_ARM][0]), arm_env.prod_env()
    assert {k: v for k, v in env.items() if k in prod} == prod
    extra = {k: v for k, v in env.items() if k not in prod}
    assert all(v in ("", "0", "5", "0.98") for v in extra.values()), extra  # stated off or at the library default
    assert extra[SPLIT] == "" and extra[STS_COLLECT] == ""
    # The sampled cut is production's (arm_env.PROD_DSPARK_CUT_ENV) and runs on the fitted STS file.
    assert {k: env[k] for k in ab.arm_env.PROD_DSPARK_CUT_ENV} == ab.arm_env.PROD_DSPARK_CUT_ENV
    assert env[CONFIDENCE_CUT] == "1" and env["SGLANG_DSPARK_USE_GREEDY_DRAFT"] == "0"
    assert ab.ARMS[PROD_ARM][1] is True and ab.REFERENCE[PROD_ARM] == PIN01_HICACHE_ARMS[5]
    assert ab.server_flags(PROD_ARM) == ab.server_flags(PIN01_HICACHE_ARMS[5]) + (STS_PATH_FLAG, arm_env.PROD_DSPARK_STS_PATH)
    assert all(flag in arm_env.ServerArgs.prod().argv() for flag in ab.server_flags(PROD_ARM))
    monkeypatch.setenv(SPLIT, "0:3,1:5")
    for arm in PROD_ARMS:
        assert ab._overrides(arm, "/out")[SPLIT] == "", arm
    monkeypatch.delenv(SPLIT)
    ran = []
    monkeypatch.setattr(ab, "run_timed", lambda arm, out, **_: ran.append(arm) or 0)
    monkeypatch.setattr(ab, "run_probe", lambda arm, out: 0)
    monkeypatch.setattr(ab, "summarize", lambda out: {})
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path), *PROD_ARMS])
    ab.main()
    assert ran == list(PROD_ARMS)


def test_the_cut_arms_are_prod_or_its_greedy_cut_and_every_other_arm_pins_the_cut_and_the_collection_off(monkeypatch):
    """Production runs the sampled cut (arm_env.PROD_DSPARK_CUT_ENV) on the fitted STS file, so every production arm
    carries the cut and the STS flag, except the collect arm, which has no STS file (collect mode: the head built, the cut
    off). -confcut and -stscollect state the greedy-only cut they ran as (the sampled path and the draft's cost off). Every
    other arm states the flag 0 and the collect path empty against a shell's. Mutant: an arm inherits the shell's flag or
    path, the collect arm passes the STS file, a production arm drops it, or a greedy cut arm takes the sampled path -- red."""
    ab = _ab()
    monkeypatch.setenv(CONFIDENCE_CUT, "1")
    monkeypatch.setenv(STS_COLLECT, "/shell/sts")
    for arm in ab.ARMS:
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides[CONFIDENCE_CUT] == ("1" if arm in PROD_ARMS else "0"), arm
            assert (overrides[STS_COLLECT] != "") == (arm == STS_COLLECT_ARM), arm
        assert (STS_PATH_FLAG in ab.server_flags(arm)) == (arm in PROD_ARMS and arm != STS_COLLECT_ARM), arm
    prod = ab.ARMS[PROD_ARM][0]
    greedy = {CONFIDENCE_CUT: "1", "SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED": "0", "SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST": "0"}
    for arm in (STS_COLLECT_ARM, CONFIDENCE_CUT_ARM):
        cut = ab.ARMS[arm][0]
        assert {k: v for k, v in cut.items() if k in greedy} == greedy, arm
        assert {k: v for k, v in cut.items() if k not in greedy} == {k: v for k, v in prod.items() if k not in greedy}, arm
        assert ab.ARMS[arm][1] is True and ab.REFERENCE[arm] == PROD_ARM
    flags = ab.server_flags(CONFIDENCE_CUT_ARM)
    assert flags == ab.server_flags(PROD_ARM)
    assert flags[-2:] == (STS_PATH_FLAG, ab.arm_env.PROD_DSPARK_STS_PATH)
    assert ab.server_flags(STS_COLLECT_ARM) == ab.server_flags(PIN01_HICACHE_ARMS[5])


def test_the_sampled_cut_arm_is_prod_and_the_block_arm_adds_block_verification_alone(monkeypatch):
    """-confcut-sampled ran the sampled A/B (2026-10-18) as -prod plus the cut; the cut shipped, so the arm now equals
    -prod and states arm_env.PROD_DSPARK_CUT_ENV. -confcut-sampled-block is -prod plus --speculative-use-block-verification
    alone. The draft stays sampled: SGLANG_DSPARK_USE_GREEDY_DRAFT is 0 in every arm, whatever the shell exports. Every
    non-production arm pins the sampled knobs off. Mutant: an arm drops a knob, turns the greedy draft on, the block arm
    changes env, or block verification reaches another arm -- red."""
    ab = _ab()
    for key in SAMPLED_KNOBS:
        monkeypatch.setenv(key, "1")
    for arm in ab.ARMS:
        for overrides in (ab._overrides(arm, "/out"), ab._probe_overrides(arm, "/out")):
            assert overrides["SGLANG_DSPARK_USE_GREEDY_DRAFT"] == "0", arm
            if arm not in PROD_ARMS:
                assert all(overrides[k] == "0" for k in SAMPLED_KNOBS), arm
        assert (BLOCK_FLAG in ab.server_flags(arm)) == (arm == SAMPLED_CUT_BLOCK_ARM), arm
    prod = ab.ARMS[PROD_ARM][0]
    assert {k: prod[k] for k in ab.arm_env.PROD_DSPARK_CUT_ENV} == ab.arm_env.PROD_DSPARK_CUT_ENV
    assert prod["SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL"] == "0" and prod["SGLANG_DSPARK_USE_GREEDY_DRAFT"] == "0"
    for arm in (SAMPLED_CUT_ARM, SAMPLED_CUT_BLOCK_ARM):
        assert ab.ARMS[arm] == ab.ARMS[PROD_ARM], arm
        assert arm in ab.PROD_ARMS
    assert ab.REFERENCE[SAMPLED_CUT_ARM] == PROD_ARM and ab.REFERENCE[SAMPLED_CUT_BLOCK_ARM] == SAMPLED_CUT_ARM
    sts = (STS_PATH_FLAG, ab.arm_env.PROD_DSPARK_STS_PATH)
    assert ab.ARM_SPEC_FLAGS[SAMPLED_CUT_ARM] == ab.ARM_SPEC_FLAGS[PROD_ARM] == sts
    assert ab.ARM_SPEC_FLAGS[SAMPLED_CUT_BLOCK_ARM] == (*sts, BLOCK_FLAG)
    assert ab.server_flags(SAMPLED_CUT_ARM) == ab.server_flags(PROD_ARM)
    assert ab.server_flags(SAMPLED_CUT_BLOCK_ARM) == ab.server_flags(PROD_ARM) + (BLOCK_FLAG,)


def test_the_production_arms_prefill_layer_major_and_every_other_arm_keeps_it_off():
    """Production prefills layer-major under DSpark since 2026-10-17 (arm_env.prod_dspark_env), so -prod and every arm
    built on it carry the 8192-token threshold and the DSpark switch, and -prod-layermajor now equals -prod. No other
    arm sets either (dspark_env() keeps the threshold 0 beneath them). Mutant: another arm turns it on, or production
    drops it -- red."""
    ab = _ab()
    threshold, switch = "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS", "SGLANG_ENABLE_LAYER_MAJOR_PREFILL_DSPARK"
    for arm, (overrides, _) in ab.ARMS.items():
        if arm == CHUNKED_ARM:
            # The parity reference: production with layer-major prefill off, and the switch unstated.
            assert overrides[threshold] == "0" and switch not in overrides
            rows = "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS"  # residency rows need layer-major prefill: off too
            assert overrides[rows] == "0"
            assert {k: v for k, v in ab.ARMS[PROD_ARM][0].items() if k not in (threshold, switch, rows)} == {
                k: v for k, v in overrides.items() if k not in (threshold, rows)
            }
        elif arm in PROD_ARMS:
            assert (overrides[threshold], overrides[switch]) == ("8192", "1"), arm
        else:
            assert overrides.get(threshold, "0") == "0" and switch not in overrides, arm
    assert ab.arm_env.dspark_env()[threshold] == "0"
    assert ab.ARMS[LAYER_MAJOR_ARM][0] == ab.ARMS[PROD_ARM][0]
    assert ab.ARMS[LAYER_MAJOR_ARM][1] is True and ab.REFERENCE[LAYER_MAJOR_ARM] == PROD_ARM
    assert ab.server_flags(LAYER_MAJOR_ARM) == ab.server_flags(PROD_ARM)


def test_the_residency_pair_differs_only_in_the_residency_rows_and_both_give_up_that_much_hot_cache():
    """-smallhot is -layermajor with the hot cache RESIDENCY_ROWS slots smaller and the startup calibration off (it
    would grow back into the buffer's memory) and the rows off; -res adds the rows with the lend off (stage 0's allocated
    buffer). Production has the rows (and the lend, by default) since 2026-10-18, so only the arms built on non-production
    recipes leave the rows unset. Mutant: the pair differs in another knob, or the hot cache keeps the memory the buffer
    needs -- red."""
    ab = _ab()
    residency, hot = "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS", "SGLANG_MOE_HOT_GPU_MB"
    lend = "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND"
    for arm, (overrides, _) in ab.ARMS.items():
        assert (residency in overrides) == (arm in PROD_ARMS), arm
        want = "0" if arm in (LAYER_MAJOR_SMALL_HOT_ARM, CHUNKED_ARM) else str(ab.RESIDENCY_ROWS)
        assert overrides.get(residency, want) == want, arm
    lm, small, res = (ab.ARMS[a][0] for a in LAYER_MAJOR_ARMS)
    assert {k: v for k, v in res.items() if k not in (residency, lend)} == {
        k: v for k, v in small.items() if k != residency
    }
    assert res[residency] == str(ab.RESIDENCY_ROWS) and res[lend] == "0" and small[residency] == "0"
    assert {k: v for k, v in small.items() if k not in (hot, ab.CALIBRATE, residency)} == {
        k: v for k, v in lm.items() if k not in (hot, ab.CALIBRATE, residency)
    }
    assert small[ab.CALIBRATE] == "0"
    given_up = (int(lm[hot]) - int(small[hot])) * 2**20
    assert ab.RESIDENCY_ROWS * ab.HOT_SLOT_BYTES <= given_up < ab.RESIDENCY_ROWS * ab.HOT_SLOT_BYTES + 2**20
    assert ab.REFERENCE[LAYER_MAJOR_RESIDENCY_ARM] == LAYER_MAJOR_SMALL_HOT_ARM
    assert all(ab.server_flags(a) == ab.server_flags(PROD_ARM) for a in LAYER_MAJOR_ARMS)


def test_the_lend_arm_is_production_with_the_lend_stated():
    """-lend keeps production's full hot cache and calibration (the decode bar is the full hot cache) and equals -prod,
    which has the rows (and the lend by default) since 2026-10-18, with the lend stated 1; only it and -res (lend 0) state
    the lend. Mutant: the arm shrinks the hot cache, drops the lend, or another arm sets it -- red."""
    ab = _ab()
    lend = "SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_LEND"
    for arm, (overrides, _) in ab.ARMS.items():
        assert (lend in overrides) == (arm in (LAYER_MAJOR_LEND_ARM, LAYER_MAJOR_RESIDENCY_ARM)), arm
    prod, arm = ab.ARMS[PROD_ARM][0], ab.ARMS[LAYER_MAJOR_LEND_ARM][0]
    assert arm[lend] == "1" and arm["SGLANG_LAYER_MAJOR_EXPERT_RESIDENCY_ROWS"] == str(ab.RESIDENCY_ROWS)
    assert {k: v for k, v in arm.items() if k != lend} == prod
    assert ab.ARMS[LAYER_MAJOR_LEND_ARM][1] is True and ab.REFERENCE[LAYER_MAJOR_LEND_ARM] == PROD_ARM
    assert ab.server_flags(LAYER_MAJOR_LEND_ARM) == ab.server_flags(PROD_ARM)
    assert LAYER_MAJOR_LEND_ARM in ab.PROD_ARMS


def test_the_collect_arm_writes_its_shards_under_out_and_the_probe_server_its_own(monkeypatch):
    """The timed server, the probe server and a capture each collect into their own directory, at the stem the fit's
    --data-glob '<stem>.*.pt' names. Mutant: the probe shares the timed stem (its shard 0 overwrites the timed one) or
    the capture leaves the path empty -- red."""
    ab = _ab()
    timed = ab._overrides(STS_COLLECT_ARM, "/out")[STS_COLLECT]
    probe = ab._probe_overrides(STS_COLLECT_ARM, "/out")[STS_COLLECT]
    assert timed == f"/out/{STS_COLLECT_ARM}/sts/confidence"
    assert probe == f"/out/{STS_COLLECT_ARM}.probe/sts/confidence"
    spec = importlib.util.spec_from_file_location(
        "spec_margin_capture", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "spec_margin_capture.py")
    )
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    assert cap.server_env("/cap", False, arm=STS_COLLECT_ARM)[STS_COLLECT] == "/cap/sts/confidence"
    assert cap.server_env("/cap", False, arm=CONFIDENCE_CUT_ARM)[STS_COLLECT] == ""
    assert STS_PATH_FLAG in cap.server_argv(CONFIDENCE_CUT_ARM)
    assert STS_PATH_FLAG not in cap.server_argv(STS_COLLECT_ARM)
    # The production arms run production's warm-ups; the others keep the prefill-only one.
    for arm in ab.ARMS:
        argv = cap.server_argv(arm)
        want = "dsv41_prefill_shapes,dsv41_decode_paths" if arm in PROD_ARMS else "dsv41_prefill_shapes"
        assert argv[argv.index("--warmups") + 1] == want, arm


def test_the_cut_arms_probe_server_gets_the_sts_file_and_no_other_arms_does(monkeypatch, tmp_path):
    """The probe server launches the arm too, so it needs the cut's STS file; without it the launch refuses the cut.
    Mutant: run_probe builds its argv without the arm's speculative flags -- red."""
    import contextlib

    ab = _ab()
    argvs = []

    class Server:
        pid = 1

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(ab.subprocess, "Popen", lambda cmd, **_: argvs.append(cmd) or Server())
    monkeypatch.setattr(ab, "_healthy", lambda port, deadline: False)  # no client: run_probe returns 1
    monkeypatch.setattr(ab.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(ab.run_locks, "take", lambda path: contextlib.nullcontext())
    for arm in PROD_ARMS:
        assert ab.run_probe(arm, str(tmp_path)) == 1
    prod, collect, cut, sampled, block, *layer_major = argvs
    for argv in (prod, cut, sampled, block):
        assert argv[argv.index(STS_PATH_FLAG) + 1] == ab.arm_env.PROD_DSPARK_STS_PATH
    assert BLOCK_FLAG in block and BLOCK_FLAG not in sampled and BLOCK_FLAG not in cut and BLOCK_FLAG not in prod
    assert all(a[a.index("--warmups") + 1] == "dsv41_prefill_shapes,dsv41_decode_paths" for a in argvs)
    assert STS_PATH_FLAG not in collect
    assert len(layer_major) == len(LAYER_MAJOR_ARMS) + 2 and all(STS_PATH_FLAG in a for a in layer_major)
