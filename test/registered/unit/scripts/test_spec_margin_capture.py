"""The prefetch margin capture's server env and its counters summary (CPU)."""

import importlib.util
import json
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _trace(env):
    return json.loads(env["SGLANG_EXPERT_STREAM_TRACE"])

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _capture():
    spec = importlib.util.spec_from_file_location(
        "spec_margin_capture", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "spec_margin_capture.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_capture_scores_where_asked_and_otherwise_runs_the_prefetch_arm():
    cap = _capture()
    cpu, gpu = cap.server_env("/out", False), cap.server_env("/out", False, "gpu")
    assert (cpu["SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"], gpu["SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"]) == ("cpu", "gpu")
    assert {k: v for k, v in gpu.items() if k != "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"} == {
        k: v for k, v in cpu.items() if k != "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"
    }
    assert _trace(gpu)["job"]["prefix"] == "/out/events"


@pytest.mark.parametrize("warn_only", [False, True])
def test_a_trace_overflow_refusal_fails_the_capture_unless_analysis_warn_only_and_then_names_the_dropped_count(
    monkeypatch, tmp_path, capsys, warn_only
):
    """Default: a nonzero layer_misses/spec_margin exit is the capture's exit. --analysis-warn-only: exit 0 with a
    WARNING that carries the per-file dropped-event count. Mutant: the flag is ignored, or applies to the default -- red."""
    import argparse

    cap = _capture()
    (tmp_path / "events.1.exl3-copy-eng.2.jsonl").write_text('{"e":1}\n{"dropped":181212}\n')
    (tmp_path / "events.1.exl3-tier0.0.jsonl").write_text('{"e":1}\n{"dropped":0}\n')
    ran = []
    monkeypatch.setattr(cap, "server_env", lambda *a, **k: {"SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH": "1"})
    monkeypatch.setattr(cap.subprocess, "run",
                        lambda argv, **_: ran.append(os.path.basename(argv[1])) or argparse.Namespace(
                            returncode=1 if "spec_margin" in argv[1] else 0))
    a = argparse.Namespace(out_dir=str(tmp_path), top_k_only=False, scorer=None, arm="x", verify_split=False,
                           analysis_warn_only=warn_only)
    assert cap.post_run(a) == (0 if warn_only else 1)
    assert ran == ["layer_misses.py", "spec_margin.py"]
    out = capsys.readouterr().out
    assert ("WARNING" in out and "181212" in out and "exl3-copy-eng" in out) == warn_only
    assert cap.dropped_events(str(tmp_path)) == {"events.1.exl3-copy-eng.2.jsonl": 181212}


def test_the_counter_summary_reads_the_last_counters_line(tmp_path):
    cap = _capture()
    log = tmp_path / "server.log"
    first = {"spec_scored": 1, "spec_score_ns": 1, "spec_used": 1, "spec_promoted": 1}
    last = {"spec_scored": 10, "spec_score_ns": 50_000, "spec_used": 40, "spec_promoted": 4, "spec_late": 2}
    log.write_text("".join("exl3 RAM miss thread counters " + json.dumps(c) + "\n" for c in (first, last)))
    s = cap.counter_summary(str(log))
    assert s["wait_or_score_us_per_record"] == pytest.approx(5.0)
    assert s["in_flight_at_use"] == pytest.approx(0.1) and s["spec_late"] == 2
    assert cap.counter_summary(str(tmp_path / "missing.log")) == {}


def test_the_counter_summary_carries_the_promotion_counters(tmp_path):
    cap = _capture()
    keys = ("rows_read", "spec_promoted_early", "spec_promote_no_delta", "spec_promote_no_room",
            "spec_promote_no_victim", "spec_promote_victim_routed", "spec_promote_used",
            "spec_promote_reading", "spec_publish_backstop", "spec_promote_row_busy", "spec_pool_served",
            "spec_skip_landed_gpu",
            "spec_skip_landed_cpu", "spec_skip_reading_gpu", "spec_skip_reading_cpu",
            "drip_issued", "drip_landed", "drip_used", "drip_abandoned", "drip_no_slot",
            "drip_paused", "drip_skipped_bytes", "drip_late", "evictions", "eager_evictions")
    log = tmp_path / "server.log"
    log.write_text("exl3 RAM miss thread counters " + json.dumps({k: i + 1 for i, k in enumerate(keys)}) + "\n")
    s = cap.counter_summary(str(log))
    assert {k: s[k] for k in keys} == {k: i + 1 for i, k in enumerate(keys)}


def test_the_capture_runs_any_ab_arm_and_leaves_its_scorer_alone_unless_asked():
    cap = _capture()
    off = cap.server_env("/out", False, arm="dspark-both")
    assert off["SGLANG_EXPERT_STREAM_ENABLE_RAM_PREFETCH"] == "0" and "SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER" not in off
    floors = cap.server_env("/out", False, arm="dspark-both-prefetch-gpu-floors")
    assert floors["SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"] == "gpu"
    assert floors["SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"].endswith("ram_prefetch_margin_floors.json")
    from sglang.srt.layers.moe.ram_prefetch import MARGIN_FLOORS

    assert floors["SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"] == MARGIN_FLOORS
    gpu = cap.server_env("/out", False, arm="dspark-both-prefetch-gpu")
    assert gpu["SGLANG_EXPERT_STREAM_RAM_PREFETCH_SCORER"] == "gpu" and gpu["SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"] == ""
    assert _trace(off)["job"]["prefix"] == "/out/events"


def test_the_capture_logs_routes_router_inputs_and_verify_accepts_unless_told_not_to():
    cap = _capture()
    on = cap.server_env("/out", False, arm="dspark-both")
    assert _trace(on)["expert_path"] == "/out/stages.jsonl"
    assert _trace(on)["router_capture_path"] == "/out/router"
    assert _trace(on)["verify_accept_path"] == "/out/verify-accept"
    off = cap.server_env("/out", False, arm="dspark-both", verify_split=False)
    assert not any(k in _trace(off) for k in ("expert_path", "router_capture_path", "verify_accept_path"))


def test_the_default_client_is_the_greedy_session_probe_and_prompts_files_make_it_sample(tmp_path):
    cap = _capture()
    default = cap.client_argv(cap.parse_args([str(tmp_path)]))
    assert "--prompts-file" not in default and "--temperature" not in default and "--top-p" not in default
    assert default[default.index("--top-logprobs") + 1] == "1"
    prompts = tmp_path / "p.json"
    prompts.write_text("[]")
    a = cap.parse_args([str(tmp_path), "--prompts-file", str(prompts), "--temperature", "0.6", "--top-p", "0.95",
                        "--max-tokens", "768"])
    argv = cap.client_argv(a)
    assert argv[argv.index("--temperature") + 1] == "0.6" and argv[argv.index("--top-p") + 1] == "0.95"
    assert argv[argv.index("--max-tokens") + 1] == "768" and argv[argv.index("--top-logprobs") + 1] == "0"
    assert argv[argv.index("--prompts-file") + 1] == str(prompts)
    assert _trace(cap.capture_env(a))["job"]["capacity"] == int(cap.LONG_TRACE_CAPACITY)  # a JSON integer in the group env


def test_sampling_flags_need_a_long_prompt_or_a_prompts_file(tmp_path):
    cap = _capture()
    with pytest.raises(SystemExit):
        cap.parse_args([str(tmp_path), "--temperature", "0.6"])
