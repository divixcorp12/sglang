"""The sampled-finish A/B report's log parsing, aggregation and advisory ship rule, and the driver's --dry-run (CPU)."""

import importlib.util
import json
import os
import subprocess
import sys

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DSPARK = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark")
DRIVER = os.path.join(DSPARK, "run_sampled_finish_ab.sh")

LOG = """\
[ERROR] `loss` is part of Qwen2_5_VLCausalLMOutputWithPast.__init__'s signature, but not documented.
[t] max_total_num_tokens=381184, chunked_prefill_size=4096, max_running_requests=1
[t] dspark confidence cut: T(w) calibration starts at eligible verify 97, after 96 held back (serving): 96 verifies
[t] dspark confidence cut: T(w) ms = [226.6, 243.4, 260.2] (a=226.6248, b=16.7763/row, r2=0.9148; per-width median ms [1.0], counts [16, 16, 16, 16, 16, 16])
[t] Decode batch, accept len: 3.20, conf cut: verifies 30, w* 4.00, E(w*) 3.00, accept len 3.00, cut rows 30 (total 30), widths [1, 2, 3, 4, 10, 10], sampled 30, cuda graph: True
[t] Decode batch, accept len: 3.20, conf cut: verifies 10, w* 6.00, E(w*) 5.00, accept len 4.00, cut rows 20 (total 50), widths [0, 0, 0, 0, 0, 10], sampled 10, cuda graph: True
[t] Triton kernel 'a' device-loaded after serving started (free device mem: 2 GiB). Pre-load it.
[t] Triton kernel 'b' device-loaded after serving started (free device mem: 2 GiB). Pre-load it.
[t] Expert residency clock dropped an uncommitted verify token count; speculative commits are not reaching the hot cache
"""


def _report():
    spec = importlib.util.spec_from_file_location("sampled_finish_report", os.path.join(DSPARK, "sampled_finish_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_server_log_parse_pulls_kv_cut_summary_calibration_and_the_warning_counts_and_ignores_noise():
    """Mutant: the parser counts the Qwen docstring [ERROR] lines, averages w* per line instead of per verify, drops the
    width counts or the calibration's sample count, or miscounts the late loads -- red."""
    log = _report().parse_server_log(LOG)
    assert log["kv"] == 381184 and log["errors"] == []
    assert log["late_loads"] == 2 and log["dropped_uncommitted"] == 1
    cut = log["cut"]
    assert cut["summaries"] == 2 and cut["verifies"] == 40 and cut["cut_rows"] == 50 and cut["sampled_verifies"] == 40
    assert abs(cut["w_star"] - (4.0 * 30 + 6.0 * 10) / 40) < 1e-9
    assert abs(cut["accept_len"] - (3.0 * 30 + 4.0 * 10) / 40) < 1e-9
    assert cut["width_counts"] == [1, 2, 3, 4, 10, 20]
    cal = log["calibration"]
    assert (cal["a"], cal["b"], cal["r2"]) == (226.6248, 16.7763, 0.9148)
    assert cal["samples"] == 96 and cal["counts"] == [16] * 6


def test_the_parse_reads_the_older_calibration_and_cut_lines_without_counts_or_widths_and_flags_real_errors():
    old = (
        "dspark confidence cut: T(w) ms = [226.6248, 243.4011] (a=226.6248, b=16.7763/row, r2=0.0148)\n"
        "Decode batch, conf cut: verifies 33, w* 4.94, E(w*) 3.57, accept len 3.55, cut rows 35 (total 35), cuda graph\n"
        "Traceback (most recent call last):\n"
        "dspark confidence cut disabled, every verify covers all rows: x\n"
    )
    log = _report().parse_server_log(old)
    assert log["calibration"]["samples"] is None and log["calibration"]["counts"] is None
    assert log["cut"]["width_counts"] is None and log["cut"]["cut_rows"] == 35
    assert len(log["errors"]) == 2 and log["kv"] is None
    assert _report().parse_server_log("nothing here")["cut"] is None


def _window(t0, t1, tokens, verifies, truncated=False):
    return {"t0_s": t0, "t1_s": t1, "tokens": tokens, "verifies": verifies, "truncated": truncated}


def _res(ms_per_token_scale):
    # 60 s windows, 10 tokens/s scaled; verifies 4 tokens each in the steady state.
    windows = [_window(0, 30, 100, 40), _window(30, 60, 100, 40)] + [
        _window(60 * k, 60 * (k + 1), int(600 / ms_per_token_scale), int(600 / ms_per_token_scale) // 4) for k in range(1, 6)
    ] + [_window(360, 420, 100, 30)]
    return {"windows": windows, "long": {"ttft_s": 10.0, "finish_reason": None, "tokens": 5000}}


def test_the_aggregate_takes_the_steady_window_from_60_to_360_s_and_the_whole_decode_from_all_windows():
    agg = _report().aggregate(_res(1.0))
    steady, whole = agg["steady"], agg["whole"]
    assert steady["seconds"] == 300 and steady["tokens"] == 5 * 600 and steady["verifies"] == 5 * 150
    assert abs(steady["ms_per_token"] - 100.0) < 1e-9 and abs(steady["tokens_per_verify"] - 4.0) < 1e-9
    assert abs(steady["ms_per_verify"] - 400.0) < 1e-9
    assert whole["seconds"] == 420 and whole["tokens"] == 200 + 3000 + 100 and agg["steady_complete"]
    short = _report().aggregate({"windows": [_window(0, 30, 10, 3), _window(30, 60, 10, 3), _window(60, 120, 10, 3)],
                                 "long": {"ttft_s": 1.0, "finish_reason": None, "tokens": 30}})
    assert not short["steady_complete"]


def test_a_dropped_copy_engine_trace_does_not_truncate_ms_per_verify_but_a_dropped_verify_file_does():
    res = _res(1.0)
    for w in res["windows"]:
        w["truncated"] = True
    r = _report()
    assert not r.aggregate({**res, "trace_dropped_files": ["events.1.exl3-copy-eng.2.jsonl"]})["steady"]["truncated"]
    assert r.aggregate({**res, "trace_dropped_files": ["events.1.verify.jsonl"]})["steady"]["truncated"]
    assert r.aggregate({**res, "trace_dropped_files": ["events.1.device-stamps.jsonl"]})["steady"]["truncated"]


def _arm(tokens_per_s_scale, tpv=4.0, errors=(), kv=381184):
    # A faster arm streams more tokens in the same windows, with tokens per verify held at ``tpv``.
    windows = [_window(60 * k, 60 * (k + 1), int(600 * tokens_per_s_scale), int(600 * tokens_per_s_scale / tpv))
               for k in range(1, 6)]
    res = {"windows": windows, "long": {"ttft_s": 10.0, "finish_reason": None, "tokens": 1}}
    return {"agg": _report().aggregate(res), "log": {"errors": list(errors), "kv": kv}, "kv_ok": kv == 381184}


def test_the_ship_rule_needs_5_percent_on_both_metrics_no_errors_and_no_lower_accept_and_block_needs_5_over_cut():
    """The advisory suggestion: cut when it clears the rule against -prod, block only when it also clears 5% over
    cut-only, otherwise the owner. Mutant: 4% passes, one metric is enough, errors or a lower accept are ignored, or
    block ships without beating cut-only -- red."""
    r = _report()
    # ms/token and ms per verify both fall as tokens per second rise at fixed tokens per verify... ms per verify is
    # the window time over verifies, so it falls with the token rate; tokens per verify stays put.
    prod = _arm(1.0)
    assert r.verdict({"prod": prod, "cut": _arm(1.06), "block": _arm(1.07)})["suggestion"] == "cut"
    assert r.verdict({"prod": prod, "cut": _arm(1.06), "block": _arm(1.2)})["suggestion"] == "block"
    close = r.verdict({"prod": prod, "cut": _arm(1.04), "block": _arm(1.04)})
    assert not close["arms"]["cut"]["clears_rule_vs_prod"] and close["suggestion"].startswith("owner")
    assert r.verdict({"prod": prod, "cut": _arm(0.9), "block": _arm(0.9)})["suggestion"] == "none"
    errs = r.verdict({"prod": prod, "cut": _arm(1.06, errors=["Traceback"]), "block": _arm(1.2)})
    assert not errs["arms"]["cut"]["clears_rule_vs_prod"] and errs["suggestion"] == "block"
    low_accept = r.verdict({"prod": prod, "cut": _arm(1.2, tpv=3.5), "block": _arm(1.2, tpv=3.5)})
    assert "tokens per verify lower than -prod" in low_accept["arms"]["cut"]["reasons"]
    wrong_kv = r.verdict({"prod": prod, "cut": _arm(1.2, kv=1), "block": _arm(1.2)})
    assert any("KV" in x for x in wrong_kv["arms"]["cut"]["reasons"])
    assert r.verdict({"prod": prod, "cut": _arm(1.06), "block": _arm(1.07)})["advisory"] is True


def test_render_prints_every_row_the_spec_asks_for_and_labels_the_verdict_advisory():
    r = _report()
    arms = {n: {**_arm(s), "log": {**_arm(s)["log"], **r.parse_server_log(LOG)}} for n, s in (("prod", 1.0), ("cut", 1.1), ("block", 1.2))}
    text = r.render(arms, r.verdict(arms))
    for needle in ("ms/token, 60-360 s", "ms per verify, 60-360 s", "tokens per verify", "long TTFT", "KV (max_total_num_tokens)",
                   "late device-loads", "conf cut w*", "conf cut rows", "T(w) -prod: a=226.6248", "ADVISORY", "the owner decides"):
        assert needle in text, needle


def test_the_driver_dry_run_prints_the_three_arms_envs_and_the_sampled_argv_and_launches_nothing(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(["bash", DRIVER, "--dry-run", "--worktree", ROOT, "--python", sys.executable, "--out", str(out)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr + r.stdout
    lines = r.stdout.splitlines()
    assert "DRY RUN: nothing launched" in lines and not out.exists()
    sections = [x for x in lines if x.startswith("##### ")]
    assert [x.split()[1] for x in sections] == ["prod", "cut", "block"]
    assert sections[0].endswith("-prod") and sections[1].endswith("-prod-confcut-sampled")
    assert sections[2].endswith("-prod-confcut-sampled-block")
    resolved = [json.loads(x.split(": ", 1)[1]) for x in lines if x.startswith("resolved-env.json: ")]
    assert len(resolved) == 3
    for res, name in zip(resolved, ("prod", "cut", "block")):
        env, argv = res[name], res["argv"]
        assert env["SGLANG_DSPARK_USE_GREEDY_DRAFT"] == "0"
        on = name != "prod"
        assert env["SGLANG_DSPARK_CONFIDENCE_CUT"] == ("1" if on else "0")
        assert env["SGLANG_DSPARK_CONFIDENCE_CUT_SAMPLED"] == ("1" if on else "0")
        assert env["SGLANG_DSPARK_CONFIDENCE_CUT_DRAFT_COST"] == ("1" if on else "0")
        assert env["SGLANG_DSPARK_CONFIDENCE_CUT_REFIT_INTERVAL"] == "0"
        assert ("--speculative-dspark-confidence-sts-path" in argv) == on
        assert ("--speculative-use-block-verification" in argv) == (name == "block")
    caps = [x for x in lines if x.strip().startswith("capture: ")]
    assert len(caps) == 3
    for cap in caps:
        for word in ("--decode-seconds 420", "--ttft-repeats 0", "--temperature 0.6", "--top-p 0.95", "SAMPLE_PROMPT_DIVIX01.md"):
            assert word in cap, (word, cap)
