"""The long-prompt, time-boxed capture: spec_margin_capture.py's flags, long_prompt.py's streaming client and its cut,
and long_compare.py's windows and two-arm alignment (CPU)."""

import http.server
import importlib.util
import json
import os
import threading
import time

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DSPARK = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DSPARK, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- spec_margin_capture.py's flags ----------------------------------------------------------------------------------


def test_the_capture_keeps_the_probe_client_unless_given_a_long_prompt():
    cap = _load("spec_margin_capture")
    a = cap.parse_args(["/out"])
    assert (a.long_prompt, a.decode_seconds, a.ttft_repeats) == (None, 300.0, 0)
    assert cap.client_argv(a) == [
        cap.arm_env.PYTHON, cap.PROBE, "--port", str(cap.PORT), "--sessions", cap.both_cpu_ab.SESSIONS,
        "--prompts", "8", "--max-tokens", "128", "--top-logprobs", "1", "--out", "/out/probe.json",
    ]


def test_the_capture_runs_the_long_prompt_client_with_its_window_and_repeats(tmp_path):
    cap = _load("spec_margin_capture")
    prompt = tmp_path / "p.md"
    prompt.write_text("hello")
    a = cap.parse_args(["/out", "--long-prompt", str(prompt), "--decode-seconds", "120", "--ttft-repeats", "3"])
    assert cap.client_argv(a) == [
        cap.arm_env.PYTHON, cap.LONG_PROMPT, "--port", str(cap.PORT), "--prompt", str(prompt),
        "--decode-seconds", "120.0", "--ttft-repeats", "3", "--out-dir", "/out",
    ]
    assert int(cap.LONG_TRACE_CAPACITY) == 1048576


def test_the_capture_passes_a_sampled_long_prompts_temperature_and_top_p(tmp_path):
    cap = _load("spec_margin_capture")
    prompt = tmp_path / "p.md"
    prompt.write_text("hello")
    a = cap.parse_args(["/out", "--long-prompt", str(prompt), "--temperature", "0.6", "--top-p", "0.95"])
    assert cap.client_argv(a)[-4:] == ["--temperature", "0.6", "--top-p", "0.95"]
    lp = _load("long_prompt")
    body = lp.request_body("hi", "rid-1", temperature=0.6, top_p=0.95)
    assert body["temperature"] == 0.6 and body["top_p"] == 0.95
    assert "top_p" not in lp.request_body("hi", "rid-1")
    for bad in (["--top-p", "0.95"], ["--temperature", "-1"], ["--temperature", "0.6", "--top-p", "0"]):
        with pytest.raises(SystemExit):
            cap.parse_args(["/out", "--long-prompt", str(prompt), *bad])
    with pytest.raises(SystemExit):
        cap.parse_args(["/out", "--temperature", "0.6"])


def test_a_sampled_run_sends_the_sampling_arguments_and_records_them(tmp_path):
    lp = _load("long_prompt")
    prompt = tmp_path / "p.md"
    prompt.write_text("PROMPT")
    sent = []

    def request(port, text, rid, seconds, **sampling):
        sent.append(sampling)
        return {"rid": rid, "ttft_s": 0.1, "first_token_ns": 5, "cut": True, "finish_reason": None,
                "abort_status": 200, "chunks": []}

    assert lp.run(1, str(prompt), str(tmp_path), 300, 1, request=request, sleep=lambda s: None,
                  temperature=0.6, top_p=0.95) == 0
    assert sent == [{"temperature": 0.6, "top_p": 0.95}] * 2
    record = json.load(open(tmp_path / "long_prompt.json"))
    assert (record["temperature"], record["top_p"]) == (0.6, 0.95)


@pytest.mark.parametrize("argv", [
    ["/out", "--decode-seconds", "60"],
    ["/out", "--ttft-repeats", "2"],
    ["/out", "--long-prompt", "/no/such/prompt.md"],
])
def test_the_capture_refuses_long_prompt_options_it_cannot_honour(argv):
    cap = _load("spec_margin_capture")
    with pytest.raises(SystemExit):
        cap.parse_args(argv)


def test_the_capture_refuses_a_non_positive_window(tmp_path):
    cap = _load("spec_margin_capture")
    prompt = tmp_path / "p.md"
    prompt.write_text("hello")
    with pytest.raises(SystemExit):
        cap.parse_args(["/out", "--long-prompt", str(prompt), "--decode-seconds", "0"])


# ---- long_prompt.py --------------------------------------------------------------------------------------------------


def test_the_request_is_greedy_streaming_with_thinking_on_and_per_chunk_usage():
    lp = _load("long_prompt")
    body = lp.request_body("hi", "rid-1")
    assert body["temperature"] == 0 and body["stream"] is True and body["max_tokens"] >= 8192
    assert body["chat_template_kwargs"] == {"thinking": True}
    assert body["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert body["rid"] == "rid-1"


def test_the_dsv41_encoder_reads_the_thinking_key_the_client_sends():
    source = open(os.path.join(ROOT, "python", "sglang", "srt", "entrypoints", "openai", "serving_chat.py")).read()
    assert '(request.chat_template_kwargs or {}).get(\n            "thinking", envs.SGLANG_DEFAULT_THINKING.get()' in source


def _chunk(tokens=None, content="", reasoning="", finish=None, choices=True):
    e = {"usage": {"completion_tokens": tokens, "prompt_tokens": 100,
                   "prompt_tokens_details": {"cached_tokens": 0}} if tokens is not None else None}
    if choices:
        e["choices"] = [{"delta": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}]
    return e


class _Clock:
    def __init__(self, times):
        self.times = iter(times)

    def __call__(self):
        return next(self.times)


def test_the_window_starts_at_the_first_text_and_cuts_once_its_seconds_have_passed():
    lp = _load("long_prompt")
    s = 1_000_000_000
    events = [_chunk(content=""), _chunk(1, "a"), _chunk(3, "bc"), _chunk(5, "de"), _chunk(7, "fg")]
    # role chunk at 0 s, first text at 1 s; 10 s window: the chunk at 11 s cuts, the one after is never read
    clock = _Clock([0, 1 * s, 6 * s, 11 * s, 12 * s, 13 * s])
    w = lp.read_window(iter(events), 10, clock=clock, wall=lambda: 7)
    assert w["first_token_ns"] == 1 * s and w["cut"] is True
    assert [c["completion_tokens"] for c in w["chunks"]] == [None, 1, 3, 5]
    assert w["end_ns"] == 12 * s


def test_a_stream_that_ends_inside_the_window_is_not_cut():
    lp = _load("long_prompt")
    events = [_chunk(1, "a"), _chunk(2, "b", finish="stop"), _chunk(2, choices=False)]
    w = lp.read_window(iter(events), 10, clock=_Clock([0, 1, 2, 3]), wall=lambda: 0)
    assert w["cut"] is False and w["finish_reason"] == "stop" and len(w["chunks"]) == 3


def test_sse_parsing_stops_at_done_and_skips_keepalives():
    lp = _load("long_prompt")
    lines = [b": keepalive\n", b"data: {\"x\": 1}\n", b"\n", "data: {\"x\": 2}\n", b"data: [DONE]\n", b"data: {\"x\": 3}\n"]
    assert list(lp.sse_events(lines)) == [{"x": 1}, {"x": 2}]


class _StreamingServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.aborts, self.disconnected, self.bodies = [], threading.Event(), []


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/abort_request":
            self.server.aborts.append(body["rid"])
            self.send_response(200)
            self.end_headers()
            return
        self.server.bodies.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b"data: " + json.dumps(_chunk(content="")).encode() + b"\n\n")
            for i in range(1, 10_000):
                time.sleep(0.01)
                self.wfile.write(b"data: " + json.dumps(_chunk(2 * i, f"t{i} ")).encode() + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.server.disconnected.set()


def test_the_client_cancels_a_stream_that_would_run_on_and_posts_its_abort():
    lp = _load("long_prompt")
    server = _StreamingServer()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        t = time.monotonic()
        r = lp.run_request(port, "the prompt", "rid-x", 0.3)
        assert time.monotonic() - t < 5
        assert r["cut"] is True and r["abort_status"] == 200 and server.aborts == ["rid-x"]
        assert server.bodies[0]["messages"] == [{"role": "user", "content": "the prompt"}]
        assert r["ttft_s"] is not None and r["ttft_s"] >= 0
        elapsed = (r["chunks"][-1]["ns"] - r["first_token_ns"]) / 1e9
        assert 0.3 <= elapsed < 1.0
        assert server.disconnected.wait(5)
    finally:
        server.shutdown()
        server.server_close()


def test_the_probe_tokens_hold_one_entry_per_completion_token():
    lp = _load("long_prompt")
    chunks = [{"completion_tokens": None, "content": "", "reasoning": ""},
              {"completion_tokens": 1, "content": "a", "reasoning": ""},
              {"completion_tokens": 4, "content": "bcd", "reasoning": "r"},
              {"completion_tokens": 4, "content": "", "reasoning": ""}]
    tokens = lp.probe_tokens(chunks)
    assert len(tokens) == 4 and "".join(t["token"] for t in tokens) == "arbcd"


def test_the_run_sends_tagged_ttft_samples_then_the_prompt_and_writes_both_files(tmp_path):
    lp = _load("long_prompt")
    prompt = tmp_path / "p.md"
    prompt.write_text("PROMPT")
    sent = []

    def request(port, text, rid, seconds):
        sent.append((text, rid, seconds))
        chunks = [{"ns": 5, "epoch_ns": 5, "completion_tokens": 2, "reasoning_tokens": None, "prompt_tokens": 1,
                   "cached_tokens": 0, "content": "xy", "reasoning": ""}]
        return {"rid": rid, "ttft_s": 0.1, "first_token_ns": 5, "cut": True, "finish_reason": None,
                "abort_status": 200, "chunks": chunks}

    assert lp.run(1, str(prompt), str(tmp_path), 300, 2, request=request, sleep=lambda s: None) == 0
    assert [s[2] for s in sent] == [lp.TTFT_DECODE_SECONDS, lp.TTFT_DECODE_SECONDS, 300]
    assert sent[-1][0] == "PROMPT" and len({s[0] for s in sent}) == 3
    record = json.load(open(tmp_path / "long_prompt.json"))
    assert [r["kind"] for r in record["requests"]] == ["ttft", "ttft", "long"]
    probe = json.load(open(tmp_path / "probe.json"))
    assert len(probe) == 3 and all(len(p["tokens"]) == 2 for p in probe)


# ---- long_compare.py -------------------------------------------------------------------------------------------------


def test_the_windows_are_30_30_then_60_s_to_the_cut():
    lc = _load("long_compare")
    assert lc.window_edges(300) == [0, 30, 60, 120, 180, 240, 300]
    assert lc.window_edges(90) == [0, 30, 60, 90]
    assert lc.window_edges(25) == [0, 25]
    assert lc.window_edges(130) == [0, 30, 60, 120, 130]
    edges = lc.window_edges(300)
    assert [lc.bucket(t, edges) for t in (0, 29.9, 30, 59, 60, 119, 120, 299.9, 300)] == [0, 0, 1, 1, 2, 2, 3, 5, 5]
    assert lc.bucket(-0.1, edges) is None and lc.bucket(300.1, edges) is None


def _chunks(texts_and_counts, step_ns, t0=0):
    return [{"ns": t0 + i * step_ns, "epoch_ns": t0 + i * step_ns, "completion_tokens": n, "content": text,
             "reasoning": "", "cached_tokens": 0} for i, (text, n) in enumerate(texts_and_counts)]


def test_the_client_windows_count_the_tokens_streamed_in_each():
    lc = _load("long_compare")
    s = 1_000_000_000
    chunks = _chunks([(f"t{i}", 2 * i + 1) for i in range(91)], s)
    w = lc.client_windows(lc.counted(chunks), 0, [0, 30, 60, 90])
    assert [x["tokens"] for x in w] == [60, 60, 60]
    assert w[0]["ms_per_token"] == pytest.approx(500.0)


def test_the_common_prefix_ends_at_the_first_chunk_that_differs():
    lc = _load("long_compare")
    a = _chunks([("ab", 2), ("cd", 4), ("ef", 6), ("gh", 8)], 10)
    b = _chunks([("abc", 3), ("de", 5), ("fX", 7), ("yz", 9)], 10)
    d = lc.divergence(a, b)
    # "abcdef" vs "abcdefX": common text "abcdef"; arm b's third chunk holds the f and the X, so b's tokens stop at 5
    assert d["common_chars"] == 6 and d["diverged"] is True and d["divergence_token"] == 5
    same = lc.divergence(a, a)
    assert same["diverged"] is False and same["divergence_token"] is None and same["common_tokens"] == 8
    shorter = lc.divergence(a, a[:2])
    assert shorter["diverged"] is False and shorter["common_tokens"] == 4


def test_ms_per_token_is_read_on_the_same_tokens_for_both_arms():
    lc = _load("long_compare")
    fast = lc.counted(_chunks([("x", n) for n in range(1, 101)], 10_000_000))
    slow = lc.counted(_chunks([("x", n) for n in range(1, 101)], 20_000_000))
    assert lc.ms_per_token_over(fast, 10, 60) == pytest.approx(10.0)
    assert lc.ms_per_token_over(slow, 10, 60) == pytest.approx(20.0)
    assert lc.ms_per_token_over(fast, 10, 1000) is None


def test_a_window_is_flagged_against_the_runs_median_window():
    lc = _load("long_compare")
    windows = [{"verifies": 10, "ms_per_stream_layer": 1.0, "cpu_ms_per_lane": {0: 1.0, 1: 1.0}} for _ in range(4)]
    windows[2]["ms_per_stream_layer"] = 1.5
    windows[3]["cpu_ms_per_lane"] = {0: 1.0, 1: 1.3}
    lc.flag_windows(windows, [0, 1], 1.3, 1.2)
    assert [w["stall"] for w in windows] == [False, False, True, False]
    assert [w["cpu_slow"] for w in windows] == [False, False, False, True]


class _FakeRun:
    """lane_tier_replay.Run's verifies and records, one verify per second of decode."""

    def __init__(self, ltr, t0, seconds, cpu_forced, gpu_nvme, stream_ms=2.0):
        s = 1_000_000_000
        self.verifies = [(t0 + i * s, t0 + i * s + s // 2) for i in range(seconds)]
        self.records = []
        for v in range(seconds):
            r = ltr.Record(v)
            r.verify, r.row, r.stream, r.wait = v, 0, stream_ms, 1.0
            r.evict_in = list(range(gpu_nvme))
            r.jobs = [{"team": 0, "part": 1, "lanes": 1, "start": 0, "end": 1_000_000}] * cpu_forced
            self.records.append(r)


def _write_run(d, text_counts, step_ns, t0, sha="abc"):
    os.makedirs(d, exist_ok=True)
    chunks = _chunks(text_counts, step_ns, t0)
    long = {"kind": "long", "rid": "long", "ttft_s": 2.0, "sent_epoch_ns": t0 - 2_000_000_000, "first_token_ns": t0,
            "cut": True, "decode_seconds": 90, "finish_reason": None, "chunks": chunks}
    with open(os.path.join(d, "long_prompt.json"), "w") as f:
        json.dump({"prompt_sha256": sha, "requests": [long]}, f)


def test_one_run_reads_as_a_diagnostic_and_two_are_paired_on_their_common_tokens(tmp_path):
    lc = _load("long_compare")
    s = 1_000_000_000
    t0 = 1000 * s
    # 95 s of decode, a chunk per second of 3 tokens; arm b streams the same text 1.5x slower and diverges at chunk 40
    a_dir, b_dir = str(tmp_path / "a"), str(tmp_path / "b")
    _write_run(a_dir, [(f"w{i} ", 3 * i + 1) for i in range(96)], s, t0)
    _write_run(b_dir, [(f"w{i} " if i < 40 else f"v{i} ", 3 * i + 1) for i in range(96)], 3 * s // 2, t0)
    ra = lc.analyse(a_dir, 1.3, 1.2, run=_FakeRun(lc.ltr, t0, 95, cpu_forced=2, gpu_nvme=5))
    rb = lc.analyse(b_dir, 1.3, 1.2, run=_FakeRun(lc.ltr, t0, 95, cpu_forced=1, gpu_nvme=3))
    assert [(w["t0_s"], w["t1_s"]) for w in ra["windows"]] == [(0, 30), (30, 60), (60, 90)]
    assert [w["verifies"] for w in ra["windows"]] == [30, 30, 31]
    assert ra["windows"][0]["cpu_forced_per_verify"] == 2 and ra["windows"][0]["gpu_nvme_per_verify"] == 5
    assert ra["windows"][0]["ms_per_token"] == pytest.approx(1000 / 3)
    assert ra["long"]["decode_s"] == 90 and ra["long"]["ttft_s"] == 2.0
    p = lc.pair(a_dir, b_dir, ra, rb)
    assert p["divergence"]["divergence_token"] == 3 * 39 + 1
    assert p["common_prefix"]["delta"] == pytest.approx(1000 / 6)
    assert p["by_window_server"][0]["cpu_forced_per_verify"] == -1 and p["by_window_server"][0]["gpu_nvme_per_verify"] == -2
    first = p["by_window_same_tokens"][0]
    assert first["ms_per_token"][0] == pytest.approx(1000 / 3) and first["ms_per_token"][1] == pytest.approx(500)
    # past the divergence there are no common tokens to compare
    assert p["by_window_same_tokens"][-1]["tokens"] is None


def test_a_prefill_is_flagged_against_the_median_prefill_of_every_run_and_not_on_too_few():
    lc = _load("long_compare")

    def run(*awaits):
        return {"prefill": [{"nvme3n1_await_ms": aw, "prefill_stall": None} for aw in awaits]}

    two = [run(5.0), run(50.0)]
    assert lc.flag_prefills(two) is None and all(p["prefill_stall"] is None for r in two for p in r["prefill"])
    pooled = [run(5.0, 6.0), run(4.0, 20.0, None)]
    assert lc.flag_prefills(pooled) == pytest.approx(5.5)
    assert [p["prefill_stall"] for r in pooled for p in r["prefill"]] == [False, False, False, True, False]


def test_a_window_past_a_truncated_job_trace_is_marked(tmp_path):
    lc = _load("long_compare")
    s = 1_000_000_000
    t0 = 1000 * s
    d = str(tmp_path / "a")
    _write_run(d, [(f"w{i} ", 3 * i + 1) for i in range(96)], s, t0)
    with open(os.path.join(d, "events.1.exl3-copy-eng.2.jsonl"), "w") as f:
        f.write(json.dumps({"schema": 1}) + "\n")
        f.write(json.dumps({"event": "copy_submit", "ns": t0 + 45 * s}) + "\n")
        f.write(json.dumps({"dropped": 12}) + "\n")
    r = lc.analyse(d, 1.3, 1.2, run=_FakeRun(lc.ltr, t0, 95, cpu_forced=0, gpu_nvme=0))
    assert r["trace_complete_through_s"] == pytest.approx(45)
    assert [w["truncated"] for w in r["windows"]] == [False, True, True]
    assert "TRUNC" in lc.window_flags(r["windows"][1])
