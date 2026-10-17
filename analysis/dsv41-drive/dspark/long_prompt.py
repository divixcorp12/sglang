"""A long-prompt, time-boxed served request, for spec_margin_capture.py's --long-prompt mode: one streaming
/v1/chat/completions request with thinking on, greedy unless --temperature (and --top-p) say otherwise, cancelled
DECODE_SECONDS after its first streamed token, so a
capture measures decode past the ~900-token break-even of a TTFT cost (DSV_OPTIMIZATION.md, share 28 vs 64) on the
same collectors as the 8 x 128-token probe.

With --ttft-repeats K, K requests of 20 s decode come first, for TTFT samples; each is the prompt behind a numbered
tag, so none shares a radix prefix with another or with the long request (which is the prompt as given, cold too).
Cancelling closes the stream and posts the request's rid to /abort_request; the server keeps running.

Per streamed chunk it records CLOCK_MONOTONIC and epoch arrival (the job trace's clocks), the cumulative completion
and reasoning tokens (continuous usage stats) and the delta text, so long_compare.py can window the decode and align
two arms by token position. Writes OUT_DIR/long_prompt.json, and OUT_DIR/probe.json in logprob_probe.py's shape (one
entry per request, its "tokens" one per completion token, a chunk's text on its first) so serve_compare.py and
lane_tier_replay.py read the run as K+1 serving prompts.

Usage: long_prompt.py --port PORT --prompt PATH --out-dir DIR [--decode-seconds 300] [--ttft-repeats 0]
                      [--temperature 0.6 [--top-p 0.95]]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import time
from typing import Callable, Iterable, Iterator

DECODE_SECONDS = 300.0
TTFT_DECODE_SECONDS = 20.0
MAX_TOKENS = 8192
# Past serve_compare's 1 s verify gap that splits sessions, and the server's 2 s disconnect abort.
GAP_SECONDS = 3.0
REQUEST_TIMEOUT_S = 900


def sampling_params(temperature: float = 0.0, top_p: float | None = None) -> dict:
    """The body's sampling fields: greedy by default, else this temperature (and top_p when given)."""
    return {"temperature": temperature, **({} if top_p is None else {"top_p": top_p})}


def request_body(prompt: str, rid: str, max_tokens: int = MAX_TOKENS, sampling: dict | None = None) -> dict:
    return {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        **(sampling or sampling_params()),
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        # The dsv41 encoder takes its mode from "thinking" (serving_chat._apply_jinja_template); logprob_probe's
        # "enable_thinking" is not read for it.
        "chat_template_kwargs": {"thinking": True},
        "rid": rid,
    }


def sse_events(lines: Iterable[bytes | str]) -> Iterator[dict]:
    for raw in lines:
        line = (raw.decode() if isinstance(raw, bytes) else raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def read_window(events: Iterable[dict], decode_seconds: float, clock: Callable[[], int] = time.monotonic_ns,
                wall: Callable[[], int] = time.time_ns) -> dict:
    """Reads chunks until ``decode_seconds`` after the first one carrying text (``cut``) or the stream's end. The cut
    is checked as each chunk lands, so it overshoots by at most one chunk interval."""
    chunks, first, finish, cut = [], None, None, False
    for event in events:
        t = clock()
        choices = event.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        usage = event.get("usage") or {}
        content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
        chunks.append({
            "ns": t,
            "epoch_ns": wall(),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "content": content,
            "reasoning": reasoning,
        })
        finish = choices[0].get("finish_reason") or finish
        if first is None and (content or reasoning):
            first = t
        if first is not None and t - first >= decode_seconds * 1e9:
            cut = True
            break
    return {"first_token_ns": first, "cut": cut, "end_ns": clock(), "finish_reason": finish, "chunks": chunks}


def abort(port: int, rid: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        conn.request("POST", "/abort_request", body=json.dumps({"rid": rid}),
                     headers={"Content-Type": "application/json"})
        return conn.getresponse().status
    finally:
        conn.close()


def run_request(port: int, prompt: str, rid: str, decode_seconds: float, max_tokens: int = MAX_TOKENS,
                connect: Callable[..., http.client.HTTPConnection] = http.client.HTTPConnection,
                abort_fn: Callable[[int, str], int] = abort, sampling: dict | None = None) -> dict:
    conn = connect("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    sent_ns, sent_epoch_ns = time.monotonic_ns(), time.time_ns()
    try:
        conn.request("POST", "/v1/chat/completions",
                     body=json.dumps(request_body(prompt, rid, max_tokens, sampling)),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{rid}: HTTP {response.status}: {response.read()[:500]!r}")
        window = read_window(sse_events(response), decode_seconds)
    finally:
        conn.close()
    abort_status = abort_fn(port, rid) if window["cut"] else None
    first = window["first_token_ns"]
    return {
        "rid": rid,
        "decode_seconds": decode_seconds,
        "sent_ns": sent_ns,
        "sent_epoch_ns": sent_epoch_ns,
        "ttft_s": (first - sent_ns) / 1e9 if first is not None else None,
        "abort_status": abort_status,
        **window,
    }


def probe_tokens(chunks: list[dict]) -> list[dict]:
    """logprob_probe.py's per-token list rebuilt from chunks: one entry per completion token, the chunk's text on its
    first. A chunk without a count (the role chunk) adds none."""
    tokens, seen = [], 0
    for c in chunks:
        n = c["completion_tokens"]
        if n is None or n <= seen:
            continue
        tokens.append({"token": c["reasoning"] + c["content"]})
        tokens.extend({"token": ""} for _ in range(n - seen - 1))
        seen = n
    return tokens


def ttft_prompt(prompt: str, i: int) -> str:
    return f"[TTFT sample {i + 1}]\n\n{prompt}"


def run(port: int, prompt_path: str, out_dir: str, decode_seconds: float = DECODE_SECONDS, ttft_repeats: int = 0,
        request: Callable[..., dict] = run_request, sleep: Callable[[float], None] = time.sleep,
        sampling: dict | None = None) -> int:
    with open(prompt_path, "rb") as f:
        raw = f.read()
    prompt = raw.decode()
    sampling = sampling or sampling_params()
    tag = f"long-prompt-{os.getpid()}"
    requests = []
    for i in range(ttft_repeats):
        requests.append({"kind": "ttft", **request(port, ttft_prompt(prompt, i), f"{tag}-ttft{i}", TTFT_DECODE_SECONDS,
                                                   sampling=sampling)})
        sleep(GAP_SECONDS)
    requests.append({"kind": "long", **request(port, prompt, f"{tag}-long", decode_seconds, sampling=sampling)})
    sleep(GAP_SECONDS)
    record = {
        "prompt": os.path.abspath(prompt_path),
        "prompt_sha256": hashlib.sha256(raw).hexdigest(),
        "sampling": sampling,
        "decode_seconds": decode_seconds,
        "ttft_repeats": ttft_repeats,
        "ttft_decode_seconds": TTFT_DECODE_SECONDS,
        "max_tokens": MAX_TOKENS,
        "clock": "CLOCK_MONOTONIC",
        "requests": requests,
    }
    with open(os.path.join(out_dir, "long_prompt.json"), "w") as f:
        json.dump(record, f)
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump([{"session_id": r["rid"], "kind": r["kind"], "tokens": probe_tokens(r["chunks"])} for r in requests], f)
    for r in requests:
        print(f"{r['rid']}: ttft {r['ttft_s']} s, {r['chunks'][-1]['completion_tokens'] if r['chunks'] else 0} tokens,"
              f" cut {r['cut']}, finish {r['finish_reason']}, abort HTTP {r['abort_status']}", flush=True)
    return 0 if all(r["first_token_ns"] is not None for r in requests) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--decode-seconds", type=float, default=DECODE_SECONDS)
    p.add_argument("--ttft-repeats", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0, help="sample at this temperature (0, the default, is greedy)")
    p.add_argument("--top-p", type=float, help="with --temperature: nucleus sampling at this top_p")
    a = p.parse_args(argv)
    if a.decode_seconds <= 0 or a.ttft_repeats < 0:
        p.error("--decode-seconds must be positive and --ttft-repeats non-negative")
    if a.top_p is not None and a.temperature <= 0:
        p.error("--top-p needs a positive --temperature")
    return run(a.port, a.prompt, a.out_dir, a.decode_seconds, a.ttft_repeats,
               sampling=sampling_params(a.temperature, a.top_p))


if __name__ == "__main__":
    sys.exit(main())
