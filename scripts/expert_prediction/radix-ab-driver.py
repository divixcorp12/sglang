#!/usr/bin/env python3
"""Radix cache A/B driver for the NVFP4 shadow server.

Stdlib only. Talks to http://127.0.0.1:<port>/v1/chat/completions.

Phase A: a logprob probe (thinking off, non-streaming) that issues P1 (cold),
P1 again (cached when radix is on), then P2 (shares the CONTEXT prefix).

Phase B: a 4-turn streaming chat session (thinking on, production default)
that measures per-turn TTFT, total time, usage, and client decode rate.

Writes one JSON file with everything recorded, for later cross-run analysis.
"""
import argparse
import json
import statistics
import time
import urllib.request

CONTEXT_FUNCTION_COUNT = 24


def build_context() -> str:
    """Build a deterministic ~3000-token synthetic backtesting module."""
    topics = ["sharpe", "drawdown", "position sizing"]
    lines = [
        "import math",
        "import statistics",
        "",
        "",
        "class BacktestState:",
        '    """Holds running state for a single backtest pass."""',
        "",
        "    def __init__(self, capital):",
        "        self.capital = capital",
        "        self.history = []",
        "",
    ]
    for i in range(CONTEXT_FUNCTION_COUNT):
        topic = topics[i % len(topics)]
        name = f"func_{i:03d}_{topic.replace(' ', '_')}"
        if topic == "sharpe":
            doc = (
                f"Compute a rolling Sharpe ratio contribution for window {i}. "
                "Sharpe ratio here is annualized excess return over return "
                "volatility; a higher value indicates better risk-adjusted "
                "performance for this slice of the backtest."
            )
            body = [
                f"    mean_return = sum(returns[-{i + 5}:]) / max(1, len(returns[-{i + 5}:]))",
                f"    vol = statistics.pstdev(returns[-{i + 5}:]) if len(returns) > 1 else 1.0",
                "    return mean_return / vol if vol else 0.0",
            ]
        elif topic == "drawdown":
            doc = (
                f"Compute the maximum drawdown for equity curve segment {i}. "
                "Drawdown is the peak-to-trough decline in portfolio equity, "
                "expressed as a fraction of the running peak value."
            )
            body = [
                "    peak = equity[0]",
                "    max_dd = 0.0",
                "    for value in equity:",
                "        peak = max(peak, value)",
                "        dd = (peak - value) / peak if peak else 0.0",
                "        max_dd = max(max_dd, dd)",
                "    return max_dd",
            ]
        else:
            doc = (
                f"Compute a position size for signal strength bucket {i}. "
                "Position sizing scales exposure by signal confidence and by "
                "the inverse of recent realized volatility, capped at max_risk."
            )
            body = [
                f"    scale = min(1.0, signal_strength * {(i % 7) + 1} / 10.0)",
                "    size = scale * capital * max_risk / (1.0 + recent_vol)",
                "    return min(size, capital * max_risk)",
            ]
        lines.append(f"def {name}(returns=None, equity=None, capital=100000.0,")
        lines.append("               signal_strength=0.5, recent_vol=0.1, max_risk=0.02):")
        lines.append(f'    """{doc}"""')
        lines.extend(body)
        lines.append("")
        lines.append("")
    return "\n".join(lines)


CONTEXT = build_context()


def post(port, path, payload, stream=False):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(req, timeout=600)


def run_nonstreaming(port, messages, max_tokens):
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = post(port, "/v1/chat/completions", payload)
    body = json.loads(resp.read().decode("utf-8"))
    wall_time = time.time() - t0

    choice = body["choices"][0]
    content_logprobs = (choice.get("logprobs") or {}).get("content") or []
    tokens = []
    for entry in content_logprobs:
        tokens.append(
            {
                "token": entry.get("token"),
                "logprob": entry.get("logprob"),
                "top_logprobs": [
                    {"token": t.get("token"), "logprob": t.get("logprob")}
                    for t in (entry.get("top_logprobs") or [])
                ],
            }
        )
    return {
        "wall_time_s": wall_time,
        "content": choice.get("message", {}).get("content"),
        "tokens": tokens,
        "usage": body.get("usage"),
    }


def run_streaming_turn(port, messages, max_tokens):
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    resp = post(port, "/v1/chat/completions", payload, stream=True)

    t_first_token = None
    t_last_token = None
    reasoning_parts = []
    content_parts = []
    usage = None

    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        now = time.time()
        if reasoning:
            reasoning_parts.append(reasoning)
            if t_first_token is None:
                t_first_token = now
            t_last_token = now
        if content:
            content_parts.append(content)
            if t_first_token is None:
                t_first_token = now
            t_last_token = now

    t_end = time.time()
    ttft = (t_first_token - t0) if t_first_token is not None else None
    total_time = t_end - t0
    completion_tokens = (usage or {}).get("completion_tokens")
    decode_rate = None
    if (
        completion_tokens
        and completion_tokens > 1
        and t_first_token is not None
        and t_last_token is not None
        and t_last_token > t_first_token
    ):
        decode_rate = (completion_tokens - 1) / (t_last_token - t_first_token)

    return {
        "ttft_s": ttft,
        "total_time_s": total_time,
        "usage": usage,
        "client_decode_tok_s": decode_rate,
        "reasoning_content": "".join(reasoning_parts),
        "content": "".join(content_parts),
    }


def phase_a(port):
    p1_messages = [
        {
            "role": "user",
            "content": CONTEXT + "\n\nSummarize what this module does in one sentence.",
        }
    ]
    p2_messages = [
        {
            "role": "user",
            "content": CONTEXT
            + "\n\nWhich function computes drawdown? Answer with its name.",
        }
    ]

    p1_cold = run_nonstreaming(port, p1_messages, max_tokens=16)
    p1_cached = run_nonstreaming(port, p1_messages, max_tokens=16)
    p2 = run_nonstreaming(port, p2_messages, max_tokens=16)

    return {
        "p1_cold": p1_cold,
        "p1_cached": p1_cached,
        "p2": p2,
    }


def phase_b(port):
    turn_prompts = [
        CONTEXT + "\n\nReview this module for look-ahead bias and list the top three risks.",
        "Rewrite the riskiest function to fix the issue. Show only the code.",
        "Add a vectorized pandas version of the Sharpe computation with a 63-day window.",
        "Explain how transaction costs would change the position sizing function's output.",
    ]

    history = []
    turns = []
    for prompt in turn_prompts:
        history.append({"role": "user", "content": prompt})
        result = run_streaming_turn(port, history, max_tokens=400)
        turns.append(result)
        history.append({"role": "assistant", "content": result["content"]})

    return {"turns": turns}


def median_decode_rate(turns):
    rates = [t["client_decode_tok_s"] for t in turns if t.get("client_decode_tok_s")]
    return statistics.median(rates) if rates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    result = {
        "port": args.port,
        "context_chars": len(CONTEXT),
        "phase_a": phase_a(args.port),
        "phase_b": phase_b(args.port),
    }
    result["phase_b_median_client_decode_tok_s"] = median_decode_rate(
        result["phase_b"]["turns"]
    )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
