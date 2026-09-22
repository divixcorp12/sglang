"""Drive multi-turn chat sessions against a capture-enabled SGLang server.

Reads sessions written by build_sessions.py, sends each turn to
/v1/chat/completions with streaming enabled, and appends one results-JSONL
line per turn (timing, token usage, scoring for ConvFinQA).
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

ANSWER_RE = re.compile(
    r"ANSWER:\s*\$?\s*([-+]?[\d,\.]+)\s*%?\s*$", re.IGNORECASE | re.MULTILINE
)


def _load_sessions(path):
    sessions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def _load_done_turns(results_path):
    done = {}
    contents = {}
    if not os.path.exists(results_path):
        return done, contents
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("session_id")
            turn = rec.get("turn")
            if sid is not None and turn is not None:
                done.setdefault(sid, set()).add(turn)
                contents.setdefault(sid, {})[turn] = rec.get("content", "")
    return done, contents


def _session_fully_done(done_turns, session):
    turns_done = done_turns.get(session["session_id"], set())
    return len(turns_done) == len(session["turns"])


def _parse_answer(text):
    if not text:
        return None
    match = None
    for m in ANSWER_RE.finditer(text):
        match = m
    if match is None:
        return None
    raw = match.group(0)
    is_percent = raw.rstrip().endswith("%")
    num_str = match.group(1).replace(",", "").replace("$", "")
    try:
        value = float(num_str)
    except ValueError:
        return None
    return value, is_percent


def _score(content, expected):
    if expected is None:
        return None
    parsed = _parse_answer(content)
    if parsed is None:
        return False
    value, is_percent = parsed
    try:
        expected_value = float(expected)
    except (TypeError, ValueError):
        return False

    def close(a, b):
        return abs(a - b) <= max(1e-3, abs(b) * 0.01)

    if close(value, expected_value):
        return True
    if is_percent and close(value / 100.0, expected_value):
        return True
    if not is_percent and close(value * 100.0, expected_value):
        return True
    return False


def _echo(text):
    print(text, end="", flush=True)


def _stream_chat(port, messages, rid, max_tokens, timeout=1800, echo=False):
    body = json.dumps(
        {
            "model": "default",
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "rid": rid,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    start = time.monotonic()
    ttft = None
    last_token = None
    reasoning_parts = []
    content_parts = []
    finish_reason = None
    usage = None
    # Raw per-chunk arrival times (seconds since `start`), one per reasoning/content
    # delta. Client-side observation only (detokenizer, serialization, socket, event
    # loop and client scheduling are all inside it) -- never call this "step latency",
    # which is an engine-side quantity measured elsewhere. Purely additive: nothing
    # else in this function changes.
    chunk_times = []

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage_chunk = chunk.get("usage")
            if usage_chunk:
                usage = usage_chunk
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            finish_reason = choices[0].get("finish_reason") or finish_reason
            reasoning_delta = delta.get("reasoning_content") or ""
            content_delta = delta.get("content") or ""
            if reasoning_delta or content_delta:
                now = time.monotonic()
                if ttft is None:
                    ttft = now - start
                last_token = now
                chunk_times.append(now - start)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
                if echo:
                    _echo(reasoning_delta)
            if content_delta:
                if echo and not content_parts:
                    _echo("\n--- answer ---\n")
                content_parts.append(content_delta)
                if echo:
                    _echo(content_delta)

    total = time.monotonic() - start
    return {
        "ttft": ttft,
        "total": total,
        "decode_seconds": None if ttft is None else last_token - start - ttft,
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "chunk_times": chunk_times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--session-ids",
        default="",
        help="Comma list of session_ids to run, in this file's order; empty runs all.",
    )
    parser.add_argument(
        "--print-stream",
        action="store_true",
        help="Echo reasoning and answers to stdout as they stream.",
    )
    parser.add_argument(
        "--stop-when-capture-stopped",
        metavar="CAPTURE_DIR",
        help="Stop issuing new sessions once CAPTURE_DIR/capture-stopped.json exists "
        "(the writer hit its byte cap and further traffic would be wasted GPU time).",
    )
    args = parser.parse_args()

    sessions = _load_sessions(args.sessions)
    if args.session_ids:
        wanted = set(args.session_ids.split(","))
        sessions = [session for session in sessions if session["session_id"] in wanted]
    done_turns, done_contents = _load_done_turns(args.results)
    stop_marker = (
        os.path.join(args.stop_when_capture_stopped, "capture-stopped.json")
        if args.stop_when_capture_stopped
        else None
    )

    run_count = 0
    with open(args.results, "a") as results_f:
        for session in sessions:
            if _session_fully_done(done_turns, session):
                continue
            if args.max_sessions is not None and run_count >= args.max_sessions:
                break
            if stop_marker is not None and os.path.exists(stop_marker):
                print(f"capture stopped ({stop_marker}); ending driver", flush=True)
                break
            run_count += 1

            history = []
            session_done_turns = done_turns.get(session["session_id"], set())
            session_done_contents = done_contents.get(session["session_id"], {})
            for turn_idx, user_text in enumerate(session["turns"]):
                if turn_idx in session_done_turns:
                    history.append({"role": "user", "content": user_text})
                    history.append(
                        {"role": "assistant", "content": session_done_contents.get(turn_idx, "")}
                    )
                    continue

                history.append({"role": "user", "content": user_text})
                rid = f"{session['session_id']}-t{turn_idx}"
                expected = (
                    session["expected"][turn_idx]
                    if turn_idx < len(session["expected"])
                    else None
                )

                try:
                    if args.print_stream:
                        print(
                            f"\n===== {rid} ({session['domain']}, {session['split']}) =====",
                            flush=True,
                        )
                    result = _stream_chat(
                        args.port, history, rid, args.max_tokens, echo=args.print_stream
                    )
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    record = {
                        "session_id": session["session_id"],
                        "domain": session["domain"],
                        "split": session["split"],
                        "turn": turn_idx,
                        "rid": rid,
                        "error": str(exc),
                    }
                    results_f.write(json.dumps(record) + "\n")
                    results_f.flush()
                    break

                content = result["content"]
                usage = result["usage"] or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                decode_tps = None
                if completion_tokens and completion_tokens > 1 and result["decode_seconds"]:
                    decode_tps = (completion_tokens - 1) / result["decode_seconds"]

                correct = (
                    _score(content, expected) if session["domain"] == "convfinqa" else None
                )

                record = {
                    "session_id": session["session_id"],
                    "domain": session["domain"],
                    "split": session["split"],
                    "turn": turn_idx,
                    "rid": rid,
                    "ttft": result["ttft"],
                    "total": result["total"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "decode_tokens_per_sec": decode_tps,
                    "finish_reason": result["finish_reason"],
                    "truncated": result["finish_reason"] == "length",
                    "reasoning": result["reasoning"],
                    "content": content,
                    "expected": expected,
                    "correct": correct,
                    "chunk_times": result["chunk_times"],
                }
                results_f.write(json.dumps(record) + "\n")
                results_f.flush()
                if args.print_stream:
                    print(
                        f"\n----- {rid}: {completion_tokens} tokens, "
                        f"finish={result['finish_reason']}, correct={correct} -----",
                        flush=True,
                    )

                history.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
