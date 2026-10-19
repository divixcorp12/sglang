"""First-turn completions with top-k logprobs, for the prefetch correctness gates: greedy by default.

``--temperature``/``--top-p`` sample instead. ``--prompts-file`` (repeatable; a JSON list of strings) replaces the
sessions with those prompts, one request each in order, thinking on as production serves (the dsv41 encoder reads
``thinking``, not ``enable_thinking``), and records each request's CLOCK_MONOTONIC span so collected rows can be mapped
back to their prompt. ``--top-logprobs 0`` asks for no logprobs.
"""

import argparse
import json
import time
import urllib.request


def sessions_from(args):
    """[(session_id, prompt text)]: the corpus sessions, or every ``--prompts-file`` prompt (``--prompts`` caps both)."""
    if not args.prompts_file:
        with open(args.sessions) as f:
            rows = [json.loads(line) for line in f][: args.prompts]
        return [(row["session_id"], row["turns"][0][:6000]) for row in rows]
    out = []
    for path in args.prompts_file:
        with open(path) as f:
            for index, text in enumerate(json.load(f)):
                out.append((f"{path.rsplit('/', 1)[-1]}:{index}", text))
    return out[: args.prompts]


def request_body(args, text):
    body = {
        "model": "default", "messages": [{"role": "user", "content": text}],
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"thinking": True} if args.prompts_file else {"enable_thinking": False},
    }
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_logprobs > 0:
        body["logprobs"] = True
        body["top_logprobs"] = args.top_logprobs
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--top-logprobs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--prompts-file", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.prompts_file and not args.sessions:
        parser.error("--sessions or --prompts-file is required")
    results = []
    for session_id, text in sessions_from(args):
        body = json.dumps(request_body(args, text)).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        sent_ns = time.monotonic_ns()
        with urllib.request.urlopen(request, timeout=900) as response:
            reply = json.loads(response.read())
        done_ns = time.monotonic_ns()
        choice = reply["choices"][0]
        entry = {"session_id": session_id}
        if args.top_logprobs > 0:
            entry["tokens"] = [{"token": t["token"], "top": [[c["token"], c["logprob"]] for c in t["top_logprobs"]]}
                               for t in choice["logprobs"]["content"]]
        if args.prompts_file:
            entry |= {"sent_ns": sent_ns, "done_ns": done_ns, "clock": "CLOCK_MONOTONIC",
                      "completion_tokens": reply.get("usage", {}).get("completion_tokens"),
                      "finish_reason": choice.get("finish_reason")}
        results.append(entry)
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
