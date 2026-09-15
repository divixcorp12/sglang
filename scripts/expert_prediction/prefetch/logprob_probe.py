"""Greedy first-turn completions with top-2 logprobs, for the prefetch correctness gates."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = []
    with open(args.sessions) as f:
        sessions = [json.loads(line) for line in f][: args.prompts]
    for session in sessions:
        body = json.dumps({
            "model": "default", "messages": [{"role": "user", "content": session["turns"][0][:6000]}],
            "temperature": 0, "max_tokens": args.max_tokens, "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.loads(response.read())["choices"][0]
        tokens = [{"token": t["token"], "top": [[c["token"], c["logprob"]] for c in t["top_logprobs"]]}
                  for t in choice["logprobs"]["content"]]
        results.append({"session_id": session["session_id"], "tokens": tokens})
    with open(args.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
