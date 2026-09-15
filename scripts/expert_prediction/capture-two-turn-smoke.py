"""Send a two-turn chat (turn 2 repeats turn 1 as history) and print token usage as JSON."""

import argparse
import json
import urllib.request


def _chat(port, messages, max_tokens):
    body = json.dumps(
        {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    messages = [
        {
            "role": "user",
            "content": "Write a Python function that computes the rolling Sharpe ratio "
            "of a pandas Series of daily returns over a 63-day window, annualized.",
        }
    ]
    first = _chat(args.port, messages, args.max_tokens)
    messages.append({"role": "assistant", "content": first["choices"][0]["message"]["content"] or ""})
    messages.append(
        {"role": "user", "content": "Now make it robust to missing days and explain the bias."}
    )
    second = _chat(args.port, messages, args.max_tokens)
    print(json.dumps({"turn1": first["usage"], "turn2": second["usage"]}))


if __name__ == "__main__":
    main()
