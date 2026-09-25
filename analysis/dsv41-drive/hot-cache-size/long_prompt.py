"""One long-prompt request: the first N tokens of a text file, greedy, streamed through /generate.

Usage: long_prompt.py --port P --model DIR --text FILE --tokens N --out OUT.json [--max-new 64]

Prefill of an N-token prompt runs as N/512 chunks (--chunked-prefill-size 512), the worst case for eager VRAM use
(MoE gather staging, attention/indexer temporaries over a ~N-token context); the decode that follows attends over
all of it. Writes prompt/completion token counts, time to first token (prefill), decode ms/token and the text.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    while len(ids) < a.tokens:
        ids = ids + ids
    ids = ids[: a.tokens]
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": a.max_new, "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    stamps, last = [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            last = json.loads(payload)
            stamps.append((time.monotonic() - t0, last["meta_info"].get("completion_tokens", 0)))
    total = time.monotonic() - t0
    meta = last["meta_info"] if last else {}
    ttft = stamps[0][0] if stamps else float("nan")
    ctoks = meta.get("completion_tokens", 0)
    decode_ms = (stamps[-1][0] - ttft) * 1e3 / max(1, stamps[-1][1] - stamps[0][1]) if len(stamps) > 1 else float("nan")
    res = {"prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": ctoks, "ttft_s": round(ttft, 2),
           "total_s": round(total, 2), "decode_ms_per_token": round(decode_ms, 1),
           "finish_reason": meta.get("finish_reason"), "text": last.get("text") if last else None}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"long prompt: {res['prompt_tokens']} prompt, {ctoks} completion, ttft {ttft:.1f}s, "
          f"decode {decode_ms:.1f} ms/token, total {total:.1f}s", flush=True)
    return 0 if ctoks == a.max_new else 1


if __name__ == "__main__":
    raise SystemExit(main())
