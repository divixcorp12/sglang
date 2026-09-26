"""Two interleaved multi-turn conversations over long documents, greedy, through /generate.

Each conversation starts from its own DOC-token slice of a text file and adds short follow-up turns; turns run
X1 Y1 X2 Y2 ... so a GPU KV pool that holds one conversation but not two must evict the other's prefix between
visits. Per turn it records prompt tokens, the server's cached_tokens, time to first token and the output ids.

Usage: multiturn.py --port P --model DIR --text FILE --doc-tokens N --turns K --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer

QUESTIONS = (
    "\n\nSummarize the section above in two sentences.",
    "\n\nWhich numbers in it matter most, and why?",
    "\n\nWhat would you measure next?",
    "\n\nList one risk the text does not address.",
)


def generate(port: int, ids: list[int], max_new: int) -> tuple[dict, float]:
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft, last = None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            last = json.loads(payload)
            if ttft is None:
                ttft = time.monotonic() - t0
    return last, ttft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--doc-tokens", type=int, default=4096)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    text = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    assert len(text) >= 2 * a.doc_tokens, "text too short for two distinct documents"
    convs = {"X": text[: a.doc_tokens], "Y": text[a.doc_tokens : 2 * a.doc_tokens]}
    with open(a.out, "w") as f:
        for turn in range(a.turns):
            for name in convs:
                ids = convs[name] + tok(QUESTIONS[turn % len(QUESTIONS)], add_special_tokens=False)["input_ids"]
                last, ttft = generate(a.port, ids, a.max_new)
                meta = last["meta_info"]
                out_ids = meta.get("output_ids") or tok(last["text"], add_special_tokens=False)["input_ids"]
                convs[name] = ids + out_ids
                row = {"conv": name, "turn": turn + 1, "prompt_tokens": meta.get("prompt_tokens"),
                       "cached_tokens": meta.get("cached_tokens"), "ttft_s": round(ttft, 3),
                       "completion_tokens": meta.get("completion_tokens"), "text": last["text"]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{name}{turn + 1}: prompt {row['prompt_tokens']} cached {row['cached_tokens']} "
                      f"ttft {row['ttft_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
