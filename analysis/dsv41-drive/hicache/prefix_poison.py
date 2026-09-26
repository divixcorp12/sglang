"""Decoder SWA bounded replay plus a mid-chunk prefix hit: does decode read late-layer SWA slots nobody wrote?

Under --enable-decoder-swa-bounded-replay a prefill writes late-layer (past the last kv_source layer) SWA KV only for
the last min(128, extend) tokens of each extend, while decode reads the full 128-token window and the radix match
checks only that SWA slots are allocated. A seed of 900 tokens prefills [0, 512) and [512, 900), so late layers
write [384, 512) and [772, 900). A warm request of its first 768 tokens plus a short suffix hits 768, and its first
decode window reaches [645, 768): 123 slots no forward of this sequence wrote.

To make those slots hold foreign KV rather than whatever is left over, each trial first poisons the SWA pool: after
/flush_cache, 255-token random-token requests (one page each, max_new 1) write late-layer KV at page offsets
127..254 of every SWA page; /flush_cache resets the free list but leaves the bytes.

  short    seed, then warm = seed[:768] + 4-token suffix: the first decode reads 123 stale slots.
  control  seed, then warm = seed[:768] + 136-token suffix: the warm extend's own tail covers every decode window.

Per trial the warm output is compared with the same prompt run cold after a flush, and a second cold run gives the
noise floor. step1 is the first decode step's logprob difference: the step with the most stale slots.
Usage: prefix_poison.py --port P --model DIR --text FILE --out OUT.jsonl [--short N] [--control N]
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer

SEED_LEN = 900
MATCH = 768
POISON_LEN = 255
POISON_REQUESTS = 20
MAX_NEW = 64


def post(port: int, path: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body or {}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return r.read()


def generate(port: int, ids: list[int], max_new: int) -> dict:
    out = json.loads(post(port, "/generate", {
        "input_ids": ids, "return_logprob": True,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}))
    meta = out["meta_info"]
    lp = meta["output_token_logprobs"]
    return {"cached": meta.get("cached_tokens"), "prompt": meta.get("prompt_tokens"),
            "ids": [t[1] for t in lp], "logprobs": [t[0] for t in lp], "text": out["text"]}


def flush(port: int) -> None:
    # The scheduler refuses a flush until it is fully idle, which lags a finished request by hicache write-back.
    for _ in range(120):
        try:
            post(port, "/flush_cache")
            return
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            time.sleep(1)
    raise RuntimeError("server never went idle enough to flush its cache")


def compare(a: dict, b: dict) -> dict:
    n = min(len(a["ids"]), len(b["ids"]))
    first = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
    common = n if first is None else first
    dlp = max((abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(common)), default=0.0)
    step1 = abs(a["logprobs"][1] - b["logprobs"][1]) if common > 1 else None
    return {"first_diff": first, "max_dlogprob": round(dlp, 4),
            "step1_dlogprob": None if step1 is None else round(step1, 4)}


def poison(port: int, rng: random.Random, vocab: int) -> None:
    flush(port)
    for _ in range(POISON_REQUESTS):
        generate(port, [rng.randrange(1000, vocab) for _ in range(POISON_LEN)], 1)
    flush(port)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", type=int, default=4)
    ap.add_argument("--control", type=int, default=2)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    text = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    vocab = min(len(tok), 120000)
    rng = random.Random(0)
    short_suffix = tok("\n\nIn short,", add_special_tokens=False)["input_ids"]
    trials = [("short", i) for i in range(a.short)] + [("control", i) for i in range(a.control)]
    gen = lambda ids, n=MAX_NEW: generate(a.port, ids, n)

    with open(a.out, "w") as f:
        for n, (kind, i) in enumerate(trials):
            start = 5000 + n * 6000
            seed = text[start : start + SEED_LEN]
            suffix = short_suffix if kind == "short" else text[start + 3000 : start + 3136]
            p = seed[:MATCH] + suffix
            poison(a.port, rng, vocab)
            gen(seed)
            warm = gen(p)
            flush(a.port); cold1 = gen(p)
            flush(a.port); cold2 = gen(p)
            assert cold1["cached"] == 0 and cold2["cached"] == 0, f"{kind}{i}: flush left a cached prefix"
            row = {"kind": kind, "trial": i, "prompt": warm["prompt"], "warm_cached": warm["cached"],
                   "warm_vs_cold": compare(warm, cold1), "cold_vs_cold": compare(cold1, cold2),
                   "warm_text": warm["text"], "cold_text": cold1["text"], "cold2_text": cold2["text"]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{kind}{i}: prompt {row['prompt']} warm cached {row['warm_cached']} "
                  f"warm-vs-cold {row['warm_vs_cold']} cold-vs-cold {row['cold_vs_cold']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
