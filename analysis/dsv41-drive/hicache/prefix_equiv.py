"""Does a prefix-cache hit change greedy output? Each case compares a warm request (its prefix served from the
cache) with the same request run cold after /flush_cache, and a second cold run gives the run-to-run noise floor.

  aligned   a 4096-token document plus a question, re-sent: the hit boundary (4096) is a prefill chunk end, so
            every late-layer SWA slot a decode window reads was written.
  midchunk  a 3900-token document seeds the cache (boundary 3840, inside the 3584..3900 chunk), then its first
            3840 tokens plus a short suffix: under decoder SWA bounded replay the late layers wrote only
            [3772, 3900) of that chunk, so the first decode windows reach unwritten slots in [3713 + suffix, 3772).
  reload    two 4096-token conversations X and Y interleaved, X revisited: with the hierarchical cache, X's
            evicted SWA tail comes back from host memory.

Per comparison: the first diverging output token and the largest |logprob| difference over the common prefix.
Usage: prefix_equiv.py --port P --model DIR --text FILE --out OUT.jsonl [--max-new N]
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from transformers import AutoTokenizer


def post(port: int, path: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body or {}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return r.read()


def generate(port: int, ids: list[int], max_new: int) -> dict:
    out = json.loads(post(port, "/generate", {
        "input_ids": ids, "return_logprob": True,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}))
    meta = out["meta_info"]
    lp = meta["output_token_logprobs"]
    return {"cached": meta.get("cached_tokens"), "prompt": meta.get("prompt_tokens"),
            "ids": [t[1] for t in lp], "logprobs": [t[0] for t in lp], "text": out["text"]}


def flush(port: int) -> None:
    post(port, "/flush_cache")


def compare(a: dict, b: dict) -> dict:
    n = min(len(a["ids"]), len(b["ids"]))
    first = next((i for i in range(n) if a["ids"][i] != b["ids"][i]), None)
    common = n if first is None else first
    dlp = max((abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(common)), default=0.0)
    return {"first_diff": first, "compared": n, "max_dlogprob": round(dlp, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    text = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    assert len(text) >= 4 * 4096, "text too short for four distinct documents"
    q = tok("\n\nSummarize the section above in two sentences.", add_special_tokens=False)["input_ids"]
    suffix = tok("\n\nIn short,", add_special_tokens=False)["input_ids"]
    docs = [text[i * 4096 : (i + 1) * 4096] for i in range(4)]
    gen = lambda ids: generate(a.port, ids, a.max_new)

    cases = {}
    p = docs[0] + q
    flush(a.port); cold1 = gen(p); warm = gen(p); flush(a.port); cold2 = gen(p)
    cases["aligned"] = (warm, cold1, cold2)

    seed_ids = docs[1][:3900]
    p = seed_ids[:3840] + suffix
    flush(a.port); gen(seed_ids); warm = gen(p)
    flush(a.port); cold1 = gen(p); flush(a.port); cold2 = gen(p)
    cases["midchunk"] = (warm, cold1, cold2)

    x, y = docs[2] + q, docs[3] + q
    flush(a.port); gen(x); gen(y); warm = gen(x)
    flush(a.port); cold1 = gen(x); flush(a.port); cold2 = gen(x)
    cases["reload"] = (warm, cold1, cold2)

    with open(a.out, "w") as f:
        for name, (warm, cold1, cold2) in cases.items():
            assert cold1["cached"] == 0 and cold2["cached"] == 0, f"{name}: flush left a cached prefix"
            row = {"case": name, "warm_cached": warm["cached"], "prompt": warm["prompt"],
                   "warm_vs_cold": compare(warm, cold1), "cold_vs_cold": compare(cold1, cold2),
                   "warm_text": warm["text"], "cold_text": cold1["text"], "cold2_text": cold2["text"]}
            f.write(json.dumps(row) + "\n")
            print(f"{name}: prompt {row['prompt']} warm cached {row['warm_cached']} "
                  f"warm-vs-cold {row['warm_vs_cold']} cold-vs-cold {row['cold_vs_cold']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
