"""Step 3 of the FlashInfer-prefill check: compare two long_prompt.py runs (spec_margin_capture.py --long-prompt, greedy).

Usage: compare_long.py TRITON_RUN_DIR FLASHINFER_RUN_DIR [--out JSON]

Each RUN_DIR holds long_prompt.json and probe.json (the capture's OUT/NAME). Reports, for the long request of each:
  prompt tokens, TTFT, completion tokens streamed, decode tokens/s over the window after the first token;
  the first generated token at which the two runs differ (probe.json's per-token list; one chunk can carry several
  tokens, so the position is in tokens, the text prefix in characters), and how many tokens each streamed.

Greedy decoding with two different attention kernels is not expected to match forever: bf16 rounding differences
flip a near-tie argmax somewhere, and everything after it differs. The verdict is therefore a prefix length, not equality:
  PASS when the prompt token counts agree and the common prefix is at least MIN_PREFIX_TOKENS tokens (or the whole
  shorter stream), and both texts are coherent (the first 600 characters of each are printed for the owner to read).
  A divergence inside the first MIN_PREFIX_TOKENS tokens is a FAIL to be explained by step 2's numbers, not waved through.
Decode tokens/s is printed for orientation only: one request is not a throughput A/B (step 4).
"""

import argparse
import json
import os
import sys

MIN_PREFIX_TOKENS = 64


def load(run_dir):
    lp = json.load(open(os.path.join(run_dir, "long_prompt.json")))
    probe = json.load(open(os.path.join(run_dir, "probe.json")))
    req = [r for r in lp["requests"] if r["kind"] == "long"][-1]
    toks = [t["token"] for t in [e for e in probe if e["kind"] == "long"][-1]["tokens"]]
    text = "".join((c.get("reasoning") or "") + (c.get("content") or "") for c in req["chunks"])
    chunks = [c for c in req["chunks"] if c.get("completion_tokens")]
    prompt_tokens = next((c["prompt_tokens"] for c in req["chunks"] if c.get("prompt_tokens")), None)
    rate = None
    if len(chunks) > 1:
        dt = (chunks[-1]["ns"] - chunks[0]["ns"]) / 1e9
        rate = (chunks[-1]["completion_tokens"] - chunks[0]["completion_tokens"]) / dt if dt > 0 else None
    return {
        "dir": run_dir, "prompt_tokens": prompt_tokens, "ttft_s": req["ttft_s"], "finish": req["finish_reason"],
        "completion_tokens": chunks[-1]["completion_tokens"] if chunks else 0, "decode_tok_s": rate,
        "tokens": toks, "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("triton")
    ap.add_argument("flashinfer")
    ap.add_argument("--out")
    a = ap.parse_args()
    t, f = load(a.triton), load(a.flashinfer)
    n = min(len(t["tokens"]), len(f["tokens"]))
    first = next((i for i in range(n) if t["tokens"][i] != f["tokens"][i]), None)
    prefix = n if first is None else first
    chars = 0
    for x, y in zip(t["text"], f["text"]):
        if x != y:
            break
        chars += 1
    ok = n > 0 and prefix >= min(MIN_PREFIX_TOKENS, n) and t["prompt_tokens"] == f["prompt_tokens"]
    res = {
        "verdict": "PASS" if ok else "FAIL",
        "common_prefix_tokens": prefix, "compared_tokens": n, "first_divergence_token": first,
        "common_prefix_chars": chars,
        "triton": {k: v for k, v in t.items() if k not in ("tokens", "text")},
        "flashinfer": {k: v for k, v in f.items() if k not in ("tokens", "text")},
    }
    print(json.dumps(res, indent=1))
    for name, r in (("triton", t), ("flashinfer", f)):
        print(f"--- {name}: first 600 chars ---\n{r['text'][:600]}")
    if first is not None:
        lo = max(0, first - 3)
        print("--- around the first divergence (token index %d) ---" % first)
        print("triton    :", t["tokens"][lo:first + 4])
        print("flashinfer:", f["tokens"][lo:first + 4])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
