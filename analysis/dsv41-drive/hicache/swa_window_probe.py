"""Direct probe of the decoder-bounded-replay prefix-hit defect: compare the SWA window rows a request's first
decode reads after a warm hit with the same rows after a cold prefill.

The server runs with SGLANG_DEBUG_SWA_WINDOW_DUMP_DIR, which dumps, after each bs=1 extend, the 576 data bytes
of positions [seq_len - 128, seq_len) for every layer. Sequence (prefix_poison.py's short case, trials 0..N-1):
poison the SWA pool, seed a 900-token prompt, warm = its first 768 tokens + 4; flush, cold; flush, cold2.
Then, for each trial, the dumps ending at seq 772 are warm, cold, cold2 in order.

Rows are decoded to 512 floats (fp8 nope, bf16 rope; block scales ignored) and compared by cosine similarity
per layer and position: early layers (< 21) and late layers (>= 21, past the last kv_source layer),
positions 645..767 (written only by the seed's non-tail extend in the warm case) and 768..771 (the suffix).
Usage: swa_window_probe.py run --port P --model DIR --text FILE [--trials N]
       swa_window_probe.py compare --dump DIR [--trials N]
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LATE_START = 21
MATCH = 768
SEED_LEN = 900
PROMPT = 772


def run(a) -> int:
    from prefix_poison import flush, generate, poison
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    text = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    vocab = min(len(tok), 120000)
    rng = random.Random(0)
    suffix = tok("\n\nIn short,", add_special_tokens=False)["input_ids"]
    for n in range(a.trials):
        start = 5000 + n * 6000
        seed = text[start : start + SEED_LEN]
        p = seed[:MATCH] + suffix
        assert len(p) == PROMPT, f"suffix is {len(suffix)} tokens, expected {PROMPT - MATCH}"
        poison(a.port, rng, vocab)
        generate(a.port, seed, 1)
        warm = generate(a.port, p, 1)
        flush(a.port)
        generate(a.port, p, 1)
        flush(a.port)
        generate(a.port, p, 1)
        print(f"trial{n}: warm cached {warm['cached']}", flush=True)
    return 0


def decode(rows: torch.Tensor) -> torch.Tensor:
    nope = rows[..., :448].contiguous().view(torch.float8_e4m3fn).float()
    rope = rows[..., 448:576].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope, rope], dim=-1)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(decode(a), decode(b), dim=-1)


def compare(a) -> int:
    files = sorted(glob.glob(os.path.join(a.dump, f"*-seq{PROMPT}-ext*.pt")))
    assert len(files) == 3 * a.trials, f"expected {3 * a.trials} seq{PROMPT} dumps, found {len(files)}"
    for n in range(a.trials):
        warm, cold, cold2 = (torch.load(f) for f in files[3 * n : 3 * n + 3])
        assert warm["start"] == cold["start"] == PROMPT - 128
        print(f"trial{n}: warm extend {warm['extend_len']}, cold extend {cold['extend_len']}")
        spans = {"645..767": slice(645 - warm["start"], MATCH - warm["start"]),
                 "768..771": slice(MATCH - warm["start"], PROMPT - warm["start"])}
        layers = {"early": slice(0, LATE_START), "late": slice(LATE_START, warm["rows"].shape[0])}
        for lname, ls in layers.items():
            for sname, ss in spans.items():
                wc = cos(warm["rows"][ls, ss], cold["rows"][ls, ss])
                cc = cos(cold["rows"][ls, ss], cold2["rows"][ls, ss])
                print(f"  {lname:5s} {sname}: warm-vs-cold cos mean {wc.mean():.4f} min {wc.min():.4f}"
                      f" | cold-vs-cold mean {cc.mean():.4f} min {cc.min():.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--trials", type=int, default=2)
    c = sub.add_parser("compare")
    c.add_argument("--dump", required=True)
    c.add_argument("--trials", type=int, default=2)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
