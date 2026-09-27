"""Layer-major vs chunked prefill: greedy 64-token outputs must be identical (plan 2026-09-27, Task 12).

Usage: equiv.py run --port P --model M --text FILE --out OUT.jsonl
       equiv.py compare A.jsonl B.jsonl
"""

import argparse
import json
import sys
import urllib.request

from transformers import AutoTokenizer

LENGTHS = [8192, 16384, 32768, 33000, 32868]


def _generate(port, ids):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 64, "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    return {"text": d["text"], "ids": d.get("output_ids"), "meta": d.get("meta_info", {})}


def run(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    while len(ids) < 70000:
        ids = ids + ids
    cases = [(f"len{n}", ids[:n]) for n in LENGTHS]
    cases.append(("prefix-warm", ids[5000:6024]))
    cases.append(("prefix", ids[5000:6024] + ids[:32768]))
    cases.append(("after", ids[100:356]))
    with open(a.out, "w") as f:
        for name, prompt in cases:
            r = _generate(a.port, prompt)
            f.write(json.dumps({"case": name, **r}) + "\n")
            f.flush()
            print(name, len(prompt), repr(r["text"][:60]), flush=True)


def compare(path_a, path_b):
    a = {json.loads(l)["case"]: json.loads(l) for l in open(path_a)}
    b = {json.loads(l)["case"]: json.loads(l) for l in open(path_b)}
    bad = 0
    for case in a:
        same = a[case]["text"] == b[case]["text"] and a[case]["ids"] == b[case]["ids"]
        bad += not same
        print(f"{case:12s} {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            first = next((i for i, (x, y) in enumerate(zip(a[case]["text"], b[case]["text"])) if x != y), None)
            print(f"   first differing character: {first}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    sys.exit(run(args) or 0 if args.cmd == "run" else compare(args.a, args.b))
