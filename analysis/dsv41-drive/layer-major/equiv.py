"""Layer-major vs chunked prefill: greedy token-0 identity is the gate (plan 2026-09-27, Task 12 fix round).

Usage: equiv.py run --port P --model M --text FILE --out OUT.jsonl --commit SHA --dirty 0/1 --min-tokens N
                    [--cases all|quick]
       equiv.py compare A.jsonl B.jsonl [--allow-head-mismatch]

Pass criterion: token-0 id equality, not full 64-token completion identity. DSV4.1's decode kernels are not
bitwise-stable across batch composition, so tokens 1..63 legitimately differ between two otherwise-identical
runs; comparing them would fail a correct layer-major implementation as often as a buggy one. Ratified by
controller ruling on the Task 12 review (5 of 7 chunked-vs-chunked cases diverged within 64 tokens).
"""

import argparse
import hashlib
import json
import sys
import urllib.request

LENGTHS = [8192, 16384, 32768, 33000, 32868]


def _generate(port, ids, *, top_logprobs_num=None):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 64, "temperature": 0, "ignore_eos": True}}
    if top_logprobs_num:
        body["return_logprob"] = True
        body["top_logprobs_num"] = top_logprobs_num
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    return {"text": d["text"], "ids": d.get("output_ids"), "meta": d.get("meta_info", {})}


def _flush(port):
    # Each case starts from an empty radix cache, so every long case is a full prefill down the layer-major path.
    urllib.request.urlopen(f"http://127.0.0.1:{port}/flush_cache?timeout=60", timeout=120).read()


def _prompt_hash(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def _top0_logprobs(meta):
    # Informational only (T12 Minor 1): the server's per-step top-k, for output step 0.
    top = meta.get("output_top_logprobs")
    return top[0] if top else None


def _all_cases(ids):
    # (name, prompt, flush first): prefix-warm -> prefix is the intended radix hit, and after follows it unflushed.
    cases = [(f"len{n}", ids[:n], True) for n in LENGTHS]
    cases.append(("prefix-warm", ids[5000:6024], True))
    cases.append(("prefix", ids[5000:6024] + ids[:32768], False))
    cases.append(("after", ids[100:356], False))
    # Radix I1: an unflushed chain, so one request's admission actually sets an SWA branch point and the
    # follow-up hits it -- the shape the original GPU crash needed, which the per-case flush above skips.
    cases.append(("chain-32768", ids[:32768], True))
    cases.append(("chain-32868", ids[:32868], False))
    cases.append(("chain-32868-again", ids[:32868], False))
    return cases


# A fast subset for iteration: one plain case, the longest and its unaligned-tail sibling, the prefix-hit
# pair, and the unflushed chain -- everything that has caught a real bug, none of the redundant lengths.
QUICK_CASES = {"len8192", "len32768", "len33000", "prefix-warm", "prefix",
              "chain-32768", "chain-32868", "chain-32868-again"}


def run(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    while len(ids) < 70000:
        ids = ids + ids
    cases = _all_cases(ids)
    if a.cases == "quick":
        cases = [c for c in cases if c[0] in QUICK_CASES]
    top0_logprobs_error = None
    with open(a.out, "w") as f:
        f.write(json.dumps({"case": "__header__", "commit": a.commit, "dirty": a.dirty,
                            "min_tokens": a.min_tokens}) + "\n")
        for name, prompt, flush in cases:
            if flush:
                _flush(a.port)
            try:
                r = _generate(a.port, prompt, top_logprobs_num=5)
                top0 = _top0_logprobs(r["meta"])
            except Exception as e:  # top-5 logprobs is informational; fall back to the plain gate on refusal.
                if top0_logprobs_error is None:
                    top0_logprobs_error = repr(e)
                r = _generate(a.port, prompt)
                top0 = None
            record = {"case": name, "prompt_len": len(prompt), "prompt_hash": _prompt_hash(prompt), **r}
            if top0 is not None:
                record["top0_logprobs"] = top0
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(name, len(prompt), repr(r["text"][:60]), flush=True)
    if top0_logprobs_error:
        print(f"top-5 logprobs refused, gate unaffected: {top0_logprobs_error}", file=sys.stderr)


def _max_abs_logprob_delta(top_a, top_b):
    # Compare by token id, since the two servers' top-5 sets need not be in the same rank order.
    if not top_a or not top_b:
        return None
    lp_a = {row[1]: row[0] for row in top_a}
    lp_b = {row[1]: row[0] for row in top_b}
    shared = set(lp_a) & set(lp_b)
    if not shared:
        return None
    return max(abs(lp_a[t] - lp_b[t]) for t in shared)


def compare(path_a, path_b, *, allow_head_mismatch=False):
    a = {json.loads(l)["case"]: json.loads(l) for l in open(path_a)}
    b = {json.loads(l)["case"]: json.loads(l) for l in open(path_b)}
    header_a, header_b = a.pop("__header__", None), b.pop("__header__", None)
    if not allow_head_mismatch:
        for name, h in (("a", header_a), ("b", header_b)):
            if h is None:
                print(f"NOT COMPARABLE:\n  {path_a if name == 'a' else path_b} has no header record")
                return 2
        if header_a["commit"] != header_b["commit"]:
            print(f"NOT COMPARABLE:\n  arms ran at different heads: {header_a['commit']} vs {header_b['commit']}"
                  "\n  pass --allow-head-mismatch to compare anyway")
            return 2
    # Refuse, rather than raise, when the two runs are not comparable case for case.
    problems = [f"case {c} missing from {path_b}" for c in a if c not in b]
    problems += [f"case {c} missing from {path_a}" for c in b if c not in a]
    for case in sorted(set(a) & set(b)):
        ha, hb = a[case].get("prompt_hash"), b[case].get("prompt_hash")
        if ha is None or hb is None or ha != hb:
            problems.append(f"case {case}: prompt hash differs or is missing (a={ha} b={hb})")
        if not a[case].get("ids") or not b[case].get("ids"):
            problems.append(f"case {case}: no output ids (a={a[case].get('ids')!r} b={b[case].get('ids')!r})")
    if problems:
        print("NOT COMPARABLE:\n  " + "\n  ".join(problems))
        return 2
    bad = 0
    for case in sorted(a):
        ids_a, ids_b = a[case]["ids"], b[case]["ids"]
        token0_same = ids_a[0] == ids_b[0]
        ids_same = ids_a == ids_b
        full_same = a[case]["text"] == b[case]["text"] and ids_same
        bad += not token0_same
        print(f"{case:12s} {'IDENTICAL' if token0_same else 'DIFFERENT'} (token 0)"
              f"{'' if full_same else '  [64-token completion differs -- decode-only, not gating]'}")
        if not token0_same:
            print(f"   token 0: a={ids_a[0]!r} b={ids_b[0]!r}")
        elif not full_same:
            if ids_same:
                print("   ids equal, text differs (detokenization only)")
            else:
                first_tok = next(i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y)
                print(f"   first differing token index (decode): {first_tok} / {len(ids_a)}")
        delta = _max_abs_logprob_delta(a[case].get("top0_logprobs"), b[case].get("top0_logprobs"))
        if delta is not None:
            print(f"   token 0 top-5 logprob max |delta|: {delta:.4g}  [informational, not gating]")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--dirty", required=True)
    r.add_argument("--min-tokens", dest="min_tokens", required=True)
    r.add_argument("--cases", choices=["all", "quick"], default="all")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--allow-head-mismatch", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        sys.exit(run(args) or 0)
    else:
        sys.exit(compare(args.a, args.b, allow_head_mismatch=args.allow_head_mismatch))
