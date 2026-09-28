"""Layer-major vs chunked prefill: greedy token-0 identity is the gate (plan 2026-09-27, Task 12 fix round).

Usage: equiv.py run --port P --model M --text FILE --out OUT.jsonl --commit SHA --dirty 0/1 --min-tokens N
                    [--cases all|quick|c1]
       equiv.py compare A.jsonl B.jsonl [--allow-head-mismatch]

Pass criterion: token-0 id equality, not full 64-token completion identity -- DSV4.1 decode is not bitwise-stable
across batch composition (DSV41_REFERENCE.md 27.19), so a full-completion compare would fail a correct pass too.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
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


def _mean_output_logprob(meta):
    # C1: mean of the sampled token's own logprob over the 64 output tokens -- token 0 alone
    # cannot see C1 (the tail is floored at the tail start), so decode quality is read from this.
    lps = meta.get("output_token_logprobs")
    if not lps:
        return None
    return sum(row[0] for row in lps) / len(lps)


def _supports_top_logprobs(port):
    # A one-time probe before any case runs: a per-case try/except would re-send a case's own prompt on
    # failure, which inside the unflushed chain changes the cache state the chain is testing.
    try:
        _generate(port, [0, 1, 2], top_logprobs_num=5)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return False
        raise


def _all_cases(ids):
    # (name, prompt, flush first): prefix-warm -> prefix is the intended radix hit, and after follows it unflushed.
    cases = [(f"len{n}", ids[:n], True) for n in LENGTHS]
    # C1: chunk*2 + 8 at chunked_prefill_size=4096 -- a final span of 8 rows, well inside the
    # SWA window, is the case the review found unreproduced on GPU (final-fix-brief.md).
    cases.append(("len8200", ids[:8200], True))
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

# C1 verification: just the short-final-span case (final-fix-brief.md).
C1_CASES = {"len8200"}

# The named --cases subsets compare() accepts against a full ("all") baseline.
_CASE_SUBSETS = {"quick": QUICK_CASES, "c1": C1_CASES}


def run(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    while len(ids) < 70000:
        ids = ids + ids
    cases = _all_cases(ids)
    if a.cases in _CASE_SUBSETS:
        cases = [c for c in cases if c[0] in _CASE_SUBSETS[a.cases]]
    top_logprobs_num = 5 if _supports_top_logprobs(a.port) else None
    if top_logprobs_num:
        _flush(a.port)  # clear the probe's own cache footprint before the first real case
    else:
        print("server refuses top-5 logprobs (HTTP 400); running without them, gate unaffected", file=sys.stderr)
    with open(a.out, "w") as f:
        f.write(json.dumps({"case": "__header__", "commit": a.commit, "dirty": a.dirty,
                            "min_tokens": a.min_tokens, "cases": a.cases}) + "\n")
        for name, prompt, flush in cases:
            if flush:
                _flush(a.port)
            r = _generate(a.port, prompt, top_logprobs_num=top_logprobs_num)
            top0 = _top0_logprobs(r["meta"]) if top_logprobs_num else None
            mean_lp = _mean_output_logprob(r["meta"])
            record = {"case": name, "prompt_len": len(prompt), "prompt_hash": _prompt_hash(prompt), **r}
            if top0 is not None:
                record["top0_logprobs"] = top0
            if mean_lp is not None:
                record["mean_output_logprob"] = mean_lp
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(name, len(prompt), repr(r["text"][:60]), flush=True)


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


def _is_dirty(v) -> bool:
    return str(v) not in ("0", "", "None")


def compare(path_a, path_b, *, allow_head_mismatch=False, allow_dirty=False):
    """Gate: token-0 id equality (see module docstring). Prints the full 64-token divergence informationally."""
    a = {json.loads(l)["case"]: json.loads(l) for l in open(path_a)}
    b = {json.loads(l)["case"]: json.loads(l) for l in open(path_b)}
    header_a, header_b = a.pop("__header__", None), b.pop("__header__", None)
    if header_a is None or header_b is None:
        missing = path_a if header_a is None else path_b
        print(f"NOT COMPARABLE:\n  {missing} has no header record")
        return 2
    if not allow_head_mismatch and header_a["commit"] != header_b["commit"]:
        print(f"NOT COMPARABLE:\n  arms ran at different heads: {header_a['commit']} vs {header_b['commit']}"
              "\n  pass --allow-head-mismatch to compare anyway")
        return 2
    if not allow_dirty and (_is_dirty(header_a.get("dirty")) or _is_dirty(header_b.get("dirty"))):
        print(f"NOT COMPARABLE:\n  a dirty worktree ran one of these arms (a={header_a.get('dirty')} "
              f"b={header_b.get('dirty')})\n  pass --allow-dirty to compare anyway")
        return 2
    if int(header_a.get("min_tokens", -1)) == 0 and int(header_b.get("min_tokens", -1)) == 0:
        print("NOT COMPARABLE:\n  both arms have min_tokens=0 (chunked); nothing ran layer-major")
        return 2
    smaller, larger, smaller_path, larger_path = (a, b, path_a, path_b) if len(a) <= len(b) else (b, a, path_b, path_a)
    smaller_header = header_a if smaller is a else header_b
    # A subset is only ever legitimate as a deliberate --cases quick/c1 set, not an accident (e.g. a crashed
    # arm's partial jsonl): require the smaller file's own header to say so and its cases to match exactly.
    if len(smaller) < len(larger):
        allowed = _CASE_SUBSETS.get(smaller_header.get("cases"))
        if allowed is None or set(smaller) != allowed:
            print(f"NOT COMPARABLE:\n  {smaller_path} has {len(smaller)} cases, {larger_path} has {len(larger)}, "
                  f"and {smaller_path} is not exactly a known subset (cases={smaller_header.get('cases')!r})")
            return 2
    problems = [f"case {c} (in {smaller_path}) missing from {larger_path}" for c in smaller if c not in larger]
    for case in sorted(smaller):
        if case not in larger:
            continue
        ha, hb = a[case].get("prompt_hash"), b[case].get("prompt_hash")
        if ha is None or hb is None or ha != hb:
            problems.append(f"case {case}: prompt hash differs or is missing (a={ha} b={hb})")
        if not a[case].get("ids") or not b[case].get("ids"):
            problems.append(f"case {case}: no output ids (a={a[case].get('ids')!r} b={b[case].get('ids')!r})")
    if problems:
        print("NOT COMPARABLE:\n  " + "\n  ".join(problems))
        return 2
    bad = 0
    for case in sorted(smaller):
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
                shorter = min(len(ids_a), len(ids_b))
                first_tok = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), shorter)
                print(f"   first differing token index (decode): {first_tok} / {len(ids_a)} vs {len(ids_b)}")
        delta = _max_abs_logprob_delta(a[case].get("top0_logprobs"), b[case].get("top0_logprobs"))
        if delta is not None:
            print(f"   token 0 top-5 logprob max |delta|: {delta:.4g}  [informational, not gating]")
        mlp_a, mlp_b = a[case].get("mean_output_logprob"), b[case].get("mean_output_logprob")
        if mlp_a is not None and mlp_b is not None:
            print(f"   mean output-token logprob (64 tokens): a={mlp_a:.4g} b={mlp_b:.4g}"
                  "  [informational, not gating: this is what C1 degrades, not token 0]")
        print(f"   first 20 tokens: a={ids_a[:20]!r}")
        print(f"                    b={ids_b[:20]!r}")
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
    r.add_argument("--cases", choices=["all", "quick", "c1"], default="all")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--allow-head-mismatch", action="store_true")
    c.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        sys.exit(run(args) or 0)
    else:
        sys.exit(compare(args.a, args.b, allow_head_mismatch=args.allow_head_mismatch, allow_dirty=args.allow_dirty))
