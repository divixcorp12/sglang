"""Post-analysis of one drive_ab.sh A/B: completions, decode ms/token, NVMe bytes, and flag-specific log proof.

Usage: python compare_ab.py <ab_dir> [--b-log-marker "MoE side stream: first fork"]
Reads <ab_dir>/<tag>-A.dirs and <tag>-B.dirs (one run dir per invocation, in run order).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def load_run(d):
    res = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"))]
    b = [json.loads(l) for l in open(os.path.join(d, "boundary-samples.jsonl"))]
    sectors = lambda s: sum(s["diskstats_sectors"].values())
    ready = next((x for x in b if x["label"] != "before_server"), b[0])
    return res, (sectors(b[-1]) - sectors(ready)) * 512, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ab_dir")
    ap.add_argument("--b-log-marker", default=None)
    args = ap.parse_args()
    arms = {}
    for f in sorted(glob.glob(os.path.join(args.ab_dir, "*.dirs"))):
        arm = f.rsplit("-", 1)[1].split(".")[0]  # A or B
        arms[arm] = [load_run(d.strip()) for d in open(f) if d.strip()]
    out = {"per_session": [], "arms": {}}
    by_sid = {}
    for arm, runs in arms.items():
        tok = sec = nvme = 0
        for res, nbytes, d in runs:
            nvme += nbytes
            for r in res:
                by_sid.setdefault((r["session_id"], r["turn"]), {})[arm] = r
                if r["completion_tokens"] and r["decode_tokens_per_sec"]:
                    tok += r["completion_tokens"] - 1
                    sec += (r["completion_tokens"] - 1) / r["decode_tokens_per_sec"]
        marker = None
        if args.b_log_marker:
            marker = [sum(args.b_log_marker in l for l in open(os.path.join(d, "server.log"), errors="replace")) for _, _, d in runs]
        out["arms"][arm] = {"decode_tokens": tok, "decode_ms_per_token_pooled": 1000 * sec / tok if tok else None,
                            "nvme_bytes_ready_to_end": nvme, "log_marker_counts": marker}
    diffs = []
    for key, v in sorted(by_sid.items()):
        if "A" not in v or "B" not in v:
            continue
        a, b = v["A"], v["B"]
        ma = 1000 / a["decode_tokens_per_sec"] if a["decode_tokens_per_sec"] else None
        mb = 1000 / b["decode_tokens_per_sec"] if b["decode_tokens_per_sec"] else None
        same = a["content"] == b["content"] and a.get("reasoning") == b.get("reasoning")
        row = {"session": key[0], "turn": key[1], "tokens": [a["completion_tokens"], b["completion_tokens"]],
               "ms_tok_A": ma, "ms_tok_B": mb, "delta_ms": (mb - ma) if ma and mb else None,
               "ttft_A": a["ttft"], "ttft_B": b["ttft"], "identical_output": same}
        out["per_session"].append(row)
        if row["delta_ms"] is not None:
            diffs.append(row["delta_ms"])
    out["median_delta_ms_per_token_B_minus_A"] = statistics.median(diffs) if diffs else None
    out["all_outputs_identical"] = all(r["identical_output"] for r in out["per_session"])
    for r in out["per_session"]:
        f = lambda v, spec=".2f": "n/a" if v is None else format(v, spec)
        print(f"{r['session'][:40]:40s} t{r['turn']} tok {r['tokens']} A {f(r['ms_tok_A'])} B {f(r['ms_tok_B'])} "
              f"d {f(r['delta_ms'], '+.2f')} ms/tok  ttft {f(r['ttft_A'])}/{f(r['ttft_B'])}  same={r['identical_output']}")
    print(json.dumps({k: v for k, v in out.items() if k != "per_session"}, indent=1))
    json.dump(out, open(os.path.join(args.ab_dir, "compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
