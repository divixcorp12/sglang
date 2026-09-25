#!/usr/bin/env python3
"""Per-session serving metrics of run_arm.sh arms, and output byte-identity between them. CPU only.

For each arm directory (run_arm.sh's run dir) and each timed session: TTFT, completion tokens, finish reason, decode
tok/s (run_capture_sessions.py's own figure, README rule 1), ms/token (its inverse), and the client-side inter-token
gaps from ``chunk_times``: p50, p90, max, and stalls (gaps of at least 0.5 s, about 4x a ~120 ms step, and at least
2 s, the RAM-miss wait deadline). The gaps are a client-side proxy measured through HTTP, not engine step latency
(client_latency.py). Identity compares each session's ``content`` across arms, byte for byte.

    arm_metrics.py NAME=RUN_DIR [NAME=RUN_DIR ...] [--json OUT]
"""

import argparse
import hashlib
import json
import os
import statistics

STALL_S = (0.5, 2.0)


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def session_metrics(row):
    chunks = row["chunk_times"]
    if isinstance(chunks, str):
        chunks = json.loads(chunks)
    gaps = [b - a for a, b in zip(chunks, chunks[1:])]
    tok_s = row["decode_tokens_per_sec"]
    return {
        "session_id": row["session_id"],
        "ttft_s": row["ttft"],
        "completion_tokens": row["completion_tokens"],
        "finish_reason": row["finish_reason"],
        "decode_tok_s": tok_s,
        "ms_per_token": None if not tok_s else 1000.0 / tok_s,
        "gap_ms_p50": 1000 * pct(gaps, 0.5) if gaps else None,
        "gap_ms_p90": 1000 * pct(gaps, 0.9) if gaps else None,
        "gap_ms_max": 1000 * max(gaps) if gaps else None,
        "gaps": len(gaps),
        **{f"stalls_ge_{t}s": sum(g >= t for g in gaps) for t in STALL_S},
        "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
        "_gaps": gaps,
        "_content": row["content"],
    }


def arm_metrics(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "results.jsonl")) if line.strip()]
    sessions = [session_metrics(r) for r in rows]
    gaps = [g for s in sessions for g in s["_gaps"]]
    tokens = sum(s["completion_tokens"] - 1 for s in sessions)
    decode_s = sum((s["completion_tokens"] - 1) / s["decode_tok_s"] for s in sessions)
    return {
        "run_dir": run_dir,
        "sessions": sessions,
        "pooled": {
            "decode_tok_s": tokens / decode_s,
            "ms_per_token": 1000 * decode_s / tokens,
            "ttft_s_mean": statistics.mean(s["ttft_s"] for s in sessions),
            "gap_ms_p50": 1000 * pct(gaps, 0.5),
            "gap_ms_p90": 1000 * pct(gaps, 0.9),
            "gap_ms_max": 1000 * max(gaps),
            **{f"stalls_ge_{t}s": sum(g >= t for g in gaps) for t in STALL_S},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=RUN_DIR")
    ap.add_argument("--json")
    a = ap.parse_args()
    arms = {}
    for spec in a.arms:
        name, run_dir = spec.split("=", 1)
        arms[name] = arm_metrics(run_dir)
    names = list(arms)
    for name, m in arms.items():
        print(f"== {name}  {m['run_dir']}")
        for s in m["sessions"]:
            print(
                f"  {s['session_id'][:45]:45s} ttft {s['ttft_s']:6.2f} s  tokens {s['completion_tokens']:3d} "
                f"({s['finish_reason']})  {s['decode_tok_s']:.3f} tok/s = {s['ms_per_token']:.1f} ms/token  "
                f"gap p50 {s['gap_ms_p50']:.1f} p90 {s['gap_ms_p90']:.1f} max {s['gap_ms_max']:.1f} ms  "
                f"stalls>=0.5s {s['stalls_ge_0.5s']} >=2s {s['stalls_ge_2.0s']}"
            )
        p = m["pooled"]
        print(
            f"  pooled: {p['decode_tok_s']:.3f} tok/s = {p['ms_per_token']:.1f} ms/token, TTFT mean {p['ttft_s_mean']:.2f} s, "
            f"gap p50 {p['gap_ms_p50']:.1f} p90 {p['gap_ms_p90']:.1f} max {p['gap_ms_max']:.1f} ms, "
            f"stalls>=0.5s {p['stalls_ge_0.5s']} >=2s {p['stalls_ge_2.0s']}"
        )
    identity = {}
    ref = names[0]
    for other in names[1:]:
        by_id = {s["session_id"]: s for s in arms[other]["sessions"]}
        for s in arms[ref]["sessions"]:
            o = by_id.get(s["session_id"])
            same = o is not None and o["_content"] == s["_content"]
            prefix = 0
            if o is not None and not same:
                prefix = next((i for i, (x, y) in enumerate(zip(s["_content"], o["_content"])) if x != y),
                              min(len(s["_content"]), len(o["_content"])))
            identity[f"{ref}~{other}:{s['session_id']}"] = {"identical": same, "common_prefix_chars": None if same else prefix}
            print(f"identity {ref} vs {other} {s['session_id']}: {'byte-identical' if same else f'DIFFERS after {prefix} chars'}")
    if a.json:
        for m in arms.values():
            for s in m["sessions"]:
                s.pop("_gaps"), s.pop("_content")
        with open(a.json, "w") as f:
            json.dump({"arms": arms, "identity": identity}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
