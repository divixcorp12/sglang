"""W (SPCC weighted 0.9) vs U (equal weights) decode pairs: per-arm numbers, paired deltas, byte identity.

Arms are named <kind><pair> (W1, U1, ...); a pair is the W and U arm sharing a number. Per arm, from mirror3_report:
pooled ms/token (total decode s / total decode tokens) and per session, TTFT, SM clocks, the RAM-miss counters (server
lifetime: warm-up, prefill and timed set) and each mirror drive's read split over the timed window (server_ready to
the last session's boundary sample, prefill included). Added here: each drive's mean device read time per request
(delta ms_reading / delta reads, /proc/diskstats) and RAM misses per generated token over the server's life (the
counters' only scope; every generated token, warm-up included, is in the denominator).

Usage: weight_pair_report.py <out_dir> <device>... -- <arm>=<run_dir>...
Writes <out_dir>/pairs-report.json and prints a summary.
"""

import json
import statistics
import sys
from pathlib import Path

import mirror3_report as m


def pairs(names):
    """{pair: (W arm, U arm)} for every number with both kinds; the order they ran in is kept by the caller."""
    by = {}
    for n in names:
        by.setdefault(n[1:], {})[n[0]] = n
    return {p: (d["W"], d["U"]) for p, d in by.items() if "W" in d and "U" in d}


def spread(values):
    return {"n": len(values), "mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "values": values}


def read_await(samples, start, end, devices):
    inside = [s for s in samples if start <= s["mono"] <= end]
    a, b = inside[0], inside[-1]
    out = {}
    for d in devices:
        reads = b[d]["reads"] - a[d]["reads"]
        out[d] = (b[d]["ms_reading"] - a[d]["ms_reading"]) / reads if reads else None
    return out


def generated_tokens(run_dir):
    run_dir = Path(run_dir)
    total = 0
    for path in [run_dir / "results.jsonl", *sorted(run_dir.glob("results-warmup-*.jsonl"))]:
        total += sum(r.get("completion_tokens") or 0 for r in m.load_jsonl(path) if "error" not in r)
    return total


def arm(out_dir, name, run_dir, devices):
    start, end = m.timed_window(run_dir)
    samples = m.load_jsonl(Path(out_dir) / f"{name}-diskstats.jsonl")
    rm = m.ram_miss(run_dir)
    tokens = generated_tokens(run_dir)
    served = (rm["thread"] or {}).get("served")
    env = json.loads((Path(run_dir) / "server-env-actual.json").read_text()) if (Path(run_dir) / "server-env-actual.json").exists() else {}
    return {
        "run_dir": run_dir,
        "weights_env": env.get("SGLANG_MOE_EXPERT_MIRROR_WEIGHTS") if isinstance(env, dict) else None,
        "decode": m.decode(run_dir),
        "clocks": {"timed_window": m.clock_summary(Path(out_dir) / f"{name}-clocks.csv", *m.timed_window_utc(run_dir)),
                   "session_start_end_mhz": m.session_clocks(run_dir)},
        "ram_miss": rm,
        "generated_tokens_lifetime": tokens,
        "ram_misses_served_per_generated_token": served / tokens if served is not None and tokens else None,
        "disk": m.drive_split(samples, start, end, devices),
        "read_ms_per_request": read_await(samples, start, end, devices),
    }


def main(argv):
    split = argv.index("--")
    out_dir, devices = argv[0], argv[1:split]
    runs = dict(kv.split("=", 1) for kv in argv[split + 1:])
    arms = {n: arm(out_dir, n, r, devices) for n, r in runs.items()}
    ref = next(n for n in runs if n.startswith("U"))
    identical = {n: m.identity(runs[ref], r) for n, r in runs.items() if n != ref}
    pooled = {n: a["decode"]["pooled_ms_per_token"] for n, a in arms.items()}
    pair_map = pairs(runs)
    deltas = {p: pooled[w] - pooled[u] for p, (w, u) in pair_map.items()}
    session_deltas = {}
    for p, (w, u) in pair_map.items():
        ws, us = arms[w]["decode"]["sessions"], arms[u]["decode"]["sessions"]
        session_deltas[p] = {s: ws[s]["pooled_ms_per_token"] - us[s]["pooled_ms_per_token"] for s in ws if s in us}
    out = {
        "arms": arms,
        "pooled_ms_per_token": pooled,
        "W": spread([pooled[n] for n in pooled if n.startswith("W")]),
        "U": spread([pooled[n] for n in pooled if n.startswith("U")]),
        "pair_delta_W_minus_U": spread(list(deltas.values())) if deltas else None,
        "pair_deltas": deltas,
        "pair_session_deltas": session_deltas,
        "identity_ref": ref,
        "identical": identical,
        "all_identical": all(all(v.values()) for v in identical.values()),
    }
    Path(out_dir, "pairs-report.json").write_text(json.dumps(out, indent=2, default=str))
    for n, a in arms.items():
        d = a["disk"]["drives"]
        print(n, f"pooled {pooled[n]:.2f}",
              "sessions", {s: round(v["pooled_ms_per_token"], 2) for s, v in a["decode"]["sessions"].items()},
              "share", {k: round(v["share_pct"], 1) for k, v in d.items()},
              "await", {k: round(v, 3) if v else v for k, v in a["read_ms_per_request"].items()},
              "miss/tok", round(a["ram_misses_served_per_generated_token"] or 0, 2))
    print("pair deltas W-U", {p: round(v, 2) for p, v in deltas.items()}, "all identical", out["all_identical"])
    return 0 if out["all_identical"] else "an arm's output differs"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
