"""Compare served capture runs (spec_margin_capture.py output dirs) on their serving prompts only, with NVMe-stall
prompts flagged, so that one A/B pair can resolve a ~2 ms/token effect.

Why a separate tool: the old estimators (ab_compare.py, step_budget.py) skip a fixed 8 verifies, which leaves 8-14
warm-up/health verifies in the sample (a count that differs per run, at 75-125 ms of stream each), and take a median
over 8 prompts whose means differ 2x. On 2026-10-15 that turned A' vs A'' (identical code and env) into stream +5.3
ms/verify; serving-only means give +0.8. Separately, the SPCC mirror (nvme3n1, /mnt/nvme4) has 20-60 s read stalls
(await 25-61 ms against ~7, every mirror capped at ~0.55 GB/s because a row waits for its slowest third). A stall that
lands in serving decode raises that prompt's stream ms per stream layer 1.5-3x and its wait with it; it explained
base vs base-r2 (+6.5 -> -0.2 ms/token), A' vs D (+11.9 -> -0.5) and drip3 vs drip4 (+11.6 -> +0.1).

Per run:
  serving prompts: the last len(probe.json) sessions (a session is the verifies between >1 s gaps, i.e. a prefill);
  per prompt: ms/token = sum of verify-to-verify steps / the prompt's tokens; stream per verify = sum over layers of
    the device stamps' stream_end - stream_start; stream layers = layers whose stream exceeds 50 us (a kMissGpu lane);
    ms per stream layer, which tracks that record's NVMe read time (svc_record c), so it is the stall detector;
  stall: a prompt whose ms per stream layer exceeds --stall-factor x the run's median prompt (1.3 flagged every
    stalled prompt and no prompt of a clean run across the 2026-10-14/15 cores-ab runs);
  drive load: from the server log's shutdown line, each root's demand busy s; a root well above the others had a stall
    somewhere in the run (prefill included), even if no serving prompt is flagged.
Pair (first run vs each other): ms/token and stream/verify over all prompts and over prompts clean in both runs, with
a prompt-block bootstrap 95% CI of the clean difference (prompts resampled together in both arms).

Usage: serve_compare.py RUN_DIR RUN_DIR [RUN_DIR ...] [--stall-factor 1.3] [--json OUT]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys

STREAM_LAYER_NS = 50_000  # a layer with no kMissGpu lane still stamps a ~2 us stream


def _one(d: str, pattern: str) -> str:
    paths = glob.glob(os.path.join(d, pattern))
    if len(paths) != 1:
        raise SystemExit(f"{d}: expected one {pattern}, found {len(paths)}")
    return paths[0]


def gpu_offset(d: str) -> int:
    """%globaltimer minus CLOCK_MONOTONIC, from the draft-clock sample with the tightest bracket."""
    samples = json.load(open(_one(d, "events.*.draft-clock.json")))["samples"]
    s = min(samples, key=lambda s: s["after"] - s["before"])
    return s["gpu"] - (s["before"] + s["after"]) // 2


def load_verifies(d: str) -> list[tuple[int, int]]:
    out, b = [], None
    with open(_one(d, "events.*.verify.jsonl")) as f:
        f.readline()
        for line in f:
            e = json.loads(line)
            if e["event"] == "verify_begin":
                b = e["ns"]
            elif e["event"] == "verify_end" and b is not None:
                out.append((b, e["ns"]))
                b = None
    return out


def load_stamps(d: str, offset: int) -> list[dict]:
    out = []
    with open(_one(d, "events.*.device-stamps.jsonl")) as f:
        points = json.loads(f.readline())["points"]
        for line in f:
            r = json.loads(line)
            out.append({"seq": r["seq"], **{p: (r[p] - offset if r[p] else 0) for p in points}})
    return out


def load_reads(d: str) -> dict[int, int]:
    """Per record seq, the slowest group's ns inside read_rows (svc_record c)."""
    out: dict[int, int] = {}
    for path in glob.glob(os.path.join(d, "events.*.exl3-tier*.jsonl")):
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("event") == "svc_record":
                    out[e["seq"]] = max(out.get(e["seq"], 0), e["c"])
    return out


def drive_load(d: str) -> list[float] | None:
    path = os.path.join(d, "server.log")
    if not os.path.exists(path):
        return None
    line = None
    with open(path, errors="replace") as f:
        for ln in f:
            if "exl3 RAM miss drive load " in ln:
                line = ln
    if line is None:
        return None
    roots = json.loads(line[line.index("{"):])["roots"]
    return [r["demand_busy_ns"] / 1e9 for r in roots]


def prompts(d: str, stall_factor: float) -> list[dict]:
    verifies = load_verifies(d)
    session, s = [], 0
    for i, (b, _) in enumerate(verifies):
        if i and b - verifies[i - 1][1] > 1_000_000_000:
            s += 1
        session.append(s)
    probe = json.load(open(os.path.join(d, "probe.json")))
    ids = sorted(set(session))[-len(probe):]
    if len(ids) != len(probe):
        raise SystemExit(f"{d}: {len(ids)} sessions for {len(probe)} prompts")
    index = {sid: i for i, sid in enumerate(ids)}
    per = [{"verifies": 0, "tokens": len(p["tokens"]), "step_ms": 0.0, "wall_ms": 0.0, "stream_ms": 0.0,
            "wait_ms": 0.0, "stream_layers": 0, "read_ms": 0.0, "reads": 0} for p in probe]
    begins = [b for b, _ in verifies]
    for k, (b, e) in enumerate(verifies):
        i = index.get(session[k])
        if i is None:
            continue
        p = per[i]
        p["verifies"] += 1
        p["wall_ms"] += (e - b) / 1e6
        if k + 1 < len(verifies) and session[k + 1] == session[k]:
            p["step_ms"] += (begins[k + 1] - b) / 1e6
    reads = load_reads(d)
    for st in load_stamps(d, gpu_offset(d)):
        t = st["post_start"]
        if not t:
            continue
        k = _bisect(begins, t)
        if k < 0 or t > verifies[k][1] or session[k] not in index:
            continue
        p = per[index[session[k]]]
        stream = st["stream_end"] - st["stream_start"] if st["stream_end"] else 0
        p["stream_ms"] += stream / 1e6
        if st["wait_end"]:
            p["wait_ms"] += (st["wait_end"] - st["wait_start"]) / 1e6
        if stream > STREAM_LAYER_NS:
            p["stream_layers"] += 1
            if st["seq"] in reads:
                p["read_ms"] += reads[st["seq"]] / 1e6
                p["reads"] += 1
    for p in per:
        p["ms_per_stream_layer"] = p["stream_ms"] / p["stream_layers"] if p["stream_layers"] else 0.0
        p["read_ms_per_stream_layer"] = p["read_ms"] / p["reads"] if p["reads"] else 0.0
    med = statistics.median(p["ms_per_stream_layer"] for p in per)
    for p in per:
        p["stall"] = p["ms_per_stream_layer"] > stall_factor * med
    return per


def _bisect(xs: list[int], x: int) -> int:
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def totals(per: list[dict], keep: list[bool]) -> dict:
    ps = [p for p, k in zip(per, keep) if k]
    v = sum(p["verifies"] for p in ps)
    layers = sum(p["stream_layers"] for p in ps)
    return {
        "prompts": len(ps),
        "ms_per_token": sum(p["step_ms"] for p in ps) / sum(p["tokens"] for p in ps),
        "tokens_per_verify": sum(p["tokens"] for p in ps) / v,
        "wall_per_verify": sum(p["wall_ms"] for p in ps) / v,
        "wait_per_verify": sum(p["wait_ms"] for p in ps) / v,
        "stream_per_verify": sum(p["stream_ms"] for p in ps) / v,
        "stream_layers_per_verify": layers / v,
        "ms_per_stream_layer": sum(p["stream_ms"] for p in ps) / layers if layers else 0.0,
    }


def paired_ci(a: list[dict], b: list[dict], keep: list[bool], n: int = 4000) -> tuple[float, float]:
    idx = [i for i, k in enumerate(keep) if k]
    rng = random.Random(0)
    diffs = []
    for _ in range(n):
        pick = [rng.choice(idx) for _ in idx]
        ka = [0] * len(a)
        for i in pick:
            ka[i] += 1
        mt = lambda per: sum(per[i]["step_ms"] * c for i, c in enumerate(ka)) / sum(per[i]["tokens"] * c for i, c in enumerate(ka))
        diffs.append(mt(b) - mt(a))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stall-factor", type=float, default=1.3)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    cols = ("ms_per_token", "tokens_per_verify", "wall_per_verify", "wait_per_verify", "stream_per_verify",
            "stream_layers_per_verify", "ms_per_stream_layer")
    for d in a.runs:
        per = prompts(d, a.stall_factor)
        busy = drive_load(d)
        out[d] = {"prompts": per, "all": totals(per, [True] * len(per)), "drive_busy_s": busy}
        name = os.path.basename(os.path.normpath(d))
        print(f"== {name}")
        print("   prompt  verifies tokens  ms/token  stream/v  layers/v  ms/layer  read/layer  wait/v  stall")
        for i, p in enumerate(per):
            print(f"   {i:6d} {p['verifies']:9d} {p['tokens']:6d} {p['step_ms'] / p['tokens']:9.2f} "
                  f"{p['stream_ms'] / p['verifies']:9.1f} {p['stream_layers'] / p['verifies']:9.2f} "
                  f"{p['ms_per_stream_layer']:9.2f} {p['read_ms_per_stream_layer']:11.2f} "
                  f"{p['wait_ms'] / p['verifies']:7.1f}  {'STALL' if p['stall'] else ''}")
        t = out[d]["all"]
        print("   all: " + "  ".join(f"{c} {t[c]:.2f}" for c in cols))
        if busy:
            med = statistics.median(busy)
            print("   drive load demand busy s per root: " + ", ".join(f"{x:.1f}" for x in busy) +
                  (f"  <- root {busy.index(max(busy))} +{max(busy) - med:.1f} s over the median: a stall in this run"
                   if max(busy) - med > 0.05 * med else ""))
    first = a.runs[0]
    pa = out[first]["prompts"]
    for d in a.runs[1:]:
        pb = out[d]["prompts"]
        if len(pa) != len(pb):
            print(f"!! {d}: {len(pb)} prompts vs {len(pa)}; not paired")
            continue
        clean = [not (x["stall"] or y["stall"]) for x, y in zip(pa, pb)]
        ta, tb = totals(pa, [True] * len(pa)), totals(pb, [True] * len(pb))
        ca, cb = totals(pa, clean), totals(pb, clean)
        lo, hi = paired_ci(pa, pb, clean)
        print(f"== {os.path.basename(os.path.normpath(d))} - {os.path.basename(os.path.normpath(first))}")
        for c in cols:
            print(f"   {c:26s} all {tb[c] - ta[c]:+7.2f}   clean ({sum(clean)} prompts) {cb[c] - ca[c]:+7.2f}")
        print(f"   ms/token clean difference, prompt-block bootstrap 95% CI: {lo:+.2f} .. {hi:+.2f}")
        out[d]["pair"] = {"clean_prompts": clean, "all": {c: tb[c] - ta[c] for c in cols},
                          "clean": {c: cb[c] - ca[c] for c in cols}, "ms_per_token_ci": [lo, hi]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
