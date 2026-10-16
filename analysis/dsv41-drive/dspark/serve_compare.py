"""Compare served capture runs (spec_margin_capture.py output dirs) on their serving prompts only, with NVMe-stall and
CPU-slow prompts flagged, so that one A/B pair can resolve a ~2 ms/token effect. Replaces the run-dir ab_compare.py
(and step_budget.py's medians) for A/B verdicts.

Why: the old estimators skip a fixed 8 verifies, which leaves 8-14 warm-up/health verifies in the sample (a count
that differs per run, at 75-125 ms of stream each); ab_compare's ms/token also sums those warm-up steps over serving
tokens only; and a median over 8 prompts whose means differ 2x is unstable (+-7 ms/verify). On 2026-10-15 that turned
A' vs A'' (identical code and env) into stream +5.3 ms/verify; serving-only means give +0.8. Separately, the SPCC mirror
(nvme3n1, /mnt/nvme4) has 20-60 s read stalls (await 25-61 ms against ~7; every mirror capped at ~0.55 GB/s because a
row waits for its slowest third). A stall in serving decode raises that prompt's stream ms per stream layer 1.5-3x and
its wait with it: it explained base vs base-r2 (+6.5 -> -0.2 ms/token) and A' vs D (+11.9 -> -0.5).

Per run:
  serving prompts: the last len(probe.json) sessions (a session is the verifies between >1 s gaps, i.e. a prefill);
  per prompt: ms/token = sum of verify-to-verify steps / the prompt's tokens; stream per verify = sum over layers of
    stream_end - stream_start (device stamps); stream layers = layers whose stream exceeds 50 us (a kMissGpu lane);
    ms per stream layer tracks the record's NVMe read time (svc_record c, also reported);
    CPU ms per lane per team = cpu_end - cpu_start over the job's lanes (cpu_shape b), jobs started in the prompt;
  flags, each relative to the run's own median prompt:
    STALL: ms per stream layer > --stall-factor x median (1.3 flagged every stalled prompt and no prompt of a clean
      run in cores-ab-20261014); corroborated by diskstats from <run>.memsample.jsonl, if present: the worst drive's
      read await in the 5 s samples overlapping the prompt, over that drive's median await in the run (`disk x`);
    CPUSLOW: either team's ms per lane > --cpu-factor x median; corroborated by the memsample's swap-in and major
      fault rate over the prompt against the run's median rate (`swap x`, system-wide counters) and node 0's lowest
      MemFree;
  drive load: each root's demand busy s from the server log's shutdown line; a root >5% over the median root had a
    stall somewhere in the run (prefill included), even if no serving prompt is flagged.
Pair (first run vs each other): per-prompt differences, then ms/token and the components over all prompts and over
prompts with no flag in either run, with a prompt-block bootstrap 95% CI of the clean ms/token difference.

Usage: serve_compare.py RUN_DIR RUN_DIR [RUN_DIR ...] [--stall-factor 1.3] [--cpu-factor 1.2] [--json OUT]
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


def _bisect(xs: list[int], x: int) -> int:
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


def gpu_offset(d: str) -> int:
    """%globaltimer minus CLOCK_MONOTONIC, from the draft-clock sample with the tightest bracket."""
    samples = json.load(open(_one(d, "events.*.draft-clock.json")))["samples"]
    s = min(samples, key=lambda s: s["after"] - s["before"])
    return s["gpu"] - (s["before"] + s["after"]) // 2


def load_verifies(d: str) -> tuple[list[tuple[int, int]], float]:
    """Verify (begin, end) CLOCK_MONOTONIC ns, and the epoch minus monotonic offset in s."""
    out, b = [], None
    with open(_one(d, "events.*.verify.jsonl")) as f:
        h = json.loads(f.readline())
        for line in f:
            e = json.loads(line)
            if e["event"] == "verify_begin":
                b = e["ns"]
            elif e["event"] == "verify_end" and b is not None:
                out.append((b, e["ns"]))
                b = None
    return out, (h["epoch_ns"] - h["monotonic_ns"]) / 1e9


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


def load_cpu_jobs(d: str) -> dict[int, list[tuple[int, float]]]:
    """Per CPU team, (start ns, ms per lane) of each completed job."""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "events.*.exl3-cpu-exp*.jsonl"))):
        team = int(path.split("cpu-exp")[1].split(".")[0])
        jobs: dict = {}
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                k = e.get("event")
                if k in ("cpu_start", "cpu_end", "cpu_shape"):
                    jobs.setdefault((e["row"], e["seq"]), {})[k] = e
        out[team] = sorted((j["cpu_start"]["ns"], (j["cpu_end"]["ns"] - j["cpu_start"]["ns"]) / 1e6 / max(j["cpu_shape"]["b"], 1))
                           for j in jobs.values() if all(k in j for k in ("cpu_start", "cpu_end", "cpu_shape")))
    return out


def load_memsample(d: str) -> list[dict]:
    path = os.path.realpath(d) + ".memsample.jsonl"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [s for s in (json.loads(line) for line in f) if "vmstat" in s]


def memsample_intervals(samples: list[dict]) -> list[dict]:
    """Consecutive memsample pairs: epoch window, per nvme read await ms (None without reads), swap-in + major fault
    rate per s, node 0 MemFree MiB at the end."""
    out = []
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        await_ms = {}
        for dev, s1 in b.get("disk", {}).items():
            s0 = a.get("disk", {}).get(dev)
            if dev.startswith("nvme") and s0 is not None:
                reads = s1[0] - s0[0]
                await_ms[dev] = (s1[2] - s0[2]) / reads if reads > 1000 else None
        fault = sum(b["vmstat"][k] - a["vmstat"][k] for k in ("pswpin", "pgmajfault")) / dt
        out.append({"t0": a["t"], "t1": b["t"], "await": await_ms, "fault": fault,
                    "node0_free_mib": b["nodes"]["0"]["MemFree"] / 1024})
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


def _median_ratio(x: float, xs: list[float]) -> float:
    m = statistics.median(xs) if xs else 0.0
    return x / m if m else 0.0


def prompts(d: str, stall_factor: float, cpu_factor: float) -> list[dict]:
    verifies, to_epoch = load_verifies(d)
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
            "wait_ms": 0.0, "stream_layers": 0, "read_ms": 0.0, "reads": 0, "t0": None, "t1": None} for p in probe]
    begins = [b for b, _ in verifies]
    for k, (b, e) in enumerate(verifies):
        i = index.get(session[k])
        if i is None:
            continue
        p = per[i]
        p["verifies"] += 1
        p["wall_ms"] += (e - b) / 1e6
        p["t0"] = b if p["t0"] is None else p["t0"]
        p["t1"] = e
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
    cpu = load_cpu_jobs(d)
    for p in per:
        p["ms_per_stream_layer"] = p["stream_ms"] / p["stream_layers"] if p["stream_layers"] else 0.0
        p["read_ms_per_stream_layer"] = p["read_ms"] / p["reads"] if p["reads"] else 0.0
        p["cpu_ms_per_lane"] = {}
        for team, jobs in cpu.items():
            lane = [ms for t, ms in jobs if p["t0"] <= t <= p["t1"]]
            p["cpu_ms_per_lane"][team] = statistics.mean(lane) if lane else 0.0
    # Flags, each against the run's own median prompt.
    for p in per:
        p["stall"] = _median_ratio(p["ms_per_stream_layer"], [q["ms_per_stream_layer"] for q in per]) > stall_factor
        p["cpu_ratio"] = max((_median_ratio(p["cpu_ms_per_lane"][t], [q["cpu_ms_per_lane"][t] for q in per]) for t in cpu),
                             default=0.0)
        p["cpu_slow"] = p["cpu_ratio"] > cpu_factor
    # Corroboration from the memsample (5 s samples): intervals overlapping the prompt's decode.
    intervals = memsample_intervals(load_memsample(d))
    if intervals:
        dev_await = {}
        for iv in intervals:
            for dev, aw in iv["await"].items():
                if aw is not None:
                    dev_await.setdefault(dev, []).append(aw)
        med_await = {dev: statistics.median(xs) for dev, xs in dev_await.items()}
        med_fault = statistics.median(iv["fault"] for iv in intervals)
        for p in per:
            a, b = p["t0"] / 1e9 + to_epoch, p["t1"] / 1e9 + to_epoch
            over = [iv for iv in intervals if iv["t0"] < b and a < iv["t1"]]
            p["disk_ratio"] = max((aw / med_await[dev] for iv in over for dev, aw in iv["await"].items()
                                   if aw is not None and med_await.get(dev)), default=None)
            fault = max((iv["fault"] for iv in over), default=None)
            p["fault_ratio"] = None if fault is None else fault / max(med_fault, 1.0)
            p["node0_free_mib"] = min((iv["node0_free_mib"] for iv in over), default=None)
    else:
        for p in per:
            p["disk_ratio"] = p["fault_ratio"] = p["node0_free_mib"] = None
    return per


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
    """Prompt-block bootstrap 95% CI of b's ms/token minus a's over the kept prompts."""
    idx = [i for i, k in enumerate(keep) if k]
    rng = random.Random(0)

    def mst(per, w):
        return sum(per[i]["step_ms"] * c for i, c in w.items()) / sum(per[i]["tokens"] * c for i, c in w.items())

    diffs = []
    for _ in range(n):
        w: dict[int, int] = {}
        for _ in idx:
            i = rng.choice(idx)
            w[i] = w.get(i, 0) + 1
        diffs.append(mst(b, w) - mst(a, w))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def flags(p: dict) -> str:
    out = []
    if p["stall"]:
        out.append("STALL" + (f"(disk x{p['disk_ratio']:.1f})" if p["disk_ratio"] else ""))
    if p["cpu_slow"]:
        out.append(f"CPUSLOW x{p['cpu_ratio']:.2f}" + (f"(swap x{p['fault_ratio']:.0f})" if p["fault_ratio"] else ""))
    return " ".join(out)


def _name(d: str) -> str:
    return os.path.basename(os.path.normpath(d))


COLS = ("ms_per_token", "tokens_per_verify", "wall_per_verify", "wait_per_verify", "stream_per_verify",
        "stream_layers_per_verify", "ms_per_stream_layer")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stall-factor", type=float, default=1.3)
    ap.add_argument("--cpu-factor", type=float, default=1.2)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for d in a.runs:
        per = prompts(d, a.stall_factor, a.cpu_factor)
        busy = drive_load(d)
        out[d] = {"prompts": per, "all": totals(per, [True] * len(per)), "drive_busy_s": busy}
        print(f"== {_name(d)}")
        print("   prompt verifies tokens ms/token stream/v layers/v ms/layer read/layer  wait/v  cpu ms/lane  "
              "disk x  swap x  node0 MiB  flags")
        for i, p in enumerate(per):
            cpu = "/".join(f"{v:.3f}" for _, v in sorted(p["cpu_ms_per_lane"].items()))
            opt = lambda v, f: f"{v:{f}}" if v is not None else "-"
            print(f"   {i:6d} {p['verifies']:8d} {p['tokens']:6d} {p['step_ms'] / p['tokens']:8.2f} "
                  f"{p['stream_ms'] / p['verifies']:8.1f} {p['stream_layers'] / p['verifies']:8.2f} "
                  f"{p['ms_per_stream_layer']:8.2f} {p['read_ms_per_stream_layer']:10.2f} {p['wait_ms'] / p['verifies']:7.1f}"
                  f"  {cpu:>11s} {opt(p['disk_ratio'], '6.1f'):>7s} {opt(p['fault_ratio'], '6.1f'):>7s}"
                  f" {opt(p['node0_free_mib'], '9.0f'):>10s}  {flags(p)}")
        t = out[d]["all"]
        print("   all: " + "  ".join(f"{c} {t[c]:.2f}" for c in COLS))
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
            print(f"!! {_name(d)}: {len(pb)} prompts vs {len(pa)}; not paired")
            continue
        clean = [not (x["stall"] or x["cpu_slow"] or y["stall"] or y["cpu_slow"]) for x, y in zip(pa, pb)]
        print(f"== {_name(d)} - {_name(first)}")
        print("   prompt  d ms/token  d stream/v  d wait/v  d tok/verify  flags (first | second)")
        for i, (x, y) in enumerate(zip(pa, pb)):
            print(f"   {i:6d} {y['step_ms'] / y['tokens'] - x['step_ms'] / x['tokens']:+11.2f}"
                  f" {y['stream_ms'] / y['verifies'] - x['stream_ms'] / x['verifies']:+11.1f}"
                  f" {y['wait_ms'] / y['verifies'] - x['wait_ms'] / x['verifies']:+9.1f}"
                  f" {y['tokens'] / y['verifies'] - x['tokens'] / x['verifies']:+13.2f}  {flags(x)} | {flags(y)}")
        ta, tb = totals(pa, [True] * len(pa)), totals(pb, [True] * len(pb))
        ca, cb = totals(pa, clean), totals(pb, clean)
        lo_all, hi_all = paired_ci(pa, pb, [True] * len(pa))
        lo, hi = paired_ci(pa, pb, clean)
        for c in COLS:
            print(f"   {c:26s} all {tb[c] - ta[c]:+7.2f}   clean ({sum(clean)} prompts) {cb[c] - ca[c]:+7.2f}")
        print(f"   ms/token 95% CI (prompt-block bootstrap): all {lo_all:+.2f} .. {hi_all:+.2f}   clean {lo:+.2f} .. {hi:+.2f}")
        out[d]["pair"] = {"against": first, "clean_prompts": clean, "all": {c: tb[c] - ta[c] for c in COLS},
                          "clean": {c: cb[c] - ca[c] for c in COLS}, "ms_per_token_ci_all": [lo_all, hi_all],
                          "ms_per_token_ci_clean": [lo, hi]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
