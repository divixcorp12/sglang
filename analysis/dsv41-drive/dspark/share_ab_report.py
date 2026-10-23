"""The served prefill-share A/B (SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SHARE) against lane_eval.py's predictions: per run, the
decode tier and cost per verify and per prompt's prefill, from the job trace and the route log.

Per run (clean serving prompts by serve_compare's flags, or the pair's with --pair):
  decode: NVMe-cold GPU lanes, forced CPU misses, stream and wait per verify, overall and by verify position after the
    prompt's prefill (0-4, 5-14, 15+);
  prefill, per serving prompt: its window in the route log (first to last extend line), the time from its first line to
    the first verify (a TTFT proxy on the server's clock), the demand reads (sum of ram_miss), their read time, and the
    prefill counters' pinned-tier reads (file_misses) and hits;
  the nvme3n1 read await in each prefill window (lane_head_ab_report.window_await), flagged PREFILL-STALL above 2x the
    run's median await or 2x its median demand read ms per row (PREFILL_CAPTURE_PLAN.md section 2).

Usage: share_ab_report.py OFF_DIR ON_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_tier_replay as ltr  # noqa: E402

STALL_FACTOR = 2.0


def nvme3n1_samples(d: str) -> list[tuple[float, int, int]]:
    path = os.path.realpath(d) + ".nvme3n1.samples"
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14:
                continue
            # epoch, then the /proc/diskstats fields: major minor name reads merged sectors ms_reading ...
            out.append((float(parts[0]), int(parts[4]), int(parts[7])))
    return out


def window_await(samples, t0: float, t1: float):
    before = [s for s in samples if s[0] <= t0]
    after = [s for s in samples if s[0] >= t1]
    if not before or not after:
        return None, 0
    a, b = before[-1], after[0]
    reads = b[1] - a[1]
    return ((b[2] - a[2]) / reads if reads else None), reads


def prefill_counters(d: str) -> dict:
    last = None
    with open(os.path.join(d, "metrics.jsonl")) as f:
        for line in f:
            last = line
    c = json.loads(last)["counters"]["prefill"]
    tot = collections.Counter()
    for layer in c.values():
        for k in ("file_misses", "pinned_hits", "miss_rows", "routed_miss_rows", "evictions"):
            tot[k] += layer.get(k) or 0
    return dict(tot)


def run_report(d: str, clean: list[bool] | None) -> dict:
    run = ltr.Run(d)
    if clean is not None:
        run.clean = clean
    first = {}
    for v in range(len(run.verifies)):
        first.setdefault(run.prompt_of[v], v)
    by = collections.defaultdict(collections.Counter)
    for r in run.serving():
        pos = r.verify - first[run.prompt_of[r.verify]]
        b = "0-4" if pos < 5 else "5-14" if pos < 15 else "15+"
        for key in ("all", b):
            by[key]["stream"] += r.stream
            by[key]["wait"] += r.wait
            by[key]["gpu_nvme"] += r.gpu_nvme
            by[key]["cpu_forced"] += ltr.cpu_misses(r)
            by[key]["verifies_x_layers"] += 1
    verifies = collections.Counter()
    for v in range(len(run.verifies)):
        p = run.prompt_of[v]
        if p >= 0 and run.clean[p]:
            pos = v - first[p]
            verifies["all"] += 1
            verifies["0-4" if pos < 5 else "5-14" if pos < 15 else "15+"] += 1
    decode = {k: {m: round(x / max(1, verifies[k]), 2) for m, x in c.items() if m != "verifies_x_layers"}
              | {"verifies": verifies[k]} for k, c in by.items()}
    for k in decode:
        decode[k]["stream_plus_wait"] = round(decode[k]["stream"] + decode[k]["wait"], 2)
    # prefills: the route log's extend lines by forward_pass_id; the serving prompts are the last len(prompts)
    eager = ltr.eager_lines(d)
    fp = collections.OrderedDict()
    for x in eager:
        if x["phase"] != "extend":
            continue
        f = fp.setdefault(x["forward_pass_id"], {"t0": x["t"], "t1": x["t"], "tokens": x["tokens"], "ram_miss": 0,
                                                 "read_ms": 0.0, "vram_miss": 0})
        f["t1"] = x["t"]
        f["ram_miss"] += x["ram_miss"]
        f["read_ms"] += x["read_ms"]
        f["vram_miss"] += x["vram_miss"]
        if x["layer"] == 0:
            f["tokens"] = x["tokens"]
    serving = list(fp.items())[-len(run.prompts):]
    begins = [b for b, _ in run.verifies]
    samples = nvme3n1_samples(d)
    # the job trace's CLOCK_MONOTONIC to epoch, for the diskstats samples
    import glob

    with open(glob.glob(os.path.join(d, "events.*.verify.jsonl"))[0]) as f:
        head = json.loads(f.readline())
    to_epoch = head["epoch_ns"] / 1e9 - head["monotonic_ns"] / 1e9
    prompts = []
    for k, (fpid, f) in enumerate(serving):
        nxt = [b for b in begins if b / 1e9 > f["t1"]]
        first_verify = nxt[0] / 1e9 if nxt else None
        aw, reads = window_await(samples, f["t0"] + to_epoch, f["t1"] + to_epoch)
        prompts.append({"prompt": k, "fpid": fpid, "tokens": f["tokens"], "prefill_s": round(f["t1"] - f["t0"], 2),
                        "to_first_verify_s": round(first_verify - f["t0"], 2) if first_verify else None,
                        "demand_reads": f["ram_miss"], "demand_read_s": round(f["read_ms"] / 1e3, 2),
                        "ms_per_demand_row": round(f["read_ms"] / max(1, f["ram_miss"]), 3),
                        "nvme3n1_await_ms": None if aw is None else round(aw, 1), "nvme3n1_reads": reads,
                        "clean": run.clean[k] if k < len(run.clean) else None})
    awaits = [p["nvme3n1_await_ms"] for p in prompts if p["nvme3n1_await_ms"] is not None]
    rates = [p["ms_per_demand_row"] for p in prompts]
    med_aw = statistics.median(awaits) if awaits else None
    med_rate = statistics.median(rates)
    for p in prompts:
        p["prefill_stall"] = bool((med_aw and p["nvme3n1_await_ms"] and p["nvme3n1_await_ms"] > STALL_FACTOR * med_aw)
                                  or p["ms_per_demand_row"] > STALL_FACTOR * med_rate)
    return {"dir": d, "decode": decode, "prompts": prompts, "prefill_counters": prefill_counters(d),
            "prompt_flags": [{"stall": p["stall"], "cpu_slow": p["cpu_slow"]} for p in run.prompts]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--json")
    a = ap.parse_args()
    off, on = ltr.Run(a.off), ltr.Run(a.on)
    pair = [x and y for x, y in zip(off.clean, on.clean)]
    res = {"pair_clean": pair, "off": run_report(a.off, pair), "on": run_report(a.on, pair)}
    d = {}
    for k in res["off"]["decode"]:
        if k in res["on"]["decode"]:
            d[k] = {m: round(res["on"]["decode"][k][m] - res["off"]["decode"][k][m], 2)
                    for m in ("gpu_nvme", "cpu_forced", "stream", "wait", "stream_plus_wait")}
    res["delta_decode"] = d
    res["delta_prompts"] = [
        {"prompt": p["prompt"], "prefill_s": round(q["prefill_s"] - p["prefill_s"], 2),
         "to_first_verify_s": (round(q["to_first_verify_s"] - p["to_first_verify_s"], 2)
                               if p["to_first_verify_s"] and q["to_first_verify_s"] else None),
         "demand_reads": q["demand_reads"] - p["demand_reads"],
         "stall_either": p["prefill_stall"] or q["prefill_stall"]}
        for p, q in zip(res["off"]["prompts"], res["on"]["prompts"])]
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
