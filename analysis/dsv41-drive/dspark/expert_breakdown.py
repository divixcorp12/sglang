"""Where every routed expert of a DSpark verify was served from and computed, per MoE layer, from a traced capture.

Inputs (one spec_margin_capture.py dir, run without ``--no-verify-split``):

* the InstrBuild job trace ``events.<pid>.*.jsonl``. Per record (row, gen), each NUMA group's ``lane_kind`` (a = expert,
  b = lane kind | 256 when spilled, c = lane) and ``svc_routed`` (a = the group's routed experts, b = all routed, c = all
  lanes); ``pool_point`` (a GPU miss streamed from the prefetch pool), ``spec_use`` (a CPU miss swapped in from it),
  ``miss_land`` (a read miss: a = expert, b = landed ns, c & 4 = CPU miss), ``drip_use``, ``copy_submit`` / ``copy_issue``
  / ``copy_dma_observed`` / ``copy_cpu_observed`` / ``gate_open``, and the CPU teams' ``cpu_submit`` / ``cpu_end``
  (a = part: 0 the RAM hits' job, 1 a miss job). Background: ``spec_land`` (NVMe prefetch into the pool), ``drip_land``
  (a drip copied to VRAM scratch), ``delta_publish`` (c = experts promoted into the RAM tier).
* ``device-stamps`` (post / S / gate-wait spans per record seq, globaltimer), mapped to CLOCK_MONOTONIC through the
  tightest ``draft-clock`` sample; ``verify.jsonl`` (verify windows); ``stages.jsonl`` ``graph_routes`` (each graph
  forward's routes per layer, joined to the job trace's forwards by layer_misses.match_forwards on the lane experts).

A routed expert is in one category: ``vram`` (resident in the VRAM hot cache: routed, no lane), ``drip`` (HIT_SCRATCH),
``sm`` (HIT_SM, C1 pulls it from RAM), ``dma`` (HIT_COPY), ``pool`` (MISS_GPU served from a landed prefetch),
``nvme_gpu`` (MISS_GPU read from NVMe, then S), ``cpu_ram`` (HIT_CPU), ``cpu_pool`` (MISS_CPU swapped in from the pool),
``cpu_nvme`` (MISS_CPU read from NVMe). Unique experts count each once per layer; pairs weight each by the verify tokens
that route it.

A layer's last dependency is what ended its wait: ``cpu_hit`` / ``cpu_miss`` (the last CPU job to end held the gate;
``cpu_miss`` split into ``:read`` when its row's NVMe landing came later than its compute and queue, else
``:compute``), ``dma``, ``host_lag`` (the gate opened over 200 us after both sides were done), ``nvme_s`` (S waited on
NVMe pieces; no gate wait), ``sm_pull`` (C1 over 50 us, nothing else), ``none``. Traced times carry the trace's own
overhead: report counts and shares, not milliseconds.

Usage: expert_breakdown.py CAPTURE_DIR [--skip 8] [--json OUT]   (default OUT: CAPTURE_DIR/expert_breakdown.json)
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_misses  # noqa: E402

KIND = {1: "dma", 2: "sm", 3: "cpu_ram", 4: "nvme_gpu", 5: "cpu_nvme", 6: "drip"}
CATS = ("vram", "drip", "sm", "dma", "pool", "nvme_gpu", "cpu_ram", "cpu_pool", "cpu_nvme")
GPU_CATS = ("vram", "drip", "sm", "dma", "pool", "nvme_gpu")
H2D_CATS = ("sm", "dma", "pool", "nvme_gpu")
WAIT_NS = 50_000  # a span under this is not a wait; arbitrary, the trace's own jitter is ~10 us
LAG_NS = 200_000
SPILL = 256


def load_events(d: str) -> dict:
    by = collections.defaultdict(list)
    dropped = 0
    for path in glob.glob(os.path.join(d, "events.*.jsonl")):
        if path.endswith((".device-stamps.jsonl", ".verify.jsonl")):
            continue
        team = int(path.split("cpu-exp")[1].split(".")[0]) if "cpu-exp" in path else None
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if "dropped" in e:
                    dropped += int(e["dropped"])
                elif "event" in e:
                    if team is not None:
                        e["team"] = team
                    by[e["event"]].append(e)
    if dropped:
        raise ValueError(f"{dropped} job-trace events dropped (raise job.capacity of SGLANG_EXPERT_STREAM_TRACE)")
    if not by["svc_routed"]:
        raise ValueError("no svc_routed events: the capture predates the lane_kind / svc_routed instrumentation")
    return by


def gpu_offset(d: str) -> int:
    (path,) = glob.glob(os.path.join(d, "events.*.draft-clock.json"))
    with open(path) as f:
        s = min(json.load(f)["samples"], key=lambda s: s["after"] - s["before"])
    return s["gpu"] - (s["before"] + s["after"]) // 2


def load_stamps(d: str, offset: int) -> dict:
    (path,) = glob.glob(os.path.join(d, "events.*.device-stamps.jsonl"))
    out = {}
    with open(path) as f:
        points = json.loads(f.readline())["points"]
        for line in f:
            r = json.loads(line)
            out[r["seq"]] = {p: (r[p] - offset if r[p] else 0) for p in points}
    return out


def load_verifies(d: str) -> list[tuple[int, int]]:
    (path,) = glob.glob(os.path.join(d, "events.*.verify.jsonl"))
    begins, ends = [], []
    with open(path) as f:
        f.readline()
        for e in map(json.loads, f):
            (begins if e["event"] == "verify_begin" else ends).append(e["ns"])
    return list(zip(begins, ends))


def build_records(ev: dict) -> dict:
    rec = collections.defaultdict(lambda: {
        "lanes": {}, "groups": collections.defaultdict(dict), "pool": set(), "spec": set(), "lands": [],
        "routed": None, "lane_count": None, "seq": None, "start": None, "gate": 0})
    for e in ev["svc_routed"]:
        r = rec[(e["row"], e["gen"])]
        r["routed"], r["lane_count"], r["seq"] = e["b"], e["c"], e["seq"]
    for e in ev["svc_record"]:
        r = rec[(e["row"], e["gen"])]
        start = e["ns"] - e["a"]
        r["start"] = start if r["start"] is None else min(r["start"], start)
    for e in ev["lane_kind"]:
        rec[(e["row"], e["gen"])]["lanes"][e["a"]] = e["b"]
    for e in ev["pool_point"]:
        rec[(e["row"], e["gen"])]["pool"].add(e["a"])
    for e in ev["spec_use"]:
        rec[(e["row"], e["gen"])]["spec"].add(e["a"])
    for e in ev["miss_land"]:
        rec[(e["row"], e["gen"])]["lands"].append((e["a"], e["b"], bool(e["c"] & 4)))
    for e in ev["copy_submit"]:
        g = rec[(e["row"], e["gen"])]["groups"][e["group"]]
        g.update(submit=e["ns"], cpu_first=e["seq"], cpu_last=e["a"], cpu_mask=e["c"], late_cpu=e["b"])
    for e in ev["copy_issue"]:
        rec[(e["row"], e["gen"])]["groups"][e["group"]].update(dma_bytes=e["a"], dma_lanes=e["b"])
    for kind, key in (("copy_dma_observed", "dma"), ("copy_cpu_observed", "cpu")):
        for e in ev[kind]:
            rec[(e["row"], e["gen"])]["groups"][e["group"]][key] = e["ns"]
    for e in ev["gate_open"]:
        rec[(e["row"], e["gen"])]["gate"] = e["ns"]
    return {k: r for k, r in rec.items() if r["routed"] is not None}


def categorize(r: dict) -> dict:
    """Each routed expert's category, by expert (vram for routed experts without a lane)."""
    cats = {}
    for expert, b in r["lanes"].items():
        cat = KIND[b & 0xFF]
        if cat == "nvme_gpu" and expert in r["pool"]:
            cat = "pool"
        elif cat == "cpu_nvme" and expert in r["spec"]:
            cat = "cpu_pool"
        cats[expert] = cat
    return cats


def last_dependency(r: dict, s: dict | None, jobs: dict) -> str:
    if s is None or not s["post_start"]:
        return "unstamped"
    wait = s["wait_end"] - s["wait_start"] if s["wait_end"] else 0
    stream = s["stream_end"] - s["stream_start"] if s["stream_end"] else 0
    c1 = s["stream_start"] - s["post_end"] if s["stream_start"] else 0
    if wait > WAIT_NS:
        groups = r["groups"].values()
        dma = max((g.get("dma", 0) for g in groups if g.get("dma_lanes")), default=0)
        cpu = max((g.get("cpu", 0) for g in groups if g.get("cpu_mask") or g.get("late_cpu")), default=0)
        if not dma and not cpu:
            return "gate_unmatched"
        if r["gate"] and r["gate"] - max(dma, cpu) > LAG_NS:
            return "host_lag"
        if dma > cpu:
            return "dma"
        last = None
        for g, gg in r["groups"].items():
            if "cpu_first" not in gg:
                continue
            for q in range(gg["cpu_first"], gg["cpu_last"] + 1):
                j = jobs.get((g, r["row"], q))
                if j and "cpu_end" in j and (last is None or j["cpu_end"] > last["cpu_end"]):
                    last = j
        if last is None:
            return "cpu_unmatched"
        if last.get("part", 0) == 0:
            return "cpu_hit"
        row_wait = last.get("cpu_submit", r["start"]) - (r["start"] or 0)
        rest = last["cpu_end"] - last.get("cpu_submit", last["cpu_end"])
        return "cpu_miss:read" if row_wait > rest else "cpu_miss:compute"
    if stream > WAIT_NS and any(not cpu for _, _, cpu in r["lands"]):
        return "nvme_s"
    if c1 > WAIT_NS:
        return "sm_pull"
    return "none"


def cpu_jobs(ev: dict) -> dict:
    jobs = {}
    for kind in ("cpu_submit", "cpu_end"):
        for e in ev[kind]:
            j = jobs.setdefault((e["team"], e["row"], e["seq"]), {})
            j[kind] = e["ns"]
            if kind == "cpu_submit":
                j["part"] = e["a"]
    return jobs


def route_counts(d: str, forwards: list[list[dict]]) -> dict:
    """(row, gen) -> {expert: tokens routing it}, for job forwards joined to a graph_routes line."""
    path = os.path.join(d, "stages.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        lines = [x for x in map(json.loads, f) if x.get("kind") == "graph_routes"]
    routes = [[set(e for e in layer if e >= 0) for layer in x["routes"]] for x in lines]
    job = [[set(r["lanes"]) for r in fwd] for fwd in forwards]
    out = {}
    for i, li in layer_misses.match_forwards(job, routes):
        for r, layer in zip(forwards[i], lines[li]["routes"]):
            out[(r["row"], r["gen"])] = collections.Counter(e for e in layer if e >= 0)
    return out


def expert_bytes(ev: dict) -> int:
    sizes = [e["a"] // e["b"] for e in ev["copy_issue"] if e["b"] > 0 and e["a"] > 0]
    if not sizes:
        raise ValueError("no copy_issue with DMA lanes: cannot size an expert")
    return int(statistics.median(sizes))


def analyze(d: str, skip: int) -> dict:
    ev = load_events(d)
    stamps = load_stamps(d, gpu_offset(d))
    all_verifies = load_verifies(d)
    verifies = all_verifies[skip:]
    starts = [b for b, _ in all_verifies]
    recs = build_records(ev)
    jobs = cpu_jobs(ev)
    ebytes = expert_bytes(ev)

    by_verify = collections.defaultdict(list)
    for (row, gen), r in recs.items():
        r["row"], r["gen"] = row, gen
        if r["start"] is None:
            continue
        k = bisect.bisect_right(starts, r["start"]) - 1
        if 0 <= k < len(all_verifies) and r["start"] <= all_verifies[k][1]:
            by_verify[k].append(r)
    rows = 1 + max(r["row"] for r in recs.values())
    # Every verify joins the route log, warm-up included: the join walks both logs from their starts.
    forwards = [sorted(rs, key=lambda r: r["row"]) for _, rs in sorted(by_verify.items())]
    forwards = [f for f in forwards if [r["row"] for r in f] == list(range(rows))]
    pairs_by_rec = route_counts(d, forwards)
    per_v = {k: rs for k, rs in by_verify.items() if k >= skip}

    lay = collections.defaultdict(collections.Counter)
    tot_v = []
    deps = collections.Counter()
    dep_wait = collections.Counter()
    drip_deps = collections.Counter()
    for k, rs in sorted(per_v.items()):
        v = collections.Counter(records=len(rs))
        paired = all((r["row"], r["gen"]) in pairs_by_rec for r in rs) and len(rs) == rows
        v["paired"] = int(paired)
        for r in rs:
            cats = categorize(r)
            L = lay[r["row"]]
            L["records"] += 1
            n = collections.Counter(cats.values())
            n["vram"] = r["routed"] - len(cats)
            for c in CATS:
                L[c] += n[c]
                v[c] += n[c]
            L["spilled"] += sum(1 for b in r["lanes"].values() if b & SPILL)
            if paired:
                counts = pairs_by_rec[(r["row"], r["gen"])]
                for expert, t in counts.items():
                    c = cats.get(expert, "vram")
                    L["pair_" + c] += t
                    v["pair_" + c] += t
            dma_bytes = sum(g.get("dma_bytes", 0) for g in r["groups"].values())
            h2d = (n["sm"] + n["pool"] + n["nvme_gpu"]) * ebytes + dma_bytes
            nvme = sum(1 for _ in r["lands"]) * ebytes
            for key, val in (("h2d_bytes", h2d), ("nvme_bytes", nvme)):
                L[key] += val
                v[key] += val
            s = stamps.get(r["seq"])
            dep = last_dependency(r, s, jobs)
            L["dep_" + dep] += 1
            deps[dep] += 1
            if s and s["wait_end"]:
                dep_wait[dep] += s["wait_end"] - s["wait_start"]
            if n["drip"]:
                drip_deps[dep] += n["drip"]
        tot_v.append(v)

    nv = len(tot_v)
    paired_v = [v for v in tot_v if v["paired"]]
    mean = lambda xs, key: statistics.mean(x[key] for x in xs) if xs else 0.0
    layers = []
    for row in range(rows):
        L = lay[row]
        n = max(1, L["records"])
        entry = {"row": row, "records": L["records"]}
        for c in CATS:
            entry[c] = round(L[c] / n, 2)
        entry["gpu"] = round(sum(L[c] for c in GPU_CATS) / n, 2)
        entry["cpu"] = round(sum(L[c] for c in CATS if c.startswith("cpu")) / n, 2)
        entry["spilled"] = round(L["spilled"] / n, 2)
        entry["h2d_mb"] = round(L["h2d_bytes"] / n / 2**20, 1)
        entry["nvme_mb"] = round(L["nvme_bytes"] / n / 2**20, 1)
        dep_counts = {key[4:]: L[key] for key in L if key.startswith("dep_")}
        entry["last_dep"] = {key: round(val / n, 3) for key, val in sorted(dep_counts.items(), key=lambda x: -x[1])}
        pair_n = sum(L["pair_" + c] for c in CATS)
        entry["pairs"] = {c: round(L["pair_" + c] / max(1, pair_n), 3) for c in CATS} if pair_n else {}
        layers.append(entry)
    total_deps = sum(deps.values())
    total_wait = sum(dep_wait.values())
    bg = background(ev, verifies, ebytes)
    return {
        "capture": os.path.abspath(d),
        "verifies": nv,
        "skipped": skip,
        "records_per_verify": round(mean(tot_v, "records"), 2),
        "verifies_with_routes": len(paired_v),
        "expert_bytes": ebytes,
        "per_verify_unique": {c: round(mean(tot_v, c), 2) for c in CATS},
        "per_verify_pairs": {c: round(mean(paired_v, "pair_" + c), 2) for c in CATS},
        "per_verify_mb": {"h2d_demand": round(mean(tot_v, "h2d_bytes") / 2**20, 1),
                          "nvme_demand": round(mean(tot_v, "nvme_bytes") / 2**20, 1), **bg["per_verify_mb"]},
        "background_per_verify": bg["counts"],
        "last_dependency_share": {key: round(val / total_deps, 4) for key, val in deps.most_common()},
        "last_dependency_wait_share": {key: round(val / max(1, total_wait), 4) for key, val in dep_wait.most_common()},
        "drip_used_by_layer_last_dependency": dict(drip_deps.most_common()),
        "layers": layers,
    }


def background(ev: dict, verifies: list, ebytes: int) -> dict:
    """Prefetch reads, drips and promotions inside the analysed span, per verify."""
    if not verifies:
        return {"counts": {}, "per_verify_mb": {}}
    lo, hi = verifies[0][0], verifies[-1][1]
    n = len(verifies)
    inside = lambda kind: sum(1 for e in ev[kind] if lo <= e["ns"] <= hi)
    promoted = sum(e["c"] for e in ev["delta_publish"] if lo <= e["ns"] <= hi)
    counts = {"spec_land": inside("spec_land"), "spec_use": inside("spec_use"), "pool_point": inside("pool_point"),
              "drip_land": inside("drip_land"), "drip_use": inside("drip_use"), "drip_abandon": inside("drip_abandon"),
              "promoted": promoted}
    return {"counts": {key: round(val / n, 2) for key, val in counts.items()},
            "per_verify_mb": {"nvme_prefetch": round(counts["spec_land"] * ebytes / n / 2**20, 1),
                              "h2d_drip": round(counts["drip_land"] * ebytes / n / 2**20, 1)}}


def report(res: dict) -> None:
    print(f"{res['capture']}: {res['verifies']} verifies (skipped {res['skipped']}), "
          f"{res['records_per_verify']} records/verify, {res['verifies_with_routes']} joined to routes, "
          f"expert {res['expert_bytes'] / 2**20:.2f} MiB")
    print("per layer, unique experts per verify by source (cpu_* computed on CPU), MiB per verify, top last dependency")
    head = ["row"] + list(CATS) + ["gpu", "cpu", "h2dMB", "nvmeMB", "last dep (share)"]
    print(" ".join(f"{h:>8s}" for h in head[:-1]), head[-1])
    for L in res["layers"]:
        top = next(iter(L["last_dep"].items()), ("-", 0))
        cells = [L["row"]] + [L[c] for c in CATS] + [L["gpu"], L["cpu"], L["h2d_mb"], L["nvme_mb"]]
        print(" ".join(f"{x:8}" for x in cells), f"{top[0]} {top[1]:.2f}")
    print("per verify, unique experts:", json.dumps(res["per_verify_unique"]))
    print("per verify, (token, expert) pairs:", json.dumps(res["per_verify_pairs"]))
    print("per verify, MiB:", json.dumps(res["per_verify_mb"]))
    print("background per verify:", json.dumps(res["background_per_verify"]))
    print("last dependency, share of layers:", json.dumps(res["last_dependency_share"]))
    print("last dependency, share of gate-wait time (traced):", json.dumps(res["last_dependency_wait_share"]))
    print("drip lanes used, by their layer's last dependency:", json.dumps(res["drip_used_by_layer_last_dependency"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture")
    p.add_argument("--skip", type=int, default=8, help="verifies skipped at the start (warm-up)")
    p.add_argument("--json")
    a = p.parse_args()
    try:
        res = analyze(a.capture, a.skip)
    except (ValueError, FileNotFoundError) as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    report(res)
    with open(a.json or os.path.join(a.capture, "expert_breakdown.json"), "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
