"""Compare RAM-miss smoke arms (bounce + pack workers vs row-image direct reads) from their stage traces.

Usage: compare_arms.py NAME=DIR [NAME=DIR ...] [--json OUT]

Each DIR is one smoke arm's output: ``stages.jsonl`` (SGLANG_DSV41_EXPERT_TRACE_PATH, schema 7) and
``responses.jsonl`` (the smoke driver's greedy responses). CPU only; run under ``taskset -c 0-63``.

What is measured, and from which fields (every clock is the host CLOCK_MONOTONIC):

  demands      ``ram_miss_request`` records with status ``served`` and rows_asked > 0, in a decode forward
               (a forward some ``graph_step`` record carries). Prefill demands are left out.
  host tail    per demand, ``done - last_cqe``: what the host still did after the final completion.
  pack tail    per demand, ``pack_end - last_cqe``. Bounce mode: the pack workers' copy left after the last
               completion. Direct mode (row images): the reader writes pack stamps = publish clocks
               (row_pack_start/end = first/last piece publish of the row, pack_end = the demand's last
               publish), so the same subtraction is "last publish - last completion".
  piece publish  schema 7 has no per-piece publish CLOCK: ``pieces[].publish`` is an event sequence number.
               Two views are given. In clock time, the row's first piece is ``row_pack_start - min piece cqe``
               and its last is ``row_pack_end - max piece cqe``. In events, every piece gets ``publish seq -
               vet seq``: the number of landings, vettings and publishes the owner handled in between. Bounce
               mode includes the piece's copy job; direct mode should publish in the vetting's owner turn.
  submit->done per demand, ``done - submit``, bucketed by rows asked (1, 2, 3+).
  per step     served demands and their rows over decode steps (the ``steps`` of the request blocks'
               graph_step records); also graph_step ``ram_miss`` (demand rows the thread read) per step.
  stalls       slowreads.py's definition: multi-row demands (rows_asked >= 2) whose first->last completion
               span exceeds 10 ms. Also the summed ``bank_stalls`` (admission waits for a bank to pack).
  ms/token     from graph_step ``t`` (time.monotonic() when the step's lagged snapshot was recorded). The
               trace alternates eager blocks (a prefill: 40 layer records) with graph blocks (the decode
               steps). The last len(responses) graph blocks are the requests; earlier ones are warm-up. Within
               a block, decode ms/token = (t_last - t_first) / (records - 1). A block's first record carries
               the previous request's last step (the snapshot read lags one step), which does not change the
               cadence. The last record of the file is the shutdown flush (final=True), which is dropped when
               the last block holds one record more than the others. Validation, per request: the driver's
               wall time ``s`` minus the preceding prefill block's span, divided by completion_tokens - 1.
  responses    byte-identical text per (prompt, rep) across arms; each arm's rep 0 vs rep 1.

A record's mode is read from the record: ``pack_workers`` 0 with ``piece_stream`` 1 is direct mode (no pack pool
exists there), otherwise bounce mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

SLOW_SPAN_US = 10000.0  # slowreads.py's default threshold


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def dist(xs):
    return {
        "n": len(xs),
        "p10": pct(xs, 10),
        "p50": pct(xs, 50),
        "p90": pct(xs, 90),
        "p99": pct(xs, 99),
        "max": max(xs) if xs else float("nan"),
        "mean": sum(xs) / len(xs) if xs else float("nan"),
    }


def graph_blocks(path):
    """Alternating eager/graph blocks of the forward records, in file order."""
    blocks = []
    for line in open(path):
        r = json.loads(line)
        kind = r.get("kind")
        if kind == "ram_miss_request":
            continue
        tag = "g" if kind == "graph_step" else "e"
        if not blocks or blocks[-1]["tag"] != tag:
            blocks.append({"tag": tag, "recs": []})
        blocks[-1]["recs"].append(r)
    return blocks


def analyse(arm_dir):
    stages = os.path.join(arm_dir, "stages.jsonl")
    responses = [json.loads(l) for l in open(os.path.join(arm_dir, "responses.jsonl"))]

    blocks = graph_blocks(stages)
    gblocks = [i for i, b in enumerate(blocks) if b["tag"] == "g"]
    request_blocks = gblocks[-len(responses):] if responses else []
    sizes = [len(blocks[i]["recs"]) for i in request_blocks]
    dropped_final = False
    if len(sizes) >= 2 and sizes[-1] == sorted(sizes[:-1])[len(sizes[:-1]) // 2] + 1:
        blocks[request_blocks[-1]]["recs"].pop()
        dropped_final = True

    decode_fwds, steps, graph_ram_rows = set(), 0, 0
    per_request = []
    intervals = []
    for n, bi in enumerate(request_blocks):
        recs = blocks[bi]["recs"]
        for r in recs:
            decode_fwds.add(r["forward"])
            steps += r.get("steps", 1)
            graph_ram_rows += r.get("ram_miss", 0)
        ts = [r["t"] for r in recs]
        intervals += [(b - a) * 1e3 for a, b in zip(ts, ts[1:])]
        ms_trace = (ts[-1] - ts[0]) * 1e3 / max(1, len(ts) - 1)
        pre = blocks[bi - 1] if bi > 0 and blocks[bi - 1]["tag"] == "e" else None
        prefill_s = (pre["recs"][-1]["t"] - pre["recs"][0]["t"]) if pre else float("nan")
        resp = responses[n]
        toks = (resp.get("usage") or {}).get("completion_tokens") or 0
        ms_wall = (resp["s"] - prefill_s) * 1e3 / (toks - 1) if toks > 1 else float("nan")
        per_request.append({"prompt": resp["prompt"], "rep": resp["rep"], "records": len(recs),
                            "ms_token_trace": ms_trace, "ms_token_wall": ms_wall, "prefill_s": prefill_s,
                            "wall_s": resp["s"], "tokens": toks})

    host_tail, pack_tail, first_pub, last_pub, pub_events = [], [], [], [], []
    host_tail1, pack_tail1 = [], []
    s2d = {1: [], 2: [], 3: []}
    demands = rows = slow = multi = bank_stalls = 0
    modes = Counter()
    statuses = Counter()
    for line in open(stages):
        r = json.loads(line)
        if r.get("kind") != "ram_miss_request" or r["forward"] not in decode_fwds:
            continue
        statuses[(r["request"]["type"], r["status"])] += 1
        if r["status"] != "served" or not r.get("rows_asked"):
            continue
        q = r["request"]
        direct = q.get("pack_workers", 0) == 0 and q.get("piece_stream", 0) == 1
        modes["direct" if direct else f"bounce(workers={q.get('pack_workers')},piece_stream={q.get('piece_stream')})"] += 1
        s = r["stages_ns"]
        demands += 1
        rows += r["rows_asked"]
        bank_stalls += r.get("bank_stalls", 0) or 0
        ht = (s["done"] - s["last_cqe"]) / 1e3
        pt = (s["pack_end"] - s["last_cqe"]) / 1e3
        host_tail.append(ht)
        pack_tail.append(pt)
        if r["rows_asked"] == 1:
            host_tail1.append(ht)
            pack_tail1.append(pt)
        s2d[min(3, r["rows_asked"])].append((s["done"] - s["submit"]) / 1e3)
        if r["rows_asked"] >= 2:
            multi += 1
            if (s["last_cqe"] - s["first_cqe"]) / 1e3 > SLOW_SPAN_US:
                slow += 1
        packs = {p["row"]: p for p in r.get("row_pack_ns", [])}
        for pc in r.get("pieces", []):
            cqes = [c for c in pc["cqe"] if c]
            rp = packs.get(pc["row"])
            if cqes and rp and rp["start"] and rp["end"]:
                first_pub.append((rp["start"] - min(cqes)) / 1e3)
                last_pub.append((rp["end"] - max(cqes)) / 1e3)
            for seq, pub in zip(pc["seq"], pc.get("publish", [])):
                if seq and pub:
                    pub_events.append(pub - seq)

    names = [(x["prompt"], x["rep"]) for x in responses]
    return {
        "dir": arm_dir,
        "modes": dict(modes),
        "decode_statuses": {f"{a}/{b}": v for (a, b), v in statuses.items()},
        "decode_steps": steps,
        "demands": demands,
        "demands_per_step": demands / steps if steps else float("nan"),
        "rows_per_step": rows / steps if steps else float("nan"),
        "graph_ram_rows_per_step": graph_ram_rows / steps if steps else float("nan"),
        "host_tail_us": dist(host_tail),
        "host_tail_single_us": dist(host_tail1),
        "pack_tail_us": dist(pack_tail),
        "pack_tail_single_us": dist(pack_tail1),
        "first_piece_publish_us": dist(first_pub),
        "last_piece_publish_us": dist(last_pub),
        "piece_publish_events": dist(pub_events),
        "submit_to_done_us": {("3+" if k == 3 else str(k)): dist(v) for k, v in s2d.items()},
        "stalls": {"slow_multi_row": slow, "multi_row": multi, "bank_stalls": bank_stalls},
        "step_interval_ms": dist(intervals),
        "ms_token_trace": sum(x["ms_token_trace"] * (x["records"] - 1) for x in per_request)
        / max(1, sum(x["records"] - 1 for x in per_request)),
        "ms_token_wall": sum(x["ms_token_wall"] for x in per_request) / max(1, len(per_request)),
        "per_request": per_request,
        "dropped_shutdown_record": dropped_final,
        "responses": {f"{p}/{k}": x["text"] for (p, k), x in zip(names, responses)},
    }


def fmt(d, keys=("n", "p50", "p90", "p99", "mean", "max")):
    return " ".join(f"{k} {d[k]:.0f}" if k == "n" else f"{k} {d[k]:.1f}" for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=DIR")
    ap.add_argument("--json", help="write the full per-arm results here")
    ap.add_argument("--per-request", action="store_true", help="print ms/token per request")
    a = ap.parse_args()
    res = {}
    for spec in a.arms:
        name, _, d = spec.partition("=")
        res[name] = analyse(d)

    for name, r in res.items():
        print(f"== {name}  ({r['dir']})")
        print(f"  modes {r['modes']}  decode steps {r['decode_steps']}  shutdown record dropped {r['dropped_shutdown_record']}")
        print(f"  decode statuses {r['decode_statuses']}")
        print(f"  demands/step {r['demands_per_step']:.2f}  rows/step {r['rows_per_step']:.2f}  "
              f"graph_step ram rows/step {r['graph_ram_rows_per_step']:.2f}")
        print(f"  host tail (done - last cqe) us:      {fmt(r['host_tail_us'])}")
        print(f"    single-row:                        {fmt(r['host_tail_single_us'])}")
        print(f"  pack tail (pack_end - last cqe) us:  {fmt(r['pack_tail_us'])}")
        print(f"    single-row:                        {fmt(r['pack_tail_single_us'])}")
        print(f"  first piece publish - its cqe us:    {fmt(r['first_piece_publish_us'])}")
        print(f"  last piece publish - its cqe us:     {fmt(r['last_piece_publish_us'])}")
        print(f"  piece publish - vet (events):        {fmt(r['piece_publish_events'])}")
        for k, v in r["submit_to_done_us"].items():
            print(f"  submit->done rows {k:2s} us:           {fmt(v)}")
        st = r["stalls"]
        print(f"  stalls: {st['slow_multi_row']} of {st['multi_row']} multi-row demands first->last > "
              f"{SLOW_SPAN_US:.0f} us; bank_stalls {st['bank_stalls']}")
        print(f"  step interval ms: {fmt(r['step_interval_ms'])}")
        print(f"  ms/token decode: trace {r['ms_token_trace']:.1f}  wall-validated {r['ms_token_wall']:.1f}")
        if a.per_request:
            for x in r["per_request"]:
                print(f"    prompt {x['prompt']} rep {x['rep']}: trace {x['ms_token_trace']:.1f} wall {x['ms_token_wall']:.1f} "
                      f"ms/token (prefill {x['prefill_s']:.2f}s, wall {x['wall_s']:.1f}s, {x['records']} records)")

    names = list(res)
    print("== responses")
    for name in names:
        rs = res[name]["responses"]
        reps = [(k, k.replace("/0", "/1")) for k in rs if k.endswith("/0")]
        same = all(rs.get(b) == rs[a] for a, b in reps)
        print(f"  {name}: rep 0 == rep 1 for every prompt: {same}")
    ref = names[0]
    for name in names[1:]:
        a_, b_ = res[ref]["responses"], res[name]["responses"]
        keys = sorted(set(a_) | set(b_))
        diff = [k for k in keys if a_.get(k) != b_.get(k)]
        print(f"  {name} vs {ref}: {len(keys) - len(diff)} of {len(keys)} responses byte-identical"
              + (f"; differ: {diff}" if diff else ""))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
