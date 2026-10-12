"""Per decode-graph replay: F-end -> next Post-start chain time, Post-start -> first CE copy, CE bytes, streams.

Input: a node-mode Nsight Systems sqlite export of one traced arm. Replays are grouped by
(correlationId, graphId) with at least MIN_KERNELS kernels; the first SKIP replays are dropped.
F = end of the lease-finalize kernel, Post = start of the RAM-miss post kernel (matched by name fragment;
the names found are printed). Node-mode timings are inflated; compare arms traced the same way only.

Usage: python trace_chain.py <export.sqlite> [--skip 5] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics

MIN_KERNELS = 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    q = lambda s, *a: c.execute(s, a).fetchall()
    names = dict(q("select id, value from StringIds"))
    fin_ids = [i for i, v in names.items() if "finalize" in v and "ram_miss" in v.lower() or "lease_finalize" in v]
    post_ids = [i for i, v in names.items() if ("ram_miss_post" in v or "_post_kernel" in v) and "gather" not in v]
    shared_hint = [i for i, v in names.items() if "gemv_int8" in v]
    print("finalize names:", sorted({names[i][:80] for i in fin_ids}))
    print("post names:", sorted({names[i][:80] for i in post_ids}))
    graphs = q(f"""select correlationId, graphId, min(start), max(end), count(*) from CUPTI_ACTIVITY_KIND_KERNEL
                  where graphId is not null and graphId != 0 group by correlationId, graphId
                  having count(*) >= {MIN_KERNELS} order by min(start)""")
    graphs = graphs[args.skip:args.skip + args.limit] if args.limit else graphs[args.skip:]
    print(f"{len(graphs)} replays after skip {args.skip}")
    memcpy_cols = [r[1] for r in q("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
    rows = []
    for corr, gid, gs, ge, nk in graphs:
        ks = q("""select start, end, shortName, demangledName, streamId from CUPTI_ACTIVITY_KIND_KERNEL
                  where correlationId=? and graphId=? order by start""", corr, gid)
        f_ends = [k[1] for k in ks if k[2] in fin_ids or k[3] in fin_ids]
        p_starts = [k[0] for k in ks if k[2] in post_ids or k[3] in post_ids]
        streams = {}
        for k in ks:
            streams[k[4]] = streams.get(k[4], 0) + 1
        chain = []
        for fe in f_ends:
            nxt = [p for p in p_starts if p > fe]
            if nxt:
                chain.append(nxt[0] - fe)
        cps = q("""select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ? and start <= ?
                   and copyKind = 1 order by start""", gs, ge)
        post_to_copy = []
        for p in p_starts:
            after = [cp[0] for cp in cps if cp[0] >= p]
            nxt_post = [x for x in p_starts if x > p]
            if after and (not nxt_post or after[0] < nxt_post[0]):
                post_to_copy.append(after[0] - p)
        union, cur_s, cur_e = 0, None, None
        for s, e, _ in cps:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    union += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            union += cur_e - cur_s
        rows.append({
            "span_ms": (ge - gs) / 1e6, "kernels": nk, "n_F": len(f_ends), "n_Post": len(p_starts),
            "F_to_nextPost_ms": sum(chain) / 1e6, "n_chain": len(chain),
            "post_to_first_copy_us_mean": statistics.mean(post_to_copy) / 1e3 if post_to_copy else None,
            "ce_copies": len(cps), "ce_MB": sum(cp[2] for cp in cps) / 1e6, "ce_union_ms": union / 1e6,
            "kernels_per_stream": streams,
        })
    summ = {}
    for key in ("span_ms", "F_to_nextPost_ms", "post_to_first_copy_us_mean", "ce_copies", "ce_MB", "ce_union_ms", "n_F", "n_Post", "n_chain"):
        v = [r[key] for r in rows if r[key] is not None]
        summ[key] = {"mean": statistics.mean(v), "median": statistics.median(v)} if v else None
    summ["streams_first_replay"] = rows[0]["kernels_per_stream"] if rows else None
    summ["replays"] = len(rows)
    print(json.dumps(summ, indent=1))
    if args.out:
        json.dump({"summary": summ, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
