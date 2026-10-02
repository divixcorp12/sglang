"""The DSpark draft's real routing, from SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH, priced at the CPU kernel's cost.

Plan 2026-10-02-dsv41-dspark-cpu-draft, Task 1 follow-up: draft_bench.py brackets a CPU draft step between 14.5 ms
(all 6 rows share 3 experts) and 26.5 ms (independent rows); this reports where the real routes fall.
  python analysis/dsv41-drive/dspark/draft_routes_report.py ROUTES.jsonl [MS_PER_PASS] [N_ROUTED]
MS_PER_PASS defaults to 0.518, draft_bench's 4-bit, 12-thread, 6-row cell. Ids >= N_ROUTED (128) are fused shared
experts that stay on the GPU; they are counted and excluded.
"""

import collections
import json
import statistics
import sys

DECODE_MAX_ROWS = 8


def passes(ids: list[int]) -> tuple[int, int]:
    """(union, weight passes): the kernel reads an expert once per 2 rows that route to it (CHUNK_M = 2)."""
    counts = collections.Counter(ids)
    return len(counts), sum((t + 1) // 2 for t in counts.values())


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def report(path: str, ms_per_pass: float, n_routed: int) -> dict:
    per_layer = collections.defaultdict(list)
    big_calls = collections.Counter()
    shared_ids = 0
    for line in open(path):
        rec = json.loads(line)
        rows = rec["ids"]
        if len(rows) > DECODE_MAX_ROWS:
            big_calls[len(rows)] += 1
            continue
        flat = []
        for row in rows:
            for e in row:
                if e >= n_routed:
                    shared_ids += 1
                elif e >= 0:
                    flat.append(e)
        per_layer[rec["layer"]].append((len(rows), flat))

    out = {"ms_per_pass": ms_per_pass, "prefill_sized_calls": dict(big_calls), "shared_ids_excluded": shared_ids}
    stage_ms = {}
    for layer in sorted(per_layer):
        calls = per_layer[layer]
        unions, wpasses, ms = [], [], []
        freq = collections.Counter()
        rows_hist = collections.Counter()
        for rows, flat in calls:
            u, p = passes(flat)
            unions.append(u)
            wpasses.append(p)
            ms.append(p * ms_per_pass)
            freq.update(flat)
            rows_hist[rows] += 1
        total = sum(freq.values())
        ranked = [c for _, c in freq.most_common()]
        stage_ms[layer] = ms
        out[f"layer_{layer}"] = {
            "calls": len(calls),
            "rows_hist": dict(sorted(rows_hist.items())),
            "union_mean": round(statistics.mean(unions), 2),
            "union_p90": pct(unions, 0.9),
            "passes_mean": round(statistics.mean(wpasses), 2),
            "pred_ms_mean": round(statistics.mean(ms), 2),
            "distinct_experts": len(freq),
            **{f"top{n}_coverage": round(sum(ranked[:n]) / total, 3) for n in (8, 16, 32)},
        }
    n = min(len(v) for v in stage_ms.values()) if stage_ms else 0
    steps = [sum(stage_ms[layer][i] for layer in stage_ms) for i in range(n)]
    if steps:
        out["draft_step_pred_ms"] = {
            "mean": round(statistics.mean(steps), 2),
            "median": round(statistics.median(steps), 2),
            "p90": round(pct(steps, 0.9), 2),
        }
    return out


if __name__ == "__main__":
    ms = float(sys.argv[2]) if len(sys.argv) > 2 else 0.518
    n_routed = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    print(json.dumps(report(sys.argv[1], ms, n_routed), indent=1))
