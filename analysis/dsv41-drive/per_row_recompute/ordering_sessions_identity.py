import json, collections, statistics as S, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *

# (2) ordering evidence, at the resolution the trace has (extent_cqe_ns stamps are per reap)
for name in ON + OFF:
    steps, used, eager = analyse(name)
    n_m2 = 0; distinct = collections.Counter(); inv_cqe = 0; inv_pack = 0; strict_multi = 0; same_reap_all = 0; pack_eq = 0
    pack_first_is_ready_first = 0; gap_hist = []
    for d, st in used:
        m = d["rows_asked"]
        if m < 2: continue
        n_m2 += 1
        rows = d["row_pack_ns"]
        # when is each row whole? max cqe over its extents
        whole = collections.defaultdict(int)
        for e in d["extent_cqe_ns"]:
            whole[e["row"]] = max(whole[e["row"]], e["cqe"])
        w = [whole[j] for j in range(m)]
        nd = len(set(w)); distinct[nd] += 1
        if nd == 1: same_reap_all += 1
        # inversion in completion time: some later ordinal strictly before an earlier one
        if any(w[j + 1] < w[j] for j in range(m - 1)): inv_cqe += 1
        order = [r["row"] for r in sorted(rows, key=lambda r: r["start"])]
        if order == list(range(m)): pack_eq += 1
        else: inv_pack += 1
        if nd > 1 and not any(w[j + 1] < w[j] for j in range(m - 1)): strict_multi += 1
    print("%s: m>=2 requests %d; distinct row-completion stamps per request %s; ALL rows in one reap %d; requests with >=2 distinct stamps and no inversion %d; completion inversions %d; pack order != ordinal %d; pack order == ordinal %d" %
          (name, n_m2, dict(sorted(distinct.items())), same_reap_all, strict_multi, inv_cqe, inv_pack, pack_eq))

# per-drive FIFO: within one request, do the two drives' extents of consecutive rows complete in submission order?
name = ON[0]; steps, used, eager = analyse(name)
reqs = [d for d, _ in used if d["rows_asked"] >= 2]
inv_part = 0; tot = 0
for d in reqs:
    by = collections.defaultdict(dict)
    for e in d["extent_cqe_ns"]:
        by[e["part"]][e["row"]] = e["cqe"]
    for part, rr in by.items():
        rows = sorted(rr)
        for a, b in zip(rows, rows[1:]):
            tot += 1
            if rr[b] < rr[a]: inv_part += 1
print("per-part (drive) consecutive-row extent completions in %s: pairs %d, out of order %d" % (name, tot, inv_part))

# (3) per-session dispersion of the modelled saving (sessions split by the largest gaps between graph_step stamps)
name = ON[0]; steps, used, eager = analyse(name)
fw = sorted(steps)
ts = [steps[f]["t"] for f in fw]
gaps = sorted(((b - a, i) for i, (a, b) in enumerate(zip(ts, ts[1:]))), reverse=True)[:3]
cuts = sorted(i for _, i in gaps)
bounds = [fw[0]] + [fw[i + 1] for i in cuts] + [fw[-1] + 1]
print("session gap seconds %s; forward boundaries %s" % ([round(g, 1) for g, _ in gaps], bounds))
T = 1000 / (sum(json.load(open(D + n + ".json"))["mean_decode_tok_s"] for n in ("task1-3-new-on-U","task1-6-new-on-U","task1c-3-new-on-U")) / 3)
for s in range(4):
    lo_f, hi_f = bounds[s], bounds[s + 1]
    tot = collections.Counter(); nst = set()
    for d, st in used:
        if not (lo_f <= d["forward"] < hi_f): continue
        m = d["rows_asked"]; sg = d["stages_ns"]
        t0, res, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
        R = sorted(r["end"] / 1e6 for r in d["row_pack_ns"])
        k = min(6, max(m, int(round(st["vram_miss"] / LAYERS))))
        b, r, t2, _ = per_req(t0, res, done, R, k, C)
        tot["b"] += b; tot["r"] += r; tot["t"] += t2; nst.add(d["forward"])
    n = max(1, len(nst))
    print("session %d: steps %d  rand %.2f ms (%.2f%%)  two-phase %.2f ms (%.2f%%)  best %.2f ms" % (s, n, tot["r"]/n, 100*tot["r"]/n/T, tot["t"]/n, 100*tot["t"]/n/T, tot["b"]/n))

# (4) identical streams across arms?
sig = {}
for name in ON + OFF:
    steps, used, eager = analyse(name)
    sig[name] = ([d["rows_asked"] for d, _ in used], [steps[f]["vram_miss"] for f in sorted(steps)], [steps[f]["ram_miss"] for f in sorted(steps)])
base = sig[ON[0]]
for name in sig:
    print("%s: rows_asked sequence identical to %s: %s; vram_miss per step identical: %s; ram_miss per step identical: %s" %
          (name, ON[0], sig[name][0] == base[0], sig[name][1] == base[1], sig[name][2] == base[2]))

# (5) eager per-layer lines: are VRAM misses concentrated in layers that also read?
steps, used, eager = analyse(ON[0])
rd = []; nr = []
for d in eager:
    if "vram_miss" in d and "ram_miss" in d and d.get("layer", -1) >= 0:
        (rd if d["ram_miss"] > 0 else nr).append(d)
print("eager per-layer lines: %d with ram_miss>0 (mean vram_miss %.2f, mean vram_miss-ram_miss %.2f), %d with ram_miss=0 (mean vram_miss %.2f); tokens %s" %
      (len(rd), S.mean(x["vram_miss"] for x in rd) if rd else -1, S.mean(x["vram_miss"] - x["ram_miss"] for x in rd) if rd else -1,
       len(nr), S.mean(x["vram_miss"] for x in nr) if nr else -1, sorted({x.get("tokens") for x in eager})[:6]))
