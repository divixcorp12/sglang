import json, sys, itertools, statistics as S
names = sys.argv[1:]
data = {}
for n in names:
    rows = []; steps = []
    with open(n + ".trace") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("kind") == "ram_miss_request":
                st = d["stages_ns"]
                rows.append(dict(seq=d["request"]["seq"], layer=d["layer"], type=d["request"]["type"], m=d["rows_asked"],
                                 observed=st["observed"], done=st["done"], pack=d["spans_ns"]["pack"],
                                 f2l=d["spans_ns"]["first_to_last_cqe"], s2f=d["spans_ns"]["submit_to_first_cqe"],
                                 cqes=tuple(e["cqe"] for e in d["extent_cqe_ns"]), t=d["t"],
                                 ends=tuple(r["end"] for r in d["row_pack_ns"])))
            elif d.get("kind") == "graph_step":
                steps.append((d["t"], d["vram_miss"], d["ram_miss"]))
    data[n] = (rows, steps)
print("files:", len(names))
for n, (rows, steps) in data.items():
    ob = [r["observed"] for r in rows]
    print(n, "requests", len(rows), "first observed(ns) %d" % ob[0], "last %d" % ob[-1], "span_s %.1f" % ((ob[-1]-ob[0])/1e9), "graph_steps", len(steps), "t0 %.2f" % rows[0]["t"])
print()
print("pairwise: (a,b) same-observed-stamp count / same-cqe-tuple count / same-pack_ns count / same non-timing key / corr(done-observed) ")
for a, b in itertools.combinations(names, 2):
    ra, rb = data[a][0], data[b][0]
    n = min(len(ra), len(rb))
    so = sum(1 for x, y in zip(ra, rb) if x["observed"] == y["observed"])
    sc = sum(1 for x, y in zip(ra, rb) if x["cqes"] == y["cqes"] and x["cqes"])
    sp = sum(1 for x, y in zip(ra, rb) if x["pack"] == y["pack"] and x["pack"])
    sk = sum(1 for x, y in zip(ra, rb) if (x["layer"], x["type"], x["m"]) == (y["layer"], y["type"], y["m"]))
    wa = [x["done"] - x["observed"] for x, y in zip(ra, rb) if x["m"] and y["m"]]
    wb = [y["done"] - y["observed"] for x, y in zip(ra, rb) if x["m"] and y["m"]]
    diffs = [abs(u - v) for u, v in zip(wa, wb)]
    print(f"{a} v {b}: observed== {so}/{n}  cqes== {sc}  pack== {sp}  key== {sk}/{n}  median |wait diff| {S.median(diffs)/1e3:.0f} us  mean waits {S.mean(wa)/1e6:.3f} / {S.mean(wb)/1e6:.3f} ms")
