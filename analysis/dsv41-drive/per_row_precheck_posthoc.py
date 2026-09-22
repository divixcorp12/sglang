import json, sys, statistics as S, collections
p = sys.argv[1]
steps = {}; reqs = []
with open(p) as fh:
    for line in fh:
        d = json.loads(line)
        if d.get("kind") == "graph_step": steps[d["forward"]] = d
        elif d.get("kind") == "ram_miss_request": reqs.append(d)
vm = [s["vram_miss"] for s in steps.values()]
rm = [s["ram_miss"] for s in steps.values()]
ts = sorted(s["t"] for s in steps.values())
dt = [b - a for a, b in zip(ts, ts[1:])]
print("steps", len(steps), "vram_miss mean %.1f p10 %d p50 %d p90 %d" % (S.mean(vm), sorted(vm)[len(vm)//10], S.median(vm), sorted(vm)[9*len(vm)//10]))
print("ram_miss mean %.1f" % S.mean(rm), "traced step dt median %.1f ms mean %.1f ms" % (1000*S.median(dt), 1000*S.mean(dt)))
kh = collections.Counter(); waits = []; nreq=0; sumhits=0
for d in reqs:
    if d["request"]["type"] != "demand" or d["rows_asked"] < 1 or d["status"] != "served": continue
    st = steps.get(d["forward"])
    if st is None: continue
    m = d["rows_asked"]; k = min(6, max(m, int(round(st["vram_miss"]/40))))
    kh[k] += 1; sumhits += k - m; nreq += 1
    waits.append((d["stages_ns"]["done"] - d["stages_ns"]["observed"]) / 1e6)
waits.sort()
print("k hist", dict(sorted(kh.items())), "mean h %.2f" % (sumhits/nreq))
print("read-wait ms p10 %.1f p50 %.1f p90 %.1f mean %.1f; sum over steps per step %.1f ms" % (waits[len(waits)//10], waits[len(waits)//2], waits[9*len(waits)//10], S.mean(waits), sum(waits)/len(steps)))
print("modelled gather per step = mean vram_miss*1.055 = %.1f ms" % (S.mean(vm)*1.055))
