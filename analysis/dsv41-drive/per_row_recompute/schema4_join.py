# Joins the measured lanes of a schema-4 trace (task1f, gen4, run under accepted CPU contention: its TIMINGS are not to be quoted)
# onto the stage timings of an earlier, less-contended trace of the same request stream (task1-2), request by request, after
# checking the two streams are the same. Then: measured k vs the A1 k, and the model's figures with each. Divides by sum(steps).
import json, sys, collections, statistics as S
from core import *

def rd(path):
    steps, reqs = {}, []
    for line in open(D + path + ".trace"):
        d = json.loads(line); k = d.get("kind")
        if k == "graph_step": steps[d["forward"]] = d
        elif k == "ram_miss_request": reqs.append(d)
    return steps, reqs

sA, rA = rd("task1-2-new-on-T"); sF, rF = rd("task1f-0-new-on-T")
sig = lambda d: (d["layer"], d["request"]["type"], d["rows_asked"], d["status"])
same = len(rA) == len(rF) and all(sig(a) == sig(b) for a, b in zip(rA, rF))
print("requests %d vs %d; (layer, type, rows_asked, status) identical request by request: %s" % (len(rA), len(rF), same))
print("requests whose forward label differs: %d (the register lag at a step boundary)" % sum(1 for a, b in zip(rA, rF) if a["forward"] != b["forward"]))
dl = [f for f in sA if (sA[f]["vram_miss"], sA[f]["ram_miss"], sA[f]["steps"]) != (sF[f]["vram_miss"], sF[f]["ram_miss"], sF[f]["steps"])]
print("graph_step lines whose (vram_miss, ram_miss, steps) differ: %d of %d; sums equal: %s" % (len(dl), len(sA),
      sum(s["vram_miss"] for s in sA.values()) == sum(s["vram_miss"] for s in sF.values()) and sum(s["ram_miss"] for s in sA.values()) == sum(s["ram_miss"] for s in sF.values())))
if not same: sys.exit("streams differ: no join")
nst = sum(s["steps"] for s in sA.values())

def kmodel(d, st, mode, lanes):
    m = d["rows_asked"]
    if mode == "A1": return min(6, max(m, int(round(st["vram_miss"] / LAYERS))))             # as registered
    if mode == "A1/steps": return min(6, max(m, int(round(st["vram_miss"] / st["steps"] / LAYERS))))
    return max(m, lanes)                                                                   # measured

def run(reqs_t, steps_t, lane_src, label):
    tot = collections.Counter()
    for d, dl in zip(reqs_t, lane_src):
        st = steps_t.get(d["forward"]); sg = d["stages_ns"]
        if st is None or d["request"]["type"] != "demand" or d["rows_asked"] < 1 or d["status"] != "served": continue
        if d["untraced"]["rows"] != 0 or len(d["row_pack_ns"]) != d["rows_asked"]: continue
        m = d["rows_asked"]; t0, res, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
        R = sorted(r["end"] / 1e6 for r in d["row_pack_ns"])
        for mode in ("A1", "A1/steps", "measured"):
            k = kmodel(d, st, mode, dl["request"]["lanes"])
            # exact permutations up to k = 6 (measured k never exceeds 6 in this trace; checked below)
            b, r, t2, _ = per_req(t0, res, done, R, k, C)
            tot[(mode, "best")] += b; tot[(mode, "rand")] += r; tot[(mode, "two")] += t2; tot[(mode, "hit")] += k - m
    print("\n%s (per %d decode steps, T_step 257.5 ms only to size the shares):" % (label, nst))
    for mode in ("A1", "A1/steps", "measured"):
        b, r, t2, h = (tot[(mode, x)] / nst for x in ("best", "rand", "two", "hit"))
        print("  k = %-9s best %.2f (%.1f%%)  rand %.2f (%.1f%%)  two-phase %.2f (%.1f%%)  hit lanes in read layers %.1f/step  per-row over two-phase: best %+.2f random %+.2f"
              % (mode, b, 100 * b / 257.5, r, 100 * r / 257.5, t2, 100 * t2 / 257.5, h, b - t2, r - t2))

print("max measured lanes over all requests:", max(d["request"]["lanes"] for d in rF))
run(rA, sA, rF, "TIMINGS of task1-2 (the trace the precheck used) with lanes measured in task1f")
run(rF, sF, rF, "TIMINGS of task1f itself (contended; shown only to separate the effect of k from the effect of timing; do not quote)")

# how wrong is the A1 k, request by request, on the read layers
err = collections.Counter(); ab = []
for d, dl in zip(rA, rF):
    if d["request"]["type"] != "demand" or d["rows_asked"] < 1: continue
    st = sA.get(d["forward"])
    if st is None: continue
    ka = kmodel(d, st, "A1/steps", 0); km = max(d["rows_asked"], dl["request"]["lanes"])
    err[ka - km] += 1; ab.append(abs(ka - km))
n = sum(err.values())
print("\nA1/steps k minus measured k, read requests (n=%d): mean abs %.2f; exact %.0f%%; A1 over %.0f%%, under %.0f%%; histogram %s" %
      (n, S.mean(ab), 100 * err[0] / n, 100 * sum(v for k, v in err.items() if k > 0) / n, 100 * sum(v for k, v in err.items() if k < 0) / n, dict(sorted(err.items()))))
# lanes by layer group: are the hit lanes evenly spread?
byl = collections.defaultdict(lambda: [0, 0, 0])
for d in rF:
    st = sF.get(d["forward"])
    if st is None: continue
    l = d["layer"]; byl[l][0] += d["request"]["lanes"]; byl[l][1] += d["rows_asked"]; byl[l][2] += 1
lay = sorted(byl); tot_l = [byl[l][0] / nst for l in lay]
print("lanes per step by layer: min %.2f max %.2f mean %.2f; first 10 layers %.2f, last 10 layers %.2f (per layer, per step)" %
      (min(tot_l), max(tot_l), S.mean(tot_l), S.mean(tot_l[:10]), S.mean(tot_l[-10:])))
print("layers 0..39 lanes/step:", " ".join("%.1f" % x for x in tot_l))
