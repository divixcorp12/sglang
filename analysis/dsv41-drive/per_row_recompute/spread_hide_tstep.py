import sys, json, statistics as S
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
steps, used, eager = analyse("task1-2-new-on-T")
sp = []; hide = 0; nn = 0; waits = []; Tsum = 0
for d, st in used:
    m = d["rows_asked"]; sg = d["stages_ns"]
    t0, res, done = sg["observed"]/1e6, sg["reserved"]/1e6, sg["done"]/1e6
    R = sorted(r["end"]/1e6 for r in d["row_pack_ns"])
    k = min(6, max(m, int(round(st["vram_miss"]/LAYERS)))); h = k - m
    if m >= 2: sp.append(R[-1] - R[0])
    hide += (done - t0 >= h * C); nn += 1
    waits.append(done - t0)
sp.sort()
print("spread p10/p50/p90 %.2f %.2f %.2f ms; frac>=c %.3f; hide_ok %.3f; read-wait p50 %.2f" % (sp[len(sp)//10], sp[len(sp)//2], sp[9*len(sp)//10], sum(1 for x in sp if x >= C)/len(sp), hide/nn, sorted(waits)[len(waits)//2]))
# hit copies finish before done? two-phase hits end vs done
late = 0
for d, st in used:
    m = d["rows_asked"]; sg = d["stages_ns"]
    t0, res, done = sg["observed"]/1e6, sg["reserved"]/1e6, sg["done"]/1e6
    k = min(6, max(m, int(round(st["vram_miss"]/LAYERS)))); h = k - m
    if h and res + h*C > done: late += 1
print("requests whose hit copies would outlast DONE (reserved + h*c > done): %.3f" % (late/len(used)))
# T_step variants
for label, names in (("registered (incl. INVALID task1-6 and disturbed task1c-3)", ["task1-3-new-on-U","task1-6-new-on-U","task1c-3-new-on-U"]),
                     ("clean on-arms only", ["task1-2-new-on-T","task1-3-new-on-U","task1b-0-new-on-T","task1c-0-new-on-T"]),
                     ("untraced clean only", ["task1-3-new-on-U"])):
    v = [json.load(open(D+n+".json"))["mean_decode_tok_s"] for n in names]
    print("T_step %s: %.1f ms" % (label, 1000/(sum(v)/len(v))))
# session-specific T from the untraced clean arm, per-session tok/s
ps = [r["decode_tok_s"] for r in json.load(open(D+"task1-3-new-on-U.json"))["per_session"]]
print("per-session step time (task1-3): %s ms" % [round(1000/x,1) for x in ps])
# traced arm's own step time (tracing inflates)
tr = json.load(open(D+"task1-2-new-on-T.json"))["mean_decode_tok_s"]
print("traced arm task1-2 mean tok/s %.4f -> %.1f ms/step" % (tr, 1000/tr))
