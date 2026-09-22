import json, collections, statistics as S, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
name = "task1-2-new-on-T"
steps, used, eager = analyse(name)
T = 1000 / (sum(json.load(open(D + n + ".json"))["mean_decode_tok_s"] for n in ("task1-3-new-on-U","task1-6-new-on-U","task1c-3-new-on-U")) / 3)
n = len({d["forward"] for d, _ in used})
# table[idx][hh] = (best, rand, two) for hh = 0 .. 6-m
table = []; hs = []
for d, st in used:
    m = d["rows_asked"]; sg = d["stages_ns"]
    t0, res, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
    R = sorted(r["end"] / 1e6 for r in d["row_pack_ns"])
    h = min(6, max(m, int(round(st["vram_miss"] / LAYERS)))) - m
    hs.append(h)
    row = []
    for hh in range(0, h + 2):            # one beyond the modelled h so interpolation is defined; cap at 6 lanes below
        k = m + hh
        if k > 6: row.append(row[-1]); continue
        b, r, t2, _ = per_req(t0, res, done, R, k, C)
        row.append((b, r, t2))
    table.append(row)
def at(f):
    tot = [0.0, 0.0, 0.0]
    for row, h in zip(table, hs):
        x = f * h; lo = int(x); fr = x - lo
        a = row[lo]; b = row[min(lo + 1, len(row) - 1)]
        for i in range(3): tot[i] += (1 - fr) * a[i] + fr * b[i]
    return [v / n for v in tot]
def credited(f): return sum(f * h for h in hs) / n
# A1-free bounds
lo_s = hi_s = 0.0; cnt = 0
for st in steps.values():
    lr = st.get("layer_ram_rows")
    if not lr: continue
    H = st["vram_miss"] - sum(lr); n0 = sum(1 for x in lr if x == 0)
    lo_s += max(0, H - 6 * n0); hi_s += min(max(H, 0), sum(6 - x for x in lr if x > 0)); cnt += 1
print("hit lanes per step: total %.1f; A1 model credits %.1f; A1-free bounds on those in read layers: min %.1f  max %.1f" %
      (S.mean(st["vram_miss"] - sum(st["layer_ram_rows"]) for st in steps.values()), credited(1.0), lo_s / cnt, hi_s / cnt))
print("T_step %.1f; f  credited/step  rand ms (rand-L)/T   two-phase ms (two-L)/T   best ms" % T)
for f in (0, .05, .1, .15, .2, .25, .3, .37, .5, .75, 1.0):
    b, r, t2 = at(f)
    print("%.2f  %5.1f   %6.2f  %5.2f%%     %6.2f  %5.2f%%    %6.2f" % (f, credited(f), r, 100 * (r - L) / T, t2, 100 * (t2 - L) / T, b))
def cross(idx, target):
    a, b = 0.0, 1.0
    for _ in range(40):
        mid = (a + b) / 2
        if (at(mid)[idx] - L) / T >= target: b = mid
        else: a = mid
    return b
fr = cross(1, .03); ft3 = cross(2, .03); ft15 = cross(2, .015)
print("floor f: registered SUPPORT (random order, 3%% after launch) = %.3f (%.1f hit lanes/step, %.2f per read request); two-phase >= 3%% = %.3f (%.1f/step); two-phase >= 1.5%% = %.3f (%.1f/step)" %
      (fr, credited(fr), fr * sum(hs) / len(hs), ft3, credited(ft3), ft15, credited(ft15)))
print("mean modelled hits per read request %.2f" % (sum(hs) / len(hs)))
