# The ~2 requests per step that the request count showed (42.0 per graph_step line against 40 layers), traced to its cause:
# 16 graph_step lines carry steps == 2 (routed_rows 480 = 2 x 240), so 495 lines cover 511 decode steps. Recomputes the
# precheck's figures with (a) vram_miss divided by the line's `steps` when choosing k, and (b) sum(steps) as the denominator.
import sys, collections
from core import *

for name in ON + OFF:
    steps, used, eager = analyse(name)
    n_lines = len(steps); n_true = sum(s["steps"] for s in steps.values())
    dbl = [f for f, s in steps.items() if s["steps"] > 1]
    print("\n== %s: %d graph_step lines, sum(steps) = %d; %d lines with steps>1; their vram_miss mean %.1f vs %.1f for the others"
          % (name, n_lines, n_true, len(dbl), sum(steps[f]["vram_miss"] for f in dbl) / max(1, len(dbl)),
             sum(s["vram_miss"] for f, s in steps.items() if f not in dbl) / max(1, n_lines - len(dbl))))
    res = {}
    for label, fix in (("as registered (k from vram_miss per line)", False), ("k from vram_miss per step", True)):
        tot = collections.Counter(); hl = 0
        for d, st in used:
            m = d["rows_asked"]; sg = d["stages_ns"]
            t0, res_, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
            R = sorted(r["end"] / 1e6 for r in d["row_pack_ns"])
            vm = st["vram_miss"] / (st["steps"] if fix else 1)
            k = min(6, max(m, int(round(vm / LAYERS))))
            b, r, t2, _ = per_req(t0, res_, done, R, k, C)
            tot["best"] += b; tot["rand"] += r; tot["two"] += t2; hl += k - m
        res[label] = (tot, hl)
        for denom_name, den in (("495 lines", n_lines), ("%d steps" % n_true, n_true)):
            print("  %-42s /%-10s best %.2f rand %.2f two-phase %.2f ms/step; hit lanes %.1f/step; per-row over two-phase best %+.2f random %+.2f"
                  % (label, denom_name, tot["best"] / den, tot["rand"] / den, tot["two"] / den, hl / den,
                     (tot["best"] - tot["two"]) / den, (tot["rand"] - tot["two"]) / den))
    T = 1000.0 / 3.905323766402863
    print("  (T_step for shares is not changed here; the registered 257.5 ms is per token, and the client saw %d decode steps in 4 sessions)" % (4 * 127))
