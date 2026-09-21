# Replaces the precheck's A1 (k = min(6, round(vram_miss/40))) with the measured planned lane count of a
# schema-4 trace (request.lanes, 4e63616666). Usage: python3 schema4_lanes.py <trace file>
# Refuses a trace whose lines are not schema 4. Does not import per_row_precheck.py.
import sys, random, collections, itertools
from core import load, chain, C, L

def rand_mean(ready, c, t0, k, seed):
    if k <= 6:
        perms = itertools.permutations(range(k)); n = 0; tot = 0.0
        for p in perms: tot += chain(p, ready, c, t0); n += 1
        return tot / n
    rng = random.Random(seed); tot = 0.0
    for _ in range(64):
        p = list(range(k)); rng.shuffle(p); tot += chain(p, ready, c, t0)
    return tot / 64

def main(path):
    steps, reqs, _ = {}, [], None
    import json
    for line in open(path):
        d = json.loads(line); kd = d.get("kind")
        if kd == "graph_step": steps[d["forward"]] = d
        elif kd == "ram_miss_request":
            if d.get("schema", 0) < 4 or "lanes" not in d["request"]:
                sys.exit("not a schema-4 trace: cannot answer the per-layer question")
            reqs.append(d)
    dec = [d for d in reqs if d["forward"] in steps]
    nst = sum(s.get("steps", 1) for s in steps.values())   # decode steps, NOT graph_step lines: a line can cover 2 steps (see merged_steps.py)
    print("decode requests %d of %d, graph_step lines %d covering %d decode steps, dropped_before nonzero %d" %
          (len(dec), len(reqs), len(steps), nst, sum(1 for d in reqs if d.get("dropped_before"))))
    # 1. the sum check that decides whether `lanes` means what the commit says
    sl = sum(d["request"]["lanes"] for d in dec); sr = sum(d["rows_asked"] for d in dec)
    sv = sum(s["vram_miss"] for s in steps.values()); sm = sum(s["ram_miss"] for s in steps.values())
    print("sum lanes %d vs sum vram_miss %d ; sum rows_asked %d vs sum ram_miss %d" % (sl, sv, sr, sm))
    # 2. where the hit lanes are
    read = [d for d in dec if d["rows_asked"] > 0]; other = [d for d in dec if d["rows_asked"] == 0]
    hit_read = sum(d["request"]["lanes"] - d["rows_asked"] for d in read)
    hit_other = sum(d["request"]["lanes"] for d in other)
    print("hit lanes per step: in read layers %.1f, in layers that read nothing %.1f (total %.1f); read requests %.1f/step" %
          (hit_read / nst, hit_other / nst, (hit_read + hit_other) / nst, len(read) / nst))
    print("lanes histogram (read requests):", dict(sorted(collections.Counter(d["request"]["lanes"] for d in read).items())))
    neg = sum(1 for d in read if d["request"]["lanes"] < d["rows_asked"])
    print("read requests with lanes < rows_asked (would break hits = lanes - rows): %d" % neg)
    # 3. the model with measured k
    tot = collections.Counter()
    for d in read:
        st = steps[d["forward"]]; sg = d["stages_ns"]
        if d["status"] != "served" or d["untraced"]["rows"] != 0 or len(d["row_pack_ns"]) != d["rows_asked"]: continue
        m = d["rows_asked"]; k = max(m, d["request"]["lanes"]); h = k - m
        t0, res, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
        R = sorted(r["end"] / 1e6 for r in d["row_pack_ns"])
        ready = [res] * h + R; tb = done + k * C
        best = tb - chain(sorted(range(k), key=lambda i: (ready[i], i)), ready, C, t0)
        rnd = tb - rand_mean(ready, C, t0, k, d["seq"] if "seq" in d else 0)
        hit_end = (max(t0, res) + h * C) if h else 0.0
        two = tb - (max(done, hit_end) + m * C)
        tot["best"] += best; tot["rand"] += rnd; tot["two"] += two
    for kname in ("best", "rand", "two"):
        print("   %-5s %.2f ms/step" % (kname, tot[kname] / nst))
    print("   per-row over two-phase: best %.2f, random %.2f ms/step; compare the A1 model, same denominator: 37.40 / 18.57 / 32.48 (per 511 steps; 38.61 / 19.17 / 33.53 per 495 lines)" %
          ((tot["best"] - tot["two"]) / nst, (tot["rand"] - tot["two"]) / nst))

main(sys.argv[1])
