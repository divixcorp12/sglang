# Independent recomputation of the Task 6 precheck. Does not import per_row_precheck.py.
import itertools, json, math, sys, collections, statistics as S
D = "/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-drive/task1-results/"
ON = ["task1-2-new-on-T", "task1b-0-new-on-T", "task1c-0-new-on-T"]
OFF = ["task1c-2-new-off-T"]
C = 1.055; L = 1.0; LAYERS = 40

def load(name):
    steps, reqs, eager = {}, [], []
    for line in open(D + name + ".trace"):
        d = json.loads(line)
        k = d.get("kind")
        if k == "graph_step": steps[d["forward"]] = d
        elif k == "ram_miss_request": reqs.append(d)
        else: eager.append(d)
    return steps, reqs, eager

def chain(order, ready, c, t0):
    e = t0
    for i in order:
        e = max(e, ready[i]) + c
    return e

def per_req(t0, res, done, R, k, c):
    """exact (no sampling) savings in ms: best, rand (mean over ALL permutations), twophase, hits_only(no spread)."""
    m = len(R); h = k - m
    ready = [res] * h + list(R)
    tb = done + k * c
    best_order = sorted(range(k), key=lambda i: (ready[i], i))
    best = tb - chain(best_order, ready, c, t0)
    perms = list(itertools.permutations(range(k)))
    rand = tb - sum(chain(p, ready, c, t0) for p in perms) / len(perms)
    # two-phase: hit lanes copied as one batch from RES; the rest as one batch after DONE
    hit_end = (max(t0, res) + h * c) if h > 0 else 0.0
    t2 = max(done, hit_end) + m * c
    two = tb - t2
    # closed form check (M = DONE virtual): min over positions in the best order
    rs = [ready[i] for i in best_order]
    cf = min((done - rs[j]) + j * c for j in range(k))   # j is 0-based: (j-1) in 1-based
    return best, rand, two, cf

def analyse(name, fscale=None):
    steps, reqs, eager = load(name)
    used = []
    for d in reqs:
        if d["request"]["type"] != "demand" or d["rows_asked"] < 1 or d["status"] != "served": continue
        if d["untraced"]["rows"] != 0: continue
        st = steps.get(d["forward"])
        if st is None or len(d["row_pack_ns"]) != d["rows_asked"]: continue
        used.append((d, st))
    return steps, used, eager

def tstep(names):
    v = [json.load(open(D + n + ".json"))["mean_decode_tok_s"] for n in names]
    return 1000.0 / (sum(v) / len(v)), v

if __name__ == "__main__":
    t_all, v_all = tstep(["task1-3-new-on-U", "task1-6-new-on-U", "task1c-3-new-on-U"])
    t_clean, v_clean = tstep(["task1-2-new-on-T", "task1-3-new-on-U", "task1b-0-new-on-T", "task1c-0-new-on-T"])
    print("T_step registered denominators %.1f ms (tok/s %s); with the four clean on-arms %.1f ms" % (t_all, [round(x,4) for x in v_all], t_clean))
    for name in ON + OFF:
        steps, used, eager = analyse(name)
        nsteps = len({d["forward"] for d, _ in used})
        tot = collections.Counter(); m_hist = collections.Counter(); hit_lanes = 0; sum_m1 = 0
        for d, st in used:
            m = d["rows_asked"]; sg = d["stages_ns"]
            t0, res, done = sg["observed"] / 1e6, sg["reserved"] / 1e6, sg["done"] / 1e6
            R = sorted(r["end"] / 1e6 for r in d["row_pack_ns"])
            k = min(6, max(m, int(round(st["vram_miss"] / LAYERS))))
            b, r, t2, cf = per_req(t0, res, done, R, k, C)
            b0, r0, t20, cf0 = per_req(t0, res, done, R, m, C)     # miss rows only (h = 0)
            g, _, _, _ = per_req(t0, res, done, R, max(6, m), C)    # generous
            tot["best"] += b; tot["rand"] += r; tot["two"] += t2; tot["rand0"] += r0; tot["best0"] += b0; tot["gen"] += g
            tot["cf_minus_best"] = max(tot["cf_minus_best"], abs(cf - b))
            tot["neg"] += (b < -1e-9) + (r < -1e-9) * 0
            hit_lanes += k - m; sum_m1 += m - 1; m_hist[m] += 1
        n = nsteps
        print("\n== %s used %d requests over %d steps; m hist %s; hit lanes credited %.1f/step; sum(m-1)*c/step = %.2f ms" %
              (name, len(used), n, dict(sorted(m_hist.items())), hit_lanes / n, sum_m1 * C / n))
        for key in ("best", "rand", "two", "rand0", "best0", "gen"):
            print("   %-6s %.2f ms/step" % (key, tot[key] / n))
        print("   closed-form vs simulated BEST, max abs diff per request: %.2e ms; negative BEST savings: %d" % (tot["cf_minus_best"], tot["neg"]))
        print("   per-row over two-phase: best %.2f ms/step, random %.2f ms/step" % ((tot["best"] - tot["two"]) / n, (tot["rand"] - tot["two"]) / n))
        T = t_all if name in ON else 1000 / 2.9039
        print("   share of step %.1f ms: rand %.2f%%  (rand-L) %.2f%%  best %.2f%%  two-phase %.2f%%  gen %.2f%%  rand0/rand %.3f  two/rand %.3f" %
              (T, 100 * tot["rand"] / n / T, 100 * (tot["rand"] / n - L) / T, 100 * tot["best"] / n / T, 100 * tot["two"] / n / T, 100 * tot["gen"] / n / T,
               tot["rand0"] / tot["rand"], tot["two"] / tot["rand"]))
