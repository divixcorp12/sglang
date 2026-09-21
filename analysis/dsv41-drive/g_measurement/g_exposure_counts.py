# How many extra stage triples V2 (per-row, S = 6 stages) has over V1 (two-phase, S = 2), per decode step, split by
# whether they can be hidden behind an NVMe read. From the MEASURED lane counts of a schema-4 trace (default: task1f).
# CPU only. Usage: python3 g_exposure_counts.py [trace]   (path defaults to the divix01 task1f trace)
import json, sys, collections
P = sys.argv[1] if len(sys.argv) > 1 else "/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-drive/task1-results/task1f-0-new-on-T.trace"
L = [json.loads(l) for l in open(P)]
st = {d["forward"]: d for d in L if d.get("kind") == "graph_step"}
reqs = [d for d in L if d.get("kind") == "ram_miss_request" and d["forward"] in st]
NST = sum(s["steps"] for s in st.values())          # decode steps (511), NOT graph_step lines (495)
S_V1, S_V2 = 2, 6
c = collections.Counter()
for d in reqs:
    k = d["request"]["lanes"]; m = d["rows_asked"]; h = k - m
    assert 0 <= k <= S_V2
    v1_active = (h > 0) + (m > 0)                     # stage [0,h) hits, stage [h,k) the rest; an all-hit layer is one active stage
    ex_active = k - v1_active                         # V2 has k active stages
    ex_empty = (S_V2 - k) - (S_V1 - v1_active)
    assert ex_active + ex_empty == S_V2 - S_V1
    if m == 0:                                        # a layer that reads nothing: nothing to hide behind
        c["B_nonread_extra_active"] += ex_active; c["A_nonread_extra_empty"] += ex_empty
    else:
        c["C_read_extra_hit_stages"] += max(0, h - 1); c["D_read_extra_miss_stages"] += m - 1; c["A_read_extra_empty"] += ex_empty
out = {k: v / NST for k, v in c.items()}
E_e = out["A_nonread_extra_empty"] + out["A_read_extra_empty"]
res = {"decode_steps": NST, "layers_per_step": len(reqs) / NST,
       "E_empty (A: tail empty stages, always exposed)": E_e,
       "E_B (extra active stages in layers that read nothing, exposed)": out["B_nonread_extra_active"],
       "E_C (extra hit stages in read layers: hidden if the hit chain fits in the read wait)": out["C_read_extra_hit_stages"],
       "E_D (extra miss-row stages in read layers, counted exposed = upper bound)": out["D_read_extra_miss_stages"]}
tot = E_e + out["B_nonread_extra_active"] + out["C_read_extra_hit_stages"] + out["D_read_extra_miss_stages"]
res["total extra triples per step (the plan's 160)"] = tot
for k, v in res.items(): print("%-95s %s" % (k, ("%.2f" % v) if isinstance(v, float) else v))
