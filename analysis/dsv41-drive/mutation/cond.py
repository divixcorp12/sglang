"""Run at launch on divix01: per-core busy over a 3 s window for cpus 0-71, loadavg, sha and host.cpp md5s -> launch-conditions.json"""
import json, sys, time, hashlib, datetime, os
D = "/data/models/slang/nvfp4-work/t2-mutants"
def snap():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); v = list(map(int, f[1:9]))
            idle = v[3] + v[4]; out[int(f[0][3:])] = (sum(v) - idle, sum(v))
    return out
a = snap(); time.sleep(3); b = snap()
busy = {c: round(100.0 * (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]), 1) for c in a}
md5 = {}
for t in ("base", "tree1x", "base2", "tree2b", "base3", "tree3"):
    p = f"{D}/{t}/python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp"
    md5[t] = hashlib.md5(open(p, "rb").read()).hexdigest()
rec = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "base3_sha": open(f"{D}/base3.sha").read().strip(), "loadavg": open("/proc/loadavg").read().split()[:3],
       "host_cpp_md5": md5, "cpu_busy_pct_3s": busy,
       "chain_cores_0_63_mean_busy_pct": round(sum(busy[c] for c in range(64)) / 64, 1),
       "cores_64_71_busy_pct": {c: busy[c] for c in range(64, 72)},
       "note": "cores 0-63 are the chain's; 64-71 are production's and must stay free; timings per mutant depend on this floor"}
json.dump(rec, open(f"{D}/launch-conditions.json", "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("utc", "base3_sha", "loadavg", "chain_cores_0_63_mean_busy_pct", "cores_64_71_busy_pct")}))
