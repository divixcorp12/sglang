"""Summary of a memsample.py run: load time, node free against the low watermark, swap and faults per phase, which
node reclaims, the scheduler's VmLck and team-thread major faults, fincore totals. Usage: memsum.py RUN_DIR SAMPLES"""
import glob
import json
import re
import sys

run, path = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(path)]
samples = [r for r in recs if "nodes" in r]
extra = [r for r in recs if "nodes" not in r]
(vp,) = glob.glob(f"{run}/events.*.verify.jsonl")
lines = open(vp).read().splitlines()
hdr = json.loads(lines[0])
off = hdr["epoch_ns"] - hdr["monotonic_ns"]
ns = [json.loads(l)["ns"] for l in lines[1:]]
d0, d1 = (min(ns) + off) / 1e9, (max(ns) + off) / 1e9
t0 = samples[0]["t"]
log = open(f"{run}/server.log", errors="replace").read()
stamp = lambda pat: re.search(r"\[(\d+-\d+-\d+ [\d:]+)\][^\n]*" + pat, log)
begin, ready = stamp("Load weight begin"), stamp("fired up and ready")
print(f"samples {len(samples)} over {samples[-1]['t'] - t0:.0f} s; decode +{d0 - t0:.0f}..+{d1 - t0:.0f} s; "
      f"load: first 'Load weight begin' {begin.group(1) if begin else '-'} -> ready {ready.group(1) if ready else '-'}")


def phase(name, lo, hi):
    ss = [s for s in samples if lo <= s["t"] <= hi]
    if len(ss) < 2:
        print(name, "no samples")
        return
    a, b = ss[0], ss[-1]
    print(f"== {name}: {len(ss)} samples, {b['t'] - a['t']:.0f} s")
    for n in ("0", "1"):
        fr = [s["zones"][n]["free"] for s in ss]
        z = ss[0]["zones"][n]
        print(f"  node {n}: zone free min {min(fr)} pages ({min(fr) * 4 / 2**20:.2f} GiB) vs low {z['low']} min {z['min']}; "
              f"FilePages {a['nodes'][n]['FilePages'] / 2**20:.1f}->{b['nodes'][n]['FilePages'] / 2**20:.1f} GiB; "
              f"AnonPages {a['nodes'][n]['AnonPages'] / 2**20:.1f}->{b['nodes'][n]['AnonPages'] / 2**20:.1f} GiB; "
              f"Mlocked {b['nodes'][n].get('Mlocked', 0) / 2**20:.1f} GiB")
    print("  vmstat deltas:", {k: b["vmstat"][k] - a["vmstat"][k] for k in a["vmstat"]})
    print("  kswapd ticks:", {k: b["kswapd"][k] - a["kswapd"].get(k, 0) for k in b["kswapd"]},
          "swap used GiB", round(a["swap_used_kb"] / 2**20, 2), "->", round(b["swap_used_kb"] / 2**20, 2),
          "max", round(max(s["swap_used_kb"] for s in ss) / 2**20, 2))
    if a.get("disk") and b.get("disk"):
        span = b["t"] - a["t"]
        print("  disk:", {dev: f"read {(b['disk'][dev][1] - a['disk'][dev][1]) * 512 / 2**30:.2f} GiB, "
                              f"busy {(b['disk'][dev][7] - a['disk'][dev][7]) / 10 / span:.0f}%"
                         for dev in b["disk"] if dev in a["disk"]})
    sa, sb = a.get("scheduler"), b.get("scheduler")
    if sa and sb and sa["pid"] == sb["pid"]:
        print("  scheduler", sb["pid"], "VmLck GiB", round((sb["vmlck_kb"] or 0) / 2**20, 2),
              "team majflt deltas", {k: sb["teams"][k]["majflt"] - sa["teams"].get(k, {}).get("majflt", 0)
                                     for k in sb["teams"]})


phase("load (first sample -> first verify)", t0, d0)
phase("decode (first -> last verify)", d0, d1)
phase("whole run", t0, samples[-1]["t"])
maps = [e for e in extra if "numa_maps" in e]
if maps:
    pick = min(maps, key=lambda e: abs(e["t"] - (d0 + d1) / 2))  # the one nearest mid-decode
    print(f"numa_maps of the scheduler (+{pick['t'] - t0:.0f} s), GiB per node:")
    for m in pick["numa_maps"][:12]:
        print(f"  {m['name'][:90]:90s} " + " ".join(f"N{k}={v * 4 / 2**20:.2f}" for k, v in sorted(m["pages"].items())))
for e in extra:
    if "numastat" in e:
        print(f"numastat (+{e['t'] - t0:.0f} s):\n{e['numastat']}")
    if "fincore" in e:
        top = sorted(e["files"], key=lambda x: -x["cached"])[:5]
        print(f"fincore {e['fincore']} (+{e['t'] - t0:.0f} s): total {e['total_cached'] / 2**30:.2f} GiB "
              f"{ {k: round(v / 2**30, 2) for k, v in e['by_group'].items()} }; FilePages kB {e['file_pages_kb']}; "
              f"top {[(x['file'].rsplit('/', 2)[-1], round(x['cached'] / 2**30, 2)) for x in top]}")
