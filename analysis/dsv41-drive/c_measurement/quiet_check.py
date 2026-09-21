#!/usr/bin/env python3
"""Pre-flight for the c window: is the box quiet enough, and on which cores? CPU only, reads /proc, touches no GPU and no drive.

    python3 quiet_check.py [--seconds 10]

Samples per-core busy time (/proc/stat), the NVMe read rate (/proc/diskstats) and the load average. It picks the quietest cores for
the harness (node 0, inside 32-63, because gpu-run.sh pins there and the GPU is on node 0) and for the reader (node 1, outside 32-63
and never 64-71), and prints GO only if every picked core is under 5% busy and the NVMe drives are idle (under 0.02 GB/s: the user's services live on /data, an LVM volume,
not on those drives). The frozen gate is 10% foreign CPU on the cores we use; this asks for half of that so the run has margin. The user's
permanent services are the STEADY-STATE condition: their CPU is recorded here and per cell and is not a reason for NO-GO.
"""
import argparse, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_harness as h

def cpu_times():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); v = [int(x) for x in f[1:9]]
            out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))            # busy = total - idle - iowait
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--seconds", type=float, default=10.0); a = p.parse_args()
    d0, s0 = cpu_times(), h.nvme_rw(); load0 = Path("/proc/loadavg").read_text().split()[:3]
    fl = h.ForeignLoad([os.getpid()], set(range(64))); fl.start(); t0 = time.monotonic()
    top = h.TopScan([os.getpid()]).sample(min(2.0, a.seconds))
    time.sleep(max(0.0, a.seconds - (time.monotonic() - t0)))
    d1, s1 = cpu_times(), h.nvme_rw(); dt = time.monotonic() - t0; fl.stop()
    busy = {c: 100.0 * (d1[c][0] - d0[c][0]) / max(1, d1[c][1] - d0[c][1]) for c in d1}
    node0 = [c for c in h.node_cpus(0) if 32 <= c <= 63]; node1 = [c for c in h.node_cpus(1) if c < 32]
    harness = sorted(sorted(node0, key=lambda c: busy[c])[:4]); reader = sorted(sorted(node1, key=lambda c: busy[c])[:4])
    rd = sum(s1[k][0] - s0.get(k, (0, 0))[0] for k in s1) / 1e9 / dt; wr = sum(s1[k][1] - s0.get(k, (0, 0))[1] for k in s1) / 1e9 / dt
    print("loadavg %s over %.0f s" % (" ".join(load0), dt))
    print("foreign CPU box-wide: %.2f cores (an upper bound: kernel worker time is included). The user's permanent services are the condition, not a NO-GO reason." % fl.last["box_foreign_cores"])
    for t in top: print("  %-16s %6.1f%%  pid %d  %s" % (t["comm"], t["cpu_pct"], t["pid"], t["cmdline"][:110]))
    print("NVMe traffic (all whole devices): read %.4f GB/s, write %.4f GB/s (limit %.2f for an idle cell)" % (rd, wr, h.IDLE_DRIVE_MAX_GBS))
    print("cores busy % (0-63): " + " ".join("%d:%.0f" % (c, busy[c]) for c in sorted(busy) if c < 64))
    print("harness cores (node 0, within 32-63), quietest 4: %s -> busy %s" % (harness, [round(busy[c]) for c in harness]))
    print("reader cores (node 1, outside 32-63 and 64-71), quietest 4: %s -> busy %s" % (reader, [round(busy[c]) for c in reader]))
    ok = all(busy[c] < 5.0 for c in harness) and all(busy[c] < 5.0 for c in reader) and (rd + wr) < h.IDLE_DRIVE_MAX_GBS
    print("GO" if ok else "NO-GO", "(every picked core under 5%% busy and the NVMe drives under %.2f GB/s)" % h.IDLE_DRIVE_MAX_GBS)
    print("then: gpu-run.sh taskset -c %s <python> c_harness.py ... --reader-cpus %s" % (",".join(map(str, harness)), ",".join(map(str, reader))))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
