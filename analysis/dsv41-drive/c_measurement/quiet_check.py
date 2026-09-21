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
REHEARSAL_MAX_BAD = 0.05     # a visit gets 3 attempts, so a per-window failure rate p costs p**3 per visit; 5% over ~470 visits is about 0.06 lost visits

def candidates(node0_cpus, node1_cpus):
    """Harness = node 0 AND within 32-63 (gpu-run.sh's mask); reader = node 1, outside 32-63, never 64-71. Membership comes from the cpulists of
    /sys/devices/system/node/node*/, which on this box are INTERLEAVED (node0 0-17,36-53; node1 18-35,54-71), never from a split point."""
    harness = [c for c in node0_cpus if 32 <= c <= 63]
    reader = [c for c in node1_cpus if c < 32 and not 64 <= c <= 71]
    return harness, reader

def rehearse(mask, spin_cores, seconds, window=0.5):
    """CPU-only dress rehearsal of the frozen gate: our own load is present on `spin_cores` (a spinner stands in for the launching thread and
    the reader), and the foreign CPU on `mask` is measured exactly as the harness measures it, per `window` seconds. Returns the list of per-window percentages."""
    import subprocess
    spinners = [subprocess.Popen(["taskset", "-c", str(c), sys.executable, "-c", "while True: pass"]) for c in spin_cores]
    try:
        fl = h.ForeignLoad([os.getpid()] + [sp.pid for sp in spinners], set(mask)); out = []; time.sleep(0.3)
        for _ in range(int(seconds / window)):
            fl.start(); time.sleep(window); out.append(fl.stop())
        return out
    finally:
        for sp in spinners: sp.kill()

def cpu_times():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split(); v = [int(x) for x in f[1:9]]
            out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))            # busy = total - idle - iowait
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--rehearse", type=float, default=0.0, help="seconds of CPU-only dress rehearsal of the frozen foreign-CPU gate on the picked cores, with a spinner as our own load")
    a = p.parse_args()
    d0, s0 = cpu_times(), h.nvme_rw(); load0 = Path("/proc/loadavg").read_text().split()[:3]
    fl = h.ForeignLoad([os.getpid()], set(range(64))); fl.start(); t0 = time.monotonic()
    top = h.TopScan([os.getpid()]).sample(min(2.0, a.seconds))
    time.sleep(max(0.0, a.seconds - (time.monotonic() - t0)))
    d1, s1 = cpu_times(), h.nvme_rw(); dt = time.monotonic() - t0; fl.stop()
    busy = {c: 100.0 * (d1[c][0] - d0[c][0]) / max(1, d1[c][1] - d0[c][1]) for c in d1}
    node0, node1 = candidates(h.node_cpus(0), h.node_cpus(1))
    print("candidates from /sys: harness (node 0 within 32-63) %s ; reader (node 1 outside 32-63, not 64-71) %s" % (node0, node1))
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
    print("IDLE-CORE CHECK (informational, my own margin): %s (every picked core under 5%% busy and the NVMe drives under %.2f GB/s)" % ("GO" if ok else "NO-GO", h.IDLE_DRIVE_MAX_GBS))
    if a.rehearse > 0:
        hm, rm = harness[:2], reader[:3]
        res_h = rehearse(hm, hm[:1], a.rehearse); res_r = rehearse(rm, rm[:1], a.rehearse)
        bad_h = sum(1 for x in res_h if x[0] > h.ENV_FOREIGN_PCT) / max(1, len(res_h)); bad_r = sum(1 for x in res_r if x[0] > h.ENV_FOREIGN_PCT) / max(1, len(res_r))
        print("REHEARSAL (the frozen gate as the harness measures it, our own load present): harness mask %s: %.0f%% of %d windows above %.0f%%; reader mask %s: %.0f%% of %d windows"
              % (hm, 100 * bad_h, len(res_h), h.ENV_FOREIGN_PCT, rm, 100 * bad_r, len(res_r)))
        ok = bad_h <= REHEARSAL_MAX_BAD and bad_r <= REHEARSAL_MAX_BAD and (rd + wr) < h.IDLE_DRIVE_MAX_GBS
        print("GO" if ok else "NO-GO", "(at most %.0f%% of rehearsal windows above the gate on either mask, NVMe idle)" % (100 * REHEARSAL_MAX_BAD))
    else:
        print("no rehearsal requested: the idle-core check above is not the registered GO; run with --rehearse 40")
    print("then: gpu-run.sh taskset -c %s <python> c_harness.py ... --reader-cpus %s" % (",".join(map(str, harness[:2])), ",".join(map(str, reader[:3]))))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
