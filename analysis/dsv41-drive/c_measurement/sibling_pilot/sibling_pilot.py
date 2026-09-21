#!/usr/bin/env python3
"""SMT-sibling pilot runner (C_MEASUREMENT_PREREG.md section 18). Needs the GPU: run it ONLY under gpu-run.sh's flock -n, with the box guarded.

One process pinned to ONE logical CPU L (the launching thread and everything else in this process live there). A spinner process is pinned to L's SMT
sibling S and is switched on and off with SIGCONT / SIGSTOP (it stays alive, so a switch costs microseconds and no fork). Cold cells, node 0 only, n = 3 and 6,
production gather kernel, exactly the harness's RealDevice.run_visit (spin kernel before every timed launch, distinct rows, 64 scratch slots).
Per rep and n: visits A, B, B, A (A = spinner stopped, B = running). Per visit it records the spinner's achieved busy fraction, the FOREIGN busy fraction of
the sibling and of L (busy ticks from /proc/stat minus our own and the spinner's ticks), and T for 100 launches after 20 warm-up launches.
It refuses to start (exit 2) if any lane of ours is running, and records the lane count again at the end.
    taskset -c L python sibling_pilot.py --out DIR --repo REPO --launch-cpu L      (S is read from /sys thread_siblings_list)
"""
import argparse, json, os, signal, statistics as S, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "proposed_amendment6"))
import c_harness as h
import quiet_check as q

REPS = 20; NS = (3, 6); VISIT_LAUNCHES = 100; WARM_SPIN_S = 0.3

class Ticks:
    """Per-CPU busy ticks (user+nice+system) and per-process ticks from /proc, over a window."""
    hz = os.sysconf("SC_CLK_TCK")
    @staticmethod
    def cpu(c):
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu%d " % c):
                f = line.split(); return int(f[1]) + int(f[2]) + int(f[3])
        raise KeyError(c)
    @staticmethod
    def proc(pid):
        f = Path("/proc/%d/stat" % pid).read_text(); r = f[f.rindex(")") + 2:].split(); return int(r[11]) + int(r[12])
    @staticmethod
    def own_all(pids):
        return sum(Ticks.proc(p) for p in pids)

def measure(launch, sib, spinner_pid, run):
    own = [os.getpid()]
    a = (Ticks.cpu(launch), Ticks.cpu(sib), Ticks.own_all(own), Ticks.proc(spinner_pid)); t0 = time.monotonic()
    T = run(); dt = time.monotonic() - t0
    b = (Ticks.cpu(launch), Ticks.cpu(sib), Ticks.own_all(own), Ticks.proc(spinner_pid))
    hz = Ticks.hz; win = hz * dt
    spin = b[3] - a[3]; own_t = b[2] - a[2]
    foreign_l = max(0, (b[0] - a[0]) - own_t); foreign_s = max(0, (b[1] - a[1]) - spin)
    pct = lambda ticks: (min(100.0 * ticks / win, 9.9) if ticks < h.ForeignLoad.MIN_TICKS else 100.0 * ticks / win)      # the harness's own resolution floor
    return T, {"window_s": dt, "spinner_ticks": spin, "sib_spinner_pct": 100.0 * spin / win, "sib_foreign_pct": pct(foreign_s), "launch_foreign_pct": pct(foreign_l)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--launch-cpu", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="no GPU: synthetic T with no sibling effect; tests the plumbing"); a = ap.parse_args()
    lanes = [] if a.dry_run else q.lane_processes()          # (a dry run on a dev machine would match unrelated command lines)
    if lanes: print("ABORT: lanes of ours are running at the START: %s" % lanes[:3]); return 2
    L = a.launch_cpu; sibs = [c for c in h.thread_siblings(L) if c != L]
    if len(sibs) != 1: print("ABORT: cpu %d has %d SMT siblings, expected 1" % (L, len(sibs))); return 2
    Sb = sibs[0]
    if 64 <= L <= 71 or 64 <= Sb <= 71: print("ABORT: cpu %d/%d is in the reserved 64-71 band" % (L, Sb)); return 2
    if os.sched_getaffinity(0) != {L}: print("ABORT: this process must be pinned to exactly cpu %d (taskset -c %d)" % (L, L)); return 2
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spinner = subprocess.Popen(["taskset", "-c", str(Sb), sys.executable, "-c", "while True: pass"]); time.sleep(0.3); os.kill(spinner.pid, signal.SIGSTOP)
    smi = None; meta = {"launch_cpu": L, "sibling_cpu": Sb, "reps": REPS, "ns": NS, "visit_launches": VISIT_LAUNCHES, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run": a.dry_run}
    try:
        if a.dry_run:
            import random; rnd = random.Random(2)
            class Dev:
                def run_visit(self, cell, launches, args):
                    time.sleep(0.05); return [(1.1 * cell.n) * (1 + rnd.gauss(0, 0.002)) for _ in range(launches)], {}
            dev = Dev(); ns_args = argparse.Namespace()
        else:
            ns_args = argparse.Namespace(nodes=(0,), repo=a.repo); dev = h.RealDevice(a.repo); meta.update(dev.setup(ns_args))
            smi = h.SmiSampler(os.getpid()); smi.start()
            for pr in smi._procs: os.sched_setaffinity(pr.pid, set(range(0, 64)) - {L, Sb})      # nvidia-smi must not inherit our single CPU
            time.sleep(2.0)
            for _ in range(3): dev.run_visit(h.Cell("sm", "cold", 0, "idle", "eager", 6), 40, ns_args)      # untimed: link and clocks up
        w0 = time.monotonic(); rows = []
        import random; rng = random.Random(20260921)
        for rep in range(REPS):
            for n in NS:
                cell = h.Cell("sm", "cold", 0, "idle", "eager", n)
                for arm in "ABBA":
                    os.kill(spinner.pid, signal.SIGCONT if arm == "B" else signal.SIGSTOP); time.sleep(WARM_SPIN_S if arm == "B" else 0.05)
                    T, m = measure(L, Sb, spinner.pid, lambda: dev.run_visit(cell, VISIT_LAUNCHES, ns_args)[0])
                    rows.append({"rep": rep, "n": n, "arm": arm, "T_ms": T, **m})
        os.kill(spinner.pid, signal.SIGSTOP); w1 = time.monotonic()
        with (out / "visits.jsonl").open("w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        if smi:
            g = [x for x in smi.gen if w0 - 0.6 <= x[0] <= w1 + 0.6]; ap_ = [x for x in smi.apps if w0 - 0.6 <= x[0] <= w1 + 0.6]
            meta["link_gen_min"] = min(x[1] for x in g) if g else None; meta["pstates"] = sorted({x[2] for x in g}); meta["other_gpu_procs_max"] = max((len(x[1]) for x in ap_), default=0)
            meta["sm_mhz"] = [min(x[3] for x in g), max(x[3] for x in g)] if g else None
        meta["lanes_end"] = [] if a.dry_run else q.lane_processes(); meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); meta["rows"] = len(rows)
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        bad = []
        if smi and (meta["link_gen_min"] != 3): bad.append("PCIe link gen %s" % meta["link_gen_min"])
        if smi and meta["other_gpu_procs_max"] > 0: bad.append("another GPU process")
        if meta["lanes_end"]: bad.append("a lane of ours at the END: %s" % meta["lanes_end"][:2])
        if bad: (out / "INVALID").write_text("; ".join(bad) + "\n"); print("INVALID:", bad); return 3
        return 0
    finally:
        try: os.kill(spinner.pid, signal.SIGKILL)
        except OSError: pass
        if smi: smi.stop()

if __name__ == "__main__":
    sys.exit(main())
