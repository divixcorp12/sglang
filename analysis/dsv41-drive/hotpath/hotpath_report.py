"""Task 18's arms report (plan 2026-09-29-hotpath-zero-overhead): A (master), B (branch), A2 (master) and C (branch
under the counting shim, not timed), from drive_hotpath_arms.sh's out dir.

Usage: hotpath_report.py <out_dir> <tier text> <perf_event_paranoid> A=<run_dir> B=<run_dir> A2=<run_dir> [C=<run_dir>]
(PYTHONPATH: analysis/dsv41-drive/mirror3 and analysis/dsv41-drive/iopoll-cuts). Writes <out_dir>/arms-report.json and
exits non-zero unless the pass condition holds: B and A2 byte-identical to A (and C, when present), read_errors == 0 in
every arm, B's pooled ms/token <= mean(A, A2) + max(1.5, |A2 - A|), and, when C ran, C's whole-run service- and
copy-thread malloc, free, mutex and cond counts all 0.

Per arm: mirror3_report.decode (session and pooled ms/token, TTFT), identity against A, the shutdown line's served,
rows_read and read_errors, the exl3-ram-miss thread's CPU seconds over the timed window (thread_sampler.report), and
the perf CSV (from "fired up" to the arm's end) divided by the lifetime `served` count. The two do not cover the same
span: `served` also counts requests served before "fired up" (the server's own warm-up), so the per-request figures
are a slight over-estimate, identically in every arm."""

import datetime
import json
import statistics
import sys
from pathlib import Path

import mirror3_report as m
import thread_sampler as ts

TIMED = ("A", "B", "A2")
ZERO_KINDS = ("malloc", "free", "mutex", "cond")


def perf(path: Path) -> dict | None:
    """perf stat -x, output: value,unit,event,run-time,pct,...; '<not counted>'/'<not supported>' stay strings."""
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 3:
            continue
        try:
            out[f[2]] = int(float(f[0]))
        except ValueError:
            out[f[2]] = f[0]
    return out


def sched(path: Path, served, start, end) -> dict | None:
    """The service thread's /proc switch and migration counters (the driver's sched_sample): the delta over every
    sample (from "fired up" to the thread's last sample) per served request, and the delta over the timed window."""
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = ("voluntary", "nonvoluntary", "migrations")

    def delta(rs):
        if len(rs) < 2:
            return None
        return {k: (rs[-1][k] - rs[0][k]) if rs[0][k] is not None and rs[-1][k] is not None else None for k in keys}

    inside = [r for r in rows if start <= datetime.datetime.fromisoformat(r["utc"]) <= end]
    whole = delta(rows)
    return {"samples": len(rows), "whole": whole,
            "whole_per_served_request": {k: round(v / served, 3) for k, v in whole.items() if v is not None}
            if whole and served else None,
            "timed_window": delta(inside), "timed_window_samples": len(inside)}


def arm(out_dir: Path, name: str, run_dir: str) -> dict:
    start, end = m.timed_window_utc(run_dir)
    rm = m.ram_miss(run_dir)
    thread = rm["thread"] or {}
    served = thread.get("served")
    counts = perf(out_dir / f"{name}-perf.csv")
    per_request = None
    if counts and served:
        per_request = {k: round(v / served, 3) for k, v in counts.items() if isinstance(v, int)}
    threads_path = out_dir / f"{name}-threads.jsonl"
    return {
        "run_dir": run_dir,
        "decode": m.decode(run_dir),
        "ms_per_token_median_turn": m.ms_per_token(run_dir)[1],
        "clocks": {"timed_window": m.clock_summary(out_dir / f"{name}-clocks.csv", start, end),
                   "session_start_end_mhz": m.session_clocks(run_dir)},
        "ram_miss": rm,
        "served": served, "rows_read": thread.get("rows_read"), "read_errors": thread.get("read_errors"),
        "threads": ts.report(threads_path, start, end) if threads_path.exists() else None,
        "perf": counts, "perf_per_served_request": per_request,
        "sched": sched(out_dir / f"{name}-sched.jsonl", served, start, end),
        "env_diff": json.loads((out_dir / f"{name}-env-diff.json").read_text())
        if (out_dir / f"{name}-env-diff.json").exists() else None,
    }


def shim(out_dir: Path) -> dict:
    paths = sorted(out_dir.glob("C-shim.json*"))
    dumps = [json.loads(p.read_text()) for p in paths]
    if len(dumps) != 1:
        return {"dumps": dumps, "files": [str(p) for p in paths],
                "ok": False, "why": f"expected exactly one dump (the scheduler's), found {len(dumps)}"}
    d = dumps[0]
    seen_ok = d["threads"].get("service", 0) >= 1 and d["threads"].get("copy", 0) >= 1
    zeros = {th: {k: d[th][k] for k in ZERO_KINDS} for th in ("service", "copy")}
    ok = seen_ok and all(v == 0 for th in zeros.values() for v in th.values())
    return {**d, "file": str(paths[0]), "ok": ok,
            "why": None if ok else ("a tracked thread was never recognized" if not seen_ok
                                    else f"nonzero malloc/free/mutex/cond: {zeros}")}


def main(argv: list[str]) -> int:
    out_dir, tier, paranoid = Path(argv[1]), argv[2], int(argv[3])
    runs = dict(kv.split("=", 1) for kv in argv[4:])
    arms = {name: arm(out_dir, name, run) for name, run in runs.items()}
    ref = runs["A"]
    identity = {n: m.identity(ref, r) for n, r in runs.items() if n != "A"}
    out = {"tier": tier, "perf_event_paranoid": paranoid, "arms": arms,
           "identical_to_A": {n: all(v.values()) for n, v in identity.items()},
           "differing_turns": {n: [k for k, same in v.items() if not same] for n, v in identity.items()}}
    pooled = {n: arms[n]["decode"]["pooled_ms_per_token"] for n in TIMED if n in arms}
    ok = True
    if all(n in pooled for n in TIMED):
        base = statistics.mean([pooled["A"], pooled["A2"]])
        drift = pooled["A2"] - pooled["A"]
        allowance = max(1.5, abs(drift))
        out.update(baseline_mean_A_A2=base, drift_A2_minus_A=drift, delta_B=pooled["B"] - base,
                   allowance_ms=allowance, b_limit_ms=base + allowance, b_not_slower=pooled["B"] <= base + allowance)
        ok &= out["b_not_slower"]
    else:
        ok = False
    out["all_identical"] = all(out["identical_to_A"].values())
    out["read_errors_all_zero"] = all(a["read_errors"] == 0 for a in arms.values())
    ok &= out["all_identical"] and out["read_errors_all_zero"]
    if "C" in runs:
        out["C_shim"] = shim(out_dir)
        ok &= out["C_shim"]["ok"]
    out["pass"] = bool(ok)
    (out_dir / "arms-report.json").write_text(json.dumps(out, indent=2, default=str))
    print("pooled ms/token", {n: round(v, 2) for n, v in pooled.items()},
          "delta_B", round(out.get("delta_B", float("nan")), 2), "drift", round(out.get("drift_A2_minus_A", float("nan")), 2),
          "identical", out["identical_to_A"], "pass", out["pass"])
    for n, a in arms.items():
        print(n, "served", a["served"], "rows_read", a["rows_read"], "read_errors", a["read_errors"],
              "ram_miss_cpu_s", (a["threads"] or {}).get("ram_miss_cpu_s"), "perf/req", a["perf_per_served_request"],
              "sched/req", (a["sched"] or {}).get("whole_per_served_request"))
    if "C_shim" in out:
        print("C shim", json.dumps({k: out["C_shim"].get(k) for k in ("threads", "service", "copy", "ok", "why")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
