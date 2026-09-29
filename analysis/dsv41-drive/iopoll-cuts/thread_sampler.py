"""Per-thread CPU of a server process tree, every `period` s, as JSON lines (plan 2026-09-28-iopoll-read-cuts Task 7).
Usage: thread_sampler.py <root pid> <out.jsonl> [period]. `report(samples, start, end)` sums, over [start, end]:
io-wq workers (comm iou-wrk*: distinct tids, max live, CPU s), the SQ thread (iou-sqp*), and the RAM-miss service
thread (comm exl3-ram-miss)."""

import datetime
import json
import os
import sys
import time

TICK = os.sysconf("SC_CLK_TCK")


def tree(root):
    children = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(pid))
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += children.get(p, [])
    return out


def sample(root):
    rows = []
    for pid in tree(root):
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
                f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            rows.append({"pid": pid, "tid": int(tid), "comm": comm, "state": f[0],
                         "cpu_s": (int(f[11]) + int(f[12])) / TICK})
    return rows


def main(root, out, period=2.0):
    with open(out, "a") as fh:
        while os.path.exists(f"/proc/{root}"):
            fh.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "tasks": sample(root)}) + "\n")
            fh.flush()
            time.sleep(period)


def report(path, start, end):
    """start/end: aware datetimes (mirror3_report.timed_window_utc)."""
    rows = [json.loads(line) for line in open(path)]
    inside = [r for r in rows if start <= datetime.datetime.fromisoformat(r["utc"]) <= end]
    if len(inside) < 2:
        return {"samples": len(inside)}

    def cpu(pred):
        first = {t["tid"]: t["cpu_s"] for t in inside[0]["tasks"] if pred(t["comm"])}
        total = 0.0
        for t in inside[-1]["tasks"]:
            if pred(t["comm"]):
                total += t["cpu_s"] - first.get(t["tid"], 0.0)  # a thread born inside counts from 0
        return round(total, 2)

    wrk = lambda c: c.startswith("iou-wrk")
    tids = {t["tid"] for r in inside for t in r["tasks"] if wrk(t["comm"])}
    return {
        "samples": len(inside),
        "window_s": (datetime.datetime.fromisoformat(inside[-1]["utc"])
                     - datetime.datetime.fromisoformat(inside[0]["utc"])).total_seconds(),
        "iowq_distinct": len(tids),
        "iowq_max_live": max(sum(1 for t in r["tasks"] if wrk(t["comm"])) for r in inside),
        "iowq_cpu_s": cpu(wrk),
        "sq_thread_cpu_s": cpu(lambda c: c.startswith("iou-sqp")),
        "ram_miss_cpu_s": cpu(lambda c: c == "exl3-ram-miss"),
    }


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
