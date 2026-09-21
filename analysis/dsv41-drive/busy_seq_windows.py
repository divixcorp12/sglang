"""Where the kBusySeq window figures in the plan come from, and how far each can be trusted.

The plan's Task 6 text cites: a busy window of 5.03 ms (mirrors on) and 10.25 ms (mirrors off) at
minimum for requests that read rows; 0.6-1.8 us for requests that read nothing; 0.2-0.9 us between
kBusySeq clearing and demand_done. None of those is a direct observation of the page word. Each is
computed here from the RAM-miss STAGE STAMPS of a trace, which bracket the window; the direct
observations of the word are in busy_seq_protocol_probe.py.

handle_demand stores kBusySeq = seq as its first action and stores 0 after set_status; pump_demand
then stores demand_done. The stamps (StageRecord, exl3_ram_miss_host.cpp) sit around that:

  observed  begin_stage: the record was found, BEFORE read_record and before the busy store
  mapped    slots marked READY and published, BEFORE set_status and before the busy clear
  done      end_stage: taken AFTER the demand_done store

so, per figure:
  read window   = mapped - observed   (served requests). The busy window is inside it at both ends,
                  so this OVERSTATES the window by the pre-store work (record read, a few hundred ns)
                  and the set_status; for ms-scale windows that is noise.
  touch window  = done - observed     (touch requests, unarmed, which never read). It also includes
                  the demand_done store and end_stage, so it overstates the busy window.
  clear->done   = done - mapped       (served). It contains set_status, the sfence, the busy clear,
                  the done store and the stamp itself: an UPPER BOUND on the clear-to-done gap, not the
                  gap. The true gap is smaller.
The read-request window is the request's own read time plus packing, so it is not independent of the
read-wait distribution (its minimum tracks the read-wait p10): it is a derived figure for THIS trace's
workload and drives, not a property of the protocol.

Usage: python busy_seq_windows.py TRACE [TRACE ...]   (schema >= 2 traces; read only)
"""

from __future__ import annotations

import hashlib
import json
import sys


def _pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q / 100))]


def _line(name, values):
    if not values:
        return f"  {name}: no samples"
    return (
        f"  {name}: n={len(values)}  min {min(values):.2f}  p1 {_pct(values, 1):.2f}  p10 {_pct(values, 10):.2f}"
        f"  p50 {_pct(values, 50):.2f}  p99 {_pct(values, 99):.2f}  max {max(values):.2f}  (us)"
    )


def analyse(path):
    read, touch, gap = [], [], []
    schemas, other = set(), 0
    for raw in open(path):
        if "ram_miss_request" not in raw:
            continue
        r = json.loads(raw)
        if r.get("kind") != "ram_miss_request":
            continue
        schemas.add(r.get("schema", 1))
        s = r["stages_ns"]
        if r["status"] == "served" and r.get("extents", 0) > 0 and s["observed"] and s["mapped"] and s["done"]:
            read.append((s["mapped"] - s["observed"]) / 1e3)
            gap.append((s["done"] - s["mapped"]) / 1e3)
        elif r["status"] == "touch" and s["observed"] and s["done"]:
            touch.append((s["done"] - s["observed"]) / 1e3)
        else:
            other += 1
    return schemas, read, touch, gap, other


def main(paths):
    for path in paths:
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        schemas, read, touch, gap, other = analyse(path)
        print(f"{path}  sha256[:16] {digest}  schemas {sorted(schemas)}  skipped {other}")
        print(_line("read-request window  = mapped - observed (DERIVED, overstates the busy window)", read))
        print(_line("touch-request window = done - observed   (DERIVED, includes the done store)", touch))
        print(_line("clear -> done        = done - mapped     (UPPER BOUND on the gap)", gap))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
