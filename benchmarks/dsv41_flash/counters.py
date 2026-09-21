"""The RAM-miss service counters of a run, and the check that the lease path really ran.

The service lives in the scheduler subprocess, so an Engine's driver cannot call service.host.counters() the way
open11_serving_path.py does. The stream trace (SGLANG_DSV41_EXPERT_TRACE_PATH) is the channel that exists: each
decode graph step appends a line carrying the service's cumulative counters, line-buffered because the scheduler
is SIGKILLed at Engine shutdown. Tracing also turns on the service's stage timestamps in both arms.
"""

from __future__ import annotations

import json

# The gap between granted and acked is read at the last SNAPSHOT_TAIL snapshots: a leak grows the gap for good,
# a healthy run returns to zero between steps, so one zero gap in the tail is evidence of a balanced ledger.
SNAPSHOT_TAIL = 8
HISTORY_KEEP = 12


class TraceCursor:
    """Reads a growing trace file from where the last read stopped, keeping the counters snapshots it sees."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.offset = 0
        self.graph_steps = 0
        self.demand_rows = 0
        self.snapshots: list[dict] = []
        self._partial = ""

    def poll(self) -> TraceCursor:
        try:
            with open(self.path) as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset = f.tell()
        except FileNotFoundError:
            return self
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()
        for line in lines:
            row = json.loads(line)
            if row.get("kind") == "graph_step":
                self.graph_steps += 1
                self.demand_rows += row["ram_miss"]
                if "thread" in row:
                    self.snapshots.append(row["thread"])
        return self

    def mark(self) -> dict:
        """The state now: what a later mark is subtracted from to get a window."""
        self.poll()
        return {
            "graph_steps": self.graph_steps,
            "demand_rows": self.demand_rows,
            "counters": dict(self.snapshots[-1]) if self.snapshots else None,
            "snapshots": len(self.snapshots),
        }


def window(start: dict, end: dict) -> dict:
    """Counter deltas between two marks; None where either mark saw no snapshot."""
    delta = None
    if start["counters"] is not None and end["counters"] is not None:
        delta = {k: end["counters"][k] - start["counters"][k] for k in end["counters"]}
    return {
        "graph_steps": end["graph_steps"] - start["graph_steps"],
        "demand_rows": end["demand_rows"] - start["demand_rows"],
        "counters_delta": delta,
    }


def verify_lease(*, arm: str, cursor: TraceCursor) -> dict:
    """Whether the in-graph RAM-miss path ran and the lease ledger is what the arm requires.

    Both arms need graph steps and a served RAM-miss request. lease_on needs leases granted, and granted balanced
    by acked; lease_off needs zero of both, which proves the arms differ. There is no fatal counter in the
    snapshot (fatal_seq is a request-page word); fail-stop kills the scheduler, so a completed run has none, and
    late_after_fatal and read_errors are checked as evidence.
    """
    reasons = []
    snaps = cursor.snapshots
    final = snaps[-1] if snaps else None
    tail = snaps[-SNAPSHOT_TAIL:]
    if cursor.graph_steps == 0 or final is None:
        reasons.append(
            "no decode graph step reached the trace: the in-graph path never ran"
        )
    else:
        if final["served"] + final["touch_only"] == 0:
            reasons.append("the RAM-miss service served nothing")
        if final["late_after_fatal"] or final["read_errors"]:
            reasons.append(
                f"late_after_fatal={final['late_after_fatal']} read_errors={final['read_errors']}"
            )
        if arm == "lease_on":
            if final["leases_granted"] <= 0:
                reasons.append(
                    "leases_granted == 0 on the lease_on arm: the lease path was never exercised"
                )
            elif min(s["leases_granted"] - s["leases_acked"] for s in tail) != 0:
                reasons.append(
                    f"leases_granted {final['leases_granted']} never balanced by leases_acked in the last {len(tail)} snapshots"
                )
        elif final["leases_granted"] or final["leases_acked"]:
            reasons.append(
                f"lease_off arm shows leases_granted={final['leases_granted']} leases_acked={final['leases_acked']}"
            )
    return {
        "ok": not reasons,
        "reasons": reasons,
        "graph_steps": cursor.graph_steps,
        "final_gap_granted_minus_acked": None
        if final is None
        else final["leases_granted"] - final["leases_acked"],
        "min_gap_in_tail": None
        if not tail
        else min(s["leases_granted"] - s["leases_acked"] for s in tail),
        "snapshot_history_tail": snaps[-HISTORY_KEEP:],
    }
