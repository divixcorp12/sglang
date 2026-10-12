"""R1 timing-sensitivity replay (D12): which of a policy's useful reads would land before demand.

A bounded model, not a measurement. Per forward-layer group the window is B0's snapshot-to-demand stamp span minus the
scorer delay; the ordered reads run back to back, each taking row_read_ms x contention. A read counts as on time only
when it finishes inside the window. The read in flight at the window's end is a partial arrival: only the part of its
stall already served is priced (partial_saved_ms). Reads of groups with no joined stamp window are left out and counted
in missing_windows. Contention, row-read time and their sources come from the frozen plan (r1_plan), never from here.

Inputs: R1Groups, the policy's ordered read ids per group, r1_stamps.corpus_windows output, TimingCases.
Usage: for case in cases_from_plan(plan.timing, scorer_delay_ms=d): res = replay(groups, reads, windows=w, case=case)
"""

from __future__ import annotations

import os
import sys

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import r1_plan  # noqa: E402
from r1_groups import R1Group  # noqa: E402
from r1_stamps import Window  # noqa: E402

LABEL = "model"
MODEL_TEXT = ("reads run back to back from the snapshot; on time = finishes within window minus scorer delay; "
              "the read in flight at that point is partial and only its served part is priced; "
              "groups without a joined stamp window are excluded")
_EPS = 1e-9


class TimingCase(msgspec.Struct, frozen=True):
    name: str
    contention: float
    row_read_ms: float
    scorer_delay_ms: float
    source: str = ""  # where the contention multiplier came from


class TimingResult(msgspec.Struct):
    case: TimingCase
    reads_admitted: int  # reads that finish inside the window
    reads_dropped_window: int  # reads that do not (partial arrivals included)
    useful: int  # reads of true misses, over groups with a window
    on_time_useful: int
    late_useful: int
    partial_saved_ms: float  # stall already served by useful partial arrivals
    missing_windows: int  # groups with no joined stamp window
    assumptions: dict
    label: str = LABEL


def cases_from_plan(timing: r1_plan.TimingAssumptions, *, scorer_delay_ms: float) -> list[TimingCase]:
    return [TimingCase(name=name, contention=float(mult), row_read_ms=timing.row_read_ms,
                       scorer_delay_ms=scorer_delay_ms, source=timing.contention_source)
            for name, mult in timing.contention.items()]


def _assumptions(case: TimingCase) -> dict:
    return {"model": MODEL_TEXT, "contention": case.contention, "contention_source": case.source,
            "row_read_ms": case.row_read_ms, "scorer_delay_ms": case.scorer_delay_ms,
            "per_read_ms": case.row_read_ms * case.contention}


def _group_window(r: R1Group, *, windows: dict) -> Window | None:
    k = r.key
    return windows.get((k.block, k.seq, k.source_layer, k.horizon))


def _split(ids: np.ndarray, *, window_ms: float, case: TimingCase) -> tuple[int, float]:
    """Full reads that fit, and the served part of the next read in ms (0 when none is in flight)."""
    room = window_ms - case.scorer_delay_ms
    per_read = case.row_read_ms * case.contention
    if room <= 0 or per_read <= 0:
        return 0, 0.0
    full = min(len(ids), int(np.floor(room / per_read + _EPS)))
    if full >= len(ids):
        return full, 0.0
    return full, max(0.0, min(per_read, room - full * per_read))


def replay(groups: list[R1Group], reads: list[np.ndarray], *, windows: dict, case: TimingCase) -> TimingResult:
    if len(groups) != len(reads):
        raise ValueError(f"{len(reads)} read lists for {len(groups)} groups")
    res = TimingResult(case=case, reads_admitted=0, reads_dropped_window=0, useful=0, on_time_useful=0,
                       late_useful=0, partial_saved_ms=0.0, missing_windows=0, assumptions=_assumptions(case))
    for r, ids in zip(groups, reads):
        w = _group_window(r, windows=windows)
        if w is None:
            res.missing_windows += 1
            continue
        ids = np.asarray(ids, dtype=np.int64)
        full, served = _split(ids, window_ms=w.window_ms, case=case)
        hit = r.labels.miss[ids] if len(ids) else np.zeros(0, dtype=bool)
        res.reads_admitted += full
        res.reads_dropped_window += len(ids) - full
        res.useful += int(hit.sum())
        res.on_time_useful += int(hit[:full].sum())
        res.late_useful += int(hit[full:].sum())
        if full < len(ids) and hit[full]:
            res.partial_saved_ms += served
    return res
