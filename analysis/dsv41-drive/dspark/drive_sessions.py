"""Per-session drive load of a served arm, from run_arm.sh's boundary samples, and the sessions a drive was slow in.

run_arm.sh brackets the warm-up (``warmup_start``/``warmup_end``) and every timed session (``session_start``/
``session_end``, with its ``session_id``) with provenance.drive_sample(): each mirror drive's cumulative
/proc/diskstats counters, read from the partition row its mount resolves to. The server's own ``exl3 RAM miss drive
load`` covers its whole lifetime, so it cannot say which session a slow drive hit (DSV41_REFERENCE.md 33.17-33.20).

Per session and drive: GB read, busy s (io_ticks), busy s per GB, reads, mean bytes per read, MB written, queue s
(time_in_queue). A session is flagged on a drive when its busy s per GB is more than FLAG_EXCESS over that drive's
median over every session of every arm compared. A counter that went backwards (a wrap or a device reset), a drive
that changed device or is missing, and an unreadable or unpaired sample skip that cell with a note; nothing raises.
"""

from __future__ import annotations

import statistics

SECTOR_BYTES = 512
FLAG_EXCESS = 0.15
_COUNTERS = ("reads", "sectors_read", "writes", "sectors_written", "io_ticks_ms", "time_in_queue_ms")


def _derive(start: dict, end: dict, wall_s: float | None) -> tuple[dict | None, str | None]:
    if start.get("device") != end.get("device"):
        return None, f"device changed from {start.get('device')} to {end.get('device')}"
    d = {}
    for key in _COUNTERS:
        if key not in start or key not in end:
            return None, f"no {key}"
        d[key] = end[key] - start[key]
        if d[key] < 0:
            return None, f"{key} went backwards ({start[key]} -> {end[key]}: a wrap or a device reset)"
    read_bytes = d["sectors_read"] * SECTOR_BYTES
    gb = read_bytes / 1e9
    busy_s = d["io_ticks_ms"] / 1000.0
    return {
        "device": end["device"],
        "gb_read": gb,
        "busy_s": busy_s,
        "busy_s_per_gb": busy_s / gb if gb > 0 else None,
        "reads": d["reads"],
        "mean_bytes_per_read": read_bytes / d["reads"] if d["reads"] else None,
        "mb_written": d["sectors_written"] * SECTOR_BYTES / 1e6,
        "queue_s": d["time_in_queue_ms"] / 1000.0,
        "busy_fraction": busy_s / wall_s if wall_s else None,
    }, None


def _interval(name: str, start: dict, end: dict, notes: list) -> dict | None:
    for row in (start, end):
        if row.get("drive_counters") is None:
            notes.append(f"{name}: {row['label']} counters unavailable ({row.get('drive_counters_unavailable')})")
            return None
    a, b = start["drive_counters"], end["drive_counters"]
    wall_s = end["monotonic"] - start["monotonic"] if "monotonic" in start and "monotonic" in end else None
    drives = {}
    for label in sorted(set(a) | set(b)):
        if label not in a or label not in b:
            drives[label] = None
            notes.append(f"{name}: {label} missing from the {'start' if label not in a else 'end'} sample")
            continue
        drives[label], why = _derive(a[label], b[label], wall_s)
        if why:
            notes.append(f"{name}: {label} skipped, {why}")
    return {"wall_s": wall_s, "drives": drives}


def session_drive_load(samples: list) -> dict:
    """{"warmup": interval or None, "sessions": [{"session_id", "wall_s", "drives": {label: figures or None}}],
    "notes": [...]}, from one run's boundary-samples.jsonl rows in file order."""
    notes = []
    starts, warmup_start, warmup, sessions = {}, None, None, []
    for row in samples:
        label = row.get("label")
        if label == "warmup_start":
            warmup_start = row
        elif label == "warmup_end":
            if warmup_start is None:
                notes.append("warm-up: an end sample with no start")
            else:
                warmup = _interval("warm-up", warmup_start, row, notes)
        elif label == "session_start":
            starts[row.get("session_id")] = row
        elif label == "session_end":
            sid = row.get("session_id")
            start = starts.pop(sid, None)
            if start is None:
                notes.append(f"session {sid}: an end sample with no start")
                continue
            interval = _interval(f"session {sid}", start, row, notes)
            if interval is not None:
                sessions.append({"session_id": sid, **interval})
    notes += [f"session {sid}: a start sample with no end (the session failed or the arm stopped)" for sid in starts]
    if not sessions and warmup is None and not notes:
        notes.append("no per-session drive counters in the boundary samples (the run predates them)")
    return {"warmup": warmup, "sessions": sessions, "notes": notes}


def _per_gb(loads: dict):
    """(arm, session, drive label, figures) for every cell with a busy s per GB."""
    for arm, load in loads.items():
        for session in load["sessions"]:
            for label, fig in session["drives"].items():
                if fig and fig["busy_s_per_gb"] is not None:
                    yield arm, session, label, fig


def flag_sessions(loads: dict, excess: float = FLAG_EXCESS) -> dict:
    """{arm: [flag]}: the sessions whose busy s per GB on a drive is more than ``excess`` over that drive's median
    over every session of every arm in ``loads`` ({arm: session_drive_load()})."""
    by_drive = {}
    for _, _, label, fig in _per_gb(loads):
        by_drive.setdefault(label, []).append(fig["busy_s_per_gb"])
    medians = {label: statistics.median(values) for label, values in by_drive.items()}
    flags = {arm: [] for arm in loads}
    for arm, session, label, fig in _per_gb(loads):
        median = medians[label]
        over = fig["busy_s_per_gb"] / median - 1 if median > 0 else 0.0
        if over > excess + 1e-9:  # a value exactly at the bound, in floating point, is not over it
            flags[arm].append({
                "session_id": session["session_id"], "drive": label, "device": fig["device"],
                "busy_s_per_gb": fig["busy_s_per_gb"], "median_busy_s_per_gb": median,
                "excess_pct": 100.0 * over, "gb_read": fig["gb_read"],
            })  # fmt: skip
    return flags


def format_table(loads: dict, flags: dict) -> str:
    """One line per arm and session: each drive's busy s per GB and GB read, '*' where flagged."""
    labels = sorted({label for load in loads.values() for s in load["sessions"] for label in s["drives"]})
    marked = {(arm, f["session_id"], f["drive"]) for arm, fs in flags.items() for f in fs}
    lines = [
        f"drive load per session: busy s per GB (GB read); * = over {100 * FLAG_EXCESS:.0f}% above the drive's "
        "median over every arm's sessions",
        f"{'arm':<30} {'#':>2} {'session':<28} " + " ".join(f"{label:>18}" for label in labels),
    ]
    for arm, load in loads.items():
        for i, session in enumerate(load["sessions"]):
            cells = []
            for label in labels:
                fig = session["drives"].get(label)
                if not fig:
                    cells.append(f"{'-':>18}")
                    continue
                per_gb = "n/a" if fig["busy_s_per_gb"] is None else f"{fig['busy_s_per_gb']:.2f}"
                star = "*" if (arm, session["session_id"], label) in marked else ""
                cells.append(f"{per_gb + star + ' (' + format(fig['gb_read'], '.2f') + ' GB)':>18}")
            sid = str(session["session_id"])
            lines.append(f"{arm:<30} {i:>2} {sid[-28:]:<28} " + " ".join(cells))
        for note in load["notes"]:
            lines.append(f"{arm:<30} note: {note}")
    return "\n".join(lines)
