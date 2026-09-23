#!/usr/bin/env python3
"""Summarize native MoE RAM-miss stages in timed DSV4.1 decode replays.

The trace's JSONL ``t`` is when the scheduler drained a native record. Window
selection uses ``stages_ns.observed/done`` instead, which share CLOCK_MONOTONIC
with the benchmark's boundary samples. Each captured replay posts one record
per streamed layer: ``demand`` when the GPU waits, ``touch`` when it does not.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _span(stages: dict, start: str, end: str) -> int | None:
    a, b = stages.get(start, 0), stages.get(end, 0)
    if not a or not b:
        return None
    if b < a:
        raise ValueError(f"invalid stage order: {start}={a} > {end}={b}")
    return b - a


def request_intervals(record: dict) -> dict[str, int | None]:
    """Return non-overlapping stage boundaries, leaving absent I/O as N/A."""
    s = record["stages_ns"]
    has_read = record.get("rows_asked", 0) > 0 and s.get("submit", 0) > 0
    last_cqe = s.get("last_cqe", 0)
    pack_end = s.get("pack_end", 0)
    pack_tail = None
    if has_read and last_cqe and pack_end:
        if pack_end < last_cqe:
            raise ValueError("pack_end precedes last_cqe")
        pack_tail = pack_end - last_cqe
    return {
        "cpu_total_ns": _span(s, "observed", "done"),
        "reserved_ns": _span(s, "observed", "reserved"),
        "reader_setup_ns": _span(s, "reserved", "submit") if has_read else None,
        "submit_first_cqe_ns": _span(s, "submit", "first_cqe") if has_read else None,
        "first_last_cqe_ns": _span(s, "first_cqe", "last_cqe") if has_read else None,
        "read_window_ns": _span(s, "submit", "last_cqe") if has_read else None,
        "pack_exposed_tail_ns": pack_tail,
        "pack_to_map_ns": _span(s, "pack_end", "mapped") if has_read else None,
        "publish_signal_ns": _span(s, "mapped", "done"),
    }


def select_decode_suffix(
    records: Iterable[dict], *, start_ns: int, end_ns: int, replays: int, layers: tuple[int, ...]
) -> list[dict]:
    """Select a whole request's final graph-replay sequence after its prefill.

    The benchmark supplies only a monotonic boundary after each session, not a
    decode-start stamp. A candidate is accepted only when its last
    ``replays * layers`` demand/touch records have the exact layer cycle and
    contiguous request sequence. The caller must report this suffix inference.
    """
    if replays < 1 or not layers or end_ns <= start_ns:
        raise ValueError("invalid decode window or replay shape")
    candidates = []
    for record in records:
        if record.get("kind") != "ram_miss_request":
            continue
        if record["request"]["type"] not in ("demand", "touch"):
            continue
        stages = record["stages_ns"]
        observed, done = stages.get("observed", 0), stages.get("done", 0)
        if start_ns <= observed <= done <= end_ns:
            candidates.append(record)
    candidates.sort(key=lambda row: (row["stages_ns"]["observed"], row["request"]["seq"]))
    expected = replays * len(layers)
    if len(candidates) < expected:
        raise ValueError(f"expected {expected} request records, found only {len(candidates)}")
    selected = candidates[-expected:]
    got_layers = tuple(record["layer"] for record in selected)
    if got_layers != layers * replays:
        raise ValueError("decode suffix does not repeat the expected layer cycle")
    seqs = [int(record["request"]["seq"]) for record in selected]
    for previous, current in zip(seqs, seqs[1:]):
        next_seq = (previous + 1) & 0xFFFFFFFF
        if next_seq == 0:
            next_seq = 1
        if current != next_seq:
            raise ValueError(f"decode suffix sequence gap: {previous} -> {current}")
    return selected


def _percentile(values: list[int], percent: int) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * percent / 100
    lo = math.floor(at)
    hi = math.ceil(at)
    return (ordered[lo] * (hi - at) + ordered[hi] * (at - lo)) if lo != hi else float(ordered[lo])


def _describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "sum_ms": sum(values) / 1e6,
        "mean_us": sum(values) / len(values) / 1e3,
        "p50_us": _percentile(values, 50) / 1e3,
        "p95_us": _percentile(values, 95) / 1e3,
        "p99_us": _percentile(values, 99) / 1e3,
        "max_us": max(values) / 1e3,
    }


def summarize(records: list[dict], *, include_layers: bool = True) -> dict:
    by_type = collections.Counter(row["request"]["type"] for row in records)
    by_status = collections.Counter(row["status"] for row in records)
    rows_asked = collections.Counter(row["rows_asked"] for row in records if row["request"]["type"] == "demand")
    metrics = [request_intervals(row) for row in records]
    keys = metrics[0].keys() if metrics else ()
    stages = {key: _describe([m[key] for m in metrics if m[key] is not None]) for key in keys}
    by_read = {}
    for label, subset in (
        ("demand_read", [row for row in records if row["request"]["type"] == "demand" and row["rows_asked"] > 0]),
        ("demand_no_read", [row for row in records if row["request"]["type"] == "demand" and row["rows_asked"] == 0]),
        ("touch", [row for row in records if row["request"]["type"] == "touch"]),
    ):
        spans = [request_intervals(row) for row in subset]
        by_read[label] = {
            "n": len(subset),
            "bytes": sum(row.get("bytes", 0) for row in subset),
            "stages": {
                key: _describe([m[key] for m in spans if m[key] is not None]) for key in keys
            },
        }
    failed = [row for row in records if row["status"] not in ("served", "touch", "no_read") or not row["request"].get("ok", 0)]
    dropped = sum(row.get("dropped_before", 0) for row in records)
    untraced_rows = sum(row.get("untraced", {}).get("rows", 0) for row in records)
    untraced_extents = sum(row.get("untraced", {}).get("extents", 0) for row in records)
    output = {
        "n": len(records),
        "request_types": dict(sorted(by_type.items())),
        "statuses": dict(sorted(by_status.items())),
        "demand_rows_asked": dict(sorted(rows_asked.items())),
        "bytes": sum(row.get("bytes", 0) for row in records),
        "dropped_before": dropped,
        "untraced_rows": untraced_rows,
        "untraced_extents": untraced_extents,
        "validity": {
            "all_served": not failed,
            "no_dropped_records": dropped == 0,
            "complete_coverage": untraced_rows == 0 and untraced_extents == 0,
            "clean_attribution": not failed and dropped == 0 and untraced_rows == 0 and untraced_extents == 0,
        },
        "stages": stages,
        "groups": by_read,
    }
    if include_layers:
        output["layers"] = {
            str(layer): summarize([row for row in records if row["layer"] == layer], include_layers=False)
            for layer in sorted({row["layer"] for row in records})
        }
        drive_bytes: collections.Counter[str] = collections.Counter()
        for row in records:
            for drive in row.get("drives", []):
                drive_bytes[str(drive["dev"])] += drive["bytes"]
        output["drive_bytes"] = dict(sorted(drive_bytes.items()))
        output["retry_bytes"] = sum(row.get("byte_split", {}).get("retried", 0) for row in records)
        output["cancelled_bytes"] = sum(row.get("byte_split", {}).get("cancelled", 0) for row in records)
        output["pack_workers"] = dict(sorted(collections.Counter(
            row["request"].get("pack_workers") for row in records
        ).items()))
        output["worst_demands"] = [
            {"seq": row["request"]["seq"], "layer": row["layer"],
             "rows_asked": row["rows_asked"], "cpu_total_us": interval["cpu_total_ns"] / 1000,
             "read_window_us": interval["read_window_ns"] / 1000 if interval["read_window_ns"] is not None else None,
             "pack_exposed_tail_us": interval["pack_exposed_tail_ns"] / 1000 if interval["pack_exposed_tail_ns"] is not None else None}
            for row, interval in sorted(
                ((row, request_intervals(row)) for row in records if row["request"]["type"] == "demand"),
                key=lambda item: item[1]["cpu_total_ns"], reverse=True,
            )[:10]
        ]
    return output


def analyze(trace: Path, results: Path, boundaries: Path, layers: tuple[int, ...]) -> tuple[dict, list[dict]]:
    lines = read_jsonl(trace)
    requests = [row for row in lines if row.get("kind") == "ram_miss_request"]
    results_rows = read_jsonl(results)
    session_ids = [row["session_id"] for row in results_rows]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("duplicate session_id: this diagnostic requires one result turn per session")
    marks = {row["label"]: row["monotonic"] for row in read_jsonl(boundaries)}
    if "server_ready" not in marks:
        raise ValueError("missing server_ready boundary")
    start = marks["server_ready"]
    sessions, selected_all, chosen_records = [], [], []
    selected_sequences: set[int] = set()
    for result in results_rows:
        session_id = result["session_id"]
        label = f"session_{session_id}"
        if label not in marks:
            raise ValueError(f"missing {label} boundary")
        end = marks[label]
        replays = int(result["completion_tokens"])
        selected = select_decode_suffix(
            requests,
            start_ns=int(start * 1e9),
            # The benchmark marks session completion while the service still
            # finishes the last streamed layers. Cap this tail at one second.
            end_ns=int(end * 1e9) + 1_000_000_000,
            replays=replays,
            layers=layers,
        )
        if any(row.get("schema") != 5 for row in selected):
            raise ValueError(f"session {session_id}: expected stage schema 5")
        if selected[0]["stages_ns"]["observed"] > int(end * 1e9):
            raise ValueError(f"session {session_id}: selected requests begin after session boundary")
        sequences = {int(row["request"]["seq"]) for row in selected}
        if selected_sequences & sequences:
            raise ValueError(f"session {session_id}: overlapping session selection")
        selected_sequences.update(sequences)
        chosen_records.extend(selected)
        sessions.append({
            "session_id": session_id,
            "window_monotonic_s": [start, end],
            "replays": replays,
            "selection": "validated final demand/touch layer cycles inside session boundary plus <=1 s service tail",
            "service_tail_after_boundary_ms": max(0, selected[-1]["stages_ns"]["done"] / 1e6 - end * 1e3),
            "first_seq": selected[0]["request"]["seq"],
            "last_seq": selected[-1]["request"]["seq"],
            "summary": summarize(selected),
        })
        for row in selected:
            selected_all.append({
                "session_id": session_id,
                "seq": row["request"]["seq"],
                "layer": row["layer"],
                "type": row["request"]["type"],
                "status": row["status"],
                "rows_asked": row["rows_asked"],
                "lanes": row["request"].get("lanes"),
                "backlog": row["request"].get("backlog"),
                "bytes": row.get("bytes", 0),
                "byte_split": row.get("byte_split", {}),
                "drives": row.get("drives", []),
                "extent_cqe_ns": row.get("extent_cqe_ns", []),
                "row_pack_ns": row.get("row_pack_ns", []),
                "stages_ns": row["stages_ns"],
                "intervals_ns": request_intervals(row),
            })
        start = end
    return {
        "trace": str(trace),
        "results": str(results),
        "boundaries": str(boundaries),
        "sessions": sessions,
        "all_selected": summarize(chosen_records),
    }, selected_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("boundaries", type=Path)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--rows-out", type=Path)
    args = parser.parse_args()
    report, rows = analyze(args.trace, args.results, args.boundaries, tuple(range(args.layers)))
    if args.rows_out:
        with args.rows_out.open("w") as target:
            for row in rows:
                target.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
