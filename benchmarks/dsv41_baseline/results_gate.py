"""The result gate (rule 5): exactly N_SESSIONS records and 0 errors, or abort.

Pure functions over parsed results-JSONL records (`run_capture_sessions.py`'s own
format); no network, no GPU.
"""

from __future__ import annotations

import json

from session_subset import N_SESSIONS


class ResultGateError(RuntimeError):
    pass


def load_results(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_result_gate(records: list[dict], *, expected_count: int = N_SESSIONS) -> None:
    errors = [r for r in records if r.get("error")]
    if len(records) != expected_count or errors:
        raise ResultGateError(
            f"result gate failed: {len(records)} records (expected {expected_count}), "
            f"{len(errors)} errors"
        )
