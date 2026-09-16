"""Summarize provenance-locked B/C/Cr/N/D experiment result files."""

import argparse
import json
import random
import statistics
from pathlib import Path


def _jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _manifest(path: str) -> dict:
    records = _jsonl(path)
    if len(records) != 1:
        raise ValueError("run manifest must contain exactly one JSON object")
    return records[0]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _last_prefetch(path: str) -> dict:
    if not path:
        return {}
    return next((r["prefetch"] for r in reversed(_jsonl(path)) if "prefetch" in r), {})


def _decode_counters(path: str) -> list[dict]:
    if not path:
        return []
    records = _jsonl(path)
    counters = records[-1].get("counters", {}) if records else {}
    return list(counters.get("decode", {}).values())


def summarize_arm(spec: str) -> dict:
    """Read ``arm=results:prediction:hot-cache:manifest`` without source guessing."""
    try:
        arm, value = spec.split("=", 1)
    except ValueError as error:
        raise ValueError("arm spec must be arm=results:prediction:hot-cache:manifest") from error
    paths = value.split(":")
    if len(paths) != 4 or not paths[0] or not paths[3]:
        raise ValueError("arm spec must be arm=results:prediction:hot-cache:manifest")
    results_path, prediction_path, hot_path, manifest_path = paths
    turns = _jsonl(results_path)
    good = [
        turn for turn in turns
        if "error" not in turn and (turn.get("completion_tokens") or 0) >= 64
        and turn.get("decode_tokens_per_sec")
    ]
    decode_ms = [1000.0 / turn["decode_tokens_per_sec"] for turn in good]
    hot_rows = _decode_counters(hot_path)
    posted = sum(row.get("side_pull_posted_rows", 0) for row in hot_rows)
    useful = sum(row.get("side_pull_useful_rows", 0) for row in hot_rows)
    wasted = sum(row.get("side_pull_wasted_rows", 0) for row in hot_rows)
    return {
        "arm": arm,
        "manifest": _manifest(manifest_path),
        "median_decode_tok_s": statistics.median([turn["decode_tokens_per_sec"] for turn in good]) if good else None,
        "p50_turn_decode_ms_per_token": _percentile(decode_ms, 50),
        "p95_turn_decode_ms_per_token": _percentile(decode_ms, 95),
        "median_ttft_s": statistics.median([turn["ttft"] for turn in good if turn.get("ttft") is not None]) if good else None,
        "turns": len(good),
        "errors": sum("error" in turn for turn in turns),
        "truncation_count": sum(turn.get("finish_reason") == "length" for turn in turns),
        "observed_miss_rate": _last_prefetch(prediction_path).get("observed_miss_rate"),
        "resident_slots": sum(row.get("resident_slots", 0) for row in hot_rows),
        "posted_rows": posted,
        "useful_rows": useful,
        "wasted_rows": wasted,
        "useful_precision": useful / posted if posted else None,
        "_session_decode_ms": {
            turn["session_id"]: 1000.0 / turn["decode_tokens_per_sec"]
            for turn in good if "session_id" in turn
        },
    }


def paired_session_bootstrap(left: dict, right: dict, *, seed: int = 20260916, resamples: int = 10_000) -> list[float]:
    """Return the paired session-cluster 95% interval for right-minus-left ms/token."""
    for key in ("commit", "cache_size"):
        if left["manifest"].get(key) != right["manifest"].get(key):
            raise ValueError(f"mixed {key}s are not comparable")
    sessions = sorted(set(left["_session_decode_ms"]) & set(right["_session_decode_ms"]))
    if not sessions:
        raise ValueError("no paired sessions")
    deltas = [right["_session_decode_ms"][s] - left["_session_decode_ms"][s] for s in sessions]
    generator = random.Random(seed)
    samples = [statistics.mean(generator.choices(deltas, k=len(deltas))) for _ in range(resamples)]
    return [_percentile(samples, 2.5), _percentile(samples, 97.5)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="arm=results:prediction:hot-cache:manifest")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    entries = list(map(summarize_arm, args.runs))
    summary = {entry["arm"]: {k: v for k, v in entry.items() if k != "_session_decode_ms"} for entry in entries}
    if args.compare:
        by_arm = {entry["arm"]: entry for entry in entries}
        summary["comparison"] = {
            "arms": args.compare,
            "paired_bootstrap_95_ci_decode_ms_per_token": paired_session_bootstrap(
                by_arm[args.compare[0]], by_arm[args.compare[1]]
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
