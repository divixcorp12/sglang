"""Paired comparison of two DSV4.1 baseline arms: the headline statistic (rule 3).

Modeled on `divix01:/data/models/slang/nvfp4-work/cc-expert-prediction/accept_paired.py`.
Joins two arms' `results.jsonl` on `session_id`, reports the per-session delta and
ratio, the win count, and the one-sided sign test. Refuses to pair arms recorded under
different card tenancy (rule 9), different SM clock profiles (this workload's decode
clock, ~2572 MHz, sits well below its at-rest clock, ~2947-2970 MHz; two arms caught
at different points in that behavior are not comparable however well their sessions
are paired — see `clock_ramp.py`), or a session any of
whose arms flagged as compile-contaminated (`compile_watch.py`; `run_arm.sh` already
hard-aborts an arm the moment this happens, so a contaminated record reaching here
means that safety net was bypassed — this is the second one).

Usage: python paired.py <arm_a_dir> <arm_b_dir>
Each <arm_dir> holds results.jsonl, clocks.jsonl, compile.jsonl, and
run-manifest.json (see run_arm.sh).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from clock_ramp import clock_profiles_compatible
from metrics import median_tok_s, one_sided_sign_test
from results_gate import check_result_gate, load_results
from session_subset import N_SESSIONS
from tenancy import parse_tenancy, tenancy_compatible


class TenancyMismatchError(RuntimeError):
    pass


class ClockProfileMismatchError(RuntimeError):
    pass


class CompileContaminationError(RuntimeError):
    pass


def _key(record: dict) -> str:
    return record["session_id"]


def _load_jsonl_by_session(path: str) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["session_id"]] = row
    return rows


def load_arm(arm_dir: str) -> tuple[list[dict], dict[str, dict], dict]:
    arm_dir = Path(arm_dir)
    manifest = json.loads((arm_dir / "run-manifest.json").read_text())
    records = load_results(str(arm_dir / "results.jsonl"))
    expected_count = len(manifest["session_ids"]) if "session_ids" in manifest else N_SESSIONS
    check_result_gate(records, expected_count=expected_count)
    if "session_ids" in manifest and [r["session_id"] for r in records] != manifest["session_ids"]:
        raise ValueError(f"{arm_dir}: result sessions do not match the run manifest")
    clocks = _load_jsonl_by_session(str(arm_dir / "clocks.jsonl"))
    missing_clocks = [r["session_id"] for r in records if r["session_id"] not in clocks]
    if missing_clocks:
        raise ValueError(f"missing clock samples for sessions: {missing_clocks}")
    compile_events = _load_jsonl_by_session(str(arm_dir / "compile.jsonl"))
    contaminated = [
        sid for sid, row in compile_events.items() if row.get("compiled_during_session")
    ]
    if contaminated:
        raise CompileContaminationError(
            f"{arm_dir}: JIT compilation occurred during these sessions' timed window: "
            f"{contaminated}; their timing is not usable"
        )
    return records, clocks, manifest


def pair_sessions(a_records: list[dict], b_records: list[dict]) -> list[tuple[dict, dict]]:
    b_by_key = {_key(r): r for r in b_records}
    return [(a_rec, b_by_key[_key(a_rec)]) for a_rec in a_records if _key(a_rec) in b_by_key]


def compare(a_dir: str, b_dir: str, *, a_name: str = "A", b_name: str = "B") -> dict:
    a_records, a_clocks, a_manifest = load_arm(a_dir)
    b_records, b_clocks, b_manifest = load_arm(b_dir)

    a_tenancy = parse_tenancy(a_manifest["tenancy_start"])
    b_tenancy = parse_tenancy(b_manifest["tenancy_start"])
    if not tenancy_compatible(a_tenancy, b_tenancy):
        raise TenancyMismatchError(
            f"{a_name} and {b_name} ran under different card tenancy "
            f"({a_manifest['tenancy_start']} vs {b_manifest['tenancy_start']}); "
            "not comparable"
        )

    a_clock_samples = [c["clock_sm_start_mhz"] for c in a_clocks.values()]
    b_clock_samples = [c["clock_sm_start_mhz"] for c in b_clocks.values()]
    if not clock_profiles_compatible(a_clock_samples, b_clock_samples):
        raise ClockProfileMismatchError(
            f"{a_name} and {b_name} ran at different, incompatible SM clock levels "
            f"(mean {statistics.fmean(a_clock_samples):.0f} MHz vs "
            f"{statistics.fmean(b_clock_samples):.0f} MHz); not comparable"
        )

    if {r["session_id"] for r in a_records} != {r["session_id"] for r in b_records}:
        raise ValueError(f"{a_name} and {b_name} used different timed session sets")
    pairs = pair_sessions(a_records, b_records)

    per_session = []
    wins = 0
    for a_rec, b_rec in pairs:
        a_tok_s = a_rec["decode_tokens_per_sec"]
        b_tok_s = b_rec["decode_tokens_per_sec"]
        wins += b_tok_s > a_tok_s
        per_session.append(
            {
                "session_id": a_rec["session_id"],
                f"{a_name}_tok_s": a_tok_s,
                f"{b_name}_tok_s": b_tok_s,
                "delta": b_tok_s - a_tok_s,
                "ratio": b_tok_s / a_tok_s,
                f"{a_name}_ttft": a_rec.get("ttft"),
                f"{b_name}_ttft": b_rec.get("ttft"),
            }
        )

    n = len(pairs)
    deltas = [row["delta"] for row in per_session]
    ratios = [row["ratio"] for row in per_session]

    return {
        "a_name": a_name,
        "b_name": b_name,
        "paired_sessions": n,
        "per_session": per_session,
        "delta_median": statistics.median(deltas),
        "ratio_median": statistics.median(ratios),
        "b_wins": wins,
        "sign_test_p_value": one_sided_sign_test(wins=wins, n=n),
        # Context only, not the comparison: each arm's own median across its sessions.
        "a_own_median_tok_s": median_tok_s([a_rec["decode_tokens_per_sec"] for a_rec, _ in pairs]),
        "b_own_median_tok_s": median_tok_s([b_rec["decode_tokens_per_sec"] for _, b_rec in pairs]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a_dir")
    parser.add_argument("b_dir")
    parser.add_argument("--a-name", default="A")
    parser.add_argument("--b-name", default="B")
    args = parser.parse_args()

    result = compare(args.a_dir, args.b_dir, a_name=args.a_name, b_name=args.b_name)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
