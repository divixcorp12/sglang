"""Accept or reject one Task 1 baseline arm from what its own json says it ran.

A result is only a baseline if the process demonstrably imported the intended tree, read with
O_DIRECT (so the ~400 GiB an arm reads cannot warm the page cache for the next arm), found the
drives idle, and had the mirror and trace settings the arm name claims. Every one of these was
unknowable for the section 19 run. The script, not the reader of a log, decides.

    task1_arm_verdict.py <arm.json> --root <worktree> --head <sha> --mirror on|off --trace T|U \\
        [--cache <arm.cache.json>]

Exit status 0 = valid, 1 = invalid (problems printed), 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Must equal provenance._SECRET_NAME (a unit test pins that): an over-matching filter hides values.
SECRET_NAME = re.compile(r"(API_?KEY|SECRET(_?KEY)?|PASSWORD|CREDENTIALS?|(AUTH|ACCESS|API|HF|BEARER)_TOKEN)$", re.I)
REDACTED = "<redacted>"
MAX_EXPERT_CACHE_GROWTH_BYTES = 1 << 30
READER = "SGLANG_MOE_EXPERT_FILE_READER"
MIRRORS_ENV = "SGLANG_MOE_EXPERT_MIRROR_DIRS"
TRACE_ENV = "SGLANG_DSV41_EXPERT_TRACE_PATH"


def check_arm(report: dict, *, root: str, head: str, mirror: bool, traced: bool, sessions: int = 4) -> tuple[list, list]:
    """(problems, notes). Any problem makes the arm unusable as a baseline."""
    problems, notes = [], []
    prov = report.get("provenance")
    if not isinstance(prov, dict):
        return ["no provenance in the arm json: the run predates the provenance harness"], notes

    package = os.path.join(os.path.realpath(root), "python", "sglang")
    sglang_file = prov.get("sglang_file")
    if not sglang_file or not os.path.realpath(sglang_file).startswith(package + os.sep):
        problems.append(f"imported sglang from {sglang_file!r}, not {package}")
    git = prov.get("git") or {}
    if git.get("head") != head:
        problems.append(f"git head {git.get('head')!r} is not the expected {head!r}")
    if git.get("dirty") or git.get("untracked_in_package_count"):
        problems.append(
            f"worktree not clean: {git.get('dirty_file_count')} tracked changes, "
            f"{git.get('untracked_in_package_count')} untracked under the package"
        )
    if prov.get("unavailable"):
        problems.append(f"provenance fields unavailable: {sorted(prov['unavailable'])}")

    resolved = prov.get("sglang_env_resolved") or {}
    for table in ("sglang_env", "sglang_env_at_exec", "sglang_env_resolved"):
        wrongly = sorted(k for k, v in (prov.get(table) or {}).items() if v == REDACTED and not SECRET_NAME.search(k))
        if wrongly:
            problems.append(f"{table} redacted knobs that are not secrets, so their values are lost: {wrongly}")
    if resolved.get(READER) != "uring_direct":
        problems.append(f"{READER} resolved to {resolved.get(READER)!r}, not 'uring_direct': reads may fill the page cache")
    env = prov.get("sglang_env") or {}
    if mirror and not env.get(MIRRORS_ENV):
        problems.append(f"arm claims mirrors on but {MIRRORS_ENV} is unset in the process")
    if not mirror and env.get(MIRRORS_ENV):
        problems.append(f"arm claims mirrors off but {MIRRORS_ENV}={env[MIRRORS_ENV]!r} in the process")
    if traced != bool(env.get(TRACE_ENV)):
        problems.append(f"arm claims traced={traced} but {TRACE_ENV} is {env.get(TRACE_ENV)!r}")
    if resolved.get("SGLANG_MOE_EXPERT_GRAPH_GATHER") is not True:
        problems.append("SGLANG_MOE_EXPERT_GRAPH_GATHER did not resolve to true: this is not the in-graph reader")

    idle = prov.get("drive_idle_check") or {}
    if idle.get("idle") is not True:
        problems.append(f"drives not idle at start: {idle}")
    if prov.get("sglang_env_drift_at_engine_ready"):
        notes.append(f"Engine launch edited the environment: {prov['sglang_env_drift_at_engine_ready']}")

    rows = report.get("per_session") or []
    if len(rows) != sessions:
        problems.append(f"{len(rows)} sessions, expected {sessions}")
    multi = 0
    for i, row in enumerate(rows):
        step = row.get("step_latency") or {}
        if "unavailable" in step:
            problems.append(f"session {i}: no step latency ({step['unavailable']})")
        multi += step.get("multi_token_chunks", 0)
        if row.get("cpu_s") is None:
            problems.append(f"session {i}: no cpu_s")
    if multi:
        notes.append(f"{multi} chunks carried more than one token: step-latency percentiles are smoothed over them")
    return problems, notes


def check_cache(cache: dict) -> list:
    """Reject an arm that left expert shard bytes in the page cache."""
    before, after = cache.get("expert_resident_bytes_before"), cache.get("expert_resident_bytes_after")
    if before is None or after is None:
        return ["expert page-cache residency was not measured"]
    if after - before > MAX_EXPERT_CACHE_GROWTH_BYTES:
        return [f"expert shard page-cache residency grew {(after - before) / (1 << 30):.2f} GiB across the arm"]
    return []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arm_json")
    p.add_argument("--root", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--mirror", choices=("on", "off"), required=True)
    p.add_argument("--trace", choices=("T", "U"), required=True)
    p.add_argument("--cache")
    args = p.parse_args()
    try:
        with open(args.arm_json) as f:
            report = json.load(f)
        cache = json.load(open(args.cache)) if args.cache else None
    except (OSError, ValueError) as e:
        print(f"UNREADABLE {e}")
        return 2
    problems, notes = check_arm(
        report, root=args.root, head=args.head, mirror=args.mirror == "on", traced=args.trace == "T"
    )
    if cache is not None:
        problems += check_cache(cache)
    for n in notes:
        print("NOTE", n)
    for x in problems:
        print("PROBLEM", x)
    print("INVALID" if problems else "VALID")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
