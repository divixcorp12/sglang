"""Judge one HTTP-driven arm with Task 1's own `check_arm` and friends, not a second copy.

Calls `task1_arm_verdict.check_arm` / `check_cache` / `contention` / `generation` /
`session_outliers` / `cross_arm_outliers` (via `task1_verdict.load_task1_verdict`)
against a report built by `report_builder.build_report`, and appends this campaign's
own two checks (compile-contamination, the SM-clock readiness note) as clearly-labeled
additions — never edits to Task 1's functions.

**Known, currently-unresolved consequence: every arm reads INVALID.**
`check_arm` requires `per_session[i]["step_latency"]` to carry real percentiles;
`report_builder.STEP_LATENCY_UNAVAILABLE` is honest about not having them (see that
module's docstring for why). `judge()` therefore always returns at least 8 PROBLEM
lines of the form "session N: no step latency (...)" until that gap closes. This is
surfaced, not hidden: `judge()` returns them like any other problem, and
`compile_and_clock_are_the_only_problems` tells a caller whether the remaining,
non-acknowledged problems are empty — the caller decides whether that is enough to
proceed, this module does not decide for it.
"""

from __future__ import annotations

STEP_LATENCY_PROBLEM_PREFIX = "session "  # "session {i}: no step latency (...)" — check_arm's own wording
STEP_LATENCY_PROBLEM_SUFFIX = "no step latency"


def is_acknowledged_step_latency_problem(problem: str) -> bool:
    return problem.startswith(STEP_LATENCY_PROBLEM_PREFIX) and STEP_LATENCY_PROBLEM_SUFFIX in problem


def compile_contamination_problems(report: dict) -> list[str]:
    """NOT in Task 1's check_arm — added 2026-09-21 after a Triton compile landed inside a served
    request post-/health, which check_arm (built for the offline Engine, which has no /health
    concept) has no way to see."""
    return [
        f"session {row['session_id']}: JIT compilation during the timed window "
        f"({row.get('compile_events')} event(s))"
        for row in report.get("per_session", [])
        if row.get("compiled_during_session")
    ]


def residency_cache_dict(report: dict) -> dict | None:
    """This campaign's `residency` (three named boundaries, one dir) reshaped into the
    `{"expert_resident_by_dir_before": {dir: bytes}, "expert_resident_by_dir_after": {dir: bytes}}`
    shape `task1_arm_verdict.check_cache` expects — Task 1's own whole-arm residency check,
    the coarser variant it already has for when per-session residency (an offline-Engine-only
    instrumentation) is unavailable. None if this report has no residency section to check.
    """
    residency = report.get("residency")
    if not residency or residency.get("dir") is None:
        return None
    before, after = residency.get("before_server"), residency.get("after_timed_set")
    if before is None or after is None:
        return None
    d = residency["dir"]
    return {"expert_resident_by_dir_before": {d: before}, "expert_resident_by_dir_after": {d: after}}


def clock_readiness_note(report: dict) -> str:
    """NOT in Task 1's check_arm — added 2026-09-21 after 82 gate failures on a same-config
    microbenchmark, traced to SM-clock instability. Task 1's arm script has no clock check at all
    (only an NVMe idle check); this records what run_arm.sh's own readiness gate already enforced
    before timing started, for the reader of this verdict rather than as a second gate."""
    samples = [
        row["clock_sm_start_mhz"]
        for row in report.get("per_session", [])
        if row.get("clock_sm_start_mhz") is not None
    ]
    if not samples:
        return "CLOCK no per-session clock samples recorded"
    return f"CLOCK per-session clock_sm_start_mhz samples: {samples}"


def judge(
    report: dict,
    *,
    root: str,
    head: str,
    mirror: bool,
    traced: bool,
    task1_module,
    manifest: dict | None = None,
    reference_arms: list[str] | None = None,
    arm_json_path: str | None = None,
) -> dict:
    """(problems, acknowledged_problems, notes, valid). `valid` is Task 1's own definition:
    True only if `problems` (all of them, acknowledged or not) is empty — this function does not
    soften that; a caller who wants to treat the acknowledged step-latency gap as non-blocking
    during the transition must say so explicitly by filtering `acknowledged_problems` out itself.
    """
    sessions = len(report.get("per_session") or [])
    problems, notes = task1_module.check_arm(
        report, root=root, head=head, mirror=mirror, traced=traced, sessions=sessions
    )
    cache = residency_cache_dict(report)
    if cache is not None:
        problems += task1_module.check_cache(cache)
        notes.append("RESIDENCY whole-arm (before_server -> after_timed_set); no per-session "
                     "instrumentation exists over HTTP, see report_builder.py")
    else:
        notes.append("RESIDENCY not judged: no residency section in this report")
    problems += compile_contamination_problems(report)
    notes.append(clock_readiness_note(report))

    contended, why = task1_module.contention(report)
    notes.append(f"CONTENDED {contended} {why}")

    notes += task1_module.session_outliers(report)

    if manifest is not None:
        gen, tree = task1_module.generation(report, root, manifest)
        notes.insert(0, f"GENERATION {gen}" + (f" (python tree {tree})" if tree else ""))

    if reference_arms is not None and arm_json_path is not None:
        cross = task1_module.cross_arm_outliers(report, arm_json_path, reference_arms)
        notes += cross if cross is not None else ["CROSS-ARM not judged: no reference arm"]

    acknowledged = [p for p in problems if is_acknowledged_step_latency_problem(p)]
    unacknowledged = [p for p in problems if not is_acknowledged_step_latency_problem(p)]

    return {
        "problems": problems,
        "acknowledged_problems": acknowledged,
        "unacknowledged_problems": unacknowledged,
        "notes": notes,
        "valid": not problems,
        "valid_except_acknowledged_gaps": not unacknowledged,
    }
