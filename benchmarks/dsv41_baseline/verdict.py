"""Judge one HTTP-driven arm with Task 1's own `check_arm` and friends, not a second copy.

Calls `task1_arm_verdict.check_arm` / `check_timed_phase` / `boot_growth` /
`contention` / `generation` /
`session_outliers` / `cross_arm_outliers` (via `task1_verdict.load_task1_verdict`)
against a report built by `report_builder.build_report`, and appends this campaign's
own checks (compile-contamination, the measured server-env stand-ins, the SM-clock readiness
note) as clearly-labeled additions — never edits to Task 1's functions.

Two kinds of gap are acknowledged rather than hidden, and both are reported: the absent
engine-side step latency (below), and the four checks `check_arm` makes of a process this
campaign drives over HTTP and cannot sample from the inside. For the second kind
`server_env_problems` asks the same question of the server's measured environment, so
acknowledging the unknowable form of the check does not drop the check.

**Known, currently-unresolved consequence: every arm reads INVALID.**
`check_arm` requires `per_session[i]["step_latency"]` to carry real percentiles;
`report_builder.STEP_LATENCY_UNAVAILABLE` is honest about not having them (see that
module's docstring for why). `judge()` therefore returns a PROBLEM for each session
lines of the form "session N: no step latency (...)" until that gap closes. This is
surfaced, not hidden: `judge()` returns them like any other problem, and
`compile_and_clock_are_the_only_problems` tells a caller whether the remaining,
non-acknowledged problems are empty — the caller decides whether that is enough to
proceed, this module does not decide for it.
"""

from __future__ import annotations

import os

STEP_LATENCY_PROBLEM_PREFIX = "session "  # "session {i}: no step latency (...)" — check_arm's own wording
STEP_LATENCY_PROBLEM_SUFFIX = "no step latency"


def is_acknowledged_step_latency_problem(problem: str) -> bool:
    return problem.startswith(STEP_LATENCY_PROBLEM_PREFIX) and STEP_LATENCY_PROBLEM_SUFFIX in problem


READER_ENV = "SGLANG_MOE_EXPERT_FILE_READER"
GATHER_ENV = "SGLANG_MOE_EXPERT_GRAPH_GATHER"
TRUE_VALUES = ("1", "true", "yes", "on")

# check_arm asks four things of a process this campaign cannot sample from the inside: the
# server's sglang import path, its resolved (default-applied) knob values, and the two knobs read
# from them. `report_builder` now reports those as unavailable instead of handing over the
# harness's own values, so check_arm says "unknown" where it used to say something false. These
# are matched by wording, which is safe only because `task1_verdict` refuses to load the file at
# all unless its sha256 still matches: a reworded check fails the pin, loudly, before it reaches
# this list. `server_env_problems` re-asks each question of the server's measured environment.
ACKNOWLEDGED_SERVER_PROVENANCE_MARKERS = (
    "imported sglang from None",
    "provenance fields unavailable",
    f"{READER_ENV} resolved to None",
    f"{GATHER_ENV} did not resolve to true",
)


def is_acknowledged_server_provenance_problem(problem: str) -> bool:
    return any(marker in problem for marker in ACKNOWLEDGED_SERVER_PROVENANCE_MARKERS)


def server_env_problems(report: dict, *, root: str) -> list[str]:
    """NOT in Task 1's check_arm — this campaign's measured stand-ins for the checks above.

    Each reads the server's own /proc/<pid>/environ (`provenance.sglang_env`), which run_arm.sh has
    already string-compared against the arm's expected env. An explicitly set knob is therefore a
    fact about the server; a knob left unset is a default applied inside a process this harness
    cannot read, and is reported as unknown rather than assumed."""
    prov = report.get("provenance") or {}
    env = prov.get("sglang_env")
    if not env:
        return ["no server environment recorded: none of the env checks below were made"]

    problems = []
    reader = env.get(READER_ENV)
    if reader is None:
        problems.append(f"{READER_ENV} is unset in the server, so its default decides the reader and this arm cannot show which")
    elif reader != "uring_direct":
        problems.append(f"{READER_ENV}={reader!r} in the server, not 'uring_direct': reads may fill the page cache")

    gather = env.get(GATHER_ENV)
    if gather is None:
        problems.append(f"{GATHER_ENV} is unset in the server, so this arm cannot show it used the in-graph reader")
    elif gather.strip().lower() not in TRUE_VALUES:
        problems.append(f"{GATHER_ENV}={gather!r} in the server: this is not the in-graph reader")

    # Stands in for check_arm's sglang_file test: the import path itself is unobservable, but the
    # only tree the server can import from is the one PYTHONPATH names.
    want = os.path.join(os.path.realpath(root), "python")
    entries = [os.path.realpath(e) for e in (env.get("PYTHONPATH") or "").split(os.pathsep) if e]
    if not entries:
        problems.append("PYTHONPATH is unset in the server, so which sglang tree it imported is unconstrained")
    elif entries[0] != want:
        problems.append(f"the server's first PYTHONPATH entry is {entries[0]!r}, not {want!r}")
    return problems


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
    """Convert the HTTP harness's three boundaries to Task 1's phase shape.

    Return None when any required measurement is absent; judge then fails closed.
    """
    residency = report.get("residency")
    if not isinstance(residency, dict) or not isinstance(residency.get("dir"), str) or not residency["dir"]:
        return None
    before = residency.get("before_server")
    ready = residency.get("server_ready")
    last = residency.get("after_timed_set")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in (before, ready, last)):
        return None
    d = residency["dir"]
    return {"before": {d: before}, "ready": {d: ready}, "last": {d: last}}


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
        problems += task1_module.check_timed_phase(cache)
        for directory, growth in task1_module.boot_growth(cache).items():
            notes.append(f"RESIDENCY startup growth in {directory}: {growth / (1 << 20):+.1f} MiB "
                         "(before_server -> server_ready; recorded, not gated)")
        notes.append("RESIDENCY timed gate (server_ready -> after_timed_set); "
                     "no per-session instrumentation exists over HTTP, see report_builder.py")
    else:
        problems.append("residency phase measurements missing or invalid: need directory and "
                        "before_server, server_ready, after_timed_set byte counts")
    problems += compile_contamination_problems(report)
    problems += server_env_problems(report, root=root)
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

    def acknowledged_gap(problem: str) -> bool:
        return is_acknowledged_step_latency_problem(problem) or is_acknowledged_server_provenance_problem(problem)

    acknowledged = [p for p in problems if acknowledged_gap(p)]
    unacknowledged = [p for p in problems if not acknowledged_gap(p)]

    return {
        "problems": problems,
        "acknowledged_problems": acknowledged,
        "unacknowledged_problems": unacknowledged,
        "notes": notes,
        "valid": not problems,
        "valid_except_acknowledged_gaps": not unacknowledged,
    }
