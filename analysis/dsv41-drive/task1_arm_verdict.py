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
    """Reject an arm that left expert shard bytes in the page cache, naming the directory."""
    before, after = cache.get("expert_resident_by_dir_before"), cache.get("expert_resident_by_dir_after")
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after):
        return ["expert page-cache residency was not measured per directory"]
    problems = []
    for d in sorted(before):
        grew = after[d] - before[d]
        if grew > MAX_EXPERT_CACHE_GROWTH_BYTES:
            problems.append(f"expert shard page-cache residency grew {grew / (1 << 30):.2f} GiB across the arm in {d}")
    return problems


def phase_residency(report: dict):
    """Per-directory residency at the phase boundaries, or None if the driver did not record them.

    boot  = before_engine -> engine_ready   (weight loading, not expert reads)
    timed = engine_ready  -> end of the last session (the phase the independence claim is about)"""
    res = report.get("expert_residency") or {}
    rows = report.get("per_session") or []
    before, ready = res.get("before_engine"), res.get("after_engine_ready")
    last = rows[-1].get("expert_resident_bytes") if rows else None
    if not all(isinstance(x, dict) and x for x in (before, ready, last)):
        return None
    if not (set(before) == set(ready) == set(last)):
        return None
    return {"before": before, "ready": ready, "last": last}


def check_timed_phase(phases: dict) -> list:
    """The independence gate: no directory's residency may grow more than the limit while timed."""
    problems = []
    for d in sorted(phases["ready"]):
        a, b = phases["ready"][d], phases["last"][d]
        if a is None or b is None:
            problems.append(f"residency of {d} could not be measured in the timed phase")
        elif b - a > MAX_EXPERT_CACHE_GROWTH_BYTES:
            problems.append(
                f"expert shard page-cache residency grew {(b - a) / (1 << 30):.2f} GiB during the timed sessions in {d}"
            )
    return problems


def boot_growth(phases: dict) -> dict:
    """{dir: bytes grown between before_engine and engine_ready}; None where unmeasured."""
    return {
        d: None if phases["before"][d] is None or phases["ready"][d] is None else phases["ready"][d] - phases["before"][d]
        for d in sorted(phases["ready"])
    }


def regime(phases) -> str:
    """Which regime the boot put the arm in, derived from its own recorded boot-phase growth."""
    if phases is None:
        return "unknown: no per-session residency recorded"
    growth = [g for g in boot_growth(phases).values()]
    if any(g is None for g in growth):
        return "unknown: boot-phase residency unmeasured"
    return "boot-populated" if max(growth) > MAX_EXPERT_CACHE_GROWTH_BYTES else "boot-warm"


def residency_notes(report: dict) -> list:
    """Where in the run each directory's residency changed, from the driver's own samples."""
    res = report.get("expert_residency") or {}
    steps = [("before_engine", res.get("before_engine")), ("engine_ready", res.get("after_engine_ready"))]
    steps += [(f"session_{i}", row.get("expert_resident_bytes")) for i, row in enumerate(report.get("per_session") or [])]
    notes, prev = [], None
    for label, sample in steps:
        if isinstance(sample, dict) and isinstance(prev, dict):
            for d in sorted(sample):
                if sample[d] is not None and prev.get(d) is not None and sample[d] != prev[d]:
                    notes.append(f"residency of {d} changed {(sample[d] - prev[d]) / (1 << 20):+.1f} MiB at {label}")
        if isinstance(sample, dict):
            prev = sample
    return notes


SECTOR_BYTES = 512
SESSION_TTFT_OUTLIER = 1.25   # x the fastest TTFT among the other sessions 1..n
P5_FLAT_KB = 256 * 1024       # Cached falling by less than this across boot counts as flat


def boot_phase(report: dict):
    """Device reads and meminfo change between the before_engine and engine_ready samples."""
    by = {b.get("label"): b for b in report.get("boundary_samples") or []}
    a, b = by.get("before_engine"), by.get("engine_ready")
    if not a or not b:
        return None
    reads = None
    if isinstance(a.get("diskstats_sectors"), dict) and isinstance(b.get("diskstats_sectors"), dict):
        reads = {k: (b["diskstats_sectors"][k] - a["diskstats_sectors"][k]) * SECTOR_BYTES for k in sorted(a["diskstats_sectors"])
                 if k in b["diskstats_sectors"]}
    mem = None
    if isinstance(a.get("meminfo_kb"), dict) and isinstance(b.get("meminfo_kb"), dict):
        mem = {k: b["meminfo_kb"][k] - a["meminfo_kb"][k] for k in sorted(a["meminfo_kb"]) if k in b["meminfo_kb"]}
    return {"device_read_bytes": reads, "meminfo_delta_kb": mem}


def p5_note(boot_growth_bytes, phase) -> str:
    """P5 (team-lead's hypothesis): a negative source-dir boot growth comes with Cached falling across boot."""
    if not phase or not phase.get("meminfo_delta_kb") or not boot_growth_bytes:
        return "P5 not testable: no boot-boundary meminfo"
    negative = {d: g for d, g in boot_growth_bytes.items() if g is not None and g < -P5_FLAT_KB * 1024}
    if not negative:
        return "P5 not testable in this arm: no directory's residency fell during boot"
    cached = phase["meminfo_delta_kb"].get("Cached")
    d, g = min(negative.items(), key=lambda kv: kv[1])
    verdict = "REFUTED (Cached flat or rising)" if cached >= -P5_FLAT_KB else "consistent (Cached fell)"
    return f"P5 {verdict}: {d} residency {g / (1 << 20):+.0f} MiB during boot, Cached {cached / 1024:+.0f} MiB"


def session_outliers(report: dict) -> list:
    """NOTES, never failures: a session whose TTFT is far above the fastest of its siblings'. Session 0 is
    exempt (it is cold and differs by up to 2x by design). The reference is the fastest sibling, not the
    median: with three sessions to compare, two disturbed ones would otherwise drag the median up and hide
    each other. Only TTFT is checked: decode tok/s differs 3.2-4.7 between sessions of a healthy arm because
    the prompts differ, so an arm-internal tok/s rule would either miss or cry wolf. An arm in which every
    session is slow is invisible to this check."""
    rows = report.get("per_session") or []
    notes = []
    for i in range(1, len(rows)):
        others = [r["ttft_s"] for j, r in enumerate(rows) if j not in (0, i) and r.get("ttft_s") is not None]
        if others and rows[i].get("ttft_s") is not None and rows[i]["ttft_s"] > SESSION_TTFT_OUTLIER * min(others):
            notes.append(
                f"OUTLIER session_{i}: ttft {rows[i]['ttft_s']:.1f} s vs {min(others):.1f} s for its fastest sibling "
                f"(decode {rows[i].get('decode_tok_s', 0):.3f} tok/s): a disturbed session, not gated"
            )
    return notes


def boundary_notes(report: dict) -> list:
    """Load and busy foreign processes seen at any boundary, so a disturbance can be explained."""
    notes = []
    for b in report.get("boundary_samples") or []:
        busy = [f"{p['name']}({p['cpu_pct']:.0f}%)" for p in (b.get("top_other_cpu") or []) if p["cpu_pct"] >= 20]
        load = (b.get("loadavg") or [None])[0]
        if busy or (load is not None and load >= 4):
            notes.append(f"at {b['label']}: load1 {load}, busy other processes: {', '.join(busy) or 'none >=20%'}")
    return notes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("arm_json")
    p.add_argument("--root", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--mirror", choices=("on", "off"), required=True)
    p.add_argument("--trace", choices=("T", "U"), required=True)
    p.add_argument("--cache")
    p.add_argument("--summary-json", help="write regime, boot-phase growth and timed-phase growth here")
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
    phases = phase_residency(report)
    if phases is not None:
        # The whole-arm figure is a note: it mixes boot weight loading with the timed sessions.
        problems += check_timed_phase(phases)
    elif cache is not None:
        problems += check_cache(cache)
        notes.append("no per-session residency: gated on the whole-arm before/after instead")
    if cache is not None and isinstance(cache.get("expert_resident_by_dir_before"), dict):
        b, a = cache["expert_resident_by_dir_before"], cache.get("expert_resident_by_dir_after") or {}
        notes += [f"whole-arm residency of {d} {(a.get(d, 0) - b[d]) / (1 << 20):+.1f} MiB" for d in sorted(b)]
    notes += residency_notes(report)
    kind = regime(phases)
    growth = boot_growth(phases) if phases is not None else None
    phase = boot_phase(report)
    notes.insert(0, f"REGIME {kind} boot_growth_bytes={json.dumps(growth)}")
    notes.insert(1, f"BOOT_PHASE {json.dumps(phase)}")
    notes.append(p5_note(growth, phase))
    notes += session_outliers(report)
    notes += boundary_notes(report)
    if args.summary_json:
        timed = None if phases is None else {d: phases["last"][d] - phases["ready"][d] for d in phases["ready"]
                                              if phases["last"][d] is not None and phases["ready"][d] is not None}
        with open(args.summary_json, "w") as f:
            json.dump({"regime": kind, "boot_growth_bytes": growth, "timed_growth_bytes": timed, "boot_phase": phase,
                       "outlier_sessions": [n for n in notes if n.startswith("OUTLIER")], "valid": not problems}, f, indent=2)
    for n in notes:
        print("NOTE", n)
    for x in problems:
        print("PROBLEM", x)
    print("INVALID" if problems else "VALID")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
