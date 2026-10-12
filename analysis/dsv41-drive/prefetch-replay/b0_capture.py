"""B0 capture driver: a session-split residency-capture corpus on /mnt/nvme1, one served session at a time.

Per sampled session (b0_split.sample_sessions over the corpus) it starts a dspark-both-misscut server with the stage
trace, router capture, the residency switch and the accept log pointed at ``OUT_DIR/<session_id>/``, serves the
session's turns through run_capture_sessions.py, and stops the launcher alone so the final ring reads land. The
session is then checked for completeness (capture_verify.check_capture, residency record per verify forward, every
turn answered). A session that fails is wiped and served once more; one that fails again moves to
``OUT_DIR/failed/<session_id>`` and is listed. A passing session gets ``DONE.json``, so a rerun resumes.

Each session dir also holds ``rids.json`` (rid -> session, turn, B0 split) and ``drive-samples.jsonl`` (mirror drive
counters around the serve, for drive_sessions.flag_sessions). At the end ``OUT_DIR/manifest.json`` is written by
b0_manifest. ``--pilot K`` serves only the first K sampled sessions and writes no manifest.

Locks: rowimg-disk.lock, then cc-gpu.lock, through run_locks.take; the server runs under taskset on SERVER_CORES.

Usage: b0_capture.py OUT_DIR [--sessions 300] [--seed 0] [--pilot 2] [--max-tokens 1024]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dspark"))
import capture_verify  # noqa: E402  (also puts arm_env, run_locks, session_subset, scripts/dsv41 on sys.path)
import arm_env  # noqa: E402
import both_cpu_ab  # noqa: E402
import drive_sessions  # noqa: E402
import run_locks  # noqa: E402
import session_subset  # noqa: E402
from b0_guards import ARM, refuse_launch  # noqa: E402
from b0_manifest import write_manifest  # noqa: E402
from b0_split import sample_sessions, split_sessions  # noqa: E402

REPO = capture_verify.REPO
PORT = 30033
DISK_LOCK = capture_verify.DISK_LOCK
RUNNER = os.path.join(REPO, "scripts", "expert_prediction", "benchmarks", "run_capture_sessions.py")
RESIDENCY_PREFIX = "router.res"
DONE_NAME = "DONE.json"
FAILED_DIR = "failed"
FAILED_NAME = "FAILED.json"
ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 5
# The acceptance bar: sessions captured complete over sessions sampled.
MIN_COMPLETE_FRACTION = 0.9
_COMPLETENESS_ERRORS = (RuntimeError, OSError, ValueError, KeyError)


def recipe_env() -> dict:
    """The arm's overrides plus the residency switch: the part of the server env that is the same for every session."""
    overrides, _ = both_cpu_ab.ARMS[ARM]
    return {**overrides, "SGLANG_DSV41_RESIDENCY_CAPTURE": "1"}


def server_env(session_dir: str) -> dict:
    """The recipe with the trace, router capture and accept log written under ``session_dir``."""
    return arm_env.arm_env({
        **recipe_env(),
        "SGLANG_DSV41_EXPERT_TRACE_PATH": os.path.join(session_dir, "trace.jsonl"),
        # The residency side files take ROUTER_CAPTURE_PATH + ".res", which is RESIDENCY_PREFIX.
        "SGLANG_DSV41_ROUTER_CAPTURE_PATH": os.path.join(session_dir, "router"),
        "SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH": os.path.join(session_dir, "verify-accept"),
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(session_dir, "metrics.jsonl"),
    })


def rid_map(session: dict, *, split_name: str) -> dict:
    """rid -> session/turn/split for every turn, rids named as run_capture_sessions.py names them."""
    sid = session["session_id"]
    return {
        f"{sid}-t{turn}": {"session_id": sid, "turn": turn, "split": split_name}
        for turn in range(len(session["turns"]))
    }


def split_of(split: dict) -> dict:
    return {sid: name for name, ids in split.items() for sid in ids}


def results_complete(session_dir: str, *, session: dict) -> None:
    """Raise RuntimeError unless results.jsonl answered every turn without an error record."""
    answered, errors = set(), []
    with open(os.path.join(session_dir, "results.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                errors.append(rec["rid"])
            else:
                answered.add(rec["turn"])
    if errors or answered != set(range(len(session["turns"]))):
        raise RuntimeError(
            f"session {session['session_id']}: {len(answered)} of {len(session['turns'])} turns answered, "
            f"errors on {errors}"
        )


def residency_complete(session_dir: str) -> None:
    """Raise RuntimeError when a verify forward in the route log has no residency record."""
    import tier_sim
    from sglang.srt.layers.moe.residency_capture import load_residency

    res = load_residency(os.path.join(session_dir, RESIDENCY_PREFIX))
    have = {int(seq) for seq in res.keys[:, 0]}
    loaded = tier_sim.load_forwards(os.path.join(session_dir, "trace.jsonl"), allow_dropped=True)
    missing = [f["seq"] for f in loaded["forwards"] if f["kind"] == "graph" and f["verify"] and f["seq"] not in have]
    if missing:
        raise RuntimeError(f"{len(missing)} verify forwards have no residency record (first seq {missing[0]})")


def check_complete(session_dir: str, *, session: dict) -> dict:
    """The session's completeness: every turn answered, no lost forward, router and residency records for each verify."""
    results_complete(session_dir, session=session)
    summary = capture_verify.check_capture(session_dir)
    residency_complete(session_dir)
    return summary


def serve_with_retry(*, session_dir: str, attempt, check, attempts: int = ATTEMPTS) -> dict:
    """Run ``attempt()`` on a clean session dir, then ``check()``; a failure wipes the dir and goes again, up to
    ``attempts`` times. Returns {"ok", "attempts", "summary" or "reasons"}."""
    reasons = []
    for n in range(1, attempts + 1):
        shutil.rmtree(session_dir, ignore_errors=True)
        os.makedirs(session_dir)
        try:
            attempt()
            return {"ok": True, "attempts": n, "summary": check()}
        except _COMPLETENESS_ERRORS as e:
            reasons.append(f"attempt {n}: {type(e).__name__}: {e}")
    return {"ok": False, "attempts": attempts, "reasons": reasons}


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _failed_path(out_dir: str, sid: str) -> str:
    return os.path.join(out_dir, FAILED_DIR, sid)


def _quarantine(out_dir: str, sid: str, result: dict) -> None:
    """Move a twice-failed session out of the corpus root, so the manifest never lists it."""
    dest = _failed_path(out_dir, sid)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(os.path.join(out_dir, sid), dest)
    _write_json(os.path.join(dest, FAILED_NAME), {"session_id": sid, **result})


def capture_all(*, out_dir: str, sessions: list[dict], split: dict, serve, check) -> dict:
    """Serve every session without a DONE marker; ``serve(session, session_dir)`` runs one attempt and
    ``check(session, session_dir)`` returns its summary or raises. Returns {"done", "failed"} session ids.
    Stops early after MAX_CONSECUTIVE_FAILURES sessions fail in a row (a systematic fault, not bad luck)."""
    names = split_of(split)
    done, failed, streak = [], [], 0
    for session in sessions:
        sid = session["session_id"]
        sdir = os.path.join(out_dir, sid)
        if os.path.exists(os.path.join(sdir, DONE_NAME)):
            done.append(sid)
            continue
        if os.path.exists(os.path.join(_failed_path(out_dir, sid), FAILED_NAME)):
            failed.append(sid)
            continue
        result = serve_with_retry(
            session_dir=sdir,
            attempt=lambda s=session, d=sdir: serve(s, d),
            check=lambda s=session, d=sdir: check(s, d),
        )
        if result["ok"]:
            _write_json(os.path.join(sdir, "rids.json"), rid_map(session, split_name=names.get(sid, "")))
            _write_json(os.path.join(sdir, DONE_NAME), {"session_id": sid, **result})
            done.append(sid)
            streak = 0
            continue
        _quarantine(out_dir, sid, result)
        failed.append(sid)
        streak += 1
        if streak >= MAX_CONSECUTIVE_FAILURES:
            break
    return {"done": done, "failed": failed}


def drive_flags(*, out_dir: str, ids: list[str]) -> list[dict]:
    """The captured sessions whose mirror-drive busy s per GB is over FLAG_EXCESS of the drive's median."""
    samples = []
    for sid in ids:
        path = os.path.join(out_dir, sid, "drive-samples.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                samples += [json.loads(line) for line in f if line.strip()]
    loads = {ARM: drive_sessions.session_drive_load(samples)}
    return drive_sessions.flag_sessions(loads)[ARM]


def _drive_sample(label: str, sid: str) -> dict:
    import provenance

    return {"label": label, "session_id": sid, **provenance.drive_sample()}


def _serve_session(session: dict, session_dir: str, *, max_tokens: int) -> None:
    """One attempt: server up, the session's turns served, launcher stopped so the rings' last read lands."""
    sid = session["session_id"]
    sessions_file = os.path.join(session_dir, "session.jsonl")
    with open(sessions_file, "w") as f:
        f.write(json.dumps(session) + "\n")
    env = os.environ | server_env(session_dir) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PORT, dspark=True).argv()
    with open(os.path.join(session_dir, "server.log"), "w") as log:
        server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                  stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
    samples = []
    try:
        if not capture_verify._healthy(PORT, time.monotonic() + arm_env.DSPARK_HEALTH_TIMEOUT_S):
            raise RuntimeError(f"session {sid}: the server never became healthy")
        samples.append(_drive_sample("session_start", sid))
        rc = subprocess.run(
            ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, RUNNER, "--port", str(PORT),
             "--sessions", sessions_file, "--session-ids", sid, "--max-tokens", str(max_tokens),
             "--results", os.path.join(session_dir, "results.jsonl")],
            cwd=REPO,
        ).returncode
        samples.append(_drive_sample("session_end", sid))
    finally:
        capture_verify.stop_server(server)
    with open(os.path.join(session_dir, "drive-samples.jsonl"), "w") as f:
        f.write("".join(json.dumps(s) + "\n" for s in samples))
    if rc != 0:
        raise RuntimeError(f"session {sid}: the session runner exited {rc}")


def _ps_lines() -> list[str]:
    return subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=True).stdout.splitlines()


def _commit() -> str:
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def _finish(*, out_dir: str, sessions: list[dict], split: dict, outcome: dict, recipe: dict) -> int:
    flagged = drive_flags(out_dir=out_dir, ids=outcome["done"])
    rids = {}
    for sid in outcome["done"]:
        with open(os.path.join(out_dir, sid, "rids.json")) as f:
            rids.update(json.load(f))
    _write_json(os.path.join(out_dir, "rids.json"), rids)
    write_manifest(out_dir, commit=_commit(), recipe=recipe, split=split, failed=outcome["failed"], flagged=flagged)
    print(json.dumps({"sampled": len(sessions), "done": len(outcome["done"]), "failed": outcome["failed"],
                      "flagged": [f["session_id"] for f in flagged]}, indent=1))
    return 0 if len(outcome["done"]) >= MIN_COMPLETE_FRACTION * len(sessions) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--sessions", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pilot", type=int, default=0, help="serve only the first K sampled sessions, no manifest")
    p.add_argument("--max-tokens", type=int, default=1024)
    a = p.parse_args(argv)
    out_dir = os.path.abspath(a.out_dir)
    refuse_launch(out_dir=out_dir, env=server_env(out_dir), arm=ARM, model_path=arm_env.MODEL_PATH,
                  ps_lines=_ps_lines())
    session_subset.verify_corpus_checksum()
    os.makedirs(out_dir, exist_ok=True)
    sampled = sample_sessions(session_subset.CORPUS_PATH, n=a.sessions, seed=a.seed)
    split = split_sessions([s["session_id"] for s in sampled], seed=a.seed)
    _write_json(os.path.join(out_dir, "split.json"), split)
    sessions = sampled[: a.pilot] if a.pilot else sampled
    by_id = {s["session_id"]: s for s in sessions}
    with run_locks.take(DISK_LOCK), run_locks.take(arm_env.GPU_LOCK):
        outcome = capture_all(
            out_dir=out_dir, sessions=sessions, split=split,
            serve=lambda s, d: _serve_session(s, d, max_tokens=a.max_tokens),
            check=lambda s, d: check_complete(d, session=by_id[s["session_id"]]),
        )
    if a.pilot:
        print(json.dumps(outcome, indent=1))
        return 0 if not outcome["failed"] else 1
    recipe = {"arm": ARM, "env": arm_env.arm_env(recipe_env()), "seed": a.seed, "sessions": a.sessions,
              "max_tokens": a.max_tokens, "model_path": arm_env.MODEL_PATH}
    return _finish(out_dir=out_dir, sessions=sessions, split=split, outcome=outcome, recipe=recipe)


if __name__ == "__main__":
    sys.exit(main())
