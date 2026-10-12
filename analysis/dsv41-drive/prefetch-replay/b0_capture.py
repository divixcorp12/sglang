"""B0 capture driver: a session-split residency-capture corpus on /mnt/nvme1, blocks of sessions per server start.

The sampled sessions (b0_split.sample_sessions over the corpus) are served in blocks of ``--block`` (default 25).
Each block starts one dspark-both-misscut server whose stage trace, router capture, residency switch, accept log
and metrics point at ``OUT_DIR/blocks/<block-id>/``. The block's sessions go to that server one after another, each
through its own run_capture_sessions.py call, and the launcher is stopped alone after the last one so the final
ring reads land. The first session served after each server start is ``warm_start`` and is kept.

Per session ``OUT_DIR/<session_id>/`` holds results.jsonl, rids.json (rid -> session, turn, B0 split),
drive-samples.jsonl (mirror drive counters around the session, for drive_sessions.flag_sessions) and DONE.json
(block, warm_start). The block is checked for completeness (capture_verify.check_capture, a residency record per
verify forward) and each session for answered turns. Sessions without DONE are re-served once in a new block;
the ones still without DONE move to ``OUT_DIR/failed/<session_id>`` and are listed. A rerun skips sessions with
DONE.json or FAILED.json and regroups the rest into new blocks.

At the end ``OUT_DIR/manifest.json`` is written by b0_manifest. ``--pilot K`` serves the first K sampled sessions
as one block and writes no manifest.

Locks: rowimg-disk.lock, then cc-gpu.lock, through run_locks.take, held across all blocks; the server runs under
taskset on SERVER_CORES.

Usage: b0_capture.py OUT_DIR [--sessions 300] [--seed 0] [--block 25] [--pilot 2] [--max-tokens 1024]
"""

from __future__ import annotations

import argparse
import functools
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
BLOCKS_DIR = "blocks"
# Sessions per server start; 25 gives 12 starts for the 300-session corpus.
DEFAULT_BLOCK = 25
# Blocks in a row whose retry left every session without DONE: a systematic fault, so the run stops.
MAX_CONSECUTIVE_FAILED_BLOCKS = 2
# The acceptance bar: sessions captured complete over sessions sampled.
MIN_COMPLETE_FRACTION = 0.9
_COMPLETENESS_ERRORS = (RuntimeError, OSError, ValueError, KeyError)


def recipe_env() -> dict:
    """The arm's overrides plus the residency switch: the part of the server env that is the same for every session."""
    overrides, _ = both_cpu_ab.ARMS[ARM]
    return {**overrides, "SGLANG_DSV41_RESIDENCY_CAPTURE": "1"}


def server_env(block_dir: str) -> dict:
    """The recipe with the trace, router capture, accept log and metrics written under ``block_dir``."""
    return arm_env.arm_env({
        **recipe_env(),
        "SGLANG_DSV41_EXPERT_TRACE_PATH": os.path.join(block_dir, "trace.jsonl"),
        # The residency side files take ROUTER_CAPTURE_PATH + ".res", which is RESIDENCY_PREFIX.
        "SGLANG_DSV41_ROUTER_CAPTURE_PATH": os.path.join(block_dir, "router"),
        "SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH": os.path.join(block_dir, "verify-accept"),
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(block_dir, "metrics.jsonl"),
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


def residency_complete(block_dir: str) -> None:
    """Raise RuntimeError when a verify forward in the route log has no residency record."""
    import tier_sim
    from sglang.srt.layers.moe.residency_capture import load_residency

    res = load_residency(os.path.join(block_dir, RESIDENCY_PREFIX))
    have = {int(seq) for seq in res.keys[:, 0]}
    loaded = tier_sim.load_forwards(os.path.join(block_dir, "trace.jsonl"), allow_dropped=True)
    missing = [f["seq"] for f in loaded["forwards"] if f["kind"] == "graph" and f["verify"] and f["seq"] not in have]
    if missing:
        raise RuntimeError(f"{len(missing)} verify forwards have no residency record (first seq {missing[0]})")


def check_block(block_dir: str) -> dict:
    """The block's completeness: no lost forward, router and residency records for each verify. Returns the summary."""
    summary = capture_verify.check_capture(block_dir)
    residency_complete(block_dir)
    return summary


def check_session(session: dict, session_dir: str) -> None:
    """Raise RuntimeError unless every turn of ``session`` was answered."""
    results_complete(session_dir, session=session)


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _failed_path(out_dir: str, name: str) -> str:
    """Where ``OUT_DIR/<name>`` goes when it fails: a session id, or ``blocks/<block-id>``."""
    return os.path.join(out_dir, FAILED_DIR, name)


def _quarantine(*, out_dir: str, name: str) -> str:
    """Move ``OUT_DIR/<name>`` under failed/, so the manifest never lists it; returns the new path."""
    dest = _failed_path(out_dir, name)
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(os.path.join(out_dir, name)):
        shutil.move(os.path.join(out_dir, name), dest)
    else:
        os.makedirs(dest)
    return dest


def plan_blocks(*, sessions: list[dict], block_size: int) -> list[list[dict]]:
    """Consecutive groups of up to ``block_size`` sessions, in the order given."""
    if block_size < 1:
        raise ValueError(f"block size {block_size} is below 1")
    return [sessions[i : i + block_size] for i in range(0, len(sessions), block_size)]


def _next_block_id(out_dir: str) -> str:
    """One past the highest id in blocks/ and failed/blocks/; its dir is created here, so the id is taken."""
    taken = []
    for parent in (os.path.join(out_dir, BLOCKS_DIR), _failed_path(out_dir, BLOCKS_DIR)):
        if os.path.isdir(parent):
            taken += [int(n[1:]) for n in os.listdir(parent) if n[:1] == "b" and n[1:].isdigit()]
    block_id = f"b{max(taken, default=-1) + 1:03d}"
    os.makedirs(os.path.join(out_dir, BLOCKS_DIR, block_id))
    return block_id


def _clean_session_dir(out_dir: str, sid: str) -> str:
    sdir = os.path.join(out_dir, sid)
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir)
    return sdir


def _serve_in_order(*, out_dir: str, block_dir: str, sessions: list[dict], start, serve_session, stop):
    """One server start, every session served on it in order, then stop. Returns (served sessions, {sid: error})."""
    served, errors = [], {}
    handle = start(block_dir)
    try:
        for session in sessions:
            sid = session["session_id"]
            try:
                serve_session(handle, session, os.path.join(out_dir, sid), block_dir)
                served.append(session)
            except _COMPLETENESS_ERRORS as e:
                errors[sid] = f"{type(e).__name__}: {e}"
    finally:
        stop(handle)
    return served, errors


def _mark_done(*, out_dir: str, session: dict, split_name: str, block_id: str, warm_start: bool, attempts: int,
               summary: dict) -> None:
    sdir = os.path.join(out_dir, session["session_id"])
    _write_json(os.path.join(sdir, "rids.json"), rid_map(session, split_name=split_name))
    _write_json(os.path.join(sdir, DONE_NAME), {"session_id": session["session_id"], "block": block_id,
                                                "warm_start": warm_start, "attempts": attempts, "summary": summary})


def _serve_block_attempt(*, out_dir: str, names: dict, block_id: str, sessions: list[dict], attempt: int, start,
                         serve_session, stop, check_block, check_session) -> dict:
    """Serve ``sessions`` on one server into ``blocks/<block_id>``; DONE goes only to sessions that pass both the
    block check and their own. Returns {"done": [sid], "errors": {sid: reason}}."""
    block_dir = os.path.join(out_dir, BLOCKS_DIR, block_id)
    rids = {}
    for session in sessions:
        rids.update(rid_map(session, split_name=names.get(session["session_id"], "")))
    _write_json(os.path.join(block_dir, "rids.json"), rids)
    for session in sessions:
        _clean_session_dir(out_dir, session["session_id"])
    tag = f"attempt {attempt} (block {block_id})"
    try:
        served, errors = _serve_in_order(out_dir=out_dir, block_dir=block_dir, sessions=sessions, start=start,
                                         serve_session=serve_session, stop=stop)
        summary = check_block(block_dir)
    except _COMPLETENESS_ERRORS as e:
        return {"done": [], "errors": {s["session_id"]: f"{tag}: {type(e).__name__}: {e}" for s in sessions}}
    errors = {sid: f"{tag}: {reason}" for sid, reason in errors.items()}
    done = []
    for session in served:
        sid = session["session_id"]
        try:
            check_session(session, os.path.join(out_dir, sid))
        except _COMPLETENESS_ERRORS as e:
            errors[sid] = f"{tag}: {type(e).__name__}: {e}"
            continue
        # warm_start marks the first session served after the server start; it is kept, not dropped.
        _mark_done(out_dir=out_dir, session=session, split_name=names.get(sid, ""), block_id=block_id,
                   warm_start=session is sessions[0], attempts=attempt, summary=summary)
        done.append(sid)
    return {"done": done, "errors": errors}


def _serve_block_with_retry(*, out_dir: str, block: list[dict], serve_attempt) -> int:
    """Serve a block, re-serve once its sessions without DONE under a new block id, and move what still lacks DONE
    to failed/. A block dir with no DONE session moves to failed/blocks/. Returns how many sessions got DONE."""
    left, reasons, n_done = block, {}, 0
    for attempt in range(1, ATTEMPTS + 1):
        block_id = _next_block_id(out_dir)
        result = serve_attempt(block_id=block_id, sessions=left, attempt=attempt)
        for sid, reason in result["errors"].items():
            reasons.setdefault(sid, []).append(reason)
        if not result["done"]:
            _quarantine(out_dir=out_dir, name=os.path.join(BLOCKS_DIR, block_id))
        n_done += len(result["done"])
        left = [s for s in left if s["session_id"] not in result["done"]]
        if not left:
            return n_done
    for session in left:
        sid = session["session_id"]
        dest = _quarantine(out_dir=out_dir, name=sid)
        _write_json(os.path.join(dest, FAILED_NAME),
                    {"session_id": sid, "attempts": ATTEMPTS, "reasons": reasons.get(sid, [])})
    return n_done


def _has_done(out_dir: str, sid: str) -> bool:
    return os.path.exists(os.path.join(out_dir, sid, DONE_NAME))


def _has_failed(out_dir: str, sid: str) -> bool:
    return os.path.exists(os.path.join(_failed_path(out_dir, sid), FAILED_NAME))


def _outcome(*, out_dir: str, sessions: list[dict]) -> dict:
    """{"done", "failed", "blocks": {sid: block}, "warm_start"} read back from every DONE.json and FAILED.json."""
    out = {"done": [], "failed": [], "blocks": {}, "warm_start": []}
    for session in sessions:
        sid = session["session_id"]
        if _has_done(out_dir, sid):
            with open(os.path.join(out_dir, sid, DONE_NAME)) as f:
                done = json.load(f)
            out["done"].append(sid)
            out["blocks"][sid] = done["block"]
            if done["warm_start"]:
                out["warm_start"].append(sid)
        elif _has_failed(out_dir, sid):
            out["failed"].append(sid)
    return out


def capture_blocks(*, out_dir: str, sessions: list[dict], split: dict, block_size: int, start, serve_session, stop,
                   check_block, check_session) -> dict:
    """Serve the sessions without DONE or FAILED in blocks of ``block_size``, one server start per block.

    ``start(block_dir)`` returns a server handle or raises; ``serve_session(handle, session, session_dir,
    block_dir)`` serves one session; ``stop(handle)`` runs after the block's last session; ``check_block(block_dir)``
    returns the block summary and ``check_session(session, session_dir)`` raises when the session is incomplete.
    Stops after MAX_CONSECUTIVE_FAILED_BLOCKS blocks in a row end with no session done.
    Returns {"done", "failed", "blocks": {sid: block_id}, "warm_start"}, earlier runs' sessions included."""
    serve_attempt = functools.partial(
        _serve_block_attempt, out_dir=out_dir, names=split_of(split), start=start, serve_session=serve_session,
        stop=stop, check_block=check_block, check_session=check_session,
    )
    pending = [s for s in sessions if not _has_done(out_dir, s["session_id"])
               and not _has_failed(out_dir, s["session_id"])]
    streak = 0
    for block in plan_blocks(sessions=pending, block_size=block_size):
        n_done = _serve_block_with_retry(out_dir=out_dir, block=block, serve_attempt=serve_attempt)
        streak = 0 if n_done else streak + 1
        if streak >= MAX_CONSECUTIVE_FAILED_BLOCKS:
            break
    return _outcome(out_dir=out_dir, sessions=sessions)


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


def _start_server(block_dir: str) -> subprocess.Popen:
    """Launch the server with its capture files in ``block_dir`` and wait for health; stops it before raising."""
    env = os.environ | server_env(block_dir) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PORT, dspark=True).argv()
    with open(os.path.join(block_dir, "server.log"), "w") as log:
        server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                  stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
    if not capture_verify._healthy(PORT, time.monotonic() + arm_env.DSPARK_HEALTH_TIMEOUT_S):
        capture_verify.stop_server(server)
        raise RuntimeError(f"block {os.path.basename(block_dir)}: the server never became healthy")
    return server


def _serve_one(server: subprocess.Popen, session: dict, session_dir: str, block_dir: str, *, max_tokens: int) -> None:
    """Serve one session's turns on the running server; the drive samples and results land in ``session_dir``."""
    sid = session["session_id"]
    if server.poll() is not None:
        raise RuntimeError(f"session {sid}: the server exited {server.returncode} before the session")
    sessions_file = os.path.join(session_dir, "session.jsonl")
    with open(sessions_file, "w") as f:
        f.write(json.dumps(session) + "\n")
    samples = [_drive_sample("session_start", sid)]
    rc = subprocess.run(
        ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, RUNNER, "--port", str(PORT),
         "--sessions", sessions_file, "--session-ids", sid, "--max-tokens", str(max_tokens),
         "--results", os.path.join(session_dir, "results.jsonl")],
        cwd=REPO,
    ).returncode
    samples.append(_drive_sample("session_end", sid))
    with open(os.path.join(session_dir, "drive-samples.jsonl"), "w") as f:
        f.write("".join(json.dumps(s) + "\n" for s in samples))
    if rc != 0:
        raise RuntimeError(f"session {sid}: the session runner exited {rc} (block {os.path.basename(block_dir)})")


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
    write_manifest(out_dir, commit=_commit(), recipe=recipe, split=split, failed=outcome["failed"], flagged=flagged,
                   session_block=outcome["blocks"], warm_start=outcome["warm_start"])
    print(json.dumps({"sampled": len(sessions), "done": len(outcome["done"]), "failed": outcome["failed"],
                      "flagged": [f["session_id"] for f in flagged]}, indent=1))
    return 0 if len(outcome["done"]) >= MIN_COMPLETE_FRACTION * len(sessions) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--sessions", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="sessions served per server start")
    p.add_argument("--pilot", type=int, default=0,
                   help="serve only the first K sampled sessions as one block, no manifest")
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
    block_size = a.pilot if a.pilot else a.block
    with run_locks.take(DISK_LOCK), run_locks.take(arm_env.GPU_LOCK):
        outcome = capture_blocks(
            out_dir=out_dir, sessions=sessions, split=split, block_size=block_size,
            start=_start_server,
            serve_session=functools.partial(_serve_one, max_tokens=a.max_tokens),
            stop=capture_verify.stop_server, check_block=check_block, check_session=check_session,
        )
    if a.pilot:
        print(json.dumps(outcome, indent=1))
        return 0 if not outcome["failed"] else 1
    recipe = {"arm": ARM, "env": arm_env.arm_env(recipe_env()), "seed": a.seed, "sessions": a.sessions,
              "max_tokens": a.max_tokens, "model_path": arm_env.MODEL_PATH}
    return _finish(out_dir=out_dir, sessions=sessions, split=split, outcome=outcome, recipe=recipe)


if __name__ == "__main__":
    sys.exit(main())
