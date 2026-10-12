"""R1 stage records: atomic DONE.json with input/output hashes, the out-root guard and the invocation log (S1.5).

A stage writes ``<out>/stages/<name>/DONE.json`` with the sha256 of its inputs and outputs; it is skipped only when
the recorded inputs still match (done_state). guard_out_root refuses any output directory outside the work root or
inside the B1 run directory. Every driver invocation is appended to ``<out>/invocations.jsonl``.

Inputs: paths and in-memory JSON objects handed in by the driver; no corpus data is read here.
Usage: imported by r1_run.py; ``from r1_stage import StageCtx, done_state, write_done``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Literal

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)

OUT_ROOT = "/mnt/nvme1/nvfp4-work"
FORBIDDEN = ("b1-run-20261012",)
DONE_NAME = "DONE.json"
INVOCATIONS_NAME = "invocations.jsonl"
_CHUNK = 1 << 20


class StageCtx(msgspec.Struct, frozen=True):
    out: str
    dataset: str
    corpus: str
    model_path: str
    b1_head: str
    device: str
    smoke: bool
    plan_sha: str | None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_default(obj):
    if isinstance(obj, msgspec.Struct):
        return msgspec.to_builtins(obj)
    if isinstance(obj, tuple | set):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json_atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stage_dir(ctx: StageCtx, name: str) -> str:
    return os.path.join(ctx.out, "stages", name)


def _done_path(ctx: StageCtx, name: str) -> str:
    return os.path.join(stage_dir(ctx, name), DONE_NAME)


def write_done(ctx: StageCtx, name: str, *, inputs: dict[str, str], outputs: dict[str, str], info: dict) -> None:
    record = {"stage": name, "inputs": dict(inputs), "outputs": dict(outputs), "info": info, "plan_sha": ctx.plan_sha}
    write_json_atomic(_done_path(ctx, name), record)


def done_state(ctx: StageCtx, name: str, *, inputs: dict[str, str]) -> Literal["missing", "match", "stale"]:
    path = _done_path(ctx, name)
    if not os.path.exists(path):
        return "missing"
    with open(path) as f:
        record = json.load(f)
    return "match" if record["inputs"] == dict(inputs) else "stale"


def guard_out_root(path: str, *, root: str = OUT_ROOT, forbidden: tuple[str, ...] = FORBIDDEN) -> str:
    real = os.path.realpath(path)
    real_root = os.path.realpath(root)
    if real == real_root or not real.startswith(real_root + os.sep):
        raise ValueError(f"--out {path!r} is outside {root!r}")
    parts = real[len(real_root) + 1 :].split(os.sep)
    for bad in forbidden:
        if bad in parts:
            raise ValueError(f"--out {path!r} is or lies inside the protected run directory {bad!r}")
    return real


def log_invocation(out: str, *, argv: list[str], exit_code: int, started: str, ended: str) -> None:
    os.makedirs(out, exist_ok=True)
    line = json.dumps({"argv": list(argv), "exit_code": exit_code, "started": started, "ended": ended})
    with open(os.path.join(out, INVOCATIONS_NAME), "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
