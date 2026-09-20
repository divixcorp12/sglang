"""Provenance for one measurement arm: what the running process actually saw.

DSV41_REFERENCE section 19 recorded an end-to-end result that could not be reproduced and
could not be reconstructed either, because the run wrote no environment, no reader mode, no
drive-idle check and no record of which tree it imported. Every arm's result json now embeds
``capture()``, taken from inside the process that ran, not from the shell that launched it.

A field that cannot be read is ``None`` and named in ``"unavailable"`` with the reason; a key
is never omitted, so a missing answer cannot be mistaken for a missing question.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re
import socket
import subprocess
import sys

SCHEMA = 1

# Prefixes of the process environment that steer the framework. SGL_ is the legacy alias family.
ENV_PREFIXES = ("SGLANG_", "SGL_")
# Not SGLANG_*, but they change what a run measures (placement, thread counts, module lookup).
OTHER_ENV = ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|API_?KEY|CREDENTIAL", re.I)
REDACTED = "<redacted>"

# mount point -> block device, as in run-mirror-arms.sh
DRIVES = {"nvme0": "nvme0n1", "nvme2": "nvme2n1", "nvme4": "nvme3n1"}
# A drive another job is using reads well above this; the arms' own boot reads are not sampled here.
IDLE_MAX_BYTES_PER_S = 1 << 20
SECTOR_BYTES = 512


def _redact(name: str, value: str) -> str:
    return REDACTED if _SECRET_NAME.search(name) else value


def _wanted(name: str) -> bool:
    return name.startswith(ENV_PREFIXES) or name in OTHER_ENV


def process_env() -> dict:
    """The live ``os.environ``, filtered: what child processes spawned now will inherit."""
    return {k: _redact(k, v) for k, v in sorted(os.environ.items()) if _wanted(k)}


def exec_env(path: str = "/proc/self/environ") -> dict | None:
    """The environment the kernel handed this process at exec, before anything in Python edited it."""
    with open(path, "rb") as f:
        raw = f.read()
    out = {}
    for item in raw.split(b"\0"):
        name, sep, value = item.decode(errors="surrogateescape").partition("=")
        if sep and _wanted(name):
            out[name] = _redact(name, value)
    return dict(sorted(out.items()))


def resolved_env() -> dict:
    """Every SGLANG_* knob with its default applied, as ``envs.X.get()`` returns it.

    An unset variable still has a value in force (the reader mode is one); the raw environment
    alone would not show it."""
    from sglang.srt.environ import Envs, EnvField

    out = {}
    for field in sorted(
        (v for v in vars(Envs).values() if isinstance(v, EnvField)), key=lambda f: f.name
    ):
        if field.secret or _SECRET_NAME.search(field.name):
            out[field.name] = REDACTED
            continue
        try:
            value = field.get()
        except Exception as e:  # a lazy default that needs a device or platform we do not have
            out[field.name] = f"<unresolvable: {type(e).__name__}: {e}>"
            continue
        out[field.name] = value if isinstance(value, (bool, int, float, str, type(None))) else str(value)
    return out


def env_drift(before: dict, after: dict) -> dict:
    """{name: [before, after]} for every variable that changed between two ``process_env`` reads."""
    return {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}


def _git(cwd: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", cwd, *args], capture_output=True, text=True, timeout=30, check=True
    ).stdout


def git_state(package_dir: str) -> dict:
    """HEAD and dirtiness of the worktree ``package_dir`` sits in.

    ``dirty`` counts tracked modifications only. New untracked files under the package are
    listed separately because an imported but uncommitted module is code that HEAD does not name."""
    top = _git(package_dir, "rev-parse", "--show-toplevel").strip()
    changed = [line for line in _git(top, "status", "--porcelain", "--untracked-files=no").splitlines() if line]
    untracked = [
        line[3:]
        for line in _git(top, "status", "--porcelain", "--untracked-files=all", "--", package_dir).splitlines()
        if line.startswith("??")
    ]
    return {
        "toplevel": top,
        "head": _git(top, "rev-parse", "HEAD").strip(),
        "branch": _git(top, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(changed),
        "dirty_files": changed[:50],
        "dirty_file_count": len(changed),
        # Names the exact uncommitted content: two "dirty" runs are the same code only if this matches.
        "tracked_diff_sha1": hashlib.sha1(_git(top, "diff", "HEAD").encode(errors="surrogateescape")).hexdigest(),
        "untracked_in_package": untracked[:50],
        "untracked_in_package_count": len(untracked),
    }


def read_sectors(path: str = "/proc/diskstats") -> dict:
    """Cumulative sectors read per drive (/proc/diskstats field 6), keyed by mount name."""
    wanted = {device: name for name, device in DRIVES.items()}
    out = {}
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) > 5 and fields[2] in wanted:
                out[wanted[fields[2]]] = int(fields[5])
    return out


def drive_idle_check(seconds: float = 2.0, diskstats: str = "/proc/diskstats", sleep=None) -> dict:
    """Bytes/s read from each drive over ``seconds`` with the arm not yet started.

    A drive another job is reading makes an arm's timing and per-drive bytes meaningless, so the
    verdict is stored with the result rather than printed to a log nobody joins to it."""
    import time

    sleep = sleep or time.sleep
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        before = read_sectors(diskstats)
        sleep(seconds)
        after = read_sectors(diskstats)
        rates = {k: (after[k] - before[k]) * SECTOR_BYTES / seconds for k in DRIVES}
    except Exception as e:
        return {"idle": None, "unavailable": f"{type(e).__name__}: {e}", "measured_at_utc": started}
    return {
        "idle": all(r <= IDLE_MAX_BYTES_PER_S for r in rates.values()),
        "bytes_per_s": rates,
        "seconds": seconds,
        "max_idle_bytes_per_s": IDLE_MAX_BYTES_PER_S,
        "measured_at_utc": started,
    }


def capture(harness_files: dict | None = None) -> dict:
    """Snapshot the running process. ``harness_files`` maps a label to a module ``__file__``,
    so a stale copy of the driver or of trace_corpus is visible afterwards."""
    unavailable = {}
    out = {
        "schema": SCHEMA,
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "harness_files": dict(harness_files or {}),
        "sglang_env": process_env(),
    }
    try:
        out["sglang_env_at_exec"] = exec_env()
    except OSError as e:
        out["sglang_env_at_exec"] = None
        unavailable["sglang_env_at_exec"] = f"{type(e).__name__}: {e}"

    out["sglang_file"] = None
    out["sglang_env_resolved"] = None
    out["git"] = None
    try:
        import sglang

        package_dir = os.path.dirname(os.path.abspath(sglang.__file__)) if sglang.__file__ else None
        if package_dir is None:  # a namespace package has no __file__
            package_dir = os.path.abspath(list(sglang.__path__)[0])
        out["sglang_file"] = sglang.__file__ or package_dir
    except Exception as e:
        unavailable["sglang_file"] = f"{type(e).__name__}: {e}"
        unavailable["git"] = "sglang did not import, so no worktree to inspect"
        package_dir = None
    if package_dir is not None:
        try:
            out["git"] = git_state(package_dir)
        except Exception as e:
            unavailable["git"] = f"{type(e).__name__}: {e}"
        try:
            out["sglang_env_resolved"] = resolved_env()
        except Exception as e:
            unavailable["sglang_env_resolved"] = f"{type(e).__name__}: {e}"
    else:
        unavailable["sglang_env_resolved"] = "sglang did not import"
    out["unavailable"] = unavailable
    return out


def timed_chunks(stream, log: list, clock=None):
    """Pass a generate stream through, appending ``(time, completion_tokens)`` per chunk to ``log``.

    ``completion_tokens`` is the cumulative count from the chunk's meta_info, or None when the
    chunk carries none. It is what lets ``step_latency`` count tokens instead of assuming one
    chunk is one token."""
    import time

    clock = clock or time.perf_counter
    for chunk in stream:
        meta = chunk.get("meta_info") if isinstance(chunk, dict) else None
        log.append((clock(), meta.get("completion_tokens") if meta else None))
        yield chunk


def _percentile(sorted_values: list, q: float) -> float:
    # Nearest-rank: with 127 samples p99 is a real observation, not an interpolation.
    return sorted_values[min(len(sorted_values) - 1, max(0, -(-len(sorted_values) * q // 100) - 1))]


def step_latency(log: list) -> dict:
    """Decode step latency from a ``timed_chunks`` log, from the first chunk (end of prefill) onward.

    Each later chunk's wall time is divided by the tokens it delivered. It is exact only where a
    chunk delivered one token; ``multi_token_chunks`` counts the rest, and if it is not 0 the
    percentiles are smoothed over those chunks and must be read that way. If any chunk lacks a
    token count the percentiles are None with the reason."""
    if len(log) < 2:
        return {"steps": 0, "unavailable": "fewer than two chunks"}
    if any(tokens is None for _, tokens in log):
        return {"steps": len(log) - 1, "unavailable": "a chunk carried no completion_tokens"}
    per_token, multi = [], 0
    for (t0, n0), (t1, n1) in zip(log, log[1:]):
        if n1 <= n0:
            return {"steps": len(log) - 1, "unavailable": f"completion_tokens went {n0} -> {n1}"}
        multi += n1 - n0 > 1
        per_token.append((t1 - t0) / (n1 - n0))
    ordered = sorted(per_token)
    return {
        "steps": len(per_token),
        "multi_token_chunks": multi,
        "step_s_p50": _percentile(ordered, 50),
        "step_s_p95": _percentile(ordered, 95),
        "step_s_p99": _percentile(ordered, 99),
        "step_s_max": ordered[-1],
        "step_s": per_token,
    }


def process_tree_cpu_s() -> float | None:
    """User+system CPU seconds so far of this process and every live descendant (the scheduler and
    its workers). Whole-process totals, not per thread: a spinning doorbell thread counts in full."""
    try:
        import psutil

        me = psutil.Process()
        total = 0.0
        for p in [me, *me.children(recursive=True)]:
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.NoSuchProcess:
                pass
        return total
    except Exception:
        return None
