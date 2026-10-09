"""A trace run's per-verify accept log (``SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH``).

One JSON line per request per verify step, in the scheduler's order: the request's ``rid``, ``k`` (its verifies
before this one), ``num_correct_drafts`` (drafts only, no bonus token), ``settled`` (False for a request retracted or
finished before this step committed) and ``ns`` (CLOCK_MONOTONIC). The k-th line of a request joins the k-th
``target_verify`` forward with that rid in the stage trace's route log (analysis/dsv41-drive/dspark/verify_split.py).
Diagnostics only: unset, nothing is opened or written.
"""

from __future__ import annotations

import collections
import json
import os
import time
from typing import Optional

from sglang.srt.environ import envs


class VerifyAcceptLog:
    def __init__(self, prefix: str) -> None:
        self._file = open(f"{prefix}.{os.getpid()}.jsonl", "a", buffering=1)
        self._k: collections.Counter = collections.Counter()

    def record(self, rid: str, num_correct_drafts: int, *, settled: bool) -> None:
        k = self._k[rid]
        self._k[rid] = k + 1
        self._file.write(
            json.dumps(
                {
                    "rid": rid,
                    "k": k,
                    "num_correct_drafts": int(num_correct_drafts),
                    "settled": bool(settled),
                    "ns": time.monotonic_ns(),
                }
            )
            + "\n"
        )

    def close(self) -> None:
        self._file.close()


_log: Optional[VerifyAcceptLog] = None
_resolved = False


def get_verify_accept_log() -> Optional[VerifyAcceptLog]:
    """The process's log, opened on first use when the path is set; None otherwise."""
    global _log, _resolved
    if not _resolved:
        prefix = envs.SGLANG_DSV41_VERIFY_ACCEPT_LOG_PATH.get()
        _log = VerifyAcceptLog(prefix) if prefix else None
        _resolved = True
    return _log


def reset_for_test() -> None:
    global _log, _resolved
    if _log is not None:
        _log.close()
    _log, _resolved = None, False
