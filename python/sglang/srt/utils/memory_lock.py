"""Lock the scheduler process's memory against swap (SGLANG_ENABLE_SCHEDULER_MLOCK).

A DSV4.1 EXL3 server holds its pinned expert tier on node 0 next to the loader's page cache; under that pressure the
kernel swapped out the CPU expert teams' working memory and the teams then took major faults in decode
(ram-prefetch/cores-ab-20261014/splitmodel-stream, 2026-10-15). mlockall(MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT) keeps
every page the process has touched, and every page it touches later, out of reclaim, without faulting anything in:
MCL_ONFAULT locks pages only as they become resident, so CUDA's large virtual reservations and lazily used allocator
arenas stay unpopulated, and memory the CUDA driver already pins is unaffected.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from typing import Callable, Optional

MCL_CURRENT, MCL_FUTURE, MCL_ONFAULT = 1, 2, 4  # <sys/mman.h> on Linux
MLOCK_FLAGS = MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT

_locked = False


def _libc_mlockall() -> Callable[[int], int]:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    return libc.mlockall


def locked_bytes(status_path: str = "/proc/self/status") -> Optional[int]:
    """The process's VmLck (bytes in locked mappings), or None where /proc does not report it."""
    try:
        with open(status_path) as f:
            for line in f:
                if line.startswith("VmLck:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def lock_process_memory(logger: logging.Logger, mlockall: Optional[Callable[[int], int]] = None) -> bool:
    """mlockall(MLOCK_FLAGS) once per process; logs the locked bytes, or warns on failure (never raises)."""
    global _locked
    if _locked:
        return True
    try:
        call = mlockall or _libc_mlockall()
        before = locked_bytes()
        if call(MLOCK_FLAGS) != 0:
            error = ctypes.get_errno()
            logger.warning("scheduler mlockall(MCL_CURRENT|MCL_FUTURE|MCL_ONFAULT) failed: %s (errno %d)",
                           os.strerror(error), error)
            return False
    except (OSError, AttributeError) as error:
        logger.warning("scheduler mlockall is unavailable: %r", error)
        return False
    _locked = True
    logger.info("scheduler mlockall(MCL_CURRENT|MCL_FUTURE|MCL_ONFAULT): VmLck %s -> %s bytes", before, locked_bytes())
    return True


def maybe_lock_process_memory(logger: logging.Logger) -> bool:
    """lock_process_memory when SGLANG_ENABLE_SCHEDULER_MLOCK is set."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_SCHEDULER_MLOCK.get():
        return False
    return lock_process_memory(logger)
