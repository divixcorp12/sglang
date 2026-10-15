"""Loading several JIT modules with their cold builds run side by side.

One process loads its modules one after another, so a cold cache costs the sum of
every build. The build lock is per module variant (``loader._build_lock``), so
separate processes can build different modules at once; each publishes into the
cache, and the loads that follow here are hits.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import logging
import subprocess
import sys
import time
from typing import Sequence, Tuple

from sglang.kernels.jit.utils.compile.loader import JitCompileForbidden
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# (importable module, function in it, positional arguments): one call that loads
# JIT modules. The arguments cross a process boundary as their repr, so they are
# literals (str, int, bool, None, tuples of those).
Load = Tuple[str, str, Tuple]

DEFAULT_WORKERS = 8

_CHILD = (
    "import ast, importlib, sys\n"
    "getattr(importlib.import_module(sys.argv[1]), sys.argv[2])"
    "(*ast.literal_eval(sys.argv[3]))\n"
)


def _call(load: Load):
    module, function, args = load
    return getattr(importlib.import_module(module), function)(*args)


def _build_in_child(load: Load) -> None:
    module, function, args = load
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, module, function, repr(tuple(args))],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        # The load below repeats the build in this process and raises the real
        # error with the compiler's output, so a failure here is only logged.
        logger.warning(
            "Parallel JIT build of %s.%s%r failed (exit %d):\n%s",
            module,
            function,
            tuple(args),
            completed.returncode,
            completed.stderr[-4000:],
        )


def load_in_parallel(loads: Sequence[Load], *, workers: int = DEFAULT_WORKERS) -> None:
    """Make every call in *loads* in this process, building cache misses first
    in up to *workers* child processes at once.

    The cache is probed in this process with ``SGLANG_CRASH_ON_JIT_COMPILE``
    set, so a load that hits is simply done, and when all of them hit no child
    is started. Under that flag or ``SGLANG_JIT_FORCE_RECOMPILE`` there is
    nothing to gain from children, and the calls are made in order as they are.
    """
    loads = list(dict.fromkeys((m, f, tuple(a)) for m, f, a in loads))
    if (
        envs.SGLANG_CRASH_ON_JIT_COMPILE.get()
        or envs.SGLANG_JIT_FORCE_RECOMPILE.get()
    ):
        for load in loads:
            _call(load)
        return

    misses = []
    with envs.SGLANG_CRASH_ON_JIT_COMPILE.override(True):
        for load in loads:
            try:
                _call(load)
            except JitCompileForbidden:
                misses.append(load)
    if not misses:
        return

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(min(len(misses), workers)) as pool:
        list(pool.map(_build_in_child, misses))
    logger.info(
        "Built %d JIT module loads in %.1f s with up to %d processes",
        len(misses),
        time.monotonic() - start,
        min(len(misses), workers),
    )
    for load in misses:
        _call(load)
