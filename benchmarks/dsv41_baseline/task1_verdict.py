"""Load the real `task1_arm_verdict.py`, sha256-pinned, rather than reimplementing it.

That file lives outside any repo on divix01 (`PIPELINE_BASELINE.md` section 2: "two
hand-copied runtime files outside any repo"), alongside `task1-baseline-arms.sh`,
Task 1's matched-baseline harness for the offline `Engine` path
(`docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`). Its provenance,
page-cache-residency, cross-arm-outlier, contention and code-generation checks are
reused here verbatim: a second implementation of the same judgment is worse than
none, and this campaign has no reason to define "VALID" differently than Task 1 does.

`task1-baseline-arms.sh` pins itself the same way (`RUNTIME_SHA256`, `harness_gate`):
a hand-copied file outside version control is only trustworthy if every reader checks
it against a hash before trusting its content, not by convention.
"""

from __future__ import annotations

import hashlib
import importlib.util
import types

# The path and hash task1-baseline-arms.sh itself checks against
# ($VERDICT in that script; verified 2026-09-21, wt-dsv41 HEAD e0a0b584d3).
TASK1_VERDICT_PATH = (
    "/data/models/slang/nvfp4-work/cc-expert-prediction"
    "/analysis/dsv41-drive/task1_arm_verdict.py"
)
TASK1_VERDICT_SHA256 = "0bcd9a26282c5c24b3e35bbe3374e49467165dcbc04bf346a32de9b2d72fb770"


class Task1VerdictHashMismatchError(RuntimeError):
    pass


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task1_verdict(
    *, path: str = TASK1_VERDICT_PATH, expected_sha256: str = TASK1_VERDICT_SHA256
) -> types.ModuleType:
    """Import `task1_arm_verdict.py` from `path`, refusing a hash mismatch.

    A silent drift here is exactly the failure `task1-baseline-arms.sh`'s own
    `harness_gate` exists to prevent, applied to this campaign's read of the same file.
    """
    got = _sha256_file(path)
    if got != expected_sha256:
        raise Task1VerdictHashMismatchError(
            f"{path} sha256 is {got}, expected {expected_sha256}: it has changed since "
            "this campaign last verified it. Re-read it before trusting its checks, "
            "then update TASK1_VERDICT_SHA256."
        )
    spec = importlib.util.spec_from_file_location("task1_arm_verdict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
