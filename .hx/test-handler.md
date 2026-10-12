# test-handler instructions for this repository

## When a check fails before any test runs

Tests in this repository run on divix01 through `/Users/dnikolaidis/.hx/checks/divix01-pytest` (see the planner
layer). A failure in environment setup is not a code bug:

- **`uv` failing to resolve or install** (for example `nvshmem4py-cu13 ... doesn't have a source distribution or
  wheel for the current platform`) means the check was written as `uv run` on the laptop. Do not "fix" it in the
  repo: no `python/.python-version`, no `python/uv.lock` change, no `pyproject.toml` change (all ask-before). Say in
  your summary that the check must use the divix01 runner, and stop.
- **`tracked files differ from HEAD; commit first`** comes from the runner: the checkout has uncommitted tracked
  changes, often a `python/uv.lock` that something rewrote. Restore it (`git checkout HEAD -- python/uv.lock`); never
  commit it.
- **`push failed` / ssh errors** are infrastructure: report them, do not change code.
- **`TimeoutExpired` in a child process** right after a C++ change is a cold JIT build, not a hang
  (`.claude/rules/divix01-run-protocol.md`, "Cold JIT builds").

Only a failure inside pytest's own output (an assertion, an import error in the repo's code) is yours to fix.
