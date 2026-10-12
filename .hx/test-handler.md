# test-handler instructions for this repository

## When a check fails before any test runs

Tests in this repository run on divix01 through `/Users/dnikolaidis/.hx/checks/divix01-pytest` (see the planner
layer). A failure in environment setup is not a code bug:

- **`uv` failing to resolve or install** (for example `nvshmem4py-cu13 ... doesn't have a source distribution or
  wheel for the current platform`) means the check was written as `uv run` on the laptop. Do not "fix" it in the
  repo: no `python/.python-version`, no `python/uv.lock` change, no `pyproject.toml` change (all ask-before). Say in
  your summary that the check must use the divix01 runner, and stop.
- The runner tests uncommitted work: it pushes a snapshot commit to a scratch ref `hx-check/<branch>/<sha>` and
  deletes it after the run, so rerun your lane before committing. If the snapshot's diff stat lists
  `python/uv.lock`, something rewrote it: restore it (`git checkout HEAD -- python/uv.lock`); never commit it.
- **`SGLANG_EXL3_SRC must point at an exllamav3 checkout`** or a missing CUDA toolkit is the runner's environment,
  not code. The runner sets the EXL3 env the GPU suites use; report it if it reappears.
- **A test that also fails at the merge-base** is not yours: check the base before changing code for it, and say so
  in your summary. On 2026-10-12, `test_dsv41_config.py::test_one_field_per_knob` failed on master (65 DSV41 env vars,
  26 config fields).
- **`push failed` / ssh errors** are infrastructure: report them, do not change code.
- **`TimeoutExpired` in a child process** right after a C++ change is a cold JIT build, not a hang
  (`.claude/rules/divix01-run-protocol.md`, "Cold JIT builds").

Only a failure inside pytest's own output (an assertion, an import error in the repo's code) is yours to fix.
