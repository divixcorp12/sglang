# planner instructions for this repository

<!-- hx appends this to its planner instructions when planning in this repository. Where it disagrees, this wins.
     Write facts about this codebase: where the project lives (in a monorepo), how its tests and typechecks run,
     which areas are risky, conventions the docs don't state. Commit it to share it with everyone who plans here. -->

## Tests run on divix01, never on the laptop

This overrides the built-in advice to write `uv run pytest ...` checks. Here that command cannot work: the laptop is
macOS, `--extra test` pulls CUDA-only wheels (`nvshmem4py-cu13` has no macOS build), importing `sglang` needs the
full serving stack, and `test/registered/unit/kernels/conftest.py` loads expert-stream JIT host modules that build
only on divix01's Linux toolchain. On F100 (2026-10-11) such a check failed every batch, and the fixer then committed
a 13k-line `python/uv.lock` rewrite and a `python/.python-version`.

- **Every pytest check** uses the divix01 runner, from the checkout root:
  `/Users/dnikolaidis/.hx/checks/divix01-pytest <test files> -q -p no:randomly`
  It pushes the checkout's branch to origin, runs pytest in a private divix01 worktree
  (`/data/models/slang/nvfp4-work/wt-<branch>`) with `PYTHONPATH`, `OMP_NUM_THREADS=8` and `taskset -c 0-63`, and
  exits with pytest's own status (`.claude/rules/divix01-run-protocol.md`).
- **GPU tests** (`test/manual/dsv41/*_cuda.py`, `*_gpu.py`, anything needing CUDA) add `--gpu`:
  `/Users/dnikolaidis/.hx/checks/divix01-pytest --gpu <files> -q -s -p no:randomly`. It waits on `cc-gpu.lock` and
  runs on cores 32-63. Run them for real in the batch that owns the kernel; a `py_compile` is not a GPU test.
- **Laptop-only checks** are limited to `python3 -m py_compile <files>` and the stdlib lint scripts
  (`python3 scripts/lint/check_registered_tests.py`, `python3 scripts/lint/check_no_bare_pytest_main.py`).
- Never write `uv run`, `uv sync` or `pip install` into a check.
- Name test files narrowly (run protocol, "Run only the tests your change affects"); never a whole directory.
  `test/registered` and `test/registered/unit` fail collection on divix01 (pyarrow); target files or
  `test/registered/unit/kernels`.
- A check that changes host C++ cold-builds every JIT module first (50-100 s each); the runner has no timeout, so
  this is slow, not a failure.

## Runs that need the server

A pilot or capture that launches the server is not a batch check: it needs a quiet box, both locks
(`rowimg-disk.lock` then `cc-gpu.lock`) and the owner's go-ahead, and production on divix01 may be stopped on purpose.
Put such runs in their own final task whose check the owner starts, and have documents record their numbers as
`pending`.
