# Mutation harness for the host-facing RAM-miss tests

The scripts behind `../MUTATION_RESULTS.md`. Nothing here is run by CI. Everything runs on divix01 in
`/data/models/slang/nvfp4-work/t2-mutants`, outside every worktree, under `taskset -c 0-63`.

- `mutate.py` holds the mutant table (`MUTANTS`: id, file, old text, new text, replace-all, tests, note).
  A mutant is a text replacement that must match exactly once (twice only where marked replace-all).
- `mutate_gen.py <base> <tree> <results.jsonl> <label-prefix> ID...` applies each mutant to `<tree>` (a copy of
  the pristine `<base>` export), runs `run_gen.sh`, appends one result row (status, kills, collected counts,
  seconds, load average at start and end), and restores the file. H14 has a base-3 text.
- `run_gen.sh <tree> <label>` runs the 14 host-facing test files against a tree (`CUDA_VISIBLE_DEVICES=9`, a
  private `SGLANG_JIT_CACHE_DIR`); `FILES_OVERRIDE` selects another set.
- `chain.sh` is the one-launch chain: base 3 in full, the gate family on base 2, H19/H20 on base 1, then the
  invocation baselines. `launch.sh` exports base 3 at HEAD, builds the trees, records `cond.py`'s conditions
  and launches it. `summarize3.py` prints the assertion line each killing test died on, flags kills whose failing
  line reads a counter, and lists every run's collected total against its baseline.

Safeguards, each there because of a failure it prevents:
- **Lock file.** `chain.sh` refuses to start if `chain.lock` exists (a double launch contaminated an earlier run).
- **Preflight.** Before anything runs, every base directory and tree must exist and every mutant must apply
  exactly once on every base (`DRYRUN=1`); otherwise the chain refuses and writes `chain.refused`. This is what
  the wrong directory name in the first chain would have tripped.
- **Stop file.** `touch stop.flag` stops the chain after the mutant that is running finishes; it writes
  `chain.stopped` and removes the lock. Never edit a running script to add a pause.
- **Resume.** `RESUME=1 ./chain.sh` keeps the result files and skips the mutants already in them (the baselines
  are re-run). Remove `stop.flag` first.
- **Invocation check.** A killed mutant is only counted if the run collected the same number of tests as its
  baseline; `summarize3.py` prints the totals. A kill whose only evidence is an absence is not a kill.
