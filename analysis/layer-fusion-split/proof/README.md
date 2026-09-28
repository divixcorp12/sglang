# Proofs for the layer-fusion-split move commits

Each script certifies one `mechanical_provable` commit by reproducing it byte for byte. Run from the repo root (the
C++ wrappers also work from any other directory):

| Script | Proves | Expected last line |
|---|---|---|
| `python3 analysis/layer-fusion-split/proof/88dee2b0e4.py` | cxx-move: `manifest-cxx.json`'s line blocks relocate exactly (`cxx_move_proof.py`) and nothing else changes | `PASS: reproduces the commit byte-for-byte (cxx_move_proof).` |
| `python3 analysis/layer-fusion-split/proof/81becdda0e.py` | cast-cxx-move: the same, for `manifest-cast.json` | `PASS: reproduces the commit byte-for-byte (cxx_move_proof).` |
| `python3 analysis/layer-fusion-split/proof/b4aa5a5a2f.py` | python-move: a `Repro` of the `dsv41_layer_fusion.py` split | `PASS: reproduces the commit byte-for-byte.` |
| `python3 analysis/layer-fusion-split/proof/a81f0d3ca6.py` | cast-python-move: a `Repro` of the cast-fusion module move | `PASS: reproduces the commit byte-for-byte.` |

Prerequisites: `pre-commit` 4.x on `PATH`, with the repo's pinned hooks installed (`.pre-commit-config.yaml`,
clang-format `v20.1.7`). The `Repro` engine runs `pre-commit run --files ...` with `check=False`, so when pre-commit
is absent it skips formatting silently, and the unformatted result then diffs against the commit as a false FAIL.

`ast_norm_eq.py <old> <new> <path>...` is the AST evidence quoted in `b54253cd01` and `501e536a6b`; it compares
`git show` blobs, so it gives the same answer at any HEAD.

Chain verification over the whole branch:

```bash
python3 .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_cli.py \
  --base a04b7e27c7 --branch layer-fusion-split --proof analysis/layer-fusion-split/proof --report <report.md>
```

The range starts at `a04b7e27c7`, not the merge-base, because the branch's first commit (the plan doc) carries no
classification word and would read as `UNCLASSIFIED`.
