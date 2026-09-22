"""This campaign's code-generation registry: `{python tree sha: label}`.

Modeled on `task1-results/clean-reference.json`'s `generations` map and
`task1-baseline-arms.sh`'s `generation_gate` (`PIPELINE_BASELINE.md` section 9, rule
7): a `python/` tree that is not registered here cannot run as an arm, so a new code
generation can never silently pass as an old one. Task 1's own history is the
argument for this being a hard gate, not an advisory print: before it was enforced, an
unregistered tree read "GENERATION unknown" and the arm still read VALID.

The lookup itself is `task1_arm_verdict.generation()`, imported via `task1_verdict.py`
and given this file's contents as its `manifest` argument — not reimplemented, only
the file this campaign's registry lives in is new.
"""

from __future__ import annotations

import json
import os

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generations.json")


class UnregisteredGenerationError(RuntimeError):
    pass


def load(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return {"generations": {}}
    with open(path) as f:
        return json.load(f)


def register(tree: str, label: str, *, path: str = DEFAULT_PATH) -> dict:
    manifest = load(path)
    manifest.setdefault("generations", {})[tree] = label
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def check_registered(tree: str, *, path: str = DEFAULT_PATH) -> str:
    """The tree's registered label, or raise if it has none."""
    manifest = load(path)
    label = manifest.get("generations", {}).get(tree)
    if label is None:
        raise UnregisteredGenerationError(
            f"python tree {tree} is registered in no generation of {path}. "
            f"Register it first: python -c \"import generations; "
            f"generations.register({tree!r}, '<label>')\""
        )
    return label
