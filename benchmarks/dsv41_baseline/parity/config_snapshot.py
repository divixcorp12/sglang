"""Snapshot of the production recipe's resolved SGLANG_* configuration, for the F145 before/after env-rename check.

Usage (CPU only, no server, no GPU; see README.md):
  CUDA_VISIBLE_DEVICES= taskset -c 0-63 python config_snapshot.py --tree WORKTREE --out base-config-snapshot.json

For each recipe (base_env(), the DSpark mode, production) the recipe's env dict is laid over a clean environment, and
every `Envs` field in python/sglang/srt/environ.py is resolved the way the server resolves it (`field.get()`).
The snapshot holds the raw dicts and `{name: resolved value}` per recipe, so a later test can map old names to the
merged config fields and compare values. SGLANG_* names a recipe sets that `Envs` does not declare are listed
separately: the server reads those straight from os.environ (C++ and launcher code).

environ.py and arm_env.py are loaded by file path from `--tree` (default: the tree this script sits in), so the
snapshot describes that tree's Envs defaults and recipe; point it at the base worktree to get the base commit's.
"""

from __future__ import annotations

import argparse
import enum
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TREE = HERE.parents[2]
SNAPSHOT_PREFIX = "SGLANG_"


def load_environ_module(tree: Path):
    spec = importlib.util.spec_from_file_location(
        "sglang_srt_environ_snapshot", tree / "python" / "sglang" / "srt" / "environ.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_arm_env(tree: Path):
    sys.path.insert(0, str(tree / "benchmarks" / "dsv41_baseline"))
    import arm_env

    return arm_env


def jsonable(value):
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def declared_fields(environ_module) -> dict:
    return {
        name: field
        for name, field in vars(environ_module.Envs).items()
        if isinstance(field, environ_module.EnvField)
    }


def resolve_field(field):
    try:
        return jsonable(field.get())
    except Exception as error:
        return {"unresolved": f"{type(error).__name__}: {error}"}


def resolve_under_env(*, environ_module, recipe_env: dict[str, str]) -> dict:
    """Resolve every declared Envs field with os.environ replaced by the recipe over a SGLANG_-free copy."""
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update({k: v for k, v in saved.items() if not k.startswith(SNAPSHOT_PREFIX)})
        os.environ.update(recipe_env)
        fields = declared_fields(environ_module)
        resolved = {name: resolve_field(field) for name, field in sorted(fields.items())}
        is_set = sorted(name for name, field in fields.items() if field.is_set())
    finally:
        os.environ.clear()
        os.environ.update(saved)
    unregistered = {k: v for k, v in sorted(recipe_env.items()) if k.startswith(SNAPSHOT_PREFIX) and k not in fields}
    return {"resolved": resolved, "set_by_recipe": is_set, "unregistered_recipe_names": unregistered}


def recipes(arm_env) -> dict[str, dict[str, str]]:
    return {
        "base": arm_env.base_env(),
        "dspark": arm_env.arm_env(arm_env.dspark_env()),
        "prod": arm_env.prod_env(),
    }


def git_commit(tree: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def build_snapshot(tree: Path) -> dict:
    environ_module = load_environ_module(tree)
    arm_env = load_arm_env(tree)
    raw = recipes(arm_env)
    return {
        "git_commit": git_commit(tree),
        "tree": str(tree),
        "environ_path": str(Path(environ_module.__file__).resolve()),
        "arm_env_path": str(Path(arm_env.__file__).resolve()),
        "raw_env": raw,
        "dspark_overrides": arm_env.dspark_env(),
        "resolved": {
            name: resolve_under_env(environ_module=environ_module, recipe_env=env) for name, env in raw.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE, help="worktree whose environ.py/arm_env.py to read")
    args = parser.parse_args(argv)
    snapshot = build_snapshot(args.tree.resolve())
    args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    counts = {name: len(entry["resolved"]) for name, entry in snapshot["resolved"].items()}
    print(f"wrote {args.out}: commit {snapshot['git_commit']}, fields resolved per recipe {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
