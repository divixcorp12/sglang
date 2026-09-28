"""Proof: the cast-cxx-move commit is a pure relocation of line blocks (cxx_move_proof.py) and touches nothing else."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET = "81becdda0e128d3d1b0368b6432d4891283dd5db"
MANIFEST = Path(__file__).resolve().parent / "manifest-cast.json"
CHECKER = Path("analysis/expert-stream-split/cxx_move_proof.py").resolve()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    parent = subprocess.run(
        ["git", "rev-parse", f"{TARGET}^"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if parent != manifest["base"]:
        print(
            f"FAIL: the manifest's base {manifest['base']} is not the commit's parent {parent}"
        )
        return 1
    allowed = set(manifest["sources"]) | set(manifest["files"])
    touched = subprocess.run(
        ["git", "diff", "--name-only", manifest["base"], TARGET],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    extra = sorted(set(touched) - allowed)
    if extra:
        print(f"FAIL: the commit touches files outside the manifest: {extra}")
        return 1
    worktree = tempfile.mkdtemp(prefix="cxx-move-proof-")
    subprocess.run(
        ["git", "worktree", "add", "--detach", worktree, TARGET],
        check=True,
        capture_output=True,
    )
    try:
        result = subprocess.run(
            [sys.executable, str(CHECKER), str(MANIFEST)],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            return 1
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", worktree], check=False)
    print("PASS: reproduces the commit byte-for-byte (cxx_move_proof).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
