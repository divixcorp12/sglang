"""Concatenate several invocations of ONE arm into a single arm directory for paired.py.

An interleaved comparison (A B B A A B B A, each invocation timing a different
`DSV41_SESSION_INDICES` pair) leaves four run directories per arm; `paired.py` takes one
directory per arm. This writes that directory: `results.jsonl`, `clocks.jsonl` and
`compile.jsonl` concatenated in the order given, and a `run-manifest.json` whose
`session_ids` is the concatenation and whose `sources` names every input.

It refuses to merge what is not one arm: members must share the commit and the server
env exactly, time disjoint sessions, and have mutually compatible tenancy (`tenancy.py`).
The merged `tenancy_start` is the first member's, which is what `paired.py` compares
against the other arm; every member's is kept under `member_tenancy_start`.

Usage: python concat_arms.py <out_dir> <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tenancy import parse_tenancy, tenancy_compatible

_JSONL = ("results.jsonl", "clocks.jsonl", "compile.jsonl")


def concat_arms(out_dir: str, run_dirs: list[str]) -> dict:
    if not run_dirs:
        raise ValueError("no run directories given")
    manifests = [json.loads((Path(d) / "run-manifest.json").read_text()) for d in run_dirs]
    first = manifests[0]
    for d, m in zip(run_dirs[1:], manifests[1:]):
        if m.get("commit") != first.get("commit"):
            raise ValueError(f"{d}: commit {m.get('commit')} differs from {first.get('commit')}")
        if m.get("env") != first.get("env"):
            diff = sorted(
                k for k in set(m.get("env", {})) | set(first.get("env", {}))
                if m.get("env", {}).get(k) != first.get("env", {}).get(k)
            )
            raise ValueError(f"{d}: server env differs from {run_dirs[0]} in {diff}; not the same arm")
        if not tenancy_compatible(parse_tenancy(first["tenancy_start"]), parse_tenancy(m["tenancy_start"])):
            raise ValueError(f"{d}: tenancy {m['tenancy_start']} incompatible with {first['tenancy_start']}")
    session_ids = [sid for m in manifests for sid in m["session_ids"]]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"members time overlapping sessions: {session_ids}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    for name in _JSONL:
        with open(out / name, "w") as dst:
            for d in run_dirs:
                text = (Path(d) / name).read_text()
                dst.write(text if not text or text.endswith("\n") else text + "\n")
    manifest = {
        **{k: first[k] for k in ("arm", "commit", "python_tree", "generation_label", "env") if k in first},
        "session_ids": session_ids,
        "tenancy_start": first["tenancy_start"],
        "tenancy_end": manifests[-1].get("tenancy_end"),
        "member_tenancy_start": [m["tenancy_start"] for m in manifests],
        "sources": [str(Path(d).resolve()) for d in run_dirs],
    }
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    manifest = concat_arms(args.out_dir, args.run_dirs)
    print(f"wrote {args.out_dir}: {len(manifest['session_ids'])} sessions from {len(args.run_dirs)} runs")


if __name__ == "__main__":
    main()
