"""B1 dataset builder: per-split x shards and per-horizon group shards from B0 block directories.

Inputs are the model path (gates via b0_integrity._gates), the block dirs, the corpus manifest.json (split, flagged,
warm_start) and the integrity report (non-tie mismatch cells to drop). Each session's x rows are written once
(``<out>/<split>/x/<session>.bin``) and its groups once per horizon (``<out>/<split>/h<h>/<session>.npz``). The
manifest ``dataset.json`` holds rows, groups, eligible rows and dropped cells per split and horizon. Warm-up verifies
(no rids) are skipped by the extractor and counted.

Usage: b1_dataset.py MODEL_PATH BLOCK_DIR... --manifest M --integrity REPORT --out D [--allow-any-out]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b0_integrity as b0  # noqa: E402
import b1_extract  # noqa: E402
from b1_rows import SPLITS, TRAINED_HORIZONS, eligible, write_groups, write_x  # noqa: E402

from sglang.srt.layers.moe.residency_capture import load_residency  # noqa: E402

OUT_ROOT = "/mnt/nvme1"
DATASET_NAME = "dataset.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flagged_sessions(manifest: dict) -> set[str]:
    # b0_manifest schema 2: each flagged entry is a drive_flags record keyed by session_id.
    return {f["session_id"] for f in manifest["flagged"]}


def _session_meta(manifest: dict) -> dict[str, dict]:
    """session -> {split, warm_start, flagged} for every session the manifest splits."""
    warm, flagged = set(manifest["warm_start"]), flagged_sessions(manifest)
    return {sid: {"split": name, "warm_start": sid in warm, "flagged": sid in flagged}
            for name, sids in manifest["split"].items() for sid in sids}


def _empty_counts() -> dict:
    return {s: {h: {"rows": 0, "groups": 0, "eligible_rows": 0, "dropped_cells": 0} for h in TRAINED_HORIZONS}
            for s in SPLITS}


def _write_session(out_dir: str, *, split: str, session: str, groups: list, x, counts: dict) -> None:
    write_x(out_dir, split=split, session=session, x=x)
    for h in TRAINED_HORIZONS:
        part = [g for g in groups if g.horizon == h]
        write_groups(out_dir, split=split, horizon=h, session=session, groups=part)
        c = counts[split][h]
        c["groups"] += len(part)
        c["rows"] += sum(len(g.tokens) for g in part)
        c["eligible_rows"] += sum(len(eligible(g)) for g in part)


def build_dataset(*, model_path, block_dirs, manifest_path, integrity_path, out_dir, gates=None) -> dict:
    """Write the shards and ``out_dir/dataset.json``; ``gates`` is an injected (W, bias) in residency row order."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(integrity_path) as f:
        integrity = json.load(f)
    meta = _session_meta(manifest)
    if gates is None:
        rows = load_residency(os.path.join(os.fspath(block_dirs[0]), b0.RESIDENCY_PREFIX)).rows
        gates = b0._gates(model_path, rows)
    W, bias = gates
    num_experts = int(bias.shape[1])
    extracted = b1_extract.ExtractCounts()
    counts = _empty_counts()
    seen: set[str] = set()
    for block_dir in block_dirs:
        drop = b1_extract.mismatch_cells(integrity, block_dir=block_dir)
        for session, groups, x in b1_extract.extract_block(
                block_dir, W=W, bias=bias, session_meta=meta, drop=drop, num_experts=num_experts,
                counts=extracted):
            if session in seen:
                raise ValueError(f"session {session} appears in more than one block")
            seen.add(session)
            _write_session(out_dir, split=meta[session]["split"], session=session, groups=groups, x=x, counts=counts)
    for split, by_horizon in extracted.dropped_by_split.items():
        for h, n in by_horizon.items():
            counts[split][h]["dropped_cells"] = n
    result = {
        "counts": counts,
        "warmup_forwards": extracted.warmup_forwards,
        "invalid_horizons": extracted.invalid_horizons,
        "dropped_cells": extracted.dropped_cells,
        "sessions_by_split": {s: sorted(manifest["split"].get(s, [])) for s in SPLITS},
        "built_sessions": sorted(seen),
        "flagged": sorted(flagged_sessions(manifest)),
        "manifest_sha256": _sha256(manifest_path),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, DATASET_NAME), "w") as f:
        json.dump(result, f, indent=1)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_path")
    p.add_argument("block_dirs", nargs="+")
    p.add_argument("--manifest", required=True)
    p.add_argument("--integrity", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-any-out", action="store_true", help="tests only: accept an --out off /mnt/nvme1")
    a = p.parse_args(argv)
    if not a.allow_any_out and not os.path.realpath(a.out).startswith(OUT_ROOT + os.sep):
        p.error(f"--out must be under {OUT_ROOT} (pass --allow-any-out in tests)")
    result = build_dataset(model_path=a.model_path, block_dirs=a.block_dirs, manifest_path=a.manifest,
                           integrity_path=a.integrity, out_dir=a.out)
    print(json.dumps({k: result[k] for k in ("counts", "warmup_forwards", "dropped_cells")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
