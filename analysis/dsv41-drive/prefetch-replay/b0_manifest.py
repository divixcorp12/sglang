"""b0_manifest.py: the B0 corpus manifest (schema 2). CPU only.

Corpus layout: capture files live per server start under ``corpus_dir/blocks/<block_id>/`` (trace, router,
residency side files ``<prefix>.json`` / ``.snap.bin`` / ``.label.bin`` / ``.seq.bin``, see ``residency_capture``),
and per-session outputs under ``corpus_dir/<sid>/``. ``manifest.json`` sits in ``corpus_dir`` and records the
session -> block mapping, the warm_start sessions, and sizes and sha256 of every listed block and session file.
"""

import hashlib
import json
import os

MANIFEST_SCHEMA = 2
MANIFEST_NAME = "manifest.json"
BLOCKS_DIR = "blocks"
SNAP_SUFFIX = ".snap.bin"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _residency_header(block_dir: str) -> dict:
    prefixes = [f[: -len(SNAP_SUFFIX)] for f in sorted(os.listdir(block_dir)) if f.endswith(SNAP_SUFFIX)]
    if len(prefixes) != 1:
        raise ValueError(f"{block_dir} holds {len(prefixes)} residency captures, expected 1")
    with open(os.path.join(block_dir, prefixes[0] + ".json")) as f:
        header = json.load(f)
    if not header.get("rows") or "validity_rule" not in header:
        raise ValueError(f"{block_dir} residency header lacks rows or validity_rule")
    return header


def _shared_header(corpus_dir: str, block_ids: list[str]) -> tuple[list[dict], str]:
    """The per-row metadata and validity rule, which must be identical in every block."""
    first = _residency_header(os.path.join(corpus_dir, BLOCKS_DIR, block_ids[0]))
    rows, rule = first["rows"], first["validity_rule"]
    for block_id in block_ids[1:]:
        header = _residency_header(os.path.join(corpus_dir, BLOCKS_DIR, block_id))
        if header["rows"] != rows or header["validity_rule"] != rule:
            raise ValueError(f"block {block_id} residency header differs from {block_ids[0]}")
    return rows, rule


def _file_table(corpus_dir: str, rel_dirs: list[str]) -> dict[str, dict]:
    files = {}
    for rel in rel_dirs:
        for root, _, names in os.walk(os.path.join(corpus_dir, rel)):
            for fname in sorted(names):
                path = os.path.join(root, fname)
                files[os.path.relpath(path, corpus_dir)] = {
                    "size": os.path.getsize(path),
                    "sha256": _sha256(path),
                }
    return dict(sorted(files.items()))


def _blocks_of(session_block: dict[str, str]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for sid in sorted(session_block):
        blocks.setdefault(session_block[sid], []).append(sid)
    return dict(sorted(blocks.items()))


def write_manifest(
    corpus_dir: str,
    *,
    commit: str,
    recipe: dict,
    split: dict,
    failed: list,
    flagged: list,
    session_block: dict[str, str],
    warm_start: list[str],
) -> dict:
    """Write and return ``corpus_dir/manifest.json``; raises ValueError on an empty mapping or block headers
    that disagree on the per-row metadata or validity rule."""
    if not session_block:
        raise ValueError(f"{corpus_dir}: empty session -> block mapping")
    blocks = _blocks_of(session_block)
    rows, rule = _shared_header(corpus_dir, list(blocks))
    sessions = sorted(session_block)
    file_dirs = [os.path.join(BLOCKS_DIR, b) for b in blocks] + sessions
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "commit": commit,
        "recipe": recipe,
        "sessions": sessions,
        "blocks": blocks,
        "session_block": {sid: session_block[sid] for sid in sessions},
        "warm_start": sorted(warm_start),
        "split": split,
        "failed": list(failed),
        "flagged": list(flagged),
        "rows": rows,
        "validity_rule": rule,
        "files": _file_table(corpus_dir, file_dirs),
    }
    with open(os.path.join(corpus_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
