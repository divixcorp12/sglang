"""b0_manifest.py: the B0 corpus manifest. CPU only.

Corpus layout: one directory per captured session under ``corpus_dir``, holding the residency side files
(``<prefix>.json``, ``.snap.bin``, ``.label.bin``, ``.seq.bin``, see ``residency_capture``) beside the router
capture and trace. ``manifest.json`` sits in ``corpus_dir``.
"""

import hashlib
import json
import os

MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"
SNAP_SUFFIX = ".snap.bin"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_dirs(corpus_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(corpus_dir)
        if os.path.isdir(os.path.join(corpus_dir, name))
        and any(f.endswith(SNAP_SUFFIX) for f in os.listdir(os.path.join(corpus_dir, name)))
    )


def _residency_header(session_dir: str) -> dict:
    prefixes = [f[: -len(SNAP_SUFFIX)] for f in sorted(os.listdir(session_dir)) if f.endswith(SNAP_SUFFIX)]
    if len(prefixes) != 1:
        raise ValueError(f"{session_dir} holds {len(prefixes)} residency captures, expected 1")
    with open(os.path.join(session_dir, prefixes[0] + ".json")) as f:
        header = json.load(f)
    if not header.get("rows") or "validity_rule" not in header:
        raise ValueError(f"{session_dir} residency header lacks rows or validity_rule")
    return header


def _shared_header(corpus_dir: str, sessions: list[str]) -> tuple[list[dict], str]:
    """The per-row metadata and validity rule, which must be identical in every session."""
    first = _residency_header(os.path.join(corpus_dir, sessions[0]))
    rows, rule = first["rows"], first["validity_rule"]
    for name in sessions[1:]:
        header = _residency_header(os.path.join(corpus_dir, name))
        if header["rows"] != rows or header["validity_rule"] != rule:
            raise ValueError(f"session {name} residency header differs from {sessions[0]}")
    return rows, rule


def _file_table(corpus_dir: str, sessions: list[str]) -> dict[str, dict]:
    files = {}
    for name in sessions:
        base = os.path.join(corpus_dir, name)
        for root, _, names in os.walk(base):
            for fname in sorted(names):
                path = os.path.join(root, fname)
                files[os.path.relpath(path, corpus_dir)] = {
                    "size": os.path.getsize(path),
                    "sha256": _sha256(path),
                }
    return dict(sorted(files.items()))


def write_manifest(corpus_dir: str, *, commit: str, recipe: dict, split: dict, failed: list, flagged: list) -> dict:
    """Write and return ``corpus_dir/manifest.json``; raises ValueError on an empty corpus or session headers
    that disagree on the per-row metadata or validity rule."""
    sessions = _session_dirs(corpus_dir)
    if not sessions:
        raise ValueError(f"{corpus_dir} holds no captured session")
    rows, rule = _shared_header(corpus_dir, sessions)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "commit": commit,
        "recipe": recipe,
        "sessions": sessions,
        "split": split,
        "failed": list(failed),
        "flagged": list(flagged),
        "rows": rows,
        "validity_rule": rule,
        "files": _file_table(corpus_dir, sessions),
    }
    with open(os.path.join(corpus_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
