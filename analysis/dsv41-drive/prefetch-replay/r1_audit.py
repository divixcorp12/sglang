"""R1 data audit: the D14 exclusion manifest, the input manifest and a seeded gate top-12 parity sample.

The exclusion manifest lists every non-tie capture mismatch cell (block, seq, layer_id, token) of the B0 integrity
report, with each train/dev/calib group whose source or target layer is the cell's layer in that forward.
``token_present`` marks the groups that still hold the token (B1 already dropped it where the layer is the source);
r1_groups.load_exclusions reads the result. The input manifest records the sha256 of every input file, the commit,
split membership, layer ids and the dev/calib recall denominators. locked_test shards are never listed or read (R4).

Inputs: a B1 dataset dir (b1_rows layout plus dataset.json), the B0 corpus dir (manifest.json, <block>/rids.json),
the B0 integrity report, the frozen B1 head and the model path (target gates).
Usage: ex = r1_audit.build_exclusions(integrity=json.load(f), dataset_dir=D, corpus_dir=C); run by r1_run.py's audit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b0_integrity  # noqa: E402
import b1_rows  # noqa: E402
import r1_gate  # noqa: E402
import r1_groups  # noqa: E402
import r1_stage  # noqa: E402

DATASET_JSON = "dataset.json"
CORPUS_MANIFEST = "manifest.json"
DENOMINATOR_SPLITS = ("dev", "calib")
_REASON = "non-tie gate recompute mismatch in the B0 capture (integrity report)"


def _mismatch_cells(integrity: dict) -> list[dict]:
    """Every non-tie mismatch of the report as {block, seq, layer_id, token, computed, captured}, sorted."""
    cells = {}
    for report in integrity["per_session"]:
        block = os.path.basename(os.path.normpath(report["out_dir"]))
        for m in report.get("mismatches", []):
            key = (block, int(m["seq"]), int(m["layer_id"]), int(m["token"]))
            cells[key] = {"block": block, "seq": key[1], "layer_id": key[2], "token": key[3],
                          "computed": list(m["computed"]), "captured": list(m["captured"])}
    return [cells[k] for k in sorted(cells)]


def _exclusions(integrity: dict) -> r1_groups.Exclusions:
    return r1_groups.Exclusions(cells=frozenset((c["block"], c["seq"], c["layer_id"], c["token"])
                                                for c in _mismatch_cells(integrity)))


def _load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _block_sessions(corpus_dir, *, block: str) -> set[str]:
    rids = b0_integrity._load_rids(os.path.join(os.fspath(corpus_dir), block))
    if rids is None:
        raise ValueError(f"{corpus_dir}/{block}: no rids.json, so the block's sessions are unknown")
    return {v["session_id"] for v in rids.values()}


def _r1_split_sessions(dataset_dir) -> dict[str, list[str]]:
    """Built sessions of each R1 split, from dataset.json; locked_test is left out."""
    ds = _load_json(os.path.join(os.fspath(dataset_dir), DATASET_JSON))
    built = set(ds["built_sessions"])
    return {s: sorted(set(ds["sessions_by_split"][s]) & built) for s in r1_groups.R1_SPLITS}


def _group_ref(g: b1_rows.Group, *, token: int) -> dict:
    return {"session": g.session, "split": g.split, "block": g.block, "seq": g.seq,
            "source_layer": g.source_layer, "target_layer": g.target_layer, "horizon": g.horizon,
            "token_present": bool(np.isin(token, g.tokens))}


def _affected(cell: dict, *, dataset_dir, sessions: dict[str, list[str]], horizons) -> tuple[list, list]:
    """(source groups, target groups) of the cell's forward whose source or target is the cell's layer."""
    source, target = [], []
    for split in r1_groups.R1_SPLITS:
        if not sessions[split]:
            continue
        for h in horizons:
            for g in b1_rows.read_groups(dataset_dir, split=split, horizon=h, sessions=sessions[split]):
                if g.block != cell["block"] or g.seq != cell["seq"]:
                    continue
                if g.source_layer == cell["layer_id"]:
                    source.append(_group_ref(g, token=cell["token"]))
                if g.target_layer == cell["layer_id"]:
                    target.append(_group_ref(g, token=cell["token"]))
    return source, target


def build_exclusions(*, integrity: dict, dataset_dir, corpus_dir, horizons=b1_rows.TRAINED_HORIZONS) -> dict:
    """The exclusion manifest in r1_groups.load_exclusions' schema, with each cell's affected groups."""
    split_sessions = _r1_split_sessions(dataset_dir)
    out = []
    for cell in _mismatch_cells(integrity):
        in_block = _block_sessions(corpus_dir, block=cell["block"])
        sessions = {s: [x for x in names if x in in_block] for s, names in split_sessions.items()}
        source, target = _affected(cell, dataset_dir=dataset_dir, sessions=sessions, horizons=tuple(horizons))
        out.append({**cell, "reason": _REASON, "source_groups": source, "target_groups": target})
    return {"cells": out, "horizons": [int(h) for h in horizons], "splits_scanned": list(r1_groups.R1_SPLITS)}


def _shard_hashes(dataset_dir) -> dict[str, str]:
    """sha256 of every train/dev/calib shard file by path relative to the dataset dir."""
    root = os.fspath(dataset_dir)
    out = {}
    for split in r1_groups.R1_SPLITS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, split)):
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                out[os.path.relpath(path, root)] = r1_stage.sha256_file(path)
    return out


def _commit(repo) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", os.fspath(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _layers(integrity: dict) -> dict:
    """Residency row order of the corpus; every block of the report must agree."""
    seen = {json.dumps([[r["layer_id"], r["stratum"], bool(r["gate_has_bias"])] for r in rep["rows"]])
            for rep in integrity["per_session"]}
    if len(seen) != 1:
        raise ValueError(f"integrity report blocks disagree on the residency rows: {len(seen)} variants")
    rows = json.loads(seen.pop())
    return {"layer_ids": [r[0] for r in rows], "strata": [r[1] for r in rows],
            "gate_has_bias": [r[2] for r in rows]}


def _denominators(dataset_dir, *, exclusions: r1_groups.Exclusions, horizons) -> dict:
    """Per dev/calib split and horizon: groups, tokens and both recall denominators after exclusions."""
    out = {}
    for split in DENOMINATOR_SPLITS:
        for h in horizons:
            c = Counter()
            for r in r1_groups.read_r1_groups(dataset_dir, split=split, horizon=h, exclusions=exclusions):
                c.update(groups=1, tokens=len(r.g.tokens), eligible_all=len(r.labels.eligible_all),
                         eligible_addr=len(r.labels.eligible_addr))
            out.setdefault(split, {})[str(h)] = {k: int(c[k]) for k in
                                                 ("groups", "tokens", "eligible_all", "eligible_addr")}
    return out


def input_manifest(*, dataset_dir, corpus_dir, integrity_path, b1_head_path, repo,
                   horizons=b1_rows.TRAINED_HORIZONS) -> dict:
    """Hashes, commit, split membership, layer ids and dev/calib denominators of the R1 inputs."""
    integrity = _load_json(integrity_path)
    dataset_json = os.path.join(os.fspath(dataset_dir), DATASET_JSON)
    corpus_manifest = os.path.join(os.fspath(corpus_dir), CORPUS_MANIFEST)
    shards = _shard_hashes(dataset_dir)
    exclusions = _exclusions(integrity)
    sessions = _r1_split_sessions(dataset_dir)
    locked = _load_json(dataset_json)["sessions_by_split"].get(b1_rows.LOCKED_SPLIT, [])
    return {
        "sha256": {"dataset_json": r1_stage.sha256_file(dataset_json),
                   "corpus_manifest": r1_stage.sha256_file(corpus_manifest),
                   "integrity": r1_stage.sha256_file(integrity_path),
                   "b1_head": r1_stage.sha256_file(b1_head_path),
                   "shards": r1_stage.sha256_json(shards)},
        "shards": shards,
        "paths": {"dataset": os.fspath(dataset_dir), "corpus": os.fspath(corpus_dir),
                  "integrity": os.fspath(integrity_path), "b1_head": os.fspath(b1_head_path)},
        **_commit(repo),
        "splits": {s: {"sessions": names, "count": len(names)} for s, names in sessions.items()},
        "locked_test": {"count": len(locked), "read": False},
        **_layers(integrity),
        "exclusion_cells": sorted(list(c) for c in exclusions.cells),
        "denominators": _denominators(dataset_dir, exclusions=exclusions, horizons=tuple(horizons)),
    }


def _sample(groups, *, n: int, seed: int) -> list[r1_groups.R1Group]:
    """Seeded reservoir sample (Algorithm R) of n groups, returned in stream order."""
    rng = np.random.default_rng(seed)
    kept: list[tuple[int, r1_groups.R1Group]] = []
    for i, r in enumerate(groups):
        if i < n:
            kept.append((i, r))
            continue
        j = int(rng.integers(0, i + 1))
        if j < n:
            kept[j] = (i, r)
    return [r for _, r in sorted(kept, key=lambda t: t[0])]


def gate_parity_sample(*, dataset_dir, model_path, split: str, horizon: int, n_groups: int, seed: int,
                       gates: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None) -> dict:
    """r1_gate.top12_parity totals over a seeded sample of the split's groups; ``gates`` injects the target gates."""
    stream = r1_groups.read_r1_groups(dataset_dir, split=split, horizon=horizon,
                                      exclusions=r1_groups.Exclusions.empty())
    picked = _sample(stream, n=n_groups, seed=seed)
    biased = sorted({r.key.target_layer for r in picked if r.g.gate_has_bias})
    if gates is None:
        gates = r1_gate.load_target_gates(model_path, layer_ids=biased)
    totals = Counter()
    xmaps: dict[str, np.memmap] = {}
    for r in picked:
        if not r.g.gate_has_bias:
            totals["no_bias_groups"] += 1
            continue
        if r.key.session not in xmaps:
            xmaps[r.key.session] = b1_rows.read_x(dataset_dir, split=split, session=r.key.session)
        W, bias = gates[r.key.target_layer]
        x = torch.from_numpy(r1_groups.x_rows(xmaps[r.key.session], r))
        p = r1_gate.top12_parity(r.g, scores=r1_gate.full_gate_scores(x, W=W, bias=bias))
        totals.update(tokens=p.tokens, exact=p.exact, tie_only=p.tie_only, mismatched=p.mismatched)
    return {"split": split, "horizon": int(horizon), "seed": int(seed), "groups": len(picked),
            **{k: int(totals[k]) for k in ("tokens", "exact", "tie_only", "mismatched", "no_bias_groups")}}
