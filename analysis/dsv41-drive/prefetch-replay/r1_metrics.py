"""R1 policy metrics: the C2 tables for one scorer's capped reads on one split (spec section 1.3).

A group reads the candidates scoring >= the threshold, best first, truncated to the cap (r1_policy.select). Per key
('pooled', 'pooled_noflag', 'flagged', 'warm', 'layer:<id>', 'stratum:<s>', 'session:<id>') it counts useful, wasted
and eligible rows (all true misses and the addressable subset), outside-top-12 recovery, whole-layer coverage, the
MISS_GPU/MISS_CPU split and the useful rows the cap drops; the pre-cap read histogram with its percentiles and bursts;
per-position coverage, anchor-only/draft-only/shared row-set attribution and block-level reliability bins. The D15
paired whole-session bootstrap compares two reports' pooled recall.

Inputs: R1Groups of one split and horizon (r1_groups.read_r1_groups) and one scorer's r1_policy.Candidates per group.
Output: a PolicyReport, its JSON form and bootstrap summaries.

Usage: rep = r1_metrics.policy_report(groups, cands, scorer="gate", cap=2, threshold=t); r1_metrics.report_to_json(rep)
"""

from __future__ import annotations

import math
import os
import sys
from typing import Literal

import msgspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_rows  # noqa: E402
import r1_policy  # noqa: E402
from r1_groups import NUM_EXPERTS, R1Group  # noqa: E402
from r1_policy import Candidates  # noqa: E402

from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import KIND_VRAM_HOT  # noqa: E402

HIST_BINS = ("0", "1", "2", "3", "4+")
PERCENTILES = (50, 90, 95, 99)
RELIABILITY_BINS = 10
BOOTSTRAP_RESAMPLES = 10_000  # D15
_MISS_KINDS = np.array([LaneKind.MISS_GPU, LaneKind.MISS_CPU], dtype=np.uint8)


class PolicyCell(msgspec.Struct):
    forwards: int = 0  # (forward, target layer) groups
    eligible_all: int = 0  # every true miss row
    eligible_addr: int = 0  # misses not VRAM-hot and not in the RAM snapshot
    read: int = 0
    useful: int = 0
    wasted: int = 0
    wasted_hot: int = 0  # read rows a token routed VRAM-hot at demand
    never_candidates: int = 0  # eligible rows outside every kept token's gate top-12
    never_recovered: int = 0
    layers_with_miss: int = 0
    layer_covered: int = 0  # groups with misses whose every eligible row was read
    miss_gpu_eligible: int = 0
    miss_cpu_eligible: int = 0
    miss_gpu_useful: int = 0
    miss_cpu_useful: int = 0
    cap_dropped_useful: int = 0  # eligible rows over the threshold that the cap left unread

    @property
    def recall(self) -> float:
        return self.useful / self.eligible_all if self.eligible_all else 0.0

    @property
    def recall_addr(self) -> float:
        return self.useful / self.eligible_addr if self.eligible_addr else 0.0

    @property
    def precision(self) -> float:
        return self.useful / self.read if self.read else 0.0

    @property
    def never_recovery(self) -> float:
        return self.never_recovered / self.never_candidates if self.never_candidates else 0.0

    @property
    def layer_coverage(self) -> float:
        return self.layer_covered / self.layers_with_miss if self.layers_with_miss else 0.0

    @property
    def mean_reads(self) -> float:
        return self.read / self.forwards if self.forwards else 0.0


class Histogram(msgspec.Struct):
    bins: dict[str, int]  # pre-cap reads per (forward, target layer): '0', '1', '2', '3', '4+'
    p50: float
    p90: float
    p95: float
    p99: float
    max: int
    # Run length -> runs of consecutive target layers of one forward whose pre-cap reads exceed the cap;
    # filled for forward-level keys ('pooled', 'session:<id>') only.
    bursts: dict[str, int] = {}
    max_burst: int = 0


class PositionCell(msgspec.Struct):
    miss_rows: int = 0  # unique (token, expert) miss routes of tokens at this position
    miss_rows_addr: int = 0
    covered: int = 0

    @property
    def coverage(self) -> float:
        return self.covered / self.miss_rows if self.miss_rows else 0.0


class RowSetCell(msgspec.Struct):
    eligible_all: int = 0
    eligible_addr: int = 0
    useful: int = 0

    @property
    def recall(self) -> float:
        return self.useful / self.eligible_all if self.eligible_all else 0.0


class ReliabilityBin(msgspec.Struct):
    lo: float
    hi: float
    candidates: int = 0  # available rows scored in [lo, hi)
    candidates_miss: int = 0
    read: int = 0  # of those, rows the capped policy read
    read_miss: int = 0


class PolicyReport(msgspec.Struct):
    scorer: str
    split: str
    cap: int
    threshold: float
    cells: dict[str, PolicyCell]
    hist: dict[str, Histogram]
    positions: dict[str, PositionCell]  # token position in the forward, '0' = anchor
    rowsets: dict[str, RowSetCell]  # 'anchor_only', 'draft_only', 'shared'
    reliability: list[ReliabilityBin]  # empty unless every candidate score lies in [0, 1]
    forward_totals: dict[str, float]  # capped reads per forward over its target layers


def _keys(r: R1Group) -> list[str]:
    g = r.g
    keys = ["pooled", "flagged" if g.flagged else "pooled_noflag"]
    if g.warm_start:
        keys.append("warm")
    return keys + [f"layer:{g.target_layer}", f"stratum:{g.target_stratum}", f"session:{g.session}"]


def _mask(ids: np.ndarray) -> np.ndarray:
    m = np.zeros(NUM_EXPERTS, dtype=bool)
    m[np.asarray(ids, dtype=np.int64)] = True
    return m


def _token_miss(r: R1Group) -> np.ndarray:
    """bool [n, 384]: token m routed e as MISS_GPU or MISS_CPU."""
    g = r.g
    ids = np.asarray(g.target_ids, dtype=np.int64)
    hit = np.isin(g.target_kinds, _MISS_KINDS) & (ids >= 0)
    out = np.zeros((len(ids), NUM_EXPERTS), dtype=bool)
    out[np.nonzero(hit)[0], ids[hit]] = True
    return out


def _group_cell(r: R1Group, *, read: np.ndarray, dropped: np.ndarray) -> PolicyCell:
    lab, g = r.labels, r.g
    ids = np.asarray(g.target_ids, dtype=np.int64)
    hot_routed = _mask(ids[(g.target_kinds == KIND_VRAM_HOT) & (ids >= 0)])
    never = _mask(b1_rows.never_candidates(g))
    useful = int(lab.miss[read].sum())
    n_elig = len(lab.eligible_all)
    return PolicyCell(
        forwards=1, eligible_all=n_elig, eligible_addr=len(lab.eligible_addr), read=len(read), useful=useful,
        wasted=len(read) - useful, wasted_hot=int(hot_routed[read].sum()), never_candidates=int(never.sum()),
        never_recovered=int(never[read].sum()), layers_with_miss=int(n_elig > 0),
        layer_covered=int(n_elig > 0 and useful == n_elig), miss_gpu_eligible=int(lab.miss_gpu.sum()),
        miss_cpu_eligible=int(lab.miss_cpu.sum()), miss_gpu_useful=int(lab.miss_gpu[read].sum()),
        miss_cpu_useful=int(lab.miss_cpu[read].sum()), cap_dropped_useful=int(lab.miss[dropped].sum()))


def _accumulate(into: PolicyCell, one: PolicyCell) -> None:
    for name in PolicyCell.__struct_fields__:
        setattr(into, name, getattr(into, name) + getattr(one, name))


def _add_positions(out: dict[str, PositionCell], *, r: R1Group, tmiss: np.ndarray, read_mask: np.ndarray) -> None:
    for m, p in enumerate(r.positions.tolist()):
        c = out.setdefault(str(p), PositionCell())
        c.miss_rows += int(tmiss[m].sum())
        c.miss_rows_addr += int((tmiss[m] & r.labels.available).sum())
        c.covered += int((tmiss[m] & read_mask).sum())


def _add_rowsets(out: dict[str, RowSetCell], *, r: R1Group, tmiss: np.ndarray, read_mask: np.ndarray) -> None:
    anchor = tmiss[r.positions == 0].any(axis=0)
    draft = tmiss[r.positions != 0].any(axis=0)
    for name, rows in (("anchor_only", anchor & ~draft), ("draft_only", draft & ~anchor), ("shared", anchor & draft)):
        c = out[name]
        c.eligible_all += int(rows.sum())
        c.eligible_addr += int((rows & r.labels.available).sum())
        c.useful += int((rows & read_mask).sum())


def _add_reliability(counts: np.ndarray, *, r: R1Group, c: Candidates, read_mask: np.ndarray) -> None:
    # counts int64 [bins, 4]: candidates, candidates_miss, read, read_miss; truncation as b1_score._calibrate does.
    bins = np.clip((c.vals.astype(np.float64) * RELIABILITY_BINS).astype(np.int64), 0, RELIABILITY_BINS - 1)
    miss = r.labels.miss[c.ids]
    read = read_mask[c.ids]
    for col, sel in enumerate((np.ones_like(miss), miss, read, read & miss)):
        counts[:, col] += np.bincount(bins[sel], minlength=RELIABILITY_BINS)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="inverted_cdf")) if len(values) else 0.0


def _bursts(layers_counts: list[tuple[int, int]], *, cap: int) -> list[int]:
    """Lengths of runs of consecutive (sorted) target layers whose pre-cap reads exceed the cap."""
    runs, run = [], 0
    for _, count in sorted(layers_counts):
        if count > cap:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    return runs + [run] if run else runs


def _histogram(counts: list[int], *, bursts: list[int] | None) -> Histogram:
    a = np.asarray(counts, dtype=np.int64)
    bins = {b: int((a == i).sum()) for i, b in enumerate(HIST_BINS[:-1])}
    bins[HIST_BINS[-1]] = int((a >= len(HIST_BINS) - 1).sum())
    pct = {f"p{q}": _percentile(a, q) for q in PERCENTILES}
    runs = bursts or []
    by_len: dict[str, int] = {}
    for n in sorted(runs):
        by_len[str(n)] = by_len.get(str(n), 0) + 1
    return Histogram(bins=bins, **pct, max=int(a.max()) if len(a) else 0, bursts=by_len,
                     max_burst=max(runs, default=0))


def _check_inputs(groups: list[R1Group], cands: list[Candidates]) -> None:
    if len(groups) != len(cands):
        raise ValueError(f"{len(cands)} candidate lists for {len(groups)} groups")
    if not groups:
        raise ValueError("no groups to report on")
    for r, c in zip(groups, cands):
        if not r.labels.available[c.ids].all():
            raise AssertionError(f"candidates for {r.key} include a skip-mask row")


def _forward_totals(per_forward: dict[tuple, int]) -> dict[str, float]:
    a = np.asarray(list(per_forward.values()), dtype=np.int64)
    out = {f"p{q}": _percentile(a, q) for q in PERCENTILES}
    return {**out, "mean": float(a.mean()) if len(a) else 0.0, "max": float(a.max()) if len(a) else 0.0,
            "forwards": float(len(a))}


def policy_report(groups: list[R1Group], cands: list[Candidates], *, scorer: str, cap: int,
                  threshold: float) -> PolicyReport:
    """Every metric of one scorer's capped policy (reads >= threshold, at most cap per group) on one split."""
    _check_inputs(groups, cands)
    cells: dict[str, PolicyCell] = {}
    precap: dict[str, list[int]] = {}
    layers: dict[tuple, list[tuple[int, int]]] = {}
    per_forward: dict[tuple, int] = {}
    positions: dict[str, PositionCell] = {}
    rowsets = {name: RowSetCell() for name in ("anchor_only", "draft_only", "shared")}
    rel = np.zeros((RELIABILITY_BINS, 4), dtype=np.int64)
    probabilistic = all(len(c.vals) == 0 or (c.vals.min() >= 0.0 and c.vals.max() <= 1.0) for c in cands)
    for r, c in zip(groups, cands):
        passing = c.ids[c.vals >= threshold]
        read, dropped = r1_policy.select(c, threshold=threshold, cap=cap), passing[cap:]
        one = _group_cell(r, read=read, dropped=dropped)
        for key in _keys(r):
            _accumulate(cells.setdefault(key, PolicyCell()), one)
            precap.setdefault(key, []).append(len(passing))
        fwd = (r.key.session, r.key.block, r.key.seq)
        layers.setdefault(fwd, []).append((r.key.target_layer, len(passing)))
        per_forward[fwd] = per_forward.get(fwd, 0) + len(read)
        read_mask, tmiss = _mask(read), _token_miss(r)
        _add_positions(positions, r=r, tmiss=tmiss, read_mask=read_mask)
        _add_rowsets(rowsets, r=r, tmiss=tmiss, read_mask=read_mask)
        if probabilistic:
            _add_reliability(rel, r=r, c=c, read_mask=read_mask)
    hist = _histograms(precap, layers=layers, cap=cap)
    width = 1.0 / RELIABILITY_BINS
    reliability = [ReliabilityBin(lo=i * width, hi=(i + 1) * width, candidates=int(rel[i, 0]),
                                  candidates_miss=int(rel[i, 1]), read=int(rel[i, 2]), read_miss=int(rel[i, 3]))
                   for i in range(RELIABILITY_BINS)] if probabilistic else []
    return PolicyReport(scorer=scorer, split=groups[0].g.split, cap=cap, threshold=float(threshold), cells=cells,
                        hist=hist, positions=dict(sorted(positions.items())), rowsets=rowsets,
                        reliability=reliability, forward_totals=_forward_totals(per_forward))


def _histograms(precap: dict[str, list[int]], *, layers: dict[tuple, list[tuple[int, int]]],
                cap: int) -> dict[str, Histogram]:
    bursts: dict[str, list[int]] = {"pooled": []}
    for (session, _, _), lc in layers.items():
        runs = _bursts(lc, cap=cap)
        bursts["pooled"] += runs
        bursts.setdefault(f"session:{session}", []).extend(runs)
    return {key: _histogram(counts, bursts=bursts.get(key)) for key, counts in precap.items()}


def _finite(x: float) -> float | None:
    # JSON has no infinity; r1_policy returns math.inf for a budget no threshold meets (reads nothing).
    return float(x) if math.isfinite(x) else None


def _cell_json(c: PolicyCell) -> dict:
    return {**msgspec.structs.asdict(c), "recall": c.recall, "recall_addr": c.recall_addr, "precision": c.precision,
            "never_recovery": c.never_recovery, "layer_coverage": c.layer_coverage, "mean_reads": c.mean_reads}


def report_to_json(r: PolicyReport) -> dict:
    return {"scorer": r.scorer, "split": r.split, "cap": r.cap, "threshold": _finite(r.threshold),
            "cells": {k: _cell_json(c) for k, c in r.cells.items()},
            "hist": {k: msgspec.structs.asdict(h) for k, h in r.hist.items()},
            "positions": {k: {**msgspec.structs.asdict(c), "coverage": c.coverage} for k, c in r.positions.items()},
            "rowsets": {k: {**msgspec.structs.asdict(c), "recall": c.recall} for k, c in r.rowsets.items()},
            "reliability": [msgspec.structs.asdict(b) for b in r.reliability],
            "forward_totals": dict(r.forward_totals)}


def _session_arrays(rep: PolicyReport, *, names: list[str], denominator: str) -> tuple[np.ndarray, ...]:
    cells = [rep.cells[k] for k in names]
    elig = [c.eligible_all if denominator == "all" else c.eligible_addr for c in cells]
    return (np.array([c.useful for c in cells], np.float64), np.array(elig, np.float64),
            np.array([c.read for c in cells], np.float64))


def session_bootstrap(cand: PolicyReport, ctrl: PolicyReport, *, n: int = BOOTSTRAP_RESAMPLES, seed: int = 0,
                      denominator: Literal["all", "addressable"] = "all") -> dict:
    """D15: pooled recall difference (cand - ctrl), sessions resampled with replacement, paired, percentile 95% CI."""
    if denominator not in ("all", "addressable"):
        raise ValueError(f"unknown denominator {denominator!r}")
    if (cand.split, cand.cap) != (ctrl.split, ctrl.cap):
        raise ValueError(f"unpaired reports: {cand.split}/cap {cand.cap} vs {ctrl.split}/cap {ctrl.cap}")
    names = sorted(k for k in cand.cells if k.startswith("session:"))
    if not names or names != sorted(k for k in ctrl.cells if k.startswith("session:")):
        raise ValueError("candidate and control must cover the same, non-empty set of sessions")
    u_c, e, r_c = _session_arrays(cand, names=names, denominator=denominator)
    u_k, e_k, r_k = _session_arrays(ctrl, names=names, denominator=denominator)
    if not np.array_equal(e, e_k):
        raise ValueError("candidate and control eligible rows differ per session; groups are not identical")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(names), size=(n, len(names)))
    w = np.zeros((n, len(names)), dtype=np.float64)
    np.add.at(w, (np.arange(n)[:, None], idx), 1.0)
    den = w @ e
    diffs = np.where(den > 0, (w @ (u_c - u_k)) / np.maximum(den, 1.0), 0.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    total = e.sum()
    rec_c = u_c.sum() / total if total else 0.0
    rec_k = u_k.sum() / total if total else 0.0
    prec_c = u_c.sum() / r_c.sum() if r_c.sum() else 0.0
    prec_k = u_k.sum() / r_k.sum() if r_k.sum() else 0.0
    out = {"diff": float(rec_c - rec_k), "lo": float(lo), "hi": float(hi),
           "ratio": float(rec_c / rec_k) if rec_k else None, "precision_diff": float(prec_c - prec_k),
           "cand_recall": float(rec_c), "ctrl_recall": float(rec_k), "sessions": len(names), "n": n, "seed": seed,
           "denominator": denominator}
    if useful_ci:
        u_lo, u_hi = np.percentile(w @ (u_c - u_k), [2.5, 97.5])
        out.update({"useful_diff": float(u_c.sum() - u_k.sum()), "useful_lo": float(u_lo), "useful_hi": float(u_hi),
                    "cand_useful": float(u_c.sum()), "ctrl_useful": float(u_k.sum()),
                    "cand_reads": float(r_c.sum()), "ctrl_reads": float(r_k.sum())})
    return out
