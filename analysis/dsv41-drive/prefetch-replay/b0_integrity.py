"""B0 corpus integrity: a tie-aware gate recompute and the record consistency of residency capture sessions.

Each session directory holds the stage trace (``trace.jsonl``), the RouterCapture files (``router.*``) and the
residency files (``router.res.*``, record layout in residency_capture_ref). Per session:

1. **Routes.** On every row whose gate has a correction bias, each verify forward's live tokens get router_score's
   gate on the captured input: sqrt(softplus(x @ W.T)) in fp32 plus the bias, top 6 compared as a set. A
   difference that is only a rank-6/7 swap within TIE_RTOL is a tie, counted; any other is kept in the report and
   fails the session. Rows without a biased gate (hash-routed or gateless) are counted unchecked.
2. **Records.** Seq gaps, verify forwards without a router or residency record, live token counts that disagree,
   valid flags that disagree with horizon_targets, a mask on a no-target horizon and label kinds above 5 fail the
   session. No-target horizons and snapshots stamped after their demand are counted.
3. **Rates.** For each valid horizon, the target layer's live routes that were not VRAM hot: of those the snapshot
   showed mapped in RAM, the share served as an NVMe miss (evicted between snapshot and demand); unmapped ones
   that hit are counted as admitted. Tie and eviction rates are reported overall, per horizon and per stratum
   (routes by the demanding layer, no-target horizons by the predicting row), and nothing is filtered on stratum.

Layouts. A capture dir is either a block dir (``OUT/blocks/<block-id>/``, the files of many sessions served by one
server start) or an F100 one-session dir; both are positional arguments and hold the same files. A dir with a
``rids.json`` (rid -> session_id, turn, split) also gets ``by_session``: each joined forward goes to the session of
its route-log ``rids``, with that session's counts, rates and split. A route-log rid missing from rids.json is
listed under ``unmapped_rids``; it is not an error. ``by_stratum`` and ``by_horizon`` stay dir-wide.

Verify forwards whose route-log ``rids`` is empty (the server's no-request warm-up forwards) are skipped by the
checks above and counted as ``no_request_forwards``, which is not a failure.

Usage: b0_integrity.py MODEL_PATH CAPTURE_DIR... [--out report.json]; exits 1 when any dir fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
import router_score  # noqa: E402
import tier_sim  # noqa: E402

from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture import ResidencyRow, load_residency  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import HORIZONS, KIND_VRAM_HOT, horizon_targets  # noqa: E402

RESIDENCY_PREFIX = "router.res"
TOPK = router_score.TOPK
# One bf16 ulp of the score: the served gate rounds where this fp32 recompute does not.
TIE_RTOL = 2.0**-7
_MISS_KINDS = np.array([LaneKind.MISS_GPU, LaneKind.MISS_CPU], dtype=np.uint8)
_HIT_KINDS = np.array([LaneKind.HIT_COPY, LaneKind.HIT_SM, LaneKind.HIT_CPU], dtype=np.uint8)
_FAILING = ("non_tie_mismatches", "seq_gaps", "missing_records", "live_mismatches", "snapshot_inconsistent",
            "label_inconsistent")


class Joined(msgspec.Struct, frozen=True):
    """The verify forwards with all three records: their seq, router record, residency record and live tokens."""

    seqs: np.ndarray
    router: np.ndarray
    residency: np.ndarray
    live: np.ndarray


def _rate(num: int, den: int):
    return num / den if den else None


def _join(loaded: dict, capture, res) -> tuple[Joined, Counter]:
    record_of_seq = {int(seq): q for q, seq in enumerate(res.keys[:, 0])}
    width = int(capture.header["tokens"])
    counts: Counter = Counter()
    joined = []
    for f in loaded["forwards"]:
        if f["kind"] != "graph" or not f["verify"]:
            continue
        counts["verify_forwards"] += 1
        router, q = f["router"], record_of_seq.get(int(f["seq"]))
        has_router = router is not None and router < len(capture.x)
        counts["missing_router"] += not has_router
        counts["missing_label"] += q is None
        if not has_router or q is None:
            continue
        live = int(f["tokens"])
        if not (1 <= live <= width and live == int(capture.tokens[router]) == int(res.keys[q, 2])):
            counts["live_mismatches"] += 1
            continue
        joined.append((int(f["seq"]), int(router), q, live))
    cols = np.asarray(joined, dtype=np.int64).reshape(-1, 4)
    return Joined(seqs=cols[:, 0], router=cols[:, 1], residency=cols[:, 2], live=cols[:, 3]), counts


def _seq_gaps(loaded: dict) -> int:
    """Forwards the reader dropped; a gap inside the logged seqs is one of them, so take the larger count."""
    seqs = sorted(f["seq"] for f in loaded["forwards"] if f["kind"] == "graph")
    inner = sum(b - a - 1 for a, b in zip(seqs, seqs[1:]))
    return max(int(loaded["dropped"]), inner)


def _is_tie(order: np.ndarray, scores: np.ndarray, captured: np.ndarray) -> bool:
    """Only the computed rank-6 expert is swapped for the rank-7 one, and their biased scores are within TIE_RTOL."""
    computed, cap = {int(e) for e in order[:TOPK]}, {int(e) for e in captured}
    if computed - cap != {int(order[TOPK - 1])} or cap - computed != {int(order[TOPK])}:
        return False
    a, b = float(scores[TOPK - 1]), float(scores[TOPK])
    return a - b <= TIE_RTOL * max(abs(a), abs(b))


def _route_check(capture, joined: Joined, *, rows: list[ResidencyRow], W, bias) -> tuple[Counter, dict, list]:
    """The gate recompute on every live token of every biased row; per-stratum cells and ties, and the mismatches."""
    W, bias = torch.as_tensor(W, dtype=torch.float32), torch.as_tensor(bias, dtype=torch.float32)
    rec = np.repeat(joined.router, joined.live)
    seq = np.repeat(joined.seqs, joined.live)
    tok = np.concatenate([np.arange(n) for n in joined.live] or [np.zeros(0, dtype=np.int64)])
    counts: Counter = Counter()
    strata: dict = defaultdict(Counter)
    mismatches = []
    for row, meta in enumerate(rows):
        if not meta.gate_has_bias:
            counts["unchecked_route_cells"] += len(rec)
            continue
        x = torch.from_numpy(router_score.bf16_to_f32(np.asarray(capture.x[rec, row, tok])))
        biased = torch.nn.functional.softplus(x @ W[row].T).sqrt() + bias[row]
        top = torch.topk(biased, TOPK + 1, dim=-1)
        order, scores = top.indices.numpy(), top.values.numpy()
        captured = np.asarray(capture.ids[rec, row, tok])
        differ = (np.sort(order[:, :TOPK], axis=1) != np.sort(captured, axis=1)).any(axis=1)
        strata[meta.stratum]["route_cells"] += len(rec)
        for i in np.nonzero(differ)[0]:
            if _is_tie(order[i], scores[i], captured[i]):
                strata[meta.stratum]["tie_count"] += 1
                continue
            strata[meta.stratum]["non_tie_mismatches"] += 1
            mismatches.append({
                "seq": int(seq[i]), "router_record": int(rec[i]), "layer_id": meta.layer_id, "token": int(tok[i]),
                "computed": sorted(int(e) for e in order[i, :TOPK]), "captured": sorted(int(e) for e in captured[i]),
                "rank6_minus_rank7": float(scores[i, TOPK - 1] - scores[i, TOPK]),
            })
    return counts, strata, mismatches


def _mapped_bits(words: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Bit ``ids`` of each forward's 384-bit mask: ``words`` [F, MASK_WORDS] uint64, ``ids`` [F, tokens, topk] >= 0."""
    flat = ids.reshape(len(ids), -1)
    word = np.take_along_axis(words, flat // 64, axis=1)
    return ((word >> (flat % 64).astype(np.uint64)) & np.uint64(1)).astype(bool).reshape(ids.shape)


def _snapshot_check(res, capture, joined: Joined, *, rows: list[ResidencyRow]) -> tuple[Counter, dict, dict]:
    """Snapshot/label consistency, no-target horizons and the eviction counts, overall, per horizon, per stratum."""
    targets, valid_ref = horizon_targets([row.layer_id for row in rows])
    q, r = joined.residency, joined.router
    width = min(int(res.kinds.shape[2]), int(capture.ids.shape[2]))
    live = (np.arange(width)[None, :] < joined.live[:, None])[:, :, None]
    counts: Counter = Counter()
    strata: dict = defaultdict(Counter)
    horizons: dict = defaultdict(Counter)
    kinds_all = res.kinds[q][:, :, :width]
    counts["label_inconsistent"] += int(((kinds_all > LaneKind.MISS_CPU) & live[:, None]).sum())
    for t, source in enumerate(rows):
        for h in range(HORIZONS):
            target, flags = targets[t][h], res.valid[q, t, h]
            counts["snapshot_inconsistent"] += int((flags != valid_ref[t][h]).sum())
            if target < 0:
                lost = int((~flags).sum())
                counts["invalid_no_target"] += lost
                strata[source.stratum]["invalid_no_target"] += lost
                counts["snapshot_inconsistent"] += int((res.masks[q, t, h] != 0).any(axis=1).sum())
                continue
            counts["stamp_inversions"] += int((res.snap_stamp[q, t] > res.label_stamp[q, target]).sum())
            ids = np.asarray(capture.ids[r, target, :width]).astype(np.int64)
            kinds = res.kinds[q, target, :width]
            counted = live & flags[:, None, None] & (ids >= 0) & (kinds != KIND_VRAM_HOT)
            mapped = _mapped_bits(res.masks[q, t, h], np.clip(ids, 0, None))
            step = Counter(
                snapshot_routes=int((counted & mapped).sum()),
                evicted=int((counted & mapped & np.isin(kinds, _MISS_KINDS)).sum()),
                admitted=int((counted & ~mapped & np.isin(kinds, _HIT_KINDS)).sum()),
            )
            counts.update(step)
            horizons[h + 1].update(step)
            strata[rows[target].stratum].update(step)
    return counts, strata, horizons


def _with_rates(counts: Counter) -> dict:
    out = dict(counts)
    out["tie_rate"] = _rate(counts["tie_count"], counts["route_cells"])
    out["eviction_rate"] = _rate(counts["evicted"], counts["snapshot_routes"])
    return out


def _load_rids(out_dir: str) -> dict | None:
    """The dir's rid -> {session_id, turn, split} map, or None for a dir without rids.json."""
    path = os.path.join(out_dir, "rids.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _by_session_groups(joined: Joined, *, loaded: dict, rids: dict) -> dict:
    """The joined forwards split by session; a forward with no mapped rid goes to the None group."""
    session_of_seq = {}
    for f in loaded["forwards"]:
        if f["kind"] == "graph":
            mapped = [rids[r]["session_id"] for r in f["rids"] if r in rids]
            session_of_seq[int(f["seq"])] = mapped[0] if mapped else None
    owners = [session_of_seq.get(int(s)) for s in joined.seqs]
    groups = {}
    for owner in dict.fromkeys(owners):
        keep = np.fromiter((o == owner for o in owners), dtype=bool, count=len(owners))
        groups[owner] = Joined(seqs=joined.seqs[keep], router=joined.router[keep],
                               residency=joined.residency[keep], live=joined.live[keep])
    return groups


def _unmapped_rids(loaded: dict, *, rids: dict) -> list[str]:
    """Route-log rids that rids.json does not name."""
    seen = {r for f in loaded["forwards"] for r in f["rids"]}
    return sorted(seen - set(rids))


def _by_session(sessions: dict, *, rids: dict) -> dict:
    """Per session counts and rates with the session's B0 split."""
    split_of = {v["session_id"]: v["split"] for v in rids.values()}
    return {sid: {**_with_rates(c), "split": split_of.get(sid)} for sid, c in sorted(sessions.items())}


def _check_part(capture, res, joined: Joined, *, W, bias) -> tuple[Counter, dict, dict, list]:
    """Route and snapshot checks of some joined forwards: totals, per-stratum, per-horizon, mismatches."""
    route_counts, route_strata, mismatches = _route_check(capture, joined, rows=res.rows, W=W, bias=bias)
    snap_counts, snap_strata, horizons = _snapshot_check(res, capture, joined, rows=res.rows)
    # The route totals live only in their strata; the snapshot totals are counted apart from theirs.
    counts: Counter = Counter()
    counts.update(route_counts)
    counts.update(snap_counts)
    strata: dict = defaultdict(Counter)
    for name, part_counts in route_strata.items():
        counts.update(part_counts)
        strata[name].update(part_counts)
    for name, part_counts in snap_strata.items():
        strata[name].update(part_counts)
    return counts, strata, horizons, mismatches


def check_session(out_dir: str, *, W, bias, residency_prefix: str = RESIDENCY_PREFIX) -> dict:
    """One capture dir's integrity report (a block dir or a one-session dir); ``W`` [rows, experts, hidden] and
    ``bias`` [rows, experts] fp32, in the residency header's row order. With a rids.json in the dir the report
    adds by_session and unmapped_rids. Raises ValueError on a residency header without per-row entries."""
    res = load_residency(os.path.join(out_dir, residency_prefix))
    capture = router_score.load_capture(os.path.join(out_dir, "router"))
    loaded = tier_sim.load_forwards(os.path.join(out_dir, "trace.jsonl"), allow_dropped=True)
    layer_ids = [row.layer_id for row in res.rows]
    if layer_ids != capture.header["layer_ids"] or layer_ids != loaded["layer_ids"]:
        raise ValueError(f"{out_dir}: residency, router capture and route log name different layers")
    if len(W) != len(res.rows) or len(bias) != len(res.rows):
        raise ValueError(f"{out_dir}: {len(W)} gates for {len(res.rows)} rows")
    joined, counts = _join(loaded, capture, res)
    rids = _load_rids(out_dir)
    groups = {None: joined} if rids is None else _by_session_groups(joined, loaded=loaded, rids=rids)
    strata: dict = defaultdict(Counter)
    horizons: dict = defaultdict(Counter)
    mismatches: list = []
    sessions: dict = {}
    for session_id, part in groups.items():
        part_counts, part_strata, part_horizons, part_mismatches = _check_part(
            capture, res, part, W=W, bias=bias)
        counts.update(part_counts)
        mismatches.extend(part_mismatches)
        for name, c in part_strata.items():
            strata[name].update(c)
        for h, c in part_horizons.items():
            horizons[h].update(c)
        if session_id is not None:
            sessions[session_id] = Counter(part_counts, joined_forwards=len(part.seqs))
    counts["seq_gaps"] = _seq_gaps(loaded)
    counts["missing_records"] = counts["missing_router"] + counts["missing_label"]
    counts["joined_forwards"] = len(joined.seqs)
    report = _with_rates(counts)
    for key in _FAILING + ("tie_count", "no_request_forwards"):
        report.setdefault(key, 0)
    report.update(
        out_dir=out_dir,
        validity_rule=res.validity_rule,
        rows=[msgspec.structs.asdict(row) for row in res.rows],
        by_stratum={name: _with_rates(c) for name, c in sorted(strata.items())},
        by_horizon={h: _with_rates(c) for h, c in sorted(horizons.items())},
        mismatches=mismatches,
    )
    if rids is not None:
        report["by_session"] = _by_session(sessions, rids=rids)
        report["unmapped_rids"] = _unmapped_rids(loaded, rids=rids)
    report["ok"] = all(report[key] == 0 for key in _FAILING)
    return report


def _gates(model_path: str, rows: list[ResidencyRow]) -> tuple[torch.Tensor, torch.Tensor]:
    """Gates in row order; rows without a biased gate stay zero and are never recomputed."""
    biased = [i for i, row in enumerate(rows) if row.gate_has_bias]
    if not biased:
        return torch.zeros((len(rows), router_score.NUM_EXPERTS, 1)), torch.zeros((len(rows), router_score.NUM_EXPERTS))
    w, b = router_score.load_gates(model_path, [rows[i].layer_id for i in biased])
    W = torch.zeros((len(rows), *w.shape[1:]))
    bias = torch.zeros((len(rows), *b.shape[1:]))
    W[biased], bias[biased] = w, b
    return W, bias


def _summarize(sessions: list[dict]) -> dict:
    total: Counter = Counter()
    strata: dict = defaultdict(Counter)
    by_session: dict = defaultdict(Counter)
    splits: dict = {}
    for report in sessions:
        total.update({k: v for k, v in report.items() if isinstance(v, int) and not isinstance(v, bool)})
        for name, counts in report.get("by_stratum", {}).items():
            strata[name].update({k: v for k, v in counts.items() if isinstance(v, int)})
        for sid, counts in report.get("by_session", {}).items():
            by_session[sid].update({k: v for k, v in counts.items() if isinstance(v, int)})
            splits[sid] = counts.get("split")
    return {
        "sessions": len(sessions),
        "failed_sessions": [r["out_dir"] for r in sessions if not r["ok"]],
        "overall": _with_rates(total),
        "by_stratum": {name: _with_rates(c) for name, c in sorted(strata.items())},
        "by_session": {sid: {**_with_rates(c), "split": splits[sid]} for sid, c in sorted(by_session.items())},
        "per_session": sessions,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_path")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--out")
    a = p.parse_args(argv)
    gates: dict = {}
    reports = []
    for out_dir in a.sessions:
        try:
            rows = load_residency(os.path.join(out_dir, RESIDENCY_PREFIX)).rows
            key = tuple(msgspec.structs.astuple(row) for row in rows)
            if key not in gates:
                gates[key] = _gates(a.model_path, rows)
            W, bias = gates[key]
            reports.append(check_session(out_dir, W=W, bias=bias))
        except (OSError, ValueError, RuntimeError) as e:
            reports.append({"out_dir": out_dir, "ok": False, "error": f"{type(e).__name__}: {e}"})
    summary = _summarize(reports)
    text = json.dumps(summary, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(json.dumps({k: summary[k] for k in ("sessions", "failed_sessions", "overall")}, indent=1))
    return 0 if not summary["failed_sessions"] else 1


if __name__ == "__main__":
    sys.exit(main())
