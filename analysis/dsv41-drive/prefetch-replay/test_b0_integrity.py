"""b0_integrity.py on synthetic sessions: a clean pass, a counted bf16 tie, a failing non-tie mismatch, a missing
residency record and a header without per-row entries. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b0_integrity as b0  # noqa: E402
from sglang.srt.layers.moe.dsv41_strata import stratum  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402
from sglang.srt.layers.moe.residency_capture import ResidencyCapture, ResidencyRow  # noqa: E402

EXPERTS, HIDDEN, WIDTH, TOPK = 16, 4, 2, 6
LAYERS = [19, 20]
LIVE = [2, 1]
ROUTE = [5, 4, 3, 2, 1, 0]  # router order, not rank order: the check compares sets
# Kinds of ROUTE at layer 20: 1 and 0 VRAM hot, 3 and 2 RAM hits, 4 an NVMe miss the snapshot showed mapped
# (evicted), 5 a miss it showed unmapped.
KINDS = [LaneKind.MISS_CPU, LaneKind.MISS_GPU, LaneKind.HIT_COPY, LaneKind.HIT_COPY, 0, 0]
MAPPED = (1 << 2) | (1 << 3) | (1 << 4)


def _gates(rank7_logit=-5.6):
    """x is one-hot on dim 0, so expert e's logit is W[:, e, 0]: experts 0..5 lead, expert 6 is rank 7."""
    logits = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, rank7_logit] + [-6.0 - 0.1 * e for e in range(7, EXPERTS)]
    W = torch.zeros(len(LAYERS), EXPERTS, HIDDEN)
    W[:, :, 0] = torch.tensor(logits)
    return W, torch.zeros(len(LAYERS), EXPERTS)


def _trace(tmp_path):
    lines = [{"kind": "graph_routes_header", "schema": 3, "run": "r", "layer_ids": LAYERS, "hot_capacity": [1, 1]}]
    lines += [{"kind": "graph_routes", "schema": 3, "seq": s, "forward_pass_id": s + 1, "phase": "target_verify",
               "rids": ["a"], "forward_tokens": live, "routes": [ROUTE, ROUTE], "misses": [0, 0], "router": s}
              for s, live in enumerate(LIVE)]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))


def _router(tmp_path, captured):
    """Live tokens get x = 1.0 on dim 0 (bf16 0x3F80); the dead token's zero input would route arbitrarily."""
    prefix = str(tmp_path / "router")
    header = {"schema": 3, "layer_ids": LAYERS, "tokens": WIDTH, "hidden": HIDDEN, "topk": TOPK}
    (tmp_path / "router.json").write_text(json.dumps(header))
    x = np.zeros((len(LIVE), len(LAYERS), WIDTH, HIDDEN), dtype=np.uint16)
    for record, live in enumerate(LIVE):
        x[record, :, :live, 0] = 0x3F80
    ids = np.empty((len(LIVE), len(LAYERS), WIDTH, TOPK), dtype=np.int32)
    ids[:] = ROUTE
    if captured is not None:
        ids[0, 0, 0] = captured
    x.tofile(prefix + ".x.bin")
    ids.tofile(prefix + ".ids.bin")
    np.zeros(ids.shape, dtype=np.float32).tofile(prefix + ".w.bin")
    keys = [[s, s + 1, live] for s, live in enumerate(LIVE)]
    np.asarray(keys, dtype=np.int64).tofile(prefix + ".seq.bin")


def _residency(tmp_path, records):
    """Row 0 (layer 19) predicts row 1 (layer 20) at h=1 only; the dead token is labelled all NVMe misses."""
    rows = [ResidencyRow(layer_id=layer, stratum=stratum(layer), gate_has_bias=True) for layer in LAYERS]
    capture = ResidencyCapture(str(tmp_path / b0.RESIDENCY_PREFIX), rows=rows, schema=3)
    for s in range(records):
        masks = torch.zeros((len(LAYERS), 4, 6), dtype=torch.int64)
        masks[0, 0, 0] = MAPPED
        valid = torch.zeros((len(LAYERS), 4), dtype=torch.uint8)
        valid[0, 0] = 1
        kinds = torch.full((len(LAYERS), WIDTH, TOPK), LaneKind.MISS_GPU, dtype=torch.uint8)
        kinds[1, : LIVE[s]] = torch.tensor([int(k) for k in KINDS], dtype=torch.uint8)
        capture.write(run="r", depth=8, keys=(s, s + 1, LIVE[s]), masks=masks, valid=valid,
                      snap_stamp=torch.tensor([100, 100]), kinds=kinds, label_stamp=torch.tensor([200, 200]))
    capture.flush()


def _session(tmp_path, *, captured=None, records=len(LIVE)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _trace(tmp_path)
    _router(tmp_path, captured)
    _residency(tmp_path, records)
    return str(tmp_path)


def test_a_clean_session_passes_and_counts_only_live_tokens(tmp_path):
    """Dead token rows (past the live count) would both mismatch the gate and add evictions if they were read."""
    W, bias = _gates()
    report = b0.check_session(_session(tmp_path), W=W, bias=bias)
    assert report["ok"]
    assert (report["tie_count"], report["non_tie_mismatches"], report["seq_gaps"], report["missing_records"]) == (0, 0, 0, 0)
    assert report["route_cells"] == 2 * sum(LIVE)
    # Per record: row 0 has 3 no-target horizons, row 1 all 4.
    assert report["invalid_no_target"] == 7 * len(LIVE)
    assert (report["snapshot_routes"], report["evicted"]) == (3 * sum(LIVE), sum(LIVE))
    assert report["eviction_rate"] == pytest.approx(1 / 3)
    assert report["by_stratum"]["19->20"]["eviction_rate"] == pytest.approx(1 / 3)
    assert report["by_horizon"][1]["evicted"] == sum(LIVE)
    assert report["stamp_inversions"] == 0


def _warmup_session(tmp_path):
    """Two rids=[] verifies (seq 0, 1; zero x and ids) ahead of the two real forwards (seq 2, 3)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    lives, real = [1, 1, *LIVE], [False, False, True, True]
    lines = [{"kind": "graph_routes_header", "schema": 3, "run": "r", "layer_ids": LAYERS, "hot_capacity": [1, 1]}]
    lines += [{"kind": "graph_routes", "schema": 3, "seq": s, "forward_pass_id": s + 1, "phase": "target_verify",
               "rids": ["a"] if real[s] else [], "forward_tokens": live, "routes": [ROUTE, ROUTE],
               "misses": [0, 0], "router": s} for s, live in enumerate(lives)]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    prefix = str(tmp_path / "router")
    header = {"schema": 3, "layer_ids": LAYERS, "tokens": WIDTH, "hidden": HIDDEN, "topk": TOPK}
    (tmp_path / "router.json").write_text(json.dumps(header))
    x = np.zeros((len(lives), len(LAYERS), WIDTH, HIDDEN), dtype=np.uint16)
    ids = np.zeros((len(lives), len(LAYERS), WIDTH, TOPK), dtype=np.int32)
    for s, live in enumerate(lives):
        if real[s]:
            x[s, :, :live, 0] = 0x3F80
            ids[s] = ROUTE
    x.tofile(prefix + ".x.bin")
    ids.tofile(prefix + ".ids.bin")
    np.zeros(ids.shape, dtype=np.float32).tofile(prefix + ".w.bin")
    np.asarray([[s, s + 1, live] for s, live in enumerate(lives)], dtype=np.int64).tofile(prefix + ".seq.bin")
    rows = [ResidencyRow(layer_id=layer, stratum=stratum(layer), gate_has_bias=True) for layer in LAYERS]
    capture = ResidencyCapture(str(tmp_path / b0.RESIDENCY_PREFIX), rows=rows, schema=3)
    for s, live in enumerate(lives):
        masks = torch.zeros((len(LAYERS), 4, 6), dtype=torch.int64)
        valid = torch.zeros((len(LAYERS), 4), dtype=torch.uint8)
        kinds = torch.full((len(LAYERS), WIDTH, TOPK), LaneKind.MISS_GPU, dtype=torch.uint8)
        if real[s]:
            masks[0, 0, 0] = MAPPED
            valid[0, 0] = 1
            kinds[1, :live] = torch.tensor([int(k) for k in KINDS], dtype=torch.uint8)
        capture.write(run="r", depth=8, keys=(s, s + 1, live), masks=masks, valid=valid,
                      snap_stamp=torch.tensor([100, 100]), kinds=kinds, label_stamp=torch.tensor([200, 200]))
    capture.flush()
    return str(tmp_path)


def test_warmup_verify_forwards_without_rids_are_counted_not_checked(tmp_path):
    W, bias = _gates()
    report = b0.check_session(_warmup_session(tmp_path), W=W, bias=bias)
    assert report["ok"] and report["no_request_forwards"] == 2
    assert report["joined_forwards"] == len(LIVE) and report["missing_records"] == 0
    assert report["route_cells"] == 2 * sum(LIVE) and report["non_tie_mismatches"] == 0
    assert (report["snapshot_routes"], report["evicted"]) == (3 * sum(LIVE), sum(LIVE))
    assert "no_request_forwards" not in b0._FAILING


def test_a_rank6_rank7_swap_within_bf16_rounding_is_a_counted_tie(tmp_path):
    W, bias = _gates(rank7_logit=2.999)
    report = b0.check_session(_session(tmp_path, captured=[0, 1, 2, 3, 4, 6]), W=W, bias=bias)
    assert report["ok"] and report["tie_count"] == 1 and report["non_tie_mismatches"] == 0
    assert report["by_stratum"][stratum(19)]["tie_count"] == 1
    assert report["tie_rate"] == pytest.approx(1 / (2 * sum(LIVE)))


def test_a_non_tie_mismatch_is_kept_and_fails_the_run(tmp_path, monkeypatch):
    """The same rank-6/7 swap is not a tie when the scores are far apart, and an expert from below rank 7 never is."""
    W, bias = _gates()
    far = b0.check_session(_session(tmp_path / "far", captured=[0, 1, 2, 3, 4, 6]), W=W, bias=bias)
    assert not far["ok"] and far["tie_count"] == 0 and far["non_tie_mismatches"] == 1
    session = _session(tmp_path / "deep", captured=[0, 1, 2, 3, 4, 9])
    monkeypatch.setattr(b0.router_score, "load_gates", lambda model_path, layer_ids: (W, bias))
    out = tmp_path / "report.json"
    assert b0.main(["model", session, "--out", str(out)]) == 1
    report = json.loads(out.read_text())
    assert report["failed_sessions"] == [session]
    (kept,) = report["per_session"][0]["mismatches"]
    assert (kept["layer_id"], kept["token"], kept["captured"]) == (19, 0, [0, 1, 2, 3, 4, 9])


def test_a_verify_without_its_residency_record_fails(tmp_path):
    W, bias = _gates()
    report = b0.check_session(_session(tmp_path, records=1), W=W, bias=bias)
    assert not report["ok"] and report["missing_records"] == 1


def test_a_residency_header_without_per_row_entries_is_refused(tmp_path):
    session = _session(tmp_path)
    header_path = tmp_path / f"{b0.RESIDENCY_PREFIX}.json"
    header = json.loads(header_path.read_text())
    del header["rows"]
    header_path.write_text(json.dumps(header))
    W, bias = _gates()
    with pytest.raises(ValueError, match="per-row"):
        b0.check_session(session, W=W, bias=bias)
