"""b1_extract.py on test_b0_integrity's synthetic block: hand-computed groups, the warm-up skip, horizon validity, a
target without a biased gate and a dropped mismatch cell. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b0_integrity as b0  # noqa: E402
import b1_extract  # noqa: E402
import test_b0_integrity as t0  # noqa: E402
from b1_rows import GATE_DEPTH  # noqa: E402

# Of the 16 logits in t0._gates, experts 0..11 lead in id order, so the gate's top-12 is 0..11.
LOGITS = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, -5.6, -6.7] + [-6.0 - 0.1 * e for e in range(8, t0.EXPERTS)])
META = {"s1": {"split": "train", "warm_start": False, "flagged": False},
        "s2": {"split": "dev", "warm_start": True, "flagged": True}}
HOT20 = [1, 0]


def _hot_block(tmp_path):
    """t0._block (s1: seq 0 with 2 live tokens, s2: seq 1 with 1) with a logged hot set per layer."""
    block = t0._block(tmp_path)
    lines = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    lines[0]["hot_layer_ids"] = t0.LAYERS
    for line in lines[1:]:
        line["hot"] = [[], HOT20]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    return block


def _extract(block, *, drop=frozenset(), horizons=(1, 2, 4), meta=META):
    W, bias = t0._gates()
    counts = b1_extract.ExtractCounts()
    out = {s: (groups, x) for s, groups, x in b1_extract.extract_block(
        block, W=W, bias=bias, session_meta=meta, drop=set(drop), horizons=horizons, num_experts=t0.EXPERTS,
        counts=counts)}
    return out, counts


def test_each_session_gets_its_layer19_to_20_group_with_the_hand_computed_fields(tmp_path):
    out, counts = _extract(_hot_block(tmp_path))
    assert sorted(out) == ["s1", "s2"]
    for session, seq, live in (("s1", 0, t0.LIVE[0]), ("s2", 1, t0.LIVE[1])):
        (g,), x = out[session]
        assert (g.session, g.split, g.block, g.seq, g.rid) == (session, META[session]["split"], tmp_path.name, seq,
                                                               f"{session}-t0")
        assert (g.warm_start, g.flagged) == (META[session]["warm_start"], META[session]["flagged"])
        assert (g.horizon, g.source_row, g.source_layer, g.target_row, g.target_layer) == (1, 0, 19, 1, 20)
        assert g.gate_has_bias and g.source_stratum and g.target_stratum
        assert g.tokens.tolist() == list(range(live)) and g.x_offset == 0
        assert x.dtype == np.uint16 and x.shape == (live, t0.HIDDEN)
        assert (x[:, 0] == 0x3F80).all() and (x[:, 1:] == 0).all()
        assert g.source_ids.tolist() == [t0.ROUTE] * live and g.target_ids.tolist() == [t0.ROUTE] * live
        assert g.target_kinds.tolist() == [[int(k) for k in t0.KINDS]] * live
        assert g.ram_mask.tolist() == [t0.MAPPED, 0, 0, 0, 0, 0]
        assert g.hot.tolist() == HOT20
        assert g.gate_ids.tolist() == [list(range(GATE_DEPTH))] * live
        want = np.sqrt(np.log1p(np.exp(LOGITS[:GATE_DEPTH])))
        np.testing.assert_allclose(g.gate_scores, np.tile(want, (live, 1)), rtol=1e-5)
    # Per forward, row 0 has no layer 21 or 23 and row 1 no layer 21, 22 or 24.
    assert (counts.warmup_forwards, counts.dropped_cells, counts.invalid_horizons) == (0, 0, 5 * len(t0.LIVE))


def test_horizons_without_a_target_row_give_no_group_and_store_no_x(tmp_path):
    out, counts = _extract(_hot_block(tmp_path), horizons=(2, 4))
    assert all(groups == [] and x.shape == (0, t0.HIDDEN) for groups, x in out.values())
    assert counts.invalid_horizons == 4 * len(t0.LIVE)


def test_warmup_forwards_are_skipped_and_counted(tmp_path):
    block = t0._warmup_session(tmp_path)
    (tmp_path / "rids.json").write_text(json.dumps({"a": {"session_id": "sa", "turn": 0, "split": "train"}}))
    out, counts = _extract(block, meta={"sa": META["s1"]})
    groups, x = out["sa"]
    assert counts.warmup_forwards == 2
    assert [g.seq for g in groups] == [2, 3]
    assert [g.x_offset for g in groups] == [0, t0.LIVE[0]] and len(x) == sum(t0.LIVE)


def test_a_listed_mismatch_cell_is_dropped_from_its_source_row_and_counted(tmp_path):
    block = _hot_block(tmp_path)
    out, counts = _extract(block, drop={(0, 19, 0)})
    (g,), x = out["s1"]
    assert g.tokens.tolist() == [1] and len(x) == 1 and g.gate_ids.shape == (1, GATE_DEPTH)
    assert counts.dropped_cells == 1 and counts.dropped_by_split == {"train": {1: 1}, "dev": {1: 0}}
    assert len(out["s2"][0]) == 1
    out, counts = _extract(block, drop={(1, 19, 0)})
    assert out["s2"][0] == [] and len(out["s2"][1]) == 0 and counts.dropped_cells == 1


def test_a_target_without_a_biased_gate_has_no_gate_top12(tmp_path):
    block = _hot_block(tmp_path)
    header_path = tmp_path / f"{b0.RESIDENCY_PREFIX}.json"
    header = json.loads(header_path.read_text())
    header["rows"][1]["gate_has_bias"] = False
    header_path.write_text(json.dumps(header))
    out, _ = _extract(block)
    (g,), _ = out["s1"]
    assert not g.gate_has_bias
    assert (g.gate_ids == -1).all() and np.isneginf(g.gate_scores).all()


def test_mismatch_cells_reads_only_the_block_s_report(tmp_path):
    cell = {"seq": 1283, "router_record": 7, "layer_id": 39, "token": 1, "computed": [], "captured": []}
    report = {"per_session": [
        {"out_dir": str(tmp_path) + "/", "mismatches": [cell]},
        {"out_dir": str(tmp_path / "other"), "mismatches": [{**cell, "seq": 5}]},
        {"out_dir": str(tmp_path), "ok": False, "error": "OSError: x"},
    ]}
    assert b1_extract.mismatch_cells(report, block_dir=tmp_path) == {(1283, 39, 1)}
