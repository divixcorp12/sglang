"""r1_audit.py on a synthetic dataset and corpus: the D14 exclusion manifest and its affected groups, the input
manifest's hashes, splits and denominators, and the seeded gate parity sample. CPU only; locked_test is never read."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_rows  # noqa: E402
import r1_audit  # noqa: E402
import r1_groups  # noqa: E402
import r1_stage  # noqa: E402
import router_score  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind  # noqa: E402

HIDDEN = 8
MISS_GPU, HIT = int(LaneKind.MISS_GPU), int(LaneKind.HIT_COPY)


def _group(*, seq, source_row, horizon=1, tokens=(0, 1, 2), x_offset=0, session="s1", split="dev", block="b000",
           gate=None, x=None):
    tok = np.array(tokens, np.int16)
    n = len(tok)
    target_ids = (np.arange(6, dtype=np.int16)[None] * 10 + tok[:, None] + 20).astype(np.int16)
    kinds = np.zeros((n, 6), np.uint8)
    kinds[:, 0], kinds[:, 1] = MISS_GPU, HIT
    if gate is None:
        gate_ids = np.tile(np.arange(12, dtype=np.int16), (n, 1))
        gate_scores = np.linspace(2.0, 1.0, 12, dtype=np.float32)[None].repeat(n, 0)
    else:
        W, bias = gate
        top = torch.topk(torch.nn.functional.softplus(torch.from_numpy(x) @ W.T).sqrt() + bias, 12, dim=-1)
        gate_ids, gate_scores = top.indices.numpy().astype(np.int16), top.values.numpy().astype(np.float32)
    return b1_rows.Group(
        session=session, split=split, block=block, seq=seq, rid=f"r{seq}", warm_start=False, flagged=False,
        horizon=horizon, source_row=source_row, source_layer=source_row + 3, source_stratum="a",
        gate_has_bias=True, target_row=source_row + horizon, target_layer=source_row + 3 + horizon,
        target_stratum="b", tokens=tok, x_offset=x_offset, source_ids=np.zeros((n, 6), np.int16),
        source_weights=np.full((n, 6), 0.25, np.float32), target_ids=target_ids, target_kinds=kinds,
        ram_mask=np.zeros(6, np.uint64), hot=np.zeros(0, np.int16), gate_ids=gate_ids, gate_scores=gate_scores)


def _rows():
    return [{"layer_id": layer, "stratum": "a", "gate_has_bias": True} for layer in range(3, 10)]


def _integrity(corpus, cells):
    mismatches = [{"seq": s, "router_record": 0, "layer_id": layer, "token": t, "computed": [1, 2],
                   "captured": [1, 3], "rank6_minus_rank7": 0.5} for s, layer, t in cells]
    return {"per_session": [{"out_dir": str(corpus / "blocks" / "b000"), "rows": _rows(), "mismatches": mismatches},
                            {"out_dir": str(corpus / "blocks" / "b001"), "rows": _rows(), "mismatches": []}]}


def _layout(tmp_path):
    """Dataset with s1 (dev, b000), s2 (calib, b000), s3 (train, b001) and a locked s4 (b000)."""
    data, corpus = tmp_path / "data", tmp_path / "corpus"
    by_split = {"dev": ["s1"], "calib": ["s2"], "train": ["s3"], "locked_test": ["s4"]}
    blocks = {"s1": "b000", "s2": "b000", "s3": "b001", "s4": "b000"}
    for split, (session,) in by_split.items():
        block = blocks[session]
        h1 = [_group(seq=7, source_row=0, session=session, split=split, block=block),
              # B1 already dropped token 1 where layer 4 is the source.
              _group(seq=7, source_row=1, tokens=(0, 2), x_offset=3, session=session, split=split, block=block),
              _group(seq=7, source_row=4, x_offset=5, session=session, split=split, block=block),
              _group(seq=8, source_row=0, x_offset=8, session=session, split=split, block=block)]
        h2 = [_group(seq=7, source_row=0, horizon=2, session=session, split=split, block=block),
              _group(seq=7, source_row=1, horizon=2, tokens=(0, 2), x_offset=3, session=session, split=split,
                     block=block)]
        for h, groups in ((1, h1), (2, h2), (4, [])):
            b1_rows.write_groups(data, split=split, horizon=h, session=session, groups=groups)
        b1_rows.write_x(data, split=split, session=session, x=np.zeros((11, HIDDEN), np.uint16))
    (data / "dataset.json").write_text(json.dumps({"sessions_by_split": by_split,
                                                   "built_sessions": sorted(blocks)}))
    for block in ("b000", "b001"):
        block_dir = corpus / "blocks" / block
        block_dir.mkdir(parents=True)
        rids = {f"rid-{s}": {"session_id": s, "turn": 0, "split": "x"} for s, b in blocks.items() if b == block}
        (block_dir / "rids.json").write_text(json.dumps(rids))
    (corpus / "manifest.json").write_text(json.dumps({"split": by_split}))
    return data, corpus


def test_exclusion_manifest_lists_source_and_target_groups_and_loads(tmp_path):
    data, corpus = _layout(tmp_path)
    got = r1_audit.build_exclusions(integrity=_integrity(corpus, [(7, 4, 1)]), dataset_dir=data, corpus_dir=corpus)
    (cell,) = got["cells"]
    assert (cell["block"], cell["seq"], cell["layer_id"], cell["token"]) == ("b000", 7, 4, 1)
    assert cell["reason"]
    src = sorted((g["split"], g["horizon"], g["source_layer"], g["target_layer"], g["token_present"])
                 for g in cell["source_groups"])
    tgt = sorted((g["split"], g["horizon"], g["source_layer"], g["target_layer"], g["token_present"])
                 for g in cell["target_groups"])
    # Train s3 sits in b001 and locked s4 is never read, so only dev and calib appear.
    assert src == [(s, h, 4, 4 + h, False) for s in ("calib", "dev") for h in (1, 2)]
    assert tgt == [("calib", 1, 3, 4, True), ("dev", 1, 3, 4, True)]
    path = tmp_path / "exclusions.json"
    r1_stage.write_json_atomic(str(path), got)
    assert r1_groups.load_exclusions(path).cells == frozenset({("b000", 7, 4, 1)})


def test_no_mismatch_gives_an_empty_manifest(tmp_path):
    data, corpus = _layout(tmp_path)
    got = r1_audit.build_exclusions(integrity=_integrity(corpus, []), dataset_dir=data, corpus_dir=corpus)
    assert got["cells"] == []


def test_input_manifest_hashes_splits_layers_and_denominators(tmp_path):
    data, corpus = _layout(tmp_path)
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text(json.dumps(_integrity(corpus, [(7, 4, 1)])))
    head = tmp_path / "head.pt"
    head.write_bytes(b"head")
    m = r1_audit.input_manifest(dataset_dir=data, corpus_dir=corpus, integrity_path=integrity_path,
                                b1_head_path=head, repo=r1_audit.REPO)
    assert m["sha256"]["b1_head"] == r1_stage.sha256_file(str(head))
    assert m["sha256"]["shards"] == r1_stage.sha256_json(m["shards"])
    assert m["shards"] and not any(k.startswith("locked_test") for k in m["shards"])
    assert {s: v["sessions"] for s, v in m["splits"].items()} == {"train": ["s3"], "dev": ["s1"], "calib": ["s2"]}
    assert m["locked_test"] == {"count": 1, "read": False}
    assert m["layer_ids"] == list(range(3, 10)) and len(m["commit"]) == 40
    assert m["exclusion_cells"] == [["b000", 7, 4, 1]]
    # h1 dev: 4 groups of 3+2+3+3 tokens, minus token 1 of the seq-7 group targeting layer 4.
    dev1 = m["denominators"]["dev"]["1"]
    assert dev1["groups"] == 4 and dev1["tokens"] == 10
    excl = r1_groups.Exclusions(cells=frozenset({("b000", 7, 4, 1)}))
    want = sum(len(r.labels.eligible_addr) for r in r1_groups.read_r1_groups(data, split="dev", horizon=1,
                                                                               exclusions=excl))
    assert dev1["eligible_addr"] == want and dev1["eligible_all"] == want
    assert m["denominators"]["calib"]["4"]["groups"] == 0 and "train" not in m["denominators"]


def _parity_layout(tmp_path):
    g = torch.Generator().manual_seed(0)
    gates = {layer: (torch.randn(384, HIDDEN, generator=g) * 0.3, torch.randn(384, generator=g) * 0.1)
             for layer in range(4, 9)}
    words = (torch.randn(15, HIDDEN, generator=g).numpy().view(np.uint32) >> 16).astype(np.uint16)
    xf = router_score.bf16_to_f32(words)
    groups = [_group(seq=s, source_row=s, x_offset=3 * s, gate=gates[s + 4], x=xf[3 * s:3 * s + 3])
              for s in range(5)]
    b1_rows.write_groups(tmp_path, split="dev", horizon=1, session="s1", groups=groups)
    b1_rows.write_x(tmp_path, split="dev", session="s1", x=words)
    return gates


def test_gate_parity_sample_is_seeded_and_exact_on_stored_gates(tmp_path):
    gates = _parity_layout(tmp_path)
    kw = dict(dataset_dir=tmp_path, model_path=None, split="dev", horizon=1, n_groups=3, gates=gates)
    a = r1_audit.gate_parity_sample(seed=0, **kw)
    assert a == r1_audit.gate_parity_sample(seed=0, **kw)
    assert a["groups"] == 3 and a["tokens"] == 9 and a["exact"] == 9
    assert a["tie_only"] == 0 and a["mismatched"] == 0 and a["no_bias_groups"] == 0
    full = r1_audit.gate_parity_sample(seed=1, **{**kw, "n_groups": 10})
    assert full["groups"] == 5 and full["exact"] == 15


def test_sample_is_seeded_and_in_stream_order():
    a = r1_audit._sample(iter(range(100)), n=5, seed=3)
    assert a == r1_audit._sample(iter(range(100)), n=5, seed=3)
    assert a == sorted(a) and len(set(a)) == 5
    assert r1_audit._sample(iter(range(4)), n=10, seed=0) == [0, 1, 2, 3]


def test_parity_refuses_locked_test(tmp_path):
    with pytest.raises(PermissionError):
        r1_audit.gate_parity_sample(dataset_dir=tmp_path, model_path=None, split="locked_test", horizon=1,
                                    n_groups=1, seed=0, gates={})
