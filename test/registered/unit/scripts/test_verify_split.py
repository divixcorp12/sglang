"""Misses split by verified position: job-trace misses joined to the route log, router ids and accept log (CPU)."""

import importlib.util
import json
import os

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _module():
    spec = importlib.util.spec_from_file_location(
        "verify_split", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "verify_split.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_miss_needed_only_after_the_last_accepted_position_counts_as_rejected():
    m = _module()
    ids = np.array([[[1, 2], [3, 4], [5, 6]]])  # one layer, three positions, top-2
    # One correct draft: positions 0 and 1 are used; 5 only position 2 needed.
    split = m.split_layer(misses={2, 3, 5}, ids_layer=ids[0], num_correct_drafts=1, live=3)
    assert split == {"misses": 3, "accept": 2, "reject_only": 1}


def test_positions_past_the_live_count_are_padding_and_never_count():
    m = _module()
    ids = np.array([[[1, 2], [3, 4], [9, 9]]])
    assert m.split_layer(misses={9}, ids_layer=ids[0], num_correct_drafts=0, live=2) == {
        "misses": 1, "accept": 0, "reject_only": 1}


def test_alignment_finds_the_offset_whose_routes_hold_the_misses():
    m = _module()
    routes = [[{1, 2}], [{3, 4}], [{5, 6}], [{7, 8}]]  # route forwards, one layer each
    misses = [[{3}], [{5, 6}], [{8}]]  # job forwards: they start at route forward 1
    assert m.align(misses, routes, max_offset=2) == (1, 1.0)


def test_each_verify_is_matched_to_its_requests_kth_accept_line():
    m = _module()
    lines = [{"kind": "graph_routes", "phase": "target_verify", "rids": [r], "router": i, "forward_tokens": 6,
              "routes": [[1]], "seq": i} for i, r in enumerate(["a", "b", "a"])]
    accepts = {("a", 0): 2, ("b", 0): 5, ("a", 1): 0}
    assert [v["num_correct_drafts"] for v in m.verify_forwards(lines, accepts)] == [2, 5, 0]


def test_router_ids_are_read_from_the_side_files(tmp_path):
    m = _module()
    prefix = str(tmp_path / "router")
    with open(prefix + ".json", "w") as f:
        json.dump({"layer_ids": [0, 1], "tokens": 3, "topk": 2}, f)
    ids = np.arange(2 * 2 * 3 * 2, dtype=np.int32).reshape(2, 2, 3, 2)
    ids.tofile(prefix + ".ids.bin")
    got = m.router_ids(prefix)
    assert got.shape == (2, 2, 3, 2) and int(got[1, 0, 2, 1]) == int(ids[1, 0, 2, 1])


LAYERS, TOKENS, TOPK, FORWARDS = 2, 2, 2, 12


def _capture(out_dir, job_routes, drop=(), base_gen=100):
    """A capture of FORWARDS target-verify forwards, one request, one correct draft each: route forward i routes
    position p of layer l to experts 1000 * l + 10 * i + 2 * p and + 1. Job forward k misses, on each layer, both
    experts of position 0 (accepted) and the first of position 1 (rejected) of route forward ``job_routes[k]``. A
    forward in ``drop`` is left incomplete (no gate on its last row), as the job trace leaves one it could not
    finish."""
    ids = np.zeros((FORWARDS, LAYERS, TOKENS, TOPK), dtype=np.int32)
    for i in range(FORWARDS):
        for layer in range(LAYERS):
            for p in range(TOKENS):
                ids[i, layer, p] = [1000 * layer + 10 * i + 2 * p, 1000 * layer + 10 * i + 2 * p + 1]
    with open(os.path.join(out_dir, "router.json"), "w") as f:
        json.dump({"layer_ids": list(range(LAYERS)), "tokens": TOKENS, "topk": TOPK}, f)
    ids.tofile(os.path.join(out_dir, "router.ids.bin"))
    with open(os.path.join(out_dir, "stages.jsonl"), "w") as f:
        for i in range(FORWARDS):
            f.write(json.dumps({"kind": "graph_routes", "phase": "target_verify", "rids": ["r"], "router": i, "seq": i,
                                "forward_tokens": TOKENS,
                                "routes": [sorted(int(e) for e in ids[i, layer].ravel()) for layer in range(LAYERS)]})
                    + "\n")
    with open(os.path.join(out_dir, "verify-accept.1.jsonl"), "w") as f:
        for i in range(FORWARDS):
            f.write(json.dumps({"rid": "r", "k": i, "num_correct_drafts": 0}) + "\n")
    with open(os.path.join(out_dir, "events.1.tier.0.jsonl"), "w") as f:
        for k, i in enumerate(job_routes):
            for layer in range(LAYERS):
                gen = base_gen + LAYERS * k + layer
                misses = [int(ids[i, layer, 0, 0]), int(ids[i, layer, 0, 1]), int(ids[i, layer, 1, 0])]
                f.write(json.dumps({"event": "copy_submit", "row": layer, "gen": gen, "ns": gen, "b": len(misses)})
                        + "\n")
                if not (k in drop and layer == LAYERS - 1):
                    f.write(json.dumps({"event": "gate_open", "row": layer, "gen": gen, "ns": gen + 1}) + "\n")
                for e in misses:
                    f.write(json.dumps({"event": "miss_expert", "row": layer, "gen": gen, "a": e}) + "\n")


def test_a_forward_the_job_trace_dropped_costs_only_that_forward(tmp_path):
    m = _module()
    # The job trace never completed route forward 5: one offset joins 0-4 or 6-11, never both (5 or 6 of 11).
    _capture(str(tmp_path), job_routes=list(range(FORWARDS)), drop={5})
    result = m.analyze(str(tmp_path))
    assert result["verifies"] == FORWARDS - 1
    assert result["alignment"]["joined"] == FORWARDS - 1 and result["alignment"]["job_forwards"] == FORWARDS - 1
    assert result["per_verify"] == {"misses": 6, "accept": 4, "reject_only": 2}


def test_a_job_trace_from_another_order_is_still_refused(tmp_path):
    m = _module()
    order = [3, 9, 0, 7, 11, 2, 5, 10, 1, 8, 4, 6]  # the job forwards in another order than the route log
    _capture(str(tmp_path), job_routes=order)
    with pytest.raises(ValueError, match="do not join"):
        m.analyze(str(tmp_path))
