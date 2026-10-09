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
