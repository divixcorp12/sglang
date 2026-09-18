"""Truncating the DeepSeek V4.1 config and index to the first N layers."""

import importlib.util
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41", "make_truncated_model.py")
_spec = importlib.util.spec_from_file_location("make_truncated_model", _PATH)
mtm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mtm)
K = mtm._KEYS


def _config():
    return {
        K["layers"]: 40,
        K["ratios"]: [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        K["kv_sources"]: [2, 8, 14, 20],
        K["index_sources"]: [2, 8, 14, 20, 24, 28, 32, 36],
        K["candidate"]: 20,
        K["engram_ids"]: [1, 14],
        K["engram_rows"]: [384006168, 384016682],
        K["nextn"]: 3,
        "hidden_size": 5120,
    }


def test_truncates_every_layer_indexed_field():
    t = mtm.truncate_text_config(_config(), 3)
    assert t[K["layers"]] == 3
    assert t[K["ratios"]] == [0, 0, 2]
    assert t[K["kv_sources"]] == [2] and t[K["index_sources"]] == [2]
    assert t[K["candidate"]] == -1
    assert t[K["engram_ids"]] == [1] and t[K["engram_rows"]] == [384006168]
    assert t[K["nextn"]] == 0 and t["hidden_size"] == 5120


def test_does_not_mutate_input():
    c = _config()
    mtm.truncate_text_config(c, 3)
    assert c[K["layers"]] == 40


def test_weight_map_keeps_first_layers_and_top_level_only():
    wm = {
        "embed.weight": "a",
        "head.trellis": "z",
        "norm.weight": "z",
        "hc_head_fn": "z",
        "layers.0.attn.wq_a.trellis": "a",
        "layers.2.ffn.experts.1.w1.trellis": "b",
        "layers.3.attn.wq_a.trellis": "c",
        "layers.12.attn.wq_a.trellis": "d",
        "mtp.0.ffn.experts.0.w1.trellis": "e",
        "vision.blocks.0.attn.wo.weight": "f",
        "aligner.w1.weight": "f",
    }
    got = mtm.select_weight_map(wm, 3)
    assert set(got) == {
        "embed.weight",
        "head.trellis",
        "norm.weight",
        "hc_head_fn",
        "layers.0.attn.wq_a.trellis",
        "layers.2.ffn.experts.1.w1.trellis",
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
