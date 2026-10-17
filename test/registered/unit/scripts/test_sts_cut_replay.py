"""sts_cut_replay.py: the cut replayed over an STS collection's verifies (CPU)."""

import importlib.util
import json
import os

import torch

from sglang.srt.speculative.dspark_components.dspark_sts import DSparkStsCalibration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _load():
    path = os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "sts_cut_replay.py")
    spec = importlib.util.spec_from_file_location("sts_cut_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_replay_caps_each_verifys_own_accept_at_the_cut_width(tmp_path):
    replay = _load()
    # Verify 0: confident, the accept took all 3 drafts. Verify 1: doubtful from draft 1, the accept took 1.
    logits = torch.tensor([[9.0, 9.0, 9.0], [-9.0, -9.0, -9.0]])
    sampled = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    torch.save(
        {"logits": logits, "prefix_mask": torch.zeros(2, 3), "sampled_prefix_mask": sampled,
         "accept_rates": torch.zeros(2, 3), "temperatures": torch.tensor([0.6, 0.6])},
        tmp_path / "c.0.pt",
    )
    sts = tmp_path / "sts.json"
    sts.write_text(DSparkStsCalibration(temperatures=[1.0, 1.0, 1.0]).to_json())
    out = tmp_path / "r.json"
    assert replay.main(["--data-glob", str(tmp_path / "c.*.pt"), "--cost", "100,10", "--json", str(out),
                        f"one={sts}"]) == 0
    report = json.loads(out.read_text())
    assert report["verifies"] == 2
    assert report["every_row"]["tokens_per_verify"] == 3.0  # (4 + 2) / 2
    assert report["every_row"]["ms_per_verify"] == 130.0
    # The cut verifies every row of verify 0 and only the anchor of verify 1: tokens 4 and 1.
    assert report["one"]["widths"] == [1, 0, 0, 1]
    assert report["one"]["tokens_per_verify"] == 2.5 and report["one"]["ms_per_verify"] == 115.0
    assert report["oracle"]["widths"] == [0, 1, 0, 1]
