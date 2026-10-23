"""r1_stamps.py on a synthetic two-row residency capture: windows, inversions, unstamped and no-target cells. CPU only."""

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import r1_stamps as st  # noqa: E402
from sglang.srt.layers.moe.expert_engine.profiles.dsv41_strata import stratum  # noqa: E402
from sglang.srt.layers.moe.residency_capture import ResidencyCapture, ResidencyRow  # noqa: E402

LAYERS = [19, 20]
WIDTH, TOPK = 2, 6
# Per record: (seq, snap stamps of rows 0/1, label stamps of rows 0/1).
RECORDS = [
    (0, [1_000_000, 0], [0, 3_500_000]),
    (1, [2_000_000, 0], [0, 1_000_000]),  # inversion
    (2, [0, 0], [0, 5_000_000]),  # unstamped snapshot
]


def _block(path: Path, *, valid_h1: bool = True) -> str:
    path.mkdir(parents=True)
    rows = [ResidencyRow(layer_id=layer, stratum=stratum(layer), gate_has_bias=True) for layer in LAYERS]
    capture = ResidencyCapture(str(path / st.RESIDENCY_PREFIX), rows=rows, schema=3)
    for seq, snap, label in RECORDS:
        valid = torch.zeros((len(LAYERS), 4), dtype=torch.uint8)
        valid[0, 0] = int(valid_h1)
        capture.write(run="r", depth=8, keys=(seq, seq + 1, 1), masks=torch.zeros((2, 4, 6), dtype=torch.int64),
                      valid=valid, snap_stamp=torch.tensor(snap), kinds=torch.zeros((2, WIDTH, TOPK), dtype=torch.uint8),
                      label_stamp=torch.tensor(label))
    capture.flush()
    return str(path)


def test_block_windows_join_and_count(tmp_path):
    windows, audit = st.block_windows(_block(tmp_path / "b"), horizons=(1, 2))
    assert windows == {(0, 19, 1): st.Window(snap_ns=1_000_000, demand_ns=3_500_000)}
    assert windows[(0, 19, 1)].window_ms == pytest.approx(2.5)
    assert (audit.joined, audit.inversions, audit.nonpositive) == (1, 1, 1)
    # h=2 has no target for either row, h=1 has none for row 1.
    assert audit.missing == 3 * 2 + 3
    assert audit.joined + audit.missing + audit.inversions + audit.nonpositive == 2 * 2 * len(RECORDS)


def test_cleared_valid_flag_is_missing(tmp_path):
    windows, audit = st.block_windows(_block(tmp_path / "b", valid_h1=False), horizons=(1,))
    assert windows == {}
    assert audit.missing == 2 * len(RECORDS)


def test_corpus_windows_key_by_block(tmp_path):
    _block(tmp_path / "blocks" / "b000")
    windows, audit = st.corpus_windows(str(tmp_path), blocks=["b000"], horizons=(1,))
    assert list(windows) == [("b000", 0, 19, 1)]
    assert audit.joined == 1
