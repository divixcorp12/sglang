"""Mirror read weights measured at startup (mirror_weight_calibration.py, SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION;
CPU): weights from synthetic throughputs, the noise and clamp guardrails, the probe's reads over real files, the
fallback, and the precedence of an explicit SGLANG_MOE_EXPERT_MIRROR_WEIGHTS."""

import logging
import math
import os

import pytest

from sglang.srt.environ import envs
from sglang.srt.layers.moe import mirror_weight_calibration as mw
from sglang.srt.layers.moe.exl3_expert_format import exl3_mirror_config
from sglang.srt.layers.moe.exl3_row_image import row_image_dir, write_manifest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GB = 1e9


@pytest.fixture(autouse=True)
def _fresh_cache():
    mw._CACHE.clear()
    yield
    mw._CACHE.clear()


def test_weights_follow_throughput_so_busy_time_evens_out():
    """read-amp's busy times (nvme3 104.5 s against 86.6 / 85.6 s for equal bytes) are throughputs of 1/104.5 :
    1/86.6 : 1/85.6; the weights are those over their mean, so bytes / throughput -- busy time -- is equal."""
    rates = [[GB / 86.6, GB / 104.5, GB / 85.6]] * 3
    weights, clamped = mw.weights_from_rates(rates)
    assert clamped == [] and math.isclose(sum(weights), 3, abs_tol=1e-3)
    busy = [w / r for w, r in zip(weights, rates[0])]
    assert max(busy) / min(busy) < 1.001
    assert weights[1] < weights[0] and weights[1] < weights[2]


def test_weights_are_clamped_to_the_range_and_the_clamp_is_reported():
    weights, clamped = mw.weights_from_rates([[10 * GB, 1 * GB, 1 * GB]])
    assert weights[0] == mw.WEIGHT_RANGE[1] and weights[1] == mw.WEIGHT_RANGE[0] and clamped == [0, 1, 2]


@pytest.mark.parametrize("rates, match", [
    ([[GB, 0.0]], "read nothing"),
    ([[GB, GB], [GB, 1.5 * GB]], "disagree"),
    ([], "no probe rounds"),
])
def test_a_probe_with_a_dead_or_noisy_root_fails(rates, match):
    with pytest.raises(mw.MirrorProbeError, match=match):
        mw.weights_from_rates(rates)


def test_a_sub_read_is_a_rows_share_of_a_root_over_its_pieces():
    assert mw.chunk_bytes(12 * 2**20, 3) == 512 * 1024
    assert mw.chunk_bytes(100, 3) == mw.ALIGN  # never below one aligned block


def _root(tmp_path, name, experts=8, stride=64 * 1024):
    root = tmp_path / name
    os.makedirs(row_image_dir(str(root)))
    for layer in (0, 1):
        with open(os.path.join(row_image_dir(str(root)), f"layer-{layer:03d}.rows"), "wb") as f:
            f.write(os.urandom(experts * stride))
    write_manifest(str(root), {"layers": {
        str(layer): {"file": f"layer-{layer:03d}.rows", "digests": ["x"] * experts} for layer in (0, 1)}})
    return str(root)


def test_the_probe_reads_every_root_at_once(tmp_path):
    roots = [_root(tmp_path, "a"), _root(tmp_path, "b")]
    files, stride = mw.root_files(roots[0])
    assert len(files) == 2 and stride == 64 * 1024
    rates, chunk = mw.probe_rates(roots, rounds=2, seconds=0.05, direct=False)
    assert chunk == mw.ALIGN and len(rates) == 2 and all(r > 0 for row in rates for r in row)


def test_a_root_without_images_fails_the_probe_and_keeps_equal_weights(tmp_path, caplog):
    roots = (_root(tmp_path, "a"), str(tmp_path))
    with caplog.at_level(logging.WARNING):
        assert mw.probed_weights(roots) == (1.0, 1.0)
    assert "keeping the static weights 1:1" in caplog.text


def test_probed_weights_are_computed_once_and_logged(caplog):
    calls = []

    def probe(roots):
        calls.append(roots)
        return [[2 * GB, 1 * GB]] * 3, 512 * 1024

    with caplog.at_level(logging.INFO):
        first = mw.probed_weights(("/a", "/b"), probe=probe)
        assert mw.probed_weights(("/a", "/b"), probe=probe) == first
    assert len(calls) == 1 and first == (1.3333, 0.6667)
    assert "/a 2.00 GB/s, /b 1.00 GB/s -> weights 1.3333:0.6667" in caplog.text


def test_explicit_weights_win_and_the_flag_is_off_by_default(tmp_path, monkeypatch):
    roots = [str(tmp_path / "x"), str(tmp_path / "y")]
    for r in roots:
        os.makedirs(r)
    monkeypatch.setattr(mw, "probed_weights", lambda roots: (0.8, 1.2))
    with envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.override(os.pathsep.join(roots)):
        assert not envs.SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION.get()
        assert exl3_mirror_config()[1] == (1.0, 1.0)
        with envs.SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION.override(True):
            assert exl3_mirror_config()[1] == (0.8, 1.2)
            with envs.SGLANG_MOE_EXPERT_MIRROR_WEIGHTS.override("16:10"):
                assert exl3_mirror_config()[1] == (16.0, 10.0)
    with envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.override(roots[0]), \
            envs.SGLANG_DSV41_ENABLE_MIRROR_WEIGHT_CALIBRATION.override(True):
        assert exl3_mirror_config()[1] == (1.0,)  # one root: nothing to weigh
