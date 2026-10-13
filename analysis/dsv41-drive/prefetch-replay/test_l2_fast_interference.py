"""GPU tests for l2_fast_interference: result schema and side-stream separation, at tiny shapes and few runs."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

import l2_fast_interference  # noqa: E402
from test_l2_fast_kernels import _folded  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="interference timing needs CUDA")

STATS = {"median_ms", "p90_ms", "p99_ms", "runs", "graphs"}


def _measure(**overrides) -> dict:
    heads = {"L2_T0": (_folded(t3=False), "sha-t0"), "L2_T3": (_folded(t3=True), "sha-t3")}
    kw = dict(heads=heads, shapes=[(64, 64), (64, 128)], main_tokens=6, width=6, slots=3, runs=5, warmup=2)
    return l2_fast_interference.measure(**{**kw, **overrides})


def test_result_schema_and_slowdown_arithmetic():
    """A renamed or dropped key breaks the contract T12 reads; slowdowns are ratios of the stored medians."""
    r = _measure()
    assert {"label", "scope", "main_load", "width", "runs", "gpu", "main_alone", "native_side", "models"} <= set(r)
    assert r["label"] == "observed offline" and r["main_load"]["label"].startswith("hypothesis")
    assert set(r["main_load"]) >= {"shapes", "bytes"} and set(r["models"]) == {"L2_T0", "L2_T3"}
    for stats in (r["main_alone"], r["native_side"]):
        assert set(stats) == STATS and stats["runs"] == 5
    for m in r["models"].values():
        assert set(m) == {"checkpoint_sha256", "fused_side", "slowdown_vs_native", "slowdown_vs_none"}
        assert m["slowdown_vs_native"] == pytest.approx(m["fused_side"]["median_ms"] / r["native_side"]["median_ms"] - 1)
        assert m["slowdown_vs_none"] == pytest.approx(m["fused_side"]["median_ms"] / r["main_alone"]["median_ms"] - 1)


def test_side_work_runs_on_a_stream_other_than_the_main_stream():
    """Side work on the main stream would serialize with the load and report no interference at all."""
    r = _measure(runs=2)
    assert r["streams"]["side_stream_ptr"] != r["streams"]["main_stream_ptr"]
    main = l2_fast_interference.build_main_load(shapes=[(64, 64)], tokens=6, device="cuda")
    with pytest.raises(ValueError, match="differ"):
        l2_fast_interference.time_main(main.graphs, None, side=torch.cuda.current_stream(), runs=1, warmup=0)


def test_main_weights_rotate_past_twice_l2():
    """One weight copy resident in L2 would time the load from cache, not HBM."""
    one_mib = 2**20
    copies = l2_fast_interference.copies_for(bytes_per_copy=one_mib)
    assert copies * one_mib >= 2 * l2_fast_interference.L2_BYTES
    assert l2_fast_interference.copies_for(bytes_per_copy=10 * l2_fast_interference.L2_BYTES) == 2
