"""GPU tests for l2_fast_cost: cost.json schema, per-slot inputs, and the timed fused graph doing the eager work."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402
import torch  # noqa: E402

import l2_fast_cost  # noqa: E402
import l2_fast_kernels  # noqa: E402
import r1_cost  # noqa: E402
from test_l2_fast_kernels import _folded  # noqa: E402

from sglang.kernels.ops.moe import spec_head  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cost timing needs CUDA")

DEV = "cuda"
STATS = {"median_ms", "p90_ms", "p99_ms", "runs", "nodes"}


def test_result_has_every_contract_key_and_incremental_is_fused_minus_native():
    """A renamed or dropped key breaks the report (T12); incremental must be fused minus native per width."""
    heads = {"L2_T0": (_folded(t3=False), "sha-t0"), "L2_T3": (_folded(t3=True), "sha-t3")}
    gates = l2_fast_cost.synthetic_gates(count=2, device=DEV)
    r = l2_fast_cost.sweep(heads=heads, gates=gates, widths=(1, 2), slots=2, runs=3, warmup=1, device=DEV)
    assert {"label", "scope", "widths", "slots", "runs", "gpu", "bar_ms_per_layer", "models",
            "native_bandwidth_gbps"} <= set(r)
    assert r["label"] == "observed offline" and r["bar_ms_per_layer"] == 0.1 and r["slots"] == 2
    assert set(r["models"]) == {"L2_T0", "L2_T3"} and set(r["native_bandwidth_gbps"]) == {"1", "2"}
    for name, m in r["models"].items():
        assert set(m) == {"checkpoint_sha256", "native", "fused", "torch", "incremental_ms"}
        assert m["checkpoint_sha256"] == {"L2_T0": "sha-t0", "L2_T3": "sha-t3"}[name]
        for config in l2_fast_cost.CONFIGS:
            assert set(m[config]) == {"1", "2"}
            for stats in m[config].values():
                assert set(stats) == STATS and stats["runs"] == 3 and stats["nodes"] >= 2
        for w in ("1", "2"):
            assert m["incremental_ms"][w] == pytest.approx(m["fused"][w]["median_ms"] - m["native"][w]["median_ms"])


def test_slots_never_share_inputs_and_gates_rotate():
    """R11: a buffer shared across slots would let replays hit L2 and understate the cost."""
    gates = l2_fast_cost.synthetic_gates(count=3, device=DEV)
    slots = l2_fast_cost.build_slots(gates=gates, width=2, slots=3, device=DEV)
    for field in ("x", "scores", "ids", "weights", "skip", "ram_slot"):
        assert len({getattr(s, field).data_ptr() for s in slots}) == 3, field
    assert l2_fast_cost.rotated_gate_bytes(slots) == 3 * l2_fast_cost.EXPERTS * r1_cost.X_DIM * 2


@pytest.mark.parametrize("t3", [False, True], ids=["T0", "T3"])
def test_fused_graph_record_equals_eager_kernel_score(t3):
    """The timed graph must run the same work as an eager KernelHead.score on that slot, not a stale or empty one."""
    folded = _folded(t3=t3)
    head = l2_fast_kernels.KernelHead(folded, device=DEV)
    slots = l2_fast_cost.build_slots(gates=l2_fast_cost.synthetic_gates(count=2, device=DEV), width=2, slots=2,
                                     device=DEV)
    threshold = torch.zeros(1, dtype=torch.float32, device=DEV)
    sel = l2_fast_cost.new_native_select(device=DEV)
    steps = [l2_fast_cost.fused_step(s, head=head, rows=len(folded.rows), sel=sel, threshold=threshold)
             for s in slots]
    with torch.no_grad():
        # Extra cast/copy nodes inside the graph would inflate the incremental cost: score + 4 head kernels only.
        score_nodes = l2_fast_cost.common.graph_nodes(lambda: l2_fast_cost._score(slots[0]))
        assert l2_fast_cost.common.graph_nodes(steps[0]) == score_nodes + 4
        graphs = [r1_cost._capture(step) for step in steps]
    for slot, graph in zip(slots, graphs):
        slot.scores.zero_()
        graph.replay()
        torch.cuda.synchronize()
        got = spec_head.read_record(head.workspace)
        assert got[0] == 1  # threshold 0 on sigmoid scores: always a read unless every expert is skipped
        count, expert, value, _block = head.score(
            scores=slot.scores, x=slot.x, src_ids=slot.ids, src_w=slot.weights, positions=slot.positions,
            row=slot.index % len(folded.rows), skip=slot.skip, threshold=0.0)
        assert (count, expert) == got[:2] and value == pytest.approx(got[2], abs=1e-6)
        assert not slot.skip[expert]
