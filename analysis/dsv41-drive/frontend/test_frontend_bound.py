"""frontend_bound.py: per-layer chain grouping and the saving bound. CPU only."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from frontend_bound import copy_metrics, saving_bound_ns, split_layers  # noqa: E402

US = 1000


def _chain(t0, w1_us, s_us, cw_us):
    """One layer at t0 (ns): post 2 us, W1, C1 1 us, A 1 us, S, A 1 us, CW, F 1 us, back to back."""
    out, t = [], t0
    for label, dur in (("post", 2), ("W1", w1_us), ("C1", 1), ("A", 1), ("S", s_us), ("A", 1), ("CW", cw_us), ("F", 1)):
        out.append((t, t + dur * US, label))
        t += dur * US
    return out


def test_split_layers_starts_a_layer_at_each_post_and_takes_the_first_a_as_a1():
    kernels = _chain(0, 50, 30, 20) + _chain(1_000_000, 5, 10, 400)
    layers = split_layers(kernels)
    assert len(layers) == 2
    assert layers[0].s_start - layers[0].post_end == (50 + 1 + 1) * US  # W1 + C1 + A1
    assert layers[1].w1_ns == 5 * US and layers[1].cw_ns == 400 * US


def test_a_cut_hidden_behind_cw_spin_saves_nothing_and_the_excess_is_the_bound():
    # CW spun 70 us past its floor: removing 50 us before it only lengthens the spin.
    assert saving_bound_ns(cut_ns=50 * US, cw_ns=80 * US, cw_floor_ns=10 * US) == 0
    # CW spun 10 us: 40 of the 50 us come off the layer.
    assert saving_bound_ns(cut_ns=50 * US, cw_ns=20 * US, cw_floor_ns=10 * US) == 40 * US
    # CW under its floor is not negative spin.
    assert saving_bound_ns(cut_ns=50 * US, cw_ns=5 * US, cw_floor_ns=10 * US) == 50 * US


def test_copy_metrics_gives_each_layer_the_copies_that_start_inside_its_chain():
    # Layer 0: post ends at 2 us, CW ends at 105 us. Layer 1: post ends at 1002 us, CW ends at 1060 us.
    layers = split_layers(_chain(0, 50, 30, 20) + _chain(1_000_000, 5, 30, 20))
    copies = [
        (2 * US, 101 * US, 1000),                # layer 0: lands 99 us after post, 4 us before CW ends
        (1_002 * US, 1_012 * US, 500),           # layer 1, first copy
        (1_022 * US, 1_050 * US, 500),           # layer 1, last: lands 48 us after post, 10 us before CW ends
        (1_070 * US, 1_080 * US, 999),           # after layer 1's CW: belongs to no layer
    ]
    r = copy_metrics([layers], copies)
    assert r["copy_bytes_per_step"] == 2000
    assert abs(r["copy_busy_ms_per_step"] - 0.137) < 1e-9
    assert r["copy_done_after_post_p50_us"] == 73.5
    assert r["cw_end_after_copy_p50_us"] == 7.0


def test_the_cli_reads_a_node_mode_export_and_reports_per_step_bounds(tmp_path):
    names = {
        1: "exl3_ram_miss_post_kernel",
        2: "exl3_ram_miss_lease_stream_hit_wait_kernel",
        3: "copy_expert_row_segments_gpu_kernel",
        4: "exl3_ram_miss_lease_stage_ack_kernel",
        5: "exl3_ram_miss_lease_stream_kernel",
        6: "exl3_ram_miss_lease_copy_wait_kernel",
        7: "exl3_ram_miss_lease_finalize_kernel",
    }
    ids = {"post": 1, "W1": 2, "C1": 3, "A": 4, "S": 5, "CW": 6, "F": 7}
    db = tmp_path / "t.sqlite"
    c = sqlite3.connect(db)
    c.execute("create table StringIds (id integer, value text)")
    c.executemany("insert into StringIds values (?, ?)", names.items())
    c.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, correlationId int, shortName int, "
              "streamId int, graphId int)")
    c.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, bytes int, streamId int, copyKind int, "
              "graphNodeId int)")
    rows, copies = [], []
    for step in range(2):
        base = step * 10_000_000
        # Layer 1: W1 polls 100 us (budget), CW at its floor. Layer 2: W1 one pass, CW spins 300 us.
        for chain in (_chain(base, 100, 30, 10), _chain(base + 1_000_000, 5, 30, 310)):
            for s, e, label in chain:
                rows.append((s, e, step + 1, ids[label], 7, 1))
            post_end = chain[0][1]
            cw_end = next(e for _, e, label in chain if label == "CW")
            copies.append((post_end, cw_end - 4 * US, 1000, 141, 1, None))  # H2D, outside the graph
    copies.append((0, 1 * US, 64, 141, 2, None))  # a D2H readback: not a copy-engine H2D
    c.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?)", rows)
    c.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?, ?)", copies)
    c.commit()
    c.close()
    out = subprocess.run([sys.executable, str(HERE / "frontend_bound.py"), str(db), "--skip", "0"],
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["steps"] == 2 and r["layers_per_step"] == 2
    assert r["w1_budget_hit_frac"] == 0.5
    # cw_floor = p5 of CW = 10 us. Layer 1: cut = 102 us, no spin. Layer 2: cut 7 us, spin 300 us -> 0.
    assert abs(r["frontend_bound_ms_per_step"] - 0.102) < 1e-9
    # W1 floor = p10 of W1 = 5 us. Layer 1: 95 us, no spin. Layer 2: 0.
    assert abs(r["hit_wait0_bound_ms_per_step"] - 0.095) < 1e-9
    assert r["copy_bytes_per_step"] == 2000 and r["cw_end_after_copy_p50_us"] == 4.0


def test_steps_with_different_layer_counts_are_refused():
    """A step whose layer lacks S or F would shrink the bound's numerator silently and understate the saving."""
    import pytest

    from frontend_bound import check_steps

    whole = split_layers(_chain(0, 50, 30, 20) + _chain(1_000_000, 5, 30, 20))
    torn = split_layers(_chain(0, 50, 30, 20) + [k for k in _chain(1_000_000, 5, 30, 20) if k[2] != "F"])
    check_steps([whole, whole])
    with pytest.raises(ValueError, match="layers per step"):
        check_steps([whole, torn])
    with pytest.raises(ValueError, match="node-mode"):
        check_steps([])
