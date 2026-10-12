"""The residency capture's snapshot and label kernels on a real GPU, against the host reference in
residency_capture_ref (record layout stated there), and the snapshot's time per layer.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import random
import statistics

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.srt.layers.moe.ram_slot_map import LaneKind, type_lanes  # noqa: E402
from sglang.srt.layers.moe.residency_capture_ref import (  # noqa: E402
    HORIZONS,
    KIND_VRAM_HOT,
    MASK_WORDS,
    horizon_targets,
    route_labels,
    snapshot_masks,
)

EXPERTS = 384  # the mask's width: every word of the six is exercised
LANES = 8
SPLIT = [0, 1, 1, 2, 3, 3, 4, 5, 5]
U64 = 0xFFFFFFFFFFFFFFFF

# Rows 18 and 19 have horizons crossing 19->20, which stay valid; 40..42 have horizons past the last row.
LAYER_IDS = [3, 4, 5, 18, 19, 20, 21, 40, 41, 42]


def _random_bank(rng, rows, experts=EXPERTS):
    return [[rng.randrange(LANES) if rng.random() < 0.4 else -1 for _ in range(experts)] for _ in range(rows)]


def _run_snapshot(bank, targets, valid):
    ram_slot = torch.tensor(bank, dtype=torch.int32, device="cuda")
    out_masks = torch.zeros((HORIZONS, MASK_WORDS), dtype=torch.int64, device="cuda")
    out_valid = torch.full((HORIZONS,), 7, dtype=torch.uint8, device="cuda")
    out_stamp = torch.zeros(1, dtype=torch.int64, device="cuda")
    es.run_residency_snapshot(
        ram_slot,
        torch.tensor(targets, dtype=torch.int32, device="cuda"),
        torch.tensor([int(v) for v in valid], dtype=torch.uint8, device="cuda"),
        out_masks,
        out_valid,
        out_stamp,
    )
    torch.cuda.synchronize()
    words = [[w & U64 for w in row] for row in out_masks.cpu().tolist()]
    return words, [bool(v) for v in out_valid.cpu().tolist()], int(out_stamp.item())


def test_snapshot_matches_the_host_reference_for_every_row():
    """Every row of a map with a 19->20 step: the kernel's masks and valid flags are the host's, a 19->20 crossing
    stays valid with its mask, and a horizon past the last row is invalid with an all-zero mask."""
    bank = _random_bank(random.Random(11), len(LAYER_IDS))
    targets, valid = horizon_targets(LAYER_IDS)
    crossing = 0
    past_end = 0
    for row, (t_row, v_row) in enumerate(zip(targets, valid)):
        masks, got_valid, stamp = _run_snapshot(bank, t_row, v_row)
        assert masks == snapshot_masks(bank, targets=t_row), row
        assert got_valid == v_row, row
        assert stamp > 0, row
        crossing += LAYER_IDS[row] <= 19 < LAYER_IDS[row] + HORIZONS and any(v_row)
        past_end += not all(v_row)
        for h, ok in enumerate(v_row):
            if not ok:
                assert masks[h] == [0] * MASK_WORDS, (row, h)
    assert crossing and past_end, "the layer ids lost the 19->20 crossing or the past-the-end rows"


def test_snapshot_stamp_is_nondecreasing_across_launches():
    """The %globaltimer stamp is read on the device at launch: a later launch never carries an earlier stamp."""
    bank = _random_bank(random.Random(3), 2)
    stamps = [_run_snapshot(bank, [1, -1, -1, -1], [True, False, False, False])[2] for _ in range(4)]
    assert stamps == sorted(stamps) and stamps[0] > 0


def _typed_case(rng):
    ram = [rng.randrange(LANES) if rng.random() < 0.3 else -1 for _ in range(EXPERTS)]
    planned = rng.sample(range(EXPERTS), rng.randint(1, LANES))
    for e in planned:
        if ram[e] >= 0:
            ram[e] = rng.randrange(LANES)
    staging = rng.sample(range(8, 14), 6)
    kinds, _ = type_lanes(
        planned, ram, staging, SPLIT, lanes=LANES, captured=True, copy_armed=False, hit_copy="sm",
        cpu_on=False, cpu_misses=False,
    )
    # Routes: some planned, some hot (outside the plan), in a shuffled order.
    hot = rng.sample([e for e in range(EXPERTS) if e not in planned], rng.randint(0, 4))
    routes = planned + hot
    rng.shuffle(routes)
    return routes, planned, [int(k) for k in kinds]


def test_route_labels_match_type_lanes_and_hot_marking():
    """200 random plans: the kernel's per-route kinds are route_labels over type_lanes' kinds, planned routes carry
    their lane's LaneKind and routes outside the plan carry 0."""
    rng = random.Random(5)
    for _ in range(200):
        routes, planned, kinds = _typed_case(rng)
        count = len(planned)
        pad = LANES - count
        planned_dev = torch.tensor(planned + [-1] * pad, dtype=torch.int64, device="cuda")
        lane_kind = torch.tensor(kinds + [0] * pad, dtype=torch.int32, device="cuda")
        out_kinds = torch.full((len(routes),), 255, dtype=torch.uint8, device="cuda")
        out_stamp = torch.zeros(1, dtype=torch.int64, device="cuda")
        es.run_route_labels(
            torch.tensor(routes, dtype=torch.int64, device="cuda"),
            planned_dev,
            torch.tensor([count], dtype=torch.int32, device="cuda"),
            lane_kind,
            out_kinds,
            out_stamp,
        )
        torch.cuda.synchronize()
        want = route_labels(routes=routes, planned=planned, count=count, lane_kind=kinds)
        assert out_kinds.cpu().tolist() == want, (routes, planned, kinds)
        assert int(out_stamp.item()) > 0
        for route, kind in zip(routes, want):
            if route not in planned:
                assert kind == KIND_VRAM_HOT
            else:
                assert kind in {int(k) for k in LaneKind}


def test_snapshot_time_per_layer_past_l2(capsys):
    """Prints the snapshot kernel's time per layer: 43 separate map banks and a 512 MiB flush before each launch keep
    the reads out of L2 (96 MiB), so the figure is not a cache-resident one."""
    rng = random.Random(1)
    layers = 43
    banks = [torch.tensor(_random_bank(rng, layers), dtype=torch.int32, device="cuda") for _ in range(layers)]
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    valid = torch.ones(HORIZONS, dtype=torch.uint8, device="cuda")
    out_masks = torch.zeros((HORIZONS, MASK_WORDS), dtype=torch.int64, device="cuda")
    out_valid = torch.zeros(HORIZONS, dtype=torch.uint8, device="cuda")
    out_stamp = torch.zeros(1, dtype=torch.int64, device="cuda")
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    es.run_residency_snapshot(banks[0], targets, valid, out_masks, out_valid, out_stamp)  # builds and warms
    torch.cuda.synchronize()
    times = []
    for bank in banks * 3:
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        es.run_residency_snapshot(bank, targets, valid, out_masks, out_valid, out_stamp)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    median_us = statistics.median(times) * 1000
    with capsys.disabled():
        print(f"\nresidency snapshot: median {median_us:.1f} us per layer (n={len(times)}, L2 flushed)")
    assert median_us > 0
