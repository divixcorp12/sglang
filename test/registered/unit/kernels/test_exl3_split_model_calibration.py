"""The split model's startup calibration and live refinement (split_model_calibration.py; CPU).

The fits recover known costs from synthetic timings; the guardrails clamp and refuse; the service publishes calibrated
words, leaves the model off when the calibration fails and skips it for a complete override; the refinement moves the
CPU costs toward the served jobs. The native half (RamTier::calibrate_split_model, the engine's fit moments) runs on the
instr build's fake CPU kernel and the test copy backend. The GPU half is test/manual/dsv41/test_split_model_calibration_cuda.py.
"""

import dataclasses
import logging
import math
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import (
    CPU_FIT_MOMENTS,
    SPLIT_CELL_COPY,
    SPLIT_CELL_HIT,
    SPLIT_CELL_LANDING,
    SPLIT_CELL_MISS,
    new_page,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe import split_model_calibration as calib
from sglang.srt.layers.moe.ram_slot_map import (
    SPLIT_MODEL_FIXTURE_MS,
    split_model_off_words,
    split_model_overrides_complete,
    split_model_words,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, fake_cpu_layer, ram_miss_setup

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Known costs, ms: two nodes with different CPU teams.
TRUE = calib.SplitModelCosts(
    hit_a=[0.02, 0.05], hit_b=[0.15, 0.18], hit_c=[0.40, 0.35], miss_a=[0.33, 0.29], miss_c=[0.19, 0.25],
    copy_lane=1.05, first_byte=0.8, per_expert=1.6, stream_lane=0.65, stream_base=0.1,
)
KINDS = {"hit": SPLIT_CELL_HIT, "miss": SPLIT_CELL_MISS, "copy": SPLIT_CELL_COPY, "landing": SPLIT_CELL_LANDING}


def _cell_ms(costs, node, kind, a, b):
    """What a cell would measure under `costs`."""
    if kind == SPLIT_CELL_HIT:
        return costs.hit_a[node] + costs.hit_b[node] * a + costs.hit_c[node] * a * b
    if kind == SPLIT_CELL_MISS:
        return costs.miss_a[node] + costs.miss_c[node] * b
    if kind == SPLIT_CELL_COPY:
        return costs.copy_lane * a
    return costs.first_byte + costs.per_expert * a


def _close(got, want, tol=1e-6):
    return all(math.isclose(g, w, abs_tol=tol) for g, w in zip(got, want))


def test_fit_hit_recovers_experts_and_pairs_from_the_grid():
    cells = calib.grid(tokens=6, slots=12, lanes=8, staging=4)
    assert cells["hit"] == [(k, t) for k in (1, 2, 4) for t in (1, 2, 6)]
    points = [(k, k * t, 0.02 + 0.15 * k + 0.4 * k * t) for k, t in cells["hit"]]
    assert _close(calib.fit_hit(points), (0.02, 0.15, 0.4))


def test_one_token_rows_fold_pairs_into_experts():
    cells = calib.grid(tokens=1, slots=12, lanes=8, staging=4)
    assert cells["hit"] == [(1, 1), (2, 1), (4, 1)] and cells["miss"] == [(1, 1)]
    a, b, c = calib.fit_hit([(k, k, 0.1 + 0.5 * k) for k, _ in cells["hit"]])
    assert _close((a, b, c), (0.1, 0.5, 0.0))


def test_the_grid_is_capped_by_slots_lanes_tokens_and_staging():
    cells = calib.grid(tokens=3, slots=2, lanes=8, staging=1)
    assert cells["hit"] == [(k, t) for k in (1, 2) for t in (1, 2, 3)]
    assert cells["copy"] == [(1, 0), (2, 0)] and cells["landing"] == [(1, 0)]
    assert "landing" not in calib.grid(tokens=3, slots=8, lanes=8, staging=0)


def test_fit_node_and_assemble_recover_every_cost():
    """Synthetic cell timings for two nodes recover each node's CPU costs, the copy lane and the landing line; the
    stream fit recovers stream_lane and its base. Mutation: assemble fitting landing per node -- still green; fitting
    copy_lane with an intercept -- red."""
    fits = []
    for node in range(2):
        cells = calib.grid(tokens=6, slots=12, lanes=8, staging=4)
        ms = {kind: [_cell_ms(TRUE, node, KINDS[kind], a, b) for a, b in c] for kind, c in cells.items()}
        fits.append(calib.fit_node(cells, ms))
    pieces = 8
    alone = TRUE.stream_base * pieces
    pull = [alone + TRUE.stream_lane * n for n in calib.STREAM_LANES]
    stream = calib.fit_stream(pull, calib.STREAM_LANES, pieces)
    got = calib.assemble(fits, stream)
    for key in ("hit_a", "hit_b", "hit_c", "miss_a", "miss_c"):
        assert _close(getattr(got, key), getattr(TRUE, key)), key
    for key in ("copy_lane", "first_byte", "per_expert", "stream_lane", "stream_base"):
        assert math.isclose(getattr(got, key), getattr(TRUE, key), abs_tol=1e-6), key


def test_assemble_needs_two_read_sizes_and_a_copy_lane():
    one = {"hit_a": 0, "hit_b": 0, "hit_c": 0, "miss_a": 0, "miss_c": 0, "copy_lane": 1.0, "landing": [(1, 2.0)]}
    with pytest.raises(calib.SplitModelCalibrationError, match="two sizes"):
        calib.assemble([one], (0.0, 0.0))
    two = {**one, "landing": [(1, 2.0), (2, 3.0)]}
    del two["copy_lane"]
    with pytest.raises(calib.SplitModelCalibrationError, match="copy lane"):
        calib.assemble([two], (0.0, 0.0))


def test_clamping_keeps_sane_values_clamps_near_ones_and_refuses_nonsense():
    costs, clamped = calib.clamp_costs(TRUE)
    assert costs == TRUE and clamped == []
    near = dataclasses.replace(TRUE, hit_b=[0.15, -0.5], copy_lane=6.0)
    costs, clamped = calib.clamp_costs(near)
    assert costs.hit_b == [0.15, 0.0] and costs.copy_lane == 5.0
    assert clamped == ["hit_b.1 -0.500->0.000", "copy_lane 6.000->5.000"]
    for bad in (dataclasses.replace(TRUE, per_expert=float("nan")), dataclasses.replace(TRUE, copy_lane=50.0),
                dataclasses.replace(TRUE, miss_c=[0.2, -3.0])):
        with pytest.raises(calib.SplitModelCalibrationError, match="sane range"):
            calib.clamp_costs(bad)


def test_words_overrides_and_the_line():
    words = TRUE.words()
    assert words == split_model_words(2, base=TRUE.base())
    assert words[:5] == [20_000, 150_000, 400_000, 330_000, 190_000] and words[10:] == [
        1_050_000, 800_000, 1_600_000, 650_000, 100_000, 0]  # the flags word last
    assert TRUE.words(residency_tiebreak=True)[-1] == 1
    assert calib.costs_from_words(words, 2) == TRUE
    over = calib.apply_overrides(TRUE, "hit_c.1=0.5,copy_lane=1.3")
    assert over.hit_c == [0.4, 0.5] and over.copy_lane == 1.3 and over.hit_b == TRUE.hit_b
    line = TRUE.line()
    assert "n1[hit 0.050+0.180k+0.350p miss 0.290+0.250t]" in line and "copy_lane 1.050" in line and "\n" not in line


def test_an_override_is_complete_only_when_it_names_every_cost_of_every_node():
    every = ",".join(f"{k}=1" for k in SPLIT_MODEL_FIXTURE_MS)
    assert split_model_overrides_complete(every, 2)
    assert not split_model_overrides_complete("copy_lane=1", 2)
    per_node0 = ",".join(f"{k}.0=1" for k in ("hit_a", "hit_b", "hit_c", "miss_a", "miss_c"))
    shared = ",".join(f"{k}=1" for k in ("copy_lane", "first_byte", "per_expert", "stream_lane", "stream_base"))
    assert not split_model_overrides_complete(per_node0 + "," + shared, 2)
    assert split_model_overrides_complete(per_node0 + "," + shared, 1)


def _moments(jobs):
    """Fit moments (CPU_FIT_MOMENTS) of (k, pairs, ms) jobs, y in ns."""
    m = [0] * len(CPU_FIT_MOMENTS)
    for k, p, ms in jobs:
        y = int(round(ms * 1e6))
        for i, v in enumerate((1, k, p, k * k, k * p, p * p, y, k * y, p * y)):
            m[i] += v
    return m


def test_refine_fit_recovers_a_window_and_needs_enough_jobs():
    jobs = [(k, k * t, 0.03 + 0.2 * k + 0.3 * k * t) for k in (1, 2, 3, 5) for t in (1, 2, 3)] * 8
    fit = calib.refine_fit(_moments(jobs), [0.03, 0.2, 0.3], miss=False)
    assert _close(fit, (0.03, 0.2, 0.3), tol=1e-3)
    assert calib.refine_fit(_moments(jobs[: calib.REFINE_MIN_JOBS - 1]), [0, 0, 0], miss=False) is None
    misses = [(1, t, 0.3 + 0.2 * t) for t in (1, 2, 4, 6)] * 20
    assert _close(calib.refine_fit(_moments(misses), [0.3, 0.2], miss=True), (0.3, 0.2), tol=1e-3)


def test_refine_fit_keeps_the_direction_a_window_cannot_see():
    """Served jobs where pairs is always 1.25 k cannot separate b from c: the ridge keeps their split near the prior
    while matching the window's total cost per expert."""
    prior = [0.0, 0.2, 0.4]
    jobs = [(k, 1.25 * k, 0.2 * k + 0.4 * 1.25 * k + 0.1 * k) for k in (2, 4, 6, 8)] * 20  # 0.1 ms/expert slower
    a, b, c = calib.refine_fit(_moments([(k, int(p), ms) for k, p, ms in jobs]), prior, miss=False)
    assert all(math.isfinite(v) for v in (a, b, c))
    assert abs(b - prior[1]) < 0.15 and abs(c - prior[2]) < 0.15


def test_the_refiner_moves_part_way_respects_overrides_and_waits_for_a_window():
    start = [[[0] * 9, [0] * 9] for _ in range(2)]
    refiner = calib.SplitModelRefiner(TRUE, start, overrides="hit_b.1=0.18")
    assert refiner.refine(start) is None
    served = dataclasses.replace(TRUE, hit_a=[0.12, 0.15], hit_b=[0.25, 0.28], hit_c=[0.40, 0.35])
    now = []
    for node in range(2):
        hits = [(k, k * t, _cell_ms(served, node, SPLIT_CELL_HIT, k, t)) for k in (1, 2, 4) for t in (1, 2, 6)] * 10
        now.append([_moments(hits), [0] * 9])
    got = refiner.refine(now)
    assert got is not None and refiner.refines == 1
    # REFINE_WEIGHT of the way to the window's fit, which the ridge holds a little short of the served 0.25.
    step = TRUE.hit_b[0] + calib.REFINE_WEIGHT * (0.25 - TRUE.hit_b[0])
    assert TRUE.hit_b[0] < got.hit_b[0] <= step and math.isclose(got.hit_b[0], step, abs_tol=0.01)
    assert got.hit_b[1] == TRUE.hit_b[1]  # named by the override
    assert got.miss_a == TRUE.miss_a and got.copy_lane == TRUE.copy_lane  # no miss window; not refined
    assert refiner.refine(now) is None  # nothing new since


class _FakeHost:
    """calibrate()'s view of an ExpertStreamHost: cells answered from `costs`, every call recorded."""

    def __init__(self, costs, nodes=2, fail=None):
        self.wire = SimpleNamespace(nodes=nodes, lanes=8)
        self.node_ranges = [[(10 * g, 10 * g + 10)] * 3 for g in range(nodes)]
        self.costs, self.fail, self.calls = costs, fail, []

    def copy_expert_bytes(self, row):
        return 64

    def calibrate_split_model(self, rows, cells, *, device, reps, scratch, timeout_s, group):
        self.calls.append((group, list(rows), list(cells), device, reps, scratch.numel() if scratch is not None else 0))
        if self.fail:
            raise RuntimeError(self.fail)
        return [_cell_ms(self.costs, group, kind, a, b) for kind, a, b in cells]


def _services(nodes=2, tokens=6, rows=(0, 2)):
    return [SimpleNamespace(group=g, layers={r: None for r in rows}, tokens=tokens, _group_slots=lambda r: 10)
            for g in range(nodes)]


def test_calibrate_runs_every_nodes_cells_and_fits_them():
    host = _FakeHost(TRUE)
    costs, clamped, seconds = calib.calibrate(host, _services(), None, device=-1, reps=3, staging_width=4)
    assert clamped == [] and seconds >= 0
    assert [c[0] for c in host.calls] == [0, 1] and all(c[1] == [0, 2] and c[3] == -1 and c[4] == 3 for c in host.calls)
    assert host.calls[0][5] == 4 * 64  # scratch for the widest copy cell
    for key in ("hit_a", "hit_b", "hit_c", "miss_a", "miss_c"):
        assert _close(getattr(costs, key), getattr(TRUE, key)), key
    assert math.isclose(costs.copy_lane, TRUE.copy_lane) and math.isclose(costs.per_expert, TRUE.per_expert)
    assert (costs.stream_lane, costs.stream_base) == (0.0, 0.0)  # no GPU: no stream side


def test_a_fully_pinned_model_calibrates_without_landing_reads():
    """staging_width 0 (model_fully_pinned): no row stages, so no landing cell runs and the model, which reads
    first_byte and per_expert only for misses, takes their lower bounds instead of failing the fit. Mutation: assemble
    still requires landing points (the split model would stay off on a fully pinned model)."""
    host = _FakeHost(TRUE)
    costs, clamped, _ = calib.calibrate(host, _services(), None, device=-1, reps=3, staging_width=0)
    assert all(kind != SPLIT_CELL_LANDING for c in host.calls for kind, _, _ in c[2])
    assert (costs.first_byte, costs.per_expert) == (calib.RANGES_MS["first_byte"][0], calib.RANGES_MS["per_expert"][0])
    assert clamped == []
    for key in ("hit_a", "hit_b", "hit_c", "miss_a", "miss_c"):
        assert _close(getattr(costs, key), getattr(TRUE, key)), key
    assert math.isclose(costs.copy_lane, TRUE.copy_lane)
    with pytest.raises(calib.SplitModelCalibrationError, match="two sizes"):
        calib.assemble([{"landing": []}], (0.0, 0.0))


def test_calibrate_refuses_a_group_count_the_model_does_not_price_and_unregistered_groups():
    with pytest.raises(calib.SplitModelCalibrationError, match="2 nodes"):
        calib.calibrate(_FakeHost(TRUE), _services(nodes=1), None, device=-1, reps=1, staging_width=4)
    with pytest.raises(calib.SplitModelCalibrationError, match="no registered"):
        calib.calibrate(_FakeHost(TRUE), _services(rows=()), None, device=-1, reps=1, staging_width=4)


def test_a_failed_native_calibration_keeps_its_scratch():
    kept = len(calib._KEPT_SCRATCH)
    with pytest.raises(RuntimeError, match="timed out"):
        calib.calibrate(_FakeHost(TRUE, fail="timed out"), _services(), None, device=-1, reps=1, staging_width=4)
    assert len(calib._KEPT_SCRATCH) == kept + 1


def _service_stub(nodes=2, fail=None):
    """An Exl3RamMissService stand-in for _calibrate_split_model and _refine_split_model."""
    from sglang.srt.layers.moe.exl3_ram_miss import Exl3RamMissService

    published = []
    stats = {"jobs": 0, "lanes": 0, "forward_ns": 0, "staged_jobs": 0, "stage_two_waits": 0, "stage_two_wait_ns": 0}
    services = [SimpleNamespace(group=g, excluded=[], exclude_calibration_stats=None) for g in range(nodes)]
    for s in services:
        s.exclude_calibration_stats = s.excluded.append
    stub = SimpleNamespace(
        wire=SimpleNamespace(nodes=nodes),
        device_side=SimpleNamespace(split_model=torch.zeros(7 * nodes + 6, dtype=torch.int64),
                                    set_split_model_costs=published.append),
        cpu_experts=SimpleNamespace(services=services),
        host=SimpleNamespace(cpu_stats=lambda g: dict(stats), cpu_fit_moments=lambda g: [[0] * 9, [0] * 9]),
        staging_width=lambda: 4,
        _split_model_refiner=None,
    )
    stub._split_model_moments = lambda: Exl3RamMissService._split_model_moments(stub)
    stub._publish_split_model = lambda costs: Exl3RamMissService._publish_split_model(stub, costs)
    return stub, published, Exl3RamMissService


def test_the_service_publishes_calibrated_words_and_starts_a_refiner(monkeypatch):
    stub, published, cls = _service_stub()
    monkeypatch.setattr(calib, "calibrate", lambda *a, **k: (TRUE, ["hit_b.1 2.5->2.0"], 0.5))
    with envs.SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS.override("copy_lane=1.2"):
        cls._calibrate_split_model(stub)
    assert published == [calib.apply_overrides(TRUE, "copy_lane=1.2").words()]
    assert stub._split_model_refiner is not None and stub._split_model_refiner.costs.copy_lane == 1.2
    assert all(len(s.excluded) == 1 for s in stub.cpu_experts.services)


def test_a_failed_calibration_leaves_the_model_off_with_a_warning(monkeypatch, caplog):
    stub, published, cls = _service_stub()

    def fail(*a, **k):
        raise calib.SplitModelCalibrationError("per_expert = nan ms is outside any sane range")

    monkeypatch.setattr(calib, "calibrate", fail)
    with caplog.at_level(logging.WARNING):
        cls._calibrate_split_model(stub)
    assert published == [] and stub._split_model_refiner is None
    assert "the model stays off" in caplog.text
    assert all(len(s.excluded) == 1 for s in stub.cpu_experts.services)


def test_a_complete_override_skips_the_calibration(monkeypatch):
    stub, published, cls = _service_stub()
    monkeypatch.setattr(calib, "calibrate", lambda *a, **k: pytest.fail("calibrated despite a complete override"))
    every = ",".join(f"{k}=0.5" for k in SPLIT_MODEL_FIXTURE_MS)
    with envs.SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS.override(every):
        cls._calibrate_split_model(stub)
    assert published == [split_model_words(2, every)]


def test_the_model_off_skips_the_calibration(monkeypatch):
    stub, published, cls = _service_stub()
    stub.device_side.split_model = torch.empty(0, dtype=torch.int64)
    monkeypatch.setattr(calib, "calibrate", lambda *a, **k: pytest.fail("calibrated with the model off"))
    cls._calibrate_split_model(stub)
    assert published == []


def test_the_service_refines_and_publishes_and_a_failure_stops_the_refinement():
    stub, published, cls = _service_stub()
    stub._split_model_refiner = SimpleNamespace(refine=lambda m: TRUE, refines=1)
    cls._refine_split_model(stub)
    assert published == [TRUE.words()]
    with envs.SGLANG_DSV41_CPU_SPLIT_MODEL_RESIDENCY_TIEBREAK.override(True):
        cls._refine_split_model(stub)  # the flags word carries the tiebreak through every publish
    assert published[-1] == TRUE.words(residency_tiebreak=True) and published[-1][-1] == 1
    del published[1:]

    def broken(m):
        raise ValueError("bad window")

    stub._split_model_refiner = SimpleNamespace(refine=broken, refines=0)
    cls._refine_split_model(stub)
    assert stub._split_model_refiner is None and len(published) == 1


def test_refinement_is_off_by_default():
    assert envs.SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES.get() == 0


# The native half: RamTier::calibrate_split_model and the engine's fit moments on the fake kernel (k * FORWARD_NS a
# job, whatever its tokens) and the test copy backend.
ROW, ROWS, DST_ROWS, HIDDEN, TOKENS = 1, 2, 6, 8, 6
FORWARD_NS = 200_000


def _host(tmp_path, *, capacity=12, register=True):
    s = ram_miss_setup(tmp_path, capacity=capacity, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    host = attached_host(s, new_page(pin=False, wire=lease.wire_layout(8)), k=3)
    host.enable_copy_engine(-1)
    dst = {n: torch.zeros((DST_ROWS,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[ROW].items()}
    table = torch.tensor(
        [[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[ROW].items()],
        dtype=torch.int64,
    )
    host.set_copy_table(ROW, table, DST_ROWS)
    x_rows = torch.zeros((ROWS, lease.cpu_row_bytes(HIDDEN, TOKENS, host.wire.lanes)), dtype=torch.uint8)
    out_rows = torch.zeros((ROWS, 2, TOKENS, HIDDEN), dtype=torch.float32)
    cores = sorted(os.sched_getaffinity(0))[:2]
    host.enable_cpu_experts(host.test_kernel_address(FORWARD_NS), [0] * 9, cores, x_rows, out_rows, threads=2)
    if register:
        host.set_cpu_layer(ROW, fake_cpu_layer(HIDDEN))
    return s, host, (dst, x_rows, out_rows)


CELLS = [(SPLIT_CELL_HIT, 1, 1), (SPLIT_CELL_HIT, 2, 6), (SPLIT_CELL_HIT, 4, 2), (SPLIT_CELL_MISS, 1, 3),
         (SPLIT_CELL_COPY, 2, 0), (SPLIT_CELL_LANDING, 1, 0), (SPLIT_CELL_LANDING, 3, 0)]


def test_the_native_cells_time_cpu_jobs_dma_and_reads_and_feed_the_fit_moments(tmp_path):
    """Every cell runs reps + 1 times: CPU cells as jobs of k lanes each routed by b tokens (the fit moments count k *
    b pairs a job, hits in part 0 and misses in part 1), copy cells through the backend, landing cells as reads of the
    row's own experts into its staging slots. Mutations: the per-lane routing ignored (pairs == k * 1.25) or a miss
    job sent to part 0 -- red."""
    _, host, _keep = _host(tmp_path)
    reps = 2
    before = host.cpu_fit_moments()
    jobs = host.cpu_stats()["jobs"]
    scratch = torch.zeros(2 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    ms = host.calibrate_split_model([ROW], CELLS, device=-1, reps=reps, scratch=scratch, timeout_s=5.0)
    assert len(ms) == len(CELLS) and all(v > 0 for v in ms), ms
    for (kind, a, _), v in zip(CELLS, ms):
        if kind in (SPLIT_CELL_HIT, SPLIT_CELL_MISS):
            assert v >= a * FORWARD_NS / 1e6, (kind, a, v)
    assert host.cpu_stats()["jobs"] - jobs == 4 * (reps + 1)
    after = host.cpu_fit_moments()
    hit = [x - y for x, y in zip(after[0], before[0])]
    miss = [x - y for x, y in zip(after[1], before[1])]
    names = dict(zip(CPU_FIT_MOMENTS, hit))
    assert names["n"] == 3 * (reps + 1) and names["k"] == 7 * (reps + 1)
    assert names["p"] == (1 * 1 + 2 * 6 + 4 * 2) * (reps + 1)
    assert names["y"] >= names["k"] * FORWARD_NS
    assert dict(zip(CPU_FIT_MOMENTS, miss))["p"] == 3 * (reps + 1) and miss[0] == reps + 1


@pytest.mark.parametrize("cells, match", [
    ([(9, 1, 1)], "kind"),
    ([(SPLIT_CELL_HIT, 9, 1)], "lanes"),
    ([(SPLIT_CELL_HIT, 1, TOKENS + 1)], "tokens"),
    ([(SPLIT_CELL_LANDING, 4, 0)], "staging"),
    ([(SPLIT_CELL_COPY, 4, 0)], "scratch"),
])
def test_the_native_calibration_refuses_cells_it_cannot_run(tmp_path, cells, match):
    _, host, _keep = _host(tmp_path)
    scratch = torch.zeros(2 * host.copy_expert_bytes(ROW), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match=match):
        host.calibrate_split_model([ROW], cells, device=-1, reps=1, scratch=scratch)


def test_the_native_calibration_refuses_an_unregistered_row(tmp_path):
    _, host, _keep = _host(tmp_path, register=False)
    with pytest.raises(RuntimeError, match="no registered CPU layer"):
        host.calibrate_split_model([ROW], [(SPLIT_CELL_HIT, 1, 1)], device=-1, reps=1, scratch=None)


def test_the_fit_moments_start_at_zero_and_count_the_calibrations_jobs(tmp_path):
    """The moments start at 0 and only grow; the calibration's own jobs show in them, which is why the service takes
    its refinement baseline after calibrating."""
    _, host, _keep = _host(tmp_path)
    assert host.cpu_fit_moments() == [[0] * 9, [0] * 9]
    host.calibrate_split_model([ROW], [(SPLIT_CELL_HIT, 2, 1)], device=-1, reps=1, scratch=None)
    first = host.cpu_fit_moments()
    assert first[0][0] == 2 and first[0][1] == 4 and first[1] == [0] * 9
