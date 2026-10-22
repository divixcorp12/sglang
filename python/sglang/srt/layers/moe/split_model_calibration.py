"""The split model's costs (SGLANG_DSV41_ENABLE_CPU_SPLIT_MODEL), measured at startup and refined while serving.

The model (ram_slot_map.model_cpu_lanes) prices each record's CPU and copy sides from cost words in ns. None of them is a
constant: when the model is on, the RAM-miss service measures every one on the loaded model just before it arms the copy
engine (``calibrate``), publishes the fitted words to the device, and can refine the CPU costs from the jobs it serves
(``SplitModelRefiner``). What is measured, and how each word is fitted:

- **CPU, per node** (``hit_a/b/c``, ``miss_a/c``). The node's own engine runs CPU jobs on its real tier rows: hit jobs
  (output part 0) of k in HIT_EXPERTS lanes, each lane routed by t in TOKENS_PER_EXPERT of the row's tokens, and miss
  jobs (part 1) of one lane with t tokens. Runs rotate over every registered row and the node's slots, so an expert's
  bytes are not in L3 from the run before, as in decode. ``hit = a + b*k + c*pairs`` (pairs = k*t) and ``miss = a +
  c*t`` by least squares. On one-token rows pairs == k, so c is 0 and b carries both.
- **Copy lane** (``copy_lane``). DMAs of m in COPY_LANES experts from each node's slots through a private copy backend
  (the copy engine's cuMemcpyAsync per table entry), alone; the slope through the origin (the model charges copy_lane
  per lane), the mean of the nodes' (both share one FIFO).
- **Landing** (``first_byte``, ``per_expert``). Reads of m in LANDING_ROWS of each node's own experts from NVMe through
  the node's reader (O_DIRECT: no page cache serves them) into its staging slots, timed until the last row lands;
  ``landing(m) = first_byte + per_expert*m`` by least squares, over every node's points. It starts at the read, not at
  the post, so the RAM thread's pickup of a record is not in it. A fully pinned model has no staging slots and no
  misses: nothing is read, and both words are 0 (``assemble(landing=False)``; ``clamp_costs`` leaves them there).
- **Stream coupling** (``stream_lane``, ``stream_base``). The stream kernel's host pulls (``calibration_pull``: S's grid
  and copy on one expert of pinned host rows) after a delay of first_byte, alone and with STREAM_LANES copy-engine
  lanes started at time 0 as a record's FIFO is; ``stream_lane`` is the slope of the pull time's growth through the
  origin, ``stream_base`` S's tail after the last landing: one piece of the pull alone (pull / STAGE_PIECES).

Guardrails: every fitted value is clamped to its RANGES_MS range, and a value outside it by more than the range's width,
or not finite, is nonsense (SplitModelCalibrationError): the service then keeps the model off and the split table types
the lanes, never fixture or offline constants. SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS still names costs for experiments: the
ones it names replace the fitted ones, and when it names every cost the calibration does not run.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Optional, Sequence

from sglang.srt.layers.moe.ram_slot_map import (
    SPLIT_MODEL_KEYS,
    SPLIT_MODEL_NODE_KEYS,
    split_model_overrides,
    split_model_words,
)

logger = logging.getLogger(__name__)

# The startup grid: CPU hit jobs of HIT_EXPERTS lanes x TOKENS_PER_EXPERT tokens a lane (capped at the rows' tokens and
# the node's slots), miss jobs of one lane x TOKENS_PER_EXPERT, DMAs of COPY_LANES experts, reads of LANDING_ROWS rows
# (capped at the staging slots), and the stream pull against STREAM_LANES copy lanes.
HIT_EXPERTS = (1, 2, 4)
TOKENS_PER_EXPERT = (1, 2, 6)
COPY_LANES = (1, 2, 4)
LANDING_ROWS = (1, 2, 4)
STREAM_LANES = (0, 1, 2, 4)
# The sane range of each cost, in ms (ram_slot_map keys). hit_a may be negative: it is a fit's intercept. Wider than any
# value measured on divix01 (2026-10): a fit outside a range by more than its width is a failed measurement.
RANGES_MS = {
    "hit_a": (-0.2, 1.0),
    "hit_b": (0.0, 2.0),
    "hit_c": (0.0, 2.0),
    "miss_a": (0.0, 3.0),
    "miss_c": (0.0, 2.0),
    "copy_lane": (0.1, 5.0),
    "first_byte": (0.0, 10.0),
    "per_expert": (0.1, 10.0),
    "stream_lane": (0.0, 3.0),
    "stream_base": (0.0, 3.0),
}
# The NVMe landing's costs, which a fully pinned model never measures or charges.
LANDING_KEYS = ("first_byte", "per_expert")
# Live refinement: a window needs this many jobs per part, its fit pulls the running estimate toward itself by
# REFINE_WEIGHT, and REFINE_PRIOR_JOBS jobs' worth of ridge toward the running estimate keep the directions a window
# cannot tell apart (served k and pairs move together) where they were.
REFINE_MIN_JOBS = 64
REFINE_WEIGHT = 0.25
REFINE_PRIOR_JOBS = 8
# Scratch buffers of failed calibrations, which a timed-out DMA may still write into.
_KEPT_SCRATCH: list = []


class SplitModelCalibrationError(RuntimeError):
    """The calibration could not run or measured nonsense; the model stays off."""


@dataclasses.dataclass
class SplitModelCosts:
    """The split model's costs in ms: the per-node lists of SPLIT_MODEL_NODE_KEYS, then SPLIT_MODEL_KEYS."""

    hit_a: list
    hit_b: list
    hit_c: list
    miss_a: list
    miss_c: list
    copy_lane: float
    first_byte: float
    per_expert: float
    stream_lane: float
    stream_base: float

    @property
    def nodes(self) -> int:
        return len(self.hit_a)

    def base(self) -> dict:
        """The costs as ram_slot_map.split_model_words' ``base``."""
        return {key: tuple(getattr(self, key)) for key in SPLIT_MODEL_NODE_KEYS} | {
            key: getattr(self, key) for key in SPLIT_MODEL_KEYS
        }

    def words(self, overrides: str = "", residency_tiebreak: bool = False) -> list[int]:
        """The device's cost words (ns) and flags word, the costs ``overrides`` (SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS)
        names replaced."""
        return split_model_words(self.nodes, overrides, base=self.base(), residency_tiebreak=residency_tiebreak)

    def line(self) -> str:
        """One compact log line, ms."""
        node = " ".join(
            f"n{n}[hit {self.hit_a[n]:.3f}+{self.hit_b[n]:.3f}k+{self.hit_c[n]:.3f}p miss {self.miss_a[n]:.3f}"
            f"+{self.miss_c[n]:.3f}t]"
            for n in range(self.nodes)
        )
        return (
            f"{node} copy_lane {self.copy_lane:.3f} landing {self.first_byte:.3f}+{self.per_expert:.3f}i "
            f"stream {self.stream_base:.3f}+{self.stream_lane:.3f}/lane"
        )


def costs_from_words(words: Sequence[int], nodes: int) -> SplitModelCosts:
    """The costs (ms) of device cost words (ns)."""
    ms = [w / 1e6 for w in words]
    per_node = {key: [ms[5 * n + i] for n in range(nodes)] for i, key in enumerate(SPLIT_MODEL_NODE_KEYS)}
    shared = {key: ms[5 * nodes + i] for i, key in enumerate(SPLIT_MODEL_KEYS)}
    return SplitModelCosts(**per_node, **shared)


def apply_overrides(costs: SplitModelCosts, overrides: str) -> SplitModelCosts:
    """``costs`` with the ones SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS names replaced, as ms."""
    return costs_from_words(costs.words(overrides), costs.nodes)


def solve(a: list, b: list) -> Optional[list]:
    """x with a x = b by Gaussian elimination with partial pivoting, or None when a is singular."""
    n = len(b)
    m = [list(map(float, row)) + [float(v)] for row, v in zip(a, b)]
    scale = max((abs(v) for row in a for v in row), default=0.0)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= 1e-12 * max(scale, 1.0):
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def least_squares(xs: Sequence[Sequence[float]], ys: Sequence[float]) -> Optional[list]:
    """Coefficients minimising sum (y - x . coef)^2 (normal equations), or None when the columns are dependent."""
    k = len(xs[0])
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)]
    return solve(ata, aty)


def fit_hit(points: Sequence[tuple]) -> tuple[float, float, float]:
    """(a, b, c) of ``ms = a + b*k + c*pairs`` over (k, pairs, ms) points. When pairs is always k (one-token rows), c is
    0 and b is the slope on k. Raises SplitModelCalibrationError when the points cannot fix the fit."""
    if all(p == k for k, p, _ in points):
        coef = least_squares([(1.0, k) for k, _, _ in points], [ms for _, _, ms in points])
        if coef is None:
            raise SplitModelCalibrationError(f"the CPU hit jobs vary in nothing: {points}")
        return coef[0], coef[1], 0.0
    coef = least_squares([(1.0, k, p) for k, p, _ in points], [ms for _, _, ms in points])
    if coef is None:
        raise SplitModelCalibrationError(f"the CPU hit jobs cannot separate experts from pairs: {points}")
    return coef[0], coef[1], coef[2]


def fit_line(points: Sequence[tuple]) -> tuple[float, float]:
    """(intercept, slope) over (x, y) points."""
    coef = least_squares([(1.0, x) for x, _ in points], [y for _, y in points])
    if coef is None:
        raise SplitModelCalibrationError(f"cannot fit a line through {points}")
    return coef[0], coef[1]


def fit_slope(points: Sequence[tuple]) -> float:
    """The slope through the origin over (x, y) points."""
    xx = sum(x * x for x, _ in points)
    if xx <= 0:
        raise SplitModelCalibrationError(f"cannot fit a slope through {points}")
    return sum(x * y for x, y in points) / xx


def clamp_costs(costs: SplitModelCosts, *, landing: bool = True) -> tuple[SplitModelCosts, list]:
    """``costs`` with each value clamped to RANGES_MS, and the names clamped. Raises SplitModelCalibrationError for a
    value that is not finite or lies outside its range by more than the range's width. ``landing=False`` (nothing was
    read from NVMe: assemble(landing=False)) keeps first_byte and per_expert at their 0, not per_expert's floor."""
    clamped = []

    def one(name: str, value: float) -> float:
        if not landing and name in LANDING_KEYS:
            return value
        lo, hi = RANGES_MS[name.split(".")[0]]
        if not math.isfinite(value) or value < lo - (hi - lo) or value > hi + (hi - lo):
            raise SplitModelCalibrationError(f"{name} = {value} ms is outside any sane range ({lo}..{hi} ms)")
        if lo <= value <= hi:
            return value
        bounded = min(max(value, lo), hi)
        if abs(value - bounded) >= 5e-4:  # a fit's rounding past a bound (-0.0001 ms) is not worth a log entry
            clamped.append(f"{name} {value:.3f}->{bounded:.3f}")
        return bounded

    per_node = {
        key: [one(f"{key}.{n}", v) for n, v in enumerate(getattr(costs, key))] for key in SPLIT_MODEL_NODE_KEYS
    }
    shared = {key: one(key, getattr(costs, key)) for key in SPLIT_MODEL_KEYS}
    return SplitModelCosts(**per_node, **shared), clamped


def grid(tokens: int, slots: int, lanes: int, staging: int) -> dict:
    """The startup cells for a node: rows of ``tokens`` tokens, ``slots`` slots in its smallest range, a ``lanes``-lane
    build and ``staging`` staging slots in its smallest row. {kind: [(a, b)]} with kinds "hit", "miss", "copy",
    "landing"; empty kinds are dropped."""
    width = min(slots, lanes)
    per_lane = sorted({min(t, tokens) for t in TOKENS_PER_EXPERT})
    cells = {
        "hit": [(k, t) for k in HIT_EXPERTS if k <= width for t in per_lane],
        "miss": [(1, t) for t in per_lane] if width >= 1 else [],
        "copy": [(m, 0) for m in COPY_LANES if m <= width],
        "landing": [(m, 0) for m in LANDING_ROWS if m <= min(staging, lanes)],
    }
    return {kind: c for kind, c in cells.items() if c}


def fit_node(cells: dict, ms: dict) -> dict:
    """One node's fits from its cells (``grid``) and their measured ms ({kind: [ms per cell]}): its CPU costs, its copy
    lane slope and its landing points."""
    if "hit" not in cells or "miss" not in cells:
        raise SplitModelCalibrationError("a node has no CPU cells")
    hit_a, hit_b, hit_c = fit_hit([(k, k * t, m) for (k, t), m in zip(cells["hit"], ms["hit"])])
    misses = [(t, m) for (_, t), m in zip(cells["miss"], ms["miss"])]
    miss_a, miss_c = fit_line(misses) if len({t for t, _ in misses}) > 1 else (misses[0][1], 0.0)
    out = {"hit_a": hit_a, "hit_b": hit_b, "hit_c": hit_c, "miss_a": miss_a, "miss_c": miss_c}
    if "copy" in cells:
        out["copy_lane"] = fit_slope([(m, v) for (m, _), v in zip(cells["copy"], ms["copy"])])
    out["landing"] = [(m, v) for (m, _), v in zip(cells.get("landing", []), ms.get("landing", []))]
    return out


def fit_stream(pull_ms: Sequence[float], lanes: Sequence[int], pieces: int) -> tuple[float, float]:
    """(stream_lane, stream_base) from the pull's ms with each of ``lanes`` copy lanes (lanes[0] == 0 alone)."""
    if not lanes or lanes[0] != 0 or len(lanes) < 2:
        raise SplitModelCalibrationError(f"the stream pull needs a run alone and with copy lanes, got lanes {lanes}")
    alone = pull_ms[0]
    lane = fit_slope([(n, t - alone) for n, t in zip(lanes[1:], pull_ms[1:])])
    return max(lane, 0.0), alone / pieces


def assemble(nodes: Sequence[dict], stream: tuple[float, float], *, landing: bool = True) -> SplitModelCosts:
    """The model's costs from each node's fit_node and the stream fit; landing pools every node's points.

    ``landing=False`` (no row stages: the model is fully pinned, so no expert can miss) measures no landing, and
    first_byte and per_expert are 0: no NVMe read is ever charged, and the model reads them only for misses."""
    if landing:
        points = [p for n in nodes for p in n["landing"]]
        if len({m for m, _ in points}) < 2:
            raise SplitModelCalibrationError(f"the landing fit needs reads of two sizes, got {points}")
        first_byte, per_expert = fit_line(points)
    else:
        first_byte, per_expert = 0.0, 0.0
    copies = [n["copy_lane"] for n in nodes if "copy_lane" in n]
    if not copies:
        raise SplitModelCalibrationError("no node timed a copy lane")
    return SplitModelCosts(
        **{key: [n[key] for n in nodes] for key in SPLIT_MODEL_NODE_KEYS},
        copy_lane=sum(copies) / len(copies),
        first_byte=first_byte,
        per_expert=per_expert,
        stream_lane=stream[0],
        stream_base=stream[1],
    )


def moments_delta(now: Sequence[Sequence[int]], before: Sequence[Sequence[int]]) -> list:
    """The fit moments (transport.CPU_FIT_MOMENTS, per part) gained between two snapshots."""
    return [[a - b for a, b in zip(p, q)] for p, q in zip(now, before)]


def refine_fit(moments: Sequence[int], prior: Sequence[float], miss: bool) -> Optional[list]:
    """A window's CPU fit, ms, from its moments (n, k, p, kk, kp, pp, y, ky, py with y in ns), ridged toward ``prior``
    ((a, b, c) for hits, (a, c) for misses) by REFINE_PRIOR_JOBS jobs per coefficient; None under REFINE_MIN_JOBS jobs.
    A miss job has one lane, so its fit is a + c*pairs."""
    n, k, p, kk, kp, pp, y, ky, py = moments
    if n < REFINE_MIN_JOBS:
        return None
    y, ky, py = y / 1e6, ky / 1e6, py / 1e6
    if miss:
        ata, aty = [[n, p], [p, pp]], [y, py]
    else:
        ata, aty = [[n, k, p], [k, kk, kp], [p, kp, pp]], [y, ky, py]
    # Ridge: lam * |coef - prior|^2, lam scaled to each column's mean square so REFINE_PRIOR_JOBS is in jobs.
    for i in range(len(aty)):
        lam = REFINE_PRIOR_JOBS * ata[i][i] / n
        ata[i][i] += lam
        aty[i] += lam * prior[i]
    return solve(ata, aty)


class SplitModelRefiner:
    """The live refinement of the CPU costs (SGLANG_DSV41_CPU_SPLIT_MODEL_REFINE_BATCHES): each ``refine`` fits every
    node's hit and miss jobs since the last call (the engines' fit moments) and moves the running costs REFINE_WEIGHT of
    the way to the fit, clamped to RANGES_MS. The copy lane, landing and stream costs keep their startup values: the
    production host reads no clock on those paths. Costs SGLANG_DSV41_CPU_SPLIT_MODEL_COSTS names stay as named."""

    def __init__(self, costs: SplitModelCosts, moments: Sequence, overrides: str = ""):
        self.costs = costs
        self.before = [list(map(list, m)) for m in moments]
        self.fixed = set(split_model_overrides(overrides, costs.nodes)[0])  # (key, node) pairs the env names
        self.refines = 0

    def refine(self, moments: Sequence) -> Optional[SplitModelCosts]:
        """The new costs from per-node moment snapshots, or None when no node had a full window."""
        changed = False
        costs = dataclasses.replace(self.costs, **{key: list(getattr(self.costs, key)) for key in SPLIT_MODEL_NODE_KEYS})
        for node, (now, before) in enumerate(zip(moments, self.before)):
            delta = moments_delta(now, before)
            for part, keys in ((0, ("hit_a", "hit_b", "hit_c")), (1, ("miss_a", "miss_c"))):
                prior = [getattr(costs, key)[node] for key in keys]
                fit = refine_fit(delta[part], prior, miss=part == 1)
                if fit is None:
                    continue
                self.before[node][part] = list(now[part])
                for key, old, new in zip(keys, prior, fit):
                    if (key, node) not in self.fixed and math.isfinite(new):
                        lo, hi = RANGES_MS[key]
                        getattr(costs, key)[node] = min(max(old + REFINE_WEIGHT * (new - old), lo), hi)
                        changed = True
        if not changed:
            return None
        self.costs = costs
        self.refines += 1
        return costs


def staging_slots(host, group: int, row: int, width: int) -> int:
    """The staging slots group ``group`` holds in ``row`` (RamTier::reserve_staging: min(width, its slots - 1))."""
    lo, hi = host.node_ranges[group][row]
    return min(width, hi - lo - 1)


def pull_table(slabs: dict, names: Sequence[str], slot: int, scratch, device):
    """The calibration pull's table for one expert: each slab name's row of ``slot`` into ``scratch`` at 16-byte
    aligned offsets, int64 [names, 3] on ``device``."""
    import torch

    rows, offset = [], 0
    for name in names:
        row = slabs[name][slot]
        size = row.numel() * row.element_size()
        rows.append((row.data_ptr(), scratch.data_ptr() + offset, size))
        offset += (size + 15) // 16 * 16
    if offset > scratch.numel():
        raise SplitModelCalibrationError(f"the pull scratch holds {scratch.numel()} bytes, one expert needs {offset}")
    return torch.tensor(rows, dtype=torch.int64, device=device)


def expert_bytes(slabs: dict, names: Sequence[str]) -> int:
    """One expert's bytes over every slab name, each rounded up to 16 as pull_table lays them out."""
    return sum((slabs[n][0].numel() * slabs[n][0].element_size() + 15) // 16 * 16 for n in names)


def measure_stream(device_side, slabs: dict, names: Sequence[str], slots: tuple, *, device: int, reps: int,
                   delay_ms: float) -> list[float]:
    """The calibration pull's ms of one expert (slot slots[0]) started ``delay_ms`` after copy-engine copies of n experts
    (slots[1..n]) for each n in STREAM_LANES, mean of ``reps`` after a warm-up. CUDA events time the pull alone."""
    import torch

    dev = torch.device("cuda", device)
    per = expert_bytes(slabs, names)
    pull_dst = torch.empty(per, dtype=torch.uint8, device=dev)
    copy_dst = torch.empty(max(STREAM_LANES) * per, dtype=torch.uint8, device=dev)
    table = pull_table(slabs, names, slots[0], pull_dst, dev)
    pull_stream, copy_stream = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
    # torch.cuda._sleep spins in clock cycles: measure the cycles a ms takes once.
    probe = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    probe[0].record()
    torch.cuda._sleep(1_000_000)
    probe[1].record()
    torch.cuda.synchronize(dev)
    cycles_per_ms = 1_000_000 / max(probe[0].elapsed_time(probe[1]), 1e-3)

    def run(lanes: int) -> float:
        go = torch.cuda.Event()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(int(cycles_per_ms))  # every launch below is queued before the streams start
        go.record()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(go)
            for j in range(lanes):
                offset = j * per
                for name in names:
                    src = slabs[name][slots[1 + j]]
                    size = src.numel() * src.element_size()
                    copy_dst[offset : offset + size].copy_(src.reshape(-1).view(torch.uint8), non_blocking=True)
                    offset += (size + 15) // 16 * 16
        with torch.cuda.stream(pull_stream):
            pull_stream.wait_event(go)
            if delay_ms > 0:
                torch.cuda._sleep(int(delay_ms * cycles_per_ms))
            start.record()
            device_side.calibration_pull(table)
            end.record()
        torch.cuda.synchronize(dev)
        return start.elapsed_time(end)

    out = []
    for lanes in STREAM_LANES:
        run(lanes)
        out.append(sum(run(lanes) for _ in range(reps)) / reps)
    return out


def calibrate(host, groups, device_side, *, device: int, reps: int, staging_width: int,
              timeout_s: float = 1.0) -> tuple[SplitModelCosts, list, float]:
    """Measure and fit every cost on the loaded model (the module doc): (clamped costs, names clamped, seconds taken).

    ``groups`` is the CPU expert services, one per node (CpuExpertGroups.services). The caller owns the tier (the RAM
    thread paused, the device drained) and has not armed the copy engine. Raises SplitModelCalibrationError, or the
    native RuntimeError, when it cannot measure."""
    import torch

    from sglang.kernels.ops.moe.expert_stream_transport import (
        SPLIT_CELL_COPY,
        SPLIT_CELL_HIT,
        SPLIT_CELL_LANDING,
        SPLIT_CELL_MISS,
        STAGE_PIECES,
    )

    began = time.monotonic()
    nodes = host.wire.nodes
    if len(groups) != nodes:
        raise SplitModelCalibrationError(f"the model prices {nodes} nodes, CPU experts run {len(groups)} groups")
    kinds = {"hit": SPLIT_CELL_HIT, "miss": SPLIT_CELL_MISS, "copy": SPLIT_CELL_COPY, "landing": SPLIT_CELL_LANDING}
    fits, scratch = [], None
    for service in groups:
        rows = sorted(service.layers)
        if not rows:
            raise SplitModelCalibrationError(f"group {service.group} has no registered CPU row")
        slots = min(service._group_slots(r) for r in rows)
        staging = min(staging_slots(host, service.group, r, staging_width) for r in rows)
        cells = grid(service.tokens, slots, host.wire.lanes, staging)
        # The copy cells read rows[0]'s copy table, the same expert size on every row.
        need = max((m for m, _ in cells.get("copy", [])), default=0) * host.copy_expert_bytes(rows[0])
        if need and (scratch is None or scratch.numel() < need):
            scratch = torch.empty(need, dtype=torch.uint8, device="cpu" if device < 0 else torch.device("cuda", device))
        flat = [(kinds[kind], a, b) for kind, c in cells.items() for a, b in c]
        try:
            ms = host.calibrate_split_model(
                rows, flat, device=device, reps=reps, scratch=scratch, timeout_s=timeout_s, group=service.group
            )
        except RuntimeError:
            # A timed-out DMA may still write into the scratch: keep its block out of the allocator for good.
            _KEPT_SCRATCH.append(scratch)
            raise
        by_kind, i = {}, 0
        for kind, c in cells.items():
            by_kind[kind] = ms[i : i + len(c)]
            i += len(c)
        logger.info("split model calibration group %d cells %s ms %s", service.group, cells, by_kind)
        fits.append(fit_node(cells, by_kind))
    landing = [p for f in fits for p in f["landing"]]
    first_byte = fit_line(landing)[0] if len({m for m, _ in landing}) > 1 else 0.0
    if device >= 0:
        service = groups[0]
        row = sorted(service.layers)[0]
        lo, hi = host.node_ranges[service.group][row]
        if hi - lo < 1 + max(STREAM_LANES):
            raise SplitModelCalibrationError(f"the stream pull needs {1 + max(STREAM_LANES)} slots, row {row} has {hi - lo}")
        pull = measure_stream(
            device_side, service.slabs_by_row[row], service.trait.slab_names, tuple(range(lo, lo + 1 + max(STREAM_LANES))),
            device=device, reps=reps, delay_ms=max(first_byte, 0.0),
        )
        logger.info("split model calibration stream pull ms by copy lanes %s: %s", list(STREAM_LANES), pull)
        stream = fit_stream(pull, STREAM_LANES, STAGE_PIECES)
    else:
        stream = (0.0, 0.0)  # the test backend has no GPU to pull with: no stream side
    # staging_width 0: the model is fully pinned, no row stages, so no landing cell ran and no miss will land.
    landing = staging_width > 0
    costs, clamped = clamp_costs(assemble(fits, stream, landing=landing), landing=landing)
    return costs, clamped, time.monotonic() - began
