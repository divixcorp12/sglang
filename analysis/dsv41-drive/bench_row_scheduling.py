"""Drive-scheduling arms for batched EXL3 row reads, on one fixed request replay.

``bench_mirror_rows.py`` times one row per call. That cannot say anything about
whole-row assignment, which only differs from splitting when a batch holds
several rows, so its 2.31x does not transfer to a batch. This harness issues
every arm the *same* rows in the *same* batches and submits each batch as one
concurrent ``UringFileReader.read``, the way ``Exl3RowReader.read_split`` does.

Arms (each turns a batch of rows into extents over the mirror roots):

  within_row  -- every row split across all roots by ``StaticSplitPolicy`` (1:1)
  whole_row   -- every row read whole from one root, chosen per batch by least
                 outstanding bytes, ties by bytes served so far (stateful)
  one_root    -- every row from one root (weights e_i)
  weighted    -- like within_row, with weights measured from one-root calibration

Three quantities are kept apart everywhere, because conflating them is how the
earlier analysis went wrong:

  application rows  -- rows the caller asked for
  extents           -- reads submitted; a split row is one extent per non-empty part
  per-drive bytes   -- what each root was asked for in one batch (its outstanding
                       bytes at submit, while the batch fits the ring)

Equal useful bytes across arms is checked, not assumed: every arm must request
the same rows, the same useful bytes (``record.nbytes``) and the same aligned
bytes, or the run stops. Data bytes are checked against the first arm, and the
first ``--verify-rows`` rows against a plain read of the source checkpoint.

The reader does not expose per-extent completion times, so per-drive *service*
time is not measured; only per-drive bytes and the batch's wall time are.

CPU only. Run under ``taskset -c 0-63`` with ``CUDA_VISIBLE_DEVICES=`` set. It
prints the bytes it will move first and refuses to exceed ``--max-gib``.

    python bench_row_scheduling.py --dry-run          # plan and byte volume, no I/O
    python bench_row_scheduling.py --rows-per-size 24 --reps 2
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_mirror_rows as base  # noqa: E402  (also puts this worktree's python/ on sys.path)
import drive_conditions as dc  # noqa: E402

import torch  # noqa: E402

from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat  # noqa: E402
from sglang.srt.layers.moe.exl3_expert_layout import (  # noqa: E402
    Exl3ExpertLayout,
    build_exl3_expert_layout,
)
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy  # noqa: E402
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader  # noqa: E402
from sglang.srt.layers.moe.exl3_shard_row_source import (  # noqa: E402
    BOUNCE_ROWS,
    Exl3ShardRowSource,
    shared_row_reader,
)
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES  # noqa: E402

DEFAULT_SIZES = (1, 2, 4, 6, 8, 32)
ARM_KINDS = ("within_row", "whole_row", "one_root", "weighted")
TAIL_MIN_SAMPLES = 100  # below this p99 is the sample maximum, not a tail estimate
GIB = 1 << 30


# ------------------------------------------------------------------- replay


class RowRequest(NamedTuple):
    layer: int
    expert: int


class RowGeometry(NamedTuple):
    path: str
    offset: int  # page-aligned file offset of the superset read
    length: int  # page-aligned superset length: what an arm must request
    start: int  # where the row begins inside the superset
    useful: int  # the row's own bytes (record.nbytes)
    file_offset: int


@dataclass(frozen=True)
class Batch:
    index: int
    rows: tuple[RowRequest, ...]

    @property
    def size(self) -> int:
        return len(self.rows)


def row_geometry(layout: Exl3ExpertLayout, row: RowRequest) -> RowGeometry:
    record = layout.records[(row.layer, row.expert)]
    offset, length, start = record.aligned_read(PAGE_BYTES)
    return RowGeometry(
        record.path, offset, length, start, record.nbytes, record.file_offset
    )


def build_replay(
    num_experts: int,
    layers: Sequence[int],
    sizes: Sequence[int],
    rows_per_size: int,
    seed: int,
) -> list[Batch]:
    """The request sequence every arm replays. Depends on nothing but its
    arguments: a batch is ``size`` distinct experts of one layer."""
    rng = random.Random(seed)
    batches: list[Batch] = []
    for size in sizes:
        if size > num_experts:
            raise ValueError(f"batch size {size} exceeds the {num_experts} experts")
        for i in range(max(2, -(-rows_per_size // size))):
            layer = layers[i % len(layers)]
            experts = rng.sample(range(num_experts), size)
            batches.append(
                Batch(len(batches), tuple(RowRequest(layer, e) for e in experts))
            )
    return batches


def replay_digest(batches: Sequence[Batch]) -> str:
    h = hashlib.sha256()
    for batch in batches:
        h.update(repr((batch.index, batch.rows)).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------- planning


class Extent(NamedTuple):
    row: int  # application row: position in the batch
    root: int  # index into the roots
    offset: int  # file offset of this extent
    dest_offset: int  # byte offset inside the row's destination slot
    length: int


class SplitPlanner:
    """Each row is divided across the roots by a length-only policy."""

    def __init__(self, weights: Sequence[float]) -> None:
        self.weights = tuple(weights)
        self._policy = StaticSplitPolicy(self.weights)

    def reset(self) -> None:
        return None

    def plan(self, geometries: Sequence[RowGeometry]) -> list[Extent]:
        extents = []
        for row, geometry in enumerate(geometries):
            split = self._policy.plan(geometry.length)
            for root, (part_start, part_bytes) in enumerate(
                zip(split.starts, split.part_bytes)
            ):
                if part_bytes:
                    extents.append(
                        Extent(
                            row, root, geometry.offset + part_start, part_start,
                            part_bytes,
                        )
                    )  # fmt: skip
        return extents


class WholeRowPlanner:
    """Each row is read whole from one root.

    The choice needs the rows already assigned in this batch and the bytes each
    root has served before it, so it is stateful and cannot be a function of one
    row's length: ``SplitPolicy.plan(length)`` has no place for either.
    """

    def __init__(self, num_roots: int) -> None:
        self.num_roots = num_roots
        self.served = [0] * num_roots

    def reset(self) -> None:
        self.served = [0] * self.num_roots

    def assign(self, lengths: Sequence[int]) -> list[int]:
        outstanding = [0] * self.num_roots
        chosen = []
        for length in lengths:
            root = min(
                range(self.num_roots),
                key=lambda r: (outstanding[r], self.served[r], r),
            )
            outstanding[root] += length
            chosen.append(root)
        for root, nbytes in enumerate(outstanding):
            self.served[root] += nbytes
        return chosen

    def plan(self, geometries: Sequence[RowGeometry]) -> list[Extent]:
        roots = self.assign([g.length for g in geometries])
        return [
            Extent(row, root, g.offset, 0, g.length)
            for row, (root, g) in enumerate(zip(roots, geometries))
        ]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str
    weights: tuple[float, ...] = ()  # split arms; empty for whole_row


def make_planner(spec: ArmSpec, num_roots: int):
    if spec.kind == "whole_row":
        return WholeRowPlanner(num_roots)
    return SplitPlanner(spec.weights)


def weight_label(weights: Sequence[float]) -> str:
    return ":".join(f"{w:g}" for w in weights)


def build_arm_specs(
    kinds: Sequence[str],
    num_roots: int,
    labels: Sequence[str],
    one_root_index: int,
    weighted: Optional[Sequence[float]],
) -> list[ArmSpec]:
    specs = []
    if "within_row" in kinds:
        even = tuple(1.0 for _ in range(num_roots))
        specs.append(ArmSpec(f"within-row {weight_label(even)}", "within_row", even))
    if "whole_row" in kinds:
        specs.append(ArmSpec("whole-row", "whole_row"))
    if "one_root" in kinds:
        only = tuple(1.0 if i == one_root_index else 0.0 for i in range(num_roots))
        specs.append(ArmSpec(f"one-root {labels[one_root_index]}", "one_root", only))
    if "weighted" in kinds:
        if weighted is None:
            raise ValueError("the weighted arm needs weights")
        specs.append(
            ArmSpec(f"weighted {weight_label(weighted)}", "weighted", tuple(weighted))
        )
    return specs


def plan_summary(
    planner, batches: Sequence[Batch], layout, num_roots: int
) -> dict:
    """What an arm would submit for a replay, with no I/O: the per-drive bytes
    and the signature that says whether two arms plan identically."""
    planner.reset()
    h = hashlib.sha256()
    drive_bytes = [0] * num_roots
    extents = rows = useful = requested = 0
    for batch in batches:
        geoms = [row_geometry(layout, r) for r in batch.rows]
        plan = planner.plan(geoms)
        h.update(repr(plan).encode())
        rows += batch.size
        extents += len(plan)
        useful += sum(g.useful for g in geoms)
        requested += sum(g.length for g in geoms)
        for e in plan:
            drive_bytes[e.root] += e.length
        if sum(e.length for e in plan) != sum(g.length for g in geoms):
            raise RuntimeError(f"batch {batch.index}: extents do not cover the rows")
    planner.reset()
    return {
        "rows": rows, "extents": extents, "useful_bytes": useful,
        "requested_bytes": requested, "drive_bytes": drive_bytes,
        "signature": h.hexdigest()[:16],
    }  # fmt: skip


def request_model(
    specs: Sequence[ArmSpec],
    batches: Sequence[Batch],
    layout: Exl3ExpertLayout,
    drives: Sequence[dc.DriveInfo],
) -> dict:
    """Per-drive extents and predicted block requests for each arm, per batch size.

    Plans exactly as ``run_replay`` does (stateful planners reset at the start of
    every batch-size group) and reads nothing but the layout. The request counts
    are ``block_requests`` of each extent, a model of the block layer, not a
    measurement.
    """
    num_roots = len(drives)
    groups = group_by_size(batches)
    out: dict = {}
    for spec in specs:
        planner = make_planner(spec, num_roots)
        per_size = {}
        for size, group in groups.items():
            planner.reset()
            nbytes = [0] * num_roots
            extents = [0] * num_roots
            requests = [0] * num_roots
            for batch in group:
                geoms = [row_geometry(layout, r) for r in batch.rows]
                for e in planner.plan(geoms):
                    nbytes[e.root] += e.length
                    extents[e.root] += 1
                    requests[e.root] += dc.block_requests(
                        length=e.length, max_request_bytes=drives[e.root].max_request_bytes
                    )  # fmt: skip
            n = len(group)
            busiest = max(requests)
            per_size[str(size)] = {
                "batches": n,
                "bytes_per_batch": [b / n for b in nbytes],
                "extents_per_batch": [x / n for x in extents],
                "requests_per_batch": [r / n for r in requests],
                "bytes_per_request": [b / r if r else None for b, r in zip(nbytes, requests)],
                "request_share": [r / sum(requests) if sum(requests) else None for r in requests],
                "busiest_over_quietest_requests": (
                    busiest / min(requests) if min(requests) else None
                ),
            }
        out[spec.name] = per_size
    return out


def format_request_model(model: dict, drives: Sequence[dc.DriveInfo]) -> str:
    lines = [
        "predicted block requests per batch (model: ceil(extent / max_sectors_kb), not measured)",
        "drives: "
        + "; ".join(
            f"[{i}] {d.path} = {d.device} ({d.model}, {d.fs_type}, max_sectors_kb={d.max_sectors_kb}, "
            f"max_segments={d.max_segments})"
            for i, d in enumerate(drives)
        ),
    ]
    header = (
        f"{'arm':<22}{'rows':>5}  {'extents/drive':<16}{'MB/drive':<18}"
        f"{'requests/drive':<18}{'req share':<16}{'max/min req':>11}"
    )
    lines += [header, "-" * len(header)]
    for name, per_size in model.items():
        for size, r in per_size.items():
            ratio = r["busiest_over_quietest_requests"]
            lines.append(
                f"{name:<22}{size:>5}  "
                f"{'/'.join(f'{x:.1f}' for x in r['extents_per_batch']):<16}"
                f"{'/'.join(f'{x / 1e6:.1f}' for x in r['bytes_per_batch']):<18}"
                f"{'/'.join(f'{x:.1f}' for x in r['requests_per_batch']):<18}"
                f"{'/'.join(f'{x:.2f}' for x in r['request_share'] if x is not None):<16}"
                f"{('inf' if ratio is None else f'{ratio:.2f}'):>11}"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------ execution


@dataclass
class BatchSample:
    arm: str
    rep: int
    size: int
    batch: int
    read_ns: int
    pack_ns: int
    plan_ns: int
    rows: int
    extents: int
    drive_bytes: list
    drive_extents: list
    useful_bytes: int
    requested_bytes: int
    completed_bytes: int
    drive_requests: list = field(default_factory=list)  # model, empty without drive limits


class LayerContext(NamedTuple):
    source: Exl3ShardRowSource  # names, row bytes, segments; its own bounce is unused
    segments: tuple


@dataclass
class Harness:
    layout: Exl3ExpertLayout
    reader: Exl3RowReader
    roots: tuple[str, ...]
    layers: dict[int, LayerContext]
    bounce: torch.Tensor
    dests: dict[str, torch.Tensor]
    root_bytes: list = field(default_factory=list)  # bytes asked of each root, all phases
    drives: tuple = ()  # dc.DriveInfo per root; empty disables the request model

    def __post_init__(self) -> None:
        self.root_bytes = [0] * len(self.roots)

    @classmethod
    def build(cls, layout, source: str, roots, layers, max_rows: int, drives=()) -> "Harness":
        roots = tuple(roots)
        reader = shared_row_reader(layout, True, source)
        contexts = {}
        for layer in layers:
            segments = Exl3ExpertFormat(layout, layer, direct=True).segment_map()
            contexts[layer] = LayerContext(
                Exl3ShardRowSource.for_layer(layout, layer, segments, direct=True),
                tuple(segments),
            )
        first = contexts[layers[0]].source
        for layer, ctx in contexts.items():
            if ctx.source.names != first.names or ctx.source.row_bytes != first.row_bytes:
                raise RuntimeError(f"layer {layer} lays a row out differently")
        slot_bytes = -(-reader.buffer_bytes // PAGE_BYTES) * PAGE_BYTES
        storage = torch.empty(max_rows * slot_bytes + PAGE_BYTES, dtype=torch.uint8)
        start = (-storage.data_ptr()) % PAGE_BYTES
        bounce = storage[start : start + max_rows * slot_bytes].view(max_rows, -1)
        return cls(
            layout, reader, roots, contexts, bounce,
            base.make_destinations(first, max_rows), drives=tuple(drives),
        )  # fmt: skip

    def open_everything(self, batches: Sequence[Batch]) -> None:
        """Open every (root, shard) the replay touches, outside any timing."""
        paths = {
            self.layout.records[(r.layer, r.expert)].path
            for b in batches
            for r in b.rows
        }
        for path in paths:
            for root in self.roots:
                self.reader._file(path, root)

    def submit(self, geoms: Sequence[RowGeometry], extents: Sequence[Extent]) -> int:
        """One ``_submit`` for the whole batch; returns the bytes owed."""
        file_ids, offsets, dests, lengths = [], [], [], []
        expected = 0
        for e in extents:
            g = geoms[e.row]
            file_id, file_bytes = self.reader._file(g.path, self.roots[e.root])
            file_ids.append(file_id)
            offsets.append(e.offset)
            dests.append(self.bounce[e.row].data_ptr() + e.dest_offset)
            lengths.append(e.length)
            expected += max(0, min(e.length, file_bytes - e.offset))
        self.reader._submit(file_ids, offsets, dests, lengths, expected)
        return expected

    def pack(self, batch: Batch, geoms: Sequence[RowGeometry], source_bounce=None) -> None:
        """The scatter ``Exl3ShardRowSource.read`` does after a read."""
        bounce = self.bounce if source_bounce is None else source_bounce
        ctx = self.layers[batch.rows[0].layer]
        for slot, geometry in enumerate(geoms):
            source = bounce[slot]
            for seg in ctx.segments:
                at = geometry.start + seg.src_offset
                self.dests[seg.name][
                    slot, seg.dst_offset : seg.dst_offset + seg.nbytes
                ].copy_(source[at : at + seg.nbytes])

    def digests(self, count: int) -> list[int]:
        return base.row_digests(self.dests, count)

    def clear(self, count: int) -> None:
        self.bounce[:count].zero_()
        for tensor in self.dests.values():
            tensor[:count].zero_()

    def run_batch(
        self, arm: ArmSpec, planner, batch: Batch, rep: int, check: bool
    ) -> tuple[BatchSample, Optional[list[int]]]:
        geoms = [row_geometry(self.layout, r) for r in batch.rows]
        num_roots = len(self.roots)
        if check:
            self.clear(batch.size)
        began = time.perf_counter_ns()
        extents = planner.plan(geoms)
        planned = time.perf_counter_ns()
        completed = self.submit(geoms, extents)
        read_done = time.perf_counter_ns()
        self.pack(batch, geoms)
        pack_done = time.perf_counter_ns()
        drive_bytes = [0] * num_roots
        drive_extents = [0] * num_roots
        drive_requests = [0] * num_roots
        for e in extents:
            drive_bytes[e.root] += e.length
            drive_extents[e.root] += 1
            if self.drives:
                drive_requests[e.root] += dc.block_requests(
                    length=e.length, max_request_bytes=self.drives[e.root].max_request_bytes
                )  # fmt: skip
        for root, nbytes in enumerate(drive_bytes):
            self.root_bytes[root] += nbytes
        sample = BatchSample(
            arm.name, rep, batch.size, batch.index,
            read_done - planned, pack_done - read_done, planned - began,
            batch.size, len(extents), drive_bytes, drive_extents,
            sum(g.useful for g in geoms), sum(g.length for g in geoms), completed,
            drive_requests if self.drives else [],
        )  # fmt: skip
        return sample, (self.digests(batch.size) if check else None)

    def reference_digests(self, rows: Sequence[RowRequest]) -> dict[RowRequest, int]:
        """Digest of each row read with ordinary ``pread`` from the source
        checkpoint and packed by the same scatter: independent of the io_uring
        path and of every mirror."""
        out = {}
        for row in rows:
            geometry = row_geometry(self.layout, row)
            file_bytes = os.path.getsize(geometry.path)
            n = max(0, min(geometry.length, file_bytes - geometry.offset))
            scratch = torch.zeros(1, self.bounce.shape[1], dtype=torch.uint8)
            with open(geometry.path, "rb", buffering=0) as f:
                f.seek(geometry.offset)
                data = f.read(n)
            scratch[0, :n] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            for tensor in self.dests.values():
                tensor[:1].zero_()
            single = Batch(0, (row,))
            self.pack(single, [geometry], source_bounce=scratch)
            out[row] = self.digests(1)[0]
        return out


# --------------------------------------------------------------------- driver


@dataclass
class RunResult:
    samples: list = field(default_factory=list)
    warmup_bytes: int = 0
    blocks: list = field(default_factory=list)  # one conditions record per (rep, size, arm)
    mismatches: list = field(default_factory=list)
    source_mismatches: list = field(default_factory=list)
    verified_rows: int = 0


def group_by_size(batches: Sequence[Batch]) -> dict[int, list[Batch]]:
    groups: dict[int, list[Batch]] = {}
    for b in batches:
        groups.setdefault(b.size, []).append(b)
    return groups


def rotate(items: Sequence, block: int) -> list:
    return base.arm_order(items, block)


def run_replay(
    harness: Harness,
    specs: Sequence[ArmSpec],
    batches: Sequence[Batch],
    reps: int,
    warmup: Sequence[Batch],
    verify_rows: int,
    verbose: bool = True,
    probe: Optional[dc.ConditionProbe] = None,
) -> RunResult:
    result = RunResult()
    num_roots = len(harness.roots)
    planners = {s.name: make_planner(s, num_roots) for s in specs}
    harness.open_everything(list(batches) + list(warmup))

    for spec in specs:
        planner = planners[spec.name]
        planner.reset()
        for batch in warmup:
            sample, _ = harness.run_batch(spec, planner, batch, -1, False)
            result.warmup_bytes += sample.requested_bytes

    distinct = list(dict.fromkeys(r for b in batches for r in b.rows))[:verify_rows]
    reference = harness.reference_digests(distinct)
    groups = group_by_size(batches)
    digests: dict[tuple[str, int], list[int]] = {}
    block = 0
    for rep in range(reps):
        for size, group in groups.items():
            for spec in rotate(list(specs), block):
                planner = planners[spec.name]
                planner.reset()
                started = probe.start() if probe else None
                for batch in group:
                    check = rep == 0
                    sample, digest = harness.run_batch(spec, planner, batch, rep, check)
                    result.samples.append(sample)
                    if digest is not None:
                        digests[(spec.name, batch.index)] = digest
                        for row, crc in zip(batch.rows, digest):
                            if row in reference:
                                result.verified_rows += 1
                                if crc != reference[row]:
                                    result.source_mismatches.append(
                                        {"arm": spec.name, "batch": batch.index,
                                         "layer": row.layer, "expert": row.expert}
                                    )  # fmt: skip
                if probe:
                    result.blocks.append(
                        {"arm": spec.name, "rep": rep, "size": size, "batches": len(group),
                         **probe.stop(started)}
                    )  # fmt: skip
                if verbose:
                    ms = [s.read_ns / 1e6 for s in result.samples
                          if s.arm == spec.name and s.size == size and s.rep == rep]
                    print(
                        f"rep {rep} size {size:>2} {spec.name:<20} "
                        f"n={len(ms)} read p50={statistics.median(ms):.3f}ms",
                        flush=True,
                    )
            block += 1

    first = specs[0].name
    for spec in specs[1:]:
        for batch in batches:
            a, b = digests[(first, batch.index)], digests[(spec.name, batch.index)]
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    result.mismatches.append(
                        {"arm": spec.name, "reference": first, "batch": batch.index,
                         "row": i, "expert": batch.rows[i].expert}
                    )  # fmt: skip
    return result


def check_equal_work(samples: Sequence[BatchSample], specs: Sequence[ArmSpec]) -> list:
    """Every arm must have asked for the same rows, useful bytes and aligned
    bytes, batch by batch. Returns the violations."""
    per_arm = {
        s.name: {
            (x.rep, x.batch): (x.rows, x.useful_bytes, x.requested_bytes, x.completed_bytes)
            for x in samples
            if x.arm == s.name
        }
        for s in specs
    }
    reference = per_arm[specs[0].name]
    errors = []
    for spec in specs[1:]:
        if per_arm[spec.name] != reference:
            errors.append(f"{spec.name} requested different work from {specs[0].name}")
    for x in samples:
        if sum(x.drive_bytes) != x.requested_bytes:
            errors.append(f"{x.arm} batch {x.batch}: per-drive bytes != requested")
    return errors


# -------------------------------------------------------------------- calibrate


def calibrate_weights(
    harness: Harness, batches: Sequence[Batch], rounds: int = 2
) -> dict:
    """Per-root one-root throughput on the same rows, roots alternating, and the
    weights it implies (rounded to two places so they are quotable)."""
    num_roots = len(harness.roots)
    harness.open_everything(batches)
    rates: list[list[float]] = [[] for _ in range(num_roots)]
    moved = 0
    for round_ in range(rounds):
        for root in rotate(list(range(num_roots)), round_):
            only = tuple(1.0 if i == root else 0.0 for i in range(num_roots))
            spec = ArmSpec(f"calib {root}", "one_root", only)
            planner = SplitPlanner(only)
            for batch in batches:
                sample, _ = harness.run_batch(spec, planner, batch, -2, False)
                rates[root].append(sample.requested_bytes / (sample.read_ns / 1e9))
                moved += sample.requested_bytes
    mb_s = [statistics.median(r) / 1e6 for r in rates]
    top = max(mb_s)
    weights = tuple(round(m / top, 2) for m in mb_s)
    return {"mb_per_s": mb_s, "weights": weights, "bytes_moved": moved, "rounds": rounds}


# ---------------------------------------------------------------------- report


def summarize_group(samples: Sequence[BatchSample], reps: int, num_roots: int) -> dict:
    n = len(samples)
    read_ms = [s.read_ns / 1e6 for s in samples]
    total_ms = [(s.read_ns + s.pack_ns) / 1e6 for s in samples]
    rows = samples[0].rows
    drive_bytes = [[s.drive_bytes[r] for s in samples] for r in range(num_roots)]
    drive_ext = [[s.drive_extents[r] for s in samples] for r in range(num_roots)]
    total_req = sum(s.requested_bytes for s in samples)
    out = {
        "n_batches": n,
        "app_rows_per_batch": rows,
        "extents_per_batch": statistics.mean(s.extents for s in samples),
        "read_ms": {f"p{p}": base.pct(read_ms, p) for p in (50, 90, 95, 99)},
        "read_plus_pack_ms": {f"p{p}": base.pct(total_ms, p) for p in (50, 90, 95, 99)},
        "read_ms_mean": statistics.mean(read_ms),
        "pack_ms_mean": statistics.mean(s.pack_ns for s in samples) / 1e6,
        "plan_ms_mean": statistics.mean(s.plan_ns for s in samples) / 1e6,
        "read_ms_per_row_p50": base.pct([m / rows for m in read_ms], 50),
        "read_mb_per_s": total_req / 1e6 / (sum(s.read_ns for s in samples) / 1e9),
        "requested_bytes_per_batch": total_req / n,
        "useful_bytes_per_batch": sum(s.useful_bytes for s in samples) / n,
        "per_drive_bytes_mean": [statistics.mean(d) for d in drive_bytes],
        "per_drive_bytes_max": [max(d) for d in drive_bytes],
        "per_drive_extents_max": [max(d) for d in drive_ext],
        "max_batch_extents": max(s.extents for s in samples),
        "drive_share": [sum(d) / total_req for d in drive_bytes],
        "rep_p50_ms": [
            statistics.median(s.read_ns / 1e6 for s in samples if s.rep == rep)
            if any(s.rep == rep for s in samples)
            else float("nan")
            for rep in range(reps)
        ],
        "tail_reliable": n >= TAIL_MIN_SAMPLES,
    }
    return out


def paired_vs_first(samples: Sequence[BatchSample], specs: Sequence[ArmSpec]) -> dict:
    """Read-time ratio of each arm to the first arm on the *same batch* in the
    *same rep* (below 1 is faster). The replay makes this pairing exact. Pairs
    from different reps of one batch share rows, so they are not independent:
    the sign-test p is an upper-bound on evidence, not a proper one."""
    ref = specs[0].name
    read = {(x.arm, x.rep, x.batch): x for x in samples}
    out: dict = {}
    for spec in specs[1:]:
        per_size: dict = {}
        for size in sorted({x.size for x in samples}):
            for label, keep in (("all_reps", 0), ("after_rep0", 1)):
                ratios = [
                    read[(spec.name, x.rep, x.batch)].read_ns / x.read_ns
                    for x in samples
                    if x.arm == ref and x.size == size and x.rep >= keep
                ]
                if not ratios:
                    continue
                wins = sum(r < 1 for r in ratios)
                n = len(ratios)
                tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
                per_size.setdefault(str(size), {})[label] = {
                    "pairs": n, "median_ratio": statistics.median(ratios),
                    "mean_ratio": statistics.mean(ratios), "faster_than_ref": wins,
                    "sign_test_p": min(1.0, 2 * tail / 2**n),
                }  # fmt: skip
        out[spec.name] = per_size
    return {"reference": ref, "arms": out}


RESIDENCY_CHANGE_BYTES = 64 << 20  # arbitrary; a block whose shard residency moved more is not cold-vs-warm comparable
REQUEST_DISAGREE = 0.25  # arbitrary; the model is a size-cap-only estimate, so a wide margin


def conditions_summary(
    blocks: Sequence[dict], samples: Sequence[BatchSample], num_roots: int
) -> dict:
    """Per (arm, size): model against measurement and the conditions it was taken under."""
    out: dict = {}
    for arm in dict.fromkeys(b["arm"] for b in blocks):
        per_size = {}
        for size in dict.fromkeys(b["size"] for b in blocks if b["arm"] == arm):
            group = [b for b in blocks if b["arm"] == arm and b["size"] == size]
            picked = [s for s in samples if s.arm == arm and s.size == size]
            drives = []
            for r in range(num_roots):
                cells = [b["drives"][r] for b in group]
                predicted = sum(s.drive_requests[r] for s in picked) if picked[0].drive_requests else None
                measured = sum(c["reads_completed"] for c in cells)
                asked = sum(s.drive_bytes[r] for s in picked)
                busy = sum(c["io_ms"] for c in cells)
                deltas = [c["residency_delta_bytes"] for c in cells]
                drives.append({
                    "device": cells[0]["device"],
                    "asked_bytes": asked,
                    "diskstats_bytes": sum(c["read_bytes"] for c in cells),
                    "predicted_requests": predicted,
                    "measured_requests": measured,
                    "measured_merged": sum(c["reads_merged"] for c in cells),
                    "requests_disagree": (
                        None if not predicted
                        else abs(measured - predicted) / predicted > REQUEST_DISAGREE
                    ),
                    "depth_when_busy": (
                        sum(c["weighted_io_ms"] for c in cells) / busy if busy else None
                    ),
                    "residency_before_min": _min_none(
                        c["residency_before"] and c["residency_before"]["resident_bytes"] for c in cells
                    ),
                    "residency_before_max": _max_none(
                        c["residency_before"] and c["residency_before"]["resident_bytes"] for c in cells
                    ),
                    "residency_unmeasured": any(d is None for d in deltas),
                    "residency_moved": any(d is None or abs(d) > RESIDENCY_CHANGE_BYTES for d in deltas),
                })  # fmt: skip
            per_size[str(size)] = {
                "blocks": len(group),
                "drives": drives,
                "load_average_1m": [
                    min(b["load_average_before"][0] for b in group),
                    max(b["load_average_after"][0] for b in group),
                ],
                "foreign_cores_low_max": max(b["foreign_cores_low"] or 0.0 for b in group),
                "foreign_cores_all_max": max(b["foreign_cores_all"] or 0.0 for b in group),
            }
        out[arm] = per_size
    return out


def _min_none(values):
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _max_none(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def format_conditions(summary: dict) -> str:
    lines = [
        "conditions per (arm, batch size): requests = block requests per drive over the whole",
        "block (model / measured reads-completed); depth = in-flight requests while busy;",
        "cache = resident shard GiB at block start (min-max), '!' if it moved > 64 MiB;",
        "load = 1-min loadavg range; foreign = other processes' cores on cpus 0-63 / all.",
    ]
    header = f"{'arm':<22}{'rows':>5}  {'device':<12}{'req model/meas':>18}{'depth':>7}{'cache GiB':>14}{'load':>12}{'foreign':>12}"
    lines += [header, "-" * len(header)]
    for arm, per_size in summary.items():
        for size, c in per_size.items():
            load = f"{c['load_average_1m'][0]:.1f}-{c['load_average_1m'][1]:.1f}"
            foreign = f"{c['foreign_cores_low_max']:.1f}/{c['foreign_cores_all_max']:.1f}"
            for i, d in enumerate(c["drives"]):
                pred = "-" if d["predicted_requests"] is None else str(d["predicted_requests"])
                flag = "?" if d["requests_disagree"] else " "
                depth = "-" if d["depth_when_busy"] is None else f"{d['depth_when_busy']:.1f}"
                lo, hi = d["residency_before_min"], d["residency_before_max"]
                cache = "unmeasured" if lo is None else f"{lo / GIB:.1f}-{hi / GIB:.1f}"
                cache += "!" if d["residency_moved"] else ""
                lines.append(
                    f"{arm if i == 0 else '':<22}{size if i == 0 else '':>5}  {d['device']:<12}"
                    f"{pred + '/' + str(d['measured_requests']) + flag:>18}{depth:>7}{cache:>14}"
                    f"{load if i == 0 else '':>12}{foreign if i == 0 else '':>12}"
                )
    return "\n".join(lines)


def build_report(
    result: RunResult,
    specs: Sequence[ArmSpec],
    batches: Sequence[Batch],
    meta: dict,
    reps: int,
    num_roots: int,
) -> dict:
    results: dict = {}
    for spec in specs:
        per_size: dict = {}
        for size in sorted({b.size for b in batches}):
            group = [s for s in result.samples if s.arm == spec.name and s.size == size]
            if group:
                per_size[str(size)] = summarize_group(group, reps, num_roots)
        results[spec.name] = per_size
    report = dict(meta)
    report["results"] = results
    if result.blocks:
        report["conditions"] = conditions_summary(result.blocks, result.samples, num_roots)
        report["condition_blocks"] = result.blocks
    report["accounting_errors"] = check_equal_work(result.samples, specs)
    report["paired"] = paired_vs_first(result.samples, specs)
    report["bytes_match"] = not result.mismatches and not result.source_mismatches
    report["mismatches"] = result.mismatches[:50]
    report["source_mismatches"] = result.source_mismatches[:50]
    report["source_verified_rows"] = result.verified_rows
    report["batch_samples"] = [
        {
            "arm": s.arm, "rep": s.rep, "size": s.size, "batch": s.batch,
            "read_ns": s.read_ns, "pack_ns": s.pack_ns, "plan_ns": s.plan_ns,
            "extents": s.extents, "drive_bytes": s.drive_bytes,
            "drive_requests_model": s.drive_requests,
        }
        for s in result.samples
    ]  # fmt: skip
    return report


def format_table(report: dict) -> str:
    lines = []
    sizes = sorted({int(s) for arm in report["results"].values() for s in arm})
    for size in sizes:
        lines.append(
            f"batch of {size} application rows  (extents and per-drive bytes are "
            f"per batch; times in ms)"
        )
        header = (
            f"{'arm':<22}{'n':>4}{'ext':>5}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}"
            f"{'pack':>7}{'ms/row':>8}{'MB/s':>7}  per-drive MB (mean/max)  rep p50"
        )
        lines += [header, "-" * len(header)]
        for name, per_size in report["results"].items():
            r = per_size.get(str(size))
            if not r:
                continue
            t = r["read_ms"]
            mean = "/".join(f"{b / 1e6:.0f}" for b in r["per_drive_bytes_mean"])
            top = "/".join(f"{b / 1e6:.0f}" for b in r["per_drive_bytes_max"])
            reps = " ".join(f"{m:.2f}" for m in r["rep_p50_ms"])
            flag = "" if r["tail_reliable"] else "*"
            lines.append(
                f"{name:<22}{r['n_batches']:>4}{r['extents_per_batch']:>5.0f}"
                f"{t['p50']:>9.3f}{t['p90']:>9.3f}{t['p95']:>9.3f}{t['p99']:>8.3f}{flag or ' '}"
                f"{r['pack_ms_mean']:>7.2f}{r['read_ms_per_row_p50']:>8.3f}"
                f"{r['read_mb_per_s']:>7.0f}  {mean} / {top}   {reps}"
            )
        lines.append("")
    if report.get("conditions"):
        lines += [format_conditions(report["conditions"]), ""]
    paired = report["paired"]
    lines.append(
        f"paired against {paired['reference']} on the same batch (ratio < 1 is faster; "
        "reps after 0; wins/pairs; sign-test p, pairs not independent):"
    )
    for name, per_size in paired["arms"].items():
        cells = []
        for size, labels in per_size.items():
            c = labels.get("after_rep0")
            if c:
                cells.append(
                    f"{size}:{c['median_ratio']:.3f} ({c['faster_than_ref']}/{c['pairs']}"
                    f", p={c['sign_test_p']:.2g})"
                )
        lines.append(f"  {name:<22}" + "  ".join(cells))
    lines.append(
        "* fewer than %d batches: p99 is the sample maximum, p95 near it; not a tail"
        % TAIL_MIN_SAMPLES
    )
    lines.append(
        "  estimate. read = the one concurrent submit; pack = the scatter that follows."
    )
    for note in report.get("identical_plans", []):
        lines.append(f"NOTE: {note}")
    if report["accounting_errors"]:
        lines.append("ACCOUNTING: " + "; ".join(report["accounting_errors"]))
    else:
        lines.append(
            "accounting: every arm requested the same rows, useful bytes and aligned "
            "bytes in every batch."
        )
    if report["bytes_match"]:
        lines.append(
            f"byte check: arms agree on every rep-0 row; {report['source_verified_rows']} "
            "row reads matched a plain read of the source."
        )
    else:
        lines.append(
            "BYTE MISMATCH: "
            f"{len(report['mismatches'])} arm/arm, {len(report['source_mismatches'])} "
            "arm/source. Timings are not trustworthy."
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ diskstats


def device_key(path: str) -> tuple[int, int]:
    st = os.stat(path).st_dev
    return os.major(st), os.minor(st)


def read_diskstats(paths: Sequence[str]) -> dict:
    """Bytes read so far by the block device under each path; None if it has no
    /proc/diskstats row (overlay, tmpfs)."""
    wanted = {p: device_key(p) for p in paths}
    rows = {}
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                fields = line.split()
                rows[(int(fields[0]), int(fields[1]))] = (
                    fields[2],
                    int(fields[5]) * 512,
                )
    except OSError:
        return {p: None for p in paths}
    return {
        p: ({"device": rows[k][0], "read_bytes": rows[k][1]} if k in rows else None)
        for p, k in wanted.items()
    }


def diskstats_delta(before: dict, after: dict) -> dict:
    return {
        p: (
            after[p]["read_bytes"] - before[p]["read_bytes"]
            if before[p] and after[p]
            else None
        )
        for p in before
    }


# ----------------------------------------------------------------------- main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default=base.DEFAULT_SOURCE)
    p.add_argument("--roots", nargs="+", default=list(base.DEFAULT_ROOTS))
    p.add_argument("--layers", nargs="+", type=int, default=[base.DEFAULT_LAYER])
    p.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES),
                   help="application rows per concurrent batch")  # fmt: skip
    p.add_argument("--rows-per-size", type=int, default=48,
                   help="rows replayed per batch size, per rep (at least 2 batches)")  # fmt: skip
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    p.add_argument("--arms", nargs="+", choices=ARM_KINDS, default=list(ARM_KINDS))
    p.add_argument("--one-root-index", type=int, default=0)
    p.add_argument("--weights", default="auto",
                   help="weights for the weighted arm: 'auto' measures them from a "
                        "one-root calibration, or e.g. 3:1")  # fmt: skip
    p.add_argument("--warmup-batches", type=int, default=4)
    p.add_argument("--calib-batches", type=int, default=2)
    p.add_argument("--verify-rows", type=int, default=16,
                   help="rows also read from the source checkpoint, as ground truth")  # fmt: skip
    p.add_argument("--max-gib", type=float, default=80.0,
                   help="refuse to run if the planned reads exceed this")  # fmt: skip
    p.add_argument("--model-requests", action="store_true",
                   help="predict per-drive block requests for the replay from sysfs limits "
                        "and the extent plan; reads only the checkpoint headers")  # fmt: skip
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan, per-drive bytes and total I/O; read nothing")  # fmt: skip
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    if args.reps < 1 or args.rows_per_size < 1 or min(args.sizes) < 1:
        p.error("--reps, --rows-per-size and --sizes must be at least 1")
    return args


def mean_row_bytes(layout, layers: Sequence[int]) -> int:
    lengths = [
        layout.records[(layer, e)].aligned_read(PAGE_BYTES)[1]
        for layer in layers
        for e in range(layout.num_experts)
    ]
    return sum(lengths) // len(lengths)


def report_request_model(args, specs, replay, layout, roots) -> int:
    if "weighted" in args.arms and args.weights == "auto":
        raise SystemExit("--model-requests needs explicit --weights for the weighted arm")
    drives = [dc.resolve_drive(r) for r in roots]
    model = request_model(specs, replay, layout, drives)
    print()
    print(format_request_model(model, drives))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "when": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "argv": sys.argv,
                    "replay_digest": replay_digest(replay),
                    "load_average": dc.load_average(),
                    "drives": [d._asdict() for d in drives],
                    "arms": [
                        {"name": s.name, "kind": s.kind, "weights": list(s.weights)} for s in specs
                    ],
                    "model": model,
                },
                f, indent=2,
            )  # fmt: skip
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source = os.path.realpath(args.source)
    roots = [os.path.realpath(r) for r in args.roots]
    if len(set(roots)) != len(roots):
        raise SystemExit(f"--roots resolve to duplicates: {roots}")
    if source in roots:
        raise SystemExit("a mirror root is the source checkpoint itself")
    if len(roots) < 2:
        raise SystemExit("scheduling arms need at least two roots")
    if not 0 <= args.one_root_index < len(roots):
        raise SystemExit(f"--one-root-index {args.one_root_index} outside the roots")
    devices = {os.stat(r).st_dev for r in roots}
    layout = build_exl3_expert_layout(source)
    for layer in args.layers:
        if not 0 <= layer < layout.num_layers:
            raise SystemExit(
                f"layer {layer} is outside the checkpoint's {layout.num_layers}"
            )
    if max(args.sizes) > layout.num_experts:
        raise SystemExit(f"batch size exceeds the {layout.num_experts} experts")

    taken: set[str] = set()
    labels = [base.root_label(r, taken) for r in roots]
    replay = build_replay(
        layout.num_experts, args.layers, args.sizes, args.rows_per_size, args.seed
    )
    warmup = build_replay(
        layout.num_experts, args.layers, [min(8, max(args.sizes))],
        8 * args.warmup_batches, args.seed + 1,
    )[: args.warmup_batches]  # fmt: skip
    calibration = build_replay(
        layout.num_experts, args.layers, [min(8, max(args.sizes))],
        8 * args.calib_batches, args.seed + 2,
    )[: args.calib_batches]  # fmt: skip

    auto = args.weights == "auto"
    if "weighted" in args.arms and not auto:
        weights = base.parse_weights(args.weights, len(roots))
    else:
        weights = tuple(1.0 for _ in roots)  # placeholder until measured

    def planned_specs(w):
        return build_arm_specs(
            args.arms, len(roots), labels, args.one_root_index,
            w if "weighted" in args.arms else None,
        )  # fmt: skip

    specs = planned_specs(weights)
    if not specs:
        raise SystemExit("no arms selected")

    replay_requested = sum(
        row_geometry(layout, r).length for b in replay for r in b.rows
    )
    warm_requested = sum(row_geometry(layout, r).length for b in warmup for r in b.rows)
    calib_requested = sum(
        row_geometry(layout, r).length for b in calibration for r in b.rows
    )
    plan_bytes = (
        len(specs) * (args.reps * replay_requested + warm_requested)
        + (2 * len(roots) * calib_requested if "weighted" in args.arms and auto else 0)
    )
    verify_bytes = args.verify_rows * mean_row_bytes(layout, args.layers)
    print(
        f"replay {replay_digest(replay)}: {len(replay)} batches, "
        f"{sum(b.size for b in replay)} application rows per rep, sizes "
        f"{sorted(set(b.size for b in replay))}"
    )
    print(
        f"planned I/O: {plan_bytes / GIB:.2f} GiB from the mirror roots "
        f"({len(specs)} arms x {args.reps} reps x {replay_requested / GIB:.2f} GiB "
        f"replay + warmup{' + calibration' if auto and 'weighted' in args.arms else ''}); "
        f"{verify_bytes / GIB:.2f} GiB from the source ({args.verify_rows} rows)"
    )
    if max(args.sizes) > BOUNCE_ROWS:
        print(
            f"NOTE: sizes above {BOUNCE_ROWS} exceed production's bounce ring "
            f"(BOUNCE_ROWS); production would run them as serial batches of "
            f"{BOUNCE_ROWS}."
        )
    if len(devices) != len(roots):
        print("WARNING: two mirror roots are on the same device.")
    if (plan_bytes + verify_bytes) / GIB > args.max_gib:
        raise SystemExit(
            f"planned I/O {(plan_bytes + verify_bytes) / GIB:.1f} GiB exceeds "
            f"--max-gib {args.max_gib}"
        )

    signatures = {}
    for spec in specs:
        signatures[spec.name] = plan_summary(
            make_planner(spec, len(roots)), replay, layout, len(roots)
        )
    for name, s in signatures.items():
        print(
            f"  {name:<22} extents={s['extents']:>5} rows={s['rows']} "
            f"per-drive GiB={[round(b / GIB, 2) for b in s['drive_bytes']]}"
        )
    if args.model_requests:
        return report_request_model(args, specs, replay, layout, roots)
    if args.dry_run:
        return 0

    drives = [dc.resolve_drive(r) for r in roots]
    print("drives: " + "; ".join(
        f"{d.path} = {d.device} ({d.model}, {d.fs_type}, max_sectors_kb={d.max_sectors_kb})"
        for d in drives
    ))  # fmt: skip
    probe = dc.ConditionProbe(drives)
    load_start = dc.load_average()
    cpus = sorted(os.sched_getaffinity(0))
    watched = roots + [source]
    disk_before = read_diskstats(watched)
    harness = Harness.build(layout, source, roots, args.layers, max(args.sizes), drives=drives)
    calib = None
    if "weighted" in args.arms and auto:
        calib = calibrate_weights(harness, calibration)
        weights = calib["weights"]
        specs = planned_specs(weights)
        print(f"measured one-root MB/s {calib['mb_per_s']} -> weights {weights}")
        signatures = {
            spec.name: plan_summary(
                make_planner(spec, len(roots)), replay, layout, len(roots)
            )
            for spec in specs
        }

    result = run_replay(harness, specs, replay, args.reps, warmup, args.verify_rows, probe=probe)
    disk_after = read_diskstats(watched)
    identical = []
    seen: dict[str, str] = {}
    for name, s in signatures.items():
        if s["signature"] in seen:
            identical.append(
                f"{name} submits exactly the same extents as {seen[s['signature']]}; "
                "their timings differ only by run-to-run noise."
            )
        seen.setdefault(s["signature"], name)

    meta = {
        "when": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv if argv is None else argv),
        "source": source,
        "roots": roots,
        "labels": labels,
        "layers": args.layers,
        "reps": args.reps,
        "seed": args.seed,
        "sizes": args.sizes,
        "replay": {
            "digest": replay_digest(replay),
            "batches": len(replay),
            "rows_per_rep": sum(b.size for b in replay),
            "requested_bytes_per_rep": replay_requested,
        },
        "arms": [
            {"name": s.name, "kind": s.kind, "weights": list(s.weights), **signatures[s.name]}
            for s in specs
        ],
        "calibration": calib,
        "direct": True,
        "ring_depth": int(os.environ.get("SGLANG_URING_FILE_READER_QUEUE_DEPTH", 128)),
        "same_device_roots": len(devices) != len(roots),
        "drives": [d._asdict() for d in drives],
        "load_average_at_start": load_start,
        "load_average_at_end": dc.load_average(),
        "cpus_allowed": [min(cpus), max(cpus), len(cpus)],  # lowest, highest, count
        "io": {
            "planned_mirror_bytes": plan_bytes,
            "harness_bytes_per_root": dict(zip(roots, harness.root_bytes)),
            "warmup_bytes_per_arm": result.warmup_bytes // len(specs),
            "diskstats_read_bytes": diskstats_delta(disk_before, disk_after),
            "diskstats_note": "whole-device counters: include any other process's reads",
        },
        "identical_plans": identical,
    }
    report = build_report(result, specs, replay, meta, args.reps, len(roots))
    output = args.output or str(
        Path(__file__).with_name(
            "row-scheduling-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
    )
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(format_table(report))
    print(f"\nresults: {output}")
    return 0 if report["bytes_match"] and not report["accounting_errors"] else 2


if __name__ == "__main__":
    sys.exit(main())
