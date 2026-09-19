"""Split vs repack: what a RAM miss of one EXL3 expert row costs on the host.

Three ways to land a sample of one layer's expert rows in host memory, each at
several batch sizes (rows per reader submission):

- ``superset_raw``: one page-aligned superset read per row (Exl3RowReader), no split;
- ``superset_split``: the same reads, split into the six per-name rows
  (Exl3ShardRowSource, what the framework runs on a RAM miss);
- ``repacked_per_name``: per-name reads from a repacked copy of the same rows,
  as ExpertFileRowReader would do over repacked files (O_DIRECT for page-multiple
  rows, buffered for the three small names).

The repacked copy is written once, sparse (each sampled row at
``expert_id * row_bytes`` of a file sized for every expert, as production's
per-name files are, so sampled rows do not share pages), into ``--work-dir``,
which must sit on the drive being measured: the run refuses a work dir on
another device or on a RAM-backed filesystem. It deletes the copy at the end.
Every read is bounded up front by ``--max-read-gb`` (3.5 GB, the plan's cap,
warmup reads included) and the write by ``--max-write-gb`` (0.35 GB).

Only the reads are timed. Before each timed pass every destination and bounce
buffer is faulted in and one row is read (the shard files are opened, the
io_uring reader is warm), so page-fault cost is outside every region. All three
arms share one byte basis, ``payload_bytes`` (the six streamed tensors of every
row, ``gb_per_s``); ``file_bytes`` / ``file_gb_per_s`` count what the drive
moved (page-aligned supersets on the shard arms). ``deterministic`` on
``superset_split`` only says the split reproduces the repack pass's split; it
does not compare with the checkpoint (the source's own tests do). ``verified``
on ``repacked_per_name`` is a real file round-trip. ``--buffered`` reads through
the page cache and is for CPU tests, never for drive numbers.

The gate is read at batch 8 (``GATE_BATCH``; ``--rows`` defaults to 16, two whole
8-row submissions). There ``superset_split`` and ``repacked_per_name`` run
``--repeats`` (5) timed passes, interleaved split, repacked, split, ...; each row
carries every pass's ``samples_ms_per_row`` with their median and min, its
``ms_per_row`` is the median, and the repacked row's ``ratio_median`` is repacked
over split medians (<= 0.8 means repack wins). Other batches and
``superset_raw`` are one pass, as context: their ratio has no error bar. The
repeats are counted in the read budget, so the default batches are 1 and 8. Run
one process per queue depth (``SGLANG_URING_FILE_READER_QUEUE_DEPTH`` on its
command line, never changed inside a process).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import torch

from bench_expert_reads import drop_superset_ranges, superset_file_bytes
from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import (  # noqa: F401 (EXL3_STREAMED_NAMES: tests)
    EXL3_STREAMED_NAMES,
    Exl3ExpertFormat,
)
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.model_loader.file_row_reader import (
    PAGE_BYTES,
    AlignedRowSource,
    read_plans,
    shared_uring_file_reader,
)


GATE_BATCH = 8  # the batch the 0.8x go/no-go is read at
DEFAULT_ROWS = 16  # two whole 8-row submissions at the gate batch
DEFAULT_REPEATS = 5
# Batches 1 and 8: with five gate repeats a third batch would not fit the 3.5 GB cap.
DEFAULT_BATCHES = [1, 8]

# A repacked copy on one of these is RAM, not the drive under test.
_RAM_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "overlay", "devtmpfs"})


def _page_aligned(nbytes: int) -> torch.Tensor:
    """A page-aligned uint8 buffer, already faulted in (zero-filled, untimed)."""
    storage = torch.zeros(nbytes + PAGE_BYTES, dtype=torch.uint8)
    start = (-storage.data_ptr()) % PAGE_BYTES
    return storage[start : start + nbytes]


def _slabs(specs, rows: int) -> dict[str, torch.Tensor]:
    """Page-aligned per-name ``[rows, *row_shape]`` host slabs."""
    return {
        spec.name: _page_aligned(rows * spec.row_bytes).view(spec.dtype).view((rows,) + spec.row_shape)
        for spec in specs
    }


def _drop_cache(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _existing_ancestor(path: str) -> str:
    path = os.path.realpath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return path


def _device_of(path: str) -> int:
    return os.stat(path).st_dev


def _unescape_mount_point(field: str) -> str:
    return field.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def _fs_type(path: str) -> str:
    """Filesystem type of the mount that holds ``path`` (from /proc/self/mountinfo)."""
    path = os.path.realpath(path)
    best, best_type = "", None
    with open("/proc/self/mountinfo") as f:
        for line in f:
            head, _, tail = line.partition(" - ")
            fields = head.split()
            if len(fields) < 5 or not tail:
                continue
            mount_point = _unescape_mount_point(fields[4])
            inside = mount_point == "/" or path == mount_point or path.startswith(mount_point + "/")
            if inside and len(mount_point) >= len(best):
                best, best_type = mount_point, tail.split()[0]
    if best_type is None:
        raise SystemExit(f"cannot tell which filesystem holds {path}")
    return best_type


def _check_work_dir(work_dir: str, shard_path: str) -> None:
    """Refuse a scratch dir that is not on the shards' drive or is RAM-backed."""
    anchor = _existing_ancestor(work_dir)
    shard = os.path.realpath(shard_path)
    if _device_of(anchor) != _device_of(shard):
        raise SystemExit(
            f"--work-dir {work_dir} is not on the drive that holds the shards ({shard}); "
            "the repacked copy must sit on the drive under test"
        )
    fs_type = _fs_type(anchor)
    if fs_type in _RAM_FILESYSTEMS:
        raise SystemExit(
            f"--work-dir {work_dir} is on a {fs_type} mount, which is not the drive under test"
        )


def _buffered_names(sources: dict, slabs: dict) -> list[str]:
    """Names whose repacked reads are buffered: not a page-multiple row, or a misaligned slab."""
    return sorted(
        name
        for name, source in sources.items()
        if not source.direct or slabs[name].data_ptr() % PAGE_BYTES
    )


def planned_read_bytes(
    buffer_bytes: int,
    streamed_bytes: int,
    rows: int,
    batches: list[int],
    *,
    warmup_rows: int = 0,
    repeats: int = 1,
    gate_batch: int = GATE_BATCH,
) -> int:
    """Bytes the run reads from the drive under test.

    The shards give the repack pass plus two superset modes per batch size; the
    repacked copy, which sits on the same drive, gives one per-name pass per
    batch size. Each arm also reads ``warmup_rows`` untimed rows per batch size,
    and the gate batch, if listed, repeats its split and repacked passes
    ``repeats`` times (``superset_raw`` stays single-pass).
    """
    timed = rows * (buffer_bytes * (1 + 2 * len(batches)) + streamed_bytes * len(batches))
    warmup = warmup_rows * len(batches) * (2 * buffer_bytes + streamed_bytes)
    extra = (repeats - 1) * rows * (buffer_bytes + streamed_bytes) if gate_batch in batches else 0
    return timed + warmup + extra


def _write_repacked(
    work_dir: str,
    layer: int,
    specs,
    truth: dict,
    experts: list[int],
    num_experts: int,
    paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """One sparse file per streamed name, sampled expert ``e``'s row at ``e * row_bytes``.

    Production's per-name files hold every expert, and a miss reads random
    rows; a compact copy of adjacent sampled rows would let the small names
    share pages. The holes cost no writes. ``paths`` (the caller's dict, if given)
    is filled as files are created, so a failed write can still be cleaned up.
    """
    paths = {} if paths is None else paths
    for spec in specs:
        path = os.path.join(work_dir, f"layer{layer}.{spec.name}.bin")
        paths[spec.name] = path
        rows = truth[spec.name].view(torch.uint8).reshape(len(experts), -1)
        with open(path, "wb") as f:
            f.truncate(num_experts * spec.row_bytes)
            for i, expert in enumerate(experts):
                f.seek(expert * spec.row_bytes)
                f.write(rows[i].numpy().tobytes())
            f.flush()
            os.fsync(f.fileno())
        _drop_cache(path)
    return paths


def run(
    expert_dir: str,
    layer: int,
    rows: int,
    batches: list[int],
    work_dir: str,
    *,
    direct: bool = True,
    seed: int = 0,
    max_read_gb: float = 3.5,
    max_write_gb: float = 0.35,
    require_same_drive: bool = True,
    repeats: int = DEFAULT_REPEATS,
    gate_batch: int = GATE_BATCH,
) -> list[dict]:
    if rows < 1:
        raise SystemExit("rows must be at least 1")
    if repeats < 1:
        raise SystemExit("repeats must be at least 1")
    if not batches or any(not 1 <= batch <= rows for batch in batches):
        raise SystemExit(f"every batch must be in [1, rows={rows}]")
    layout = build_exl3_expert_layout(expert_dir)
    fmt = Exl3ExpertFormat(layout, layer, direct=direct)
    specs = fmt.tensor_specs(None)
    reader = Exl3RowReader(layout, direct=direct)
    streamed_bytes = sum(spec.row_bytes for spec in specs)
    planned = planned_read_bytes(
        reader.buffer_bytes, streamed_bytes, rows, batches,
        warmup_rows=1, repeats=repeats, gate_batch=gate_batch,
    )
    if planned > max_read_gb * 1e9:
        raise SystemExit(
            f"this run would read {planned / 1e9:.2f} GB (repeats {repeats} at batch "
            f"{gate_batch}), over --max-read-gb {max_read_gb}"
        )
    if rows * streamed_bytes > max_write_gb * 1e9:
        raise SystemExit(
            f"the repacked copy would write {rows * streamed_bytes / 1e9:.2f} GB, "
            f"over --max-write-gb {max_write_gb}"
        )
    if rows > layout.num_experts:
        raise SystemExit(f"layer {layer} has only {layout.num_experts} experts")
    experts = sorted(random.Random(seed).sample(range(layout.num_experts), rows))
    keys = [(layer, expert) for expert in experts]
    if require_same_drive:
        _check_work_dir(work_dir, layout.records[keys[0]].path)
    context = {
        "layer": layer,
        "queue_depth": int(envs.SGLANG_URING_FILE_READER_QUEUE_DEPTH.get()),
    }
    file_bytes = superset_file_bytes(layout, keys)
    payload_bytes = rows * streamed_bytes

    created = not os.path.exists(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    paths = {}
    results = []
    try:
        # The repacked copy: this layer's sampled rows, one file per streamed name.
        # Written inside the try, so a failed write still removes what it left.
        truth = _slabs(specs, rows)
        Exl3ShardRowSource(reader, layer, fmt.segment_map()).read(torch.tensor(experts), truth)
        _write_repacked(work_dir, layer, specs, truth, experts, layout.num_experts, paths)
        uring = shared_uring_file_reader()
        repacked = {
            spec.name: AlignedRowSource(
                uring, paths[spec.name], spec.row_bytes, layout.num_experts, direct=direct
            )
            for spec in specs
        }
        for path in {layout.records[key].path for key in keys}:
            reader._file(path)  # opens the shard (and sizes it) before any clock starts
        ids0 = torch.zeros(1, dtype=torch.int64)
        ids0 = torch.zeros(1, dtype=torch.int64)
        for batch in batches:
            passes = repeats if batch == gate_batch else 1
            # superset_raw: single pass, context for the gate
            bounce = _page_aligned(batch * reader.buffer_bytes).view(batch, reader.buffer_bytes)
            reader.read(keys[:1], [bounce[0].data_ptr()])  # untimed warm read
            if not direct:
                drop_superset_ranges(layout, keys)
            began = time.perf_counter()
            for start in range(0, rows, batch):
                chunk = keys[start : start + batch]
                reader.read(chunk, [bounce[i].data_ptr() for i in range(len(chunk))])
            raw_seconds = time.perf_counter() - began
            results.append(
                _row("superset_raw", batch, rows, [raw_seconds], payload_bytes, file_bytes, direct, context)
            )

            # Untimed warmup of both arms, then ``passes`` timed passes of each in turn
            # (split, repacked, split, ...) so drift and cache state hit both alike.
            split_slabs = _slabs(specs, rows)
            source = Exl3ShardRowSource(reader, layer, fmt.segment_map(), bounce_rows=batch)
            source.bounce.zero_()  # the shared bounce is only as warm as its last user
            source.read(torch.tensor(experts[:1]), split_slabs)
            repack_slabs = _slabs(specs, rows)
            read_plans(uring, [repacked[n].plan(torch.tensor(experts[:1]), repack_slabs[n], ids0) for n in repack_slabs])
            split_seconds, repack_seconds, split_stats = [], [], []
            deterministic = verified = True
            for _ in range(passes):
                # Untimed: the pass must fill zeroed (already faulted) slabs, cold caches.
                for slab in split_slabs.values():
                    slab.zero_()
                if not direct:
                    drop_superset_ranges(layout, keys)
                began = time.perf_counter()
                stats = source.read(torch.tensor(experts), split_slabs)
                split_seconds.append(time.perf_counter() - began)
                split_stats.append(stats)
                deterministic &= all(
                    torch.equal(split_slabs[n].view(torch.uint8), truth[n].view(torch.uint8))
                    for n in split_slabs
                )

                for slab in repack_slabs.values():
                    slab.zero_()
                for path in paths.values():
                    _drop_cache(path)
                began = time.perf_counter()
                for start in range(0, rows, batch):
                    file_rows = torch.tensor(experts[start : start + batch], dtype=torch.int64)
                    slots = torch.arange(start, start + file_rows.numel(), dtype=torch.int64)
                    read_plans(uring, [repacked[n].plan(file_rows, repack_slabs[n], slots) for n in repack_slabs])
                repack_seconds.append(time.perf_counter() - began)
                verified &= all(
                    torch.equal(repack_slabs[n].view(torch.uint8), truth[n].view(torch.uint8))
                    for n in repack_slabs
                )

            row = _row("superset_split", batch, rows, split_seconds, payload_bytes, file_bytes, direct, context)
            row["read_ms_per_row"] = statistics.median(st.read_ns for st in split_stats) / 1e6 / rows
            row["split_ms_per_row"] = statistics.median(st.split_ns for st in split_stats) / 1e6 / rows
            row["split_share"] = statistics.median(
                st.split_ns / max(st.read_ns + st.split_ns, 1) for st in split_stats
            )
            row["deterministic"] = deterministic
            results.append(row)
            split_row = row

            buffered = _buffered_names(repacked, repack_slabs)
            row = _row(
                "repacked_per_name", batch, rows, repack_seconds, payload_bytes, payload_bytes,
                direct and not buffered, context,
            )
            row["buffered_names"] = buffered
            row["verified"] = verified
            # The gate: repacked over split, both medians (<= 0.8 means repack wins).
            row["ratio_median"] = row["median_ms_per_row"] / split_row["median_ms_per_row"]
            results.append(row)
    finally:
        for path in paths.values():
            if os.path.exists(path):
                os.unlink(path)
        # Remove the scratch dir only if this run made it and nothing else is in it.
        if created and os.path.isdir(work_dir) and not os.listdir(work_dir):
            os.rmdir(work_dir)
    return results


def _row(
    mode: str,
    batch: int,
    rows: int,
    samples: list[float],
    payload_bytes: int,
    file_bytes: int,
    direct: bool,
    context: dict,
) -> dict:
    """One output row from ``samples`` (seconds of each timed pass); the row's
    ``seconds``, ``ms_per_row`` and byte rates are the median pass's."""
    seconds = statistics.median(samples)
    ms = [sample * 1e3 / rows for sample in samples]
    return {
        "mode": mode,
        "batch": batch,
        "rows": rows,
        **context,
        "direct": bool(direct),
        "repeats": len(samples),
        "samples_ms_per_row": ms,
        "median_ms_per_row": statistics.median(ms),
        "min_ms_per_row": min(ms),
        "seconds": seconds,
        "ms_per_row": seconds * 1e3 / rows,
        "bytes_basis": "streamed_payload",
        "payload_bytes": payload_bytes,
        "gb_per_s": payload_bytes / seconds / 1e9,
        "file_bytes": file_bytes,
        "file_gb_per_s": file_bytes / seconds / 1e9,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert-dir", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--batch", type=int, nargs="+", default=DEFAULT_BATCHES)
    p.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"timed passes of the split and repacked arms at batch {GATE_BATCH}, interleaved",
    )
    p.add_argument("--work-dir", required=True, help="scratch dir on the drive under test")
    p.add_argument("--max-read-gb", type=float, default=3.5)
    p.add_argument("--max-write-gb", type=float, default=0.35)
    p.add_argument("--buffered", action="store_true")
    args = p.parse_args()
    for row in run(
        args.expert_dir,
        args.layer,
        args.rows,
        args.batch,
        args.work_dir,
        direct=not args.buffered,
        max_read_gb=args.max_read_gb,
        max_write_gb=args.max_write_gb,
        repeats=args.repeats,
    ):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
