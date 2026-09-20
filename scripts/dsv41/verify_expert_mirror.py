"""Verify that mirror copies of an EXL3 checkpoint hold the source's bytes.

The mirror row source (``SGLANG_MOE_EXPERT_MIRROR_DIRS``) trusts each root to
hold a byte-identical copy of the checkpoint. Its own guards are O(1): a size
check. A copy of the right size that is only partly filled (an interrupted or
preallocated copy, a sparse hole) passes them and reads back zeros. This tool
is what the copies are trusted on. It compares content, and it says exactly
what it compared.

What is compared, for every root separately:

* **Sizes** of every file the layout reads, against the source (always all of
  them, whatever else is chosen; it costs one ``stat`` per file and root). A
  missing file, a different size, and a mirror file that is the same file as
  the source (a symlink or hard link, which would verify nothing) are all
  reported.
* **Content**: every chosen expert row, read through ``Exl3MirrorRowSource``
  (one root at a time, so a mismatch names its drive) and through
  ``Exl3ShardRowSource`` on the source, compared byte for byte over the six
  streamed tensors of the row. The source is read once per row, however many
  roots there are. The three 4-byte ``mul1`` scalars of a row are not read by
  serving and so are not compared, and files outside the layout (attention
  shards, the index) are not looked at.

"Verified" therefore means every byte the runtime reads matches, not that the
files are identical. That is safe while the layout (offsets, headers, the
index) is built from the source directory. If ``SGLANG_DSV41_EXPERT_DIR`` is
ever pointed at a mirror, the layout would be built from headers and an index
this tool never compared: verify with a whole-file comparison first.

The default is a FULL verification: every expert row of every layer, with
O_DIRECT reads, so the bytes come from the drives and not the page cache.
``--layers`` and ``--sample-experts`` bound the work for a fast pre-check; the
output then says PARTIAL, gives the coverage, and the exit status is not 0.
``--buffered`` reads through the page cache (for tests, never for a verdict on
a drive) and is likewise never a clean pass.

Exit status: 0 only for a FULL, direct-read pass; 2 if anything mismatched;
3 if what ran passed but was not a full direct-read verification; 1 for a
usage or configuration error.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.exl3_expert_format import (
    Exl3ExpertFormat,
    parse_mirror_roots,
)
from sglang.srt.layers.moe.exl3_expert_layout import (
    Exl3ExpertLayout,
    build_exl3_expert_layout,
)
from sglang.srt.layers.moe.exl3_mirror_row_source import Exl3MirrorRowSource
from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource

# Rows compared per step: two buffers of 16 x 13.3 MB on the real checkpoint.
BATCH_ROWS = 16
# Bytes of the source and the mirror shown around a mismatch.
CONTEXT_BYTES = 16

EXIT_OK, EXIT_USAGE, EXIT_MISMATCH, EXIT_NOT_COMPLETE = 0, 1, 2, 3


@dataclass(frozen=True)
class SizeProblem:
    """A layout file whose copy in ``root`` is not the source's file.

    ``kind``: ``missing``, ``unreadable`` (``detail`` is the OS error),
    ``size`` or ``same-file`` (the mirror path is the source file itself).
    """

    root: str
    path: str  # the source file
    mirror_path: str
    source_bytes: int
    mirror_bytes: Optional[int]
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class RowMismatch:
    """The first differing byte of one expert row, and what was found there."""

    root: str
    layer: int
    expert: int
    source_path: str
    mirror_path: str
    file_offset: int  # absolute, in both files
    row_offset: int  # from the start of the expert's on-disk row
    tensor: str  # the on-disk tensor holding it, e.g. w1.trellis
    tensor_offset: int
    expected: bytes  # source bytes from the mismatch on, up to CONTEXT_BYTES
    found: bytes  # the mirror's, same length
    bad_bytes_in_row: int  # streamed bytes of this row that differ
    mirror_row_all_zero: bool  # every streamed byte the mirror gave was 0


@dataclass(frozen=True)
class ReadFailure:
    root: str
    layer: int
    message: str


@dataclass
class RootReport:
    root: str
    size_problems: list[SizeProblem] = field(default_factory=list)
    mismatches: list[RowMismatch] = field(default_factory=list)
    read_failures: list[ReadFailure] = field(default_factory=list)
    layers_skipped: list[int] = field(default_factory=list)
    rows_checked: int = 0
    bytes_compared: int = 0
    read_seconds: float = 0.0
    stopped_early: bool = False

    @property
    def ok(self) -> bool:
        return not (self.size_problems or self.mismatches or self.read_failures)


@dataclass
class VerifyResult:
    source: str
    num_layers: int
    num_experts: int
    layers: tuple[int, ...]
    sampled: dict[int, tuple[int, ...]]  # the experts compared, per chosen layer
    sample_experts: Optional[int]
    seed: int
    direct: bool
    keep_going: bool
    files_checked: int
    streamed_row_bytes: int
    root_reports: list[RootReport]
    elapsed_seconds: float = 0.0

    @property
    def full(self) -> bool:
        """Every expert row of every layer was in the plan."""
        return len(self.layers) == self.num_layers and all(
            len(experts) == self.num_experts for experts in self.sampled.values()
        )

    @property
    def ok(self) -> bool:
        return all(report.ok for report in self.root_reports)

    @property
    def complete(self) -> bool:
        """A pass that is a full verification of the drives' contents."""
        return self.ok and self.full and self.direct

    @property
    def exit_code(self) -> int:
        if not self.ok:
            return EXIT_MISMATCH
        return EXIT_OK if self.complete else EXIT_NOT_COMPLETE


# --- Selecting what to compare -------------------------------------------------


def parse_layers(spec: str) -> Optional[list[int]]:
    """``all`` (None), or ``0,3,10-12`` as a sorted list. Whether each layer
    exists is checked by ``verify_mirror``, which names the one that does not."""
    if spec.strip() == "all":
        return None
    layers: set[int] = set()
    for token in spec.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", token)
        if match is None or (
            match.group(2) is not None and int(match.group(2)) < int(match.group(1))
        ):
            raise ValueError(
                f"invalid layers {spec!r}: {token!r} is not a layer or an "
                "ascending range like 3 or 10-12; use 'all' or a comma-separated list"
            )
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        layers.update(range(first, last + 1))
    return sorted(layers)


def _sample(layer: int, num_experts: int, count: int, seed: int) -> tuple[int, ...]:
    """``count`` experts of ``layer``, sorted. Seeded per layer, so a layer's
    sample does not depend on which other layers are chosen."""
    if count >= num_experts:
        return tuple(range(num_experts))
    return tuple(
        sorted(random.Random(f"{seed}:{layer}").sample(range(num_experts), count))
    )


def _ranges(values: Sequence[int]) -> str:
    """[0, 1, 2, 5] -> '0-2,5'."""
    parts, values = [], sorted(values)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[end] + 1:
            end += 1
        parts.append(
            str(values[start]) if end == start else f"{values[start]}-{values[end]}"
        )
        start = end + 1
    return ",".join(parts)


# --- Sizes ------------------------------------------------------------------------


def _mirror_path(root: str, source: str, path: str) -> str:
    return os.path.join(root, os.path.relpath(path, source))


def check_sizes(
    layout: Exl3ExpertLayout, source: str, roots: Sequence[str]
) -> tuple[dict[str, list[SizeProblem]], int]:
    """Every problem with a root's copy of every layout file, in path order,
    and the number of files checked."""
    paths = sorted({record.path for record in layout.records.values()})
    source_stat = {path: os.stat(path) for path in paths}
    problems: dict[str, list[SizeProblem]] = {root: [] for root in roots}
    for root in roots:
        for path in paths:
            mirror = _mirror_path(root, source, path)
            expected = source_stat[path]
            try:
                found = os.stat(mirror)
            except FileNotFoundError:
                problems[root].append(
                    SizeProblem(root, path, mirror, expected.st_size, None, "missing")
                )
                continue
            except OSError as error:
                problems[root].append(
                    SizeProblem(
                        root,
                        path,
                        mirror,
                        expected.st_size,
                        None,
                        "unreadable",
                        str(error),
                    )
                )
                continue
            if (found.st_dev, found.st_ino) == (expected.st_dev, expected.st_ino):
                kind = "same-file"
            elif found.st_size != expected.st_size:
                kind = "size"
            else:
                continue
            problems[root].append(
                SizeProblem(root, path, mirror, expected.st_size, found.st_size, kind)
            )
    return problems, len(paths)


# --- Content ------------------------------------------------------------------------


class _Comparer:
    """Reads a batch of one layer's rows from the source and from a mirror and
    compares the six streamed tensors byte for byte."""

    def __init__(self, layout: Exl3ExpertLayout, source: str, batch_rows: int) -> None:
        fmt = Exl3ExpertFormat(layout, 0, direct=False)
        self.layout = layout
        self.source = source
        self.segments = fmt.segment_map()
        self.specs = fmt.tensor_specs(None)
        self.names = tuple(spec.name for spec in self.specs)
        self.streamed_row_bytes = sum(segment.nbytes for segment in self.segments)
        self.batch_rows = batch_rows
        self._spans = sorted(layout.tensors, key=lambda span: span.rel_offset)
        self._source_buffers = self._buffers()
        self._mirror_buffers = self._buffers()

    def _buffers(self) -> dict[str, torch.Tensor]:
        return {
            spec.name: torch.zeros(
                (self.batch_rows,) + tuple(spec.row_shape), dtype=spec.dtype
            )
            for spec in self.specs
        }

    def read_source(self, row_source, experts: Sequence[int]) -> None:
        self._read(row_source, self._source_buffers, experts)

    def read_mirror(self, row_source, experts: Sequence[int]) -> float:
        """Seconds the read took."""
        return self._read(row_source, self._mirror_buffers, experts)

    @staticmethod
    def _read(row_source, buffers, experts: Sequence[int]) -> float:
        # Zero first: a read that silently skipped bytes must not leave the
        # previous batch's (or root's) correct bytes behind. That includes the
        # bounce ring, which the source read and every mirror read share.
        for buffer in buffers.values():
            buffer.zero_()
        row_source.bounce.zero_()
        began = time.perf_counter()
        row_source.read(torch.tensor(list(experts), dtype=torch.int64), buffers)
        return time.perf_counter() - began

    @staticmethod
    def _flat(buffer: torch.Tensor, rows: int) -> torch.Tensor:
        return buffer[:rows].contiguous().view(torch.uint8).reshape(rows, -1)

    def compare(
        self, root: str, layer: int, experts: Sequence[int], first_only: bool
    ) -> list[RowMismatch]:
        """Mismatching rows of the batch in the source and mirror buffers, in
        row order; with ``first_only`` at most the first."""
        rows = len(experts)
        source = {name: self._flat(b, rows) for name, b in self._source_buffers.items()}
        mirror = {name: self._flat(b, rows) for name, b in self._mirror_buffers.items()}
        differs = {}
        for name in self.names:
            if not torch.equal(source[name], mirror[name]):
                differs[name] = source[name] != mirror[name]
        if not differs:
            return []
        bad_rows = torch.stack([d.any(dim=1) for d in differs.values()]).any(dim=0)
        found = []
        for index in torch.nonzero(bad_rows).flatten().tolist():
            found.append(
                self._describe(
                    root, layer, experts[index], index, source, mirror, differs
                )
            )
            if first_only:
                break
        return found

    def _describe(
        self, root, layer, expert, index, source, mirror, differs
    ) -> RowMismatch:
        record = self.layout.records[(layer, expert)]
        best = None  # (row_offset, name, dst_offset, segment)
        bad_bytes = 0
        for name, mask in differs.items():
            row_mask = mask[index]
            count = int(row_mask.sum())
            if not count:
                continue
            bad_bytes += count
            dst = int(torch.nonzero(row_mask)[0])
            segment = next(
                s
                for s in self.segments
                if s.name == name and s.dst_offset <= dst < s.dst_offset + s.nbytes
            )
            row_offset = segment.src_offset + (dst - segment.dst_offset)
            if best is None or row_offset < best[0]:
                best = (row_offset, name, dst, segment)
        row_offset, name, dst, segment = best
        span = next(
            s
            for s in self._spans
            if s.rel_offset <= row_offset < s.rel_offset + s.nbytes
        )
        end = min(dst + CONTEXT_BYTES, segment.dst_offset + segment.nbytes)
        return RowMismatch(
            root=root,
            layer=layer,
            expert=expert,
            source_path=record.path,
            mirror_path=_mirror_path(root, self.source, record.path),
            file_offset=record.file_offset + row_offset,
            row_offset=row_offset,
            tensor=span.name,
            tensor_offset=row_offset - span.rel_offset,
            expected=bytes(source[name][index, dst:end].tolist()),
            found=bytes(mirror[name][index, dst:end].tolist()),
            bad_bytes_in_row=bad_bytes,
            mirror_row_all_zero=not any(bool(m[index].any()) for m in mirror.values()),
        )


def verify_mirror(
    source_dir: str,
    roots: Sequence[str],
    *,
    layers: Optional[Sequence[int]] = None,
    sample_experts: Optional[int] = None,
    seed: int = 0,
    direct: bool = True,
    keep_going: bool = False,
    batch_rows: int = BATCH_ROWS,
    log: Optional[Callable[[str], None]] = None,
) -> VerifyResult:
    """Compare every root's copy against ``source_dir``.

    Defaults to every expert row of every layer read with O_DIRECT. ``layers``
    and ``sample_experts`` (per layer, seeded) shrink that; ``VerifyResult.full``
    and ``.complete`` say whether they did. A root stops at its first mismatch
    unless ``keep_going``. Other roots are always still compared.
    """
    log = log or (lambda _line: None)
    started = time.perf_counter()
    roots = list(roots)
    if not roots:
        raise ValueError(
            "no mirror roots given: pass --roots or set SGLANG_MOE_EXPERT_MIRROR_DIRS"
        )
    source = os.path.realpath(source_dir)
    real = [os.path.realpath(root) for root in roots]
    for root, path in zip(roots, real):
        if not os.path.isdir(root):
            raise ValueError(f"mirror root {root} is not a directory")
        if path == source:
            raise ValueError(
                f"mirror root {root} is the same directory as the source {source}; "
                "comparing a directory with itself verifies nothing"
            )
    for i, path in enumerate(real):
        if path in real[:i]:
            first = roots[real.index(path)]
            raise ValueError(
                f"mirror root {roots[i]} is the same directory as mirror root "
                f"{first}; a copy compared twice verifies nothing new"
            )

    layout = build_exl3_expert_layout(source)
    if layers is None:
        chosen = tuple(range(layout.num_layers))
    else:
        chosen = tuple(sorted(set(layers)))
        if not chosen:
            raise ValueError("no layers selected")
        for layer in chosen:
            if not 0 <= layer < layout.num_layers:
                raise ValueError(
                    f"layer {layer} is outside the checkpoint's {layout.num_layers} "
                    f"layers (0-{layout.num_layers - 1})"
                )
    if sample_experts is not None and sample_experts < 1:
        raise ValueError(f"--sample-experts must be at least 1, got {sample_experts}")
    sampled = {
        layer: (
            tuple(range(layout.num_experts))
            if sample_experts is None
            else _sample(layer, layout.num_experts, sample_experts, seed)
        )
        for layer in chosen
    }

    size_problems, files_checked = check_sizes(layout, source, roots)
    reports = {
        root: RootReport(root, size_problems=size_problems[root]) for root in roots
    }
    comparer = _Comparer(layout, source, batch_rows)
    layer_paths: dict[int, set[str]] = {}
    for (layer, _expert), record in layout.records.items():
        layer_paths.setdefault(layer, set()).add(record.path)

    result = VerifyResult(
        source=source,
        num_layers=layout.num_layers,
        num_experts=layout.num_experts,
        layers=chosen,
        sampled=sampled,
        sample_experts=sample_experts,
        seed=seed,
        direct=direct,
        keep_going=keep_going,
        files_checked=files_checked,
        streamed_row_bytes=comparer.streamed_row_bytes,
        root_reports=[],
    )
    log(render_header(result))
    log("")
    for root in roots:
        problems = len(reports[root].size_problems)
        log(
            f"sizes {root}: "
            + (
                f"{problems} of {files_checked} files wrong"
                if problems
                else f"all {files_checked} files match"
            )
        )

    for layer in chosen:
        live, mirrors = [], {}
        for root in roots:
            report = reports[root]
            if report.stopped_early:
                continue
            bad = {p.path for p in report.size_problems}
            if layer_paths[layer] & bad:
                # The mirror source refuses these files; say so, do not pass them.
                report.layers_skipped.append(layer)
                log(f"layer {layer:>3} {root}: skipped, a shard fails the size check")
                continue
            try:
                mirrors[root] = Exl3MirrorRowSource.for_mirrored_layer(
                    layout,
                    layer,
                    comparer.segments,
                    direct=direct,
                    roots=[root],
                    policy=StaticSplitPolicy([1.0]),
                    source_root=source,
                    bounce_rows=8,
                )
            except (OSError, RuntimeError, ValueError) as error:
                report.read_failures.append(ReadFailure(root, layer, str(error)))
                report.stopped_early = not keep_going
                log(f"layer {layer:>3} {root}: cannot open: {error}")
                continue
            live.append(root)
        if not live:
            continue
        source_rows = Exl3ShardRowSource.for_layer(
            layout, layer, comparer.segments, direct=direct
        )
        experts = sampled[layer]
        before = {
            root: (reports[root].rows_checked, len(reports[root].mismatches))
            for root in live
        }
        for start in range(0, len(experts), batch_rows):
            batch = experts[start : start + batch_rows]
            live = [root for root in live if not reports[root].stopped_early]
            if not live:
                break
            comparer.read_source(source_rows, batch)
            for root in live:
                report = reports[root]
                try:
                    report.read_seconds += comparer.read_mirror(mirrors[root], batch)
                except (OSError, RuntimeError, ValueError) as error:
                    report.read_failures.append(ReadFailure(root, layer, str(error)))
                    report.stopped_early = not keep_going
                    log(f"layer {layer:>3} {root}: read failed: {error}")
                    continue
                found = comparer.compare(root, layer, batch, first_only=not keep_going)
                report.mismatches.extend(found)
                if found and not keep_going:
                    counted = batch.index(found[0].expert) + 1
                    report.stopped_early = True
                else:
                    counted = len(batch)
                report.rows_checked += counted
                report.bytes_compared += counted * comparer.streamed_row_bytes
        for root in before:
            report = reports[root]
            rows = report.rows_checked - before[root][0]
            bad_rows = len(report.mismatches) - before[root][1]
            state = "MISMATCH" if bad_rows else "ok"
            log(f"layer {layer:>3} {root}: {rows} rows compared, {state}")

    result.root_reports.extend(reports[root] for root in roots)
    result.elapsed_seconds = time.perf_counter() - started
    return result


# --- Report ----------------------------------------------------------------------------


def _gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:.3f} GB"


def _reads_tag(result: VerifyResult) -> str:
    return "" if result.direct else ", BUFFERED (not a drive verification)"


def _pass_label(result: VerifyResult) -> str:
    """PASS, qualified when what ran was not a full direct-read verification."""
    qualifiers = ([] if result.full else ["partial"]) + (
        [] if result.direct else ["buffered"]
    )
    return "PASS" + (f" ({', '.join(qualifiers)})" if qualifiers else "")


def _mode_line(result: VerifyResult) -> str:
    rows = sum(len(experts) for experts in result.sampled.values())
    total = result.num_layers * result.num_experts
    per_root = f"{rows} of {total} expert rows ({_gb(rows * result.streamed_row_bytes)}) per root"
    if result.full:
        return (
            f"MODE: FULL{_reads_tag(result)}: every expert row of every layer, "
            f"{per_root}"
        )
    layers = (
        f"all {result.num_layers} layers"
        if len(result.layers) == result.num_layers
        else f"layers {_ranges(result.layers)} of {result.num_layers}"
    )
    experts = (
        "every expert"
        if result.sample_experts is None or result.sample_experts >= result.num_experts
        else f"{result.sample_experts} of {result.num_experts} experts per layer "
        f"(seed {result.seed})"
    )
    return (
        f"MODE: PARTIAL{_reads_tag(result)}: {layers}; {experts}; {per_root}. "
        "This is NOT a full verification."
    )


def _size_line(problem: SizeProblem) -> str:
    if problem.kind == "missing":
        return f"missing: {problem.mirror_path} does not exist (source {problem.path}, {problem.source_bytes} bytes)"
    if problem.kind == "unreadable":
        return f"unreadable: {problem.mirror_path}: {problem.detail}"
    if problem.kind == "same-file":
        return (
            f"same file: {problem.mirror_path} is the same file as the source "
            f"{problem.path} (a symlink or hard link), so it verifies nothing"
        )
    delta = problem.mirror_bytes - problem.source_bytes
    return (
        f"size mismatch: {problem.mirror_path} is {problem.mirror_bytes} bytes, "
        f"its source {problem.path} is {problem.source_bytes} bytes "
        f"({'long' if delta > 0 else 'short'} by {abs(delta)})"
    )


def _mismatch_lines(m: RowMismatch, indent: str = "    ") -> list[str]:
    lines = [
        f"{indent}mirror file : {m.mirror_path}",
        f"{indent}source file : {m.source_path}",
        f"{indent}file offset : {m.file_offset} (0x{m.file_offset:x})",
        f"{indent}row offset  : {m.row_offset} (0x{m.row_offset:x}) into the expert's "
        f"row = {m.tensor} + {m.tensor_offset}",
        f"{indent}expected    : {m.expected.hex(' ')}  (source, {len(m.expected)} bytes from there)",
        f"{indent}found       : {m.found.hex(' ')}  (mirror)",
        f"{indent}streamed bytes of this row that differ: {m.bad_bytes_in_row}",
    ]
    if m.mirror_row_all_zero:
        lines.append(
            f"{indent}every streamed byte the mirror gave for this row is 0: "
            "an unfilled (sparse or preallocated) copy"
        )
    return lines


def render_header(result: VerifyResult) -> str:
    """What is about to run: source, mode, read mode, what is compared."""
    return "\n".join(
        [
            f"source: {result.source} ({result.num_layers} layers x {result.num_experts} experts)",
            _mode_line(result),
            "reads: "
            + (
                "O_DIRECT (page cache bypassed; the bytes came from the drives)"
                if result.direct
                else "BUFFERED (through the page cache: says nothing about what is on the "
                "drives; never a verdict on a copy)"
            ),
            f"compared: sizes of all {result.files_checked} files the layout reads, and each "
            "chosen row's six streamed tensors byte for byte (the 3 mul1 scalars per row are "
            "never read by serving and are not compared; files outside the layout are not "
            "looked at)",
        ]
    )


def render(result: VerifyResult, header: bool = True) -> str:
    """The report as text: the header (unless the caller already printed it),
    one section per root, a verdict."""
    out = [render_header(result), ""] if header else []
    for report in result.root_reports:
        out.append(
            f"ROOT {report.root}: {_pass_label(result) if report.ok else 'FAIL'}"
        )
        if report.size_problems:
            out.append(
                f"  sizes: {len(report.size_problems)} of {result.files_checked} files wrong:"
            )
            out += [f"    - {_size_line(p)}" for p in report.size_problems]
        else:
            out.append(f"  sizes: all {result.files_checked} files match the source")
        out.append(
            f"  content: {report.rows_checked} rows compared "
            f"({_gb(report.bytes_compared)}), "
            f"{len(report.mismatches)} mismatching, "
            f"read {report.read_seconds:.1f} s"
            + (
                "; STOPPED at the first mismatch, the rows after it were not compared"
                if report.stopped_early and report.mismatches
                else ""
            )
        )
        if report.layers_skipped:
            out.append(
                f"  skipped {len(report.layers_skipped)} layer(s), not compared because "
                "a shard they read fails the size check above: "
                + ", ".join(f"layer {layer}" for layer in report.layers_skipped)
            )
        for failure in report.read_failures:
            out.append(f"  READ FAILED at layer {failure.layer}: {failure.message}")
        if report.mismatches:
            first = report.mismatches[0]
            out.append(f"  FIRST MISMATCH: layer {first.layer}, expert {first.expert}")
            out += _mismatch_lines(first)
            for other in report.mismatches[1:21]:
                out.append(
                    f"  also: layer {other.layer}, expert {other.expert}, row offset "
                    f"{other.row_offset}, file offset {other.file_offset} "
                    f"({other.bad_bytes_in_row} bytes differ)"
                )
            if len(report.mismatches) > 21:
                out.append(
                    f"  ... and {len(report.mismatches) - 21} more mismatching rows"
                )
        out.append("")
    scope = _mode_line(result).split(":")[1].strip()
    if not result.ok:
        bad = [r.root for r in result.root_reports if not r.ok]
        out.append(
            f"VERDICT: FAILED [{scope}]: {len(bad)} of {len(result.root_reports)} "
            f"roots do not match the source: {', '.join(bad)}"
        )
    elif result.complete:
        out.append(
            f"VERDICT: VERIFIED, meaning every byte the runtime reads matches the "
            f"source: the six streamed tensors of every expert row of every layer, on "
            f"all {len(result.root_reports)} roots, read with O_DIRECT, and every "
            "layout file's size.\n"
            "  It does NOT mean the files are identical: the mul1 scalars, safetensors "
            "headers, alignment padding, the index and files outside the layout were "
            "not compared.\n"
            "  If SGLANG_DSV41_EXPERT_DIR is ever pointed at a mirror, the layout would "
            "be built from headers and an index this tool never checked."
        )
    elif not result.full:
        out.append(
            "VERDICT: PASSED A PARTIAL CHECK. This is NOT a full verification: the rows "
            "and layers not chosen were not compared. Do not trust these copies on this "
            "evidence alone."
            + (
                ""
                if result.direct
                else " The reads were also BUFFERED, so even the rows compared say "
                "nothing about the drives."
            )
        )
    else:
        out.append(
            "VERDICT: PASSED, BUT THE READS WERE BUFFERED. The page cache, not the drives, "
            "may have served these bytes; rerun without --buffered before trusting the copies."
        )
    out.append(f"elapsed: {result.elapsed_seconds:.1f} s")
    return "\n".join(out)


# --- Command line ----------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Exit status: 0 full direct-read pass, 2 mismatch, 3 passed but not "
        "a full direct-read verification, 1 usage or configuration error.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="the source checkpoint directory (default: SGLANG_DSV41_EXPERT_DIR)",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="the mirror roots (default: SGLANG_MOE_EXPERT_MIRROR_DIRS)",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="'all' (default), or layers such as 0,3,10-12 for a PARTIAL check",
    )
    parser.add_argument(
        "--sample-experts",
        type=int,
        default=None,
        help="compare only N random experts per layer (PARTIAL); default every expert",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    parser.add_argument(
        "--buffered",
        action="store_true",
        help="read through the page cache instead of O_DIRECT (tests only; never a clean pass)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="do not stop a root at its first mismatch; compare and list every bad row",
    )
    parser.add_argument(
        "--batch-rows", type=int, default=BATCH_ROWS, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        source = args.source or envs.SGLANG_DSV41_EXPERT_DIR.get()
        if not source:
            raise ValueError(
                "no source given: pass --source or set SGLANG_DSV41_EXPERT_DIR"
            )
        roots = args.roots
        if roots is None:
            dirs = envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.get()
            roots = list(parse_mirror_roots(dirs)) if dirs else []
        layers = parse_layers(args.layers)
        result = verify_mirror(
            source,
            roots,
            layers=layers,
            sample_experts=args.sample_experts,
            seed=args.seed,
            direct=not args.buffered,
            keep_going=args.keep_going,
            batch_rows=args.batch_rows,
            log=lambda line: print(line, flush=True),
        )
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    print()
    print(render(result, header=False), flush=True)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
