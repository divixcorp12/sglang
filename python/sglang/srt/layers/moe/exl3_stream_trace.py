"""G and f for streamed EXL3 experts, from the expert streaming framework's gather stats.

G is VRAM misses per decode token, f the share of those that also miss RAM and
read the shards (DSV41_REFERENCE §9.4). Each streamed MoE layer call reports
its gather's ``ExpertGatherStats``: ``miss_rows`` (rows not in the hot cache)
and ``host_read_rows`` (rows read through the row source, i.e. not in the
pinned host tier either). A new forward starts when a call's layer id is not
above the previous call's; a call with one token row is a decode call.
Optionally every call is appended to a JSONL trace for scripts/dsv41/tier_sim.py.
Each line carries a ``time.monotonic()`` stamp, so the gaps between forwards
show the residency-boundary promotion stalls that tok/s includes.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

LOG_EVERY_FORWARDS = 256


def capturing_graphs() -> bool:
    """True during the decode runner's warmup and capture forwards, False at replay.

    ``get_is_capture_mode()`` is also true at every breakable replay, so only the
    module flag that ``model_capture_mode()`` sets separates capture from serving.
    """
    from sglang.srt.model_executor.runner_utils import capture_mode

    return bool(capture_mode.is_capture_mode)


# Bumped whenever a ram_miss_request field changes meaning OR the record layout changes (a field
# added or removed): a JSONL file has no field check like the native record's, so this integer is the
# only mechanical answer to which fields a file has and which quantities it compares.
# 2: stage stamps and spans cover the whole read, not the first io_uring batch, and packing can fall
# inside first_to_last_cqe. 3: adds row_pack[].admit, extent_cqe[].submit/attempts and dropped_before;
# no schema-2 field changed meaning. 4: adds request.lanes, the planned lane count the device posted;
# every schema-3 field keeps its meaning, so spans, byte totals and per-drive shares compare across 3 and 4.
# 5: adds request.pack_workers and request.pack_split, the packing mode the reader ran in. No earlier field
# changed meaning, but a worker-mode record's pack stamps mean something else than an inline one's (a row's
# pack_start is after the worker woke, its spans overlap). A file written before schema 5 does not say which
# mode wrote it; it is ASSUMED inline, because the workers were barred from any run that writes a stage trace.
# 6: adds request.piece_stream, extent_cqe[].sub and pieces. 7: adds pieces[].publish and pieces_published,
# pieces_out_of_order and piece_publish_refused; with piece streaming a row's row_pack span covers its pieces' jobs.
RAM_MISS_TRACE_SCHEMA = 7


def _stream_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class GraphRouteLog:
    """Every graph forward's routed experts and VRAM misses per streamed layer, for the stage trace only.

    A replayed decode graph runs no Python, so its routes never reach ``Exl3StreamTrace.record``.
    ``record`` is called from each layer's in-graph gather (``Exl3RamMissRowBackend.post``) and is
    captured with it: the first streamed layer (row 0) takes the next ring slot and bumps ``seq``, then
    every layer copies its routes and its plan's miss count into that slot. Each forward thus writes
    its own entry, numbered by ``seq`` (the graph forwards that ran before it), whatever the scheduler's
    per-batch cadence: no per-batch read can fold or skip one.

    ``poll`` runs at the per-batch check. It reads the ring back through pinned buffers without waiting,
    one batch behind (like ``Exl3RamMissService._graph_rows``), and writes the entries finished since
    the last read. The copy is on the host's current stream, which need not be the stream the graph
    replays on, so an entry is trusted only when the ``seq`` read before the ring shows a later forward
    has started (entries below ``seq - 1``), and only if it is ``margin`` entries clear of the slots
    later forwards may overwrite while the ring copy runs. ``final`` (after shutdown's device barrier)
    reads everything.

    Warmup forwards before capture execute these copies too. ``warmup`` counts them, from the Python
    calls made in capture mode outside a stream capture (a capture records the copies but runs none),
    and no entry below it is written.
    """

    def __init__(self, layers: int, width: int, device, depth: int = 64, margin: int = 4) -> None:
        if depth <= margin + 1:
            raise ValueError("the route ring must be deeper than its safety margin")
        self.layers, self.width, self.depth, self.margin = layers, width, depth, margin
        self.layer_ids: list[int] = [-1] * layers
        self.capacities: list[int] = [0] * layers  # hot slots per row, for the replay's allocation
        self.routes = torch.full((depth, layers, width), -1, dtype=torch.int64, device=device)
        self.misses = torch.full((depth, layers), -1, dtype=torch.int32, device=device)
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)
        self.slot = torch.zeros(1, dtype=torch.int64, device=device)
        self.warmup = 0
        self.next_seq = 0
        self.dropped = 0
        self._header_written = False
        self._host: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._event = None
        self._pending = False

    def bind(self, row: int, layer_id: int, capacity: int) -> None:
        self.layer_ids[row] = int(layer_id)
        self.capacities[row] = int(capacity)

    def record(self, row: int, routes: torch.Tensor, count: torch.Tensor) -> None:
        """Log one layer's routes (``routes[:n]``, -1 past them) and plan miss count; captured in the graph."""
        if row == 0:
            if capturing_graphs() and not _stream_capturing():
                self.warmup += 1
            torch.remainder(self.seq, self.depth, out=self.slot)
            self.seq.add_(1)
        n = min(routes.numel(), self.width)
        self.routes[:, row, :n].index_copy_(0, self.slot, routes[:n].view(1, n))
        self.misses[:, row].index_copy_(0, self.slot, count.reshape(-1)[:1])

    def read_seq(self) -> int:
        """Graph forwards run so far. Called by eager forwards only, which already wait for the stream."""
        return int(self.seq.item())

    def tensors(self) -> list[torch.Tensor]:
        """Every buffer a copy may still be writing, for the service's quarantine."""
        owned = [self.routes, self.misses, self.seq, self.slot]
        return owned + (list(self._host) if self._host is not None else [])

    def poll(self, trace: "Exl3StreamTrace", *, final: bool = False) -> None:
        if self.seq.device.type != "cuda":
            self._emit(trace, int(self.seq[0]), self.routes, self.misses, complete=True)
            return
        if not final and torch.cuda.is_current_stream_capturing():
            return
        if self._pending:
            if final:
                self._event.synchronize()
            elif not self._event.query():
                return
            self._pending = False
            seq, routes, misses = self._host
            self._emit(trace, int(seq[0]), routes, misses, complete=False)
        if final:
            self._emit(trace, int(self.seq.item()), self.routes.cpu(), self.misses.cpu(), complete=True)
            return
        if self._host is None:
            self._host = tuple(torch.empty_like(t, device="cpu").pin_memory() for t in (self.seq, self.routes, self.misses))
            self._event = torch.cuda.Event(enable_timing=False)
        for host, device in zip(self._host, (self.seq, self.routes, self.misses)):
            host.copy_(device, non_blocking=True)  # seq first: the stream runs these in order
        self._event.record(torch.cuda.current_stream(self.seq.device))
        self._pending = True

    def _emit(self, trace: "Exl3StreamTrace", seq: int, routes: torch.Tensor, misses: torch.Tensor, *, complete: bool) -> None:
        if not self._header_written:
            trace.record_graph_routes_header(self)
            self._header_written = True
        end = seq if complete else seq - 1
        start = max(self.next_seq, self.warmup)
        oldest = max(seq - self.depth + (0 if complete else self.margin), 0)
        lost = max(oldest - start, 0)
        start += lost
        self.dropped += lost
        for s in range(start, end):
            slot = s % self.depth
            trace.record_graph_route_step(
                s,
                [[int(e) for e in row if e >= 0] for row in routes[slot].tolist()],
                [int(m) for m in misses[slot].tolist()],
                dropped_before=lost if s == start else 0,
            )
        self.next_seq = max(self.next_seq, end)


class Exl3StreamTrace:
    def __init__(self, path: str = "", log_every: int = LOG_EVERY_FORWARDS) -> None:
        # Line-buffered: the scheduler process may exit without running atexit.
        self._file = open(path, "a", buffering=1) if path else None
        self.log_every = log_every
        # Set by the RAM-miss service when it logs graph routes: stamps each eager forward with the
        # graph forwards that ran before it, so a replay can interleave the two (tier_sim.load_forwards).
        self.graph_seq_source: Optional[Callable[[], int]] = None
        self._graph_seq: Optional[int] = None
        self._last_layer: Optional[int] = None
        self._background: dict[int, int] = {}
        self.forwards = 0
        self.decode_tokens = 0
        self.decode_vram_misses = 0
        self.decode_ram_misses = 0
        self.vram_misses = 0
        self.ram_misses = 0
        self.read_ms = 0.0
        self.split_ms = 0.0
        self.background_read_rows = 0

    def record(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        stats: Optional[Any],
        background_read_rows: int,
    ) -> None:
        """Count one streamed MoE call; ``stats`` is its gather's stats, or None
        when the call gathered nothing. ``background_read_rows`` is the layer's
        cumulative count of rows read outside gathers (promotions, seeding)."""
        if capturing_graphs():
            return
        tokens = int(topk_ids.shape[0])
        if self._last_layer is None or layer_id <= self._last_layer:
            self.forwards += 1
            if tokens == 1:
                self.decode_tokens += 1
            if self.graph_seq_source is not None:
                self._graph_seq = self.graph_seq_source()
            if self.forwards % self.log_every == 0:
                self.log()
        self._last_layer = layer_id

        vram_miss = int(getattr(stats, "miss_rows", 0))
        ram_miss = int(getattr(stats, "host_read_rows", 0))
        read_ms = getattr(stats, "host_read_ns", 0) / 1e6
        split_ms = getattr(stats, "host_split_ns", 0) / 1e6
        background = int(background_read_rows) - self._background.get(layer_id, 0)
        self._background[layer_id] = int(background_read_rows)
        self.vram_misses += vram_miss
        self.ram_misses += ram_miss
        self.read_ms += read_ms
        self.split_ms += split_ms
        self.background_read_rows += background
        if tokens == 1:
            self.decode_vram_misses += vram_miss
            self.decode_ram_misses += ram_miss
        if self._file is not None:
            flat = topk_ids.reshape(-1)
            experts, counts = torch.unique(flat[flat >= 0], return_counts=True)
            line = {
                "forward": self.forwards,
                "layer": layer_id,
                "tokens": tokens,
                "experts": experts.tolist(),
                "counts": counts.tolist(),
                "vram_miss": vram_miss,
                "ram_miss": ram_miss,
                "read_ms": round(read_ms, 4),
                "split_ms": round(split_ms, 4),
                "background_rows": background,
                "t": round(time.monotonic(), 6),
            }
            if self._graph_seq is not None:
                line["graph_seq"] = self._graph_seq
            self._file.write(json.dumps(line) + "\n")

    @property
    def enabled(self) -> bool:
        """A trace file is open (SGLANG_DSV41_EXPERT_TRACE_PATH was set)."""
        return self._file is not None

    def record_graph_step(
        self,
        layer_rows_delta,
        routed_rows: int,
        routed_misses: int,
        thread: Optional[dict] = None,
        steps: int = 1,
    ) -> None:
        """One graph decode step: its VRAM misses (G) and the demand rows the RAM-miss
        thread read (f). In-graph MoE layers produce no host gather stats; the option C
        service calls this once per batch with the step's deltas. The line is shaped
        like a one-token forward, so tier_sim.live_summary counts it as a decode token.
        ``thread``: the thread's cumulative counters, written as given. An Engine's
        scheduler is SIGKILLed at shutdown, so the last line is where a run reads them.
        ``steps``: the decode steps the deltas cover (a lagged per-batch read folds one
        step into the next line); live_summary counts them as decode tokens.
        """
        ram = int(sum(layer_rows_delta))
        self.forwards += 1
        self.decode_tokens += int(steps)
        self.decode_vram_misses += int(routed_misses)
        self.decode_ram_misses += ram
        self.vram_misses += int(routed_misses)
        self.ram_misses += ram
        self._last_layer = None  # the next eager call starts a new forward
        if self._file is not None:
            line = {
                "forward": self.forwards,
                "layer": -1,
                "tokens": 1,
                "kind": "graph_step",
                "experts": [],
                "counts": [],
                "vram_miss": int(routed_misses),
                "ram_miss": ram,
                "routed_rows": int(routed_rows),
                "layer_ram_rows": [int(v) for v in layer_rows_delta],
                "steps": int(steps),
                "t": round(time.monotonic(), 6),
            }
            if thread is not None:
                line["thread"] = thread
            self._file.write(json.dumps(line) + "\n")

    def record_graph_routes_header(self, log: GraphRouteLog) -> None:
        """Once, before the first route step: what a ``graph_routes`` line's rows are."""
        if self._file is None:
            return
        line = {
            "kind": "graph_routes_header",
            "layer_ids": list(log.layer_ids),
            "hot_capacity": list(log.capacities),
            "width": log.width,
            "depth": log.depth,
            "warmup_forwards": log.warmup,
            "t": round(time.monotonic(), 6),
        }
        self._file.write(json.dumps(line) + "\n")

    def record_graph_route_step(self, seq: int, routes: list[list[int]], misses: list[int], dropped_before: int = 0) -> None:
        """One graph forward: ``routes[row]`` its routed experts in router order, ``misses[row]`` the
        experts its gather found outside VRAM (the plan's count). ``seq`` counts the graph forwards before
        it, warmup included; an eager forward line's ``graph_seq`` is on the same count. Not a forward
        call: no ``tokens``, and tier_sim.load_trace skips it."""
        if self._file is None:
            return
        line = {"kind": "graph_routes", "seq": seq, "routes": routes, "misses": misses, "t": round(time.monotonic(), 6)}
        if dropped_before:
            line["dropped_before"] = dropped_before
        self._file.write(json.dumps(line) + "\n")

    def record_ram_miss_requests(self, records: list[dict], layer_ids: list[int]) -> None:
        """One line per RAM-miss request the native service served, from ``Exl3RamMissHost.drain_trace``.

        ``layer_ids[row]`` names each record's streamed row. The line is not a forward call: it has
        no ``tokens``, and ``kind`` is ``ram_miss_request`` (tier_sim.load_trace skips it). Every
        ``stages_ns`` value and ``prev_done_ns`` is the host's CLOCK_MONOTONIC in ns, which is what
        ``t`` (``time.monotonic()``) reads too; 0 is a stage the request never reached. Nothing here
        compares a GPU clock with the host's.

        ``schema`` is RAM_MISS_TRACE_SCHEMA. In schema 1 the stamps below ``submit`` were the FIRST io_uring batch's,
        ``pack_end`` the last's, and ``spans_ns`` summed the per-batch spans, so packing never fell
        inside ``first_to_last_cqe``. From schema 2 every stamp spans the whole read and a row packs
        as soon as its own extents land, so ``first_to_last_cqe`` can contain the packing of earlier
        rows and is no longer pure drive latency. **The two are not comparable**; compare across the
        boundary with ``extent_cqe_ns``, whose meaning is unchanged, or re-baseline.

        ``bytes`` is the completed total; ``byte_split`` names the rest. ``row_pack_ns`` and
        ``extent_cqe_ns`` are per-row and per-extent stamps, bounded (``untraced`` counts the rest).
        ``schema`` 3 adds ``row_pack_ns[].admit``, ``extent_cqe_ns[].submit`` and ``.attempts``. A
        row's stamps are compared with its own extents', never as one sorted list: rows overlap.
        ``extent_cqe_ns[].cqe`` is when the wait that reaped the extent returned, not a per-completion time.
        ``dropped_before`` is how many records the native trace ring dropped, for being full, just
        before this line's record: nonzero means lines are missing at this point of the file.
        ``schema`` 4 adds ``request.lanes``: the layer's planned lane count for that request (RAM hits and
        misses; ``rows_asked`` counts only the misses read), which with the line's ``layer`` gives lanes per layer.
        ``schema`` 5 adds ``request.pack_workers`` and ``request.pack_split``: the packing mode. ``pack_workers``
        0 is the inline reader; above 0 the rows are packed by workers, and ``row_pack_ns`` starts, ``spans_ns.pack``
        and anything derived from the gap between an extent's reap and its row's pack start are then not
        comparable with an inline trace's (analysis/dsv41-drive/overlap_timeline.py refuses them).
        ``schema`` 6 adds ``request.piece_stream``, ``extent_cqe_ns[].sub`` and ``pieces``. With piece streaming on each
        extent is one sub-read of its part (``sub``), and ``pieces`` has one entry per row: when each sub-read landed and
        each piece was vetted, as shared sequence numbers, and the vetting's clock (``stage_records``). Off, ``sub`` is
        0 and ``pieces`` is empty, so every schema-5 field keeps its meaning.
        ``schema`` 7 adds ``pieces[].publish`` (when each piece was published, on the same sequence) and
        ``piece_publish``: ``published``, ``out_of_order`` and ``refused``. With piece streaming each piece is packed by
        its own job, so a row's ``row_pack_ns`` spans its pieces and can start before the row's last sub-read landed;
        with it off nothing changes.
        """
        if self._file is None or not records:
            return
        from sglang.kernels.ops.moe.exl3_ram_miss import STAGE_ORDER

        for record in records:
            line = {
                "forward": self.forwards,
                "layer": layer_ids[record["row"]],
                "kind": "ram_miss_request",
                "schema": RAM_MISS_TRACE_SCHEMA,
                "request": {
                    "seq": record["seq"],
                    "type": record["kind"],
                    "ok": record["ok"],
                    "rows": record["rows"],
                    "batches": record["batches"],
                    "backlog": record["backlog"],
                    "lanes": record["lanes"],
                    "pack_workers": record["pack_workers"],
                    "pack_split": record["pack_split"],
                    "piece_stream": record["piece_stream"],
                },
                "status": record["status"],
                "rows_asked": record["rows_asked"],
                "stages_ns": {name: record[name] for name in STAGE_ORDER},
                "missing_stages": record["missing_stages"],
                "prev_done_ns": record["prev_done"],
                "spans_ns": {
                    "submit_to_first_cqe": record["submit_to_first_cqe_ns"],
                    "first_to_last_cqe": record["first_to_last_cqe_ns"],
                    "pack": record["pack_ns"],
                },
                "bytes": record["bytes"],
                "byte_split": {
                    "useful": record["useful_bytes"],
                    "submitted": record["submitted_bytes"],
                    "completed": record["bytes"],
                    "retried": record["retried_bytes"],
                    "cancelled": record["cancelled_bytes"],
                },
                "row_pack_ns": record["row_pack"],
                "extent_cqe_ns": record["extent_cqe"],
                "pieces": record["pieces"],
                "piece_publish": {
                    "published": record["pieces_published"],
                    "out_of_order": record["pieces_out_of_order"],
                    "refused": record["piece_publish_refused"],
                },
                "untraced": {"rows": record["rows_untraced"], "extents": record["extents_untraced"]},
                "dropped_before": record["dropped_before"],
                "extents": record["extents"],
                "drives": record["drives"],
                "t": round(time.monotonic(), 6),
            }
            self._file.write(json.dumps(line) + "\n")

    def stats(self) -> dict:
        return {
            "forwards": self.forwards,
            "decode_tokens": self.decode_tokens,
            "decode_vram_misses": self.decode_vram_misses,
            "decode_ram_misses": self.decode_ram_misses,
            "G": self.decode_vram_misses / self.decode_tokens if self.decode_tokens else 0.0,
            "f": (
                self.decode_ram_misses / self.decode_vram_misses
                if self.decode_vram_misses
                else 0.0
            ),
            "vram_misses": self.vram_misses,
            "ram_misses": self.ram_misses,
            "read_ms": round(self.read_ms, 3),
            "split_ms": round(self.split_ms, 3),
            "background_read_rows": self.background_read_rows,
        }

    def log(self) -> None:
        logger.info("exl3 expert stream: %s", json.dumps(self.stats()))

    def close(self) -> None:
        if self.forwards:
            self.log()
        if self._file is not None:
            self._file.close()
            self._file = None


_TRACE: Optional[Exl3StreamTrace] = None
_TRACE_LOCK = threading.Lock()


def get_exl3_stream_trace() -> Exl3StreamTrace:
    """The process-wide trace, writing to SGLANG_DSV41_EXPERT_TRACE_PATH when set."""
    global _TRACE
    with _TRACE_LOCK:
        if _TRACE is None:
            _TRACE = Exl3StreamTrace(envs.SGLANG_DSV41_EXPERT_TRACE_PATH.get())
            atexit.register(_TRACE.close)
        return _TRACE
