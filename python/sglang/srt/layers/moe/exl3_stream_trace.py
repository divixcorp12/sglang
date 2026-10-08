"""G and f for streamed EXL3 experts, from the expert streaming framework's gather stats.

G is VRAM misses per decode token, f the share of those that also miss RAM and read
the shards (DSV41_REFERENCE.md section 9.4). This file holds:

* :class:`Exl3StreamTrace`, the process-wide counter and optional JSONL trace.
  Each streamed MoE layer call reports its gather's ``ExpertGatherStats``:
  ``miss_rows`` (rows not in the hot cache) and ``host_read_rows`` (rows read
  through the row source, i.e. not in the pinned host tier either). A new forward
  starts when a call's layer id is not above the previous call's; a call with one
  token row is a decode call. Every line carries a ``time.monotonic()`` stamp, so
  the gaps between forwards show the residency-boundary promotion stalls that tok/s
  includes. The trace feeds ``scripts/dsv41/tier_sim.py``.
* :class:`GraphRouteLog`, which records the routes of replayed decode graphs, whose
  Python never runs, into a device ring for the trace.
* :class:`RouterCapture`, the binary side files of the router's input.

The trace is enabled by ``SGLANG_DSV41_EXPERT_TRACE_PATH``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
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


# Bumped whenever a ram_miss_request field changes meaning or the record layout
# changes (a field added or removed). A JSONL file has no field check like the native
# record's, so this integer is the only mechanical answer to which fields a file has
# and which quantities it compares.
#
# 2: stage stamps and spans cover the whole read, not the first io_uring batch, and
#    packing can fall inside first_to_last_cqe.
# 3: adds row_pack[].admit, extent_cqe[].submit/attempts and dropped_before; no
#    schema-2 field changed meaning.
# 4: adds request.lanes, the planned lane count the device posted. Every schema-3
#    field keeps its meaning, so spans, byte totals and per-drive shares compare
#    across 3 and 4.
# 5: adds request.pack_workers and request.pack_split, the packing mode the reader
#    ran in. No earlier field changed meaning, but a worker-mode record's pack
#    stamps mean something else than an inline one's (a row's pack_start is after
#    the worker woke, its spans overlap). A file written before schema 5 does not
#    say which mode wrote it; it is assumed inline, because the workers were barred
#    from any run that writes a stage trace.
# 6: adds request.piece_stream, extent_cqe[].sub and pieces.
# 7: adds pieces[].publish and pieces_published, pieces_out_of_order and
#    piece_publish_refused. With piece streaming a row's row_pack span covers its
#    pieces' jobs.
RAM_MISS_TRACE_SCHEMA = 7


# Bumped whenever a graph_routes / graph_routes_header field changes meaning or the
# layout changes.
# 1: the first layout (hot-cache-policy capture).
# 2: adds graph_routes.router, the forward's record number in the RouterCapture side
#    files. With router capture off it is never written, and every schema-1 field
#    keeps its meaning.
ROUTE_LOG_SCHEMA = 2
# The RouterCapture side files' own layout; <prefix>.json states it.
ROUTER_CAPTURE_SCHEMA = 2


def _stream_capturing() -> bool:
    """True while the current CUDA stream is being captured into a graph."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _verify_tokens(forward_batch) -> int:
    """A target verify's live token count, the graph's padding excluded.

    init_new leaves ``extend_num_tokens`` None for a verify; only the DSpark verify
    layout knows how many rows are real. Its host copy is used when the planner kept
    one; otherwise the device lengths are read, a sync this diagnostic path accepts.
    """
    layout = getattr(forward_batch.spec_info, "ragged_verify_layout", None)
    if layout is None:
        return 0
    if layout.verify_lens_cpu is not None:
        return int(sum(layout.verify_lens_cpu))
    return int(layout.verify_lens.sum())


class RouterCapture:
    """Binary side files of every graph forward's router input.

    One record per forward, in ``seq`` order, appended by ``GraphRouteLog``:

    * ``<prefix>.x.bin``: bf16 ``[records, layers, tokens, hidden]`` (raw
      little-endian 16-bit words), each streamed layer's normalized MoE input as its
      gate saw it, for every token row of the graph (``tokens`` is the graph's width).
    * ``<prefix>.ids.bin``: int32 ``[records, layers, tokens, topk]``, the top-k ids.
    * ``<prefix>.w.bin``: fp32 ``[records, layers, tokens, topk]``, the top-k
      weights in route order.
    * ``<prefix>.seq.bin``: int64 ``[records, 3]``, the forward's ``seq``, pass id
      and live token count; the first two are the keys of its ``graph_routes`` line
      (which names the record as ``router``), and rows past the live count are the
      graph's padding.
    * ``<prefix>.json``: the shapes and layer ids. The record count is the files'
      size.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.records = 0
        self._files: Optional[list] = None

    def write(
        self,
        log: GraphRouteLog,
        seq: int,
        pass_id: int,
        tokens: int,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> int:
        """Append one forward's record for ``log``; returns its record number.

        ``x`` is ``[layers, M_max, hidden]``, ``ids`` and ``weights`` ``[layers, M_max, topk]``; ``tokens`` is the
        forward's live count, the rows past it being the graph's padding. The first call writes the ``.json``
        header and opens the data files.
        """
        if self._files is None:
            header = {
                "schema": ROUTER_CAPTURE_SCHEMA,
                "run": log.run,
                "layer_ids": list(log.layer_ids),
                "tokens": int(x.shape[-2]),
                "hidden": int(x.shape[-1]),
                "topk": int(weights.shape[-1]),
                "x_dtype": "bfloat16",
                "ids_dtype": "int32",
                "w_dtype": "float32",
                "depth": log.depth,
            }
            with open(self.prefix + ".json", "w") as f:
                json.dump(header, f)
            self._files = [
                open(self.prefix + suffix, "ab")
                for suffix in (".x.bin", ".ids.bin", ".w.bin", ".seq.bin")
            ]
        x_file, ids_file, w_file, seq_file = self._files
        x_file.write(x.contiguous().view(torch.int16).numpy().tobytes())
        ids_file.write(ids.contiguous().to(torch.int32).numpy().tobytes())
        w_file.write(weights.contiguous().numpy().tobytes())
        seq_file.write(
            torch.tensor([seq, pass_id, tokens], dtype=torch.int64).numpy().tobytes()
        )
        self.records += 1
        return self.records - 1

    def flush(self) -> None:
        """Flush the data files."""
        for f in self._files or []:
            f.flush()


class GraphRouteLog:
    """Every graph forward's routed experts and VRAM misses per streamed layer.

    For the stage trace only. A replayed decode graph runs no Python, so its routes
    never reach ``Exl3StreamTrace.record``. ``record`` is instead called from each
    layer's in-graph gather (``Exl3RamMissRowBackend.post``) and is captured with it:
    the first streamed layer (row 0) takes the next ring slot and bumps ``seq``, then
    every layer copies its routes and its plan's miss count into that slot. Each
    forward thus writes its own entry, numbered by ``seq`` (the graph forwards that
    ran before it), whatever the scheduler's per-batch cadence: no per-batch read can
    fold or skip one.

    Row 0 also stores the forward's pass id and, when ``bind_hot`` named the GPU
    residency bank, the hot set every layer held when the forward started.
    ``on_pre_forward`` writes the pass id to the device before the forward is queued,
    so the forward's phase and request come from the scheduler's forward mode, never
    from a token count.

    ``poll`` runs at the per-batch check. It reads the ring back through pinned
    buffers without waiting, one batch behind (like ``Exl3RamMissService._graph_rows``),
    and writes the entries finished since the last read. The copy is on the host's
    current stream, which need not be the stream the graph replays on, so an entry is
    trusted only when the ``seq`` read before the ring shows a later forward has
    started (entries below ``seq - 1``), and only if it is ``margin`` entries clear
    of the slots later forwards may overwrite while the ring copy runs. An entry
    overwritten before it was read is counted in ``dropped`` and in the next line's
    ``dropped_before``; a replay must refuse such a run. ``final`` (after shutdown's
    device barrier) reads everything.

    ``enable_router`` adds three rings on the same slots: ``record_router`` copies each
    layer's router input, top-k ids and top-k weights for every token into them, and every emitted forward is
    also written to the RouterCapture side files. Without it no router tensor exists
    and nothing calls ``record_router``.

    Warmup forwards before capture execute these copies too. ``warmup`` counts them,
    from the Python calls made in capture mode outside a stream capture (a capture
    records the copies but runs none), and no entry below it is written.
    """

    def __init__(
        self, layers: int, width: int, device, depth: int = 64, margin: int = 4
    ) -> None:
        if depth <= margin + 1:
            raise ValueError("the route ring must be deeper than its safety margin")
        self.layers, self.width, self.depth, self.margin = layers, width, depth, margin
        self.run = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
        self.layer_ids: list[int] = [-1] * layers
        self.capacities: list[int] = [0] * layers  # hot slots per row
        self.routes = torch.full(
            (depth, layers, width), -1, dtype=torch.int64, device=device
        )
        self.misses = torch.full((depth, layers), -1, dtype=torch.int32, device=device)
        self.pass_ids = torch.full((depth,), -1, dtype=torch.int64, device=device)
        self.pass_id = torch.full((1,), -1, dtype=torch.int64, device=device)
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)
        self.slot = torch.zeros(1, dtype=torch.int64, device=device)
        self.hot_bank: Optional[torch.Tensor] = None
        self.hot_layer_ids: list[int] = []
        self.hot: Optional[torch.Tensor] = None
        self.warmup = 0
        self.next_seq = 0
        self.dropped = 0
        self._meta: dict[int, dict] = {}
        self._current: Optional[dict] = None
        self._header_written = False
        self._host: Optional[list[torch.Tensor]] = None
        self._event = None
        self._pending = False
        self.router: Optional[RouterCapture] = None
        self.router_x: Optional[torch.Tensor] = None
        self.router_ids: Optional[torch.Tensor] = None
        self.router_w: Optional[torch.Tensor] = None

    def bind(self, row: int, layer_id: int, capacity: int) -> None:
        """Name ring row ``row``: its layer id and hot-slot capacity."""
        self.layer_ids[row] = int(layer_id)
        self.capacities[row] = int(capacity)

    def bind_hot(self, bank: torch.Tensor, layer_ids: list[int]) -> None:
        """Snapshot ``bank`` once per forward.

        ``bank`` is ``GpuResidencyUpdater.slot_to_expert``, one row per ``layer_ids``.
        """
        self.hot_bank = bank
        self.hot_layer_ids = [int(layer) for layer in layer_ids]
        self.hot = torch.full(
            (self.depth, *bank.shape), -1, dtype=bank.dtype, device=bank.device
        )

    def enable_router(self, prefix: str) -> None:
        """Also capture each layer's router input, top-k ids and weights into ``prefix`` files.

        The rings are sized by the first ``record_router`` call, a warmup forward's.
        """
        self.router = RouterCapture(prefix)

    def ring_bytes(self) -> int:
        """Device bytes held by the rings."""
        return sum(t.numel() * t.element_size() for t in self._ring())

    def on_pre_forward(self, forward_pass_id: int, forward_batch) -> None:
        """The expert distribution recorder's pre-forward observer.

        Records the forward's identity on the host and the device. Runs before the
        forward is queued, on its stream, outside any capture.
        """
        mode = forward_batch.forward_mode
        if not mode.is_extend():
            tokens = forward_batch.batch_size
        elif mode.is_target_verify():
            tokens = _verify_tokens(forward_batch)
        else:
            tokens = forward_batch.extend_num_tokens
        meta = {
            "forward_pass_id": int(forward_pass_id),
            "phase": mode.name.lower(),
            "rids": list(forward_batch.rids or []),
            "tokens": int(tokens or 0),
        }
        self._current = meta
        if _stream_capturing():
            return  # a capture would freeze this pass id into every replay
        self._meta[meta["forward_pass_id"]] = meta
        self.pass_id.fill_(meta["forward_pass_id"])

    def current_meta(self) -> Optional[dict]:
        """The identity of the forward the observer saw last, or None."""
        return self._current

    def record(self, row: int, routes: torch.Tensor, count: torch.Tensor) -> None:
        """Log one layer's routes (``routes[:n]``, -1 past them) and plan miss count.

        Captured in the graph.
        """
        if row == 0:
            if capturing_graphs() and not _stream_capturing():
                self.warmup += 1
            torch.remainder(self.seq, self.depth, out=self.slot)
            self.seq.add_(1)
            self.pass_ids.index_copy_(0, self.slot, self.pass_id)
            if self.hot is not None:
                # Before this layer's gather commits, so it is the residency every
                # layer held at the forward's start.
                self.hot.index_copy_(0, self.slot, self.hot_bank.unsqueeze(0))
        n = min(routes.numel(), self.width)
        self.routes[:, row, :n].index_copy_(0, self.slot, routes[:n].view(1, n))
        self.misses[:, row].index_copy_(0, self.slot, count.reshape(-1)[:1])

    def record_router(
        self, row: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        """Log one layer's router input, top-k ids and weights for every token of the forward into the slot.

        Captured in the graph. Must follow the layer's ``record``: row 0's takes the
        slot. The rings take the first call's token count: a graph is captured at its
        widest verify, so every replay writes that many rows and the live count travels
        in the forward's meta.
        """
        tokens, hidden = int(x.shape[0]) if x.dim() > 1 else 1, int(x.shape[-1])
        topk = int(topk_weights.shape[-1])
        if self.router_x is None:
            if _stream_capturing() or self._host is not None:
                raise RuntimeError(
                    "the router rings must be allocated by a warmup forward, before capture and reads"
                )
            device = self.routes.device
            shape = (self.depth, self.layers, tokens)
            self.router_x = torch.zeros((*shape, hidden), dtype=torch.bfloat16, device=device)
            # The router's own id dtype: a cast here would add a kernel to the captured graph.
            self.router_ids = torch.zeros((*shape, topk), dtype=topk_ids.dtype, device=device)
            self.router_w = torch.zeros((*shape, topk), dtype=torch.float32, device=device)
        want = tuple(self.router_x.shape[-2:]), tuple(self.router_w.shape[-2:])
        if (tokens, hidden) != want[0] or (tokens, topk) != want[1]:
            raise ValueError(
                f"router capture holds {want[0][0]} tokens of [{want[0][1]}] and [{want[1][1]}], got "
                f"{tokens} tokens of {tuple(x.shape)} and {tuple(topk_weights.shape)}"
            )
        self.router_x[:, row].index_copy_(
            0, self.slot, x.reshape(1, tokens, hidden).to(torch.bfloat16)
        )
        self.router_ids[:, row].index_copy_(
            0, self.slot, topk_ids.reshape(1, tokens, topk).to(self.router_ids.dtype)
        )
        self.router_w[:, row].index_copy_(
            0, self.slot, topk_weights.reshape(1, tokens, topk).float()
        )

    def read_seq(self) -> int:
        """Graph forwards run so far.

        Called by eager forwards only, which already wait for the stream.
        """
        return int(self.seq.item())

    def _ring(self) -> list[torch.Tensor]:
        """The device buffers read back together, ``seq`` first."""
        ring = [self.seq, self.routes, self.misses, self.pass_ids]
        ring += [self.hot] if self.hot is not None else []
        ring += (
            [self.router_x, self.router_ids, self.router_w]
            if self.router_x is not None
            else []
        )
        return ring

    def tensors(self) -> list[torch.Tensor]:
        """Every buffer a copy may still be writing, for the service's quarantine."""
        return self._ring() + [self.slot, self.pass_id] + (self._host or [])

    def poll(self, trace: Exl3StreamTrace, *, final: bool = False) -> None:
        """Write the entries finished since the last read, then queue the next copy.

        Without ``final`` this never waits: it emits the previous batch's copy once
        its event completes and starts another. ``final`` waits and reads everything.
        """
        if self.seq.device.type != "cuda":
            self._emit(trace, self._ring(), complete=True)
            return
        if not final and torch.cuda.is_current_stream_capturing():
            return
        if self._pending:
            if final:
                self._event.synchronize()
            elif not self._event.query():
                return
            self._pending = False
            self._emit(trace, self._host, complete=False)
        if final:
            self._emit(trace, [tensor.cpu() for tensor in self._ring()], complete=True)
            return
        if self._host is None:
            self._host = [
                torch.empty_like(t, device="cpu").pin_memory() for t in self._ring()
            ]
            self._event = torch.cuda.Event(enable_timing=False)
        # seq is first: the stream runs these copies in order.
        for host, device in zip(self._host, self._ring()):
            host.copy_(device, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.seq.device))
        self._pending = True

    def _emit(
        self, trace: Exl3StreamTrace, ring: list[torch.Tensor], *, complete: bool
    ) -> None:
        """Write the entries ``ring`` holds that are safe and not yet written.

        ``complete`` means every slot is settled (a final or device-less read);
        otherwise the newest entry and ``margin`` slots of the ring are held back.
        Entries lost to overwrite are added to ``dropped``.
        """
        if not self._header_written:
            trace.record_graph_routes_header(self)
            self._header_written = True
        seq_tensor, routes, misses, pass_ids = ring[:4]
        hot = ring[4] if self.hot is not None else None
        router_x, router_ids, router_w = (
            ring[-3:] if self.router_x is not None else (None, None, None)
        )
        seq = int(seq_tensor[0])
        end = seq if complete else seq - 1
        start = max(self.next_seq, self.warmup)
        oldest = max(seq - self.depth + (0 if complete else self.margin), 0)
        lost = max(oldest - start, 0)
        start += lost
        self.dropped += lost
        for s in range(start, end):
            slot = s % self.depth
            pass_id = int(pass_ids[slot])
            meta = self._meta.pop(pass_id, None)
            for stale in [key for key in self._meta if key < pass_id]:
                del self._meta[stale]  # eager forwards never log here
            record = None
            if router_x is not None:
                record = self.router.write(
                    self,
                    s,
                    pass_id,
                    int((meta or {}).get("tokens", 0)),
                    router_x[slot],
                    router_ids[slot],
                    router_w[slot],
                )
            trace.record_graph_route_step(
                s,
                [[int(e) for e in row if e >= 0] for row in routes[slot].tolist()],
                [int(m) for m in misses[slot].tolist()],
                dropped_before=lost if s == start else 0,
                meta={
                    "run": self.run,
                    **(meta if meta is not None else {"forward_pass_id": pass_id}),
                },
                hot=None
                if hot is None
                else [
                    sorted(int(e) for e in row if e >= 0) for row in hot[slot].tolist()
                ],
                router=record,
            )
        self.next_seq = max(self.next_seq, end)
        if self.router is not None:
            # The scheduler may be SIGKILLed at shutdown, like the stage trace's lines.
            self.router.flush()


class Exl3StreamTrace:
    """Counters for G and f, plus an optional JSONL trace, of streamed EXL3 experts.

    One instance per process (``get_exl3_stream_trace``). Eager forwards report
    through ``record``; graph decode steps through ``record_graph_step`` and the
    ``record_graph_route*`` methods, which the RAM-miss service calls. With ``path``
    empty the counters still run and no file is written. ``stats`` reports the
    counters and ``log`` writes them to the logger every ``log_every`` forwards.

    ``graph_seq_source`` and ``forward_meta_source`` are set by the RAM-miss service
    when it logs graph routes. They stamp each eager forward with the graph forwards
    that ran before it, so a replay can interleave the two
    (``tier_sim.load_forwards``), and with its pass id, phase and requests.
    """

    def __init__(self, path: str = "", log_every: int = LOG_EVERY_FORWARDS) -> None:
        # Line-buffered: the scheduler process may exit without running atexit.
        self._file = open(path, "a", buffering=1) if path else None
        self.log_every = log_every
        self.graph_seq_source: Optional[Callable[[], int]] = None
        self.forward_meta_source: Optional[Callable[[], Optional[dict]]] = None
        self._graph_seq: Optional[int] = None
        self._meta: Optional[dict] = None
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
        """Count one streamed MoE call and, if tracing, write its line.

        ``stats`` is its gather's stats, or None when the call gathered nothing.
        ``background_read_rows`` is the layer's cumulative count of rows read outside
        gathers (promotions, seeding). Ignored during graph capture.
        """
        if capturing_graphs():
            return
        tokens = int(topk_ids.shape[0])
        if self._last_layer is None or layer_id <= self._last_layer:
            self.forwards += 1
            if tokens == 1:
                self.decode_tokens += 1
            if self.graph_seq_source is not None:
                self._graph_seq = self.graph_seq_source()
            if self.forward_meta_source is not None:
                self._meta = self.forward_meta_source()
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
            if self._meta is not None:
                line.update(
                    {
                        key: self._meta[key]
                        for key in ("forward_pass_id", "phase", "rids")
                    }
                )
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
        """One graph decode step: its VRAM misses (G) and the demand rows read (f).

        In-graph MoE layers produce no host gather stats, so the RAM-miss service
        calls this once per batch with the step's deltas. The line is shaped like a
        one-token forward, so ``tier_sim.live_summary`` counts it as a decode token.

        ``thread`` holds the RAM-miss thread's cumulative counters, written as given.
        An Engine's scheduler is SIGKILLed at shutdown, so the last line is where a
        run reads them. ``steps`` is the decode steps the deltas cover (a lagged
        per-batch read folds one step into the next line); ``live_summary`` counts
        them as decode tokens.
        """
        ram = int(sum(layer_rows_delta))
        self.forwards += 1
        self.decode_tokens += int(steps)
        self.decode_vram_misses += int(routed_misses)
        self.decode_ram_misses += ram
        self.vram_misses += int(routed_misses)
        self.ram_misses += ram
        self._last_layer = None  # so the next eager call starts a new forward
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
            "schema": ROUTE_LOG_SCHEMA,
            "run": log.run,
            "layer_ids": list(log.layer_ids),
            "hot_capacity": list(log.capacities),
            "width": log.width,
            "depth": log.depth,
            "warmup_forwards": log.warmup,
            "hot_layer_ids": list(log.hot_layer_ids),
            "t": round(time.monotonic(), 6),
        }
        if log.router is not None:
            line["router_prefix"] = log.router.prefix
        self._file.write(json.dumps(line) + "\n")

    def record_graph_route_step(
        self,
        seq: int,
        routes: list[list[int]],
        misses: list[int],
        dropped_before: int = 0,
        meta: Optional[dict] = None,
        hot: Optional[list[list[int]]] = None,
        router: Optional[int] = None,
    ) -> None:
        """One graph forward: its routed experts and misses per streamed layer.

        ``routes[row]`` is the row's routed experts in router order and
        ``misses[row]`` the experts its gather found outside VRAM (the plan's count).
        ``seq`` counts the graph forwards before it, warmup included; an eager forward
        line's ``graph_seq`` is on the same count. ``meta`` is the forward's pass id,
        and its phase, requests and tokens when the pre-forward observer saw it.
        ``hot`` is the experts each ``hot_layer_ids`` layer held when the forward
        started. ``dropped_before`` is the entries lost just before this one (a replay
        must refuse the run). ``router`` is the forward's record in the RouterCapture
        side files, when router capture is on.

        Not a forward call: the line has no ``tokens`` key (the forward's token count
        is ``forward_tokens``), and ``tier_sim.load_trace`` skips it.
        """
        if self._file is None:
            return
        line = {"kind": "graph_routes", "schema": ROUTE_LOG_SCHEMA, "seq": seq}
        line.update(meta or {})
        line.update(
            {"routes": routes, "misses": misses, "t": round(time.monotonic(), 6)}
        )
        if meta is not None and "tokens" in meta:
            # Keeps load_trace's rule that "tokens" means a forward call.
            line["forward_tokens"] = line.pop("tokens")
        if hot is not None:
            line["hot"] = hot
        if router is not None:
            line["router"] = router
        if dropped_before:
            line["dropped_before"] = dropped_before
        self._file.write(json.dumps(line) + "\n")

    def record_ram_miss_requests(
        self, records: list[dict], layer_ids: list[int]
    ) -> None:
        """One line per RAM-miss request the native service served.

        The records come from ``ExpertStreamHost.drain_trace``; ``layer_ids[row]``
        names each record's streamed row. The line is not a forward call: it has no
        ``tokens``, and ``kind`` is ``ram_miss_request`` (``tier_sim.load_trace``
        skips it). Every ``stages_ns`` value and ``prev_done_ns`` is the host's
        CLOCK_MONOTONIC in ns, which is what ``t`` (``time.monotonic()``) reads too;
        0 is a stage the request never reached. Nothing here compares a GPU clock
        with the host's.

        ``schema`` is ``RAM_MISS_TRACE_SCHEMA``, which lists what each version added.
        Two points matter when comparing files:

        * From schema 2 every stamp spans the whole read and a row packs as soon as
          its own extents land, so ``first_to_last_cqe`` can contain the packing of
          earlier rows and is no longer pure drive latency. In schema 1 the stamps
          below ``submit`` were the first io_uring batch's, ``pack_end`` the last's,
          and ``spans_ns`` summed the per-batch spans. The two are not comparable;
          compare across the boundary with ``extent_cqe_ns``, whose meaning is
          unchanged, or re-baseline.
        * ``pack_workers`` 0 is the inline reader; above 0 the rows are packed by
          workers, and ``row_pack_ns`` starts, ``spans_ns.pack`` and anything derived
          from the gap between an extent's reap and its row's pack start are then not
          comparable with an inline trace's
          (``analysis/dsv41-drive/overlap_timeline.py`` refuses them).

        Field notes:

        * ``bytes`` is the completed total; ``byte_split`` names the rest.
        * ``row_pack_ns`` and ``extent_cqe_ns`` are per-row and per-extent stamps,
          bounded (``untraced`` counts the rest). A row's stamps are compared with its
          own extents', never as one sorted list: rows overlap.
          ``extent_cqe_ns[].cqe`` is when the wait that reaped the extent returned,
          not a per-completion time.
        * ``dropped_before`` is how many records the native trace ring dropped, for
          being full, just before this line's record: nonzero means lines are missing
          at this point of the file.
        * ``request.lanes`` is the layer's planned lane count for that request (RAM
          hits and misses; ``rows_asked`` counts only the misses read), which with the
          line's ``layer`` gives lanes per layer.
        * With piece streaming on, each extent is one sub-read of its part (``sub``),
          and ``pieces`` has one entry per row: when each sub-read landed, when each
          piece was vetted (as shared sequence numbers) and published, and the vetting's
          clock (``stage_records``). Each piece is packed by its own job, so a row's
          ``row_pack_ns`` spans its pieces and can start before the row's last sub-read
          landed. ``piece_publish`` counts ``published``, ``out_of_order`` and
          ``refused``. With it off, ``sub`` is 0 and ``pieces`` is empty.
        """
        if self._file is None or not records:
            return
        from sglang.kernels.ops.moe.expert_stream_transport import STAGE_ORDER

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
                "untraced": {
                    "rows": record["rows_untraced"],
                    "extents": record["extents_untraced"],
                },
                "dropped_before": record["dropped_before"],
                "extents": record["extents"],
                "drives": record["drives"],
                "t": round(time.monotonic(), 6),
            }
            self._file.write(json.dumps(line) + "\n")

    def stats(self) -> dict:
        """The counters, with G and f derived from the decode ones."""
        return {
            "forwards": self.forwards,
            "decode_tokens": self.decode_tokens,
            "decode_vram_misses": self.decode_vram_misses,
            "decode_ram_misses": self.decode_ram_misses,
            "G": self.decode_vram_misses / self.decode_tokens
            if self.decode_tokens
            else 0.0,
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
        """Log ``stats`` at info level."""
        logger.info("exl3 expert stream: %s", json.dumps(self.stats()))

    def close(self) -> None:
        """Log the final counters and close the trace file; safe to call twice."""
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
