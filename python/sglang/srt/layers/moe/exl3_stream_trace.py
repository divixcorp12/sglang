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
from typing import Any, Optional

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


class Exl3StreamTrace:
    def __init__(self, path: str = "", log_every: int = LOG_EVERY_FORWARDS) -> None:
        # Line-buffered: the scheduler process may exit without running atexit.
        self._file = open(path, "a", buffering=1) if path else None
        self.log_every = log_every
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

    def record_ram_miss_requests(self, records: list[dict], layer_ids: list[int]) -> None:
        """One line per RAM-miss request the native service served, from ``Exl3RamMissHost.drain_trace``.

        ``layer_ids[row]`` names each record's streamed row. The line is not a forward call: it has
        no ``tokens``, and ``kind`` is ``ram_miss_request`` (tier_sim.load_trace skips it). Every
        ``stages_ns`` value and ``prev_done_ns`` is the host's CLOCK_MONOTONIC in ns, which is what
        ``t`` (``time.monotonic()``) reads too; 0 is a stage the request never reached. Nothing here
        compares a GPU clock with the host's. The stamps below ``submit`` are the first io_uring
        batch's and ``pack_end`` the last's; ``spans_ns`` sums every batch (see StageRecord).
        """
        if self._file is None or not records:
            return
        from sglang.kernels.ops.moe.exl3_ram_miss import STAGE_ORDER

        for record in records:
            line = {
                "forward": self.forwards,
                "layer": layer_ids[record["row"]],
                "kind": "ram_miss_request",
                "request": {
                    "seq": record["seq"],
                    "type": record["kind"],
                    "ok": record["ok"],
                    "rows": record["rows"],
                    "batches": record["batches"],
                    "backlog": record["backlog"],
                },
                "stages_ns": {name: record[name] for name in STAGE_ORDER},
                "prev_done_ns": record["prev_done"],
                "spans_ns": {
                    "submit_to_first_cqe": record["submit_to_first_cqe_ns"],
                    "first_to_last_cqe": record["first_to_last_cqe_ns"],
                    "pack": record["pack_ns"],
                },
                "bytes": record["bytes"],
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
