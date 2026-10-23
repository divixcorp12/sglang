"""A read-only Engram hash table served from its safetensors shard.

The layer-1 table alone is 101.5 GB, more than the host RAM budget, and a batch
touches 24 rows per token. Rows are gathered on the CPU, copied to the device
and dequantized there. Eager, or an eager break of a breakable graph. By default rows come from an np.memmap and
the page cache does the caching; with SGLANG_DSV41_ENGRAM_RAM_GIB set, a bounded
RAM row cache shared by every Engram layer does, and misses read with O_DIRECT.
"""

from __future__ import annotations

import atexit
import glob
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.layers.moe.exl3_expert_layout import read_safetensors_header
from sglang.srt.layers.quantization.exl3.ops import assert_not_capturing
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.engram_row_cache import EngramRowCache

logger = logging.getLogger(__name__)

# How often the native store's counters are logged.
NATIVE_STATS_INTERVAL_S = 60.0
_LOGGED_STORES: set = set()


def native_cache_bytes() -> int:
    """SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB in bytes; 0 is a store that keeps no rows."""
    gib = envs.SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB.get()
    if gib < 0:
        raise ValueError(f"SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB must be >= 0, got {gib}")
    return int(gib * (1 << 30))


def native_stats_line(now: dict, before: Optional[dict]) -> str:
    """One log line: the window's and the run's accesses, hits, unique misses and hit rate."""
    def part(acc: int, hits: int, misses: int) -> str:
        return f"{acc} accesses, {hits} hits, {misses} unique misses, hit rate {hits / acc if acc else 0.0:.3f}"

    keys = ("accesses", "hits", "unique_misses")
    window = [now[k] - (before[k] if before else 0) for k in keys]
    return (f"engram native cache ({now['cache_bytes'] / (1 << 30):.2f} GiB, {now['capacity_rows']} rows): "
            f"window {part(*window)}; total {part(*(now[k] for k in keys))}")


def log_native_stats(store, interval_s: float = NATIVE_STATS_INTERVAL_S) -> None:
    """Log the shared native store's counters every interval_s (a daemon thread) and at exit; once per store."""
    if id(store) in _LOGGED_STORES:
        return
    _LOGGED_STORES.add(id(store))
    state = {"before": None}

    def emit() -> None:
        now = dict(store.stats())
        if now["accesses"] != (state["before"] or {}).get("accesses", 0):
            logger.info(native_stats_line(now, state["before"]))
        state["before"] = now

    def loop() -> None:
        while True:
            time.sleep(interval_s)
            emit()

    threading.Thread(target=loop, name="engram-cache-stats", daemon=True).start()
    atexit.register(emit)


_LOGGED_RING_EXTENSIONS: set = set()
_RING_KEYS = ("waits", "spins", "spin_ns")


def ring_layer_totals(lookups) -> dict:
    """ring_stats() summed per Engram layer. It has one row per device-wait lookup, so one per captured graph that
    holds the layer; spin_us_max is the largest of them."""
    totals: dict = {}
    for row in lookups:
        t = totals.setdefault(int(row["layer"]), dict.fromkeys(_RING_KEYS + ("spin_us_max",), 0))
        for k in _RING_KEYS:
            t[k] += int(row[k])
        t["spin_us_max"] = max(t["spin_us_max"], int(row["spin_us_max"]))
    return totals


def ring_stats_line(layer: int, now: dict, before: Optional[dict], seconds: float) -> str:
    """One layer's device-wait line: over the window since ``before``, or the run total when it is None. A
    spinning wait found the lookup unserved; spin_us_max is the longest spin since start (not per window)."""
    waits, spins, spin_ns = (now[k] - (before[k] if before else 0) for k in _RING_KEYS)
    return (f"engram ring stats layer {layer}{'' if before else ' (run total)'}: waits {waits}, spinning waits "
            f"{spins}, spin_ns {spin_ns}, mean {spin_ns / waits if waits else 0.0:.0f} ns per wait "
            f"({spin_ns / spins if spins else 0.0:.0f} per spinning wait), max {now['spin_us_max']} us since start, "
            f"over {seconds:.1f} s")


def log_ring_stats(extension, interval_s: float = NATIVE_STATS_INTERVAL_S) -> None:
    """Log each Engram layer's device-wait counters every interval_s (a daemon thread), and the run total at exit;
    once per extension. The device wait kernel writes them into each lookup's pinned control block, and
    ring_stats() reads them there with host loads: no CUDA call, no device sync, nothing on the batch path."""
    if id(extension) in _LOGGED_RING_EXTENSIONS:
        return
    _LOGGED_RING_EXTENSIONS.add(id(extension))
    start = time.monotonic()
    state = {"before": {}, "at": start}

    def emit_window() -> None:
        now, at = ring_layer_totals(extension.ring_stats()), time.monotonic()
        for layer, totals in sorted(now.items()):
            before = state["before"].get(layer, dict.fromkeys(_RING_KEYS, 0))
            if totals["waits"] > before["waits"]:
                logger.info(ring_stats_line(layer, totals, before, at - state["at"]))
        if now:
            state["before"], state["at"] = now, at

    def emit_total() -> None:
        # The last window's snapshot stands in if the lookups were already closed.
        now = ring_layer_totals(extension.ring_stats()) or state["before"]
        for layer, totals in sorted(now.items()):
            logger.info(ring_stats_line(layer, totals, None, time.monotonic() - start))

    def loop() -> None:
        while True:
            time.sleep(interval_s)
            emit_window()

    threading.Thread(target=loop, name="engram-ring-stats", daemon=True).start()
    atexit.register(emit_total)


class EngramFileTable:
    def __init__(
        self,
        path: str,
        weight_key: str,
        scale_key: str,
        num_embeddings: int,
        dim: int,
        block: int = 32,
        cache: Optional["EngramRowCache"] = None,
        cache_tag: int = 0,
        reader=None,
        direct: bool = True,
    ):
        data_start, header = read_safetensors_header(path)
        w, s = header[weight_key], header[scale_key]
        if (w["dtype"], s["dtype"]) != ("F8_E4M3", "F8_E8M0"):
            raise ValueError(f"{path}: expected F8_E4M3/F8_E8M0, got {w['dtype']}/{s['dtype']}")
        if w["shape"] != [num_embeddings, dim] or s["shape"] != [num_embeddings, dim // block]:
            raise ValueError(
                f"{path}: rows/shape {w['shape']} {s['shape']} != [{num_embeddings}, {dim}]"
            )
        self.dim, self.block = dim, block
        self.path = path
        self.weight_offset = data_start + w["data_offsets"][0]
        self.scale_offset = data_start + s["data_offsets"][0]
        self.num_embeddings = num_embeddings
        self._host_node_extension = None
        self._native_store = None
        self.weight = np.memmap(path, np.uint8, "r", data_start + w["data_offsets"][0], (num_embeddings, dim))
        self.scale = np.memmap(
            path, np.uint8, "r", data_start + s["data_offsets"][0], (num_embeddings, dim // block)
        )
        self.cache = cache
        # Row ids of different layers share one cache; the tag keeps them apart.
        self._tag = cache_tag << 40
        if cache is not None:
            from sglang.srt.model_loader.file_row_reader import (
                PagedRowSource,
                shared_uring_file_reader,
            )

            reader = reader if reader is not None else shared_uring_file_reader()
            self._weight_rows = PagedRowSource(
                reader, path, dim, num_embeddings, direct=direct,
                base_offset=self.weight_offset,
            )
            self._scale_rows = PagedRowSource(
                reader, path, dim // block, num_embeddings, direct=direct,
                base_offset=self.scale_offset,
            )

    def _fetch(self, keys: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(keys - self._tag)
        # The file reader needs CPU destinations; a default device (the reference
        # oracle sets "cuda") must not move them.
        weight = torch.empty((ids.numel(), self.dim), dtype=torch.uint8, device="cpu")
        scale = torch.empty((ids.numel(), self.dim // self.block), dtype=torch.uint8, device="cpu")
        self._weight_rows.read_rows(ids, weight)
        self._scale_rows.read_rows(ids, scale)
        return torch.cat([weight, scale], dim=1).numpy()

    def _rows(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cache is None:
            return self.weight[flat], self.scale[flat]
        rows = self.cache.lookup(flat.astype(np.int64) + self._tag, self._fetch)
        return np.ascontiguousarray(rows[:, : self.dim]), np.ascontiguousarray(rows[:, self.dim :])

    @classmethod
    def open(cls, table_dir: str, layer_id: int, num_embeddings: int, dim: int) -> "EngramFileTable":
        if (
            envs.SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT.get()
            and not envs.SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING.get()
        ):
            raise ValueError(
                "SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT serves from the native store of "
                "SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING; set both"
            )
        weight_key = f"layers.{layer_id}.engram.embed.weight"
        for path in sorted(glob.glob(os.path.join(table_dir, "*.safetensors"))):
            _, header = read_safetensors_header(path)
            if weight_key in header:
                from sglang.srt.layers.engram_row_cache import shared_engram_row_cache

                scale_key = f"layers.{layer_id}.engram.embed.scale"
                table = cls(
                    path, weight_key, scale_key, num_embeddings, dim,
                    cache=(None if envs.SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING.get()
                           else shared_engram_row_cache(dim + dim // 32)), cache_tag=layer_id,
                )
                if (
                    envs.SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING.get()
                ):
                    if torch.version.cuda is None or not torch.cuda.is_available():
                        raise RuntimeError("SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING requires CUDA")
                    try:
                        from sglang.srt.layers.engram_host_node import native_engram_host_node

                        table._host_node_extension = native_engram_host_node()
                        table._native_store = table._host_node_extension.get_shared_store(
                            native_cache_bytes(), dim + dim // 32
                        )
                        log_native_stats(table._native_store)
                        log_ring_stats(table._host_node_extension)
                        table._native_store.register_table(
                            path, table.weight_offset, table.scale_offset,
                            num_embeddings, dim, dim // table.block, layer_id << 40,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Engram CUDA host-node io_uring route initialization failed"
                        ) from exc
                return table
        raise FileNotFoundError(f"{weight_key} not found in {table_dir}")

    def _native_stats(self) -> dict:
        """The native store's counters and each device-wait lookup's wait statistics."""
        stats = self._native_store.stats()
        stats["device_waits"] = list(self._host_node_extension.ring_stats())
        return stats

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        assert_not_capturing("EngramFileTable.lookup")
        from sglang.srt.layers.moe.hot_cache_calibration import calibrating

        if calibrating():
            # The startup hot-cache calibration's dummy prefill: every lookup would insert into the shared row
            # cache, so it gets zero rows of the same shape instead (no read, no insert).
            return torch.zeros(*indices.shape, self.dim, dtype=torch.bfloat16, device=indices.device)
        flat = indices.reshape(-1).cpu().numpy()
        if self._native_store is not None:
            rows = self._native_store.lookup(flat.astype(np.int64), self._tag)
            if self.cache is None:
                from sglang.srt.layers.engram_row_cache import cache_stats_sink

                sink = cache_stats_sink()
                if sink is not None:
                    sink.maybe_write("engram", self._native_stats)
            weight_rows = np.ascontiguousarray(rows[:, : self.dim])
            scale_rows = np.ascontiguousarray(rows[:, self.dim :])
            weight = torch.from_numpy(weight_rows).to(indices.device).view(torch.float8_e4m3fn)
            scale = torch.from_numpy(scale_rows).to(indices.device).view(torch.float8_e8m0fnu)
            values = weight.float().unflatten(-1, (-1, self.block)) * scale.float().unsqueeze(-1)
            return values.flatten(-2).to(torch.bfloat16).reshape(*indices.shape, self.dim)
        weight_rows, scale_rows = self._rows(flat)
        weight = torch.from_numpy(weight_rows).to(indices.device).view(torch.float8_e4m3fn)
        scale = torch.from_numpy(scale_rows).to(indices.device).view(torch.float8_e8m0fnu)
        values = weight.float().unflatten(-1, (-1, self.block)) * scale.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).reshape(*indices.shape, self.dim)
