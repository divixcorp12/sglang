"""A bounded, least-recently-used host-RAM tier of raw EXL3 expert rows.

Each slot is page-aligned and holds one aligned superset read, so the row sits
at a per-expert offset inside it and RAM carries no padding (DSV41_REFERENCE
§14.1(c)). The VRAM tier copies from here; promotion keeps the RAM copy, and
``is_pinned`` keeps every VRAM-resident row in RAM, so the hierarchy stays
inclusive.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable, Optional

import torch

from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader
from sglang.srt.model_loader.file_row_reader import PAGE_BYTES

Key = tuple[int, int]


class Exl3RamExpertCache:
    def __init__(
        self,
        reader: Exl3RowReader,
        capacity: int,
        *,
        register: bool = False,
        is_pinned: Optional[Callable[[Key], bool]] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("RAM expert cache needs at least one slot")
        self.reader = reader
        self.capacity = capacity
        self.slot_bytes = self.slot_bytes_for(reader)
        self._storage = torch.empty(capacity * self.slot_bytes + PAGE_BYTES, dtype=torch.uint8)
        start = (-self._storage.data_ptr()) % PAGE_BYTES
        self.buffer = self._storage[start : start + capacity * self.slot_bytes].view(
            capacity, self.slot_bytes
        )
        if register:
            from sglang.srt.mem_cache.pool_host.common import _cuda_host_register

            _cuda_host_register(self.buffer, registration_granularity_bytes=self.slot_bytes)
        self._is_pinned = is_pinned or (lambda key: False)
        self._slots: OrderedDict[Key, tuple[int, int]] = OrderedDict()
        self._free = list(range(capacity - 1, -1, -1))
        self.hits = 0
        self.misses = 0

    @staticmethod
    def slot_bytes_for(reader: Exl3RowReader) -> int:
        return -(-reader.buffer_bytes // PAGE_BYTES) * PAGE_BYTES

    def contains(self, key: Key) -> bool:
        return key in self._slots

    def ensure(self, keys: Iterable[Key]) -> list[torch.Tensor]:
        keys = list(keys)
        distinct = list(dict.fromkeys(keys))
        if len(distinct) > self.capacity:
            raise ValueError(
                f"{len(distinct)} experts in one batch exceed the RAM cache capacity {self.capacity}"
            )
        protected = set(distinct)
        missing = []
        for key in distinct:
            if key in self._slots:
                self._slots.move_to_end(key)
                self.hits += 1
            else:
                missing.append(key)
        if missing:
            slots = [self._take_slot(protected) for _ in missing]
            try:
                starts = self.reader.read(missing, [self.buffer[slot].data_ptr() for slot in slots])
            except BaseException:
                self._free.extend(slots)
                raise
            for key, slot, start in zip(missing, slots, starts):
                self._slots[key] = (slot, start)
            self.misses += len(missing)
        row_bytes = self.reader.layout.row_bytes
        out = []
        for key in keys:
            slot, start = self._slots[key]
            out.append(self.buffer[slot, start : start + row_bytes])
        return out

    def _take_slot(self, protected: set[Key]) -> int:
        if self._free:
            return self._free.pop()
        for key in self._slots:  # oldest first
            if key not in protected and not self._is_pinned(key):
                slot, _ = self._slots.pop(key)
                return slot
        raise RuntimeError("RAM expert cache: every slot is pinned or in use by this batch")
