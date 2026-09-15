"""Chain token hashes per request so identical prefixes share keys across requests."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Sequence

UNKNOWN_PREFIX = 0
_BROKEN_CHAIN = -1


def _step(previous: int, token_id: int) -> int:
    digest = hashlib.blake2b(
        previous.to_bytes(8, "little", signed=True)
        + token_id.to_bytes(8, "little", signed=True),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=True) or 1


class PrefixHasher:
    """``hash_rows`` returns the chain hash of tokens ``0..position`` for each row."""

    def __init__(self, *, max_requests: int) -> None:
        self._max_requests = max_requests
        # rid -> (next expected position, chain hash so far)
        self._state: OrderedDict[str, tuple[int, int]] = OrderedDict()

    def hash_rows(
        self, *, rid: str, positions: Sequence[int], token_ids: Sequence[int]
    ) -> list[int]:
        next_position, value = self._state.pop(rid, (0, UNKNOWN_PREFIX))
        hashes = []
        for position, token_id in zip(positions, token_ids):
            if next_position == _BROKEN_CHAIN or position != next_position:
                next_position = _BROKEN_CHAIN
                hashes.append(UNKNOWN_PREFIX)
                continue
            value = _step(value, token_id)
            hashes.append(value)
            next_position = position + 1
        self._state[rid] = (next_position, value)
        while len(self._state) > self._max_requests:
            self._state.popitem(last=False)
        return hashes


class SeenPrefixes:
    """Drop prefill rows whose exact token prefix was already captured."""

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def keep_mask(self, *, hashes: Sequence[int], is_prefill: bool) -> list[bool]:
        keep = []
        for value in hashes:
            known = value != UNKNOWN_PREFIX
            keep.append(not (is_prefill and known and value in self._seen))
            if known:
                self._seen.add(value)
        return keep
