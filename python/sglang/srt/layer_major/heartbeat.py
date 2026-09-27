"""Progress of the running layer-major pass, read by the scheduler watchdog: one pass is a single forward that can
outlast the watchdog timeout, and forward_ct only moves between forwards."""

from __future__ import annotations

import itertools


class PassHeartbeat:
    def __init__(self):
        self._counter = itertools.count(1)
        self._value = 0

    def tick(self) -> None:
        # itertools.count is atomic under the GIL; the watchdog thread only reads _value.
        self._value = next(self._counter)

    @property
    def value(self) -> int:
        return self._value


_HEARTBEAT = PassHeartbeat()


def current_heartbeat() -> PassHeartbeat:
    return _HEARTBEAT


def pass_progress() -> int:
    return _HEARTBEAT.value
