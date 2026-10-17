"""The scheduler's end of the server's startup warm-up: arm the EXL3 RAM-miss copy engine once it is over.

The HTTP process sends ``EndStartupWarmupReqInput`` after its last warm-up request and before it reports ready, and
waits for the reply. The copy engine's arming calibrates the CPU split and split model, whose timings any other CPU,
PCIe or NVMe traffic skews, so it runs here with the scheduler idle and the box quiet (``copy_engine_quiet``), bounded
by ``QUIET_TIMEOUT_S``. With fewer than ``COPY_ENGINE_ARM_DECODES`` decode forwards by then, the reply goes out at once
and the arming waits for the first idle moment after enough decodes.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Callable, Optional

from sglang.srt.layers.moe.copy_engine_quiet import QUIET_TIMEOUT_S
from sglang.srt.managers.io_struct import (
    EndStartupWarmupReqInput,
    EndStartupWarmupReqOutput,
)

logger = logging.getLogger(__name__)

_RAM_MISS_MODULE = "sglang.srt.layers.moe.exl3_ram_miss"


def ram_miss_service():
    """The EXL3 RAM-miss service when one exists, without importing its module."""
    module = sys.modules.get(_RAM_MISS_MODULE)
    return None if module is None else module.Exl3RamMissService._instance


def arm_decodes() -> int:
    return sys.modules[_RAM_MISS_MODULE].COPY_ENGINE_ARM_DECODES


class SchedulerStartupWarmupEnd:
    def __init__(
        self,
        *,
        is_fully_idle: Callable[[], bool],
        send_output: Callable[[EndStartupWarmupReqOutput, EndStartupWarmupReqInput], None],
        service: Callable[[], object] = ram_miss_service,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        timeout_s: float = QUIET_TIMEOUT_S,
    ) -> None:
        self._is_fully_idle = is_fully_idle
        self._send_output = send_output
        self._service = service
        self._clock = clock
        self._wall_clock = wall_clock
        self._timeout_s = timeout_s
        self._warmup_ended = False
        self._warmup_ended_at: Optional[float] = None
        # The request awaiting its reply (the HTTP process holds ready until it lands).
        self._pending: Optional[EndStartupWarmupReqInput] = None
        # time.monotonic() when the arming became due (warm-up over, enough decodes), and its quiet deadline.
        self._due_at: Optional[float] = None
        self._busy_since: Optional[float] = None
        self._scheduler_busy_s = 0.0

    def handle(self, recv_req: EndStartupWarmupReqInput) -> Optional[EndStartupWarmupReqOutput]:
        if self._warmup_ended:
            return EndStartupWarmupReqOutput(success=True, message="startup warm-up already ended")
        self._warmup_ended = True
        self._warmup_ended_at = recv_req.warmup_ended_at
        self._pending = recv_req
        return self._step(reply_inline=True)

    def check_pending(self) -> None:
        """Run from every scheduler loop iteration: arm once due, idle and quiet."""
        if self._warmup_ended and (self._pending is not None or self._due_at is not None or self._arm_pending()):
            self._step(reply_inline=False)

    def _arm_pending(self) -> bool:
        service = self._service()
        return service is not None and service.copy_engine_arm_pending

    def _reply(self, output: EndStartupWarmupReqOutput, *, reply_inline: bool) -> Optional[EndStartupWarmupReqOutput]:
        pending, self._pending = self._pending, None
        if reply_inline:
            return output
        if pending is not None:
            self._send_output(output, pending)
        return None

    def _step(self, *, reply_inline: bool) -> Optional[EndStartupWarmupReqOutput]:
        service = self._service()
        if service is None or not service.copy_engine_arm_pending:
            self._due_at = None
            return self._reply(EndStartupWarmupReqOutput(success=True, message="no copy engine to arm"),
                               reply_inline=reply_inline)
        now = self._clock()
        if self._due_at is None:
            decodes = service.copy_decodes
            if decodes < arm_decodes():
                if self._pending is not None:
                    logger.info(
                        "exl3 RAM miss copy engine: startup warm-up ended after %d of %d decode forwards; arming at "
                        "the first idle moment after enough decodes",
                        decodes,
                        arm_decodes(),
                    )
                return self._reply(
                    EndStartupWarmupReqOutput(success=True, message=f"copy engine arming deferred ({decodes} decodes)"),
                    reply_inline=reply_inline,
                )
            self._due_at = now
        deadline = self._due_at + self._timeout_s
        if not self._is_fully_idle():
            if self._busy_since is None:
                self._busy_since = now
            if now < deadline:
                return None
        if self._busy_since is not None:
            self._scheduler_busy_s += now - self._busy_since
            self._busy_since = None
        busy_s = {"scheduler_requests": self._scheduler_busy_s} if self._scheduler_busy_s else {}
        ended_s_ago = None if self._warmup_ended_at is None else self._wall_clock() - self._warmup_ended_at
        report = service.arm_copy_engine_when_quiet(deadline=deadline, busy_s=busy_s, warmup_ended_s_ago=ended_s_ago)
        self._due_at = None
        return self._reply(
            EndStartupWarmupReqOutput(success=True, message=f"copy engine armed; {report.line()}"),
            reply_inline=reply_inline,
        )
