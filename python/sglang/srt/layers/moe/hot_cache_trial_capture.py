"""A throwaway CUDA graph capture that measures what the real one will cost (SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP).

The calibration sizes the hot cache before capture, and nothing can shrink after it (the graphs bake the slot count),
so it needs capture's cost up front. An eager dummy forward under-reads it (calib6/7: 30 MiB against 214 measured):
capture keeps static outputs and blocks it cannot reuse across streams. This captures the target verify graph and the
draft graph for real, into a private graph pool swapped in for the process-wide one, measures free memory and torch's
reserve while they are alive, then tears them down (each graph backend's ``cleanup``) and gives the memory back.

What a capture leaves behind and how this handles it:
* the GPU residency updater: only ModelRunner.init_cuda_graphs enables it (discard_graph_capture_routes); the target
  is captured through ``capture_cuda_graphs`` directly, so it stays disabled. The caller resets the warm-up's routes
  and counters (``reset_trial_routes``) and checks the updater is still disabled;
* the DSpark draft's sampler hook: built and appended once (dspark_worker_v2), so the real capture reuses it;
* the cached Exl3FusedMoE pointer tables: the caller drops them (forget_graph_path_state) before any resize;
* the DSV4 attention backend's capture metadata keeps some trial blocks until the real capture replaces it: measured
  as ``residue_torch`` and not credited back to the estimate (the pessimistic side).
Observability forward hooks would be registered twice, so with them configured the trial is skipped.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

MIB = 1 << 20


@dataclass(frozen=True)
class TrialCapture:
    free_drop: int  # free device memory the captured graphs took while alive
    torch_reserved: int  # of it: torch's reserve (the graph pool)
    persistent: int  # free memory not returned after teardown (module loads, the residue below)
    residue_torch: int  # torch reserve still held after teardown (the attention backend's capture metadata)
    seconds: float

    @property
    def nontorch(self) -> int:
        return self.free_drop - self.torch_reserved

    @property
    def estimate(self) -> int:
        """What the real capture will take from free memory measured after the trial: the trial's drop, less what
        stayed taken outside torch (module loads, already out of that free memory). The pinned residue is not
        credited, though the real capture releases it."""
        return max(0, self.free_drop - max(0, self.persistent - self.residue_torch))

    def summary(self) -> dict:
        return {
            "free_drop_mib": round(self.free_drop / MIB, 1),
            "torch_reserved_mib": round(self.torch_reserved / MIB, 1),
            "outside_torch_mib": round(self.nontorch / MIB, 1),
            "persistent_mib": round(self.persistent / MIB, 1),
            "residue_torch_mib": round(self.residue_torch / MIB, 1),
            "estimate_mib": round(self.estimate / MIB, 1),
            "seconds": round(self.seconds, 2),
        }


def _state(device):
    torch.cuda.synchronize(device)
    return torch.cuda.mem_get_info(device)[0], torch.cuda.memory_reserved(device)


def _release(runner) -> None:
    backend = getattr(runner, "backend", None)
    cleanup = getattr(backend, "cleanup", None)
    if cleanup is not None:
        cleanup()


def trial_capture(model_runner, draft_worker=None) -> Optional[TrialCapture]:
    """Capture the target's graphs (and the draft worker's, when given) into a private pool, measure, tear down.
    None when it cannot run safely (observability forward hooks configured)."""
    from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import capture_cuda_graphs
    from sglang.srt.model_executor.runner_utils.pool import (
        get_global_graph_memory_pool,
        set_global_graph_memory_pool,
    )
    from sglang.srt.runtime_context import get_observability

    if get_observability().forward_hooks:
        logger.warning("hot cache trial capture skipped: observability forward hooks would be registered twice")
        return None
    device = model_runner.device
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free0, reserved0 = _state(device)
    saved_pool = get_global_graph_memory_pool()
    set_global_graph_memory_pool(torch.cuda.graph_pool_handle())
    runners, draft_runners = [], ()
    try:
        target = capture_cuda_graphs(model_runner=model_runner)
        runners += [target.decode.runner, target.prefill.runner]
        del target
        if draft_worker is not None:
            draft_worker.init_cuda_graphs()
            draft_runners = tuple(draft_worker._draft_model_runners())
            for runner in draft_runners:
                runners += [runner.decode_cuda_graph_runner, runner.prefill_cuda_graph_runner]
        free1, reserved1 = _state(device)
    finally:
        for runner in runners:
            _release(runner)
        for runner in draft_runners:  # the real capture assigns them again
            runner.decode_cuda_graph_runner = None
            runner.prefill_cuda_graph_runner = None
        runners.clear()
        gc.collect()  # a graph backend and its runner reference each other
        set_global_graph_memory_pool(saved_pool)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    free2, reserved2 = _state(device)
    return TrialCapture(
        free_drop=free0 - free1,
        torch_reserved=reserved1 - reserved0,
        persistent=max(0, free0 - free2),
        residue_torch=max(0, reserved2 - reserved0),
        seconds=time.perf_counter() - started,
    )


def reset_trial_routes(manager) -> None:
    """Undo what the trial's warm-up and capture forwards recorded, the way the real capture does
    (discard_graph_capture_routes), but leave the GPU residency updater disabled: enabling it is the real capture's,
    and once enabled no resize is allowed."""
    manager.discard_graph_capture_routes()
    updater = getattr(manager, "gpu_residency", None)
    if updater is not None:
        updater.enabled.fill_(False)
