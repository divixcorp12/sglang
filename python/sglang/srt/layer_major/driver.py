"""Layer-major prefill: every chunk through a layer before the next layer."""

from __future__ import annotations

from typing import Any

from sglang.srt.layer_major.heartbeat import current_heartbeat
from sglang.srt.layer_major.protocols import ExpertResidency, LayerMajorModelAdapter
from sglang.srt.layer_major.state_store import StateStore


def run_pass(
    adapter: LayerMajorModelAdapter,
    residency: ExpertResidency,
    forward_batch: Any,
    schedule_batch: Any,
    store: StateStore,
) -> Any:
    heartbeat = current_heartbeat()
    handle = adapter.begin_pass(forward_batch, schedule_batch, store)
    failed = True
    restored = False
    output = None
    pass_exc: BaseException | None = None
    try:
        layers = adapter.layer_ids(handle)
        residency.begin(layers)
        for layer_id in layers:
            residency.make_resident(layer_id)
            for chunk in range(adapter.num_chunks(handle)):
                adapter.run_layer(handle, layer_id, chunk, store)
                heartbeat.tick()
        # Mark before calling: a failing restore() must not be retried by the cleanup below.
        restored = True
        residency.restore()
        output = adapter.finish_pass(handle, store)
        failed = False
    except BaseException as exc:
        pass_exc = exc

    # Every cleanup step must run even if an earlier one raises, and the pass's own exception (if
    # any) always wins over one raised during cleanup; a cleanup-only exception propagates after
    # every step below has run.
    cleanup_exc: BaseException | None = None
    try:
        # Borrowed expert bytes must be back before anything else reads them, even on failure.
        if not restored:
            residency.restore()
    except BaseException as exc:
        cleanup_exc = exc
    try:
        residency.end()
    except BaseException as exc:
        cleanup_exc = cleanup_exc if cleanup_exc is not None else exc
    try:
        adapter.release_pass(handle, store, failed=failed)
    except BaseException as exc:
        cleanup_exc = cleanup_exc if cleanup_exc is not None else exc

    if pass_exc is not None:
        raise pass_exc
    if cleanup_exc is not None:
        raise cleanup_exc
    return output
