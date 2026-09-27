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
    try:
        layers = adapter.layer_ids(handle)
        residency.begin(layers)
        for layer_id in layers:
            residency.make_resident(layer_id)
            for chunk in range(adapter.num_chunks(handle)):
                adapter.run_layer(handle, layer_id, chunk, store)
                heartbeat.tick()
        residency.restore()
        restored = True
        output = adapter.finish_pass(handle, store)
        failed = False
        return output
    finally:
        # Borrowed expert bytes must be back before anything else reads them, even on failure.
        if not restored:
            residency.restore()
        residency.end()
        adapter.release_pass(handle, store, failed=failed)
