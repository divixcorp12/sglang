"""Where a TP worker hands a layer-major batch to the strategy, in place of model_runner.forward."""

from __future__ import annotations

import logging
from typing import Any

import msgspec

from sglang.srt.environ import envs
from sglang.srt.layer_major.driver import run_pass
from sglang.srt.layer_major.protocols import NullResidency
from sglang.srt.layer_major.state_store import StateStore

logger = logging.getLogger(__name__)


class LayerMajorRuntime(msgspec.Struct):
    adapter: Any
    store: StateStore
    residency: Any

    def release(self) -> None:
        """Release the runtime's StateStore. The store is always built pinned (layer_major_runtime
        passes pin=True), so its host memory must not be freed while a copy queued against it is
        still in flight: the CUDA stream is synchronized first."""
        import torch

        torch.cuda.current_stream().synchronize()
        self.store.release()


def layer_major_runtime(model_runner) -> LayerMajorRuntime | None:
    if envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get() <= 0:
        return None
    make = getattr(type(model_runner.model), "make_layer_major_adapter", None)
    if make is None:
        raise ValueError(f"{type(model_runner.model).__name__} has no layer-major adapter")
    adapter = make(model_runner.model, model_runner)
    store = StateStore(
        adapter.field_specs(),
        model_runner.model_config.context_len,
        numa_node=envs.SGLANG_LAYER_MAJOR_STATE_NUMA_NODE.get(),
        pin=True,
    )
    return LayerMajorRuntime(adapter=adapter, store=store, residency=NullResidency())


def run_layer_major_prefill(runtime: LayerMajorRuntime, model_runner, schedule_batch, forward_batch):
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput

    with forward_context(ForwardContext(attn_backend=model_runner.attn_backend)):
        with get_global_expert_distribution_recorder().with_forward_pass(model_runner.forward_pass_id, forward_batch):
            logits_output = run_pass(runtime.adapter, runtime.residency, forward_batch, schedule_batch, runtime.store)

    # Logged only once run_pass has actually returned: schedule_batch.layer_major_ring_tokens and
    # server_args are real fields on the request that ran, not on a pass that raised.
    tokens = schedule_batch.layer_major_ring_tokens
    num_chunks = -(-tokens // model_runner.server_args.chunked_prefill_size) if tokens else 0
    logger.info("layer-major prefill: %d tokens in %d chunks", tokens or 0, num_chunks)
    return ModelRunnerOutput(logits_output=logits_output, can_run_graph=False)
