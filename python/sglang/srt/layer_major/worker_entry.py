"""Where a TP worker hands a layer-major batch to the strategy, in place of model_runner.forward."""

from __future__ import annotations

import logging
from typing import Any

import msgspec
import torch

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
        passes pin=True), so its host memory must not be freed while a copy queued against it on any
        stream is still in flight: torch.cuda.synchronize() (not current_stream().synchronize(), since
        the strategy's own copies may be queued on a side stream) runs first."""
        torch.cuda.synchronize()
        self.store.release()


def layer_major_runtime(model_runner) -> LayerMajorRuntime | None:
    if envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get() <= 0:
        return None
    make = getattr(type(model_runner.model), "make_layer_major_adapter", None)
    if make is None:
        raise ValueError(f"{type(model_runner.model).__name__} has no layer-major adapter")
    # These features tap or capture inside the model's normal per-layer forward call; the strategy
    # calls layers directly instead (driver.run_pass), so nothing here can promise their hooks still
    # see every token in the right place. Refuse rather than silently produce wrong shadow scores or
    # captured top-k.
    if model_runner.expert_prediction_runtime is not None:
        raise ValueError(
            "layer-major prefill cannot be combined with MoE expert-prediction shadow mode yet "
            "(SGLANG_MOE_EXPERT_PREDICTOR, SGLANG_MOE_EXPERT_PREFETCH_PREDICTOR or "
            "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR is set)"
        )
    if _capture_enabled():
        raise ValueError("layer-major prefill cannot be combined with routed-experts/indexer top-k capture yet")
    adapter = make(model_runner.model, model_runner)
    store = StateStore(
        adapter.field_specs(),
        model_runner.model_config.context_len,
        numa_node=envs.SGLANG_LAYER_MAJOR_STATE_NUMA_NODE.get(),
        pin=True,
    )
    return LayerMajorRuntime(adapter=adapter, store=store, residency=NullResidency())


def layer_major_runtime_for_worker(model_runner, *, is_draft_worker: bool) -> LayerMajorRuntime | None:
    """A draft worker never runs layer-major prefill. MTP's DeepseekV4ForCausalLMNextN inherits DSV4's
    `make_layer_major_adapter` and would otherwise get its own bogus adapter and a second pinned,
    full-context_len StateStore; a draft worker with no hook at all (DSpark, standalone) would instead
    die at startup on `layer_major_runtime`'s "no layer-major adapter" check. Neither belongs here: only
    the target worker's last-rank forward ever takes the layer-major path."""
    if is_draft_worker:
        return None
    return layer_major_runtime(model_runner)


def takes_layer_major_path(batch, forward_batch) -> bool:
    """Whether TpModelWorker's last-rank branch should route through run_layer_major_prefill instead
    of model_runner.forward. `batch.layer_major_ring_tokens` marks admission for the *request*; a
    decode step on that same request still carries the field (it lives on the ScheduleBatch, not
    cleared between steps) but must take the normal per-token decode path, so forward_mode must be
    extend too."""
    return (
        batch is not None
        and batch.layer_major_ring_tokens is not None
        and forward_batch.forward_mode.is_extend()
    )


def _capture_enabled() -> bool:
    """The launch flags the capturer factories consult (RoutedExpertsCapturer.create,
    routed_experts.py; create_indexer_capturer, indexer_topk.py), not the capturers themselves:
    those are only installed later in startup (ModelRunner._init_post_memory_pool_components, after
    TpModelWorker.__init__ -- and this check -- already ran), so reading them here would never see
    a launch that enables the feature."""
    from sglang.srt.runtime_context import get_exec

    features = get_exec().features
    return features.enable_return_routed_experts or features.enable_return_indexer_topk


def run_layer_major_prefill(runtime: LayerMajorRuntime, model_runner, schedule_batch, forward_batch):
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput

    # Mirrors model_runner.forward's own per-forward bookkeeping around _forward_raw (model_runner.py
    # ~1953-2028, frozen): forward_pass_id advances once per pass, and the recorder's per-pass output
    # dict and the EPLB manager's rebalance counter are driven exactly as forward() drives them, in the
    # same order, from outside model_runner. expert_prediction_runtime and the routed-experts/indexer
    # capturers are not mirrored (see layer_major_runtime): construction already refused them.
    model_runner.forward_pass_id += 1
    with forward_context(ForwardContext(attn_backend=model_runner.attn_backend)):
        with get_global_expert_distribution_recorder().with_forward_pass(
            model_runner.forward_pass_id, forward_batch
        ) as recorder_outputs:
            logits_output = run_pass(runtime.adapter, runtime.residency, forward_batch, schedule_batch, runtime.store)
    output = ModelRunnerOutput(
        logits_output=logits_output,
        can_run_graph=False,
        expert_distribution_metrics=recorder_outputs.get("metrics"),
    )
    if model_runner.eplb_manager is not None:
        model_runner.eplb_manager.on_forward_pass_end()

    tokens = schedule_batch.layer_major_ring_tokens
    num_chunks = -(-tokens // model_runner.server_args.chunked_prefill_size)
    logger.info("layer-major prefill: %d tokens in %d chunks", tokens, num_chunks)
    return output
