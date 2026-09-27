"""Which requests the layer-major prefill takes, and which launches cannot use it at all."""

from __future__ import annotations

import msgspec

from sglang.srt.environ import envs


class LayerMajorGate(msgspec.Struct, frozen=True):
    min_tokens: int
    max_tokens: int

    def admits(self, *, extend_len: int, wants_prompt_logprobs: bool, wants_hidden_states: bool) -> bool:
        # Prompt logprobs and hidden states need every prompt row past the last layer; the pass computes the tail only.
        if wants_prompt_logprobs or wants_hidden_states:
            return False
        return self.min_tokens <= extend_len <= self.max_tokens


def gate_from_env(*, max_tokens: int) -> LayerMajorGate | None:
    min_tokens = envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get()
    if min_tokens <= 0:
        return None
    return LayerMajorGate(min_tokens=min_tokens, max_tokens=max_tokens)


def launch_refusal(
    *,
    max_running_requests: int | None,
    speculative_algorithm: str | None,
    enable_dp_attention: bool,
    attn_cp_size: int,
    enable_two_batch_overlap: bool,
    pp_size: int = 1,
) -> str | None:
    # One blocking pass per request; nothing may run between its chunks.
    if max_running_requests != 1:
        return "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS requires --max-running-requests 1"
    for feature, enabled in (
        ("speculative decoding", speculative_algorithm is not None),
        ("DP attention", enable_dp_attention),
        ("context parallelism", attn_cp_size > 1),
        ("two-batch overlap", enable_two_batch_overlap),
        # The PP event loop (scheduler_pp_mixin.py) has no exception containment
        # and cannot consume run_batch's placeholder failure result.
        ("pipeline parallelism", pp_size > 1),
    ):
        if enabled:
            return f"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS cannot be combined with {feature} yet"
    return None


def scheduler_layer_major_refusal(
    *,
    raw_chunked_prefill_size: int | None,
    effective_chunked_prefill_size: int | None,
    is_hybrid_swa_allocator: bool,
    is_swa_req_ring: bool,
    adapter_missing_model_name: str | None,
) -> str | None:
    """The refusals that need scheduler/allocator/model state gate_from_env's caller can only
    gather once init_model_worker has run, not just the raw server args launch_refusal checks."""
    if effective_chunked_prefill_size is None:
        return "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS requires chunked prefill to be enabled (--chunked-prefill-size > 0)"
    if effective_chunked_prefill_size != raw_chunked_prefill_size:
        # The adapter sizes its ring from the launch's raw chunked_prefill_size
        # (models/deepseek_v4_layer_major.py); a scheduler-side override (e.g. the
        # multimodal-transformers-backend disable in init_chunked_prefill) would
        # size the ring differently from what the adapter actually reads.
        return (
            "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS: effective chunked_prefill_size "
            f"({effective_chunked_prefill_size}) differs from the launch value "
            f"({raw_chunked_prefill_size}) the layer-major adapter reads for its ring size"
        )
    if not is_hybrid_swa_allocator:
        # check_prefill_ring only exists on SWAPrefillBudget.
        return "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS requires a hybrid-SWA KV allocator"
    if is_swa_req_ring:
        # The per-request SWA ring (unified-KV DeepSeekV4TokenToKVPool mode) hands
        # every window slot to its owning request up front; there are no shared
        # window slots left for the layer-major ring to claim.
        return (
            "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS cannot be combined with the "
            "per-request SWA ring allocator (unified-KV mode)"
        )
    if adapter_missing_model_name is not None:
        return (
            f"{adapter_missing_model_name} has no layer-major adapter; unset "
            "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS or use a supported model"
        )
    return None
