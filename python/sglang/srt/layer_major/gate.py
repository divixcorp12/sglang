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
) -> str | None:
    # One blocking pass per request; nothing may run between its chunks.
    if max_running_requests != 1:
        return "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS requires --max-running-requests 1"
    for feature, enabled in (
        ("speculative decoding", speculative_algorithm is not None),
        ("DP attention", enable_dp_attention),
        ("context parallelism", attn_cp_size > 1),
        ("two-batch overlap", enable_two_batch_overlap),
    ):
        if enabled:
            return f"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS cannot be combined with {feature} yet"
    return None
