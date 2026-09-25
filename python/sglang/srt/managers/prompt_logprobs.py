"""Admission checks for prompt-token logprobs (``logprob_start_len`` inside the prompt).

Kept out of scheduler.py so it can be tested without importing the scheduler's dependencies.
"""

from typing import Optional


def prompt_logprob_refusal(
    *,
    mlx_sampling: bool,
    decoder_swa_bounded_replay: bool,
    return_logprob: bool,
    logprob_start_len: int,
    input_len: int,
) -> Optional[str]:
    """Why a request asking for prompt-token logprobs must be refused, or None.

    Each refused path would otherwise fail inside the forward, where an exception stops the
    scheduler and so the whole server, for every request in flight.
    """
    if not (return_logprob and 0 <= logprob_start_len < input_len):
        return None
    if mlx_sampling:
        # The MLX sampling path computes output logprobs only; the prefill result carries
        # no input_token_logprobs, so letting this through would crash output processing.
        return (
            "Prompt input logprobs (logprob_start_len) are not supported "
            "on the MLX sampling path; omit logprob_start_len to get "
            "output logprobs."
        )
    if decoder_swa_bounded_replay:
        # Rows outside the bounded replay's tail are never computed past the last kv_source
        # layer, so their logits do not exist; the model raises in
        # _check_late_layer_tail_readers.
        return (
            "Prompt input logprobs (logprob_start_len, echo with logprobs) are not "
            "supported with --enable-decoder-swa-bounded-replay; omit "
            "logprob_start_len to get output logprobs."
        )
    return None
