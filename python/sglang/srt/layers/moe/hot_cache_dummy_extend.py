"""One dummy EXTEND prefill through a speculative target model, for the pre-capture hot cache calibration.

BaseRunner._dummy_run cannot do this: a speculative runner forces TARGET_VERIFY, and the runners that own it exist only
after graph capture. This builds the one-sequence EXTEND ForwardBatch the way _dummy_run's extend branch does, but with
the real DSpark prefill's capture mode (CaptureHiddenMode.FULL) and no speculative input, single rank only (no DP/PP
gather, no LoRA). Token ids are a fixed in-vocabulary sequence, positions run 0..N-1, and out_cache_loc and
req_pool_indices are 0: the KV writes land in the reserved slot 0, as graph capture's dummy forwards do. It calls
model.forward directly, so neither the expert-distribution recorder nor the forward-pass counter runs.
"""

from __future__ import annotations

from typing import Callable

import torch


def dummy_extend_forward(model_runner, tokens: int) -> Callable[[], None]:
    """A callable that runs the dummy EXTEND forward of ``tokens`` tokens; built once, run as often as needed."""
    from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len, set_is_extend_in_batch
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.runtime_context import get_parallel

    mr = model_runner
    if get_parallel().tp_size != 1 or get_parallel().pp_size != 1 or get_parallel().dp_size != 1:
        raise RuntimeError("hot cache calibration's dummy prefill supports a single rank only")
    device = mr.device
    vocab = mr.model_config.vocab_size
    with torch.device(device):
        input_ids = (torch.arange(tokens, dtype=torch.int64) * 7919 + 1000) % (vocab - 2000) + 1000
        positions = torch.arange(tokens, dtype=torch.int64)
        out_cache_loc = torch.zeros(tokens, dtype=torch.int64)
        req_pool_indices = torch.zeros(1, dtype=torch.int64)
        seq_lens = torch.full((1,), tokens, dtype=torch.int64)
        extend_seq_lens = torch.full((1,), tokens, dtype=torch.int32)
        extend_prefix_lens = torch.zeros(1, dtype=torch.int32)
        extend_start_loc = torch.zeros(1, dtype=torch.int32)
    seq_lens_cpu = seq_lens.cpu()

    def run() -> None:
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=tokens,
            return_logprob=False,
            positions=positions,
            extend_num_tokens=tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[tokens],
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            spec_algorithm=mr.spec_algorithm,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            global_forward_mode=ForwardMode.EXTEND,
        )
        forward_batch = mr.prepare_dummy_forward_batch(forward_batch)
        mr.attn_backend.init_forward_metadata(forward_batch)
        set_dp_buffer_len(None, tokens, False, None)
        set_is_extend_in_batch(True)
        with forward_context(ForwardContext(attn_backend=mr.attn_backend)):
            with torch.inference_mode():
                mr.model.forward(input_ids, forward_batch.positions, forward_batch)

    return run
