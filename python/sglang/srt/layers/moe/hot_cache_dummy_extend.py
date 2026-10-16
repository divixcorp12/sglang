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


def dummy_verify_forward(model_runner) -> Callable[[], None]:
    """A callable that runs graph capture's own dummy forward (TARGET_VERIFY on a speculative target, one request)
    eagerly: BaseRunner._dummy_run reads nothing but ``self.model_runner``, so it runs on a stand-in. Its peak
    transient estimates what the captured graph's private pool will hold. Afterwards, ``forget_graph_path_state``
    drops what a graph-path forward builds lazily from the hot cache's pre-retire capacity."""
    from types import SimpleNamespace

    from sglang.srt.model_executor.runner.base_runner import BaseRunner

    runner = SimpleNamespace(model_runner=model_runner)
    per_req = model_runner.decode_num_tokens_per_req()
    buffers = BaseRunner._alloc_dummy_decode_buffers(
        runner, 1, num_tokens_per_req=per_req, allocate_logits_buffer=False
    )
    return lambda: BaseRunner._dummy_run(runner, 1, buffers=buffers)


def forget_graph_path_state(model) -> int:
    """Delete the per-layer Exl3FusedMoE an eager graph-path forward caches (its slot pointer tables span the hot
    cache's capacity at the time); the first real graph-path forward rebuilds it. Returns how many it dropped."""
    dropped = 0
    for module in model.modules():
        if hasattr(module, "_exl3_fused_moe"):
            del module._exl3_fused_moe
            dropped += 1
    return dropped


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
