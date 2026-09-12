import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)


def test_logits_processor_output_buffers_tensor_fields_and_preserves_metadata():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    metadata = {"request": ["warmup"]}
    top_logprobs = [[("token", -0.5)]]
    output = LogitsProcessorOutput(
        next_token_logits=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        hidden_states=torch.tensor([[7.0], [8.0], [9.0]]),
        next_token_logprobs=torch.tensor([0.1, 0.2, 0.3]),
        input_token_logprobs=torch.tensor([0.4, 0.5, 0.6]),
        full_logits=torch.tensor([[10.0], [11.0], [12.0]]),
        mm_input_embeds=torch.tensor([[13.0], [14.0], [15.0]]),
        next_token_top_logprobs_val=top_logprobs,
        customized_info=metadata,
    )

    buffer = backend._alloc_full_buffer(output, 5)

    assert isinstance(buffer, LogitsProcessorOutput)
    assert buffer.next_token_logits.shape == (5, 2)
    assert buffer.hidden_states.shape == (5, 1)
    assert buffer.next_token_logprobs.shape == (5,)
    assert buffer.input_token_logprobs.shape == (5,)
    assert buffer.full_logits.shape == (5, 1)
    assert buffer.mm_input_embeds.shape == (5, 1)
    assert buffer.next_token_top_logprobs_val is top_logprobs
    assert buffer.customized_info is metadata

    backend._copy_output_to_buffer(output, buffer, 3)
    sliced = backend._slice_output(buffer, 2)

    assert torch.equal(sliced.next_token_logits, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(sliced.hidden_states, torch.tensor([[7.0], [8.0]]))
    assert torch.equal(sliced.next_token_logprobs, torch.tensor([0.1, 0.2]))
    assert torch.equal(sliced.input_token_logprobs, torch.tensor([0.4, 0.5]))
    assert torch.equal(sliced.full_logits, torch.tensor([[10.0], [11.0]]))
    assert torch.equal(sliced.mm_input_embeds, torch.tensor([[13.0], [14.0]]))
    assert sliced.next_token_top_logprobs_val is top_logprobs
    assert sliced.customized_info is metadata
