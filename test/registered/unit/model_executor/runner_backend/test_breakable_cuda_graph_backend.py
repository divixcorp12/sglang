from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend import (
    breakable_cuda_graph_backend as backend_module,
)
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


class _Key:
    def __init__(self, size):
        self.size = size


class _RowsPerRequestRunner:
    def __init__(self, rows_per_request):
        self.rows_per_request = rows_per_request

    def capture_output_rows(self, size):
        return size * self.rows_per_request

    def cuda_graph_output_rows(self, output):
        return None

    def cuda_graph_output_capacity_rows(self, output):
        return None


def test_capture_output_rows_come_from_the_runner_and_default_to_the_key_size():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)

    backend._cuda_graph_runner = object()
    assert backend._capture_output_rows(_Key(1)) == 1

    backend._cuda_graph_runner = _RowsPerRequestRunner(4)
    assert backend._capture_output_rows(_Key(1)) == 4
    assert backend._capture_output_rows(_Key(2)) == 8


def test_shared_output_buffer_grows_for_a_wider_capture_and_keeps_earlier_slices():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._cuda_graph_runner = _RowsPerRequestRunner(1)
    backend._shared_output_buffer = None
    backend._shared_output_rows = 0
    narrow = LogitsProcessorOutput(
        next_token_logits=torch.full((1, 3), 7.0), hidden_states=torch.full((1, 2), 7.0)
    )
    wide = LogitsProcessorOutput(
        next_token_logits=torch.arange(12.0).reshape(4, 3),
        hidden_states=torch.arange(8.0).reshape(4, 2),
    )

    first = backend._reserve_shared_output_buffer(narrow, 1)
    backend._copy_output_to_buffer(narrow, first, 1)
    earlier = backend._slice_output(first, 1)
    second = backend._reserve_shared_output_buffer(wide, 4)

    assert second is not first
    assert second.next_token_logits.shape == (4, 3)
    assert second.hidden_states.shape == (4, 2)
    assert backend._reserve_shared_output_buffer(narrow, 1) is second

    backend._copy_output_to_buffer(wide, second, backend._output_rows(wide, 4))
    stored = backend._slice_output(second, backend._output_rows(wide, 4))
    assert torch.equal(stored.next_token_logits, wide.next_token_logits)
    assert torch.equal(stored.hidden_states, wide.hidden_states)
    assert torch.equal(earlier.next_token_logits, torch.full((1, 3), 7.0))
    assert torch.equal(earlier.hidden_states, torch.full((1, 2), 7.0))


def test_copy_rejects_a_field_with_fewer_rows_than_the_capture_stores():
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._cuda_graph_runner = _RowsPerRequestRunner(1)
    output = LogitsProcessorOutput(
        next_token_logits=torch.zeros(4, 3), hidden_states=torch.zeros(1, 2)
    )
    buffer = backend._alloc_full_buffer(output, 4)

    with pytest.raises(ValueError, match="1 rows but the capture stores 4"):
        backend._copy_output_to_buffer(output, buffer, backend._output_rows(output, 4))


class _Precarve:
    @contextmanager
    def measure(self):
        yield

    def mint(self):
        pass


class _Graph:
    def __init__(self, deduped_cuda_graph):
        self._segments = []
        self._break_fns = []


def test_capture_one_stores_every_verify_row_of_a_request_keyed_graph(monkeypatch):
    monkeypatch.setattr(backend_module, "BreakableCUDAGraph", _Graph)
    monkeypatch.setattr(
        backend_module, "BreakableCUDAGraphCapture", lambda **kwargs: nullcontext()
    )
    monkeypatch.setattr(backend_module, "graph_pool_capture_scope", nullcontext)
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._cuda_graph_runner = _RowsPerRequestRunner(4)
    backend._device_module = SimpleNamespace(synchronize=lambda: None)
    backend._tp_group = SimpleNamespace(barrier=lambda: None)
    backend._precarve = _Precarve()
    backend._debug_eager = False
    backend._pool = None
    backend._capture_stream = None
    backend._shared_output_buffer = None
    backend._shared_output_rows = 0
    backend._graphs, backend._outputs, backend._capture_inputs = {}, {}, {}
    verify = LogitsProcessorOutput(
        next_token_logits=torch.arange(8.0).reshape(4, 2),
        hidden_states=torch.arange(12.0).reshape(4, 3),
    )
    key = _Key(1)

    backend.capture_one(key, lambda: verify)

    stored = backend._outputs[key]
    assert torch.equal(stored.next_token_logits, verify.next_token_logits)
    assert torch.equal(stored.hidden_states, verify.hidden_states)
