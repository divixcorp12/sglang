"""C1: a layer-major pass must run the late layers over every span whose tail rows fall inside
decode's reach [s - window, s), not just the final span. Mirrors chunked prefill, where every
extend runs its own tail. RED at 43d7813a41: tail_run_spans does not exist there, and finish_pass
always called forward_late_tail exactly once, on the final span only."""

import types
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v4_layer_major import (
    DSV4_WINDOW,
    DeepseekV4LayerMajorAdapter,
    chunk_spans,
    tail_run_spans,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 4096


class TestTailRunSpans(unittest.TestCase):
    """Pure write-coverage check: the spans selected must cover [s - window, s)."""

    def _spans(self, seq_len):
        return chunk_spans(prefix_len=0, seq_len=seq_len, chunk=CHUNK)

    def test_exact_multiple_of_chunk_needs_only_the_final_span(self):
        spans = self._spans(CHUNK * 2)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [1])

    def test_final_span_of_8_rows_also_needs_the_penultimate_tail(self):
        spans = self._spans(CHUNK * 2 + 8)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [1, 2])

    def test_final_span_of_127_rows_also_needs_the_penultimate_tail(self):
        spans = self._spans(CHUNK * 2 + 127)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [1, 2])

    def test_final_span_of_exactly_the_window_needs_no_penultimate_tail(self):
        spans = self._spans(CHUNK * 2 + DSV4_WINDOW)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [2])

    def test_single_span_suffix_needs_only_itself(self):
        # No penultimate span exists; this is the prefix case (review focus item 1), not C1.
        spans = self._spans(8)
        needed = tail_run_spans(spans, window=DSV4_WINDOW, prefix_len=0)
        self.assertEqual([s.index for s in needed], [0])


class _RecordingModel:
    """Stand-in causal_lm: records which span (and how many rows) the late layers ran on."""

    def __init__(self):
        self.calls = []

    def forward_late_tail(self, *, forward_batch, hidden_states, prev_pre, hash_ids):
        self.calls.append((forward_batch.span_index, hidden_states.shape[0]))
        return SimpleNamespace(output_for=forward_batch.span_index)


class _RecordingStore:
    def __init__(self, meta_by_index):
        self._meta_by_index = meta_by_index

    def read_into(self, name, offset, tensor, stream=None):
        pass

    def unpark(self, index, device):
        return self._meta_by_index[index]


def _run_finish_pass(*, seq_len, prefix_len=0, chunk=CHUNK, window=DSV4_WINDOW):
    spans = chunk_spans(prefix_len=prefix_len, seq_len=seq_len, chunk=chunk)
    tail_spans = tail_run_spans(spans, window=window, prefix_len=prefix_len)
    tail_by_span = {s.index: f"TAIL{s.index}" for s in tail_spans}
    forward_batches = [SimpleNamespace(span_index=s.index) for s in spans]
    handle = SimpleNamespace(
        spans=spans,
        forward_batches=forward_batches,
        hash_ids=[None for _ in spans],
        schedule_batch=SimpleNamespace(prefix_lens=[prefix_len]),
        tail_by_span=tail_by_span,
        finalized=False,
    )
    finalize_calls = []
    model = _RecordingModel()
    backend_calls = []
    backend = SimpleNamespace(
        install_forward_metadata=lambda meta, tail_metadata=None: backend_calls.append((meta, tail_metadata))
    )
    meta_by_index = {s.index: f"META{s.index}" for s in spans}
    store = _RecordingStore(meta_by_index)
    adapter_self = SimpleNamespace(
        runner=SimpleNamespace(device="cpu"),
        model=SimpleNamespace(hc_mult=1, hidden_size=4),
        backend=backend,
        causal_lm=model,
        _finalize_ring=lambda h: finalize_calls.append(h),
    )
    adapter_self._run_late_layers = types.MethodType(DeepseekV4LayerMajorAdapter._run_late_layers, adapter_self)
    output = DeepseekV4LayerMajorAdapter.finish_pass(adapter_self, handle, store)
    return SimpleNamespace(
        calls=model.calls, output=output, tail_spans=tail_spans,
        finalize_calls=finalize_calls, backend_calls=backend_calls,
    )


class TestFinishPassRunsEveryNeededTail(unittest.TestCase):
    def test_full_final_chunk_runs_only_its_own_tail(self):
        r = _run_finish_pass(seq_len=CHUNK * 2)
        self.assertEqual([c[0] for c in r.calls], [1])
        self.assertEqual(r.output.output_for, 1)
        self.assertEqual(len(r.finalize_calls), 1)

    def test_short_final_chunk_runs_the_penultimate_tail_before_the_final_tail(self):
        r = _run_finish_pass(seq_len=CHUNK * 2 + 8)
        # Order check: penultimate (index 1) runs before the true final span (index 2).
        self.assertEqual([c[0] for c in r.calls], [1, 2])
        # Only the true final span's output is returned; the penultimate's is discarded.
        self.assertEqual(r.output.output_for, 2)
        self.assertEqual(r.backend_calls, [("META1", "TAIL1"), ("META2", "TAIL2")])
        self.assertEqual(len(r.finalize_calls), 1)

    def test_final_chunk_of_127_rows_still_needs_the_penultimate_tail(self):
        r = _run_finish_pass(seq_len=CHUNK * 2 + 127)
        self.assertEqual([c[0] for c in r.calls], [1, 2])
        self.assertEqual(r.output.output_for, 2)

    def test_boundary_final_chunk_of_exactly_the_window_needs_no_penultimate_tail(self):
        r = _run_finish_pass(seq_len=CHUNK * 2 + DSV4_WINDOW)
        self.assertEqual([c[0] for c in r.calls], [2])
        self.assertEqual(r.output.output_for, 2)


if __name__ == "__main__":
    unittest.main()
