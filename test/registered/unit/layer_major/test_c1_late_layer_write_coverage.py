"""C1 write-coverage RED/GREEN test: calls the unbound finish_pass with stand-ins for
causal_lm/backend/store, and checks it ran the late layers over every position in
[max(prefix_len, s-window), s), s = the final span's end. "Ran over" here means the stand-in
causal_lm recorded the position, not a verified KV write (that is what the GPU check is for).

Deliberately uses no symbol the fix adds (no tail_run_spans, no tail_by_span import): only
chunk_spans, ChunkSpan, DSV4_WINDOW and DeepseekV4LayerMajorAdapter, all present at 43d7813a41. This
lets the same file run unchanged against that commit, where it fails on the +8 and +127 cases
because the old finish_pass calls forward_late_tail on the final span only.

The handle gives every span that MIGHT need to run a tail (the final span, plus the one before it
when it exists) real tail metadata; which of those finish_pass actually calls forward_late_tail on
is what each commit's own code decides, and is what this test observes through the stand-in.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.models.deepseek_v4_layer_major import (
    DSV4_WINDOW,
    DeepseekV4LayerMajorAdapter,
    chunk_spans,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 4096


class _FakeTail:
    """One span's own last min(window, rows) positions -- what LateLayerTail selects in production."""

    def __init__(self, span, window):
        rows = span.end - span.start
        k = min(window, rows)
        self.positions = list(range(span.end - k, span.end))


class _RecordingBackend:
    def __init__(self):
        self.tail_forward_metadata = None
        self.calls = []  # (metadata, tail_metadata), in install order

    def install_forward_metadata(self, metadata, *, tail_metadata=None):
        self.tail_forward_metadata = tail_metadata
        self.calls.append((metadata, tail_metadata))


class _RecordingCausalLM:
    """Records the (layer, positions) the late layers ran on: one call per span, all late layers
    at once, at the positions the currently-installed tail names."""

    def __init__(self, backend):
        self._backend = backend
        self.calls = []  # (span_index, positions)

    def forward_late_tail(self, *, forward_batch, hidden_states, prev_pre, hash_ids):
        positions = list(self._backend.tail_forward_metadata.late_layer_tail.positions)
        self.calls.append((forward_batch.span_index, positions))
        return SimpleNamespace(output_for=forward_batch.span_index)


class _FakeStore:
    def read_into(self, name, offset, tensor, stream=None):
        pass

    def unpark(self, index, device):
        return f"META{index}"


def _run_finish_pass(*, seq_len, prefix_len=0, chunk=CHUNK, window=DSV4_WINDOW):
    spans = chunk_spans(prefix_len=prefix_len, seq_len=seq_len, chunk=chunk)
    # Every span that could possibly be needed (final, plus the one before it) gets real tail
    # metadata; a single-span suffix has no "one before it".
    tail_candidates = spans[-2:] if len(spans) >= 2 else spans[-1:]
    tail_meta_by_index = {s.index: SimpleNamespace(late_layer_tail=_FakeTail(s, window)) for s in tail_candidates}
    forward_batches = [SimpleNamespace(span_index=s.index) for s in spans]
    backend = _RecordingBackend()
    causal_lm = _RecordingCausalLM(backend)
    handle = SimpleNamespace(
        spans=spans,
        forward_batches=forward_batches,
        hash_ids=[None for _ in spans],
        schedule_batch=SimpleNamespace(prefix_lens=[prefix_len]),
        # Old field name (43d7813a41): the final span's own tail.
        final_tail_metadata=tail_meta_by_index[spans[-1].index],
        # New field name (the fix): every span a pass might run a tail over.
        tail_by_span=tail_meta_by_index,
        finalized=False,
    )
    finalize_calls = []
    adapter_self = SimpleNamespace(
        runner=SimpleNamespace(device="cpu"),
        model=SimpleNamespace(hc_mult=1, hidden_size=4),
        backend=backend,
        causal_lm=causal_lm,
        _finalize_ring=lambda h: finalize_calls.append(h),
    )
    # Bind unconditionally: unused (AttributeError never triggered) unless the code under test
    # calls self._run_late_layers, which only the fix's finish_pass does.
    adapter_self._run_late_layers = lambda h, s, span: DeepseekV4LayerMajorAdapter._run_late_layers(
        adapter_self, h, s, span
    )
    output = DeepseekV4LayerMajorAdapter.finish_pass(adapter_self, handle, _FakeStore())
    covered = set()
    for _, positions in causal_lm.calls:
        covered.update(positions)
    required = set(range(max(prefix_len, spans[-1].end - window), spans[-1].end))
    return SimpleNamespace(
        calls=causal_lm.calls, output=output, covered=covered, required=required,
        finalize_calls=finalize_calls, backend_calls=backend.calls, tail_meta_by_index=tail_meta_by_index,
    )


class TestLateLayerWriteCoverage(unittest.TestCase):
    def test_exact_multiple_of_chunk_is_fully_covered(self):
        r = _run_finish_pass(seq_len=CHUNK * 2)
        self.assertEqual(r.covered, r.required)
        self.assertEqual(len(r.finalize_calls), 1)

    def test_final_chunk_of_8_rows_is_fully_covered(self):
        r = _run_finish_pass(seq_len=CHUNK * 2 + 8)
        missing = r.required - r.covered
        self.assertEqual(missing, set(), f"late-layer SWA slots never written for positions {sorted(missing)}")

    def test_final_chunk_of_127_rows_is_fully_covered(self):
        r = _run_finish_pass(seq_len=CHUNK * 2 + 127)
        missing = r.required - r.covered
        self.assertEqual(missing, set(), f"late-layer SWA slots never written for positions {sorted(missing)}")

    def test_final_chunk_of_exactly_the_window_is_fully_covered(self):
        r = _run_finish_pass(seq_len=CHUNK * 2 + DSV4_WINDOW)
        self.assertEqual(r.covered, r.required)

    def test_call_order_and_metadata_pairing_for_a_short_final_chunk(self):
        # Pins existing (already-fixed) behavior; not a RED case like the two above.
        r = _run_finish_pass(seq_len=CHUNK * 2 + 8)
        self.assertEqual(r.calls, [(1, list(range(8064, 8192))), (2, list(range(8192, 8200)))])
        self.assertEqual(
            r.backend_calls,
            [("META1", r.tail_meta_by_index[1]), ("META2", r.tail_meta_by_index[2])],
        )

    def test_the_requests_own_output_still_comes_from_the_true_final_span(self):
        spans = chunk_spans(prefix_len=0, seq_len=CHUNK * 2 + 8, chunk=CHUNK)
        r = _run_finish_pass(seq_len=CHUNK * 2 + 8)
        self.assertEqual(r.output.output_for, spans[-1].index)


if __name__ == "__main__":
    unittest.main()
