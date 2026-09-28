from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.models.deepseek_v4_layer_major import (
    ChunkSpan,
    DeepseekV4LayerMajorAdapter,
)


class TestInstallForwardMetadata(unittest.TestCase):
    def _backend(self, window=None):
        b = object.__new__(DeepseekV4AttnBackend)
        b.encoder_replay = True
        b.forward_metadata = None
        b.tail_forward_metadata = "stale"
        b.token_to_kv_pool = SimpleNamespace(request_window=window)
        return b

    def test_installs_and_clears_per_forward_state(self):
        b = self._backend()
        meta = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout=None))
        b.install_forward_metadata(meta)
        self.assertIs(b.forward_metadata, meta)
        self.assertIsNone(b.tail_forward_metadata)
        self.assertFalse(b.encoder_replay)

    def test_activates_request_window_when_present(self):
        activated = []
        window = SimpleNamespace(activate=activated.append)
        b = self._backend(window=window)
        meta = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout="L"))
        b.install_forward_metadata(meta, tail_metadata="T")
        self.assertEqual((activated, b.tail_forward_metadata), (["L"], "T"))


class _RecordingBackend:
    """Records each install_forward_metadata call; that is all run_layer needs from it here."""

    def __init__(self):
        self.calls = []
        self.forward_metadata = None
        self.tail_forward_metadata = None

    def install_forward_metadata(self, metadata, *, tail_metadata=None):
        self.calls.append((metadata, tail_metadata))
        self.forward_metadata = metadata
        self.tail_forward_metadata = tail_metadata


class _FakeLayer:
    engram = None

    def forward_hc_pre_from_prev(self, *, hidden_states, prev_pre, **_unused):
        rows, hc_mult = hidden_states.shape[0], hidden_states.shape[1]
        return hidden_states, torch.zeros(rows, hc_mult)


class _FakeStore:
    def __init__(self, meta_by_index):
        self._meta_by_index = meta_by_index
        self.parked = {}

    def read_into(self, name, offset, tensor, stream=None):
        pass

    def write_from(self, name, offset, tensor, stream=None):
        pass

    def unpark(self, index, device):
        return self._meta_by_index[index]

    def park(self, index, value):
        self.parked[index] = value


class TestRunLayerTailMetadata(unittest.TestCase):
    """run_layer must hand the backend a tail only on the pass's final chunk (fix
    round 1, item 6): earlier chunks have no tail and rely on
    layer_major_skip_candidates instead."""

    def _run(self):
        spans = [ChunkSpan(index=0, start=0, end=4), ChunkSpan(index=1, start=4, end=7)]
        handle = SimpleNamespace(
            spans=spans,
            forward_batches=[
                SimpleNamespace(positions=None, input_ids=None) for _ in spans
            ],
            schedule_batch=SimpleNamespace(prefix_lens=[0]),
            tail_by_span={1: "TAIL"},
        )
        store = _FakeStore({0: "META0", 1: "META1"})
        backend = _RecordingBackend()
        model = SimpleNamespace(
            hc_mult=1, hidden_size=4, start_layer=0, layers=[_FakeLayer()]
        )
        adapter_self = SimpleNamespace(
            runner=SimpleNamespace(device="cpu"), model=model, backend=backend
        )
        recorder = SimpleNamespace(
            with_current_layer=lambda layer_id: contextlib.nullcontext()
        )
        with mock.patch(
            "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            for chunk in range(len(spans)):
                DeepseekV4LayerMajorAdapter.run_layer(
                    adapter_self, handle, layer_id=0, chunk=chunk, store=store
                )
        return backend

    def test_only_the_final_chunk_installs_the_tail(self):
        backend = self._run()
        self.assertEqual(backend.calls, [("META0", None), ("META1", "TAIL")])


class TestRunLayerPenultimateTailMetadata(unittest.TestCase):
    """C1: when the final span is short, the penultimate span is also a needed tail. run_layer
    must hand the candidate-source layer THAT span's own tail metadata (distinct from the final
    span's), not None and not the final one's -- this is what lets it publish tail-only candidate
    masks scoped to exactly the penultimate span's rows (the general publish/consume mechanism
    itself is pinned generically in test_dsv4_candidate_indexer.py's tail-publish tests)."""

    def test_each_needed_span_gets_its_own_tail_not_the_final_ones(self):
        # Mirrors a real C1 case: chunk 0 (not needed), chunk 1 (penultimate, needed),
        # chunk 2 (the true final span, needed) -- two distinct tails installed.
        spans = [
            ChunkSpan(index=0, start=0, end=4096),
            ChunkSpan(index=1, start=4096, end=8192),
            ChunkSpan(index=2, start=8192, end=8200),
        ]
        handle = SimpleNamespace(
            spans=spans,
            forward_batches=[SimpleNamespace(positions=None, input_ids=None) for _ in spans],
            schedule_batch=SimpleNamespace(prefix_lens=[0]),
            tail_by_span={1: "TAIL_PENULTIMATE", 2: "TAIL_FINAL"},
        )
        store = _FakeStore({0: "META0", 1: "META1", 2: "META2"})
        backend = _RecordingBackend()
        model = SimpleNamespace(hc_mult=1, hidden_size=4, start_layer=0, layers=[_FakeLayer()])
        adapter_self = SimpleNamespace(runner=SimpleNamespace(device="cpu"), model=model, backend=backend)
        recorder = SimpleNamespace(with_current_layer=lambda layer_id: contextlib.nullcontext())
        with mock.patch(
            "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            for chunk in range(len(spans)):
                DeepseekV4LayerMajorAdapter.run_layer(adapter_self, handle, layer_id=0, chunk=chunk, store=store)
        self.assertEqual(
            backend.calls,
            [("META0", None), ("META1", "TAIL_PENULTIMATE"), ("META2", "TAIL_FINAL")],
        )


if __name__ == "__main__":
    unittest.main()
