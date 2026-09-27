import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend


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


if __name__ == "__main__":
    unittest.main()
