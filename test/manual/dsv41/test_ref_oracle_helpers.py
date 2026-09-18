"""CPU-only checks for `ref_oracle.reference_args` against the real reference `config.json`.

Needs the official DeepSeek-V4.1-Flash snapshot on disk (divix01); skips cleanly if it is
absent, and skips (rather than fails) if the reference's own `inference/model.py` cannot be
imported without a GPU.
"""

import importlib.util
import json
import os
import sys

import pytest

_SNAP = os.environ.get(
    "DSV41_SNAP",
    "/mnt/nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/"
    "snapshots/dba1be0a40aa45a94ad051997016db3960a90277",
)
_CONFIG_PATH = os.path.join(_SNAP, "inference", "config.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_CONFIG_PATH), reason=f"needs the official snapshot at {_SNAP}"
)

_ORACLE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "dsv41", "ref_oracle.py")
_spec = importlib.util.spec_from_file_location("ref_oracle", _ORACLE_PATH)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


def test_reference_args_truncates_to_three_layers():
    with open(_CONFIG_PATH) as f:
        ref_config = json.load(f)
    kwargs = oracle.reference_args(ref_config, n_layers=3, max_seq_len=4096)

    inference_dir = os.path.join(_SNAP, "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    try:
        import model as ref  # the reference's own inference/model.py
    except Exception as exc:  # needs CUDA at import time on this host -- skip, don't fail
        pytest.skip(f"reference model.py did not import without a GPU: {exc}")

    ref.ModelArgs(**kwargs)  # must construct without raising
    assert kwargs["compress_ratios"] == [0, 0, 2]
    assert kwargs["kv_source_layers"] == [2]
    assert kwargs["engram_layer_ids"] == [1]
    assert kwargs["candidate_source_layer"] == -1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
