"""Upstream SGLang's Engram hashing equals DeepSeek's reference inference/engram.py."""

import hashlib
import os
import types

import numpy as np
import pytest
import torch

TOKENIZER_DIR = os.environ.get(
    "DSV41_TOKENIZER_DIR", "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
)
GOLDEN = os.path.join(os.path.dirname(__file__), "dsv41_engram_golden.npz")
ENGRAM_CONFIG = dict(
    engram_layer_ids=[1, 14],
    engram_vocab_size=16_000_000,
    engram_num_embeddings=[384006168, 384016682],
    engram_max_ngram_size=4,
    engram_pad_id=2,
    engram_compressed_vocab_size=99092,
    engram_n_heads=8,
    engram_head_dim=256,
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(TOKENIZER_DIR, "tokenizer.json")),
    reason="needs the DeepSeek-V4.1 tokenizer",
)


def prefill_window(seq, n):
    """Predecessor table [T, n] and look-back mask for one sequence starting at 0."""
    t = torch.arange(len(seq)).unsqueeze(-1)
    shifts = torch.arange(n)
    tokens = torch.as_tensor(seq, dtype=torch.int64)[(t - shifts).clamp_min(0)]
    return tokens, t < shifts


@pytest.fixture(scope="module")
def hasher():
    from transformers import AutoTokenizer

    from sglang.srt.layers.engram import EngramHasher, EngramLayout

    config = types.SimpleNamespace(**ENGRAM_CONFIG)
    layout = EngramLayout.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return EngramHasher(
        layout, tokenizer, config.engram_pad_id, config.engram_compressed_vocab_size
    )


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


def test_token_map_matches_reference(hasher, golden):
    digest = hashlib.sha256(hasher.token_map.numpy().astype(np.int64).tobytes()).digest()
    assert np.frombuffer(digest, dtype=np.uint8).tolist() == golden["token_map_sha256"].tolist()


@pytest.mark.parametrize("name", ["text", "random", "repeat"])
def test_hash_ids_match_reference(hasher, golden, name):
    from sglang.srt.layers.engram import compute_engram_hash_ids

    seq = golden[f"tokens_{name}"].tolist()
    tokens, blocked = prefill_window(seq, hasher.max_ngram_size)
    ids = compute_engram_hash_ids(
        tokens, blocked, hasher.pad_id, hasher.token_map,
        hasher.multipliers, hasher.primes, hasher.offsets,
    )
    assert torch.equal(ids, torch.from_numpy(golden[f"ids_{name}"]))


def test_ids_stay_inside_each_table(hasher, golden):
    from sglang.srt.layers.engram import compute_engram_hash_ids

    tokens, blocked = prefill_window(golden["tokens_random"].tolist(), hasher.max_ngram_size)
    ids = compute_engram_hash_ids(
        tokens, blocked, hasher.pad_id, hasher.token_map,
        hasher.multipliers, hasher.primes, hasher.offsets,
    )
    limits = torch.tensor(ENGRAM_CONFIG["engram_num_embeddings"]).view(1, 2, 1)
    assert bool((ids >= 0).all()) and bool((ids < limits).all())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
