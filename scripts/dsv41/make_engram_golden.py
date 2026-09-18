"""Golden Engram hash ids from DeepSeek's reference inference/engram.py.

Run on divix01, where the official snapshot, sympy and the DeepSeek-V4.1 tokenizer
live. test/manual/dsv41/test_engram_parity.py checks upstream SGLang's EngramHasher
against this output, bit for bit.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import types

import numpy as np
import torch
from transformers import AutoTokenizer

TEXT = (
    "Answer using the financial report excerpt below. Net income was $ 403.1 million "
    "in 2015 , up from $ 244.9 million ; THE Net Income , the net income."
)


def _load_reference(snapshot):
    path = os.path.join(snapshot, "inference", "engram.py")
    spec = importlib.util.spec_from_file_location("dsv41_reference_engram", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.snapshot, "config.json")) as f:
        text_config = json.load(f)["text_config"]
    engram = {k: v for k, v in text_config.items() if k.startswith("engram_")}
    # The reference's NgramHashState.__init__ reads `args.engram_pad_id`, but the
    # checkpoint's config.json names the same field `engram_pad_token_id`. This is
    # a call-site field-name mismatch, not a hashing difference: map it here so the
    # reference gets the pad id under the name it expects.
    engram["engram_pad_id"] = engram["engram_pad_token_id"]
    reference = _load_reference(args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    rng = np.random.default_rng(0)
    sequences = {
        "text": tokenizer(TEXT, add_special_tokens=False)["input_ids"],
        "random": rng.integers(0, len(tokenizer), size=64).tolist(),
        "repeat": ([5, 5, 5, 6] * 8),
    }
    token_map, vocab = reference.build_compressed_token_map(tokenizer)
    assert vocab == engram["engram_compressed_vocab_size"], vocab

    arrays = {
        "token_map_sha256": np.frombuffer(
            hashlib.sha256(np.asarray(token_map, dtype=np.int64).tobytes()).digest(),
            dtype=np.uint8,
        ),
    }
    for name, seq in sequences.items():
        ref_args = types.SimpleNamespace(max_batch_size=1, max_seq_len=len(seq), **engram)
        layout = reference.EngramLayout.from_args(ref_args)
        state = reference.NgramHashState(ref_args, layout, tokenizer)
        ids = state(torch.tensor([seq], dtype=torch.int64), start_pos=0)[0]
        arrays[f"tokens_{name}"] = np.asarray(seq, dtype=np.int64)
        arrays[f"ids_{name}"] = ids.numpy()
    np.savez(args.out, **arrays)
    print({k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
