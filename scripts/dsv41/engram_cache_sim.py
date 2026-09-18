"""Exact-LRU hit rates for a host-RAM Engram row cache over our benchmark corpus.

Rows are ids from upstream SGLang's EngramHasher (proven equal to DeepSeek's reference
by test/manual/dsv41/test_engram_parity.py). One cache serves both Engram layers; an
entry is a 264-byte row (256 B FP8 weights + 8 B E8M0 scales). LRU is exact: a reuse
at stack distance d hits iff d < capacity. Run on divix01 under taskset and a memory cap.
"""

import argparse
import json
import os
import types

import numba
import numpy as np
import torch

ROW_BYTES = 256 + 8
LAYER_KEY_SHIFT = 32  # row ids are < 2**29; the layer index goes above them


@numba.njit(cache=True)
def _fenwick_add(tree, index, delta):
    i = index + 1
    while i < tree.shape[0]:
        tree[i] += delta
        i += i & (-i)


@numba.njit(cache=True)
def _fenwick_prefix(tree, count):
    total, i = 0, count
    while i > 0:
        total += tree[i]
        i -= i & (-i)
    return total


@numba.njit(cache=True)
def _reuse_distances(dense):
    n = dense.shape[0]
    last = np.full(dense.max() + 1, -1, np.int64)
    tree = np.zeros(n + 1, np.int32)
    distances = np.empty(n, np.int64)
    prev_index = np.empty(n, np.int64)
    for i in range(n):
        key = dense[i]
        p = last[key]
        prev_index[i] = p
        if p < 0:
            distances[i] = -1
        else:
            distances[i] = _fenwick_prefix(tree, i) - _fenwick_prefix(tree, p + 1)
            _fenwick_add(tree, p, -1)
        _fenwick_add(tree, i, 1)
        last[key] = i
    return distances, prev_index


def reuse_distances(keys):
    _, dense = np.unique(keys, return_inverse=True)
    return _reuse_distances(dense.astype(np.int64).ravel())


def lru_hits(distances, capacity):
    return (distances >= 0) & (distances < capacity)


def session_repeat_hits(prev_index, session_start):
    return prev_index >= session_start


def _session_texts(sessions_path, results_path):
    completions = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            completions[(r["session_id"], r["turn"])] = (r.get("reasoning") or "") + (
                r.get("content") or ""
            )
    with open(sessions_path) as f:
        for line in f:
            s = json.loads(line)
            for turn, prompt in enumerate(s["turns"]):
                yield s["session_id"], prompt + completions.get((s["session_id"], turn), "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="/mnt/nvme2/nvfp4-work/benchmarks/full")
    parser.add_argument("--tokenizer-dir", default="/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw")
    parser.add_argument("--config", required=True, help="official config.json (text_config)")
    parser.add_argument("--max-tokens", type=int, default=1_500_000)
    parser.add_argument("--budgets-gb", type=float, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sglang.srt.layers.engram import EngramHasher, EngramLayout, compute_engram_hash_ids

    with open(args.config) as f:
        config = types.SimpleNamespace(**json.load(f)["text_config"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    # NOTE: the brief's config.engram_pad_id does not exist on the official
    # config.json's text_config; the key is engram_pad_token_id (see
    # EngramHasher.from_config at python/sglang/srt/layers/engram.py:~271).
    # EngramHasher.from_config itself is not usable standalone here because it
    # resolves its tokenizer from the live serving runtime (get_serving()/
    # get_model()), which does not exist outside a running server. So we keep
    # the brief's explicit constructor call and only fix the key name.
    hasher = EngramHasher(
        EngramLayout.from_config(config),
        tokenizer,
        config.engram_pad_token_id,
        config.engram_compressed_vocab_size,
    )
    n = hasher.max_ngram_size
    layer_offsets = (torch.arange(hasher.primes.shape[0]) << LAYER_KEY_SHIFT).view(1, -1, 1)

    session_ids, per_session = [], {}
    for sid, text in _session_texts(
        os.path.join(args.corpus_dir, "sessions.jsonl"),
        os.path.join(args.corpus_dir, "results.jsonl"),
    ):
        if sid not in per_session:
            session_ids.append(sid)
            per_session[sid] = []
        per_session[sid].extend(tokenizer(text, add_special_tokens=False)["input_ids"])

    key_chunks, start_chunks, total_tokens, access = [], [], 0, 0
    for sid in session_ids:
        seq = per_session[sid][: max(0, args.max_tokens - total_tokens)]
        if not seq:
            break
        t = torch.arange(len(seq)).unsqueeze(-1)
        shifts = torch.arange(n)
        tokens = torch.as_tensor(seq, dtype=torch.int64)[(t - shifts).clamp_min(0)]
        ids = compute_engram_hash_ids(
            tokens, t < shifts, hasher.pad_id, hasher.token_map,
            hasher.multipliers, hasher.primes, hasher.offsets,
        )
        keys = (ids + layer_offsets).reshape(-1).numpy()
        key_chunks.append(keys)
        start_chunks.append(np.full(keys.shape[0], access, np.int64))
        access += keys.shape[0]
        total_tokens += len(seq)

    keys = np.concatenate(key_chunks)
    session_start = np.concatenate(start_chunks)
    per_token = keys.shape[0] // total_tokens
    distances, prev_index = reuse_distances(keys)
    unique_rows = int((prev_index < 0).sum())

    budgets = []
    for gb in args.budgets_gb:
        capacity = int(gb * 1e9 // ROW_BYTES)
        misses = ~lru_hits(distances, capacity)
        per_token_misses = misses.reshape(total_tokens, per_token).sum(axis=1)
        budgets.append(
            {
                "gb": gb,
                "rows": capacity,
                "hit_rate": float(1 - misses.mean()),
                "misses_per_token_mean": float(per_token_misses.mean()),
                "misses_per_token_p99": float(np.percentile(per_token_misses, 99)),
            }
        )
    result = {
        "tokens": total_tokens,
        "sessions": len(key_chunks),
        "accesses": int(keys.shape[0]),
        "accesses_per_token": per_token,
        "unique_rows": unique_rows,
        "unique_set_gb": unique_rows * ROW_BYTES / 1e9,
        "hit_ceiling": 1 - unique_rows / keys.shape[0],
        "session_cold_hit_rate": float(session_repeat_hits(prev_index, session_start).mean()),
        "budgets": budgets,
        "notes": "prompts plus the 530 recorded completions; no chat template",
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
