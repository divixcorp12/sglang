"""Engram lookback + history commit after partial accept in spec verify (CPU).

Oracle: stateless. For a committed token sequence and a verify block [anchor, d1..dk]
the n-gram inputs of row j are the tokens at absolute index len+j-s (s < n), PAD
(blocked) below index 0 -- exactly what plain one-token decode sees. The system under
test is EngramHasher's real forward (extend / target-verify / decode) and
commit_after_verify, driven through steps with accept counts 0..k.
commit_lens follows the DSpark contract: accepted drafts + bonus (a + 1); the bonus is
the next block's anchor, so the history takes anchor + a drafts.
"""

import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.engram import (
    EngramHasher,
    compute_engram_hash_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

N, L, H, VOCAB, CVOCAB = 4, 2, 2, 50, 20
K = 3  # drafts per block; verify block = K + 1 rows
BLOCK = K + 1
SLOTS = 6
PAD = 2


class _Mode:
    def __init__(self, kind):
        self.kind = kind

    def is_decode(self):
        return self.kind == "decode"

    def is_target_verify(self):
        return self.kind == "verify"

    def is_extend(self):
        return self.kind == "extend"


def _hasher(seed=0):
    g = torch.Generator().manual_seed(seed)
    h = object.__new__(EngramHasher)
    nn.Module.__init__(h)
    h.max_ngram_size = N
    h.pad_id = int(PAD)
    h.register_buffer("token_map", torch.randint(0, CVOCAB, (VOCAB,), generator=g))
    h.register_buffer(
        "multipliers", torch.randint(1, 10**6, (L, N), generator=g) * 2 + 1
    )
    h.register_buffer("primes", torch.tensor([[[101, 103]] * (N - 1)] * L))
    h.register_buffer(
        "offsets", torch.arange(L * (N - 1) * H).reshape(L, (N - 1) * H) * 1000
    )
    h.image_token_id = None
    h.history = None
    h.pad_row = 0
    h.init_history(SLOTS, "cpu")
    return h


def _fb(kind, ids, positions, slots, **kw):
    return SimpleNamespace(
        forward_mode=_Mode(kind),
        req_pool_indices=torch.tensor(slots),
        positions=torch.tensor(positions),
        out_cache_loc=None,
        engram_history=None,
        **kw,
    )


def _oracle_tokens(full, first_pos, n_rows):
    """Predecessor table [rows, N] and blocked mask for rows at abs index first_pos+j."""
    toks, blk = [], []
    for j in range(n_rows):
        i = first_pos + j
        toks.append([full[i - s] if i - s >= 0 else 0 for s in range(N)])
        blk.append([i - s < 0 for s in range(N)])
    return torch.tensor(toks), torch.tensor(blk)


def _oracle_hash(h, toks, blk):
    return compute_engram_hash_ids(
        toks, blk, h.pad_id, h.token_map, h.multipliers, h.primes, h.offsets
    )


def _oracle_history(committed):
    """Last N-1 committed tokens, oldest first; slots before the start are unused."""
    return [committed[i] if i >= 0 else None for i in range(len(committed) - (N - 1), len(committed))]


def _extend(h, prompts, slots):
    ids = torch.tensor([t for p in prompts for t in p])
    pos = [i for p in prompts for i in range(len(p))]
    lens = torch.tensor([len(p) for p in prompts])
    starts = torch.cumsum(lens, 0) - lens
    fb = _fb("extend", ids, pos, slots, extend_seq_lens=lens, extend_start_loc=starts)
    h(ids, fb)


def _check_history(h, slots, committed):
    for slot, seq in zip(slots, committed):
        want = _oracle_history(seq)
        got = h.history[slot].tolist()
        for c, (w, g) in enumerate(zip(want, got)):
            assert w is None or w == g, (slot, c, want, got)


def _run(prompt_lens, steps, seed, accepts=None):
    rng = random.Random(seed)
    h = _hasher(seed)
    slots = [3, 1, 4, 0][: len(prompt_lens)]
    committed = [[rng.randrange(VOCAB) for _ in range(p)] for p in prompt_lens]
    _extend(h, committed, slots)
    _check_history(h, slots, committed)
    pending = [rng.randrange(VOCAB) for _ in slots]  # first sampled token = anchor
    for step in range(steps):
        block_ids = [[a] + [rng.randrange(VOCAB) for _ in range(K)] for a in pending]
        acc = [
            (accepts[step][r] if accepts else rng.randint(0, K)) for r in range(len(slots))
        ]
        ids = torch.tensor([t for b in block_ids for t in b])
        pos = [len(c) + j for c in committed for j in range(BLOCK)]
        fb = _fb(
            "verify", ids, pos, slots, spec_info=SimpleNamespace(draft_token_num=BLOCK)
        )
        got = h(ids, fb)
        for r, c in enumerate(committed):
            toks, blk = _oracle_tokens(c + block_ids[r], len(c), BLOCK)
            want = _oracle_hash(h, toks, blk)
            rows = slice(r * BLOCK, (r + 1) * BLOCK)
            for j in range(BLOCK):
                assert torch.equal(got[rows][j], want[j]), (
                    f"step {step} req {r} offset {j} accept {acc[r]}: verify hash "
                    f"differs from stateless oracle (prefix len {len(c)})"
                )
        h.commit_after_verify(
            torch.tensor(block_ids),
            torch.tensor(slots),
            torch.tensor([a + 1 for a in acc], dtype=torch.int32),
        )
        for r, c in enumerate(committed):
            c.extend(block_ids[r][: acc[r] + 1])
        pending = [rng.randrange(VOCAB) for _ in slots]  # bonus
        # the bonus must be the next anchor; the oracle keeps it outside `committed`
        _check_history(h, slots, committed)
    return h, slots, committed, pending


@pytest.mark.parametrize("seed", range(6))
def test_random_accepts_match_stateless_oracle(seed):
    _run([1, 2, 3, 6], steps=8, seed=seed)


@pytest.mark.parametrize("a", range(K + 1))
def test_each_accept_count_over_consecutive_steps(a):
    _run([1, 2, 5], steps=5, seed=100 + a, accepts=[[a] * 3] * 5)


def test_mixed_accepts_short_history():
    # prompts shorter than the lookback, then reject / partial / full in turn
    _run([1, 1], steps=4, seed=7, accepts=[[0, K], [K, 0], [1, 2], [2, 1]])


def test_oracle_matches_plain_decode():
    """The oracle itself: one-token decode of the same sequence hashes identically."""
    rng = random.Random(3)
    h = _hasher(3)
    seq = [rng.randrange(VOCAB) for _ in range(9)]
    _extend(h, [seq[:1]], [2])
    for i in range(1, len(seq)):
        fb = _fb("decode", None, [i], [2])
        fb.out_cache_loc = torch.tensor([7])
        got = h(torch.tensor([seq[i]]), fb)
        toks, blk = _oracle_tokens(seq, i, 1)
        assert torch.equal(got[0], _oracle_hash(h, toks, blk)[0]), i
