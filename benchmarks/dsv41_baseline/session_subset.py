"""The DSV4.1 baseline corpus: real text, truncated to a fixed short shape, served over HTTP.

The corpus is the real one behind the recorded phase 3a/3b arms
(`divix01:/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`, produced by
`build_sessions.py`, generated 2026-09-15) — NOT the Qwen3.8 "prefetch-shadow" pinned
corpus. `analysis/dsv41-phase3a/wc-step7.sh` is
the exact command that produced `corpus-cold.json`: `trace_corpus.py --n 8 --skip 0
--prompt-tokens 256 --new-tokens 128`. `synthetic_corpus.py` reuses that recipe — the
first-turn text of each of these 8 sessions, truncated to 256 tokens — but re-encodes
it as a one-turn chat session so it can be driven over `/v1/chat/completions` like the
Qwen campaign, instead of fed as raw token ids to an offline Engine.

The serving context length comes from `arm_env.CONTEXT_LENGTH`, currently 32,768 to
match production. The corpus still uses short first-turn prompts and 128 generated
tokens. Historical 4,096-token results are not directly comparable: the longer
context changes memory allocation, and the current launch also enables prefix caching.

`corpus-c.json` (the recorded 2.781 tok/s baseline, DSV41_REFERENCE.md section 17.6)
used only the first 4 of these 8 sessions. `N_SESSIONS=8` keeps those 4 first and
unchanged, preserving corpus overlap, but the new launch settings require a fresh
throughput baseline. The extra 4 sessions buy statistical power for the paired sign
test (8 sessions of a clean sweep gives p ~= 0.0039 vs 4 sessions' p = 0.0625,
short of conventional significance).
"""

from __future__ import annotations

import hashlib
import json

from arm_env import CONTEXT_LENGTH

CORPUS_PATH = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
CORPUS_SHA256 = "249e8a73a32b69aff563471dbae2f4f3a2a9beaa1a3ae5cb03b4c2c549c16c72"

N_SESSIONS = 8
SKIP = 0
PROMPT_TOKENS = 256
NEW_TOKENS = 128

# The 8 session_ids at (skip=0, n=8) in this corpus. The first 4 are exactly
# `corpus-c.json`'s sessions.
EXPECTED_SESSION_IDS = (
    "cfq-train-Single_CDW/2015/page_35.pdf-2",
    "cfq-train-Single_ETR/2004/page_261.pdf-1",
    "cfq-train-Single_TSCO/2018/page_31.pdf-1",
    "cfq-train-Double_BKR/2017/page_47.pdf",
    "cfq-val-Single_K/2013/page_62.pdf-1",
    "cfq-train-Single_DISCA/2016/page_11.pdf-1",
    "cfq-train-Single_WRK/2019/page_49.pdf-1",
    "cfq-train-Single_VLO/2012/page_27.pdf-2",
)
BASELINE_4_SESSION_IDS = EXPECTED_SESSION_IDS[:4]

# Session 8 (the 9th row) of the same real corpus: a real, un-invented session that is
# not one of the 8 timed ones, used only to discard the first request's one-time cost
# and to bring the idle card's SM clock up before timing starts (see clock_ramp.py).
WARMUP_SESSION_INDEX = 8
WARMUP_SESSION_ID = "fb-financebench_id_04209"


class CorpusChecksumError(RuntimeError):
    pass


def verify_corpus_checksum(path: str = CORPUS_PATH, *, expected: str = CORPUS_SHA256) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    got = digest.hexdigest()
    if got != expected:
        raise CorpusChecksumError(
            f"corpus checksum mismatch for {path}: got {got}, expected {expected}"
        )
    return got


def load_raw_sessions(path: str = CORPUS_PATH, *, n: int, skip: int = 0) -> list[dict]:
    """Load `n` raw corpus sessions starting at `skip`, in file order."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if i >= skip + n:
                break
            out.append(json.loads(line))
    return out


def load_expected_sessions(path: str = CORPUS_PATH, *, n: int = N_SESSIONS, skip: int = SKIP) -> list[str]:
    """The session_ids these (skip, n) would draw, in file order."""
    return [s["session_id"] for s in load_raw_sessions(path, n=n, skip=skip)]
