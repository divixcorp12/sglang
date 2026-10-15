"""A trace run's per-verify draft confidence log (``SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH``).

One JSON line per request per DSpark verify step, in step order: ``rid``, ``k`` (its verifies before this one, the
ordinal verify_accept_log.py also writes, so the two join on ``(rid, k)``), and per draft position ``top_prob`` and
``second_prob`` (the draft's softmax top-1 and top-2 probability over its Markov-corrected step logits) and
``argmax_agree`` (whether that top-1 is the drafted token), then ``num_correct_drafts``.

The folded greedy proposal keeps no logits, so they are recomputed eagerly from the draft's hidden states with the
drafted tokens teacher-forced: one extra LM-head pass per verify. Diagnostics only: unset, nothing is opened,
computed or written.
"""

from __future__ import annotations

import atexit
import collections
import json
import os
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.kv_canary.runner.future_tensor import FutureTensors


def draft_top_probs(
    *,
    draft_model,
    raw_hidden: torch.Tensor,
    draft_hidden: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """[bs, gamma] top-1/top-2 probabilities of the draft's step logits, with the drafted tokens fed back as the
    Markov head's previous tokens (what the in-graph proposal fed itself)."""
    bs, gamma = draft_tokens.shape
    base_logits, _ = draft_model.compute_base_logits(raw_hidden)
    forced = draft_tokens.long()
    _, corrected = draft_model.markov_head.sample_block(
        base_logits.view(bs, gamma, -1),
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=lambda step_logits, step_idx: forced[:, step_idx],
        collect_corrected=True,
    )
    top = torch.softmax(corrected.float(), dim=-1).topk(2, dim=-1)
    return {
        "top_prob": top.values[..., 0],
        "second_prob": top.values[..., 1],
        "argmax_agree": top.indices[..., 0] == forced,
    }


class DraftTopProbLog:
    def __init__(self, prefix: str) -> None:
        self._file = open(f"{prefix}.{os.getpid()}.jsonl", "a", buffering=1)
        self._k: collections.Counter = collections.Counter()
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        self._pending: Optional[tuple[list[str], FutureTensors]] = None

    def record(
        self,
        *,
        rids: list[str],
        probs: dict[str, torch.Tensor],
        num_correct_drafts: torch.Tensor,
    ) -> None:
        """Stage this step's rows for an async copy; the previous step's rows are written first."""
        self._drain()
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=num_correct_drafts.device)
        self._pending = (
            list(rids),
            FutureTensors.device_to_host(
                {**probs, "num_correct_drafts": num_correct_drafts},
                d2h_stream=self._d2h_stream,
            ),
        )

    def write_rows(self, *, rids: list[str], host: dict[str, torch.Tensor]) -> None:
        top = host["top_prob"].tolist()
        second = host["second_prob"].tolist()
        agree = host["argmax_agree"].tolist()
        correct = host["num_correct_drafts"].tolist()
        for row, rid in enumerate(rids):
            k = self._k[rid]
            self._k[rid] = k + 1
            self._file.write(
                json.dumps(
                    {
                        "rid": rid,
                        "k": k,
                        "top_prob": [round(p, 5) for p in top[row]],
                        "second_prob": [round(p, 5) for p in second[row]],
                        "argmax_agree": [bool(a) for a in agree[row]],
                        "num_correct_drafts": int(correct[row]),
                    }
                )
                + "\n"
            )

    def _drain(self) -> None:
        if self._pending is None:
            return
        rids, future = self._pending
        self._pending = None
        self.write_rows(rids=rids, host=future.wait())

    def close(self) -> None:
        self._drain()
        self._file.close()


_log: Optional[DraftTopProbLog] = None
_resolved = False


def get_draft_top_prob_log() -> Optional[DraftTopProbLog]:
    """The process's log, opened on first use when the path is set; None otherwise."""
    global _log, _resolved
    if not _resolved:
        prefix = envs.SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH.get()
        _log = DraftTopProbLog(prefix) if prefix else None
        if _log is not None:
            atexit.register(_log.close)
        _resolved = True
    return _log


def reset_for_test() -> None:
    global _log, _resolved
    if _log is not None:
        atexit.unregister(_log.close)
        _log.close()
    _log, _resolved = None, False
