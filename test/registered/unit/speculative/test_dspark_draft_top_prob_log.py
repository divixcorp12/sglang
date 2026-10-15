"""The per-verify draft top-probability log a trace run joins to its accept log (CPU)."""

import json

import torch

from sglang.srt.environ import envs
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components import dspark_draft_top_prob_log as top_prob_log
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

VOCAB, HIDDEN, GAMMA = 11, 4, 3


class _Draft:
    def __init__(self):
        torch.manual_seed(0)
        self.head = torch.randn(VOCAB, HIDDEN)
        self.markov_head = VanillaMarkov(vocab_size=VOCAB, markov_rank=2)

    def compute_base_logits(self, x):
        return x @ self.head.T, None


def _step_logits(draft, hidden, anchor, tokens):
    base = draft.compute_base_logits(hidden)[0].view(1, GAMMA, VOCAB)
    prev = [anchor] + tokens[:-1]
    bias = draft.markov_head.markov_w2(draft.markov_head.markov_w1(torch.tensor(prev)))
    return base[0] + bias


@torch.no_grad()
def test_probs_come_from_the_markov_step_logits_with_the_drafted_tokens_fed_back():
    draft = _Draft()
    hidden = torch.randn(GAMMA, HIDDEN)
    anchor = 5
    drafted = [3, 7, 1]
    probs = top_prob_log.draft_top_probs(
        draft_model=draft, raw_hidden=hidden, draft_hidden=hidden.view(1, GAMMA, HIDDEN),
        anchor_tokens=torch.tensor([anchor]), draft_tokens=torch.tensor([drafted]))
    expect = torch.softmax(_step_logits(draft, hidden, anchor, drafted), dim=-1).sort(dim=-1, descending=True)
    assert torch.allclose(probs["top_prob"][0], expect.values[:, 0])
    assert torch.allclose(probs["second_prob"][0], expect.values[:, 1])
    assert probs["argmax_agree"][0].tolist() == [int(i) == t for i, t in zip(expect.indices[:, 0], drafted)]


def test_each_verify_of_a_request_gets_its_ordinal(tmp_path):
    log = top_prob_log.DraftTopProbLog(str(tmp_path / "top"))

    def host(top, correct):
        return {"top_prob": torch.tensor(top), "second_prob": torch.tensor(top) / 2,
                "argmax_agree": torch.tensor(top) > 0.5, "num_correct_drafts": torch.tensor(correct)}

    log.write_rows(rids=["r1", "r2"], host=host([[0.9, 0.4], [0.75, 0.6]], [2, 0]))
    log.write_rows(rids=["r1"], host=host([[0.25, 0.125]], [1]))
    log.close()
    (path,) = tmp_path.glob("top.*.jsonl")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(e["rid"], e["k"], e["num_correct_drafts"]) for e in lines] == [("r1", 0, 2), ("r2", 0, 0), ("r1", 1, 1)]
    assert lines[0]["top_prob"] == [0.9, 0.4] and lines[0]["argmax_agree"] == [True, False]
    assert lines[2]["second_prob"] == [0.125, 0.0625]


def test_the_log_exists_only_when_its_path_is_set(tmp_path):
    top_prob_log.reset_for_test()
    with envs.SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH.override(""):
        assert top_prob_log.get_draft_top_prob_log() is None
    top_prob_log.reset_for_test()
    with envs.SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH.override(str(tmp_path / "top")):
        log = top_prob_log.get_draft_top_prob_log()
        assert log is not None and log is top_prob_log.get_draft_top_prob_log()
    top_prob_log.reset_for_test()
