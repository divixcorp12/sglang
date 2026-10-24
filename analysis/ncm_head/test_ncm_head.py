"""Tiny CPU correctness fixtures, not acceptance/training experiments."""

import copy
import json
import math
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest
import torch

from .data import CaptureWriter, read_records
from .model import (
    Confidence,
    DenseProjector,
    Head,
    HeadConfig,
    Memory,
    active_nonvocab_count,
    capacity_width,
    fixed_future_targets,
    planning_loss,
    segments,
)
from .protocol import (
    paired_report,
    preflight,
    screen_budget,
    useful_gain,
    validate_groups,
)
from .sampling import (
    Admission,
    Mode,
    Sampling,
    expected_accepted,
    overlap,
    probabilities,
    propose,
)
from .train import Trainer, label_projection


def fixtures(arm="F", gamma=5, seed=7, width=8):
    torch.manual_seed(seed)
    v, d, r, k = 11, 12, width, 4
    emb = torch.randn(v, d)
    proj = DenseProjector(torch.randn(v, d), torch.randn(d), 1e-6)
    head = Head(
        HeadConfig(d, d, gamma, r, arm, 32),
        emb,
        proj,
        torch.randn(v, k),
        torch.randn(v, k),
    )
    h = torch.randn(gamma, d)
    history = torch.randn(19, d)
    return head, h, history


@pytest.mark.parametrize(
    "arm", ["C", "P", "PM", "PH", "F", "F-noaux", "F-flat", "F-flat-aux"]
)
def test_zero_init_recovers_stock(arm):
    stock, h, history = fixtures("B0")
    head, _, _ = fixtures(arm)
    s = stock.prepare(h, history, 2)
    p = head.prepare(h, history, 2)
    tokens = torch.tensor([3, 4, 5, 6, 1])
    torch.testing.assert_close(
        stock.teacher_logits(s, 2, tokens), head.teacher_logits(p, 2, tokens)
    )


def test_memory_chunking_startup_and_watermark():
    m = Memory(12, 8)
    x = torch.randn(15, 12)
    initial = m.empty("request-3")
    full = m.ingest(initial, x, start=0, generation="request-3")
    a = m.ingest(initial, x[:7], start=0, generation="request-3")
    b = m.ingest(a, x[7:], start=7, generation="request-3")
    torch.testing.assert_close(full.slots, b.slots)
    assert (
        initial.processed == 0
        and b.processed == 15
        and full.accumulator.dtype == torch.float32
    )
    first = m.ingest(initial, x[:1], start=0, generation="request-3")
    torch.testing.assert_close(first.slots, first.latest.expand(4, -1))
    for start, generation in [(0, "request-3"), (15, "request-4")]:
        with pytest.raises(ValueError, match="watermark"):
            m.ingest(b, x[:1], start=start, generation=generation)


@pytest.mark.parametrize("arm", ["F", "F-flat-aux"])
@pytest.mark.parametrize("gamma", [1, 5])
def test_centering_and_flat_aux_identity(arm, gamma):
    head, h, history = fixtures(arm, gamma)
    plan = head.prepare(h, history, 2)
    _ids, w = segments(gamma)
    torch.testing.assert_close(
        (w[:, None] * plan.residual).sum(0), torch.zeros(8), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(plan.position.mean(0), plan.coarse)
    targets = (torch.randn(8), torch.randn(len(w), 8))
    residual = targets[1] - (w[:, None] * targets[1]).sum(0)
    targets = (targets[0], residual)
    loss = planning_loss(plan, targets, gamma)
    expected = (
        w[:, None] * (plan.coarse + plan.residual - targets[0] - targets[1]).square()
    ).sum() / 8
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_eos_target_padding_and_truncation():
    p = label_projection(8, 12, 11)
    fs = torch.randn(2, 12)
    assert fixed_future_targets(fs, [2, 3], p, 5, 1) is None
    g, d = fixed_future_targets(fs, [2, 1], p, 5, 1)
    normalized = fs / (fs.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    torch.testing.assert_close(g, (normalized @ p.T).sum(0) / 5)
    _, w = segments(5)
    torch.testing.assert_close(
        (w[:, None] * d).sum(0), torch.zeros(8), atol=1e-6, rtol=0
    )
    with pytest.raises(ValueError, match="after EOS"):
        fixed_future_targets(torch.randn(3, 12), [2, 1, 3], p, 5, 1)
    assert not g.requires_grad
    torch.testing.assert_close(p.norm(dim=1), torch.ones(8))


def test_prefix_distinguishes_same_last_token():
    head, h, history = fixtures()
    plan = head.prepare(h, history, 2)
    initial = head.initial_state(plan)
    left = head.observe(plan, head.observe(plan, initial, 3, 0), 5, 1)
    right = head.observe(plan, head.observe(plan, initial, 4, 0), 5, 1)
    assert not torch.allclose(left, right)
    assert head.initial_state(plan).equal(initial)


def test_confidence_cannot_train_proposal():
    head, h, history = fixtures()
    plan = head.prepare(h, history, 2)
    state = head.initial_state(plan)
    _, v = head.step_logits(plan, state, 2, 0)
    conf = Confidence(8, 4, 3)
    out = conf(plan, state, v, head.position_embedding.weight[0], torch.zeros(3), 0)
    torch.nn.functional.binary_cross_entropy_with_logits(
        out, torch.tensor(0.6)
    ).backward()
    assert all(p.grad is None for p in head.parameters())
    assert all(p.grad is not None for p in conf.parameters())


def test_causal_stop_never_inspects_suffix_and_invalid_falls_forward():
    def forbidden():
        raise AssertionError("looked into future")

    policy = Admission(0.5)
    assert policy.visit(lambda: 0.6)
    assert not policy.visit(lambda: 0.4)
    assert not policy.visit(forbidden) and policy.mode == Mode.STOPPED
    policy = Admission(0.5)
    assert policy.visit(lambda: float("nan"))
    assert policy.visit(forbidden) and policy.mode == Mode.FULL_REMAINDER
    assert Admission(0.5, compatible=False).visit(forbidden)
    assert Admission(0.5).visit(lambda: 0.5)


def test_proposal_cut_happens_before_draw():
    head, h, history = fixtures()
    plan = head.prepare(h, history, 2)
    gen = torch.Generator().manual_seed(8)
    before = gen.get_state().clone()
    tokens, qs, mode = propose(
        head, plan, 2, Sampling(), gen, 1, threshold=0.5, confidence=lambda *a: 0.1
    )
    assert tokens == [] and qs == [] and mode == Mode.STOPPED
    assert gen.get_state().equal(before)


def test_sampled_survival_is_not_product_of_overlaps():
    p = [torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0])]
    q = [torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0])]
    assert expected_accepted([0, 0], p, q, 99) == 2.0
    assert float(overlap(p[0], q[0])) == pytest.approx(0.6)
    assert expected_accepted([0, 0], p, q, 0) == 1.0


def test_sampling_filters_and_gradients():
    x = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    q = probabilities(x, Sampling(top_p=0.8))
    assert q[2] == 0 and float(q.sum().detach()) == pytest.approx(1.0)
    q[0].backward()
    assert torch.isfinite(x.grad).all() and x.grad[2] == 0
    q = probabilities(torch.tensor([0.0, 0.0, 0.0]), Sampling(top_k=1))
    assert q.tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        probabilities(torch.tensor([float("nan"), 0.0]), Sampling())


def enumerate_round(biased=False):
    # Exact rational distribution of a full 3-token stream, completing after one
    # speculative round with target draws. EOS=2; gamma=2. No Monte Carlo.
    target = lambda prefix: (
        [Q(1, 2), Q(1, 3), Q(1, 6)]
        if not prefix or prefix[-1] == 0
        else [Q(1, 4), Q(1, 2), Q(1, 4)]
    )
    draft = lambda prefix: [Q(1, 3)] * 3 if not prefix else [Q(1, 2), Q(1, 4), Q(1, 4)]
    out = {}

    def finish(prefix, mass):
        if len(prefix) == 3 or prefix and prefix[-1] == 2:
            out[tuple(prefix)] = out.get(tuple(prefix), Q(0)) + mass
            return
        for t, p in enumerate(target(prefix)):
            finish(prefix + [t], mass * p)

    def verify_path(tokens, prob, i=0, accepted=None):
        accepted = [] if accepted is None else accepted
        if i == len(tokens):
            finish(accepted, prob)
            return
        p, q = target(accepted), draft(tokens[:i])
        y = tokens[i]
        a = min(Q(1), p[y] / q[y])
        if y == 2:
            finish(accepted + [y], prob * a)
        else:
            verify_path(tokens, prob * a, i + 1, accepted + [y])
        residue = [max(Q(0), u - v) for u, v in zip(p, q)]
        total = sum(residue)
        if total:
            for t, r in enumerate(residue):
                if r:
                    finish(accepted + [t], prob * (1 - a) * r / total)

    def candidates(prefix, mass):
        # Legitimate cutoff at position 2 depends only on already sampled position 1.
        if len(prefix) == 2 or prefix and (prefix[-1] == 2 or prefix[0] == 1):
            verify_path([] if biased and prefix[0] == 0 else prefix, mass)
            return
        for t, q in enumerate(draft(prefix)):
            candidates(prefix + [t], mass * q)

    candidates([], Q(1))
    result = out.copy()
    out.clear()
    finish([], Q(1))
    return result, out


def test_exact_enumeration_causal_cut_and_negative_control():
    actual, target = enumerate_round()
    assert actual == target and sum(actual.values()) == 1
    mutant, _ = enumerate_round(biased=True)
    assert mutant != target


def test_capacity_match_counts_only_active_proposal():
    full, _, _ = fixtures("F", width=128)
    control, _, _ = fixtures("C", width=128)
    target = active_nonvocab_count(full)
    width = capacity_width(target, active_nonvocab_count(control), 32, 128, "C")
    c = replace(control.config, flat_hidden=width)
    replacement = Head(
        c,
        control.embedding.weight,
        copy.deepcopy(control.projector),
        control.dep_embedding.weight,
        control.dep_output.weight,
    )
    assert abs(active_nonvocab_count(replacement) - target) / target < 0.05


def test_group_leak_and_preflight_fail_closed():
    rows = [
        {
            "id": "a",
            "group": "same",
            "split": "train",
            "domain": "code",
            "length_bin": "short",
            "source": "fixture",
            "text": "a",
        },
        {
            "id": "b",
            "group": "same",
            "split": "test",
            "domain": "code",
            "length_bin": "short",
            "source": "fixture",
            "text": "b",
        },
    ]
    with pytest.raises(ValueError, match="leakage"):
        validate_groups(rows)
    cfg = json.loads(Path(__file__).with_name("pilot.json").read_text())
    assert preflight(cfg)["ready"] is False
    assert screen_budget(0.5, 0.1)["required_prompt_groups"] == 200
    assert useful_gain(3, 0.02, 0.05) == pytest.approx(0.2210526316)


def test_crossed_prompt_pairing_and_missing_pairs():
    rows = [
        {
            "seed": s,
            "prompt_id": p,
            "arm": a,
            "stratum": "code-short",
            "accepted": 2.0 + (0.2 if a == "F" else 0),
        }
        for s in [1, 2, 3]
        for p in ["a", "b", "c"]
        for a in ["C", "F"]
    ]
    report = paired_report(rows, draws=100)
    assert report["mean_gain"] == pytest.approx(0.2)
    assert report["crossed_95"] == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError, match="incomplete"):
        paired_report(rows[:-1], draws=10)


def test_capture_roundtrip_and_training_split_guard(tmp_path):
    head, h, history = fixtures()
    writer = CaptureWriter(tmp_path / "capture", {"model": "synthetic-fixture"})
    writer.add(
        record_id="r1",
        group_id="g1",
        split="train",
        kind="reference",
        anchor_id=2,
        processed_ids=list(range(len(history))),
        history=history,
        hidden=h,
        tokens=[3, 1],
        future_features=torch.randn(2, 12),
        label_law={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "greedy": False},
    )
    writer.close()
    records = list(read_records(tmp_path / "capture", "train", "reference"))
    config = json.loads(Path(__file__).with_name("pilot.json").read_text())
    # Two tiny optimizer steps validate gradients/state update, not model efficacy.
    config["updates"] = {"warmstart": 2}
    config["effective_batch"] = 1
    trainer = Trainer(head, config, label_projection(8, 12, 17), 1)
    logs = trainer.run_stage(records, [], "warmstart")
    assert len(logs) == 2 and all(math.isfinite(x["loss"]) for x in logs)
    assert head.up.weight.abs().sum() > 0
    assert not head.embedding.weight.requires_grad
    forbidden = (dict(records[0][0], split="test"), records[0][1])
    with pytest.raises(ValueError, match="cannot access"):
        trainer.reference_loss(forbidden)


def test_memory_keeps_fp32_coefficients_after_dtype_conversion():
    m = Memory(12, 8).to(torch.bfloat16)
    assert m.alpha.dtype == torch.float32
    state = m.ingest(m.empty(), torch.randn(2, 12), start=0, generation="")
    assert (
        state.accumulator.dtype == torch.float32 and state.mass.dtype == torch.float32
    )


def test_native_verifier_reference_eos_and_correction():
    from .sampling import verify

    calls = []

    def target(prefix):
        calls.append(prefix.copy())
        return torch.tensor([0.0, 1.0, 0.0])

    gen = torch.Generator().manual_seed(11)
    assert verify([1], [torch.tensor([0.0, 1.0, 0.0])], target, gen, 1, 5) == [1]
    assert len(calls) == 1  # accepted EOS never emits a bonus
    assert verify([0], [torch.tensor([1.0, 0.0, 0.0])], target, gen, 1, 5) == [1]


def test_group_counts_are_independent_groups():
    rows = [
        {
            "id": str(i),
            "group": "family",
            "split": "train",
            "domain": "code",
            "length_bin": "short",
            "source": "fixture",
            "text": str(i),
        }
        for i in range(3)
    ]
    assert validate_groups(rows)["train"] == 1


def test_preflight_does_not_trust_boolean_proofs_without_artifacts(tmp_path):
    cfg = json.loads(Path(__file__).with_name("pilot.json").read_text())
    manifest = {
        k: True
        for k in [
            "feature_alignment_validated",
            "native_logits_parity_validated",
            "input_gradient_validated",
            "checkpoint_hashes_complete",
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = preflight(cfg, bundle=tmp_path)
    assert not report["ready"]
    assert "native frozen weights missing or hash mismatch" in report["blockers"]


def test_native_bridge_backward_on_cpu_oracle(monkeypatch):
    # Exercise bridge plumbing without importing SGLang, CUDA or the EXL3 extension.
    import sys
    import types

    from .native import NativeProjector

    class Tensors:
        def __init__(self, trellis, suh, svh, mul1):
            self.trellis, self.suh, self.svh, self.mul1 = trellis, suh, svh, mul1

    module = types.ModuleType("sglang.srt.layers.quantization.exl3.ops")
    module.Exl3Tensors = Tensors
    w = torch.randn(12, 11)
    module.exl3_linear = lambda x, t: x @ w
    monkeypatch.setitem(sys.modules, module.__name__, module)
    tensors = Tensors(
        torch.zeros(1, dtype=torch.int16), torch.ones(12), torch.ones(11), True
    )
    bridge = NativeProjector(tensors, torch.ones(12), 1e-6, w)
    dense = DenseProjector(w.T, torch.ones(12), 1e-6)
    x = torch.randn(5, 12, requires_grad=True)
    a = bridge(x)
    b = dense(x)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.square().sum(), x)[0]
    gb = torch.autograd.grad(b.square().sum(), x)[0]
    torch.testing.assert_close(ga, gb)


def test_crossed_resampling_preserves_common_prompt_effects():
    rows = [
        {
            "seed": s,
            "prompt_id": str(p),
            "arm": a,
            "stratum": "code-short",
            "accepted": float(p) if a == "F" else 0.0,
        }
        for s in [1, 2, 3]
        for p in range(8)
        for a in ["C", "F"]
    ]
    result = paired_report(rows, draws=200)
    # Replicating identical checkpoints must not average away shared prompt variation.
    assert result["crossed_95"] == pytest.approx(result["conditional_95"])
    assert result["crossed_95"][1] > result["crossed_95"][0]
