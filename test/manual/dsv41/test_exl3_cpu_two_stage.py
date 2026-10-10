"""Two-stage CPU misses on the EXL3 kernel (SGLANG_DSV41_CPU_TWO_STAGE): the forward a staged job runs, its stage-two
wait between GateUp and Middle, gives the one-stage forward's output bit for bit. Needs the optimized ext build
(SGLANG_EXL3_SRC, SGLANG_DSV41_CPU_EXPERTS=1) and 4 cores in the affinity mask.

Each case holds the routed slots' w2 back: their w2 slabs are poisoned, and kernel_forward_two_stage's helper thread
restores them 50 ms into the call, then opens the gate. The forward starts at once, runs PrepareGateUp and GateUp on
the w13 bytes, waits, and runs Middle (the down input reads w2_suh), Down and Accumulate on the restored w2: a forward
that read any w2 byte before its wait would read the poison."""

import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

sys.path.insert(0, os.path.dirname(__file__))

from test_cpu_expert_engines_exl3 import CAP, CORES, HIDDEN, INTER, _kernel, _on_threads, _random_slabs  # noqa: E402

DELAY_S = 0.05
SECOND = ("w2_trellis", "w2_suh", "w2_svh")

# (rows' slots, rows' weights, the slots whose w2 is still landing): one token on one expert; four verify tokens
# sharing an expert (two kernel chunks of it); one token on three experts; three tokens over five experts of which two
# were already whole (a record's misses landing apart).
CASES = [
    ([[2]], [[1.0]], {2}),
    ([[3], [3], [3], [3]], [[1.0], [0.5], [0.25], [2.0]], {3}),
    ([[0, 2, 5]], [[0.5, 0.3, 0.2]], {0, 2, 5}),
    ([[0, 1, 4], [1, 4, -1], [0, 3, 4]], [[0.5, 0.3, 0.2], [0.6, 0.4, 0.0], [0.2, 0.2, 0.6]], {1, 4}),
]
CASE_IDS = ["one_token", "verify_tokens", "three_experts", "mixed_landing"]


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("slots, weights, landing", CASES, ids=CASE_IDS)
def test_a_staged_forward_matches_the_one_stage_forward_bit_for_bit(monkeypatch, slots, weights, landing, accumulate):
    """Mutants: call stage_two after Middle -- red (Middle's down input reads the poisoned w2_suh); after Down -- red;
    never -- red on the call count and the output."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    slabs = _random_slabs(20261010, HIDDEN, INTER)
    layer = es.kernel_layer(trait.kernel_address(), trait.layer_spec(slabs, CAP), variant="instr")
    g = torch.Generator().manual_seed(20261010)
    x = (torch.randn(len(slots), HIDDEN, generator=g) * 4.0).half()
    start = torch.randn(len(slots), HIDDEN, generator=g)
    saved = {(name, slot): slabs[name][slot].clone() for name in SECOND for slot in landing}

    def forward(staged):
        out = start.clone() if accumulate else torch.full((len(slots), HIDDEN), float("nan"))
        args = (layer, x, torch.tensor(slots, dtype=torch.int32), torch.tensor(weights, dtype=torch.float32), out)
        if not staged:
            status, why = es.kernel_forward(*args, threads=4, cores=cores, accumulate=accumulate, variant="instr")
            return status, why, None, out
        for (name, slot), _ in saved.items():
            slabs[name][slot].view(torch.uint8).fill_(0xA5)
        restore = [
            (slabs[name][slot].data_ptr(), copy.data_ptr(), copy.numel() * copy.element_size())
            for (name, slot), copy in saved.items()
        ]
        status, why, info = es.kernel_forward_two_stage(
            *args, threads=4, cores=cores, accumulate=accumulate, restore=restore, delay_s=DELAY_S, variant="instr"
        )
        return status, why, info, out

    try:
        (results,) = _on_threads(lambda _: [forward(False), forward(True), forward(False)], [None])
        (s1, w1, _, one), (s2, w2, info, two), (s3, w3, _, again) = results
        assert (s1, w1, s2, w2, s3, w3) == (0, "", 0, "", 0, "")
        assert torch.isfinite(one).all() and torch.equal(one, again)
        assert info["calls"] == 1 and info["waited_ns"] >= DELAY_S * 1e9 / 2, info
        assert torch.equal(two.view(torch.int32), one.view(torch.int32))
        for (name, slot), copy in saved.items():
            assert torch.equal(slabs[name][slot].view(torch.uint8), copy.view(torch.uint8))
    finally:
        es.kernel_drop(layer, variant="instr")


def test_the_poison_reaches_a_forward_that_reads_w2_without_waiting(monkeypatch):
    """The control: the same poisoned w2 read by a one-stage forward changes its output, so the cases above can see a
    forward that read w2 before its wait."""
    from sglang.kernels.ops.moe import expert_stream_transport as es

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    trait, cores = _kernel()
    slabs = _random_slabs(20261010, HIDDEN, INTER)
    layer = es.kernel_layer(trait.kernel_address(), trait.layer_spec(slabs, CAP), variant="instr")
    x = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(1)).half()
    slots, weights = torch.tensor([[2]], dtype=torch.int32), torch.tensor([[1.0]])

    def forward():
        out = torch.full((1, HIDDEN), float("nan"))
        assert es.kernel_forward(layer, x, slots, weights, out, threads=4, cores=cores, variant="instr")[0] == 0
        return out

    try:
        (clean,) = _on_threads(lambda _: forward(), [None])
        for name in SECOND:
            slabs[name][2].view(torch.uint8).fill_(0xA5)
        (poisoned,) = _on_threads(lambda _: forward(), [None])
        assert not torch.equal(clean, poisoned)
    finally:
        es.kernel_drop(layer, variant="instr")
