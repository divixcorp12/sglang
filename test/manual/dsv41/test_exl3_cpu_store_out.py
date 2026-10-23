"""The EXL3 CPU kernel's store-first overwrite (ForwardCall::store_first, SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT) gives the
bits of the zero-then-accumulate it replaces (csrc/exl3/optimized/forward_plan.hpp).

The CPU expert engine (cpu_experts.h run_job) ran a verify's per-token job as: zero the job's token rows, then an
accumulating forward. With the flag it hands the kernel an overwrite with store_first: each token's first sum is a
store, the rest add, and a token no route writes reads 0. Every case runs both on the same call, the old one into
zeroed rows and the new one into rows full of NaN (whatever an earlier record left), plus the kernel's own overwrite
(store_first off, the memset inside the team), and compares the bits (int32 views, so a -0 against +0 fails). A
second, accumulating job on the same rows (a record's later CPU misses into part 1) must then agree too.

Needs SGLANG_EXL3_SRC, SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1 and the bw tier (SGLANG_EXPERT_STREAM_CPU_MAX_ISA=bw on a host above it). Run on
divix01 under taskset -c 0-51,56-107.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from test_exl3_cpu_dsv41_plan_multitoken import CAP, H, _slabs  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("SGLANG_EXL3_SRC"), reason="needs SGLANG_EXL3_SRC")

LIMIT = 10.0

# (name, routes [tokens][k]; -1 is no route). A verify's per-token job: a token holds the job's lanes it routes.
CASES = [
    ("1-token", [[4, 0, 5]]),
    ("1-token-1-lane", [[7]]),
    ("2-tokens-distinct", [[4, 0], [5, 1]]),
    ("2-tokens-shared", [[3, 6], [3, 6]]),
    ("2-tokens-one-unrouted", [[3, 6], [-1, -1]]),
    ("6-tokens-distinct", [[0, -1], [1, -1], [2, -1], [3, -1], [4, -1], [5, -1]]),
    ("6-tokens-one-expert", [[2]] * 6),  # past CHUNK_M: two chunks of the one expert, each token stored once
    ("6-tokens-shared", [[1, 2], [2, 1], [1, -1], [-1, 2], [1, 2], [2, -1]]),
    ("6-tokens-unrouted", [[1, -1], [-1, -1], [1, 3], [-1, -1], [3, -1], [-1, -1]]),  # tokens 1, 3, 5 route none
    ("6-tokens-dead-route", [[1, 2], [1, 2], [2, -1], [-1, -1], [1, -1], [2, 1]]),  # token 0's route to 1 weighs 0
    ("1-token-twice", [[1, 1, 2]]),  # the same expert on two lanes of one token: a store, then an add
]
ZERO_WEIGHT = {"6-tokens-dead-route": (0, 0)}


@pytest.fixture(scope="module")
def layer():
    os.environ.setdefault("SGLANG_EXL3_MOE_CPU_PIN", "0")
    from sglang.kernels.ops.moe import expert_stream_transport as es
    from sglang.srt.layers.quantization.exl3.ext import cpu_act_defines, exl3_ext, optimized_cpu
    from sglang.srt.layers.quantization.exl3.schemes import Exl3CpuQuantTrait

    if not optimized_cpu(cpu_act_defines()):
        pytest.skip("the kernel under test is the optimized one: set SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS=1")
    ext = exl3_ext()
    trait = Exl3CpuQuantTrait(ext, act_limit=LIMIT)
    handle = es.kernel_layer(trait.kernel_address(), trait.layer_spec(_slabs(20261015), CAP), variant="instr")
    yield handle
    es.kernel_drop(handle, variant="instr")


def _inputs(name, routes, seed=0):
    g = torch.Generator().manual_seed(sum(map(ord, name)) + seed)
    tokens, k = len(routes), len(routes[0])
    x = (torch.randn(tokens, H, generator=g) * 4.0).half()
    weights = torch.rand(tokens, k, generator=g) + 0.05
    slots = torch.tensor(routes, dtype=torch.int32)
    weights[slots < 0] = 0.0  # as expand_tokens writes an unrouted lane
    if name in ZERO_WEIGHT:
        weights[ZERO_WEIGHT[name]] = 0.0
    return x, slots, weights


def _forward(layer, x, slots, weights, out, *, threads, accumulate, store_first):
    from sglang.kernels.ops.moe import expert_stream_transport as es

    status, why = es.kernel_forward(
        layer, x, slots, weights, out, threads=threads, accumulate=accumulate, store_first=store_first, variant="instr"
    )
    assert (status, why) == (0, "")
    return out


def _bits(t):
    return t.view(torch.int32)


@pytest.mark.parametrize("threads", [1, 4, 10])
@pytest.mark.parametrize("name,routes", CASES, ids=[c[0] for c in CASES])
def test_store_first_gives_the_bits_of_zero_then_accumulate(layer, name, routes, threads):
    """Mutations: Accumulate stores every row (a shared token loses its first expert's sum); dispatch leaves the
    unrouted tokens out of the zeroing (their rows keep NaN)."""
    x, slots, weights = _inputs(name, routes)
    tokens = len(routes)
    old = _forward(layer, x, slots, weights, torch.zeros(tokens, H), threads=threads, accumulate=True, store_first=False)
    new = _forward(
        layer, x, slots, weights, torch.full((tokens, H), float("nan")), threads=threads, accumulate=False,
        store_first=True,
    )
    own = _forward(
        layer, x, slots, weights, torch.full((tokens, H), float("nan")), threads=threads, accumulate=False,
        store_first=False,
    )
    assert torch.isfinite(old).all(), name
    assert torch.equal(_bits(new), _bits(old)), name
    assert torch.equal(_bits(own), _bits(old)), name
    for t in range(tokens):
        if (slots[t] < 0).all():
            assert torch.equal(_bits(new[t]), _bits(torch.zeros(H))), (name, t)
    # A later job of the record adds into the same rows: accumulate, so store_first changes nothing there.
    x2, slots2, weights2 = _inputs(name, routes, seed=1)
    _forward(layer, x2, slots2, weights2, old, threads=threads, accumulate=True, store_first=False)
    _forward(layer, x2, slots2, weights2, new, threads=threads, accumulate=True, store_first=True)
    assert torch.equal(_bits(new), _bits(old)), name


def test_a_zero_product_stores_plus_zero_as_the_old_add_did(layer):
    """A route whose weight is 0 makes w * src exactly +-0; the old 0 + (-0) is +0, so the store must be +0 too, not
    the -0 a bare product can give. Mutation: the store writes w * src without the opaque zero (expect -0 bits where
    src is negative, if the compiler drops the add)."""
    name, routes = "zero-weight", [[1], [1], [-1]]
    x, slots, weights = _inputs(name, routes)
    weights[:] = 0.0
    weights[slots < 0] = 0.0
    old = _forward(layer, x, slots, weights, torch.zeros(3, H), threads=4, accumulate=True, store_first=False)
    new = _forward(layer, x, slots, weights, torch.full((3, H), float("nan")), threads=4, accumulate=False,
                   store_first=True)
    assert torch.equal(_bits(new), _bits(old))
    assert (_bits(new) == 0).all()
