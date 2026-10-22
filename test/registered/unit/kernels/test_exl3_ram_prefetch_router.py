"""The router seam of the RAM prefetch scorer (csrc/moe/expert_stream/router.h): each router's host score against a
NumPy reference, through the host test export expert_stream_router_score (CPU).

Compile-time contract, checked when the module builds rather than here: router.h static_asserts that
SqrtSoftplusBias supports the learned head and SigmoidBias does not, and SpecHeadKernelFor<R> (spec_head.cuh)
static_asserts R::kSupportsLearnedHead. A negative-compile test is impractical in the JIT flow, so the refusal is
exercised by hand: instantiating SpecHeadKernelFor<SigmoidBias> in a device TU must fail with the message "the learned
L+2 head was trained on DSV4 routing"."""

import numpy as np
import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as es
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _sqrt_softplus_reference(dot, bias):
    z = dot.astype(np.float64)
    softplus = np.where(z > 20.0, z, np.log1p(np.exp(np.minimum(z, 20.0))))
    return (np.sqrt(softplus) + bias.astype(np.float64)).astype(np.float32)


def _sigmoid_reference(dot, bias):
    z = dot.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-z)) + bias.astype(np.float64)).astype(np.float32)


REFERENCES = {"sqrt_softplus": _sqrt_softplus_reference, "sigmoid": _sigmoid_reference}


def _case(seed):
    rng = np.random.default_rng(seed)
    # Both sides of softplus' threshold of 20, and logits sigmoid saturates on.
    dots = np.concatenate(
        [
            rng.normal(0.0, 3.0, 4096),
            rng.uniform(15.0, 25.0, 512),
            rng.uniform(-60.0, -20.0, 256),
            np.array([0.0, 20.0, 20.000002, -0.0]),
        ]
    ).astype(np.float32)
    biases = rng.normal(0.0, 0.1, dots.shape).astype(np.float32)
    return dots, biases


@pytest.mark.parametrize("router", sorted(REFERENCES))
@pytest.mark.parametrize("seed", range(4))
def test_the_host_score_matches_the_numpy_reference(router, seed):
    dots, biases = _case(seed)
    got = es.router_score(router, torch.from_numpy(dots), torch.from_numpy(biases)).numpy()
    np.testing.assert_allclose(got, REFERENCES[router](dots, biases), rtol=2e-6, atol=1e-6)


def test_the_two_routers_differ():
    """Mutant: router_score dispatching both indices to one router -- red."""
    dots, biases = _case(0)
    a = es.router_score("sqrt_softplus", torch.from_numpy(dots), torch.from_numpy(biases))
    b = es.router_score("sigmoid", torch.from_numpy(dots), torch.from_numpy(biases))
    assert not torch.equal(a, b)


def test_an_unknown_router_is_refused():
    with pytest.raises(ValueError, match="router"):
        es.router_score("softmax", torch.zeros(2), torch.zeros(2))
