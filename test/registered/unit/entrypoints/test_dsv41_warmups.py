"""The DSV4.1 warm-ups' request sequences: dsv41_prefill_shapes prefills only; dsv41_decode_paths decodes a two-chunk
greedy prompt, then sampled ones, so the decode-only Triton kernels load before serving."""

import asyncio
from types import SimpleNamespace

from sglang.srt.constants import WARMUP_RID_PREFIX
from sglang.srt.entrypoints import warmup
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 4096  # production's --chunked-prefill-size (benchmarks/dsv41_baseline/arm_env.py)


def _requests(name, disaggregation_mode="null"):
    sent = []

    async def generate_request(req, _request):
        sent.append(req)
        yield {}

    asyncio.run(
        warmup.execute_warmups(disaggregation_mode, [name], SimpleNamespace(generate_request=generate_request))
    )
    return sent


def test_prefill_shapes_prefill_each_size_then_one_grammar_request():
    sent = _requests("dsv41_prefill_shapes")
    sizes = tuple(len(r.input_ids) for r in sent)
    assert sizes == warmup.DSV41_PREFILL_WARMUP_SIZES + (16,)
    assert all(r.sampling_params["max_new_tokens"] == 1 for r in sent[:-1])
    assert sent[-1].sampling_params["regex"] == "(yes|no)"
    assert all(r.rid.startswith(WARMUP_RID_PREFIX) for r in sent)


def test_decode_paths_decode_a_two_chunk_greedy_prompt_then_sampled_ones():
    sent = _requests("dsv41_decode_paths")
    assert [len(r.input_ids) for r in sent] == [n for n, _ in warmup.DSV41_DECODE_WARMUP_REQUESTS]
    for r in sent:
        assert r.sampling_params["max_new_tokens"] >= 8
        assert r.sampling_params["ignore_eos"] is True
        # The confidence cut's T(w) calibration skips WARMUP_ rids; these requests' verifies must not reach it.
        assert r.rid.startswith(WARMUP_RID_PREFIX)
    assert len({r.rid for r in sent}) == len(sent)
    greedy, *sampled = sent
    # Two prefill chunks, the first a full one, which has to evict the SWA KV that dsv41_prefill_shapes left cached.
    assert CHUNK < len(greedy.input_ids) <= 2 * CHUNK
    assert greedy.sampling_params["temperature"] == 0.0
    # Sampled requests take DSpark's sampling draft and rejection-sampling accept; one also takes the top-p/top-k path.
    assert sampled and all(r.sampling_params["temperature"] > 0 for r in sampled)
    assert any("top_p" in r.sampling_params and "top_k" in r.sampling_params for r in sampled)
    # No extend exceeds the chunk the prefill warm-up already ran, so the hot-cache calibration's replay still holds.
    assert max(warmup.DSV41_PREFILL_WARMUP_SIZES) == CHUNK
    assert all(len(r.input_ids) < CHUNK for r in sampled)


def test_decode_paths_carry_the_fake_bootstrap_under_disaggregation():
    for r in _requests("dsv41_decode_paths", disaggregation_mode="prefill"):
        assert r.bootstrap_room == 0 and r.bootstrap_host == warmup.FAKE_BOOTSTRAP_HOST
