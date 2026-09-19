"""The expert-caching server-args gate accepts Window C's EXL3 launches (CPU)."""

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups import memory_hook
from sglang.srt.arg_groups.expert_stream_requirements import (
    expert_stream_requirements_for,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BREAKABLE_BS1 = CudaGraphConfig(
    decode=PhaseConfig(backend="breakable", bs=[1], max_bs=1),
    prefill=PhaseConfig(backend="disabled"),
)

# Plan Task 17's env.sh, less the paths: the dynamic arm.
WINDOW_C_ENV = {
    "SGLANG_DSV41_EXPERT_STREAM": True,
    "SGLANG_MOE_EXPERT_ROW_SOURCE": "shards",
    "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
    "SGLANG_MOE_PINNED_HOST_MB": 71680,
    "SGLANG_MOE_HOT_GPU_MB": 16384,
    "SGLANG_MOE_HOT_DYNAMIC": True,
    "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS": 256,
    "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": 32,
    "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": 8,
    "SGLANG_MOE_HOT_ASYNC_PROMOTIONS": False,
    "SGLANG_MOE_HOT_LOG_INTERVAL": 64,
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": False,
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": False,
    "SGLANG_MOE_EXPERT_DOORBELL": False,
    "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": 0,
}


@pytest.fixture
def model_dir(tmp_path):
    config = {"architectures": ["DeepseekV4ForCausalLM"], "quantization_config": {"quant_method": "exl3", "bits": 3.02}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _launch(model_dir, **changes):
    """The server arguments of Window C's Engines (disable_cuda_graph=True resolves to both phases disabled)."""
    values = dict(
        model_path=model_dir,
        quantization=None,
        ple_offload_embedding=False,
        cpu_offload_gb=0,
        offload_group_size=0,
        ple_offload_backend=None,
        moe_runner_backend="auto",
        tp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        disable_overlap_schedule=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        max_running_requests=4,
        expert_distribution_recorder_mode="per_pass",
        enable_waterfill=False,
        enable_eplb=False,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled"),
            prefill=PhaseConfig(backend="disabled"),
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _gate(args, **env_changes):
    env = {**WINDOW_C_ENV, **env_changes}
    error = None
    with ExitStack() as stack:
        for name, value in env.items():
            stack.enter_context(getattr(envs, name).override(value))
        stack.enter_context(patch.object(memory_hook, "resolving_view", lambda a: a))
        try:
            memory_hook.handle_offload_compatibility(args)
        except ValueError as caught:
            # EnvField.override restores the environment only on a clean exit, so
            # the error leaves the with-block as a value and is raised after it.
            error = caught
    if error is not None:
        raise error


def test_the_exl3_method_selects_the_exl3_requirements(model_dir):
    args = _launch(model_dir)
    assert expert_stream_requirements_for(args, args).label == "EXL3"


def test_window_c_launches_pass(model_dir):
    _gate(_launch(model_dir))  # dynamic residency, per-pass recorder
    _gate(_launch(model_dir, expert_distribution_recorder_mode=None), SGLANG_MOE_HOT_DYNAMIC=False)  # static seeded arm
    _gate(_launch(model_dir), SGLANG_MOE_HOT_GPU_MB=0)  # pinned tier only


@pytest.mark.parametrize(
    "launch_changes, env_changes, match",
    [
        ({}, {"SGLANG_DSV41_EXPERT_STREAM": False}, "requires SGLANG_DSV41_EXPERT_STREAM=1"),
        ({}, {"SGLANG_MOE_PINNED_HOST_MB": 0, "SGLANG_MOE_EXPERT_HOST_ARENA": True}, "SGLANG_MOE_EXPERT_HOST_ARENA"),
        ({"expert_distribution_recorder_mode": None}, {}, "stat or per_pass"),
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="full", bs=[1], max_bs=1), prefill=PhaseConfig(backend="disabled"))}, {}, "cannot run the Engram"),
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="breakable", bs=[1, 2], max_bs=2), prefill=PhaseConfig(backend="disabled"))}, {}, "max batch size 1"),
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="breakable", bs=[1], max_bs=1), prefill=PhaseConfig(backend="breakable"))}, {}, "runs prefill eagerly"),
    ],
)
def test_unsupported_launches_are_refused(model_dir, launch_changes, env_changes, match):
    with pytest.raises(ValueError, match=match):
        _gate(_launch(model_dir, **launch_changes), **env_changes)


def test_breakable_decode_at_batch_size_one_passes(model_dir):
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1))
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1, disable_overlap_schedule=True))


MULTI_STREAM = "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP"


@pytest.fixture
def multi_stream_unset(monkeypatch):
    """The env unset for the test; monkeypatch restores whatever the gate sets."""
    monkeypatch.setenv(MULTI_STREAM, "1")
    monkeypatch.delenv(MULTI_STREAM)


def test_breakable_decode_turns_the_alt_stream_overlap_off(model_dir, multi_stream_unset):
    # Captured in a breakable decode graph, DSV4's alt-stream overlap still yields wrong
    # output after the stats-stream re-fork (graph_parity on the truncated model: the
    # layer-0 MoE input differs at the first decode step); with it off, graph and eager
    # agree bit for bit.
    assert envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() is True
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1))
    assert envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() is False


def test_eager_launches_leave_the_alt_stream_overlap_alone(model_dir, multi_stream_unset):
    _gate(_launch(model_dir))
    assert not envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.is_set()


def test_an_explicit_alt_stream_overlap_with_breakable_decode_is_refused(model_dir, monkeypatch):
    monkeypatch.setenv(MULTI_STREAM, "1")
    with pytest.raises(ValueError, match=MULTI_STREAM):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1))
    monkeypatch.setenv(MULTI_STREAM, "0")
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
