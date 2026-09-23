"""The expert-caching server-args gate accepts Window C's EXL3 launches (CPU)."""

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups import memory_hook, pipeline
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
        moe_offload_preset="off",
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled"),
            prefill=PhaseConfig(backend="disabled"),
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _under_window_c_env(env_changes, run):
    """``run()`` under Window C's environment, a ``ValueError`` from it raised after the env is restored."""
    env = {**WINDOW_C_ENV, **env_changes}
    error = None
    with ExitStack() as stack:
        for name, value in env.items():
            stack.enter_context(getattr(envs, name).override(value))
        stack.enter_context(patch.object(memory_hook, "resolving_view", lambda a: a))
        try:
            run()
        except ValueError as caught:
            # EnvField.override restores the environment only on a clean exit, so
            # the error leaves the with-block as a value and is raised after it.
            error = caught
    if error is not None:
        raise error


def _gate(args, **env_changes):
    _under_window_c_env(env_changes, lambda: memory_hook.handle_offload_compatibility(args))


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
        ({}, {"SGLANG_MOE_HOT_ASYNC_PROMOTIONS": True}, "SGLANG_MOE_HOT_ASYNC_PROMOTIONS"),
        ({"cuda_graph_config": CudaGraphConfig(decode=PhaseConfig(backend="breakable", bs=[1], max_bs=1), prefill=PhaseConfig(backend="disabled"))}, {"SGLANG_MOE_HOT_ASYNC_PROMOTIONS": True}, "SGLANG_MOE_HOT_ASYNC_PROMOTIONS"),
        ({"speculative_algorithm": "DSPARK", "cuda_graph_config": BREAKABLE_BS1}, {}, "DSpark"),
    ],
)
def test_unsupported_launches_are_refused(model_dir, launch_changes, env_changes, match):
    with pytest.raises(ValueError, match=match):
        _gate(_launch(model_dir, **launch_changes), **env_changes)


def test_dspark_with_a_decode_graph_names_the_remedy(model_dir):
    # The refusal message must name both the algorithm and the remedy so a launch
    # operator knows what to change, not just that something is wrong.
    with pytest.raises(ValueError) as exc_info:
        _gate(_launch(model_dir, speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1))
    message = str(exc_info.value)
    assert "DSpark" in message
    assert "--cuda-graph-backend-decode disabled" in message


def test_dspark_speculation_passes_with_decode_disabled(model_dir):
    # DSpark's verify step runs up to block_size + 1 tokens; option C's in-graph scratch
    # and RAM-miss posting are sized for one token per step, so eager decode is required,
    # but is otherwise unaffected by speculative decoding being enabled.
    _gate(_launch(model_dir, speculative_algorithm="DSPARK"))


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


def test_graph_gather_over_the_pinned_tier_needs_breakable_decode(model_dir):
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)
    with pytest.raises(ValueError, match="GRAPH_GATHER"):
        _gate(_launch(model_dir), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)
    with pytest.raises(ValueError, match="PINNED_HOST_MB"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, SGLANG_MOE_PINNED_HOST_MB=0)
    with pytest.raises(ValueError, match="HOT_GPU_MB"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, SGLANG_MOE_HOT_GPU_MB=0)


def test_only_exl3_direct_graph_gather_admits_gpu_residency_update(model_dir):
    direct = dict(
        SGLANG_MOE_GPU_RESIDENCY_UPDATE=True,
        SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2,
        SGLANG_MOE_EXPERT_GRAPH_GATHER=True,
    )
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **direct)
    with pytest.raises(ValueError, match="GPU_RESIDENCY_UPDATE"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**direct, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 1})
    with pytest.raises(ValueError, match="GPU_RESIDENCY_UPDATE"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**direct, "SGLANG_MOE_EXPERT_GRAPH_GATHER": False})
    with pytest.raises(ValueError, match="GRAPH_GATHER"):
        _gate(_launch(model_dir), **direct)


def test_the_pre_parse_offload_pass_leaves_graph_checks_to_the_second_pass(model_dir):
    # run_resolution_pipeline runs handle_offload_compatibility twice; the first pass
    # comes before parse_cuda_graph_config, while cuda_graph_config is still the raw
    # CLI value (None for flag-only launches). That pass must not read it as eager
    # decode and refuse graph gather (the Task 16 option C Engine launch failed this way).
    raw = None
    _gate(_launch(model_dir, cuda_graph_config=raw), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)
    _gate(
        _launch(model_dir, cuda_graph_config=raw),
        SGLANG_MOE_EXPERT_GRAPH_GATHER=True,
        SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=True,
    )


class _PipelineStopped(Exception):
    pass


def _resolution_pipeline_hooks(args, hook_for=None):
    """Run ``run_resolution_pipeline`` on ``args`` with every step replaced, and return the step names in order.

    ``hook_for(name)`` supplies a stand-in for a step (called with ``args``); every
    other step is a no-op. The run stops at the first ``handle_offload_compatibility`` that follows
    ``handle_cuda_graph_config``, so it never reaches the direct calls that need a real ServerArgs.
    """
    names = []

    def record(hook, server_args):
        name = hook.__name__
        names.append(name)
        if hook_for is not None and hook_for(name) is not None:
            hook_for(name)(server_args)
        if name == "handle_offload_compatibility" and "handle_cuda_graph_config" in names:
            raise _PipelineStopped

    with patch.object(pipeline, "run_hook", record):
        try:
            pipeline.run_resolution_pipeline(args)
        except _PipelineStopped:
            pass
    return names


def test_the_resolution_pipeline_checks_offload_after_the_cuda_graph_config_is_parsed(model_dir):
    # The gate's graph checks run only once cuda_graph_config is parsed (they are skipped on
    # the raw CLI value), so they depend on the offload pass that follows handle_cuda_graph_config.
    # If a merge moved that pass before it, or dropped it, every EXL3 graph rule would go unchecked.
    names = _resolution_pipeline_hooks(_launch(model_dir))
    offload = [i for i, name in enumerate(names) if name == "handle_offload_compatibility"]
    assert names.count("handle_cuda_graph_config") == 1
    assert len(offload) == 2, names
    parsed_at = names.index("handle_cuda_graph_config")
    assert offload[0] < parsed_at < offload[1]


def test_a_flag_only_full_decode_graph_launch_passes_pass_one_and_is_refused_after_parsing(model_dir):
    # Flag-only launch: cuda_graph_config is None until handle_cuda_graph_config parses it. The
    # first offload pass must accept it; the second, on the parsed decode `full` config, must refuse it.
    parsed = CudaGraphConfig(
        decode=PhaseConfig(backend="full", bs=[1], max_bs=1),
        prefill=PhaseConfig(backend="disabled"),
    )
    args = _launch(model_dir, cuda_graph_config=None)
    seen = []

    def parse(server_args):
        seen.append("parsed")
        server_args.cuda_graph_config = parsed

    def offload(server_args):
        memory_hook.handle_offload_compatibility(server_args)
        seen.append("offload passed")

    hooks = {"handle_cuda_graph_config": parse, "handle_offload_compatibility": offload}
    with pytest.raises(ValueError, match="cannot run the Engram"):
        _under_window_c_env(
            {"SGLANG_MOE_EXPERT_GRAPH_GATHER": True},
            lambda: _resolution_pipeline_hooks(args, hooks.get),
        )
    assert seen == ["offload passed", "parsed"]  # pass 1 accepted; pass 2 raised inside its own call
    # The parsed config alone is what refuses it: the same launch with the config already parsed.
    with pytest.raises(ValueError, match="cannot run the Engram"):
        _gate(_launch(model_dir, cuda_graph_config=parsed), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)


def test_graph_gather_keeps_the_alt_stream_overlap_off(model_dir, multi_stream_unset):
    # The fused MoE's temp buffers are shared by every layer: sound only on one stream.
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)
    assert envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() is False


def test_prefetch_needs_graph_gather_under_breakable_decode(model_dir):
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=True)
    with pytest.raises(ValueError, match="SGLANG_DSV41_ENABLE_EXPERT_PREFETCH"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=True)
    with pytest.raises(ValueError, match="SGLANG_DSV41_ENABLE_EXPERT_PREFETCH"):
        _gate(_launch(model_dir), SGLANG_DSV41_ENABLE_EXPERT_PREFETCH=True)  # eager decode: no in-graph MoE to post from


def test_prefetch_is_off_unless_the_env_var_is_set():
    from sglang.srt.layers.moe.exl3_expert_format import prefetch_enabled

    assert prefetch_enabled() is False
    with envs.SGLANG_DSV41_ENABLE_EXPERT_PREFETCH.override(True):
        assert prefetch_enabled() is True


def test_the_exl3_requirements_read_graph_gathers_from_the_pinned_tier(model_dir):
    args = _launch(model_dir)
    assert expert_stream_requirements_for(args, args).graph_gather_host_source == "pinned_tier"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
