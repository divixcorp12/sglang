"""The expert-caching server-args gate accepts Window C's EXL3 launches (CPU)."""

import json
import os
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
    "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": 0,
}


# The residency the RAM-miss service needs: the in-graph updater at insert-on-miss stage 2.
DIRECT = {"SGLANG_MOE_GPU_RESIDENCY_UPDATE": True, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 2}


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


def test_the_removed_doorbell_variable_does_not_gate_an_exl3_launch(model_dir, monkeypatch):
    # Assert the resolved format first: a gate that silently fell back to NVFP4 proves nothing (divix01-run-protocol).
    monkeypatch.setenv("SGLANG_MOE_EXPERT_DOORBELL", "1")
    args = _launch(model_dir)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    _gate(args)


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
    # An eager DSpark verify needs none of the graphed verify's configuration (§33.8).
    _gate(_launch(model_dir, speculative_algorithm="DSPARK"))


# DSpark's verify in the breakable decode graph (DSV41_REFERENCE.md §33.8): DIRECT residency at W miss lanes.
GRAPHED_VERIFY = {"SGLANG_MOE_EXPERT_GRAPH_GATHER": True, **DIRECT, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 8}


def test_dspark_verify_in_the_breakable_decode_graph_passes(model_dir):
    _gate(_launch(model_dir, speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1), **GRAPHED_VERIFY)


@pytest.mark.parametrize(
    "launch_changes, env_changes, match",
    [
        ({"speculative_algorithm": "EAGLE"}, GRAPHED_VERIFY, "graphs the verify of DSpark only"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 0}, "MISS_LANES=1-64"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 65}, "MISS_LANES=1-64"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 1}, "INSERT_ON_MISS_STAGE=2"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER": False}, "SGLANG_MOE_EXPERT_GRAPH_GATHER=1"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_RAGGED_VERIFY_MODE": "compact"}, "SGLANG_RAGGED_VERIFY_MODE=static"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_DSV41_CPU_EXPERTS": True}, "SGLANG_DSV41_CPU_EXPERTS needs"),
    ],
)
def test_a_graphed_dspark_verify_needs_its_configuration(model_dir, launch_changes, env_changes, match):
    launch = dict(speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1) | launch_changes
    with pytest.raises(ValueError, match=match) as raised:
        _gate(_launch(model_dir, **launch), **env_changes)
    if match != "SGLANG_DSV41_CPU_EXPERTS needs":
        assert "--cuda-graph-backend-decode disabled" in str(raised.value)


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
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, **DIRECT)
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


@pytest.mark.parametrize(
    "off",
    [
        {"SGLANG_MOE_GPU_RESIDENCY_UPDATE": False},
        {"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 1},
        {"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 0},
    ],
    ids=["no_updater", "stage_1", "stage_0"],
)
def test_graph_gather_needs_direct_residency(model_dir, off):
    """The RAM-miss service reads the VRAM-hot set from the DIRECT updater's records: no other residency mode feeds it.
    Mutation: the gate admits a graph-gather launch whose residency is not DIRECT."""
    direct = dict(
        SGLANG_MOE_GPU_RESIDENCY_UPDATE=True,
        SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2,
        SGLANG_MOE_EXPERT_GRAPH_GATHER=True,
    )
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **direct)
    with pytest.raises(ValueError, match="needs DIRECT residency.*GPU_RESIDENCY_UPDATE=1.*INSERT_ON_MISS_STAGE=2"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**direct, **off})


def test_the_pre_parse_offload_pass_leaves_graph_checks_to_the_second_pass(model_dir):
    # run_resolution_pipeline runs handle_offload_compatibility twice; the first pass
    # comes before parse_cuda_graph_config, while cuda_graph_config is still the raw
    # CLI value (None for flag-only launches). That pass must not read it as eager
    # decode and refuse graph gather (the Task 16 option C Engine launch failed this way).
    raw = None
    _gate(_launch(model_dir, cuda_graph_config=raw), SGLANG_MOE_EXPERT_GRAPH_GATHER=True)


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
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_MOE_EXPERT_GRAPH_GATHER=True, **DIRECT)
    assert envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() is False


def test_the_exl3_requirements_read_graph_gathers_from_the_pinned_tier(model_dir):
    args = _launch(model_dir)
    assert expert_stream_requirements_for(args, args).graph_gather_host_source == "pinned_tier"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))


CPU_EXPERTS_ENV = dict(
    SGLANG_MOE_EXPERT_GRAPH_GATHER=True,
    SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=True,
    SGLANG_DSV41_ENABLE_LAYER_FUSION=True,
    SGLANG_MOE_GPU_RESIDENCY_UPDATE=True,
    SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2,
    SGLANG_MOE_EXPERT_FUSED_PLAN=True,
    SGLANG_DSV41_CPU_EXPERTS=True,
)


@pytest.mark.parametrize("missing", [name for name in CPU_EXPERTS_ENV if name != "SGLANG_DSV41_CPU_EXPERTS"])
def test_cpu_experts_name_every_missing_prerequisite(model_dir, missing):
    """Each is load-bearing: without it the CPU lanes are never completed (the copy engine), never left out of the fused
    MoE (layer fusion), not the lowest-scored misses (DIRECT residency's keys, which only the fused plan sorts by), or
    run on no cores; the refusal names the one that is off."""
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **CPU_EXPERTS_ENV)
    off = {"SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 1}.get(missing, False)
    with pytest.raises(ValueError, match=f"SGLANG_DSV41_CPU_EXPERTS needs {missing}"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**CPU_EXPERTS_ENV, missing: off})


@pytest.mark.parametrize("update", [True, False])
def test_cpu_experts_without_direct_residency_are_refused(model_dir, update):
    """Stage 1 is not DIRECT.

    With the residency update on, the update's own EXL3 rule would say to turn it off.
    The CPU-experts rule runs first and names what to set instead: the update on, at stage 2.
    """
    env = {**CPU_EXPERTS_ENV, "SGLANG_MOE_GPU_RESIDENCY_UPDATE": update, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 1}
    wanted = ("" if update else "SGLANG_MOE_GPU_RESIDENCY_UPDATE=1, ") + "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2"
    with pytest.raises(ValueError, match=f"SGLANG_DSV41_CPU_EXPERTS needs {wanted}$") as refused:
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **env)
    assert "set it to 0" not in str(refused.value)


FULL_BS1 = CudaGraphConfig(
    decode=PhaseConfig(backend="full", bs=[1], max_bs=1),
    prefill=PhaseConfig(backend="disabled"),
)


# With decode graphs disabled, the generic rule "graph gather requires decode CUDA graphs" refuses first;
# graph gather off is how a disabled launch reaches the EXL3 rules.
_EAGER_DECODE = {"SGLANG_MOE_EXPERT_GRAPH_GATHER": False}


@pytest.mark.parametrize(
    "changes, env",
    [
        ({}, _EAGER_DECODE),  # both phases disabled
        ({"cuda_graph_config": FULL_BS1}, {}),
        ({"speculative_algorithm": "DSPARK"}, _EAGER_DECODE),
    ],
    ids=["disabled", "full", "spec-disabled"],
)
def test_cpu_experts_need_the_breakable_decode_graph(model_dir, changes, env):
    """Refused before the speculative and backend rules; the refusal never suggests disabled decode graphs."""
    args = _launch(model_dir, **changes)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    with pytest.raises(ValueError, match="pass --cuda-graph-backend-decode breakable") as refused:
        _gate(args, **{**CPU_EXPERTS_ENV, **env})
    assert "disabled" not in str(refused.value)


def test_cpu_misses_need_cpu_experts(model_dir):
    """SGLANG_DSV41_CPU_EXPERTS_MISSES only widens which lanes the CPU may take; without CPU experts no CPU thread
    exists to compute them, and the device would type misses CPU that nothing completes. With CPU experts it passes."""
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **CPU_EXPERTS_ENV, SGLANG_DSV41_CPU_EXPERTS_MISSES=True)
    with pytest.raises(ValueError, match="SGLANG_DSV41_CPU_EXPERTS_MISSES needs SGLANG_DSV41_CPU_EXPERTS=1"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_DSV41_CPU_EXPERTS_MISSES=True)


@pytest.mark.parametrize("value", ["ce", "sm"])
def test_either_hit_copy_passes_with_cpu_experts(model_dir, value):
    """CPU completion uses the copy thread and the gate whichever way the hits are copied."""
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_HIT_COPY=value)


def test_an_unknown_hit_copy_is_refused(model_dir):
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_HIT_COPY must be ce or sm"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), SGLANG_DSV41_RAM_HIT_COPY="dma")


def test_cpu_experts_refuse_the_prefetch_pull_join(model_dir):
    """The join moves a route to the pulled slot after planning: a CPU lane's route would match no plan slot and be
    computed on the GPU as well as on the CPU."""
    with pytest.raises(ValueError, match="PREFETCH_PULL_MODE"):
        _gate(
            _launch(model_dir, cuda_graph_config=BREAKABLE_BS1),
            **{**CPU_EXPERTS_ENV, "SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE": "always"},
        )


DSPARK_CPU_ENV = dict(
    SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=True,
    SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES="18-27",
    SGLANG_EXL3_CPU_ACT_RESIDUAL=True,
    SGLANG_EXL3_CPU_ACT_BLOCK=128,
)


@pytest.fixture
def cpu_pin_off(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")


def test_dspark_cpu_experts_pass_with_dspark_and_cores(model_dir, cpu_pin_off):
    _gate(_launch(model_dir, speculative_algorithm="DSPARK"), **DSPARK_CPU_ENV)


def test_dspark_cpu_experts_without_dspark_are_refused(model_dir):
    with pytest.raises(ValueError, match="--speculative-algorithm DSPARK"):
        _gate(_launch(model_dir), **DSPARK_CPU_ENV)


def test_dspark_cpu_experts_named_cores_need_two(model_dir):
    with pytest.raises(ValueError, match="SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES"):
        _gate(
            _launch(model_dir, speculative_algorithm="DSPARK"),
            **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": "18"},
        )


def test_dspark_cpu_experts_without_named_cores_pass(model_dir, cpu_pin_off):
    """Unset cores: ThreadingConfig derives the draft's cores at its start, so the launch does not ask for them."""
    _gate(
        _launch(model_dir, speculative_algorithm="DSPARK"),
        **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": ""},
    )


def test_a_bad_resident_file_is_refused_at_launch(model_dir, tmp_path, cpu_pin_off):
    bad = tmp_path / "resident.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="resident.json"):
        _gate(
            _launch(model_dir, speculative_algorithm="DSPARK"),
            **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": str(bad)},
        )


def test_dspark_cpu_experts_need_the_kernel_pin_off(model_dir, monkeypatch):
    monkeypatch.delenv("EXL3_MOE_CPU_PIN", raising=False)
    with pytest.raises(ValueError, match="EXL3_MOE_CPU_PIN=0"):
        _gate(_launch(model_dir, speculative_algorithm="DSPARK"), **DSPARK_CPU_ENV)


def test_dspark_cpu_experts_refuse_more_threads_than_cores(model_dir, cpu_pin_off):
    with pytest.raises(ValueError, match="SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS"):
        _gate(
            _launch(model_dir, speculative_algorithm="DSPARK"),
            **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS": 13},
        )


def test_dspark_cpu_experts_refuse_the_nvme_interrupt_cores(model_dir, cpu_pin_off):
    """Cores 64-71 take the NVMe completion interrupts the RAM-miss reads wait on; a spinning draft worker there would
    stall them, so the launch refuses rather than the first draft call."""
    with pytest.raises(ValueError, match="64"):
        _gate(
            _launch(model_dir, speculative_algorithm="DSPARK"),
            **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": "62-65"},
        )


@pytest.mark.parametrize(
    "unset", [{"SGLANG_EXL3_CPU_ACT_RESIDUAL": False}, {"SGLANG_EXL3_CPU_ACT_BLOCK": 0}, {"SGLANG_EXL3_CPU_ACT_BLOCK": 64}]
)
def test_dspark_cpu_experts_need_the_optimized_cpu_kernel_build(model_dir, cpu_pin_off, unset):
    """Only the optimized EXL3 build exports a CpuExpertKernel (sglang_exl3_cpu::kernel_address), and with the target's
    CPU experts refused under speculation the residual/128 defines are what select it; without them the draft would
    fail at its first step, after the model loaded."""
    with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_ACT_RESIDUAL=1"):
        _gate(_launch(model_dir, speculative_algorithm="DSPARK"), **{**DSPARK_CPU_ENV, **unset})


DSPARK_BREAKABLE = dict(speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1)
# The recipe's DSpark mode: a lane per route (MISS_LANES unset), 8 victim lanes.
SPILL = {"SGLANG_MOE_EXPERT_GRAPH_GATHER": True, **DIRECT, "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": 8}


@pytest.mark.parametrize(
    "name, value",
    [("SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES", "12-15"), ("SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS", 4)],
)
def test_draft_cores_are_refused_when_the_draft_shares_the_cpu_team(model_dir, name, value):
    env = {
        **CPU_EXPERTS_ENV,
        **SPILL,
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": True,
        "SGLANG_EXL3_CPU_ACT_RESIDUAL": True,
        "SGLANG_EXL3_CPU_ACT_BLOCK": 128,
        name: value,
    }
    # The refusal is an EXL3 rule, so matching it also shows the gate resolved the EXL3 requirements, not the NVFP4
    # fallback a non-directory model_path gets.
    with pytest.raises(ValueError, match="shares the GPU node's CPU expert team"):
        _gate(_launch(model_dir, **DSPARK_BREAKABLE), **env)


def _exl3_launch(model_dir, **changes):
    # Assert the resolved format first: a gate that silently fell back to NVFP4 proves nothing (divix01-run-protocol).
    args = _launch(model_dir, **changes)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    return args


def test_cpu_experts_with_a_graphed_dspark_verify_pass(model_dir):
    # The envs share SGLANG_MOE_EXPERT_GRAPH_GATHER and the DIRECT pair, so they merge rather than splat twice.
    _gate(_exl3_launch(model_dir, **DSPARK_BREAKABLE), **{**CPU_EXPERTS_ENV, **SPILL})


def test_a_graphed_dspark_verify_with_cpu_experts_needs_victim_lanes(model_dir):
    """A narrowed MISS_LANES verify with CPU experts and no spill was never run on a GPU; the tested shape is a lane per
    route plus VICTIM_LANES."""
    with pytest.raises(ValueError, match="SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES") as refused:
        _gate(_exl3_launch(model_dir, **DSPARK_BREAKABLE), **{**CPU_EXPERTS_ENV, **GRAPHED_VERIFY})
    assert "graphed DSpark verify" in str(refused.value)


@pytest.mark.parametrize(
    "launch, env, match",
    [
        ({"speculative_algorithm": "EAGLE"}, GRAPHED_VERIFY, "graphs the verify of DSpark only"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 0}, "MISS_LANES=1-64"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_RAGGED_VERIFY_MODE": "compact"}, "SGLANG_RAGGED_VERIFY_MODE=static"),
    ],
)
def test_cpu_experts_under_speculation_need_the_graphed_dspark_verify(model_dir, launch, env, match):
    """The graphed verify's own rules, with a remedy for a CPU-experts launch: the verify must be graphed, so the
    eager-decode remedy is never offered."""
    with pytest.raises(ValueError, match=match) as refused:
        _gate(_exl3_launch(model_dir, **(DSPARK_BREAKABLE | launch)), **{**CPU_EXPERTS_ENV, **env})
    assert "--cuda-graph-backend-decode disabled" not in str(refused.value)
    assert "SGLANG_DSV41_CPU_EXPERTS" in str(refused.value)


@pytest.mark.parametrize(
    "env, match",
    [
        ({"SGLANG_DSV41_CPU_EXPERTS": False}, "needs SGLANG_DSV41_CPU_EXPERTS=1"),
        ({"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 36}, "unset SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES"),
    ],
)
def test_victim_lanes_need_cpu_experts_and_a_lane_per_route(model_dir, env, match):
    """Spill's victimless lanes are the CPU's, and its record has a lane per route, so a 6-token verify never
    outnumbers its lanes (Review Focus 5)."""
    with pytest.raises(ValueError, match=match):
        _gate(_exl3_launch(model_dir, **DSPARK_BREAKABLE), **{**CPU_EXPERTS_ENV, **SPILL, **env})


def test_victim_lanes_need_a_dspark_launch(model_dir):
    """Spill is the graphed DSpark verify's mode; on a non-speculative batch-size-1 launch V=4 would spill two of six
    routes, an untested mode."""
    with pytest.raises(ValueError, match="--speculative-algorithm DSPARK"):
        _gate(_exl3_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**CPU_EXPERTS_ENV, **SPILL})


@pytest.fixture
def top_k_6_model_dir(model_dir):
    path = os.path.join(model_dir, "config.json")
    with open(path) as stream:
        config = json.load(stream)
    config["num_experts_per_tok"] = 6
    with open(path, "w") as stream:
        json.dump(config, stream)
    return model_dir


@pytest.mark.parametrize("victims, passes", [(35, True), (36, False), (64, False)])
def test_victim_lanes_stay_below_the_verify_routes(top_k_6_model_dir, victims, passes):
    """Block size 5 verifies 6 tokens x top-k 6 = 36 routes; _init_insert_direct refuses V >= 36 after the load, so the
    gate refuses it first."""
    launch = _exl3_launch(top_k_6_model_dir, speculative_dspark_block_size=5, **DSPARK_BREAKABLE)
    env = {**CPU_EXPERTS_ENV, **SPILL, "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": victims}
    if passes:
        _gate(launch, **env)
    else:
        with pytest.raises(ValueError, match="below the 36 routes"):
            _gate(launch, **env)


def test_victim_lanes_are_unbounded_when_the_top_k_is_unknown(model_dir):
    """A config.json without num_experts_per_tok gives the gate no top-k; it bounds nothing rather than guess."""
    launch = _exl3_launch(model_dir, speculative_dspark_block_size=5, **DSPARK_BREAKABLE)
    _gate(launch, **{**CPU_EXPERTS_ENV, **SPILL, "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": 64})


def test_the_recipe_budget_a_passes_with_both_cpu_expert_clients(top_k_6_model_dir, cpu_pin_off):
    """benchmarks/dsv41_baseline/arm_env.py dspark_env(): block size 5 (6 tokens x top-k 6 = 36 lanes), 8 victim lanes,
    the target's and the draft's CPU experts together on the optimized build."""
    launch = _exl3_launch(top_k_6_model_dir, speculative_dspark_block_size=5, **DSPARK_BREAKABLE)
    _gate(
        launch,
        **{
            **CPU_EXPERTS_ENV,
            **SPILL,
            "SGLANG_RAGGED_VERIFY_MODE": "static",
            "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": True,
            "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": "",
            "SGLANG_EXL3_CPU_ACT_RESIDUAL": True,
            "SGLANG_EXL3_CPU_ACT_BLOCK": 128,
        },
    )


def test_row_weighted_assignment_needs_the_cpu_experts_it_names(model_dir, cpu_pin_off):
    """The option weights the EXL3 kernel's tiles for the draft's or the target's layers; naming a source whose CPU
    experts are off would log a change no kernel runs."""
    draft = dict(DSPARK_CPU_ENV, SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT="draft")
    _gate(_launch(model_dir, speculative_algorithm="DSPARK"), **draft)
    with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT=target needs SGLANG_DSV41_CPU_EXPERTS"):
        _gate(_launch(model_dir, speculative_algorithm="DSPARK"), **{**draft, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "target"})
    target = dict(CPU_EXPERTS_ENV, SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT="target")
    _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **target)
    with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT=both needs SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**target, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "both"})
    with pytest.raises(ValueError, match="SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT"):
        _gate(_launch(model_dir, cuda_graph_config=BREAKABLE_BS1), **{**target, "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT": "1"})


def test_ram_prefetch_options_default_off_at_the_replays_budget():
    """Off until a served A/B is accepted; one candidate per token, one row per layer (DSV41_REFERENCE.md 33.13)."""
    assert envs.SGLANG_DSV41_RAM_PREFETCH.get() is False
    assert envs.SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN.get() == 1
    assert envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.get() == 1
    assert envs.SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE.get() == 2


def test_ram_prefetch_needs_cpu_experts(model_dir):
    """Only a record with a CPU lane stages the input the scorer reads, and only a forced CPU miss uses the pool."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True)
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS=1"):
        _gate(args, SGLANG_DSV41_RAM_PREFETCH=True)


@pytest.mark.parametrize(
    "name, value",
    [
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", 0),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", 13),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", 0),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", 9),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", 0),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", 5),
    ],
)
def test_ram_prefetch_options_outside_the_hosts_bounds_are_refused(model_dir, name, value):
    """Refused at launch, not at the service's start, where the host would refuse the same bound."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    with pytest.raises(ValueError, match=f"{name} must be in"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True, **{name: value})


def test_the_ram_prefetch_scorer_defaults_to_the_cpu():
    """The GPU scorer is opt-in until its A/B is accepted (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design)."""
    assert envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get() == "cpu"


def test_the_gpu_scorer_runs_on_the_prefetch_and_the_captured_copy_engine_post(model_dir):
    """gpu needs the prefetch it feeds, and the copy-engine post whose capture the scoring kernels follow."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu needs SGLANG_DSV41_RAM_PREFETCH=1"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")
    no_copy_engine = {**CPU_EXPERTS_ENV, "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": False}
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu runs after the copy-engine post"):
        _gate(args, **no_copy_engine, SGLANG_DSV41_RAM_PREFETCH=True, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")


@pytest.mark.parametrize("prefetch", [False, True])
def test_an_unknown_ram_prefetch_scorer_is_refused_whether_or_not_the_prefetch_is_on(model_dir, prefetch):
    """A typo must not fall back to the CPU scorer silently, nor pass while the prefetch is off."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER must be one of"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=prefetch, SGLANG_DSV41_RAM_PREFETCH_SCORER="tpu")


def test_margin_floors_need_the_gpu_scorer_and_a_readable_file(model_dir, tmp_path):
    """The floors filter the GPU scorer's candidate page; the CPU scorer would ignore them silently."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    path = tmp_path / "floors.json"
    path.write_text('{"min_margin": {"1": 0.25}}')
    gpu = dict(**CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")
    _gate(args, **gpu, SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS=str(path))
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS needs SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True, SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS=str(path))
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"):
        _gate(args, **gpu, SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS=str(tmp_path / "missing.json"))


THREE_ROOTS = "/mnt/nvme0/dsv41_flash:/mnt/nvme4/dsv41_flash:/mnt/nvme2/dsv41_flash"


def test_the_dynamic_mirror_root_choice_defaults_off():
    """Off until its served A/B is accepted (design 2026-10-09-dsv41-drive-aware-reads, change (3))."""
    assert envs.SGLANG_MOE_EXPERT_MIRROR_DYNAMIC.get() is False
    assert envs.SGLANG_MOE_EXPERT_MIRROR_CAPS.get() == ""


def test_dynamic_mirror_caps_pass_with_one_positive_cap_per_root(model_dir):
    args = _exl3_launch(model_dir)
    _gate(args, SGLANG_MOE_EXPERT_MIRROR_DIRS=THREE_ROOTS, SGLANG_MOE_EXPERT_MIRROR_DYNAMIC=True,
          SGLANG_MOE_EXPERT_MIRROR_CAPS="4,2,4")
    _gate(args, SGLANG_MOE_EXPERT_MIRROR_DIRS=THREE_ROOTS)  # static: no caps needed


@pytest.mark.parametrize(
    "env, match",
    [
        ({"SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": True}, "needs SGLANG_MOE_EXPERT_MIRROR_CAPS"),
        ({"SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": True, "SGLANG_MOE_EXPERT_MIRROR_CAPS": "4,2"}, "lists 2 caps but"),
        ({"SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": True, "SGLANG_MOE_EXPERT_MIRROR_CAPS": "4,0,4"}, "positive integer"),
        ({"SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": True, "SGLANG_MOE_EXPERT_MIRROR_CAPS": "4,x,4"}, "positive integer"),
        ({"SGLANG_MOE_EXPERT_MIRROR_DYNAMIC": True, "SGLANG_MOE_EXPERT_MIRROR_CAPS": "4,-1,4"}, "positive integer"),
        ({"SGLANG_MOE_EXPERT_MIRROR_CAPS": "4,2,4"}, "silently ignored"),
    ],
)
def test_bad_dynamic_mirror_caps_are_refused_at_launch(model_dir, env, match):
    args = _exl3_launch(model_dir)
    with pytest.raises(ValueError, match=match):
        _gate(args, SGLANG_MOE_EXPERT_MIRROR_DIRS=THREE_ROOTS, **env)


def test_dynamic_mirror_caps_need_mirror_roots(model_dir):
    args = _exl3_launch(model_dir)
    with pytest.raises(ValueError, match="needs SGLANG_MOE_EXPERT_MIRROR_DIRS"):
        _gate(args, SGLANG_MOE_EXPERT_MIRROR_DYNAMIC=True, SGLANG_MOE_EXPERT_MIRROR_CAPS="4")


def test_the_mirror_caps_parser():
    from sglang.srt.layers.moe.exl3_read_split import mirror_caps

    assert mirror_caps(False, "", THREE_ROOTS) is None
    assert mirror_caps(True, " 4, 2 ,4 ", THREE_ROOTS) == (4, 2, 4)
    with pytest.raises(ValueError, match="more than 4"):
        mirror_caps(True, "1,1,1,1,1", ":".join(f"/r{i}" for i in range(5)))
