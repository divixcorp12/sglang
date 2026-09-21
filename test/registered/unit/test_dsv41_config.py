"""Dsv41Config re-reads the environment on every from_envs() call."""

from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_from_envs_defaults():
    assert Dsv41Config.from_envs() == Dsv41Config(
        reasoning_effort=None,
        engram_host_table=False,
        engram_host_table_layout="shared",
        engram_table_dir="",
        engram_ram_gib=0.0,
        expert_stream=False,
        expert_dir="",
        expert_trace_path="",
        ram_miss_timeout_ms=2000,
        ram_miss_fault="",
        enable_expert_prefetch=False,
        torch_prefill_indexer=False,
        fused_wo_a=True,
    )


def test_from_envs_observes_overrides_and_their_exit():
    before = Dsv41Config.from_envs()
    with (
        envs.SGLANG_DSV41_EXPERT_STREAM.override(True),
        envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(40_000),
        envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override("3:1.5"),
        envs.SGLANG_DSV41_REASONING_EFFORT.override("high"),
    ):
        inside = Dsv41Config.from_envs()
        assert inside.expert_stream is True
        assert inside.ram_miss_timeout_ms == 40_000
        assert inside.ram_miss_fault == "3:1.5"
        assert inside.reasoning_effort == "high"
    assert Dsv41Config.from_envs() == before
