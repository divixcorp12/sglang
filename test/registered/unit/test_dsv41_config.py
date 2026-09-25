"""Dsv41Config resolves the DSV41 knobs afresh on every call, so envs.X.override() is observed."""

import msgspec
import pytest

from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_defaults_match_the_env_declarations():
    cfg = Dsv41Config.from_envs()
    assert cfg == Dsv41Config(
        reasoning_effort=None,
        engram_host_table=False,
        engram_host_table_layout="shared",
        engram_table_dir="",
        engram_ram_gib=0.0,
        engram_host_node_cache_uring=False,
        enable_engram_device_wait=False,
        expert_stream=False,
        expert_dir="",
        expert_trace_path="",
        router_capture_path="",
        ram_miss_timeout_ms=2000,
        ram_miss_pack_workers=0,
        ram_miss_fault="",
        enable_expert_prefetch=False,
        enable_ram_miss_leases=False,
        enable_ram_miss_two_phase=False,
        ram_miss_hit_wait_us=100,
        enable_ram_miss_piece_stream=False,
        enable_ram_miss_row_images=False,
        enable_ram_miss_copy_engine=False,
        enable_native_prefetch=False,
        enable_prefill_fills=False,
        enable_prefill_share=False,
        enable_moe_side_stream=False,
        enable_layer_fusion=False,
        enable_exl3_cast_fusion=False,
        torch_prefill_indexer=False,
        fused_wo_a=True,
    )


def test_one_field_per_knob():
    declared = {
        name
        for name in vars(type(envs))
        if name.startswith(("SGLANG_DSV41_", "SGLANG_ENABLE_DSV41_", "SGLANG_TEST_DSV41_"))
    }
    assert len(declared) == len(msgspec.structs.fields(Dsv41Config))


def test_the_config_is_frozen():
    cfg = Dsv41Config.from_envs()
    with pytest.raises(AttributeError):
        cfg.expert_stream = True


def test_from_envs_observes_an_override_and_reverts_after():
    assert Dsv41Config.from_envs().enable_ram_miss_leases is False
    with envs.SGLANG_DSV41_ENABLE_RAM_MISS_LEASES.override(True):
        assert Dsv41Config.from_envs().enable_ram_miss_leases is True
        with envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(40_000):
            inner = Dsv41Config.from_envs()
            assert inner.ram_miss_timeout_ms == 40_000 and inner.enable_ram_miss_leases is True
        assert Dsv41Config.from_envs().ram_miss_timeout_ms == 2000
    assert Dsv41Config.from_envs().enable_ram_miss_leases is False


def test_from_envs_maps_the_renamed_fields():
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override("3:0.5"),
        envs.SGLANG_DSV41_RAM_MISS_PACK_WORKERS.override(4),
    ):
        cfg = Dsv41Config.from_envs()
    assert cfg.engram_host_table is True
    assert cfg.ram_miss_fault == "3:0.5"
    assert cfg.ram_miss_pack_workers == 4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
