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
        ram_miss_fault="",
        enable_ram_miss_copy_engine=False,
        enable_ram_miss_sm_small_copies=False,
        enable_prefill_share=False,
        enable_moe_side_stream=False,
        torch_prefill_indexer=False,
        fused_wo_a=True,
    )


def test_one_field_per_knob():
    declared = {
        name
        for name in vars(type(envs))
        if name.startswith(
            ("SGLANG_DSV41_", "SGLANG_ENABLE_DSV41_", "SGLANG_TEST_DSV41_")
        )
    }
    assert len(declared) == len(msgspec.structs.fields(Dsv41Config))


def test_the_config_is_frozen():
    cfg = Dsv41Config.from_envs()
    with pytest.raises(AttributeError):
        cfg.expert_stream = True


def test_from_envs_observes_an_override_and_reverts_after():
    assert Dsv41Config.from_envs().enable_ram_miss_copy_engine is False
    with envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.override(True):
        assert Dsv41Config.from_envs().enable_ram_miss_copy_engine is True
        with envs.SGLANG_DSV41_RAM_MISS_TIMEOUT_MS.override(40_000):
            inner = Dsv41Config.from_envs()
            assert (
                inner.ram_miss_timeout_ms == 40_000
                and inner.enable_ram_miss_copy_engine is True
            )
        assert Dsv41Config.from_envs().ram_miss_timeout_ms == 2000
    assert Dsv41Config.from_envs().enable_ram_miss_copy_engine is False


def test_from_envs_maps_the_renamed_fields():
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_TEST_DSV41_RAM_MISS_FAULT.override("3:0.5"),
    ):
        cfg = Dsv41Config.from_envs()
    assert cfg.engram_host_table is True
    assert cfg.ram_miss_fault == "3:0.5"


REMOVED_RAM_MISS_KNOBS = (
    "SGLANG_DSV41_RAM_MISS_PACK_WORKERS",
    "SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES",
    "SGLANG_DSV41_ENABLE_RAM_MISS_LEASES",
    "SGLANG_DSV41_ENABLE_RAM_MISS_TWO_PHASE",
    "SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM",
    "SGLANG_DSV41_ENABLE_EXPERT_PREFETCH",
    "SGLANG_DSV41_ENABLE_NATIVE_PREFETCH",
)


def test_removed_ram_miss_knobs_warn(monkeypatch):
    """The packed path's knobs, the lease switch and the lease chain's modes are gone (2026-09-29): a launch that still
    sets one (every archived arm does) warns, starts, and gets the one protocol whatever value it set."""
    import warnings

    from sglang.srt import environ

    monkeypatch.setenv("SGLANG_DSV41_RAM_MISS_PACK_WORKERS", "8")
    monkeypatch.setenv("SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES", "0")
    monkeypatch.setenv("SGLANG_DSV41_ENABLE_RAM_MISS_LEASES", "0")
    for name in REMOVED_RAM_MISS_KNOBS[3:]:
        monkeypatch.setenv(name, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        environ._handle_deprecated_envs()
    text = " ".join(str(w.message) for w in caught)
    for name in REMOVED_RAM_MISS_KNOBS:
        assert f"{name} is deprecated" in text, name
        assert not hasattr(environ.envs, name), name
        # Warn-only: nothing forwards the value anywhere, so it is ignored.
        assert environ._DEPRECATED_ENVS[name].replacement is None, name
    assert "build_row_images.py" in text and "lease mode is always on" in text
    fields = {f.name for f in msgspec.structs.fields(Dsv41Config)}
    assert (
        not {
            "ram_miss_pack_workers",
            "enable_ram_miss_row_images",
            "enable_ram_miss_leases",
            "enable_ram_miss_two_phase",
            "enable_ram_miss_piece_stream",
            "enable_expert_prefetch",
            "enable_native_prefetch",
        }
        & fields
    )
    Dsv41Config.from_envs()  # the set values do not break the config a service starts from


ALWAYS_ON_KNOBS = (
    "SGLANG_DSV41_ENABLE_LAYER_FUSION",
    "SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION",
    "SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN",
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS",
    "SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER",
    "SGLANG_DSV41_ENABLE_LEASE_PDL",
)


def test_always_on_knobs_are_deprecated_without_replacement(monkeypatch):
    """The layer fusion, cast fusion, prefill route plan, fills and split gather are always on, and the lease chain's PDL
    follows the GPU: a launch that still sets one warns, starts, and the value is ignored."""
    import warnings

    from sglang.srt import environ

    for name in ALWAYS_ON_KNOBS:
        monkeypatch.setenv(name, "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        environ._handle_deprecated_envs()
    text = " ".join(str(w.message) for w in caught)
    fields = {f.name for f in msgspec.structs.fields(Dsv41Config)}
    for name in ALWAYS_ON_KNOBS:
        assert name in environ._DEPRECATED_ENVS, name
        assert environ._DEPRECATED_ENVS[name].replacement is None, name
        assert not hasattr(environ.envs, name), name
        assert f"{name} is deprecated" in text, name
    assert (
        not {
            "enable_layer_fusion",
            "enable_exl3_cast_fusion",
            "enable_prefill_route_plan",
            "enable_prefill_fills",
            "enable_prefill_split_gather",
            "enable_lease_pdl",
        }
        & fields
    )
    Dsv41Config.from_envs()  # the set values do not break the config a service starts from


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
