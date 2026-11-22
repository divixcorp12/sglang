"""Dsv41Config resolves the DSV41 knobs afresh on every call, so envs.X.override() is observed."""

import os

import msgspec
import pytest

from sglang.srt.dsv41_config import Dsv41Config
from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_engine.config import ExpertStreamConfig
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
        engram_verify_in_graph=True,
        ram_miss_fault="",
        torch_prefill_indexer=False,
        torch_prefill_indexer_score_budget_mb=0,
        fused_wo_a=True,
    )


# The DSV41-prefixed knobs that are not Dsv41Config fields: the DSV4.1 profile's own (Dsv41ProfileConfig), and the
# Engram native-cache size, which its table reads directly.
PROFILE_OR_DIRECT_KNOBS = {
    "SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM",
    "SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH",
    "SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS",
    "SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD",
    "SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH",
    "SGLANG_DSV41_ENGRAM_NATIVE_CACHE_GIB",
}


def test_one_field_per_knob():
    declared = {
        name
        for name in vars(type(envs))
        if name.startswith(("SGLANG_DSV41_", "SGLANG_ENABLE_DSV41_", "SGLANG_TEST_DSV41_"))
    }
    assert PROFILE_OR_DIRECT_KNOBS <= declared
    assert len(declared - PROFILE_OR_DIRECT_KNOBS) == len(msgspec.structs.fields(Dsv41Config))


def test_the_config_is_frozen():
    cfg = Dsv41Config.from_envs()
    with pytest.raises(AttributeError):
        cfg.engram_host_table = True


def test_from_envs_observes_an_override_and_reverts_after():
    assert Dsv41Config.from_envs().engram_ram_gib == 0.0
    with envs.SGLANG_DSV41_ENGRAM_RAM_GIB.override(2.0):
        assert Dsv41Config.from_envs().engram_ram_gib == 2.0
        with envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("per_rank"):
            inner = Dsv41Config.from_envs()
            assert inner.engram_host_table_layout == "per_rank" and inner.engram_ram_gib == 2.0
        assert Dsv41Config.from_envs().engram_host_table_layout == "shared"
    assert Dsv41Config.from_envs().engram_ram_gib == 0.0


def test_the_stream_config_observes_an_override_and_reverts_after():
    """The expert-streaming knobs moved to ExpertStreamConfig (F145): it resolves afresh too."""
    assert ExpertStreamConfig.from_envs().copy.timeout_ms == 2000
    with envs.SGLANG_EXPERT_STREAM_COPY_MODE.override("copy_engine"):
        assert ExpertStreamConfig.from_envs().copy.mode.value == "copy_engine"
        with envs.SGLANG_EXPERT_STREAM_RAM_MISS_TIMEOUT_MS.override(40_000):
            inner = ExpertStreamConfig.from_envs()
            assert inner.copy.timeout_ms == 40_000 and inner.copy.mode.value == "copy_engine"
        assert ExpertStreamConfig.from_envs().copy.timeout_ms == 2000
    assert ExpertStreamConfig.from_envs().copy.mode.value == "sm_copy"


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
    assert not {
        "ram_miss_pack_workers", "enable_ram_miss_row_images", "enable_ram_miss_leases", "enable_ram_miss_two_phase",
        "enable_ram_miss_piece_stream", "enable_expert_prefetch", "enable_native_prefetch",
    } & fields
    Dsv41Config.from_envs()  # the set values do not break the config a service starts from


RETIRED_ALWAYS_ON_KNOBS = (
    "SGLANG_DSV41_ENABLE_LAYER_FUSION",
    "SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION",
    "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_FILLS",
    "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_ROUTE_PLAN",
    "SGLANG_EXPERT_STREAM_ENABLE_PREFILL_SPLIT_GATHER",
    "SGLANG_EXPERT_STREAM_ENABLE_LEASE_PDL",
)
# Their pre-rename spellings: F145 refuses them at startup, now with no replacement.
REFUSED_PRE_RENAME_KNOBS = (
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS",
    "SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN",
    "SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER",
    "SGLANG_DSV41_ENABLE_LEASE_PDL",
)


@pytest.mark.parametrize("value", ["0", "1"])
def test_retired_always_on_knobs_warn_and_are_ignored(monkeypatch, value):
    """Six switches production always set are unconditional: a launch that still sets one, to either value, warns and
    starts, and no config field reads it."""
    import warnings

    from sglang.srt import environ
    from sglang.srt.layers.moe.expert_engine import removed_env

    for name in RETIRED_ALWAYS_ON_KNOBS:
        monkeypatch.setenv(name, value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        environ._handle_deprecated_envs()
    text = " ".join(str(w.message) for w in caught)
    for name in RETIRED_ALWAYS_ON_KNOBS:
        assert f"{name} is deprecated" in text, name
        assert not hasattr(environ.envs, name), name
        assert environ._DEPRECATED_ENVS[name].replacement is None, name
    assert "Always on" in text
    removed_env.check_removed_env(dict(os.environ))  # warned, not refused
    assert ExpertStreamConfig.from_envs() == ExpertStreamConfig()
    fields = {f.name for f in msgspec.structs.fields(Dsv41Config)}
    assert not {"enable_layer_fusion", "enable_exl3_cast_fusion"} & fields


@pytest.mark.parametrize("name", REFUSED_PRE_RENAME_KNOBS)
def test_the_pre_rename_spellings_still_refuse_startup(name):
    from sglang.srt import environ
    from sglang.srt.layers.moe.expert_engine import removed_env

    assert name not in environ._DEPRECATED_ENVS
    assert removed_env.REMOVED_ENV[name].env == removed_env.NO_REPLACEMENT
    with pytest.raises(removed_env.RemovedEnvError, match=f"{name} \\(removed, no replacement\\): always on"):
        removed_env.check_removed_env({name: "1"})


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
