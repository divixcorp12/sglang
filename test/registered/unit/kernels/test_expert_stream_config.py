"""The expert engine's config (F145): env to struct resolution, group envs, the copy-mode merge, the removed-name table,
and the production recipe against the base commit's resolved values.

CPU only. ``golden/expert_stream_base_config.json`` is the resolved value of every in-scope env name at the base commit
570958cb71 (`benchmarks/dsv41_baseline/parity/config_snapshot.py` run on that tree), for a clean env and for the
base / dspark / prod recipes; it is the only source of "today's" values in this file.
"""

import enum
import json
import os
import sys
import typing
from pathlib import Path

import msgspec
import pytest

from sglang.srt.environ import InsertOnMissStage, Envs, EnvField
from sglang.srt.layers.moe.expert_engine import config as cfg
from sglang.srt.layers.moe.expert_engine import removed_env as rem
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
FIXTURE = json.loads((Path(__file__).parent / "golden" / "expert_stream_base_config.json").read_text())
sys.path.insert(0, str(REPO / "benchmarks" / "dsv41_baseline"))

NEW_PREFIXES = ("SGLANG_EXPERT_STREAM_", "SGLANG_EXL3_MOE_")
OLD_PREFIXES = ("SGLANG_DSV41_", "SGLANG_MOE_", "SGLANG_EXL3_", "SGLANG_EXPERT_", "SGLANG_DRAFT_", "SGLANG_CPU_EXPERT_",
                "SGLANG_DSPARK_DEBUG", "EXL3_MOE_", "NVFP4_CPU_")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(NEW_PREFIXES + OLD_PREFIXES):
            monkeypatch.delenv(key)


def _set(monkeypatch, env: dict):
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _path(obj, dotted: str):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _builtin(value):
    """A config value as plain data: enums by value, tuples as lists."""
    return json.loads(msgspec.json.encode(value))


# --- defaults --------------------------------------------------------------------------------------------------


def test_clean_env_resolves_to_the_struct_defaults():
    # Every Envs default equals the struct default, so a launch that sets nothing gets the same config as no env.
    assert cfg.ExpertStreamConfig.from_envs() == cfg.ExpertStreamConfig()
    assert cfg.Dsv41ProfileConfig.from_envs() == cfg.Dsv41ProfileConfig()
    assert cfg.Exl3FormatConfig.from_envs() == cfg.Exl3FormatConfig()
    assert cfg.MimoV26ProfileConfig.from_envs() == cfg.MimoV26ProfileConfig()


def _old_to_builtin(path: str, old_value):
    """The base commit's resolved value for a field, as the config's plain-data form."""
    annotation = rem._field_type(path)
    if annotation is InsertOnMissStage:
        # The base resolves a default as the member's name and a set value as the plain int.
        return int(InsertOnMissStage[old_value]) if isinstance(old_value, str) else old_value
    if typing.get_origin(annotation) is tuple:
        return [int(x) if annotation == tuple[int, ...] else x for x in old_value]
    return old_value


SPECIAL_OLD = {  # merged or reshaped: checked by their own tests
    "SGLANG_MOE_EXPERT_STREAM", "SGLANG_DSV41_EXPERT_STREAM", "SGLANG_MOE_EXPERT_COPY_BACKEND",
    "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE", "SGLANG_EXPERT_NUMA_CORES", "SGLANG_DSV41_CPU_EXPERTS_CORES",
    "SGLANG_DSV41_RAM_MISS_SPIN_CORE", "SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU", "SGLANG_MOE_EXPERT_PREFETCH_PULL",
    "SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE", "SGLANG_MOE_HOT_INSERT_ON_MISS", "SGLANG_DSV41_EXPERT_RAM_GIB",
}


def test_defaults_equal_the_base_commits_envs_defaults():
    config = _builtin(cfg.ExpertStreamConfig())
    checked = 0
    for old, value in FIXTURE["defaults"].items():
        if old in SPECIAL_OLD or old not in rem.REMOVED_ENV:
            continue
        field = rem.REMOVED_ENV[old].field
        if field.startswith("exl3."):
            continue
        actual = config
        for part in field.split("."):
            actual = actual[part]
        assert actual == _old_to_builtin(field, value), old
        checked += 1
    assert checked > 100


def test_pull_mode_default_is_the_legacy_default():
    assert FIXTURE["defaults"]["SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE"] == "off"
    assert cfg.ShadowPrefetchConfig().pull_mode is cfg.PullMode.OFF


# --- every env name fills its field -------------------------------------------------------------------------------


def _alt(path: str, annotation, default):
    """A valid value, in env spelling, different from the default."""
    forced = {
        "copy.mode": "dma", "copy.ram_hit_copy": "sm", "prefetch.ram.scorer": "cpu",
        "residency.insert_on_miss_stage": "2", "tiers.pinned_host_full_layers": "3,5",
    }
    if path in forced:
        return forced[path]
    if annotation is bool:
        return "0" if default else "1"
    if annotation in (int, typing.Optional[int]):
        return str((default or 0) + 7)
    if annotation is float:
        return str(default + 0.5)
    return "alt"


def _env_alt_expected(path: str, annotation, text: str):
    if path == "copy.mode":
        return cfg.CopyMode.DMA
    if path == "residency.insert_on_miss_stage":
        return InsertOnMissStage.DIRECT
    if path == "tiers.pinned_host_full_layers":
        return (3, 5)
    if annotation is bool:
        return text == "1"
    if annotation in (int, typing.Optional[int]):
        return int(text)
    if annotation is float:
        return float(text)
    return text


INDIVIDUAL = {env: path for env, path in cfg.env_fields().items() if env not in rem.GROUP_PATHS}


@pytest.mark.parametrize("env_name", sorted(INDIVIDUAL))
def test_each_individual_env_fills_its_field(monkeypatch, env_name):
    path = INDIVIDUAL[env_name]
    annotation = rem._field_type(path)
    default = _path(cfg.ExpertStreamConfig(), path)
    text = _alt(path, annotation, default)
    monkeypatch.setenv(env_name, text)
    config = cfg.ExpertStreamConfig.from_envs()
    assert _path(config, path) == _env_alt_expected(path, annotation, text)
    # nothing else moved
    changed = [k for k in _builtin(config) if _builtin(config)[k] != _builtin(cfg.ExpertStreamConfig())[k]]
    assert changed == [path.split(".")[0]]


def test_new_env_names_are_registered_with_the_config_default():
    for env_name, path in INDIVIDUAL.items():
        field = Envs.__dict__[env_name]
        assert isinstance(field, EnvField), env_name
        default = _path(cfg.ExpertStreamConfig(), path)
        if isinstance(default, enum.Enum):
            default = default.value if not isinstance(default, enum.IntEnum) else default
        if isinstance(default, tuple):
            default = tuple(str(x) for x in default)
        assert field.default == default, env_name


# --- group envs ------------------------------------------------------------------------------------------------


def test_group_env_decodes_into_its_struct(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_URING", '{"mode": "sqpoll", "queue_depth": 256, "slab_arena": true}')
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_TRACE", '{"job": {"prefix": "/tmp/j"}, "draft_trigger": {"forward_us": 5000}}')
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", '{"predictor": "apex", "pull_mode": "always", "predictors": ["a", "b"]}')
    config = cfg.ExpertStreamConfig.from_envs()
    assert (config.tiers.uring.mode, config.tiers.uring.queue_depth, config.tiers.uring.slab_arena) == ("sqpoll", 256, True)
    assert config.tiers.uring.read_cuts == "auto"  # a key the object leaves out keeps today's default
    assert (config.trace.job.prefix, config.trace.job.capacity, config.trace.draft_trigger.forward_us) == ("/tmp/j", 131072, 5000)
    assert config.prefetch.shadow.pull_mode is cfg.PullMode.ALWAYS
    assert config.prefetch.shadow.predictors == ("a", "b")


@pytest.mark.parametrize(
    "env_name,payload,key",
    [
        ("SGLANG_EXPERT_STREAM_URING", '{"mod": "sqpoll"}', "mod"),
        ("SGLANG_EXPERT_STREAM_TRACE", '{"job": {"prefx": "/x"}}', "prefx"),
        ("SGLANG_EXPERT_STREAM_MIRROR", '{"dirs": "a", "bogus": 1}', "bogus"),
        ("SGLANG_EXPERT_STREAM_CPU_SPLIT", '{"cpu_msec": 1.0}', "cpu_msec"),
        ("SGLANG_EXPERT_STREAM_CPU_TWO_STAGE", '{"w3_weights": "1"}', "w3_weights"),
        ("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", '{"predictors": [], "pull": "off"}', "pull"),
    ],
)
def test_group_env_rejects_an_unknown_key_naming_env_and_key(monkeypatch, env_name, payload, key):
    monkeypatch.setenv(env_name, payload)
    with pytest.raises(ValueError) as error:
        cfg.ExpertStreamConfig.from_envs()
    assert env_name in str(error.value) and key in str(error.value)


@pytest.mark.parametrize(
    "env_name,payload",
    [
        ("SGLANG_EXPERT_STREAM_URING", '{"mode": "turbo"}'),
        ("SGLANG_EXPERT_STREAM_URING", '{"queue_depth": "many"}'),
        ("SGLANG_EXPERT_STREAM_TRACE", "not json"),
        ("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", '{"pull_mode": "sometimes", "predictor": "apex"}'),
        ("SGLANG_EXPERT_STREAM_SHADOW_PREFETCH", '{"pull_mode": "always"}'),  # a pull mode needs a predictor
    ],
)
def test_group_env_rejects_a_bad_value_naming_the_env(monkeypatch, env_name, payload):
    monkeypatch.setenv(env_name, payload)
    with pytest.raises(ValueError, match=env_name):
        cfg.ExpertStreamConfig.from_envs()


def test_every_group_member_is_a_key_of_its_group():
    # The removed table sends each old group member to <group>.<field path>; the key must exist in the struct.
    for old, target in rem.REMOVED_ENV.items():
        if target.env in rem.GROUP_PATHS:
            assert rem._field_type(target.field) is not None, old


# --- copy mode -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend,engine,mode",
    [("gpu", False, "sm_copy"), ("dma", False, "dma"), ("gpu", True, "copy_engine"), ("dma", True, "copy_engine")],
)
def test_copy_mode_reproduces_the_old_backend_and_engine_pairs(monkeypatch, backend, engine, mode):
    assert rem.legacy_copy_mode(backend, engine) == mode
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_COPY_MODE", mode)
    assert cfg.CopyConfig.from_envs().mode is cfg.CopyMode(mode)


def test_copy_mode_default_is_the_old_defaults_pair():
    assert FIXTURE["defaults"]["SGLANG_MOE_EXPERT_COPY_BACKEND"] == "gpu"
    assert FIXTURE["defaults"]["SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE"] is False
    assert cfg.CopyConfig().mode is cfg.CopyMode(rem.legacy_copy_mode("gpu", False))


def test_copy_mode_rejects_an_unknown_value_naming_the_env(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERT_STREAM_COPY_MODE", "gpu")
    with pytest.raises(ValueError, match="SGLANG_EXPERT_STREAM_COPY_MODE.*sm_copy"):
        cfg.CopyConfig.from_envs()


def test_old_names_for_the_same_mode_translate_to_one_name():
    new = rem.translate_env({"SGLANG_MOE_EXPERT_COPY_BACKEND": "dma"}, lambda cpu: 0)
    assert new == {"SGLANG_EXPERT_STREAM_COPY_MODE": "dma"}
    new = rem.translate_env(
        {"SGLANG_MOE_EXPERT_COPY_BACKEND": "gpu", "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": "1"}, lambda cpu: 0
    )
    assert new == {"SGLANG_EXPERT_STREAM_COPY_MODE": "copy_engine"}


# --- watchdog formula ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("timeout_ms", [100, 2000, 10000, 20000])
def test_copy_timeouts_are_the_formula_the_service_wrote_inline(timeout_ms):
    # max(30 s, 3T), 2T + 1 s, T + 5 s and watchdog + 5 s, as exl3_ram_miss.py spelled them in three places.
    copy = cfg.CopyConfig(timeout_ms=timeout_ms)
    assert copy.watchdog_wait_s() == max(30.0, 3.0 * timeout_ms / 1000)
    assert copy.eager_pause_bound_s() == 2 * timeout_ms / 1000 + 1.0
    assert copy.completion_deadline_s() == timeout_ms / 1000 + 5.0
    assert copy.fill_timeout_s() == max(30.0, 3.0 * timeout_ms / 1000) + 5.0


def test_watchdog_outlasts_the_wait_and_the_pause_bound():
    for timeout_ms in range(1, 60001, 997):
        copy = cfg.CopyConfig(timeout_ms=timeout_ms)
        assert copy.watchdog_wait_s() > copy.eager_pause_bound_s()
        assert copy.watchdog_wait_s() > timeout_ms / 1000


# --- removed names ---------------------------------------------------------------------------------------------


def test_a_clean_env_passes_the_removed_name_check():
    rem.check_removed_env({})
    rem.check_removed_env({"SGLANG_EXPERT_STREAM_ENABLE": "1", "SGLANG_EXPERT_STREAM_URING": "{}"})


def test_the_retired_hit_wait_is_not_a_removed_name():
    # It keeps warning and being ignored (environ._DEPRECATED_ENVS); the recipe just stops setting it.
    assert "SGLANG_DSV41_RAM_MISS_HIT_WAIT_US" not in rem.REMOVED_ENV
    rem.check_removed_env({"SGLANG_DSV41_RAM_MISS_HIT_WAIT_US": "100"})


@pytest.mark.parametrize("old", sorted(rem.REMOVED_ENV))
def test_each_removed_name_raises_naming_its_replacement(old):
    with pytest.raises(rem.RemovedEnvError) as error:
        rem.check_removed_env({old: "1", "UNRELATED": "x"})
    assert old in str(error.value) and rem.REMOVED_ENV[old].env in str(error.value)


def test_every_set_removed_name_is_listed_at_once():
    with pytest.raises(rem.RemovedEnvError) as error:
        rem.check_removed_env({"SGLANG_MOE_HOT_GPU_MB": "1", "EXL3_MOE_CPU_PIN": "0", "SGLANG_DRAFT_FORWARD_TRIGGER_US": "9"})
    message = str(error.value)
    assert "3 removed" in message
    assert "SGLANG_EXPERT_STREAM_HOT_GPU_MB" in message and "SGLANG_EXL3_MOE_CPU_PIN" in message
    assert """SGLANG_EXPERT_STREAM_TRACE='{"draft_trigger": {"forward_us": <value>}}'""" in message


def test_a_vendored_name_the_engine_set_is_not_refused_but_any_other_still_is():
    """exl3/ext.py sets EXL3_MOE_CPU_PIN for the vendored kernel and lists it in the marker env; a child that inherits
    it must start, while the same name set by a launch script (no marker) or a name the marker does not list is refused."""
    marked = {rem.VENDORED_ENV_MARKER: "EXL3_MOE_CPU_PIN,EXL3_MOE_CPU_MAX_ISA", "EXL3_MOE_CPU_PIN": "0"}
    rem.check_removed_env({**marked, "EXL3_MOE_CPU_MAX_ISA": "avx2"})
    with pytest.raises(rem.RemovedEnvError, match="EXL3_MOE_CPU_SMALL_WORKERS"):
        rem.check_removed_env({**marked, "EXL3_MOE_CPU_SMALL_WORKERS": "2"})
    with pytest.raises(rem.RemovedEnvError, match="EXL3_MOE_CPU_PIN"):
        rem.check_removed_env({"EXL3_MOE_CPU_PIN": "0"})


def test_the_vendored_kernel_env_ext_sets_passes_the_removed_name_check(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import ext

    # A private copy of the environment, so nothing the engine sets leaks into other tests.
    private = {
        k: v
        for k, v in os.environ.items()
        if k not in ("EXL3_MOE_CPU_PIN", "EXL3_MOE_CPU_SMALL_WORKERS", "EXL3_MOE_CPU_MAX_ISA", rem.VENDORED_ENV_MARKER)
    }
    monkeypatch.setattr(os, "environ", private)
    config = cfg.ResolvedConfig(
        stream=cfg.ExpertStreamConfig(),
        dsv41=cfg.Dsv41ProfileConfig(),
        exl3=cfg.Exl3FormatConfig(cpu=cfg.Exl3CpuConfig(pin="0", small_workers=2)),
    )
    ext.set_vendored_kernel_env(config)
    assert os.environ["EXL3_MOE_CPU_PIN"] == "0" and os.environ["EXL3_MOE_CPU_SMALL_WORKERS"] == "2"
    assert "EXL3_MOE_CPU_MAX_ISA" not in os.environ, "an unconfigured cap leaves the kernel's default"
    rem.check_removed_env(dict(os.environ))  # what a child process sees
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = "avx2"  # set by the user, not the engine
    with pytest.raises(rem.RemovedEnvError, match="EXL3_MOE_CPU_MAX_ISA"):
        rem.check_removed_env(dict(os.environ))


def _resolves(field: str) -> bool:
    if field.startswith("exl3."):
        root, rest = cfg.Exl3FormatConfig, field[len("exl3."):]
    else:
        root, rest = cfg.ExpertStreamConfig, field
    current = root
    for part in rest.split("."):
        types = {info.name: info.type for info in msgspec.structs.fields(current)}
        if part not in types:
            return False
        current = types[part]
    return True


def test_every_replacement_is_a_real_env_and_config_field():
    for old, target in rem.REMOVED_ENV.items():
        assert target.env in Envs.__dict__ and isinstance(Envs.__dict__[target.env], EnvField), old
        assert _resolves(target.field), old
        assert target.env.startswith(NEW_PREFIXES), old


def test_the_new_envs_are_exactly_the_individual_names_plus_the_groups():
    reachable = set(cfg.env_fields())
    assert set(rem.GROUP_PATHS) <= reachable
    # No old name is read by the config.
    assert not reachable & set(rem.REMOVED_ENV)


# --- the production recipe against the base commit ------------------------------------------------------------


def _expected_config(recipe: str):
    """The config the base commit's resolved values mean, built from the fixture only."""
    resolved = FIXTURE["recipes"][recipe]["resolved"]
    raw = FIXTURE["recipes"][recipe]["raw"]
    flat = _builtin(cfg.ExpertStreamConfig())
    exl3 = _builtin(cfg.Exl3FormatConfig())
    dsv41 = _builtin(cfg.Dsv41ProfileConfig())

    def put(tree, field, value):
        parts = field.split(".")
        for part in parts[:-1]:
            tree = tree[part]
        tree[parts[-1]] = value

    for old, value in resolved.items():
        if old in SPECIAL_OLD or old not in rem.REMOVED_ENV:
            continue
        field = rem.REMOVED_ENV[old].field
        if field.startswith("exl3."):
            put(exl3, field[len("exl3."):], value)
        else:
            put(flat, field, _old_to_builtin(field, value))
    flat["enabled"] = resolved["SGLANG_MOE_EXPERT_STREAM"] or resolved["SGLANG_DSV41_EXPERT_STREAM"]
    flat["copy"]["mode"] = rem.legacy_copy_mode(
        resolved["SGLANG_MOE_EXPERT_COPY_BACKEND"], resolved["SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE"]
    )
    # pull_mode: the recipe sets neither PULL nor PULL_MODE; the base resolves "off"
    flat["prefetch"]["shadow"]["pull_mode"] = resolved["SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE"]
    import arm_env

    cores = {}
    for old, key in (("SGLANG_DSV41_CPU_EXPERTS_CORES", "cpu"), ("SGLANG_DSV41_RAM_MISS_SPIN_CORE", "ram")):
        if resolved[old] not in ("", None):
            cores.setdefault(arm_env.node_of_cpu(int(str(resolved[old]).replace("-", ",").split(",")[0])), {})[key] = str(resolved[old])
    flat["tiers"]["numa_cores"] = ";".join(
        f"{node}:" + ",".join(f"{k}={plan[k]}" for k in ("ram", "cpu") if k in plan) for node, plan in sorted(cores.items())
    ) or resolved["SGLANG_EXPERT_NUMA_CORES"]
    for old, name in (
        ("SGLANG_DSV41_ENABLE_MOE_SIDE_STREAM", "moe_side_stream"),
    ):
        dsv41[name] = resolved[old]
    dsv41["ram_prefetch"]["margin_floors_path"] = resolved["SGLANG_DSV41_RAM_PREFETCH_MARGIN_FLOORS"]
    dsv41["ram_prefetch"]["learned_head_path"] = resolved["SGLANG_DSV41_RAM_PREFETCH_LEARNED_HEAD"]
    dsv41["dspark_draft_graph"] = not resolved["SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH"]
    dsv41["trace"]["draft_top_prob_path"] = resolved["SGLANG_DSV41_DRAFT_TOP_PROB_LOG_PATH"]
    for field, old in (("src", "SGLANG_EXL3_SRC"), ("dir", "SGLANG_EXL3_BUILD_DIR"), ("cxx", "SGLANG_EXL3_CPU_CXX")):
        exl3["build"][field] = resolved[old]
    for field, old in (("act_residual", "SGLANG_EXL3_CPU_ACT_RESIDUAL"), ("act_block", "SGLANG_EXL3_CPU_ACT_BLOCK"),
                       ("max_m", "SGLANG_EXL3_CPU_MAX_M"), ("row_weighted", "SGLANG_EXL3_CPU_ROW_WEIGHTED_ASSIGNMENT")):
        exl3["cpu"][field] = resolved[old]
    exl3["cpu"]["pin"] = raw.get("EXL3_MOE_CPU_PIN")
    return flat, dsv41, exl3


@pytest.mark.parametrize("recipe", ["base", "dspark", "prod"])
def test_the_recipe_under_the_new_names_resolves_to_the_base_commits_settings(monkeypatch, recipe):
    import arm_env

    new_env = {"base": arm_env.base_env(), "dspark": arm_env.arm_env(arm_env.dspark_env()), "prod": arm_env.prod_env()}[recipe]
    rem.check_removed_env(new_env)  # the recipe sets no removed name
    assert "SGLANG_DSV41_RAM_MISS_HIT_WAIT_US" not in new_env
    _set(monkeypatch, {k: v for k, v in new_env.items() if k.startswith(("SGLANG_", "EXL3_"))})
    flat, dsv41, exl3 = _expected_config(recipe)
    assert _builtin(cfg.ExpertStreamConfig.from_envs()) == flat
    assert _builtin(cfg.Dsv41ProfileConfig.from_envs()) == dsv41
    assert _builtin(cfg.Exl3FormatConfig.from_envs()) == exl3


def test_the_recipe_still_means_what_the_fixture_froze():
    # The fixture's raw env is the base commit's recipe under the removed names. Translated to the new names it must
    # still be what today's arm_env sets on the in-scope names, or the test above would compare two different recipes.
    import arm_env

    now = {"base": arm_env.base_env(), "dspark": arm_env.arm_env(arm_env.dspark_env()), "prod": arm_env.prod_env()}
    sizes = ("SGLANG_EXPERT_STREAM_PINNED_HOST_MB", "SGLANG_EXPERT_STREAM_PINNED_HOST_NUMA_MB",
             "SGLANG_EXPERT_STREAM_HOT_GPU_MB")
    for recipe, frozen in FIXTURE["recipes"].items():
        for key, value in arm_env.new_names_env(frozen["raw"]).items():
            if key in sizes:
                continue  # the recipe's sizes move with the hardware; the resolved comparison above is what counts
            if key in rem.GROUP_PATHS:
                assert json.loads(now[recipe][key]) == json.loads(value), (recipe, key)
            else:
                assert now[recipe].get(key) == value, (recipe, key)


def test_the_prod_recipe_group_and_merged_values():
    import arm_env

    new = arm_env.prod_env()
    assert new["SGLANG_EXPERT_STREAM_NUMA_CORES"] == "0:ram=26"
    assert new["SGLANG_EXPERT_STREAM_COPY_MODE"] == "copy_engine"
    assert new["SGLANG_EXPERT_STREAM_ENABLE"] == "1"
    assert json.loads(new["SGLANG_EXPERT_STREAM_CPU_SPLIT"]) == {
        "model": True, "model_costs": "", "model_refine_batches": 0, "calibration": True, "miss_cut": 1, "miss_cut_max": 3,
    }
    assert json.loads(new["SGLANG_EXPERT_STREAM_MIRROR"]) == {"dirs": arm_env.EXPERT_MIRROR_DIRS}


# --- engine modules read no env ---------------------------------------------------------------------------------


def test_no_engine_module_but_config_reads_the_env():
    package = REPO / "python/sglang/srt/layers/moe/expert_engine"
    # removed_env.py is the startup check itself: it reads the process env by definition.
    exempt = {"config.py", "removed_env.py"}
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if path.name in exempt:
            continue
        text = path.read_text()
        if "envs." in text or "os.environ" in text or "os.getenv" in text:
            offenders.append(str(path.relative_to(package)))
    assert offenders == []
