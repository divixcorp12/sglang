"""ThreadingConfig on fake sysfs trees: the design's derivation, today's one-node runtime, the override and every
refusal (spec 2026-10-03-numa-node-distributor-design, Part 2)."""

import logging

import pytest

from sglang.srt.layers.moe.cpu_experts import threading_config as tc
from sglang.srt.layers.moe.cpu_experts.threading_config import (
    CoreSettings,
    NodePlan,
    ThreadingConfig,
    Topology,
    check_engine_cores,
    check_not_reserved,
    parse_cpu_list,
    parse_numa_cores,
    pci_numa_node,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DIVIX01 = {0: "0-17,36-53", 1: "18-35,54-71"}
SERVER = frozenset(parse_cpu_list("0-7,16,36-52"))


def fake_sysfs(root, nodes, pairs):
    """A /sys/devices/system tree: each node's cpulist, with cpu c and c + pairs one physical core (c < pairs)."""
    for node, cpus in nodes.items():
        node_dir = root / "node" / f"node{node}"
        node_dir.mkdir(parents=True)
        (node_dir / "cpulist").write_text(cpus + "\n")
        for cpu in parse_cpu_list(cpus):
            topology = root / "cpu" / f"cpu{cpu}" / "topology"
            topology.mkdir(parents=True)
            low = cpu % pairs
            (topology / "thread_siblings_list").write_text(f"{low},{low + pairs}\n")
    return Topology.from_sysfs(str(root))


@pytest.fixture
def divix01(tmp_path):
    return fake_sysfs(tmp_path, DIVIX01, 36)


def resolve(topology, *, nodes=(0, 1), affinity=SERVER, gpu_node=0, **settings):
    return ThreadingConfig.resolve(
        nodes=list(nodes), gpu_node=gpu_node, affinity=affinity, topology=topology, settings=CoreSettings(**settings)
    )


def test_divix01_derives_the_designs_plan(divix01):
    config = resolve(divix01, cpu_experts=True, threads=16)
    assert config.plans == (
        NodePlan(group=0, node=0, ram=17, cpu=tuple(range(8, 15)), sq=None, busy_poll=True),
        NodePlan(group=1, node=1, ram=35, cpu=tuple(range(18, 34)), sq=None, busy_poll=True),
    )
    assert config.copy_cpus == (15,)
    assert config.log_lines() == [
        "numa node0: ram=17 cpu=8-14 (7) sq=-",
        "numa node1: ram=35 cpu=18-33 (16) sq=-",
        "numa copy thread: node0 cpus=15",
    ]


def test_the_copy_thread_takes_a_shared_core_and_leaves_the_free_one_to_the_ram_thread(divix01):
    """Node 0's only core whose sibling is outside the server's affinity is 17, which the busy-polling RAM thread
    needs; the copy thread, which spins with PAUSE, takes 15 (sibling 51 is the server's). Mutant: drop the `shared`
    preference in _copy_core -- red (node 0 has no core for its RAM thread)."""
    config = resolve(divix01, cpu_experts=True)
    assert config.copy_cpus == (15,) and config.plans[0].ram == 17
    assert 15 not in config.plans[0].cpu


def test_without_a_threads_cap_the_engine_takes_every_remaining_core(divix01):
    config = resolve(divix01, cpu_experts=True)
    assert config.plans[1].cpu == tuple(range(18, 35)) and config.plans[1].threads == 17


def test_sqpoll_takes_the_next_usable_core_below_the_ram_core(divix01):
    config = resolve(divix01, cpu_experts=True, sqpoll=True)
    assert [(p.ram, p.sq, p.cpu[0], p.cpu[-1]) for p in config.plans] == [(17, 14, 8, 13), (35, 34, 18, 33)]


def test_two_nodes_without_cpu_experts_pin_only_the_ram_threads(divix01):
    config = resolve(divix01, spin_core=17)
    assert [(p.ram, p.cpu, p.busy_poll) for p in config.plans] == [(17, (), True), (35, (), True)]


def test_one_node_without_numa_or_cpu_experts_is_todays_runtime(divix01):
    everything = frozenset(range(72))
    config = resolve(divix01, nodes=(0,), affinity=everything)
    assert config.plans == (NodePlan(group=0, node=0, ram=None, cpu=(), sq=None, busy_poll=False),)
    assert config.copy_cpus == ()
    pinned = resolve(divix01, nodes=(0,), affinity=everything, spin_core=17, sq_thread_cpu=15, sqpoll=True)
    assert pinned.plans[0] == NodePlan(group=0, node=0, ram=17, cpu=(), sq=15, busy_poll=True)


def test_a_reserved_sq_core_is_refused_without_the_derivation(divix01):
    with pytest.raises(ValueError, match="64-71"):
        resolve(divix01, nodes=(0,), affinity=frozenset(range(72)), sqpoll=True, sq_thread_cpu=70)


def test_the_override_replaces_the_plans_of_the_nodes_it_names(divix01):
    config = resolve(divix01, cpu_experts=True, sqpoll=True, numa_cores="1:ram=34,cpu=18-25,27,sq=33")
    assert config.plans[1] == NodePlan(
        group=1, node=1, ram=34, cpu=(18, 19, 20, 21, 22, 23, 24, 25, 27), sq=33, busy_poll=True
    )
    assert (config.plans[0].ram, config.plans[0].sq) == (17, 14), "node 0 is still derived"


def test_cpu_experts_cores_is_the_cpu_override_of_its_node(divix01):
    config = resolve(divix01, cpu_experts=True, cores="20-23")
    assert config.plans[1].cpu == (20, 21, 22, 23) and config.plans[1].ram == 35


def test_a_server_affinity_covering_a_node_is_refused(divix01):
    """Review Focus 4: the server's affinity takes all of node 1. Start is refused naming the node; nothing falls back
    to node 0's cores or to an unpinned thread."""
    with pytest.raises(ValueError, match="node 1 has no usable core"):
        resolve(divix01, affinity=SERVER | frozenset(parse_cpu_list("18-35,54-71")), cpu_experts=True)


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"numa_cores": "1:ram=35,cpu=18-33", "affinity": SERVER | {20}}, "core 20 is in the server's affinity"),
        ({"numa_cores": "1:ram=64,cpu=18-33"}, "core 64 is reserved"),
        ({"numa_cores": "1:ram=35,cpu=18"}, "at least 2 cores"),
        ({"numa_cores": "1:ram=17,cpu=18-33"}, "core 17 is on node 0"),
        ({"numa_cores": "1:ram=35,cpu=18-33,54"}, "cores 18 and 54 share a physical core"),
        ({"numa_cores": "2:ram=35,cpu=18-33"}, "node 2 is not one of the tier's nodes"),
        ({"numa_cores": "1:ram=35,cpu=18-33", "cores": "20-23"}, "both"),
        ({"cores": "10-11,20-21"}, "two nodes"),
        ({"cores": "18-29", "nodes": (0,)}, "core 18 is on node 1"),
        ({"spin_core": 8}, "core 8 shares a physical core with the server's affinity"),
        ({"affinity": SERVER | frozenset(range(8, 18))}, "node 0 has no core .* for the copy thread"),
        ({"numa_cores": "1:ram=35,cpu=18-33,sq=34"}, "node 1 sets an SQPOLL core, but .* is not sqpoll"),
        (
            {"numa_cores": "1:ram=34,cpu=18-33", "spin_core": 35},
            "node 1's RAM core is set both by SGLANG_DSV41_RAM_MISS_SPIN_CORE and SGLANG_EXPERT_NUMA_CORES",
        ),
        (
            {"numa_cores": "0:sq=15,cpu=8-14", "nodes": (0,), "sq_thread_cpu": 14},
            "node 0's SQPOLL core is set both by SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU and SGLANG_EXPERT_NUMA_CORES",
        ),
    ],
)
def test_each_refusal(divix01, settings, match):
    settings = dict(settings)
    affinity = settings.pop("affinity", SERVER)
    nodes = settings.pop("nodes", (0, 1))
    with pytest.raises(ValueError, match=match):
        resolve(divix01, nodes=nodes, affinity=affinity, cpu_experts=True, **settings)


def test_threads_one_is_refused_in_words_that_say_what_the_cap_did(divix01):
    """THREADS truncates the node's core list, and an engine needs two cores. Mutation: the bare engine-core refusal
    ("CPU experts need at least 2 cores, got [..]") is raised, which never mentions the setting."""
    with pytest.raises(ValueError, match=r"^THREADS=1 leaves one core for node 0's engine; CPU experts need at least 2$"):
        resolve(divix01, cpu_experts=True, threads=1)


def test_an_engine_core_refusal_names_the_node(divix01):
    """Mutation: check_engine_cores is called unwrapped, so the refusal does not say which node's plan failed."""
    with pytest.raises(ValueError, match=r"^node 1: CPU experts need at least 2 cores"):
        resolve(divix01, cpu_experts=True, numa_cores="1:ram=35,cpu=18")


@pytest.mark.parametrize(
    "spec, entry",
    [("1:ram=-1", "ram=-1"), ("1:ram=35,cpu=a-b", "cpu=a-b"), ("1:ram=35,cpu=18-x", "cpu=18-x")],
)
def test_a_malformed_core_list_names_the_variable_and_the_entry(spec, entry):
    """Mutation: parse_cpu_list's bare `invalid literal for int()` escapes."""
    with pytest.raises(ValueError, match=r"SGLANG_EXPERT_NUMA_CORES") as caught:
        parse_numa_cores(spec)
    assert entry in str(caught.value) and "invalid literal" not in str(caught.value)


@pytest.mark.parametrize(
    "spec, match",
    [
        ("1:ram=35;1:ram=34", "twice"),
        ("x:ram=1", "node"),
        ("1:foo=3", "unknown key"),
        ("1:ram=34-35", "one core"),
        ("1:18-33", "key"),
    ],
)
def test_a_malformed_override_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_numa_cores(spec)


def test_the_override_parses_lists_inside_a_key():
    assert parse_numa_cores("1:ram=35,cpu=18-20,22,sq=34; 0:ram=17") == {
        1: {"ram": [35], "cpu": [18, 19, 20, 22], "sq": [34]},
        0: {"ram": [17]},
    }


def test_the_shared_core_checks():
    assert parse_cpu_list("36-38, 40,36") == [36, 37, 38, 40]
    with pytest.raises(ValueError, match="backwards"):
        parse_cpu_list("9-3")
    with pytest.raises(ValueError, match="64-71"):
        check_not_reserved(71)
    check_not_reserved(63)
    with pytest.raises(ValueError, match="at least 2 cores"):
        check_engine_cores([4, 4], 1)
    with pytest.raises(ValueError, match="3 CPU expert threads on 2 cores"):
        check_engine_cores([4, 5], 3)


def test_the_gpus_node_comes_from_its_pci_device(tmp_path):
    (tmp_path / "0000:41:00.0").mkdir()
    (tmp_path / "0000:41:00.0" / "numa_node").write_text("1\n")
    (tmp_path / "0000:01:00.0").mkdir()
    (tmp_path / "0000:01:00.0" / "numa_node").write_text("-1\n")
    assert pci_numa_node("0000:41:00.0", str(tmp_path)) == 1
    assert pci_numa_node("0000:01:00.0", str(tmp_path)) is None, "-1: the platform reports no node"


def test_from_env_reads_every_core_setting(monkeypatch):
    from sglang.srt.environ import envs

    seen = {}
    monkeypatch.setattr(ThreadingConfig, "resolve", staticmethod(lambda **kw: seen.update(kw) or "resolved"))
    monkeypatch.setattr(Topology, "from_sysfs", classmethod(lambda cls, root=tc.SYSFS: "topology"))
    monkeypatch.setattr(tc, "_affinity", lambda: frozenset({0, 1}))
    monkeypatch.setattr(tc, "gpu_numa_node", lambda device: None)
    with envs.SGLANG_DSV41_CPU_EXPERTS_CORES.override("18-29"), \
            envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.override(4), \
            envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.override(17), \
            envs.SGLANG_EXPERT_STREAM_URING_MODE.override("sqpoll_iopoll"), \
            envs.SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU.override(15), \
            envs.SGLANG_EXPERT_NUMA_CORES.override("1:ram=35,cpu=18-33"), \
            envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("1:1024,0:1024"), \
            envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS.override(True), \
            envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override("6-11"), \
            envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.override(3), \
            envs.SGLANG_DSV41_RAM_PREFETCH.override(True):
        assert ThreadingConfig.from_env(cpu_experts=True, device=None) == "resolved"
    assert seen == {
        "nodes": [1, 0],
        "gpu_node": None,
        "affinity": frozenset({0, 1}),
        "topology": "topology",
        "settings": CoreSettings(
            cpu_experts=True, cores="18-29", threads=4, spin_core=17, sqpoll=True, sq_thread_cpu=15,
            numa_cores="1:ram=35,cpu=18-33", draft=True, draft_cores="6-11", draft_threads=3, ram_prefetch=True,
        ),
    }


# The recipe's server cores (benchmarks/dsv41_baseline/arm_env.py SERVER_CORES): node 0's 6-17 and their siblings free.
RECIPE_SERVER = frozenset(parse_cpu_list("0-5,36-41"))


def test_the_draft_takes_the_gpu_nodes_cores_left_after_the_copy_and_ram_threads(divix01):
    """Only the server's affinity and the draft switch are given; every thread's core is derived. Mutant: derive the
    draft before the node plans -- red (the draft takes 16 and 17's neighbours, the RAM thread lands lower)."""
    config = resolve(divix01, affinity=RECIPE_SERVER, draft=True)
    assert config.copy_cpus == (17,)
    assert [p.ram for p in config.plans] == [16, 35]
    assert config.draft_cpus == tuple(range(6, 16))
    assert config.log_lines()[-1] == "numa dspark draft: node0 cpus=6-15 (10)"


def test_the_draft_threads_cap_takes_the_lowest_free_cores(divix01):
    config = resolve(divix01, affinity=RECIPE_SERVER, draft=True, draft_threads=4)
    assert config.draft_cpus == (6, 7, 8, 9)


def test_a_one_node_tier_with_the_draft_on_is_derived(divix01):
    """Without the draft this tier is today's underived runtime; with it, the draft's cores must be kept off the copy
    and RAM threads, so the plan is derived."""
    config = resolve(divix01, nodes=(0,), affinity=RECIPE_SERVER, draft=True)
    assert (config.copy_cpus, config.plans[0].ram, config.draft_cpus) == ((17,), 16, tuple(range(6, 16)))


def test_without_the_draft_there_are_no_draft_cores(divix01):
    assert resolve(divix01, affinity=RECIPE_SERVER, spin_core=17).draft_cpus == ()


def test_named_draft_cores_are_left_out_of_every_derived_role(divix01):
    """D2-3's graphed arm: the draft held 6-17 by name and the copy and RAM threads were placed among them. Named draft
    cores are taken first, so the derived threads go below them. Mutant: leave the draft out of `taken` -- red."""
    config = resolve(divix01, affinity=RECIPE_SERVER, draft=True, draft_cores="8-17")
    assert config.copy_cpus == (7,) and config.plans[0].ram == 6
    assert config.draft_cpus == tuple(range(8, 18))


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"spin_core": 17}, "core 17 is named both for the DSpark draft"),
        ({"spin_core": 17, "draft_cores": "6-16,53"}, "core 17 is named both for the DSpark draft"),
        ({"numa_cores": "0:ram=12"}, "core 12 is named both for the DSpark draft"),
    ],
)
def test_a_given_role_on_a_named_draft_core_is_refused(divix01, settings, match):
    settings = {"draft_cores": "6-17", **settings}
    with pytest.raises(ValueError, match=match):
        resolve(divix01, affinity=RECIPE_SERVER, draft=True, **settings)


def test_too_few_cores_left_for_the_draft_is_refused(divix01):
    with pytest.raises(ValueError, match=r"node 0: \[15\] is left for the DSpark draft's CPU experts"):
        resolve(divix01, affinity=RECIPE_SERVER | frozenset(range(6, 15)), draft=True)


def test_under_cpu_experts_the_draft_has_no_cores_of_its_own(divix01):
    """One team per node: with the target's CPU experts on, the draft is a job source on node 0's team, so the plan
    derives no draft role and node 0's team keeps every free core (the recipe's 6-15)."""
    config = resolve(divix01, affinity=RECIPE_SERVER, cpu_experts=True, threads=10, spin_core=17, draft=True)
    assert config.draft_cpus == ()
    assert [(p.ram, p.cpu) for p in config.plans] == [(17, tuple(range(6, 16))), (35, tuple(range(18, 28)))]
    assert config.copy_cpus == (16,)


@pytest.mark.parametrize("settings", [{"draft_cores": "12-15"}, {"draft_threads": 4}])
def test_under_cpu_experts_named_draft_cores_are_refused(divix01, settings):
    with pytest.raises(ValueError, match="shares the GPU node's CPU expert team"):
        resolve(divix01, affinity=RECIPE_SERVER, cpu_experts=True, threads=10, spin_core=17, draft=True, **settings)


def test_a_draft_only_launch_still_derives_the_draft_cores(divix01):
    config = resolve(divix01, affinity=RECIPE_SERVER, draft=True)
    assert config.draft_cpus == tuple(range(6, 16))


def test_ram_prefetch_takes_each_nodes_spare_affinity_cores_or_its_ram_core(divix01):
    """Spare: the server's affinity on the node less every assigned core and its SMT sibling (CPU experts 8-14 take
    44-50 with them, the copy core 15 takes 51). Node 1 has no affinity core, so its thread shares the RAM core."""
    config = resolve(divix01, cpu_experts=True, threads=16, ram_prefetch=True)
    assert config.plans[0].spec == (*range(0, 8), 16, *range(36, 44), 52)
    assert config.plans[1].spec == (35,)
    assert config.log_lines()[:2] == [
        "numa node0: ram=17 cpu=8-14 (7) sq=- spec=0-7,16,36-43,52",
        "numa node1: ram=35 cpu=18-33 (16) sq=- spec=35",
    ]


def test_without_ram_prefetch_no_plan_names_spec_cores(divix01):
    config = resolve(divix01, cpu_experts=True, threads=16)
    assert [plan.spec for plan in config.plans] == [(), ()]
    assert "spec=" not in " ".join(config.log_lines())


def test_a_spec_thread_sharing_the_ram_core_is_warned_naming_the_node_and_core(divix01, caplog):
    """Node 1 has no spare core, so its speculative thread time-slices the busy-polling RAM core 35; node 0 has spares."""
    with caplog.at_level(logging.WARNING, logger=tc.__name__):
        resolve(divix01, cpu_experts=True, threads=16, ram_prefetch=True)
    warnings = [r.getMessage() for r in caplog.records if r.name == tc.__name__ and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "node1" in warnings[0] and "core 35" in warnings[0] and "node0" not in warnings[0]


def test_ram_prefetch_without_a_shared_core_warns_nothing(divix01, caplog):
    with caplog.at_level(logging.WARNING, logger=tc.__name__):
        resolve(divix01, nodes=(0,), cpu_experts=True, threads=16, ram_prefetch=True)
        resolve(divix01, cpu_experts=True, threads=16)
    assert [r for r in caplog.records if r.name == tc.__name__] == []


def test_ram_prefetch_spec_cores_leave_out_the_reserved_cores(divix01):
    """An affinity spanning 64-71: node 1's only spare would be a reserved core, so its thread shares the RAM core."""
    config = resolve(divix01, affinity=SERVER | set(range(64, 72)), cpu_experts=True, threads=16, ram_prefetch=True)
    assert config.plans[1].spec == (config.plans[1].ram,)
    assert not set(config.plans[0].spec) & tc.RESERVED_CORES
