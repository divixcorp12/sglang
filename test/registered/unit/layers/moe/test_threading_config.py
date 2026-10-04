"""ThreadingConfig on fake sysfs trees: the design's derivation, today's one-node runtime, the override and every
refusal (spec 2026-10-03-numa-node-distributor-design, Part 2)."""

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
        NodePlan(group=0, node=0, ram=17, cpu=tuple(range(8, 16)), sq=None, busy_poll=True),
        NodePlan(group=1, node=1, ram=35, cpu=tuple(range(18, 34)), sq=None, busy_poll=True),
    )
    assert config.copy_cpus == tuple(sorted(SERVER))
    assert config.log_lines() == [
        "numa node0: ram=17 cpu=8-15 (8) sq=-",
        "numa node1: ram=35 cpu=18-33 (16) sq=-",
        "numa copy thread: node0 cpus=0-7,16,36-52",
    ]


def test_without_a_threads_cap_the_engine_takes_every_remaining_core(divix01):
    config = resolve(divix01, cpu_experts=True)
    assert config.plans[1].cpu == tuple(range(18, 35)) and config.plans[1].threads == 17


def test_sqpoll_takes_the_next_usable_core_below_the_ram_core(divix01):
    config = resolve(divix01, cpu_experts=True, sqpoll=True)
    assert [(p.ram, p.sq, p.cpu[0], p.cpu[-1]) for p in config.plans] == [(17, 15, 8, 14), (35, 34, 18, 33)]


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
    assert (config.plans[0].ram, config.plans[0].sq) == (17, 15), "node 0 is still derived"


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
        ({"omp_thread_limit": 16}, "OMP_THREAD_LIMIT"),
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
    monkeypatch.setenv("OMP_THREAD_LIMIT", "24")
    with envs.SGLANG_DSV41_CPU_EXPERTS_CORES.override("18-29"), \
            envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.override(4), \
            envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.override(17), \
            envs.SGLANG_EXPERT_STREAM_URING_MODE.override("sqpoll_iopoll"), \
            envs.SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU.override(15), \
            envs.SGLANG_EXPERT_NUMA_CORES.override("1:ram=35,cpu=18-33"), \
            envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.override("1:1024,0:1024"):
        assert ThreadingConfig.from_env(cpu_experts=True, device=None) == "resolved"
    assert seen == {
        "nodes": [1, 0],
        "gpu_node": None,
        "affinity": frozenset({0, 1}),
        "topology": "topology",
        "settings": CoreSettings(
            cpu_experts=True, cores="18-29", threads=4, spin_core=17, sqpoll=True, sq_thread_cpu=15,
            numa_cores="1:ram=35,cpu=18-33", omp_thread_limit=24,
        ),
    }
