"""Where the expert stream's threads run: one plan per NUMA node of the pinned tier.

ThreadingConfig is the only reader of the core settings (SGLANG_DSV41_CPU_EXPERTS_CORES and _THREADS,
SGLANG_DSV41_RAM_MISS_SPIN_CORE, SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU, SGLANG_EXPERT_NUMA_CORES), of the process
affinity and of the reserved cores; C++ receives resolved core lists only. The rules are the design's
(docs/superpowers/specs/2026-10-03-numa-node-distributor-design.md, Part 2). ``resolve`` reads nothing, so it runs
on any topology; ``from_env`` gathers the machine's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence

# NVMe completion interrupts are pinned to these cores on divix01.
RESERVED_CORES = frozenset(range(64, 72))
SYSFS = "/sys/devices/system"
PCI_DEVICES = "/sys/bus/pci/devices"


def parse_cpu_list(spec: str) -> list[int]:
    """Cores from a sysfs or taskset list such as "0-17,36-53", sorted and unique."""
    cores: set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        first, last = int(lo), int(hi) if sep else int(lo)
        if last < first:
            raise ValueError(f"core range {part!r} runs backwards")
        cores.update(range(first, last + 1))
    return sorted(cores)


def _format_cpus(cores: Sequence[int]) -> str:
    """The inverse of parse_cpu_list: "8-15", "0-7,16,36-52", or "-" for none."""
    runs: list[list[int]] = []
    for core in sorted(cores):
        if runs and core == runs[-1][1] + 1:
            runs[-1][1] = core
        else:
            runs.append([core, core])
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in runs) or "-"


def parse_numa_cores(spec: str) -> dict[int, dict[str, list[int]]]:
    """SGLANG_EXPERT_NUMA_CORES as {node: {"ram" | "cpu" | "sq": cores}}; raises ValueError when malformed."""
    plans: dict[int, dict[str, list[int]]] = {}
    for entry in filter(None, (e.strip() for e in spec.split(";"))):
        node_text, sep, body = entry.partition(":")
        if not sep or not node_text.strip().isdigit():
            raise ValueError(f"SGLANG_EXPERT_NUMA_CORES entry {entry!r} does not start with a node, as in 1:ram=35")
        node = int(node_text)
        if node in plans:
            raise ValueError(f"SGLANG_EXPERT_NUMA_CORES names node {node} twice")
        items: dict[str, list[str]] = {}
        key = None
        for item in (i.strip() for i in body.split(",")):
            name, eq, value = item.partition("=")
            if eq:
                key = name.strip()
                if key not in ("ram", "cpu", "sq"):
                    raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: unknown key {key!r}; expected ram, cpu or sq")
                if key in items:
                    raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: node {node} sets {key} twice")
                items[key] = [value]
            elif key is None:
                raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: {item!r} on node {node} follows no key")
            else:
                items[key].append(item)
        plans[node] = {}
        for k, v in items.items():
            try:
                plans[node][k] = parse_cpu_list(",".join(v))
            except ValueError as refusal:
                raise ValueError(
                    f"SGLANG_EXPERT_NUMA_CORES: node {node}'s entry {k}={','.join(v)} is not a taskset core list"
                ) from refusal
        for k in ("ram", "sq"):
            if k in plans[node] and len(plans[node][k]) != 1:
                raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: node {node}'s {k} is one core")
    return plans


def check_not_reserved(core: int) -> None:
    """Raise ValueError for a core in RESERVED_CORES."""
    if core in RESERVED_CORES:
        raise ValueError(f"core {core} is reserved: cores 64-71 take NVMe completion interrupts")


def check_engine_cores(cores: Sequence[int], threads: int) -> None:
    """Raise ValueError unless a CPU expert engine can run ``threads`` workers on ``cores``."""
    distinct = sorted(set(cores))
    if len(distinct) < 2:
        # Spinning workers sharing one core livelock (DSV41_REFERENCE.md section 28.2).
        raise ValueError(f"CPU experts need at least 2 cores, got {distinct}")
    if not 1 <= threads <= len(distinct):
        raise ValueError(f"{threads} CPU expert threads on {len(distinct)} cores")


def pci_numa_node(bus_id: str, root: str = PCI_DEVICES) -> Optional[int]:
    """The NUMA node of PCI device ``bus_id`` ("0000:41:00.0"), None when the platform reports none (-1)."""
    with open(os.path.join(root, bus_id, "numa_node")) as f:
        node = int(f.read())
    return node if node >= 0 else None


def gpu_numa_node(device: Optional[int]) -> Optional[int]:
    """The NUMA node of CUDA device ``device``; None without a device or a reported node."""
    if device is None:
        return None
    import torch

    p = torch.cuda.get_device_properties(device)
    return pci_numa_node(f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0")


def _affinity() -> frozenset[int]:
    return frozenset(os.sched_getaffinity(0))


@dataclass(frozen=True)
class Topology:
    node_cpus: Mapping[int, tuple[int, ...]]  # node -> its CPUs, ascending
    siblings: Mapping[int, frozenset[int]]  # CPU -> its SMT siblings, itself included

    @classmethod
    def from_sysfs(cls, root: str = SYSFS) -> "Topology":
        node_dir = os.path.join(root, "node")
        node_cpus: dict[int, tuple[int, ...]] = {}
        for name in sorted(os.listdir(node_dir)):
            if name.startswith("node") and name[4:].isdigit():
                with open(os.path.join(node_dir, name, "cpulist")) as f:
                    node_cpus[int(name[4:])] = tuple(parse_cpu_list(f.read()))
        siblings = {}
        for cpus in node_cpus.values():
            for cpu in cpus:
                with open(os.path.join(root, "cpu", f"cpu{cpu}", "topology", "thread_siblings_list")) as f:
                    siblings[cpu] = frozenset(parse_cpu_list(f.read()))
        return cls(node_cpus, siblings)

    def node_of(self, cpu: int) -> Optional[int]:
        return next((node for node, cpus in self.node_cpus.items() if cpu in cpus), None)

    def physical(self, node: int) -> list[int]:
        """The node's physical cores, each named by its lowest-numbered SMT thread."""
        return [cpu for cpu in self.node_cpus[node] if min(self.siblings[cpu]) == cpu]


@dataclass(frozen=True)
class CoreSettings:
    cpu_experts: bool = False  # SGLANG_DSV41_CPU_EXPERTS
    cores: str = ""  # SGLANG_DSV41_CPU_EXPERTS_CORES
    threads: int = 0  # SGLANG_DSV41_CPU_EXPERTS_THREADS, 0: no cap
    spin_core: Optional[int] = None  # SGLANG_DSV41_RAM_MISS_SPIN_CORE
    sqpoll: bool = False  # SGLANG_EXPERT_STREAM_URING_MODE names a sqpoll mode
    sq_thread_cpu: Optional[int] = None  # SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU
    numa_cores: str = ""  # SGLANG_EXPERT_NUMA_CORES
    omp_thread_limit: Optional[int] = None  # OMP_THREAD_LIMIT


@dataclass(frozen=True)
class NodePlan:
    group: int  # the wire's node axis: home(expert) == group
    node: int  # the NUMA node id
    ram: Optional[int]  # the RAM/NVMe thread's core; None inherits the server's affinity
    cpu: tuple[int, ...]  # the CPU expert engine's cores, cpu[0] its thread (worker 0); () without CPU experts
    sq: Optional[int]  # the io_uring SQPOLL thread's core; None unpinned
    busy_poll: bool

    @property
    def threads(self) -> int:
        return len(self.cpu)

    def cores(self) -> list[int]:
        return [c for c in (self.ram, self.sq) if c is not None] + list(self.cpu)

    def log_line(self) -> str:
        ram = "-" if self.ram is None else str(self.ram)
        sq = "-" if self.sq is None else str(self.sq)
        return f"numa node{self.node}: ram={ram} cpu={_format_cpus(self.cpu)} ({self.threads}) sq={sq}"


@dataclass(frozen=True)
class ThreadingConfig:
    plans: tuple[NodePlan, ...]  # one per node of the tier, in placement order
    copy_cpus: tuple[int, ...]  # the copy thread's core, which it spins on; () inherits the server's affinity
    gpu_node: int

    @property
    def nodes(self) -> int:
        return len(self.plans)

    def log_lines(self) -> list[str]:
        lines = [plan.log_line() for plan in self.plans]
        if self.copy_cpus:
            lines.append(f"numa copy thread: node{self.gpu_node} cpus={_format_cpus(self.copy_cpus)}")
        return lines

    @classmethod
    def from_env(cls, *, cpu_experts: bool, device: Optional[int]) -> "ThreadingConfig":
        """The machine's plan: its sysfs topology, this process's affinity and the core env vars."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.host_numa import parse_placement

        limit = os.environ.get("OMP_THREAD_LIMIT")  # OpenMP's own variable, outside Envs
        settings = CoreSettings(
            cpu_experts=cpu_experts,
            cores=envs.SGLANG_DSV41_CPU_EXPERTS_CORES.get(),
            threads=envs.SGLANG_DSV41_CPU_EXPERTS_THREADS.get(),
            spin_core=envs.SGLANG_DSV41_RAM_MISS_SPIN_CORE.get(),
            sqpoll="sqpoll" in envs.SGLANG_EXPERT_STREAM_URING_MODE.get(),
            sq_thread_cpu=envs.SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU.get(),
            numa_cores=envs.SGLANG_EXPERT_NUMA_CORES.get(),
            omp_thread_limit=int(limit) if limit else None,
        )
        placement = parse_placement(envs.SGLANG_MOE_PINNED_HOST_NUMA_MB.get())
        gpu_node = gpu_numa_node(device)
        return cls.resolve(
            nodes=[node for node, _ in placement] or [gpu_node if gpu_node is not None else 0],
            gpu_node=gpu_node,
            affinity=_affinity(),
            topology=Topology.from_sysfs(),
            settings=settings,
        )

    @staticmethod
    def resolve(
        *,
        nodes: Sequence[int],
        gpu_node: Optional[int],
        affinity: Iterable[int],
        topology: Topology,
        settings: CoreSettings,
    ) -> "ThreadingConfig":
        """Every node's plan by the design's rules, validated; raises ValueError naming the node and the rule."""
        affinity = frozenset(affinity)
        gpu = gpu_node if gpu_node is not None else nodes[0]
        overrides = parse_numa_cores(settings.numa_cores)
        for node in overrides:
            if node not in nodes:
                raise ValueError(f"SGLANG_EXPERT_NUMA_CORES: node {node} is not one of the tier's nodes {list(nodes)}")
        if settings.cores:
            cores = parse_cpu_list(settings.cores)
            homes = {topology.node_of(c) for c in cores}
            if len(homes) != 1:
                raise ValueError(f"SGLANG_DSV41_CPU_EXPERTS_CORES {settings.cores!r} spans two nodes")
            home = homes.pop()
            if home not in nodes:
                raise ValueError(f"SGLANG_DSV41_CPU_EXPERTS_CORES: core {cores[0]} is on node {home}, off the tier")
            if "cpu" in overrides.get(home, {}):
                raise ValueError(
                    f"node {home}'s CPU cores are set both by SGLANG_DSV41_CPU_EXPERTS_CORES and SGLANG_EXPERT_NUMA_CORES"
                )
            overrides.setdefault(home, {})["cpu"] = cores
        if settings.spin_core is not None:
            check_not_reserved(settings.spin_core)
            home = topology.node_of(settings.spin_core)
            if home not in nodes:
                raise ValueError(f"SGLANG_DSV41_RAM_MISS_SPIN_CORE: core {settings.spin_core} is on node {home}")
            if "ram" in overrides.get(home, {}):
                raise ValueError(
                    f"node {home}'s RAM core is set both by SGLANG_DSV41_RAM_MISS_SPIN_CORE and SGLANG_EXPERT_NUMA_CORES"
                )
            overrides.setdefault(home, {})["ram"] = [settings.spin_core]
        if settings.sq_thread_cpu is not None and settings.sq_thread_cpu >= 0:
            if len(nodes) > 1:
                raise ValueError(
                    "SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU names one core for several NUMA groups; "
                    "set each group's sq= in SGLANG_EXPERT_NUMA_CORES"
                )
            if "sq" in overrides.get(nodes[0], {}):
                raise ValueError(
                    f"node {nodes[0]}'s SQPOLL core is set both by SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU "
                    "and SGLANG_EXPERT_NUMA_CORES"
                )
            check_not_reserved(settings.sq_thread_cpu)
            overrides.setdefault(nodes[0], {})["sq"] = [settings.sq_thread_cpu]
        if not settings.sqpoll:
            for node, plan_keys in overrides.items():
                if "sq" in plan_keys:
                    raise ValueError(f"node {node} sets an SQPOLL core, but SGLANG_EXPERT_STREAM_URING_MODE is not sqpoll")
        derive = len(nodes) > 1 or settings.cpu_experts or bool(settings.numa_cores)
        if not derive:
            ram = settings.spin_core
            sq = overrides.get(nodes[0], {}).get("sq", [None])[0]
            plan = NodePlan(group=0, node=nodes[0], ram=ram, cpu=(), sq=sq, busy_poll=ram is not None)
            return ThreadingConfig((plan,), (), gpu)
        copy = _copy_core(gpu, overrides, topology, affinity)
        plans = []
        taken: set[int] = set(topology.siblings[copy])
        for group, node in enumerate(nodes):
            plan = _derive(group, node, overrides.get(node, {}), topology, affinity, settings, taken)
            _check_plan(plan, topology, affinity, settings)
            taken.update(s for c in plan.cores() for s in topology.siblings[c])
            plans.append(plan)
        workers = sum(p.threads for p in plans)
        if settings.omp_thread_limit is not None and settings.omp_thread_limit < workers:
            raise ValueError(
                f"OMP_THREAD_LIMIT={settings.omp_thread_limit} is below the {workers} CPU expert workers of all nodes; "
                "two engines' teams run at once"
            )
        return ThreadingConfig(tuple(plans), (copy,), gpu)


def _copy_core(gpu, overrides, topology, affinity) -> int:
    """The copy thread's core on the GPU's node, outside the server's affinity, since the thread never sleeps. Chosen
    before the nodes' plans, so CPU experts that take every free core leave it alone. It spins with PAUSE, so it
    prefers a core whose SMT sibling is the server's and leaves the fully free cores to the busy-polling RAM threads."""
    named = {c for plan in overrides.values() for cores in plan.values() for c in cores}
    free = [
        c
        for c in reversed(topology.physical(gpu))
        if c not in affinity and c not in RESERVED_CORES and not (topology.siblings[c] & named)
    ]
    shared = [c for c in free if topology.siblings[c] & affinity]
    if not free:
        raise ValueError(f"node {gpu} has no core outside the server's affinity for the copy thread")
    return (shared or free)[0]


def _derive(group, node, override, topology, affinity, settings, taken) -> NodePlan:
    """One node's plan: the override's keys as given, every other role from the node's usable cores."""
    usable = [
        c
        for c in reversed(topology.physical(node))
        if c not in affinity and c not in RESERVED_CORES and not (topology.siblings[c] & taken)
    ]
    named = {c for cores in override.values() for c in cores}
    free = [c for c in usable if c not in named]

    def dedicated(core: int) -> bool:
        return not (topology.siblings[core] & affinity)

    if "ram" in override:
        ram = override["ram"][0]
    else:
        ram = next((c for c in free if dedicated(c)), None)
        if ram is None:
            raise ValueError(
                f"node {node} has no usable core for its RAM thread"
                if not usable
                else f"node {node} has no usable core whose physical core is outside the server's affinity"
            )
        free.remove(ram)
    sq = None
    if settings.sqpoll:
        if "sq" in override:
            sq = override["sq"][0]
        elif free:
            sq = free.pop(0)
    cpu: tuple[int, ...] = ()
    if settings.cpu_experts:
        listed = override["cpu"] if "cpu" in override else sorted(free)
        cpu = tuple(listed[: settings.threads] if settings.threads else listed)
    return NodePlan(group=group, node=node, ram=ram, cpu=cpu, sq=sq, busy_poll=True)


def _check_plan(plan: NodePlan, topology: Topology, affinity: frozenset[int], settings: CoreSettings) -> None:
    """The design's refusals for one plan, derived or given."""
    cores = plan.cores()
    for core in cores:
        try:
            check_not_reserved(core)
        except ValueError as refusal:
            raise ValueError(f"node {plan.node}: {refusal}") from None
        home = topology.node_of(core)
        if home != plan.node:
            raise ValueError(f"node {plan.node}: core {core} is on node {home}")
        if core in affinity:
            raise ValueError(f"node {plan.node}: core {core} is in the server's affinity")
    for a, b in combinations(cores, 2):
        if b in topology.siblings[a]:
            raise ValueError(f"node {plan.node}: cores {min(a, b)} and {max(a, b)} share a physical core")
    if plan.ram is not None and topology.siblings[plan.ram] & affinity:
        raise ValueError(
            f"node {plan.node}: core {plan.ram} shares a physical core with the server's affinity, "
            "so its busy-polling RAM thread would not have the core to itself"
        )
    if settings.cpu_experts:
        if settings.threads == 1 and plan.threads == 1:
            raise ValueError(
                f"THREADS=1 leaves one core for node {plan.node}'s engine; CPU experts need at least 2"
            )
        try:
            check_engine_cores(plan.cpu, plan.threads)
        except ValueError as refusal:
            raise ValueError(f"node {plan.node}: {refusal}") from None
