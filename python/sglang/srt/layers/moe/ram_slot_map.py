"""Reference model of the device slot map (analysis/dsv41-drive/LEASE_PROTOCOL.md): lane typing in the post and
delta application. The CUDA post kernel and map_bulk_apply transcribe it; the parity tests compare the two."""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable, Optional, Sequence

from sglang.kernels.ops.moe.expert_lease_block import wire_layout


class LaneKind(IntEnum):
    """lease_layout.h kKind*: what moves a lane's bytes and what the device waits on."""

    HIT_COPY = 1
    HIT_SM = 2
    HIT_CPU = 3
    MISS_GPU = 4
    MISS_CPU = 5
    HIT_SCRATCH = 6


class LaneOverflow(ValueError):
    """Forced lanes (DIRECT found them no VRAM victim, the post's spill) cannot be CPU lanes: no host lanes (the copy
    engine is not armed) or no CPU layer. The post then serves the unforced prefix and flags the forward
    (expert_stream_post_kernel)."""


def split_miss_cut(cut: int, cut_max: int, *, cpu_experts: bool) -> tuple[int, int]:
    """The split miss cut the device applies: none without CPU experts, where there is no CPU split to cut."""
    return (cut if cpu_experts else 0), cut_max


# The model-based split's per-node cost words (ns): a CPU hit job of k experts and p (token, expert) pairs costs
# hit_a + hit_b * k + hit_c * p, a CPU miss of t tokens miss_a + miss_c * t.
SPLIT_MODEL_NODE_KEYS = ("hit_a", "hit_b", "hit_c", "miss_a", "miss_c")
SPLIT_MODEL_KEYS = ("copy_lane", "first_byte", "per_expert", "stream_lane", "stream_base")
# Fixture costs in ms, for unit tests and synthetic fixtures only: a served model takes its costs from the startup
# calibration (split_model_calibration.py), never from here. They are the 2026-10-15 offline fits on
# cores-ab-20261014/pin01-r2's CPU jobs (analysis/dsv41-drive/dspark/split_replay.py), scaled by 0.93 to the untraced
# cost per lane; miss i (1-based) of a node lands first_byte + per_expert * i after the post. With m GPU-streamed misses
# in the record, the GPU's stream kernel (S) ends at stream_base + first_byte + per_expert * m + stream_lane * copy
# lanes: its pulls share the link with the copy engine. stream_lane is the fit on pin01-r2, splitmodel, splitmodel-r2
# and split-ab A2/B1 (S grows 0.3-1.4 ms per added copy lane between runs, 0.44-1.04 within one; node-independent); 0
# drops the stream side.
SPLIT_MODEL_FIXTURE_MS = {
    "hit_a": (0.016, -0.050),
    "hit_b": (0.154, 0.179),
    "hit_c": (0.395, 0.390),
    "miss_a": (0.331, 0.291),
    "miss_c": (0.193, 0.245),
    "copy_lane": 1.05,
    "first_byte": 0.8,
    "per_expert": 1.6,
    "stream_lane": 0.65,
    "stream_base": 0.0,
}
# A copy_lane word below 0 turns the model off on the device: the split table and the miss cut type the CPU's lanes.
# The service publishes it until its calibration has run, and keeps it when the calibration fails.
SPLIT_MODEL_OFF = -1
# The flags word's residency tiebreak (lease_device.cuh kSplitModelResidencyTiebreak): among a node's candidates with
# the same tokens, which the model prices alike, the CPU takes the lowest residency keys and the copy engine the rest.
SPLIT_MODEL_RESIDENCY_TIEBREAK = 1


def split_model_overrides(overrides: str, nodes: int) -> tuple[dict, dict]:
    """``cpu_experts.split.model_costs`` parsed: ({(key, node): ms} for per-node costs, {key: ms} for shared ones).

    ``overrides`` is ``name=ms`` pairs, comma-separated; a per-node name takes ``.node`` (``hit_c.1=0.4``), and without
    it sets every node. Raises ValueError for a pair it cannot read."""
    per_node, shared = {}, {}
    for item in filter(None, (x.strip() for x in overrides.split(","))):
        name, sep, value = item.partition("=")
        key, _, node = name.strip().partition(".")
        if not sep or key not in SPLIT_MODEL_NODE_KEYS + SPLIT_MODEL_KEYS or (node and key in SPLIT_MODEL_KEYS):
            raise ValueError(f"split model costs: cannot read {item!r} (name[.node]=ms with names "
                             f"{', '.join(SPLIT_MODEL_NODE_KEYS + SPLIT_MODEL_KEYS)})")
        ms = float(value)
        if key in SPLIT_MODEL_KEYS:
            if ms < 0:
                raise ValueError(f"split model costs: {key} must be >= 0, got {ms}")
            shared[key] = ms
        elif node:
            if not node.isdigit() or int(node) >= nodes:
                raise ValueError(f"split model costs: {item!r} names a node outside 0..{nodes - 1}")
            per_node[(key, int(node))] = ms
        else:
            for n in range(nodes):
                per_node[(key, n)] = ms
    return per_node, shared


def split_model_overrides_complete(overrides: str, nodes: int) -> bool:
    """Whether ``overrides`` names every cost of every node, so the model needs no calibration."""
    per_node, shared = split_model_overrides(overrides, nodes)
    return len(per_node) == len(SPLIT_MODEL_NODE_KEYS) * nodes and len(shared) == len(SPLIT_MODEL_KEYS)


def split_model_words(nodes: int, overrides: str = "", base: Optional[dict] = None,
                      residency_tiebreak: bool = False) -> list[int]:
    """The model-based split's cost words, in ns: per node ``SPLIT_MODEL_NODE_KEYS``, then ``SPLIT_MODEL_KEYS``, then
    a flags word (``SPLIT_MODEL_RESIDENCY_TIEBREAK``, cpu_experts.split.model_residency_tiebreak).

    ``base`` maps each key to ms, a per-node key to one value per node (node n past them takes the last's): the
    calibrated costs in service (split_model_calibration.SplitModelCosts.base), the fixture costs when omitted, which
    only tests and synthetic fixtures do. ``overrides`` (cpu_experts.split.model_costs, split_model_overrides)
    replace the costs they name."""
    base = SPLIT_MODEL_FIXTURE_MS if base is None else base
    per_node = [
        {key: base[key][min(n, len(base[key]) - 1)] for key in SPLIT_MODEL_NODE_KEYS} for n in range(nodes)
    ]
    shared = {key: base[key] for key in SPLIT_MODEL_KEYS}
    node_overrides, shared_overrides = split_model_overrides(overrides, nodes)
    for (key, n), ms in node_overrides.items():
        per_node[n][key] = ms
    shared.update(shared_overrides)
    ns = lambda ms: int(round(ms * 1_000_000))
    return [ns(costs[key]) for costs in per_node for key in SPLIT_MODEL_NODE_KEYS] + [
        ns(shared[key]) for key in SPLIT_MODEL_KEYS
    ] + [SPLIT_MODEL_RESIDENCY_TIEBREAK if residency_tiebreak else 0]


def split_model_off_words(nodes: int) -> list[int]:
    """Cost words the device reads as the model off (SPLIT_MODEL_OFF in copy_lane): the split table types the lanes."""
    words = [0] * (5 * nodes + 6)
    words[5 * nodes] = SPLIT_MODEL_OFF
    return words


def split_model_on(words: Optional[Sequence[int]], nodes: int) -> bool:
    """Whether cost words type lanes by the model: present, and their copy_lane not SPLIT_MODEL_OFF."""
    return words is not None and len(words) > 5 * nodes and words[5 * nodes] >= 0


def split_model_stats_words(nodes: int) -> int:
    """The device's split-model counters after the cost words: records typed by the model, then per node the lanes it
    put on the CPU that the table would not and the lanes it took off the CPU."""
    return 1 + 2 * nodes


def model_node_finish(words: Sequence[int], node: int, k: int, pairs: int, miss_tokens: Sequence[int],
                      nodes: int) -> int:
    """The model's predicted end (ns after the post) of node's CPU: a hit job of k experts and `pairs` (token, expert)
    pairs, then each forced miss i (1-based, plan order) after it lands at first_byte + per_expert * i."""
    hit_a, hit_b, hit_c, miss_a, miss_c = words[5 * node : 5 * node + 5]
    first_byte, per_expert = words[5 * nodes + 1], words[5 * nodes + 2]
    t = hit_a + hit_b * k + hit_c * pairs if k > 0 else 0
    for i, tokens in enumerate(miss_tokens, 1):
        t = max(t, first_byte + per_expert * i) + miss_a + miss_c * tokens
    return t


def model_cpu_lanes(
    node_of: Sequence[int],
    tokens: Sequence[int],
    hit: Sequence[bool],
    eligible: Sequence[bool],
    forced: Sequence[bool],
    copyable: Sequence[bool],
    words: Sequence[int],
    nodes: int,
    streamed: int = 0,
) -> list[bool]:
    """The model-based split's unforced CPU lanes, in integer ns as the device computes them.

    Each node's CPU starts with its forced hits (and, as CPU lanes, eligible lanes the copy engine cannot take); its
    forced misses follow the hit job. The copyable eligible hits, heaviest-token first (ties: plan order), each go to
    their node's CPU when that node would finish no later than the copy side would with the lane, else to the copy
    engine. The copy side is the copy engine's FIFO (one copy_lane a lane, both nodes sharing it) and, with `streamed`
    GPU-streamed misses and stream_lane > 0, the stream kernel: stream_base + first_byte + per_expert * streamed +
    stream_lane * copy lanes."""
    k, pairs = [0] * nodes, [0] * nodes
    misses = [[] for _ in range(nodes)]
    cpu = [False] * len(node_of)
    candidates = []
    for j, node in enumerate(node_of):
        if forced[j]:
            if hit[j]:
                k[node] += 1
                pairs[node] += tokens[j]
            else:
                misses[node].append(tokens[j])
        elif eligible[j]:
            if hit[j] and copyable[j]:
                candidates.append(j)
            else:
                cpu[j] = True
                if hit[j]:
                    k[node] += 1
                    pairs[node] += tokens[j]
    copy_lane, first_byte, per_expert, stream_lane, stream_base = words[5 * nodes : 5 * nodes + 5]
    stream0 = stream_base + first_byte + per_expert * streamed
    copy, lanes = 0, 0
    for j in sorted(candidates, key=lambda j: (-tokens[j], j)):
        node = node_of[j]
        t = model_node_finish(words, node, k[node] + 1, pairs[node] + tokens[j], misses[node], nodes)
        side = copy + copy_lane
        if streamed > 0 and stream_lane > 0:
            side = max(side, stream0 + stream_lane * (lanes + 1))
        if t <= side:
            cpu[j] = True
            k[node] += 1
            pairs[node] += tokens[j]
        else:
            copy += copy_lane
            lanes += 1
    if words[5 * nodes + 5] & SPLIT_MODEL_RESIDENCY_TIEBREAK:
        # A node's candidates with the same tokens cost the model the same: keep how many of them the CPU takes, and
        # give it the highest plan indexes (plan order is residency key, highest first).
        groups: dict = {}
        for j in candidates:
            groups.setdefault((node_of[j], tokens[j]), []).append(j)
        for members in groups.values():
            on_cpu = sum(cpu[j] for j in members)
            for rank, j in enumerate(sorted(members, reverse=True)):
                cpu[j] = rank < on_cpu
    return cpu


def type_lanes(
    experts: Sequence[int],
    ram_slot: Sequence[int],
    staging: Sequence[int],
    split: Sequence[int],
    *,
    lanes: int,
    captured: bool,
    copy_armed: bool,
    hit_copy: str,
    cpu_on: bool,
    cpu_misses: bool,
    ce_ok: bool = True,
    cpu_ok: bool = True,
    dst_ok: Optional[Sequence[bool]] = None,
    nodes: int = 1,
    forced_from: Optional[int] = None,
    miss_cut: int = 0,
    miss_cut_max: int = 3,
    scratch: Optional[Sequence[int]] = None,
    split_model: Optional[Sequence[int]] = None,
    tokens: Optional[Sequence[int]] = None,
    model_stats: Optional[list] = None,
) -> tuple[list[LaneKind], list[int]]:
    """Each lane's kind and source slot: its RAM slot for a hit, the next staging slot of its home node for a miss.

    ``staging`` is node-major, ``nodes * lanes`` slots, and ``split`` node-major, ``nodes * (lanes + 1)`` counts. A
    miss takes the next slot of its home node's list; node n's CPU takes the last ``split[n][k]`` of its k eligible
    lanes in plan order (miss_keys descending, so the lowest-scored): RAM hits, plus NVMe misses with
    ``cpu_misses``.

    Lanes from ``forced_from`` on (spill, copy.graph_gather_victim_lanes) found no VRAM victim: each is a
    CPU lane whatever the split. A forced hit runs from its RAM slot; a forced miss gets slot -1 and no staging slot,
    since the host reads it into a RAM victim (RamTier::reserve_victims_locked). The split counts only the lanes before
    ``forced_from``. Raises LaneOverflow when forced lanes cannot be CPU lanes (no host lanes, no CPU layer).

    Experiment (cpu_experts.split.miss_cut / miss_cut_max): a forced CPU miss runs on its node's CPU behind the record's
    CPU-hit job, so a node with 1..``miss_cut_max`` forced misses takes ``miss_cut`` fewer of its split's lanes (at
    least 0). Forced lanes stay CPU lanes; ``miss_cut`` 0 is the typing without the cut.

    GPU drip (host/gpu_drip.h): ``scratch[j]`` is the scratch slot whose published word names lane j's expert for this
    row, else -1. With host lanes, an unforced lane with a scratch slot is HIT_SCRATCH at that slot, before any RAM
    lookup: it takes no staging slot and is not eligible, so the split sees one lane fewer.

    Experiment (cpu_experts.split.model): with ``split_model`` (split_model_words) and the copy engine able
    to take hits, model_cpu_lanes picks the unforced CPU lanes in place of the split and the miss cut, from each lane's
    ``tokens`` (the routes that name its expert); words whose copy_lane is SPLIT_MODEL_OFF keep the table's typing, and
    count nothing. ``model_stats`` (split_model_stats_words long) counts what the device
    counts: records with an eligible lane, then per node the lanes moved onto and off the CPU against the table."""
    if split_model is not None and (tokens is None or len(tokens) != len(experts)):
        raise ValueError("the split model needs each lane's tokens")
    if len(experts) > lanes:
        raise ValueError(f"a request has at most {lanes} lanes, got {len(experts)}")
    if len(set(experts)) != len(experts):
        raise ValueError(f"a request names an expert twice: {list(experts)}")
    forced_from = len(experts) if forced_from is None else forced_from
    host_lanes = captured and copy_armed
    scratched = [
        host_lanes and scratch is not None and j < forced_from and scratch[j] >= 0 for j in range(len(experts))
    ]
    home = wire_layout(lanes, nodes).home
    slots, hit, taken = [], [], [0] * nodes
    for j, e in enumerate(experts):
        if scratched[j]:
            slots.append(scratch[j])
            hit.append(False)
            continue
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
            continue
        if j >= forced_from:
            slots.append(-1)  # host-placed: read into a RAM victim, never a staging slot
            hit.append(False)
            continue
        node = home(e)
        m = taken[node]
        if m >= lanes or staging[node * lanes + m] < 0:
            raise ValueError(f"a miss lane has no staging slot on node {node}")
        slots.append(staging[node * lanes + m])
        hit.append(False)
        taken[node] += 1
    can_cpu = host_lanes and cpu_on and cpu_ok
    if forced_from < len(experts) and not can_cpu:
        raise LaneOverflow(f"lanes {forced_from}.. found no victim and cannot be CPU lanes")
    eligible = [
        j < forced_from and can_cpu and not scratched[j] and (h or cpu_misses) for j, h in enumerate(hit)
    ]
    take = [0] * nodes
    for node in range(nodes):
        n = sum(1 for e, ok in zip(experts, eligible) if ok and home(e) == node)
        take[node] = split[node * (lanes + 1) + n] if n else 0
        forced_misses = sum(1 for j, e in enumerate(experts) if j >= forced_from and not hit[j] and home(e) == node)
        if miss_cut > 0 and 1 <= forced_misses <= miss_cut_max:
            take[node] = max(0, take[node] - miss_cut)
    table = [False] * len(experts)
    for j in reversed(range(len(experts))):
        node = home(experts[j])
        if take[node] > 0 and eligible[j]:
            table[j] = True
            take[node] -= 1
    copy_ok = host_lanes and hit_copy == "ce" and ce_ok
    cpu = [j >= forced_from or table[j] for j in range(len(experts))]
    if split_model_on(split_model, nodes) and copy_ok:
        node_of = [home(e) for e in experts]
        copyable = [True if dst_ok is None else bool(dst_ok[j]) for j in range(len(experts))]
        forced = [j >= forced_from for j in range(len(experts))]
        # GPU-streamed misses (MISS_GPU): unforced, unscratched misses the CPU does not take.
        streamed = sum(1 for j in range(len(experts))
                       if not forced[j] and not scratched[j] and not hit[j] and not eligible[j])
        model = model_cpu_lanes(node_of, tokens, hit, eligible, forced, copyable, split_model, nodes, streamed)
        cpu = [forced[j] or model[j] for j in range(len(experts))]
        if model_stats is not None and any(eligible):
            model_stats[0] += 1
            for j, node in enumerate(node_of):
                if eligible[j] and model[j] and not table[j]:
                    model_stats[1 + 2 * node] += 1
                elif eligible[j] and table[j] and not model[j]:
                    model_stats[2 + 2 * node] += 1
    kinds = []
    for j, (h, c) in enumerate(zip(hit, cpu)):
        if scratched[j]:
            kinds.append(LaneKind.HIT_SCRATCH)
        elif c:
            kinds.append(LaneKind.HIT_CPU if h else LaneKind.MISS_CPU)
        elif h:
            dst = True if dst_ok is None else dst_ok[j]
            kinds.append(LaneKind.HIT_COPY if copy_ok and dst else LaneKind.HIT_SM)
        else:
            kinds.append(LaneKind.MISS_GPU)
    return kinds, slots


class MapReplica:
    """The device's map bank: ram_slot [rows][experts], staging [rows][nodes * lanes], map_chain and map_applied [rows].

    map_chain starts at 1 and the attach delta has tag 1, so a zero-filled delta record never matches."""

    def __init__(self, rows: int, experts: int, lanes: int, nodes: int = 1):
        self.ram_slot = [[-1] * experts for _ in range(rows)]
        self.staging = [[-1] * (nodes * lanes) for _ in range(rows)]
        self.map_chain = [1] * rows
        self.map_applied = [0] * rows

    def apply_delta(self, row: int, tag: int, staging: Sequence[int], entries: Iterable[tuple[int, int]]) -> None:
        if self.map_applied[row] == tag:
            return
        for e, s in entries:
            self.ram_slot[row][e] = s
        self.staging[row] = list(staging)
        self.map_applied[row] = tag

    def apply_bulk(self, entries: Iterable[tuple[int, int, int]]) -> None:
        for row, e, s in entries:
            self.ram_slot[row][e] = s
