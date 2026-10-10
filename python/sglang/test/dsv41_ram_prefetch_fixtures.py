"""Hosts for the NVMe-to-RAM prefetch tests (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
Phase 1): an instrumented ExpertStreamHost with a speculative pool, the copy engine's test backend and CPU experts on
the fake kernel, every eligible lane the CPU's, driven by ChainSim."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import (
    ExpertStreamHost,
    candidate_offset,
    new_candidate_page,
    new_hot_page,
    new_page,
)
from sglang.test.dsv41_chain_sim import ChainSim, SimRequest
from sglang.test.dsv41_ram_miss_fixtures import RamMissSetup, fake_cpu_layer, ram_miss_setup

ROWS = 2
HIDDEN = 8  # the fake CPU kernel's hidden size, and the test gates'
LANES = 8
# Two groups over an 8-slot row: node 0 slots 0-3, node 1 slots 4-7.
HALVES = [[(0, 4)] * ROWS, [(4, 8)] * ROWS]


@dataclass
class PrefetchRig:
    setup: RamMissSetup
    page: torch.Tensor
    host: ExpertStreamHost
    sim: ChainSim
    x_rows: torch.Tensor
    out_rows: torch.Tensor
    tokens: int


def x_token_bytes() -> int:
    """Bytes between two tokens' staged inputs (enable_cpu_experts: fp16, padded to 16 bytes)."""
    return (2 * HIDDEN + 15) // 16 * 16


def prefetch_rig(
    tmp_path, *, nodes=1, capacity=None, staging=None, share=2, pool=True, tokens=1, hot=False, rows=ROWS,
    mirror_weights=None,
) -> PrefetchRig:
    """One node: 7 slots a row, 3 staging, `share` pooled, 6 experts. Two nodes: 8 slots split by HALVES, 1 staging and
    `share` pooled per group, 8 experts. CPU experts on every group (one fake-kernel worker each), every row
    registered, split[n] = n so every eligible lane is the CPU's; rows of `tokens` staged inputs; with `hot`, a GPU hot
    page; `rows` streamed rows; `mirror_weights` one weight per mirror root (one root when None)."""
    experts = 6 if nodes == 1 else 8
    capacity = (7 if nodes == 1 else 8) if capacity is None else capacity
    staging = (3 if nodes == 1 else 1) if staging is None else staging
    s = ram_miss_setup(tmp_path, capacity=capacity, experts=experts, layers=rows, mirror_weights=mirror_weights)
    page = new_page(pin=False, wire=wire_layout(LANES, nodes))
    host = ExpertStreamHost(
        s.tables,
        page=page,
        slot_map=torch.full((rows, experts), -1, dtype=torch.int32),
        variant="instr",
        hot_page=new_hot_page(experts, pin=False) if hot else None,
        **({"node_ranges": [[(0, 4)] * rows, [(4, 8)] * rows]} if nodes == 2 else {}),
    )
    try:
        host.reserve_staging(staging)
        if pool:
            host.reserve_spec_pool(share)
        host.enable_copy_engine(-1)
        host.arm_copy_engine()
        table = 16 + 4 * LANES * (1 + tokens) if tokens > 1 else 0
        x_rows = torch.zeros((rows, tokens * x_token_bytes() + table), dtype=torch.uint8)
        shape = (rows, 2 * nodes, HIDDEN) if tokens == 1 else (rows, 2 * nodes, tokens, HIDDEN)
        out_rows = torch.zeros(shape, dtype=torch.float32)
        kernel = host.test_kernel_address()
        cores = sorted(os.sched_getaffinity(0))
        for g in range(nodes):
            host.enable_cpu_experts(
                kernel, list(range(LANES + 1)), [cores[g % len(cores)]], x_rows, out_rows, threads=1, group=g
            )
        for row in range(rows):
            host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
    except BaseException:
        host.stop()
        raise
    return PrefetchRig(s, page, host, ChainSim(host, page, s.slabs), x_rows, out_rows, tokens)


def _served(rig: PrefetchRig, req: SimRequest) -> None:
    if rig.host.threaded:
        assert rig.sim.wait_handled(req, timeout_s=10.0)
    else:
        assert rig.host.pump() == 1
    assert rig.sim.wait_served(req, timeout_s=10.0) and rig.sim.copy_wait(req, timeout_s=10.0)
    if rig.host.threaded and rig.sim.needs_copy_wait(req):
        rig.sim.open_copy_gate(req)


def load(rig: PrefetchRig, row: int, experts) -> SimRequest:
    """Makes `experts` resident in `row` through an uncaptured post (each a GPU miss into staging)."""
    req = rig.sim.post(row, list(experts))
    _served(rig, req)
    return req


def forced(rig: PrefetchRig, row: int, experts, *, serve: bool = True) -> SimRequest:
    """A captured post whose lanes are all forced (spill): a hit is a CPU lane at its RAM slot, a miss a CPU miss with
    slot -1 that the host reads into a RAM victim, or swaps in from the pool."""
    req = rig.sim.post(row, list(experts), captured=True, cpu_on=True, cpu_misses=True, forced_from=0)
    if serve:
        _served(rig, req)
    return req


# Token 0's logits over the one-node rig's six experts: expert 2 first, then 4, 0, 1, 3, 5.
LOGITS = [[30, 25, 40, 22, 35, 21]]


def gate(logits) -> tuple[torch.Tensor, torch.Tensor]:
    """A gate as bf16 [1, experts, HIDDEN] and its zero fp32 bias: token t's input is the unit vector e_t (write_x), so
    its logit for expert e is logits[t][e]. Logits above 20 keep softplus the identity: the score order is theirs."""
    experts = len(logits[0])
    w = torch.zeros((1, experts, HIDDEN), dtype=torch.bfloat16)
    for t, row in enumerate(logits):
        w[0, :, t] = torch.tensor(row, dtype=torch.bfloat16)
    return w, torch.zeros((1, experts), dtype=torch.float32)


def enable(
    rig: PrefetchRig,
    logits,
    *,
    top_k=2,
    per_token=1,
    per_layer=1,
    cores=None,
    top_k_only=False,
    idle_deadline_s=None,
    idle_spread=False,
    idle_pieces=1,
) -> None:
    """Row 0 targets row 1 with `logits`' gate; every later row targets nothing. `idle_deadline_s`: idle-drive
    speculative reads with that deadline, `idle_spread` and `idle_pieces` (None: the shared reader)."""
    w, bias = gate(logits)
    targets = torch.tensor([[1, 0]] + [[-1, -1]] * (rig.x_rows.shape[0] - 1), dtype=torch.int64)
    rig.host.enable_ram_prefetch(
        targets,
        w,
        bias,
        top_k=top_k,
        per_token=per_token,
        per_layer=per_layer,
        cores=cores if cores is not None else [[] for _ in range(rig.host.nodes)],
        top_k_only=top_k_only,
        **(
            {}
            if idle_deadline_s is None
            else {
                "idle_drive": True,
                "idle_deadline_s": idle_deadline_s,
                "idle_spread": idle_spread,
                "idle_pieces": idle_pieces,
            }
        ),
    )


def write_x(rig: PrefetchRig, row: int, tokens: int) -> None:
    """Stages `row`'s input as the post kernel does: token t is the unit vector e_t (fp16); a multi-token row's table
    counts the live tokens."""
    stride = x_token_bytes()
    for t in range(tokens):
        x = torch.zeros(HIDDEN, dtype=torch.float16)
        x[t] = 1.0
        rig.x_rows[row, t * stride : t * stride + 2 * HIDDEN] = x.view(torch.uint8)
    if rig.tokens > 1:
        table = rig.tokens * stride
        rig.x_rows[row, table : table + 4] = torch.tensor([tokens], dtype=torch.int32).view(torch.uint8)


def trigger(rig: PrefetchRig, *, tokens: int = 1, resident: int = 5) -> SimRequest:
    """A captured post of row 0 with one CPU hit (`resident`, loaded first if absent), after staging row 0's input:
    the service hands it to every group's speculative thread."""
    if rig.host.mapping(0)[resident] < 0:
        load(rig, 0, [resident])
    write_x(rig, 0, tokens)
    req = rig.sim.post(0, [resident], captured=True, cpu_on=True)
    _served(rig, req)
    return req


def enable_gpu(rig: PrefetchRig, *, targets=None, top_k=2, per_token=1, per_layer=1, min_margin=None) -> torch.Tensor:
    """The GPU scorer on the rig (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design): row 0 targets row 1 unless
    `targets` (int64 [rows, 2]) says otherwise. Returns the candidate page, which the test writes as the select kernel
    would (write_candidate_slot). `min_margin` (fp32 [rows]) is the per-target-row margin floor."""
    rows = rig.x_rows.shape[0]
    if targets is None:
        targets = torch.tensor([[1, 0]] + [[-1, -1]] * (rows - 1), dtype=torch.int64)
    page = new_candidate_page(pin=False)
    rig.host.enable_ram_prefetch(
        targets,
        None,
        None,
        top_k=top_k,
        per_token=per_token,
        per_layer=per_layer,
        cores=[[] for _ in range(rig.host.nodes)],
        candidates=page,
        min_margin=min_margin,
    )
    return page


def write_candidate_slot(page: torch.Tensor, seq: int, picks, *, flags: int = 0, slot_seq=None) -> None:
    """Writes `seq`'s slot as the select kernel does: the count, flags and (expert, rank, margin) entries, then the seq
    word. `slot_seq` stores another word instead: a later record's (a lapping select) or 0 (a slot still open)."""
    off = candidate_offset(seq)
    payload = struct.pack("<HH", len(picks), flags) + b"".join(struct.pack("<HBxf", e, r, m) for e, r, m in picks)
    page[off + 4 : off + 4 + len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    word = (seq if slot_seq is None else slot_seq) & 0xFFFFFFFF
    page[off : off + 4] = torch.frombuffer(bytearray(struct.pack("<I", word)), dtype=torch.uint8)


GATE_DEPTH = 12  # GateScorer::kDepth


def gate_reference(x, w, bias, skip, *, top_k, per_token, per_layer, top_k_only=False, meta=False):
    """A torch reference of the Phase 0 replay's ranking, which GateScorer::choose and the GPU select kernel both
    implement: sqrt(softplus(W x)) + b per live token, each token's top 12 walked in score order past the skipped
    experts for per_token picks, each pick's margin its score less the token's top_k-th, the union by margin (ties by
    id), the first per_layer."""
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    scores = torch.where(scores.isnan(), torch.tensor(float("-inf")), scores)  # a NaN score ranks below every other
    best, ranks = {}, {}
    experts = w.shape[0]
    for m in range(x.shape[0]):
        s = scores[m]
        order = sorted(range(experts), key=lambda e: (-float(s[e]), e))[: min(GATE_DEPTH, experts)]
        walk = order[:top_k] if top_k_only else order
        kth = s[order[top_k - 1]]
        picked = 0
        for rank, e in enumerate(walk):
            if picked >= per_token:
                break
            if skip[e]:
                continue
            # an fp32 subtraction, as the host's; equal scores (also two infinities) give 0, -inf stays above unpicked
            margin = 0.0 if s[e] == kth else max(float(s[e] - kth), -3.4028234663852886e38)
            # an expert's rank is its position in the token that gave its best margin, the lowest on a tie
            if e not in best or margin > best[e] or (margin == best[e] and rank < ranks[e]):
                ranks[e] = rank
            best[e] = max(best.get(e, float("-inf")), margin)
            picked += 1
    chosen = [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:per_layer]
    return [(e, ranks[e], best[e]) for e in chosen] if meta else chosen


def exact_gate_case(seed, tokens=3, experts=16, hidden=64):
    """Small non-negative integers with a constant 21 term: every logit is an integer above 20, exact in fp32 in any
    summation order, so softplus is the identity and host, GPU and torch produce the same fp32 scores bit for bit."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 21
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias


def lead_gate_case(seed, tokens=3, experts=16, hidden=16):
    """exact_gate_case with a lead term of 128, so x may go negative while every logit stays an exact integer above
    20."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-2, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 128
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias
