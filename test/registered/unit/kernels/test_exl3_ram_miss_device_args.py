"""The device wrapper refuses buffers the kernels cannot address, and the wire header is the Python layout (CPU)."""

import ast
import operator
import re
import types
from pathlib import Path

import pytest
import torch

import sglang.kernels.ops.moe.expert_stream_transport as ram_miss
from sglang.kernels.ops.moe import expert_lease_block as lease
from sglang.kernels.ops.moe.expert_stream_transport import PAGE_BYTES, STATE_WORDS, ExpertStreamDevice
from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import device_sources, host_sources, joined_text, wire_header

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CSRC = Path(ram_miss.__file__).resolve().parents[2] / "jit" / "csrc" / "moe"


def _runs(layers=2, experts=4):
    return torch.zeros((layers, experts, ram_miss.STAGE_PIECES, 1, 2), dtype=torch.int32)


def _device(layers=2, experts=4, page=None, **kwargs):
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8) if page is None else page
    kwargs.setdefault("piece_runs", _runs(layers, experts))
    kwargs.setdefault("row_capacities", [5, 7][:layers] + [3] * max(0, layers - 2))
    kwargs.setdefault("timeout_ms", 10)
    return ExpertStreamDevice(
        page, lease.new_lease_block(layers, pin=False), device="cpu", layers=layers, experts=experts, **kwargs
    )


def _args(lanes=8):
    return dict(
        planned=torch.zeros(lanes, dtype=torch.int64),
        count=torch.zeros(1, dtype=torch.int32),
        routes=torch.full((lanes,), -1, dtype=torch.int64),
        dst_slots=torch.zeros(lanes, dtype=torch.int32),
    )


def _post(dev, a, row=0):
    dev.post(row, a["planned"], a["count"], a["routes"], a["dst_slots"])


def test_state_words_are_distinct_and_dense():
    assert sorted(STATE_WORDS.values()) == list(range(len(STATE_WORDS)))


def test_a_page_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="page"):
        _device(page=torch.zeros(10, dtype=torch.uint8))


def test_the_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        _device(timeout_ms=0)


def test_more_experts_than_a_record_id_carries_are_refused():
    with pytest.raises(ValueError, match="32767"):
        _device(experts=ram_miss.RECORD_ID_MAX + 1)


def test_a_row_capacity_a_record_slot_cannot_carry_is_refused():
    with pytest.raises(ValueError, match="32767"):
        _device(row_capacities=[5, ram_miss.RECORD_ID_MAX + 1])


def test_a_page_off_16_byte_alignment_is_refused():
    # The post writes the record with 16-byte stores; a misaligned one faults inside the captured graph.
    with pytest.raises(ValueError, match="16-byte"):
        _device(page=torch.zeros(PAGE_BYTES + 1, dtype=torch.uint8)[1:])


@pytest.mark.parametrize("experts, capacity", [(ram_miss.RECORD_ID_MAX + 1, 5), (4, ram_miss.RECORD_ID_MAX + 1)])
def test_the_host_refuses_tables_its_deltas_cannot_carry(experts, capacity):
    # The host writes expert ids and slots into the i16 map delta; checked before anything else is read.
    tables = types.SimpleNamespace(
        starts=torch.zeros((1, experts), dtype=torch.int64), capacity=torch.tensor([capacity], dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="32767"):
        ram_miss.ExpertStreamHost(tables, page=torch.zeros(PAGE_BYTES, dtype=torch.uint8),
                                  slot_map=torch.full((1, experts), -1, dtype=torch.int32))


def test_piece_runs_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="piece_runs"):
        _device(piece_runs=_runs(layers=3))
    with pytest.raises(ValueError, match="piece_runs"):
        _device(piece_runs=_runs(experts=5))


def test_an_unpinned_page_is_refused_for_a_cuda_device():
    # Checked before any CUDA call: the kernels read it through UVA.
    with pytest.raises(ValueError, match="pinned"):
        ExpertStreamDevice(
            torch.zeros(PAGE_BYTES, dtype=torch.uint8), lease.new_lease_block(2, pin=False), device="cuda", layers=2,
            experts=4, timeout_ms=10, piece_runs=_runs(), row_capacities=[5, 7],
        )


def test_a_row_outside_the_streamed_layers_is_refused():
    dev, a = _device(layers=2), _args()
    for row in (-1, 2):
        with pytest.raises(ValueError, match="row"):
            _post(dev, a, row=row)
        with pytest.raises(ValueError, match="row"):
            dev.set_row_copy(row, 4)


def test_buffers_of_the_wrong_dtype_are_refused():
    dev = _device()
    for name, dtype in (("planned", torch.int32), ("count", torch.int64), ("routes", torch.int32), ("dst_slots", torch.int64)):
        a = _args()
        a[name] = a[name].to(dtype)
        with pytest.raises(ValueError, match=name):
            _post(dev, a)


def test_the_device_sequence_continues_from_the_page_head():
    """A device built over a used page posts the next sequence, not 1, which the thread would never serve."""
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    page[ram_miss.WORDS["demand_head"] : ram_miss.WORDS["demand_head"] + 4].view(torch.int32)[0] = 7
    assert int(_device(page=page).state[STATE_WORDS["posted"]]) == 7


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.LShift: operator.lshift}


def _evaluate(node, known):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return known[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand, known)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left, known), _evaluate(node.right, known))
    raise ValueError(f"unsupported constant expression {ast.dump(node)}")


def _constants(*paths: Path, known: dict[str, int] | None = None) -> dict[str, int]:
    """Every ``constexpr <type> kName = <integer expression>;`` of the given C++ sources, read as one text; a name
    defined twice fails. ``known`` seeds names defined elsewhere (the wire header)."""
    seeded = dict(known or {})
    found: dict[str, int] = {}
    pattern = r"^\s*(?:static\s+)?constexpr\s+[\w:]+\s+(k\w+)\s*=\s*([^;]+);"
    for name, expression in re.findall(pattern, joined_text(paths), re.MULTILINE):
        assert name not in found, f"{name} is defined twice: the layout check cannot tell which one applies"
        expression = re.sub(r"(?<=\d)[uU][lL]*\b", "", expression.strip())
        found[name] = seeded[name] = _evaluate(ast.parse(expression, mode="eval").body, seeded)
    return found


def _wire():
    return _constants(wire_header())


_NAME = re.compile(r"^\s*(?:static\s+)?constexpr\s+[\w:]+\s+(k\w+)\s*=", re.MULTILINE)

# The wire header, in full: every constant it defines and the Python value it must equal.
PYTHON_WIRE = {
    "kDemandHead": ram_miss.WORDS["demand_head"],
    "kDemandRing": ram_miss.DEMAND_RING,
    "kDemandRecords": ram_miss.DEMAND_RECORDS,
    "kRecordBytes": ram_miss.RECORD_BYTES,
    "kMaxIds": ram_miss.MAX_IDS,
    "kRecSeq": ram_miss.RECORD_FIELDS["seq"],
    "kRecRow": ram_miss.RECORD_FIELDS["row"],
    "kRecCount": ram_miss.RECORD_FIELDS["count"],
    "kRecFlags": ram_miss.RECORD_FIELDS["flags"],
    "kRecFlagCaptured": ram_miss.RECORD_FLAG_CAPTURED,
    "kRecChain": ram_miss.RECORD_FIELDS["chain"],
    "kRecChainHi": ram_miss.RECORD_FIELDS["chain_hi"],
    "kRecEpoch": ram_miss.RECORD_FIELDS["epoch"],
    "kRecProtectCount": ram_miss.RECORD_FIELDS["protect_count"],
    "kRecProtect": ram_miss.RECORD_FIELDS["protect"],
    "kRecLanes": ram_miss.RECORD_FIELDS["lanes"],
    "kLaneBytes": ram_miss.LANE_BYTES,
    "kLaneExpert": ram_miss.LANE_FIELDS["expert"],
    "kLaneSlot": ram_miss.LANE_FIELDS["slot"],
    "kLaneDst": ram_miss.LANE_FIELDS["dst"],
    "kLaneWeight": ram_miss.LANE_FIELDS["weight"],
    "kRecKinds": ram_miss.RECORD_FIELDS["kinds"],
    "kPageBytes": PAGE_BYTES,
    "kKindHitCopy": LaneKind.HIT_COPY,
    "kKindHitSm": LaneKind.HIT_SM,
    "kKindHitCpu": LaneKind.HIT_CPU,
    "kKindMissGpu": LaneKind.MISS_GPU,
    "kKindMissCpu": LaneKind.MISS_CPU,
    "kHotHeaderBytes": ram_miss.HOT_HEADER_BYTES,
    "kHotAlignment": ram_miss.HOT_ALIGNMENT,
    "kHotRecords": ram_miss.HOT_RECORDS,
    "kLeaseRing": lease.RING,
    "kLeaseLanes": lease.LANES,
    "kLeaseBlockAlign": lease.BLOCK_ALIGN,
    "kLeasePieceMask": lease.PIECE_MASK,
    "kLeasePieceMaskLineBytes": lease.PIECE_MASK_LINE_BYTES,
    "kLeaseCopyDone": lease.COPY_DONE,
    "kLeaseCopyDoneBytes": lease.COPY_DONE_BYTES,
    "kLeaseCopyGate": lease.COPY_GATE,
    "kLeaseGateClosed": lease.GATE["closed"],
    "kLeaseGateOpen": lease.GATE["open"],
    "kLeaseGateSeqShift": lease.GATE_SEQ_SHIFT,
    "kLeaseGateSeqMask": lease.GATE_SEQ_MASK,
    "kCopyArmed": lease.COPY_ARMED,
    "kSplit": lease.SPLIT,
    "kLeaseBlockBytes": lease.BLOCK_BYTES,
    "kDeltaBase": lease.DELTA_BASE,
    "kDeltaStride": lease.DELTA_STRIDE,
    "kDeltaTag": lease.DELTA_FIELDS["tag"],
    "kDeltaCount": lease.DELTA_FIELDS["count"],
    "kDeltaStaging": lease.DELTA_FIELDS["staging"],
    "kDeltaEntries": lease.DELTA_FIELDS["entries"],
    "kDeltaMaxEntries": lease.DELTA_MAX_ENTRIES,
}


def test_the_wire_header_is_the_python_layout():
    """The request page and the lease block: one C++ home, equal to Python, and nothing in it Python does not mirror."""
    assert _wire() == PYTHON_WIRE
    assert PYTHON_WIRE["kLeaseRing"] == PYTHON_WIRE["kDemandRecords"] and PYTHON_WIRE["kLeaseLanes"] == PYTHON_WIRE["kMaxIds"]


def test_no_other_source_defines_a_wire_constant():
    """A layout constant re-added beside its user compiles (an ambiguous name errors only where it is used) and then
    drifts; this names the file that re-added it."""
    wire = set(_wire())
    for path in (*host_sources(), *device_sources()):
        clash = wire & set(_NAME.findall(path.read_text()))
        assert not clash, f"{path.name} redefines wire constants {sorted(clash)}: define them only in lease_layout.h"


def test_the_device_state_words_are_the_python_state_words():
    """The device state block agrees with Python's STATE_WORDS; this is the only check of it."""
    device = _constants(*device_sources(), known=_wire())
    state = {
        "kPosted": "posted",
        "kPending": "pending",
        "kEpoch": "epoch",
        "kPendingEpoch": "pending_epoch",
        "kDeadlineLo": "deadline_lo",
        "kDeadlineHi": "deadline_hi",
    }
    assert {word: device[name] for name, word in state.items()} == STATE_WORDS
    assert device["kStateWords"] == len(STATE_WORDS)
    # The service's counters are read positionally into COUNTERS: a counter appended on one side only shifts every name.
    counters = re.search(r"enum Counter : int \{(.*?)\bkCounterCount\b", joined_text(host_sources()), re.S)
    assert len(re.findall(r"^\s*(k\w+)", re.sub(r"//[^\n]*", "", counters.group(1)), re.M)) == len(ram_miss.COUNTERS)


def test_the_stream_kernel_refuses_copy_targets_off_16_byte_alignment():
    """stream_copy_slice falls back to 1-byte copies when a destination or row size is off 16-byte alignment, which
    would silently slow every piece; the segment map refuses such a copy table instead (plan 4.2)."""
    from types import SimpleNamespace

    tables = SimpleNamespace(
        slabs=torch.tensor([[0x10000, 0x20000]]),
        row_bytes=torch.tensor([32, 64]),
        segments=torch.tensor([[0, 0, 0, 32], [1, 0, 32, 64]]),
    )

    def table(*rows):
        return SimpleNamespace(table=torch.tensor(rows, dtype=torch.int64))

    good = ram_miss.stream_segment_map(table([0x10000, 0x40000, 32], [0x20000, 0x80000, 64]), tables, 0)
    assert good.tolist() == [0, 1, 0, 0]
    with pytest.raises(ValueError, match="16-byte"):
        ram_miss.stream_segment_map(table([0x10000, 0x40008, 32], [0x20000, 0x80000, 64]), tables, 0)
    tables.row_bytes = torch.tensor([40, 64])
    with pytest.raises(ValueError, match="16-byte"):
        ram_miss.stream_segment_map(table([0x10000, 0x40000, 40], [0x20000, 0x80000, 64]), tables, 0)


def test_hot_sidecar_layout_and_384_expert_size_match_the_native_abi():
    wire = _wire()
    assert wire["kHotHeaderBytes"] == ram_miss.HOT_HEADER_BYTES == 8
    assert wire["kHotAlignment"] == ram_miss.HOT_ALIGNMENT == 64
    assert wire["kHotRecords"] == ram_miss.HOT_RECORDS == ram_miss.DEMAND_RECORDS == 16
    assert ram_miss.hot_record_bytes(384) == 64
    assert ram_miss.new_hot_page(384, pin=False).numel() == 1024


def test_hot_sidecar_rejects_wrong_stride_before_kernel_launch():
    with pytest.raises(ValueError, match="hot_page"):
        _device(layers=1, experts=384, hot_page=torch.zeros(16 * 128, dtype=torch.uint8))


# Lease-chain PDL (SGLANG_DSV41_ENABLE_LEASE_PDL, LEASE_PROTOCOL.md "PDL"): the chain kernels, their source file, and
# the Python method that launches each. C1 and CC are plain launches.
PDL_KERNELS = {
    "exl3_ram_miss_post_kernel": ("lease_kernels.cuh", "expert_stream_post"),
    "exl3_ram_miss_lease_stream_kernel": ("row_copy_kernels.cuh", "expert_stream_lease_stream"),
    "exl3_ram_miss_lease_copy_wait_kernel": ("row_copy_kernels.cuh", "expert_stream_lease_copy_wait"),
}


def _body_statements(text: str, kernel: str) -> list[str]:
    """The first statements of `kernel`'s body, comments dropped; asserts it is templated on `bool kUsePDL`."""
    match = re.search(r"template <bool kUsePDL>\s*__global__[^{;]*?\b" + kernel + r"\([^)]*\)\s*\{", text)
    assert match, f"{kernel} is not a template <bool kUsePDL> kernel"
    body = text[match.end():]
    lines = [re.sub(r"//.*", "", line).strip() for line in body.splitlines()]
    return [line for line in lines if line][:2]


@pytest.mark.parametrize("kernel", sorted(PDL_KERNELS))
def test_each_chain_kernel_waits_first_and_triggers_right_after_the_wait(kernel):
    """The wait is the first statement: nothing may touch global memory before it. The trigger comes right after it,
    never before: a dependent launched early is safe only because its own wait covers this kernel's completion, and
    this kernel's wait covers its primary's (transitivity). Red when a trigger moves above its wait."""
    text = (CSRC / "expert_stream" / PDL_KERNELS[kernel][0]).read_text()
    assert _body_statements(text, kernel) == [
        "device::PDLWaitPrimary<kUsePDL>();",
        "device::PDLTriggerSecondary<kUsePDL>();",
    ]


@pytest.mark.parametrize("kernel", sorted(PDL_KERNELS))
def test_each_chain_launcher_picks_the_instantiation_and_the_launch_attribute_from_one_flag(kernel):
    text = (CSRC / "expert_stream" / PDL_KERNELS[kernel][0]).read_text()
    launch = re.search(r"\.enable_pdl\(use_pdl != 0\)\(\s*use_pdl != 0 \? " + kernel + r"<true> : " + kernel
                       + r"<false>", text)
    assert launch, f"{kernel}'s launcher does not launch {kernel}<use_pdl> with .enable_pdl(use_pdl)"


def test_no_other_expert_stream_kernel_uses_pdl():
    text = joined_text(device_sources())
    waits = re.findall(r"PDLWaitPrimary<", text)
    triggers = re.findall(r"PDLTriggerSecondary<", text)
    assert len(waits) == len(triggers) == len(PDL_KERNELS)
    assert len(re.findall(r"\.enable_pdl\(", text)) == len(PDL_KERNELS)


def test_the_copy_wait_is_cw_a_stream_wait_on_the_gate_and_a_plain_commit_kernel():
    """CW (PDL), then cuStreamWaitValue32 on area C's gate (GEQ open), then CC with no launch attribute. Red when the
    wait moves, compares against another value, or CC gains PDL (a programmatic edge would let it run before the wait
    node ends)."""
    text = (CSRC / "expert_stream" / "row_copy_kernels.cuh").read_text()
    launcher = text[text.index("static void lease_copy_wait("):]
    arm = launcher.index("exl3_ram_miss_lease_copy_wait_kernel<true>")
    wait = launcher.index("stream_wait_value32()(")
    commit = launcher.index("(exl3_ram_miss_lease_copy_commit_kernel, commit)")
    assert arm < wait < commit
    call = launcher[wait:launcher.index(";", wait)]
    assert "kLeaseCopyGate" in call and "kLeaseGateOpen" in call and "kStreamWaitValueGeq" in call, call
    assert re.search(r"LaunchKernel\(1, device::expert_stream::kBlock, stream\)\(exl3_ram_miss_lease_copy_commit_kernel", launcher)
    body = text[text.index("void exl3_ram_miss_lease_copy_commit_kernel("):]
    body = body[: body.index("\n}\n")]
    assert "PDL" not in body and "while" not in body and "__nanosleep" not in body, "the commit kernel must not wait"
    assert '"cuStreamWaitValue32_v2"' in text


def test_cw_closes_the_gate_after_its_sm_reads_and_fences_before_the_copydone_load():
    """CW's SM reads of the kHitCopy lanes come before the gate close (behind the block barrier), and the Dekker fence
    separates the close from the CopyDone load. No Done word is left: nothing on the host waits for one."""
    text = (CSRC / "expert_stream" / "row_copy_kernels.cuh").read_text()
    body = text[text.index("void exl3_ram_miss_lease_copy_wait_kernel("):]
    body = body[: body.index("\n}\n")]
    reads = body.index("copy_wait_read(")
    barrier = body.index("__syncthreads();", reads)
    close = body.index("kLeaseGateClosed")
    fence = body.index("__threadfence_system();", close)
    load = body.index("kLeaseCopyDone", fence)
    assert reads < barrier < close < fence < load
    assert "kLeaseDone" not in text and "lane_kind" in body


def test_the_python_side_passes_the_pdl_flag_to_exactly_the_chain_launchers():
    tree = ast.parse(Path(ram_miss.__file__).read_text())
    last_args = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("expert_stream_") and isinstance(node.func.value, ast.Call)
                and getattr(node.func.value.func, "attr", None) == "_kernels"):
            last_args.setdefault(node.func.attr, []).append(ast.unparse(node.args[-1]) if node.args else "")
    pdl = {name for name, args in last_args.items() if any("lease_pdl" in a for a in args)}
    assert pdl == {launcher for _, launcher in PDL_KERNELS.values()}
    for name in pdl:
        assert last_args[name] == ["int(self.lease_pdl)"] * len(last_args[name]), name


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


@pytest.mark.parametrize("lease_pdl", [False, True])
def test_the_device_side_passes_its_pdl_flag_to_the_post_launch(lease_pdl):
    kwargs = {} if not lease_pdl else {"lease_pdl": True}
    dev = _device(**kwargs)
    assert dev.lease_pdl is lease_pdl  # off unless asked for
    recorder = _Recorder()
    dev._kernels = lambda: recorder
    _post(dev, _args())
    ((name, args),) = recorder.calls
    assert name == "expert_stream_post" and args[-1] == int(lease_pdl)


def test_the_post_and_s_take_the_row_capacity_as_a_kernel_argument():
    """The post (typing) and S bound every host slot by the capacity in their params, frozen into the captured graph."""
    lease_src = (CSRC / "expert_stream" / "lease_kernels.cuh").read_text()
    rows = (CSRC / "expert_stream" / "row_copy_kernels.cuh").read_text()
    for text, params in ((lease_src, "PostParams"), (rows, "StreamParams")):
        struct = text[text.index(f"struct {params} {{"):]
        assert "uint32_t row_capacity;" in struct[:struct.index("};")], params


def test_the_device_side_passes_each_rows_capacity_to_the_post_launch():
    dev = _device()
    recorder = _Recorder()
    dev._kernels = lambda: recorder
    a = _args()
    for row in (0, 1):
        _post(dev, a, row=row)
    assert [(name, args[22]) for name, args in recorder.calls] == [("expert_stream_post", 5), ("expert_stream_post", 7)]


def test_the_chain_has_no_hit_wait_and_the_post_fills_c1s_compaction():
    """The post types the lanes and writes C1's compacted SM hits itself: there is no W1 to launch."""
    dev = _device()
    assert not hasattr(dev, "hit_wait") and not hasattr(dev, "claimed")
    text = (CSRC / "expert_stream" / "lease_kernels.cuh").read_text()
    assert "hit_wait" not in text and "p.host_rows_1[go]" in text and "p.go_1[0]" in text


def test_the_device_map_starts_empty_with_chain_one():
    """The map is never copied whole (R1-6): ram_slot starts at -1, and map_chain at the attach delta's tag 1."""
    dev = _device()
    bank = dev.map_bank
    assert bool((bank["ram_slot"] == -1).all()) and bool((bank["staging"] == -1).all())
    assert bank["map_chain"].tolist() == [1, 1] and bank["map_applied"].tolist() == [0, 0]
    assert bank["ce_ok"].tolist() == [0, 0] and bank["cpu_ok"].tolist() == [0, 0]


@pytest.mark.parametrize("name", ["exl3_ram_miss_host.cpp", "exl3_ram_miss_host_instr.cpp"])
def test_the_exl3_host_file_is_only_bindings(name):
    """Every export body lives once, in HostExports or HostTestExports (expert_stream/host/ffi_exports.h,
    ffi_test_exports.h); each EXL3 file (one per build) only names its layout, reader and build. Red when a body grows
    back into one of them."""
    path = CSRC / name
    lines = path.read_text().splitlines()
    bodies = [line for line in lines if re.match(r"^\w.*\)\s*\{$", line) and not line.startswith("namespace")]
    assert not bodies, f"{path.name} defines functions: {bodies}"
    assert len(lines) < 40, f"{path.name} has {len(lines)} lines; it should hold only bindings"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
