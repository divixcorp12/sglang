"""The device wrapper refuses pages and slot maps the kernels cannot address (CPU)."""

import ast
import operator
import re
from pathlib import Path

import pytest
import torch

import sglang.kernels.ops.moe.expert_stream_transport as ram_miss
from sglang.kernels.ops.moe import expert_lease_block
from sglang.kernels.ops.moe.expert_stream_transport import PAGE_BYTES, STATE_WORDS, ExpertStreamDevice
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import (
    device_sources,
    host_sources,
    joined_text,
    native_prefetch_source,
    wire_header,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CSRC = Path(ram_miss.__file__).resolve().parents[2] / "jit" / "csrc" / "moe"


def test_state_words_are_distinct_and_dense():
    assert sorted(STATE_WORDS.values()) == list(range(len(STATE_WORDS)))


def test_a_page_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="page"):
        ExpertStreamDevice(torch.zeros(10, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=10, advise=False)


def test_a_slot_map_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="slot_map"):
        ExpertStreamDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((3, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=10, advise=False)


def test_the_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout"):
        ExpertStreamDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cpu", layers=2, timeout_ms=0, advise=False)


def _device(layers=2, experts=4, page=None):
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8) if page is None else page
    slot_map = torch.full((layers, experts), -1, dtype=torch.int32)
    return ExpertStreamDevice(page, slot_map, device="cpu", layers=layers, timeout_ms=10, advise=False)


def _args(lanes=6):
    return dict(
        planned=torch.zeros(lanes, dtype=torch.int64),
        count=torch.zeros(1, dtype=torch.int32),
        routes=torch.full((lanes,), -1, dtype=torch.int64),
        host_rows=torch.zeros(lanes, dtype=torch.int64),
        keep=torch.ones(1, dtype=torch.float32),
        ram_miss=torch.zeros(1, dtype=torch.int64),
    )


def _post(dev, a, row=0, next_row=-1):
    dev.post(row, a["planned"], a["count"], a["routes"], next_row)


def _wait(dev, a, row=0):
    dev.wait(row, a["planned"], a["count"], a["host_rows"], a["keep"], a["ram_miss"])


def test_an_unpinned_page_or_slot_map_is_refused_for_a_cuda_device():
    # Checked before any CUDA call: the kernels read both through UVA.
    with pytest.raises(ValueError, match="pinned"):
        ExpertStreamDevice(torch.zeros(PAGE_BYTES, dtype=torch.uint8), torch.zeros((2, 4), dtype=torch.int32), device="cuda", layers=2, timeout_ms=10, advise=False)


def test_a_row_outside_the_streamed_layers_is_refused():
    dev, a = _device(layers=2), _args()
    for row in (-1, 2):
        with pytest.raises(ValueError, match="row"):
            _post(dev, a, row=row)
        with pytest.raises(ValueError, match="row"):
            _wait(dev, a, row=row)
    for next_row in (-2, 2):
        with pytest.raises(ValueError, match="next_row"):
            _post(dev, a, next_row=next_row)


def test_buffers_of_the_wrong_dtype_are_refused():
    dev = _device()
    for name, dtype in (("planned", torch.int32), ("count", torch.int64), ("routes", torch.int32)):
        a = _args()
        a[name] = a[name].to(dtype)
        with pytest.raises(ValueError, match=name):
            _post(dev, a)
    for name, dtype in (("planned", torch.int32), ("count", torch.int64), ("host_rows", torch.int32), ("keep", torch.float16), ("ram_miss", torch.int32)):
        a = _args()
        a[name] = a[name].to(dtype)
        with pytest.raises(ValueError, match=name):
            _wait(dev, a)


def test_planned_shorter_than_host_rows_is_refused():
    dev, a = _device(), _args()
    a["planned"] = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="planned"):
        _wait(dev, a)


def test_the_device_sequences_continue_from_the_page_heads():
    """A device built over a used page posts the next sequence, not 1, which the thread would never serve."""
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    page[ram_miss.WORDS["demand_head"] : ram_miss.WORDS["demand_head"] + 4].view(torch.int32)[0] = 7
    page[ram_miss.WORDS["advise_head"] : ram_miss.WORDS["advise_head"] + 4].view(torch.int32)[0] = -5  # 2**32 - 5
    dev = _device(page=page)
    assert int(dev.state[STATE_WORDS["posted"]]) == 7
    assert int(dev.state[STATE_WORDS["advised"]]) & 0xFFFFFFFF == 2**32 - 5


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


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


def test_the_wire_header_is_the_python_layout():
    """The page, lease block and prefetch page (LEASE_PROTOCOL.md section 4): one C++ home, equal to Python."""
    wire = _wire()
    python = {
        "kDemandHead": ram_miss.WORDS["demand_head"], "kDemandDone": ram_miss.WORDS["demand_done"],
        "kFatal": ram_miss.WORDS["fatal"], "kAdviseHead": ram_miss.WORDS["advise_head"],
        "kAdviseDone": ram_miss.WORDS["advise_done"], "kBusySeq": ram_miss.WORDS["busy_seq"],
        "kHeartbeat": ram_miss.WORDS["heartbeat"], "kRecordBytes": ram_miss.RECORD_BYTES,
        "kDemandRing": ram_miss.DEMAND_RING, "kDemandRecords": ram_miss.DEMAND_RECORDS,
        "kAdviseRing": ram_miss.ADVISE_RING, "kAdviseRecords": ram_miss.ADVISE_RECORDS, "kMaxIds": ram_miss.MAX_IDS,
        "kServed": ram_miss.STATUS["served"], "kPageBytes": PAGE_BYTES,
        "kHotHeaderBytes": ram_miss.HOT_HEADER_BYTES, "kHotAlignment": ram_miss.HOT_ALIGNMENT,
        "kHotRecords": ram_miss.HOT_RECORDS,
        "kPfReqGen": ram_miss.PREFETCH_FIELDS["req_gen"], "kPfReqRow": ram_miss.PREFETCH_FIELDS["req_row"],
        "kPfReqExpert": ram_miss.PREFETCH_FIELDS["req_expert"], "kPfReqDst": ram_miss.PREFETCH_FIELDS["req_dst"],
        "kPfDoneGen": ram_miss.PREFETCH_FIELDS["done_gen"], "kPfDoneReason": ram_miss.PREFETCH_FIELDS["done_reason"],
        "kPrefetchPageBytes": ram_miss.PREFETCH_PAGE_BYTES, "kPfTagRequest": ram_miss.PREFETCH_TAG_REQUEST,
        "kPfTagCopied": ram_miss.PREFETCH_TAG_COPIED, "kPfTagSkipped": ram_miss.PREFETCH_TAG_SKIPPED,
        "kPfSkipUnarmed": ram_miss.PREFETCH_SKIP_REASONS["unarmed"],
        "kPfSkipNotReady": ram_miss.PREFETCH_SKIP_REASONS["not_ready"],
        "kPfSkipInvalid": ram_miss.PREFETCH_SKIP_REASONS["invalid"],
        "kLeaseBlockAlign": expert_lease_block.BLOCK_ALIGN,
        **_lease_python_constants(), **_lease_device_only_constants(),
    }
    assert {name: wire.get(name) for name in python} == python
    assert wire["kLeaseRing"] == wire["kDemandRecords"] and wire["kLeaseLanes"] == wire["kMaxIds"]


def test_no_other_source_defines_a_wire_constant():
    """A layout constant re-added beside its user compiles (an ambiguous name errors only where it is used) and then
    drifts; this names the file that re-added it."""
    wire = set(_wire())
    for path in (*host_sources(), *device_sources(), native_prefetch_source()):
        clash = wire & set(_NAME.findall(path.read_text()))
        assert not clash, f"{path.name} redefines wire constants {sorted(clash)}: define them only in lease_layout.h"


def test_the_device_state_words_are_the_python_state_words():
    """The two-stage device state block agrees with Python's STATE_WORDS; this is the only check of it."""
    device = _constants(*device_sources(), known=_wire())
    state = {
        "kPosted": "posted",
        "kPending": "pending",
        "kTimeouts": "timeouts",
        "kFailures": "failures",
        "kWaits": "waits",
        "kPolls": "polls",
        "kSticky": "sticky",
        "kAdvised": "advised",
        "kUnservedMisses": "unserved_misses",
        "kEpoch": "epoch",
        "kPendingEpoch": "pending_epoch",
        # Task 6 V1 two-phase (D5, D6). This mapping is hand-maintained and asserted for EQUALITY
        # against STATE_WORDS, so a word added on both sides of the boundary still fails here until
        # it is added here too -- which is the point: this test is the only thing that checks the
        # .cuh and Python agree on the state layout, and it caught D5/D6 adding four words.
        "kDeadlineLo": "deadline_lo",
        "kDeadlineHi": "deadline_hi",
        "kReqFailed": "req_failed",
        "kFailReason": "fail_reason",
        "kStreamPieces": "stream_pieces",
        "kStreamPolls": "stream_polls",
        "kW1Passes": "w1_passes",
        "kCopyWaits": "copy_waits",
        "kCopySpun": "copy_spun",
    }
    assert {word: device[name] for name, word in state.items()} == STATE_WORDS
    # The service's counters are read positionally into COUNTERS: a counter appended on one side only shifts every name.
    counters = re.search(r"enum Counter : int \{(.*?)\bkCounterCount\b", joined_text(host_sources()), re.S)
    assert len(re.findall(r"^\s*(k\w+)", re.sub(r"//[^\n]*", "", counters.group(1)), re.M)) == len(ram_miss.COUNTERS)


def test_the_stream_kernels_fault_words_are_the_device_sources():
    """The stream kernel reads its test-only fault tensor by these indices; a word moved on one side only would
    silently inject a different fault (or none) in the GPU tests that kill M13, M15 and M16."""
    device = _constants(*device_sources())
    names = {
        "abort_block": "kStreamFaultAbortBlock",
        "abort_delay_ns": "kStreamFaultAbortDelay",
        "stall_ns": "kStreamFaultStall",
        "count_delay_ns": "kStreamFaultCountDelay",
    }
    assert {word: device[name] for word, name in names.items()} == ram_miss.STREAM_FAULT_WORDS
    assert device["kStreamFaultWords"] == len(ram_miss.STREAM_FAULT_WORDS)


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
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    slot_map = torch.full((1, 384), -1, dtype=torch.int32)
    with pytest.raises(ValueError, match="hot_page"):
        ExpertStreamDevice(page, slot_map, device="cpu", layers=1, timeout_ms=10,
                          advise=False, hot_page=torch.zeros(16 * 128, dtype=torch.uint8))


def _lease_python_constants():
    from sglang.kernels.ops.moe import expert_lease_block as lease

    return {
        "kLeaseRing": lease.RING,
        "kLeaseLanes": lease.LANES,
        "kLeaseHeaderRing": lease.HEADER["ring"],
        "kLeaseHeaderLanes": lease.HEADER["lanes"],
        "kLeaseHeaderShutdown": lease.HEADER["shutdown"],
        "kLeaseHeaderSlotGenOffset": lease.HEADER["slot_gen_offset"],
        "kLeaseHeaderDOffset": lease.HEADER["d_offset"],
        "kLeaseHeaderPieceOffset": lease.HEADER["piece_offset"],
        "kLeaseHeaderCopyOffset": lease.HEADER["copy_offset"],
        "kLeaseRowTable": lease.ROW_TABLE,
        "kLeaseRowResult": lease.ROW_RESULT,
        "kLeaseRowResultBytes": lease.ROW_RESULT_BYTES,
        "kLeaseRrReady": lease.ROW_RESULT_FIELDS["ready"],
        "kLeaseRrSlotGeneration": lease.ROW_RESULT_FIELDS["slot_generation"],
        "kLeaseRrHostSlot": lease.ROW_RESULT_FIELDS["host_slot"],
        "kLeaseRrExpert": lease.ROW_RESULT_FIELDS["expert"],
        "kLeaseSlotGen": lease.SLOT_GEN,
        "kLeaseLaneRequest": lease.LANE_REQUEST,
        "kLeaseLaneRequestBytes": lease.LANE_REQUEST_BYTES,
        "kLeaseLrGen": lease.LANE_REQUEST_FIELDS["gen"],
        "kLeaseLrCount": lease.LANE_REQUEST_FIELDS["count"],
        "kLeaseLrRow": lease.LANE_REQUEST_FIELDS["row"],
        "kLeaseLrExpert": lease.LANE_REQUEST_FIELDS["expert"],
        "kLeaseLrDst": lease.LANE_REQUEST_FIELDS["dst_slot"],
        "kLeaseLrFlags": lease.LANE_REQUEST_FIELDS["flags"],
        "kLeaseLrFlagCopyEngine": lease.LANE_REQUEST_FLAG_COPY_ENGINE,
        "kLeaseLaneAck": lease.LANE_ACK,
        "kLeaseLaneAckBytes": lease.LANE_ACK_BYTES,
        "kLeaseTerminal": lease.TERMINAL,
        "kLeaseTerminalBytes": lease.TERMINAL_BYTES,
        "kLeaseTermSkippedMask": lease.TERMINAL_FIELDS["skipped_mask"],
        "kLeaseTermReason": lease.TERMINAL_FIELDS["reason"],
        "kLeaseTermGen": lease.TERMINAL_FIELDS["gen"],
        "kLeaseStreamProbe": lease.STREAM_PROBE,
        "kLeaseStreamProbeBytes": lease.STREAM_PROBE_BYTES,
        "kLeaseSmAck": lease.SM_ACK,
        "kLeaseSmAckBytes": lease.SM_ACK_BYTES,
        "kLeaseRowTableBytes": lease.ROW_TABLE_ENTRY_BYTES,
        "kLeasePieceMaskLineBytes": lease.PIECE_MASK_LINE_BYTES,
        "kLeasePieceMaskBytes": lease.PIECE_MASK_BYTES,
        "kLeaseAreaPieceMaskBytes": lease.AREA_PIECE_MASK_BYTES,
        "kLeaseCopyDoneBytes": lease.COPY_DONE_BYTES,
        "kLeaseCdMask": lease.COPY_DONE_FIELDS["mask"],
        "kLeaseCdGen": lease.COPY_DONE_FIELDS["gen"],
        "kLeaseAreaCopyDoneBytes": lease.AREA_COPY_DONE_BYTES,
    }


def _lease_device_only_constants():
    """Tags and Terminal reasons: every one is in the device source; the host may name some (the RowResult ready tags
    it writes), and then must agree, but is not required to define any."""
    from sglang.kernels.ops.moe import expert_lease_block as lease

    return {
        "kLeaseTagDemand": lease.DEMAND_TAG,
        "kLeaseTagReady": lease.READY,
        "kLeaseTagLoading": lease.LOADING,
        "kLeaseTagCopying": lease.COPYING,
        "kLeaseTagCopied": lease.COPIED,
        "kLeaseTagConsumed": lease.CONSUMED,
        "kLeaseTagViolated": lease.VIOLATED,
        "kLeaseTagTerminal": lease.TERMINAL_TAG,
        "kLeaseTagStreamed": lease.STREAM_PROBE_TAG,
        "kLeaseTagSmAck": lease.SM_ACK_TAG,
        "kLeaseReasonTimeout": lease.TERMINAL_REASONS["timeout"],
        "kLeaseReasonAborted": lease.TERMINAL_REASONS["aborted"],
        "kLeaseReasonFailed": lease.TERMINAL_REASONS["failed"],
        "kLeaseReasonIdentity": lease.TERMINAL_REASONS["identity"],
        "kLeaseReasonCount": lease.TERMINAL_REASONS["count"],
    }



# Lease-chain PDL (SGLANG_DSV41_ENABLE_LEASE_PDL, LEASE_PROTOCOL.md "PDL on the lease chain"): the six chain kernels,
# their source file, and the Python method that launches each.
PDL_KERNELS = {
    "exl3_ram_miss_post_kernel": ("lease_kernels.cuh", "expert_stream_post"),
    "exl3_ram_miss_lease_stream_hit_wait_kernel": ("lease_kernels.cuh", "expert_stream_lease_stream_hit_wait"),
    "exl3_ram_miss_lease_stage_ack_kernel": ("lease_kernels.cuh", "expert_stream_lease_stage_ack"),
    "exl3_ram_miss_lease_finalize_kernel": ("lease_kernels.cuh", "expert_stream_lease_finalize"),
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


def test_the_python_side_passes_the_pdl_flag_to_exactly_the_six_chain_launchers():
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
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    slot_map = torch.full((2, 4), -1, dtype=torch.int32)
    kwargs = {} if not lease_pdl else {"lease_pdl": True}
    dev = ExpertStreamDevice(page, slot_map, device="cpu", layers=2, timeout_ms=10, advise=False, **kwargs)
    assert dev.lease_pdl is lease_pdl  # off unless asked for
    recorder = _Recorder()
    dev._kernels = lambda: recorder
    _post(dev, _args())
    ((name, args),) = recorder.calls
    assert name == "expert_stream_post" and args[-1] == int(lease_pdl)


# The row-table capacity word (lease block, row table, +4): written once by the host when the service is built.
CAPACITY_WORD = "kLeaseRowTable + row * kLeaseRowTableBytes + 4"


def test_w1_and_s_take_the_row_capacity_as_a_kernel_argument_not_from_the_pinned_row_table():
    """W1 (lease_hit_wait_body, shared by both hit-wait kernels) and S read the capacity from their params, frozen into
    the captured graph, instead of a host-pinned load on the critical path. Red when either reads the word again."""
    for name in ("lease_device.cuh", "row_copy_kernels.cuh"):
        assert CAPACITY_WORD not in (CSRC / "expert_stream" / name).read_text(), name
    lease = (CSRC / "expert_stream" / "lease_kernels.cuh").read_text()
    rows = (CSRC / "expert_stream" / "row_copy_kernels.cuh").read_text()
    for text, params in ((lease, "HitWaitParams"), (lease, "StreamHitWaitParams"), (rows, "StreamParams")):
        struct = text[text.index(f"struct {params} {{"):]
        assert "uint32_t row_capacity;" in struct[:struct.index("};")], params


def test_the_device_side_passes_the_lease_layouts_row_capacity_to_the_hit_wait_launch():
    from sglang.kernels.ops.moe import expert_lease_block as lease

    layout = lease.lease_layout([5, 7])
    page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    slot_map = torch.full((2, 4), -1, dtype=torch.int32)
    dev = ExpertStreamDevice(page, slot_map, device="cpu", layers=2, timeout_ms=10, advise=False,
                             lease_block=lease.new_lease_block(layout, pin=False), lease_layout=layout)
    recorder = _Recorder()
    dev._kernels = lambda: recorder
    a = _args()
    for row in (0, 1):
        dev.hit_wait(row, a["planned"], a["count"], torch.zeros(6, dtype=torch.int32), 1000)
    assert [(name, args[-1]) for name, args in recorder.calls] == [("expert_stream_lease_hit_wait", 5),
                                                                  ("expert_stream_lease_hit_wait", 7)]


def test_the_exl3_host_file_is_only_bindings():
    """Every export body lives once, in HostExports (expert_stream/host/ffi_exports.h); the EXL3 file only names its
    layout and reader. Red when a body grows back into exl3_ram_miss_host.cpp."""
    path = CSRC / "exl3_ram_miss_host.cpp"
    lines = path.read_text().splitlines()
    bodies = [line for line in lines if re.match(r"^\w.*\)\s*\{$", line) and not line.startswith("namespace")]
    assert not bodies, f"{path.name} defines functions: {bodies}"
    assert len(lines) < 40, f"{path.name} has {len(lines)} lines; it should hold only bindings"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
