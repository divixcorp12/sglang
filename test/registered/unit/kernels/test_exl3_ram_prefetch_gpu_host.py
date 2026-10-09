"""The host's half of the GPU scorer (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design, "The host in GPU mode"),
driven on the test's thread with spec_pump: every record of a row with a target feeds each group's ring; the job waits
for the record's candidate slot, which the test writes as the select kernel would, then reads in the GPU's order the
first per_layer candidates still unmapped and not pooled before, its own group's only (CPU, ChainSim)."""

import json
import struct
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import CAND_FLAG_OVERSIZE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_prefetch_fixtures import (
    LOGITS,
    enable,
    enable_gpu,
    load,
    prefetch_rig,
    trigger,
    write_candidate_slot,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _landed(host, row, group=None):
    return sorted(
        e["expert"] for e in host.spec_pool(row) if e["state"] == "landed" and (group is None or e["group"] == group)
    )


def _record(rig, row=0):
    """A record of `row` with no lane, served: with the GPU scorer it still feeds every group's ring."""
    req = rig.sim.post(row, [])
    assert rig.host.pump() == 1 and rig.sim.wait_handled(req)
    return req


def _counts(rig, *names):
    c = rig.host.counters()
    return tuple(c[n] for n in names)


def test_a_ready_slot_is_read_in_the_gpus_order_past_mapped_and_pooled_experts(tmp_path):
    """4 is mapped, 0 pooled for an earlier record, 2 hot in the host's view (the GPU filtered hot; the host does not):
    per_layer 2 reads 2 and 1. Mutants: drop the host's map check, or its pooled_before check -- red (each spends the
    budget on a candidate spec_read then drops)."""
    rig = prefetch_rig(tmp_path, capacity=9, share=3)
    try:
        page = enable_gpu(rig, per_layer=2)
        load(rig, 1, [4])
        rig.host.spec_place(1, 0)
        rig.host.set_hot(1, [2])
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.5), (4, 1, 1.0), (0, 0, 0.5), (1, 2, 0.25), (3, 3, 0.0)])
        assert rig.host.spec_pump(0) and not rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == [0, 1, 2]
        assert _counts(rig, "spec_issued", "spec_landed", "spec_late", "spec_dropped") == (2, 2, 0, 0)
    finally:
        rig.host.stop()


@pytest.mark.parametrize("slot_seq", [None, 0])
def test_a_slot_never_ready_counts_late_after_the_wait(tmp_path, slot_seq):
    """Never written, or left open (seq word 0): the job waits kCandWait (200 us) and reads nothing."""
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        if slot_seq is not None:
            write_candidate_slot(page, req.seq, [(2, 0, 1.0)], slot_seq=slot_seq)
        start = time.perf_counter()
        assert rig.host.spec_pump(0)
        assert time.perf_counter() - start >= 200e-6
        assert _counts(rig, "spec_late", "spec_issued", "spec_dropped") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_a_lapped_slot_is_dropped(tmp_path):
    """The slot holds the record 16 seqs later: its candidates are another layer's. Mutant: treat any other seq as not
    yet -- red (spec_late instead)."""
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0)], slot_seq=req.seq + 16)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_dropped", "spec_late", "spec_issued") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_a_count_of_zero_is_dropped(tmp_path):
    """An oversize record (prefill) publishes count 0 with the oversize flag."""
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [], flags=CAND_FLAG_OVERSIZE)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_dropped", "spec_late", "spec_issued") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_every_record_of_a_row_with_a_target_feeds_the_ring_and_no_other_does(tmp_path):
    """Row 0 has a target and its record has no CPU lane: a job. Row 1 has none: no job. Mutant: keep the CPU
    scorer's staged-lane condition in GPU mode -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable_gpu(rig)
        _record(rig, 1)
        assert not rig.host.spec_pump(0)
        _record(rig, 0)
        assert rig.host.spec_pump(0)
    finally:
        rig.host.stop()


def test_a_job_whose_target_was_served_first_is_dropped_without_waiting(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0)])
        _record(rig, 1)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_dropped", "spec_late", "spec_issued") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_a_slot_that_never_comes_costs_its_own_job_not_the_next(tmp_path):
    """Review Focus 2. Rows 0 -> 1 and 2 -> 3: row 0's slot never comes, row 2's does. Mutant: return early from the
    ring on a late slot -- red (row 2's job is left)."""
    rig = prefetch_rig(tmp_path, rows=4)
    try:
        targets = torch.tensor([[1, 0], [-1, -1], [3, 1], [-1, -1]], dtype=torch.int64)
        page = enable_gpu(rig, targets=targets)
        _record(rig, 0)
        second = _record(rig, 2)
        write_candidate_slot(page, second.seq, [(2, 0, 1.0)])
        assert rig.host.spec_pump(0) and rig.host.spec_pump(0)
        assert _counts(rig, "spec_late", "spec_issued") == (1, 1) and _landed(rig.host, 3) == [2]
    finally:
        rig.host.stop()


def test_both_groups_read_the_same_list_and_each_its_own_experts(tmp_path):
    """Two groups, per_layer 2 over [2 (group 0), 3 (group 1), 5 (group 1)]: group 0 reads 2, group 1 reads 3."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        page = enable_gpu(rig, per_layer=2)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0), (3, 0, 0.5), (5, 1, 0.25)])
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 0) == [2] and _landed(rig.host, 1, 1) == [3]
        assert _counts(rig, "spec_issued") == (2,)
    finally:
        rig.host.stop()


def test_the_layer_budget_holds_over_both_groups(tmp_path):
    """per_layer 2 over [2, 4 (both group 0), 3 (group 1)]: group 0 reads both, group 1 nothing."""
    rig = prefetch_rig(tmp_path, nodes=2, share=2)
    try:
        page = enable_gpu(rig, per_layer=2)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0), (4, 1, 0.5), (3, 0, 0.25)])
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 0) == [2, 4] and _landed(rig.host, 1, 1) == []
        assert _counts(rig, "spec_issued") == (2,)
    finally:
        rig.host.stop()


def test_a_group_that_reads_first_does_not_shift_the_other_groups_list(tmp_path):
    """Review Focus 1. per_layer 1 over [2 (group 0), 3 (group 1)]: group 0 lands 2 first; group 1 must still count 2
    and read nothing. Mutant (ram_prefetch.h pooled_before): count an entry issued for the same seq -- red (group 1
    reads 3)."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        page = enable_gpu(rig, per_layer=1)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0), (3, 0, 0.5)])
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1, 0) == [2]
        assert rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 1) == [] and _counts(rig, "spec_issued") == (1,)
    finally:
        rig.host.stop()


def test_the_cpu_scorer_never_counts_late(tmp_path):
    """The default scorer's path is Phase 1's: a CPU record scores and reads; nothing waits for a slot."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_late", "spec_issued") == (0, 1)
    finally:
        rig.host.stop()


def test_spec_submit_carries_the_gpus_rank_and_margin_and_the_wait_is_metered(tmp_path, monkeypatch):
    """gen = rank << 32 | the margin's fp32 bits, from the slot; spec_scored / spec_score_ns count the host's wait."""
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 3, 0.75)])
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert c["spec_scored"] == 1 and c["spec_score_ns"] > 0
    finally:
        rig.host.stop()
    files = list(tmp_path.glob("jobs.*.exl3-spec0.*.jsonl"))
    assert len(files) == 1
    events = list(map(json.loads, files[0].read_text().splitlines()[1:-1]))
    submit = next(e for e in events if e["event"] == "spec_submit")
    assert (submit["a"], submit["seq"]) == (2, req.seq)
    assert submit["gen"] == (3 << 32) | struct.unpack("<I", struct.pack("<f", 0.75))[0]


def test_a_gpu_scorer_takes_no_host_gates_and_a_page_of_the_slots_size(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        targets = torch.tensor([[1, 0], [-1, -1]], dtype=torch.int64)
        cores = [[]]
        with pytest.raises(ValueError, match="gates=None and bias=None"):
            rig.host.enable_ram_prefetch(
                targets, torch.zeros((1, 6, 8), dtype=torch.bfloat16), torch.zeros((1, 6)), top_k=2, per_token=1,
                per_layer=1, cores=cores, candidates=torch.zeros(2048, dtype=torch.uint8),
            )
        with pytest.raises(ValueError, match="2048 bytes"):
            rig.host.enable_ram_prefetch(
                targets, None, None, top_k=2, per_token=1, per_layer=1, cores=cores,
                candidates=torch.zeros(1024, dtype=torch.uint8),
            )
    finally:
        rig.host.stop()
