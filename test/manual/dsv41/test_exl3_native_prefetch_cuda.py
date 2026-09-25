"""Native next-layer prefetch (plan 2026-09-25-dsv41-native-prefetch): the plan and commit kernels, and the whole path
against the real C++ service and its copy engine. GPU only.

- The plan kernel against a Python reference of the rule: the router's biased score, top 6, the first that is neither
  resident nor missing from the pinned tier, and the first extended victim that holds none of the six.
- The commit kernel: COPIED maps, SKIPPED does not, a missing done word fails stop and takes the slot out.
- Byte identity: the same decode steps replayed from captured graphs with prefetch on and off give the same bytes to
  the "MoE" (every routed expert's destination row), equal to the checkpoint's rows.

Run on divix01 holding cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import random

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import exl3_native_prefetch as kernels  # noqa: E402
from sglang.kernels.ops.moe.exl3_ram_miss import (  # noqa: E402
    PREFETCH_FIELDS,
    PREFETCH_TAG_COPIED,
    PREFETCH_TAG_REQUEST,
    PREFETCH_TAG_SKIPPED,
    WORDS,
    new_page,
    new_prefetch_page,
)
from test_exl3_piece_stream_cuda import EXPERTS, TOP_K, StreamService  # noqa: E402

READY = 3
MASK56 = (1 << 56) - 1
COUNTER = {name: i for i, name in enumerate(kernels.NATIVE_PREFETCH_COUNTERS)}


def reference_plan(logits, bias, mapping, slot_to_expert, victims, valid, ram_map):
    """(expert, slot) the plan kernel must post, or None: the rule in exl3_native_prefetch.cuh, in fp32."""
    x = logits.float()
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    key = torch.sqrt(sp) + bias.float()
    key = torch.where(torch.isnan(key), torch.full_like(key, float("inf")), key)
    order = sorted(range(len(key)), key=lambda e: (-float(key[e]), e))[: kernels.TOPK]
    candidate = next((e for e in order if mapping[e] < 0 and ram_map[e] >= 0), None)
    if candidate is None:
        return None
    for slot, ok in zip(victims.tolist(), valid.tolist()):
        if ok and slot_to_expert[slot] not in order:
            return candidate, slot
    return None


class Buffers:
    def __init__(self, experts=64, slots=12, extra=7):
        dev = "cuda"
        self.page = new_page(pin=True)
        self.pf = new_prefetch_page(pin=True)
        self.logits = torch.zeros(experts, dtype=torch.float32, device=dev)
        self.bias = torch.zeros(experts, dtype=torch.bfloat16, device=dev)
        self.mapping = torch.full((experts + 1,), -1, dtype=torch.int64, device=dev)
        self.slot_to_expert = torch.full((slots + 1,), -1, dtype=torch.int64, device=dev)
        self.slot_state = torch.zeros(slots + 1, dtype=torch.uint8, device=dev)
        self.slot_gen = torch.zeros(slots + 1, dtype=torch.int64, device=dev)
        self.victims = torch.zeros(extra, dtype=torch.int64, device=dev)
        self.valid = torch.zeros(extra, dtype=torch.bool, device=dev)
        self.ram = torch.full((experts,), -1, dtype=torch.int32).pin_memory()
        self.pending = torch.zeros(kernels.PENDING_WORDS, dtype=torch.int64, device=dev)
        self.gen = torch.zeros(1, dtype=torch.int64, device=dev)
        self.counters = torch.zeros(len(kernels.NATIVE_PREFETCH_COUNTERS), dtype=torch.int64, device=dev)
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device=dev)

    def plan(self, row=3):
        kernels.plan(self.logits, self.bias, self.mapping, self.slot_to_expert, self.victims, self.valid, self.ram,
                     self.page, self.pf, row, self.pending, self.gen, self.counters)

    def commit(self, timeout_ns=200_000_000):
        kernels.commit(self.pending, self.pf, self.page, 0, timeout_ns, self.mapping, self.slot_to_expert,
                       self.slot_state, self.slot_gen, READY, self.routes, self.counters)

    def pf_i32(self, name):
        o = PREFETCH_FIELDS[name]
        return int(self.pf[o : o + 4].view(torch.int32)[0])

    def pf_u64(self, name):
        o = PREFETCH_FIELDS[name]
        return int(self.pf[o : o + 8].view(torch.int64)[0]) & ((1 << 64) - 1)

    def set_done(self, tag, gen):
        o = PREFETCH_FIELDS["done_gen"]
        self.pf[o : o + 8].view(torch.int64)[0] = (tag << 56) | gen


def _randomize(b, rng, experts=64, slots=12):
    b.logits.copy_(torch.randn(experts, generator=rng) * 3)
    b.bias.copy_((torch.randn(experts, generator=rng) * 0.5).to(torch.bfloat16))
    b.mapping.fill_(-1)
    b.slot_to_expert.fill_(-1)
    resident = torch.randperm(experts, generator=rng)[: slots - 1].tolist()
    for slot, e in enumerate(resident):
        b.mapping[e] = slot
        b.slot_to_expert[slot] = e
    b.ram.copy_(torch.where(torch.rand(experts, generator=rng) < 0.7, 1, -1).to(torch.int32))
    b.victims.copy_(torch.randperm(slots, generator=rng)[: b.victims.numel()])
    b.valid.copy_(torch.rand(b.victims.numel(), generator=rng) < 0.8)


def test_the_plan_kernel_follows_the_rule_on_random_cases():
    b = Buffers()
    rng = torch.Generator().manual_seed(1)
    posted = 0
    for case in range(300):
        _randomize(b, rng)
        want = reference_plan(b.logits.cpu(), b.bias.cpu(), b.mapping.cpu().tolist(), b.slot_to_expert.cpu().tolist(),
                              b.victims.cpu(), b.valid.cpu(), b.ram.tolist())
        before = int(b.gen.item())
        b.plan(row=5)
        torch.cuda.synchronize()
        pending = b.pending.cpu().tolist()
        if want is None:
            assert pending[0] == 0, case
            assert int(b.gen.item()) == before
            continue
        posted += 1
        assert pending[:3] == [1, want[0], want[1]], (case, pending, want)
        assert pending[3] == before + 1
        assert b.pf_u64("req_gen") == (PREFETCH_TAG_REQUEST << 56) | pending[3]
        assert (b.pf_i32("req_row"), b.pf_i32("req_expert"), b.pf_i32("req_dst")) == (5, want[0], want[1])
    assert 100 < posted < 300, posted  # both branches exercised
    c = b.counters.cpu().tolist()
    assert c[COUNTER["posted"]] == posted and c[COUNTER["ram_filtered"]] > 0 and c[COUNTER["no_candidate"]] > 0


def test_the_plan_posts_nothing_once_the_page_is_fatal():
    b = Buffers()
    _randomize(b, torch.Generator().manual_seed(2))
    b.ram.fill_(1)
    b.valid.fill_(True)
    o = WORDS["fatal"]
    b.page[o : o + 4].view(torch.int32)[0] = 7
    b.plan()
    torch.cuda.synchronize()
    assert int(b.pending[0]) == 0 and b.pf_u64("req_gen") == 0


def _one_pending(b, expert=10, slot=4, old=33):
    b.mapping.fill_(-1)
    b.slot_to_expert.fill_(-1)
    b.mapping[old] = slot
    b.slot_to_expert[slot] = old
    b.pending.copy_(torch.tensor([1, expert, slot, 42, 0, 0]))


def test_commit_maps_a_copied_row_and_counts_it_used():
    b = Buffers()
    _one_pending(b)
    b.routes[:2] = torch.tensor([10, 3])
    b.set_done(PREFETCH_TAG_COPIED, 42)
    b.commit()
    torch.cuda.synchronize()
    assert int(b.mapping[10]) == 4 and int(b.mapping[33]) == -1 and int(b.slot_to_expert[4]) == 10
    assert int(b.slot_state[4]) == READY and int(b.slot_gen[4]) == 1 and int(b.pending[0]) == 0
    c = b.counters.cpu().tolist()
    assert c[COUNTER["copied"]] == 1 and c[COUNTER["used"]] == 1


def test_commit_leaves_residency_alone_for_a_skipped_row():
    b = Buffers()
    _one_pending(b)
    b.set_done(PREFETCH_TAG_SKIPPED, 42)
    b.commit()
    torch.cuda.synchronize()
    assert int(b.mapping[33]) == 4 and int(b.mapping[10]) == -1 and int(b.slot_to_expert[4]) == 33
    assert int(b.counters[COUNTER["skipped"]]) == 1


def test_commit_ignores_an_older_done_word_times_out_fails_stop_and_takes_the_slot_out():
    b = Buffers()
    _one_pending(b)
    b.set_done(PREFETCH_TAG_COPIED, 41)  # the previous request's
    b.commit(timeout_ns=20_000_000)
    torch.cuda.synchronize()
    o = WORDS["fatal"]
    assert int(b.page[o : o + 4].view(torch.int32)[0]) != 0
    assert int(b.mapping[33]) == -1 and int(b.mapping[10]) == -1 and int(b.slot_to_expert[4]) == -1
    assert int(b.counters[COUNTER["aborted"]]) == 1


# ---- byte identity, against the real service and copy engine -------------------------------------------------------

SLOTS = TOP_K  # the harness's destination rows are the hot slots
ROUTES = 3
STEPS = 120
# ~40 ms of copy engine per job: longer than the host takes between two steps, so a commit that does not wait maps a
# slot the next step reads before its bytes land.
BALLAST_BYTES = 512 << 20


class Decode:
    """Captured decode steps over one streamed row. G1 is the commit (prefetch on only); G2 the demand chain, the
    "MoE" gather of every routed expert's destination row, and (prefetch on) the plan for the next step. Between them
    the host plans the demand from the device residency, as DIRECT's planner does on the device in production."""

    def __init__(self, tmp_path, prefetch: bool):
        self.s = StreamService(tmp_path, copy_engine=True, native_prefetch=prefetch)
        self.prefetch = prefetch
        dev = "cuda"
        self.mapping = torch.full((EXPERTS + 1,), -1, dtype=torch.int64, device=dev)
        self.slot_to_expert = torch.full((SLOTS + 1,), -1, dtype=torch.int64, device=dev)
        self.slot_state = torch.zeros(SLOTS + 1, dtype=torch.uint8, device=dev)
        self.slot_gen = torch.zeros(SLOTS + 1, dtype=torch.int64, device=dev)
        self.route_slots = torch.zeros(ROUTES, dtype=torch.int64, device=dev)
        self.out = {n: torch.zeros((ROUTES,) + tuple(self.s.dest[n].shape[1:]), dtype=self.s.dest[n].dtype, device=dev)
                    for n in self.s.names}
        self.logits = torch.full((EXPERTS,), -10.0, dtype=torch.float32, device=dev)
        self.bias = torch.zeros(EXPERTS, dtype=torch.bfloat16, device=dev)
        self.victims = torch.zeros(7, dtype=torch.int64, device=dev)
        self.valid = torch.zeros(7, dtype=torch.bool, device=dev)
        self.pending = torch.zeros(kernels.PENDING_WORDS, dtype=torch.int64, device=dev)
        self.gen = torch.zeros(1, dtype=torch.int64, device=dev)
        self.counters = torch.zeros(len(kernels.NATIVE_PREFETCH_COUNTERS), dtype=torch.int64, device=dev)
        self.lru = list(range(SLOTS))  # host LRU of slots, oldest first
        self.s.plan([])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.g1 = torch.cuda.CUDAGraph() if prefetch else None
        self.g2 = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            if prefetch:
                with torch.cuda.graph(self.g1, stream=stream):
                    self._commit()
            with torch.cuda.graph(self.g2, stream=stream):
                self._step()
        torch.cuda.synchronize()

    def _commit(self):
        kernels.commit(self.pending, self.s.prefetch_page, self.s.page, int(self.s.host.lease_block.data_ptr()),
                       2_000_000_000, self.mapping, self.slot_to_expert, self.slot_state, self.slot_gen, READY,
                       self.s.routes, self.counters)

    def _step(self):
        s = self.s
        s.post()
        s.hit_wait()
        s.copy1()
        s.ack1()
        s.stream()
        s.ack2()
        s.copy_wait()
        s.finalize()
        s.total()
        for n in s.names:
            torch.index_select(s.dest[n], 0, self.route_slots, out=self.out[n])
        if self.prefetch:
            kernels.plan(self.logits, self.bias, self.mapping, self.slot_to_expert, self.victims, self.valid,
                         s.slot_map[s.row], s.page, s.prefetch_page, s.row, self.pending, self.gen, self.counters)

    def run(self, routes, predicted):
        s = self.s
        s.routes.fill_(-1)
        s.routes[: len(routes)] = torch.tensor(routes, dtype=torch.int64)  # the commit counts a used row against them
        if self.prefetch:
            self.g1.replay()
            torch.cuda.synchronize()
        mapping = self.mapping.cpu().tolist()
        slot_to_expert = self.slot_to_expert.cpu().tolist()
        misses = [e for e in routes if mapping[e] < 0]
        hit_slots = {mapping[e] for e in routes if mapping[e] >= 0}
        free = [slot for slot in self.lru if slot not in hit_slots][: len(misses)]
        s.plan(misses, routes=routes)
        s.dest_slots[: len(free)] = torch.tensor(free, dtype=torch.int32)
        where = {e: mapping[e] for e in routes if mapping[e] >= 0}
        where.update(zip(misses, free))
        self.route_slots.copy_(torch.tensor([where[e] for e in routes]))
        for slot in [where[e] for e in routes]:
            self.lru.remove(slot)
            self.lru.append(slot)
        # The next step's prediction and victims: DIRECT's order past the "shortlist" (the oldest 2 slots).
        self.logits.fill_(-10.0)
        self.logits[predicted] = 10.0
        order = [slot for slot in self.lru if slot not in set(where.values())]
        victims = order[2:] + [SLOTS] * 7
        self.victims.copy_(torch.tensor(victims[:7]))
        self.valid.copy_(torch.tensor([v < SLOTS for v in victims[:7]]))
        # DIRECT's commit, on the host and before the replay: the plan in G2 must see this step's inserts, as the next
        # layer's plan sees the residency its own gather will read.
        for e, slot in zip(misses, free):
            old = slot_to_expert[slot]
            if old >= 0:
                self.mapping[old] = -1
            self.mapping[e] = slot
            self.slot_to_expert[slot] = e
            slot_to_expert[slot] = e
        self.g2.replay()
        torch.cuda.synchronize()
        assert s.keep.item() == 1.0, (s.counters(), s.stats())
        return {n: t.cpu().clone() for n, t in self.out.items()}, misses

    def close(self):
        self.s.close()


@pytest.mark.parametrize("ballast", [False, True])
def test_replay_with_prefetch_on_is_byte_identical_to_replay_with_it_off(tmp_path, ballast):
    """Each step routes 3 of 16 experts; the prefetch predicts the next step's first expert half the time and a random
    one otherwise, so copied rows are used, wasted and (outside the 8-row pinned tier) filtered. The bytes the MoE reads
    must be the checkpoint's in both arms; prefetch on must also remove demand misses. With the ballast every copy job
    completes ~40 ms after it is issued, so a commit that does not wait for the done word maps a slot before its bytes
    land (mutant: the commit kernel's wait removed -- red on the byte check)."""
    rng = random.Random(5)
    steps = [rng.sample(range(EXPERTS), ROUTES) for _ in range(STEPS + 1)]
    predicted = [steps[i + 1][0] if rng.random() < 0.5 else rng.randrange(EXPERTS) for i in range(STEPS)]
    outs, misses = {}, {}
    for arm in (False, True):
        (tmp_path / f"arm{int(arm)}").mkdir()
        d = Decode(tmp_path / f"arm{int(arm)}", prefetch=arm)
        ballast_src = torch.empty(BALLAST_BYTES, dtype=torch.uint8).pin_memory() if ballast else None
        ballast_dst = torch.empty(BALLAST_BYTES, dtype=torch.uint8, device="cuda") if ballast else None
        if ballast:
            d.s.host.copy_engine_ballast(ballast_dst, ballast_src)
        try:
            expected = d.s.expected(list(range(EXPERTS)))
            outs[arm], misses[arm] = [], 0
            for i in range(STEPS):
                out, missed = d.run(steps[i], predicted[i])
                misses[arm] += len(missed)
                for lane, e in enumerate(steps[i]):
                    for n in d.s.names:
                        assert torch.equal(out[n][lane].view(torch.uint8), expected[e][n].view(torch.uint8)), (arm, i, e, n)
                outs[arm].append(out)
            if arm:
                d.g1.replay()  # settle the last step's prefetch
                torch.cuda.synchronize()
                assert d.s.until(lambda: d.s.counters()["prefetch_requests"] == int(d.counters[COUNTER["posted"]]))
                c, dc = d.s.counters(), d.counters.cpu().tolist()
                assert dc[COUNTER["posted"]] > 15, dc
                assert dc[COUNTER["copied"]] == c["prefetch_copied"] > 5, (dc, c)
                assert dc[COUNTER["aborted"]] == 0 and c["copy_errors"] == 0, (dc, c)
                assert c["prefetch_skipped_not_ready"] + dc[COUNTER["ram_filtered"]] > 0, (dc, c)
                assert dc[COUNTER["used"]] > 0 and c["prefetch_used"] + c["prefetch_wasted"] > 0, (dc, c)
        finally:
            if ballast:
                d.s.host.copy_engine_ballast(None, None)
            d.close()
    for i in range(STEPS):
        for n in outs[False][i]:
            assert torch.equal(outs[False][i][n].view(torch.uint8), outs[True][i][n].view(torch.uint8)), (i, n)
    assert misses[True] < misses[False], misses
