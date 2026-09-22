"""Explicit-state model of the Task 5 lease protocol (LEASE_PROTOCOL.md), small enough to search exhaustively.

Two threads and a store: the *device* (one captured stream: post, wait, copy, acknowledge, consume) and the
*service* (one thread: observe, reserve slots, load, grant leases, publish, retire), over a handful of pinned
slots that can also be evicted and reloaded by an advisory whenever the service is idle. Every step is one
atomic action; a breadth-first search visits every interleaving up to the configured bound and reports the
shortest trace to a violation.

What the model is for. The protocol's claim is invariant I1: the bytes of a slot do not change between the
service publishing a lane and the service retiring that lane's lease. The model checks I1 directly (ghost
state: which slot each device lane holds), and separately the observable that this codebase once got wrong:
a request accepted as successful whose delivered bytes are not the requested experts'. A model that cannot
find a known bug is too weak, so ``Config`` also builds the *mutants* (one protocol rule removed each) and the
protocol *as it exists today*; the tests require the model to find what is known to be wrong.

Task 8's side (LEASE_PROTOCOL.md 17.1): a *promoter* takes host leases on ready slots, copies from them on
its own stream and releases them, and an *eager* caller pauses the service and assigns a slot by hand
(``before_host_use``). ``pause_counts_host`` and ``copy_waits_on_serving`` are the two rules Task 8 depends on:
the pause counts graph-lane leases only, and a lease holder's copy never waits on the serving stream.

What it does not model. It is sequentially consistent except for one relaxation: the device's stores to the
acknowledgement and terminal words reach the service in any order across different words (same word stays in
order). Missing fences, ``ld.global.nc`` staleness and PCIe reordering of the service's stores are outside it,
so a pass is a statement about the protocol's logic, not about the memory model (LEASE_PROTOCOL.md section 6).
Sizes are tiny on purpose: the 32-bit sequence is ``seq_space`` values and starts near its wrap.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

# Device program counters.
IDLE, P_LREQ, P_REC, P_HEAD, WAIT, STATUS, RD, COMMIT, COPY_B, COPY_E, ACK, MOE, HALT, ERR = range(14)
# Service program counters.
S_IDLE, S_LOAD_B, S_LOAD_E, S_GRANT, S_STATUS, S_DONE, S_ADV_B, S_ADV_E = range(8)
# Slot states.
FREE, LOADING, READY = range(3)
# Lane states in the service's Outstanding entry.
NONE, GRANTED, ACKED, VOID = range(4)
CONSUMED, VIOLATED = 1, 2
TORN = -1
# Promoter program counters, and the eager caller's.
H_IDLE, H_COPY_B, H_COPY_E, H_REL = range(4)
E_IDLE, E_TAKE, E_WB, E_WE, E_RESUME, E_SYNC = range(6)


@dataclass(frozen=True)
class Config:
    """The protocol under test and the size of the world. Defaults are the designed protocol."""

    lanes: int = 2
    slots: int = 2
    experts: int = 3
    ring: int = 2
    seq_space: int = 7  # sequences 1..seq_space; the counter is modulo seq_space + 1 and skips 0
    start: int = 6  # both sides start having served this sequence, so the run crosses the wrap
    requests: int = 3  # requests the device posts
    advisories: int = 1  # evict-and-reload pressure the service applies while idle
    timeouts: bool = True  # the device may give up waiting at any point
    io_faults: bool = True  # a row read may fail
    cuda_error: bool = False  # the device may die mid-kernel
    shutdown: bool = False  # the service may be shut down, and Python then frees or quarantines
    max_states: int = 6_000_000
    promotions: int = 0  # host-lease acquisitions the promoter makes (Task 8); 0 leaves the actor out
    eager_uses: int = 0  # pause-assign-resume cycles of the eager caller; 0 leaves the actor out
    host_guard: bool = True  # eviction by the service respects host leases
    eager_host_guard: bool = True  # the eager assign respects host leases
    pause_counts_host: bool = False  # True: the pause also waits for host leases (R2 says it must not)
    copy_waits_on_serving: bool = False  # True: the promotion copy waits on the serving stream (A3 says it must not)
    host_release: bool = True  # False: the promoter never releases its lease
    blocking_eager: bool = False  # the eager caller is the scheduler thread and blocks in synchronize() before pausing
    poll_release: bool = False  # host leases are acquired and released by the scheduler thread's poll step, not a callback
    host_admission_closed: bool = True  # after shutdown no new host lease is taken
    free_waits_executor: bool = True  # Python frees only once the promotion copies and leases are done
    stale_ack: Optional[tuple] = None  # (idx, lane, gen): an acknowledgement left by an idle lane a full sequence cycle ago
    menu: Optional[tuple] = None  # the lane lists a request may take (default: every list of up to ``lanes`` experts)

    # Protocol rules. Each False (or True for the mutants) is one rule removed.
    leases: bool = True  # eviction requires leases == 0
    fail_closed: bool = True  # a failed wait sets the copy count to 0
    ack_after_copy: bool = True  # False: the acknowledgement is published at commit
    gen64: bool = True  # False: acknowledgements and readiness are keyed by the 32-bit sequence only
    skip0: bool = True  # the service skips sequence 0 on wrap as the device does
    skip0_lap: bool = True  # ... and also when a lap resume lands on it (head - ring + 2 can be 0)
    echo_gen: bool = True  # the service takes the request generation from the device's lane request; False: it counts epochs itself
    defer_reuse: bool = True  # a request slot is served only when its previous lease row is retired
    detector: bool = True  # the acknowledgement re-checks the slot generation
    exclusion: bool = False  # an advisory never evicts while the device is between wait and consume (today's implicit rule)
    free_needs_sync: bool = True  # Python frees only after the device has completed
    protocol: bool = True  # False: none of leases, row results, acknowledgements, terminal (today)
    retire: bool = True  # False: the service never retires a lease (the liveness mutant)
    defer_leased: bool = True  # False: a demand whose only victims are leased fails instead of waiting for the acks

    def modulus(self) -> int:
        return self.seq_space + 1


def today(**kw) -> Config:
    """The service and kernels as they are before Task 5: no leases, the device reads the map, the copy runs
    whatever the wait returned, and the only protection is the implicit temporal exclusion."""
    base = dict(protocol=False, leases=False, fail_closed=False, exclusion=True, detector=False)
    base.update(kw)
    return Config(**base)


def reached(cfg: Config, observed: int, seq: int) -> bool:
    """The signed-difference compare of the real code, over the model's modulus."""
    m = cfg.modulus()
    return (observed - seq) % m < m // 2


def next_seq_device(cfg: Config, seq: int, epoch: int) -> tuple[int, int]:
    nxt = (seq + 1) % cfg.modulus()
    if nxt == 0:
        return 1, epoch + 1
    return nxt, epoch


def next_seq_service(cfg: Config, seq: int, epoch: int) -> tuple[int, int]:
    nxt = (seq + 1) % cfg.modulus()
    if nxt == 0:
        if cfg.skip0:
            return 1, epoch + 1
        return 0, epoch + 1  # the unfixed loop: a phantom sequence 0 is expected next
    return nxt, epoch


def ring_index(cfg: Config, seq: int) -> int:
    """(seq - 1) % ring in unsigned arithmetic: the modulus plays the part of 2**32."""
    return ((seq - 1) % cfg.modulus()) % cfg.ring


def gen_key(cfg: Config, epoch: int, seq: int):
    return (epoch, seq) if cfg.gen64 else (0, seq)


# State: a flat tuple; field indices below keep the transition code readable.
FIELDS = (
    "head done fatal shut rec lreq rowres sgen term ack pend map "
    "sst sexp slease scont "
    "nd sep spc sreq splan out advleft overruns freed quar dead "
    "dseq dep dk dpc dreq di dok dgo dctx dhold dread ddel dstop "
    "hl hpc hslot hexp hsnap hneed hleft paused epc eplan eleft "
    "viol"
).split()
IX = {name: i for i, name in enumerate(FIELDS)}


def put(t: tuple, i: int, v) -> tuple:
    return t[:i] + (v,) + t[i + 1 :]


class Model:
    def __init__(self, cfg: Config):
        self.stats: dict[str, int] = {}
        # The real counter is 2**32 wide and the ring divides it; in flight requests are far below half of it.
        # The model keeps those ratios, or ``reached`` would misread a lag as a lead.
        assert cfg.modulus() % cfg.ring == 0, "the ring must divide the sequence modulus, as 16 divides 2**32"
        assert cfg.requests < cfg.modulus() // 2, "more requests in flight than half the sequence range"
        self.c = cfg

    # ---- construction ----
    def initial(self) -> tuple:
        c = self.c
        idle_lanes = tuple(None for _ in range(c.lanes))
        s = dict(
            head=c.start, done=c.start, fatal=0, shut=0,
            rec=tuple((0, 0, 0, 0) for _ in range(c.ring)),
            lreq=tuple(None for _ in range(c.ring)),
            rowres=tuple(idle_lanes for _ in range(c.ring)),
            sgen=tuple(0 for _ in range(c.slots)),
            term=tuple(None for _ in range(c.ring)),
            ack=tuple(idle_lanes for _ in range(c.ring)),
            pend=(),
            map=tuple(-1 for _ in range(c.experts)),
            sst=tuple(FREE for _ in range(c.slots)),
            sexp=tuple(-1 for _ in range(c.slots)),
            slease=tuple(0 for _ in range(c.slots)),
            scont=tuple((-1, 0) for _ in range(c.slots)),
            nd=next_seq_service(c, c.start, 0)[0] if c.start else 1, sep=0, spc=S_IDLE, sreq=None, splan=(),
            out=tuple(None for _ in range(c.ring)), advleft=c.advisories, overruns=0, freed=0, quar=0, dead=0,
            dseq=c.start, dep=0, dk=0, dpc=IDLE, dreq=None, di=0, dok=True, dgo=0,
            dctx=idle_lanes, dhold=tuple(-1 for _ in range(c.lanes)), dread=idle_lanes,
            ddel=idle_lanes, dstop=0,
            hl=tuple(0 for _ in range(c.slots)), hpc=H_IDLE, hslot=-1, hexp=-1, hsnap=None, hneed=-1,
            hleft=c.promotions, paused=0, epc=E_IDLE, eplan=(), eleft=c.eager_uses,
            viol=(),
        )
        if c.stale_ack is not None:
            idx, lane, gen = c.stale_ack
            row = list(s["ack"][idx])
            row[lane] = (gen, CONSUMED)
            s["ack"] = put(s["ack"], idx, tuple(row))
        # The service opens as ``done + 1`` and skips 0 (host.cpp RamTier::open).
        nd = (c.start + 1) % c.modulus()
        s["nd"] = 1 if nd == 0 else nd
        s["sep"] = 1 if nd == 0 else 0
        s["dep"] = 1 if nd == 0 else 0
        return tuple(s[f] for f in FIELDS)

    # ---- helpers ----
    def note(self, event: str) -> None:
        """Coverage: which situations the exploration actually reached (not part of the state)."""
        self.stats[event] = self.stats.get(event, 0) + 1

    def d(self, s: tuple) -> dict:
        return dict(zip(FIELDS, s))

    def flag(self, s: dict, name: str) -> None:
        if name not in s["viol"]:
            s["viol"] = s["viol"] + (name,)

    def emit(self, s: dict) -> tuple:
        if s["pend"]:
            # Stores to different words may land in any order, so only the per-word order is state.
            s["pend"] = tuple(sorted(s["pend"], key=lambda w: (w[0], w[1], w[2])))
        if s["dpc"] in (IDLE, HALT) and s["dreq"] is not None:
            blank = tuple(None for _ in range(self.c.lanes))
            s["dreq"], s["dctx"], s["dread"], s["ddel"], s["dgo"], s["di"] = None, blank, blank, blank, 0, 0
        return tuple(s[f] for f in FIELDS)

    def resident(self, s: dict, expert: int) -> int:
        for slot in range(self.c.slots):
            if s["sst"][slot] == READY and s["sexp"][slot] == expert:
                return slot
        return -1

    def holds(self, s: dict, slot: int) -> bool:
        return slot in s["dhold"]

    def demand_visible(self, s: dict) -> bool:
        return s["head"] != 0 and reached(self.c, s["head"], s["nd"])

    def in_device_window(self, s: dict) -> bool:
        """From the device posting its demand to the end of the consume. Today an advisory for the layer is stale
        once that layer's demand is posted, so no eviction on the row starts inside this window."""
        return s["dpc"] in (WAIT, STATUS, RD, COMMIT, COPY_B, COPY_E, ACK, MOE)

    # ---- the whole transition relation ----
    def successors(self, s: tuple) -> Iterator[tuple[str, tuple]]:
        st = self.d(s)
        if st["dead"]:
            return
        yield from self.device(st)
        yield from self.service(st)
        yield from self.deliveries(st)
        yield from self.environment(st)
        yield from self.host(st)
        yield from self.eager(st)

    def deliveries(self, st: dict):
        seen = set()
        for j, (kind, idx, lane, val) in enumerate(st["pend"]):
            key = (kind, idx, lane)
            if key in seen:
                continue  # same word: the earlier store lands first
            seen.add(key)
            n = dict(st)
            n["pend"] = st["pend"][:j] + st["pend"][j + 1 :]
            if kind == "ack":
                a = list(n["ack"][idx])
                a[lane] = val
                n["ack"] = put(n["ack"], idx, tuple(a))
            else:
                n["term"] = put(n["term"], idx, val)
            yield (f"deliver {kind}[{idx}][{lane}]", self.emit(n))

    def environment(self, st: dict):
        c = self.c
        if st["fatal"] and not st["dead"]:
            n = dict(st)
            n["dead"] = 1
            yield ("watchdog aborts the process", self.emit(n))
        if c.shutdown and not st["shut"]:
            n = dict(st)
            n["shut"] = 1
            yield ("shutdown: admission closed", self.emit(n))
        if c.shutdown and st["shut"] and not st["freed"] and not st["quar"]:
            finished = st["dpc"] in (HALT,) or (st["dpc"] == IDLE and st["dk"] >= c.requests)
            executor_done = st["hpc"] == H_IDLE and not any(st["hl"])
            if (finished and (executor_done or not c.free_waits_executor)) or not c.free_needs_sync:
                n = dict(st)
                n["freed"] = 1
                if any(h >= 0 for h in st["dhold"]) or st["dpc"] in (COPY_B, COPY_E, ACK):
                    self.flag(n, "FreedWhileReading")
                if not executor_done:
                    self.flag(n, "FreedWhileReading")  # a promotion copy or lease is still outstanding
                yield ("python frees memory", self.emit(n))
            if st["dpc"] == ERR:
                n = dict(st)
                n["quar"] = 1
                yield ("sync fails: quarantine", self.emit(n))

    # ---- device ----
    def device(self, st: dict):
        c = self.c
        pc = st["dpc"]
        if pc in (HALT, ERR):
            return
        if c.cuda_error and pc in (COPY_B, COPY_E, ACK):
            n = dict(st)
            n["dpc"] = ERR
            yield ("CUDA error: the kernel dies", self.emit(n))
        if pc == IDLE:
            if st["dk"] >= c.requests or st["dstop"] or st["fatal"] or st["shut"]:
                n = dict(st)
                n["dpc"] = HALT
                yield ("device finished", self.emit(n))
                return
            if st["paused"]:
                return  # the eager caller owns the slots: no graph replay runs meanwhile
            seq, ep = next_seq_device(c, st["dseq"], st["dep"])
            idx = ring_index(c, seq)
            shapes = c.menu if c.menu is not None else [
                e for count in range(c.lanes + 1) for e in itertools.product(range(c.experts), repeat=count)
            ]
            for experts in shapes:
                    count = len(experts)
                    n = dict(st)
                    n["dseq"], n["dep"], n["dk"] = seq, ep, st["dk"] + 1
                    n["dreq"] = (idx, gen_key(c, ep, seq), seq, count, experts, 1 if count > 0 else 0)
                    n["dpc"] = P_LREQ if c.protocol else P_REC
                    n["dhold"] = tuple(-1 for _ in range(c.lanes))
                    n["dctx"] = tuple(None for _ in range(c.lanes))
                    n["ddel"] = tuple(None for _ in range(c.lanes))
                    n["dread"] = tuple(None for _ in range(c.lanes))
                    n["dok"], n["dgo"] = True, 0
                    yield (f"post seq {seq} lanes {experts}", self.emit(n))
            return
        idx, gen, seq, count, experts, armed = st["dreq"]
        n = dict(st)
        if pc == P_LREQ:
            n["lreq"] = put(st["lreq"], idx, (gen, count, experts))
            n["dpc"] = P_REC
            yield ("device writes the lane request", self.emit(n))
        elif pc == P_REC:
            n["rec"] = put(st["rec"], idx, (seq, count, armed, 0))
            n["dpc"] = P_HEAD
            yield ("device writes the record", self.emit(n))
        elif pc == P_HEAD:
            n["head"] = seq
            n["dpc"] = WAIT if armed else MOE
            if not armed:
                n["dgo"] = 0
            yield ("device release-stores demand_head", self.emit(n))
        elif pc == WAIT:
            if st["fatal"] or st["shut"]:
                yield from self.abort(st, "wait sees fatal or shutdown")
            elif reached(c, st["done"], seq):
                n["dpc"] = STATUS
                yield ("wait sees demand_done", self.emit(n))
            elif c.timeouts:
                yield from self.abort(st, "wait times out")
        elif pc == STATUS:
            if st["rec"][idx][3] != 1:
                yield from self.abort(st, "status is not served")
            else:
                n["dpc"], n["di"] = (RD if count > 0 else COMMIT), 0
                yield ("status served", self.emit(n))
        elif pc == RD:
            i = st["di"]
            if c.protocol:
                r = st["rowres"][idx][i]
                good = (
                    r is not None and r[0] == gen and r[1] == 1 and r[4] == experts[i]
                )
                if not good:
                    yield from self.abort(st, f"lane {i}: row result missing, stale or for another expert")
                    return
                n["dctx"] = put(st["dctx"], i, (r[2], r[3]))
                n["dhold"] = put(st["dhold"], i, r[2])
            else:
                slot = st["map"][experts[i]]
                if slot < 0:
                    n["dok"] = False
                    slot = 0
                n["dctx"] = put(st["dctx"], i, (slot, 0))
                n["dhold"] = put(st["dhold"], i, slot)
            n["di"] = i + 1
            if n["di"] >= count:
                n["dpc"] = COMMIT
            yield (f"device resolves lane {i}", self.emit(n))
        elif pc == COMMIT:
            n["dgo"] = count
            n["di"] = 0
            n["dpc"] = COPY_B if count > 0 else MOE
            if c.protocol and not c.ack_after_copy:
                for i in range(count):
                    n["pend"] = n["pend"] + (("ack", idx, i, (gen, CONSUMED)),)
            yield ("commit: copy count set", self.emit(n))
        elif pc == COPY_B:
            i = st["di"]
            slot = st["dctx"][i][0]
            n["dread"] = put(st["dread"], i, (slot, st["scont"][slot]))
            n["dpc"] = COPY_E
            yield (f"copy lane {i} begins on slot {slot}", self.emit(n))
        elif pc == COPY_E:
            i = st["di"]
            slot, snap = st["dread"][i]
            now = st["scont"][slot]
            if now != snap or now[0] == TORN:
                self.flag(n, "BytesChangedUnderCopy")
            n["ddel"] = put(st["ddel"], i, now[0])
            if now[0] != experts[i]:
                self.flag(n, "WrongBytesRead")
            n["di"] = i + 1
            n["dpc"] = COPY_B if n["di"] < st["dgo"] else (ACK if c.protocol and c.ack_after_copy else MOE)
            if n["dpc"] == ACK:
                n["di"] = 0
            yield (f"copy lane {i} ends", self.emit(n))
        elif pc == ACK:
            i = st["di"]
            slot, sgen = st["dctx"][i]
            outcome = CONSUMED
            if c.detector and st["sgen"][slot] != sgen:
                outcome = VIOLATED
                n["dok"] = False
                n["fatal"] = 1
            n["dhold"] = put(st["dhold"], i, -1)
            n["pend"] = n["pend"] + (("ack", idx, i, (gen, outcome)),)
            n["di"] = i + 1
            n["dpc"] = ACK if n["di"] < st["dgo"] else MOE
            yield (f"ack lane {i}", self.emit(n))
        elif pc == MOE:
            if st["dok"] and not st["fatal"]:
                for i in range(count):
                    if st["ddel"][i] != experts[i]:
                        self.flag(n, "WrongBytesAccepted")
            n["dhold"] = tuple(-1 for _ in range(c.lanes))
            n["dpc"] = IDLE
            if not st["dok"] or st["fatal"]:
                n["dstop"] = 1
            yield ("fused MoE consumes", self.emit(n))

    def abort(self, st: dict, why: str):
        """The wait kernel gives up. Designed: zero the copy count, publish the terminal record, raise fatal."""
        c = self.c
        idx, gen, seq, count, experts, armed = st["dreq"]
        n = dict(st)
        n["dok"] = False
        n["fatal"] = 1
        n["dstop"] = 1
        if c.protocol:
            mask = tuple(range(count))
            n["pend"] = n["pend"] + (("term", idx, 0, (gen, mask)),)
        if c.fail_closed:
            n["dgo"] = 0
            n["dhold"] = tuple(-1 for _ in range(c.lanes))
            n["dpc"] = MOE
        else:
            # Today (and the fail-open mutant): the translate loop and the copy still run, and keep = 0 is the only
            # consequence. With the protocol on, the terminal above is published and the copy runs anyway.
            n["dpc"], n["di"] = (RD if count > 0 else COMMIT), 0
        yield (f"device aborts: {why}", self.emit(n))

    # ---- service ----
    def service(self, st: dict):
        c = self.c
        pc = st["spc"]
        if st["dead"] or st["paused"]:
            return
        if pc == S_IDLE:
            yield from self.retire(st)
            if st["shut"]:
                return
            if self.demand_visible(st):
                yield from self.observe(st)
            elif st["advleft"] > 0 and not (c.exclusion and self.in_device_window(st)):
                yield from self.advise(st)
        elif pc == S_LOAD_B:
            yield from self.retire(st)
            slot, expert = st["splan"][0]
            n = dict(st)
            self.check_evict(n, slot)
            n["scont"] = put(st["scont"], slot, (TORN, st["scont"][slot][1]))
            n["spc"] = S_LOAD_E
            yield (f"service begins writing slot {slot}", self.emit(n))
        elif pc == S_LOAD_E:
            slot, expert = st["splan"][0]
            n = dict(st)
            n["scont"] = put(st["scont"], slot, (expert, (st["scont"][slot][1] + 1) % 4))
            n["sst"] = put(st["sst"], slot, READY)
            n["map"] = put(st["map"], expert, slot)
            n["splan"] = st["splan"][1:]
            n["spc"] = S_LOAD_B if n["splan"] else (S_GRANT if c.protocol else S_STATUS)
            n["sreq"] = st["sreq"][:6] + (0, st["sreq"][7])
            yield (f"service finishes slot {slot} (expert {expert})", self.emit(n))
        elif pc == S_GRANT:
            yield from self.retire(st)
            idx, gen, seq, count, experts, armed, lane, failed = st["sreq"]
            if lane >= count:
                n = dict(st)
                n["spc"] = S_STATUS
                yield ("all lanes granted", self.emit(n))
                return
            slot = self.resident(st, experts[lane])
            n = dict(st)
            if slot < 0:
                self.flag(n, "InternalIdentity")
                slot = 0
            n["slease"] = put(st["slease"], slot, st["slease"][slot] + 1)
            if n["slease"][slot] >= 2:
                self.note("two leases on one slot")
            n["rowres"] = put(
                st["rowres"], idx, tuple(
                    (gen, 1, slot, st["sgen"][slot], experts[lane]) if j == lane else st["rowres"][idx][j]
                    for j in range(c.lanes)
                ),
            )
            entry = st["out"][idx]
            lanes = list(entry[1]) if entry and entry[0] == gen else [(NONE, -1, 0)] * c.lanes
            lanes[lane] = (GRANTED, slot, st["sgen"][slot])
            n["out"] = put(st["out"], idx, (gen, tuple(lanes), False))
            n["sreq"] = st["sreq"][:6] + (lane + 1, failed)
            yield (f"service grants and publishes lane {lane} on slot {slot}", self.emit(n))
        elif pc == S_STATUS:
            idx = st["sreq"][0]
            n = dict(st)
            r = st["rec"][idx]
            n["rec"] = put(st["rec"], idx, (r[0], r[1], r[2], 2 if st["sreq"][7] else 1))
            n["spc"] = S_DONE
            yield ("service sets the record status", self.emit(n))
        elif pc == S_DONE:
            idx, gen, seq = st["sreq"][0], st["sreq"][1], st["sreq"][2]
            n = dict(st)
            n["done"] = seq
            entry = st["out"][idx]
            if entry and entry[0] == gen:
                n["out"] = put(st["out"], idx, (entry[0], entry[1], True))
            n["nd"], n["sep"] = next_seq_service(c, st["nd"], st["sep"])
            n["sreq"], n["spc"] = None, S_IDLE
            yield (f"service stores demand_done {seq}", self.emit(n))
        elif pc == S_ADV_B:
            slot, expert = st["splan"][0]
            n = dict(st)
            self.check_evict(n, slot)
            n["scont"] = put(st["scont"], slot, (TORN, st["scont"][slot][1]))
            n["spc"] = S_ADV_E
            yield (f"advisory begins writing slot {slot}", self.emit(n))
        elif pc == S_ADV_E:
            slot, expert = st["splan"][0]
            n = dict(st)
            n["scont"] = put(st["scont"], slot, (expert, (st["scont"][slot][1] + 1) % 4))
            n["sst"] = put(st["sst"], slot, READY)
            n["map"] = put(st["map"], expert, slot)
            n["splan"], n["spc"] = (), S_IDLE
            yield (f"advisory finishes slot {slot} (expert {expert})", self.emit(n))

    def leased(self, st: dict, slot: int, eager: bool = False) -> bool:
        """The eviction predicate: a graph-lane lease, or a host lease on the path that respects them."""
        c = self.c
        host_guard = c.eager_host_guard if eager else c.host_guard
        return (c.leases and st["slease"][slot] > 0) or (host_guard and st["hl"][slot] > 0)

    def check_evict(self, n: dict, slot: int) -> None:
        """Ghost check of I1 at the first byte store: no lease of either kind and no reader may hold the slot."""
        if n["slease"][slot] > 0 or n["hl"][slot] > 0 or self.holds(n, slot):
            self.flag(n, "RecycledUnderReader")

    def advise(self, st: dict):
        c = self.c
        for slot in range(c.slots):
            for expert in range(c.experts):
                if self.resident(st, expert) >= 0:
                    continue
                if st["sst"][slot] == LOADING:
                    continue
                if st["sst"][slot] == READY and self.leased(st, slot):
                    continue
                n = dict(st)
                if st["sst"][slot] == READY:
                    old = st["sexp"][slot]
                    n["map"] = put(st["map"], old, -1)
                n["sgen"] = put(st["sgen"], slot, (st["sgen"][slot] + 1) % 4)
                n["sst"] = put(st["sst"], slot, LOADING)
                n["sexp"] = put(st["sexp"], slot, expert)
                n["splan"] = ((slot, expert),)
                n["advleft"] = st["advleft"] - 1
                n["spc"] = S_ADV_B
                yield (f"advisory takes slot {slot} for expert {expert}", self.emit(n))

    def observe(self, st: dict):
        c = self.c
        head, nd = st["head"], st["nd"]
        n = dict(st)
        if (head - nd) % c.modulus() >= c.ring:
            n["overruns"] = min(2, st["overruns"] + 1)
            nd = (head - (c.ring - 2)) % c.modulus()
            if nd == 0 and c.skip0 and c.skip0_lap:
                nd = 1
                n["sep"] = st["sep"] + 1
            n["nd"] = nd
            yield ("service resumes after a lap", self.emit(n))
            return
        idx = ring_index(c, nd)
        rec = st["rec"][idx]
        if nd == 0:
            # The device never posts sequence 0. On a used page the read below fails, on a fresh page the empty
            # record of slot (0 - 1) % ring reads as a request: either way the service is handling nothing.
            self.flag(n, "PhantomSequence")
        if rec[0] != nd:
            n["overruns"] = min(2, st["overruns"] + 1)
            n["done"] = nd
            n["nd"], n["sep"] = next_seq_service(c, nd, st["sep"])
            yield (f"service finds no record for seq {nd}: overrun", self.emit(n))
            return
        seq = nd
        gen = gen_key(c, st["sep"], seq)
        if c.protocol and c.echo_gen:
            lr0 = st["lreq"][idx]
            if lr0 is not None and lr0[0][1] == seq:
                gen = lr0[0]
        if c.protocol:
            entry = st["out"][idx]
            if c.defer_reuse and entry is not None and not (entry[2] and all(l[0] != GRANTED for l in entry[1])):
                self.note("deferred: request slot not retired")
                return  # deferred: the request slot still holds an unretired lease row
            term = st["term"][idx]
            if term is not None and term[0] == gen:
                self.note("dropped: the device already gave up")
                n["done"] = seq
                n["nd"], n["sep"] = next_seq_service(c, nd, st["sep"])
                yield (f"service drops seq {seq}: the device already gave up", self.emit(n))
                return
            lr = st["lreq"][idx]
            if lr is None or lr[0] != gen:
                # The device already wrote a later request's lanes over this slot: the seqlock re-check fails.
                # Harmless for an unarmed record (nothing waits); an armed one must never be lapped.
                if rec[2] == 1:
                    self.flag(n, "ArmedRequestLapped")
                n["overruns"] = min(2, st["overruns"] + 1)
                n["done"] = nd
                n["nd"], n["sep"] = next_seq_service(c, nd, st["sep"])
                yield (f"service finds the lane request of seq {seq} overwritten: overrun", self.emit(n))
                return
            count, experts = lr[1], lr[2]
        else:
            count, experts = rec[1], self.peek_lanes(st, idx)
        if rec[2] == 0 or count == 0:
            n["sreq"] = (idx, gen, seq, 0, (), 0, 0, False)
            n["spc"] = S_STATUS
            yield (f"service touches seq {seq}", self.emit(n))
            return
        wanted = tuple(dict.fromkeys(experts))
        missing = [e for e in wanted if self.resident(st, e) < 0]
        base = st["sst"]
        free = [sl for sl in range(c.slots) if base[sl] == FREE]
        evictable = [
            sl for sl in range(c.slots)
            if base[sl] == READY and st["sexp"][sl] not in wanted and not self.leased(st, sl)
        ]
        blocked = [
            sl for sl in range(c.slots)
            if base[sl] == READY and st["sexp"][sl] not in wanted and self.leased(st, sl)
        ]
        pool = free + evictable
        if len(pool) < len(missing):
            if len(pool) + len(blocked) >= len(missing) and (c.leases or c.host_guard) and c.defer_leased:
                self.note("deferred: victims are leased")
                return  # deferred: it would fit if leases retired
            n["sreq"] = (idx, gen, seq, count, experts, 1, 0, True)
            n["spc"] = S_STATUS
            yield (f"service fails seq {seq}: no victim", self.emit(n))
            return
        for chosen in itertools.permutations(pool, len(missing)):
            m = dict(st)
            plan = []
            for sl, e in zip(chosen, missing):
                if m["sst"][sl] == READY:
                    m["map"] = put(m["map"], m["sexp"][sl], -1)
                m["sgen"] = put(m["sgen"], sl, (m["sgen"][sl] + 1) % 4)
                m["sst"] = put(m["sst"], sl, LOADING)
                m["sexp"] = put(m["sexp"], sl, e)
                plan.append((sl, e))
            m["splan"] = tuple(plan)
            m["sreq"] = (idx, gen, seq, count, experts, 1, 0, False)
            m["spc"] = S_LOAD_B if plan else (S_GRANT if c.protocol else S_STATUS)
            yield (f"service reserves {plan} for seq {seq}", self.emit(m))
            if c.io_faults and plan:
                f = dict(m)
                for sl, _ in plan:
                    f["sst"] = put(f["sst"], sl, FREE)
                    f["sexp"] = put(f["sexp"], sl, -1)
                f["splan"] = ()
                f["sreq"] = (idx, gen, seq, count, experts, 1, 0, True)
                f["spc"] = S_STATUS
                yield (f"service read of seq {seq} fails", self.emit(f))

    def peek_lanes(self, st: dict, idx: int) -> tuple:
        # Today the service learns the lanes from need/protect; the model hands it the device's lanes.
        return st["dreq"][4] if st["dreq"] is not None and st["dreq"][0] == idx else ()

    def retire(self, st: dict):
        c = self.c
        if not c.protocol or not c.retire:
            return
        for idx in range(c.ring):
            entry = st["out"][idx]
            if entry is None:
                continue
            gen, lanes, final = entry
            for lane in range(c.lanes):
                state, slot, sgen = lanes[lane]
                if state != GRANTED:
                    continue
                a = st["ack"][idx][lane]
                t = st["term"][idx]
                for why, hit in (("ack", a is not None and a[0] == gen), ("terminal", t is not None and t[0] == gen and lane in t[1])):
                    if not hit:
                        continue
                    n = dict(st)
                    if st["slease"][slot] <= 0:
                        self.flag(n, "LeaseUnderflow")
                    n["slease"] = put(st["slease"], slot, st["slease"][slot] - 1)
                    self.note(f"retired by {why}")
                    if why == "ack" and a[1] == VIOLATED:
                        self.flag(n, "LeaseViolationCounted")
                    new = list(lanes)
                    new[lane] = (ACKED if why == "ack" else VOID, slot, sgen)
                    done = final and all(x[0] != GRANTED for x in new)
                    n["out"] = put(st["out"], idx, None if done else (gen, tuple(new), final))
                    yield (f"service retires lane {lane} of request slot {idx} by {why}", self.emit(n))

    # ---- Task 8: the promoter (host leases) ----
    def host(self, st: dict):
        c = self.c
        pc = st["hpc"]
        if pc == H_IDLE:
            if st["hleft"] <= 0:
                return
            if st["shut"] and c.host_admission_closed:
                return  # shutdown closed admission: no new lease
            if c.poll_release and st["epc"] == E_SYNC:
                return  # the scheduler thread is blocked in synchronize() and cannot run the poll step
            for slot in range(c.slots):
                if st["sst"][slot] != READY:
                    continue
                n = dict(st)
                n["hl"] = put(st["hl"], slot, st["hl"][slot] + 1)
                n["hslot"], n["hexp"], n["hleft"] = slot, st["sexp"][slot], st["hleft"] - 1
                if st["freed"]:
                    self.flag(n, "HostLeaseAfterFree")
                in_flight = st["dpc"] not in (IDLE, HALT, ERR)
                # A3 violated: the copy is ordered after the serving stream's queued work (wait_stream).
                n["hneed"] = st["dk"] if (c.copy_waits_on_serving and in_flight) else -1
                n["hpc"] = H_COPY_B
                self.note("host lease taken")
                yield (f"promoter leases slot {slot} (expert {st['sexp'][slot]})", self.emit(n))
        elif pc == H_COPY_B:
            waits = st["hneed"] >= 0 and not (st["dpc"] in (IDLE, HALT, ERR) or st["dk"] > st["hneed"])
            if waits:
                return  # the copy is queued behind the serving stream
            slot = st["hslot"]
            n = dict(st)
            n["hsnap"] = st["scont"][slot]
            n["hpc"] = H_COPY_E
            yield (f"promotion copy begins on slot {slot}", self.emit(n))
        elif pc == H_COPY_E:
            slot = st["hslot"]
            now = st["scont"][slot]
            n = dict(st)
            if now != st["hsnap"] or now[0] == TORN:
                self.flag(n, "BytesChangedUnderCopy")
            if now[0] != st["hexp"]:
                self.flag(n, "WrongBytesRead")
            n["hpc"] = H_REL
            yield (f"promotion copy ends on slot {slot}", self.emit(n))
        elif pc == H_REL:
            if c.poll_release and st["epc"] == E_SYNC:
                return  # the release is the scheduler thread's poll step, which is blocked
            slot = st["hslot"]
            n = dict(st)
            if c.host_release:
                if st["hl"][slot] <= 0:
                    self.flag(n, "LeaseUnderflow")
                n["hl"] = put(st["hl"], slot, st["hl"][slot] - 1)
            n["hpc"], n["hslot"], n["hsnap"] = H_IDLE, -1, None
            yield (f"promoter releases slot {slot}", self.emit(n))

    # ---- the eager caller: pause, assign by hand, resume (before_host_use) ----
    def eager(self, st: dict):
        c = self.c
        pc = st["epc"]
        if pc == E_IDLE and c.blocking_eager:
            if st["eleft"] <= 0 or st["paused"] or st["fatal"]:
                return
            n = dict(st)
            n["epc"] = E_SYNC
            yield ("scheduler enters before_host_use and blocks in synchronize", self.emit(n))
            return
        if pc == E_SYNC:
            base = (
                st["dpc"] in (IDLE, HALT) and not st["pend"] and all(o is None for o in st["out"])
                and st["spc"] == S_IDLE and not st["fatal"]
            )
            if not base:
                return
            if any(x > 0 for x in st["hl"]) and c.pause_counts_host:
                n = dict(st)
                self.flag(n, "PauseBlockedByHostLease")
                yield ("the eager pause is refused only because a promotion holds a lease", self.emit(n))
                return
            n = dict(st)
            n["paused"], n["epc"] = 1, E_TAKE
            if any(x > 0 for x in st["hl"]):
                self.note("pause granted while a host lease is held")
            yield ("eager caller pauses the service", self.emit(n))
            return
        if pc == E_IDLE:
            if st["eleft"] <= 0 or st["paused"] or st["spc"] != S_IDLE:
                return
            # The current stream was synchronized (no graph work in flight, device stores landed) and the
            # pause acknowledgement retired what it could: graph-lane leases must be zero.
            base = st["dpc"] in (IDLE, HALT) and not st["pend"] and all(o is None for o in st["out"])
            if not base or st["fatal"]:
                return
            if any(x > 0 for x in st["hl"]) and c.pause_counts_host:
                n = dict(st)
                self.flag(n, "PauseBlockedByHostLease")
                yield ("the eager pause is refused only because a promotion holds a lease", self.emit(n))
                return
            n = dict(st)
            n["paused"], n["epc"] = 1, E_TAKE
            if any(x > 0 for x in st["hl"]):
                self.note("pause granted while a host lease is held")
            yield ("eager caller pauses the service", self.emit(n))
        elif pc == E_TAKE:
            n = dict(st)
            n["epc"] = E_RESUME
            yield ("eager caller assigns nothing", self.emit(n))
            for slot in range(c.slots):
                for expert in range(c.experts):
                    if self.resident(st, expert) >= 0:
                        continue
                    if st["sst"][slot] == LOADING:
                        continue
                    if st["sst"][slot] == READY and self.leased(st, slot, eager=True):
                        continue
                    n = dict(st)
                    if st["sst"][slot] == READY:
                        n["map"] = put(st["map"], st["sexp"][slot], -1)
                    n["sgen"] = put(st["sgen"], slot, (st["sgen"][slot] + 1) % 4)
                    n["sst"] = put(st["sst"], slot, LOADING)
                    n["sexp"] = put(st["sexp"], slot, expert)
                    n["eplan"] = ((slot, expert),)
                    n["epc"] = E_WB
                    yield (f"eager assign takes slot {slot} for expert {expert}", self.emit(n))
        elif pc == E_WB:
            slot, expert = st["eplan"][0]
            n = dict(st)
            self.check_evict(n, slot)
            n["scont"] = put(st["scont"], slot, (TORN, st["scont"][slot][1]))
            n["epc"] = E_WE
            yield (f"eager assign begins writing slot {slot}", self.emit(n))
        elif pc == E_WE:
            slot, expert = st["eplan"][0]
            n = dict(st)
            n["scont"] = put(st["scont"], slot, (expert, (st["scont"][slot][1] + 1) % 4))
            n["sst"] = put(st["sst"], slot, READY)
            n["map"] = put(st["map"], expert, slot)
            n["eplan"], n["epc"] = (), E_RESUME
            yield (f"eager assign finishes slot {slot}", self.emit(n))
        elif pc == E_RESUME:
            n = dict(st)
            n["paused"], n["epc"], n["eleft"] = 0, E_IDLE, st["eleft"] - 1
            yield ("eager caller resumes the service", self.emit(n))

    # ---- classification of states ----
    def quiescent_problem(self, st: dict) -> Optional[str]:
        """Called for a state with no successor: a deadlock, or a leak in a healthy run, or nothing."""
        c = self.c
        if st["dead"] or st["quar"] or st["freed"]:
            return None
        healthy_done = (
            st["dpc"] in (HALT,) and st["spc"] == S_IDLE and not st["pend"] and not self.demand_visible(st)
            and st["hpc"] == H_IDLE and st["epc"] == E_IDLE and not st["paused"]
        )
        if healthy_done and not st["fatal"]:
            if any(x > 0 for x in st["hl"]):
                return "LeakedLease"
            if c.protocol and (any(x > 0 for x in st["slease"]) or any(o is not None for o in st["out"])):
                return "LeakedLease"
            return None
        return "Deadlock"


VIOLATION_KINDS = (
    "RecycledUnderReader", "WrongBytesRead", "WrongBytesAccepted", "BytesChangedUnderCopy", "FreedWhileReading",
    "LeaseUnderflow", "InternalIdentity", "PhantomSequence", "ArmedRequestLapped", "SpuriousFatal", "LeakedLease", "Deadlock",
    "PauseBlockedByHostLease", "HostLeaseAfterFree",
)


@dataclass
class Result:
    violation: Optional[str]
    trace: list[str]
    states: int
    complete: bool  # every reachable state was visited (no state cap hit)
    seen_violations: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)  # situations reached, see ``Model.note``


def explore(cfg: Config, stop_on: Optional[Callable[[str], bool]] = None, _traced: bool = False) -> Result:
    """Breadth-first over every interleaving. The first pass keeps only state hashes (a 64-bit collision could
    hide a state; at these sizes that is about 1e-5 per run); a violation makes it run again keeping parents,
    to return the shortest trace. ``stop_on`` says which kinds end the search; others are recorded and it goes on."""
    m = Model(cfg)
    init = m.initial()
    parent = {init: None} if _traced else None
    seen = {hash(init)} if not _traced else None
    queue = deque([init])
    found: dict[str, tuple] = {}
    count = 1

    def trace_of(state: tuple) -> list[str]:
        out = []
        while parent[state] is not None:
            state, label = parent[state]
            out.append(label)
        return list(reversed(out))

    def report(name: str, state: tuple) -> Result:
        if not _traced:
            again = explore(cfg, lambda k, name=name: k == name, _traced=True)
            return Result(name, again.trace, count, True, {name: again.trace}, dict(m.stats))
        return Result(name, trace_of(state), count, True, {name: trace_of(state)}, dict(m.stats))

    while queue:
        s = queue.popleft()
        st = m.d(s)
        if st["fatal"] and not (cfg.timeouts or cfg.io_faults or cfg.cuda_error or cfg.shutdown):
            st["viol"] = st["viol"] + ("SpuriousFatal",)  # no fault was injected, yet the run failed stop
        for name in st["viol"]:
            found.setdefault(name, s)
            if stop_on is None or stop_on(name):
                return report(name, s)
        succ = list(m.successors(s))
        if not succ:
            problem = m.quiescent_problem(st)
            if problem:
                found.setdefault(problem, s)
                if stop_on is None or stop_on(problem):
                    return report(problem, s)
            continue
        for label, nxt in succ:
            m.note("step: " + "".join(ch for ch in label.split(":")[0] if not ch.isdigit()).replace("[]", "").replace("()", "").replace("(,)", "").replace("  ", " ").strip())
            if _traced:
                if nxt in parent:
                    continue
                parent[nxt] = (s, label)
            else:
                h = hash(nxt)
                if h in seen:
                    continue
                seen.add(h)
            count += 1
            if count >= cfg.max_states:
                return Result(None, [], count, False, {k: [] for k in found}, dict(m.stats))
            queue.append(nxt)
    return Result(None, [], count, True, {k: [] for k in found}, dict(m.stats))


def main() -> None:  # pragma: no cover - a convenience for reading a counterexample
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="{}", help="JSON overrides of Config fields")
    ap.add_argument("--today", action="store_true", help="start from the protocol as it is today")
    args = ap.parse_args()
    overrides = json.loads(args.config)
    if overrides.get("menu") is not None:
        overrides["menu"] = tuple(tuple(shape) for shape in overrides["menu"])
    cfg = today(**overrides) if args.today else Config(**overrides)
    res = explore(cfg)
    print(f"states {res.states} complete {res.complete} violation {res.violation}")
    for step in res.trace:
        print("  ", step)


if __name__ == "__main__":  # pragma: no cover
    main()
