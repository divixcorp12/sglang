"""OPEN 11: the per-layer cost of arming every `count > 0` record when advise is off.

LEASE_PROTOCOL.md 15 requires lease mode to arm every request with lanes, because the handshake is where the
lease is granted. Without advise, today's all-hit records are UNARMED and skip the service round trip entirely;
in lease mode they cannot. Section 20.1 item 13 registers this as "a measurement, not a test", bundled with
OPEN 5 (the added fences and the acknowledgement kernel).

What is compared, on one layer, with every planned expert already resident (the case that changes):

  A  lease OFF : post -> legacy wait (unarmed: no poll)            -> copy
  C  lease ON  : post -> lease wait (armed: waits for the service) -> copy -> ack

C - A is the per-layer cost. It bundles the round trip (OPEN 11) with the ack kernel and fences (OPEN 5); the
phase breakdown below attributes it, at the price of a synchronize between phases, which inflates absolutes.

Not measured here: the real backend. `Exl3RamMissRowBackend` has no lease `post` override and there is no
environment switch, so step 5 of section 20.1 has not landed and a switch-on/switch-off serving comparison is
not possible. This measures the kernels and the real C++ service thread, which is the level item 13 names.

Run on divix01 under gpu-run.sh (holds cc-gpu.lock), taskset -c 0-63, OMP_NUM_THREADS=1.
"""
import statistics as S
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "test" / "manual" / "dsv41"))

from sglang.kernels.ops.moe.exl3_ram_miss import Exl3RamMissDevice, Exl3RamMissHost, new_page  # noqa: E402
from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu, expert_row_segments  # noqa: E402

LAYERS, EXPERTS, CAPACITY, TOP_K = 2, 16, 8, 6
REPS, WARMUP = 300, 30
HITS = [3, 5, 7]          # all resident after the priming step: the all-hit case OPEN 11 is about


class Rig:
    """The Service harness of test_exl3_lease_kernels_cuda.py, parameterised by lease on/off."""

    def __init__(self, tmp, lease: bool, timeout_ms=2000):
        from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
        from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
        from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
        from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
        from sglang.test.dsv41_fake_exl3 import write_fake_exl3

        self.lease = lease
        write_fake_exl3(str(tmp), num_layers=LAYERS, num_experts=EXPERTS, hidden=1024, inter=512, finite=True)
        self.layout = build_exl3_expert_layout(str(tmp))
        self.fmt = Exl3ExpertFormat(self.layout, 0, direct=False)
        self.specs = {s.name: s for s in self.fmt.tensor_specs(None)}
        self.names = EXL3_STREAMED_NAMES
        self.slabs = {lid: {n: allocate_host_slab(CAPACITY, self.specs[n].row_shape, self.specs[n].dtype, register=True)
                            for n in self.names} for lid in range(LAYERS)}
        tables = exl3_ram_miss_tables(self.layout, self.fmt.segment_map(), self.slabs)
        self.page = new_page(pin=True)
        slot_map = torch.full((LAYERS, EXPERTS), -1, dtype=torch.int32).pin_memory()
        self.host = Exl3RamMissHost(tables, page=self.page, slot_map=slot_map, direct=False)
        kw = {}
        if lease:
            self.host.enable_lease_mode()
            kw = {"lease_block": self.host.lease_block, "lease_layout": self.host.lease_layout}
        self.host.start_thread(fatal_wait_s=60.0)
        self.dev = Exl3RamMissDevice(self.page, slot_map, device="cuda", layers=LAYERS,
                                     timeout_ms=timeout_ms, advise=False, **kw)
        self.planned = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.routes = torch.full((TOP_K,), -1, dtype=torch.int64, device="cuda")
        self.host_rows = torch.zeros(TOP_K, dtype=torch.int64, device="cuda")
        self.keep = torch.ones(1, dtype=torch.float32, device="cuda")
        self.ram_miss = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.dest = {n: torch.zeros((TOP_K,) + self.specs[n].row_shape, dtype=self.specs[n].dtype, device="cuda")
                     for n in self.names}
        self.segments = expert_row_segments([(self.slabs[0][n], self.dest[n]) for n in self.names])
        self.dest_slots = torch.arange(TOP_K, dtype=torch.int32, device="cuda")

    def plan(self, experts):
        self.planned.fill_(-1); self.planned[: len(experts)] = torch.tensor(experts, dtype=torch.int64)
        self.count.fill_(len(experts))
        self.routes.fill_(-1); self.routes[: len(experts)] = torch.tensor(experts, dtype=torch.int64)

    def step(self, row=0):
        self.dev.post(row, self.planned, self.count, self.routes, -1)
        self.dev.wait(row, self.planned, self.count, self.host_rows, self.keep, self.ram_miss)
        n = self.dev.go_count if self.lease else self.count
        copy_expert_row_segments_gpu(self.segments, self.host_rows, self.dest_slots, n)
        if self.lease:
            self.dev.ack(self.keep)

    def phases(self, row=0):
        """Per-phase, with a synchronize between each: attribution, not a wall-clock figure."""
        out = {}
        for name, fn in (("post", lambda: self.dev.post(row, self.planned, self.count, self.routes, -1)),
                         ("wait", lambda: self.dev.wait(row, self.planned, self.count, self.host_rows, self.keep, self.ram_miss)),
                         ("copy", lambda: copy_expert_row_segments_gpu(
                             self.segments, self.host_rows, self.dest_slots,
                             self.dev.go_count if self.lease else self.count)),
                         ("ack", (lambda: self.dev.ack(self.keep)) if self.lease else None)):
            if fn is None:
                out[name] = None; continue
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); out[name] = (time.perf_counter() - t0) * 1e3
        return out

    def close(self):
        from sglang.srt.layers.moe.expert_host_tier import release_host_slabs
        try:
            if self.host is not None:
                self.host.stop()
        finally:
            release_host_slabs([slab for names in self.slabs.values() for slab in names.values()])


def measure(tmp, lease):
    rig = Rig(tmp, lease)
    try:
        rig.plan(HITS)
        for _ in range(3):                     # prime: first step misses and makes the rows resident
            rig.step(); torch.cuda.synchronize(); time.sleep(0.05)
        before = rig.host.counters()["rows_read"]
        for _ in range(WARMUP):
            rig.step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            rig.step()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        after = rig.host.counters()["rows_read"]
        ph = [rig.phases() for _ in range(30)]
        return {
            "lease": lease,
            "p50": S.median(ts), "p90": sorted(ts)[int(0.9 * len(ts))], "min": min(ts), "mean": S.fmean(ts),
            "rows_read_during_timing": after - before,   # must be 0: these are hits
            "phases_p50": {k: (S.median([p[k] for p in ph]) if ph[0][k] is not None else None) for k in ph[0]},
            "counters": {k: rig.host.counters()[k] for k in ("leases_granted", "leases_acked", "leases_voided")
                         if k in rig.host.counters()},
        }
    finally:
        rig.close()


if __name__ == "__main__":
    import json, tempfile
    torch.set_num_threads(1)
    out = []
    for lease in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            r = measure(Path(tmp), lease)
            out.append(r)
            print(json.dumps(r), flush=True)
    a, c = out[0], out[1]
    print()
    print("A lease OFF (all-hit, unarmed): p50 %.4f ms  p90 %.4f  min %.4f  rows_read %d"
          % (a["p50"], a["p90"], a["min"], a["rows_read_during_timing"]))
    print("C lease ON  (all-hit, armed)  : p50 %.4f ms  p90 %.4f  min %.4f  rows_read %d"
          % (c["p50"], c["p90"], c["min"], c["rows_read_during_timing"]))
    print("DELTA per layer (C - A)       : p50 %+.4f ms  p90 %+.4f  min %+.4f"
          % (c["p50"] - a["p50"], c["p90"] - a["p90"], c["min"] - a["min"]))
    print("  x40 layers per step         : p50 %+.3f ms" % (40 * (c["p50"] - a["p50"])))
    print("phases p50 (ms, synchronised, attribution only):")
    print("  OFF", {k: (None if v is None else round(v, 4)) for k, v in a["phases_p50"].items()})
    print("  ON ", {k: (None if v is None else round(v, 4)) for k, v in c["phases_p50"].items()})
