# The expert-id round trip in `_prepare_promotion`: closed as not worth doing for throughput (2026-09-21)

Plan: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, Task 8, first item: "Preserve
CPU expert IDs for admission instead of the avoidable CPU->GPU->CPU metadata round trip."

**Decision: closed. The round trip is real, avoidable and behaviour-identical, and it costs about
54 microseconds per promotion chunk, under 1% of the residency-boundary excess. Nothing was
implemented.** This is a judgement about *throughput*. It does not settle the item on other
grounds (see "What this does not close").

Read-only: nothing under `python/` was touched, no GPU, no drive load. The measurement is a
read-only query of an existing Nsight Systems export (`expert_id_roundtrip_probe.py`).

## Where it is

The ids are Python ints from the start. `ExpertResidencyPolicy._decide_from_host` produces
`ResidencyDecision.desired_experts` (a tuple of ints, `expert_residency.py`), which
`ExpertHotCache.stage_reassign` turns into `reserve()` placements and
`HotCacheSlotTicket.expert_id`. Then, in `ExpertHotCache._prepare_promotion`
(`expert_hot_cache.py`):

```python
expert_rows = [ticket.expert_id for ticket in tickets]                  # a list of ints
...
pinned_cache.ensure_rows(torch.tensor(expert_rows, device=self.device))  # list -> CUDA tensor
```

and in `ExpertPinnedHostCache.ensure_rows` (`expert_stream.py`):

```python
requested = list(dict.fromkeys(int(value) for value in source_ids.tolist()))   # CUDA tensor -> list
```

The tensor's only other use is `source_ids.numel()`. After `ensure_rows`, `_prepare_promotion`
reads `pinned_cache._expert_to_slot` (host only) and passes the same `expert_rows` list to
`FixedRowTransferPlan.set_rows`. So the list exists throughout; the tensor only carries it to the
device and back. `gather_rows` also calls `ensure_rows` with a CUDA tensor, but those ids are real
routed experts that live on the device, so its `.tolist()` is the demand path and is not part of
this item.

## It is avoidable, and a pure refactor

The device makes no decision on the way. `torch.tensor(list_of_ints)` is int64; `.tolist()`
returns the same ints in the same order; `dict.fromkeys(int(v) ...)` is the same order-preserving
de-duplication the host can do directly; and the `numel() == 0` early return equals
`len(list) == 0` (`_load_reserved` returns early on no tickets, and a chunk is never empty). It is
an identity on the values and their order, not a routing or selection step. I found no case where
"preserve the CPU ids" would change behaviour.

## What it costs

Measured from an existing trace, with no GPU and no new run:
`analysis/dsv41-overlap/prof-node.sqlite` on divix01, the option-C node-mode window that contains
the 2.1 s stall (section 18.2 and 18.6 of `DSV41_REFERENCE.md`), taken at `wt-dsv41` `76829dff55`.
Neither `expert_hot_cache.py` nor `expert_stream.py` has a commit since that revision, so the trace
is of the code that runs today.

In the trace the round trip is one `cudaMemcpyAsync` host-to-device of 8n bytes, a
`cudaStreamSynchronize`, a `cudaMemcpyAsync` device-to-host of the same size, and a second
`cudaStreamSynchronize`, on the scheduler thread, in the window before each chunk's six copy
kernels. `expert_id_roundtrip_probe.py` finds and times those pairs:

```
chunks (copy-kernel clusters): 37; windows kept: 35
pairs per kept window: [1]
ids per chunk: min 18 median 21 max 23
round trip us (H2D start to D2H sync end): median 53.5 mean 55.7 p90 60.2 max 92.3
total round trip ms: 1.950
preparation window ms: median 32.7
```

- **Per chunk, 54 us median** (max 92 us). No outlier: the stream was idle every time, as the outer
  `host_use` synchronize implies.
- **Across the whole 19.7 s trace, 1.95 ms.**
- **About 0.16% of the preparation window** (median 32.7 ms, dominated by the NVMe reads in
  `read_host_rows`).
- **About 0.8 ms per residency boundary** (roughly 15 chunks per boundary in this window), against
  the 82-115 ms boundary excess recorded in section 18.6: under 1%. About 0.025 ms per token,
  against 3.6 ms per token of boundary excess.
- For scale, the other tiny host-device copies in the same window (the `_publish_slots` upload, the
  `_refresh_mapping` pageable copy and sync, the `set_rows` uploads) add roughly 100-200 us per
  chunk. Removing all of them would save about 3% of the boundary excess. Those belong to the B
  items in `PROMOTION_ASYNC.md`, not to this one.

Caveats. It is one window of one session, at one revision. Node-mode tracing adds a little per
API call, so the figures are upper bounds. The pairing rule cannot distinguish this pair from the
demand path's lookup pairs of the same size, so windows longer than 200 ms (the first chunk and one
8.2 s window) were skipped; the first candidates I saw at 30-140 ms were those demand-path pairs,
not this. No other configuration, session mix or boundary length was measured.

## Where this disagrees with `PROMOTION_ASYNC.md`

Agreement: the round trip exists as section 3 states, and "M0 saves almost no wall time" (the first
review's confirmation) is now measured.

**1. Section 3's justification does not survive the design's own later sections.** Section 3 says
removing the round trip matters because "in M1 and later the promotion path must not synchronize".
By sections 6 and 7 the asynchronous path never calls `ensure_rows` at all: M1 leases rows already
resident in the pinned tier, and M2 admits through the service. So M0 is not a prerequisite for M1
or M2. It improves only the synchronous path that they replace. The document argues for work on
grounds that its own later sections remove.

**2. The section 11.2 M0 test cannot detect the regression it exists for.** It reads: "`ensure_rows(list)`
on a CPU-tier `ExpertPinnedHostCache` performs no CUDA call (a spy on `torch.tensor` with a `device`
argument and on `Tensor.tolist` of CUDA tensors)". A CPU tier has no CUDA device, so neither the old
code nor the new can make a CUDA call there. Run CPU-only, the test would fail on the old code only
because a list has no `.tolist()` or `.numel()`: an API-existence check with a behaviour test's
name. It follows the same pattern as the first finding of `PROMOTION_ASYNC_REVIEW_2.md` (B1): a
safeguard that cannot fail on the bug it is named for.

**What a discriminating M0 test would have to assert.** The regression is that
`_prepare_promotion` hands the pinned tier a CUDA tensor built from the host list. So the test must
assert *what `_prepare_promotion` passes*, not what `ensure_rows` does with it:

- Call `ExpertHotCache._prepare_promotion` on a lightweight stand-in for `self` (it needs
  `_transfer_plan`, `streamer`, `_copy_routes_for`, `_slot_batch`, `begin_loading`,
  `_cancel_tickets`, `promotion_in_flight` and `device`; `ExpertHotCache.__init__` itself needs a GPU,
  so a real instance cannot be built on a CPU-only runner: see `PROMOTION_ASYNC_REVIEW_2.md` B2).
- Give it a recording fake pinned cache whose `ensure_rows` (or the new host-ids entry) stores its
  argument.
- Assert the argument is a list of Python ints in ticket order, not a `torch.Tensor`.

That assertion fails on the current code (it passes a tensor) and passes on the change. A parity
test over admission decisions (same slots, same evictions, same `populated_rows` for the same ids,
including duplicates and more misses than slots, on a CPU tier) is worth having but cannot
discriminate: the old and new paths are identical by construction there, and it cannot detect a
regression in latency or in what crosses the device boundary, which it does not measure.
Whoever verifies M0 later must not use the section 11.2 test as evidence.

`PROMOTION_ASYNC.md` is owned by its author and is being revised; this note is for them to fold in,
not something I edited there.

## What this does not close

Closing the item is a judgement about throughput, at about 1% of the boundary excess. If someone
later wants the change for another reason, that is a different argument and this document does not
settle it. Two candidates: clarity (the list is already the truth, and the tensor is noise), and
removing a stream synchronize from a path being made asynchronous (if any promotion path that
survives into the async design still calls `ensure_rows` with a CUDA tensor, the first point above
stops applying and the synchronize becomes a real cost there). The change would then be about 15
lines: a host-ids entry holding `ensure_rows`' body, the tensor entry delegating after its own
`.tolist()`, `_prepare_promotion` passing `expert_rows`, plus the argument-type test above.

## Not checked

Any GPU measurement of my own. Other sessions, boundary lengths or promotion bursts. Whether the
option-C trace's tiny-copy costs match a non-node-mode run. `lease_model.py`, `LEASE_PROTOCOL.md`
and the scheduler.
