// Device side of the option C RAM-miss service (DSV41 Phase 3b plan, D10-D15, D22).
//
// post: one block; thread 0 builds the layer's request (need = planned VRAM misses
// whose host-mapped slot map entry is -1; protect = every routed expert), writes it
// into the page's demand ring as a seqlock (seq cleared and fenced, volatile payload,
// seq release-stored) and release-stores demand_head. The record is posted for every MoE layer (the thread
// uses touch-only records for LRU recency); the wait is armed only when something is
// needed or advisories are on, and the record says so (kRecArmed): the thread only
// touches for an unarmed record, since nothing orders it before the next gathers. With `advise`, it also remembers this token's routes
// for `row` and posts the previous token's routes of `next_row` that are not in RAM
// as an advisory record.
// wait: one block; thread 0 polls demand_done with ld.acquire.sys and __nanosleep
// until it reaches the armed sequence or `timeout_ns` of %globaltimer passes, then
// translates the planned experts to pinned slots from the slot map. A timeout, a
// failed request, a planned row still not in RAM, or a fatal word already raised on
// the page raises the page's fatal word
// (sticky: later posts post nothing and later waits return at once) and sets keep
// to 0, which drops the layer's routed output for this forward.
//
// Lease mode (LEASE_PROTOCOL.md; the wrappers take a lease block, and a null one leaves everything above as it was):
// post also writes the request's LaneRequest into the lease block and arms every request that has planned lanes;
// lease_wait replaces wait's translate half: it validates each lane's RowResult and commits `go_count[0] = count`
// or fails closed with go_count 0 and a Terminal record; lease_ack is launched after the copy kernel, in the
// same stream, and release-stores one LaneAck word per committed lane (section 6.4 says why it is a separate kernel).
// Copy engine (LEASE_PROTOCOL.md 7.6): the service may publish a hit lane as COPYING and copy it with the DMA engine
// itself; copy_wait then waits for the service's CopyDone word instead of any kernel copying or acknowledging it.

#include "expert_stream/lease_kernels.cuh"
#include "expert_stream/row_copy_kernels.cuh"
#include "exl3/exl3_row_layout.h"
