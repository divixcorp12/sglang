// Device side of the RAM-miss service for EXL3 streamed experts: the slot-map chain of one decode layer,
// post -> C1 -> S -> CW -> stream wait on the gate -> CC (analysis/dsv41-drive/LEASE_PROTOCOL.md).
//
// post applies the row's pending map delta, types every lane from the device's map, and writes the record and
// demand_head. C1 copies the HIT_SM lanes (copy_expert_row_segments_gpu); S copies the MISS_GPU lanes from their
// staging slots piece by piece as the read publishes them, and traps at its deadline. CW reads the small tensors of
// the HIT_COPY lanes and closes the gate when the copy thread or the CPU still owes work; CC checks CopyDone after the
// stream wait. Every failure is fail-stop: a kernel traps, the host aborts. There is no error word.

#include "exl3_row_layout.h"
#include "../expert_stream/lease_kernels.cuh"
#include "../expert_stream/row_copy_kernels.cuh"
