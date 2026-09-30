// Device side of the RAM-miss service for EXL3 streamed experts: the lease chain of one decode layer,
// post -> W1 -> C1 -> S -> CW -> stream wait on the gate -> CC (analysis/dsv41-drive/LEASE_PROTOCOL.md).
//
// post writes the layer's LaneRequest and demand record and publishes demand_head. W1 claims the lanes the service
// granted before its read and compacts the READY ones for C1 (copy_expert_row_segments_gpu). S copies the rest piece
// by piece as the read publishes them, and traps at its deadline. CW reads the small tensors of the service's DMA
// lanes, publishes Done (no kernel of the request reads a leased slot after it) and closes the gate when the service
// still owes copies; CC checks CopyDone after the stream wait. Every failure is fail-stop: a kernel traps, the host
// aborts. There is no error word.

#include "exl3/exl3_row_layout.h"
#include "expert_stream/lease_kernels.cuh"
#include "expert_stream/row_copy_kernels.cuh"
