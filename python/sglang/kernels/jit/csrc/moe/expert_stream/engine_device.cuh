// Device side of the expert-stream engine, one translation unit for every format and router: the slot-map chain of one
// decode layer, post -> C1 -> S -> CW -> stream wait on the gate -> CC (analysis/dsv41-drive/LEASE_PROTOCOL.md).
//
// post applies the row's pending map delta, types every lane from the device's map, and writes the record and
// demand_head. C1 copies the HIT_SM lanes (copy_expert_row_segments_gpu); S copies the MISS_GPU lanes from their
// staging slots piece by piece as the read publishes them, and traps at its deadline. CW reads the small tensors of
// the HIT_COPY lanes and closes the gate when the copy thread or the CPU still owes work; CC checks CopyDone after the
// stream wait. Every failure is fail-stop: a kernel traps, the host aborts. There is no error word.
//
// The format and router come from engine_select.h (-DSGLANG_EXPERT_FORMAT / -DSGLANG_EXPERT_ROUTER). The python wrappers
// (expert_stream_transport._device_wrappers) instantiate RowCopyKernel<EngineLayout> and SpecScoreKernelFor<EngineRouter>.

#include "engine_select.h"
#include "lease_kernels.cuh"
#include "row_copy_kernels.cuh"
#include "spec_score.cuh"
#include "residency_snapshot.cuh"
