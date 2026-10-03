// Test-only: prints LeaseLayout's members for the lane/node pairs test_expert_stream_lease_layout checks.
#include <sgl_kernel/tensor.h>

#include <string>

#include "lease_layout.h"

namespace {

using namespace ::sglang::expert_stream::wire;

template <class L>
std::string members() {
  std::string out;
  auto put = [&](const char* name, int64_t value) { out += std::string(name) + "=" + std::to_string(value) + "\n"; };
#define P(name) put(#name, static_cast<int64_t>(L::name))
  P(kLanes); P(kNodes); P(kDemandHead); P(kDemandRing); P(kDemandRecords); P(kRecSeq); P(kRecRow); P(kRecCounts);
  P(kRecFlags); P(kRecFlagCaptured); P(kRecChain); P(kRecEpoch); P(kRecKinds); P(kKindWords); P(kPackedCounts);
  P(kRecProtectCount); P(kRecHeaderBytes); P(kRecProtect); P(kRecLaneExpert); P(kRecLaneSlot); P(kRecLaneDst);
  P(kRecLaneWeight); P(kRecPayloadEnd); P(kRecordBytes); P(kRecIdMax); P(kPageBytes); P(kKindHitCopy); P(kKindHitSm);
  P(kKindHitCpu); P(kKindMissGpu); P(kKindMissCpu); P(kHotHeaderBytes); P(kHotAlignment); P(kHotRecords);
  P(kLeaseBlockAlign); P(kLeasePieceMask); P(kLeasePieceMaskLineBytes); P(kLeaseCopyDone); P(kLeaseCopyDoneBytes);
  P(kLeaseCopyGate); P(kLeaseGateClosed); P(kLeaseGateOpen); P(kLeaseGateSeqShift); P(kLeaseGateSeqMask);
  P(kCopyArmed); P(kSplit); P(kSplitStride); P(kLeaseBlockBytes); P(kDeltaBase); P(kDeltaTag); P(kDeltaCount);
  P(kDeltaStaging); P(kDeltaEntries); P(kDeltaMaxEntries); P(kDeltaStride);
#undef P
  out.pop_back();
  return out;
}

template <int N>
std::string for_nodes(int64_t nodes) {
  return nodes == 1 ? members<LeaseLayout<N, 1>>() : members<LeaseLayout<N, 2>>();
}

std::string probe(int64_t lanes, int64_t nodes) {
  if (nodes != 1 && nodes != 2) return "";
  switch (lanes) {
    case 1: return for_nodes<1>(nodes);
    case 6: return for_nodes<6>(nodes);
    case 8: return for_nodes<8>(nodes);
    case 13: return for_nodes<13>(nodes);
    case 32: return for_nodes<32>(nodes);
    default: return "";
  }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lease_layout_probe, probe);
