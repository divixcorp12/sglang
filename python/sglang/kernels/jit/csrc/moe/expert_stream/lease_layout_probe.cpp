// Test-only: prints LeaseLayout's members for the lane/node pairs test_expert_stream_lease_layout checks.
#include <sgl_kernel/tensor.h>

#include <string>

#include "draft_channel.h"
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
  put("home7", L::home(7));
  out.pop_back();
  return out;
}

template <int N>
std::string for_nodes(int64_t nodes) {
  return nodes == 1 ? members<LeaseLayout<N, 1>>() : members<LeaseLayout<N, 2>>();
}

template <class L>
std::string channel_members() {
  using C = TargetChannelOf<L>;
  std::string out;
  auto put = [&](const std::string& name, int64_t value) { out += name + "=" + std::to_string(value) + "\n"; };
  put("chan_head", C::kHead);
  put("chan_ring", C::kRing);
  put("chan_records", C::kRecords);
  put("chan_record_bytes", C::kRecordBytes);
  put("chan_done", C::kDone);
  put("chan_gate", C::kGate);
  for (uint32_t seq : {1u, 2u, 0x1FFFFFFFu, 0x20000001u}) {
    put("gate_open_" + std::to_string(seq), ::sglang::expert_stream::channel::gate_word(seq, L::kLeaseGateOpen));
    put("gate_closed_" + std::to_string(seq), ::sglang::expert_stream::channel::gate_word(seq, L::kLeaseGateClosed));
  }
  namespace d = ::sglang::expert_stream::draft;
  using D = d::DraftChannel;
  put("draft_head", D::kHead);
  put("draft_ring", D::kRing);
  put("draft_records", D::kRecords);
  put("draft_record_bytes", D::kRecordBytes);
  put("draft_done", D::kDone);
  put("draft_gate", D::kGate);
  put("draft_channel_bytes", d::kChannelBytes);
  put("draft_max_rows", d::kMaxRows);
  put("draft_max_k", d::kMaxK);
  put("draft_rec_stage", d::kRecStage);
  put("draft_rec_rows", d::kRecRows);
  put("draft_rec_k", d::kRecK);
  put("draft_rec_epoch", d::kRecEpoch);
  out.pop_back();
  return out;
}

template <int N>
std::string channel_for_nodes(int64_t nodes) {
  return nodes == 1 ? channel_members<LeaseLayout<N, 1>>() : channel_members<LeaseLayout<N, 2>>();
}

// The target's lease channel (TargetChannelOf), the shared gate encoding and the draft's channel
// (draft_channel.h), for test_lease_channel_layout and test_dspark_draft_channel_layout.
std::string channel_probe(int64_t lanes, int64_t nodes) {
  if (nodes != 1 && nodes != 2) return "";
  switch (lanes) {
    case 8: return channel_for_nodes<8>(nodes);
    case 24: return channel_for_nodes<24>(nodes);
    case 32: return channel_for_nodes<32>(nodes);
    default: return "";
  }
}

std::string probe(int64_t lanes, int64_t nodes) {
  if (nodes != 1 && nodes != 2) return "";
  switch (lanes) {
    case 1: return for_nodes<1>(nodes);
    case 6: return for_nodes<6>(nodes);
    case 8: return for_nodes<8>(nodes);
    case 13: return for_nodes<13>(nodes);
    case 24: return for_nodes<24>(nodes);
    case 32: return for_nodes<32>(nodes);
    default: return "";
  }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lease_layout_probe, probe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_lease_channel_probe, channel_probe);
