// The EXL3 instantiation of the expert-stream host transport: its row layout and file reader, and the FFI exports.
#include "exl3/exl3_row_layout.h"
#include "expert_stream/host/faulty_reader.h"
#include "expert_stream/host/ffi_exports.h"
#include "expert_stream/host/uring_reader.h"

namespace sglang {

using Exl3Reader = expert_stream::FaultyReader<expert_stream::UringReader>;
static_assert(expert_stream::AsyncFileReader<Exl3Reader>);
using Exl3HostExports = expert_stream::HostExports<exl3::Exl3RowLayout, Exl3Reader>;

EXPERT_STREAM_HOST_EXPORTS(Exl3HostExports)

}  // namespace sglang
