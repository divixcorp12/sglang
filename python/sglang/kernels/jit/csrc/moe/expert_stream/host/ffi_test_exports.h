// The expert-stream host exports only tests call; ffi_exports.h holds the surface the server uses. Every site that
// expands EXPERT_STREAM_HOST_EXPORTS(Exports) also expands EXPERT_STREAM_HOST_TEST_EXPORTS(Exports).
#pragma once

#include "ffi_exports.h"

namespace sglang::expert_stream {

template <class Exports>
struct HostTestExports;

/// \brief The test-only host exports of one HostExports instantiation. Derived from it, so a handle its open()
/// returned resolves here: both use the same function-local registries.
template <ExpertRowLayout Layout, AsyncFileReader Reader, class Build>
struct HostTestExports<HostExports<Layout, Reader, Build>> : HostExports<Layout, Reader, Build> {
  using Base = HostExports<Layout, Reader, Build>;
  using typename Base::Source;
  using Base::check_table_tensors;
  using Base::find;
};

}  // namespace sglang::expert_stream

// The inner macro takes the HostTestExports type, so each line reads Exports::name like EXPERT_STREAM_HOST_EXPORTS'.
#define EXPERT_STREAM_HOST_TEST_EXPORTS(Exports) \
  EXPERT_STREAM_HOST_TEST_EXPORTS_OF(::sglang::expert_stream::HostTestExports<Exports>)
#define EXPERT_STREAM_HOST_TEST_EXPORTS_OF(Exports)
