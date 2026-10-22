// The engine's compile-time format and router: the one switch both template translation units (engine_host.cpp,
// engine_device.cuh) share.
//
//   -DSGLANG_EXPERT_FORMAT=<name>   the format traits at csrc/moe/formats/<name>/format.h (format.h's ExpertFormatTraits)
//   -DSGLANG_EXPERT_ROUTER=<name>   the router router.h names routers::<name> (ExpertRouter)
//
// expert_stream_transport.py passes both from its (format, router) key, so a build exists only for a registered pair.
// The header is chosen by the macro, so this file names no format.
#pragma once

#ifndef SGLANG_EXPERT_FORMAT
#error "the engine needs -DSGLANG_EXPERT_FORMAT=<a directory under csrc/moe/formats>"
#endif
#ifndef SGLANG_EXPERT_ROUTER
#error "the engine needs -DSGLANG_EXPERT_ROUTER=<a router of expert_stream/router.h's namespace routers>"
#endif

#define SGLANG_EXPERT_STREAM_STR_(x) #x
#define SGLANG_EXPERT_STREAM_STR(x) SGLANG_EXPERT_STREAM_STR_(x)

#include "format.h"
#include "router.h"
#include SGLANG_EXPERT_STREAM_STR(../formats/SGLANG_EXPERT_FORMAT/format.h)

namespace sglang::expert_stream {

using EngineFormat = ::sglang::formats::SGLANG_EXPERT_FORMAT::Format;
using EngineLayout = EngineFormat::Layout;
using EngineRouter = routers::SGLANG_EXPERT_ROUTER;

static_assert(ExpertFormatTraits<EngineFormat>);
static_assert(ExpertRouter<EngineRouter>);
static_assert(
    EngineFormat::kName == SGLANG_EXPERT_STREAM_STR(SGLANG_EXPERT_FORMAT),
    "the format's kName is the directory name -DSGLANG_EXPERT_FORMAT selects it by");
static_assert(
    EngineRouter::kName == SGLANG_EXPERT_STREAM_STR(SGLANG_EXPERT_ROUTER),
    "the router's kName is the name -DSGLANG_EXPERT_ROUTER selects it by");

}  // namespace sglang::expert_stream
