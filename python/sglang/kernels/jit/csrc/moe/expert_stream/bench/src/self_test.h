#pragma once

#include <filesystem>

#include "placement.h"

namespace fullstack {

// --self-test: placement rules, DeviceSim's records and typing on a standalone page, row images, then the real stack
// (this binary's build) on synthetic rows with a fake forward. Returns the number of failed checks.
int run_self_test(const Placement& placement, const std::filesystem::path& image_dir);

}  // namespace fullstack
