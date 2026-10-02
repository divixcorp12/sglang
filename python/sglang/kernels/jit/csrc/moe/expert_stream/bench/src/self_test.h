// The bench's --self-test: checks the harness itself, with no fixture file and no GPU.
#pragma once

#include "placement.h"
#include <filesystem>

namespace fullstack {

// Runs, in order: the placement rules, DeviceSim's records and lane typing on a standalone page, the row images, then
// the real stack (this binary's build) on synthetic rows with a fake forward. Returns the number of failed checks.
int run_self_test(const Placement& placement, const std::filesystem::path& image_dir);

}  // namespace fullstack
