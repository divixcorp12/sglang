// The bench's fixture file: eight DSV4.1 layers of real EXL3 weights with synthetic FP16 activations, and the bit-exact
// reference outputs the forwards are checked against. The reader is strict: any other shape or length is rejected.
#pragma once
#include <ATen/Tensor.h>

#include <array>
#include <filesystem>
#include <vector>

// One layer: its FP16 input and, per expert, the nine matrices (gate, up, down) x (trellis, suh, svh), grouped by
// projection: matrices[projection * 3 + {0: trellis, 1: suh, 2: svh}][expert].
struct LayerFixture {
  at::Tensor input;
  std::array<std::vector<at::Tensor>, 9> matrices;
};

// The parsed fixture file (magic EXL3BW01): H5120/I2304, eight layers, five experts per layer, unswizzled 3-bit.
struct Fixture {
  int hidden = 0;
  int experts = 0;
  std::vector<LayerFixture> layers;
  // Reads and validates `path`. Throws on a bad magic, shape, length or trailing bytes.
  explicit Fixture(const std::filesystem::path& path);
};

// Checks `actual` against the FP32 reference file at `path`, bit for bit. Throws, naming the file, on a mismatch.
void compare_reference(const std::filesystem::path& path, const std::vector<float>& actual);
