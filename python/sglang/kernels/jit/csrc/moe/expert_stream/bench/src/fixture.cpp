#include "fixture.h"

#include <ATen/ATen.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {
uint32_t read_u32(std::ifstream& input) {
  uint32_t value;
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("Truncated fixture header");
  return value;
}

at::Tensor read_tensor(std::ifstream& input, at::IntArrayRef shape, at::ScalarType dtype) {
  auto tensor = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
  if (!input.read(static_cast<char*>(tensor.data_ptr()), tensor.nbytes()))
    throw std::runtime_error("Truncated fixture tensor");
  return tensor;
}
}  // namespace

Fixture::Fixture(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open fixture: " + path.string());
  char magic[8];
  if (!input.read(magic, 8) || std::memcmp(magic, "EXL3BW01", 8))
    throw std::runtime_error("Fixture must have EXL3BW01 magic");
  const auto count = read_u32(input);
  experts = read_u32(input);
  hidden = read_u32(input);
  const auto intermediate = read_u32(input);
  if (count != 8 || experts != 5 || hidden != 5120 || intermediate != 2304)
    throw std::runtime_error("Expected the eight-layer DSV4.1 fixture (5 experts, H5120/I2304)");
  // Reject malformed lengths before allocating the ~550 MB fixture.
  constexpr uint64_t matrix_bytes = uint64_t(5120) * 2304 * 3 / 8 + (5120 + 2304) * 2;
  constexpr uint64_t expected = 24 + 8 * (5120 * 2 + 3 * 5 * (matrix_bytes + 16));
  if (std::filesystem::file_size(path) != expected) throw std::runtime_error("Unexpected fixture length");
  layers.reserve(count);
  for (uint32_t layer = 0; layer < count; ++layer) {
    LayerFixture data;
    data.input = read_tensor(input, {1, hidden}, at::kHalf);
    for (int projection = 0; projection < 3; ++projection) {
      for (int expert = 0; expert < experts; ++expert) {
        const auto k = read_u32(input), n = read_u32(input);
        const auto bits = read_u32(input), swizzled = read_u32(input);
        if (k != (projection == 2 ? intermediate : uint32_t(hidden)) ||
            n != (projection == 2 ? uint32_t(hidden) : intermediate) || bits != 3 || swizzled != 0)
          throw std::runtime_error("Fixture requires original unswizzled 3-bit matrices");
        data.matrices[projection * 3].push_back(read_tensor(input, {k / 16, n / 16, 48}, at::kShort));
        data.matrices[projection * 3 + 1].push_back(read_tensor(input, {k}, at::kHalf));
        data.matrices[projection * 3 + 2].push_back(read_tensor(input, {n}, at::kHalf));
      }
    }
    layers.push_back(std::move(data));
  }
  if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing fixture data");
}

void compare_reference(const std::filesystem::path& path, const std::vector<float>& actual) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Missing reference: " + path.string());
  const size_t bytes = actual.size() * sizeof(float);
  if (std::filesystem::file_size(path) != bytes) throw std::runtime_error("Reference length mismatch");
  std::vector<float> expected(actual.size());
  if (!input.read(reinterpret_cast<char*>(expected.data()), bytes) ||
      std::memcmp(expected.data(), actual.data(), bytes))
    throw std::runtime_error("Bit-exact reference check failed: " + path.string());
}
