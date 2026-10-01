#pragma once
#include <ATen/Tensor.h>
#include <array>
#include <filesystem>
#include <vector>

struct LayerFixture {
  at::Tensor input;
  std::array<std::vector<at::Tensor>, 9> matrices;
};

struct Fixture {
  int hidden = 0;
  int experts = 0;
  std::vector<LayerFixture> layers;
  explicit Fixture(const std::filesystem::path& path);
};

void compare_reference(const std::filesystem::path& path, const std::vector<float>& actual);
