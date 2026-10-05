#include "stack_fixture.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "aligned.h"
#include "fixture.h"
#include "kernel.h"
#include "moe_mul1.h"
#include "placement.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <fstream>
#include <omp.h>
#include <stdexcept>
#include <string>

namespace fullstack {
namespace fs = std::filesystem;

// The slabs are aligned zeroed buffers; `inputs` keeps each layer's FP16 input so write_x can restore it after a
// forward.
struct StackFixture::Impl {
  int64_t rows = 0;
  int64_t experts = 0;
  int64_t hidden = 0;
  int64_t intermediate = 0;
  RowSet set;
  std::vector<std::array<AlignedBuffer, kNames>> slabs;
  AlignedBuffer x;
  int64_t x_stride = 0;
  AlignedBuffer out;
  int64_t out_stride = 0;
  std::vector<uint8_t> inputs;  // [rows][2 * hidden]: each layer's FP16 fixture input
};

namespace {

// Byte size of a tensor as int64_t.
int64_t nbytes(const at::Tensor& t) {
  return static_cast<int64_t>(t.nbytes());
}

// For each name, the fixture matrices (indexes into a layer's nine, see LayerFixture) its slab row concatenates, in the
// pinned tier's row shape: w13_* hold gate then up ([slot, 2, ...]), w2_* hold down ([slot, 1, ...]); -1 is unused.
constexpr std::array<std::array<int, 2>, kNames> kSources = {{{0, 3}, {1, 4}, {2, 5}, {6, -1}, {7, -1}, {8, -1}}};

// The stamp that lets a later run reuse the row-image files: the fixture's path, size and mtime plus the image size.
std::string fixture_stamp(const fs::path& fixture, const ImageLayout& layout) {
  const fs::path absolute = fs::absolute(fixture);
  return absolute.string() + "\n" + std::to_string(fs::file_size(absolute)) + "\n" +
         std::to_string(fs::last_write_time(absolute).time_since_epoch().count()) + "\nimage_bytes " +
         std::to_string(layout.image_bytes) + "\n";
}

}  // namespace

StackFixture::StackFixture(const fs::path& fixture_path, const fs::path& image_dir, std::vector<int> group_nodes)
    : impl_(std::make_unique<Impl>()) {
  Impl& f = *impl_;
  constexpr int kNodes = ::sglang::expert_stream::wire::Wire::kNodes;
  if (kNodes > 1 && static_cast<int>(group_nodes.size()) != kNodes)
    throw std::runtime_error("give one NUMA node per group (" + std::to_string(kNodes) + ")");
  require_o_direct(image_dir);
  const Fixture fixture(fixture_path);
  f.rows = static_cast<int64_t>(fixture.layers.size());
  f.experts = fixture.experts;
  f.hidden = fixture.hidden;
  const auto& m = fixture.layers[0].matrices;
  f.intermediate = m[2][0].size(0);  // gate svh: [n] = I
  const std::array<int64_t, kNames> row_bytes = {
      2 * nbytes(m[0][0]), 2 * nbytes(m[1][0]), 2 * nbytes(m[2][0]), nbytes(m[6][0]), nbytes(m[7][0]), nbytes(m[8][0])};
  f.set.layout = image_layout(row_bytes);
  f.set.experts = f.experts;
  f.set.capacity = capacity();
  const ImageLayout& layout = f.set.layout;
  const std::string stamp = fixture_stamp(fixture_path, layout);
  f.slabs.resize(static_cast<size_t>(f.rows));
  for (int64_t row = 0; row < f.rows; ++row) {
    std::array<uint8_t*, kNames> bases{};
    for (int n = 0; n < kNames; ++n) {
      f.slabs[row][n] = aligned_zeroed(capacity() * row_bytes[n]);
      bases[n] = f.slabs[row][n].get();
      if (kNodes > 1)
        for (int g = 0; g < kNodes; ++g)
          bind_pages(bases[n] + g * kGroupSlots * row_bytes[n], kGroupSlots * row_bytes[n], group_nodes[g]);
    }
    f.set.slabs.push_back(bases);
    char name[40];
    std::snprintf(name, sizeof(name), "fixture-layer-%03lld.rows", static_cast<long long>(row));
    const fs::path path = image_dir / name;
    const auto& matrices = fixture.layers[row].matrices;
    write_row_image(
        path,
        layout,
        f.experts,
        [&](int64_t e, uint8_t* image) {
          for (int n = 0; n < kNames; ++n) {
            uint8_t* cursor = image + layout.name_offsets[n];
            for (int source : kSources[n]) {
              if (source < 0) continue;
              const at::Tensor& t = matrices[source][e];
              std::memcpy(cursor, t.data_ptr(), t.nbytes());
              cursor += t.nbytes();
            }
          }
        },
        stamp);
    f.set.paths.push_back(path.string());
  }
  f.x_stride = round_up(2 * f.hidden, 16);  // as cpu_experts/service.py pads: FP16 rows to 16 bytes
  f.x = aligned_zeroed(f.rows * f.x_stride);
  f.inputs.resize(static_cast<size_t>(f.rows * 2 * f.hidden));
  for (int64_t row = 0; row < f.rows; ++row) {
    std::memcpy(f.inputs.data() + row * 2 * f.hidden, fixture.layers[row].input.data_ptr(), 2 * f.hidden);
    std::memcpy(f.x.get() + row * f.x_stride, fixture.layers[row].input.data_ptr(), 2 * f.hidden);
  }
  f.out_stride = 2 * kNodes * f.hidden * static_cast<int64_t>(sizeof(float));
  f.out = aligned_zeroed(f.rows * f.out_stride);
}

StackFixture::~StackFixture() = default;

int64_t StackFixture::rows() const {
  return impl_->rows;
}
int64_t StackFixture::experts() const {
  return impl_->experts;
}
int64_t StackFixture::hidden() const {
  return impl_->hidden;
}
const RowSet& StackFixture::row_set() const {
  return impl_->set;
}
uint8_t* StackFixture::x_row(int64_t row) const {
  return impl_->x.get() + row * impl_->x_stride;
}
int64_t StackFixture::x_stride() const {
  return impl_->x_stride;
}
float* StackFixture::out_row(int64_t row) const {
  return reinterpret_cast<float*>(impl_->out.get() + row * impl_->out_stride);
}
int64_t StackFixture::out_stride() const {
  return impl_->out_stride;
}

void StackFixture::write_x(int64_t row) const {
  std::memcpy(x_row(row), impl_->inputs.data() + row * 2 * impl_->hidden, 2 * impl_->hidden);
}

// Mirrors Exl3CpuQuantTrait.layer_spec: the row's six slabs by base pointer and row stride (kNames is
// EXL3_STREAMED_NAMES' order), activation 0 (silu) with cpu_forward.cpp's limit 10, unswizzled 3-bit, made into a layer
// by the EXL3 kernel. Views only: the fixture's slabs outlive it.
std::unique_ptr<::sglang::cpu_experts::CpuExpertLayer> StackFixture::make_layer(int64_t row) const {
  const Impl& f = *impl_;
  const SglangExl3CpuParams params{3, 0};  // bits, swizzled
  ::sglang::cpu_experts::LayerSlabs d;
  d.capacity = static_cast<int32_t>(capacity());
  d.hidden = static_cast<int32_t>(f.hidden);
  d.intermediate = static_cast<int32_t>(f.intermediate);
  d.activation = 0;
  d.act_limit = 10.0f;
  d.slab_count = kNames;
  for (int n = 0; n < kNames; ++n) {
    d.slabs[n] = f.set.slabs[row][n];
    d.slot_bytes[n] = static_cast<uint64_t>(f.set.layout.row_bytes[n]);
  }
  return ::sglang::exl3_cpu::exl3_cpu_kernel().make_layer(d, std::as_bytes(std::span<const SglangExl3CpuParams>(&params, 1)));
}

void StackFixture::preload_slots(int64_t row) const {
  const Impl& f = *impl_;
  const ImageLayout& layout = f.set.layout;
  if (f.experts > kGroupSlots - kStaging) throw std::runtime_error("more experts than mappable slots");
  const std::string& path = f.set.paths[static_cast<size_t>(row)];
  std::ifstream in(path, std::ios::binary);
  std::vector<char> image(static_cast<size_t>(layout.image_bytes));
  for (int64_t e = 0; e < f.experts; ++e) {
    in.seekg(e * layout.row_stride);
    if (!in.read(image.data(), static_cast<std::streamsize>(image.size())))
      throw std::runtime_error("cannot read expert " + std::to_string(e) + "'s image from " + path);
    for (int n = 0; n < kNames; ++n)
      std::memcpy(
          f.set.slabs[row][n] + slot_of(e) * layout.row_bytes[n],
          image.data() + layout.name_offsets[n],
          static_cast<size_t>(layout.row_bytes[n]));
  }
}

void configure_cpu_kernel_runtime() {
  // The kernel detects its ISA tier at the first query, here, after EXL3_MOE_CPU_MAX_ISA was set at launch.
  if (!exl3_moe_cpu_has_avx512_bw() || exl3_moe_cpu_has_avx512_vnni() || exl3_moe_cpu_has_avx512_vbmi())
    throw std::runtime_error("This study requires AVX512BW; set EXL3_MOE_CPU_MAX_ISA=bw before launch");
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  omp_set_dynamic(0);
}

void release_kernel_team() {
  if (omp_pause_resource_all(omp_pause_soft) != 0)
    throw std::runtime_error("Cannot release the bare caller's OpenMP team");
}

void check_reference(const fs::path& path, const std::vector<float>& actual) {
  compare_reference(path, actual);
}

void check_reference_close(const fs::path& path, const std::vector<float>& actual, float tol) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Missing reference: " + path.string());
  const size_t bytes = actual.size() * sizeof(float);
  if (fs::file_size(path) != bytes) throw std::runtime_error("Reference length mismatch");
  std::vector<float> expected(actual.size());
  if (!input.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("Cannot read reference: " + path.string());
  for (size_t i = 0; i < actual.size(); ++i)
    if (!(std::fabs(actual[i] - expected[i]) <= tol * std::max(1.0f, std::fabs(expected[i]))))
      throw std::runtime_error("Reference check failed beyond " + std::to_string(tol) + ": " + path.string());
}

}  // namespace fullstack
