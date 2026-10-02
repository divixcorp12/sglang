#include "stack_fixture.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <omp.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "aligned.h"
#include "fixture.h"
#include "moe_mul1.h"

namespace fullstack {
namespace fs = std::filesystem;

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

int64_t nbytes(const at::Tensor& t) {
  return static_cast<int64_t>(t.nbytes());
}

// Each name's slab row from the fixture's nine matrices per expert (gate, up, down x trellis, suh, svh), in the pinned
// tier's row shape: w13_* hold gate then up ([slot, 2, ...]), w2_* hold down ([slot, 1, ...]).
constexpr std::array<std::array<int, 2>, kNames> kSources = {{{0, 3}, {1, 4}, {2, 5}, {6, -1}, {7, -1}, {8, -1}}};

std::string fixture_stamp(const fs::path& fixture, const ImageLayout& layout) {
  const fs::path absolute = fs::absolute(fixture);
  return absolute.string() + "\n" + std::to_string(fs::file_size(absolute)) + "\n" +
         std::to_string(fs::last_write_time(absolute).time_since_epoch().count()) + "\nimage_bytes " +
         std::to_string(layout.image_bytes) + "\n";
}

}  // namespace

StackFixture::StackFixture(const fs::path& fixture_path, const fs::path& image_dir) : impl_(std::make_unique<Impl>()) {
  Impl& f = *impl_;
  require_o_direct(image_dir);
  const Fixture fixture(fixture_path);
  f.rows = static_cast<int64_t>(fixture.layers.size());
  f.experts = fixture.experts;
  f.hidden = fixture.hidden;
  const auto& m = fixture.layers[0].matrices;
  f.intermediate = m[2][0].size(0);  // gate svh: [n] = I
  const std::array<int64_t, kNames> row_bytes = {2 * nbytes(m[0][0]), 2 * nbytes(m[1][0]), 2 * nbytes(m[2][0]),
                                                 nbytes(m[6][0]),     nbytes(m[7][0]),     nbytes(m[8][0])};
  f.set.layout = image_layout(row_bytes);
  f.set.experts = f.experts;
  f.set.capacity = kCapacity;
  const ImageLayout& layout = f.set.layout;
  const std::string stamp = fixture_stamp(fixture_path, layout);
  f.slabs.resize(static_cast<size_t>(f.rows));
  for (int64_t row = 0; row < f.rows; ++row) {
    std::array<uint8_t*, kNames> bases{};
    for (int n = 0; n < kNames; ++n) {
      f.slabs[row][n] = aligned_zeroed(kCapacity * row_bytes[n]);
      bases[n] = f.slabs[row][n].get();
    }
    f.set.slabs.push_back(bases);
    char name[40];
    std::snprintf(name, sizeof(name), "fixture-layer-%03lld.rows", static_cast<long long>(row));
    const fs::path path = image_dir / name;
    const auto& matrices = fixture.layers[row].matrices;
    write_row_image(path, layout, f.experts, [&](int64_t e, uint8_t* image) {
      for (int n = 0; n < kNames; ++n) {
        uint8_t* cursor = image + layout.name_offsets[n];
        for (int source : kSources[n]) {
          if (source < 0) continue;
          const at::Tensor& t = matrices[source][e];
          std::memcpy(cursor, t.data_ptr(), t.nbytes());
          cursor += t.nbytes();
        }
      }
    }, stamp);
    f.set.paths.push_back(path.string());
  }
  f.x_stride = round_up(2 * f.hidden, 16);  // cpu_experts/service.py: FP16 rows padded to 16 bytes
  f.x = aligned_zeroed(f.rows * f.x_stride);
  f.inputs.resize(static_cast<size_t>(f.rows * 2 * f.hidden));
  for (int64_t row = 0; row < f.rows; ++row) {
    std::memcpy(f.inputs.data() + row * 2 * f.hidden, fixture.layers[row].input.data_ptr(), 2 * f.hidden);
    std::memcpy(f.x.get() + row * f.x_stride, fixture.layers[row].input.data_ptr(), 2 * f.hidden);
  }
  f.out_stride = 2 * f.hidden * static_cast<int64_t>(sizeof(float));
  f.out = aligned_zeroed(f.rows * f.out_stride);
}

StackFixture::~StackFixture() = default;

int64_t StackFixture::rows() const { return impl_->rows; }
int64_t StackFixture::experts() const { return impl_->experts; }
int64_t StackFixture::hidden() const { return impl_->hidden; }
const RowSet& StackFixture::row_set() const { return impl_->set; }
uint8_t* StackFixture::x_row(int64_t row) const { return impl_->x.get() + row * impl_->x_stride; }
int64_t StackFixture::x_stride() const { return impl_->x_stride; }
float* StackFixture::out_row(int64_t row) const {
  return reinterpret_cast<float*>(impl_->out.get() + row * impl_->out_stride);
}
int64_t StackFixture::out_stride() const { return impl_->out_stride; }

void StackFixture::write_x(int64_t row) const {
  std::memcpy(x_row(row), impl_->inputs.data() + row * 2 * impl_->hidden, 2 * impl_->hidden);
}

// cpu_experts/exl3.py::register_layer over the row's slots: gate is w13 part 0, up part 1, down w2's one part; each
// view is contiguous. Activation 0 (silu) with cpu_forward.cpp's limit 10, unswizzled.
int64_t StackFixture::register_layer(int64_t row) const {
  const Impl& f = *impl_;
  const int64_t H = f.hidden, I = f.intermediate;
  const auto& rb = f.set.layout.row_bytes;
  auto view = [&](int name, int64_t slot, int64_t offset, at::IntArrayRef shape, at::ScalarType dtype) {
    return at::from_blob(f.set.slabs[row][name] + slot * rb[name] + offset, shape, at::TensorOptions().dtype(dtype));
  };
  std::array<std::vector<at::Tensor>, 9> m;
  for (int64_t s = 0; s < kCapacity; ++s) {
    for (int part = 0; part < 2; ++part) {
      m[3 * part + 0].push_back(view(0, s, part * rb[0] / 2, {H / 16, I / 16, 48}, at::kShort));
      m[3 * part + 1].push_back(view(1, s, part * rb[1] / 2, {H}, at::kHalf));
      m[3 * part + 2].push_back(view(2, s, part * rb[2] / 2, {I}, at::kHalf));
    }
    m[6].push_back(view(3, s, 0, {I / 16, H / 16, 48}, at::kShort));
    m[7].push_back(view(4, s, 0, {I}, at::kHalf));
    m[8].push_back(view(5, s, 0, {H}, at::kHalf));
  }
  return exl3_moe_cpu_make_layer(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], {}, {}, {}, 0, 10.0, 0);
}

void StackFixture::free_layer(int64_t handle) {
  exl3_moe_cpu_free_layer(handle);
}

void configure_cpu_kernel_runtime() {
  // ISA detection occurs during kernel static initialization, before main.
  if (!exl3_moe_cpu_has_avx512_bw() || exl3_moe_cpu_has_avx512_vnni() || exl3_moe_cpu_has_avx512_vbmi())
    throw std::runtime_error("This study requires AVX512BW; set EXL3_MOE_CPU_MAX_ISA=bw before launch");
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  omp_set_dynamic(0);
}

void check_reference(const fs::path& path, const std::vector<float>& actual) {
  compare_reference(path, actual);
}

}  // namespace fullstack
